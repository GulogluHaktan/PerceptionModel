"""Corner refiner: a small, separate network that sharpens the gate corners of the main model's dense gate head.

Why (handoff Updates 103-125): the dense head regresses all 4 corners as offsets from one centre cell, so its error grows with gate size
(DroneRacing loop: 1.8 / 4.1 / 9.4 px for 0-60 / 60-120 / 120-250 px gates vs GateNet 0.8 / 1.3 / 2.3 px). A corner, seen locally, looks the
same at any gate size, so a patch-level corner locator does not have that problem. The main model stays frozen and unchanged (general metrics
cannot move); this module has its own weights and checkpoint and is applied after decoding.

Canonical patch: centred on the coarse corner, rotated so that the direction corner -> gate centroid points to the patch's bottom-right
(every inner corner then looks alike: opening towards bottom-right, whatever the corner type or camera roll), sampled at s image px per
patch px with s = clamp(gate size / SIZE_DIV, 1, S_MAX) (larger gates: thicker bars and larger coarse errors, so a wider view).
Output: a heatmap over the patch, read out by soft-argmax (sub-pixel), plus a logit "corner visible in this patch"."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 64            # patch side (patch px)
SIZE_DIV = 96.0
S_MAX = 4.0
CANON = math.atan2(1.0, 1.0)     # canonical direction to the centroid: bottom-right (image y down)


def patch_frames(corner, centroid, size):
    """corner, centroid [N,2] image px; size [N] gate size px -> (scale s [N], rotation [N,2,2] mapping canonical patch px -> image px direction)."""
    d = centroid - corner
    ang = torch.atan2(d[:, 1], d[:, 0]) - CANON          # rotate canonical frame by this angle to reach the image frame
    c, s_ = ang.cos(), ang.sin()
    R = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2)      # [N,2,2]
    s = (size / SIZE_DIV).clamp(1.0, S_MAX)
    return s, R


def sample_patches(img, bidx, corner, centroid, size):
    """img [B,3,H,W] (0..1); bidx [N] image index per corner -> patches [N,3,P,P], plus (s, R) for mapping outputs back."""
    _, _, H, W = img.shape
    s, R = patch_frames(corner, centroid, size)
    t = torch.arange(P, device=img.device, dtype=torch.float32) - (P - 1) / 2      # patch px centres, 0 at the corner
    qy, qx = torch.meshgrid(t, t, indexing="ij")
    q = torch.stack([qx, qy], -1).view(1, P * P, 2)                                 # canonical patch coords
    pts = corner[:, None] + s[:, None, None] * torch.einsum("nij,nkj->nki", R, q.expand(len(corner), -1, -1))   # [N,P*P,2] image px
    grid = torch.stack([(pts[..., 0] + 0.5) / W * 2 - 1, (pts[..., 1] + 0.5) / H * 2 - 1], -1).view(-1, P, P, 2)
    patches = F.grid_sample(img[bidx].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return patches, s, R


def to_image(offset_patch, corner, s, R):
    """offset in canonical patch px [N,2] -> refined corner in image px [N,2]."""
    return corner + s[:, None] * torch.einsum("nij,nj->ni", R, offset_patch)


def to_patch(target, corner, s, R):
    """target image px [N,2] -> canonical patch px offset [N,2] (inverse of to_image)."""
    return torch.einsum("nji,nj->ni", R, (target - corner) / s[:, None])


def block(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CornerRefiner(nn.Module):
    """Small U-Net-lite: 64x64 patch -> 64x64 heatmap (soft-argmax) + visibility logit. ~0.3 M parameters."""

    def __init__(self, w: int = 32):
        super().__init__()
        self.e1 = nn.Sequential(block(5, w), block(w, w))                 # 64  (3 colour + 2 coordinate channels)
        self.e2 = nn.Sequential(block(w, 2 * w, 2), block(2 * w, 2 * w))  # 32
        self.e3 = nn.Sequential(block(2 * w, 4 * w, 2), block(4 * w, 4 * w), block(4 * w, 4 * w))   # 16
        self.d2 = nn.Sequential(block(4 * w + 2 * w, 2 * w), block(2 * w, 2 * w))
        self.d1 = nn.Sequential(block(2 * w + w, w), block(w, w))
        self.heat = nn.Conv2d(w, 1, 1)
        self.vis = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 1))
        t = torch.arange(P, dtype=torch.float32) - (P - 1) / 2
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("gx", gx.reshape(-1), persistent=False)
        self.register_buffer("gy", gy.reshape(-1), persistent=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, patches):
        x = (patches - self.mean) / self.std
        n = x.shape[0]
        co = torch.stack([self.gx.view(P, P), self.gy.view(P, P)]).expand(n, -1, -1, -1) / (P / 2)
        x = torch.cat([x, co], 1)
        a = self.e1(x); b = self.e2(a); c = self.e3(b)
        u = self.d2(torch.cat([F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False), b], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False), a], 1))
        h = self.heat(u).flatten(1)                     # [N,P*P]
        p = h.softmax(1)
        off = torch.stack([(p * self.gx).sum(1), (p * self.gy).sum(1)], -1)      # canonical patch px, 0 = coarse corner
        return off, self.vis(c).squeeze(1), h


@torch.no_grad()
def refine_gates(model, img, gates, margin=2.0, vis_thr=0.0):
    """img [B,3,H,W] 0..1; gates: list (per image) of [k,8] corner px (x,y)*4 -> same list with refined corners.
    Corners outside the image (beyond `margin` px) and corners whose visibility logit is below vis_thr keep the coarse value."""
    dev = img.device; _, _, H, W = img.shape
    bidx, cor, cen, size, where = [], [], [], [], []
    for b, g in enumerate(gates):
        if len(g) == 0: continue
        c = g.view(-1, 4, 2).float()
        ctr = c.mean(1); sz = (c.amax(1) - c.amin(1)).amax(-1)
        for i in range(len(c)):
            for j in range(4):
                x, y = c[i, j]
                if -margin <= x < W + margin and -margin <= y < H + margin:
                    bidx.append(b); cor.append(c[i, j]); cen.append(ctr[i]); size.append(sz[i]); where.append((b, i, j))
    if not cor: return gates
    bidx = torch.tensor(bidx, device=dev); cor = torch.stack(cor); cen = torch.stack(cen); size = torch.stack(size)
    patches, s, R = sample_patches(img, bidx, cor, cen, size)
    off, vis, _ = model(patches)
    new = to_image(off, cor, s, R)
    out = [g.clone() for g in gates]
    for k, (b, i, j) in enumerate(where):
        if vis[k] >= vis_thr: out[b][i, 2 * j:2 * j + 2] = new[k].to(out[b].dtype)
    return out
