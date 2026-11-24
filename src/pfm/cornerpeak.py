"""Single-corner peak head on the FROZEN PerceptionModel (add-on; the main model's weights are not touched).

Why: the gate head returns whole gates only. A close gate that fills or leaves the frame gets a low score, so in the DroneRacing loop
near gates are lost (found 9 % for gates > 120 px vs 71 % for GateNet, Update 131). GateNet survives there with per-corner heatmaps that feed
the EKF's partial-gate update. This head gives the same thing from our features: 4 corner-type heatmaps (tl, tr, br, bl of the gate,
geometric order in the upright image) at stride 4 plus a shared sub-cell offset, CenterNet style.

Input: the stride-8 fused pyramid map of the main model (128 ch, detached) + a light stride-4 image branch for exact localisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN = 448      # main-model input side
STRIDE = 4


def fused_features(model, x):
    """x normalised (B,3,448,448) -> (B,128,56,56): the same stride-8 fusion the dense heads use. No grad through the main model."""
    with torch.no_grad():
        fmap, _ = model.backbone(x)
        p4, p3, p2 = model.pyramid(fmap)
        f = p4 + F.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False) \
               + F.interpolate(p2, size=p4.shape[-2:], mode="bilinear", align_corners=False)
    return f.float()


def _cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CornerPeakHead(nn.Module):
    def __init__(self, fdim: int = 128):
        super().__init__()
        self.img = nn.Sequential(_cbr(3, 24, 2), _cbr(24, 32, 2), _cbr(32, 32))          # stride 4 image cue
        self.f = nn.Sequential(_cbr(fdim, 96), _cbr(96, 96))
        self.fuse = nn.Sequential(_cbr(96 + 32, 96), _cbr(96, 96), _cbr(96, 64))
        self.heat = nn.Conv2d(64, 4, 1)
        self.off = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.heat.bias, -2.19)

    def forward(self, img01, feat):
        """img01 (B,3,448,448) in 0..1, feat (B,128,56,56) -> heat logits (B,4,112,112), offsets (B,2,112,112) in cells."""
        a = self.img(img01 - 0.5)
        b = F.interpolate(self.f(feat), size=a.shape[-2:], mode="bilinear", align_corners=False)
        h = self.fuse(torch.cat((b, a), 1))
        return self.heat(h), self.off(h)


@torch.no_grad()
def decode_peaks(heat, off, out_hw, thr: float = 0.3, k: int = 3, max_per_type: int = 6):
    """-> list (per image) of [n,4] (type 0..3, score, u, v) in px of an out_hw (h, w) image the 448 input was resized from."""
    hs = heat.float().sigmoid(); B, _, H, W = hs.shape
    pk = (F.max_pool2d(hs, k, 1, k // 2) == hs) & (hs > thr)
    sy, sx = out_hw[0] / H, out_hw[1] / W; res = []
    for b in range(B):
        rows = []
        for t in range(4):
            yx = pk[b, t].nonzero()
            if len(yx) == 0: continue
            sc = hs[b, t, yx[:, 0], yx[:, 1]]
            o = sc.argsort(descending=True)[:max_per_type]; yx, sc = yx[o], sc[o]
            ox, oy = off[b, 0, yx[:, 0], yx[:, 1]].float(), off[b, 1, yx[:, 0], yx[:, 1]].float()
            u = (yx[:, 1].float() + 0.5 + ox) * sx; v = (yx[:, 0].float() + 0.5 + oy) * sy
            rows.append(torch.stack((torch.full_like(sc, t), sc, u, v), -1))
        res.append(torch.cat(rows) if rows else torch.zeros(0, 4, device=heat.device))
    return res


class NearGateHead(nn.Module):
    """Add-on whole-gate head (same dense format as DenseHeads.gdet: centre logit + 8 corner offsets, image-normalised, relative to the cell centre)
    trained on DroneRacing on-policy crops where close gates are cut by the frame but all 4 corners are labelled (also outside the image).
    Frozen main model features + the stride-4 image branch, output at stride 4 (112 x 112 for a 448 input). Corners in simulator identity order."""

    def __init__(self, fdim: int = 128):
        super().__init__()
        self.img = nn.Sequential(_cbr(3, 24, 2), _cbr(24, 32, 2), _cbr(32, 32))
        self.f = nn.Sequential(_cbr(fdim, 128), _cbr(128, 128, 1))
        self.ctx = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 128, 3, 1, 4, dilation=4, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))      # wide context: a close gate spans the whole map
        self.fuse = nn.Sequential(_cbr(128 + 32, 96), _cbr(96, 96))
        self.out = nn.Conv2d(96, 9, 1)
        nn.init.constant_(self.out.bias[:1], -2.19)

    def forward(self, img01, feat):
        a = self.img(img01 - 0.5)
        b = self.f(feat); b = b + self.ctx(b)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.fuse(torch.cat((b, a), 1)))
