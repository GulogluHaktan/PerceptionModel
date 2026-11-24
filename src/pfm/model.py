"""Student model: backbone + teacher-specific readout heads + dense task heads.

Decision D1: two students are trained from the same cached teacher features:
  * "vit"    - plain ViT-B (ViTDet-style simple feature pyramid)
  * "hybrid" - conv/attention hybrid (E-RADIO-style stand-in, see TODO)
Teacher-specific readout heads exist from day one (GRAFT idea) so teachers can be
added later without retraining the backbone from scratch.
Decision D2: runtime outputs are dense heads only (depth, class-aware segmentation,
gate mask + corners). No prompt-based SAM at run time.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .classes import CLASSES  # noqa: E402,F401  (D2 class list lives in classes.py)


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style: build a multi-scale pyramid from the single stride-16 ViT map."""

    def __init__(self, dim: int, out_dim: int = 128):
        super().__init__()
        self.up2 = nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, 2), nn.GroupNorm(8, dim // 2), nn.GELU())
        self.p4 = nn.Conv2d(dim // 2, out_dim, 1)   # stride 8
        self.p3 = nn.Conv2d(dim, out_dim, 1)        # stride 16
        self.p2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(dim, out_dim, 1))  # stride 32

    def forward(self, x):
        return [self.p4(self.up2(x)), self.p3(x), self.p2(x)]


class Backbone(nn.Module):
    """Returns a stride-16 (or hybrid-equivalent) feature map of shape (B, C, H/16, W/16)."""

    def __init__(self, kind: str = "vit", pretrained: bool = False, img_size: int = 224,
                 hybrid_name: str = "fastvit_sa24"):
        super().__init__()
        self.kind = kind
        if kind == "vit":
            # Plain ViT-B/16, dynamic image size so we can change resolution (RADIOv2.5 trap).
            self.net = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained, num_classes=0,
                img_size=img_size, dynamic_img_size=True, global_pool="")
            self.dim = self.net.embed_dim
            self.num_prefix = self.net.num_prefix_tokens
        elif kind == "hybrid":
            # Conv+attention hybrid student (TensorRT-friendly). NVIDIA E-RADIO itself is 390M
            # params (see teachers.ERadio), so it is used as a teacher; this is the Orin student.
            # Swap `hybrid_name` for any timm features_only model (e.g. fastvit_sa36, efficientvit_b2).
            self.net = timm.create_model(
                hybrid_name, pretrained=pretrained, features_only=True, out_indices=(2,))
            self.dim = self.net.feature_info.channels()[-1]
            self.num_prefix = 0
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "vit":
            tokens = self.net.forward_features(x)           # (B, P+N, C)
            b, _, c = tokens.shape
            h, w = x.shape[-2] // 16, x.shape[-1] // 16
            summary = tokens[:, 0]
            fmap = tokens[:, self.num_prefix:].transpose(1, 2).reshape(b, c, h, w)
            return fmap, summary
        feats = self.net(x)[-1]
        return feats, feats.mean((2, 3))


class TeacherHead(nn.Module):
    """Teacher-specific readout: maps student features to one teacher's feature space.
    Adding a teacher = adding one of these (backbone can stay frozen/continue)."""

    def __init__(self, dim: int, teacher_dim: int):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, teacher_dim, 1))
        self.summary = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, teacher_dim))

    def forward(self, fmap, summary):
        return self.spatial(fmap), self.summary(summary)


class DenseHeads(nn.Module):
    """Depth (relative first, D1), segmentation (D2 classes), gate corner heatmaps."""

    def __init__(self, in_dim: int = 128, n_classes: int = len(CLASSES)):
        super().__init__()

        def head(out):
            return nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1), nn.GELU(), nn.Conv2d(in_dim, out, 1))

        self.depth = head(1)
        self.seg = head(n_classes)
        self.gate_corners = head(4)  # 4 corner heatmaps
        # Slot head: K gates x [score logit + 4 corners (u,v)] in image-normalised coordinates, values may lie OUTSIDE [0,1]
        # (a close gate that is partly out of view still has all four corners). Same format as droneracing's det_features input.
        self.n_slots = 4
        g = int(os.environ.get("PFM_SLOT_GRID", "6"))      # 6 = original coarse head (old checkpoints); larger = finer grid with a conv before the MLP
        self.slot_pool = nn.AdaptiveAvgPool2d((g, g))
        if g == 6:
            self.slots = nn.Sequential(nn.Flatten(), nn.Linear(in_dim * 36, 512), nn.GELU(), nn.Linear(512, self.n_slots * 9))
        else:
            self.slots = nn.Sequential(nn.Conv2d(in_dim, 64, 3, padding=1), nn.GELU(), nn.Flatten(), nn.Linear(64 * g * g, 512), nn.GELU(), nn.Linear(512, self.n_slots * 9))

        # Dense gate head (PFM_GATE_HEAD=dense): per stride-8 cell a gate-centre logit + 8 corner offsets (image-normalised, relative to the cell centre).
        # Translation-equivariant, so it localises far better than the global MLP above; decoded into the same K x 9 slot format.
        self.gdet = None; self.gdet_levels = len([t for t in os.environ.get("PFM_GDET_LEVELS", "").split(",") if t]) + 1     # size levels: one 9-channel map per level, so nested gates of different size no longer fight for one cell
        if os.environ.get("PFM_GATE_HEAD", "") == "dense":
            layers = [nn.Conv2d(in_dim, in_dim, 3, padding=1), nn.GELU(), nn.Conv2d(in_dim, in_dim, 3, padding=1), nn.GELU(), nn.Conv2d(in_dim, 9 * self.gdet_levels, 1)]
            if os.environ.get("PFM_GDET_STRIDE", "8") == "4":     # finer grid: far gates seen through each other collide less often in one cell
                layers = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)] + layers
            elif os.environ.get("PFM_GDET_STRIDE", "8") == "2":   # stride 2: 224 x 224 output at 448 input (corner offsets regressed from a 4x finer grid)
                layers = [nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)] + layers
            self.gdet = nn.Sequential(*layers)
            nn.init.constant_(self.gdet[-1].bias[::9], -2.19)

    def decode_gdet(self, raw):
        """raw (B,9,H,W) -> (B,K,9): [score logit, 4 corners (u,v) image-normalised], best first. Peak NMS (PFM_GDET_NMS, default 3x3), then
        duplicate removal on the decoded corners (PFM_GDET_DEDUP, default 0.25; 0 = off): of the 4K best peaks, one whose mean corner distance to a
        better-scoring kept one is below 0.25 x the larger gate size is dropped (same gate found twice); a small gate seen through a large one survives."""
        raw = raw.float(); b, _, h, w = raw.shape; K = int(os.environ.get("PFM_GDET_K", self.n_slots))      # gates returned per frame (default 4); frames with more gates than K lose the weakest ones
        k = int(os.environ.get("PFM_GDET_NMS", "3")); dd = float(os.environ.get("PFM_GDET_DEDUP", "0.25")); L = raw.shape[1] // 9; rl = raw.view(b, L, 9, h, w)
        heat = rl[:, :, 0].sigmoid(); peak = (F.max_pool2d(heat, k, 1, k // 2) == heat)      # (B,L,H,W): peaks per size level
        M = min(4 * K if dd > 0 else K, L * h * w)
        score, idx = (heat * peak).flatten(1).topk(M, dim=1); pos = idx % (h * w)
        ii, jj = pos // w, pos % w
        cx, cy = (jj.float() + 0.5) / w, (ii.float() + 0.5) / h
        off = rl[:, :, 1:].permute(0, 1, 3, 4, 2).reshape(b, L * h * w, 8).gather(1, idx[..., None].expand(-1, -1, 8))      # (B,M,8)
        coords = off + torch.stack([cx, cy], -1).repeat(1, 1, 4)
        if dd > 0:
            c = coords.view(b, M, 4, 2); size = (c.max(2).values - c.min(2).values).max(-1).values                   # (B,M)
            dist = (c[:, :, None] - c[:, None]).norm(dim=-1).mean(-1)                                                    # (B,M,M) mean corner distance
            close = dist < dd * torch.maximum(size[:, :, None], size[:, None]); keep = torch.ones_like(score, dtype=torch.bool)
            for q in range(1, M):       # greedy, best first: dropped if close to a better one that is itself kept
                keep[:, q] = ~(close[:, q, :q] & keep[:, :q]).any(1)
            merged = None
            if L > 1 and os.environ.get("PFM_GDET_MERGE", "1") == "1":      # size levels: a gate near a size threshold is split over two maps; its two peaks are one gate, so their scores are combined (1 - product of misses) on the kept one
                lv = idx // (h * w); miss = 1 - score
                for q in range(1, M):
                    first = (close[:, q, :q] & keep[:, :q]).float(); first = first * (first.cumsum(1) == 1)      # the best kept gate this dropped peak duplicates
                    other = (lv[:, :q] != lv[:, q:q + 1]).float() * first * (~keep[:, q:q + 1]).float()
                    miss = torch.cat([miss[:, :q] * (1 - other * score[:, q:q + 1]), miss[:, q:]], 1)
                merged = (1 - miss).clamp(1e-6, 1 - 1e-6)
            score = (merged if merged is not None else score) * keep; score, o = score.topk(K, dim=1); idx = idx.gather(1, o); coords = coords.gather(1, o[..., None].expand(-1, -1, 8))
            if merged is not None: return torch.cat([torch.where(score > 0, torch.logit(score.clamp(1e-6, 1 - 1e-6)), torch.full_like(score, -20.0))[..., None], coords], -1)
        logit = torch.where(score > 0, rl[:, :, 0].flatten(1).gather(1, idx), torch.full_like(score, -20.0))
        return torch.cat([logit[..., None], coords], -1)

    def forward(self, pyramid, out_size):
        # Fuse pyramid at stride 8 (cheap FPN-lite).
        p4, p3, p2 = pyramid
        f = p4 + F.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False) \
               + F.interpolate(p2, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        up = lambda t: F.interpolate(t, size=out_size, mode="bilinear", align_corners=False)
        slots = self.slots(self.slot_pool(f)).view(-1, self.n_slots, 9)
        out = {"depth": up(self.depth(f)), "seg": up(self.seg(f)), "corners": up(self.gate_corners(f)), "slots": slots}
        if self.gdet is not None:
            raw = self.gdet(f); out["gdet"] = raw; out["slots_mlp"] = slots; out["slots"] = self.decode_gdet(raw)
        return out


class PerceptionModel(nn.Module):
    def __init__(self, backbone: str = "vit", teacher_dims: dict[str, int] | None = None,
                 pretrained: bool = False, hybrid_name: str | None = None):
        super().__init__()
        hybrid_name = hybrid_name or os.environ.get("PFM_HYBRID_NAME", "fastvit_sa24")      # Update 165: PFM_HYBRID_NAME=fastvit_sa12 = compatible student core
        self.backbone = Backbone(backbone, pretrained, hybrid_name=hybrid_name)
        d = self.backbone.dim
        self.teacher_heads = nn.ModuleDict({k: TeacherHead(d, v) for k, v in (teacher_dims or {}).items()})
        self.pyramid = SimpleFeaturePyramid(d)
        self.dense = DenseHeads()

    def add_teacher(self, name: str, teacher_dim: int):
        """Continual distillation hook (D1/GRAFT): new teacher = new readout head."""
        self.teacher_heads[name] = TeacherHead(self.backbone.dim, teacher_dim)

    def forward(self, x, with_teacher_heads: bool = True):
        fmap, summary = self.backbone(x)
        out = self.dense(self.pyramid(fmap), x.shape[-2:])
        if with_teacher_heads:
            out["teacher"] = {k: h(fmap, summary) for k, h in self.teacher_heads.items()}
        return out
