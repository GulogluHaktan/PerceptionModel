"""Teacher registry. A teacher maps normalized images (B,3,H,W) -> (spatial map, summary).
Depth teachers return a (B,1,H,W) relative-depth map as the spatial map and zeros as summary.

kind:
  fake    - frozen random conv net (pipeline tests, no download)
  dinov2  - timm DINOv2 ViT-S/14 (spatial semantics teacher)      [needs weights download]
  depth_da2 - Depth Anything V2 via HF transformers (depth teacher) [needs transformers + download]
Add SigLIP/SAM2/Grounding-DINO teachers the same way on the 5090 (D1/D2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeTeacher(nn.Module):
    def __init__(self, dim=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(3, dim, 14, 14)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * 0.1)
        self.dim = dim

    def forward(self, x):
        f = self.conv(x)
        return f, f.mean((2, 3))


class DinoV2(nn.Module):
    def __init__(self, name="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True, global_pool="")
        self.dim = self.net.embed_dim

    def forward(self, x):
        t = self.net.forward_features(x)
        b, _, c = t.shape
        h = x.shape[-2] // 14
        w = x.shape[-1] // 14
        n = self.net.num_prefix_tokens
        return t[:, n:].transpose(1, 2).reshape(b, c, h, w), t[:, 0]


class DepthAnythingV2(nn.Module):
    def __init__(self, name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.net = AutoModelForDepthEstimation.from_pretrained(name)
        self.dim = 1

    def forward(self, x):
        d = self.net(pixel_values=x).predicted_depth[:, None]
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return d, torch.zeros(x.shape[0], 1, device=x.device)


class ERadio(nn.Module):
    """NVIDIA E-RADIO (390M params, ~31 ms @448 on a laptop GPU: a TEACHER, not an Orin student).
    Needs transformers<5 (remote code) + einops + open_clip_torch. Spatial map is 1536-d: big cache
    (about 2.4 MB/img at 448 px in fp16) - consider size 224 for this teacher.
    Input: [0,1] images; its own input conditioner is applied inside, so we undo ImageNet norm outside."""

    def __init__(self, name="nvidia/E-RADIO"):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.dim = 1536

    def forward(self, x):
        from .data import MEAN, STD
        x01 = x * STD.to(x.device) + MEAN.to(x.device)
        summary, feats = self.net(x01)          # feats: (B, N, C) tokens, stride 16
        b, n, c = feats.shape
        h = x.shape[-2] // 16
        return feats.transpose(1, 2).reshape(b, c, h, n // h), summary


REGISTRY = {"fake": FakeTeacher, "dinov2": DinoV2, "depth_da2": DepthAnythingV2, "eradio": ERadio}
DEPTH_TEACHERS = {"depth_da2", "fake_depth"}


def build_teacher(kind, device="cuda"):
    if kind == "fake_depth":
        class FD(FakeTeacher):
            def forward(self, x):
                f, _ = super().forward(x)
                d = F.interpolate(f.mean(1, keepdim=True), size=x.shape[-2:], mode="bilinear")
                return d, torch.zeros(x.shape[0], 1, device=x.device)
        t = FD(dim=8, seed=1)
    else:
        t = REGISTRY[kind]()
    return t.eval().to(device).requires_grad_(False)
