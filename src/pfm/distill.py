"""Multi-teacher distillation losses working from CACHED teacher features.

Teachers are expensive; we run them once and store features on disk (cache.py), then
both students train from the cache. Losses follow AM-RADIO style: cosine + smooth-L1 on
spatial features, cosine on summary tokens. Per-teacher weights counter the
"teacher imbalance" trap (RADIOv2.5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def feat_loss(student_fmap, teacher_fmap):
    """Spatial feature loss. Student map is resized to the teacher grid."""
    if student_fmap.shape[-2:] != teacher_fmap.shape[-2:]:
        student_fmap = F.interpolate(student_fmap, size=teacher_fmap.shape[-2:], mode="bilinear",
                                     align_corners=False)
    cos = 1 - F.cosine_similarity(student_fmap, teacher_fmap, dim=1).mean()
    sl1 = F.smooth_l1_loss(student_fmap, teacher_fmap)
    return cos + sl1


def summary_loss(student_vec, teacher_vec):
    return 1 - F.cosine_similarity(student_vec, teacher_vec, dim=-1).mean()


def distill_loss(out_teacher: dict, batch_teacher: dict, weights: dict | None = None):
    """out_teacher: {name: (fmap, summary)} from the student.
    batch_teacher: {name: (fmap, summary)} from the cache. Returns (total, per_teacher)."""
    total, parts = 0.0, {}
    for name, (s_map, s_vec) in out_teacher.items():
        if name not in batch_teacher:
            continue
        t_map, t_vec = batch_teacher[name]
        l = feat_loss(s_map, t_map) + summary_loss(s_vec, t_vec)
        parts[name] = l.detach()
        total = total + (weights or {}).get(name, 1.0) * l
    return total, parts


def self_distill_loss(out_new: dict, out_old: dict):
    """D5: previous model as teacher against forgetting (dense head outputs)."""
    l = F.l1_loss(out_new["depth"], out_old["depth"])
    l = l + F.kl_div(F.log_softmax(out_new["seg"], 1), F.softmax(out_old["seg"], 1), reduction="batchmean")
    return l
