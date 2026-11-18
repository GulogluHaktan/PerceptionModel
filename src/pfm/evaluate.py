"""Golden-set evaluation: seg mIoU + depth AbsRel (scale-shift aligned, so relative depth is fine)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data import normalize


def align_ss(pred, gt, mask):
    """Least-squares scale+shift of pred onto gt over valid pixels (per image)."""
    out = torch.zeros_like(pred)
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() < 10:
            out[i] = pred[i]
            continue
        p, g = pred[i][m], gt[i][m]
        A = torch.stack([p, torch.ones_like(p)], 1)
        sol = torch.linalg.lstsq(A, g[:, None]).solution[:, 0]
        out[i] = pred[i] * sol[0] + sol[1]
    return out


@torch.no_grad()
def evaluate(model, golden, n_classes, device="cuda", bs=8):
    model.eval()
    inter = torch.zeros(n_classes)
    union = torch.zeros(n_classes)
    rels = []
    for x, seg, depth in DataLoader(golden, batch_size=bs):
        out = model(normalize(x.to(device)), with_teacher_heads=False)
        pred = out["seg"].argmax(1).cpu()
        valid = seg >= 0
        for c in range(n_classes):
            p, g = (pred == c) & valid, (seg == c) & valid
            inter[c] += (p & g).sum()
            union[c] += (p | g).sum()
        if depth.abs().sum() > 0:
            d = out["depth"].float().cpu()
            mask = depth > 1e-3
            a = align_ss(d, depth, mask)
            rels.append(((a - depth).abs() / depth.clamp(min=1e-3))[mask].mean().item())
    present = union > 0
    res = {"seg_miou": (inter[present] / union[present]).mean().item() if present.any() else 0.0}
    if rels:
        res["depth_absrel"] = sum(rels) / len(rels)
    return res


def match_gates(pred, tgt, thr=0.5, w=256.0, h=192.0):
    """Order-free matching of predicted gate slots to labelled gates. pred (K,9) [score logit, 8 coords], tgt (K,9) [valid, 8 coords] (image-normalised).
    Greedy one-to-one, labelled gates in their own order (largest first), each takes the nearest unused prediction with score > thr.
    Returns ([(tgt index, mean corner error in px or None if no prediction left)], number of unmatched predictions)."""
    import torch
    sc = pred[:, 0].sigmoid(); P = [q for q in range(pred.shape[0]) if sc[q] > thr]; used = set(); out = []; s = torch.tensor([w, h])
    for k in range(tgt.shape[0]):
        if tgt[k, 0] < 0.5: continue
        c = tgt[k, 1:].view(4, 2) * s; best = (None, None)
        for q in P:
            if q in used: continue
            e = ((pred[q, 1:].view(4, 2) * s - c).norm(dim=1)); e = e[torch.isfinite(e)].mean().item()
            if best[0] is None or e < best[0]: best = (e, q)
        if best[1] is not None: used.add(best[1])
        out.append((k, best[0]))
    return out, len(P) - len(used)


def absrel_both(pred, gt, valid):
    """AbsRel of a relative depth prediction against metric depth on the valid pixels (numpy arrays), after a per-image least-squares scale+shift fit done in two ways:
    'lin' = fit pred to depth (the protocol of all numbers up to Update 46), 'inv' = fit pred to 1/depth and invert (right for a network that outputs inverse depth, as the teacher does).
    Returns (lin, inv)."""
    import numpy as np
    d = pred[valid].astype(np.float64); g = gt[valid].astype(np.float64); A = np.stack([d, np.ones_like(d)], 1)
    s, b = np.linalg.lstsq(A, g, rcond=None)[0]; lin = float(np.mean(np.abs(np.clip(s * d + b, 1e-3, None) - g) / g))
    s, b = np.linalg.lstsq(A, 1.0 / g, rcond=None)[0]; inv = float(np.mean(np.abs(1.0 / np.clip(s * d + b, 1.0 / (10 * g.max()), None) - g) / g))
    return lin, inv
