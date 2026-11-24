"""Gate-head decoding split for the device: the cheap tensor part stays in the graph, the tiny sequential part runs in numpy.

pfm.model.DenseHeads.decode_gdet does peak NMS + top-M on the GPU and then two greedy loops over the M = 4K candidates (duplicate removal,
size-level score merge). Exported to ONNX those loops unroll into ~hundreds of tiny serial kernels (84 % of the system engine's GPU time on the
5090 profile, Update 156) - on the Orin each costs a kernel launch on a slow CPU core. Here:
  gdet_candidates(raw, K)   (torch, in the graph)  -> (B, M, 11+M): score (heat * is_peak), level, raw logit at the peak, 8 corner coords (normalised), 'same gate' row
  gdet_select(cand, K, ...) (numpy, on the host)   -> (K, 9): logit + corners, identical to decode_gdet (tested: deploy/export_system.py check)."""
import numpy as np
import torch
import torch.nn.functional as F


def gdet_candidates(raw, K=8, nms=3, dedup=0.25):
    raw = raw.float(); b, _, h, w = raw.shape; L = raw.shape[1] // 9; rl = raw.view(b, L, 9, h, w)
    heat = rl[:, :, 0].sigmoid(); peak = (F.max_pool2d(heat, nms, 1, nms // 2) == heat)
    M = min(4 * K if dedup > 0 else K, L * h * w)
    score, idx = (heat * peak).flatten(1).topk(M, dim=1); pos = idx % (h * w); lv = idx // (h * w)
    ii, jj = pos // w, pos % w; cx, cy = (jj.float() + 0.5) / w, (ii.float() + 0.5) / h
    off = rl[:, :, 1:].permute(0, 1, 3, 4, 2).reshape(b, L * h * w, 8).gather(1, idx[..., None].expand(-1, -1, 8))
    coords = off + torch.stack([cx, cy], -1).repeat(1, 1, 4)
    logit = rl[:, :, 0].flatten(1).gather(1, idx)
    out = [score[..., None], lv.float()[..., None], logit[..., None], coords]
    if dedup > 0:          # pairwise "same gate" matrix (strictly lower: only better-ranked ones), cheap on the GPU, ~60 us in numpy
        c = coords.view(b, M, 4, 2); size = (c.max(2).values - c.min(2).values).max(-1).values
        dist = (c[:, :, None] - c[:, None]).norm(dim=-1).mean(-1); close = dist < dedup * torch.maximum(size[:, :, None], size[:, None])
        out.append((close & torch.ones(M, M, dtype=torch.bool, device=raw.device).tril(-1)).float())
    return torch.cat(out, -1)          # (B, M, 11 [+ M])


def gdet_select(c, K=8, levels=1, dedup=0.25, merge=True):
    """c (M, 11) numpy float32 for ONE image -> (K, 9) [logit (-20 = empty), 8 coords]."""
    c = np.asarray(c, np.float32)
    n = int((c[:, 0] > 0).sum())                          # zero-score candidates (not peaks) rank last: they cannot change keep / merge of a scored one
    if n < len(c):
        o = np.argsort(-c[:, 0], kind="stable")[:n]; c2 = c[o]
        if c.shape[1] > 11: c2 = np.concatenate([c2[:, :11], c2[:, 11:][:, o]], 1)
        out = gdet_select(c2, K, levels, dedup, merge) if n else np.zeros((0, 9), np.float32)
        return np.concatenate([out, np.concatenate([np.full((K - len(out), 1), -20.0, np.float32), np.zeros((K - len(out), 8), np.float32)], 1)]) if len(out) < K else out
    score = c[:, 0].copy(); lv = c[:, 1]; raw_logit = c[:, 2]; coords = c[:, 3:11]; M = len(c)
    if dedup <= 0:
        s = score; o = np.argsort(-s, kind="stable")[:K]
        return np.concatenate([np.where(s[o] > 0, raw_logit[o], -20.0)[:, None], coords[o]], 1).astype(np.float32)
    if c.shape[1] >= 11 + M: close = c[:, 11:11 + M] > 0.5                  # computed in the graph (gdet_candidates)
    else:
        cc = coords.reshape(M, 4, 2); size = (cc.max(1) - cc.min(1)).max(-1)
        close = (np.linalg.norm(cc[:, None] - cc[None], axis=-1).mean(-1) < dedup * np.maximum(size[:, None], size[None])) & np.tri(M, M, -1, dtype=bool)
    keep = np.ones(M, bool)                               # greedy "dropped if close to a better KEPT one" as a fixed point: close is strictly lower-
    for _ in range(M):                                    # triangular (a DAG), so Jacobi iteration from all-True converges to the greedy result
        k2 = ~(close & keep[None]).any(1)
        if (k2 == keep).all(): break
        keep = k2
    merged = None
    if levels > 1 and merge:                              # each dropped peak q of another size level lifts the FIRST kept gate it duplicates: miss_j *= 1 - s_q
        ck = close & keep[None]; has = ck.any(1); first = np.zeros((M, M), np.float32); first[np.nonzero(has)[0], ck[has].argmax(1)] = 1.0
        other = first * (lv[None] != lv[:, None]) * (~keep)[:, None]          # (q, j)
        miss = (1 - score) * np.prod(1 - other * score[:, None], axis=0)
        merged = np.clip(1 - miss, 1e-6, 1 - 1e-6).astype(np.float32)
    s = ((merged if merged is not None else score) * keep).astype(np.float32); o = np.argsort(-s, kind="stable")[:K]; s = s[o]
    if len(o) < K: o = np.concatenate([o, np.full(K - len(o), -1)]); s = np.concatenate([s, np.zeros(K - len(s), np.float32)]); coords = np.concatenate([coords, np.zeros((1, 8), np.float32)]); raw_logit = np.concatenate([raw_logit, [-20.0]]).astype(np.float32)
    if merged is not None:
        sc = np.clip(s, 1e-6, 1 - 1e-6); lg = np.where(s > 0, np.log(sc / (1 - sc)), -20.0)
    else:
        lg = np.where(s > 0, raw_logit[o], -20.0)
    return np.concatenate([lg[:, None], coords[o]], 1).astype(np.float32)
