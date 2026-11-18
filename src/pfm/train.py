"""Distillation training from the teacher feature cache (D1). Resumable, bf16, grad clipping."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .cache import FeatureCache
from .data import ImageFolder, normalize
from .distill import distill_loss
from .evaluate import evaluate
from .gate import GateConfig, decide, log_provenance
from .model import CLASSES, PerceptionModel


def corner_heatmap(corners, size, sigma):
    """corners (N,4,2) normalised -> (4,size,size) gaussian heatmaps (tl,tr,br,bl)."""
    ys, xs = np.mgrid[0:size, 0:size]
    hm = np.zeros((4, size, size), np.float32)
    for g in corners:
        for k in range(4):
            cx, cy = g[k, 0] * size, g[k, 1] * size
            hm[k] = np.maximum(hm[k], np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
    return torch.from_numpy(hm)


def _clip_len(a, b, w, h):
    """Length of the part of segment a-b that lies inside the rectangle [0,w) x [0,h) (Liang-Barsky), 0 if none."""
    d = b - a; t0, t1 = 0.0, 1.0
    for p_, q_ in ((-d[0], a[0]), (d[0], w - a[0]), (-d[1], a[1]), (d[1], h - a[1])):
        if p_ == 0:
            if q_ < 0: return 0.0
        else:
            t = q_ / p_
            if p_ < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
    return float(max(t1 - t0, 0.0) * np.linalg.norm(d))


def frame_in_view(c, w, h):
    """Longest piece of the gate outline (4 known corners, any of them may be outside) that lies inside the image, in pixels."""
    return max(_clip_len(c[i], c[(i + 1) % 4], w, h) for i in range(4))


def half_gate_crop(c, n_in, rng=random, tries=30, s_range=(1.6, 4.0)):
    """Zoom-in crop (s, x0, y0) of the unit image that leaves exactly n_in (0 or 1) corners of the gate c (4,2 image-normalised, all known) in view while a piece of the
    outline at least 0.3 crop widths long stays visible; None if no such crop is found. The crop never leaves the image. Used by the training augmentation and by scripts/eval_half_gates.py."""
    for _ in range(tries):
        s = rng.uniform(*s_range); k = rng.randrange(4); e = 1 / s
        if n_in == 1:       # put corner k somewhere in the crop
            x0 = c[k, 0] - rng.uniform(0.1, 0.9) * e; y0 = c[k, 1] - rng.uniform(0.1, 0.9) * e
        else:               # put a point of edge k -> k+1 somewhere in the crop
            pt = c[k] + rng.uniform(0.2, 0.8) * (c[(k + 1) % 4] - c[k]); x0 = pt[0] - rng.uniform(0.2, 0.8) * e; y0 = pt[1] - rng.uniform(0.2, 0.8) * e
        if not (0 <= x0 <= 1 - e and 0 <= y0 <= 1 - e): continue
        q = (c - np.array([x0, y0], np.float32)) * s; ins = ((q[:, 0] >= 0) & (q[:, 0] < 1) & (q[:, 1] >= 0) & (q[:, 1] < 1)).sum()
        if ins == n_in and frame_in_view(q, 1.0, 1.0) >= 0.3: return float(s), float(x0), float(y0)
    return None


def gate_list(rows, w=256.0, h=192.0):
    """corners_full rows -> list of (size, corners (4,2) image-normalised, NaN = hidden corner without a label), biggest first.
    A gate counts if it is in front of the camera, has at least 2 known corners and at least one known corner inside the image; a gate with all 4 corners known and none of them
    inside also counts when a piece of its outline at least 0.15 image widths long crosses the image (a bar of a close gate: the 'half gate' case)."""
    out = []
    for r in rows:
        c = np.array([[np.nan, np.nan] if p is None else p for p in r["corners_px"]], np.float32).reshape(4, 2); known = ~np.isnan(c[:, 0])
        if known.sum() < 2 or (np.array(r.get("depths", [1] * 4)) < 0.3).any() or (np.abs(c[known]) > 4 * max(w, h)).any():
            continue
        ck = c[known]
        if not ((ck[:, 0] >= 0) & (ck[:, 0] < w) & (ck[:, 1] >= 0) & (ck[:, 1] < h)).any() and not (known.all() and frame_in_view(c, w, h) >= 0.15 * w):
            continue
        out.append(((ck.max(0) - ck.min(0)).max(), c / np.array([w, h], np.float32)))
    out.sort(key=lambda t: -t[0])
    return out


def slot_targets(rows, k=4, w=256.0, h=192.0):
    """corners_full rows (race_to_training: corners_px at 256x192, depths, ...) -> (k,9) [valid, 8 coords normalised, may be outside 0..1], nearest (largest) first.
    A corner may be None in the row (real labels: hidden corner, no coordinate): it becomes NaN and slot_loss skips it. At least 2 known corners are needed."""
    t = np.zeros((k, 9), np.float32)
    for i, (_, c) in enumerate(gate_list(rows, w, h)[:k]):
        t[i, 0] = 1; t[i, 1:] = c.reshape(-1)
    return torch.from_numpy(t)


def complete_square(c, w=256.0, h=192.0):
    """Guess the hidden corners of a gate (4,2 image-normalised, NaN = hidden; order TL TR BR BL, clockwise in the image) so that its centre can be placed like in the sim labels, where all 4 corners are known:
    3 known -> parallelogram; 2 neighbouring corners -> a square built on that side (in pixels), on the side the corner order says; otherwise unchanged. Only used for the centre and the size, never as a regression target."""
    c = c.copy(); k = ~np.isnan(c[:, 0]); px = np.array([w, h], np.float32)
    if k.sum() == 3:
        i = int(np.where(~k)[0][0]); c[i] = c[(i - 1) % 4] + c[(i + 1) % 4] - c[(i + 2) % 4]
    elif k.sum() == 2:
        for i in range(4):
            if k[i] and k[(i + 1) % 4]:
                v = (c[(i + 1) % 4] - c[i]) * px; r = np.array([-v[1], v[0]], np.float32) / px; c[(i + 2) % 4] = c[(i + 1) % 4] + r; c[(i + 3) % 4] = c[i] + r; break
    return c


def gdet_targets(rows, g, w=256.0, h=192.0, complete=False, levels=()):
    """Dense gate-head targets on a g x g grid: heat (g,g) with a gaussian per gate centre (peak cell exactly 1) and off (8,g,g) = corner - cell centre
    (image-normalised) in the 3x3 cells around each centre, NaN elsewhere and for hidden corners. Centre = mean of the known corners, clamped into the image."""
    if levels:      # size levels (thresholds in px of the larger side at 256 width): one target pair per level, a gate within 20% of a threshold goes to both neighbours
        hs, os_ = [], []
        for lv in range(len(levels) + 1):
            lo = levels[lv - 1] / 1.2 if lv > 0 else 0.0; hi = levels[lv] * 1.2 if lv < len(levels) else 1e9
            sel = [(sz, c) for sz, c in gate_list(rows, w, h) if lo <= _gate_ext_px(c, w, h, complete) < hi]
            a, b_ = _gdet_targets_one(sel, g, w, h, complete); hs.append(a); os_.append(b_)
        return torch.from_numpy(np.stack(hs)), torch.from_numpy(np.concatenate(os_))
    return tuple(torch.from_numpy(a) for a in _gdet_targets_one(gate_list(rows, w, h), g, w, h, complete))


def _gate_ext_px(c, w, h, complete):
    cc_ = complete_square(c, w, h) if complete else c; kk = ~np.isnan(cc_[:, 0]); return float(((cc_[kk].max(0) - cc_[kk].min(0)) * np.array([256.0, 192.0])).max())      # larger side in px at 256x192


def _gdet_targets_one(gates, g, w, h, complete):
    heat = np.zeros((g, g), np.float32); off = np.full((8, g, g), np.nan, np.float32); owned = np.zeros((g, g), bool); ys, xs = np.mgrid[0:g, 0:g]
    for size, c in gates:
        known = ~np.isnan(c[:, 0]); cc_ = complete_square(c, w, h) if complete else c; kk = ~np.isnan(cc_[:, 0])     # complete=True: centre of the guessed full square (same convention as the sim labels) instead of the mean of the visible corners
        ctr = np.clip(cc_[kk].mean(0), 0, 1 - 1e-4); cj, ci = int(ctr[0] * g), int(ctr[1] * g)
        ext = (cc_[kk].max(0) - cc_[kk].min(0)).max() * g; sig = max(1.0, min(ext, 2.0 * g) / 6)
        heat = np.maximum(heat, np.exp(-((xs - cj) ** 2 + (ys - ci) ** 2) / (2 * sig ** 2))); heat[ci, cj] = 1
        for i in range(max(ci - 1, 0), min(ci + 2, g)):
            for j in range(max(cj - 1, 0), min(cj + 2, g)):
                if owned[i, j]: continue
                owned[i, j] = True; off[:, i, j] = (c - np.array([(j + 0.5) / g, (i + 0.5) / g], np.float32)).reshape(-1)
    return heat.astype(np.float32), off


def gdet_loss(raw, heat, off, off_weight=10.0):
    """CenterNet focal loss on the centre heatmap + L1 on the corner offsets where a target exists (finite)."""
    if raw.shape[1] > 9: L = raw.shape[1] // 9; raw = raw.reshape(-1, 9, *raw.shape[2:]); heat = heat.reshape(-1, *heat.shape[2:]); off = off.reshape(-1, 8, *off.shape[2:])      # size levels: each level is its own map
    p = raw[:, 0].sigmoid().clamp(1e-4, 1 - 1e-4); pos = heat >= 0.999
    l_pos = ((1 - p) ** 2 * p.log())[pos].sum(); l_neg = ((1 - heat) ** 4 * p ** 2 * (1 - p).log())[~pos].sum()
    focal = -(l_pos + l_neg) / pos.sum().clamp(min=1)
    d = (raw[:, 1:] - off).abs(); d = d[torch.isfinite(d)]
    return focal + off_weight * (d.mean() if d.numel() else raw.sum() * 0)


def slot_loss(pred, tgt, coord_weight=5.0):
    """pred (B,K,9) logits+coords, tgt (B,K,9). BCE on the score of every slot, L1 on the coords of the slots that hold a gate."""
    v = tgt[..., 0]
    bce = F.binary_cross_entropy_with_logits(pred[..., 0], v)
    m = v > 0.5
    d = (pred[..., 1:][m] - tgt[..., 1:][m]).abs(); d = d[torch.isfinite(d)]     # NaN target = hidden corner without a label
    l1 = d.mean() if d.numel() else pred.sum() * 0
    return bce + coord_weight * l1


def not_gate_loss(seg_logits, mask, gate_idx=CLASSES.index("gate")):
    """-log(1 - P(class)) on the pixels of `mask` (B,H,W bool): pushes one class (default gate) down where a pixel is known not to be it, without saying which class it is."""
    p = seg_logits.softmax(1)[:, gate_idx][mask]
    return -(1 - p).clamp(min=1e-6).log().mean()


def _warp(t, theta, mode):
    """t (C,H,W) float -> same size, sampled through the affine theta (2,3) (output -> input, normalised coords)."""
    if theta.shape[0] == 3:      # full 3x3 projective map (side-view augmentation): identity grid pushed through the matrix
        g0 = F.affine_grid(torch.eye(2, 3)[None], (1, t.shape[0], t.shape[1], t.shape[2]), align_corners=False)
        q = torch.cat([g0, torch.ones_like(g0[..., :1])], -1) @ theta.T; grid = q[..., :2] / q[..., 2:]
    else: grid = F.affine_grid(theta[None], (1, t.shape[0], t.shape[1], t.shape[2]), align_corners=False)
    return F.grid_sample(t[None], grid, mode=mode, padding_mode="border", align_corners=False)[0]


def close_gate_indices(images, labels, px=100.0, prefixes=("real_",)):
    """Indices of frames (id starts with one of `prefixes`) that hold a gate whose in-view corners (>= 2) span at least `px` (256x192 scale): the last moments before passing a gate."""
    out = []
    for i, p in enumerate(images.files):
        sid = images.sid(p)
        if not sid.startswith(tuple(prefixes)): continue
        cf = Path(labels) / "corners_full" / f"{sid}.json"
        if not cf.exists(): continue
        for r in json.load(open(cf)):
            c = np.array([q for q in r["corners_px"] if q is not None and 0 <= q[0] < 256 and 0 <= q[1] < 192], np.float32)
            if len(c) >= 2 and float(np.ptp(c, axis=0).max()) >= px: out.append(i); break
    return out


class CachedDS(Dataset):
    """Image + cached teacher features (+ pseudo labels from labeler.py when `labels` is given).
    Unlabelled pixels are 255 (ignored). Images without a label file are skipped for seg/corner loss.
    aug=True: random zoom-in crop (1-1.6x, never leaves the image), horizontal flip (gate corner order swapped) and photometric jitter; labels, exact depth and the
    cached teacher maps are warped with the same crop (teacher maps only approximately: their summary vector is kept).
    neg_prefixes: ids starting with one of these and without a corners_full file are known gate-free images (general indoor/outdoor sets) -> empty gate targets."""

    def __init__(self, images: ImageFolder, cache: FeatureCache, teachers: list[dict], labels=None, aug=False, grid=56, neg_prefixes=()):
        self.images, self.cache, self.teachers = images, cache, teachers
        self.labels = Path(labels) if labels else None
        self.aug, self.grid, self.neg = aug, grid, tuple(neg_prefixes)
        self.centre_complete = False    # config key gate_centre_complete: place the centre of partly labelled gates at the centre of the guessed full square
        self.half_zero = 0.5    # share of the half-gate crops that leave no corner in view (the rest leave one); config key half_gate_zero_share
        self.levels = ()        # size thresholds of the dense gate head levels (config key gdet_levels)
        self.side_view = 0.0    # share of gate frames given a side-view warp; config key side_view_prob
        self.zoom_out = 0.0     # share of gate frames that are shrunk instead of cropped; config key zoom_out_prob
        self.half_prob = 0.0    # share of gate frames that get a 'half gate' crop (0-1 corners of one gate left in view); set from the config key half_gate_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        sid, x = self.images[i]
        tf = {t["name"]: self.cache.get(t["name"], sid) for t in self.teachers}
        size = x.shape[-1]
        s, x0, y0, flip = 1.0, 0.0, 0.0, False
        if self.aug:
            if random.random() < 0.6:
                s = random.uniform(1.0, 1.6); x0 = random.uniform(0, 1 - 1 / s); y0 = random.uniform(0, 1 - 1 / s)
            flip = random.random() < 0.5
        cf = self.labels / "corners_full" / f"{sid}.json" if self.labels is not None else None
        if self.aug and self.zoom_out > 0 and cf is not None and cf.exists() and random.random() < self.zoom_out:     # shrink a gate frame (0.5-1x, edge pixels repeated around it): more small gates from real footage
            s = random.uniform(0.5, 1.0); x0 = random.uniform(1 - 1 / s, 0); y0 = random.uniform(1 - 1 / s, 0)
        if self.aug and self.half_prob > 0 and random.random() < self.half_prob and cf.exists():
            full = [np.array(r["corners_px"], np.float32) / np.array([256.0, 192.0], np.float32) for r in json.load(open(cf))
                    if all(p is not None for p in r["corners_px"]) and not (np.array(r.get("depths", [1] * 4)) < 0.3).any()]
            full = [c for c in full if ((c >= 0) & (c < 1)).all()]      # start from gates that are fully in view
            if full:
                hc = half_gate_crop(random.choice(full), 0 if random.random() < self.half_zero else 1)
                if hc is not None: s, x0, y0 = hc
        geo = s != 1.0 or flip
        theta = torch.tensor([[(-1.0 if flip else 1.0) / s, 0.0, 2 * x0 + 1 / s - 1], [0.0, 1 / s, 2 * y0 + 1 / s - 1]])
        Minv = None
        if self.aug and self.side_view > 0 and s == 1.0 and cf is not None and cf.exists() and random.random() < self.side_view:
            # side view: the picture is squeezed sideways (up to 2x) and given a keystone, as if the gates were seen from an angle. A zoom-in keeps the sampled area inside the picture.
            k = random.uniform(0.15, 0.4) * random.choice((-1, 1)); zs = random.uniform(1 / (1 - abs(k)), 2.2); a = random.uniform(1.0, min(2.0, zs * (1 - abs(k))))
            A = torch.tensor([[(-1.0 if flip else 1.0) * a / zs, 0.0, 0.0], [0.0, 1 / zs, 0.0], [0.0, 0.0, 1.0]]); theta = A @ torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [k, 0.0, 1.0]])
            Minv = np.linalg.inv(theta.numpy().astype(np.float64)); geo = True

        def tc(c):      # corners (...,4,2) image-normalised (NaN kept) -> after crop/flip, order TL TR BR BL restored
            c = np.array(c, np.float32).copy()
            if Minv is not None:      # projective: picture point -> point in the warped picture (inverse of the sampling map); the flip is inside the matrix, only the corner order is restored here
                q = np.concatenate([2 * c - 1, np.ones_like(c[..., :1])], -1) @ Minv.T; c = np.where(q[..., 2:] > 0.05, ((q[..., :2] / np.where(q[..., 2:] > 0.05, q[..., 2:], 1.0)) + 1) / 2, np.nan)      # a corner far outside, behind the horizon of the warp, becomes unknown
                return c[..., [1, 0, 3, 2], :].astype(np.float32) if flip else c.astype(np.float32)
            c[..., 0] = (c[..., 0] - x0) * s; c[..., 1] = (c[..., 1] - y0) * s
            if flip:
                c[..., 0] = 1 - c[..., 0]; c = c[..., [1, 0, 3, 2], :]
            return c

        if geo:
            x = _warp(x, theta, "bilinear")
            tf = {k: ((_warp(a, theta, "bilinear") if a.dim() == 3 else a), b) for k, (a, b) in tf.items()}
        seg, hm, has, has_c = torch.full((size, size), 255, dtype=torch.long), torch.zeros(4, size, size), torch.tensor(False), torch.tensor(False)
        depth, has_d = torch.zeros(size, size), torch.tensor(False)
        slots, has_s = torch.zeros(4, 9), torch.tensor(False)
        L = len(self.levels) + 1; g_heat, g_off = (torch.zeros(L, self.grid, self.grid), torch.full((8 * L, self.grid, self.grid), float("nan"))) if self.levels else (torch.zeros(self.grid, self.grid), torch.full((8, self.grid, self.grid), float("nan")))
        rows = None
        if self.labels is not None and (self.labels / "corners_full" / f"{sid}.json").exists():    # all gate corners incl. outside the image (racing sets)
            rows = json.load(open(self.labels / "corners_full" / f"{sid}.json"))
            if geo:
                for r in rows:
                    c = np.array([[np.nan, np.nan] if p is None else p for p in r["corners_px"]], np.float32) / np.array([256.0, 192.0], np.float32)
                    c = tc(c) * np.array([256.0, 192.0], np.float32); r["corners_px"] = [None if np.isnan(p[0]) else [float(p[0]), float(p[1])] for p in c]
        elif sid.startswith(self.neg):
            rows = []
        if rows is not None:
            slots = slot_targets(rows); has_s = torch.tensor(True); g_heat, g_off = gdet_targets(rows, self.grid, complete=self.centre_complete, levels=self.levels)
        if self.labels is not None and (self.labels / "depth" / f"{sid}.npy").exists():     # exact depth from the data factory (metres, 0 = no surface)
            dm = np.load(self.labels / "depth" / f"{sid}.npy").astype(np.float32)
            depth = F.interpolate(torch.from_numpy(dm)[None, None], size=(size, size), mode="nearest")[0, 0]; has_d = torch.tensor(True)
            if geo: depth = _warp(depth[None], theta, "nearest")[0]
        if self.labels is not None and (self.labels / "seg" / f"{sid}.png").exists():
            im = Image.open(self.labels / "seg" / f"{sid}.png")
            if im.size != (size, size):   # labels made at another resolution: resize, never crash mid-run
                im = im.resize((size, size), Image.NEAREST)
            seg = torch.from_numpy(np.asarray(im).copy()).long()
            if geo: seg = _warp(seg[None].float(), theta, "nearest")[0].long()
            cp = self.labels / "corners" / f"{sid}.npy"
            if cp.exists():
                cn = np.load(cp)
                hm = corner_heatmap(tc(cn) if geo and len(cn) else cn, size, sigma=max(2.0, size / 112)); has_c = torch.tensor(True)     # no corner file = corners UNKNOWN for this image (no corner loss)
            has = torch.tensor(True)
        if self.aug and random.random() < 0.8:      # photometric: exposure, gamma, colour cast, noise, light blur
            x = (x * random.uniform(0.6, 1.4)).clamp(0, 1) ** random.uniform(0.7, 1.5)
            x = x * torch.tensor([random.uniform(0.9, 1.1) for _ in range(3)]).view(3, 1, 1) + torch.randn_like(x) * random.uniform(0, 0.03)
            if random.random() < 0.2: x = F.avg_pool2d(x[None], 3, 1, 1)[0]
            x = x.clamp(0, 1)
        return x, tf, seg, hm, has, has_c, depth, has_d, slots, has_s, g_heat, g_off


def gate_cfg_from(cfg):
    """Gate floors come from the config (`gate:` key) so real numbers can be set after the first runs."""
    g = cfg.get("gate", {})
    return GateConfig(**({"floors": g["floors"]} if "floors" in g else {}), **({"max_regress": g["max_regress"]} if "max_regress" in g else {}),
                      **({"max_total_regress": g["max_total_regress"]} if "max_total_regress" in g else {}))


def ssi_depth_loss(pred, tgt):
    """Scale-shift invariant L1 vs. teacher relative depth (median/MAD normalised)."""
    def norm(d):
        f = d.flatten(1)
        med = f.median(1).values[:, None]
        mad = (f - med).abs().mean(1, keepdim=True).clamp(min=1e-6)
        return ((f - med) / mad)
    return F.l1_loss(norm(pred), norm(tgt))


def masked_ssi_depth_loss(pred, tgt):
    """Scale-shift invariant L1 against exact depth on the pixels that have a surface (tgt > 0): per image, both sides normalised by their masked median and mean absolute deviation."""
    tot, n = 0.0, 0
    for p, t in zip(pred, tgt):
        m = t > 0
        if int(m.sum()) < 100:
            continue
        a, b = p[m], t[m]
        na = (a - a.median()) / (a - a.median()).abs().mean().clamp(min=1e-6); nb = (b - b.median()) / (b - b.median()).abs().mean().clamp(min=1e-6)
        tot = tot + F.l1_loss(na, nb); n += 1
    return tot / max(n, 1)


def run(cfg_path):
    """cfg_path: yaml path or an already-loaded dict. Returns golden metrics (dict, empty without golden)."""
    cfg = cfg_path if isinstance(cfg_path, dict) else yaml.safe_load(open(cfg_path))
    out_dir = Path(cfg["out_dir"]) / cfg["student"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0)); random.seed(cfg.get("seed", 0))

    images = ImageFolder(cfg["images"], cfg["size"])
    cache = FeatureCache(cfg["cache"])
    cache.check_size(cfg["size"])
    feat_teachers = [t for t in cfg["teachers"] if not t["kind"].endswith("depth") and t["kind"] != "depth_da2"]
    depth_teacher = next((t for t in cfg["teachers"] if t not in feat_teachers), None)
    dims = {}
    for t in feat_teachers:
        fmap, _ = cache.get(t["name"], images[0][0])
        dims[t["name"]] = fmap.shape[0]
    os.environ["PFM_SLOT_GRID"] = str(cfg.get("slot_grid", 6)); os.environ["PFM_GATE_HEAD"] = str(cfg.get("gate_head", "")); os.environ["PFM_GDET_STRIDE"] = str(cfg.get("gdet_stride", 8)); os.environ["PFM_GDET_LEVELS"] = ",".join(str(t) for t in cfg.get("gdet_levels", ()))
    model = PerceptionModel(cfg["student"], teacher_dims=dims, pretrained=cfg.get("pretrained", False),
                            hybrid_name=cfg.get("hybrid_name", "fastvit_sa24")).to(dev)

    cfg["batch"] = min(cfg["batch"], len(images))  # tiny datasets: avoid zero steps with drop_last
    all_t = feat_teachers + ([depth_teacher] if depth_teacher else [])
    ds = CachedDS(images, cache, all_t, cfg.get("labels"), aug=cfg.get("aug", False), grid=cfg["size"] // cfg.get("gdet_stride", 8), neg_prefixes=cfg.get("gate_free_prefixes", ())); ds.half_prob = float(cfg.get("half_gate_prob", 0.0)); ds.side_view = float(cfg.get("side_view_prob", 0.0)); ds.levels = tuple(float(t) for t in cfg.get("gdet_levels", ())); ds.zoom_out = float(cfg.get("zoom_out_prob", 0.0)); ds.half_zero = float(cfg.get("half_gate_zero_share", 0.5)); ds.centre_complete = bool(cfg.get("gate_centre_complete", False))
    if int(cfg.get("close_gate_boost", 1)) > 1 and cfg.get("labels"):      # real frames with a gate about to be passed are about 1% of the set: show them close_gate_boost times per epoch
        ci = close_gate_indices(images, cfg["labels"], float(cfg.get("close_gate_px", 100))); print(f"close-gate frames: {len(ci)} x {int(cfg['close_gate_boost'])}", flush=True)
        ds = torch.utils.data.ConcatDataset([ds] + [torch.utils.data.Subset(ds, ci)] * (int(cfg["close_gate_boost"]) - 1))
    dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=True,
                    num_workers=cfg.get("workers", 4), drop_last=True, persistent_workers=cfg.get("workers", 4) > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.05)
    steps = cfg["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.05)
    weights = {t["name"]: t.get("weight", 1.0) for t in feat_teachers}

    seg_w = None
    if cfg.get("seg_class_weights"):       # {class name: weight}, others 1: lifts rare classes (person, car, lane_marking, opponent_drone)
        seg_w = torch.ones(len(CLASSES), device=dev)
        for k, v in cfg["seg_class_weights"].items(): seg_w[CLASSES.index(k)] = float(v)
    start = 0
    ck = out_dir / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); start = s["epoch"] + 1
        print(f"resumed from epoch {start}")

    golden = None
    if cfg.get("golden"):
        from .data import GoldenSet
        golden = GoldenSet(cfg["golden"], cfg["size"])

    for ep in range(start, cfg["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0
        for x, tf, seg, hm, has, has_c, depth_gt, has_d, slots_t, has_s, g_heat, g_off in dl:
            x = normalize(x.to(dev))
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(x)
            batch_t = {k: (v[0].to(dev), v[1].to(dev)) for k, v in tf.items() if k in dims}
            loss, parts = distill_loss({k: (a.float(), b.float()) for k, (a, b) in out["teacher"].items()}, batch_t, weights)
            if depth_teacher is not None:
                loss = loss + cfg.get("depth_weight", 1.0) * ssi_depth_loss(out["depth"].float(), tf[depth_teacher["name"]][0].to(dev))
            if has.any():
                hseg = has.to(dev)
                seg_d = seg.to(dev)[hseg]; hc = has_c.to(dev)[hseg]; hm_d = hm.to(dev)[hseg]
                seg_o = out["seg"].float()[hseg]; ng = seg_d == 254       # 254 = surely not gate, class otherwise unknown (weak labels of real gate frames, pfm.gate_seg)
                nl = seg_d == 253                                         # 253 = surely not lane_marking (real lane frames with line labels only, scripts/tusimple_prepare.py)
                seg_t = seg_d.masked_fill(ng | nl, 255)
                if (seg_t != 255).any():
                    loss = loss + cfg.get("seg_weight", 1.0) * F.cross_entropy(seg_o, seg_t, weight=seg_w, ignore_index=255)
                if ng.any():
                    loss = loss + cfg.get("notgate_weight", 1.0) * not_gate_loss(seg_o, ng)
                if nl.any():
                    loss = loss + cfg.get("notlane_weight", 1.0) * not_gate_loss(seg_o, nl, CLASSES.index("lane_marking"))
                pos = hm_d > 0.5   # corner heatmaps are sparse: up-weight positives
                cl = F.mse_loss(out["corners"].float()[hseg].sigmoid(), hm_d, reduction="none")
                if hc.any():
                    loss = loss + cfg.get("corner_weight", 5.0) * (cl[hc] * (1 + 20 * pos[hc])).mean()
            if cfg.get("depth_gt_weight", 0.0) > 0 and has_d.any():
                hd = has_d.to(dev); dg = depth_gt.to(dev)[hd][:, None]
                if cfg.get("depth_gt_space", "metric") == "inverse":      # the teacher (Depth Anything v2) gives INVERSE depth (near = large): put the exact depth into the same space,
                    dg = torch.where(dg > 0, 1.0 / dg.clamp(min=0.1), dg)   # otherwise the two depth losses pull the one output in opposite directions (Update 47). 0 stays "no surface".
                loss = loss + cfg["depth_gt_weight"] * masked_ssi_depth_loss(out["depth"].float()[hd], dg)
            if cfg.get("slot_weight", 1.0) > 0 and has_s.any():
                hs = has_s.to(dev); loss = loss + cfg.get("slot_weight", 1.0) * slot_loss(out.get("slots_mlp", out["slots"]).float()[hs], slots_t.to(dev)[hs])
                if "gdet" in out:
                    loss = loss + cfg.get("gdet_weight", 1.0) * gdet_loss(out["gdet"].float()[hs], g_heat.to(dev)[hs], g_off.to(dev)[hs])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        print(json.dumps({"epoch": ep, "loss": round(tot / len(dl), 4), "sec": round(time.time() - t0, 1),
                          "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else 0}), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, ck)

    torch.save(model.state_dict(), out_dir / "final.pt")
    metrics = evaluate(model, golden, len(CLASSES), dev) if golden else {}
    verdict = decide(metrics, None, gate_cfg_from(cfg)) if metrics else {}
    log_provenance(out_dir / "provenance.jsonl", {"student": cfg["student"], "config": cfg, "metrics": metrics, "gate": verdict})
    print("golden:", metrics, verdict)
    return metrics
