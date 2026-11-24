"""Corner error of the main model's gates before / after the corner refiner, on real golden frames (never used to train the refiner).
Matching as in eval_real_golden.py (visible corners, best cyclic order, 256x192 scale); only gates matched within 20 px before refinement
are compared, so the numbers isolate precision (same gates on both sides). Also reports native-pixel errors.
Usage: CKPT=runs/release_v4/bigY_h_hybrid.pt REF=runs/refiner_v1/refiner.pt [GOLDEN=data/real_mrg/golden FLIGHTS=red_s1_f1,...] python scripts/eval_refiner.py"""
import json, os, sys
import numpy as np, torch
from PIL import Image
for k, v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(k, v)
sys.path.insert(0, "src")
from pfm.model import PerceptionModel   # noqa: E402
from pfm.data import normalize          # noqa: E402
from pfm.refine import CornerRefiner, refine_gates   # noqa: E402
dev = "cuda"; CK = os.environ.get("CKPT", "runs/release_v4/bigY_h_hybrid.pt"); REF = os.environ.get("REF", "runs/refiner_v1/refiner.pt")
G = os.environ.get("GOLDEN", "data/real_mrg/golden"); FL = [x for x in os.environ.get("FLIGHTS", "red_s1_f1,red_s1_f2,red_s1_f3").split(",") if x]
cands = [c for c in json.load(open(f"{G}/candidates.json")) if not FL or any(c["id"].startswith(f) for f in FL)]
m = PerceptionModel("hybrid", teacher_dims={"dino": 384}); m.load_state_dict(torch.load(CK if CK.endswith(".pt") else f"{CK}/hybrid/final.pt", map_location="cpu")); m = m.to(dev).eval()
r = CornerRefiner(); r.load_state_dict(torch.load(REF, map_location="cpu")); r = r.to(dev).eval()
VT = float(os.environ.get("VIS_THR", "-1e9"))


def best_err(pc, gc, vis, sx, sy):
    best = None
    for s in range(4):
        d = (np.roll(pc, s, 0) - gc)[vis]
        e256 = np.linalg.norm(d * [256 / sx, 192 / sy], axis=1); en = np.linalg.norm(d, axis=1)
        if best is None or e256.mean() < best[0].mean(): best = (e256, en, s)
    return best


rows = []
for c in cands:
    im = Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB"); W, H = im.size
    full = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None].to(dev) / 255
    x = torch.nn.functional.interpolate(full, size=(448, 448), mode="bilinear", align_corners=False)
    with torch.no_grad(): s = m(normalize(x), with_teacher_heads=False)["slots"].float()[0]
    keep = [q for q in range(s.shape[0]) if s[q, 0].sigmoid() > 0.5]
    if not keep: continue
    coarse = torch.stack([s[q, 1:] * torch.tensor([W, H] * 4, device=dev) for q in keep])
    fine = refine_gates(r, full, [coarse], vis_thr=VT)[0]
    cn, fn = coarse.cpu().numpy().reshape(-1, 4, 2), fine.cpu().numpy().reshape(-1, 4, 2)
    for g in c["gates"]:
        gc = np.array([[a * W, b * H] for a, b, _ in g["corners"]]); vis = np.array([v > 0 for *_, v in g["corners"]])
        if vis.sum() < 2: continue
        bi = min(range(len(cn)), key=lambda i: best_err(cn[i], gc, vis, W, H)[0].mean())
        e0 = best_err(cn[bi], gc, vis, W, H)
        if e0[0].mean() >= 20: continue
        d1 = (np.roll(fn[bi], e0[2], 0) - gc)[vis]
        e1 = (np.linalg.norm(d1 * [256 / W, 192 / H], axis=1), np.linalg.norm(d1, axis=1))
        size = (gc[vis].max(0) - gc[vis].min(0)).max()
        for k in range(len(e0[0])): rows.append((e0[0][k], e1[0][k], e0[1][k], e1[1][k], size))
a = np.array(rows)
out = {"golden": G, "flights": FL, "corners": len(a), "median_256_before": round(float(np.median(a[:, 0])), 2), "median_256_after": round(float(np.median(a[:, 1])), 2),
       "median_native_before": round(float(np.median(a[:, 2])), 2), "median_native_after": round(float(np.median(a[:, 3])), 2),
       "p90_native_before": round(float(np.percentile(a[:, 2], 90)), 2), "p90_native_after": round(float(np.percentile(a[:, 3], 90)), 2),
       "lt2px_native_before": round(float((a[:, 2] < 2).mean()), 3), "lt2px_native_after": round(float((a[:, 3] < 2).mean()), 3)}
for lo, hi in ((0, 60), (60, 120), (120, 250), (250, 5000)):
    k = (a[:, 4] >= lo) & (a[:, 4] < hi)
    if k.any(): out[f"size{lo}-{hi}_native_median"] = [int(k.sum()), round(float(np.median(a[k, 2])), 2), round(float(np.median(a[k, 3])), 2)]
print(json.dumps(out), flush=True)
