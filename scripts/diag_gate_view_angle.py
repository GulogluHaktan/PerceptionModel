"""Gate detection against viewing angle on the never-seen place (MonoRaceGate red). A square gate seen from the side looks narrow: view ratio = shorter mean side / longer mean side
of the labelled quad (a square gate: 1 = frontal, 0.5 = about 60 degrees off; the gates of this place are taller than wide, so frontal is about 0.75 there). Per ratio bin: found within 20 / 10 px (256x192 scale) and median corner error. With TRAIN=1 it only counts how many
training gates fall in each bin (real and sim). Usage: PFM_GDET_K=8 PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 [PFM_GDET_LEVELS=56] CKPT=runs/x KIND=hybrid python scripts/diag_gate_view_angle.py"""
import json, os, sys
import numpy as np
BINS = [(0.85, 1.01, "ratio > 0.85"), (0.7, 0.85, "ratio 0.70-0.85"), (0.5, 0.7, "ratio 0.50-0.70"), (0.0, 0.5, "ratio < 0.50 (steep side view)")]
def ratio(c):      # c (4,2) px, order TL TR BR BL
    w = (np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[3] - c[2])) / 2; h = (np.linalg.norm(c[0] - c[3]) + np.linalg.norm(c[1] - c[2])) / 2; return min(w, h) / max(w, h, 1e-6)
if os.environ.get("TRAIN"):
    L = "data/mix/labels/corners_full"; st = {}
    for f in os.listdir(L):
        p = "real" if f.startswith("real_") else "sim"
        if p == "sim" and hash(f) % 5: continue
        for r in json.load(open(f"{L}/{f}")):
            if any(q is None for q in r["corners_px"]): continue
            c = np.array(r["corners_px"], float)
            if not ((c[:, 0] >= 0) & (c[:, 0] < 256) & (c[:, 1] >= 0) & (c[:, 1] < 192)).all() or np.ptp(c, axis=0).max() < 12: continue
            b = next(n for lo, hi, n in BINS if lo <= ratio(c) < hi); st[(p, b)] = st.get((p, b), 0) + 1
    for p in ("real", "sim"):
        tot = sum(v for (q, _), v in st.items() if q == p)
        for _, _, n in BINS: print(json.dumps({"training gates": p + (" (every 5th file)" if p == "sim" else ""), "view": n, "n": st.get((p, n), 0), "share": round(st.get((p, n), 0) / max(tot, 1), 3)}))
    sys.exit()
import torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); W, H = 256.0, 192.0; G = "data/real_mrg/golden"
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval(); st = {n: [] for _, _, n in BINS}
for x in json.load(open(f"{G}/candidates.json")):
    if not x["seq"].startswith("red"): continue
    t = torch.from_numpy(np.asarray(Image.open(f"{G}/images/{x['id']}.jpg").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): s = m(normalize(t.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
    preds = [s[q, 1:].view(4, 2).numpy() * [W, H] for q in range(s.shape[0]) if s[q, 0].sigmoid() > 0.5]
    for g in x["gates"]:
        c = np.array([a[:2] for a in g["corners"]], float) * [W, H]
        if not all(a[2] > 0 for a in g["corners"]): continue
        e = min((min(np.linalg.norm(np.roll(p, r, 0) - c, axis=1).mean() for r in range(4)) for p in preds), default=1e9)
        st[next(n for lo, hi, n in BINS if lo <= ratio(c) < hi)].append(e)
for n, v in st.items():
    v = np.array(v); print(json.dumps({"model": CK, "view": n, "n": len(v), "found<20px": round(float((v < 20).mean()), 3) if len(v) else None, "found<10px": round(float((v < 10).mean()), 3) if len(v) else None, "found<5px": round(float((v < 5).mean()), 3) if len(v) else None, "median_err_px": round(float(np.median(v[v < 20])), 2) if (v < 20).any() else None}))
