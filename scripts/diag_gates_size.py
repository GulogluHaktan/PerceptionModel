"""Where does the dense gate head lose sim gates? Per labelled gate (4 largest, held-out sim racing frames): gate size bin (larger side in px at 256x192), found within 20 px at THR 0.5 / 0.3 / 0.1,
strict = error below max(3 px, a quarter of the gate size); heat score at the labelled centre (max over the 3x3 cells), share where that score is > 0.5 but no detection matched (lost by NMS / top-K / offsets), unmatched predictions per frame per threshold.
Usage: PFM_GATE_HEAD=dense [PFM_GDET_STRIDE=4] [PFM_GDET_NMS=3] CKPT=runs/bigC python scripts/diag_gates_size.py [set]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.train import slot_targets
from pfm.evaluate import match_gates
S = sys.argv[1] if len(sys.argv) > 1 else "sim_racec"; R = Path("data/heldout"); CK = os.environ["CKPT"]; KIND = os.environ.get("KIND", "vit"); W, H = 256.0, 192.0; THS = [0.5, 0.3, 0.1]; NS = int(os.environ.get("NSLOTS", "4"))
names = sorted(p.stem for p in (R / "images" / S).glob("*__clear.png"))
m = PerceptionModel(KIND, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{KIND}/final.pt", map_location="cpu")); m = m.cuda().eval()
rec = []; un = {t: 0 for t in THS}
with torch.no_grad():
    for i in range(0, len(names), 16):
        ch = names[i:i + 16]; x = torch.stack([torch.from_numpy(np.asarray(Image.open(R / "images" / S / f"{n}.png").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for n in ch])
        m.dense.n_slots = 4; o = m(normalize(x.cuda()), with_teacher_heads=False)
        if "gdet" in o: raw = o["gdet"]; m.dense.n_slots = NS; pred = m.dense.decode_gdet(raw).cpu(); heat = raw[:, 0].float().sigmoid().cpu()
        else: pred = o["slots"].float().cpu(); heat = torch.full((len(ch), 56, 56), float("nan"))      # MLP slot head: no heatmap
        g = heat.shape[-1]
        for j, n in enumerate(ch):
            t = slot_targets(json.load(open(R / "labels/corners_full" / f"{S}__{n}.json")))
            if t[:, 0].sum() == 0: continue
            res = {}
            for th in THS:
                ms, u = match_gates(pred[j], t, th); un[th] += u; res[th] = dict(ms)
            for k in range(4):
                if t[k, 0] < 0.5: continue
                c = t[k, 1:].view(4, 2); px = c * torch.tensor([W, H]); part = bool(((px[:, 0] < 0) | (px[:, 0] >= W) | (px[:, 1] < 0) | (px[:, 1] >= H)).any())
                ctr = c.mean(0).clamp(0, 1 - 1e-4); cj, ci = int(ctr[0] * g), int(ctr[1] * g); hs = heat[j, max(ci - 1, 0):ci + 2, max(cj - 1, 0):cj + 2].max().item()
                rec.append((float((px.max(0).values - px.min(0).values).max()), part, hs) + tuple((res[th][k] is not None and res[th][k] < 20) for th in THS) + tuple((res[th][k] is not None and res[th][k] < max(3.0, 0.25 * float((px.max(0).values - px.min(0).values).max()))) for th in THS))
r = np.array(rec, float); full = r[:, 1] == 0; out = {"ckpt": CK, "set": S, "frames": len(names), "unmatched_per_frame": {str(t): round(un[t] / len(names), 3) for t in THS}, "bins": []}
print(CK, S, "unmatched/frame", out["unmatched_per_frame"])
for lo, hi in [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]:
    q = r[full & (r[:, 0] >= lo) & (r[:, 0] < hi)]
    if not len(q): continue
    b = {"size_px": f"{lo}-{hi if hi < 1e8 else 'inf'}", "n": len(q), "found20_thr0.5": round(q[:, 3].mean(), 3), "thr0.3": round(q[:, 4].mean(), 3), "thr0.1": round(q[:, 5].mean(), 3), "strict_thr0.5": round(q[:, 6].mean(), 3), "strict_thr0.3": round(q[:, 7].mean(), 3), "centre_score>0.5": round((q[:, 2] > 0.5).mean(), 3), "centre>0.5_but_missed": round(((q[:, 2] > 0.5) & (q[:, 3] == 0)).mean(), 3)}
    out["bins"].append(b); print(b)
q = r[~full]; print("partly out: n", len(q), "found20", q[:, 3:6].mean(0).round(3), "| all full-view found20:", r[full][:, 3:6].mean(0).round(3), "strict:", r[full][:, 6:].mean(0).round(3)); out["full_view_found20"] = r[full][:, 3:6].mean(0).round(3).tolist(); out["full_view_strict"] = r[full][:, 6:].mean(0).round(3).tolist()
json.dump(out, open(f"runs/eval/diag_size_{S}_{os.path.basename(CK)}_{KIND}{os.environ.get('DTAG', '')}.json", "w"), indent=1)
