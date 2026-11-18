"""What the planner would get: metric position of each gate from the 4 predicted corners (PnP with the known 1.5 m inner square and the camera intrinsics), compared with the same PnP on the true corners.
SIM frames (held-out racing set, rendered like the training data: sim-to-sim, NOT a real-world number); it shows how pixel errors turn into metres at each distance.
Per true gate with 4 corners in front of the camera: best prediction with score > THR (mean corner error, best cyclic order); it counts when within max(20 px, size/4). Reported per distance bin:
share found, median position error (m and % of the distance), median sideways (x, y) and along-view (z) error, median angle between the true and the estimated gate normal.
The focal length is taken per frame from the labels (the value that makes the true inner square 1.5 m), i.e. a calibrated camera is assumed.
Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid python scripts/eval_gate_pose_sim.py [set=sim_racec] [step]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
S = sys.argv[1] if len(sys.argv) > 1 else "sim_racec"; step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; R = Path("data/heldout"); CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); THR = float(os.environ.get("THR", "0.5")); W, H, G = 256.0, 192.0, 1.5
OBJ = np.array([[-G / 2, G / 2, 0], [G / 2, G / 2, 0], [G / 2, -G / 2, 0], [-G / 2, -G / 2, 0]], np.float64)
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
def fx_of(c, z):      # focal length that makes the labelled square 1.5 m on a side
    best = None
    for hf in np.arange(80.0, 165.0, 0.5):
        fx = W / 2 / np.tan(np.radians(hf) / 2); P = np.stack([(c[:, 0] - W / 2) / fx * z, (c[:, 1] - H / 2) / fx * z, z], 1); e = np.abs([np.linalg.norm(P[i] - P[(i + 1) % 4]) - G for i in range(4)]).mean()
        if best is None or e < best[0]: best = (e, fx)
    return best
def pnp(c, fx):
    ok, rv, tv = cv2.solvePnP(OBJ, c.astype(np.float64), np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]]), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (tv[:, 0], cv2.Rodrigues(rv)[0][:, 2]) if ok else (None, None)
BINS = [(0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)]; acc = {b: {"n": 0, "found": 0, "pos": [], "rel": [], "lat": [], "z": [], "ang": []} for b in BINS}
names = sorted(p.stem for p in (R / "images" / S).glob("*__clear.png"))[::step]
with torch.no_grad():
    for i in range(0, len(names), 16):
        ch = names[i:i + 16]; x = torch.stack([torch.from_numpy(np.asarray(Image.open(R / "images" / S / f"{n}.png").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for n in ch])
        pred = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()
        for j, n in enumerate(ch):
            rows = json.load(open(R / "labels/corners_full" / f"{S}__{n}.json")); ps = [pred[j, q, 1:].view(4, 2).numpy() * [W, H] for q in range(pred.shape[1]) if pred[j, q, 0].sigmoid() > THR]
            for r in rows:
                if any(p is None for p in r["corners_px"]) or min(r.get("depths", [0])) < 0.3: continue
                c = np.array(r["corners_px"], np.float64); z = np.array(r["depths"], np.float64)
                if not ((c[:, 0] >= 0) & (c[:, 0] < W) & (c[:, 1] >= 0) & (c[:, 1] < H)).any(): continue
                fe, fx = fx_of(c, z)
                if fe > 0.08: continue      # label not consistent with a 1.5 m square (should not happen)
                tg, ng = pnp(c, fx)
                if tg is None: continue
                d = float(np.linalg.norm(tg)); b = next(bb for bb in BINS if bb[0] <= d < bb[1]); a = acc[b]; a["n"] += 1
                size = max(np.linalg.norm(c[q] - c[(q + 1) % 4]) for q in range(4)); cand = [(np.linalg.norm(np.roll(p, s, 0) - c, axis=1).mean(), k, s) for k, p in enumerate(ps) for s in range(4)]
                if not cand: continue
                e, k, s = min(cand)
                if e >= max(20.0, size / 4): continue
                tp, npd = pnp(np.roll(ps[k], s, 0), fx)
                if tp is None: continue
                a["found"] += 1; dv = tp - tg; a["pos"].append(np.linalg.norm(dv)); a["rel"].append(np.linalg.norm(dv) / d); a["lat"].append(np.linalg.norm(dv[:2])); a["z"].append(abs(dv[2]))
                a["ang"].append(np.degrees(np.arccos(np.clip(abs(float(npd @ ng)), 0, 1))))
out = {}
for b, a in acc.items():
    if a["n"] < 10: continue
    md = lambda v: round(float(np.median(v)), 3) if v else None
    out[f"{b[0]}-{b[1] if b[1] < 1e8 else 'up'} m"] = {"gates": a["n"], "found": round(a["found"] / a["n"], 3), "pos_err_m": md(a["pos"]), "pos_err_rel": md(a["rel"]), "sideways_m": md(a["lat"]), "along_view_m": md(a["z"]), "normal_deg": md(a["ang"]),
                                                    "pos_err_m_p90": round(float(np.percentile(a["pos"], 90)), 3) if a["pos"] else None}
print(f"{CK} {K} gate pose from predicted corners ({S}, sim-to-sim): {json.dumps(out)}")
