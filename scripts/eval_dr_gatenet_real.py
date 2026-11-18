"""Evaluate the DroneRacing GateNet (g17) on the same real golden frames and with the same matching as eval_real_golden.py (256x192 pixel scale,
visible corners, best cyclic order). Usage: DR=<DroneRacing checkout> GOLDEN=data/real_mrg/golden FLIGHTS=red_s1_f1,... python scripts/eval_dr_gatenet_real.py
Input to GateNet: the frame resized to 256x192 (its training resolution), RGB 0..1; threshold THR (default 0.3, its decode default)."""
import json, os, sys, itertools
import numpy as np, torch
from PIL import Image
DR = os.environ["DR"]; sys.path.insert(0, DR)
from droneracing.gatenet import GateNet, decode, decode_refined
G = os.environ.get("GOLDEN", "data/real_mrg/golden"); W, H = 256, 192; dev = "cuda"; THR = float(os.environ.get("THR", "0.3"))
FL = [x for x in os.environ.get("FLIGHTS", "").split(",") if x]
cands = [c for c in json.load(open(f"{G}/candidates.json")) if (not FL or c["id"][7:9] in FL or c.get("seq") in FL)]
ck = torch.load(f"{DR}/releases/v3_FAST_soup85_nov5/gatenet/g17_gatenet.pt", map_location=dev); ref = ck.get("refine", True)
gn = GateNet(refine=ref).to(dev); gn.load_state_dict(ck["net"]); gn.eval()
def err(pc, gc, vis): return min(np.linalg.norm((np.roll(pc, s, 0) - gc)[vis], axis=1).mean() for s in range(4))
errs = []; part = []; n_fp = 0; n_pred = 0
for c in cands:
    x = torch.from_numpy(np.asarray(Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB").resize((W, H), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].to(dev) / 255
    with torch.no_grad():
        d = gn.detect(x, thr=THR)[0]
    preds = [p[1:].view(4, 2).cpu().numpy() for p in d]; n_pred += len(preds); used = set()
    for g in c["gates"]:
        gc = np.array([[x_ * W, y_ * H] for x_, y_, v in g["corners"]]); vis = np.array([v > 0 for _, _, v in g["corners"]])
        if vis.sum() < 2: continue
        best = (1e9, None)
        for i, p in enumerate(preds):
            e = err(p, gc, vis)
            if e < best[0]: best = (e, i)
        errs.append(best[0]); part.append(vis.sum() < 4)
        if best[1] is not None: used.add(best[1])
    n_fp += len(preds) - len(used)
e = np.array(errs); p = np.array(part); r = lambda t, m=slice(None): round(float((e[m] < t).mean()), 3)
print(json.dumps({"model": "DroneRacing GateNet g17", "frames": len(cands), "gt_gates": len(e), "recall<10px": r(10), "recall<20px": r(20), "recall<40px": r(40), "median_err_px": round(float(np.median(e)), 1), "full<20": r(20, ~p), "partial<20": r(20, p), "n_partial": int(p.sum()), "unmatched_pred_per_frame": round(n_fp / len(cands), 2)}))
