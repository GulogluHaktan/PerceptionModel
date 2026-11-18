"""'Half gate' test in a place never trained on (MonoRaceGate red): gates of which only one corner (n_in=1) or no corner, just a piece of the frame (n_in=0), is in view.
The red frames have all 4 corners labelled, so each such view is made by cropping (zoom 1.6-4x, pfm.train.half_gate_crop, fixed seed, 2 crops per gate and case) around a gate that is fully visible
and at least 0.12 of the frame wide; the true position of the corners outside the crop is known. A gate counts as found when a prediction with score > THR has all 4 corners (also the unseen ones) within
a fraction of the gate size (mean corner error / largest side of the gate in the crop; best cyclic corner order). Reported: found at 0.5 / 0.25 / 0.1 of the size, for n_in=1 the error of the visible corner,
and predictions per crop that match no gate with a visible corner or bar. CAUTION: a crop of a far gate is not the same as a real close pass (no wide-angle distortion change, no motion blur), so this
is a proxy for the close-pass case. Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=vit python scripts/eval_half_gates.py"""
import json, os, random, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.train import half_gate_crop, frame_in_view
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); G = "data/real_mrg/golden"; W, H = 256.0, 192.0; THR = float(os.environ.get("THR", "0.5")); rng = random.Random(0)
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
def err(pc, gc, idx=slice(None)): return min(np.linalg.norm((np.roll(pc, s, 0) - gc)[idx], axis=1).mean() for s in range(4))
res = {0: [], 1: []}; vis1 = []; extra = extra_near = n_crop = 0
for c in json.load(open(f"{G}/candidates.json")):
    if not c["id"].startswith("red_"): continue
    im = Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB"); gs = [np.array([[x, y] for x, y, v in g["corners"]], np.float32) for g in c["gates"] if all(v > 0 for _, _, v in g["corners"])]
    for gi, g in enumerate(gs):
        if not ((g >= 0) & (g < 1)).all() or (g.max(0) - g.min(0)).max() < 0.12: continue
        for n_in in (0, 1):
            for _ in range(2):
                hc = half_gate_crop(g, n_in, rng)
                if hc is None: continue
                s, x0, y0 = hc; e = 1 / s; cr = im.crop((round(x0 * im.width), round(y0 * im.height), round((x0 + e) * im.width), round((y0 + e) * im.height))).resize((448, 448), Image.BICUBIC)
                x = torch.from_numpy(np.asarray(cr).copy()).permute(2, 0, 1).float()[None] / 255
                with torch.no_grad(): sl = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
                preds = [sl[q, 1:].view(4, 2).numpy() * [W, H] for q in range(sl.shape[0]) if sl[q, 0].sigmoid() > THR]; n_crop += 1
                q_all = [(o - np.array([x0, y0], np.float32)) * s * [W, H] for o in gs]; tgt = q_all[gi]; size = max(np.linalg.norm(tgt[i] - tgt[(i + 1) % 4]) for i in range(4))
                es = [err(p, tgt) for p in preds]; best = min(es) / size if es else 9.0; res[n_in].append(best)
                if n_in == 1 and es:
                    inside = ((tgt[:, 0] >= 0) & (tgt[:, 0] < W) & (tgt[:, 1] >= 0) & (tgt[:, 1] < H)); p = preds[int(np.argmin(es))]
                    vis1.append(min(np.linalg.norm((np.roll(p, r, 0) - tgt)[inside], axis=1).mean() for r in range(4)))
                # predictions that match no gate having a corner or a bar in view
                valid = [q for q in q_all if (((q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)).any() or frame_in_view(q, W, H) >= 0.15 * W)]
                nd = lambda p, q: err(p, q) / max(np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4))
                ex = [p for p in preds if not any(nd(p, q) < 0.5 for q in valid)]; extra += len(ex)
                extra_near += sum(1 for p in ex if any(nd(p, q) < 1.5 for q in q_all))   # a rough guess for a labelled gate (in view or cut) rather than a gate invented on background
out = {"crops": n_crop, "extra_pred_per_crop": round(extra / max(n_crop, 1), 3), "of_which_rough_guess_of_a_labelled_gate": round(extra_near / max(n_crop, 1), 3)}
for n_in in (0, 1):
    a = np.array(res[n_in]); out[f"corners_in_view_{n_in}"] = {"n": len(a), "found<0.5size": round(float((a < 0.5).mean()), 3), "found<0.25size": round(float((a < 0.25).mean()), 3), "found<0.1size": round(float((a < 0.1).mean()), 3), "no_prediction": round(float((a >= 9).mean()), 3)}
out["corners_in_view_1"]["visible_corner_median_err_px"] = round(float(np.median(vis1)), 1) if vis1 else None
print(f"{CK} {K} half gates on red (never seen): {json.dumps(out)}")
