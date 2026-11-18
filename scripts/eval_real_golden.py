"""Evaluate sim-trained students on the REAL TII golden candidates (data/real_tii/golden). Per gate: match predicted slots (score>0.5) to labelled gates using only visible corners and the best cyclic corner order; report recall within 10/20/40 px (256x192 scale), median corner error, false slots per frame. Usage: CKPT=runs/sim_v2 python scripts/eval_real_golden.py"""
import itertools, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel   # noqa: E402
from pfm.data import normalize          # noqa: E402
CK = os.environ.get("CKPT", "runs/sim_v2"); G = os.environ.get("GOLDEN", "data/real_tii/golden"); W, H = 256.0, 192.0; dev = "cuda"
EXCL = set(json.load(open(f"{G}/review_flags.json"))["suspect_by_claude_first_pass"]) if os.environ.get("EXCLUDE_FLAGGED") and os.path.exists(f"{G}/review_flags.json") else set()
FL = [x for x in os.environ.get("FLIGHTS", "").split(",") if x]
cands = [c for c in json.load(open(f"{G}/candidates.json")) if c["id"] not in EXCL and (not FL or c["id"][7:9] in FL or c.get("seq") in FL)]
def load(kind):
    m = PerceptionModel(kind, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{kind}/final.pt", map_location="cpu")); return m.to(dev).eval()
def err(pc, gc, vis):
    return min(np.linalg.norm((np.roll(pc, s, 0) - gc)[vis], axis=1).mean() for s in range(4))
out = {}
for kind in os.environ.get("KINDS", "vit,hybrid").split(","):
    m = load(kind); errs = []; n_gt = 0; n_fp = 0; part = []; n_pred = 0
    for c in cands:
        x = torch.from_numpy(np.asarray(Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): s = m(normalize(x.to(dev)), with_teacher_heads=False)["slots"].float().cpu()[0]
        keep = [q for q in range(4) if s[q, 0].sigmoid() > float(os.environ.get("THR", "0.5"))]; preds = [(s[q, 1:].view(4, 2).numpy() * [W, H]) for q in keep]; n_pred += len(preds); used = set()
        for g in c["gates"]:
            gc = np.array([[x_ * W, y_ * H] for x_, y_, v in g["corners"]]); vis = np.array([v > 0 for _, _, v in g["corners"]])
            if vis.sum() < 2: continue
            n_gt += 1; is_part = vis.sum() < 4; best = (1e9, None)
            for i, p in enumerate(preds):
                e = err(p, gc, vis)
                if e < best[0]: best = (e, i)
            errs.append(best[0]); part.append(is_part)
            if best[1] is not None: used.add(best[1])
        n_fp += len(preds) - len(used)
    e = np.array(errs); p = np.array(part)
    r = lambda t, mask=slice(None): round(float((e[mask] < t).mean()), 3)
    out[kind] = {"frames": len(cands), "gt_gates": n_gt, "recall<10px": r(10), "recall<20px": r(20), "recall<40px": r(40), "median_err_px": round(float(np.median(e)), 1), "full_gates_recall<20px": r(20, ~p), "partial_gates_recall<20px": r(20, p), "n_partial": int(p.sum()), "pred_per_frame": round(n_pred / len(cands), 2), "unmatched_pred_per_frame": round(n_fp / len(cands), 2)}
    print(kind, json.dumps(out[kind]), flush=True)
json.dump(out, open(f"runs/eval/real_golden_{os.path.basename(CK.rstrip('/'))}{'_fl' + '-'.join(FL) if FL else ''}{'_thr' + os.environ['THR'] if 'THR' in os.environ else ''}{'_' + os.path.basename(os.path.dirname(G)) if "GOLDEN" in os.environ else ''}_{'clean' if EXCL else 'all'}.json", "w"), indent=1)
