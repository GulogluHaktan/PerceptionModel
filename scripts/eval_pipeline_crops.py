"""Offline check of the full DroneRacing add-on pipeline (~/DroneRacing_eval/ours_detector.py: main model + OURS_NG + OURS_CREF + OURS_CPK_CONFIRM)
on held-out on-policy crops (default seed 24, never used for training). The detector reads its switches from the environment at import, so run
one process per variant. Prints found-rate per gate size (384 px scale) with two criteria:
  strict: mean visible-corner error < max(4 px, 5 % size)   diag: >= 2 visible corners within 4 px (like the in-loop PERCEPTION line)
and false detections per frame. Corner order is ignored (best of 8 cyclic / mirror orders).
Run: OURS_THR=0.05 OURS_NG=... OURS_CREF=... OURS_CPK=... OURS_CPK_CONFIRM=1 OURS_CPK_SNAP=0 env -u PYTHONPATH .venv/bin/python scripts/eval_pipeline_crops.py NAME
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, os.path.expanduser("~/DroneRacing_eval"))
import torch
import train_cornerpeak as T
import ours_detector as D
GN = None
if os.environ.get("EVAL_DET") == "gatenet":      # reference: their GateNet g17 on the same crops (decode_refined, as in eval_vision.py)
    from droneracing.gatenet import GateNet, decode_refined
    _ck = torch.load(os.path.expanduser("~/DroneRacing_eval/runs_host/g17/gatenet.pt"), map_location="cuda")
    GN = GateNet(refine=_ck.get("refine", False)).cuda().eval(); GN.load_state_dict(_ck["net"])

name = sys.argv[1] if len(sys.argv) > 1 else "run"
seeds = {int(s) for s in os.environ.get("EVAL_SEEDS", "24").split(",")}
imgs, cors, viss = T.load(seeds, int(os.environ.get("EVAL_STRIDE", "1")))
dev = "cuda"; st = {}; fp = 0; nf = 0; dup = 0; serr = {}      # per bucket: matched mean corner err (px@384) and |size ratio - 1|
orders = [list(torch.roll(torch.arange(4), r).tolist()) for r in range(4)] + [list(torch.roll(torch.arange(4).flip(0), r).tolist()) for r in range(4)]
with torch.no_grad():
    for i0 in range(0, len(imgs), 16):
        x = imgs[i0:i0 + 16].to(dev).float().div(255); B, _, h, w = x.shape
        if GN is not None:
            dets = decode_refined(*GN(x))
        else:
            dets = D.detect(x)
            if D.CPK: D.cpk_peaks(x, dets)          # CONFIRM filters / snaps dets in place
        C = cors[i0:i0 + 16].to(dev) * torch.tensor([w / T.IN, h / T.IN], device=dev); V = viss[i0:i0 + 16].to(dev)
        sc = 384 / max(h, w)
        for b in range(B):
            d = dets[b]; nf += 1; pc = d[:, 1:].view(-1, 4, 2) if len(d) else None; used = torch.zeros(len(d), dtype=torch.bool, device=dev)
            for g in range(C.shape[1]):
                v = V[b, g].float()
                if v.sum() < 2: continue
                size = float((C[b, g].amax(0) - C[b, g].amin(0)).max()) * sc
                key = "0-60" if size < 60 else "60-120" if size < 120 else "120-250" if size < 250 else "250+"
                e = st.setdefault(key, [0, 0, 0, 0]); e[0] += 1
                if pc is None: continue
                err = torch.stack([(pc[:, o] - C[b, g][None]).norm(dim=-1) for o in orders]) * sc      # [8,k,4]
                mean_v = ((err * v).sum(-1) / v.sum()).amin(0)                                             # [k]
                n4 = ((err < 4) & (v > 0)).sum(-1).amax(0)
                k = int(mean_v.argmin())
                if mean_v[k] < max(4.0, 0.05 * size): e[1] += 1
                if mean_v[k] < 0.25 * size and v.sum() == 4:
                    ps = float((pc[k].amax(0) - pc[k].amin(0)).max()) * sc; serr.setdefault(key, []).append((float(mean_v[k]), abs(ps / size - 1)))
                if int(n4.max()) >= 2: e[2] += 1
                ev = ((err[0] * v).sum(-1) / v.sum())      # identity order only (what the EKF uses)
                if float(ev.min()) < max(4.0, 0.05 * size): e[3] += 1
                near = err.mean(-1).amin(0) < 0.25 * size; used |= near; dup += max(0, int(near.sum()) - 1)      # extra detections of the same gate
            fp += int((~used).sum())
def q(a, p): a = sorted(a); return round(a[min(len(a) - 1, int(p * len(a)))], 3) if a else None
res = {k: dict(n=v[0], strict=round(v[1] / v[0], 3), diag=round(v[2] / v[0], 3), exact_order=round(v[3] / v[0], 3),
               err_med=q([a for a, _ in serr.get(k, [])], 0.5), err_p90=q([a for a, _ in serr.get(k, [])], 0.9),
               size_err_med=q([b for _, b in serr.get(k, [])], 0.5), size_err_p90=q([b for _, b in serr.get(k, [])], 0.9)) for k, v in sorted(st.items())}
print(json.dumps(dict(name=name, found=res, false_per_frame=round(fp / max(1, nf), 3), dup_per_frame=round(dup / max(1, nf), 3))), flush=True)
