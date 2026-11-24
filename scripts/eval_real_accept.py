"""REAL-data acceptance for the condition modules (the sim acceptance in eval_cond_router.py is not enough: Update 147).
Frames: runs/selfimprove/real_dark_split.json (real labelled dark frames, val / test) + a bright control (real day frames, val / test).
Per frame the router (ROUTER npz, optional in-distribution abstain) picks a condition; seg confusion is cached for core and every module,
so any set of enabled modules is scored by summation. Acceptance on REAL VAL: backward elimination from all modules, dropping the module
whose removal raises mean(val dark mIoU, val bright mIoU) the most, while that gain >= TOL. Reported on REAL TEST.
Run: env -u PYTHONPATH ROUTER=runs/selfimprove/router7.npz MODULES="night_real:...,nir:...,..." .venv/bin/python scripts/eval_real_accept.py"""
import os, sys, json
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from train_cond_adapter import CondAdapter

K = 11; TOL = float(os.environ.get("TOL", "0.0005"))
R = np.load(os.environ.get("ROUTER", "runs/selfimprove/router6a.npz")); mus, Si, conds = R["mus"], R["Si"], [str(c) for c in R["conds"]]
THR = R["thr"] if ("thr" in R.files and os.environ.get("ABSTAIN", "0") == "1") else None
MODS = dict(kv.split(":") for kv in os.environ["MODULES"].split(","))
core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.cuda().eval()
ad = {}
for c, p in MODS.items(): A = CondAdapter().cuda(); A.load_state_dict(torch.load(p, map_location="cpu")); ad[c] = A.eval()
S = json.loads(Path("runs/selfimprove/real_dark_split.json").read_text())
br = S["bright"][::4]; sets = dict(val_dark=S["val"], test_dark=S["test"], val_bright=br[0::2][:60], test_bright=br[1::2][:60])

def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].cuda() / 255
def route(x):
    f = fused_features(core, normalize(x)).float().mean((2, 3))[0].cpu().numpy(); d = f - mus; dd = np.einsum("kd,de,ke->k", d, Si, d); k = int(np.argmin(dd))
    return conds[k] if (THR is None or dd[k] <= THR[k]) else "abstain"
def cm_of(o, lab):
    pr = o[0].float().argmax(0).cpu().numpy(); v = (lab != 255) & (lab < K); return np.bincount(lab[v] * K + pr[v], minlength=K * K).reshape(K, K)
def miou(cm): iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); return float(iu[cm.sum(1) > 0].mean())

cache = {}
with torch.no_grad():
    for name, fr in sets.items():
        rows = []
        for p, lp in fr:
            x = load(p); lab = np.asarray(Image.open(lp).resize((448, 448), Image.NEAREST)).astype(np.int64)
            row = dict(route=route(x), core=cm_of(core(normalize(x), with_teacher_heads=False)["seg"], lab))
            for c, A in ad.items(): row[c] = cm_of(core(normalize(A(x)), with_teacher_heads=False)["seg"], lab)
            rows.append(row)
        cache[name] = rows; print(name, len(rows), "routes", {c: sum(r["route"] == c for r in rows) for c in set(r["route"] for r in rows)}, flush=True)

def score(name, enabled): return round(miou(sum(r[r["route"]] if r["route"] in enabled else r["core"] for r in cache[name])), 4)
def forced(name, c): return round(miou(sum(r[c] for r in cache[name])), 4)
def val(en): return (score("val_dark", en) + score("val_bright", en)) / 2
en = set(ad); hist = [(sorted(en), val(en))]
while en:
    best = max(en, key=lambda m: val(en - {m}))
    if val(en - {best}) - val(en) < -TOL: break          # removal must not cost more than TOL (ties -> fewer modules)
    en = en - {best}; hist.append((sorted(en), val(en)))
if en and val(en) < val(set()) + TOL: hist.append(([], val(set()))); en = set()   # final guard: the enabled set must beat core alone on real val
out = dict(elimination=hist, real_accepted=sorted(en),
           forced={n: {c: forced(n, c) for c in ["core"] + list(ad)} for n in sets},
           test={n: dict(core=score(n, set()), all_modules=score(n, set(ad)), real_accepted=score(n, en)) for n in ("test_dark", "test_bright")},
           val={n: dict(core=score(n, set()), all_modules=score(n, set(ad)), real_accepted=score(n, en)) for n in ("val_dark", "val_bright")})
print(json.dumps(out, indent=1)); Path(os.environ.get("OUTJ", "runs/selfimprove/real_accept.json")).write_text(json.dumps(out, indent=1))
