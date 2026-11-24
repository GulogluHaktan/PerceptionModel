"""Fit and save the condition router used at run time (same LDA as scripts/eval_cond_router.py): pooled stride-8 fused features of the
frozen core on general TRAINING images rendered under {clean, fog, night, nir, blur}. Saves runs/selfimprove/router.npz
(mus, Si, conds) — loaded by ~/DroneRacing_eval/ours_detector.py when OURS_ROUTER is set.
Run: env -u PYTHONPATH .venv/bin/python scripts/fit_cond_router.py"""
import os, random, sys
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from pfm.conditions import apply_condition, CONDS
CONDS = os.environ.get("CONDS", ",".join(CONDS)).split(","); OUTR = os.environ.get("OUTR", "runs/selfimprove/router.npz")

core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.cuda().eval()
gen = torch.Generator(device="cuda").manual_seed(0)
tr = [p for p in Path("data/general/train/images").rglob("*.jpg")]; random.Random(1).shuffle(tr); tr = tr[:int(os.environ.get("NFIT", "200"))]
X, Y = [], []
with torch.no_grad():
    for p in tr:
        x0 = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].cuda() / 255
        for k, c in enumerate(CONDS):
            X.append(fused_features(core, normalize(apply_condition(x0, c, gen, core))).float().mean((2, 3))[0].cpu().numpy()); Y.append(k)
# extra classes from REAL unlabelled images (no effect applied), e.g. EXTRA="real_night:data/general/raw_acdc/datasets/night/trainB,real_fog:..."
for kv in [e for e in os.environ.get("EXTRA", "").split(",") if e]:
    name, d = kv.split(":"); fs = sorted(p for p in Path(d).glob("*") if p.suffix.lower() in (".jpg", ".png")); random.Random(2).shuffle(fs); CONDS = CONDS + [name]
    with torch.no_grad():
        for p in fs[:int(os.environ.get("NFIT", "200"))]:
            x0 = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].cuda() / 255
            X.append(fused_features(core, normalize(x0)).float().mean((2, 3))[0].cpu().numpy()); Y.append(len(CONDS) - 1)
X = np.stack(X); Y = np.array(Y); mus = np.stack([X[Y == k].mean(0) for k in range(len(CONDS))])
Sw = sum(np.cov((X[Y == k] - mus[k]).T) for k in range(len(CONDS))) / len(CONDS) + 1e-3 * np.eye(X.shape[1])
Si = np.linalg.inv(Sw); dk = np.array([np.einsum("nd,de,ne->n", X[Y == k] - mus[k], Si, X[Y == k] - mus[k]) for k in range(len(CONDS))])
thr = np.percentile(dk, float(os.environ.get("PCT", "99")), axis=1)   # in-distribution radius per class: farther -> abstain (core alone)
np.savez(OUTR, mus=mus, Si=Si, conds=np.array(CONDS), thr=thr, dk=dk); print("saved", OUTR)
