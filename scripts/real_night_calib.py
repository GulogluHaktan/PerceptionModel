"""How close is a synthetic night to REAL night, seen by the frozen core? Real dark pool = labelled dark frames (mean luminance < 0.25)
of eval_bdd + eval_iddaw + real_bdd/dark_{val,test}; split by sorted index (even = val, odd = test), saved to runs/selfimprove/real_dark_split.json.
Metric: distance between the mean pooled core feature of generated frames (bright BDD/IDD frames rendered with the effect) and of
REAL-VAL dark frames, in the router's within-class-whitened space (lower = closer), plus a 1-NN real-vs-fake classifier accuracy
(0.5 = indistinguishable). Run: env -u PYTHONPATH .venv/bin/python scripts/real_night_calib.py"""
import os, sys, json
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from pfm import effects as E

def lum(p): return np.asarray(Image.open(p).convert("L").resize((112, 112))).mean() / 255
def pairs(d): return [(p, d / "seg" / (p.stem + ".png")) for p in sorted((d / "images").glob("*")) if (d / "seg" / (p.stem + ".png")).exists()]

def real_dark_split():
    sp = Path("runs/selfimprove/real_dark_split.json")
    if sp.exists(): return json.loads(sp.read_text())
    dark, bright = [], []
    for d in [Path("data/general/eval_bdd"), Path("data/general/eval_iddaw"), Path("data/general/real_bdd/dark_val"), Path("data/general/real_bdd/dark_test")]:
        for p, l in pairs(d):
            y = lum(p); (dark if y < 0.25 else bright if (y > 0.35 and "eval_" in str(d)) else []).append([str(p), str(l)])
    out = dict(val=dark[0::2], test=dark[1::2], bright=bright); sp.write_text(json.dumps(out, indent=0)); return out

if __name__ == "__main__":
    S = real_dark_split(); print({k: len(v) for k, v in S.items()}, flush=True)
    core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.cuda().eval()
    R = np.load("runs/selfimprove/router6.npz"); Si = R["Si"]
    def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].cuda() / 255
    def feat(x): return fused_features(core, normalize(x)).float().mean((2, 3))[0].cpu().numpy()
    gen = torch.Generator(device="cuda").manual_seed(0); rng = np.random.default_rng(0)
    with torch.no_grad():
        RD = os.environ.get("REAL_DIR")      # e.g. ACDC night trainB (unlabelled real night) instead of the labelled real-val dark frames
        real = np.stack([feat(load(p)) for p in (sorted(Path(RD).glob("*"))[::3][:150] if RD else [q for q, _ in S["val"]])])
        src = [p for p, _ in S["bright"]][::3][:120]; F = {"clean": [], "sensor_night": [], "night_real": []}
        for p in src:
            x = load(p); inv = core(normalize(x), with_teacher_heads=False)["depth"][0, 0].float()
            F["clean"].append(feat(x)); F["sensor_night"].append(feat(E.sensor(x, ev=-4.0, gain_ev=3.0, iso=3200, gen=gen).clamp(0, 1)))
            F["night_real"].append(feat(night := E.night_real(x, inv, gen, rng)))
    L = np.linalg.cholesky(Si); res = {}
    for k, v in F.items():
        v = np.stack(v); d = (v.mean(0) - real.mean(0)) @ L; X = np.concatenate([v, real]) @ L; y = np.r_[np.zeros(len(v)), np.ones(len(real))]
        D = ((X[:, None] - X[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf); acc = float((y[D.argmin(1)] == y).mean())
        res[k] = dict(mean_dist=round(float(np.sqrt((d ** 2).sum())), 2), nn_real_vs_fake_acc=round(acc, 3)); print(k, res[k], flush=True)
    Path(os.environ.get("OUTJ", "runs/selfimprove/real_night_calib.json")).write_text(json.dumps(res, indent=1))
