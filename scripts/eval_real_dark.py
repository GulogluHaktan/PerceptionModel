"""Real-data check of the condition router + modules (no simulated effect): pick genuinely dark frames from the labelled road
segmentation eval sets (BDD / IDD-AW / ... in data/general/eval_*), by mean luminance, and compare seg mIoU of
  core alone | routed system (router6.npz + accepted modules) | each module forced on.
A bright control subset of the same size checks that the router leaves normal frames alone.
Run: env -u PYTHONPATH .venv/bin/python scripts/eval_real_dark.py   (env DARK=0.20 BRIGHT=0.35 NMAX=300)"""
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

DARK = float(os.environ.get("DARK", "0.20")); BRIGHT = float(os.environ.get("BRIGHT", "0.35")); NMAX = int(os.environ.get("NMAX", "300"))
M = Path("runs/selfimprove"); R = np.load(os.environ.get("ROUTER", str(M / "router6.npz"))); mus, Si, conds = R["mus"], R["Si"], [str(c) for c in R["conds"]]
THR = R["thr"] if ("thr" in R.files and os.environ.get("ABSTAIN", "1") == "1") else None; ABST = float(os.environ.get("ABST", "1.0"))
MODS = dict(nir=M / "cond_nir/adapter.pt", night=M / "cond_night/adapter.pt", fog=M / "cond_fog/adapter.pt", blur=M / "cond_blur/adapter.pt", night_blur=M / "cond_night_blur_v2/adapter.pt")
core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.cuda().eval()
ad = {}
for c, p in MODS.items(): A = CondAdapter().cuda(); A.load_state_dict(torch.load(p, map_location="cpu")); ad[c] = A.eval()

# collect labelled frames, dedup by file name
seen, dark, bright = set(), [], []
for d in [d for s_ in os.environ.get("SETS", "eval_bdd,eval_iddaw").split(",") for d in sorted(Path("data/general").glob(s_ + "*"))]:
    if not (d / "seg").is_dir(): continue
    for p in sorted((d / "images").glob("*")):
        lp = d / "seg" / (p.stem + ".png")
        if p.name in seen or not lp.exists(): continue
        seen.add(p.name)
        y = np.asarray(Image.open(p).convert("L").resize((112, 112))).mean() / 255
        if y < DARK: dark.append((p, lp, d.name))
        elif y > BRIGHT: bright.append((p, lp, d.name))
dark = dark[:NMAX]; bright = bright[::max(1, len(bright) // max(1, len(dark)))][:len(dark)]
print(f"dark {len(dark)} (sets {sorted(set(s for *_, s in dark))}), bright control {len(bright)}", flush=True)


def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].cuda() / 255
def route(x):
    f = fused_features(core, normalize(x)).float().mean((2, 3))[0].cpu().numpy(); d = f - mus
    dd = np.einsum("kd,de,ke->k", d, Si, d); k = int(np.argmin(dd))
    return conds[k] if (THR is None or dd[k] <= THR[k] * ABST) else "abstain"


def run(frames):
    arms = ["core", "system"] + list(ad); cms = {a: None for a in arms}; routes = {}
    with torch.no_grad():
        for p, lp, _ in frames:
            x = load(p); lab = np.asarray(Image.open(lp).resize((448, 448), Image.NEAREST)).astype(np.int64); v = (lab != 255) & (lab < 11)
            c = route(x); routes[c] = routes.get(c, 0) + 1
            for a in arms:
                A = ad.get(c) if a == "system" else (None if a == "core" else ad[a])
                o = core(normalize(A(x) if A is not None else x), with_teacher_heads=False)["seg"]
                pr = o[0].float().argmax(0).cpu().numpy(); K = o.shape[1]
                b = np.bincount(lab[v] * K + pr[v], minlength=K * K).reshape(K, K); cms[a] = b if cms[a] is None else cms[a] + b
    out = {}
    for a, cm in cms.items():
        iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); out[a] = round(float(iu[cm.sum(1) > 0].mean()), 3)
    return dict(miou=out, routes=routes)


res = dict(dark=run(dark), bright=run(bright), n_dark=len(dark), n_bright=len(bright), thr=dict(dark=DARK, bright=BRIGHT))
print(json.dumps(res, indent=1)); (M / "real_dark_eval.json").write_text(json.dumps(res, indent=1))
