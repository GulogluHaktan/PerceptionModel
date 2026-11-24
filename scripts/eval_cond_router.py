"""Self-improvement loop with per-condition modules and a condition-aware router (docs/SELF_IMPROVE_PLAN.md, item 1).
1. Condition classifier (label-free at run time): LDA on pooled stride-8 fused features of the frozen core, fitted on general TRAINING
   images rendered under {clean, fog, night, nir, blur} with pfm.effects (no red / TII / BDD-val image is used).
2. Acceptance (validation split = even red/TII frames + first half of the BDD val subset): module m_c is accepted for condition c only if,
   under c, gate recall and BDD mIoU are both >= core - TOL and at least one improves by >= GAIN.
3. Test (odd red/TII frames + second half of BDD): the full system classifies each frame; if the predicted condition has an accepted
   module it is plugged in, otherwise the core runs alone. Reported per true condition: core vs system, router accuracy.
Run: env -u PYTHONPATH .venv/bin/python scripts/eval_cond_router.py   (env MODULES="nir:runs/.../adapter.pt,night:...,fog:...")
"""
import json, os, random, sys
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from train_cond_adapter import CondAdapter
from pfm import effects as E

# same conditions / data / metrics as scripts/eval_cond_adapter.py (copied: that file runs its evaluation at import)
W, H, THR = 256.0, 192.0, 0.5; NSEG = int(os.environ.get("NSEG", "300"))
core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to("cuda").eval()
gen = torch.Generator(device="cuda")
def cond(x, name, inv):
    if name == "clean": return x
    if name == "fog": d = 1.0 / inv.clamp(min=1e-3); d = d * (8.0 / d.median()); return E.fog(x, d, beta=0.12, airlight=0.8)
    if name == "night": return E.sensor(x, ev=-4.0, gain_ev=3.0, iso=3200, gen=gen)
    if name == "nir":
        g = (0.1 * x[:, 0:1] + 0.3 * x[:, 1:2] + 0.6 * x[:, 0:1]).clamp(0, 1); return E.lens(g.repeat(1, 3, 1, 1), 0.0, 0.0, 0.6)
    if name == "blur":
        Hh, Ww = x.shape[-2:]; flow = torch.zeros(Hh, Ww, 2, device="cuda"); flow[..., 0] = 1500.0; flow[..., 1] = 400.0
        return E.motion_blur(x, flow, exposure_s=0.012)
    if name == "night_blur":
        Hh, Ww = x.shape[-2:]; flow = torch.zeros(Hh, Ww, 2, device="cuda"); flow[..., 0] = 1500.0; flow[..., 1] = 400.0
        return E.sensor(E.motion_blur(x, flow, exposure_s=0.012), ev=-4.0, gain_ev=3.0, iso=3200, gen=gen)
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].to("cuda") / 255
def gate_hits(slots, gs):
    preds = [slots[q, 1:].view(4, 2).cpu().numpy() * [W, H] for q in range(slots.shape[0]) if slots[q, 0].sigmoid() > THR]
    hit = n = 0
    for g in gs:
        v = g[:, 2] > 0
        if v.sum() < 2: continue
        n += 1; tgt = g[:, :2] * [W, H]
        es = [min(np.linalg.norm((np.roll(p, s, 0) - tgt)[v], axis=1).mean() for s in range(4)) for p in preds]
        hit += int(bool(es) and min(es) < 10.0)
    return hit, n
gate_frames = []
for G, pref in (("data/real_mrg/golden", "red_"), ("data/real_tii/golden", "")):
    for c in json.load(open(f"{G}/candidates.json")):
        if c["id"].startswith(pref):
            gate_frames.append((f"{G}/images/{c['id']}.jpg", [np.array([[x, y, v] for x, y, v in g["corners"]], np.float32) for g in c.get("gates", [])]))
segd = Path("data/general/eval_bdd3"); seg_frames = sorted((segd / "images").iterdir())[:NSEG]

dev = "cuda"; TOL = float(os.environ.get("TOL", "0.005")); GAIN = float(os.environ.get("GAIN", "0.01")); NFIT = int(os.environ.get("NFIT", "200"))
CONDS = os.environ.get("CONDS", "clean,fog,night,nir,blur").split(",")
MODS = dict(kv.split(":") for kv in os.environ.get("MODULES", "nir:runs/selfimprove/cond_nir/adapter.pt,night:runs/selfimprove/cond_night/adapter.pt,fog:runs/selfimprove/cond_fog/adapter.pt").split(","))
ad = {}
for c, p in MODS.items():
    if os.path.exists(p): A = CondAdapter().to(dev); A.load_state_dict(torch.load(p, map_location="cpu")); ad[c] = A.eval()
print("modules", list(ad), flush=True)

def pooled(xn): return fused_features(core, xn).float().mean((2, 3))[0].cpu().numpy()

with torch.no_grad():
    # 1. condition classifier on training images
    tr = [p for p in Path("data/general/train/images").rglob("*.jpg")]; random.Random(1).shuffle(tr); tr = tr[:NFIT]
    X, Y = [], []
    for p in tr:
        x0 = load(p); inv = core(normalize(x0), with_teacher_heads=False)["depth"][0, 0].float()
        for k, c in enumerate(CONDS): X.append(pooled(normalize(cond(x0, c, inv).clamp(0, 1)))); Y.append(k)
    X = np.stack(X); Y = np.array(Y); mus = np.stack([X[Y == k].mean(0) for k in range(len(CONDS))])
    Sw = sum(np.cov((X[Y == k] - mus[k]).T) for k in range(len(CONDS))) / len(CONDS) + 1e-3 * np.eye(X.shape[1]); Si = np.linalg.inv(Sw)
    def classify(f): d = f - mus; return int(np.argmin(np.einsum("kd,de,ke->k", d, Si, d)))

    def score(frames_g, frames_s, c, use):
        """use: callable(x, pred_cond) -> adapter or None. Returns gate recall, mIoU, router accuracy"""
        gen.manual_seed(0); gh = [0, 0]; cm = None; ok = n = 0
        for p, gs in frames_g:
            x0 = load(p); inv = core(normalize(x0), with_teacher_heads=False)["depth"][0, 0].float(); x = cond(x0, c, inv).clamp(0, 1)
            pc = classify(pooled(normalize(x))); ok += int(CONDS[pc] == c); n += 1; A = use(pc)
            o = core(normalize(A(x) if A is not None else x), with_teacher_heads=False); h, m = gate_hits(o["slots"][0].float(), gs); gh[0] += h; gh[1] += m
        for p in frames_s:
            lab = np.asarray(Image.open(segd / "seg" / (p.stem + ".png")).resize((448, 448), Image.NEAREST)).astype(np.int64)
            x0 = load(p); inv = core(normalize(x0), with_teacher_heads=False)["depth"][0, 0].float(); x = cond(x0, c, inv).clamp(0, 1)
            pc = classify(pooled(normalize(x))); ok += int(CONDS[pc] == c); n += 1; A = use(pc)
            o = core(normalize(A(x) if A is not None else x), with_teacher_heads=False)
            pr = o["seg"][0].float().argmax(0).cpu().numpy(); K = o["seg"].shape[1]; v = lab != 255
            b = np.bincount(lab[v] * K + pr[v], minlength=K * K).reshape(K, K); cm = b if cm is None else cm + b
        iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); miou = float(iu[cm.sum(1) > 0].mean())
        return round(gh[0] / max(1, gh[1]), 3), round(miou, 3), round(ok / n, 3)

    half = len(seg_frames) // 2
    val_g, val_s, te_g, te_s = gate_frames[0::2], seg_frames[:half], gate_frames[1::2], seg_frames[half:]
    # 2. acceptance on validation (module forced on under its own condition)
    accepted = {}; log = {}
    for c, A in ad.items():
        g0, s0, _ = score(val_g, val_s, c, lambda pc: None); g1, s1, _ = score(val_g, val_s, c, lambda pc, A=A: A)
        acc = g1 >= g0 - TOL and s1 >= s0 - TOL and (g1 >= g0 + GAIN or s1 >= s0 + GAIN)
        log[c] = dict(val_core=(g0, s0), val_module=(g1, s1), accepted=acc); print("accept?", c, log[c], flush=True)
        if acc: accepted[c] = A
    # 3. test: full routed system vs core
    res = {}
    for c in CONDS:
        gc, sc, ra = score(te_g, te_s, c, lambda pc: None)
        gs_, ss, _ = score(te_g, te_s, c, lambda pc: accepted.get(CONDS[pc]))
        res[c] = dict(core=dict(gate=gc, seg=sc), system=dict(gate=gs_, seg=ss), router_acc=ra); print("test", c, res[c], flush=True)
    out = dict(acceptance=log, test=res, accepted=list(accepted))
    Path("runs/selfimprove/router_eval.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
