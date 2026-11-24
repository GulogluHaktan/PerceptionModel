"""Self-improvement loop, step 1: can the FROZEN core tell by itself that it is in a harder condition (distribution shift)?
Real labelled gate frames (never-seen 'red' place + TII golden) are rendered under conditions with pfm.effects (fog, night, NIR-like
grayscale, motion blur). Per frame we record the gate recall (within 10 px at 256x192, best cyclic order, visible corners) and
label-free signals from the frozen model: seg entropy, gate slot confidence, feature distance (Mahalanobis of pooled stride-8
fused features against clean reference frames). Reports per condition: recall, AUROC(clean vs condition) per signal and the
Spearman correlation signal <-> per-frame recall. No training, core untouched.
Run: PFM_GDET_LEVELS=56 PFM_GDET_K=8 env -u PYTHONPATH .venv/bin/python scripts/shift_signals.py
"""
import json, math, os, sys
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from pfm import effects as E

dev = "cuda"; W, H = 256.0, 192.0; THR = 0.5
CK = os.environ.get("CKPT", "runs/release_v4/bigY_h_hybrid.pt")
m = PerceptionModel("hybrid", teacher_dims={"dino": 384}); m.load_state_dict(torch.load(CK, map_location="cpu")); m = m.to(dev).eval()

frames = []
for G, pref in (("data/real_mrg/golden", "red_"), ("data/real_tii/golden", "")):
    for c in json.load(open(f"{G}/candidates.json")):
        if not c["id"].startswith(pref): continue
        gs = [np.array([[x, y, v] for x, y, v in g["corners"]], np.float32) for g in c.get("gates", [])]
        frames.append((f"{G}/images/{c['id']}.jpg", gs))
print(f"frames {len(frames)}", flush=True)
gen = torch.Generator(device=dev).manual_seed(0)

def cond(x, name, inv):
    if name == "clean": return x
    if name == "fog":   # relative inverse depth -> metres with median 8 m
        d = 1.0 / inv.clamp(min=1e-3); d = d * (8.0 / d.median()); return E.fog(x, d, beta=0.12, airlight=0.8)
    if name == "night": return E.sensor(x, ev=-4.0, gain_ev=3.0, iso=3200, gen=gen)
    if name == "nir":   # IR-like: single channel, vegetation/skin bright, low colour contrast, active-light vignetting
        g = (0.1 * x[:, 0:1] + 0.3 * x[:, 1:2] + 0.6 * x[:, 0:1].clamp(max=1)).clamp(0, 1); g = g.repeat(1, 3, 1, 1)
        return E.lens(g, 0.0, 0.0, 0.6)
    if name == "blur":
        Hh, Ww = x.shape[-2:]; flow = torch.zeros(Hh, Ww, 2, device=dev); flow[..., 0] = 1500.0; flow[..., 1] = 400.0
        return E.motion_blur(x, flow, exposure_s=0.012)
    raise ValueError(name)

def gate_recall(slots, gs):
    preds = [slots[q, 1:].view(4, 2).cpu().numpy() * [W, H] for q in range(slots.shape[0]) if slots[q, 0].sigmoid() > THR]
    hit = n = 0
    for g in gs:
        v = g[:, 2] > 0
        if v.sum() < 2: continue
        n += 1; tgt = g[:, :2] * [W, H]
        es = [min(np.linalg.norm((np.roll(p, s, 0) - tgt)[v], axis=1).mean() for s in range(4)) for p in preds]
        hit += int(bool(es) and min(es) < 10.0)
    return hit, n

CONDS = ["clean", "fog", "night", "nir", "blur"]
rec = {c: [] for c in CONDS}
with torch.no_grad():
    for path, gs in frames:
        x0 = torch.from_numpy(np.asarray(Image.open(path).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].to(dev) / 255
        inv = m(normalize(x0), with_teacher_heads=False)["depth"][0, 0].float()
        for c in CONDS:
            x = cond(x0, c, inv).clamp(0, 1)
            xn = normalize(x); out = m(xn, with_teacher_heads=False)
            p = out["seg"].float().softmax(1); ent = float((-(p * p.clamp(min=1e-8).log()).sum(1)).mean())
            sc = out["slots"][0, :, 0].float().sigmoid(); conf = float(sc.max())
            amb = float(((sc > 0.2) & (sc < 0.8)).float().sum())          # undecided slots
            f = fused_features(m, xn).float().mean((2, 3))[0].cpu().numpy()
            h, n = gate_recall(out["slots"][0].float(), gs)
            rec[c].append(dict(ent=ent, conf=conf, amb=amb, feat=f, hit=h, n=n))

ref = np.stack([r["feat"] for r in rec["clean"][::2]])                      # clean reference = even frames (odd frames are the clean test set)
mu = ref.mean(0); cov = np.cov(ref.T) + 1e-3 * np.eye(ref.shape[1]); ci = np.linalg.inv(cov)
for c in CONDS:
    for r in rec[c]: d = r["feat"] - mu; r["maha"] = float(math.sqrt(d @ ci @ d))

def auroc(neg, pos):
    a = np.array(neg); b = np.array(pos); return float(((b[:, None] > a[None]).mean() + 0.5 * (b[:, None] == a[None]).mean()))
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1]) if np.std(rb) > 0 else float("nan")

clean_test = rec["clean"][1::2]
sig = {"seg_entropy": lambda r: r["ent"], "low_gate_conf": lambda r: -r["conf"], "undecided_slots": lambda r: r["amb"], "feature_dist": lambda r: r["maha"]}
res = {}
for c in CONDS:
    R = rec[c] if c != "clean" else clean_test
    h = sum(r["hit"] for r in R); n = sum(r["n"] for r in R)
    row = dict(gate_recall_10px=round(h / max(1, n), 3), n_gates=n)
    if c != "clean":
        row["auroc_vs_clean"] = {k: round(auroc([f(r) for r in clean_test], [f(r) for r in R]), 3) for k, f in sig.items()}
    fr = [r for r in R if r["n"] > 0]
    row["spearman_signal_vs_frame_recall"] = {k: round(spearman([f(r) for r in fr], [r["hit"] / r["n"] for r in fr]), 3) for k, f in sig.items()}
    res[c] = row
pooled = [r for c in CONDS for r in (rec[c] if c != "clean" else clean_test) if r["n"] > 0]
res["pooled_spearman_signal_vs_frame_recall"] = {k: round(spearman([f(r) for r in pooled], [r["hit"] / r["n"] for r in pooled]), 3) for k, f in sig.items()}
print(json.dumps(res, indent=1)); Path("runs/selfimprove").mkdir(parents=True, exist_ok=True)
json.dump(res, open("runs/selfimprove/shift_signals.json", "w"), indent=1)
