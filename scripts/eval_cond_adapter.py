"""Evaluate the condition module (scripts/train_cond_adapter.py) on data it never saw:
gates = never-seen 'red' place + TII golden (recall within 10 px @256x192), segmentation = BDD100K val sky-aware (eval_bdd3, mIoU).
Conditions use FIXED strengths (same as shift_signals.py). Arms:
  core      frozen core alone
  always    adapter always plugged in
  routed    adapter plugged in only when the shift detector fires (feature Mahalanobis distance above the 95th percentile of clean
            reference frames; reference = even red/TII frames, clean) -> clean frames must stay exactly as 'core'
Run: env -u PYTHONPATH .venv/bin/python scripts/eval_cond_adapter.py [adapter.pt]
"""
import json, math, os, sys
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from pfm import effects as E
from train_cond_adapter import CondAdapter

dev = "cuda"; W, H = 256.0, 192.0; THR = 0.5; NSEG = int(os.environ.get("NSEG", "300"))
AP = sys.argv[1] if len(sys.argv) > 1 else "runs/selfimprove/cond_adapter/adapter.pt"
core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
A = CondAdapter().to(dev); A.load_state_dict(torch.load(AP, map_location="cpu")); A.eval()
gen = torch.Generator(device=dev)

def cond(x, name, inv):
    if name == "clean": return x
    if name == "fog": d = 1.0 / inv.clamp(min=1e-3); d = d * (8.0 / d.median()); return E.fog(x, d, beta=0.12, airlight=0.8)
    if name == "night": return E.sensor(x, ev=-4.0, gain_ev=3.0, iso=3200, gen=gen)
    if name == "nir":
        g = (0.1 * x[:, 0:1] + 0.3 * x[:, 1:2] + 0.6 * x[:, 0:1]).clamp(0, 1); return E.lens(g.repeat(1, 3, 1, 1), 0.0, 0.0, 0.6)
    if name == "blur":
        Hh, Ww = x.shape[-2:]; flow = torch.zeros(Hh, Ww, 2, device=dev); flow[..., 0] = 1500.0; flow[..., 1] = 400.0
        return E.motion_blur(x, flow, exposure_s=0.012)

def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].to(dev) / 255

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

with torch.no_grad():
    # shift detector reference: clean even gate frames
    ref = np.stack([fused_features(core, normalize(load(p))).float().mean((2, 3))[0].cpu().numpy() for p, _ in gate_frames[::2]])
    mu = ref.mean(0); ci = np.linalg.inv(np.cov(ref.T) + 1e-3 * np.eye(ref.shape[1]))
    dist = lambda xn: float(math.sqrt((lambda d: d @ ci @ d)(fused_features(core, xn).float().mean((2, 3))[0].cpu().numpy() - mu)))
    thr = float(np.percentile([dist(normalize(load(p))) for p, _ in gate_frames[::2]], 95))
    print(f"router threshold (95th pct clean ref) {thr:.2f}", flush=True)

    def run(x):
        """-> dict arm -> model output; plus whether the router fired"""
        xn = normalize(x); fire = dist(xn) > thr
        o_core = core(xn, with_teacher_heads=False); o_ad = core(normalize(A(x)), with_teacher_heads=False)
        return {"core": o_core, "always": o_ad, "routed": o_ad if fire else o_core}, fire

    res = {}
    for c in ["clean", "fog", "night", "nir", "blur"]:
        gen.manual_seed(0)
        gh = {a: [0, 0] for a in ("core", "always", "routed")}; fires = 0; nf = 0
        for p, gs in gate_frames[1::2]:          # odd frames: test (even frames were the router reference)
            x0 = load(p); inv = core(normalize(x0), with_teacher_heads=False)["depth"][0, 0].float()
            outs, fire = run(cond(x0, c, inv).clamp(0, 1)); fires += fire; nf += 1
            for a, o in outs.items(): h, n = gate_hits(o["slots"][0].float(), gs); gh[a][0] += h; gh[a][1] += n
        inter = {a: None for a in gh}; conf = {a: None for a in gh}
        for p in seg_frames:
            lab = np.asarray(Image.open(segd / "seg" / (p.stem + ".png")).resize((448, 448), Image.NEAREST)).astype(np.int64)
            x0 = load(p); inv = core(normalize(x0), with_teacher_heads=False)["depth"][0, 0].float()
            outs, _ = run(cond(x0, c, inv).clamp(0, 1))
            for a, o in outs.items():
                pr = o["seg"][0].float().argmax(0).cpu().numpy(); K = o["seg"].shape[1]; v = lab != 255
                cm = np.bincount(lab[v] * K + pr[v], minlength=K * K).reshape(K, K)
                conf[a] = cm if conf[a] is None else conf[a] + cm
        miou = {}
        for a, cm in conf.items():
            iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); present = cm.sum(1) > 0; miou[a] = round(float(iu[present].mean()), 3)
        res[c] = dict(gate_recall={a: round(h / max(1, n), 3) for a, (h, n) in gh.items()}, seg_miou_bdd=miou, router_fire_rate=round(fires / nf, 3))
        print(c, json.dumps(res[c]), flush=True)
    json.dump(res, open(Path(AP).parent / "eval.json", "w"), indent=1)
