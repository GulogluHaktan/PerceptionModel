"""Stress test of the sim-trained students on held-out RACING frames: the clear frames are degraded in controlled ways (motion blur at different speeds and turn rates, light levels, fog, rain, JPEG, and a worst-case mix) with the project's own effects pipeline,
then gate IoU, race seg mIoU, gate recall and corner error are measured per condition. Also writes labelled-vs-predicted pictures.
Training saw only five fixed conditions (clear, blur 1/40 s, fog 0.03, rain, low light ev-3 gain+3): stronger levels here are OUT of its training range, which is the point.
Usage: [CKPT=runs/sim_v2] python scripts/eval_robustness_sim.py [step=4]   -> runs/eval/robustness_sim_race.json, runs/eval/vis/*.png"""
import json, math, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES      # noqa: E402
from pfm.data import normalize                      # noqa: E402
from pfm.effects import pipeline                    # noqa: E402
from pfm.train import slot_targets                  # noqa: E402
from pfm.evaluate import match_gates                # noqa: E402
R = Path("data/heldout"); S = "sim_race"; CK = os.environ.get("CKPT", "runs/sim_v2"); SIZE = 448; NC = len(CLASSES); GATE = CLASSES.index("gate"); dev = "cuda"; W_, H_ = 256.0, 192.0
step = int(sys.argv[1]) if len(sys.argv) > 1 else 4
fwd = lambda v: ({"v": (0, 0, v)}, {"exposure_s": 1 / 60}); yaw = lambda w: ({"v": (0, 0, 0), "w": (0, w, 0)}, {"exposure_s": 1 / 60})
light = lambda ev, gain=0.0, iso=800.0: ({}, {"ev": ev, "gain_ev": gain, "iso": iso, "exposure_s": 1 / 100})
COND = {"clear (as rendered)": None, "blur forward 10 m/s": fwd(10), "blur forward 20 m/s": fwd(20), "blur forward 30 m/s": fwd(30), "blur forward 40 m/s": fwd(40), "blur forward 60 m/s": fwd(60),
        "blur turning 0.5 rad/s": yaw(0.5), "blur turning 1 rad/s": yaw(1.0), "blur turning 2 rad/s": yaw(2.0),
        "light +1 stop": light(1.0, 0, 400), "light -1 stop": light(-1.0), "light -2 stops": light(-2.0), "light -3 stops": light(-3.0), "light -4 stops": light(-4.0, 0, 1600), "low light (train: -3, gain +3)": light(-3.0, 3.0, 3200.0),
        "fog 0.02": ({}, {"exposure_s": 1 / 250, "fog_beta": 0.02, "fog_airlight": 0.78}), "fog 0.05": ({}, {"exposure_s": 1 / 250, "fog_beta": 0.05, "fog_airlight": 0.78}),
        "rain 0.7": ({}, {"exposure_s": 1 / 250, "rain_density": 0.7, "fog_beta": 0.008}), "JPEG quality 15": ({}, {"jpeg_quality": 15}),
        "worst mix: 30 m/s + dark + fog": ({"v": (0, 0, 30)}, {"exposure_s": 1 / 60, "ev": -3.0, "gain_ev": 3.0, "iso": 3200.0, "fog_beta": 0.03, "fog_airlight": 0.78})}
def paths(s, n): return R / "images" / s / f"{n}.png", R / "labels/seg" / f"{s}__{n}.png", R / "labels/depth" / f"{s}__{n}.npy", R / "labels/corners_full" / f"{s}__{n}.json"
def degrade(s, n, cond, fx, seed):
    ip, _, dp, _ = paths(s, n); img = np.asarray(Image.open(ip).convert("RGB")).astype(np.float32) / 255
    if COND[cond] is None: return img
    motion, cfg = COND[cond]; return pipeline(img, np.load(dp).astype(np.float32), fx, motion, cfg, seed=seed)
def to_input(img): return torch.from_numpy(np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
def load_model(kind):
    m = PerceptionModel(kind, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{kind}/final.pt", map_location="cpu")); return m.to(dev).eval()
def predict(m, imgs):
    with torch.no_grad():
        o = m(normalize(torch.stack([to_input(i) for i in imgs]).to(dev)), with_teacher_heads=False)
    return o["seg"].argmax(1).cpu(), o["slots"].float().cpu()
names = sorted(p.stem for p in (R / "images" / S).glob("*__clear.png"))[::step]; W0, H0 = Image.open(paths(S, names[0])[0]).size; fx = (W0 / 2) / math.tan(math.radians(130 / 2)); print("frames", len(names), "image", W0, H0, "fx", round(fx, 1), flush=True)
KINDS = os.environ.get("KINDS", "vit,hybrid").split(","); models = {k: load_model(k) for k in KINDS}; res = {}
for cond in COND:
    st = {k: {"conf": torch.zeros(NC, NC, dtype=torch.long), "full": [0, 0, []], "part": [0, 0, []]} for k in models}
    for i in range(0, len(names), 16):
        chunk = names[i:i + 16]; imgs = [degrade(S, n, cond, fx, 1000 + i + j) for j, n in enumerate(chunk)]
        seg = torch.stack([torch.from_numpy(np.asarray(Image.open(paths(S, n)[1]).resize((SIZE, SIZE), Image.NEAREST)).copy()).long() for n in chunk]); v = seg != 255
        for k, m in models.items():
            pred, slots = predict(m, imgs); st[k]["conf"] += torch.bincount(seg[v] * NC + pred[v], minlength=NC * NC).view(NC, NC)
            for j, n in enumerate(chunk):
                t = slot_targets(json.load(open(paths(S, n)[3])))
                for q, e in match_gates(slots[j], t)[0]:        # order-free matching; a gate counts as found when its matched prediction is within 20 px mean corner error
                    c = t[q, 1:].view(4, 2) * torch.tensor([W_, H_]); part = bool(((c[:, 0] < 0) | (c[:, 0] >= W_) | (c[:, 1] < 0) | (c[:, 1] >= H_)).any()); b = st[k]["part" if part else "full"]; b[0] += 1
                    if e is not None and e < 20: b[1] += 1; b[2].append(e)
    res[cond] = {}
    for k in models:
        cf = st[k]["conf"]; inter = cf.diag().float(); gt = cf.sum(1).float(); un = gt + cf.sum(0).float() - inter; pres = gt > 0
        res[cond][k] = {"gate_iou": round((inter[GATE] / un[GATE]).item(), 3), "race_miou": round((inter[pres] / un[pres]).mean().item(), 3), "gate_recall": round(st[k]["full"][1] / max(st[k]["full"][0], 1), 3),
                        "corner_err_median_px": round(float(np.median(st[k]["full"][2])), 1) if st[k]["full"][2] else None, "partly_out_n": st[k]["part"][0], "partly_out_recall": round(st[k]["part"][1] / max(st[k]["part"][0], 1), 3)}
    print(f"{cond:34s} " + " | ".join(f"{k} IoU {res[cond][k]['gate_iou']:.3f} recall {res[cond][k]['gate_recall']:.2f} err {res[cond][k]['corner_err_median_px']}" for k in KINDS), flush=True)
Path("runs/eval").mkdir(parents=True, exist_ok=True); json.dump({"ckpt": CK, "frames": len(names), "step": step, "results": res}, open(f"runs/eval/robustness_sim_race{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1)
# ---------- pictures: label (left) versus vit prediction (right), one row per condition
VIS = ["clear (as rendered)", "blur forward 30 m/s", "blur forward 60 m/s", "low light (train: -3, gain +3)", "light -4 stops", "fog 0.05", "worst mix: 30 m/s + dark + fog"]
def partial_frames(s):
    out = []
    for n in sorted(p.stem for p in (R / "images" / s).glob("*__clear.png")):
        t = slot_targets(json.load(open(paths(s, n)[3])))
        for q in range(4):
            if t[q, 0] > 0.5:
                c = t[q, 1:].view(4, 2) * torch.tensor([W_, H_])
                if bool(((c[:, 0] < 0) | (c[:, 0] >= W_) | (c[:, 1] < 0) | (c[:, 1] >= H_)).any()) and int(t[:, 0].sum()) >= 1: out.append(n); break
    return out
def pick(s, partial):
    cands = partial_frames(s) if partial else [n for n in sorted(p.stem for p in (R / "images" / s).glob("*__clear.png")) if int(slot_targets(json.load(open(paths(s, n)[3])))[:, 0].sum()) >= 2]
    return cands[len(cands) // 2] if cands else None
font = ImageFont.truetype("/usr/share/fonts/noto/NotoSans-Bold.ttf", 15) if os.path.exists("/usr/share/fonts/noto/NotoSans-Bold.ttf") else ImageFont.load_default()
def panel(img, segmap, corners_gt, corners_pred, title, cw=320):
    base = Image.fromarray((img * 255).astype(np.uint8)).resize((cw, int(cw * H0 / W0)), Image.BILINEAR).convert("RGBA"); ov = Image.new("RGBA", base.size, (0, 0, 0, 0)); sm = Image.fromarray(segmap.astype(np.uint8)).resize(base.size, Image.NEAREST); a = np.asarray(sm)
    col = np.zeros((*a.shape, 4), np.uint8); col[a == GATE] = (40, 220, 80, 120); col[a == CLASSES.index("other_obstacle")] = (230, 60, 60, 120); ov = Image.alpha_composite(base, Image.fromarray(col)); d = ImageDraw.Draw(ov); sx, sy = base.size[0] / W_, base.size[1] / H_
    for c in corners_gt: d.line([(x * sx, y * sy) for x, y in c] + [(c[0][0] * sx, c[0][1] * sy)], fill=(255, 255, 255, 255), width=2)
    for c in corners_pred: d.line([(x * sx, y * sy) for x, y in c] + [(c[0][0] * sx, c[0][1] * sy)], fill=(255, 70, 40, 255), width=2)
    d.rectangle([0, 0, base.size[0], 20], fill=(0, 0, 0, 170)); d.text((4, 2), title, fill=(255, 255, 255, 255), font=font); return ov.convert("RGB")
Path("runs/eval/vis").mkdir(parents=True, exist_ok=True); mv = models[KINDS[0]]
for tag, s, partial in (("normal", "sim_race", False), ("partly_out_of_view", "sim_racep", True)):
    n = pick(s, partial)
    if n is None: continue
    gt = slot_targets(json.load(open(paths(s, n)[3]))); gcs = [(gt[q, 1:].view(4, 2) * torch.tensor([W_, H_])).tolist() for q in range(4) if gt[q, 0] > 0.5]; seg_gt = np.asarray(Image.open(paths(s, n)[1])); rows = []
    for cond in VIS:
        img = degrade(s, n, cond, fx, 7); pred, slots = predict(mv, [img]); pm = pred[0].numpy(); sc = slots[0, :, 0].sigmoid(); pcs = [(slots[0, q, 1:].view(4, 2) * torch.tensor([W_, H_])).tolist() for q in range(4) if sc[q] > 0.5]
        pr = np.asarray(Image.fromarray(pm.astype(np.uint8)).resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST)); v = seg_gt != 255; iou = ((pr == GATE) & (seg_gt == GATE) & v).sum() / max((((pr == GATE) | (seg_gt == GATE)) & v).sum(), 1)
        rows.append((panel(img, seg_gt, gcs, [], f"TRUTH  {cond}"), panel(img, pr, gcs, pcs, f"MODEL  gate IoU {iou:.2f}")))
    w, h = rows[0][0].size; sheet = Image.new("RGB", (2 * w + 6, len(rows) * (h + 4)), (20, 20, 20))
    for r, (a, b) in enumerate(rows): sheet.paste(a, (0, r * (h + 4))); sheet.paste(b, (w + 6, r * (h + 4)))
    sheet.save(f"runs/eval/vis/robustness_{tag}{os.environ.get('OUT_TAG', '')}.png"); print("saved", tag, n, sheet.size)
