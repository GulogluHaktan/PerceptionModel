"""Appearance + lens augmentation of EXISTING sim racing frames, labels kept consistent (no re-render):
 - gate recolour: pixels of the seg class `gate` get a random hue/saturation/darkening (palette includes TII-like purple, black, orange, red, blue),
 - barrel (fisheye-like) warp  src = z*out*(1+k r^2), z = 1/(1+2k) so the whole source fits: image bilinear, seg/depth nearest, gate corners moved with the inverse map (fixed-point).
Writes data/mix_aug/images/sim_raceaug/<id>.png and labels/{seg,depth,corners_full}/sim_raceaug__<id>.* . No corner heatmaps (no corner loss on these).
Usage: python scripts/make_aug_sim.py [n=1500]"""
import colorsys, json, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import CLASSES
GATE = CLASSES.index("gate"); S = Path("data/train_sim"); O = Path("data/mix_aug"); n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; rng = random.Random(1)
files = sorted(p for d in ("sim_race", "sim_racep") for p in (S / "images" / d).glob("*.png") if p.stem.endswith(("clear", "blur", "lowlight"))); rng.shuffle(files)
PAL = [(0.75, 0.35, 0.9), (0.72, 0.6, 0.5), (0.0, 0.0, 0.18), (0.08, 0.9, 0.9), (0.0, 0.85, 0.8), (0.6, 0.8, 0.8), (0.33, 0.7, 0.7), (0.95, 0.4, 0.6)]   # (hue, sat, value-scale)
for sub in ("images/sim_raceaug", "labels/seg", "labels/depth", "labels/corners_full"): (O / sub).mkdir(parents=True, exist_ok=True)
done = 0
for p in files:
    if done >= n: break
    sid = f"{p.parent.name}__{p.stem}"; sp, dp, cp = S / f"labels/seg/{sid}.png", S / f"labels/depth/{sid}.npy", S / f"labels/corners_full/{sid}.json"
    if not (sp.exists() and dp.exists() and cp.exists()): continue
    img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255; H, W = img.shape[:2]; seg = np.asarray(Image.open(sp)); dep = np.load(dp).astype(np.float32)
    if seg.shape != (H, W) or dep.shape != (H, W): continue
    m = seg == GATE
    if m.sum() > 50:
        h, s, v = rng.choice(PAL); h = (h + rng.uniform(-0.03, 0.03)) % 1; s = min(max(s * rng.uniform(0.8, 1.1), 0), 1); sc = rng.uniform(0.7, 1.3)
        lum = img[m].mean(1, keepdims=True); shade = np.clip(lum / max(float(lum.mean()), 1e-3), 0.5, 1.5); tint = np.array(colorsys.hsv_to_rgb(h, s, 1.0), np.float32)
        img[m] = np.clip(tint[None] * min(v * sc, 1.0) * 0.8 * shade, 0, 1)
    k = rng.uniform(0.0, 0.7) if rng.random() < 0.8 else 0.0; z = 1 / (1 + 2 * k)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"); r2 = xs ** 2 + ys ** 2; f = z * (1 + k * r2); grid = torch.stack([xs * f, ys * f], -1)[None]
    wi = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0].permute(1, 2, 0).numpy()
    ws = F.grid_sample(torch.from_numpy(seg.astype(np.float32))[None, None], grid, mode="nearest", padding_mode="border", align_corners=True)[0, 0].numpy().astype(np.uint8)
    wd = F.grid_sample(torch.from_numpy(dep)[None, None], grid, mode="nearest", padding_mode="border", align_corners=True)[0, 0].numpy().astype(np.float16)
    rows = json.load(open(cp)); new = []
    for r in rows:
        c = np.array(r["corners_px"], np.float32) / np.array([256.0, 192.0]) * 2 - 1; o = c.copy()
        for _ in range(12): o = c / (z * (1 + k * (o ** 2).sum(1, keepdims=True)))
        r2_ = dict(r); r2_["corners_px"] = [[round(float((a + 1) / 2 * 256), 2), round(float((b + 1) / 2 * 192), 2)] for a, b in o]; new.append(r2_)
    nid = f"{p.stem}"; Image.fromarray((np.clip(wi, 0, 1) * 255).astype(np.uint8)).save(O / f"images/sim_raceaug/{nid}.png"); Image.fromarray(ws).save(O / f"labels/seg/sim_raceaug__{nid}.png"); np.save(O / f"labels/depth/sim_raceaug__{nid}.npy", wd); json.dump(new, open(O / f"labels/corners_full/sim_raceaug__{nid}.json", "w")); done += 1
print("wrote", done)
