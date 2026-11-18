"""Combined-degradation augmentation of existing sim racing frames (stress-test gap: fog, worst mix): effects.pipeline with random combinations of forward motion blur (15-50 m/s), fog (beta 0.015-0.06), rain, light level (ev -4..+1.5 with gain), JPEG; labels reused (symlinked under new ids).
Writes data/mix_mix/images/sim_racemix/<set>_<stem>.png. Usage: python scripts/make_mix_aug_sim.py [n=1500]"""
import math, os, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "src")
from pfm.effects import pipeline
S = Path("data/train_sim"); O = Path("data/mix_mix"); L = Path("data/mix/labels"); n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; rng = random.Random(9)
files = sorted(p for d in ("sim_race", "sim_racep") for p in (S / "images" / d).glob("*__clear.png")); rng.shuffle(files); (O / "images/sim_racemix").mkdir(parents=True, exist_ok=True); done = 0
for p in files:
    if done >= n: break
    sid = f"{p.parent.name}__{p.stem}"; dp = S / f"labels/depth/{sid}.npy"
    if not dp.exists(): continue
    img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255; H, W = img.shape[:2]; fx = (W / 2) / math.tan(math.radians(65))
    cfg = {"exposure_s": 1 / rng.choice([40, 60, 100, 250])}; motion = {}
    if rng.random() < 0.7: motion = {"v": (0, 0, rng.uniform(15, 50))}
    if rng.random() < 0.6: cfg.update(fog_beta=rng.uniform(0.015, 0.06), fog_airlight=rng.uniform(0.6, 0.85))
    if rng.random() < 0.3: cfg.update(rain_density=rng.uniform(0.3, 0.9), fog_beta=cfg.get("fog_beta", 0.008))
    r = rng.random()
    if r < 0.45: cfg.update(ev=rng.uniform(-4.0, -1.0), gain_ev=rng.uniform(0, 4), iso=rng.choice([800.0, 1600.0, 3200.0]))
    elif r < 0.6: cfg.update(ev=rng.uniform(0.5, 1.5))
    if rng.random() < 0.3: cfg.update(jpeg_quality=rng.choice([15, 30, 50]))
    out = pipeline(img, np.load(dp).astype(np.float32), fx, motion, cfg, seed=done); nid = f"{p.parent.name}_{p.stem}"
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(O / f"images/sim_racemix/{nid}.png")
    for sub, ext in (("seg", "png"), ("corners", "npy"), ("corners_full", "json"), ("depth", "npy")):
        f = S / f"labels/{sub}/{sid}.{ext}"; d = L / sub / f"sim_racemix__{nid}.{ext}"
        if f.exists() and not d.exists() and not d.is_symlink(): d.parent.mkdir(parents=True, exist_ok=True); os.symlink(f.resolve(), d)
    done += 1
print("wrote", done)
