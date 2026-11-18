"""Light-level augmentation of existing sim racing frames for the stress-test gaps (over-exposure, deep dark): effects.pipeline with ev in [+0.5,+2] (40%) or [-4,-1.5] with gain/iso (60%) on the clear frames; labels are the same files (symlinked under the new ids).
Writes data/mix_light/images/sim_racelight/<set>_<stem>.png and links labels into data/mix/labels. Usage: python scripts/make_light_aug_sim.py [n=1500]"""
import math, os, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "src")
from pfm.effects import pipeline
S = Path("data/train_sim"); O = Path("data/mix_light"); L = Path("data/mix/labels"); n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; rng = random.Random(4)
files = sorted(p for d in ("sim_race", "sim_racep") for p in (S / "images" / d).glob("*__clear.png")); rng.shuffle(files); (O / "images/sim_racelight").mkdir(parents=True, exist_ok=True); done = 0
for p in files:
    if done >= n: break
    sid = f"{p.parent.name}__{p.stem}"; dp = S / f"labels/depth/{sid}.npy"
    if not dp.exists(): continue
    img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255; H, W = img.shape[:2]; fx = (W / 2) / math.tan(math.radians(65))
    if rng.random() < 0.4: cfg = {"ev": rng.uniform(0.5, 2.0), "exposure_s": 1 / 100}
    else: cfg = {"ev": rng.uniform(-4.0, -1.5), "gain_ev": rng.uniform(0, 4), "iso": rng.choice([800.0, 1600.0, 3200.0]), "exposure_s": 1 / 100}
    out = pipeline(img, np.load(dp).astype(np.float32), fx, {}, cfg, seed=done); nid = f"{p.parent.name}_{p.stem}"
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(O / f"images/sim_racelight/{nid}.png")
    for sub, ext in (("seg", "png"), ("corners", "npy"), ("corners_full", "json"), ("depth", "npy")):
        f = S / f"labels/{sub}/{sid}.{ext}"; d = L / sub / f"sim_racelight__{nid}.{ext}"
        if f.exists() and not d.exists() and not d.is_symlink(): d.parent.mkdir(parents=True, exist_ok=True); os.symlink(f.resolve(), d)
    done += 1
print("wrote", done)
