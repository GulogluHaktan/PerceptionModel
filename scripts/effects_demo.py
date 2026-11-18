"""Apply the camera/weather effects to one rendered frame (RGB + depth) and write a labelled grid.
Usage: python scripts/effects_demo.py <orbit_dir> <frame_id> <out.png>   (orbit_dir from orin_3dgs_orbit.py: rgb/, depth/, transforms_orbit.json)"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "src")
from pfm.effects import pipeline

root, fid, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
img = np.asarray(Image.open(root / "rgb" / f"{fid}.png").convert("RGB")).astype(np.float32) / 255
depth = np.asarray(Image.open(root / "depth" / f"{fid}.png")).astype(np.float32) / 1000       # scene units, treated as metres
depth[depth < 0.05] = np.inf                                                                  # no surface
W = img.shape[1]; fx = 0.5 * W / math.tan(0.5 * json.load(open(root / "transforms_orbit.json"))["camera_angle_x"])
cases = [
    ("1 temiz", {}, {}),
    ("2 yavas hiz 3 m/s, 1/250 s", {"exposure_s": 1 / 250}, {"v": (3, 0, 0)}),
    ("3 hizli 12 m/s, 1/100 s", {"exposure_s": 1 / 100}, {"v": (12, 0, 2)}),
    ("4 rolling shutter 12 m/s", {"exposure_s": 0.0005, "readout_s": 1 / 30}, {"v": (12, 0, 0)}),
    ("5 donus 1.5 rad/s, 1/60 s", {"exposure_s": 1 / 60}, {"w": (0, 1.5, 0)}),
    ("6 sis beta 0.35", {"fog_beta": 0.35}, {}),
    ("7 yogun sis beta 0.8", {"fog_beta": 0.8}, {}),
    ("8 dusuk isik -4 EV, ISO 1600", {"ev": -4, "gain_ev": 4, "iso": 1600}, {}),
    ("9 yagmur + hafif sis", {"rain_density": 0.6, "fog_beta": 0.1}, {}),
    ("10 hepsi: hiz+sis+yagmur+gurultu+JPEG", {"exposure_s": 1 / 100, "fog_beta": 0.3, "rain_density": 0.4, "ev": -2, "gain_ev": 2, "iso": 800, "vignette": 0.5, "jpeg_quality": 60}, {"v": (12, 0, 2)}),
]
S = 360; cols = 5
sheet = Image.new("RGB", (S * cols, (S + 22) * 2), (255, 255, 255)); d = ImageDraw.Draw(sheet)
for k, (name, cfg, mo) in enumerate(cases):
    r = pipeline(img, depth, fx, mo, cfg, seed=k)
    im = Image.fromarray((r * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)
    x, y = (k % cols) * S, (k // cols) * (S + 22)
    d.text((x + 4, y + 4), name, fill=(0, 0, 0)); sheet.paste(im, (x, y + 22))
sheet.save(out); print("saved", out, sheet.size)
