"""Rain-only comparison on one rendered frame. Usage: python scripts/rain_demo.py <orbit_dir> <frame_id> <out.png>"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "src")
from pfm.effects import pipeline
root, fid, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
img = np.asarray(Image.open(root / "rgb" / f"{fid}.png").convert("RGB")).astype(np.float32) / 255
depth = np.asarray(Image.open(root / "depth" / f"{fid}.png")).astype(np.float32) / 1000; depth[depth < 0.05] = np.inf
meta = json.load(open(root / "transforms_orbit.json")); W = img.shape[1]; fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
c2w = np.array(next(f for f in meta["frames"] if f["file_path"].endswith(fid))["transform_matrix"]); c2w[:3, 1:3] *= -1      # OpenGL -> OpenCV
down = tuple(c2w[:3, :3].T @ np.array([0, 0, -1.0]))                                                                         # gravity in the camera frame (world is Z-up)
img = img * 0.82                                                                                                              # overcast scene, so the white background is not clipped
cases = [("hafif yagmur, duran kamera, 1/250 s", {"rain_density": 0.25, "exposure_s": 1 / 250}, {}),
         ("siddetli yagmur, duran kamera, 1/60 s", {"rain_density": 0.9, "exposure_s": 1 / 60}, {}),
         ("siddetli yagmur, 1/1000 s (kisa pozlama)", {"rain_density": 0.9, "exposure_s": 1 / 1000}, {}),
         ("siddetli + yana 12 m/s ucus, 1/100 s", {"rain_density": 0.9, "exposure_s": 1 / 100}, {"v": (12, 0, 0)}),
         ("siddetli + ruzgar 8 m/s + sis", {"rain_density": 0.9, "exposure_s": 1 / 100, "wind_x": -8, "fog_beta": 0.12}, {}),
         ("siddetli + ileri 15 m/s ucus, 1/100 s", {"rain_density": 0.9, "exposure_s": 1 / 100}, {"v": (0, 0, 15)})]
S = 480; sheet = Image.new("RGB", (S * 3, (S + 22) * 2), (255, 255, 255)); d = ImageDraw.Draw(sheet)
for k, (name, cfg, mo) in enumerate(cases):
    r = pipeline(img, depth, fx, {**mo, "down": down}, cfg, seed=k, device="cpu")
    x, y = (k % 3) * S, (k // 3) * (S + 22); d.text((x + 4, y + 4), name, fill=(0, 0, 0))
    sheet.paste(Image.fromarray((r * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS), (x, y + 22))
sheet.save(out); print("saved", out)
