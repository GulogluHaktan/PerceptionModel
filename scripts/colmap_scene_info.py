"""Suggest scenario fields for a COLMAP scene: up vector (from the camera poses), scene centre and camera distance in an up-aligned frame, hfov.
Usage: python scripts/colmap_scene_info.py <scene_dir with sparse/0> <meters_per_unit ASSUMPTION> [width=640]
The scale is NOT estimated here: pass a guess (or a real value from pfm.scale) and treat results in metres accordingly."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from pfm.colmap_io import camera_to_world_cv, read_model
from pfm.frames import estimate_up_from_poses, rotation_from_up

root = Path(sys.argv[1]); mpu = float(sys.argv[2]); W = int(sys.argv[3]) if len(sys.argv) > 3 else 640
cams, imgs, xyz = read_model(root / "sparse" / "0"); poses = [camera_to_world_cv(i) for i in imgs]
up = estimate_up_from_poses(poses); R = rotation_from_up(up)
pts = (xyz @ R.T); centres = np.array([(R @ p[:3, 3]) for p in poses])
core = pts[(np.abs(pts - np.median(pts, 0)) < 3 * np.median(np.abs(pts - np.median(pts, 0)), 0) + 1e-9).all(1)]            # robust core of the SfM cloud
c = np.median(core, 0); dist = np.linalg.norm(centres[:, :2] - c[:2], axis=1); h = centres[:, 2] - c[2]
w0, h0, fx, _, _, _ = cams[imgs[0][3]]
out = {"scene": {"up": [round(float(x), 5) for x in up], "meters_per_unit": mpu},
       "centre_aligned_units": [round(float(x), 3) for x in c], "camera_distance_units_median": round(float(np.median(dist)), 3),
       "camera_height_above_centre_units_median": round(float(np.median(h)), 3), "hfov_deg": round(math.degrees(2 * math.atan(0.5 * w0 / fx)), 2),
       "suggest": {"trajectory": {"center": [round(float(x) * mpu, 3) for x in c], "radius_m": round(float(np.median(dist)) * mpu, 3), "altitude_m": round(float(c[2] * mpu + np.median(h) * mpu), 3)},
                   "camera": {"width": W, "height": int(round(W * h0 / w0)), "hfov_deg": round(math.degrees(2 * math.atan(0.5 * w0 / fx)), 2)}}}
print(json.dumps(out, indent=1))
