"""Turns the Isaac city frames (runs/city_job/out/<run>) into a labelled multi-condition set: for every frame, five conditions made with the exact depth
(clear, fast motion blur, fog, rain, low light). Labels are copied unchanged (effects do not move pixels except blur, which is the point).
Output: <out>/images/<run>__<k>__<cond>.png, <out>/labels/<same>.png (class ids, 255 = unlabelled), <out>/index.json.
Usage: python scripts/city_effects.py <out_dir> <run_dir> [<run_dir> ...]"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import math, zlib
from PIL import Image
sys.path.insert(0, "src")
from pfm.effects import pipeline

CONDS = {"clear": {"exposure_s": 1 / 500}, "blur": {"exposure_s": 1 / 40},
         "fog": {"exposure_s": 1 / 250, "fog_beta": 0.03, "fog_airlight": 0.78},
         "rain": {"exposure_s": 1 / 250, "rain_density": 0.7, "fog_beta": 0.008},
         "lowlight": {"exposure_s": 1 / 100, "ev": -3.0, "gain_ev": 3.0, "iso": 3200.0}}
def _rcv(m):
    return np.stack([m[:3, 0], -m[:3, 1], -m[:3, 2]], 1)                                                         # OpenGL (r, u, -f) -> OpenCV axes (right, down, forward)


def cam_motion(frames, k):
    """Velocity and angular rate in the camera frame from consecutive poses (speed_mps gives the time step: dt = |dp| / speed). A camera that circles an object while looking at it moves sideways and turns: forward-only motion would blur the wrong way."""
    k0 = min(k, len(frames) - 2); a, b = np.array(frames[k0]["transform_matrix"]), np.array(frames[k0 + 1]["transform_matrix"]); sp = float(frames[k]["speed_mps"])
    dp = b[:3, 3] - a[:3, 3]; dist = float(np.linalg.norm(dp))
    if dist < 1e-9 or sp < 1e-9: return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    dt = dist / sp; Ra, Rb = _rcv(a), _rcv(b); v = Ra.T @ dp / dt; Rr = Ra.T @ Rb
    ang = math.acos(max(-1.0, min(1.0, (np.trace(Rr) - 1) / 2)))
    w = np.zeros(3) if ang < 1e-9 else ang / (2 * math.sin(ang)) * np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / dt
    return tuple(map(float, v)), tuple(map(float, w))


INDOOR_SKIP = {"rain", "fog"}
INDOOR_CFG = {"lowlight": {"exposure_s": 1 / 100, "ev": -3.0, "gain_ev": 1.0, "iso": 3200.0}}               # net 2 stops darker (gain 3 would cancel the ev and only add noise)
out = Path(sys.argv[1]); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True); index = []
for rd in map(Path, sys.argv[2:]):
    meta = json.load(open(rd / "poses.json")); fx = meta["fx"]; corners = json.load(open(rd / "corners.json")) if (rd / "corners.json").exists() else None   # racing runs: gate-corner labels per frame
    for k, fr in enumerate(meta["frames"]):
        img = np.asarray(Image.open(rd / "rgb" / f"{k:04d}.png").convert("RGB")).astype(np.float32) / 255; dep = np.load(rd / "depth" / f"{k:04d}.npy")
        m = np.array(fr["transform_matrix"]); r_cv = np.stack([m[:3, 0], -m[:3, 1], m[:3, 2] * -1], 1)             # OpenGL (r, u, -f) -> OpenCV axes (right, down, forward)
        motion = {"v": (0.0, 0.0, float(fr["speed_mps"])), "w": (0.0, 0.0, 0.0), "down": tuple(r_cv.T @ np.array([0, 0, -1.0]))}
        if meta["traj"] in ("indoor", "outdoor", "race") and len(meta["frames"]) > 1:
            motion["v"], motion["w"] = cam_motion(meta["frames"], k)
        for ci, (cn, cfg) in enumerate(CONDS.items()):
            if (meta["traj"] == "indoor" or meta.get("setting") == "hall") and cn in INDOOR_SKIP:                                                        # no weather inside a room
                continue
            res = pipeline(img, dep, fx, motion, INDOOR_CFG.get(cn, cfg) if meta["traj"] == "indoor" else cfg, seed=zlib.crc32(f"{rd.name}:{k}:{ci}".encode()))
            sid = f"{rd.name}__{k:04d}__{cn}"; Image.fromarray((res * 255).astype(np.uint8)).save(out / "images" / f"{sid}.png"); shutil.copy(rd / "label" / f"{k:04d}.png", out / "labels" / f"{sid}.png")
            index.append({"id": sid, "run": rd.name, "frame": k, "cond": cn, "light": meta["light"], "traj": meta["traj"], "speed_mps": fr["speed_mps"], **({"gates": corners[k]} if corners else {})})
json.dump(index, open(out / "index.json", "w")); print(len(index), "images ->", out)
