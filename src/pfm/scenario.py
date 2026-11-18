"""Strict scenario schema (plan step 5): what an LLM scenario manager (or a human) may fill in. Nothing else is accepted.
`validate_scenario` fills defaults, rejects unknown keys / wrong types / out-of-range values, validates the effects block with
pfm.effects.validate_cfg, and returns the normalised scenario; `scenario_hash` is the provenance id written next to every generated frame."""
from __future__ import annotations

import hashlib
import json

from .effects import validate_cfg

OBJECT_TYPES = {"gate", "pole", "tree", "box"}
TOP = {"scene", "objects", "trajectory", "camera", "effects", "seed"}
SCENE = {"id": str, "meters_per_unit": float, "up": list, "ground_z": float}      # ground_z (optional): height of the surface to stand objects on, in the up-aligned frame in scene units; default = RANSAC ground plane
OBJ = {"type": str, "pos": list, "yaw_deg": float, "size": list, "on_ground": bool}
TRAJ = {"kind": str, "frames": int, "fps": float, "max_speed_mps": float, "center": list, "radius_m": float, "altitude_m": float}
CAM = {"width": int, "height": int, "hfov_deg": float}
TRAJ_KINDS = {"spiral", "line"}


def _check(d, schema, name, defaults):
    bad = set(d) - set(schema)
    if bad:
        raise ValueError(f"{name}: unknown keys {sorted(bad)}")
    out = {**defaults, **d}
    for k, t in schema.items():
        if k in out and not isinstance(out[k], (t, int) if t is float else t):
            raise ValueError(f"{name}.{k} must be {t.__name__}, got {type(out[k]).__name__}")
    return out


def _rng(name, v, lo, hi):
    if not lo <= v <= hi:
        raise ValueError(f"{name}={v} outside [{lo}, {hi}]")


def validate_scenario(sc: dict) -> dict:
    bad = set(sc) - TOP
    if bad:
        raise ValueError(f"unknown top-level keys {sorted(bad)}")
    scene = _check(sc.get("scene", {}), SCENE, "scene", {"id": "unnamed", "meters_per_unit": 1.0, "up": [0.0, 0.0, 1.0]})
    _rng("scene.meters_per_unit", float(scene["meters_per_unit"]), 1e-3, 1e3)
    if len(scene["up"]) != 3 or sum(float(v) ** 2 for v in scene["up"]) < 1e-12:
        raise ValueError("scene.up must be 3 numbers, not all zero")
    objs = []
    if len(sc.get("objects", [])) > 50:
        raise ValueError("at most 50 objects")
    for i, o in enumerate(sc.get("objects", [])):
        o = _check(o, OBJ, f"objects[{i}]", {"yaw_deg": 0.0, "size": [3.0, 3.0], "on_ground": True})
        if o["type"] not in OBJECT_TYPES:
            raise ValueError(f"objects[{i}].type must be one of {sorted(OBJECT_TYPES)}")
        if len(o["pos"]) != 3 or len(o["size"]) not in (2, 3):
            raise ValueError(f"objects[{i}]: pos needs 3 numbers, size 2 or 3")
        for v in list(o["pos"]) + list(o["size"]):
            _rng(f"objects[{i}] value", float(v), -1e4, 1e4)
        objs.append(o)
    tr = _check(sc.get("trajectory", {}), TRAJ, "trajectory",
                {"kind": "spiral", "frames": 120, "fps": 30.0, "max_speed_mps": 12.0, "center": [0.0, 0.0, 0.5], "radius_m": 4.0, "altitude_m": 1.8})
    if tr["kind"] not in TRAJ_KINDS:
        raise ValueError(f"trajectory.kind must be one of {sorted(TRAJ_KINDS)}")
    _rng("trajectory.frames", tr["frames"], 2, 5000); _rng("trajectory.fps", float(tr["fps"]), 1, 240)
    _rng("trajectory.max_speed_mps", float(tr["max_speed_mps"]), 0, 40); _rng("trajectory.radius_m", float(tr["radius_m"]), 0.1, 500)
    cam = _check(sc.get("camera", {}), CAM, "camera", {"width": 640, "height": 480, "hfov_deg": 90.0})
    _rng("camera.width", cam["width"], 64, 4096); _rng("camera.height", cam["height"], 64, 4096); _rng("camera.hfov_deg", float(cam["hfov_deg"]), 20, 140)
    seed = sc.get("seed", 0)
    if not isinstance(seed, int):
        raise ValueError("seed must be an int")
    return {"scene": scene, "objects": objs, "trajectory": tr, "camera": cam, "effects": validate_cfg(sc.get("effects", {})), "seed": seed}


def scenario_hash(sc: dict) -> str:
    return hashlib.sha1(json.dumps(validate_scenario(sc), sort_keys=True).encode()).hexdigest()[:12]
