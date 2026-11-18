"""Scenario objects -> boxes / gate corners in scene units (shared by the proxy renderer, the Isaac export and the dataset script)."""
from __future__ import annotations

import numpy as np

from .ground import place_on_plane
from .proxy_fg import gate_boxes, gate_corners


def build_objects(sc: dict, plane, mpu: float = 1.0):
    """sc: validated scenario. Returns (boxes, gates3d). Sizes in the scenario are metres, positions are scene units (x, y on the ground, z = offset above it)."""
    boxes, gates = [], []
    for o in sc["objects"]:
        x, y, zoff = o["pos"]; pos = place_on_plane(plane, (x, y)) + np.array([0, 0, zoff])
        if o["type"] == "gate":
            w, h = o["size"][0] / mpu, o["size"][1] / mpu
            boxes += gate_boxes(pos, o["yaw_deg"], w, h); gates.append(gate_corners(pos, o["yaw_deg"], w, h))
        else:
            r, h = (o["size"][0] / mpu, o["size"][1] / mpu) if len(o["size"]) == 2 else (o["size"][0] / mpu / 2, o["size"][2] / mpu)
            col, lab = ((0.3, 0.3, 0.32), 2) if o["type"] == "pole" else ((0.15, 0.4, 0.15), 2) if o["type"] == "tree" else ((0.5, 0.5, 0.55), 5)
            boxes.append({"center": pos + np.array([0, 0, h / 2]), "size": np.array([2 * r, 2 * r, h]), "yaw": float(np.radians(o.get("yaw_deg", 0.0))), "color": col, "label": lab})
    return boxes, gates
