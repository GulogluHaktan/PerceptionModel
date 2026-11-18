"""Scenes that are not Z-up (COLMAP worlds): everything scenario-related (trajectory, ground, objects) is built in an ALIGNED frame where the scene's up
vector is +Z, then moved to the scene frame with a pure rotation (aligned -> scene = R^T). Renderers (gsplat, proxy, shadow) work in the scene frame."""
from __future__ import annotations

import math

import numpy as np


def rotation_from_up(up) -> np.ndarray:
    """3x3 rotation R (scene -> aligned) with R @ up = +Z (Rodrigues)."""
    u = np.asarray(up, float); u = u / np.linalg.norm(u); z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z); c = float(u @ z); s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def estimate_up_from_poses(c2w_cv_list) -> np.ndarray:
    """Cameras are usually held upright: the world 'up' is minus the mean camera y axis (OpenCV y points down)."""
    y = np.mean([np.asarray(m, float)[:3, 1] for m in c2w_cv_list], axis=0)
    return -y / np.linalg.norm(y)


def aligned_to_scene(R_scene_to_aligned) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = np.asarray(R_scene_to_aligned, float).T
    return T


def yaw_matrix(yaw: float) -> np.ndarray:
    return np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])


def transform_boxes(boxes, T):
    """Boxes (yaw-only or with a full 'R') -> boxes in the target frame with a full 'R'."""
    out = []
    for b in boxes:
        R = np.asarray(b["R"], float) if "R" in b else yaw_matrix(b["yaw"])
        nb = dict(b); nb["R"] = T[:3, :3] @ R; nb["center"] = T[:3, :3] @ np.asarray(b["center"], float) + T[:3, 3]; out.append(nb)
    return out
