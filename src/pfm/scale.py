"""Metric scale for scenes whose unit is not the metre (COLMAP / Blender scenes) (plan step 2). Two ways to get `meters_per_unit`:
1. a known real distance between two scene points (a gate width, a marked baseline);
2. telemetry: camera centres in scene units versus the same positions in metres (GPS/VIO/odometry): similarity alignment (Umeyama) gives scale, and
   the residual tells how trustworthy it is (a large residual means the telemetry and the reconstruction disagree)."""
from __future__ import annotations

import numpy as np


def scale_from_distance(p_a, p_b, known_m: float) -> float:
    d = float(np.linalg.norm(np.asarray(p_a, float) - np.asarray(p_b, float)))
    if d < 1e-9 or known_m <= 0:
        raise ValueError("need two distinct points and a positive distance")
    return known_m / d


def umeyama(src, dst):
    """Similarity transform dst ~ s * R @ src + t (least squares). Returns s, R, t, rms residual."""
    x, y = np.asarray(src, float), np.asarray(dst, float)
    mx, my = x.mean(0), y.mean(0); xc, yc = x - mx, y - my
    u, d, vt = np.linalg.svd(yc.T @ xc / len(x))
    s_ = np.eye(3); s_[2, 2] = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    r = u @ s_ @ vt; var = (xc ** 2).sum() / len(x); sc = float(np.trace(np.diag(d) @ s_) / var); t = my - sc * r @ mx
    res = float(np.sqrt(((y - (sc * (r @ x.T).T + t)) ** 2).sum(1).mean()))
    return sc, r, t, res


def scale_from_telemetry(cam_centres_units, positions_m):
    """Returns (meters_per_unit, rms_residual_m). Needs at least 4 non-collinear matched positions."""
    x = np.asarray(cam_centres_units, float)
    if len(x) < 4 or np.linalg.matrix_rank(x - x.mean(0)) < 2:
        raise ValueError("need at least 4 matched, non-collinear positions")
    sc, _, _, res = umeyama(x, positions_m)
    return sc, res
