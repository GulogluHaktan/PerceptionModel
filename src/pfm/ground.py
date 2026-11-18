"""Ground plane stand-in for the collision mesh (plan step 3): RANSAC plane through the Gaussian centres (or any 3D points), and object placement on it.
A TSDF mesh from rendered depth is the later, better version. The plane is returned as (unit normal pointing UP, offset d) with n.x + d = 0."""
from __future__ import annotations

import numpy as np


def fit_ground_plane(points, up=(0.0, 0.0, 1.0), iters=300, thresh=0.05, min_up_cos=0.8, seed=0):
    """Largest plane whose normal is within acos(min_up_cos) of `up`. points (N,3)."""
    p = np.asarray(points, float); p = p[np.isfinite(p).all(1)]
    rng = np.random.default_rng(seed); up = np.asarray(up, float) / np.linalg.norm(up)
    best, best_n = (None, -1)
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n = n / L
        if n @ up < 0:
            n = -n
        if n @ up < min_up_cos:
            continue
        d = -n @ a; cnt = int((np.abs(p @ n + d) < thresh).sum())
        if cnt > best_n:
            best, best_n = (n, d), cnt
    if best is None:
        raise ValueError("no ground-like plane found")
    n, d = best; inl = np.abs(p @ n + d) < thresh                           # refit on the inliers (least squares)
    q = p[inl]; c0 = q.mean(0); u, s, vt = np.linalg.svd(q - c0, full_matrices=False); n2 = vt[2] if vt[2] @ up > 0 else -vt[2]
    return n2, float(-n2 @ c0), int(inl.sum())


def place_on_plane(plane, xy, up=(0.0, 0.0, 1.0)):
    """3D point on the plane above/below the horizontal position xy (x, y with z the up axis)."""
    n, d = plane[0], plane[1]
    z = -(n[0] * xy[0] + n[1] * xy[1] + d) / n[2]
    return np.array([xy[0], xy[1], z])
