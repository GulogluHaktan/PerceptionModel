"""Contact shadows of inserted objects on the ground of the 3DGS scene (plan: shadows). The footprint is the convex hull of the object corners projected along the
light direction onto the ground plane, drawn into the image, blurred, and limited to background pixels that lie on the ground plane (not on other objects).
Approximation: one directional light, no shadows cast by splat objects onto the inserted objects, no soft penumbra model beyond a blur."""
from __future__ import annotations

import itertools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .camera import Camera, project


def plane_depth(cam: Camera, c2w_cv, plane):
    """z-depth of the ground plane along every pixel ray (0 where the ray does not hit it in front of the camera)."""
    n, d = np.asarray(plane[0], float), float(plane[1]); c2w = np.asarray(c2w_cv, float)
    vv, uu = np.meshgrid(np.arange(cam.height), np.arange(cam.width), indexing="ij")
    dc = np.stack([(uu + 0.5 - cam.cx) / cam.fx, (vv + 0.5 - cam.cy) / cam.fx, np.ones_like(uu, dtype=float)], -1)
    dw = dc @ c2w[:3, :3].T; den = dw @ n
    t = -(n @ c2w[:3, 3] + d) / np.where(np.abs(den) < 1e-9, 1e-9, den)
    return np.where((t > 0) & (np.abs(den) > 1e-9), t, 0.0)


def _hull(pts):
    pts = sorted(map(tuple, pts))
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def box_corners(b):
    if "R" in b:
        r = np.asarray(b["R"], float)
    else:
        y = b["yaw"]; r = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1.0]])
    h = np.asarray(b["size"], float) / 2
    return np.array([np.asarray(b["center"], float) + r @ (np.array(s) * h) for s in itertools.product((-1, 1), repeat=3)])


def ground_shadow(cam: Camera, c2w_cv, boxes, plane, light_dir, bg_depth, blur=2.0, tol=0.08):
    """Float mask (H,W) in [0,1]. light_dir: direction the light travels (pointing down into the scene), world frame."""
    n, d = np.asarray(plane[0], float), float(plane[1]); ld = np.asarray(light_dir, float) / np.linalg.norm(light_dir)
    m = Image.new("L", (cam.width, cam.height), 0); dr = ImageDraw.Draw(m)
    if abs(n @ ld) < 1e-6:
        return np.zeros((cam.height, cam.width), np.float32)
    for b in boxes:
        c = box_corners(b); t = -(c @ n + d) / (n @ ld); foot = c + t[:, None] * ld                       # corners moved along the light onto the plane
        u, v, z = project(cam, c2w_cv, foot)
        if (z < 0.05).any():
            continue
        hull = _hull(np.c_[u, v])
        if len(hull) >= 3:
            dr.polygon(hull, fill=255)
    mask = np.asarray(m.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255
    pd = plane_depth(cam, c2w_cv, plane)
    on_ground = (pd > 0) & (bg_depth > 0) & (np.abs(bg_depth - pd) < np.maximum(tol, 0.03 * pd))                # only where the splat surface is the ground
    return mask * on_ground
