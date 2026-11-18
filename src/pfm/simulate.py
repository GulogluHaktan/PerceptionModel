"""One generated frame: background (3DGS render) + foreground (objects) -> depth-aware compose -> effects, plus exact gate-corner labels (plan steps 4, 5, 7).
`background` is a dict with rgb (H,W,3 float), depth (H,W metres, 0 = empty), label (H,W uint8). The foreground comes from pfm.proxy_fg now and from Isaac Sim renders later."""
from __future__ import annotations

import numpy as np

from .camera import Camera, project
from .compose import compose
from .effects import pipeline
from .proxy_fg import render_boxes


def project_corners(cam: Camera, c2w_cv, corners3d, comp_depth, tol=0.1):
    """Pixel coordinates of 3D corners with a visibility flag (inside the image and not hidden behind a nearer surface in the composed depth)."""
    u, v, z = project(cam, c2w_cv, corners3d); out = []
    for ui, vi, zi in zip(u, v, z):
        inside = zi > 0.05 and 0 <= ui < cam.width and 0 <= vi < cam.height
        vis = bool(inside and (comp_depth[int(vi), int(ui)] <= 0 or abs(comp_depth[int(vi), int(ui)] - zi) < max(tol, 0.03 * zi)))
        out.append([round(float(ui), 2), round(float(vi), 2), vis])
    return out


def generate_frame(cam: Camera, c2w_cv, background: dict, boxes, gates3d, effects_cfg, motion, seed=0, device="cuda", fg=None, shadow=None, depth_scale=1.0):
    """depth_scale: metres per scene unit (background/foreground depth are composited in scene units, converted for the effects).
    shadow: optional (plane, light_dir, boxes) for ground contact shadows (pfm.shadow).
    fg: optional (rgb, depth, label, alpha) from an external renderer (Isaac Sim); otherwise the analytic proxy renders `boxes`."""
    fg_rgb, fg_depth, fg_label, fg_alpha = fg if fg is not None else render_boxes(cam, c2w_cv, boxes, device="cpu")
    sh = None
    if shadow is not None:                       # shadow = (plane, light_dir, boxes_that_cast)
        from .shadow import ground_shadow
        sh = ground_shadow(cam, c2w_cv, shadow[2], shadow[0], shadow[1], background["depth"])
    rgb, depth, label, vis = compose(background["rgb"], background["depth"], background["label"], fg_rgb, fg_depth, fg_label, fg_alpha, shadow=sh)
    corners = [project_corners(cam, c2w_cv, g, depth) for g in gates3d]                      # in scene units, like the renderers
    depth = (depth * depth_scale).astype(np.float32)                                          # scene units -> metres for the effects and the files
    d_eff = np.where(depth > 0, depth, np.inf).astype(np.float32)
    final = pipeline(rgb, d_eff, cam.fx, motion, effects_cfg, seed=seed, device=device)
    return final, depth, label, corners, vis
