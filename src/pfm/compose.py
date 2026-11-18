"""Depth-aware compositing of a rendered foreground (Isaac Sim objects) over a rendered background (3DGS scene) (plan step 4).
Both sides are rendered from the same camera pose (src/pfm/camera.py). Per pixel the nearer surface wins; foreground alpha gives soft edges.
Labels: foreground label where the foreground wins, background label elsewhere (255 = none). Optional `shadow` (H,W in 0..1) darkens the
background before blending (a contact shadow; real shadow casting from the 3DGS scene is NOT modelled)."""
from __future__ import annotations

import numpy as np


def compose(bg_rgb, bg_depth, bg_label, fg_rgb, fg_depth, fg_label, fg_alpha=None, shadow=None, shadow_strength=0.45, depth_eps=0.02):
    """rgb float (H,W,3) in [0,1]; depth metres (0 = no surface); label uint8 (255 = none); fg_alpha (H,W) in [0,1] (default: foreground depth > 0).
    Returns rgb, depth, label, visible_fg (bool mask of foreground pixels that survive the depth test)."""
    fg_valid = fg_depth > 0
    alpha = fg_valid.astype(np.float32) if fg_alpha is None else np.where(fg_valid, fg_alpha, 0.0).astype(np.float32)
    in_front = fg_valid & ((bg_depth <= 0) | (fg_depth < bg_depth - depth_eps))
    a = (alpha * in_front)[..., None]
    bg = bg_rgb * (1 - shadow_strength * shadow[..., None]) if shadow is not None else bg_rgb
    rgb = bg * (1 - a) + fg_rgb * a
    win = (alpha * in_front) > 0.5
    depth = np.where(win, fg_depth, bg_depth)
    label = np.where(win, fg_label, bg_label).astype(np.uint8)
    return rgb.astype(np.float32), depth.astype(np.float32), label, win
