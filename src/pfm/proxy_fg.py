"""Analytic foreground renderer (stand-in for Isaac Sim until its renders are wired in): oriented boxes by ray-box intersection.
Same interface as an Isaac foreground pass: rgb, depth (z along the optical axis, metres, 0 = empty), label, alpha, all for one camera pose.
Gate = four box bars, pole/tree = a thin box, box = box. Flat shading with one light: this is NOT photoreal, it only exercises the pipeline (depth test, labels, corners)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .camera import Camera

LIGHT = np.array([0.4, 0.3, 0.86]) / np.linalg.norm([0.4, 0.3, 0.86])


def gate_boxes(pos, yaw_deg, width, height, bar=0.08, color=(1.0, 0.45, 0.05), label=1):
    """Four bars of a square gate standing at pos (ground point), opening normal along the local +x axis, width along local y."""
    y = math.radians(yaw_deg); r = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1.0]]); p = np.asarray(pos, float)
    def b(off, size):
        return {"center": p + r @ np.asarray(off, float), "size": np.asarray(size, float), "yaw": y, "color": color, "label": label}
    return [b((0, -width / 2, height / 2), (bar, bar, height)), b((0, width / 2, height / 2), (bar, bar, height)),
            b((0, 0, height - bar / 2), (bar, width + bar, bar)), b((0, 0, bar / 2), (bar, width + bar, bar))]


def gate_corners(pos, yaw_deg, width, height):
    """3D outer corners (top-left, top-right, bottom-right, bottom-left as seen from the +x side) of the gate frame."""
    y = math.radians(yaw_deg); r = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1.0]]); p = np.asarray(pos, float)
    return np.array([p + r @ np.array([0, s * width / 2, z]) for s, z in ((-1, height), (1, height), (1, 0), (-1, 0))])


def render_boxes(cam: Camera, c2w_cv, boxes, light=LIGHT, device="cpu"):
    H, W = cam.height, cam.width
    c2w = torch.tensor(np.asarray(c2w_cv), dtype=torch.float32, device=device)
    vv, uu = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    d_cam = torch.stack([(uu + 0.5 - cam.cx) / cam.fx, (vv + 0.5 - cam.cy) / cam.fx, torch.ones_like(uu, dtype=torch.float32)], -1).float()   # z = 1: t is the z-depth
    d_w = d_cam @ c2w[:3, :3].T; o_w = c2w[:3, 3]
    depth = torch.zeros(H, W, device=device); rgb = torch.zeros(H, W, 3, device=device); label = torch.full((H, W), 255, dtype=torch.uint8, device=device)
    best = torch.full((H, W), float("inf"), device=device)
    for bx in boxes:
        if "R" in bx:
            R = torch.tensor(np.asarray(bx["R"]), dtype=torch.float32, device=device)
        else:
            y = bx["yaw"]; R = torch.tensor([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1.0]], dtype=torch.float32, device=device)
        c = torch.tensor(np.asarray(bx["center"]), dtype=torch.float32, device=device); h = torch.tensor(np.asarray(bx["size"]), dtype=torch.float32, device=device) / 2
        o = R.T @ (o_w - c); d = d_w @ R                                   # ray in the box frame
        inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t1, t2 = (-h - o) * inv, (h - o) * inv
        tn, tf = torch.minimum(t1, t2), torch.maximum(t1, t2)
        t_near, axis = tn.max(-1); t_far = tf.min(-1).values
        hit = (t_far > t_near.clamp(min=0)) & (t_near > 0) & (t_near < best)
        if not hit.any():
            continue
        n_local = torch.zeros(H, W, 3, device=device); sign = -torch.sign(torch.gather(d, -1, axis[..., None]))[..., 0]
        n_local.scatter_(-1, axis[..., None], sign[..., None]); n_w = n_local @ R.T
        shade = 0.35 + 0.65 * (n_w @ torch.tensor(light, dtype=torch.float32, device=device)).clamp(min=0)
        col = torch.tensor(bx["color"], dtype=torch.float32, device=device)
        best = torch.where(hit, t_near, best); depth = torch.where(hit, t_near, depth)
        rgb = torch.where(hit[..., None], shade[..., None] * col, rgb); label = torch.where(hit, torch.tensor(bx["label"], dtype=torch.uint8, device=device), label)
    return rgb.cpu().numpy(), depth.cpu().numpy(), label.cpu().numpy(), (depth > 0).float().cpu().numpy()
