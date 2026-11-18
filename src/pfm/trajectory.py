"""Flight paths with a speed limit (plan step 6). Poses are OpenGL camera-to-world, Z-up world, in METRES. Velocities for the effects come from consecutive poses."""
from __future__ import annotations

import math

import numpy as np

from .camera import to_opengl
from .effects import velocity_from_poses


def look_at_cv(eye, target, up=(0, 0, 1.0)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def spiral(frames, center, radius, altitude, turns=1.25, shrink=0.4, wave=None, wander=None):
    """wave (altitude oscillation) and wander (look-at drift) default to fractions of the radius, so a 0.5 m scene and a 4 m scene both keep the subject in view."""
    wave = 0.12 * radius if wave is None else wave; wander = 0.08 * radius if wander is None else wander
    out = []
    for k in range(frames):
        t = k / max(1, frames - 1); a = 2 * math.pi * turns * t; r = radius * (1 - shrink * t)
        eye = np.array(center, float) + [r * math.cos(a), r * math.sin(a), altitude - center[2] + wave * math.sin(4 * math.pi * t)]
        out.append(look_at_cv(eye, np.array(center, float) + [wander * math.sin(6 * t), wander * math.cos(5 * t), 0.0]))
    return out


def line(frames, center, radius, altitude):
    a = np.array(center, float) + [-radius, 0, altitude - center[2]]; b = np.array(center, float) + [radius, 0, altitude - center[2]]
    return [look_at_cv(a + (b - a) * k / max(1, frames - 1), np.array(center, float) + [radius * 2, 0, 0]) for k in range(frames)]


def max_speed(poses_cv, fps):
    d = [np.linalg.norm(poses_cv[i + 1][:3, 3] - poses_cv[i][:3, 3]) * fps for i in range(len(poses_cv) - 1)]
    return max(d) if d else 0.0


def retime(poses_cv, fps, limit_mps):
    """Speed limit by changing the frame rate of the path: returns the effective fps so that the fastest step equals limit (poses unchanged)."""
    v = max_speed(poses_cv, fps)
    return fps if v <= limit_mps or v == 0 else fps * limit_mps / v


def motions(poses_cv, fps):
    """Per-frame camera-frame velocity and angular rate (central differences)."""
    out = []
    for k in range(len(poses_cv)):
        a, b = max(0, k - 1), min(len(poses_cv) - 1, k + 1)
        v, w = velocity_from_poses(poses_cv[a], poses_cv[b], (b - a) / fps)
        out.append({"v": v, "w": w})
    return out


def make_trajectory(tr: dict):
    """From a validated scenario trajectory block -> (poses_opencv list, effective_fps)."""
    f = spiral if tr["kind"] == "spiral" else line
    poses = f(tr["frames"], tr["center"], tr["radius_m"], tr["altitude_m"])
    return poses, retime(poses, float(tr["fps"]), float(tr["max_speed_mps"]))
