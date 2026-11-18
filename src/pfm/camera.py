"""Camera contract shared by the 3DGS renderer, Isaac Sim and the effects library (plan step 1 and 2).
Poses are stored as OpenGL camera-to-world 4x4 (x right, y UP, z BACK; the NeRF/Blender convention used in every json of this project).
Everything inside the code that projects or unprojects uses OpenCV axes (x right, y DOWN, z FORWARD): use `to_opencv` / `to_opengl`.
Pinhole, square pixels, principal point at the image centre; depth is z along the optical axis in metres.
`meters_per_unit` converts a scene whose unit is not a metre (COLMAP scenes, Blender scenes) to metres; get it from a known distance or telemetry."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float

    @staticmethod
    def from_angle_x(width: int, height: int, angle_x: float) -> "Camera":
        return Camera(width, height, 0.5 * width / math.tan(0.5 * angle_x))

    @staticmethod
    def from_hfov_deg(width: int, height: int, hfov_deg: float) -> "Camera":
        return Camera.from_angle_x(width, height, math.radians(hfov_deg))

    @property
    def angle_x(self) -> float:
        return 2 * math.atan(0.5 * self.width / self.fx)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2


def to_opencv(c2w_opengl) -> np.ndarray:
    m = np.array(c2w_opengl, dtype=np.float64).copy(); m[:3, 1:3] *= -1
    return m


to_opengl = to_opencv                                   # the flip is its own inverse


def scale_pose(c2w, meters_per_unit: float) -> np.ndarray:
    m = np.array(c2w, dtype=np.float64).copy(); m[:3, 3] *= meters_per_unit
    return m


def project(cam: Camera, c2w_cv, pts_world):
    """World points (N,3) -> continuous pixel coordinates u = fx*x/z + cx, v likewise (pixel i covers [i, i+1), its centre is i + 0.5) and depth z."""
    pc = (np.linalg.inv(c2w_cv) @ np.c_[np.asarray(pts_world, float), np.ones(len(pts_world))].T).T
    z = pc[:, 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return cam.fx * pc[:, 0] / zz + cam.cx, cam.fx * pc[:, 1] / zz + cam.cy, z


def unproject(cam: Camera, c2w_cv, depth):
    """Depth image (H,W) -> world points (H,W,3); pixels with depth <= 0 give NaN."""
    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = np.where(depth > 0, depth, np.nan)
    pc = np.stack([(uu + 0.5 - cam.cx) / cam.fx * z, (vv + 0.5 - cam.cy) / cam.fx * z, z, np.ones_like(z)], -1)      # pixel centres at +0.5, like Isaac Sim and gsplat
    return (pc.reshape(-1, 4) @ np.asarray(c2w_cv, float).T)[:, :3].reshape(h, w, 3)
