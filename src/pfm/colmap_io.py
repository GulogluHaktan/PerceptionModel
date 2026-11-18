"""Minimal COLMAP binary reader (cameras, images, points3D) for the data factory (a copy of the reader in scripts/orin_3dgs_colmap.py)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_model(d):
    d = Path(d)
    with open(d / "cameras.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; cams = {}
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24)); npar = {0: 3, 1: 4}[mid]; par = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            fx, fy, cx, cy = (par[0], par[0], par[1], par[2]) if mid == 0 else par; cams[cid] = (w, h, fx, fy, cx, cy)
    imgs = []
    with open(d / "images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(64)); name = b""
            while (c := f.read(1)) != b"\x00": name += c
            npts = struct.unpack("<Q", f.read(8))[0]; f.read(24 * npts)
            imgs.append((name.decode(), np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid))
    with open(d / "points3D.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; xyz = np.zeros((n, 3), np.float32)
        for i in range(n):
            f.read(8); xyz[i] = struct.unpack("<ddd", f.read(24)); f.read(3); f.read(8); f.read(8 * struct.unpack("<Q", f.read(8))[0])
    return cams, sorted(imgs), xyz


def camera_to_world_cv(img):
    """(name, qvec, tvec, cid) -> 4x4 camera-to-world in OpenCV axes (COLMAP world-to-camera inverted)."""
    _, q, t, _ = img; R = qvec2rot(q); m = np.eye(4); m[:3, :3] = R.T; m[:3, 3] = -R.T @ t
    return m
