"""Gravity alignment for generated meshes of things that stand flat on the ground (vehicles, beds, sofas, tables). numpy only (loaded by scripts/image_to_asset.py in the Hunyuan env).
A generator copies the viewpoint of the photo: an aerial photo of a car gives a rolled/pitched car. For these tags the smallest principal axis of the vertex cloud is 'up', the largest is 'forward' (-> +X); the
side that is wider in the horizontal plane (wheels/chassis vs roof) is down. NOT for chairs, poles, lamps, planes, bikes (their up axis is not the smallest PCA axis)."""
import numpy as np

FLAT_TAGS = {"car", "jeep", "minivan", "pickup_truck", "truck", "ambulance", "bus", "cab", "police_car", "race_car", "school_bus", "sofa", "bed", "table"}


def upright_flat(v: np.ndarray) -> np.ndarray:
    """Return vertices rotated (proper rotation, no mirroring) so that Y is up (min-variance axis), X is the longest horizontal axis, and the wider end is at the bottom."""
    c = v - v.mean(0); w, E = np.linalg.eigh(np.cov(c.T)); E = E[:, np.argsort(w)[::-1]]          # columns: largest, middle, smallest
    R = np.stack([E[:, 0], E[:, 2], E[:, 1]], 1)                                                  # new x = longest, new y = smallest (up), new z = middle
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    p = c @ R; y = p[:, 1]; lo, hi = np.percentile(y, 20), np.percentile(y, 80)
    spread = lambda m: float(p[m][:, [0, 2]].std(0).prod()) if m.sum() > 10 else 0.0
    if spread(y >= hi) > spread(y <= lo):                                                         # roof wider than the bottom: flip 180 deg about X (keeps handedness)
        p = p * np.array([1, -1, -1])
    return p
