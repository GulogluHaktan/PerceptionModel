import numpy as np
import pytest
import torch

from pfm.effects import fog, motion_blur, pipeline, pixel_flow, validate_cfg, velocity_from_poses


def test_cfg_is_strict():
    assert validate_cfg({})["iso"] == 100.0
    with pytest.raises(ValueError, match="unknown"):
        validate_cfg({"magic": 1})
    with pytest.raises(ValueError, match="outside"):
        validate_cfg({"fog_beta": 5})


def test_flow_physics():
    z = torch.full((8, 8), 10.0)
    f = pixel_flow(z, fx=100.0, v=(1.0, 0, 0), w=(0, 0, 0))      # moving right: the scene moves left, flow = -fx*vx/z
    assert torch.allclose(f[..., 0], torch.full((8, 8), -10.0), atol=1e-4) and f[..., 1].abs().max() < 1e-4
    assert pixel_flow(z, 100.0, (0, 0, 0), (0, 0, 0)).abs().max() == 0


def test_blur_keeps_flat_images_and_blurs_edges():
    img = torch.zeros(1, 3, 16, 16); img[..., 8:] = 1
    flow = torch.zeros(16, 16, 2); flow[..., 0] = 200.0
    out = motion_blur(img, flow, 0.05)                             # 10 px displacement
    assert out[0, 0, 4, 1] < 1e-3 and abs(float(out[0, 0, 4, 14]) - 1) < 1e-3 and 0.05 < float(out[0, 0, 4, 8]) < 0.95
    assert torch.allclose(motion_blur(torch.ones(1, 3, 8, 8), flow[:8, :8], 0.05), torch.ones(1, 3, 8, 8), atol=1e-4)


def test_fog_limits():
    img = torch.zeros(1, 3, 4, 4); d = torch.full((4, 4), 1e-2)
    assert float(fog(img, d, 0.0, 0.8).max()) == 0.0
    d = torch.full((4, 4), 100.0)
    assert abs(float(fog(img, d, 1.0, 0.8).mean()) - 0.8) < 1e-3


def test_pipeline_deterministic_and_poses():
    rng = np.random.default_rng(1); img = rng.random((32, 32, 3)).astype(np.float32); depth = np.full((32, 32), 5.0, np.float32)
    cfg = {"fog_beta": 0.1, "iso": 800, "ev": -2, "gain_ev": 2}
    a = pipeline(img, depth, 40.0, {"v": (2, 0, 0)}, cfg, seed=3, device="cpu"); b = pipeline(img, depth, 40.0, {"v": (2, 0, 0)}, cfg, seed=3, device="cpu")
    assert np.array_equal(a, b) and a.shape == (32, 32, 3)
    p0 = np.eye(4); p1 = np.eye(4); p1[0, 3] = 1.0                 # 1 m along the camera x axis in 0.5 s
    v, w = velocity_from_poses(p0, p1, 0.5)
    assert abs(v[0] - 2.0) < 1e-6 and max(abs(np.array(w))) < 1e-6
