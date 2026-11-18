import json
import numpy as np
import pytest

from pfm.camera import Camera, project, scale_pose, to_opencv, unproject
from pfm.compose import compose
from pfm.dataset_writer import DatasetWriter
from pfm.ground import fit_ground_plane, place_on_plane
from pfm.scenario import scenario_hash, validate_scenario
from pfm.trajectory import make_trajectory, max_speed, motions


def test_camera_roundtrip_and_conventions():
    cam = Camera.from_hfov_deg(640, 480, 90.0); assert abs(cam.fx - 320.0) < 1e-9
    c2w_gl = np.eye(4); c2w_gl[:3, 3] = [1, 2, 3]
    cv = to_opencv(c2w_gl); assert np.allclose(cv[:3, 1], [0, -1, 0]) and np.allclose(to_opencv(cv), c2w_gl)
    u, v, z = project(cam, np.eye(4), np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 2.0]]))   # a point on the axis hits the image centre
    assert abs(u[0] - 320) < 1e-9 and abs(v[0] - 240) < 1e-9 and abs(u[1] - (320 + 160)) < 1e-9
    d = np.full((480, 640), 4.0); pts = unproject(cam, np.eye(4), d)
    u2, v2, z2 = project(cam, np.eye(4), pts.reshape(-1, 3)); assert np.allclose(z2, 4.0) and abs(u2[240 * 640 + 100] - 100.5) < 1e-6
    assert np.allclose(scale_pose(np.eye(4), 2.0)[:3, 3], 0)


def test_compose_depth_test_and_labels():
    h = w = 8
    bg_rgb = np.zeros((h, w, 3), np.float32); bg_d = np.full((h, w), 5.0, np.float32); bg_l = np.full((h, w), 5, np.uint8)
    fg_rgb = np.ones((h, w, 3), np.float32); fg_l = np.full((h, w), 1, np.uint8)
    fg_d = np.zeros((h, w), np.float32); fg_d[:, :4] = 3.0; fg_d[:, 4:6] = 7.0                         # left in front, middle behind, right empty
    rgb, d, l, vis = compose(bg_rgb, bg_d, bg_l, fg_rgb, fg_d, fg_l)
    assert (l[:, :4] == 1).all() and (l[:, 4:] == 5).all() and vis[:, :4].all() and not vis[:, 4:].any()
    assert rgb[0, 0, 0] == 1.0 and rgb[0, 5, 0] == 0.0 and d[0, 0] == 3.0 and d[0, 5] == 5.0
    rgb2, *_ = compose(bg_rgb, bg_d, bg_l, fg_rgb, fg_d, fg_l, fg_alpha=np.full((h, w), 0.5, np.float32))
    assert abs(rgb2[0, 0, 0] - 0.5) < 1e-6


def test_ground_plane():
    rng = np.random.default_rng(0)
    g = np.c_[rng.uniform(-5, 5, (2000, 2)), 0.3 + 0.01 * rng.standard_normal(2000)]; clutter = rng.uniform(-5, 5, (500, 3)) + [0, 0, 2]
    n, d, inl = fit_ground_plane(np.vstack([g, clutter]))
    assert n[2] > 0.99 and abs(place_on_plane((n, d), (1.0, 1.0))[2] - 0.3) < 0.03 and inl > 1500


def test_scenario_is_strict_and_hash_stable():
    sc = {"objects": [{"type": "gate", "pos": [10, 0, 0]}], "effects": {"fog_beta": 0.2}, "seed": 3}
    v = validate_scenario(sc); assert v["trajectory"]["frames"] == 120 and scenario_hash(sc) == scenario_hash(json.loads(json.dumps(sc)))
    for bad in ({"objects": [{"type": "dragon", "pos": [0, 0, 0]}]}, {"magic": 1}, {"trajectory": {"max_speed_mps": 99}},
                {"camera": {"width": "wide"}}, {"effects": {"fog_beta": 9}}, {"seed": 1.5}):
        with pytest.raises(ValueError):
            validate_scenario(bad)


def test_trajectory_speed_limit_and_writer(tmp_path):
    sc = validate_scenario({"trajectory": {"frames": 40, "fps": 30.0, "max_speed_mps": 5.0, "radius_m": 6.0}})
    poses, fps = make_trajectory(sc["trajectory"])
    assert max_speed(poses, fps) <= 5.0 + 1e-6 and len(poses) == 40
    m = motions(poses, fps); assert abs(np.linalg.norm(m[10]["v"]) - 5.0) < 3.0 and len(m) == 40
    w = DatasetWriter(tmp_path, {"seed": 1}); w.add(0, np.zeros((4, 4, 3)), np.ones((4, 4)), np.full((4, 4), 255), np.eye(4), m[0]); h = w.close()
    man = json.load(open(tmp_path / "manifest.json")); assert man["scenario_hash"] == h and man["n_frames"] == 1


def test_proxy_fg_depth_and_simulate():
    from pfm.proxy_fg import gate_boxes, gate_corners, render_boxes
    from pfm.simulate import generate_frame
    cam = Camera.from_hfov_deg(64, 48, 90.0); c2w = np.eye(4)                       # camera at the origin looking along +z (OpenCV)
    box = {"center": np.array([0.0, 0.0, 5.0]), "size": np.array([2.0, 2.0, 2.0]), "yaw": 0.0, "color": (1, 0, 0), "label": 7}
    rgb, d, lab, a = render_boxes(cam, c2w, [box])
    assert abs(d[24, 32] - 4.0) < 1e-3 and lab[24, 32] == 7 and lab[0, 0] == 255 and d[0, 0] == 0 and rgb[24, 32, 0] > 0.3      # front face at z = 4
    bg = {"rgb": np.zeros((48, 64, 3), np.float32), "depth": np.full((48, 64), 3.0, np.float32), "label": np.full((48, 64), 5, np.uint8)}    # wall at 3 m hides the box at 4 m
    out, depth, label, corners, vis = generate_frame(cam, c2w, bg, [box], [], {}, {}, device="cpu")
    assert not vis.any() and (label == 5).all()
    bg["depth"][:] = 9.0                                                              # wall behind the box: now the box wins
    out, depth, label, corners, vis = generate_frame(cam, c2w, bg, [box], [], {}, {}, device="cpu"); assert vis[24, 32] and label[24, 32] == 7
    g = gate_corners((0.0, 0.0, 4.0), 0.0, 2.0, 1.5); assert g.shape == (4, 3) and abs(g[:, 2].max() - 5.5) < 1e-9
    from pfm.trajectory import look_at_cv                                           # a Z-up camera that faces the gate (the earlier pose saw it edge-on)
    c2 = look_at_cv((-5.0, 0.0, 1.0), (0.0, 0.0, 0.8)); gb = gate_boxes((0, 0, 0.0), 0, 2.0, 1.5); gc = [gate_corners((0, 0, 0.0), 0, 2.0, 1.5)]
    empty = {"rgb": bg["rgb"], "depth": np.zeros((48, 64), np.float32), "label": bg["label"]}
    out, depth, label, corners, vis = generate_frame(cam, c2, empty, gb, gc, {}, {}, device="cpu")
    assert len(corners[0]) == 4 and sum(c[2] for c in corners[0]) == 4 and (label == 1).any() and vis.sum() > 20
    assert corners[0][0][1] < corners[0][3][1]                                       # the top corners are above the bottom corners in the image
