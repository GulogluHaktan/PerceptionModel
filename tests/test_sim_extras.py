import json
import numpy as np
import pytest

from pfm.camera import Camera
from pfm.llm_scenario import coverage_matrix, extract_json, generate_scenarios
from pfm.replay import Entry, Library, plan_round
from pfm.scale import scale_from_distance, scale_from_telemetry
from pfm.shadow import ground_shadow, plane_depth
from pfm.trajectory import look_at_cv


def test_scale_from_distance_and_telemetry():
    assert abs(scale_from_distance([0, 0, 0], [2, 0, 0], 1.0) - 0.5) < 1e-12
    rng = np.random.default_rng(0); units = rng.uniform(-3, 3, (30, 3)); th = 0.7
    r = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]]); metres = 0.37 * (r @ units.T).T + [5, -2, 1] + rng.normal(0, 0.01, (30, 3))
    s, res = scale_from_telemetry(units, metres); assert abs(s - 0.37) < 0.01 and res < 0.05
    with pytest.raises(ValueError):
        scale_from_telemetry(units[:3], metres[:3])


def test_replay_rules():
    lib = Library()
    for i, (c, rnd) in enumerate([("clear", 0), ("clear", 0), ("fog", 0), ("fog", 0), ("rain", 1), ("rain", 1)]):
        lib.add(Entry(f"h{i}", c, rnd, last_score=0.5, best_score=0.5))
    lib.entries[3].last_score = 0.3                                                # fog regressed
    p = plan_round(lib, round_no=1, n_total=1000, min_real=0.3)
    assert p["real"] >= 300 and p["real"] + p["n_synthetic"] == 1000 and set(p["suite"]) == {"h0", "h2", "h4"}
    assert not (set(p["synthetic"]) & set(p["suite"]))                              # the regression suite is never trained on
    assert p["synthetic"].get("h5", 0) > 0                                          # the new condition is learned (h4 is its held-out suite scenario)
    assert p["synthetic"].get("h1", 0) > 0 and p["synthetic"]["h3"] > p["synthetic"]["h1"]    # replay, regressed condition gets more
    with pytest.raises(ValueError):
        lib.add(Entry("h0", "x", 1))


def test_llm_manager_validates_and_retries():
    good = {"objects": [{"type": "gate", "pos": [1, 0, 0]}], "effects": {"fog_beta": 0.2}, "seed": 1}
    bad = {"objects": [{"type": "dragon", "pos": [0, 0, 0]}]}
    calls = []
    def ask(prompt):
        calls.append(prompt)
        return "```json\n" + json.dumps([good, bad]) + "\n```" if len(calls) == 1 else json.dumps([{**good, "seed": 2}])
    valid, rejected = generate_scenarios("fast flight in fog", 2, ask)
    assert len(valid) == 2 and len(rejected) == 1 and "dragon" in rejected[0]["error"] or "type" in rejected[0]["error"]
    assert len(calls) == 2 and "rejected" in calls[1]                                  # the errors went back to the LLM
    assert extract_json('text [{"a": 1}] more')[0]["a"] == 1
    with pytest.raises(ValueError):
        extract_json("no json here")
    assert len(coverage_matrix()) == 3 * 3 * 2 * 2 and len({json.dumps(s, sort_keys=True) for s in coverage_matrix()}) == 36


def test_shadow_only_on_ground():
    cam = Camera.from_hfov_deg(96, 72, 80.0); c2w = look_at_cv((-6, 0, 3), (0, 0, 0)); plane = (np.array([0, 0, 1.0]), 0.0)
    box = {"center": np.array([0, 0, 0.5]), "size": np.array([1.0, 1.0, 1.0]), "yaw": 0.0}
    pd = plane_depth(cam, c2w, plane); assert pd[50, 48] > 0 and pd[0, 48] == 0                                    # lower rows hit the ground, the sky row does not
    sh = ground_shadow(cam, c2w, [box], plane, (0.4, 0.2, -0.9), pd)                                               # background = bare ground
    assert sh.max() > 0.5 and sh[:10].sum() == 0
    sh2 = ground_shadow(cam, c2w, [box], plane, (0.4, 0.2, -0.9), np.full((72, 96), 2.0, np.float32))              # background surface far from the plane: no shadow
    assert sh2.max() == 0


def test_frames_up_alignment_and_rotated_boxes():
    from pfm.camera import Camera
    from pfm.frames import aligned_to_scene, estimate_up_from_poses, rotation_from_up, transform_boxes
    from pfm.proxy_fg import render_boxes
    up = np.array([0.3, -0.8, 0.5]); up /= np.linalg.norm(up); R = rotation_from_up(up)
    assert np.allclose(R @ up, [0, 0, 1], atol=1e-9) and np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    poses = []                                                                       # upright cameras around an axis `up`: their y axes point along -up
    for a in np.linspace(0, 6, 8):
        z = np.cross(up, [1, 0, 0]) * np.cos(a) + np.cross(up, np.cross(up, [1, 0, 0])) * np.sin(a); z /= np.linalg.norm(z)
        y = -up + 0.05 * z; y /= np.linalg.norm(y); x = np.cross(y, z); m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2] = x / np.linalg.norm(x), y, z; poses.append(m)
    assert np.allclose(estimate_up_from_poses(poses), up, atol=0.08)
    cam = Camera.from_hfov_deg(64, 48, 90.0); T = aligned_to_scene(R)
    box_a = {"center": np.array([0.0, 0.0, 1.0]), "size": np.array([2.0, 2.0, 2.0]), "yaw": 0.3, "color": (1, 0, 0), "label": 3}
    # a camera and the box moved by the SAME rigid motion must give the same picture
    from pfm.trajectory import look_at_cv
    c2w_a = look_at_cv((-6, 0, 1.5), (0, 0, 1.0)); c2w_s = T @ c2w_a
    _, d_a, l_a, _ = render_boxes(cam, c2w_a, [box_a]); _, d_s, l_s, _ = render_boxes(cam, c2w_s, transform_boxes([box_a], T))
    assert (l_a == 3).sum() > 50 and np.array_equal(l_a, l_s) and np.allclose(d_a, d_s, atol=1e-4)
