import importlib.util, math, sys
sp = importlib.util.spec_from_file_location("layout_race", "src/pfm/layout_race.py"); L = importlib.util.module_from_spec(sp); sys.modules["layout_race"] = L; sp.loader.exec_module(L)

G2 = [{"id": "a", "at": [0, 0, 1.5], "yaw_deg": 0}, {"id": "b", "after": "a", "dist": 12, "turn_deg": 20, "dz": 0.3}]


def test_resolves_and_labels():
    r = L.resolve({"gates": G2, "flight": {"speed_mps": 20, "fps": 40, "focus_m": 8}}); assert r["ok"], r["errors"]
    s = r["scene"]; assert len(s["labels"]) == len(s["camera"]["poses"]); assert s["stats"]["frames_gate_seen"] > 0


def test_corners_symmetric_head_on():
    pose = {"pos": [-5.0, 0.0, 1.5], "fwd": [1.0, 0, 0], "up": [0, 0, 1.0]}
    px = L.project(pose, 130, L.inner_corners({"at": [0, 0, 1.5], "yaw_deg": 0}))
    assert abs(px[0][0] + px[1][0] - L.IMG_W) < 1e-6 and abs(px[0][1] - px[1][1]) < 1e-6 and px[0][1] < px[3][1]   # TL left of TR, top above bottom
    assert abs(px[0][0] - (L.IMG_W / 2 - (L.IMG_W / 2 / math.tan(math.radians(65))) * 0.75 / 5)) < 1e-6


def test_rejects():
    assert any("outside" in e for e in L.resolve({"gates": G2, "flight": {"speed_mps": 50}})["errors"])
    assert any("dist" in e for e in L.resolve({"gates": [G2[0], {"id": "b", "after": "a", "dist": 2}]})["errors"])
    assert any("height" in e or "z" in e for e in L.resolve({"gates": [{"id": "a", "at": [0, 0, 0.3], "yaw_deg": 0}, G2[1]]})["errors"])
    back = [G2[0], {"id": "b", "at": [-10, 0, 1.5], "yaw_deg": 0}]
    assert any("off the passage" in e or "behind" in e or "back" in e for e in L.resolve({"gates": back})["errors"])
    assert any("unknown" in e for e in L.resolve({"gates": G2, "code": "x"})["errors"])


def test_generator_valid_and_partial_gates_present():
    for sd in range(3):
        lay = L.generate(sd, speed=30, focus=8); r = L.resolve(lay); assert r["ok"]; assert r["scene"]["stats"]["frames_partial_gate"] > 0
