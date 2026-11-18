import json
from pathlib import Path
import pytest
pytestmark = pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "assets" / "models" / "dims_outdoor.json").exists(), reason="run sim/isaac_outdoor_dims.py first")
from pfm import layout_out as L


def _ids(group):
    return [c["id"] for c in L.catalogue() if c["group"] == group]


def _bld():
    return [c["id"] for c in L.catalogue() if c["group"] == "building" and max(c["size_m"][:2]) < 60][0]


def base(**kw):
    tree, car, bld = _ids("tree")[0], _ids("car")[0], _bld()
    d = {"area": {"size": 120}, "light": "noon", "roads": [{"id": "A", "from": [-100, 0], "to": [100, 0], "width": 10}],
         "items": [{"id": "t", "asset": tree, "at": {"road": "A", "s": 0.5, "lane": "kerb_left"}}, {"id": "c", "asset": car, "at": {"road": "A", "s": 0.4, "lane": "lane_right"}},
                   {"id": "b", "asset": bld, "at": [0, 60]}],
         "camera": {"waypoints": [[-60, -30, 20], [60, -30, 20]], "look_at": [0, 0, 2], "speed_mps": 15, "fps": 10}}
    d.update(kw); return d


def test_valid_layout_and_system_decides_size_and_yaw():
    r = L.resolve(base()); assert r["ok"], r["errors"]
    it = {i["id"]: i for i in r["scene"]["items"]}; assert 6 <= it["t"]["height_m"] <= 14 and abs(abs(it["c"]["yaw_deg"]) % 360) in (0.0, 180.0)
    assert abs(it["c"]["pos"][1]) == pytest.approx(2.5)                                    # lane centre = width / 4
    assert r["scene"]["camera"]["poses"][0]["forward"][2] < 0


def test_rules_reject_with_messages():
    car, tree = _ids("car")[0], _ids("tree")[0]
    assert any("carriageway" in e for e in L.resolve(base(items=[{"id": "t", "asset": tree, "at": [0, 1]}]))["errors"])
    assert any("must stand on a carriageway" in e for e in L.resolve(base(items=[{"id": "c", "asset": car, "at": [0, 50]}]))["errors"])
    assert any("setback" in e or "carriageway" in e for e in L.resolve(base(items=[{"id": "b", "asset": _bld(), "at": [0, 8]}]))["errors"])
    assert any("not in the catalogue" in e for e in L.resolve(base(items=[{"id": "x", "asset": "nope", "at": [1, 1]}]))["errors"])
    assert not L.resolve(base(items=[{"id": "t", "asset": tree, "at": [0, 40], "size": 3}]))["ok"]                      # no sizes from the LLM
    assert any("lane" in e for e in L.resolve(base(items=[{"id": "t", "asset": tree, "at": {"road": "A", "s": 0.5, "lane": "lane_left"}}]))["errors"])


def test_camera_rules():
    assert any("outside the area" in e or "height range" in e for e in L.resolve(base(camera={"waypoints": [[0, -30, 0.2]]}))["errors"])
    assert any("path of" in e for e in L.resolve(base(camera={"waypoints": [[-100, -30, 20], [100, -30, 20]], "speed_mps": 1, "fps": 120}))["errors"])
    assert any("looks" in e for e in L.resolve(base(camera={"waypoints": [[0, 40, 30]], "look_at": [0, 40.1, 0]}))["errors"])
    assert L.resolve(base(camera={"waypoints": [[-60, -30, 20], [60, -30, 20]], "look_at": [0, 0, 2], "speed_mps": 30, "fps": 10}))["ok"]               # 30 m/s is allowed


def test_generator_modes_are_valid():
    from pfm.layout_gen_out import MODES, generate
    for m in MODES:
        lay = generate(m, 1); assert L.resolve(lay)["ok"] and lay["items"], m
    assert generate("drone", 4) == generate("drone", 4)


def test_rover_generator_survives_a_crowded_lane():
    from pfm.layout_gen_out import generate
    from pfm.layout_out import resolve
    for seed in (1001, 1005, 1009):                      # 1001 used to fail: a car or tree stood on the rover's lane
        assert resolve(generate("rover", seed))["ok"]


def test_auto_placement_on_a_crowded_kerb():
    from pfm.layout_out import resolve
    from pfm.layout_out import catalogue
    tree = next(c["id"] for c in catalogue() if c["group"] == "tree"); car = next(c["id"] for c in catalogue() if c["group"] == "car")
    base = {"area": {"size": 100.0}, "light": "noon", "roads": [{"id": "A", "from": [-80.0, 0.0], "to": [80.0, 0.0], "width": 10.0}], "seed": 3}
    crowded = dict(base, items=[{"id": f"t{i}", "asset": tree, "at": {"road": "A", "s": 0.5, "lane": "kerb_left"}} for i in range(6)])      # six trees at the same s: the resolver must reject
    assert not resolve(crowded)["ok"]
    auto = dict(base, items=[{"id": f"t{i}", "asset": tree, "at": {"road": "A", "s": "auto", "lane": "kerb_left"}} for i in range(6)] + [{"id": f"c{i}", "asset": car, "at": {"road": "A", "s": "auto", "lane": "lane_right"}} for i in range(4)])
    r = resolve(auto); assert r["ok"], r["errors"][:2]
    xs = sorted(round(i["pos"][0], 1) for i in r["scene"]["items"] if i["group"] == "tree"); assert len(set(xs)) == 6                       # six different spots
    assert resolve(auto)["scene"]["hash"] == r["scene"]["hash"] and [i["pos"] for i in resolve(auto)["scene"]["items"]] == [i["pos"] for i in r["scene"]["items"]]   # deterministic
    full = dict(base, items=[{"id": f"t{i}", "asset": tree, "at": {"road": "A", "s": "auto", "lane": "kerb_left"}} for i in range(100)])
    assert not resolve(full)["ok"] and any("no free spot" in e for e in resolve(full)["errors"])                                               # a full lane is reported, not silently dropped


def test_area_auto_for_roadless_scenes_and_buildings():
    from pfm.layout_out import resolve, catalogue
    tree = next(c["id"] for c in catalogue() if c["group"] == "tree"); bld = min((c for c in catalogue() if c["group"] == "building"), key=lambda c: max(c["size_m"][:2]))["id"]          # the smallest building (a 190 m complex cannot fit a 150 m area, and the resolver says so)
    park = {"area": {"size": 150.0}, "light": "noon", "roads": [], "seed": 5, "items": [{"id": f"t{i}", "asset": tree, "at": {"area": "auto"}} for i in range(20)]}
    r = resolve(park); assert r["ok"], r["errors"][:2]
    assert len({(round(i["pos"][0], 1), round(i["pos"][1], 1)) for i in r["scene"]["items"]}) == 20                                   # 20 different spots
    town = {"area": {"size": 150.0}, "light": "noon", "roads": [{"id": "A", "from": [-100.0, 0.0], "to": [100.0, 0.0], "width": 10.0}], "seed": 5,
            "items": [{"id": "b0", "asset": bld, "at": {"area": "auto"}}, {"id": "t0", "asset": tree, "at": {"area": "auto"}}]}
    rt = resolve(town); assert rt["ok"], rt["errors"][:2]                                                                              # buildings keep their setback, trees stay off the road
    car = next(c["id"] for c in catalogue() if c["group"] == "car"); bad = dict(town, items=[{"id": "c0", "asset": car, "at": {"area": "auto"}}])
    assert not resolve(bad)["ok"]                                                                                                       # cars need a lane
