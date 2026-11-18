from pfm import layout as L

SOFA = "indoor_sofa_pink_sofa_5ec9"; BED = "indoor_bed_single_bed_2_81ad"; TABLE = "indoor_table_teatime_table_c66c"


def base(**kw):
    d = {"room": {"w": 6, "d": 5, "h": 2.6}, "light": "day", "items": [{"id": "s", "asset": SOFA, "at": {"wall": "south", "along": 0.5}},
         {"id": "t", "asset": TABLE, "at": {"near": "s", "side": "north", "gap": 0.5}}], "camera": {"waypoints": [[1, 4, 1.2], [5, 4, 1.2]], "look_at": [3, 1, 0.6], "speed_mps": 1.0, "fps": 10}}
    d.update(kw); return d


def test_catalogue_lists_only_indoor_with_sizes():
    c = L.catalogue(); assert len(c) >= 30 and all(len(x["size_m"]) == 3 for x in c)


def test_valid_layout_resolves_and_sizes_come_from_us():
    r = L.resolve(base()); assert r["ok"], r["errors"]
    s = {i["id"]: i for i in r["scene"]["items"]}; assert abs(s["s"]["pos"][1] - (s["s"]["footprint"][1] / 2 + 0.02)) < 1e-6   # sofa touches the south wall
    assert s["t"]["pos"][1] > s["s"]["pos"][1] and r["scene"]["camera"]["poses"][0]["forward"][1] < -0.5                         # looks towards the sofa (south)


def test_unknown_asset_overlap_and_outside_are_rejected_with_messages():
    r = L.resolve(base(items=[{"id": "a", "asset": "nope", "at": [1, 1]}])); assert not r["ok"] and "not in the catalogue" in r["errors"][0]
    r = L.resolve(base(items=[{"id": "a", "asset": SOFA, "at": [3, 2]}, {"id": "b", "asset": SOFA, "at": [3.2, 2]}])); assert any("overlaps" in e for e in r["errors"])
    r = L.resolve(base(items=[{"id": "a", "asset": SOFA, "at": [0.1, 0.1]}])); assert any("leaves the room" in e for e in r["errors"])
    r = L.resolve(base(items=[{"id": "a", "asset": SOFA, "at": [3, 2], "size": 5}])); assert not r["ok"]                                  # the LLM may not set sizes


def test_camera_rules():
    r = L.resolve(base(camera={"waypoints": [[0.05, 1, 1]]})); assert any("outside the allowed camera box" in e for e in r["errors"])
    r = L.resolve(base(items=[{"id": "b", "asset": BED, "at": [3, 2.5]}], camera={"waypoints": [[3, 2.5, 0.5]]})); assert any("too close to item" in e for e in r["errors"])
    r = L.resolve(base(camera={"waypoints": [[1, 4, 1], [5, 4, 1]], "speed_mps": 100})); assert not r["ok"]
    r = L.resolve(base(camera={"waypoints": [[1, 4, 1.2]], "look_at": [3, 2, 0.5], "roll_deg": 180})); assert r["ok"] and r["scene"]["camera"]["roll_deg"] == 180


def test_near_cycle_and_missing_reference():
    r = L.resolve(base(items=[{"id": "a", "asset": SOFA, "at": {"near": "b", "side": "east", "gap": 0.2}}, {"id": "b", "asset": TABLE, "at": {"near": "a", "side": "east", "gap": 0.2}}]))
    assert any("circular" in e for e in r["errors"])
    r = L.resolve(base(items=[{"id": "a", "asset": SOFA, "at": {"near": "zz", "side": "east", "gap": 0.2}}])); assert any("does not exist" in e for e in r["errors"])


def test_stacking_and_openings():
    tv = "indoor_television_set_vintage_tv_free_ed92"; desk = "indoor_desk_old_simple_desk_23fa"
    lay = base(items=[{"id": "d", "asset": desk, "at": {"wall": "north", "along": 0.5}}, {"id": "v", "asset": tv, "at": {"on": "d", "dx": 0.0, "dy": 0.0}}],
               openings=[{"id": "door", "kind": "door", "wall": "south", "along": 0.2}, {"id": "w1", "kind": "window", "wall": "east", "along": 0.5}], camera={"waypoints": [[1, 2, 1.2], [5, 2, 1.2]], "look_at": [3, 4.5, 0.8]})
    r = L.resolve(lay); assert r["ok"], r["errors"]
    v = [i for i in r["scene"]["items"] if i["id"] == "v"][0]; assert v["pos"][2] > 0.3 and len(r["scene"]["openings"]) == 2
    bad = base(items=[{"id": "s", "asset": SOFA, "at": {"wall": "south", "along": 0.2}}], openings=[{"id": "door", "kind": "door", "wall": "south", "along": 0.2}])
    assert any("door is blocked" in e for e in L.resolve(bad)["errors"])
    big = base(items=[{"id": "d", "asset": desk, "at": [3, 2]}, {"id": "b", "asset": BED, "at": {"on": "d", "dx": 0, "dy": 0}}])
    assert any("cannot sit on" in e for e in L.resolve(big)["errors"])
    two = base(items=[], openings=[{"id": "a", "kind": "door", "wall": "south", "along": 0.5}, {"id": "b", "kind": "window", "wall": "south", "along": 0.5}])
    assert any("overlaps opening" in e for e in L.resolve(two)["errors"])


def test_generator_always_returns_valid_layouts():
    from pfm.layout_gen import KINDS, generate
    for k in KINDS:
        for s in range(5):
            lay = generate(k, s); assert L.resolve(lay)["ok"] and lay["items"], (k, s)
    assert generate("bedroom", 3) == generate("bedroom", 3)                                  # same seed, same layout


def test_llm_loop_repairs_with_a_fake_model():
    import json
    from pfm.layout_gen import generate
    from pfm.layout_llm import layout_schema, make_layout
    good = generate("bedroom", 1); bad = json.loads(json.dumps(good)); bad["items"][0]["asset"] = "indoor_sofa_pink_sofa_5ec9"; bad["items"].append({"id": "x", "asset": bad["items"][0]["asset"], "at": [0.1, 0.1]})
    replies = iter([json.dumps(bad), "not json", json.dumps(good)]); seen = []
    def ask(system, user, schema): seen.append(user); return next(replies)
    r = make_layout("a bedroom", ask, max_repairs=3, system="s"); assert r["ok"] and r["rounds"] == 2 and "broke these rules" in seen[1] or "not valid JSON" in seen[2]
    s = layout_schema(); assert len(s["properties"]["items"]["items"]["properties"]["asset"]["enum"]) >= 30


def test_camera_may_not_stare_at_floor():
    r = L.resolve(base(items=[], camera={"waypoints": [[3, 2.5, 2.2]], "look_at": [3, 2.6, 0.0]})); assert any("looks" in e and "down" in e for e in r["errors"])
    r = L.resolve(base(items=[], camera={"waypoints": [[1, 1, 1.5]], "look_at": [5, 4, 1.0]})); assert r["ok"], r["errors"]


def test_camera_must_see_something():
    r = L.resolve(base(camera={"waypoints": [[1, 4, 1.2], [5, 4, 1.2]], "look_at": [3, 5, 1.2]})); assert any("sees an item" in e for e in r["errors"])
