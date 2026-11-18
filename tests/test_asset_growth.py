import json
import numpy as np
from pfm import asset_growth as G


def box_mesh(a, b, c):
    v = np.array([[x, y, z] for x in (0, a) for y in (0, b) for z in (0, c)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    for _ in range(6):                                              # subdivide to pass the face-count floor
        new = []
        for t in f:
            m = (v[t[0]] + v[t[1]] + v[t[2]]) / 3; v = np.vstack([v, m]); k = len(v) - 1; new += [[t[0], t[1], k], [t[1], t[2], k], [t[2], t[0], k]]
        f = np.array(new)
        if len(f) > 800:
            break
    return v, f


def test_detections_validated():
    raw = json.dumps({"objects": [{"tag": "chair", "box": [100, 200, 400, 900], "importance": 0.9},       # 0..1000 coords are rescaled
                                  {"tag": "dragon", "box": [0, 0, 1, 1], "importance": 1},                 # tag not in the table (also blocked by the schema)
                                  {"tag": "sofa", "box": [0.1, 0.1, 0.11, 0.11], "importance": 0.9},       # too small
                                  {"tag": "table", "box": [0.1, 0.1, 0.6, 0.6], "importance": 0.1},        # unimportant
                                  {"tag": "lamp", "box": [0.1, 0.1, 0.6, 0.6], "importance": 0.8}]})
    d = G.parse_detections(raw, known_tags=["lamp"])
    assert [o["tag"] for o in d] == ["chair"] and abs(d[0]["box"][2] - 0.4) < 1e-9
    assert G.parse_detections("not json") == []
    assert set(G.detection_schema()["properties"]["objects"]["items"]["properties"]["tag"]["enum"]) == set(G.TAG_SIZE_M)


def test_gate_accepts_matching_and_rejects_wrong_shape():
    v, f = box_mesh(1.0, 2.0, 0.5)                                   # a slab, Y up
    s = G.silhouette(v, f, 0, 0)                                     # its own front silhouette is the "crop"
    assert G.gate(v, f, s, "table")["ok"]
    wide = np.zeros((60, 60), bool); wide[28:32, 5:55] = True        # crop of a thin horizontal bar: nothing like a tall slab from any view
    r = G.gate(v, f, wide, "table"); assert not r["ok"] and any("IoU" in x for x in r["reasons"])
    assert not G.gate(v[:4], f[:2], s, "table")["ok"]                # tiny mesh


def test_proportion_prior_and_duplicates():
    D = {"a": [1, 1, 1], "b": [1, 0.8, 0.8], "c": [9, 9, 9]}; R = {"a": {"tag": "box"}, "b": {"tag": "box"}, "c": {"tag": "other"}}
    p = G.proportion_prior("box", D, R); assert p is not None and G.proportion_prior("other", D, R) is None
    v, f = box_mesh(1.0, 1.0, 20.0); r = G.gate(v, f, G.silhouette(v, f, 0, 0), "box", p); assert any("proportions" in x for x in r["reasons"])
    from PIL import Image
    im = Image.fromarray((np.random.default_rng(0).random((40, 40, 3)) * 255).astype(np.uint8)); h = G.dhash(im)
    assert G.is_duplicate(h, "chair", [{"name": "x", "status": "accepted", "tag": "chair", "crop_dhash": f"{h:x}"}]) == "x"
    assert G.is_duplicate(h, "chair", [{"name": "x", "status": "rejected", "tag": "chair", "crop_dhash": f"{h:x}"}]) is None
    assert G.is_duplicate(h, "sofa", [{"name": "x", "status": "accepted", "tag": "chair", "crop_dhash": f"{h:x}"}]) is None


def test_seed_consistency_and_gate():
    v, f = box_mesh(1.0, 2.0, 0.5); v2, f2 = box_mesh(1.0, 0.5, 2.0)             # same mesh vs a different shape
    assert G.seed_consistency(v, f, v, f) > 0.999 and G.seed_consistency(v, f, v2, f2) < 0.8
    s = G.silhouette(v, f, 0, 0)
    assert G.gate(v, f, s, "table", alt=(v, f))["ok"]
    r = G.gate(v, f, s, "table", alt=(v2, f2)); assert not r["ok"] and any("two seeds" in x for x in r["reasons"])


def test_upright_flat_recovers_a_rolled_car():
    from pfm.mesh_upright import upright_flat
    rng = np.random.default_rng(0)
    body = rng.uniform([-2.2, 0.2, -0.9], [2.2, 0.8, 0.9], (3000, 3))                 # long wide low body (x long, y up, z wide)
    roof = rng.uniform([-1.0, 0.8, -0.7], [1.0, 1.4, 0.7], (1500, 3))                  # narrower roof on top
    v = np.vstack([body, roof]); a = np.radians(50)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    out = upright_flat(v @ Rz.T @ Rx.T); ext = out.max(0) - out.min(0)
    assert ext[0] > ext[2] > ext[1] * 0.9 and abs(ext[1] - 1.2) < 0.3 and abs(ext[0] - 4.4) < 0.3          # long along x, height ~1.2 m, not rolled
    y = out[:, 1]; assert out[y > np.percentile(y, 80)][:, [0, 2]].std(0).prod() < out[y < np.percentile(y, 20)][:, [0, 2]].std(0).prod()   # roof on top
