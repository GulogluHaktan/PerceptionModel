"""Declarative OUTDOOR layout (no Isaac import), same principle as pfm.layout: the LLM picks asset ids and coordinates, never code or sizes; this module resolves and checks.

LLM writes:  area size, light, roads (straight segments), items (asset id + where), camera waypoints.
We resolve:  real size (assets/models/dims_outdoor.json, measured in Isaac), tree/pole height, road-relative placement (lane, kerb, yaw along the road), sidewalks and lane markings
             (added by the renderer), overlap and setback checks, camera path (speed up to 40 m/s, fps, roll), clearance from buildings, item visibility, pitch limit.
Rules reject instead of silently fixing; every error names the item and the limit so the reply can go straight back to the LLM (same repair loop as indoors).
Frame: Z up, ground z=0, area [-S, S]^2. Roads are straight segments with a carriageway of `width` m plus 3 m sidewalks on both sides.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .assets import REGISTRY
from .layout import _num, _vec

DIMS = json.loads((Path(__file__).resolve().parents[2] / "assets" / "models" / "dims_outdoor.json").read_text())
_LC = Path(__file__).resolve().parents[2] / "assets" / "models" / "label_coverage.json"
UNLABELLED = {a for a, v in json.loads(_LC.read_text()).items() if v["labelled_frac"] < 0.6} if _LC.exists() else set()      # measured on real renders (scripts/outdoor_label_coverage.py): their leaves/parts get no semantic label, so a label from the class would be wrong
GROUPS = {"tree", "shrub", "rock", "prop", "pole", "building", "car"}
LIGHTS = {"noon", "overcast", "dusk"}
LIMITS = {"area": (40.0, 300.0), "road_w": (5.0, 16.0), "roads": 6, "items": 120, "speed": (0.0, 40.0), "fps": (1.0, 120.0), "frames": (2, 600), "hfov": (20.0, 140.0), "alt": (1.0, 150.0)}
SIDEWALK, ITEM_GAP, BUILD_GAP, CAM_ITEM, MAX_PITCH_DEG, MIN_SEEN_FRAC = 3.0, 0.3, 3.0, 2.0, 80.0, 0.4
TREE_H = (6.0, 14.0)                                  # tree height range; the exact value is a hash of the item id (deterministic)
LANES = {"lane_left": -1, "lane_right": 1}
KERBS = {"kerb_left": -1, "kerb_right": 1}
TOP = {"area", "light", "roads", "items", "camera", "seed"}


def _tag(name: str) -> str:
    a = REGISTRY[name]; g = a["group"]
    return a.get("tag") or (name.split("_", 1)[-1] if g in ("prop", "pole") else g)


def catalogue() -> list[dict]:
    out = []
    for n, d in DIMS.items():
        if n in REGISTRY and d["ok"] and n not in UNLABELLED and REGISTRY[n]["group"] in GROUPS:
            sz = d["size_m"]; out.append({"id": n, "tag": _tag(n), "group": REGISTRY[n]["group"], "size_m": sz, "height_range": TREE_H if REGISTRY[n]["group"] == "tree" else None})
    return out


def catalogue_text() -> str:
    return "\n".join(f'{c["id"]} | {c["group"]}/{c["tag"]} | ' + (f'height {TREE_H[0]:g}-{TREE_H[1]:g} m (system picks)' if c["height_range"] else f'{c["size_m"][0]:.1f} x {c["size_m"][1]:.1f} x {c["size_m"][2]:.1f} m') for c in catalogue())


SCHEMA_TEXT = f"""Reply with ONLY a JSON object. Keys: area, light, roads, items, camera, seed. No code, no sizes, no text outside the JSON.
area: {{size}} half-extent in metres, {LIMITS['area']}; the world spans -size..size in x and y, ground z=0.
light: one of {sorted(LIGHTS)}.
roads: list (max {LIMITS['roads']}) of {{id, from: [x,y], to: [x,y], width: {LIMITS['road_w']} m}}; sidewalks (3 m) and lane markings are added by the system.
items: list (max {LIMITS['items']}) of {{id: unique, asset: catalogue id, at: [x, y] OR {{road: road id, s: 0..1 along the road OR the word \"auto\" (the system picks a free spot: use it for cars, trees, poles and props so they never overlap), lane: lane_left|lane_right (cars) or kerb_left|kerb_right (trees, poles, props)}} OR {{area: "auto"}} (the system picks a free spot anywhere in the area; for buildings, rocks and every scene WITHOUT roads) OR {{near: item id, side: north|south|east|west, gap: metres}}, yaw_deg}}.
  Cars go on lanes (the system aligns them with the road), trees/poles/props on kerbs or anywhere off the carriageway, buildings away from roads. Tree height is chosen by the system.
camera: {{waypoints: [[x,y,z],...] (1..20), look_at: [x,y,z] or null (null = look along the motion), roll_deg: -180..180 (180 = inverted), speed_mps: {LIMITS['speed']}, fps: {LIMITS['fps']}, hfov_deg: {LIMITS['hfov']}}}; z in {LIMITS['alt']}.
Rules: items may not overlap or leave the area, nothing but cars on a carriageway, buildings keep {BUILD_GAP} m from the sidewalk, the camera keeps {CAM_ITEM} m from items, may not look more than {MAX_PITCH_DEG:g} degrees up/down, and at least {MIN_SEEN_FRAC:.0%} of frames must show an item.
seed: int."""


def _h01(s: str) -> float:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _size(name: str, iid: str) -> tuple[float, float, float, float | None]:
    d = DIMS[name]; sx, sy, sz = d["size_m"]; g = REGISTRY[name]["group"]
    if g == "tree":
        h = TREE_H[0] + (TREE_H[1] - TREE_H[0]) * _h01(iid); k = h / d["nominal_height"]; return sx * k, sy * k, h, h
    return sx, sy, sz, (d["nominal_height"] if g == "pole" else None)


def _rot(fx, fy, yaw):
    a = math.radians(yaw); c, s = abs(math.cos(a)), abs(math.sin(a)); return fx * c + fy * s, fx * s + fy * c


def _seg_dist(p, a, b):
    ax, ay = a; bx, by = b; dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2)); return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy), t


def _road_edge_dist(x, y, hx, hy, roads):
    """Smallest distance from the item's rectangle (centre x,y half-sizes hx,hy) to the carriageway edge of any road (negative = overlapping the carriageway); also the sidewalk-inclusive distance."""
    best = (1e9, None)
    for r in roads:
        d, _ = _seg_dist((x, y), r["from"], r["to"]); d -= r["width"] / 2 + math.hypot(hx, hy) * 0.7
        if d < best[0]:
            best = (d, r)
    return best


def _auto_s(rd, lane, grp, sx, sy, yaw, iid, placed, rl, S, seed):
    """First free position (as s along the road) for an item on a lane / kerb: 60 candidates in a seeded shuffle; free = inside the area, right side of the carriageway edge, no overlap with placed items (same rules as the explicit path). None if the lane is full."""
    import random
    if rd is None or (grp == "car" and lane not in LANES) or (grp != "car" and lane not in KERBS):
        return None
    cands = [0.04 + 0.92 * k / 59 for k in range(60)]; random.Random(f"{seed}:{iid}").shuffle(cands); dx, dy = rd["dir"]; nx, ny = -dy, dx
    for s in cands:
        cx = rd["from"][0] + dx * rd["length"] * s; cy = rd["from"][1] + dy * rd["length"] * s
        if grp == "car":
            pos = [cx + nx * LANES[lane] * rd["width"] / 4, cy + ny * LANES[lane] * rd["width"] / 4]; base = math.degrees(math.atan2(dy, dx)) + (0 if lane == "lane_right" else 180)
        else:
            off = KERBS[lane] * (rd["width"] / 2 + SIDEWALK + max(sx, sy) / 2 + 0.5); pos = [cx + nx * off, cy + ny * off]; base = 0.0
        fx, fy = _rot(sx, sy, yaw + base)
        if abs(pos[0]) + fx / 2 > S or abs(pos[1]) + fy / 2 > S:
            continue
        d_edge, _ = _road_edge_dist(pos[0], pos[1], fx / 2, fy / 2, rl)
        if (grp == "car" and rl and d_edge > -1.0) or (grp != "car" and rl and d_edge < 0.3):
            continue
        gap_m = BUILD_GAP if grp == "building" else ITEM_GAP
        if any((grp == "car") == (o["group"] == "car") and abs(pos[0] - o["pos"][0]) < (fx + o["footprint"][0]) / 2 + max(gap_m, BUILD_GAP if o["group"] == "building" else ITEM_GAP)
               and abs(pos[1] - o["pos"][1]) < (fy + o["footprint"][1]) / 2 + max(gap_m, BUILD_GAP if o["group"] == "building" else ITEM_GAP) for o in placed.values()):
            continue
        return s
    return None


def _auto_xy(grp, sx, sy, yaw, iid, placed, rl, S, seed):
    """First free position anywhere in the area for a non-car item (trees, rocks, props, poles, buildings): 400 seeded candidates; free = inside the area, off the carriageway (buildings also behind the setback), no overlap with placed items. None if the area is full."""
    import random
    if grp == "car":
        return None
    rng = random.Random(f"{seed}:{iid}:area"); fx, fy = _rot(sx, sy, yaw); gap_m = BUILD_GAP if grp == "building" else ITEM_GAP
    for _ in range(400):
        pos = [rng.uniform(-S + fx / 2, S - fx / 2), rng.uniform(-S + fy / 2, S - fy / 2)]
        d_edge, _r = _road_edge_dist(pos[0], pos[1], fx / 2, fy / 2, rl) if rl else (99.0, None)
        if d_edge < (SIDEWALK + BUILD_GAP if grp == "building" else 0.3):
            continue
        if any(abs(pos[0] - o["pos"][0]) < (fx + o["footprint"][0]) / 2 + max(gap_m, BUILD_GAP if o["group"] == "building" else ITEM_GAP)
               and abs(pos[1] - o["pos"][1]) < (fy + o["footprint"][1]) / 2 + max(gap_m, BUILD_GAP if o["group"] == "building" else ITEM_GAP) for o in placed.values() if o["group"] != "car"):
            continue
        return pos
    return None


def resolve(layout: dict) -> dict:
    errs: list[str] = []
    if not isinstance(layout, dict):
        return {"ok": False, "errors": ["layout must be a JSON object"], "scene": None}
    if set(layout) - TOP:
        errs.append(f"unknown top-level keys {sorted(set(layout) - TOP)}")
    ar = layout.get("area", {}); S = None
    if not isinstance(ar, dict) or set(ar) != {"size"}:
        errs.append("area must have exactly the key size")
    else:
        S = _num(ar["size"], "area.size", errs, *LIMITS["area"])
    light = layout.get("light", "noon")
    if light not in LIGHTS:
        errs.append(f"light must be one of {sorted(LIGHTS)}")
    if S is None or errs:
        return {"ok": False, "errors": errs, "scene": None}
    roads: dict[str, dict] = {}; r_in = layout.get("roads", [])
    if not isinstance(r_in, list) or len(r_in) > LIMITS["roads"]:
        errs.append(f"roads must be a list of at most {LIMITS['roads']}"); r_in = []
    for i, r in enumerate(r_in):
        t = f"roads[{i}]"
        if not isinstance(r, dict) or set(r) != {"id", "from", "to", "width"} or not isinstance(r["id"], str) or r["id"] in roads:
            errs.append(f"{t}: needs exactly id (unique), from, to, width"); continue
        a, b, w = _vec(r["from"], 2, f"{t}.from", errs), _vec(r["to"], 2, f"{t}.to", errs), _num(r["width"], f"{t}.width", errs, *LIMITS["road_w"])
        if None in (a, b, w):
            continue
        if any(abs(v) > S for v in a + b):
            errs.append(f"{t}: endpoints must lie inside the area (|x|,|y| <= {S:g})"); continue
        if math.dist(a, b) < 10:
            errs.append(f"{t}: a road must be at least 10 m long"); continue
        L = math.dist(a, b); roads[r["id"]] = {"id": r["id"], "from": a, "to": b, "width": w, "dir": [(b[0] - a[0]) / L, (b[1] - a[1]) / L], "length": L}
    items_in = layout.get("items", [])
    if not isinstance(items_in, list) or len(items_in) > LIMITS["items"]:
        errs.append(f"items must be a list of at most {LIMITS['items']}"); items_in = []
    placed: dict[str, dict] = {}; pending = list(enumerate(items_in)); rl = list(roads.values())
    for _ in range(len(pending) + 1):
        progress = False
        for idx, it in list(pending):
            tag = f"items[{idx}]"
            if not isinstance(it, dict) or set(it) - {"id", "asset", "at", "yaw_deg"} or not {"id", "asset", "at"} <= set(it):
                errs.append(f"{tag}: needs id, asset, at (optional yaw_deg), nothing else"); pending.remove((idx, it)); continue
            iid, asset = it["id"], it["asset"]
            if not isinstance(iid, str) or not iid or iid in placed:
                errs.append(f"{tag}: id {iid!r} is empty or duplicated"); pending.remove((idx, it)); continue
            if asset not in DIMS or not DIMS[asset]["ok"] or asset in UNLABELLED or REGISTRY.get(asset, {}).get("group") not in GROUPS:
                errs.append(f"{tag} ({iid}): asset {asset!r} is not in the catalogue"); pending.remove((idx, it)); continue
            yaw = _num(it.get("yaw_deg", 0.0), f"{tag}.yaw_deg", errs, -360.0, 360.0)
            if yaw is None:
                pending.remove((idx, it)); continue
            grp = REGISTRY[asset]["group"]; sx, sy, sz, nominal = _size(asset, iid); at = it["at"]; pos = None; base_yaw = 0.0
            if isinstance(at, list):
                pos = _vec(at, 2, f"{tag}.at", errs)
            elif isinstance(at, dict) and set(at) == {"road", "s", "lane"}:
                rd = roads.get(at["road"])
                if at["s"] == "auto":                                          # intent only: the system picks a free spot along the road (deterministic per layout seed and item id)
                    s = _auto_s(rd, at["lane"], grp, sx, sy, yaw, iid, placed, rl, S, int(layout.get("seed", 0)) if isinstance(layout.get("seed", 0), int) else 0)
                    if rd is not None and s is None:
                        errs.append(f"{tag} ({iid}): no free spot along road {at['road']!r} for lane {at['lane']!r} (try the other lane or road, or fewer items)")
                        pending.remove((idx, it)); progress = True; continue
                else:
                    s = _num(at["s"], f"{tag}.at.s", errs, 0.0, 1.0)
                if rd is None:
                    errs.append(f"{tag} ({iid}): road {at['road']!r} does not exist")
                elif s is not None:
                    lane = at["lane"]; dx, dy = rd["dir"]; nx, ny = -dy, dx; cx = rd["from"][0] + dx * rd["length"] * s; cy = rd["from"][1] + dy * rd["length"] * s
                    if grp == "car" and lane in LANES:
                        off = LANES[lane] * rd["width"] / 4; pos = [cx + nx * off, cy + ny * off]; base_yaw = math.degrees(math.atan2(dy, dx)) + (0 if lane == "lane_right" else 180)
                    elif grp != "car" and lane in KERBS:
                        off = KERBS[lane] * (rd["width"] / 2 + SIDEWALK + max(sx, sy) / 2 + 0.5); pos = [cx + nx * off, cy + ny * off]
                    else:
                        errs.append(f"{tag} ({iid}): lane {lane!r} does not fit a {grp} (cars: lane_left|lane_right, others: kerb_left|kerb_right)")
            elif isinstance(at, dict) and set(at) == {"area"} and at["area"] == "auto":
                pos = _auto_xy(grp, sx, sy, yaw, iid, placed, rl, S, int(layout.get("seed", 0)) if isinstance(layout.get("seed", 0), int) else 0)
                if pos is None:
                    errs.append(f"{tag} ({iid}): no free spot in the area for {asset} ({'cars need at road/lane' if grp == 'car' else 'the area is full: use fewer items'})")
                    pending.remove((idx, it)); progress = True; continue
            elif isinstance(at, dict) and set(at) == {"near", "side", "gap"} and at["side"] in ("north", "south", "east", "west"):
                ref = placed.get(at["near"])
                if ref is None:
                    if at["near"] in {x.get("id") for _, x in pending if isinstance(x, dict)}:
                        continue
                    errs.append(f"{tag} ({iid}): near {at['near']!r} does not exist"); pending.remove((idx, it)); continue
                gap = _num(at["gap"], f"{tag}.at.gap", errs, 0.0, 30.0)
                if gap is not None:
                    fx, fy = _rot(sx, sy, yaw); rx, ry = ref["pos"][:2]; rw, rd_ = ref["footprint"][:2]
                    pos = {"north": [rx, ry + rd_ / 2 + gap + fy / 2], "south": [rx, ry - rd_ / 2 - gap - fy / 2], "east": [rx + rw / 2 + gap + fx / 2, ry], "west": [rx - rw / 2 - gap - fx / 2, ry]}[at["side"]]
            else:
                errs.append(f"{tag} ({iid}): at must be [x, y], {{road, s, lane}}, {{area: auto}} or {{near, side, gap}}")
            pending.remove((idx, it)); progress = True
            if pos is None:
                continue
            tot_yaw = yaw + base_yaw; fx, fy = _rot(sx, sy, tot_yaw)
            if abs(pos[0]) + fx / 2 > S or abs(pos[1]) + fy / 2 > S:
                errs.append(f"{tag} ({iid}): {asset} ({fx:.1f} x {fy:.1f} m) centred at ({pos[0]:.1f}, {pos[1]:.1f}) leaves the area (|x|,|y| <= {S:g})"); continue
            d_edge, rd_near = _road_edge_dist(pos[0], pos[1], fx / 2, fy / 2, rl)
            if grp == "car" and rl and d_edge > -1.0:
                errs.append(f"{tag} ({iid}): a car must stand on a carriageway (use at road/lane)"); continue
            if grp != "car" and rl and d_edge < 0.3:
                errs.append(f"{tag} ({iid}): {asset} stands on the carriageway of road {rd_near['id']!r}; move it by at least {0.3 - d_edge:.1f} m or use kerb_left/kerb_right"); continue
            if grp == "building" and rl and d_edge < SIDEWALK + BUILD_GAP:
                errs.append(f"{tag} ({iid}): building is {d_edge:.1f} m from the carriageway, needs {SIDEWALK + BUILD_GAP:g} m (sidewalk + setback)"); continue
            gap_m = BUILD_GAP if grp == "building" else ITEM_GAP
            for oid, o in placed.items():
                if (grp == "car") != (o["group"] == "car"):
                    continue                                              # cars live on carriageways, everything else off them: they cannot touch
                m = max(gap_m, BUILD_GAP if o["group"] == "building" else ITEM_GAP)
                if abs(pos[0] - o["pos"][0]) < (fx + o["footprint"][0]) / 2 + m and abs(pos[1] - o["pos"][1]) < (fy + o["footprint"][1]) / 2 + m:
                    errs.append(f"{tag} ({iid}) overlaps item {oid!r} (keep {m:g} m apart)"); break
            else:
                placed[iid] = {"id": iid, "asset": asset, "group": grp, "tag": _tag(asset), "cls": REGISTRY[asset]["cls"], "uri": REGISTRY[asset]["uri"], "kind": REGISTRY[asset]["kind"],
                               "pos": [pos[0], pos[1], 0.0], "yaw_deg": tot_yaw, "footprint": [fx, fy, sz], "height_m": sz if grp == "tree" else None, "pole_height_m": 7.0 if nominal and grp == "pole" else None}
        if not pending or not progress:
            break
    for _, it in pending:
        errs.append(f"item {it.get('id') if isinstance(it, dict) else it!r}: circular or unresolved 'near' reference")
    cam = layout.get("camera"); poses = _camera(cam, S, placed, roads, errs) if cam is not None else None
    if errs:
        return {"ok": False, "errors": errs, "scene": None}
    scene = {"area": S, "light": light, "roads": list(roads.values()), "items": list(placed.values()), "camera": poses, "seed": int(layout.get("seed", 0))}
    scene["hash"] = hashlib.sha1(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:12]
    return {"ok": True, "errors": [], "scene": scene}


def _camera(cam, S, placed, roads, errs):
    keys = {"waypoints", "look_at", "roll_deg", "speed_mps", "fps", "hfov_deg"}
    if not isinstance(cam, dict) or set(cam) - keys or "waypoints" not in cam:
        errs.append(f"camera needs waypoints; allowed keys {sorted(keys)}"); return None
    wp = cam["waypoints"]
    if not isinstance(wp, list) or not 1 <= len(wp) <= 20:
        errs.append("camera.waypoints must hold 1..20 points"); return None
    pts = [_vec(p, 3, f"camera.waypoints[{i}]", errs) for i, p in enumerate(wp)]
    if None in pts:
        return None
    roll = _num(cam.get("roll_deg", 0.0), "camera.roll_deg", errs, -180.0, 180.0); speed = _num(cam.get("speed_mps", 5.0), "camera.speed_mps", errs, *LIMITS["speed"])
    fps = _num(cam.get("fps", 30.0), "camera.fps", errs, *LIMITS["fps"]); hfov = _num(cam.get("hfov_deg", 90.0), "camera.hfov_deg", errs, *LIMITS["hfov"])
    look = cam.get("look_at"); look = None if look is None else _vec(look, 3, "camera.look_at", errs)
    if None in (roll, speed, fps, hfov) or (cam.get("look_at") is not None and look is None):
        return None
    length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:])); n = 1 if len(pts) == 1 or speed == 0 else max(2, math.ceil(length / speed * fps))
    if n != 1 and not LIMITS["frames"][0] <= n <= LIMITS["frames"][1]:
        errs.append(f"path of {length:.0f} m at {speed} m/s and {fps} fps gives {n} frames; allowed {LIMITS['frames']}"); return None
    samples = []
    for k in range(n):
        d = length * k / max(n - 1, 1); acc = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = math.dist(a, b)
            if acc + seg >= d - 1e-9 or b is pts[-1]:
                t = 0.0 if seg < 1e-9 else min(max((d - acc) / seg, 0.0), 1.0); samples.append([a[i] + (b[i] - a[i]) * t for i in range(3)]); break
            acc += seg
        else:
            samples.append(list(pts[0]))
    for k, p in enumerate(samples):
        what = f"camera frame {k}" if n > 1 else "camera position"
        if abs(p[0]) > S or abs(p[1]) > S or not LIMITS["alt"][0] <= p[2] <= LIMITS["alt"][1]:
            errs.append(f"{what} {[round(v, 1) for v in p]} is outside the area (|x|,|y| <= {S:g}) or the height range {LIMITS['alt']}"); return None
        for o in placed.values():
            top = o["pos"][2] + o["footprint"][2]
            if abs(p[0] - o["pos"][0]) < o["footprint"][0] / 2 + CAM_ITEM and abs(p[1] - o["pos"][1]) < o["footprint"][1] / 2 + CAM_ITEM and p[2] < top + CAM_ITEM:
                errs.append(f"{what} {[round(v, 1) for v in p]} is inside or within {CAM_ITEM:g} m of item {o['id']!r} (its top is {top:.1f} m); fly higher or move sideways"); return None
    poses = []
    for k, p in enumerate(samples):
        if look is not None:
            fwd = [look[i] - p[i] for i in range(3)]
        else:
            q = samples[k + 1] if k + 1 < n else p; r = samples[k - 1] if k else p; fwd = [q[i] - r[i] for i in range(3)]
            if math.hypot(*fwd) < 1e-9:
                fwd = [1.0, 0.0, 0.0]
        nrm = math.sqrt(sum(v * v for v in fwd)) or 1.0; pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2] / nrm))))
        if abs(pitch) > MAX_PITCH_DEG:
            errs.append(f"camera frame {k} looks {pitch:.0f} deg {'down' if pitch < 0 else 'up'} (limit {MAX_PITCH_DEG:g})"); return None
        poses.append({"pos": p, "forward": [v / nrm for v in fwd]})
    if placed:
        hh = math.radians(hfov / 2); hv = math.atan(math.tan(hh) * 0.75); seen = 0
        for p_ in poses:
            f = p_["forward"]; yf = math.atan2(f[1], f[0]); pf = math.asin(max(-1.0, min(1.0, f[2])))
            for o in placed.values():
                v = [o["pos"][0] - p_["pos"][0], o["pos"][1] - p_["pos"][1], o["footprint"][2] / 2 - p_["pos"][2]]; hd = math.hypot(v[0], v[1])
                if 1.0 < hd < 150 and abs((math.atan2(v[1], v[0]) - yf + math.pi) % (2 * math.pi) - math.pi) < hh and abs(math.atan2(v[2], hd) - pf) < hv:
                    seen += 1; break
        if seen < MIN_SEEN_FRAC * len(poses):
            errs.append(f"camera sees an item in only {seen} of {len(poses)} frames (need {MIN_SEEN_FRAC:.0%}); aim the path or look_at at the objects"); return None
    return {"poses": poses, "roll_deg": roll, "speed_mps": speed, "fps": fps, "hfov_deg": hfov, "length_m": length}
