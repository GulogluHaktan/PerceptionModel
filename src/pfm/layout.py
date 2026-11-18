"""Declarative indoor layout (no Isaac import): the LLM never writes code or sizes, it only picks asset ids and coordinates; this module resolves everything else.

LLM writes:   room size, light, items (asset id + where), camera waypoints.
We resolve:   real-world size (from assets/models/dims.json, the model is already rescaled to its tag's typical size), floor contact, wall anchoring, yaw footprint,
              overlap and clearance checks, camera path sampling (speed, fps, roll). Rules reject instead of silently fixing, and every error names the item and the limit,
              so the reply can go straight back to the LLM for a repair round (same loop as pfm.llm_scenario).
Frame: Z up, floor at z=0, the room spans x in [0, w], y in [0, d], ceiling at z=h. Walls: north = y=d, south = y=0, east = x=w, west = x=0.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .assets import REGISTRY

DIMS = json.loads((Path(__file__).resolve().parents[2] / "assets" / "models" / "dims.json").read_text())
ALLOWED_GROUPS = {"indoor"}                        # extend (two_wheeler, prop, ...) once their size/orientation are checked
LIMITS = {"w": (2.5, 12.0), "d": (2.5, 12.0), "h": (2.2, 4.0), "items": 40, "speed": (0.0, 8.0), "fps": (1.0, 120.0), "frames": (2, 600), "hfov": (20.0, 140.0)}
LIGHTS = {"day", "lamp", "dark"}
WALLS = {"north", "south", "east", "west"}
WALL_GAP, ITEM_GAP, CAM_WALL, CAM_ITEM = 0.02, 0.05, 0.25, 0.15
MIN_SEEN_FRAC = 0.4                                    # share of camera frames that must have at least one item in view (frames of empty walls teach nothing)
MAX_PITCH_DEG = 55.0                                   # a camera staring at the floor or ceiling gives useless frames for perception
TOP = {"room", "light", "items", "openings", "camera", "seed"}
OPENING = {"door": {"w": 0.9, "z0": 0.0, "z1": 2.0}, "window": {"w": 1.2, "z0": 0.9, "z1": 1.9}}        # sizes are ours, the LLM only picks kind, wall and position
SUPPORT_TAGS = {"table", "desk", "dresser", "stove", "cabinet", "bookcase"}                              # things you may put stuff on
SMALL_MAX_M, DOOR_CLEAR_M, OPEN_MARGIN = 0.9, 0.8, 0.15


def catalogue(groups=ALLOWED_GROUPS) -> list[dict]:
    """What the LLM is shown: id, tag and the real size (x, y, z in metres). Nothing else about the asset is its business."""
    return [{"id": n, "tag": a["tag"], "size_m": DIMS[n]} for n, a in REGISTRY.items() if a["group"] in groups and n in DIMS]


def catalogue_text(groups=ALLOWED_GROUPS) -> str:
    return "\n".join(f'{c["id"]} | {c["tag"]} | {c["size_m"][0]:.2f} x {c["size_m"][1]:.2f} x {c["size_m"][2]:.2f} m' for c in catalogue(groups))


SCHEMA_TEXT = f"""Reply with ONLY a JSON object. Keys: room, light, items, camera, seed. No code, no sizes, no text outside the JSON.
room: {{w, d, h}} metres, w,d in {LIMITS['w']}, h in {LIMITS['h']}. x runs 0..w, y runs 0..d, z up, floor z=0.
light: one of {sorted(LIGHTS)}.
items: list (max {LIMITS['items']}) of {{id: unique short name, asset: an id from the catalogue, at: [x, y] OR {{wall: north|south|east|west, along: 0..1}} OR {{near: <other item id>, side: north|south|east|west, gap: metres}} OR {{on: <table/desk/dresser/stove/cabinet/bookcase item id>, dx, dy: offset from its centre}} (only items up to {SMALL_MAX_M} m can sit on something), yaw_deg: float}}.
camera: {{waypoints: [[x,y,z], ...] (1..20 points; x in [0.25, w-0.25], y in [0.25, d-0.25], z in [0.3, h-0.2], so never above the ceiling), look_at: [x,y,z] or null (null = look along the motion), roll_deg: -180..180 (180 = upside down), speed_mps: {LIMITS['speed']}, fps: {LIMITS['fps']}, hfov_deg: {LIMITS['hfov']}}}.
openings: optional list of {{id, kind: door|window, wall: north|south|east|west, along: 0..1}} (the system sets their size; a door needs {DOOR_CLEAR_M} m of free floor in front, tall items may not cover a window).
Tips: at least 40% of the camera frames must have an item in view; the camera may not look more than 55 degrees up or down, so look_at must be far enough away horizontally (several metres) and at a similar height; put chairs, stools and small tables NEXT to big furniture with near (gap 0.1-0.4), put big furniture against walls with wall, never give two items the same [x, y]; windows and doors go in openings, not in items.
seed: int. Items may not overlap, may not leave the room; the camera keeps {CAM_WALL} m from walls and {CAM_ITEM} m from items."""


def _footprint(name: str, yaw_deg: float) -> tuple[float, float, float]:
    dx, dy, dz = DIMS[name]; a = math.radians(yaw_deg); c, s = abs(math.cos(a)), abs(math.sin(a))
    return dx * c + dy * s, dx * s + dy * c, dz                          # axis-aligned extent of the rotated box (conservative)


def _num(v, name, errs, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        errs.append(f"{name} must be a finite number, got {v!r}"); return None
    if lo is not None and not lo <= v <= hi:
        errs.append(f"{name}={v} outside [{lo}, {hi}]"); return None
    return float(v)


def _vec(v, n, name, errs):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        errs.append(f"{name} must be a list of {n} numbers"); return None
    out = [_num(x, f"{name}[{i}]", errs) for i, x in enumerate(v)]
    return None if None in out else out


def _free_spot(pos, fx, fy, W, D, placed, z0):
    """Nearest centre (0.1 m grid) to `pos` where an item of this footprint fits in the room and touches no item on the same level: the hint that goes into overlap errors."""
    best = None
    for gx in range(int(fx / 2 * 10) + 1, int((W - fx / 2) * 10)):
        for gy in range(int(fy / 2 * 10) + 1, int((D - fy / 2) * 10)):
            x, y = gx / 10, gy / 10
            if any(abs(o["pos"][2] - z0) < 1e-6 and abs(x - o["pos"][0]) < (fx + o["footprint"][0]) / 2 + ITEM_GAP and abs(y - o["pos"][1]) < (fy + o["footprint"][1]) / 2 + ITEM_GAP for o in placed.values()):
                continue
            d = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
            if best is None or d < best[0]:
                best = (d, [x, y])
    return best[1] if best else None


def resolve(layout: dict) -> dict:
    """Returns {ok, errors, scene}. `scene` (only when ok) is what the renderer consumes: room, light, items (pos, yaw, uri, class, tag, bbox), camera poses."""
    errs: list[str] = []
    if not isinstance(layout, dict):
        return {"ok": False, "errors": ["layout must be a JSON object"], "scene": None}
    if set(layout) - TOP:
        errs.append(f"unknown top-level keys {sorted(set(layout) - TOP)}")
    room = layout.get("room", {}); R = {}
    if not isinstance(room, dict) or set(room) != {"w", "d", "h"}:
        errs.append("room must have exactly the keys w, d, h")
    else:
        for k in "wdh":
            R[k] = _num(room[k], f"room.{k}", errs, *LIMITS[k])
    light = layout.get("light", "day")
    if light not in LIGHTS:
        errs.append(f"light must be one of {sorted(LIGHTS)}")
    items_in = layout.get("items", [])
    if not isinstance(items_in, list) or len(items_in) > LIMITS["items"]:
        errs.append(f"items must be a list of at most {LIMITS['items']}"); items_in = []
    if errs or None in R.values():
        return {"ok": False, "errors": errs, "scene": None}
    W, D, H = R["w"], R["d"], R["h"]
    placed: dict[str, dict] = {}; pending = list(enumerate(items_in))
    for _ in range(len(pending) + 1):                                    # "near" items wait for the item they refer to
        progress = False
        for idx, it in list(pending):
            tag = f"items[{idx}]"
            if not isinstance(it, dict) or set(it) - {"id", "asset", "at", "yaw_deg"} or not {"id", "asset", "at"} <= set(it):
                errs.append(f"{tag}: needs id, asset, at (optional yaw_deg), nothing else"); pending.remove((idx, it)); continue
            iid, asset = it["id"], it["asset"]
            if not isinstance(iid, str) or not iid or iid in placed:
                errs.append(f"{tag}: id {iid!r} is empty or duplicated"); pending.remove((idx, it)); continue
            if asset not in DIMS or REGISTRY.get(asset, {}).get("group") not in ALLOWED_GROUPS:
                hint = " (windows and doors go in `openings`, not in `items`)" if any(w in f"{asset} {iid}".lower() for w in ("window", "door")) else ""
                errs.append(f"{tag} ({iid}): asset {asset!r} is not in the catalogue{hint}"); pending.remove((idx, it)); continue
            yaw = _num(it.get("yaw_deg", 0.0), f"{tag}.yaw_deg", errs, -360.0, 360.0)
            if yaw is None:
                pending.remove((idx, it)); continue
            fx, fy, fz = _footprint(asset, yaw); at = it["at"]; pos = None; z0 = 0.0
            if isinstance(at, list):
                v = _vec(at, 2, f"{tag}.at", errs); pos = v
            elif isinstance(at, dict) and set(at) == {"wall", "along"} and at["wall"] in WALLS:
                t = _num(at["along"], f"{tag}.at.along", errs, 0.0, 1.0)
                if t is not None:
                    wall = at["wall"]
                    pos = {"south": [fx / 2 + (W - fx) * t, fy / 2 + WALL_GAP], "north": [fx / 2 + (W - fx) * t, D - fy / 2 - WALL_GAP],
                           "west": [fx / 2 + WALL_GAP, fy / 2 + (D - fy) * t], "east": [W - fx / 2 - WALL_GAP, fy / 2 + (D - fy) * t]}[wall]
            elif isinstance(at, dict) and set(at) == {"near", "side", "gap"} and at["side"] in WALLS:
                ref = placed.get(at["near"])
                if ref is None:
                    if at["near"] in {x.get("id") for _, x in pending if isinstance(x, dict)}:
                        continue                                         # reference not placed yet: try again next sweep
                    errs.append(f"{tag} ({iid}): near {at['near']!r} does not exist"); pending.remove((idx, it)); continue
                gap = _num(at["gap"], f"{tag}.at.gap", errs, 0.0, 3.0)
                if gap is not None:
                    rx, ry = ref["pos"][:2]; rw, rd = ref["footprint"][:2]
                    pos = {"north": [rx, ry + rd / 2 + gap + fy / 2], "south": [rx, ry - rd / 2 - gap - fy / 2], "east": [rx + rw / 2 + gap + fx / 2, ry], "west": [rx - rw / 2 - gap - fx / 2, ry]}[at["side"]]
            elif isinstance(at, dict) and set(at) == {"on", "dx", "dy"}:
                host = placed.get(at["on"]); dx_ = _num(at["dx"], f"{tag}.at.dx", errs); dy_ = _num(at["dy"], f"{tag}.at.dy", errs)
                if host is None:
                    if at["on"] in {x.get("id") for _, x in pending if isinstance(x, dict)}:
                        continue
                    errs.append(f"{tag} ({iid}): host {at['on']!r} does not exist"); pending.remove((idx, it)); continue
                if dx_ is not None and dy_ is not None:
                    if host["tag"] not in SUPPORT_TAGS or max(DIMS[asset]) > SMALL_MAX_M or host["pos"][2] > 0:
                        errs.append(f"{tag} ({iid}): cannot sit on {at['on']!r} ({host['tag']}) or is bigger than {SMALL_MAX_M} m or the host is itself stacked")
                    elif abs(dx_) > (host["footprint"][0] - fx) / 2 + 1e-6 or abs(dy_) > (host["footprint"][1] - fy) / 2 + 1e-6:
                        errs.append(f"{tag} ({iid}): leaves the top of {at['on']!r} (|dx| <= {(host['footprint'][0] - fx) / 2:.2f}, |dy| <= {(host['footprint'][1] - fy) / 2:.2f})")
                    else:
                        pos = [host["pos"][0] + dx_, host["pos"][1] + dy_]; z0 = host["footprint"][2]
            else:
                errs.append(f"{tag} ({iid}): at must be [x, y], {{wall, along}}, {{near, side, gap}} or {{on, dx, dy}}")
            pending.remove((idx, it)); progress = True
            if pos is None:
                continue
            if z0 + fz > H - 0.05:
                errs.append(f"{tag} ({iid}): {asset} reaches {z0 + fz:.2f} m, the room is only {H} m high")
            if pos[0] - fx / 2 < -1e-6 or pos[0] + fx / 2 > W + 1e-6 or pos[1] - fy / 2 < -1e-6 or pos[1] + fy / 2 > D + 1e-6:
                errs.append(f"{tag} ({iid}): {asset} ({fx:.2f} x {fy:.2f} m at yaw {yaw:g}) centred at ({pos[0]:.2f}, {pos[1]:.2f}) leaves the room "
                            f"(centre must be in x [{fx / 2:.2f}, {W - fx / 2:.2f}], y [{fy / 2:.2f}, {D - fy / 2:.2f}])"); continue
            for oid, o in placed.items():
                if abs(o["pos"][2] - z0) > 1e-6:
                    continue                                              # different level (stacked vs floor): the host check above covers it
                if abs(pos[0] - o["pos"][0]) < (fx + o["footprint"][0]) / 2 + ITEM_GAP and abs(pos[1] - o["pos"][1]) < (fy + o["footprint"][1]) / 2 + ITEM_GAP:
                    spot = _free_spot(pos, fx, fy, W, D, placed, z0)
                    errs.append(f"{tag} ({iid}) overlaps item {oid!r} (keep {ITEM_GAP} m apart)" + (f"; nearest free centre for it: {spot}, or use at near/wall" if spot else "; no free floor left")); break
            else:
                placed[iid] = {"id": iid, "asset": asset, "pos": [pos[0], pos[1], z0], "yaw_deg": yaw, "footprint": [fx, fy, fz], "uri": REGISTRY[asset]["uri"],
                               "cls": REGISTRY[asset]["cls"], "tag": REGISTRY[asset]["tag"], "target_size_m": max(DIMS[asset])}
        if not pending or not progress:
            break
    for _, it in pending:
        errs.append(f"item {it.get('id') if isinstance(it, dict) else it!r}: circular or unresolved 'near' reference")
    ops = _openings(layout.get("openings", []), W, D, H, placed, errs)
    cam = layout.get("camera")
    poses = None
    if cam is not None:
        poses = _camera(cam, W, D, H, placed, errs)
    if errs:
        return {"ok": False, "errors": errs, "scene": None}
    scene = {"room": R, "light": light, "items": list(placed.values()), "openings": ops, "camera": poses, "seed": int(layout.get("seed", 0))}
    scene["hash"] = hashlib.sha1(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:12]
    return {"ok": True, "errors": [], "scene": scene}


def _camera(cam, W, D, H, placed, errs):
    keys = {"waypoints", "look_at", "roll_deg", "speed_mps", "fps", "hfov_deg"}
    if not isinstance(cam, dict) or set(cam) - keys or "waypoints" not in cam:
        errs.append(f"camera needs waypoints; allowed keys {sorted(keys)}"); return None
    wp = cam["waypoints"]
    if not isinstance(wp, list) or not 1 <= len(wp) <= 20:
        errs.append("camera.waypoints must hold 1..20 points"); return None
    pts = [_vec(p, 3, f"camera.waypoints[{i}]", errs) for i, p in enumerate(wp)]
    if None in pts:
        return None
    roll = _num(cam.get("roll_deg", 0.0), "camera.roll_deg", errs, -180.0, 180.0); speed = _num(cam.get("speed_mps", 1.0), "camera.speed_mps", errs, *LIMITS["speed"])
    fps = _num(cam.get("fps", 30.0), "camera.fps", errs, *LIMITS["fps"]); hfov = _num(cam.get("hfov_deg", 90.0), "camera.hfov_deg", errs, *LIMITS["hfov"])
    look = cam.get("look_at"); look = None if look is None else _vec(look, 3, "camera.look_at", errs)
    if None in (roll, speed, fps, hfov) or (cam.get("look_at") is not None and look is None):
        return None

    def bad(p, what):
        if not (CAM_WALL <= p[0] <= W - CAM_WALL and CAM_WALL <= p[1] <= D - CAM_WALL and 0.3 <= p[2] <= H - 0.2):
            return f"{what} {[round(v, 2) for v in p]} is outside the allowed camera box x [{CAM_WALL}, {W - CAM_WALL:g}], y [{CAM_WALL}, {D - CAM_WALL:g}], z [0.3, {H - 0.2:g}]"
        for o in placed.values():
            if abs(p[0] - o["pos"][0]) < o["footprint"][0] / 2 + CAM_ITEM and abs(p[1] - o["pos"][1]) < o["footprint"][1] / 2 + CAM_ITEM and p[2] < o["pos"][2] + o["footprint"][2] + CAM_ITEM:
                return f"{what} {[round(v, 2) for v in p]} is inside or too close to item {o['id']!r}"
        return None
    if look is not None and not (0 <= look[0] <= W and 0 <= look[1] <= D and 0 <= look[2] <= H):
        errs.append("camera.look_at is outside the room"); return None
    length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    n = 1 if len(pts) == 1 or speed == 0 else max(2, math.ceil(length / speed * fps))
    if not LIMITS["frames"][0] <= n <= LIMITS["frames"][1] and n != 1:
        errs.append(f"path of {length:.1f} m at {speed} m/s and {fps} fps gives {n} frames; allowed {LIMITS['frames']}"); return None
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
        e = bad(p, f"camera frame {k}" if n > 1 else "camera position")
        if e:
            errs.append(e); return None
    poses = []
    for k, p in enumerate(samples):
        if look is not None:
            fwd = [look[i] - p[i] for i in range(3)]
        else:
            q = samples[min(k + 1, n - 1)] if k + 1 < n else p; r = samples[k - 1] if k else p; fwd = [q[i] - r[i] for i in range(3)]
            if math.hypot(*fwd) < 1e-9:
                fwd = [math.cos(0.0), math.sin(0.0), 0.0]
        nrm = math.sqrt(sum(v * v for v in fwd)) or 1.0
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2] / nrm))))
        if abs(pitch) > MAX_PITCH_DEG:
            errs.append(f"camera frame {k} looks {pitch:.0f} deg {'down' if pitch < 0 else 'up'} (limit {MAX_PITCH_DEG:g}); raise/lower the look_at point or use look_at null with a mostly horizontal path"); return None
        poses.append({"pos": p, "forward": [v / nrm for v in fwd]})
    if placed:
        half_h = math.radians(hfov / 2); half_v = math.atan(math.tan(half_h) * 0.75); seen = 0
        for p_ in poses:
            f = p_["forward"]; yaw_f = math.atan2(f[1], f[0]); pit_f = math.asin(max(-1.0, min(1.0, f[2])))
            for o in placed.values():
                v = [o["pos"][0] - p_["pos"][0], o["pos"][1] - p_["pos"][1], o["pos"][2] + o["footprint"][2] / 2 - p_["pos"][2]]; hd = math.hypot(v[0], v[1])
                dy = abs((math.atan2(v[1], v[0]) - yaw_f + math.pi) % (2 * math.pi) - math.pi); dp = abs(math.atan2(v[2], hd) - pit_f)
                if hd > 0.3 and dy < half_h and dp < half_v:
                    seen += 1; break
        if seen < MIN_SEEN_FRAC * len(poses):
            errs.append(f"camera sees an item in only {seen} of {len(poses)} frames (need {MIN_SEEN_FRAC:.0%}); aim the path or look_at at the furniture"); return None
    return {"poses": poses, "roll_deg": roll, "speed_mps": speed, "fps": fps, "hfov_deg": hfov, "length_m": length}


def _openings(op_in, W, D, H, placed, errs):
    """Doors and windows cut into a wall. Rectangle in wall coordinates: u0..u1 along the wall (south/north: x, west/east: y), z0..z1 up."""
    if not isinstance(op_in, list) or len(op_in) > 12:
        errs.append("openings must be a list of at most 12"); return []
    out = []
    for i, o in enumerate(op_in):
        tag = f"openings[{i}]"
        if not isinstance(o, dict) or set(o) != {"id", "kind", "wall", "along"} or o["kind"] not in OPENING or o["wall"] not in WALLS:
            errs.append(f"{tag}: needs exactly id, kind (door|window), wall, along"); continue
        t = _num(o["along"], f"{tag}.along", errs, 0.0, 1.0)
        if t is None:
            continue
        spec = OPENING[o["kind"]]; L = W if o["wall"] in ("south", "north") else D; w = spec["w"]
        if o["kind"] == "door" and H < 2.1:
            errs.append(f"{tag}: a door needs a room height of at least 2.1 m"); continue
        if L < w + 2 * OPEN_MARGIN:
            errs.append(f"{tag}: wall {o['wall']} is too short for a {o['kind']}"); continue
        c = OPEN_MARGIN + w / 2 + (L - w - 2 * OPEN_MARGIN) * t; u0, u1 = c - w / 2, c + w / 2
        for q in out:
            if q["wall"] == o["wall"] and u0 < q["u1"] + 0.1 and u1 > q["u0"] - 0.1:
                errs.append(f"{tag} ({o['id']}) overlaps opening {q['id']!r} on the {o['wall']} wall"); break
        else:
            for it in placed.values():
                fx, fy, fz = it["footprint"]; x, y = it["pos"][:2]
                lo_u, hi_u = (x - fx / 2, x + fx / 2) if o["wall"] in ("south", "north") else (y - fy / 2, y + fy / 2)
                depth = {"south": y - fy / 2, "north": D - (y + fy / 2), "west": x - fx / 2, "east": W - (x + fx / 2)}[o["wall"]]      # item distance from that wall
                if lo_u < u1 and hi_u > u0 and o["kind"] == "door" and depth < DOOR_CLEAR_M and it["pos"][2] == 0:
                    errs.append(f"{tag} ({o['id']}): door is blocked by item {it['id']!r} (needs {DOOR_CLEAR_M} m of free floor in front)"); break
                if lo_u < u1 and hi_u > u0 and o["kind"] == "window" and depth < 0.15 and it["pos"][2] + fz > spec["z0"]:
                    errs.append(f"{tag} ({o['id']}): window is covered by item {it['id']!r} (taller than the {spec['z0']} m sill, against the wall)"); break
            else:
                out.append({"id": o["id"], "kind": o["kind"], "wall": o["wall"], "u0": u0, "u1": u1, "z0": spec["z0"], "z1": min(spec["z1"], H - 0.1)})
    return out
