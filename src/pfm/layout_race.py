"""Declarative RACING-TRACK layout (no Isaac import, no numpy), same principle as pfm.layout / pfm.layout_out: the LLM picks gate positions and flight settings, never code or sizes.

LLM writes:  setting, light, gates (absolute [x,y,z,yaw] or relative to the previous gate: dist, turn, dz), flight (speed up to 30 m/s, fps, camera tilt, lateral offset through the gates, focus window).
We resolve:  the A2RL x DCL gate size (2.7 m outer, 1.5 m opening, 0.6 m bar, 0.1 m deep: same numbers as /home/haktan/Projects/DroneRacing isaac/vision_rig.py), the smooth flight line through
             the gates, constant-speed camera poses, and the EXACT ground-truth labels in DroneRacing format: per frame the 4 inner corners (TL, TR, BR, BL; seen from the side the drone approaches)
             of every gate in front, in 256x192 px at 130 deg, also when the corner is outside the image (so close / partly visible gates keep a target), plus a visible flag per corner.
Conventions as DroneRacing: gate yaw = direction of passage, normal = (cos y, sin y, 0), right = (sin y, -cos y, 0), Z up, gate centre z = 1.45 m for a low gate.
Rules reject instead of silently fixing; every error names the gate and the limit (repair loop as indoors).
"""
from __future__ import annotations

import hashlib
import json
import math

GATE_OUTER, GATE_INNER, GATE_BAR, GATE_DEPTH = 2.7, 1.5, 0.6, 0.1
IMG_W, IMG_H = 256, 192                                      # DroneRacing camera
SETTINGS = {"field", "hall"}
LIGHTS = {"noon", "overcast", "dusk"}
LIMITS = {"gates": (2, 24), "gap": (3.0, 40.0), "z": (0.8, 6.0), "area": 150.0, "turn": 100.0, "dz": 3.0, "speed": (1.0, 30.0), "fps": (10.0, 120.0), "tilt": (0.0, 50.0),
          "offset": 0.6, "focus": (3.0, 25.0), "frames": (4, 600), "hfov": (90.0, 160.0), "roll": 60.0}
TOP = {"setting", "light", "gates", "flight", "seed", "template"}
# fixed real tracks (system-owned, trusted): the A2RL x DCL 2025 final lap as DroneRacing reads it off the KAIST plot (droneracing.track.a2rl_2025_lap): x, y, z, yaw_deg, shifted to the hall centre
TEMPLATES = {"a2rl_2025": [(14.0, 25.0, 1.45, 30.0), (30.5, 19.5, 1.45, -18.0), (46.0, 22.0, 1.45, 10.0), (63.0, 20.0, 4.15, 0.0), (85.0, 18.0, 4.15, 0.0), (85.0, 18.0, 1.45, 180.0),
                         (68.0, 13.0, 1.45, -165.0), (55.0, 7.0, 1.45, 180.0), (37.0, 12.0, 1.45, 180.0), (19.0, 7.0, 1.45, 180.0), (8.0, 14.0, 1.45, 90.0)]}
KEYS = {"id", "at", "yaw_deg", "after", "dist", "turn_deg", "dz"}
FKEYS = {"speed_mps", "fps", "tilt_deg", "offset_m", "focus_m", "hfov_deg", "roll_deg"}

SCHEMA_TEXT = f"""Reply with ONLY a JSON object. Keys: setting, light, gates, flight, seed. No code, no sizes, no text outside the JSON.
template: optional, one of {sorted(TEMPLATES)}; the system then uses the real track and ignores `gates`.
setting: one of {sorted(SETTINGS)} (field = open grass, hall = indoor-like floor). light: one of {sorted(LIGHTS)}.
gates: list ({LIMITS['gates'][0]}..{LIMITS['gates'][1]}) in flight order of {{id: unique, at: [x, y, z] and yaw_deg (direction you fly THROUGH the gate), OR after: previous gate id with dist: {LIMITS['gap'][0]}..{LIMITS['gap'][1]} m, turn_deg: -{LIMITS['turn']:g}..{LIMITS['turn']:g} (left positive), dz: -{LIMITS['dz']:g}..{LIMITS['dz']:g} m}}.
  The first gate needs at+yaw_deg. Gate size is fixed by the system (2.7 m outer, 1.5 m opening). Gate centre height {LIMITS['z'][0]}..{LIMITS['z'][1]} m. Gates stay inside +-{LIMITS['area']:g} m.
flight: {{speed_mps: {LIMITS['speed']}, fps: {LIMITS['fps']}, tilt_deg: camera up-tilt {LIMITS['tilt']}, offset_m: lateral offset of the flight line from the gate centre, |x| <= {LIMITS['offset']} (the opening half-width is 0.75 m and the drone body about 0.2 m),
  focus_m: only frames within this distance of a gate are kept {LIMITS['focus']}, hfov_deg: {LIMITS['hfov']} (130 = DroneRacing), roll_deg: bank limit {LIMITS['roll']:g}}}.
Rules: consecutive gates {LIMITS['gap'][0]:g}..{LIMITS['gap'][1]:g} m apart and the turn to the next gate at most {LIMITS['turn']:g} degrees against the passage direction, no two gates closer than 2.5 m, between {LIMITS['frames'][0]} and {LIMITS['frames'][1]} frames in the end.
seed: int."""


def _num(v, name, lo, hi, errs):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        errs.append(f"{name}: must be a number, got {v!r}"); return None
    if v < lo or v > hi:
        errs.append(f"{name}: {v} outside {lo:g}..{hi:g}"); return None
    return float(v)


def _sub(a, b): return [x - y for x, y in zip(a, b)]
def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def _unit(a):
    n = math.sqrt(_dot(a, a)); return [x / n for x in a]


def gate_frame(g):
    y = math.radians(g["yaw_deg"]); return [math.cos(y), math.sin(y), 0.0], [math.sin(y), -math.cos(y), 0.0]


def inner_corners(g):
    """World coordinates of the 4 inner corners TL, TR, BR, BL (seen by the approaching drone)."""
    n, r = gate_frame(g); h = GATE_INNER / 2; c = g["at"]; u = [0, 0, 1.0]
    return [[c[i] + s * h * r[i] + t * h * u[i] for i in range(3)] for s, t in ((-1, 1), (1, 1), (1, -1), (-1, -1))]


def _place(gates, errs):
    out = []; byid = {}
    for k, g in enumerate(gates):
        if not isinstance(g, dict):
            errs.append(f"gates[{k}]: must be an object"); continue
        gid = g.get("id", f"g{k}"); bad = set(g) - KEYS
        if bad:
            errs.append(f"gate {gid}: unknown keys {sorted(bad)}"); continue
        if gid in byid:
            errs.append(f"gate {gid}: id repeated"); continue
        if "at" in g:
            at = g["at"]
            if not (isinstance(at, list) and len(at) == 3):
                errs.append(f"gate {gid}: at must be [x,y,z]"); continue
            x = _num(at[0], f"gate {gid} x", -LIMITS["area"], LIMITS["area"], errs); y = _num(at[1], f"gate {gid} y", -LIMITS["area"], LIMITS["area"], errs); z = _num(at[2], f"gate {gid} z", *LIMITS["z"], errs)
            yaw = _num(g.get("yaw_deg", 0.0), f"gate {gid} yaw_deg", -720, 720, errs)
            if None in (x, y, z, yaw):
                continue
        elif "after" in g:
            p = byid.get(g["after"])
            if p is None:
                errs.append(f"gate {gid}: after '{g['after']}' is not an earlier gate"); continue
            d = _num(g.get("dist"), f"gate {gid} dist", *LIMITS["gap"], errs); t = _num(g.get("turn_deg", 0.0), f"gate {gid} turn_deg", -LIMITS["turn"], LIMITS["turn"], errs)
            dz = _num(g.get("dz", 0.0), f"gate {gid} dz", -LIMITS["dz"], LIMITS["dz"], errs)
            if None in (d, t, dz):
                continue
            yaw = p["yaw_deg"] + t; a = math.radians(yaw); x = p["at"][0] + d * math.cos(a); y = p["at"][1] + d * math.sin(a); z = p["at"][2] + dz
            if not LIMITS["z"][0] <= z <= LIMITS["z"][1]:
                errs.append(f"gate {gid}: height {z:.2f} m outside {LIMITS['z'][0]}..{LIMITS['z'][1]} m (previous gate z {p['at'][2]:.2f} + dz {dz:g})"); continue
            if abs(x) > LIMITS["area"] or abs(y) > LIMITS["area"]:
                errs.append(f"gate {gid}: lands at ({x:.0f},{y:.0f}), outside +-{LIMITS['area']:g} m; turn back toward the centre"); continue
        else:
            errs.append(f"gate {gid}: needs 'at' (+yaw_deg) or 'after' (+dist, turn_deg, dz)"); continue
        r = {"id": gid, "at": [x, y, z], "yaw_deg": yaw}; byid[gid] = r; out.append(r)
    return out


def _check_gates(G, errs):
    for i, g in enumerate(G):
        for j in range(i):
            d = math.dist(g["at"], G[j]["at"])
            if d < 2.5:
                errs.append(f"gate {g['id']}: only {d:.1f} m from gate {G[j]['id']} (min 2.5 m)")
        if i:
            p = G[i - 1]; v = _sub(g["at"], p["at"]); d = math.hypot(v[0], v[1])
            if not LIMITS["gap"][0] <= d <= LIMITS["gap"][1]:
                errs.append(f"gate {g['id']}: {d:.1f} m from gate {p['id']}, allowed {LIMITS['gap'][0]:g}..{LIMITS['gap'][1]:g} m")
            n, _ = gate_frame(p); ang = math.degrees(math.acos(max(-1, min(1, (v[0] * n[0] + v[1] * n[1]) / max(d, 1e-9)))))
            if ang > LIMITS["turn"]:
                errs.append(f"gate {g['id']}: lies {ang:.0f} deg off the passage direction of gate {p['id']} (max {LIMITS['turn']:g}); the drone cannot fly it")
            ng, _ = gate_frame(g)
            if v[0] * ng[0] + v[1] * ng[1] <= 0:
                errs.append(f"gate {g['id']}: its passage direction points back toward gate {p['id']}; the drone would have to enter from behind")


def _flight_line(G, offset):
    """Waypoints: 8 m before the first gate, every gate centre shifted sideways by `offset` (alternating sign = slalom through the opening), 8 m after the last gate. Catmull-Rom through them."""
    pts = []; n0, _ = gate_frame(G[0]); pts.append([G[0]["at"][i] - 8 * n0[i] for i in range(3)])
    for k, g in enumerate(G):
        n, r = gate_frame(g); s = offset * (1 if k % 2 == 0 else -1); pts.append([g["at"][i] + s * r[i] for i in range(3)])
    n1, _ = gate_frame(G[-1]); pts.append([G[-1]["at"][i] + 8 * n1[i] for i in range(3)])
    return pts


def _catmull(p, n=24):
    P = [p[0]] + p + [p[-1]]; out = []
    for i in range(1, len(P) - 2):
        a, b, c, d = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t * t * t
            out.append([0.5 * ((2 * b[j]) + (-a[j] + c[j]) * t + (2 * a[j] - 5 * b[j] + 4 * c[j] - d[j]) * t2 + (-a[j] + 3 * b[j] - 3 * c[j] + d[j]) * t3) for j in range(3)])
    out.append(p[-1]); return out


def _resample(line, step):
    """Equal arc-length samples every `step` metres along the polyline."""
    out = [line[0]]; acc = 0.0
    for a, b in zip(line, line[1:]):
        seg = math.dist(a, b)
        while acc + seg >= step:
            t = (step - acc) / seg; a = [a[i] + t * (b[i] - a[i]) for i in range(3)]; out.append(a); seg = math.dist(a, b); acc = 0.0
        acc += seg
    return out


def project(pose, hfov, pts):
    """Pinhole at IMG_W x IMG_H. pose has pos, fwd, up. Returns [(u, v, depth)] (depth<=0: behind the camera)."""
    f = pose["fwd"]; u = pose["up"]; r = _unit(_cross(f, u)); u = _cross(r, f); fx = IMG_W / 2 / math.tan(math.radians(hfov) / 2); res = []
    for p in pts:
        q = _sub(p, pose["pos"]); z = _dot(q, f)
        res.append((IMG_W / 2 + fx * _dot(q, r) / z, IMG_H / 2 - fx * _dot(q, u) / z, z) if z > 1e-6 else (0.0, 0.0, z))
    return res


def resolve(layout: dict) -> dict:
    errs: list[str] = []
    if not isinstance(layout, dict):
        return {"ok": False, "errors": ["layout must be an object"], "scene": None}
    bad = set(layout) - TOP
    if bad:
        errs.append(f"unknown top-level keys {sorted(bad)}; allowed {sorted(TOP)}")
    setting = layout.get("setting", "field"); light = layout.get("light", "noon")
    if setting not in SETTINGS: errs.append(f"setting '{setting}' not in {sorted(SETTINGS)}")
    if light not in LIGHTS: errs.append(f"light '{light}' not in {sorted(LIGHTS)}")
    gates = layout.get("gates"); tpl = layout.get("template")
    if tpl is not None:
        if tpl not in TEMPLATES:
            errs.append(f"template '{tpl}' not in {sorted(TEMPLATES)}"); return {"ok": False, "errors": errs, "scene": None}
        gates = [{"id": f"g{i}", "at": [x - 50.0, y - 15.0, z], "yaw_deg": w} for i, (x, y, z, w) in enumerate(TEMPLATES[tpl])]
    if not isinstance(gates, list) or not LIMITS["gates"][0] <= len(gates) <= LIMITS["gates"][1]:
        errs.append(f"gates: list of {LIMITS['gates'][0]}..{LIMITS['gates'][1]} required"); return {"ok": False, "errors": errs, "scene": None}
    fl = layout.get("flight", {}) or {}
    badf = set(fl) - FKEYS
    if badf: errs.append(f"flight: unknown keys {sorted(badf)}")
    speed = _num(fl.get("speed_mps", 15.0), "flight speed_mps", *LIMITS["speed"], errs); fps = _num(fl.get("fps", 30.0), "flight fps", *LIMITS["fps"], errs)
    tilt = _num(fl.get("tilt_deg", 25.0), "flight tilt_deg", *LIMITS["tilt"], errs); off = _num(fl.get("offset_m", 0.0), "flight offset_m", -LIMITS["offset"], LIMITS["offset"], errs)
    focus = _num(fl.get("focus_m", 12.0), "flight focus_m", *LIMITS["focus"], errs); hfov = _num(fl.get("hfov_deg", 130.0), "flight hfov_deg", *LIMITS["hfov"], errs)
    G = _place(gates, errs)
    if errs or len(G) < 2:
        return {"ok": False, "errors": errs or ["need at least 2 valid gates"], "scene": None}
    if tpl is None:
        _check_gates(G, errs)
    if errs:
        return {"ok": False, "errors": errs, "scene": None}
    line = _resample(_catmull(_flight_line(G, off)), speed / fps); n_all = len(line)
    poses = []; t = math.radians(tilt)
    for k, p in enumerate(line):
        a, b = line[max(k - 1, 0)], line[min(k + 1, n_all - 1)]; v = _unit(_sub(b, a)); hz = math.hypot(v[0], v[1]) or 1e-9; h = [v[0] / hz, v[1] / hz, 0.0]
        pitch = math.atan2(v[2], hz) + t                           # follow the climb, then tilt the camera up (racing cameras are mounted up-tilted)
        f = [h[0] * math.cos(pitch), h[1] * math.cos(pitch), math.sin(pitch)]; up = _unit(_cross(_cross(f, [0, 0, 1.0]), f)) if abs(f[2]) < 0.999 else [0, 1.0, 0]
        poses.append({"pos": p, "fwd": f, "up": up, "k": k})
    keep = [q for q in poses if min(math.dist(q["pos"], g["at"]) for g in G) <= focus]
    if not LIMITS["frames"][0] <= len(keep) <= LIMITS["frames"][1]:
        errs.append(f"{len(keep)} frames kept (focus_m {focus:g}, speed {speed:g}, fps {fps:g}); need {LIMITS['frames'][0]}..{LIMITS['frames'][1]}: lower focus_m or fps, or use fewer gates")
        return {"ok": False, "errors": errs, "scene": None}
    labels = []; seen_any = 0
    for q in keep:
        rows = []
        for gi, g in enumerate(G):
            n, _ = gate_frame(g); rel = _sub(q["pos"], g["at"])
            if _dot(rel, n) > 0.2:                              # drone is already past this gate (it sees the back)
                continue
            if math.dist(q["pos"], g["at"]) > 60:
                continue
            px = project(q, hfov, inner_corners(g)); front = all(c[2] > 0.05 for c in px)
            if not front:                                       # a corner behind the image plane: the gate straddles the camera plane, no valid 2D target
                continue
            vis = [0 <= c[0] < IMG_W and 0 <= c[1] < IMG_H for c in px]
            rows.append({"gate": gi, "corners_px": [[round(c[0], 2), round(c[1], 2)] for c in px], "visible": vis, "n_visible": sum(vis), "dist_m": round(math.dist(q["pos"], g["at"]), 2),
                         "depths": [round(c[2], 2) for c in px]})
        rows.sort(key=lambda r: r["dist_m"]); labels.append(rows); seen_any += any(r["n_visible"] > 0 for r in rows)
    if seen_any < 0.5 * len(keep):
        errs.append(f"only {seen_any}/{len(keep)} frames show a gate corner; raise tilt_deg or reduce offset_m")
        return {"ok": False, "errors": errs, "scene": None}
    two = sum(any(0 < r["n_visible"] < 4 for r in rows) for rows in labels)
    scene = {"setting": setting, "light": light, "gates": G, "outer": GATE_OUTER, "inner": GATE_INNER, "bar": GATE_BAR, "depth": GATE_DEPTH, "img": [IMG_W, IMG_H], "hfov_deg": hfov,
             "camera": {"poses": [{"pos": [round(c, 4) for c in q["pos"]], "forward": [round(c, 5) for c in q["fwd"]], "up": [round(c, 5) for c in q["up"]]} for q in keep], "speed_mps": speed, "fps": fps, "tilt_deg": tilt},
             "labels": labels, "stats": {"frames": len(keep), "frames_partial_gate": two, "frames_gate_seen": seen_any}}
    scene["hash"] = hashlib.sha1(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:10]
    return {"ok": True, "errors": [], "scene": scene}


def generate(seed: int = 0, n_gates: int = 8, speed: float = 20.0, setting: str = "field", light: str = "noon", offset: float = 0.3, fps: float | None = None, focus: float = 12.0, tilt: float = 25.0) -> dict:
    """Rule-based baseline: gates alternate gentle turns so the track stays inside the area; returns a valid layout (retried until the resolver accepts)."""
    import random
    for attempt in range(60):
        rng = random.Random(seed * 1000 + attempt)
        gs = [{"id": "g0", "at": [-60.0, rng.uniform(-30, 30), rng.uniform(0.9, 4.2)], "yaw_deg": rng.uniform(-30, 30)}]
        zs = [gs[0]["at"][2]]
        for i in range(1, n_gates):
            p = gs[-1]
            gs.append({"id": f"g{i}", "after": p["id"], "dist": rng.uniform(10, 24), "turn_deg": rng.uniform(-70, 70), "dz": (lambda t: (zs.append(t), t - zs[-2])[1])(rng.uniform(0.9, 4.2))})
        lay = {"setting": setting, "light": light, "gates": gs, "flight": {"speed_mps": speed, "fps": fps or max(10.0, min(120.0, speed * 2)), "tilt_deg": tilt, "offset_m": offset, "focus_m": focus}, "seed": seed}
        if resolve(lay)["ok"]:
            return lay
    raise RuntimeError("no valid track found")

