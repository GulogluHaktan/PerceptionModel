"""Rule-based OUTDOOR layout generator (no LLM): deterministic baseline, few-shot examples and test bed for pfm.layout_out.resolve. Every returned layout has passed the resolver.
Rules (each one a block in `generate`):
  O1  1-3 straight roads of width 8-14 m that cross inside the area; sidewalks and lane markings come from the renderer.
  O2  buildings are placed by rejection sampling away from the roads (the resolver enforces setback and overlap); trees, lamps and benches stand on kerbs at 12-16 m spacing.
  O3  cars stand on lanes (lane_left / lane_right) and are aligned with the road by the resolver; parked cars are never on a kerb.
  O4  camera: drone (circle around the centre, 15-30 m high, up to 30 m/s), rover (along a road, 1.3 m high, 8 m/s) or inverted drone (roll 180); light noon / overcast / dusk.
Items are added one by one and kept only if the resolver still accepts the layout, so crowding never makes the whole layout fail."""
from __future__ import annotations

import math
import random
import re

from .layout_out import catalogue, resolve

MODES = ("drone", "rover", "inverted", "fast_drone")


def _pools():
    d: dict[str, list[str]] = {}
    for c in catalogue():
        d.setdefault(c["group"], []).append(c["id"])
    return d


def generate(mode: str = "drone", seed: int = 0, light: str | None = None, buildings: int = 8, trees: int = 30, cars: int = 8, max_tries: int = 40) -> dict:
    rng = random.Random(f"{mode}:{seed}"); pool = _pools()
    for _ in range(max_tries):
        S = float(rng.choice([100, 120, 150]))
        roads = [{"id": "A", "from": [-S * 0.9, round(rng.uniform(-10, 10), 1)], "to": [S * 0.9, round(rng.uniform(-10, 10), 1)], "width": float(rng.choice([8, 10, 12, 14]))}]        # O1
        if rng.random() < 0.8:
            roads.append({"id": "B", "from": [round(rng.uniform(-10, 10), 1), -S * 0.9], "to": [round(rng.uniform(-10, 10), 1), S * 0.9], "width": float(rng.choice([8, 10, 12]))})
        lay = {"area": {"size": S}, "light": light or rng.choice(["noon", "noon", "overcast", "dusk"]), "roads": roads, "items": [], "seed": seed}

        def add(item):
            lay["items"].append(item)
            if not resolve(lay)["ok"]:
                lay["items"].pop(); return False
            return True
        for i in range(buildings):                                                                                                                                          # O2
            for _t in range(12):
                if add({"id": f"b{i}", "asset": rng.choice(pool["building"]), "at": [round(rng.uniform(-S * 0.85, S * 0.85), 1), round(rng.uniform(-S * 0.85, S * 0.85), 1)], "yaw_deg": float(rng.choice([0, 90, 180, 270]))}):
                    break
        for i in range(trees):
            rd = rng.choice(roads); add({"id": f"t{i}", "asset": rng.choice(pool["tree"]), "at": {"road": rd["id"], "s": round(rng.uniform(0.05, 0.95), 3), "lane": rng.choice(["kerb_left", "kerb_right"])}, "yaw_deg": round(rng.uniform(0, 360), 0)})
        for i in range(6):
            rd = rng.choice(roads); kind = "pole" if i % 2 == 0 else "prop"; add({"id": f"p{i}", "asset": rng.choice(pool[kind]), "at": {"road": rd["id"], "s": round(rng.uniform(0.05, 0.95), 3), "lane": rng.choice(["kerb_left", "kerb_right"])}})
        for i in range(cars):                                                                                                                                                 # O3
            rd = rng.choice(roads); add({"id": f"c{i}", "asset": rng.choice(pool["car"]), "at": {"road": rd["id"], "s": round(rng.uniform(0.05, 0.95), 3), "lane": rng.choice(["lane_left", "lane_right"])}})
        for _c in range(30):                                                                                                                                                  # O4
            if mode in ("drone", "inverted", "fast_drone"):
                R, alt = rng.uniform(40, 70), rng.uniform(15, 30); sp = rng.choice([30.0, 22.0]) if mode == "fast_drone" else rng.choice([6.0, 10.0, 15.0]); a0 = rng.uniform(0, 6.28)
                wp = [[round(R * math.cos(a0 + k * 1.05), 1), round(R * math.sin(a0 + k * 1.05), 1), round(alt, 1)] for k in range(6)]
                cam = {"waypoints": wp, "look_at": [0.0, 0.0, 2.0], "roll_deg": 180.0 if mode == "inverted" else 0.0, "speed_mps": sp if mode != "inverted" else 12.0, "fps": 10.0, "hfov_deg": 90.0}
            else:
                rd = roads[0]; y0 = rd["from"][1] - rd["width"] / 4; cam = {"waypoints": [[-S * 0.7, round(y0, 1), 1.3], [S * 0.7, round(y0, 1), 1.3]], "look_at": None, "roll_deg": 0.0, "speed_mps": 8.0, "fps": 10.0, "hfov_deg": 90.0}
            lay["camera"] = cam
            r = resolve(lay)
            if r["ok"]:
                return lay
            if mode == "rover":                                                    # the rover path is fixed (a lane along road A): drop the items it runs into (parked car, kerb tree) and try again instead of re-sampling the same path 30 times
                bad = {m.group(1) for e in r["errors"] for m in [re.search(r"within 2 m of item '([^']+)'", e)] if m}
                if bad:
                    lay["items"] = [it for it in lay["items"] if it["id"] not in bad]
        lay.pop("camera", None)
    raise RuntimeError(f"no valid outdoor layout for mode {mode} seed {seed} in {max_tries} tries")
