"""Rule-based indoor layout generator (no LLM): the deterministic baseline, the few-shot examples for a small LLM, and the test bed for the resolver. Every layout it returns has passed pfm.layout.resolve.
Environment rules (each one is a line in `generate`):
  R1  big things (bed, sofa, wardrobe, bookcase, fridge, desk) stand against a wall; small things (lamp, plant, stool) go to corners (along 0 or 1) or on a host.
  R2  a coffee table sits north of the sofa with a 0.4-0.7 m gap; a TV/lamp/plant may sit on a table, desk or dresser.
  R3  a door and a window never share a wall; both keep clear of furniture (the resolver enforces it).
  R4  the camera path is a short walk between 2-3 free points at 0.9-1.8 m height; the resolver rejects any point near walls/items, so points are sampled until accepted.
  R5  light, seed, camera speed and roll vary per seed (roll 180 = inverted mount for a share of the layouts).
Kinds: living_room, bedroom, office, kitchen. A kind only fixes the item pool and the room size range."""
from __future__ import annotations

import random

from .layout import catalogue, resolve

KINDS = {
    "living_room": {"size": ((4.0, 7.0), (3.5, 6.0)), "must": ["sofa"], "maybe": ["table", "television_set", "armchair", "bookcase", "lamp", "flowerpot", "stool", "cabinet", "dresser"]},
    "bedroom": {"size": ((3.0, 5.0), (3.0, 5.0)), "must": ["bed"], "maybe": ["wardrobe", "dresser", "lamp", "stool", "flowerpot", "chair", "desk", "bookcase"]},
    "office": {"size": ((3.0, 6.0), (3.0, 5.0)), "must": ["desk", "chair"], "maybe": ["bookcase", "cabinet", "lamp", "flowerpot", "sofa", "television_set", "stool"]},
    "kitchen": {"size": ((3.0, 5.0), (2.8, 4.5)), "must": ["refrigerator", "stove"], "maybe": ["table", "chair", "sink", "microwave_oven", "cabinet", "stool", "flowerpot"]},
}
CORNER = {"lamp", "flowerpot", "stool"}
HOSTABLE = {"television_set", "lamp", "flowerpot", "microwave_oven"}
HOSTS = {"table", "desk", "dresser", "stove", "cabinet", "bookcase"}


def _by_tag():
    d: dict[str, list[str]] = {}
    for c in catalogue():
        d.setdefault(c["tag"], []).append(c["id"])
    return d


def _try(kind: str, rng: random.Random, seed: int, light: str | None) -> dict:
    spec, pool = KINDS[kind], _by_tag()
    w = round(rng.uniform(*spec["size"][0]), 1); d = round(rng.uniform(*spec["size"][1]), 1); h = round(rng.choice([2.4, 2.6, 2.8, 3.0]), 1)
    tags = list(spec["must"]) + rng.sample(spec["maybe"], k=min(len(spec["maybe"]), rng.randint(2, 5)))
    items, used_walls, hosts = [], {}, []
    for n, tag in enumerate(tags):
        if tag not in pool:
            continue
        iid = f"{tag}{n}"; asset = rng.choice(pool[tag]); it = {"id": iid, "asset": asset, "yaw_deg": float(rng.choice([0, 90, 180, 270]))}
        if tag == "table" and any(i["id"].startswith("sofa") for i in items):
            sofa = next(i for i in items if i["id"].startswith("sofa")); it["at"] = {"near": sofa["id"], "side": "north", "gap": round(rng.uniform(0.4, 0.7), 2)}      # R2
        elif tag in HOSTABLE and hosts and rng.random() < 0.7:
            it["at"] = {"on": rng.choice(hosts), "dx": 0.0, "dy": 0.0}                                                                                                   # R2
        elif tag in CORNER:
            it["at"] = {"wall": rng.choice(["north", "south", "east", "west"]), "along": float(rng.choice([0.0, 1.0]))}                                                  # R1
        elif tag in HOSTS or tag in {"bed", "sofa", "wardrobe", "bookcase", "refrigerator", "desk", "armchair", "chair"}:
            wall = rng.choice(["north", "south", "east", "west"]); it["at"] = {"wall": wall, "along": round(rng.uniform(0.1, 0.9), 2)}                                    # R1
        else:
            it["at"] = [round(rng.uniform(0.8, w - 0.8), 2), round(rng.uniform(0.8, d - 0.8), 2)]
        if tag in HOSTS:
            hosts.append(iid)
        items.append(it)
    walls = ["north", "south", "east", "west"]; rng.shuffle(walls)
    openings = [{"id": "door", "kind": "door", "wall": walls[0], "along": round(rng.uniform(0.1, 0.9), 2)}, {"id": "win", "kind": "window", "wall": walls[1], "along": round(rng.uniform(0.1, 0.9), 2)}]   # R3
    wp = [[round(rng.uniform(0.6, w - 0.6), 2), round(rng.uniform(0.6, d - 0.6), 2), round(rng.uniform(0.9, min(1.8, h - 0.3)), 2)] for _ in range(rng.randint(2, 3))]                        # R4
    cam = {"waypoints": wp, "look_at": None, "roll_deg": float(rng.choice([0, 0, 0, 180])), "speed_mps": round(rng.uniform(0.5, 2.5), 1), "fps": 6.0, "hfov_deg": float(rng.choice([80, 90, 100]))}   # R5
    return {"room": {"w": w, "d": d, "h": h}, "light": light or rng.choice(["day", "day", "lamp", "dark"]), "items": items, "openings": openings, "camera": cam, "seed": seed}


def generate(kind: str = "living_room", seed: int = 0, light: str | None = None, max_tries: int = 400) -> dict:
    """First layout from seeded attempts that passes every resolver rule; raises if none does in max_tries (a rule set that is too tight shows up here)."""
    rng = random.Random(f"{kind}:{seed}"); last = None
    for _ in range(max_tries):
        lay = _try(kind, rng, seed, light); r = resolve(lay)
        if r["ok"]:
            return lay
        last = r["errors"]
    raise RuntimeError(f"no valid {kind} layout in {max_tries} tries; last errors: {last}")
