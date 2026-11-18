"""Racing layouts aimed at STEEP SIDE VIEWS of gates (Update 83). Red (never seen) shows 7% of real gates at width/height ratio < 0.5, the normal sim set 0.1%.
Gates are placed close together (3.5-7 m) with sharp alternating turns (65-100 deg, the resolver limit is 100), so the next gate is seen from the side while the drone flies the current one. Frames within 10 m of a gate, 60 fps.
Usage: python scripts/gen_race_steep.py <n> [prefix=s_race]  -> runs/race_job/layouts/<prefix>_<i>.json (each resolved first)."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, "src")
from pfm.layout_race import resolve
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30; pre = sys.argv[2] if len(sys.argv) > 2 else "s_race"; out = Path("runs/race_job/layouts"); out.mkdir(parents=True, exist_ok=True); frames = []
for i in range(n):
    rng = random.Random(f"steep:{i}"); speed = rng.choice([6.0, 10.0, 14.0]); setting = ["field", "hall"][i % 2]
    for attempt in range(200):
        r2 = random.Random(f"steep:{i}:{attempt}"); ng = r2.choice(range(5, 10)); sgn = r2.choice([-1, 1]); zs = [r2.uniform(1.2, 3.5)]
        gs = [{"id": "g0", "at": [-20.0, r2.uniform(-10, 10), zs[0]], "yaw_deg": r2.uniform(-20, 20)}]
        for k in range(1, ng):
            z = r2.uniform(1.2, 3.5); gs.append({"id": f"g{k}", "after": gs[-1]["id"], "dist": r2.uniform(3.5, 7.0), "turn_deg": sgn * r2.uniform(65, 100) * (1 if r2.random() < 0.7 else -1), "dz": round(z - zs[-1], 3)}); zs.append(z)
        lay = {"setting": setting, "light": ["noon", "overcast", "dusk"][i % 3], "gates": gs, "flight": {"speed_mps": speed, "fps": 60.0, "tilt_deg": r2.choice([15.0, 25.0, 35.0]), "offset_m": round(r2.choice([-0.6, -0.3, 0.0, 0.3, 0.6]), 1), "focus_m": 10.0}, "seed": 4000 + i}
        r = resolve(lay)
        if r["ok"]: break
    assert r["ok"], (i, r["errors"][:2]); json.dump(lay, open(out / f"{pre}_{i}.json", "w")); frames.append(len(r["scene"]["camera"]["poses"]))
print(n, "layouts, frames per layout: min", min(frames), "median", sorted(frames)[len(frames) // 2], "max", max(frames), "total", sum(frames))
