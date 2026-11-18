"""Racing layouts aimed at PARTLY VISIBLE gates. Lesson of 2026-11-14: with the 130 deg camera the inner corners leave the image only within ~0.6 m of the gate plane unless the camera is tilted UP:
at tilt t the lower image edge is (58 - t) deg below the horizon, so a corner 0.75 m below the camera is out of view at distances up to 0.75 / tan(58 - t) (1.8 m at 35 deg, 5.3 m at 50 deg).
This generator uses tilt 40-50 (the resolver limit is 50), low speed 4-12 m/s with 120 fps (dense frames near the gate), frames within 3 m of a gate, offset up to 0.6 m.
CAUTION: the real camera tilt of the target drone is not known here; high tilt shifts the data distribution. Usage: python scripts/gen_race_partial.py <n> [prefix=p_race]"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, "src")
from pfm.layout_race import generate, resolve
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30; pre = sys.argv[2] if len(sys.argv) > 2 else "p_race"; out = Path("runs/race_job/layouts"); out.mkdir(parents=True, exist_ok=True); frames = []
for i in range(n):
    rng = random.Random(f"partial:{i}"); speed = rng.choice([4.0, 8.0, 12.0]); max_g = max(3, int(560 / (720.0 / speed)))
    for attempt in range(30):
        try:
            lay = generate(3000 + i + 1000 * attempt, n_gates=rng.choice(range(3, min(9, max_g) + 1)), speed=speed, setting=["field", "hall"][i % 2], light=["noon", "overcast", "dusk"][i % 3], offset=round(rng.choice([-0.6, -0.3, 0.3, 0.6]), 1), fps=120.0, focus=3.0, tilt=rng.choice([40.0, 45.0, 50.0]))
            r = resolve(lay)
            if r["ok"]: break
        except RuntimeError:
            continue
    assert r["ok"], (i, r["errors"][:2]); json.dump(lay, open(out / f"{pre}_{i}.json", "w")); frames.append(len(r["scene"]["camera"]["poses"]))
print(n, "layouts, frames per layout: min", min(frames), "median", sorted(frames)[len(frames) // 2], "max", max(frames), "total", sum(frames))
