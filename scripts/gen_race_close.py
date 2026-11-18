"""'Close pass' racing layouts: only frames within 3 m of a gate (focus_m minimum), sampled at 120 fps, lateral offset up to +-0.6 m (the resolver's limit), so gates are often cut by the image border: the case the slot head exists for.
The normal racing set keeps frames within 8 m at 20-60 fps, where only about 3% of the gates have a corner outside the image (measured 2026-11-14 on held-out frames).
Writes runs/race_job/layouts/c_race_<i>.json (every layout is resolved first). Render with sim/run_race_set.sh c_race_0 ... ; see docs/MONDAY.md for the conversion."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, "src")
from pfm.layout_race import generate, resolve
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30; out = Path("runs/race_job/layouts"); out.mkdir(parents=True, exist_ok=True); frames = []
for i in range(n):
    rng = random.Random(f"close:{i}"); speed = rng.choice([10.0, 20.0, 30.0]); per_gate = 720.0 / speed; max_g = int(560 / per_gate)       # frames per gate = 2 x 3 m x 120 fps / speed; the resolver allows at most 600 frames
    for attempt in range(20):
        try:
            lay = generate(2000 + i + 1000 * attempt, n_gates=rng.choice(range(5, min(10, max_g) + 1)), speed=speed, setting=["field", "hall"][i % 2], light=["noon", "overcast", "dusk"][i % 3], offset=round(rng.choice([-0.6, -0.4, 0.4, 0.6]), 1), fps=120.0, focus=3.0, tilt=rng.choice([15.0, 25.0, 35.0]))
            r = resolve(lay)
            if r["ok"]: break
        except RuntimeError:
            continue
    assert r["ok"], (i, r["errors"][:2]); json.dump(lay, open(out / f"c_race_{i}.json", "w")); frames.append(len(r["scene"]["camera"]["poses"]))
print(n, "layouts, frames per layout: min", min(frames), "median", sorted(frames)[len(frames) // 2], "max", max(frames), "total", sum(frames))
