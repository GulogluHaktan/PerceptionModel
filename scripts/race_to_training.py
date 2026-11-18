"""Racing sets -> what the trainer reads (same layout as factory_to_training.py). Input: the multi-condition set from scripts/city_effects.py (images/, labels/, index.json with the gate-corner rows) plus the raw render dir for the metric depth.
Usage: python scripts/race_to_training.py <out_root> <effects_set_dir> <raw_out_root> [name]
Corner rule (as factory_to_training): gate counts only if all 4 inner corners are visible (depth-checked) and the gate is the nearest 4 at most; a frame with a partly visible gate keeps seg + depth but gets NO corner file (trainer skips the corner loss);
a frame with no visible corner gets an empty file (true negative). The partial-gate frames are exactly the 'blind spells' of DroneRacing: see the full-corner file written next to it (<id>.full.json: pixel corners incl. outside the image, 256x192 scale) for an offset/extrapolation head later."""
import json, shutil, sys
from pathlib import Path
import numpy as np
out, eff, raw = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); name = sys.argv[4] if len(sys.argv) > 4 else eff.name
for sub in ("images/" + name, "labels/seg", "labels/corners", "labels/depth", "labels/corners_full"):
    (out / sub).mkdir(parents=True, exist_ok=True)
idx = json.load(open(eff / "index.json")); st = {"frames": 0, "corners_known": 0, "partial_gate_frames": 0, "with_full_gate": 0}
for e in idx:
    sid = e["id"]; shutil.copy(eff / "images" / f"{sid}.png", out / "images" / name / f"{sid}.png"); shutil.copy(eff / "labels" / f"{sid}.png", out / "labels/seg" / f"{name}__{sid}.png")
    np.save(out / "labels/depth" / f"{name}__{sid}.npy", np.load(raw / e["run"] / "depth" / f"{e['frame']:04d}.npy").astype(np.float16)); st["frames"] += 1
    rows = e.get("gates", [])[:4]; json.dump(rows, open(out / "labels/corners_full" / f"{name}__{sid}.json", "w"))          # incl. out-of-image corners
    if any(0 < r["n_visible_depth"] < 4 for r in rows):
        st["partial_gate_frames"] += 1; continue                                 # corners unknown for the trainer's heatmap loss
    full = [np.array([[u / 256, v / 192] for u, v in r["corners_px"]], np.float32) for r in rows if r["n_visible_depth"] == 4]
    np.save(out / "labels/corners" / f"{name}__{sid}.npy", np.stack(full) if full else np.zeros((0, 4, 2), np.float32)); st["corners_known"] += 1; st["with_full_gate"] += int(bool(full))
print(json.dumps(st))
