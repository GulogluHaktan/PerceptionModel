"""Convert data-factory datasets (rgb/, depth/, label/, manifest.json) into what the trainer reads (decision 9 / plan step 8 plumbing).
Usage: python scripts/factory_to_training.py <out_root> <name=dataset_dir> [name=dataset_dir ...] [--golden name=dataset_dir]
Writes <out_root>/images/<name>/NNNN.png, <out_root>/labels/{seg/<name>__NNNN.png, corners/<name>__NNNN.npy} for the trainer (ids = <name>__NNNN), and for --golden
<out_root>/golden/{images, seg, depth(.npy metres)}. Corner rule: a gate counts only if all 4 corners are visible; a frame with a partly visible gate keeps its
segmentation but gets NO corner file (corners unknown: the trainer skips the corner loss); frames without any visible gate corner keep an empty corner file (a true negative)."""
import json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

args = sys.argv[1:]; goldens = []
while "--golden" in args:
    i = args.index("--golden"); goldens.append(args[i + 1]); args = args[:i] + args[i + 2:]       # --golden group=dir (repeat for several datasets in one group)
out = Path(args[0]); sets = dict(a.split("=", 1) for a in args[1:])
for sub in ("images", "labels/seg", "labels/corners", "labels/depth"):
    (out / sub).mkdir(parents=True, exist_ok=True)


def frames(d):
    man = json.load(open(Path(d) / "manifest.json")); W, H = Image.open(Path(d) / "rgb" / "0000.png").size
    for k, fr in enumerate(man["frames"]):
        gates = fr["extra"]["gate_corners"] if fr.get("extra") else []
        vis = [[bool(c[2]) for c in g] for g in gates]
        if any(any(v) and not all(v) for v in vis):
            yield k, None, (W, H), "partial"; continue                            # partly visible gate: corners unknown for this frame
        full = [np.array([[c[0] / W, c[1] / H] for c in g], np.float32) for g, v in zip(gates, vis) if all(v)]
        yield k, (np.stack(full) if full else np.zeros((0, 4, 2), np.float32)), (W, H), "ok"


stats = {}
for name, d in sets.items():
    kept = dropped = with_gate = 0; (out / "images" / name).mkdir(parents=True, exist_ok=True)
    for k, corners, wh, status in frames(d):
        shutil.copy(Path(d) / "rgb" / f"{k:04d}.png", out / "images" / name / f"{k:04d}.png")
        shutil.copy(Path(d) / "label" / f"{k:04d}.png", out / "labels/seg" / f"{name}__{k:04d}.png"); kept += 1
        np.save(out / "labels/depth" / f"{name}__{k:04d}.npy", (np.asarray(Image.open(Path(d) / "depth" / f"{k:04d}.png")).astype(np.float32) / 1000).astype(np.float16))   # metres, 0 = no surface
        if status != "ok":
            dropped += 1; continue                                                # no corner file: the trainer skips the corner loss for this frame
        np.save(out / "labels/corners" / f"{name}__{k:04d}.npy", corners); with_gate += int(len(corners) > 0)
    stats[name] = {"frames": kept, "corners_unknown_partial_gate": dropped, "corners_known": kept - dropped, "of_which_with_full_gate": with_gate}
if goldens:
    for sub in ("images", "seg", "depth"):
        (out / "golden" / sub).mkdir(parents=True, exist_ok=True)
    for g in goldens:
        gname, gd = g.split("=", 1); n = 0
        for k, corners, wh, status in frames(gd):
            gid = f"{gname}__{k:04d}"
            shutil.copy(Path(gd) / "rgb" / f"{k:04d}.png", out / "golden/images" / f"{gid}.png"); shutil.copy(Path(gd) / "label" / f"{k:04d}.png", out / "golden/seg" / f"{gid}.png")
            np.save(out / "golden/depth" / f"{gid}.npy", np.asarray(Image.open(Path(gd) / "depth" / f"{k:04d}.png")).astype(np.float32) / 1000); n += 1
        stats["golden_" + gname] = {"frames": n}
print(json.dumps(stats))
