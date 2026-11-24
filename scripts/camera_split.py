"""Split the teacher-labelled own-camera frames (data/general/train/cam_split.json) into outdoor / indoor sets for the two modules:
a frame is outdoor when >30 % of its valid outdoor-teacher pixels are road / sidewalk / vegetation / terrain / sky. Within each set, val =
every 5th block of 30 consecutive frames per recording (neighbouring frames are near duplicates: blocks keep val from leaking into train).
rot180 frames are left out. Writes cam_split_out.json and cam_split_in.json. Run: python scripts/camera_split.py"""
import json, re, numpy as np
from pathlib import Path
from PIL import Image
T = Path("data/general/train"); sp = json.loads((T / "cam_split.json").read_text())
OUTDOORISH = [0, 1, 8, 9, 10]
sets = {"out": [], "in": []}
for n in sorted(sp["train"] + sp["val"]):
    if n.startswith("rot180"): continue
    lo = np.asarray(Image.open(T / "labels_cam_out" / f"{n}.png")); v = lo != 255
    sets["out" if v.any() and np.isin(lo[v], OUTDOORISH).mean() > 0.3 else "in"].append(n)
for k, names in sets.items():
    res = {"train": [], "val": []}; byrec = {}
    for n in names: byrec.setdefault(re.sub(r"_\d+$", "", n), []).append(n)
    for rec, ns in byrec.items():
        for i, n in enumerate(sorted(ns)): res["val" if (i // 30) % 5 == 4 else "train"].append(n)
    (T / f"cam_split_{k}.json").write_text(json.dumps(res)); print(k, {p: len(v) for p, v in res.items()}, "recordings", len(byrec))
