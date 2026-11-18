"""Aeroscapes (HF dronefreak/Aeroscapes, CC BY-SA 4.0; low-altitude drone views, official train/val split by sequence) -> our classes.
Map: person -> person; car -> car; drone -> opponent_drone; construction -> building; road -> ground_road; bike, boat, animal, obstacle -> other_obstacle;
background, vegetation, sky -> 255 (ignored: background is 'everything else', vegetation mixes trees and grass which are different classes here).
Train (2,621) -> data/general/train/images/aero + labels/seg/aero__<id>.png ; val: every 2nd (324) -> data/general/eval_aero/{images,seg}."""
import os
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
import time
for attempt in range(20):       # HF rate-limits anonymous bulk downloads of many small files: go slowly and resume
    try:
        root = snapshot_download("dronefreak/Aeroscapes", repo_type="dataset", local_dir="data/general/aero_raw", allow_patterns=["data/*", "data.yaml"], max_workers=2)
        cnt = {k: len(os.listdir(f"{root}/data/{a}/{b}")) if os.path.isdir(f"{root}/data/{a}/{b}") else 0 for k, (a, b) in {"it": ("images", "train"), "mt": ("masks", "train"), "iv": ("images", "val"), "mv": ("masks", "val")}.items()}; print(cnt, flush=True)
        if cnt["it"] >= 2621 and cnt["mt"] >= 2621 and cnt["iv"] >= 648 and cnt["mv"] >= 648: break
        time.sleep(120)
    except Exception as e:
        print("retry", attempt, str(e)[:120], flush=True); time.sleep(180)
else:
    raise SystemExit("aeroscapes download failed")
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}
lut = np.full(256, 255, np.uint8)
for k, c in {1: "person", 2: "other_obstacle", 3: "car", 4: "opponent_drone", 5: "other_obstacle", 6: "other_obstacle", 7: "other_obstacle", 8: "building", 10: "ground_road"}.items(): lut[k] = ix[c]
TR = "data/general/train"; EV = "data/general/eval_aero"
for d in (f"{TR}/images/aero", f"{TR}/labels/seg", f"{EV}/images", f"{EV}/seg"): os.makedirs(d, exist_ok=True)
n = m = 0
for split in ("train", "val"):
    for i, f in enumerate(sorted(os.listdir(f"{root}/data/images/{split}"))):
        sid = f[:-4]; mk = f"{root}/data/masks/{split}/{sid}.png"
        if not os.path.exists(mk) or (split == "val" and i % 2): continue
        im = Image.open(f"{root}/data/images/{split}/{f}").convert("RGB").resize((640, 480), Image.BICUBIC); l = lut[np.asarray(Image.open(mk).resize((640, 480), Image.NEAREST))]
        if split == "train": im.save(f"{TR}/images/aero/{sid}.jpg", quality=95); Image.fromarray(l).save(f"{TR}/labels/seg/aero__{sid}.png"); n += 1
        else: im.save(f"{EV}/images/{sid}.jpg", quality=95); Image.fromarray(l).save(f"{EV}/seg/{sid}.png"); m += 1
print("aeroscapes train", n, "eval", m)
