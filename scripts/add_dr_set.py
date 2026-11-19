"""Adds the DroneRacing renderer's frames (isaac/collect_percep.py data.pt: 616x820 RGB, true inner gate corners as image-normalised coordinates) to the shared bigG image set as set `dr1`.
Gate labels only (corners_full json, scale 256x192 like the other racing sets); no seg / depth labels (the loss skips them; teacher DINO / depth are cached for distillation).
A gate is labelled when >= 2 of its corners are visible (in view and not occluded); its other corners keep their geometric position when they lie within 3 image sizes of the frame (outside the picture), and are None (hidden, no label) when they are inside the picture but occluded.
Append only. Usage: python scripts/add_dr_set.py <data.pt> [set=dr1] [max_frames=12000]"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image
src = sys.argv[1]; S = sys.argv[2] if len(sys.argv) > 2 else "dr1"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
M = Path("data/mix"); T = Path("data/train_sim/images") / S; T.mkdir(parents=True, exist_ok=True)
for d in (M / "bigG/images" / S, M / "cache_src" / S, M / "labels/corners_full"): d.mkdir(parents=True, exist_ok=True)
d = torch.load(src, map_location="cpu", mmap=True, weights_only=False); n = min(N, d["img"].shape[0]); n_g = 0
for i in range(n):
    sid = f"{S}__{i:05d}"; f = T / f"{sid}.jpg"
    if not f.exists(): Image.fromarray(d["img"][i].permute(1, 2, 0).numpy()).resize((410, 308), Image.BICUBIC).save(f, quality=93)
    for dd in (M / "bigG/images" / S, M / "cache_src" / S):
        l = dd / f.name
        if not (l.exists() or l.is_symlink()): os.symlink(f.resolve(), l)
    cf = M / "labels/corners_full" / f"{sid}.json"
    if cf.exists(): continue
    rows = []; cor = d["corners"][i].float().numpy(); vis = d["vis"][i].numpy(); sv = d["slot_valid"][i].numpy()
    for k in range(cor.shape[0]):
        if not sv[k] or vis[k].sum() < 2: continue
        c = cor[k]; pts = []
        for j in range(4):
            inside = (0 <= c[j, 0] < 1) and (0 <= c[j, 1] < 1)
            ok = vis[k][j] or (not inside and abs(c[j]).max() < 3)
            pts.append([float(c[j, 0] * 256), float(c[j, 1] * 192)] if ok else None)
        if sum(p is not None for p in pts) >= 2: rows.append({"corners_px": pts, "depths": [1.0] * 4}); n_g += 1
    json.dump(rows, open(cf, "w"))
print(n, "frames", n_g, "gates labelled")
base = yaml.safe_load(open(M / "cache_bdd.yaml")); yaml.safe_dump(dict(base, images=str(M / "cache_src"), cache=str(M / "cache"), labels=str(M / "labels")), open(M / f"cache_{S}.yaml", "w"))
subprocess.run([".venv/bin/python", "scripts/cache_teachers.py", str(M / f"cache_{S}.yaml")], check=True); print("added", S)
