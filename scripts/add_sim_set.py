"""Adds a rendered sim set (data/train_sim/images/<set>, labels/{seg,depth,corners,corners_full}/<set>__*) to the shared bigG image set and the label folders, and caches the teacher outputs (DINO, depth) for the new images. Append only.
Only the effects in EFFECTS (default clear,blur,lowlight) are used, so the set does not swamp the mix.
Usage: [EFFECTS=clear,blur,lowlight] python scripts/add_sim_set.py <set>   (then train with bige_chain.py, which reads data/mix/bigG/images and data/mix/labels_J)"""
import os, subprocess, sys
from pathlib import Path
import yaml
S = sys.argv[1]; EFF = tuple(os.environ.get("EFFECTS", "clear,blur,lowlight").split(",")); M = Path("data/mix"); T = Path("data/train_sim"); root = Path.cwd()
src = [f for f in sorted((T / "images" / S).glob("*.png")) if f.stem.endswith(EFF)]; print(len(src), "images", EFF)
def link(f, d):
    d.parent.mkdir(parents=True, exist_ok=True)
    if not (d.exists() or d.is_symlink()): os.symlink(f.resolve(), d)
lab = {sub: {f.stem: f for f in (T / "labels" / sub).iterdir() if f.name.startswith(S + "__")} for sub in ("seg", "depth", "corners", "corners_full")}      # one directory listing per folder (a glob per image is quadratic)
for f in src:
    link(f, M / "bigG/images" / S / f.name); link(f, M / "cache_src" / S / f.name)
    for sub, d in lab.items():
        lf = d.get(f"{S}__{f.stem}")
        if lf is None: continue
        link(lf, M / "labels" / sub / lf.name)
        if sub == "seg": link(lf, M / "labels_J/seg" / lf.name)
base = yaml.safe_load(open("data/mix/cache_bdd.yaml")); yaml.safe_dump(dict(base, images=str(M / "cache_src"), cache=str(M / "cache"), labels=str(M / "labels")), open(M / f"cache_{S}.yaml", "w"))
subprocess.run([".venv/bin/python", "scripts/cache_teachers.py", str(M / f"cache_{S}.yaml")], check=True)
print("added", S)
