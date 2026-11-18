"""sim_v5: the full sim_v2 recipe (all of data/train_sim, 8 epochs) plus 6,000 light-varied and 6,000 combined-degradation race frames (fog, blur, rain, light, JPEG). vit then hybrid, then stress test and held-out checks against sim_v2. Unattended."""
import os, subprocess, sys
from pathlib import Path
import yaml
M = Path("data/mix"); PY = ".venv/bin/python"
def sh(cmd, env=None):
    print(">>", cmd, flush=True)
    if subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}).returncode: print("FAILED", cmd, flush=True); sys.exit(1)
base = yaml.safe_load(open("configs/sim_v2_5090.yaml"))
if len(list(Path("data/mix_light/images/sim_racelight").glob("*.png"))) < 6000: sh(f"{PY} scripts/make_light_aug_sim.py 6000")
if len(list((M / "cache/dino").glob("sim_racelight__*.pt"))) < 6000:
    yaml.safe_dump(dict(base, images="data/mix_light/images", cache=str(M / "cache"), labels=str(M / "labels")), open(M / "cache_light.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_light.yaml")
if len(list(Path("data/mix_mix/images/sim_racemix").glob("*.png"))) < 6000: sh(f"{PY} scripts/make_mix_aug_sim.py 6000")
if len(list((M / "cache/dino").glob("sim_racemix__*.pt"))) < 6000:
    yaml.safe_dump(dict(base, images="data/mix_mix/images", cache=str(M / "cache"), labels=str(M / "labels")), open(M / "cache_mix.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_mix.yaml")
full = M / "full_mix/images"; sh(f"rm -rf {full} && mkdir -p {full}")
for d in list(Path("data/train_sim/images").iterdir()) + [Path("data/mix_light/images/sim_racelight"), Path("data/mix_mix/images/sim_racemix")]: sh(f"mkdir -p {full}/{d.name} && cp -as {d.resolve()}/. {full}/{d.name}/")
# labels + cache: train_sim ones plus the light ones (already linked into data/mix/labels / data/mix/cache): build merged label and cache dirs by symlinking directory entries
lab = M / "labels_full5"; cache = M / "cache_full5"; sh(f"rm -rf {lab} {cache} && mkdir -p {lab} {cache}")
for sub in ("seg", "corners", "corners_full", "depth"):
    sh(f"mkdir -p {lab}/{sub} && cp -as $(pwd)/data/train_sim/labels/{sub}/. {lab}/{sub}/ && cp -as $(pwd)/data/mix/labels/{sub}/. {lab}/{sub}/ 2>/dev/null; true")
for t in ("dino", "depth"):
    sh(f"mkdir -p {cache}/{t} && cp -as $(pwd)/data/train_sim/cache/{t}/. {cache}/{t}/ && for f in data/mix/cache/{t}/sim_racelight__*.pt data/mix/cache/{t}/sim_racemix__*.pt; do ln -sf $(pwd)/$f {cache}/{t}/; done")
sh(f"cp data/train_sim/cache/meta.json {cache}/meta.json")
for kind in ("vit", "hybrid"):
    cp = M / f"sim_v5_{kind}.yaml"; yaml.safe_dump(dict(base, images=str(full), labels=str(lab), cache=str(cache), out_dir="runs/sim_v5", student=kind), open(cp, "w"))
    if not Path(f"runs/sim_v5/{kind}/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
sh(f"CKPT=runs/sim_v5 OUT_TAG=_sim_v5 {PY} scripts/eval_robustness_sim.py 4 | tee ~/sim_v5_robust.txt")
for s in ("sim_indoor", "sim_outdoor"): sh(f"EVAL_ROOT=data/heldout CKPT=runs/sim_v5 TAG=sim_v5 {PY} scripts/eval_students_sim.py {s}")
sh(f"EVAL_ROOT=data/heldout CKPT=runs/sim_v5 TAG=sim_v5 {PY} scripts/eval_slots_sim.py sim_racec")
print("CHAIN DONE", flush=True)
