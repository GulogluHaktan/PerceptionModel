"""Does light augmentation close the stress-test gaps (over-exposure, deep dark)? vit, N=3000, 16 epochs, sim only.
light_a = 1500 sim + 1500 light-augmented race frames; baseline = real_mix_f0 (3000 sim, same seed/recipe). Both through eval_robustness_sim (257 held-out race frames)."""
import os, random, subprocess, sys
from pathlib import Path
import yaml
M = Path("data/mix"); PY = ".venv/bin/python"; N = 3000
def sh(cmd, env=None):
    print(">>", cmd, flush=True)
    if subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}).returncode: print("FAILED", cmd, flush=True); sys.exit(1)
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink(): os.symlink(Path(src).resolve(), dst)
base = yaml.safe_load(open("configs/sim_v2_5090.yaml")); sim_root = Path("data/train_sim"); sim_pick = random.Random(0).sample(sorted((sim_root / "images").rglob("*.png")), N)
if len(list((M / "cache/dino").glob("sim_racelight__*.pt"))) < 1500:
    yaml.safe_dump(dict(base, images="data/mix_light/images", cache=str(M / "cache"), labels=str(M / "labels")), open(M / "cache_light.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_light.yaml")
img = M / "light_a/images"; sh(f"rm -rf {img}")
for p in sim_pick[:1500]: link(p, img / p.relative_to(sim_root / "images"))
for p in sorted(Path("data/mix_light/images/sim_racelight").glob("*.png"))[:1500]: link(p, img / "sim_racelight" / p.name)
cp = M / "light_a.yaml"; yaml.safe_dump(dict(base, images=str(img), labels=str(M / "labels"), cache=str(M / "cache"), out_dir="runs/light_a", epochs=16, student="vit"), open(cp, "w"))
if not Path("runs/light_a/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
for ck, tag in (("runs/light_a", "_light_a"), ("runs/real_mix_f0", "_base")): sh(f"KINDS=vit CKPT={ck} OUT_TAG={tag} {PY} scripts/eval_robustness_sim.py 4 | tee ~/light{tag}.txt")
