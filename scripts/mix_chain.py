"""Does combined-degradation augmentation lift fog and the worst mix? vit, N=3000, 16 epochs: mix_a = 1500 sim + 1500 combined-degradation race frames; compared with light_a / base from light_chain (same stress test, runs/eval/robustness_sim_race_*.json)."""
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
sh(f"{PY} scripts/make_mix_aug_sim.py 1500")
yaml.safe_dump(dict(base, images="data/mix_mix/images", cache=str(M / "cache"), labels=str(M / "labels")), open(M / "cache_mix.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_mix.yaml")
img = M / "mix_a/images"; sh(f"rm -rf {img}")
for p in sim_pick[:1500]: link(p, img / p.relative_to(sim_root / "images"))
for p in sorted(Path("data/mix_mix/images/sim_racemix").glob("*.png"))[:1500]: link(p, img / "sim_racemix" / p.name)
cp = M / "mix_a.yaml"; yaml.safe_dump(dict(base, images=str(img), labels=str(M / "labels"), cache=str(M / "cache"), out_dir="runs/mix_a", epochs=16, student="vit"), open(cp, "w"))
if not Path("runs/mix_a/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
sh(f"KINDS=vit CKPT=runs/mix_a OUT_TAG=_mix_a {PY} scripts/eval_robustness_sim.py 4 | tee ~/mix_a.txt")
print("CHAIN DONE", flush=True)
