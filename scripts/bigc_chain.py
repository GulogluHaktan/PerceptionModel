"""bigC = the bigB data mix (sim 20k + light/mix frames + TII + MonoRaceGate without red + DIODE + SUN + ADE) with three changes:
dense gate head (centre heatmap + corner offsets), augmentation (zoom-in crop, flip, photometric), general sets as gate-free negatives. vit, 8 epochs. Waits for the bigB chain.
Evaluations use PFM_GATE_HEAD=dense so that `slots` is the decoded dense head. Usage: python scripts/bigc_chain.py [tag=bigC] [extra yaml 'k: v' ...]"""
import os, subprocess, sys, time
from pathlib import Path
import yaml
tag = sys.argv[1] if len(sys.argv) > 1 else "bigC"; PY = ".venv/bin/python"; M = Path("data/mix")
def sh(cmd, check=True):
    print(">>", cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode and check: print("FAILED", cmd, flush=True); sys.exit(1)
T0 = time.time()
while "CHAIN_END" not in Path(os.path.expanduser("~/bigb.log")).read_text(): time.sleep(30)
# Aeroscapes (drone-view real segmentation) added to the mix
import random
while not Path(os.path.expanduser("~/aero_done.txt")).exists() and time.time() - T0 < 1500: time.sleep(30)      # separate tmux session downloads Aeroscapes
HAVE_AERO = len(list(Path("data/general/train/images/aero").glob("*.jpg"))) >= 2600; print("aeroscapes in mix:", HAVE_AERO, flush=True)
base = yaml.safe_load(open("configs/sim_v2_5090.yaml"))
if HAVE_AERO and len(list((M / "cache/dino").glob("aero__*.pt"))) < 2600:
    yaml.safe_dump(dict(base, images="data/general/train/images", cache=str(M / "cache"), labels="data/general/train/labels"), open(M / "cache_aero.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_aero.yaml")
for f in Path("data/general/train/labels/seg").glob("aero__*"):
    d = M / "labels/seg" / f.name
    if not d.exists() and not d.is_symlink(): os.symlink(f.resolve(), d)
img = M / "bigC/images"
if not img.exists():
    sh(f"mkdir -p {img} && cp -as $(pwd)/data/mix/bigB/images/. {img}/")
    if HAVE_AERO: sh(f"mkdir -p {img}/aero && cp -as $(pwd)/data/general/train/images/aero/. {img}/aero/")
cfg = yaml.safe_load(open(M / "bigB.yaml")); cfg.update(images=str(img), out_dir=f"runs/{tag}", gate_head="dense", aug=True, gate_free_prefixes=["diode__", "sun__", "ade__", "aero__"], workers=12)
for kv in sys.argv[2:]: cfg.update(yaml.safe_load(kv))
cp = M / f"{tag}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
if not Path(f"runs/{tag}/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
E_ = f"PFM_GATE_HEAD={cfg['gate_head']} KINDS=vit KIND=vit CKPT=runs/{tag}"
sh(f"{E_} GOLDEN=data/real_mrg/golden FLIGHTS=red_s1_f1,red_s1_f2,red_s1_f3 {PY} scripts/eval_real_golden.py")
sh(f"{E_} THR=0.3 GOLDEN=data/real_mrg/golden FLIGHTS=red_s1_f1,red_s1_f2,red_s1_f3 {PY} scripts/eval_real_golden.py")
sh(f"{E_} {PY} scripts/eval_real_golden.py")
sh(f"{E_} EVAL_ROOT=data/heldout TAG={tag} {PY} scripts/eval_slots_sim.py sim_racec")
sh(f"{E_} EVAL_ROOT=data/heldout TAG={tag} {PY} scripts/eval_slots_sim.py sim_racep", check=False)
sh(f"{E_} {PY} scripts/eval_depth_diode.py")
sh(f"{E_} {PY} scripts/eval_seg_sun.py")
sh(f"{E_} EVDIR=data/general/eval_ade {PY} scripts/eval_seg_sun.py")
sh(f"{E_} EVDIR=data/general/eval_aero {PY} scripts/eval_seg_sun.py", check=False)
sh(f"{E_} {PY} scripts/eval_real_negatives.py runs/{tag}", check=False)
sh(f"{E_} EVAL_ROOT=data/heldout TAG={tag} {PY} scripts/eval_students_sim.py sim_outdoor", check=False)
sh(f"{E_} OUT_TAG=_{tag} {PY} scripts/eval_robustness_sim.py 4 | tee ~/{tag}_robust.txt", check=False)
print("CHAIN DONE", flush=True)
