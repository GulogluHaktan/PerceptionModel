"""Does sim appearance/lens augmentation help real transfer? Same equal-compute setup as real_mix_chain (N=3000, 16 epochs, vit): half of the sim share is replaced by augmented race frames (data/mix_aug).
Runs: aug_r0 (0 real) and aug_r300 (300 real). Compare with real_mix_f0 (0.00 golden recall) and real_mix_f10 (0.23). Summary: runs/real_aug_summary.md"""
import json, os, random, subprocess, sys
from pathlib import Path
import yaml
M = Path("data/mix"); A = Path("data/mix_aug"); N = 3000; PY = ".venv/bin/python"; NA = 1500
def sh(cmd, env=None):
    print(">>", cmd, flush=True)
    if subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}).returncode: print("FAILED", cmd, flush=True); sys.exit(1)
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink(): os.symlink(Path(src).resolve(), dst)
base = yaml.safe_load(open("configs/sim_v2_5090.yaml")); rng = random.Random(0); sim_root = Path("data/train_sim")
sim_pick = rng.sample(sorted((sim_root / "images").rglob("*.png")), N); real = sorted(Path("data/train_real/images").rglob("*.jpg")); rng.shuffle(real)
if len(list((M / "cache/dino").glob("sim_raceaug__*.pt"))) < NA:
    yaml.safe_dump(dict(base, images=str(A / "images"), cache=str(M / "cache"), labels=str(A / "labels")), open(M / "cache_aug.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_aug.yaml")
lab = M / "labels"
for sub in ("seg", "depth", "corners_full"):
    for f in (A / "labels" / sub).glob("*"): link(f, lab / sub / f.name)
aug = sorted((A / "images/sim_raceaug").glob("*.png"))[:NA]; res = []
for tag, nr in (("aug_r0", 0), ("aug_r300", 300)):
    nsim = (N - nr) // 2; naug = N - nr - nsim; img = M / tag / "images"
    sh(f"rm -rf {img}")
    for p in sim_pick[:nsim]: link(p, img / p.relative_to(sim_root / "images"))
    for p in aug[:naug]: link(p, img / "sim_raceaug" / p.name)
    for p in real[:nr]: link(p, img / "real_tii" / p.name)
    cfg = dict(base, images=str(img), labels=str(lab), cache=str(M / "cache"), out_dir=f"runs/{tag}", epochs=16, student="vit"); cp = M / f"{tag}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
    if not Path(f"runs/{tag}/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
    sh(f"KINDS=vit CKPT=runs/{tag} {PY} scripts/eval_real_golden.py")
    sh(f"KINDS=vit EVAL_ROOT=data/heldout CKPT=runs/{tag} TAG={tag} {PY} scripts/eval_slots_sim.py sim_racec")
    res.append((tag, nr, json.load(open(f"runs/eval/real_golden_{tag}_all.json"))["vit"]))
L = ["# Sim appearance + fisheye augmentation (vit, N=3000, 16 epochs, golden = 102 real frames)", "", "| run | real | aug frames | recall<20px | recall<10px | recall<40px | full<20px | partial<20px | unmatched/frame |", "|---|---|---|---|---|---|---|---|---|"]
for tag, nr, g in res: L.append(f"| {tag} | {nr} | {N - nr - (N - nr) // 2} | {g['recall<20px']} | {g['recall<10px']} | {g['recall<40px']} | {g['full_gates_recall<20px']} | {g['partial_gates_recall<20px']} | {g['unmatched_pred_per_frame']} |")
Path("runs/real_aug_summary.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
