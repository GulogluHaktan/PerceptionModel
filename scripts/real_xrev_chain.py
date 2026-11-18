"""Partly visible real gates (hidden corners = NaN, masked in slot_loss) + negatives. Compare with neg_r300 / neg_r900 (full-view real gates only). Original docstring: Fix of the 'real frame = gate' shortcut: real positives AND real gate-free crops (empty slot target). N=3000, 16 epochs, vit.
Runs: neg_r300 (300 pos + 150 neg + 2550 sim), neg_r900 (900 pos + 450 neg + 1650 sim). Eval: golden recall, false alarms (real crops from golden flights, painted-out golden, Aeroscapes), sim held-out. Summary runs/real_xrev_summary.md"""
import json, os, random, subprocess, sys
from pathlib import Path
import yaml
M = Path("data/mix"); N = 3000; PY = ".venv/bin/python"
def sh(cmd, env=None):
    print(">>", cmd, flush=True)
    if subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}).returncode: print("FAILED", cmd, flush=True); sys.exit(1)
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink(): os.symlink(Path(src).resolve(), dst)
base = yaml.safe_load(open("configs/sim_v2_5090.yaml")); rng = random.Random(0); sim_root = Path("data/train_sim")
sim_pick = rng.sample(sorted((sim_root / "images").rglob("*.png")), N); TRAIN_FL = ("14", "15", "17"); real = sorted(p for p in Path("data/train_real_p/images/real_tii_p").glob("*.jpg") if p.stem[7:9] in TRAIN_FL); rng.shuffle(real); neg = sorted(p for p in Path("data/train_real/images/real_tii_neg").glob("*.jpg") if p.stem[7:9] in TRAIN_FL); random.Random(5).shuffle(neg)
if len(list((M / "cache/dino").glob("real_tii_p__*.pt"))) < 5000:
    yaml.safe_dump(dict(base, images="data/train_real_p/images", cache=str(M / "cache"), labels="data/train_real_p/labels"), open(M / "cache_realp.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_realp.yaml")
if len(list((M / "cache/dino").glob("real_tii_neg__*.pt"))) < len(neg):
    yaml.safe_dump(dict(base, images="data/train_real/images", cache=str(M / "cache"), labels="data/train_real/labels"), open(M / "cache_real2.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_real2.yaml")
lab = M / "labels"
for p in real[:900]: link(f"data/train_real_p/labels/corners_full/real_tii_p__{p.stem}.json", lab / "corners_full" / f"real_tii_p__{p.stem}.json")
for p in neg: link(f"data/train_real/labels/corners_full/real_tii_neg__{p.stem}.json", lab / "corners_full" / f"real_tii_neg__{p.stem}.json")
res = []
for tag, nr, nn, sd, tfl in (("xrev_r900", 900, 300, 0, "01,05,09"), ("xtrack_r900_s1", 900, 300, 1, "13,16,18")):
    TRAIN_FL = ("14", "15", "17") if tag == "xrev_r900" else ("02", "03", "04", "06", "07", "08", "10", "11", "12")
    real = sorted(p for p in Path("data/train_real_p/images/real_tii_p").glob("*.jpg") if p.stem[7:9] in TRAIN_FL); random.Random(0).shuffle(real); neg = sorted(p for p in Path("data/train_real/images/real_tii_neg").glob("*.jpg") if p.stem[7:9] in TRAIN_FL); random.Random(5).shuffle(neg)
    img = M / tag / "images"; sh(f"rm -rf {img}")
    for p in sim_pick[: N - nr - nn]: link(p, img / p.relative_to(sim_root / "images"))
    for p in real[:nr]: link(p, img / "real_tii_p" / p.name)
    for p in neg[:nn]: link(p, img / "real_tii_neg" / p.name)
    cfg = dict(base, images=str(img), labels=str(lab), cache=str(M / "cache"), out_dir=f"runs/{tag}", epochs=16, student="vit", seed=sd); cp = M / f"{tag}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
    if not Path(f"runs/{tag}/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
    sh(f"KINDS=vit CKPT=runs/{tag} FLIGHTS={tfl} {PY} scripts/eval_real_golden.py")
    sh(f"KINDS=vit EVAL_ROOT=data/heldout CKPT=runs/{tag} TAG={tag} {PY} scripts/eval_slots_sim.py sim_racec")
    pass
    res.append((tag, nr, nn, json.load(open(f"runs/eval/real_golden_{tag}_fl{tfl.replace(',', '-')}_all.json"))["vit"]))
L = ["| run | real pos | real neg | recall<20px | <10px | <40px | full<20px | partial<20px | unmatched/frame |", "|---|---|---|---|---|---|---|---|---|"]
for tag, nr, nn, g in res: L.append(f"| {tag} | {nr} | {nn} | {g['recall<20px']} | {g['recall<10px']} | {g['recall<40px']} | {g['full_gates_recall<20px']} | {g['partial_gates_recall<20px']} | {g['unmatched_pred_per_frame']} |")
Path("runs/real_xrev_summary.md").write_text("\n".join(L)); print("\n".join(L), flush=True)
