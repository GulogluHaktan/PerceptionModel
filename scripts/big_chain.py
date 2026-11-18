"""Consolidated recipe, red held out: 20,000 sim + 3,000 TII partial positives + 608 TII negatives + MonoRaceGate (ADNEC+DHL+marina) + all DIODE/SUN/ADE frames, 8 epochs. Previous: + DIODE (600), SUN RGB-D (600) and ADE20K outdoor (600) general frames in the mix; same split A as mrgA. Original: Leave-one-place-out with MonoRaceGate in the mix. N=3000, 16 epochs, vit. Split A: train on ADNEC+DHL+marina, test on red; split B: train on red+DHL+marina, test on ADNEC.
Mix = TII partial positives (450) + TII negatives (150) + all MonoRaceGate positives/negative crops of the training places + sim for the rest. Baseline (no MonoRaceGate): neg_r300 (red 0.420, ADNEC 0.411 within 20 px). Summary runs/mrg_summary.md"""
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
base = yaml.safe_load(open("configs/sim_v2_5090.yaml")); sim_root = Path("data/train_sim"); sim_pick = random.Random(0).sample(sorted((sim_root / "images").rglob("*.png")), 20000)
if len(list((M / "cache/dino").glob("diode__*.pt"))) < 2700:
    yaml.safe_dump(dict(base, images="data/general/train/images", cache=str(M / "cache"), labels="data/general/train/labels"), open(M / "cache_diode.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_diode.yaml")
for f in Path("data/general/train/labels/depth").glob("*.npy"): link(f, M / "labels/depth" / f.name)
if len(list((M / "cache/dino").glob("sun__*.pt"))) < 3900:
    yaml.safe_dump(dict(base, images="data/general/train/images", cache=str(M / "cache"), labels="data/general/train/labels"), open(M / "cache_sun.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_sun.yaml")
for sub in ("seg", "depth"):
    for f in Path(f"data/general/train/labels/{sub}").glob("sun__*"): link(f, M / "labels" / sub / f.name)
if len(list((M / "cache/dino").glob("ade__*.pt"))) < 3000:
    yaml.safe_dump(dict(base, images="data/general/train/images", cache=str(M / "cache"), labels="data/general/train/labels"), open(M / "cache_ade.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_ade.yaml")
for f in Path("data/general/train/labels/seg").glob("ade__*"): link(f, M / "labels/seg" / f.name)
ade = sorted(Path("data/general/train/images/ade").glob("*.jpg")); random.Random(9).shuffle(ade)
sun = sorted(Path("data/general/train/images/sun").glob("*.jpg")); random.Random(8).shuffle(sun)
dio = sorted(Path("data/general/train/images/diode").glob("*.jpg")); random.Random(7).shuffle(dio)
if len(list((M / "cache/dino").glob("real_mrg*__*.pt"))) < 500:
    yaml.safe_dump(dict(base, images="data/train_mrg/images", cache=str(M / "cache"), labels="data/train_mrg/labels"), open(M / "cache_mrg.yaml", "w")); sh(f"{PY} scripts/cache_teachers.py {M}/cache_mrg.yaml")
lab = M / "labels"
for p in sim_pick:
    s = str(p.relative_to(sim_root / "images")).rsplit(".", 1)[0].replace("/", "__")
    for t in ("dino", "depth"): link(sim_root / f"cache/{t}/{s}.pt", M / f"cache/{t}/{s}.pt")
    for sub, ext in (("seg", "png"), ("corners", "npy"), ("corners_full", "json"), ("depth", "npy")):
        f = sim_root / f"labels/{sub}/{s}.{ext}"
        if f.exists(): link(f, lab / sub / f"{s}.{ext}")
for f in Path("data/train_real_p/labels/corners_full").glob("*.json"): link(f, lab / "corners_full" / f.name)
for f in Path("data/train_mrg/labels/corners_full").glob("*.json"): link(f, lab / "corners_full" / f.name)
tii = sorted(Path("data/train_real_p/images/real_tii_p").glob("*.jpg")); random.Random(0).shuffle(tii); tneg = sorted(Path("data/train_real/images/real_tii_neg").glob("*.jpg")); random.Random(5).shuffle(tneg)
mrg = sorted(Path("data/train_mrg/images/real_mrg").glob("*.jpg")); mneg = sorted(Path("data/train_mrg/images/real_mrg_neg").glob("*.jpg"))
res = []
for tag, train_pl, test_seq in (("bigA", ("adnec", "dhl", "marina"), "red_s1_f1,red_s1_f2,red_s1_f3"),):
    mp = [p for p in mrg if p.stem.split("_")[0] in train_pl]; mn = [p for p in mneg if p.stem.split("_")[0] in train_pl]; nsim = 20000
    img = M / tag / "images"; sh(f"rm -rf {img}")
    for p in sim_pick[:nsim]: link(p, img / p.relative_to(sim_root / "images"))
    for p in tii[:3000]: link(p, img / "real_tii_p" / p.name)
    for p in tneg[:608]: link(p, img / "real_tii_neg" / p.name)
    for p in mp: link(p, img / "real_mrg" / p.name)
    for p in mn: link(p, img / "real_mrg_neg" / p.name)
    for p in dio[:2700]: link(p, img / "diode" / p.name)
    for p in sun[:3964]: link(p, img / "sun" / p.name)
    for p in ade[:3000]: link(p, img / "ade" / p.name)
    print(tag, "mrg pos", len(mp), "mrg neg", len(mn), "sim", nsim, flush=True)
    cp = M / f"{tag}.yaml"; yaml.safe_dump(dict(base, images=str(img), labels=str(lab), cache=str(M / "cache"), out_dir=f"runs/{tag}", epochs=8, student="vit"), open(cp, "w"))
    if not Path(f"runs/{tag}/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
    sh(f"GOLDEN=data/real_mrg/golden FLIGHTS={test_seq} KINDS=vit CKPT=runs/{tag} {PY} scripts/eval_real_golden.py")
    sh(f"KINDS=vit CKPT=runs/{tag} {PY} scripts/eval_real_golden.py")
    sh(f"KINDS=vit EVAL_ROOT=data/heldout CKPT=runs/{tag} TAG={tag} {PY} scripts/eval_slots_sim.py sim_racec")
    sh(f"CKPT=runs/{tag} KIND=vit {PY} scripts/eval_depth_diode.py")
    sh(f"CKPT=runs/{tag} KIND=vit {PY} scripts/eval_seg_sun.py")
    sh(f"EVDIR=data/general/eval_ade CKPT=runs/{tag} KIND=vit {PY} scripts/eval_seg_sun.py")
    sh(f"KINDS=vit EVAL_ROOT=data/heldout CKPT=runs/{tag} TAG={tag} {PY} scripts/eval_students_sim.py sim_outdoor")
    res.append((tag, test_seq))
print("CHAIN DONE", flush=True)
