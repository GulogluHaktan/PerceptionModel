"""Real-fraction fine-tune chain (protocol decision 9, small version). Equal compute: N=3000 images per run, 16 epochs, vit from the ImageNet start (same as sim_v2).
Real = 942 TII frames from flights that are NOT in the golden candidates (data/train_real). Sim = random frames of data/train_sim. Fractions of real: 0, 0.1, 0.3 (300 / 900 real).
After training: golden (102 real frames, other flights) and sim held-out slot recall (forgetting check). Results: runs/eval/real_golden_real_mix_f*_*.json, runs/eval/sim_racec_slots_real_mix_f*.json, runs/real_mix_summary.md"""
import json, os, random, subprocess, sys
from pathlib import Path
import yaml
sys.path.insert(0, "src")
M = Path("data/mix"); N = 3000; rng = random.Random(0); PY = ".venv/bin/python"
def sh(cmd, env=None):
    print(">>", cmd, flush=True); r = subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})})
    if r.returncode: print("FAILED", cmd, flush=True); sys.exit(1)
base = yaml.safe_load(open("configs/sim_v2_5090.yaml"))
# 1) teacher cache for the real frames
rc = dict(base, images="data/train_real/images", cache=str(M / "cache"), labels="data/train_real/labels"); (M).mkdir(parents=True, exist_ok=True); yaml.safe_dump(rc, open(M / "cache_real.yaml", "w"))
if not (M / "cache/dino").exists() or len(list((M / "cache/dino").glob("real_tii__*.pt"))) < 900: sh(f"{PY} scripts/cache_teachers.py {M}/cache_real.yaml")
# 2) pools
sim_root = Path("data/train_sim"); sim = sorted(p for p in (sim_root / "images").rglob("*.png")); sim_pick = rng.sample(sim, N)
real = sorted((Path("data/train_real/images")).rglob("*.jpg")); rng.shuffle(real)
def sid(p, root): return str(p.relative_to(root)).rsplit(".", 1)[0].replace("/", "__")
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink(): os.symlink(src.resolve(), dst)
lab = M / "labels"
for p in sim_pick:
    s = sid(p, sim_root / "images")
    for t in ("dino", "depth"):
        link(sim_root / f"cache/{t}/{s}.pt", M / f"cache/{t}/{s}.pt")
    for sub, ext in (("seg", "png"), ("corners", "npy"), ("corners_full", "json"), ("depth", "npy")):
        f = sim_root / f"labels/{sub}/{s}.{ext}"
        if f.exists(): link(f, lab / sub / f"{s}.{ext}")
for p in real:
    s = "real_tii__" + p.stem; link(Path(f"data/train_real/labels/corners_full/{s}.json"), lab / "corners_full" / f"{s}.json")
(M / "cache/meta.json").write_text(json.dumps({"size": 448, "teachers": {"dino": "dinov2", "depth": "depth_da2"}}))
res = []
for frac in (0.0, 0.1, 0.3):
    nr = int(N * frac); tag = f"real_mix_f{int(frac * 100)}"; img = M / tag / "images"
    if img.exists(): sh(f"rm -rf {img}")
    for p in sim_pick[: N - nr]: link(p, img / p.relative_to(sim_root / "images"))
    for p in real[:nr]: link(p, img / "real_tii" / p.name)
    cfg = dict(base, images=str(img), labels=str(lab), cache=str(M / "cache"), out_dir=f"runs/{tag}", epochs=16, student="vit"); cp = M / f"{tag}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
    if not Path(f"runs/{tag}/vit/final.pt").exists(): sh(f"{PY} scripts/train.py {cp}")
    sh(f"KINDS=vit CKPT=runs/{tag} {PY} scripts/eval_real_golden.py")
    sh(f"KINDS=vit CKPT=runs/{tag} {PY} scripts/eval_real_golden.py", {"EXCLUDE_FLAGGED": "1"})
    sh(f"KINDS=vit EVAL_ROOT=data/heldout CKPT=runs/{tag} TAG={tag} {PY} scripts/eval_slots_sim.py sim_racec")
    g = json.load(open(f"runs/eval/real_golden_{tag}_all.json"))["vit"]; res.append((tag, nr, g))
lines = ["# Real-fraction fine-tune (vit, N=3000, 16 epochs, golden = 102 real frames from other flights)", "", "| run | real frames | recall<20px | recall<10px | median err px | partial recall<20px | unmatched pred/frame |", "|---|---|---|---|---|---|---|"]
for tag, nr, g in res: lines.append(f"| {tag} | {nr} | {g['recall<20px']} | {g['recall<10px']} | {g['median_err_px']} | {g['partial_gates_recall<20px']} | {g['unmatched_pred_per_frame']} |")
Path("runs/real_mix_summary.md").write_text("\n".join(lines)); print("\n".join(lines), flush=True)
