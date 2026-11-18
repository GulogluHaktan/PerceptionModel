"""Decision 9: real-fraction test with EQUAL COMPUTE.
Usage: python scripts/real_fraction_test.py <config.yaml> <real_sources> <synth_sources> <N> [fractions=0,10,30,50,100]
Sources are sub-folder names under the config's `images` root (as written by fetch_data.py or your own
flight/sim folders). For every real fraction f a mix of N images is built with symlinks that keep the
`<source>/<file>` path, so the existing teacher cache and labels are reused. Equal compute = the same N,
batch, size, epochs and therefore the same number of optimiser steps for every fraction. GPU-seconds are
logged too. Results go to runs/real_fraction/results.csv (golden-set metrics per fraction).
Needs a golden set in the config, otherwise there is nothing to compare."""
import csv, os, random, sys, time
from pathlib import Path
import yaml
sys.path.insert(0, "src")
import torch
from pfm.data import EXT
from pfm.train import run

cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg.get("golden"), "real-fraction test needs a golden set in the config"
real_s, synth_s = sys.argv[2].split(","), sys.argv[3].split(",")
N = int(sys.argv[4])
fracs = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,10,30,50,100").split(",")]
root = Path(cfg["images"])


def files(srcs):
    return sorted(p.relative_to(root) for s in srcs for p in (root / s).rglob("*") if p.suffix.lower() in EXT)


real, synth = files(real_s), files(synth_s)
rng = random.Random(0)
rng.shuffle(real); rng.shuffle(synth)
for f in fracs:
    nr, ns = round(N * f / 100), N - round(N * f / 100)
    assert nr <= len(real) and ns <= len(synth), f"not enough images for {f}% real (need {nr} real, {ns} synth)"

out_root = Path("runs/real_fraction"); out_root.mkdir(parents=True, exist_ok=True)
rows = []
for f in fracs:
    nr = round(N * f / 100)
    pick = real[:nr] + synth[:N - nr]
    mix = Path(f"data/mix_real{f}")
    if mix.exists():
        import shutil; shutil.rmtree(mix)
    for rel in pick:
        dst = mix / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink((root / rel).resolve(), dst)
    c = dict(cfg, images=str(mix), out_dir=str(out_root / f"real{f}"))
    t = time.time()
    m = run(c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rows.append({"real_pct": f, "n_images": N, "gpu_seconds": round(time.time() - t, 1), **m})
    print("RESULT", rows[-1], flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("real_pct", "n_images", "gpu_seconds"), k))
with open(out_root / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", out_root / "results.csv")
