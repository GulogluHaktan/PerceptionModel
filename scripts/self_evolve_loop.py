"""Usage: python scripts/self_evolve_loop.py <config.yaml> <real_sources> <pool_sources> <n_real> <n_pool> <cycles> <real_labels_dir> [conf=0.8]
The 5-cycle collapse test of decision 9 (use cycles=5; run it at several n_real/n_pool ratios for the real-fraction axis)."""
import sys, yaml
sys.path.insert(0, "src")
from pfm.selfloop import run_loop
a = sys.argv
cfg = yaml.safe_load(open(a[1]))
run_loop(cfg, a[2].split(","), a[3].split(","), int(a[4]), int(a[5]), int(a[6]), a[7], conf=float(a[8]) if len(a) > 8 else 0.8)
