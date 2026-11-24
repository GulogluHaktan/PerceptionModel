"""Orin side of scripts/car_data_round.sh: every STEP-th frame of a raw dump (frames.raw + meta.json) -> one .npy stack (N,H,W) uint16.
Run on the Orin: python3 orin_sub_raw.py <recording dir> <out.npy> <step>"""
import sys, json, numpy as np
from pathlib import Path
src, dst, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
meta = json.loads((src / "meta.json").read_text()); W, H = meta["w"], meta["h"]; fb = W * H * 2
raw = np.memmap(src / "frames.raw", np.uint16, "r"); n = raw.size * 2 // fb
idx = list(range(0, n, step)); out = np.lib.format.open_memmap(dst, "w+", np.uint16, (len(idx), H, W))
for k, i in enumerate(idx): out[k] = raw[i * W * H:(i + 1) * W * H].reshape(H, W)
out.flush(); print(src.name, n, "->", len(idx))
