"""DIODE (ahd03/diode-depth-3k on Hugging Face, indoor+outdoor, 16-bit depth, 0 = invalid) -> data/general/train/images/diode/<id>.jpg + labels/depth/diode__<id>.npy (float16, 640x480, nearest) for 2,700 rows; every 10th row of all shards (300, indoor and outdoor mixed) -> data/general/eval_diode (images + depth npy + indoor flag) for a held-out depth test.
Usage: python scripts/diode_prepare.py"""
import io, json, os, subprocess
import numpy as np, pyarrow.parquet as pq
from PIL import Image
RAW = "data/general/diode_raw"; TR = "data/general/train"; EV = "data/general/eval_diode"
for d in (RAW, f"{TR}/images/diode", f"{TR}/labels/depth", f"{EV}/images", f"{EV}/depth"): os.makedirs(d, exist_ok=True)
meta = {}
for k in range(10):
    p = f"{RAW}/d{k}.parquet"
    if not os.path.exists(p) or os.path.getsize(p) < 1e8: subprocess.run(["curl", "-sL", "-o", p, f"https://huggingface.co/datasets/ahd03/diode-depth-3k/resolve/main/data/train-{k:05d}-of-00010.parquet"], check=True)
    for i, r in enumerate(pq.read_table(p).to_pylist()):
        sid = f"s{k}_{i:03d}"; im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").resize((640, 480), Image.BICUBIC); dp = np.asarray(Image.open(io.BytesIO(r["depth"]["bytes"]))).astype(np.float32)
        dp = np.asarray(Image.fromarray(dp).resize((640, 480), Image.NEAREST)).astype(np.float16)
        if (k * 300 + i) % 10: im.save(f"{TR}/images/diode/{sid}.jpg", quality=95); np.save(f"{TR}/labels/depth/diode__{sid}.npy", dp)
        else: im.save(f"{EV}/images/{sid}.jpg", quality=95); np.save(f"{EV}/depth/{sid}.npy", dp); meta[sid] = bool(r["indoor"])
    print("shard", k, flush=True)
json.dump(meta, open(f"{EV}/indoor.json", "w")); print("done", len(os.listdir(f"{TR}/images/diode")), "train", len(meta), "eval")
