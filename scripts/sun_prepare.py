"""SUN RGB-D (wyrx/SUNRGBD_seg on Hugging Face; verified by overlay: label = id2label index + 1, 0 = unlabelled) -> our 11 classes.
Map (label id after the +1 shift): wall 1, ceiling 22, door 8, window 9, blinds 13 -> building; floor 2, floor mat 20 -> ground_road; person 31 -> person; 0 -> 255 (ignore); every other furniture/prop -> other_obstacle.
Train: 3 shards (first 3,900 rows) -> data/general/train/images/sun, labels/seg/sun__<id>.png (our class ids, 640x480 nearest), labels/depth/sun__<id>.npy (uint16 sensor depth, 0 invalid, float16).
Eval: 300 rows of test shard 0 -> data/general/eval_sun/{images,seg,depth}. Usage: python scripts/sun_prepare.py"""
import io, os, subprocess
import numpy as np, pyarrow.parquet as pq
from PIL import Image
RAW = "data/general/sun_raw"; TR = "data/general/train"; EV = "data/general/eval_sun"; B = "https://huggingface.co/datasets/wyrx/SUNRGBD_seg/resolve/main/data/"
for d in (RAW, f"{TR}/images/sun", f"{TR}/labels/seg", f"{TR}/labels/depth", f"{EV}/images", f"{EV}/seg", f"{EV}/depth"): os.makedirs(d, exist_ok=True)
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}
lut = np.full(256, ix["other_obstacle"], np.uint8); lut[0] = 255
for k in (1, 22, 8, 9, 13): lut[k] = ix["building"]
for k in (2, 20): lut[k] = ix["ground_road"]
lut[31] = ix["person"]
def get(name):
    p = f"{RAW}/{name}"
    if not os.path.exists(p) or os.path.getsize(p) < 1e8: subprocess.run(["curl", "-sL", "-o", p, B + name], check=True)
    return pq.read_table(p).to_pylist()
def emit(r, im_dir, seg_dir, dp_dir, sid, prefix):
    im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").resize((640, 480), Image.BICUBIC); l = np.asarray(Image.open(io.BytesIO(r["label"]["bytes"])).resize((640, 480), Image.NEAREST))
    d = np.asarray(Image.open(io.BytesIO(r["depth"]["bytes"])).resize((640, 480), Image.NEAREST)).astype(np.float16)
    im.save(f"{im_dir}/{sid}.jpg", quality=95); Image.fromarray(lut[l]).save(f"{seg_dir}/{prefix}{sid}.png"); np.save(f"{dp_dir}/{prefix}{sid}.npy", d)
n = 0
for k in range(3):
    for i, r in enumerate(get(f"train-{k:05d}-of-00004.parquet")): emit(r, f"{TR}/images/sun", f"{TR}/labels/seg", f"{TR}/labels/depth", f"t{k}_{i:04d}", "sun__"); n += 1
    print("train shard", k, flush=True)
for i, r in enumerate(get("test-00000-of-00004.parquet")[::4][:300]): emit(r, f"{EV}/images", f"{EV}/seg", f"{EV}/depth", f"e{i:03d}", "")
print("done", n, "train")
