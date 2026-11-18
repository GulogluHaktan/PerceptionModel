"""ADE20K (1aurent/ADE20K on Hugging Face, BSD) outdoor scenes -> our classes. Segmentation RGB decode: class = R//10*256 + G (matches the `name_ndx` of `objects`, checked); 0 = unlabelled.
Name rules (object `name`): building/house/skyscraper/tower/church/castle/wall/hovel/bridge -> building; road/sidewalk/grass/earth/ground/field/path/sand/pavement/dirt/hill/lawn/runway/floor -> ground_road; tree/palm/pole/streetlight/street lamp -> tree_pole; car/truck/bus/van/automobile -> car; person -> person; rock/stone -> rock;
sky/water/sea/river/lake/mountain/plant/flora/unlabelled -> 255 (ignored; sim outdoor labels also leave sky unlabelled); everything else -> other_obstacle.
Train: outdoor scenes of train shards 0-9 (up to 3,000) -> data/general/train/images/ade, labels/seg/ade__<id>.png; eval: 300 outdoor validation frames -> data/general/eval_ade/{images,seg}. Usage: python scripts/ade_prepare.py"""
import io, os, re, subprocess
import numpy as np, pyarrow.parquet as pq
from PIL import Image
RAW = "data/general/ade_raw"; TR = "data/general/train"; EV = "data/general/eval_ade"; B = "https://huggingface.co/datasets/1aurent/ADE20K/resolve/main/data/"
for d in (RAW, f"{TR}/images/ade", f"{TR}/labels/seg", f"{EV}/images", f"{EV}/seg"): os.makedirs(d, exist_ok=True)
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}
RULES = [("sky|water|sea|river|lake|mountain|plant|flora|waterfall|pond|swimming", 255), ("person", ix["person"]), ("tree|palm|pole|streetlight|street lamp", ix["tree_pole"]), ("^car|automobile|truck|^bus|^van|minibike|motorcycle", ix["car"]),
         ("building|house|skyscraper|tower|church|castle|^wall|hovel|bridge|edifice|garage|shed", ix["building"]), ("road|sidewalk|grass|earth|ground|field|path|sand|pavement|dirt|hill|lawn|runway|^floor|crosswalk|terrain", ix["ground_road"]), ("rock|stone", ix["rock"])]
def name2cls(n):
    n = n.lower()
    for pat, c in RULES:
        if re.search(pat, n): return c
    return ix["other_obstacle"]
def get(name):
    p = f"{RAW}/{name}"
    if not os.path.exists(p) or os.path.getsize(p) < 1e8: subprocess.run(["curl", "-sL", "-o", p, B + name], check=True)
    return pq.ParquetFile(p)
def convert(pf, quota, img_dir, seg_dir, prefix, tag, seen):
    n = 0
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g, columns=["image", "segmentations", "filename", "scene", "objects"]).to_pylist():
            if n >= quota: return n
            if not r["scene"] or r["scene"][0] != "outdoor" or not r["segmentations"]: continue
            seg = np.asarray(Image.open(io.BytesIO(r["segmentations"][0]["bytes"])).convert("RGB")); cls = seg[..., 0].astype(np.int32) // 10 * 256 + seg[..., 1].astype(np.int32)
            names = {o["name_ndx"]: o["name"] for o in r["objects"] if o["parts"]["part_level"] == 0}; out = np.full(cls.shape, ix["other_obstacle"], np.uint8)
            for k in np.unique(cls):
                out[cls == k] = 255 if k == 0 else name2cls(names.get(int(k), "unknown")) if int(k) in names else 255
                seen.setdefault(int(k), names.get(int(k), "?"))
            sid = os.path.splitext(r["filename"])[0].replace("ADE_", "") + tag
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").resize((640, 480), Image.BICUBIC).save(f"{img_dir}/{sid}.jpg", quality=95)
            Image.fromarray(out).resize((640, 480), Image.NEAREST).save(f"{seg_dir}/{prefix}{sid}.png"); n += 1
    return n
seen = {}; n = 0
for k in range(10):
    n += convert(get(f"train-{k:05d}-of-00010.parquet"), 3000 - n, f"{TR}/images/ade", f"{TR}/labels/seg", "ade__", "", seen); print("train shard", k, n, flush=True)
    if n >= 3000: break
m = convert(get("validation-00000-of-00001.parquet"), 300, f"{EV}/images", f"{EV}/seg", "", "", seen); print("done", n, "train", m, "eval")
from collections import Counter
c = Counter(name2cls(v) for v in seen.values()); print({(CL[k] if k < 255 else "ignore"): v for k, v in c.items()}); print("sample mapping:", {v[:22]: (CL[name2cls(v)] if name2cls(v) < 255 else "ignore") for v in list(seen.values())[:25]})
