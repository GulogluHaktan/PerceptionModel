"""Outdoor 19-class (Cityscapes ids) labels from the raw BDD100K seg maps, with the same 4:3 centre crop + 640x480 as scripts/bdd_prepare.py:
  train: data/general/train/labels_out/seg/bdd__<id>.png (7000)   test: data/general/eval_bdd_out/{images,seg} (= the eval_bdd frames, every 3rd val)
  real dark test: data/general/eval_dark_out/{images,seg} (the test half of runs/selfimprove/real_dark_split.json, raw BDD val ids)."""
import os, json
import numpy as np
from pathlib import Path
from PIL import Image
RAW = Path("data/general/raw_bdd/data"); TR = Path("data/general/train"); LO = TR / "labels_out/seg"; LO.mkdir(parents=True, exist_ok=True)


def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)


n = 0
for f in sorted((RAW / "seg_maps/labels/train/shard_000").glob("*_train_id.png")):
    sid = f.name[:-len("_train_id.png")]
    if (TR / "images/bdd" / f"{sid}.jpg").exists(): crop43(Image.open(f), Image.NEAREST).save(LO / f"bdd__{sid}.png"); n += 1
print("train labels", n)
for name, stems in (("eval_bdd_out", [p.stem for p in sorted(Path("data/general/eval_bdd/images").glob("*.jpg"))]),
                    ("eval_dark_out", [Path(a).stem for a, _ in json.loads(Path("runs/selfimprove/real_dark_split.json").read_text())["test"]])):
    D = Path("data/general") / name; (D / "images").mkdir(parents=True, exist_ok=True); (D / "seg").mkdir(parents=True, exist_ok=True); m = 0
    for sid in stems:
        lf = RAW / "seg_maps/labels/val/shard_000" / f"{sid}_train_id.png"; jf = RAW / "images_10k/10k/val/shard_000" / f"{sid}.jpg"
        if not lf.exists() or not jf.exists(): continue
        crop43(Image.open(jf).convert("RGB"), Image.BICUBIC).save(D / "images" / f"{sid}.jpg", quality=95); crop43(Image.open(lf), Image.NEAREST).save(D / "seg" / f"{sid}.png"); m += 1
    print(name, m)
