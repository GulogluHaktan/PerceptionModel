"""Pseudo-labels for the seg module from the public Cityscapes Mask2Former teacher (scripts/eval_seg_teacher.py: BDD 0.81 / IDD-AW 0.67 vs core+SegRefine
0.745 / 0.628, person +0.2). Unlabelled driving images, none of them in any of our eval sets:
  ACDC (raw_acdc/datasets/{fog,night,rain,snow}/{trainA,trainB,testA,testB}: adverse frames + clear reference frames, 8012)  -> src "acdc"
  Mapillary Vistas shard 0 (raw_mvistas parquet, 8329 street-level frames from many countries, 400x300)                    -> src "mvist"
Images centre-cropped to 4:3 (640x480, Mapillary kept at 400x300) -> data/general/train/images/<src>/<id>.jpg ; labels (v5 + v3 sky/vegetation, pixels with
teacher confidence < CONF -> 255) -> data/general/train/labels_pseudo/seg/<src>__<id>.png.   Run: env -u PYTHONPATH .venv/bin/python scripts/make_seg_pseudo.py  (env CONF)"""
import os, sys, io, glob
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); from eval_seg_teacher import Teacher
CONF = float(os.environ.get("CONF", "0.6")); TR = Path("data/general/train"); LO = TR / "labels_pseudo/seg"; LO.mkdir(parents=True, exist_ok=True)


def crop43(im):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; im = im.crop((x0, y0, x0 + cw, y0 + ch))
    return im.resize((640, 480), Image.BICUBIC) if cw > 640 else im


def items():
    for p in sorted(glob.glob("data/general/raw_acdc/datasets/*/*/*.png")):
        c, s = p.split("/")[-3:-1]; yield "acdc", f"{c}_{s}_{Path(p).stem}", lambda p=p: Image.open(p).convert("RGB")
    import pyarrow.parquet as pq
    for f in sorted(glob.glob("data/general/raw_mvistas/data/*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for i, r in enumerate(pf.read_row_group(g, columns=["image"]).to_pylist()):
                yield "mvist", f"{Path(f).stem[:11]}_{g}_{i}", lambda b=r["image"]["bytes"]: Image.open(io.BytesIO(b)).convert("RGB")


def main():
    T = Teacher(); n = 0; stats = {}
    for src, sid, get in items():
        lp = LO / f"{src}__{sid}.png"; ip = TR / "images" / src / f"{sid}.jpg"
        if lp.exists() and ip.exists(): continue
        im = crop43(get()); lab, c = T.label(im, CONF); ip.parent.mkdir(parents=True, exist_ok=True); im.save(ip, quality=95); Image.fromarray(lab).save(lp)
        s = stats.setdefault(src, [0, 0.0]); s[0] += 1; s[1] += float((lab != 255).mean()); n += 1
        if n % 500 == 0: print(n, {k: (v[0], round(v[1] / v[0], 3)) for k, v in stats.items()}, flush=True)
    print("done", {k: (v[0], round(v[1] / max(1, v[0]), 3)) for k, v in stats.items()}, flush=True)


if __name__ == "__main__":
    main()
