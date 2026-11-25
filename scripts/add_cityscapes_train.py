"""Cityscapes TRAIN split (human fine labels, trainIds = our 19 outdoor classes) -> data/general/train/images/cityscapes/<id>.jpg and
labels_out/seg/cityscapes__<id>.png, so the outdoor module trainer uses it as human-labelled data (pool(): real). The val split stays
out of training (data/general/eval_cityscapes is the test set). Run: python scripts/add_cityscapes_train.py"""
import glob
from pathlib import Path
from PIL import Image
T = Path("data/general/train"); I = T / "images/cityscapes"; L = T / "labels_out/seg"; I.mkdir(parents=True, exist_ok=True); n = 0
for f in sorted(glob.glob("data/general/raw_cityscapes/cityscapes/leftImg8bit/train/*/*_leftImg8bit.png")):
    st = Path(f).name.replace("_leftImg8bit.png", ""); lab = f.replace("leftImg8bit/train", "gtFine/train").replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
    if not Path(lab).exists() or (L / f"cityscapes__{st}.png").exists(): continue
    Image.open(f).convert("RGB").save(I / f"{st}.jpg", quality=94); Image.open(lab).save(L / f"cityscapes__{st}.png"); n += 1
print("added", n)
