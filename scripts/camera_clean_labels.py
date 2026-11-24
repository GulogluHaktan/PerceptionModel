"""Pseudo-label cleaning for own-camera frames (Update 167): the Cityscapes teacher labels window frames, walls, desk edges and ceilings
as "car" (it reads a view through a window as the inside of a vehicle: ego-vehicle prior). 109 of 131 outdoor val frames had car pixels,
up to half the frame. Vehicle classes (car, truck, bus, train) in connected regions larger than AREA of the image are set to 255 (ignored)
in labels_cam_out and labels_cam_out_cb (train and val alike, so the metric stays consistent). Originals are kept in *_raw.
Run: python scripts/camera_clean_labels.py"""
import shutil, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
T = Path("data/general/train"); AREA = 0.05; VEH = (13, 14, 15, 16)
for d in ("labels_cam_out", "labels_cam_out_cb"):
    src = T / d; raw = T / f"{d}_raw"
    if not src.exists(): continue
    if not raw.exists(): shutil.copytree(src, raw)
    nf = npx = 0
    for f in sorted(raw.glob("*.png")):
        y = np.array(Image.open(f)); veh = np.isin(y, VEH); hit = False
        if veh.any():
            lab, n = ndimage.label(veh); sizes = ndimage.sum(veh, lab, range(1, n + 1))
            for k, s in enumerate(sizes, 1):
                if s > AREA * y.size: y[lab == k] = 255; npx += int(s); hit = True
        nf += hit; Image.fromarray(y).save(src / f.name)
    print(d, "frames cleaned", nf, "pixels ignored", npx)
