"""Label set v5 = v4 + two-wheelers as vehicles. Until v4 a motorcycle / bicycle was other_obstacle while its rider was person; the model then paints whole motorcycles as person or splits them
between person, car and other_obstacle (user remark on the IDD-AW sheet, 2026-11-16; confusion table in Update 58: 25% of the pixels predicted person on IDD-AW are other_obstacle). For a drone a motorcycle
is a moving vehicle, so motorcycle and bicycle now map to car; the rider stays person. Class list unchanged.
Writes new files only: data/general/train/labels_v5/seg/bdd__*.png (from labels_v4 + BDD train ids 17, 18) and the evaluation label sets data/general/eval_bdd5, eval_iddaw5 (from the v3 sets + raw ids;
IDD level-3 ids 6 motorcycle, 7 bicycle). ADE: run `SKY=1 TWO=1 scripts/ade_relabel_v2.py` (labels_v5 / eval_ade5). Not changed: Aeroscapes 'bike' and RUGD 'bicycle' stay other_obstacle (few pixels).
Usage: python scripts/two_wheel_labels_v5.py"""
import glob, os
import numpy as np
from PIL import Image
TR = "data/general/train"; V5 = f"{TR}/labels_v5/seg"; os.makedirs(V5, exist_ok=True); CAR = 7
def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)
def link_images(src, dst):
    os.makedirs(f"{dst}/seg", exist_ok=True)
    if not os.path.exists(f"{dst}/images"): os.symlink(os.path.realpath(f"{src}/images"), f"{dst}/images")
link_images("data/general/eval_bdd", "data/general/eval_bdd5"); link_images("data/general/eval_iddaw", "data/general/eval_iddaw5")
for split, old_dir, pre, out_dir in (("train", f"{TR}/labels_v4/seg", "bdd__", V5), ("val", "data/general/eval_bdd3/seg", "", "data/general/eval_bdd5/seg")):
    n = px = fr = 0
    for f in sorted(glob.glob(f"{old_dir}/{pre}*.png")):
        sid = os.path.basename(f)[len(pre):-4]; raw = np.asarray(crop43(Image.open(f"data/general/raw_bdd/data/seg_maps/labels/{split}/shard_000/{sid}_train_id.png"), Image.NEAREST))
        lab = np.asarray(Image.open(f)).copy(); m = (raw == 17) | (raw == 18); lab[m] = CAR; Image.fromarray(lab).save(f"{out_dir}/{pre}{sid}.png"); n += 1; px += m.mean(); fr += m.any()
    print(f"bdd {split}: {n} frames, {fr} with a two-wheeler, mean pixel share {px / n:.5f}", flush=True)
n = px = fr = 0
for f in sorted(glob.glob("data/general/eval_iddaw3/seg/*.png")):
    sid = os.path.basename(f)[:-4]; drive, k = sid.split("_"); raw = np.asarray(crop43(Image.open(f"data/general/raw_iddaw/val/gt_labels/{drive}/{k}_labellevel3Ids.png"), Image.NEAREST))
    lab = np.asarray(Image.open(f)).copy(); m = (raw == 6) | (raw == 7); lab[m] = CAR; Image.fromarray(lab).save(f"data/general/eval_iddaw5/seg/{sid}.png"); n += 1; px += m.mean(); fr += m.any()
print(f"iddaw eval: {n} frames, {fr} with a two-wheeler, mean pixel share {px / n:.5f}")
