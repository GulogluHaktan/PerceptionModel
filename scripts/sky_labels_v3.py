"""Label set v3: sky becomes class 0 (background) instead of 'ignored', and BDD / IDD-AW vegetation becomes tree_pole. Reason (Update 54): with sky ignored everywhere the model paints sky as tree, ground or building
(visible in every output sheet); for obstacle avoidance a tree in the sky is a false obstacle. Class list unchanged (0 = background was unused).
Writes new files only: data/general/train/labels_v3/seg/{bdd__,ade__,rugd__}*.png, and evaluation label sets data/general/eval_{bdd3,ade3,rugd_heldout3,iddaw3}/seg (images linked to the existing ones).
ADE: scripts/ade_relabel_v2.py rules + sky/cloud -> 0 (run that script with SKY=1). Here: BDD (train id 10 sky -> 0, 8 vegetation -> tree_pole), RUGD (sky colour -> 0), IDD-AW (25 sky -> 0, 24 vegetation -> tree_pole)."""
import glob, os
import numpy as np
from PIL import Image
TR = "data/general/train"; V3 = f"{TR}/labels_v3/seg"; os.makedirs(V3, exist_ok=True); TREE = 2
def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)
def link_images(src, dst):
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f"{dst}/images"): os.symlink(os.path.realpath(f"{src}/images"), f"{dst}/images")
    os.makedirs(f"{dst}/seg", exist_ok=True)
# BDD
n = 0
for split, old_dir, pre, out_dir in (("train", f"{TR}/labels/seg", "bdd__", V3), ("val", "data/general/eval_bdd/seg", "", "data/general/eval_bdd3/seg")):
    if split == "val": link_images("data/general/eval_bdd", "data/general/eval_bdd3")
    for f in sorted(glob.glob(f"{old_dir}/{pre}*.png")):
        sid = os.path.basename(f)[len(pre):-4]; rawp = f"data/general/raw_bdd/data/seg_maps/labels/{split}/shard_000/{sid}_train_id.png"
        if not os.path.exists(rawp): continue
        raw = np.asarray(crop43(Image.open(rawp), Image.NEAREST)); lab = np.asarray(Image.open(f)).copy(); lab[raw == 10] = 0; lab[raw == 8] = TREE; Image.fromarray(lab).save(f"{out_dir}/{pre}{sid}.png"); n += 1
print("bdd", n, flush=True)
# RUGD (train frames and the held-out eval frames)
n = 0; link_images("data/general/eval_rugd_heldout", "data/general/eval_rugd_heldout3")
for old_dir, pre, out_dir in ((f"{TR}/labels_v2/seg", "rugd__", V3), ("data/general/eval_rugd_heldout/seg", "", "data/general/eval_rugd_heldout3/seg")):
    for f in sorted(glob.glob(f"{old_dir}/{pre}*.png")):
        sid = os.path.basename(f)[len(pre):-4]; seq = sid.rsplit("_", 1)[0]; rawp = f"data/general/raw_rugd/RUGD_annotations/{seq}/{sid}.png"
        if not os.path.exists(rawp): continue
        a = np.asarray(crop43(Image.open(rawp).convert("RGB"), Image.NEAREST)); lab = np.asarray(Image.open(f)).copy(); lab[(a[..., 0] == 0) & (a[..., 1] == 0) & (a[..., 2] == 255)] = 0; Image.fromarray(lab).save(f"{out_dir}/{pre}{sid}.png"); n += 1
print("rugd", n, flush=True)
# IDD-AW (evaluation only)
n = 0; link_images("data/general/eval_iddaw", "data/general/eval_iddaw3")
for f in sorted(glob.glob("data/general/eval_iddaw/seg/*.png")):
    sid = os.path.basename(f)[:-4]; drive, fr = sid.split("_"); rawp = f"data/general/raw_iddaw/val/gt_labels/{drive}/{fr}_labellevel3Ids.png"
    raw = np.asarray(crop43(Image.open(rawp), Image.NEAREST)); lab = np.asarray(Image.open(f)).copy(); lab[raw == 25] = 0; lab[raw == 24] = TREE; Image.fromarray(lab).save(f"data/general/eval_iddaw3/seg/{sid}.png"); n += 1
print("iddaw", n)
