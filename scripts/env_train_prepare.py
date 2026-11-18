"""Training data from the two new environments, kept apart from their test data (Update 52).
  UAVid train split (200 frames, sequences other than the val/test sequences): each 4K frame gives the whole view plus its four quadrants (objects twice as large) -> 1,000 images
      data/general/train/images/uav + data/general/train/labels_v2/seg/uav__<id>.png. Same class map as scripts/env_prepare.py.
      The same tiling of the 70 val frames -> data/general/eval_uavid_tiles (280 images, evaluation only): tells how much of the UAVid weakness is object size.
  RUGD: sequences creek, park-2, trail-4, village are held out for scoring (their frames of data/general/eval_rugd are linked into data/general/eval_rugd_heldout);
      every 8th annotated frame of the other 14 sequences -> data/general/train/images/rugd + labels_v2/seg/rugd__<id>.png."""
import os, time
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}; api = HfApi(); TR = "data/general/train"
def fetch(rid, rel, raw):
    for a in range(60):
        try: return hf_hub_download(rid, rel, repo_type="dataset", local_dir=raw)
        except Exception as e:
            print("retry", a, str(e)[:90], flush=True); time.sleep(45)
    raise SystemExit("download failed")
def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)
def views(im):      # whole frame + four quadrants
    W, H = im.size; yield "", im
    for k, (x, y) in enumerate(((0, 0), (W // 2, 0), (0, H // 2), (W // 2, H // 2))): yield f"_q{k}", im.crop((x, y, x + W // 2, y + H // 2))
for d in (f"{TR}/images/uav", f"{TR}/images/rugd", f"{TR}/labels_v2/seg", "data/general/eval_uavid_tiles/images", "data/general/eval_uavid_tiles/seg", "data/general/eval_rugd_heldout/images", "data/general/eval_rugd_heldout/seg"): os.makedirs(d, exist_ok=True)
rid, raw = "dronefreak/UAVid-2020", "data/general/raw_uavid"; lut = np.full(256, 255, np.uint8)
for k, c in {1: "building", 2: "ground_road", 3: "car", 4: "tree_pole", 6: "person", 7: "car"}.items(): lut[k] = ix[c]
allf = api.list_repo_files(rid, repo_type="dataset")
for split, io_, lo_, pre in (("train", f"{TR}/images/uav", f"{TR}/labels_v2/seg", "uav__"), ("val", "data/general/eval_uavid_tiles/images", "data/general/eval_uavid_tiles/seg", "")):
    for f in sorted(x for x in allf if x.startswith(f"images/{split}/")):
        sid = os.path.basename(f)[:-4]
        if os.path.exists(f"{lo_}/{pre}{sid}_q3.png"): continue
        im = Image.open(fetch(rid, f, raw)).convert("RGB"); mk = Image.open(fetch(rid, f.replace("images/", "masks/"), raw))
        for (t, a), (_, b) in zip(views(im), views(mk)):
            if split == "val" and t == "": continue
            crop43(a, Image.BICUBIC).save(f"{io_}/{sid}{t}.jpg", quality=95); Image.fromarray(lut[np.asarray(crop43(b, Image.NEAREST))]).save(f"{lo_}/{pre}{sid}{t}.png")
    print("uavid", split, len(os.listdir(io_)), flush=True)
rid, raw = "WilliamBonilla62/RUGD", "data/general/raw_rugd"; HELD = ("creek", "park-2", "trail-4", "village"); cmap = {}
for ln in open(fetch(rid, "RUGD_annotation-colormap.txt", raw)):
    v = ln.split()
    if len(v) == 5: cmap[(int(v[2]), int(v[3]), int(v[4]))] = v[1]
M = {**{k: "ground_road" for k in ("dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "concrete")}, "tree": "tree_pole", "pole": "tree_pole", "vehicle": "car", "building": "building", "bridge": "building", "person": "person", "rock": "rock",
     **{k: "other_obstacle" for k in ("container/generic-object", "log", "bicycle", "fence", "bush", "sign", "picnic-table")}}
for f in sorted(os.listdir("data/general/eval_rugd/images")):
    if f.rsplit("_", 1)[0] in HELD:
        for sub, ext in (("images", ".jpg"), ("seg", ".png")):
            d = f"data/general/eval_rugd_heldout/{sub}/{f[:-4]}{ext}"
            if not os.path.exists(d): os.symlink(os.path.abspath(f"data/general/eval_rugd/{sub}/{f[:-4]}{ext}"), d)
print("rugd held-out eval frames", len(os.listdir("data/general/eval_rugd_heldout/images")), flush=True)
fs = [f for f in sorted(x for x in api.list_repo_files(rid, repo_type="dataset") if x.startswith("RUGD_frames-with-annotations/")) if f.split("/")[1] not in HELD][::8]
for f in fs:
    sid = os.path.basename(f)[:-4]
    if os.path.exists(f"{TR}/labels_v2/seg/rugd__{sid}.png"): continue
    a = np.asarray(crop43(Image.open(fetch(rid, f.replace("RUGD_frames-with-annotations", "RUGD_annotations"), raw)).convert("RGB"), Image.NEAREST)).astype(np.int32); out = np.full(a.shape[:2], 255, np.uint8)
    for (r, g, b), n in cmap.items():
        if n in M: out[(a[..., 0] == r) & (a[..., 1] == g) & (a[..., 2] == b)] = ix[M[n]]
    crop43(Image.open(fetch(rid, f, raw)).convert("RGB"), Image.BICUBIC).save(f"{TR}/images/rugd/{sid}.jpg", quality=95); Image.fromarray(out).save(f"{TR}/labels_v2/seg/rugd__{sid}.png")
print("rugd train", len(os.listdir(f"{TR}/images/rugd")), flush=True)
