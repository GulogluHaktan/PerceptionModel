"""BDD100K semantic segmentation subset (HF dronefreak/BDD100K-Semantic-Segmentation, unofficial mirror; BDD100K data licence: research / non-commercial use; 7,000 train + 1,000 val dashcam frames,
19 Cityscapes-style train ids) -> our classes. Map: road, sidewalk, terrain -> ground_road; building, wall -> building; pole, traffic light -> tree_pole; person, rider -> person;
car, truck, bus -> car; fence, traffic sign, train, motorcycle, bicycle -> other_obstacle; vegetation (trees and bushes mixed), sky -> 255 (ignored).
BDD labels painted lane markings as road. Our class list has lane_marking (sim labels only), so road pixels that are clearly brighter than the road around them
(white top-hat on the grey image) are set to 255 = ignored instead of ground_road: the model is then not taught that real lane paint is plain road. No lane_marking label is invented.
Images are centre-cropped from 16:9 to 4:3 and stored at 640x480. Train -> data/general/train/images/bdd + labels/seg/bdd__<id>.png ; val: every 3rd (334) -> data/general/eval_bdd/{images,seg}."""
import os, time
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import snapshot_download
RAW = "data/general/raw_bdd"
for attempt in range(300):       # anonymous bulk download of many small files gets rate-limited: retry and resume
    try:
        snapshot_download("dronefreak/BDD100K-Semantic-Segmentation", repo_type="dataset", local_dir=RAW, allow_patterns=["id2label.json", "data/images_10k/10k/train/*", "data/images_10k/10k/val/*", "data/seg_maps/labels/*"], max_workers=4)
        cnt = [len(os.listdir(f"{RAW}/data/{p}/shard_000")) if os.path.isdir(f"{RAW}/data/{p}/shard_000") else 0 for p in ("images_10k/10k/train", "seg_maps/labels/train", "images_10k/10k/val", "seg_maps/labels/val")]; print(cnt, flush=True)
        if cnt[0] >= 7000 and cnt[1] >= 7000 and cnt[2] >= 1000 and cnt[3] >= 1000: break
        time.sleep(120)
    except Exception as e:
        print("retry", attempt, str(e)[:160], flush=True); time.sleep(180)
else:
    raise SystemExit("bdd download failed")
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}
lut = np.full(256, 255, np.uint8)
for k, c in {0: "ground_road", 1: "ground_road", 9: "ground_road", 2: "building", 3: "building", 5: "tree_pole", 6: "tree_pole", 11: "person", 12: "person", 13: "car", 14: "car", 15: "car",
             4: "other_obstacle", 7: "other_obstacle", 16: "other_obstacle", 17: "other_obstacle", 18: "other_obstacle"}.items(): lut[k] = ix[c]
TR = "data/general/train"; EV = "data/general/eval_bdd"
for d in (f"{TR}/images/bdd", f"{TR}/labels/seg", f"{EV}/images", f"{EV}/seg"): os.makedirs(d, exist_ok=True)
n = m = 0; paint = []
for split in ("train", "val"):
    idir = f"{RAW}/data/images_10k/10k/{split}/shard_000"; ldir = f"{RAW}/data/seg_maps/labels/{split}/shard_000"
    for i, f in enumerate(sorted(x for x in os.listdir(idir) if x.endswith(".jpg"))):
        sid = f[:-4]; mk = f"{ldir}/{sid}_train_id.png"
        if not os.path.exists(mk) or (split == "val" and i % 3): continue
        im = Image.open(f"{idir}/{f}").convert("RGB"); W, H = im.size; cw = H * 4 // 3; x0 = (W - cw) // 2
        im = im.crop((x0, 0, x0 + cw, H)).resize((640, 480), Image.BICUBIC); raw = np.asarray(Image.open(mk).crop((x0, 0, x0 + cw, H)).resize((640, 480), Image.NEAREST)); l = lut[raw]
        g = np.asarray(im.convert("L")).astype(np.float32); th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))); road = raw == 0
        if road.sum() > 500:
            p = road & (th > max(25.0, 0.25 * float(np.median(g[road])))); l = l.copy(); l[p] = 255; paint.append(p.sum() / road.sum())
        if split == "train": im.save(f"{TR}/images/bdd/{sid}.jpg", quality=95); Image.fromarray(l).save(f"{TR}/labels/seg/bdd__{sid}.png"); n += 1
        else: im.save(f"{EV}/images/{sid}.jpg", quality=95); Image.fromarray(l).save(f"{EV}/seg/{sid}.png"); m += 1
print("bdd train", n, "eval", m, "mean share of road pixels set to ignore (bright paint):", round(float(np.mean(paint)), 4))
