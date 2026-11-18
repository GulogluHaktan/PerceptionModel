"""Three real environments that NO training run has seen, for zero-shot tests of the general heads ("different environments", Update 51). Evaluation only: nothing goes into data/general/train.
  UAVid (HF dronefreak/UAVid-2020, CC BY-NC-SA 4.0): city from a drone at height, 70 val + 50 test frames -> data/general/eval_uavid. Map: Building -> building; Road -> ground_road; Static/Moving Car -> car; Tree -> tree_pole; Human -> person;
      Clutter and (low) Vegetation -> 255 ignored (mixed content: grass, bushes, everything else).
  RUGD (HF WilliamBonilla62/RUGD; RUGD is free for research use): off-road robot, trails / creek / park / village, every 25th frame (about 300) -> data/general/eval_rugd. Map: dirt, sand, grass, asphalt, gravel, mulch, rock-bed, concrete -> ground_road;
      tree, pole -> tree_pole; vehicle -> car; building, bridge -> building; person; rock -> rock; container, log, bicycle, fence, bush, sign, picnic-table -> other_obstacle; void, water, sky -> 255.
  KITTI depth (HF dangth2004/KITTI-Depth-Estimation; KITTI: CC BY-NC-SA 3.0): road scenes with lidar depth, every 11th validation frame (about 310) -> data/general/eval_kitti (images + depth .npy in metres, 0 = no lidar return).
Images are centre-cropped to 4:3 and stored at 640x480 like the other sets (KITTI's 3.3:1 frame loses its sides; UAVid's 4K frame is shrunk 5x, so cars are about 10 px)."""
import os, time
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}; api = HfApi()
def fetch(rid, rel, raw):
    for a in range(60):
        try: return hf_hub_download(rid, rel, repo_type="dataset", local_dir=raw)
        except Exception as e:
            print("retry", a, str(e)[:90], flush=True); time.sleep(45)
    raise SystemExit("download failed")
def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)
def dirs(ev, *subs):
    for s in subs: os.makedirs(f"{ev}/{s}", exist_ok=True)
# UAVid
ev, raw, rid = "data/general/eval_uavid", "data/general/raw_uavid", "dronefreak/UAVid-2020"; dirs(ev, "images", "seg"); lut = np.full(256, 255, np.uint8)
for k, c in {1: "building", 2: "ground_road", 3: "car", 4: "tree_pole", 6: "person", 7: "car"}.items(): lut[k] = ix[c]
fs = sorted(f for f in api.list_repo_files(rid, repo_type="dataset") if f.startswith("images/val/")) + sorted(f for f in api.list_repo_files(rid, repo_type="dataset") if f.startswith("images/test/"))[::3]
for f in fs:
    sid = os.path.basename(f)[:-4]
    if os.path.exists(f"{ev}/seg/{sid}.png"): continue
    crop43(Image.open(fetch(rid, f, raw)).convert("RGB"), Image.BICUBIC).save(f"{ev}/images/{sid}.jpg", quality=95); Image.fromarray(lut[np.asarray(crop43(Image.open(fetch(rid, f.replace("images/", "masks/"), raw)), Image.NEAREST))]).save(f"{ev}/seg/{sid}.png")
print("uavid", len(os.listdir(f"{ev}/seg")), flush=True)
# RUGD
ev, raw, rid = "data/general/eval_rugd", "data/general/raw_rugd", "WilliamBonilla62/RUGD"; dirs(ev, "images", "seg")
cmap = {}; names = {}
for ln in open(fetch(rid, "RUGD_annotation-colormap.txt", raw)):
    v = ln.split()
    if len(v) == 5: cmap[(int(v[2]), int(v[3]), int(v[4]))] = v[1]
M = {**{k: "ground_road" for k in ("dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "concrete")}, "tree": "tree_pole", "pole": "tree_pole", "vehicle": "car", "building": "building", "bridge": "building", "person": "person", "rock": "rock",
     **{k: "other_obstacle" for k in ("container/generic-object", "log", "bicycle", "fence", "bush", "sign", "picnic-table")}}
fs = sorted(f for f in api.list_repo_files(rid, repo_type="dataset") if f.startswith("RUGD_frames-with-annotations/"))[::25]
for f in fs:
    sid = os.path.basename(f)[:-4]
    if os.path.exists(f"{ev}/seg/{sid}.png"): continue
    a = np.asarray(crop43(Image.open(fetch(rid, f.replace("RUGD_frames-with-annotations", "RUGD_annotations"), raw)).convert("RGB"), Image.NEAREST)).astype(np.int32); out = np.full(a.shape[:2], 255, np.uint8)
    for (r, g, b), n in cmap.items():
        if n in M: out[(a[..., 0] == r) & (a[..., 1] == g) & (a[..., 2] == b)] = ix[M[n]]
    crop43(Image.open(fetch(rid, f, raw)).convert("RGB"), Image.BICUBIC).save(f"{ev}/images/{sid}.jpg", quality=95); Image.fromarray(out).save(f"{ev}/seg/{sid}.png")
print("rugd", len(os.listdir(f"{ev}/seg")), flush=True)
# KITTI
ev, raw, rid = "data/general/eval_kitti", "data/general/raw_kitti", "dangth2004/KITTI-Depth-Estimation"; dirs(ev, "images", "depth"); allf = set(api.list_repo_files(rid, repo_type="dataset"))
fs = sorted(f for f in allf if f.startswith("data_depth_annotated/val/") and "/image_02/" in f)[::11]
for f in fs:
    drive = f.split("/")[2]; num = os.path.basename(f)[:-4]; rgb = f"raw_data/{drive[:10]}/{drive}/image_02/data/{num}.png"; sid = f"{drive}_{num}"
    if rgb not in allf or os.path.exists(f"{ev}/depth/{sid}.npy"): continue
    d = np.asarray(Image.open(fetch(rid, f, raw))).astype(np.float32) / 256.0; dm = Image.fromarray(d)
    crop43(Image.open(fetch(rid, rgb, raw)).convert("RGB"), Image.BICUBIC).save(f"{ev}/images/{sid}.jpg", quality=95); np.save(f"{ev}/depth/{sid}.npy", np.asarray(crop43(dm, Image.NEAREST)))
print("kitti", len(os.listdir(f"{ev}/depth")), flush=True)
