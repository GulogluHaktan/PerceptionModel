"""Second round of unseen environments (evaluation only): IDD-AW (HF Furqan7007/IDDAW_OFFICIAL, IDD-AW research licence): Indian roads in rain, fog, low light and snow, dense labels (26 classes).
Every 2nd validation frame (about 238) -> data/general/eval_iddaw {images, seg}. Map: road, drivable fallback, sidewalk, non drivable fallback, curb -> ground_road; person, rider -> person;
autorickshaw, car, truck, bus, vehicle fallback -> car; wall, building, bridge -> building; pole, traffic light -> tree_pole; motorcycle, bicycle, fence, guard rail, billboard, traffic sign, obs-str-bar-fallback -> other_obstacle;
vegetation, sky -> 255 (ignored, as in the BDD mapping). The mirror has no per-drive weather tag, so the score is over all conditions together. Images centre-cropped to 4:3, 640x480."""
import os, time
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}; rid, raw, ev = "Furqan7007/IDDAW_OFFICIAL", "data/general/raw_iddaw", "data/general/eval_iddaw"
for s in ("images", "seg"): os.makedirs(f"{ev}/{s}", exist_ok=True)
def fetch(rel):
    for a in range(60):
        try: return hf_hub_download(rid, rel, repo_type="dataset", local_dir=raw)
        except Exception as e:
            print("retry", a, str(e)[:90], flush=True); time.sleep(45)
    raise SystemExit("download failed")
def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)
lut = np.full(256, 255, np.uint8)
for ids, c in (((0, 1, 2, 3, 13), "ground_road"), ((4, 5), "person"), ((8, 9, 10, 11, 12), "car"), ((14, 22, 23), "building"), ((19, 20), "tree_pole"), ((6, 7, 15, 16, 17, 18, 21), "other_obstacle")):
    for k in ids: lut[k] = ix[c]
fs = sorted(f for f in HfApi().list_repo_files(rid, repo_type="dataset") if f.startswith("val/rgb/"))[::2]
for f in fs:
    sid = f.split("/")[2] + "_" + os.path.basename(f)[:8]
    if os.path.exists(f"{ev}/seg/{sid}.png"): continue
    crop43(Image.open(fetch(f)).convert("RGB"), Image.BICUBIC).save(f"{ev}/images/{sid}.jpg", quality=95)
    Image.fromarray(lut[np.asarray(crop43(Image.open(fetch(f.replace("/rgb/", "/gt_labels/").replace("_rgb.png", "_labellevel3Ids.png"))), Image.NEAREST))]).save(f"{ev}/seg/{sid}.png")
print("iddaw", len(os.listdir(f"{ev}/seg")))
