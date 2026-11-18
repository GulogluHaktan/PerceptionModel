"""TuSimple lane benchmark (HF dotblue/TuSimple, Apache-2.0; US highway dashcam, lane lines as x positions per image row) -> real labels for our lane_marking class, the first real ones.
Only the labelled frame of each clip (20.jpg) is used. The HF mirror holds only part of the benchmark: 267 labelled train frames and 233 test frames (all test frames are held out for scoring).
Label per pixel (640x480 after a centre crop from 16:9 to 4:3): the lane lines are drawn as thin bands (width grows towards the camera); inside a band, pixels brighter than the road around them
(white top-hat) = lane_marking, the other band pixels = 255 (the label line also runs through the gaps of dashed lines); outside the bands, pixels that are not locally bright = 253 = "surely not lane_marking"
(class otherwise unknown, see train.py), locally bright pixels outside the bands = 255 (unlabelled paint, reflections). Nothing else is labelled (no road / car classes).
Train -> data/general/train/images/tus + data/general/train/labels_v2/seg/tus__<id>.png ; eval -> data/general/eval_tus/{images,seg}."""
import json, os, time
import numpy as np, cv2
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
RAW = "data/general/raw_tusimple"; TR = "data/general/train"; EV = "data/general/eval_tus"; LANE, NOT_LANE = 9, 253
for d in (f"{TR}/images/tus", f"{TR}/labels_v2/seg", f"{EV}/images", f"{EV}/seg"): os.makedirs(d, exist_ok=True)
from huggingface_hub import HfApi
have = set(HfApi().list_repo_files("dotblue/TuSimple", repo_type="dataset"))      # the mirror holds only part of the clips (267 labelled train frames, 233 test frames)
rows = [(json.loads(l), "train") for f in ("label_data_0313.json", "label_data_0531.json", "label_data_0601.json") for l in open(f"{RAW}/{f}")] + [(json.loads(l), "eval") for l in open(f"{RAW}/test_label.json")]
rows = [(r, s_) for r, s_ in rows if r["raw_file"] in have]; print("labelled frames present in the mirror:", len(rows), flush=True)
def fetch(rel):
    for a in range(40):
        try: return hf_hub_download("dotblue/TuSimple", rel, repo_type="dataset", local_dir=RAW)
        except Exception as e:
            print("retry", a, str(e)[:100], flush=True); time.sleep(60)
    raise SystemExit("download failed")
n = {"train": 0, "eval": 0}; share = []
for i, (r, split) in enumerate(rows):
    sid = r["raw_file"].replace("clips/", "").replace("/20.jpg", "").replace("/", "_"); io_ = f"{TR}/images/tus/{sid}.jpg" if split == "train" else f"{EV}/images/{sid}.jpg"; lo = f"{TR}/labels_v2/seg/tus__{sid}.png" if split == "train" else f"{EV}/seg/{sid}.png"
    if os.path.exists(io_) and os.path.exists(lo): n[split] += 1; continue
    im = Image.open(fetch(r["raw_file"])).convert("RGB"); W, H = im.size; band = Image.new("L", (W, H), 0); wide = Image.new("L", (W, H), 0); db, dw = ImageDraw.Draw(band), ImageDraw.Draw(wide)
    for lane in r["lanes"]:
        pts = [(x, y) for x, y in zip(lane, r["h_samples"]) if x >= 0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            w = 4 + 16 * max(0.0, (y0 - 250) / 470); db.line([x0, y0, x1, y1], fill=1, width=int(round(w))); dw.line([x0, y0, x1, y1], fill=1, width=int(round(3 * w)))
    g = np.asarray(im.convert("L")).astype(np.float32); th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 15))); band = np.asarray(band) > 0; wide = np.asarray(wide) > 0; bright = th > 20
    lab = np.full((H, W), 255, np.uint8); lab[~wide & (th < 8)] = NOT_LANE; lab[band & bright] = LANE
    cw = H * 4 // 3; x0 = (W - cw) // 2; im.crop((x0, 0, x0 + cw, H)).resize((640, 480), Image.BICUBIC).save(io_, quality=95); Image.fromarray(lab[:, x0:x0 + cw]).resize((640, 480), Image.NEAREST).save(lo)
    n[split] += 1; share.append((lab == LANE).mean())
    if i % 50 == 0: print(i, len(rows), n, flush=True)
print("tusimple", n, "mean lane_marking pixel share", round(float(np.mean(share)), 4) if share else "cached")
