"""Real condition data for the self-improvement loop's REAL acceptance step: all 1000 BDD100K val frames (none is in training;
bdd_prepare.py put every 3rd in eval_bdd), same crop / class map / lane-paint ignore as bdd_prepare.py + label set v3 (sky -> 0, vegetation -> tree_pole), split by mean luminance:
  dark  (< DARK)   -> data/general/real_bdd/dark_{val,test}/{images,seg}
  bright(> BRIGHT) -> data/general/real_bdd/bright_{val,test}/{images,seg}
val / test alternate by sorted index (val = acceptance decisions, test = reported numbers only).
Run: .venv/bin/python scripts/make_real_dark_split.py"""
import os
import numpy as np, cv2
from PIL import Image
RAW = "data/general/raw_bdd"; OUT = "data/general/real_bdd"; DARK = float(os.environ.get("DARK", "0.25")); BRIGHT = float(os.environ.get("BRIGHT", "0.35"))
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}
lut = np.full(256, 255, np.uint8)
for k, c in {0: "ground_road", 1: "ground_road", 9: "ground_road", 2: "building", 3: "building", 5: "tree_pole", 6: "tree_pole", 11: "person", 12: "person", 13: "car", 14: "car", 15: "car",
             4: "other_obstacle", 7: "other_obstacle", 16: "other_obstacle", 17: "other_obstacle", 18: "other_obstacle"}.items(): lut[k] = ix[c]
idir = f"{RAW}/data/images_10k/10k/val/shard_000"; ldir = f"{RAW}/data/seg_maps/labels/val/shard_000"
cnt = {}; k = {"dark": 0, "bright": 0}
for f in sorted(x for x in os.listdir(idir) if x.endswith(".jpg")):
    sid = f[:-4]; mk = f"{ldir}/{sid}_train_id.png"
    if not os.path.exists(mk): continue
    im = Image.open(f"{idir}/{f}").convert("RGB"); W, H = im.size; cw = H * 4 // 3; x0 = (W - cw) // 2
    im = im.crop((x0, 0, x0 + cw, H)).resize((640, 480), Image.BICUBIC); y = np.asarray(im.convert("L")).mean() / 255
    kind = "dark" if y < DARK else ("bright" if y > BRIGHT else None)
    if kind is None: continue
    raw = np.asarray(Image.open(mk).crop((x0, 0, x0 + cw, H)).resize((640, 480), Image.NEAREST)); l = lut[raw].copy()
    l[raw == 10] = 0; l[raw == 8] = 2       # label set v3 (sky_labels_v3.py): sky -> background, vegetation -> tree_pole (what the model is trained on)
    g = np.asarray(im.convert("L")).astype(np.float32); th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))); road = raw == 0
    if road.sum() > 500: l[road & (th > max(25.0, 0.25 * float(np.median(g[road]))))] = 255
    sp = f"{kind}_{'val' if k[kind] % 2 == 0 else 'test'}"; k[kind] += 1
    os.makedirs(f"{OUT}/{sp}/images", exist_ok=True); os.makedirs(f"{OUT}/{sp}/seg", exist_ok=True)
    im.save(f"{OUT}/{sp}/images/{sid}.jpg", quality=95); Image.fromarray(l).save(f"{OUT}/{sp}/seg/{sid}.png"); cnt[sp] = cnt.get(sp, 0) + 1
print(cnt)
