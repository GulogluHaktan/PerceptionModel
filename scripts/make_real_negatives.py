"""Real gate-free negatives from real TII frames: random 4:3 crops (35-65% of the width) that do not touch any labelled gate (corner extent expanded 40% each side), resized to 640x480.
Train negatives from training-flight frames -> data/train_real/images/real_tii_neg + empty corners_full json. Eval negatives from the golden frames (other flights) -> data/real_tii/neg_crops.
Usage: python scripts/make_real_negatives.py"""
import glob, json, os, random
from PIL import Image
rng = random.Random(2)
def crop_neg(img, boxes, tries=80):
    W, H = img.size
    for _ in range(tries):
        w = int(W * rng.uniform(0.35, 0.65)); h = int(w * 3 / 4); x0 = rng.randint(0, W - w); y0 = rng.randint(0, H - h)
        if all(x0 + w <= bx0 or x0 >= bx1 or y0 + h <= by0 or y0 >= by1 for bx0, by0, bx1, by1 in boxes): return img.crop((x0, y0, x0 + w, y0 + h)).resize((640, 480), Image.BICUBIC)
def ext(pts, W, H, e=0.4):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]; w, h = max(xs) - min(xs), max(ys) - min(ys); return (min(xs) - e * w, min(ys) - e * h, max(xs) + e * w, max(ys) + e * h)
os.makedirs("data/train_real/images/real_tii_neg", exist_ok=True); os.makedirs("data/real_tii/neg_crops", exist_ok=True); n = 0
for f in sorted(glob.glob("data/train_real/labels/corners_full/real_tii__*.json")):
    sid = os.path.basename(f)[len("real_tii__"):-5]; img = Image.open(f"data/train_real/images/real_tii/{sid}.jpg").convert("RGB"); W, H = img.size
    boxes = [ext([(x / 256 * W, y / 192 * H) for x, y in r["corners_px"]], W, H) for r in json.load(open(f))]; c = crop_neg(img, boxes)
    if c is None: continue
    c.save(f"data/train_real/images/real_tii_neg/{sid}.jpg", quality=95); json.dump([], open(f"data/train_real/labels/corners_full/real_tii_neg__{sid}.json", "w")); n += 1
print("train negatives", n); m = 0
for cand in json.load(open("data/real_tii/golden/candidates.json")):
    img = Image.open(f"data/real_tii/golden/images/{cand['id']}.jpg").convert("RGB"); W, H = img.size; boxes = []
    for g in cand["gates"]:
        cx, cy, w, h = g["box"]; boxes.append(ext([((cx - w / 2) * W, (cy - h / 2) * H), ((cx + w / 2) * W, (cy + h / 2) * H)], W, H, 0.15))
    for _ in range(2):
        c = crop_neg(img, boxes)
        if c is not None: c.save(f"data/real_tii/neg_crops/{cand['id']}_{m}.jpg", quality=95); m += 1
print("eval negatives (golden flights)", m)
