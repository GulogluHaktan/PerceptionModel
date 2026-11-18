"""MonoRaceGate frames as training data: images -> data/train_mrg/images/real_mrg/<id>.jpg, labels/corners_full/real_mrg__<id>.json (corners_px at 256x192, all four corners known even when outside the image), plus gate-free 4:3 crops (labels []) as real_mrg_neg. Ids start with the sequence name, so a split by place is a filter on the id."""
import json, os, random
from PIL import Image
G = "data/real_mrg/golden"; O = "data/train_mrg"; rng = random.Random(3)
for sub in ("images/real_mrg", "images/real_mrg_neg", "labels/corners_full"): os.makedirs(f"{O}/{sub}", exist_ok=True)
def ext(pts, e=0.4):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]; w, h = max(xs) - min(xs), max(ys) - min(ys); return (min(xs) - e * w, min(ys) - e * h, max(xs) + e * w, max(ys) + e * h)
n = k = 0
for c in json.load(open(f"{G}/candidates.json")):
    img = Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB"); W, H = img.size
    rows = [{"gate": i, "corners_px": [[round(x * 256, 2), round(y * 192, 2)] for x, y, _ in g["corners"]], "visible": [v > 0 for _, _, v in g["corners"]], "n_visible": sum(v > 0 for _, _, v in g["corners"]), "n_visible_depth": 4, "depths": [5.0] * 4} for i, g in enumerate(c["gates"])]
    img.save(f"{O}/images/real_mrg/{c['id']}.jpg", quality=95); json.dump(rows, open(f"{O}/labels/corners_full/real_mrg__{c['id']}.json", "w")); n += 1
    boxes = [ext([(x * W, y * H) for x, y, _ in g["corners"]]) for g in c["gates"]]
    for _ in range(80):
        w = int(W * rng.uniform(0.35, 0.65)); h = int(w * 3 / 4)
        if h >= H: continue
        x0 = rng.randint(0, W - w); y0 = rng.randint(0, H - h)
        if all(x0 + w <= a or x0 >= b or y0 + h <= c0 or y0 >= d for a, c0, b, d in boxes):
            img.crop((x0, y0, x0 + w, y0 + h)).resize((640, 480), Image.BICUBIC).save(f"{O}/images/real_mrg_neg/{c['id']}.jpg", quality=95); json.dump([], open(f"{O}/labels/corners_full/real_mrg_neg__{c['id']}.json", "w")); k += 1; break
print("positives", n, "negative crops", k)
