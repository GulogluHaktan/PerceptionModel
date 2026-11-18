"""MonoRaceGate (TU Delft, MIT) -> golden-candidate format (same as data/real_tii/golden): images + candidates.json (corners normalised, v=2 inside the image, 0 outside), overlay sheets for review.
Source: data/real_mrg/repo/data/<seq>/{img_*.png, corners.csv (name,x0,y0,..,x3,y3 = TL,TR,BR,BL inner corners, pixels)}. Writes data/real_mrg/golden/{images,candidates.json,sheets}."""
import csv, glob, json, os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
R = "data/real_mrg/repo/data"; O = "data/real_mrg/golden"; os.makedirs(f"{O}/images", exist_ok=True); os.makedirs(f"{O}/sheets", exist_ok=True); cands = []
for d in sorted(glob.glob(f"{R}/*/")):
    seq = os.path.basename(d.rstrip("/")); rows = defaultdict(list)
    for r in csv.reader(open(f"{d}/corners.csv")):
        if len(r) == 9:
            try: rows[r[0]].append([float(x) for x in r[1:]])
            except ValueError: pass
    for name, gs in sorted(rows.items()):
        p = f"{d}/{name}"
        if not os.path.exists(p): continue
        im = Image.open(p).convert("RGB"); W, H = im.size; gates = []
        for g in gs:
            cs = [(g[2 * i] / W, g[2 * i + 1] / H, 2 if 0 <= g[2 * i] < W and 0 <= g[2 * i + 1] < H else 0) for i in range(4)]
            xs, ys = [c[0] for c in cs], [c[1] for c in cs]; gates.append({"box": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)], "corners": cs})
        sid = f"{seq}_{name[4:-4]}"; im.save(f"{O}/images/{sid}.jpg", quality=95); cands.append({"id": sid, "seq": seq, "gates": gates})
json.dump(cands, open(f"{O}/candidates.json", "w"))
from collections import Counter
print(len(cands), "images", sum(len(c["gates"]) for c in cands), "gates", Counter(c["seq"] for c in cands), "partial gates", sum(any(v == 0 for _, _, v in g["corners"]) for c in cands for g in c["gates"]))
font = ImageFont.load_default(size=18)
def draw(c):
    im = Image.open(f"{O}/images/{c['id']}.jpg").convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    for g in c["gates"]:
        pts = [(x * W, y * H) for x, y, _ in g["corners"]]; d.line(pts + [pts[0]], fill=(255, 60, 40), width=max(2, W // 300))
        for i, p in enumerate(pts): d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=[(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)][i])
    d.rectangle([0, 0, 260, 26], fill=(0, 0, 0)); d.text((4, 3), c["id"], fill=(255, 255, 255), font=font); return im
step = max(len(cands) // 90, 1); sel = cands[::step][:90]
for s in range(0, len(sel), 6):
    ims = [draw(c).resize((480, 360)) for c in sel[s:s + 6]]; sh = Image.new("RGB", (960, 1080))
    for i, im in enumerate(ims): sh.paste(im, ((i % 2) * 480, (i // 2) * 360))
    sh.save(f"{O}/sheets/sheet_{s // 6 + 1:02d}.jpg", quality=88)
fs = sorted(glob.glob(f"{O}/sheets/*.jpg")); ims = [Image.open(f).convert("RGB") for f in fs]; ims[0].save(f"{O}/mrg_review_sheets.pdf", save_all=True, append_images=ims[1:], resolution=100); print(len(ims), "sheets")
