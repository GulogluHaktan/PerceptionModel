"""Pick real TII-Racing frames (CC BY 4.0, github.com/tii-racing/drone-racing-dataset) with gate labels, draw the labelled corners, make mobile contact sheets for human review.
Corner order in labels: tl, tr, br, bl; v=2 inside image, v=0 absent (stored as 0 0). Writes data/real_tii/golden/{images,sheets,candidates.json}."""
import glob, json, os, random
from PIL import Image, ImageDraw, ImageFont
R = "data/real_tii/autonomous"; O = "data/real_tii/golden"; random.seed(0)
FL = ["flight-01a-ellipse", "flight-05a-ellipse", "flight-09a-lemniscate", "flight-13a-trackRATM", "flight-16a-trackRATM", "flight-18a-trackRATM"]; PERF = {"flight-01a-ellipse": 6, "flight-05a-ellipse": 8}; PER = 22
os.makedirs(f"{O}/images", exist_ok=True); os.makedirs(f"{O}/sheets", exist_ok=True); cands = []
def parse(p):
    out = []
    for ln in open(p):
        v = ln.split()
        if len(v) < 17: continue
        v = [float(x) for x in v]; out.append({"box": v[1:5], "corners": [(v[5 + 3 * i], v[6 + 3 * i], int(v[7 + 3 * i])) for i in range(4)]})
    return out
for f in FL:
    labs = sorted(glob.glob(f"{R}/{f}/label_{f}/*.txt")); ok = [l for l in labs if parse(l)]; n = PERF.get(f, PER); step = max(len(ok) // n, 1); pick = ok[step // 2::step][:n]
    for l in pick:
        stem = os.path.basename(l)[:-4]; img = glob.glob(f"{R}/{f}/camera_{f}/{stem}.*")
        if not img: continue
        cands.append({"id": f"{f[:9]}_{stem[:5]}", "flight": f, "image": img[0], "gates": parse(l)})
font = ImageFont.load_default(size=18) if hasattr(ImageFont, "load_default") else None
def draw(c):
    im = Image.open(c["image"]).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    for g in c["gates"]:
        pts = [(x * W, y * H) for x, y, v in g["corners"]]; vis = [v for _, _, v in g["corners"]]
        for i in range(4):
            j = (i + 1) % 4
            if vis[i] and vis[j]: d.line([pts[i], pts[j]], fill=(255, 60, 40), width=3)
        for i, (p, v) in enumerate(zip(pts, vis)):
            if v: d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=[(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)][i])
    d.rectangle([0, 0, 190, 24], fill=(0, 0, 0)); d.text((4, 2), c["id"], fill=(255, 255, 255), font=font); return im
for c in cands: Image.open(c["image"]).convert("RGB").save(f"{O}/images/{c['id']}.jpg", quality=95)
json.dump(cands, open(f"{O}/candidates.json", "w"), indent=1)
per = 6
for s in range(0, len(cands), per):
    ch = cands[s:s + per]; ims = [draw(c).resize((480, 360)) for c in ch]; sh = Image.new("RGB", (960, 360 * 3), (0, 0, 0))
    for i, im in enumerate(ims): sh.paste(im, ((i % 2) * 480, (i // 2) * 360))
    sh.save(f"{O}/sheets/sheet_{s // per + 1:02d}.jpg", quality=88)
print(len(cands), "frames", (len(cands) + per - 1) // per, "sheets")
