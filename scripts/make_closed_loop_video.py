"""Video of one closed-loop simulation flight (scripts/closed_loop_race.py with SNAP_EVERY=1 and TRAJ): left the camera frame the model saw with its predicted gates (green), right the top view with the path so far.
Usage: python scripts/make_closed_loop_video.py <track> <out.mp4> [fps=60]"""
import glob, json, sys
import numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
T, out = sys.argv[1], sys.argv[2]; fps = float(sys.argv[3]) if len(sys.argv) > 3 else 60; D = "runs/closed_loop/traj"; R = json.load(open(f"{D}/{T}_percept.json")); fs = sorted(glob.glob(f"{D}/snapall_{T}/*.jpg"))
tr = np.array(R["traj"])[:, :2]; gc = np.array([g["c"][:2] for g in R["gates"]]); allp = np.concatenate([tr, gc]); lo, hi = allp.min(0) - 6, allp.max(0) + 6; W, H, BAR = 640, 480, 44; sc = min(W / (hi[0] - lo[0]), H / (hi[1] - lo[1]))
ox, oy = (W - sc * (hi[0] - lo[0])) / 2, (H - sc * (hi[1] - lo[1])) / 2
def px(p): return (ox + (p[0] - lo[0]) * sc, H - oy - (p[1] - lo[1]) * sc)
try: F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15); FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception: F = FS = ImageFont.load_default()
base = Image.new("RGB", (W, H), (250, 250, 248)); d = ImageDraw.Draw(base)
for i, g in enumerate(R["gates"]):
    c = np.array(g["c"][:2]); a = np.array(g["a"][:2]); r = np.array([a[1], -a[0]]); d.line([px(c - r * R["outer"] / 2), px(c + r * R["outer"] / 2)], fill=(107, 70, 193), width=5); d.line([px(c - r * R["inner"] / 2), px(c + r * R["inner"] / 2)], fill=(250, 250, 248), width=5)
    d.text((px(c)[0] + 8, px(c)[1] - 8), str(i + 1), fill=(90, 90, 90), font=FS)
w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=None); n = len(fs)
for k, f in enumerate(fs):
    cam = Image.open(f).convert("RGB").resize((W, H)); top = base.copy(); dd = ImageDraw.Draw(top); j = min(int(k / max(n - 1, 1) * (len(tr) - 1)), len(tr) - 1)
    if j > 1: dd.line([px(p) for p in tr[:j + 1]], fill=(47, 111, 222), width=3)
    x, y = px(tr[j]); dd.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(217, 60, 40)); gi = int(f.rsplit("gate", 1)[1][:-4])
    fr = Image.new("RGB", (2 * W, H + BAR), (20, 20, 20)); fr.paste(cam, (0, BAR)); fr.paste(top, (W, BAR)); t = ImageDraw.Draw(fr)
    t.text((10, 5), "Kapalı döngü SİMÜLASYON (gerçek uçuş değil) · sürüm 1 küçük model (bigL_h) · 12 m/s", fill=(255, 255, 255), font=F)
    t.text((10, 24), f"sol: modelin gördüğü kare, yeşil = bulduğu kapılar · sağ: kuşbakışı yol · geçilen kapı {gi}/{len(R['gates'])}", fill=(200, 200, 200), font=FS)
    w.append_data(np.asarray(fr))
w.close(); print("wrote", out, n, "frames")
