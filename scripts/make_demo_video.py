"""Demo video: the student's three outputs on a sequence of frames. Per frame three panels: camera image + gates (score > THR, green), class map blended on the image, relative depth.
Usage: [DEVICE=cpu|cuda] [THR=0.5] PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/bigD KIND=vit python scripts/make_demo_video.py <out.mp4> <fps> <title> <glob or dir> [step=1] [max_frames]
Frames are sorted by name. H.264 through imageio-ffmpeg. The speed of the video is the playback fps given here, NOT the model speed (see scripts/bench_fps.py)."""
import glob, os, sys
import numpy as np, torch, imageio
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
out, fps, title, src = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]; step = int(sys.argv[5]) if len(sys.argv) > 5 else 1; mx = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); dev = os.environ.get("DEVICE", "cuda"); THR = float(os.environ.get("THR", "0.5")); PW, PH = 480, 360
files = sorted(glob.glob(src if "*" in src else f"{src}/*"), key=lambda p: [int(t) if t.isdigit() else t for t in __import__("re").split(r"(\d+)", os.path.basename(p))])[::step][:mx]
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.to(dev).eval()
COL = {"background": (0, 0, 0), "gate": (255, 0, 255), "tree_pole": (0, 140, 0), "person": (255, 60, 60), "ground_road": (120, 120, 120), "building": (70, 110, 200), "opponent_drone": (255, 255, 0), "car": (0, 220, 220), "rock": (150, 90, 40), "lane_marking": (255, 255, 255), "other_obstacle": (255, 150, 0)}
TR = {"gate": "kapı", "tree_pole": "ağaç/direk", "person": "insan", "ground_road": "zemin/yol", "building": "bina/duvar", "opponent_drone": "drone", "car": "araba", "rock": "kaya", "lane_marking": "şerit", "other_obstacle": "engel"}
lut = np.array([COL[c] for c in CLASSES], np.uint8); cmap = colormaps["turbo"]; font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15); fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=None, pixelformat="yuv420p")
with torch.no_grad():
    for i in range(0, len(files), 8):
        ims = [Image.open(f).convert("RGB") for f in files[i:i + 8]]
        x = torch.stack([torch.from_numpy(np.asarray(im.resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for im in ims]).to(dev)
        o = m(normalize(x), with_teacher_heads=False); slots = o["slots"].float().cpu(); seg = o["seg"].argmax(1).cpu().numpy(); dep = o["depth"][:, 0].float().cpu().numpy()
        for j, im in enumerate(ims):
            a = im.resize((PW, PH), Image.BICUBIC); d = ImageDraw.Draw(a); n = 0
            for q in range(slots.shape[1]):
                sc = float(slots[j, q, 0].sigmoid())
                if sc <= THR: continue
                p = [(float(slots[j, q, 1 + 2 * k]) * PW, float(slots[j, q, 2 + 2 * k]) * PH) for k in range(4)]; d.line(p + [p[0]], fill=(0, 255, 0), width=3); d.text((min(max(p[0][0], 2), PW - 40), min(max(p[0][1] - 16, 2), PH - 16)), f"{sc:.2f}", fill=(0, 255, 0), font=fs); n += 1
            sc_ = np.asarray(Image.fromarray(lut[seg[j]]).resize((PW, PH), Image.NEAREST)).astype(np.float32); base = np.asarray(im.resize((PW, PH), Image.BICUBIC)).astype(np.float32)
            b = Image.fromarray((0.4 * base + 0.6 * sc_).astype(np.uint8)); lo, hi = np.percentile(dep[j], [2, 98]); dn = np.clip((dep[j] - lo) / max(hi - lo, 1e-6), 0, 1)
            c = Image.fromarray((cmap(dn)[..., :3] * 255).astype(np.uint8)).resize((PW, PH), Image.BILINEAR)
            fr = Image.new("RGB", (3 * PW, PH + 74), (16, 16, 16)); [fr.paste(t, (k * PW, 48)) for k, t in enumerate((a, b, c))]; g = ImageDraw.Draw(fr)
            g.text((6, 4), title, fill=(255, 255, 255), font=font); g.text((6, 26), f"kapı tespiti (yeşil, puan > {THR})  |  kare {i + j + 1}/{len(files)}  |  kapı: {n}", fill=(200, 200, 200), font=fs); g.text((PW + 6, 26), "sınıf haritası", fill=(200, 200, 200), font=fs); g.text((2 * PW + 6, 26), "göreli derinlik (kırmızı = yakın)", fill=(200, 200, 200), font=fs)
            xx = 6
            for cn in CLASSES[1:]:
                g.rectangle([xx, PH + 54, xx + 12, PH + 66], fill=COL[cn]); g.text((xx + 16, PH + 52), TR[cn], fill=(230, 230, 230), font=fs); xx += 28 + int(g.textlength(TR[cn], font=fs))
            w.append_data(np.asarray(fr))
        print(i + len(ims), "/", len(files), flush=True)
w.close(); print("wrote", out, os.path.getsize(out) // 1024, "kB")
