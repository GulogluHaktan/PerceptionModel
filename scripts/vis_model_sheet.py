"""One picture of what a trained student outputs on held-out real images of different kinds: per row camera image + gates (green, score > 0.5), class map blended on the image, relative depth (red = near).
Rows: unseen race place (MonoRaceGate red), TII flight 13 (hall seen, flight not), BDD100K val road scene, TuSimple test highway, SUN RGB-D indoor, ADE20K outdoor, Aeroscapes drone view, DIODE outdoor. None of these frames was trained on.
Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/bigE KIND=vit python scripts/vis_model_sheet.py -> runs/eval/vis/model_sheet_<ckpt>_<kind>.jpg"""
import glob, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); PW, PH = 400, 300
def pick(pat, i): fs = sorted(glob.glob(pat)); return fs[min(i, len(fs) - 1)] if fs else None
ROWS = [("Yarış kapısı: hiç görülmemiş yer (red)", pick("data/real_mrg/golden/images/red_s1_f1_*.jpg", 38)), ("Yarış kapısı: red, yakın geçiş", pick("data/real_mrg/golden/images/red_s1_f3_*.jpg", 12)),
        ("Yarış salonu: TII uçuş 13 (bu uçuş eğitimde yok)", pick("runs/eval/video/tii13_tur/*.jpg", 630)), ("Şehir içi yol (BDD100K, test)", pick("data/general/eval_bdd/images/*.jpg", 40)),
        ("Otoyol (TuSimple, test)", pick("data/general/eval_tus/images/*.jpg", 60)), ("İç mekân (SUN RGB-D, test)", pick("data/general/eval_sun/images/*.jpg", 25)),
        ("Dış mekân (ADE20K, test)", pick("data/general/eval_ade/images/*.jpg", 140)), ("Drone bakışı (Aeroscapes, test)", pick("data/general/eval_aero/images/*.jpg", 200)), ("Dış mekân (DIODE, test)", pick("data/general/eval_diode/images/*.jpg", 150))]
if os.environ.get("SHEET") == "env":      # environments no training run has seen at all (scripts/env_prepare.py)
    ROWS = [("Yüksekten drone bakışı, şehir (UAVid)", pick("data/general/eval_uavid/images/*.jpg", 10)), ("Yüksekten drone bakışı, şehir (UAVid)", pick("data/general/eval_uavid/images/*.jpg", 60)), ("Yüksekten drone bakışı, şehir (UAVid)", pick("data/general/eval_uavid/images/*.jpg", 100)),
            ("Arazi / orman yolu (RUGD)", pick("data/general/eval_rugd/images/*.jpg", 20)), ("Arazi / dere yatağı (RUGD)", pick("data/general/eval_rugd/images/*.jpg", 150)), ("Park / köy (RUGD)", pick("data/general/eval_rugd/images/*.jpg", 290)),
            ("Araç kamerası, yol (KITTI)", pick("data/general/eval_kitti/images/*.jpg", 30)), ("Araç kamerası, yol (KITTI)", pick("data/general/eval_kitti/images/*.jpg", 160)), ("Araç kamerası, yol (KITTI)", pick("data/general/eval_kitti/images/*.jpg", 280))]
if os.environ.get("SHEET") == "env2":     # adverse-weather roads never trained on (scripts/env2_prepare.py)
    ROWS = [("Kötü havada yol (IDD-AW): yağmur / sis / loş ışık / kar", pick("data/general/eval_iddaw/images/*.jpg", i)) for i in (5, 35, 65, 95, 125, 155, 185, 215, 235)]
ROWS = [r for r in ROWS if r[1]]
COL = {"background": (0, 0, 0), "gate": (255, 0, 255), "tree_pole": (0, 140, 0), "person": (255, 60, 60), "ground_road": (120, 120, 120), "building": (70, 110, 200), "opponent_drone": (255, 255, 0), "car": (0, 220, 220), "rock": (150, 90, 40), "lane_marking": (255, 255, 255), "other_obstacle": (255, 150, 0)}
TR = {"background": "arka plan", "gate": "kapı", "tree_pole": "ağaç/direk", "person": "insan", "ground_road": "zemin/yol", "building": "bina/duvar", "opponent_drone": "drone", "car": "araba", "rock": "kaya", "lane_marking": "şerit", "other_obstacle": "engel"}
lut = np.array([COL[c] for c in CLASSES], np.uint8); cmap = colormaps["turbo"]; F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; font = ImageFont.truetype(F, 15); fs_ = ImageFont.truetype(F, 13)
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
sheet = Image.new("RGB", (3 * PW + 8, len(ROWS) * (PH + 24) + 64), (18, 18, 18)); g = ImageDraw.Draw(sheet)
g.text((6, 6), f"{os.path.basename(CK)} ({K}) modelinin {'hiç görmediği ortamlardaki' if os.environ.get('SHEET') else 'eğitimde görmediği gerçek görüntülerdeki'} çıktıları", fill=(255, 255, 255), font=font)
for k, t in enumerate(("kamera görüntüsü + kapı tespiti (yeşil)", "sınıf haritası", "göreli derinlik (kırmızı = yakın)")): g.text((k * (PW + 4) + 6, 26), t, fill=(200, 200, 200), font=fs_)
for r, (title, f) in enumerate(ROWS):
    im = Image.open(f).convert("RGB"); x = torch.from_numpy(np.asarray(im.resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): o = m(normalize(x.cuda()), with_teacher_heads=False)
    s = o["slots"][0].float().cpu(); seg = o["seg"][0].argmax(0).cpu().numpy(); dep = o["depth"][0, 0].float().cpu().numpy(); a = im.resize((PW, PH), Image.BICUBIC); d = ImageDraw.Draw(a)
    for q in range(s.shape[0]):
        if float(s[q, 0].sigmoid()) > 0.5:
            p = [(float(s[q, 1 + 2 * j]) * PW, float(s[q, 2 + 2 * j]) * PH) for j in range(4)]; d.line(p + [p[0]], fill=(0, 255, 0), width=3)
    b = Image.fromarray((0.4 * np.asarray(im.resize((PW, PH), Image.BICUBIC)).astype(np.float32) + 0.6 * np.asarray(Image.fromarray(lut[seg]).resize((PW, PH), Image.NEAREST)).astype(np.float32)).astype(np.uint8))
    lo, hi = np.percentile(dep, [2, 98]); c = Image.fromarray((cmap(np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)).resize((PW, PH), Image.BILINEAR); y = 46 + r * (PH + 24)
    g.text((6, y + 2), title, fill=(255, 255, 255), font=fs_); [sheet.paste(t, (j * (PW + 4), y + 22)) for j, t in enumerate((a, b, c))]
xx = 6; y = sheet.height - 18
for cn in CLASSES:
    g.rectangle([xx, y, xx + 12, y + 12], fill=COL[cn], outline=(90, 90, 90)); g.text((xx + 16, y - 2), TR[cn], fill=(230, 230, 230), font=fs_); xx += 28 + int(g.textlength(TR[cn], font=fs_))
out = f"runs/eval/vis/model_sheet_{os.path.basename(CK)}_{K}{'_' + os.environ['SHEET'] if os.environ.get('SHEET') else ''}.jpg"; sheet.save(out, quality=90); print("wrote", out, sheet.size)
