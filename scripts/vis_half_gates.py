"""Picture for the half-gate test (scripts/eval_half_gates.py): the same crops of the unseen place (MonoRaceGate red) with the predictions of two models side by side.
Each tile shows the crop in the middle of a larger grey canvas so that corners outside the view can be drawn: white = true gate, green = prediction (score > 0.5).
Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 A=runs/bigH B=runs/bigJ [KIND=vit] python scripts/vis_half_gates.py -> runs/eval/vis/half_gates_<A>_<B>.jpg"""
import json, os, random, sys
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.train import half_gate_crop
A, B = os.environ["A"], os.environ["B"]; K = os.environ.get("KIND", "vit"); G = "data/real_mrg/golden"; rng = random.Random(0); S = 200; PAD = 110; items = []
def load(ck): m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{ck}/{K}/final.pt", map_location="cpu")); return m.cuda().eval()
ms = [load(A), load(B)]
for c in json.load(open(f"{G}/candidates.json")):
    if not c["id"].startswith("red_"): continue
    for g in c["gates"]:
        g_ = np.array([[x, y] for x, y, v in g["corners"]], np.float32)
        if not all(v > 0 for _, _, v in g["corners"]) or not ((g_ >= 0) & (g_ < 1)).all() or (g_.max(0) - g_.min(0)).max() < 0.12: continue
        for n_in in (0, 1):
            for _ in range(2):
                hc = half_gate_crop(g_, n_in, rng)
                if hc is not None: items.append((c["id"], g_, n_in, hc))
pick = [it for it in items if it[2] == 1][::47][:6] + [it for it in items if it[2] == 0][::13][:6]; rows = []
for cid, g, n_in, (s, x0, y0) in pick:
    im = Image.open(f"{G}/images/{cid}.jpg").convert("RGB"); e = 1 / s; cr = im.crop((round(x0 * im.width), round(y0 * im.height), round((x0 + e) * im.width), round((y0 + e) * im.height)))
    x = torch.from_numpy(np.asarray(cr.resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255; tiles = []
    for m in ms:
        with torch.no_grad(): sl = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
        cv = Image.new("RGB", (S + 2 * PAD, S * 3 // 4 + 2 * PAD), (60, 60, 60)); cv.paste(cr.resize((S, S * 3 // 4), Image.BICUBIC), (PAD, PAD)); d = ImageDraw.Draw(cv)
        def poly(q, col, w):
            pts = [(PAD + float(a) * S, PAD + float(b) * S * 3 / 4) for a, b in q]; d.line(pts + [pts[0]], fill=col, width=w)
        poly((g - [x0, y0]) * s, (255, 255, 255), 2)
        for q in range(sl.shape[0]):
            if sl[q, 0].sigmoid() > 0.5: poly(sl[q, 1:].view(4, 2).numpy(), (0, 255, 0), 3)
        d.rectangle((PAD, PAD, PAD + S, PAD + S * 3 // 4), outline=(255, 200, 0)); tiles.append(np.asarray(cv))
    rows.append(np.concatenate(tiles, 1))
half = len(rows) // 2; sheet = np.concatenate([np.concatenate(rows[:half], 0), np.concatenate(rows[half:], 0)], 1); out = f"runs/eval/vis/half_gates_{os.path.basename(A)}_{os.path.basename(B)}.jpg"
Image.fromarray(sheet).save(out, quality=88); print("wrote", out, sheet.shape[:2], "| per pair: left", A, "right", B, "| left block: 1 corner in view, right block: no corner in view")
