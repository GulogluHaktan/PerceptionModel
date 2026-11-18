"""Fair picture of gate detection: evenly spaced whole frames (no picking) from the never-seen place (MonoRaceGate red), label corners as white dots, predictions (score > 0.5) as green outlines.
Usage: PFM_GDET_K=8 PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid python scripts/vis_fair_gate_sheet.py out.jpg [n=12]"""
import json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
G = "data/real_mrg/golden"; c = [x for x in json.load(open(f"{G}/candidates.json")) if x["seq"].startswith("red")]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
sel = [c[int(i)] for i in np.linspace(0, len(c) - 1, n)]; CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit")
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
TW, TH = 640, 480; sheet = Image.new("RGB", (3 * (TW + 6), ((n + 2) // 3) * (TH + 6)), "white")
for i, x in enumerate(sel):
    im = Image.open(f"{G}/images/{x['id']}.jpg").convert("RGB"); t = torch.from_numpy(np.asarray(im.resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): s = m(normalize(t.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
    im = im.resize((TW, TH), Image.BICUBIC); d = ImageDraw.Draw(im)
    for q in range(s.shape[0]):
        if s[q, 0].sigmoid() > 0.5: p = s[q, 1:].view(4, 2).numpy() * [TW, TH]; d.line([tuple(a) for a in p] + [tuple(p[0])], fill=(0, 255, 0), width=3)
    for g in x.get("gates", []):
        for a in g.get("corners", []):
            if a is not None and (len(a) < 3 or a[2] > 0): d.ellipse((a[0] * TW - 4, a[1] * TH - 4, a[0] * TW + 4, a[1] * TH + 4), outline="white", width=2)
    sheet.paste(im, ((i % 3) * (TW + 6), (i // 3) * (TH + 6)))
sheet.save(sys.argv[1], quality=88); print("saved", sys.argv[1])
