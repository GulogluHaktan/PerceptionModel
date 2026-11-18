"""Side-by-side on held-out real frames (MonoRaceGate place 'red', never trained on): left bigB (MLP slot head), right bigC (dense head). White = label, coloured = prediction (score > 0.5). Writes runs/eval/vis/real_gates_compare.jpg"""
import json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
G = "data/real_mrg/golden"; c = [x for x in json.load(open(f"{G}/candidates.json")) if x["seq"].startswith("red")]; sel = c[5::33][:6]
code = r'''
import json, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
ck, ids = sys.argv[1], sys.argv[2].split(",")
m = PerceptionModel("vit", teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{ck}/vit/final.pt", map_location="cpu")); m = m.cuda().eval(); out = {}
for i in ids:
    x = torch.from_numpy(np.asarray(Image.open(f"data/real_mrg/golden/images/{i}.jpg").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): s = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
    out[i] = [[float(s[q, 0].sigmoid())] + s[q, 1:].tolist() for q in range(4) if s[q, 0].sigmoid() > 0.5]
print("JSON" + json.dumps(out))
'''
def run(ck, env):
    r = subprocess.run([".venv/bin/python", "-c", code, ck, ",".join(x["id"] for x in sel)], capture_output=True, text=True, env={**os.environ, **env}); return json.loads([l for l in r.stdout.splitlines() if l.startswith("JSON")][0][4:])
A = run("runs/bigB", {"PFM_GATE_HEAD": ""}); B = run("runs/bigC", {"PFM_GATE_HEAD": "dense"}); font = ImageFont.load_default(size=16)
def panel(x, preds, title, col):
    im = Image.open(f"{G}/images/{x['id']}.jpg").convert("RGB").resize((480, 270)); d = ImageDraw.Draw(im); W, H = im.size
    for g in x["gates"]:
        p = [(a * W, b * H) for a, b, _ in g["corners"]]; d.line(p + [p[0]], fill=(255, 255, 255), width=3)
    for pr in preds:
        p = [(pr[1 + 2 * k] * W, pr[2 + 2 * k] * H) for k in range(4)]; d.line(p + [p[0]], fill=col, width=2)
    d.rectangle([0, 0, 250, 22], fill=(0, 0, 0)); d.text((4, 2), title, fill=(255, 255, 255), font=font); return im
sh = Image.new("RGB", (966, len(sel) * 274), (20, 20, 20))
for r, x in enumerate(sel):
    sh.paste(panel(x, A[x["id"]], "ONCE (bigB)  beyaz=etiket", (255, 60, 40)), (0, r * 274)); sh.paste(panel(x, B[x["id"]], "SONRA (bigC) yesil=tahmin", (40, 255, 80)), (486, r * 274))
os.makedirs("runs/eval/vis", exist_ok=True); sh.save("runs/eval/vis/real_gates_compare.jpg", quality=90); print("saved", sh.size)
