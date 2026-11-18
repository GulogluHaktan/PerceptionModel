"""What do the students output on REAL frames? Rows: image | seg (gate green) | depth | for sim_v2 (sim only) and real_mix_f10 (300 real). Writes runs/eval/vis/real_outputs.png"""
import json, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
G = "data/real_tii/golden"; c = json.load(open(f"{G}/candidates.json")); ids = [c[i]["id"] for i in (30, 45, 60, 75, 90, 100)]
pal = np.array([[90, 90, 90]] * len(CLASSES), np.uint8); pal[CLASSES.index("gate")] = (40, 230, 80)
for k, cl in enumerate(CLASSES):
    if cl != "gate": pal[k] = np.array([(31,119,180),(255,127,14),(148,103,189),(140,86,75),(227,119,194),(188,189,34),(23,190,207),(214,39,40),(127,127,127),(174,199,232)][k % 10], np.uint8)
def run(ck):
    m = PerceptionModel("vit", teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{ck}/vit/final.pt", map_location="cpu")); m = m.cuda().eval(); out = []
    for i in ids:
        x = torch.from_numpy(np.asarray(Image.open(f"{G}/images/{i}.jpg").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): o = m(normalize(x.cuda()), with_teacher_heads=False)
        sg = pal[o["seg"][0].argmax(0).cpu().numpy()] if o["seg"].dim() == 4 and o["seg"].shape[1] == len(CLASSES) else None
        d = o["depth"][0].float().squeeze().cpu().numpy(); d = (d - d.min()) / (d.max() - d.min() + 1e-6); out.append((Image.fromarray(sg).resize((224, 168)), Image.fromarray(np.stack([255 * d, 255 * d ** 2, 255 * (1 - d)], -1).astype(np.uint8)).resize((224, 168)), float(o["slots"][0, :, 0].sigmoid().max())))
    return out
A, B = run("runs/sim_v2"), run("runs/real_mix_f10"); sh = Image.new("RGB", (224 * 5, 168 * len(ids)))
for r, i in enumerate(ids):
    sh.paste(Image.open(f"{G}/images/{i}.jpg").convert("RGB").resize((224, 168)), (0, r * 168))
    sh.paste(A[r][0], (224, r * 168)); sh.paste(A[r][1], (448, r * 168)); sh.paste(B[r][0], (672, r * 168)); sh.paste(B[r][1], (896, r * 168))
sh.save("runs/eval/vis/real_outputs.png"); print([round(a[2], 2) for a in A], [round(b[2], 2) for b in B])
