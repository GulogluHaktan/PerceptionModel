"""lane_marking on REAL highway frames: 233 TuSimple test frames (never trained on; labels from scripts/tusimple_prepare.py: 9 = lane paint on a labelled lane line, 253 = surely not lane paint, 255 = unknown).
Reports recall (labelled lane-paint pixels predicted lane_marking), precision (predicted lane_marking pixels on scored pixels that are labelled lane paint) and IoU over the scored pixels.
Label noise: line labels also cross occluding cars; unlabelled paint is 'unknown'. Usage: CKPT=runs/x KIND=vit python scripts/eval_lane_tusimple.py"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); EV = "data/general/eval_tus"; li = CLASSES.index("lane_marking"); tp = fp = fn = 0
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval(); files = sorted(glob.glob(f"{EV}/images/*.jpg"))
for p in files:
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): pr = m(normalize(x.cuda()), with_teacher_heads=False)["seg"][0].argmax(0).cpu().numpy() == li
    g = np.asarray(Image.open(f"{EV}/seg/{os.path.basename(p)[:-4]}.png").resize((448, 448), Image.NEAREST)); pos = g == 9; neg = g == 253
    tp += (pr & pos).sum(); fn += (~pr & pos).sum(); fp += (pr & neg).sum()
res = {"frames": len(files), "recall": round(float(tp / max(tp + fn, 1)), 3), "precision": round(float(tp / max(tp + fp, 1)), 3), "iou": round(float(tp / max(tp + fp + fn, 1)), 3)}
print(CK, K, "TuSimple test lane_marking:", json.dumps(res)); json.dump(res, open(f"runs/eval/lane_tusimple_{os.path.basename(CK)}_{K}.json", "w"))
