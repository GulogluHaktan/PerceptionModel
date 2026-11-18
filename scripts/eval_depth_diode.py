"""Held-out real depth test on DIODE shard 9 (300 frames, indoor+outdoor): per-image least-squares scale+shift alignment of the predicted depth to the label on valid pixels, then AbsRel (mean |pred-gt|/gt) for the valid pixels. Prints indoor / outdoor / all. Usage: CKPT=runs/x KIND=vit python scripts/eval_depth_diode.py"""
import json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.evaluate import absrel_both
EV = "data/general/eval_diode"; CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); ind = json.load(open(f"{EV}/indoor.json"))
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval(); res = {"indoor": [], "outdoor": []}
for sid, isin in ind.items():
    x = torch.from_numpy(np.asarray(Image.open(f"{EV}/images/{sid}.jpg").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): d = m(normalize(x.cuda()), with_teacher_heads=False)["depth"][0, 0].float().cpu().numpy()
    gt = np.load(f"{EV}/depth/{sid}.npy").astype(np.float32); gt = np.asarray(Image.fromarray(gt).resize((448, 448), Image.NEAREST)); v = gt > 0
    if v.sum() < 500: continue
    res["indoor" if isin else "outdoor"].append(absrel_both(d, gt, v))
a = {k: np.array(v) for k, v in res.items()}; al = np.concatenate([a["indoor"], a["outdoor"]])
print(f"{CK} {K} DIODE-held-out AbsRel: indoor {a['indoor'][:, 0].mean():.3f} (n={len(a['indoor'])}) outdoor {a['outdoor'][:, 0].mean():.3f} (n={len(a['outdoor'])}) all {al[:, 0].mean():.3f} | fit in inverse depth: indoor {a['indoor'][:, 1].mean():.3f} outdoor {a['outdoor'][:, 1].mean():.3f} all {al[:, 1].mean():.3f}")
