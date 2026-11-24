"""Depth gap check (same protocol as scripts/eval_depth_diode.py / eval_depth_dir.py: 448x448 input, per-image scale+shift fit, AbsRel lin / inverse fit):
frozen core (release_v4) vs a public relative-depth model (env TEACHER, default Depth-Anything-V2-Small; its output is inverse depth / disparity like ours).
Run: env -u PYTHONPATH .venv/bin/python scripts/eval_depth_teacher.py"""
import os, sys, json, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.evaluate import absrel_both
from transformers import AutoModelForDepthEstimation
TEACHER = os.environ.get("TEACHER", "depth-anything/Depth-Anything-V2-Small-hf")
core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.cuda().eval()
da = AutoModelForDepthEstimation.from_pretrained(TEACHER).cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


@torch.no_grad()
def preds(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None].cuda() / 255
    d0 = core(normalize(x), with_teacher_heads=False)["depth"][0, 0].float().cpu().numpy()
    d1 = da(pixel_values=(F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False) - MEAN) / STD).predicted_depth
    d1 = F.interpolate(d1[:, None], size=(448, 448), mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
    return d0, d1


def run(items):
    r0, r1 = [], []
    for p, g in items:
        g = np.asarray(Image.fromarray(np.load(g).astype(np.float32)).resize((448, 448), Image.NEAREST)); v = g > 0
        if v.sum() < 500: continue
        d0, d1 = preds(p); r0.append(absrel_both(d0, g, v)); r1.append(absrel_both(d1, g, v))
    r0, r1 = np.array(r0), np.array(r1); return dict(n=len(r0), core=[round(float(x), 4) for x in r0.mean(0)[:2]], teacher=[round(float(x), 4) for x in r1.mean(0)[:2]])


ind = json.load(open("data/general/eval_diode/indoor.json")); E = "data/general/eval_diode"
res = {"diode_indoor": run([(f"{E}/images/{s}.jpg", f"{E}/depth/{s}.npy") for s, i in ind.items() if i]),
       "diode_outdoor": run([(f"{E}/images/{s}.jpg", f"{E}/depth/{s}.npy") for s, i in ind.items() if not i]),
       "kitti": run([(p, p.replace("/images/", "/depth/")[:-4] + ".npy") for p in sorted(glob.glob("data/general/eval_kitti/images/*.jpg"))])}
for k, v in res.items(): print(k, json.dumps(v), "(AbsRel [lin fit, inverse fit])", flush=True)
json.dump(dict(teacher=TEACHER, res=res), open("runs/modules/depth_teacher_eval.json", "w"), indent=1)
