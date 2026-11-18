"""Depth on a folder with images/*.jpg and depth/*.npy (metres, 0 = no measurement), e.g. data/general/eval_kitti (sparse lidar). Per image scale+shift fit, AbsRel, both fits (pfm.evaluate.absrel_both:
lin = prediction fitted to depth, inv = fitted to inverse depth). Usage: CKPT=runs/x KIND=vit EVDIR=data/general/eval_kitti python scripts/eval_depth_dir.py"""
import glob, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.evaluate import absrel_both
EV = os.environ["EVDIR"]; CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval(); r = []
for p in sorted(glob.glob(f"{EV}/images/*.jpg")):
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): d = m(normalize(x.cuda()), with_teacher_heads=False)["depth"][0, 0].float().cpu().numpy()
    g = np.asarray(Image.fromarray(np.load(f"{EV}/depth/{os.path.basename(p)[:-4]}.npy").astype(np.float32)).resize((448, 448), Image.NEAREST)); v = g > 0
    if v.sum() >= 200: r.append(absrel_both(d, g, v))
r = np.array(r); print(f"{CK} {K} {os.path.basename(EV)} depth AbsRel (n={len(r)}): lin fit {r[:, 0].mean():.3f} | inverse fit {r[:, 1].mean():.3f}")
