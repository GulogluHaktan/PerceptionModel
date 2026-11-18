"""What the model calls motorcycles / bicycles and their riders on IDD-AW (no IDD data in training): share of the raw 'motorcycle + bicycle' pixels (level-3 ids 6, 7), 'rider' pixels (5) and
'person' pixels (4) per predicted class. Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=vit python scripts/diag_two_wheelers.py"""
import glob, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); EV = "data/general/eval_iddaw"; m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
def crop43(im, resample):
    W, H = im.size; cw = min(W, H * 4 // 3); ch = min(H, cw * 3 // 4); x0 = (W - cw) // 2; y0 = (H - ch) // 2; return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((640, 480), resample)
cnt = {k: np.zeros(len(CLASSES)) for k in ("two-wheeler", "rider", "pedestrian")}
for p in sorted(glob.glob(f"{EV}/images/*.jpg")):
    sid = os.path.basename(p)[:-4]; drive, fr = sid.split("_"); raw = np.asarray(crop43(Image.open(f"data/general/raw_iddaw/val/gt_labels/{drive}/{fr}_labellevel3Ids.png"), Image.NEAREST))
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): pr = m(normalize(x.cuda()), with_teacher_heads=False)["seg"][0].argmax(0).cpu().numpy()
    pr = np.asarray(Image.fromarray(pr.astype(np.uint8)).resize((640, 480), Image.NEAREST))
    for k, msk in (("two-wheeler", (raw == 6) | (raw == 7)), ("rider", raw == 5), ("pedestrian", raw == 4)): cnt[k] += np.bincount(pr[msk], minlength=len(CLASSES))
for k, c in cnt.items():
    s = c / max(c.sum(), 1); print(f"{CK} {K} IDD-AW {k:12s} ({int(c.sum())} px) -> " + ", ".join(f"{CLASSES[j]} {s[j]:.2f}" for j in np.argsort(-s)[:5] if s[j] >= 0.02))
