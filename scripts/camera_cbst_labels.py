"""Class-balanced pseudo-label thresholds (CBST, Zou et al. ECCV 2018) for the own-camera outdoor TRAIN frames (Update 167): the global 0.5
teacher-confidence cut kept only 44 % of car, 57 % of pole and 24 % of fence pixels (big classes 92-100 %), so the student hardly saw small
classes. Threshold per class c = clip(20th percentile of the teacher's confidence on pixels it assigns to c, 0.2, 0.5) -> each class keeps
~80 % of its pixels. Writes data/general/train/labels_cam_out_cb/<name>.png for the train frames of cam_split_out.json (val labels unchanged).
Run: env -u PYTHONPATH .venv/bin/python scripts/camera_cbst_labels.py"""
import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation
T = Path("data/general/train"); OUT = T / "labels_cam_out_cb"; OUT.mkdir(exist_ok=True); dev = "cuda"; TH, TW = 512, 1376
sp = json.loads((T / "cam_split_out.json").read_text())
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
m = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(dev).eval().half()
res = {}
for n in sp["train"]:
    x = torch.from_numpy(np.asarray(Image.open(T / "images/cam" / f"{n}.jpg").convert("RGB"))).permute(2, 0, 1)[None].to(dev).float() / 255
    with torch.no_grad():
        o = m(pixel_values=((F.interpolate(x, size=(TH, TW), mode="bilinear", align_corners=False) - MEAN) / STD).half())
        p = torch.einsum("bqc,bqhw->bchw", o.class_queries_logits.float().softmax(-1)[..., :-1], o.masks_queries_logits.float().sigmoid())
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)[0]; p = p / p.sum(0, keepdim=True).clamp(min=1e-6)
        conf, lab = p.max(0); valid = x[0].sum(0) > 0.02
    res[n] = (conf.half().cpu(), lab.byte().cpu(), valid.cpu())
allc = torch.cat([r[0][r[2]].float() for r in res.values()]); alll = torch.cat([r[1][r[2]] for r in res.values()])
thr = torch.full((256,), 0.5)
for c in range(19):
    v = allc[alll == c]
    if v.numel() > 100: thr[c] = float(torch.quantile(v[torch.randperm(v.numel())[:200000]], 0.2).clamp(0.2, 0.5))
print("class thresholds:", {c: round(float(thr[c]), 2) for c in range(19) if thr[c] < 0.5})
kept = np.zeros(19); tot = np.zeros(19)
for n, (conf, lab, valid) in res.items():
    keep = (conf.float() >= thr[lab.long()]) & valid; out = lab.clone(); out[~keep] = 255
    Image.fromarray(out.numpy()).save(OUT / f"{n}.png")
    for c in range(19): tot[c] += ((lab == c) & valid).sum().item(); kept[c] += ((out == c)).sum().item()
print("kept share:", {c: f"{kept[c] / tot[c]:.0%}" for c in range(19) if tot[c] > 1000})
