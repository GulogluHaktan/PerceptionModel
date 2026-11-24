"""Teacher class PROBABILITIES for our own camera frames (Update 167, soft distillation as in CWD / DUDA-style KD): the Swin-L teachers'
per-class probability maps at 96x256 (uint8, x255) -> data/general/train/soft_cam_out/<name>.npy (19 ch) and soft_cam_in/<name>.npy (22 ch).
finetune_m2f_indoor.py (wide_pair) reads these for images/cam frames and adds the KL term (KD weight) on top of the hard labels.
Run: env -u PYTHONPATH .venv/bin/python scripts/camera_soft_labels.py"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pfm.indoor_classes import K_IN, coco_lut
from transformers import Mask2FormerForUniversalSegmentation
T = ROOT / "data/general/train"; dev = "cuda"; TH, TW = 512, 1376; SH, SW = 96, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
names = sorted(p.stem for p in (T / "images/cam").glob("*.jpg") if not p.stem.startswith("rot180"))
for cs, mname in (("out", "facebook/mask2former-swin-large-cityscapes-semantic"), ("in", "facebook/mask2former-swin-large-coco-panoptic")):
    m = Mask2FormerForUniversalSegmentation.from_pretrained(mname).to(dev).eval().half(); A = None
    if cs == "in":
        lut = torch.tensor(coco_lut(m.config.id2label)); A = torch.zeros(K_IN, len(lut), device=dev); A[lut, torch.arange(len(lut))] = 1.0
    od = T / f"soft_cam_{cs}"; od.mkdir(exist_ok=True)
    for n in names:
        x = torch.from_numpy(np.asarray(Image.open(T / "images/cam" / f"{n}.jpg").convert("RGB"))).permute(2, 0, 1)[None].to(dev).float() / 255
        with torch.no_grad():
            o = m(pixel_values=((F.interpolate(x, size=(TH, TW), mode="bilinear", align_corners=False) - MEAN) / STD).half())
            c = o.class_queries_logits.float().softmax(-1)[..., :-1]
            if A is not None: c = c @ A.t()
            s = torch.einsum("bqc,bqhw->bchw", c, o.masks_queries_logits.float().sigmoid())
            s = F.interpolate(s, size=(SH, SW), mode="bilinear", align_corners=False)[0]; s = s / s.sum(0, keepdim=True).clamp(min=1e-6)
        np.save(od / f"{n}.npy", (s * 255).round().byte().cpu().numpy())
    print(cs, len(names), "soft maps", flush=True); del m; torch.cuda.empty_cache()
