"""Extra outdoor sources for the 19-class outdoor module (Update 167): label data/general/train/images/<src>/*.jpg with the Swin-L Cityscapes
teacher -> data/general/train/labels_out/seg/<src>__<id>.png (19 Cityscapes ids, 255 = low confidence < CONF). The module trainer's pool()
puts every non-BDD source in the pseudo pool (P_PSEUDO share of each batch). Default sources: acdc (night / fog / rain / snow) and mvist
(Mapillary Vistas, worldwide cameras). Existing label files are skipped (resumable).
Run: env -u PYTHONPATH .venv/bin/python scripts/teacher_label_sources.py [src ...]"""
import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation
T = Path("data/general/train"); OUT = T / "labels_out/seg"; SRCS = sys.argv[1:] or ["acdc", "mvist"]; CONF = 0.5; dev = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
m = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(dev).eval().half()
for src in SRCS:
    fs = sorted((T / "images" / src).glob("*.jpg")); done = 0
    for f in fs:
        op = OUT / f"{src}__{f.stem}.png"
        if op.exists(): continue
        im = Image.open(f).convert("RGB"); W, H = im.size; s = 1024 / max(W, H); h, w = int(round(H * s / 32)) * 32, int(round(W * s / 32)) * 32
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].to(dev).float() / 255
        with torch.no_grad():
            o = m(pixel_values=((F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) - MEAN) / STD).half())
            p = torch.einsum("bqc,bqhw->bchw", o.class_queries_logits.float().softmax(-1)[..., :-1], o.masks_queries_logits.float().sigmoid())
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0]; p = p / p.sum(0, keepdim=True).clamp(min=1e-6)
            conf, lab = p.max(0); lab[conf < CONF] = 255
        Image.fromarray(lab.byte().cpu().numpy()).save(op); done += 1
        if done % 1000 == 0: print(src, done, flush=True)
    print(src, "labelled", done, "of", len(fs), flush=True)
