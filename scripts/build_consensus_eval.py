"""Consensus reference labels for own-camera frames (Update 167): scoring against ONE teacher punishes the student wherever it is right
and the teacher wrong (window frames labelled "car", desk edges "other"). Three independent large models label each frame:
  Mask2Former Swin-L Cityscapes (outdoor only), Mask2Former Swin-L COCO panoptic, OneFormer Swin-L COCO
COCO's 133 classes are mapped to our 22 indoor classes (coco_lut) or to the 19 Cityscapes classes (COCO2CS below). A pixel gets a label
only if >= 2 models agree (outdoor: 2 of 3; indoor: both COCO models); otherwise 255 (not scored). Large vehicle regions (> 5 % of the
frame) are dropped (ego-vehicle prior on window views). Frames: 24 outdoor + 24 indoor from the held-out camera val splits, spread in time.
Writes data/general/eval_cam_out|eval_cam_in/{images,seg}. Run: env -u PYTHONPATH .venv/bin/python scripts/build_consensus_eval.py"""
import json, sys, shutil, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from pfm.indoor_classes import coco_lut
from transformers import Mask2FormerForUniversalSegmentation, OneFormerForUniversalSegmentation, OneFormerProcessor
T = ROOT / "data/general/train"; dev = "cuda"; TH, TW = 512, 1376; NPER = 24
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
CS = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person", "rider",
      "car", "truck", "bus", "train", "motorcycle", "bicycle"]
COCO2CS = {"road": 0, "pavement-merged": 1, "building-other-merged": 2, "house": 2, "roof": 2, "wall-brick": 3, "wall-stone": 3, "wall-other-merged": 3,
           "fence-merged": 4, "traffic light": 6, "stop sign": 7, "tree-merged": 8, "grass-merged": 9, "dirt-merged": 9, "gravel": 9, "sand": 9,
           "sky-other-merged": 10, "person": 11, "car": 13, "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18, "mountain-merged": 9}


def probs_m2f(m, x, A=None):
    with torch.no_grad():
        o = m(pixel_values=((F.interpolate(x, size=(TH, TW), mode="bilinear", align_corners=False) - MEAN) / STD).half())
        c = o.class_queries_logits.float().softmax(-1)[..., :-1]
        if A is not None: c = c @ A.t()
        s = torch.einsum("bqc,bqhw->bchw", c, o.masks_queries_logits.float().sigmoid())
        return F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)[0].argmax(0)


def probs_one(m, proc, img, A):
    inp = proc(images=img, task_inputs=["semantic"], return_tensors="pt").to(dev)
    with torch.no_grad():
        o = m(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()})
        c = o.class_queries_logits.float().softmax(-1)[..., :-1] @ A.t()
        s = torch.einsum("bqc,bqhw->bchw", c, o.masks_queries_logits.float().sigmoid())
        return F.interpolate(s, size=img.size[::-1], mode="bilinear", align_corners=False)[0].argmax(0)


m_cs = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(dev).eval().half()
m_co = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic").to(dev).eval().half()
proc = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
m_on = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(dev).eval().half()
id2 = m_co.config.id2label; n133 = len(id2)
lut_in = torch.tensor(coco_lut(id2)); A_in = torch.zeros(22, n133, device=dev); A_in[lut_in, torch.arange(n133)] = 1.0
A_cs = torch.zeros(20, n133, device=dev)            # row 19 = "not a Cityscapes class" (never agrees with anything)
for i in range(n133): A_cs[COCO2CS.get(id2[i], 19), i] = 1.0
for cs, key in (("out", "out"), ("in", "in")):
    names = json.loads((T / f"cam_split_{key}.json").read_text())["val"]; names = names[:: max(1, len(names) // NPER)][:NPER]
    D = ROOT / f"data/general/eval_cam_{cs}"; shutil.rmtree(D, ignore_errors=True); (D / "images").mkdir(parents=True); (D / "seg").mkdir()
    agree_share = []
    for n in names:
        img = Image.open(T / "images/cam" / f"{n}.jpg").convert("RGB"); x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].to(dev).float() / 255
        valid = x[0].sum(0) > 0.02
        if cs == "out":
            a, b, c = probs_m2f(m_cs, x), probs_m2f(m_co, x, A_cs), probs_one(m_on, proc, img, A_cs)
            lab = torch.full_like(a, 255); ab = (a == b); ac = (a == c); bc = (b == c) & (b < 19)
            lab[bc] = b[bc]; lab[ac] = a[ac]; lab[ab] = a[ab]
        else:
            b, c = probs_m2f(m_co, x, A_in), probs_one(m_on, proc, img, A_in)
            lab = torch.where(b == c, b, torch.full_like(b, 255))
        lab[~valid] = 255; y = lab.byte().cpu().numpy()
        if cs == "out":
            veh = np.isin(y, (13, 14, 15, 16)); lb, k = ndimage.label(veh)
            for j, s in enumerate(ndimage.sum(veh, lb, range(1, k + 1)), 1):
                if s > 0.05 * y.size: y[lb == j] = 255
        agree_share.append(float((y != 255).sum() / max(1, valid.sum().item())))
        img.save(D / "images" / f"{n}.jpg", quality=95); Image.fromarray(y).save(D / "seg" / f"{n}.png")
    print(cs, len(names), "frames, scored pixel share (models agree) %.2f" % np.mean(agree_share), flush=True)
