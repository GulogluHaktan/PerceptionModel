"""Our own IMX219 recordings -> training pairs labelled by the Swin-L teachers (Update 167: self-training on the target camera).
Raw frames (subsampled .npy stacks from the Orin raw dumps) go through the SAME ISP as the vehicle (RawISP: row-class fix, demosaic, WB,
denoise, undistort), are saved as jpg, and labelled by
  outdoor: facebook/mask2former-swin-large-cityscapes-semantic  -> data/general/train/labels_cam_out/<name>.png (19 Cityscapes ids)
  indoor : facebook/mask2former-swin-large-coco-panoptic (133 -> 22 via coco_lut, summed member probabilities) -> labels_cam_in/<name>.png
Pixels outside the undistorted image (black border) and low-confidence pixels (max prob < CONF) are 255 (ignored).
The last 15 % of every recording (in time) is the held-out camera val split (names listed in data/general/train/cam_split.json).
Run: env -u PYTHONPATH .venv/bin/python scripts/camera_pseudo_labels.py <dir with *.npy> [calib json]"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy")); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, "/home/shannon/Projects/IMX219_Camera_Driver/capture")
import raw_isp, camera
from pfm.indoor_classes import K_IN, coco_lut
from transformers import Mask2FormerForUniversalSegmentation
SRC = Path(sys.argv[1]); CAL = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "deploy/camera_imx219_cam1_1640x616.json")
T = ROOT / "data/general/train"; IMG = T / "images/cam"; LO = T / "labels_cam_out"; LI = T / "labels_cam_in"
for d in (IMG, LO, LI): d.mkdir(parents=True, exist_ok=True)
CONF = 0.5; dev = "cuda"; TH, TW = 512, 1376
grid = camera.Undistort(CAL, 100).gpu(dev).grid
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
mo = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(dev).eval().half()
mi = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic").to(dev).eval().half()
lut = torch.tensor(coco_lut(mi.config.id2label)); A = torch.zeros(K_IN, len(lut), device=dev); A[lut, torch.arange(len(lut))] = 1.0


def teach(m, x, A=None):
    with torch.no_grad():
        o = m(pixel_values=((F.interpolate(x, size=(TH, TW), mode="bilinear", align_corners=False) - MEAN) / STD).half())
        c = o.class_queries_logits.float().softmax(-1)[..., :-1]
        if A is not None: c = c @ A.t()
        s = torch.einsum("bqc,bqhw->bchw", c, o.masks_queries_logits.float().sigmoid())
        s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        p = s / s.sum(0, keepdim=True).clamp(min=1e-6); conf, lab = p.max(0)
        return lab, conf


split = {"train": [], "val": []}
for f in sorted(SRC.glob("*.npy")):
    stack = np.load(f, mmap_mode="r"); rot = 180 if "rot180" in f.stem else 0
    isp = raw_isp.RawISP(grid=grid, rotate=rot) if "rotate" in raw_isp.RawISP.__init__.__code__.co_varnames else raw_isp.RawISP(grid=grid)
    if rot and "rotate" not in raw_isp.RawISP.__init__.__code__.co_varnames: isp.rotate = rot
    for _ in range(40): isp(np.ascontiguousarray(stack[0]))                       # converge the WB / digital gain EMA
    n = len(stack); nv = int(0.15 * n)
    for k in range(n):
        rgb = isp(np.ascontiguousarray(stack[k])).clamp(0, 1)                     # (1,3,616,1640) float, what the vehicle sees
        name = f"{f.stem}_{k:05d}"; valid = rgb[0].sum(0) > 0.02
        Image.fromarray((rgb[0].permute(1, 2, 0) * 255).byte().cpu().numpy()).save(IMG / f"{name}.jpg", quality=94)
        for m_, AA, d in ((mo, None, LO), (mi, A, LI)):
            lab, conf = teach(m_, rgb, AA); lab[(conf < CONF) | ~valid] = 255
            Image.fromarray(lab.byte().cpu().numpy()).save(d / f"{name}.png")
        split["val" if k >= n - nv else "train"].append(name)
    print(f.stem, n, "frames", flush=True)
(T / "cam_split.json").write_text(json.dumps(split)); print("train", len(split["train"]), "val", len(split["val"]))
