"""Offline pseudo-label factory (D2/D6): Grounding DINO names the class, SAM2 cuts the mask.

For every image, per class prompt: boxes (score >= box_thr) -> SAM2 masks (iou score >= mask_thr).
Output (all at the training resolution, same square resize as ImageFolder):
  <out>/seg/<id>.png      uint8 class index (see model.CLASSES); 255 = unlabelled (NOT background:
                          our class list is open-world, uncovered pixels stay ignored)
  <out>/corners/<id>.npy  float32 (N,4,2) normalised (x,y) gate corners tl,tr,br,bl; (0,4,2) when no gate
This is the HEAVY teacher path that runs on the 5090. The Orin-side tiered labeller (D6) is separate.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .model import CLASSES

# class name -> Grounding DINO text prompt. 'background' and 'other_obstacle' get no prompt:
# unknown obstacles are caught by depth (D2), not by this labeller.
PROMPTS = {
    "gate": ["racing gate", "square gate", "arch", "frame gate"],
    "tree_pole": ["tree", "pole", "tree trunk", "lamp post"],
    "person": ["person"],
    "ground_road": ["road", "ground", "grass", "path", "field", "snow", "sand"],
    "building": ["building", "house"],
    "opponent_drone": ["drone"],
    "car": ["car", "truck", "bus"],
    "rock": ["rock", "boulder"],
    "lane_marking": ["lane marking", "road line", "crosswalk"],
}
# Single-phrase queries score far higher than "a. b. c." multi-phrase ones (measured:
# 0.47-0.5 vs 0.26-0.38 on the same image), so each phrase is one query.
# Classes prone to false positives need a higher box threshold (skis were labelled "drone" at 0.35).
BOX_THR = {"opponent_drone": 0.5, "gate": 0.45, "lane_marking": 0.4, "rock": 0.4, "building": 0.4}
# Only ground/road may cover most of the frame (nadir views: grass, water, snow). A "building" or
# "tree" mask over >60% of the image is a texture mismatch, not an object (seen on aerial frames).
MAX_FRAC = 0.6


def order_corners(pts):
    """pts (4,2) -> tl, tr, br, bl."""
    pts = np.asarray(pts, dtype=np.float32)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.stack([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])


def mask_to_corners(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 64:
        return None
    return order_corners(cv2.boxPoints(cv2.minAreaRect(c)))


class Labeler:
    def __init__(self, device="cuda", gd="IDEA-Research/grounding-dino-tiny", sam="facebook/sam2.1-hiera-tiny",
                 box_thr=0.3, text_thr=0.25, mask_thr=0.7):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.dev, self.box_thr, self.text_thr, self.mask_thr = device, box_thr, text_thr, mask_thr
        self.gdp = AutoProcessor.from_pretrained(gd)
        self.gdm = AutoModelForZeroShotObjectDetection.from_pretrained(gd).to(device).eval()
        self.samp = Sam2Processor.from_pretrained(sam)
        self.sam = Sam2Model.from_pretrained(sam).to(device).eval()

    @torch.no_grad()
    def label(self, img: Image.Image):
        w, h = img.size
        seg = np.full((h, w), 255, np.uint8)
        gates, found = [], []  # found: (area, class_idx, mask)
        for cname, phrases in PROMPTS.items():
            thr = max(self.box_thr, BOX_THR.get(cname, 0.0)) if cname in BOX_THR else self.box_thr
            boxes = []
            for ph in phrases:
                inp = self.gdp(images=img, text=ph.lower() + ".", return_tensors="pt").to(self.dev)
                out = self.gdm(**inp)
                res = self.gdp.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=thr, text_threshold=self.text_thr, target_sizes=[(h, w)])[0]
                boxes += res["boxes"].cpu().tolist()
            if not boxes:
                continue
            boxes = boxes[:32]
            sin = self.samp(images=img, input_boxes=[boxes], return_tensors="pt").to(self.dev)
            sout = self.sam(**sin, multimask_output=False)
            masks = self.samp.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0]  # (N,1,H,W)
            ious = sout.iou_scores.cpu().reshape(len(boxes), -1)[:, 0]
            for m, iou in zip(masks[:, 0].numpy(), ious):
                if iou < self.mask_thr or m.sum() < 64:
                    continue
                if cname != "ground_road" and m.mean() > MAX_FRAC:
                    continue
                found.append((int(m.sum()), CLASSES.index(cname), m))
                if cname == "gate":
                    gates.append(m)
        # big first, small last, so small objects (person on road) overwrite the large stuff regions
        for _, ci, m in sorted(found, key=lambda t: -t[0]):
            seg[m] = ci
        corners = [c for c in (mask_to_corners(m) for m in gates) if c is not None]
        corners = np.stack(corners) / np.array([w, h], np.float32) if corners else np.zeros((0, 4, 2), np.float32)
        return seg, corners.astype(np.float32)


def run_folder(images_root, out_root, size=448, device="cuda", **kw):
    from .data import ImageFolder
    ds = ImageFolder(images_root, size)
    out = Path(out_root)
    (out / "seg").mkdir(parents=True, exist_ok=True)
    (out / "corners").mkdir(exist_ok=True)
    lab = Labeler(device, **kw)
    from tqdm import tqdm
    for i in tqdm(range(len(ds))):
        sid, x = ds[i]
        if (out / "seg" / f"{sid}.png").exists():
            continue
        img = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        seg, corners = lab.label(img)
        Image.fromarray(seg).save(out / "seg" / f"{sid}.png")
        np.save(out / "corners" / f"{sid}.npy", corners)
