"""Upper-bound check for pseudo-labelling: a strong public Cityscapes segmenter (Mask2Former Swin-L, HF facebook/mask2former-swin-large-cityscapes-semantic)
mapped to our 11-class v5 set with the same rules as the BDD / IDD-AW preparation (+ v3: vegetation -> tree_pole, sky -> background), scored on our
driving eval sets. If it is clearly above core+SegRefine, its labels on unlabelled driving images (ACDC, Mapillary) can train a better seg module.
Run: env -u PYTHONPATH .venv/bin/python scripts/eval_seg_teacher.py   (env TEACHER, SIDE=short side px, N=frames per set)"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
TEACHER = os.environ.get("TEACHER", "facebook/mask2former-swin-large-cityscapes-semantic"); SIDE = int(os.environ.get("SIDE", "1024")); N = int(os.environ.get("N", "0")) or None
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; K = 11; ix = {c: i for i, c in enumerate(CL)}
CS2OURS = np.full(256, 255, np.uint8)        # Cityscapes train ids -> v5 (+ v3 sky / vegetation)
for ids, c in (((0, 1, 9), "ground_road"), ((2, 3), "building"), ((5, 6, 8), "tree_pole"), ((10,), "background"), ((11, 12), "person"), ((13, 14, 15, 17, 18), "car"), ((4, 7, 16), "other_obstacle")):
    for k in ids: CS2OURS[k] = ix[c]


class Teacher:
    def __init__(self, dev="cuda"):
        self.p = AutoImageProcessor.from_pretrained(TEACHER); self.m = Mask2FormerForUniversalSegmentation.from_pretrained(TEACHER, torch_dtype=torch.float16).to(dev).eval(); self.dev = dev
    @torch.no_grad()
    def probs(self, im):
        """PIL RGB -> (19, H, W) class probabilities at the image size (semantic = sum over queries of class prob x mask prob)."""
        W, H = im.size; s = SIDE / min(W, H); x = self.p(images=im.resize((round(W * s), round(H * s)), Image.BICUBIC), return_tensors="pt", do_resize=False)["pixel_values"].to(self.dev, torch.float16)
        o = self.m(pixel_values=x); cls = o.class_queries_logits.float().softmax(-1)[0, :, :-1]; mk = o.masks_queries_logits.float().sigmoid()[0]
        seg = torch.einsum("qc,qhw->chw", cls, mk); return F.interpolate(seg[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    def label(self, im, conf=0.0):
        pr = self.probs(im); pr = pr / pr.sum(0, keepdim=True).clamp(min=1e-6); c, a = pr.max(0); lab = CS2OURS[a.cpu().numpy()]
        if conf > 0: lab = lab.copy(); lab[(c < conf).cpu().numpy()] = 255
        return lab, c.cpu().numpy()


def miou(cm):
    iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); pres = cm.sum(1) > 0; return float(iu[pres].mean()), {CL[i]: round(float(iu[i]), 3) for i in np.nonzero(pres)[0]}


def main():
    T = Teacher(); S = json.loads(Path("runs/selfimprove/real_dark_split.json").read_text())
    sets = {n: [(p, Path(f"data/general/eval_{n}/seg") / (p.stem + ".png")) for p in sorted(Path(f"data/general/eval_{n}/images").glob("*"))] for n in ("bdd5", "iddaw5")}
    sets["real_dark"] = [(Path(a), Path(b)) for a, b in S["val"] + S["test"]]
    res = {}
    for name, pairs in sets.items():
        cm = np.zeros((K, K), np.int64)
        for p, lp in pairs[:N]:
            if not lp.exists(): continue
            im = Image.open(p).convert("RGB"); y = np.asarray(Image.open(lp)).astype(np.int64); lab, _ = T.label(im)
            if lab.shape != y.shape: lab = np.asarray(Image.fromarray(lab).resize(y.shape[::-1], Image.NEAREST))
            v = (y != 255) & (y < K) & (lab != 255); cm += np.bincount(y[v] * K + lab[v], minlength=K * K).reshape(K, K)
        m, pc = miou(cm); res[name] = dict(miou=round(m, 4), per_class=pc); print(name, json.dumps(res[name]), flush=True)
    Path("runs/modules/seg_teacher_eval.json").write_text(json.dumps(dict(teacher=TEACHER, side=SIDE, res=res), indent=1))


if __name__ == "__main__":
    main()
