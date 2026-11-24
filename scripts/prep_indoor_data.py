"""Data for the indoor segmentation module (src/pfm/indoor_classes.py, 22 classes).
  coco   : COCO panoptic val2017 REAL labels (data/general/raw_coco) -> indoor ids. Indoor images (>= 40 % of labelled pixels in indoor classes)
           sorted by id: the last 1000 -> data/general/eval_coco_in (held-out test with real labels), the rest -> train (labels_in/seg/coco__<id>.png).
  pseudo : COCO-panoptic Mask2Former Swin-L teacher on unlabelled indoor images (SUN RGB-D train, ADE train, sim indoor renders) -> labels_in/seg/<src>__<id>.png,
           pixels with teacher confidence < CONF -> 255. The teacher never saw COCO val, so its score on eval_coco_in is an honest upper reference.
  teacher_eval : teacher vs real labels on eval_coco_in.
Run: env -u PYTHONPATH .venv/bin/python scripts/prep_indoor_data.py coco|pseudo|teacher_eval   (env CONF, SRCS="sun,ade,simin")"""
import os, sys, io, json, glob, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.indoor_classes import IN_CLASSES, K_IN, COCO2IN, coco_lut
RAW = Path("data/general/raw_coco"); TR = Path("data/general/train"); LO = TR / "labels_in/seg"; EV = Path("data/general/eval_coco_in")
TEACHER = "facebook/mask2former-swin-large-coco-panoptic"; CONF = float(os.environ.get("CONF", "0.5"))


def coco():
    if not (RAW / "val2017").exists(): zipfile.ZipFile(RAW / "val2017.zip").extractall(RAW)
    if not (RAW / "annotations/panoptic_val2017.json").exists(): zipfile.ZipFile(RAW / "panoptic_ann.zip").extractall(RAW)
    if not (RAW / "annotations/panoptic_val2017").exists(): zipfile.ZipFile(RAW / "annotations/panoptic_val2017.zip").extractall(RAW / "annotations")
    J = json.load(open(RAW / "annotations/panoptic_val2017.json")); cat = {c["id"]: c["name"] for c in J["categories"]}
    rows = []
    for a in J["annotations"]:
        pan = np.asarray(Image.open(RAW / "annotations/panoptic_val2017" / a["file_name"]), np.uint32); pid = pan[..., 0] + 256 * pan[..., 1] + 65536 * pan[..., 2]
        lab = np.full(pid.shape, 255, np.uint8)
        for sgm in a["segments_info"]: lab[pid == sgm["id"]] = COCO2IN.get(cat[sgm["category_id"]], 0)
        v = lab != 255; share = float(((lab > 0) & v).sum() / max(1, v.sum())); rows.append((a["image_id"], a["file_name"][:-4], lab, share))
    indoor = sorted([r for r in rows if r[3] >= 0.4], key=lambda r: r[0]); ev = {r[0] for r in indoor[-1000:]}
    (EV / "images").mkdir(parents=True, exist_ok=True); (EV / "seg").mkdir(parents=True, exist_ok=True); (TR / "images/coco").mkdir(parents=True, exist_ok=True); LO.mkdir(parents=True, exist_ok=True)
    n_tr = 0
    for iid, stem, lab, share in rows:
        im = Image.open(RAW / "val2017" / f"{stem}.jpg").convert("RGB")
        if iid in ev: im.save(EV / "images" / f"{stem}.jpg", quality=95); Image.fromarray(lab).save(EV / "seg" / f"{stem}.png")
        else: im.save(TR / "images/coco" / f"{stem}.jpg", quality=95); Image.fromarray(lab).save(LO / f"coco__{stem}.png"); n_tr += 1
    cnt = np.zeros(K_IN, np.int64)
    for f in glob.glob(str(EV / "seg/*.png")): l = np.asarray(Image.open(f)); cnt += np.bincount(l[l != 255], minlength=K_IN)[:K_IN]
    print(f"coco: {len(rows)} images, indoor {len(indoor)}, eval {len(ev)}, train {n_tr}")
    print("eval pixel share:", {c: round(float(x / cnt.sum()), 4) for c, x in zip(IN_CLASSES, cnt)})


class Teacher:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch = torch; self.p = AutoImageProcessor.from_pretrained(TEACHER)
        self.m = Mask2FormerForUniversalSegmentation.from_pretrained(TEACHER, torch_dtype=torch.float16).cuda().eval(); self.lut = np.array(coco_lut(self.m.config.id2label), np.uint8)

    def label_tta(self, im, sides=(600, 800, 1000)):
        """Multi-scale + flip teacher (COCO indoor test: 0.757 vs 0.746 single pass); returns indoor ids, low confidence -> 255."""
        torch = self.torch; import torch.nn.functional as F
        W, H = im.size; lut = torch.tensor(self.lut, device="cuda").long(); acc = torch.zeros(K_IN, H, W, device="cuda"); n = 0
        with torch.no_grad():
            for side in sides:
                s = side / min(W, H); x = self.p(images=im.resize((round(W * s), round(H * s)), Image.BICUBIC), return_tensors="pt", do_resize=False)["pixel_values"].cuda().half()
                for flip in (False, True):
                    o = self.m(pixel_values=torch.flip(x, [3]) if flip else x); cls = o.class_queries_logits.float().softmax(-1)[0, :, :-1]; mk = o.masks_queries_logits.float().sigmoid()[0]
                    seg = torch.einsum("qc,qhw->chw", cls, mk); seg = torch.flip(seg, [2]) if flip else seg
                    seg = F.interpolate(seg[None], size=(H, W), mode="bilinear", align_corners=False)[0]; seg = seg / seg.sum(0, keepdim=True).clamp(min=1e-6)
                    acc.index_add_(0, lut[:seg.shape[0]], seg); n += 1
        c, a = (acc / n).max(0); lab = a.to(torch.uint8).cpu().numpy(); lab[(c < CONF).cpu().numpy()] = 255; return lab

    def label(self, im, side=800):
        torch = self.torch; import torch.nn.functional as F
        W, H = im.size; s = side / min(W, H)
        with torch.no_grad():
            x = self.p(images=im.resize((round(W * s), round(H * s)), Image.BICUBIC), return_tensors="pt", do_resize=False)["pixel_values"].cuda().half()
            o = self.m(pixel_values=x); cls = o.class_queries_logits.float().softmax(-1)[0, :, :-1]; mk = o.masks_queries_logits.float().sigmoid()[0]
            seg = torch.einsum("qc,qhw->chw", cls, mk); seg = F.interpolate(seg[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            seg = seg / seg.sum(0, keepdim=True).clamp(min=1e-6); c, a = seg.max(0)
        lab = self.lut[a.cpu().numpy()]; lab[(c < CONF).cpu().numpy()] = 255; return lab


def pseudo():
    T = Teacher(); LO.mkdir(parents=True, exist_ok=True); srcs = os.environ.get("SRCS", "sun,ade,simin").split(","); n = 0
    for src in srcs:
        if src == "simin": files = sorted(glob.glob("runs/indoor_job/out/*/rgb/*.png"))[::2]
        else: files = sorted(glob.glob(str(TR / "images" / src / "*")))
        for f in files:
            sid = Path(f).stem if src != "simin" else f"{Path(f).parts[-3]}_{Path(f).stem}"; lp = LO / f"{src}__{sid}.png"
            if lp.exists() and os.environ.get("OVERWRITE") != "1": continue
            im = Image.open(f).convert("RGB")
            if src == "simin": (TR / "images/simin").mkdir(parents=True, exist_ok=True); im.save(TR / "images/simin" / f"{sid}.jpg", quality=95)
            Image.fromarray(T.label_tta(im) if os.environ.get("TTA") == "1" else T.label(im)).save(lp); n += 1
            if n % 500 == 0: print(src, n, flush=True)
    print("pseudo done", n, flush=True)


def coco_train():
    """COCO panoptic train2017 REAL labels, indoor images only (>= 40 % indoor pixels) -> images/cocot + labels_in/seg/cocot__<id>.png."""
    import multiprocessing as mp
    if not (RAW / "annotations/panoptic_train2017.json").exists(): zipfile.ZipFile(RAW / "panoptic_ann.zip").extract("annotations/panoptic_train2017.json", RAW); zipfile.ZipFile(RAW / "panoptic_ann.zip").extract("annotations/panoptic_train2017.zip", RAW)
    if not (RAW / "annotations/panoptic_train2017").exists(): zipfile.ZipFile(RAW / "annotations/panoptic_train2017.zip").extractall(RAW / "annotations")
    J = json.load(open(RAW / "annotations/panoptic_train2017.json")); global CAT; CAT = {c["id"]: c["name"] for c in J["categories"]}
    (TR / "images/cocot").mkdir(parents=True, exist_ok=True); LO.mkdir(parents=True, exist_ok=True)
    with mp.Pool(16) as pool: res = pool.map(_one_train, J["annotations"], chunksize=64)
    print("coco train2017: annotations", len(res), "indoor kept", sum(res), flush=True)


CAT = {}
def _one_train(a):
    stem = a["file_name"][:-4]; lp = LO / f"cocot__{stem}.png"
    if lp.exists(): return 1
    pan = np.asarray(Image.open(RAW / "annotations/panoptic_train2017" / a["file_name"]), np.uint32); pid = pan[..., 0] + 256 * pan[..., 1] + 65536 * pan[..., 2]
    lab = np.full(pid.shape, 255, np.uint8)
    for sgm in a["segments_info"]: lab[pid == sgm["id"]] = COCO2IN.get(CAT[sgm["category_id"]], 0)
    v = lab != 255
    if float(((lab > 0) & v).sum() / max(1, v.sum())) < 0.4: return 0
    zf = zipfile.ZipFile(RAW / "train2017.zip"); (TR / "images/cocot" / f"{stem}.jpg").write_bytes(zf.read(f"train2017/{stem}.jpg")); Image.fromarray(lab).save(lp); return 1


def soft():
    """Teacher (single pass, side 800) class probabilities for every training image of the indoor pool, for distillation:
    data/general/train/soft_in/<src>__<id>.npy, uint8 (prob * 255), shape (22, 112, 112) = the stride-4 grid of the student's 448 input
    over the WHOLE image (the student's crops are taken from it with the same box)."""
    import torch, torch.nn.functional as F
    T = Teacher(); SO = TR / "soft_in"; SO.mkdir(parents=True, exist_ok=True); lut = torch.tensor(T.lut, device="cuda").long(); n = 0
    srcs = os.environ.get("SRCS", "coco,cocot,sun,ade").split(",")
    shard, nsh = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    for i, lp in enumerate(sorted(LO.glob("*.png"))):
        if i % nsh != shard: continue
        src, sid = lp.stem.split("__", 1)
        if src not in srcs: continue
        op = SO / f"{lp.stem}.npy"; ip = TR / "images" / src / f"{sid}.jpg"
        if op.exists() or not ip.exists(): continue
        im = Image.open(ip).convert("RGB"); W, H = im.size; s = 800 / min(W, H)
        with torch.no_grad():
            x = T.p(images=im.resize((round(W * s), round(H * s)), Image.BICUBIC), return_tensors="pt", do_resize=False)["pixel_values"].cuda().half()
            o = T.m(pixel_values=x); cls = o.class_queries_logits.float().softmax(-1)[0, :, :-1]; mk = o.masks_queries_logits.float().sigmoid()[0]
            seg = torch.einsum("qc,qhw->chw", cls, mk); seg = F.interpolate(seg[None], size=(112, 112), mode="bilinear", align_corners=False)[0]
            seg = seg / seg.sum(0, keepdim=True).clamp(min=1e-6); q = torch.zeros(K_IN, 112, 112, device="cuda"); q.index_add_(0, lut[:seg.shape[0]], seg)
        np.save(op, (q.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()); n += 1
        if n % 2000 == 0: print("soft", n, flush=True)
    print("soft done", n, flush=True)


def teacher_eval():
    T = Teacher(); cm = np.zeros((K_IN, K_IN), np.int64)
    for f in sorted(glob.glob(str(EV / "images/*.jpg"))):
        y = np.asarray(Image.open(EV / "seg" / (Path(f).stem + ".png"))).astype(np.int64); p = T.label(Image.open(f).convert("RGB")).astype(np.int64)
        v = (y != 255) & (p != 255); cm += np.bincount(y[v] * K_IN + p[v], minlength=K_IN * K_IN).reshape(K_IN, K_IN)
    iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); pres = cm.sum(1) > 0
    res = dict(miou=round(float(iu[pres].mean()), 4), per_class={IN_CLASSES[i]: round(float(iu[i]), 3) for i in range(K_IN) if pres[i]})
    print(json.dumps(res)); Path("runs/modules/indoor_teacher_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    dict(coco=coco, coco_train=coco_train, pseudo=pseudo, soft=soft, teacher_eval=teacher_eval)[sys.argv[1]]()
