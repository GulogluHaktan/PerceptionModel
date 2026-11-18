"""Held-out real indoor test (SUN RGB-D, 300 frames, our class ids): per-class IoU and mIoU over building / ground_road / person / other_obstacle (classes present), pixel accuracy, depth AbsRel (scale+shift aligned). CKPT=runs/x KIND=vit"""
import os, sys, glob
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
from pfm.evaluate import absrel_both
EV = os.environ.get("EVDIR", "data/general/eval_sun"); CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
conf = np.zeros((len(CLASSES), len(CLASSES)), np.int64); ar = []
for p in sorted(glob.glob(f"{EV}/images/*.jpg")):
    sid = os.path.basename(p)[:-4]; x = torch.from_numpy(np.asarray(Image.open(p).resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): o = m(normalize(x.cuda()), with_teacher_heads=False)
    pred = o["seg"][0].argmax(0).cpu().numpy(); d = o["depth"][0, 0].float().cpu().numpy()
    gt = np.asarray(Image.open(f"{EV}/seg/{sid}.png").resize((448, 448), Image.NEAREST)); v = gt != 255; np.add.at(conf, (gt[v], pred[v]), 1)
    if not os.path.isdir(f"{EV}/depth"): continue
    dg = np.asarray(Image.fromarray(np.load(f"{EV}/depth/{sid}.npy").astype(np.float32)).resize((448, 448), Image.NEAREST)); vd = dg > 0
    if vd.sum() > 500:
        ar.append(absrel_both(d, dg, vd))
iou = {}
for c in ("background", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "other_obstacle"):
    k = CLASSES.index(c)
    if conf[k].sum() == 0: continue
    tp = conf[k, k]; un = conf[k].sum() + conf[:, k].sum() - tp; iou[c] = round(float(tp / un), 3)
print(f"{CK} {K} {os.path.basename(EV)} held-out: IoU {iou} mIoU {np.mean([v for v in iou.values() if v is not None]):.3f} pixel-acc {conf.diagonal().sum() / conf.sum():.3f} depth AbsRel {np.mean([a[0] for a in ar]) if ar else float('nan'):.3f} (fit in inverse depth {np.mean([a[1] for a in ar]) if ar else float('nan'):.3f})")
if os.environ.get("CONF") == "1":  # where the pixels of each true class go (row share) and what each predicted class is made of
    for k, c in enumerate(CLASSES):
        if conf[k].sum() == 0: continue
        row = conf[k] / conf[k].sum(); print(f"  true {c:15s} share {conf[k].sum() / conf.sum():.3f} -> " + ", ".join(f"{CLASSES[j]} {row[j]:.2f}" for j in np.argsort(-row)[:4] if row[j] >= 0.02))
    for k, c in enumerate(CLASSES):
        if conf[:, k].sum() == 0: continue
        col = conf[:, k] / conf[:, k].sum(); print(f"  pred {c:15s} share {conf[:, k].sum() / conf.sum():.3f} <- " + ", ".join(f"{CLASSES[j]} {col[j]:.2f}" for j in np.argsort(-col)[:4] if col[j] >= 0.02))
