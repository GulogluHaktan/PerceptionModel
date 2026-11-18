"""Label set v4 = v3 + WEAK road-paint labels on the BDD training frames. BDD has no lane labels in our mirror; bdd_prepare.py already sets road pixels that are clearly brighter than the road around them
(white top-hat) to 255 = ignored. Here the strong ones (top-hat > max(40, 0.35 * median road brightness), label 255, with road on most sides (75% of a 41 px window), component at least 12 px) become lane_marking (9); the weaker ones stay ignored.
This is a proxy, not a ground truth: it also marks crosswalks, arrows and stop lines (all road paint), and can mark reflections on wet roads or headlight patches at night. Frames whose candidate share of the road is above 12%
(glare, wet road) get no paint label at all. Writes data/general/train/labels_v4/seg/bdd__*.png (v3 files are not touched). Usage: python scripts/lane_labels_v4.py"""
import glob, os
import numpy as np, cv2
from PIL import Image
TR = "data/general/train"; OUT = f"{TR}/labels_v4/seg"; os.makedirs(OUT, exist_ok=True); ROAD, LANE = 4, 9; n = used = 0; share = []
for f in sorted(glob.glob(f"{TR}/labels_v3/seg/bdd__*.png")):
    l = np.asarray(Image.open(f)).copy(); im = Image.open(f"{TR}/images/bdd/{os.path.basename(f)[5:-4]}.jpg").convert("L").resize(l.shape[::-1], Image.BILINEAR); g = np.asarray(im).astype(np.float32)
    road = l == ROAD; n += 1
    if road.sum() < 2000: Image.fromarray(l).save(f"{OUT}/{os.path.basename(f)}"); continue
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))); near = cv2.blur((road | (l == 255) & (th > 25)).astype(np.float32), (41, 41)) > 0.75  # road (or other paint) on most sides: drops the hood edge, kerbs and car outlines
    p = (l == 255) & near & (th > max(40.0, 0.35 * float(np.median(g[road])))); k, cc, st, _ = cv2.connectedComponentsWithStats(p.astype(np.uint8), connectivity=8); p &= st[cc, cv2.CC_STAT_AREA] >= 12
    p[int(0.85 * p.shape[0]):] = False  # bottom 15% of the frame: the bright edge of the ego car's hood passes the paint rule, so nothing is labelled there (stays ignored)
    s = p.sum() / (road.sum() + p.sum())
    if 0 < s <= 0.12: l[p] = LANE; used += 1; share.append(s)
    Image.fromarray(l).save(f"{OUT}/{os.path.basename(f)}")
print(f"{n} BDD frames, {used} with weak paint labels, median paint share of road {np.median(share):.4f}, mean {np.mean(share):.4f}")
