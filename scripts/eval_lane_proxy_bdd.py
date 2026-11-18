"""Weak check of the lane_marking class on REAL road frames (BDD100K val, never trained on). There are no real lane labels, so a proxy: road pixels that are clearly brighter than the road around them
(the same white top-hat rule that bdd_prepare.py uses to set lane paint to 'ignored') are taken as 'probably paint'. Reports the share of those pixels predicted as lane_marking, and the share of the remaining
road pixels predicted as lane_marking (should stay near 0). A proxy, not a ground truth: bright pixels also include reflections and worn patches. Usage: CKPT=runs/x KIND=vit python scripts/eval_lane_proxy_bdd.py"""
import glob, os, sys
import numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); EV = "data/general/eval_bdd"; li = CLASSES.index("lane_marking"); gi = CLASSES.index("ground_road")
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval(); a = np.zeros(4)
for p in sorted(glob.glob(f"{EV}/images/*.jpg")):
    im = Image.open(p).convert("RGB"); seg = np.asarray(Image.open(f"{EV}/seg/{os.path.basename(p)[:-4]}.png")); g = np.asarray(im.convert("L")).astype(np.float32)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))); road = seg == gi; paint = (seg == 255) & (th > 25) & (cv2.dilate(road.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    x = torch.from_numpy(np.asarray(im.resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): pr = m(normalize(x.cuda()), with_teacher_heads=False)["seg"][0].argmax(0).cpu().numpy()
    pr = np.asarray(Image.fromarray(pr.astype(np.uint8)).resize((640, 480), Image.NEAREST)) == li; a += [(pr & paint).sum(), paint.sum(), (pr & road).sum(), road.sum()]
print(f"{CK} {K} BDD val lane proxy: bright road paint predicted lane_marking {a[0] / max(a[1], 1):.3f} (paint pixels {int(a[1])}); plain road predicted lane_marking {a[2] / max(a[3], 1):.4f}")
