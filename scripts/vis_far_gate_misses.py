"""Contact sheet of small real gates (8-16 px at 256x192) on the TII video flights: missed ones (red label box) and found ones (green), cropped around the gate and enlarged.
Shows what a missed far gate looks like. Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid python scripts/vis_far_gate_misses.py out.jpg"""
import glob, os, random, sys
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); R = "data/real_tii/autonomous"; W, H = 256.0, 192.0; LO, HI = float(os.environ.get("LO", "8")), float(os.environ.get("HI", "16"))
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
miss, hit = [], []; stats = {"miss": [], "hit": []}
for fl in "13,16,18".split(","):
    d = glob.glob(f"{R}/flight-{fl}*")[0]; f = os.path.basename(d)
    for l in sorted(glob.glob(f"{d}/label_{f}/*.txt"))[::8]:
        gts = []
        for ln in open(l):
            v = ln.split()
            if len(v) < 17: continue
            v = [float(x) for x in v]; cs = np.array([[v[5 + 3 * k] * W, v[6 + 3 * k] * H] for k in range(4)]); vis = np.array([v[7 + 3 * k] > 0 for k in range(4)])
            if vis.sum() >= 2 and LO <= np.ptp(cs[vis], axis=0).max() < HI: gts.append((cs, vis))
        img = glob.glob(f"{d}/camera_{f}/{os.path.basename(l)[:-4]}.*")
        if not gts or not img: continue
        im = Image.open(img[0]).convert("RGB")
        x = torch.from_numpy(np.asarray(im.resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): s = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
        preds = [s[q, 1:].view(4, 2).numpy() * [W, H] for q in range(s.shape[0]) if s[q, 0].sigmoid() > 0.5]
        for cs, vis in gts:
            if os.environ.get("DRAW_PRED"):
                im = im.copy(); dr0 = ImageDraw.Draw(im)
                for p in preds: dr0.polygon([tuple(q) for q in p * [im.size[0] / W, im.size[1] / H]], outline="cyan")
            e = min((min(np.linalg.norm((np.roll(p, r, 0) - cs)[vis], axis=1).mean() for r in range(4)) for p in preds), default=1e9)
            sx, sy = im.size[0] / W, im.size[1] / H; c = cs[vis].mean(0) * [sx, sy]; r = int(os.environ.get("R", "48"))
            box = (int(c[0] - r), int(c[1] - r), int(c[0] + r), int(c[1] + r)); cr = im.crop(box).resize((192, 192), Image.BICUBIC); g = np.asarray(cr.convert("L"), np.float32)
            (hit if e < 20 else miss).append((cr, (cs * [sx, sy] - [box[0], box[1]]) * 2, vis)); stats["hit" if e < 20 else "miss"].append((g.mean(), g.std(), float(np.abs(np.diff(g, axis=1)).mean()), c[0] / im.size[0], c[1] / im.size[1], int(vis.sum())))
random.seed(0); random.shuffle(miss); random.shuffle(hit)
for k, v in stats.items():
    a = np.array(v); print(k, len(a), "brightness %.0f contrast %.0f sharpness %.2f x %.2f y %.2f corners %.2f; share at frame edge (x<0.1|x>0.9|y<0.1|y>0.9) %.2f" % (*a.mean(0), float(((a[:, 3] < 0.1) | (a[:, 3] > 0.9) | (a[:, 4] < 0.1) | (a[:, 4] > 0.9)).mean())))
sheet = Image.new("RGB", (8 * 196, 6 * 196 + 8), "white")
for row, (pool, col) in enumerate([(miss, "red")] * 4 + [(hit, "lime")] * 2):
    for j in range(8):
        i = row * 8 + j if row < 4 else (row - 4) * 8 + j
        if i >= len(pool): continue
        cr, cs, vis = pool[i]; cr = cr.copy(); dr = ImageDraw.Draw(cr)
        for q in range(4):
            if vis[q]: dr.ellipse((cs[q, 0] - 3, cs[q, 1] - 3, cs[q, 0] + 3, cs[q, 1] + 3), outline=col)
        sheet.paste(cr, (j * 196, row * 196 + (8 if row >= 4 else 0)))
sheet.save(sys.argv[1], quality=88); print("saved", sys.argv[1])
