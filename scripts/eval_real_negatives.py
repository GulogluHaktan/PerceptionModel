"""False-alarm checks on real images WITHOUT gates: (a) golden frames with every labelled gate box painted over (box +15%), (b) 200 real Aeroscapes drone photos (outdoor, no gates).
Reports share of images with any slot score > 0.5 and mean max score. Usage: python scripts/eval_real_negatives.py ckpt1 ckpt2 ..."""
import glob, json, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
G = "data/real_tii/golden"; masked = []
for c in json.load(open(f"{G}/candidates.json")):
    im = np.asarray(Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB")).copy(); H, W = im.shape[:2]
    for g in c["gates"]:
        cx, cy, w, h = g["box"]; x0, x1, y0, y1 = [int(v) for v in (max((cx - w * .575) * W, 0), min((cx + w * .575) * W, W), max((cy - h * .575) * H, 0), min((cy + h * .575) * H, H))]
        if x1 > x0 and y1 > y0: im[y0:y1, x0:x1] = im.reshape(-1, 3).mean(0).astype(np.uint8)
    masked.append(im)
aero = sorted(glob.glob("data/images/aeroscapes/*"))[::10][:200]
neg = sorted(glob.glob("data/real_tii/neg_crops/*.jpg"))
sets = {"real gate-free crops (golden flights)": [np.asarray(Image.open(p).convert("RGB")) for p in neg], "golden frames, gate boxes painted out": masked, "real Aeroscapes photos (no gates)": [np.asarray(Image.open(p).convert("RGB")) for p in aero]}
import os; KIND = os.environ.get("KIND", "vit")
for nm, pat in (("BDD100K val frames (dashcam, no gates, never trained on)", "data/general/eval_bdd/images/*.jpg"), ("ADE20K outdoor held-out frames (no gates)", "data/general/eval_ade/images/*.jpg")):     # held-out general sets: independent of the gate-free training negatives
    fs = sorted(glob.glob(pat))[:300]
    if fs: sets[nm] = [np.asarray(Image.open(p).convert("RGB")) for p in fs]
tx = lambda ims: torch.stack([torch.from_numpy(np.asarray(Image.fromarray(i).resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for i in ims])
for ck in sys.argv[1:]:
    m = PerceptionModel(KIND, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{ck}/{KIND}/final.pt", map_location="cpu")); m = m.cuda().eval()
    for name, ims in sets.items():
        X = tx(ims); mx = []
        for i in range(0, len(X), 16):
            with torch.no_grad(): mx += m(normalize(X[i:i + 16].cuda()), with_teacher_heads=False)["slots"][:, :, 0].float().sigmoid().max(1).values.cpu().tolist()
        mx = np.array(mx); print(f"{ck:20s} {name:40s} n={len(mx)} false gate (score>0.5): {(mx > 0.5).mean():.2f}  mean max score {mx.mean():.2f}", flush=True)
