"""Narrower field of view, simulated by cropping: the never-seen place (MonoRaceGate red) is cut to its centre 1/z and resized to the network input, so gates look z times larger and the outer gates leave the picture.
(The target camera IMX219 is about 62 deg wide; training used 130 deg, a centre zoom of about 3.6 would be the same picture size of a gate in the middle, but a crop is not a real lens: no distortion change, less detail.)
Per z: gates with all four corners inside the crop: found within 20 / 10 px (256x192 scale), median error, and the number of gates; also unmatched predictions per frame.
Usage: PFM_GDET_K=8 PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 [PFM_GDET_LEVELS=56] CKPT=runs/x KIND=hybrid [ZOOMS=1,1.5,2,3] python scripts/eval_fov_zoom.py"""
import json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); W, H = 256.0, 192.0; G = "data/real_mrg/golden"; Z = [float(z) for z in os.environ.get("ZOOMS", "1,1.5,2,3").split(",")]
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
cands = [x for x in json.load(open(f"{G}/candidates.json")) if x["seq"].startswith("red")]
for z in Z:
    errs = []; n_pred = 0; used_frames = 0
    for x in cands:
        im = Image.open(f"{G}/images/{x['id']}.jpg").convert("RGB"); w, h = im.size; lo = 0.5 - 0.5 / z
        gts = []
        for g in x["gates"]:
            c = np.array([a[:2] for a in g["corners"]], float); c = (c - lo) * z
            if all(a[2] > 0 for a in g["corners"]) and (c >= 0).all() and (c <= 1).all(): gts.append(c * [W, H])
        if not gts: continue
        used_frames += 1; crop = im.crop((lo * w, lo * h, (lo + 1 / z) * w, (lo + 1 / z) * h)).resize((448, 448), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): s = m(normalize(t.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
        preds = [s[q, 1:].view(4, 2).numpy() * [W, H] for q in range(s.shape[0]) if s[q, 0].sigmoid() > 0.5]; n_pred += len(preds)
        for c in gts: errs.append(min((min(np.linalg.norm(np.roll(p, r, 0) - c, axis=1).mean() for r in range(4)) for p in preds), default=1e9))
    e = np.array(errs); print(json.dumps({"model": CK, "zoom": z, "frames": used_frames, "gates": len(e), "<20px": round(float((e < 20).mean()), 3), "<10px": round(float((e < 10).mean()), 3), "median_err_px": round(float(np.median(e)), 1), "preds_per_frame": round(n_pred / max(used_frames, 1), 2)}), flush=True)
