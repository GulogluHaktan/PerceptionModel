"""Label rendered 3DGS frames and check them with geometry (decision 6: geometric reprojection check before accepting labels).
Usage: python scripts/label_consistency.py <orbit_dir> <frame_ids,comma> <out_dir>
<orbit_dir>: rgb/NNNN.png, depth/NNNN.png (16-bit, value/1000 = scene units), transforms_orbit.json (OpenGL camera-to-world).
Each frame is labelled with Grounding DINO + SAM2 (src/pfm/labeler.py, prompts overridden for the lego demo). A labelled pixel of
frame i is moved into frame j with depth_i and the two poses; it is ACCEPTED if the visible target pixel carries the same class,
REJECTED if it carries another class, and ignored where it is occluded or unlabelled in j. Prints agreement stats, writes a sheet."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "src")
import pfm.labeler as L
from pfm.model import CLASSES

L.PROMPTS = {"car": ["bulldozer", "toy truck", "yellow lego bulldozer", "lego excavator", "construction vehicle"], "ground_road": ["lego baseplate", "flat plate"]}   # demo prompts for the lego scene (more car phrases: the first version missed the bulldozer in some views)
root, ids, out = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
meta = json.load(open(root / "transforms_orbit.json"))
fr = {Path(f["file_path"]).name: f for f in meta["frames"]}
im0 = Image.open(root / "rgb" / f"{ids[0]}.png"); W = im0.size[0]
fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"]); cx = cy = W / 2
lab = L.Labeler("cuda", box_thr=0.22); data = {}      # lower box threshold: recover bulldozer detections that were dropped at 0.3
for i in ids:
    img = Image.open(root / "rgb" / f"{i}.png").convert("RGB")
    seg, _ = lab.label(img)
    d = np.asarray(Image.open(root / "depth" / f"{i}.png")).astype(np.float32) / 1000
    c2w = np.array(fr[i]["transform_matrix"], dtype=np.float64); c2w[:3, 1:3] *= -1          # OpenGL -> OpenCV
    data[i] = dict(img=np.asarray(img), seg=seg, depth=d, c2w=c2w)
    np.save(out / f"labels_{i}.npy", seg)
    print("labelled", i, {CLASSES[c]: int((seg == c).sum()) for c in np.unique(seg) if c != 255}, flush=True)
uu, vv = np.meshgrid(np.arange(W), np.arange(W))


def check(i, j):
    a, b = data[i], data[j]
    z = a["depth"]; m = (z > 0.5) & (a["seg"] != 255)
    pc = np.stack([(uu - cx) / fx * z, (vv - cy) / fx * z, z, np.ones_like(z)], -1)[m]
    pj = (np.linalg.inv(b["c2w"]) @ (a["c2w"] @ pc.T)).T
    zj = pj[:, 2]; u = np.round(fx * pj[:, 0] / zj + cx).astype(int); v = np.round(fx * pj[:, 1] / zj + cy).astype(int)
    ok = (zj > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < W)
    res = np.zeros(m.sum(), np.int8)                                   # 0 unseen/ignored, 1 agree, -1 disagree
    uo, vo = u[ok], v[ok]
    vis = np.abs(b["depth"][vo, vo * 0 + uo] - zj[ok]) < 0.04 * zj[ok] if False else np.abs(b["depth"][vo, uo] - zj[ok]) < 0.04 * zj[ok]
    lj = b["seg"][vo, uo]; li = a["seg"][m][ok]
    use = vis & (lj != 255)
    r = np.zeros(ok.sum(), np.int8); r[use & (lj == li)] = 1; r[use & (lj != li)] = -1
    res[ok] = r
    full = np.zeros((W, W), np.int8); full[m] = res
    return full


stats, rows = [], []
for k, i in enumerate(ids):
    nb = [ids[x] for x in (k - 1, k + 1) if 0 <= x < len(ids)]
    chk = [check(i, j) for j in nb]
    agree = sum((c == 1) for c in chk); dis = sum((c == -1) for c in chk)
    accepted = (agree > 0) & (dis == 0); rejected = dis > 0
    lab_px = int((data[i]["seg"] != 255).sum()); tested = int(((agree + dis) > 0).sum())
    stats.append({"frame": i, "label_px": lab_px, "checked_px": tested, "accepted_px": int(accepted.sum()), "rejected_px": int(rejected.sum()),
                  "agreement_of_checked": round(float(accepted.sum() / max(1, accepted.sum() + rejected.sum())), 3)})
    pal = np.array([[60, 60, 60], [255, 0, 0], [0, 140, 0], [255, 200, 0], [120, 120, 120], [0, 0, 255], [255, 0, 255], [0, 255, 255], [160, 80, 0], [255, 255, 255], [255, 128, 128]], np.uint8)
    sg = data[i]["seg"]; col = np.where((sg == 255)[..., None], data[i]["img"], pal[np.minimum(sg, 10)])
    ck = np.full((W, W, 3), 235, np.uint8); ck[(sg != 255)] = (150, 150, 150); ck[accepted] = (40, 170, 70); ck[rejected] = (220, 40, 40)
    rows.append(np.concatenate([data[i]["img"], (0.5 * data[i]["img"] + 0.5 * col).astype(np.uint8), ck], 1))
    print(json.dumps(stats[-1]), flush=True)
sheet = Image.fromarray(np.concatenate(rows, 0)); sheet.resize((sheet.width * 360 // W, sheet.height * 360 // W), Image.LANCZOS).save(out / "consistency_sheet.png")
tot = {k: sum(s[k] for s in stats) for k in ("label_px", "checked_px", "accepted_px", "rejected_px")}
tot["agreement_of_checked"] = round(tot["accepted_px"] / max(1, tot["accepted_px"] + tot["rejected_px"]), 3)
json.dump({"frames": stats, "total": tot}, open(out / "stats.json", "w")); print("TOTAL", json.dumps(tot))
