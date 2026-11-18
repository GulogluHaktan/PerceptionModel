"""Real TII frames for fine-tuning: flights NOT used in the golden candidates (01,05,09,13,16,18). Partly visible gates are kept: hidden corners are null (NaN in the target, skipped by slot_loss); frames with a gate that has exactly one known corner are skipped. Frames with no gate label are skipped.
Writes data/train_real/images/real_tii_p/<flight>_<n>.jpg and labels/corners_full/real_tii_p__<flight>_<n>.json (corners_px at 256x192, TL TR BR BL, same convention as the sim)."""
import glob, json, os, subprocess, sys
from PIL import Image
R = "data/real_tii/autonomous"; O = "data/train_real_p"; EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TEST = {"01", "05", "09", "13", "16", "18"}; os.makedirs(f"{O}/images/real_tii_p", exist_ok=True); os.makedirs(f"{O}/labels/corners_full", exist_ok=True); tot = 0; flights = 0
for d in sorted(glob.glob(f"{R}/flight-*")):
    f = os.path.basename(d); num = f.split("-")[1][:2]
    if num in TEST: continue
    for z in ("camera", "label"):
        if not os.path.isdir(f"{d}/{z}_{f}"): subprocess.run(["unzip", "-q", "-o", f"{d}/{z}_{f}.zip", "-d", d], env={**os.environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"}, check=True)
    labs = sorted(glob.glob(f"{d}/label_{f}/*.txt")); n = 0; flights += 1
    for i, l in enumerate(labs):
        if i % EVERY: continue
        rows = []; ok = True
        for ln in open(l):
            v = ln.split()
            if len(v) < 17: continue
            v = [float(x) for x in v]; cs = [(v[5 + 3 * k], v[6 + 3 * k], int(v[7 + 3 * k])) for k in range(4)]
            nv = sum(c[2] > 0 for c in cs)
            if nv == 1: ok = False; break      # one known corner: cannot localise, skip the whole frame
            if nv == 0: continue
            rows.append({"gate": len(rows), "corners_px": [[round(x * 256, 2), round(y * 192, 2)] if v else None for x, y, v in cs], "visible": [bool(v) for _, _, v in cs], "n_visible": nv, "n_visible_depth": nv, "depths": [5.0] * 4})
        if not ok or not rows: continue
        stem = os.path.basename(l)[:-4]; img = glob.glob(f"{d}/camera_{f}/{stem}.*")
        if not img: continue
        sid = f"{f[:9]}_{stem[:5]}"; Image.open(img[0]).convert("RGB").save(f"{O}/images/real_tii_p/{sid}.jpg", quality=95)
        json.dump(rows, open(f"{O}/labels/corners_full/real_tii_p__{sid}.json", "w")); n += 1
    print(f, n, flush=True); tot += n
print("flights", flights, "frames", tot)
