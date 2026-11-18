"""Weak segmentation labels (pfm.gate_seg) for the real gate frames of the training mix, which had corner labels only: without them the segmentation head never saw a real race hall
and painted ceilings as 'gate' (seen in the demo video, Update 45). Reads data/mix/labels/corners_full/<prefix>__*.json (corners at 256x192, None = hidden corner; gate-free crops have an empty list),
writes data/general/train/labels_v2/seg/<id>.png (512x384; 1 = gate band, 254 = not gate, 255 = unknown). Nothing existing is changed."""
import glob, json, os
import numpy as np
from PIL import Image
import sys; sys.path.insert(0, "src")
from pfm.gate_seg import weak_gate_seg
out = "data/general/train/labels_v2/seg"; os.makedirs(out, exist_ok=True); n = 0; stat = np.zeros(3)
for pre in ("real_tii_p", "real_tii_neg", "real_mrg", "real_mrg_neg"):
    for f in sorted(glob.glob(f"data/mix/labels/corners_full/{pre}__*.json")):
        rows = json.load(open(f)); gates = [np.array([[np.nan, np.nan] if p is None else p for p in r["corners_px"]], np.float32) / np.array([256.0, 192.0], np.float32) for r in rows]
        s = weak_gate_seg(gates); Image.fromarray(s).save(f"{out}/{os.path.basename(f)[:-5]}.png"); n += 1; stat += [(s == 1).mean(), (s == 254).mean(), (s == 255).mean()]
print("frames", n, "mean share gate band / not gate / unknown:", (stat / n).round(4))
