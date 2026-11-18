"""Weak segmentation labels for real gate frames that only have corner labels (inner opening corners, no pixel mask).
Values: 1 = gate (a thin band just outside the opening: surely gate frame), 254 = surely NOT gate (class otherwise unknown), 255 = unknown (ignored).
Used for training (train.py: 254 pixels get a 'not gate' loss) and for the real-image gate-pixel check (scripts/eval_gate_seg_real.py)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

NOT_GATE, IGNORE, GATE = 254, 255, 1


def weak_gate_seg(gates, w=512, h=384, band=1.12, halo=2.4):
    """gates: list of (4,2) arrays in image-normalised coordinates, NaN = unknown corner. Returns (h,w) uint8.
    Largest gate first: a halo (opening scaled by `halo` about its centre; for a gate with unknown corners a generous box around the known ones) becomes unknown,
    the opening itself (see-through) becomes not-gate; at the end the band between the opening and `band` x opening becomes gate for every fully labelled gate."""
    im = Image.new("L", (w, h), NOT_GATE); d = ImageDraw.Draw(im); gs = []
    for c in gates:
        c = np.asarray(c, np.float32) * np.array([w, h], np.float32); known = ~np.isnan(c[:, 0])
        if known.sum() == 0: continue
        ck = c[known]; gs.append((float((ck.max(0) - ck.min(0)).max()), c, known))
    gs.sort(key=lambda t: -t[0]); sc = lambda c, s: [tuple(p) for p in (c.mean(0) + s * (c - c.mean(0)))]
    for size, c, known in gs:
        if known.all():
            d.polygon(sc(c, halo), fill=IGNORE); d.polygon(sc(c, 0.97), fill=NOT_GATE)
        else:
            ck = c[known]; m = max(1.5 * size, 0.12 * w); d.rectangle([ck[:, 0].min() - m, ck[:, 1].min() - m, ck[:, 0].max() + m, ck[:, 1].max() + m], fill=IGNORE)
    for size, c, known in gs:
        if known.all():
            ring = Image.new("L", (w, h), 0); r = ImageDraw.Draw(ring); r.polygon(sc(c, band), fill=1); r.polygon(sc(c, 1.0), fill=0); im.paste(GATE, mask=ring.point(lambda v: 255 * v))
    return np.asarray(im).copy()
