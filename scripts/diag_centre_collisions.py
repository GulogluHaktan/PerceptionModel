"""How often do two labelled gates share (almost) the same centre, and are those the missed ones? The dense head gives one gate per grid cell, so nested gates (one seen through another)
with centres closer than about one cell (1/112 of the image at stride 4) cannot both come out. Sets: the never-seen place (MonoRaceGate red) and the TII video flights.
Usage: PFM_GDET_K=8 PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid python scripts/diag_centre_collisions.py"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); W, H = 256.0, 192.0
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
def frames():
    G = "data/real_mrg/golden"
    for x in json.load(open(f"{G}/candidates.json")):
        if x["seq"].startswith("red"): yield "red", f"{G}/images/{x['id']}.jpg", [(np.array([a[:2] for a in g["corners"]], float), np.array([a[2] > 0 for a in g["corners"]])) for g in x["gates"]]
    R = "data/real_tii/autonomous"
    for fl in "13,16,18".split(","):
        d = glob.glob(f"{R}/flight-{fl}*")[0]; f = os.path.basename(d)
        for l in sorted(glob.glob(f"{d}/label_{f}/*.txt"))[::8]:
            gts = []
            for ln in open(l):
                v = ln.split()
                if len(v) < 17: continue
                v = [float(q) for q in v]; gts.append((np.array([[v[5 + 3 * k], v[6 + 3 * k]] for k in range(4)]), np.array([v[7 + 3 * k] > 0 for k in range(4)])))
            img = glob.glob(f"{d}/camera_{f}/{os.path.basename(l)[:-4]}.*")
            if img: yield "tii_video", img[0], gts
st = {}
for name, path, gts in frames():
    gts = [(c * [W, H], v) for c, v in gts if v.sum() >= 2]
    if not gts: continue
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): s = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
    preds = [s[q, 1:].view(4, 2).numpy() * [W, H] for q in range(s.shape[0]) if s[q, 0].sigmoid() > 0.5]; used = set(); ctr = [c[v].mean(0) for c, v in gts]
    for i, (c, v) in enumerate(gts):
        e = sorted((min(np.linalg.norm((np.roll(p, r, 0) - c)[v], axis=1).mean() for r in range(4)), j) for j, p in enumerate(preds) if j not in used); hit = bool(e) and e[0][0] < 20
        if hit: used.add(e[0][1])
        dmin = min((np.linalg.norm(ctr[i] - ctr[j]) for j in range(len(gts)) if j != i), default=1e9)      # px at 256x192; one stride-4 cell is 2.3 px wide
        b = "centre of another gate within 2 cells (<4.6 px)" if dmin < 4.6 else "within 2-5 cells" if dmin < 11.5 else "no close centre"
        a = st.setdefault((name, b), [0, 0]); a[0] += int(hit); a[1] += 1
for (name, b), (f, n) in sorted(st.items()): print(json.dumps({"set": name, "gates": b, "n": n, "found<20px": round(f / n, 3), "missed": int(n - f)}))
