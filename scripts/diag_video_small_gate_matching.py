"""Why are small real gates 'missed' on the TII video? Per size bin: recall with one prediction per label (closest pairs first), recall when a prediction may serve several labels,
and for labels that lose only because their prediction is taken: distance to the label that took it. Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid [STEP=4] python scripts/diag_video_small_gate_matching.py"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); R = "data/real_tii/autonomous"; W, H = 256.0, 192.0; STEP = int(os.environ.get("STEP", "4"))
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
BINS = [(0, 8), (8, 16), (16, 40), (40, 100), (100, 1e9)]; st = [{"n": 0, "one": 0, "any": 0, "twin": []} for _ in BINS]
def err(p, cs, vis): return min(np.linalg.norm((np.roll(p, r, 0) - cs)[vis], axis=1).mean() for r in range(4))
for fl in "13,16,18".split(","):
    d = glob.glob(f"{R}/flight-{fl}*")[0]; f = os.path.basename(d)
    for l in sorted(glob.glob(f"{d}/label_{f}/*.txt"))[::STEP]:
        gts = []
        for ln in open(l):
            v = ln.split()
            if len(v) < 17: continue
            v = [float(x) for x in v]; cs = np.array([[v[5 + 3 * k] * W, v[6 + 3 * k] * H] for k in range(4)]); vis = np.array([v[7 + 3 * k] > 0 for k in range(4)])
            if vis.sum() >= 2: gts.append((cs, vis))
        img = glob.glob(f"{d}/camera_{f}/{os.path.basename(l)[:-4]}.*")
        if not gts or not img: continue
        x = torch.from_numpy(np.asarray(Image.open(img[0]).convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): s = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
        preds = [s[q, 1:].view(4, 2).numpy() * [W, H] for q in range(s.shape[0]) if s[q, 0].sigmoid() > 0.5]
        E = np.array([[err(p, cs, vis) for p in preds] for cs, vis in gts]).reshape(len(gts), len(preds)); one = np.zeros(len(gts), bool)
        if len(preds):
            up = set()      # one prediction per label, closest pairs first
            for i_, j_ in sorted(((i_, j_) for i_ in range(len(gts)) for j_ in range(len(preds)) if E[i_, j_] < 20), key=lambda q: E[q]):
                if not one[i_] and j_ not in up: one[i_] = True; up.add(j_)
        for i, (cs, vis) in enumerate(gts):
            sz = float(np.ptp(cs[vis], axis=0).max()); b = next(k for k, (lo, hi) in enumerate(BINS) if lo <= sz < hi); a = len(preds) > 0 and E[i].min() < 20
            st[b]["n"] += 1; st[b]["one"] += int(one[i]); st[b]["any"] += int(a)
            if a and not one[i]:
                st[b]["twin"].append(min(np.linalg.norm(cs[vis].mean(0) - o[0][o[1]].mean(0)) for j, o in enumerate(gts) if j != i))
for (lo, hi), s_ in zip(BINS, st):
    t = np.array(s_["twin"]); print(json.dumps({"size_px": f"{lo}-{hi}", "labels": s_["n"], "recall_one_pred_per_label": round(s_["one"] / max(1, s_["n"]), 3), "recall_pred_may_be_shared": round(s_["any"] / max(1, s_["n"]), 3),
        "lost_to_a_neighbour_label": len(t), "median_px_to_that_neighbour": round(float(np.median(t)), 1) if len(t) else None}))
