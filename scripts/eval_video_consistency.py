"""Temporal behaviour of the gate detector on REAL flight video (TII flights 13, 16, 18: about 120 frames/s, every frame labelled; these flights were never trained on, but the HALL was:
this measures frame-to-frame behaviour, not generalisation to a new place). Single-frame scores hide what matters in flight: does a gate flicker, does it drop out while the drone approaches, do the corners jitter.
Per frame: predictions with score > THR are matched to labelled gates with >= 2 visible corners (mean error over the visible corners < 20 px at 256x192, best cyclic order, one prediction per gate).
Labelled gates are linked into tracks over consecutive frames (nearest box centre, moved < 0.06 of the frame). Reported:
  recall per gate-frame; recall for fully visible (4 corners) and partly visible (2-3) gates; single-frame dropouts (missed at t, found at t-1 and t+1) per 1000 gate-frames;
  share of gate-frames inside miss runs of >= 5 and >= 12 frames (40 / 100 ms); recall if a detection is held for up to HOLD frames (what a trivial tracker would give);
  corner jitter = |(pred - label)_t - (pred - label)_{t-1}| over consecutive found frames (median, p90; removes the true motion); recall in the last 12 frames (100 ms) before a gate leaves the labels;
  unmatched predictions per frame. Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid [FLIGHTS=13,16,18] [STEP=1] python scripts/eval_video_consistency.py"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); R = "data/real_tii/autonomous"; W, H = 256.0, 192.0; THR = float(os.environ.get("THR", "0.5")); STEP = int(os.environ.get("STEP", "1")); HOLD = int(os.environ.get("HOLD", "3")); SIZE = int(os.environ.get("SIZE", "448")); ZOOM = os.environ.get("ZOOM") == "1"
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
def best(p, g, vis):
    c = [(np.linalg.norm((np.roll(p, s, 0) - g)[vis], axis=1).mean(), s) for s in range(4)]; return min(c)
tot = {"gf": 0, "found": 0, "full": [0, 0], "part": [0, 0], "drop1": 0, "run5": 0, "run12": 0, "hold": 0, "last": [0, 0], "frames": 0, "extra": 0}; jit = []; per_flight = {}; BINS = [(0, 8), (8, 16), (16, 40), (40, 100), (100, 1e9)]; bins = [[0, 0, 0, 0] for _ in BINS]   # by gate size (largest side of the visible corners, px at 256x192): found, gate-frames, gate-frames in miss runs >= 12
for fl in os.environ.get("FLIGHTS", "13,16,18").split(","):
    d = glob.glob(f"{R}/flight-{fl}*")[0]; f = os.path.basename(d); labs = sorted(glob.glob(f"{d}/label_{f}/*.txt"))[::STEP]; tracks = []; open_tr = []; n0 = tot["gf"]; f0 = tot["found"]
    for t, l in enumerate(labs):
        gts = []
        for ln in open(l):
            v = ln.split()
            if len(v) < 17: continue
            v = [float(x) for x in v]; cs = np.array([[v[5 + 3 * k] * W, v[6 + 3 * k] * H] for k in range(4)]); vis = np.array([v[7 + 3 * k] > 0 for k in range(4)])
            if vis.sum() >= 2: gts.append((np.array([v[1], v[2]]), cs, vis))
        img = glob.glob(f"{d}/camera_{f}/{os.path.basename(l)[:-4]}.*")
        if not img: continue
        im = Image.open(img[0]).convert("RGB")
        def run(pic):
            x = torch.from_numpy(np.asarray(pic.resize((SIZE, SIZE), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
            with torch.no_grad(): s = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
            return [s[q, 1:].view(4, 2).numpy() for q in range(s.shape[0]) if s[q, 0].sigmoid() > THR]
        preds = [q * [W, H] for q in run(im)]
        if ZOOM:      # second pass on the centre half of the frame (2x zoom, diagnostic for far gates); keep gates fully inside the crop that the full pass did not give
            iw, ih = im.size
            for q in run(im.crop((iw // 4, ih // 4, iw - iw // 4, ih - ih // 4))):
                if q.min() < 0.03 or q.max() > 0.97: continue
                q = (q * 0.5 + 0.25) * [W, H]
                if all(np.linalg.norm(q.mean(0) - o.mean(0)) > 6 for o in preds): preds.append(q)
        used = set(); tot["frames"] += 1; new_open = []
        for ctr, cs, vis in gts:
            cand = sorted((best(p, cs, vis) + (i,) for i, p in enumerate(preds) if i not in used), key=lambda z: z[0]); hit = bool(cand) and cand[0][0] < 20
            resid = None
            if hit: used.add(cand[0][2]); resid = (np.roll(preds[cand[0][2]], cand[0][1], 0) - cs)
            # link to a track
            j = min(range(len(open_tr)), key=lambda q: np.linalg.norm(open_tr[q]["ctr"] - ctr), default=None)
            if j is not None and np.linalg.norm(open_tr[j]["ctr"] - ctr) < 0.06 and open_tr[j]["t"] == t - 1: tr = open_tr.pop(j)
            else: tr = {"hits": [], "full": [], "res": [], "vis": []}; tracks.append(tr)
            tr.setdefault("size", []).append(float(np.ptp(cs[vis], axis=0).max())); tr.setdefault("rel", []).append(bool(cand) and cand[0][0] < max(20.0, 0.25 * float(np.ptp(cs[vis], axis=0).max()))); tr["ctr"] = ctr; tr["t"] = t; tr["hits"].append(hit); tr["full"].append(bool(vis.all())); tr["res"].append(resid); tr["vis"].append(vis); new_open.append(tr)
        open_tr = new_open; tot["extra"] += len(preds) - len(used)
    for tr in tracks:
        h = np.array(tr["hits"]); fu = np.array(tr["full"]); n = len(h); tot["gf"] += n; tot["found"] += int(h.sum())
        tot["full"][0] += int(h[fu].sum()); tot["full"][1] += int(fu.sum()); tot["part"][0] += int(h[~fu].sum()); tot["part"][1] += int((~fu).sum())
        sz = np.array(tr["size"]); longmiss = np.zeros(n, bool)
        for bi, (lo, hi) in enumerate(BINS):
            mk = (sz >= lo) & (sz < hi); bins[bi][0] += int(h[mk].sum()); bins[bi][1] += int(mk.sum()); bins[bi][3] += int(np.array(tr["rel"])[mk].sum())
        if n >= 3: tot["drop1"] += int((~h[1:-1] & h[:-2] & h[2:]).sum())
        i = 0
        while i < n:      # miss runs
            if h[i]: i += 1; continue
            j = i
            while j < n and not h[j]: j += 1
            if j - i >= 5: tot["run5"] += j - i
            if j - i >= 12: tot["run12"] += j - i; longmiss[i:j] = True
            i = j
        for bi, (lo, hi) in enumerate(BINS): bins[bi][2] += int((longmiss & (sz >= lo) & (sz < hi)).sum())
        tot["hold"] += int(sum(h[max(0, q - HOLD):q + 1].any() for q in range(n)))
        if n >= 24: tot["last"][0] += int(h[-12:].sum()); tot["last"][1] += 12
        for q in range(1, n):
            if h[q] and h[q - 1]:
                both = tr["vis"][q] & tr["vis"][q - 1]
                if both.any(): jit.extend(np.linalg.norm((tr["res"][q] - tr["res"][q - 1])[both], axis=1).tolist())
    per_flight[fl] = {"gate_frames": tot["gf"] - n0, "recall": round((tot["found"] - f0) / max(tot["gf"] - n0, 1), 3), "tracks": len(tracks)}
g = max(tot["gf"], 1); j = np.array(jit)
out = {"frames": tot["frames"], "gate_frames": tot["gf"], "recall<20px": round(tot["found"] / g, 3), "recall_full": round(tot["full"][0] / max(tot["full"][1], 1), 3), "recall_partly": round(tot["part"][0] / max(tot["part"][1], 1), 3), "n_partly": tot["part"][1],
       "single_frame_dropouts_per_1000": round(1000 * tot["drop1"] / g, 1), "share_in_miss_runs>=5": round(tot["run5"] / g, 3), "share_in_miss_runs>=12": round(tot["run12"] / g, 3), f"recall_with_hold_{HOLD}": round(tot["hold"] / g, 3),
       "recall_last_12_frames_of_track": round(tot["last"][0] / max(tot["last"][1], 1), 3), "corner_jitter_px_median": round(float(np.median(j)), 2), "corner_jitter_px_p90": round(float(np.percentile(j, 90)), 2),
       "unmatched_pred_per_frame": round(tot["extra"] / max(tot["frames"], 1), 3), "per_flight": per_flight, "by_gate_size_px": {f"{int(lo)}-{int(hi) if hi < 1e8 else 'up'}": {"gate_frames": b[1], "recall": round(b[0] / max(b[1], 1), 3), "recall_within_max(20px,size/4)": round(b[3] / max(b[1], 1), 3), "share_of_long_misses": round(b[2] / max(tot["run12"], 1), 3)} for (lo, hi), b in zip(BINS, bins)}}
print(f"{CK} {K} video consistency (TII flights, hall seen, flights not): {json.dumps(out)}")
