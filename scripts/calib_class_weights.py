"""Per-class score weights for the indoor / outdoor module (Update 167): seg score of class c is multiplied by w_c before the argmax, so
over-predicted classes (extra labels) are damped and under-predicted ones (missing labels) boosted. Weights are tuned on the training
script's val split (camera-degraded 2.66:1 crops at 288x768, coordinate search over log w), then reported on the untouched test set
(clean and camera-degraded). The weights fold into the exported graph's class matrix A (export_m2f_indoor.py CLASS_W=...): zero cost.
Run: env -u PYTHONPATH CLASSSET=in|out CK=<model.pt> .venv/bin/python scripts/calib_class_weights.py  -> <ck dir>/class_weights.json"""
import os, sys, json, random, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
os.environ["CAMWIDE"] = "1"; sys.argv = ["x"]; sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_m2f_indoor as FT
B = FT.B; K = FT.K_IN; dev = "cuda"; NAMES = FT.IN_CLASSES
CK = Path(os.environ["CK"]); m = FT.build(); m.load_state_dict(torch.load(CK, map_location="cpu")); m = m.to(dev).eval()
real, _ = B.pool() if B.CLASSSET == "out" else FT.pool(); random.seed(0); random.shuffle(real); val = real[:300]
test = [(p, Path(f"data/general/{B.TESTDIR}/seg") / (p.stem + ".png")) for p in sorted(Path(f"data/general/{B.TESTDIR}/images").glob("*.jpg"))]
DS = (72, 192)                                    # scores / labels cached at 1/4 resolution


def scores(pairs, cam):
    S, Y = [], []
    with torch.no_grad():
        for i, (p, lp) in enumerate(pairs):
            x, y = FT.wide_pair(p, lp, False, False)
            if cam: random.seed(1000 + i); torch.manual_seed(1000 + i); x = FT.cam_aug(x)
            s = F.interpolate(FT.predict(m, ((x - FT.MEAN) / FT.STD)[None].to(dev)), size=DS, mode="bilinear", align_corners=False)[0]
            S.append(s.half().cpu()); Y.append(F.interpolate(y[None, None].float(), size=DS, mode="nearest")[0, 0].long())
    return torch.stack(S).to(dev).float(), torch.stack(Y).to(dev)


def metrics(S, Y, w):
    pr = (S * w.view(1, K, 1, 1)).argmax(1); v = (Y != 255) & (Y < K)
    cm = torch.bincount((Y[v] * K + pr[v]), minlength=K * K).view(K, K).double()
    iu = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp(min=1); pres = cm.sum(1) > 0
    n = v.flatten(1).sum(1).clamp(min=1).double(); miss = gtc = extra = prc = 0
    for c in range(K):
        g = ((Y == c) & v).flatten(1).sum(1).double(); p = ((pr == c) & v).flatten(1).sum(1).double(); hit = ((Y == c) & (pr == c)).flatten(1).sum(1).double()
        gi = g >= 0.01 * n; pi = p >= 0.01 * n
        gtc += int(gi.sum()); miss += int((gi & (hit < 0.2 * g)).sum()); prc += int(pi.sum()); extra += int((pi & (g == 0)).sum())
    return float(iu[pres].mean()), miss / max(1, gtc), extra / max(1, prc), iu.cpu().numpy()


def objective(S, Y, w):
    mi, mr, er, _ = metrics(S, Y, w); return mi - 0.25 * (mr + er)       # mIoU, plus a push on the missing / extra rates the user cares about


Sv, Yv = scores(val, True); w = torch.ones(K, device=dev); best = objective(Sv, Yv, w); print("val start", round(best, 4), metrics(Sv, Yv, w)[:3], flush=True)
for rnd in range(3):
    for c in range(K):
        for f in (0.5, 0.7, 0.85, 1.2, 1.4, 2.0):
            w2 = w.clone(); w2[c] *= f; o = objective(Sv, Yv, w2)
            if o > best + 1e-4: best, w = o, w2
    print("round", rnd, round(best, 4), flush=True)
w = w / w.max()
res = {}
for cam in (False, True):
    St, Yt = scores(test, cam); a = metrics(St, Yt, torch.ones(K, device=dev)); b = metrics(St, Yt, w)
    res["cam" if cam else "clean"] = dict(before=dict(miou=round(a[0], 4), missing=round(a[1], 3), extra=round(a[2], 3)), after=dict(miou=round(b[0], 4), missing=round(b[1], 3), extra=round(b[2], 3)))
    print("test", "camera-degraded" if cam else "clean", res["cam" if cam else "clean"], flush=True)
    del St, Yt
out = dict(classes=NAMES, weights=[round(float(v), 4) for v in w], test=res)
(CK.parent / "class_weights.json").write_text(json.dumps(out, indent=1)); print("weights", {n: round(float(v), 2) for n, v in zip(NAMES, w) if abs(float(v) - 1) > 0.05})
