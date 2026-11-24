"""Train the add-on whole-gate head for close / cut gates (src/pfm/cornerpeak.py NearGateHead) on the FROZEN main model, DroneRacing on-policy crops
(training seeds only; corners in simulator identity order, also outside the image). Logs held-out AND training-subset metrics every eval (overfitting check).

env: CROPS, VAL_SEED, CROP_STRIDE, ITERS, BS, LR, P_ROT, ROT_MAX
Run: ORDER=label P_ROT=0.7 env -u PYTHONPATH .venv/bin/python scripts/train_neargate.py runs/neargate_v1
"""
import json, os, sys, time
from pathlib import Path
os.environ["ORDER"] = "label"; os.environ.setdefault("P_ROT", "0.7")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
import train_cornerpeak as T
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.cornerpeak import NearGateHead, fused_features, IN, STRIDE

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/neargate_v1"); OUT.mkdir(parents=True, exist_ok=True)
G = IN // STRIDE; dev = "cuda"; ITERS = T.ITERS; BS = T.BS; LR = T.LR


def targets(C, V):
    """C [B,Gt,4,2] px@448 (all corners), V [B,Gt,4] visible -> heat [B,G,G], off [B,8,G,G] (NaN = no target). Built in numpy on the CPU (many tiny writes)."""
    import numpy as np
    Cn = (C / IN).cpu().numpy(); Vn = V.cpu().numpy(); B = Cn.shape[0]
    heat = np.zeros((B, G, G), np.float32); off = np.full((B, 8, G, G), np.nan, np.float32); ys, xs = np.mgrid[0:G, 0:G]
    for b in range(B):
        owned = np.zeros((G, G), bool)
        for g in range(Cn.shape[1]):
            if Vn[b, g].sum() < 2: continue
            c = Cn[b, g]; ctr = np.clip(c.mean(0), 0, 1 - 1e-4); cj, ci = int(ctr[0] * G), int(ctr[1] * G)
            ext = float((c.max(0) - c.min(0)).max()) * G; sig = max(1.0, min(ext, 2.0 * G) / 6)
            heat[b] = np.maximum(heat[b], np.exp(-((xs - cj) ** 2 + (ys - ci) ** 2) / (2 * sig ** 2))); heat[b, ci, cj] = 1
            for i in range(max(ci - 1, 0), min(ci + 2, G)):
                for j in range(max(cj - 1, 0), min(cj + 2, G)):
                    if owned[i, j]: continue
                    owned[i, j] = True; off[b, :, i, j] = (c - np.array([(j + 0.5) / G, (i + 0.5) / G], np.float32)).reshape(-1)
    return torch.from_numpy(heat).to(C.device), torch.from_numpy(off).to(C.device)


def loss_fn(raw, heat, off, w_off=10.0):
    p = raw[:, 0].float().sigmoid().clamp(1e-4, 1 - 1e-4); pos = heat >= 0.999
    focal = -(((1 - p) ** 2 * p.log())[pos].sum() + ((1 - heat) ** 4 * p ** 2 * (1 - p).log())[~pos].sum()) / pos.sum().clamp(min=1)
    d = (raw[:, 1:].float() - off).abs(); d = d[torch.isfinite(d)]
    return focal + w_off * (d.mean() if d.numel() else raw.sum() * 0)


@torch.no_grad()
def evaluate(model, head, imgs, cors, viss, thr=0.3):
    head.eval(); st = {}; fp = 0; nf = 0
    for i0 in range(0, len(imgs), 64):
        ii = torch.arange(i0, min(len(imgs), i0 + 64)); x, C, V = T.batch(imgs, cors, viss, ii)
        with torch.autocast("cuda", torch.bfloat16):
            raw = head(x, fused_features(model, normalize(x)))
        S = model.dense.decode_gdet(raw.float())       # (B,K,9) logit + corners normalised
        for b in range(len(ii)):
            dets = S[b][S[b, :, 0].sigmoid() > thr]; pc = dets[:, 1:].view(-1, 4, 2) * IN; used = torch.zeros(len(dets), dtype=torch.bool, device=dev); nf += 1
            for g in range(C.shape[1]):
                if V[b, g].sum() < 2: continue
                size = float((C[b, g].amax(0) - C[b, g].amin(0)).max()) * 384 / IN
                key = "0-60" if size < 60 else "60-120" if size < 120 else "120-250" if size < 250 else "250+"
                e = st.setdefault(key, [0, 0]); e[0] += 1
                if len(dets):
                    dv = ((pc - C[b, g][None]).norm(dim=-1) * V[b, g][None]).sum(-1) / V[b, g].sum() * 384 / IN      # mean error over visible corners, 384 px
                    k = int(dv.argmin())
                    if dv[k] < max(4.0, 0.05 * size): e[1] += 1; used[k] = True
                    da = (pc - C[b, g][None]).norm(dim=-1).mean(-1) * 384 / IN
                    used |= da < 0.25 * size
            fp += int((~used).sum())
    head.train()
    return {k: [v[0], round(v[1] / v[0], 3)] for k, v in sorted(st.items())}, round(fp / max(1, nf), 3)


def main():
    seeds = set(range(1, 100))
    tr = T.load(seeds - {T.VAL_SEED}, T.CROP_STRIDE); va = T.load({T.VAL_SEED}, 3); trs = T.load({5, 18}, 6)
    print(f"train {len(tr[0])}, held-out {len(va[0])}, train-subset {len(trs[0])}", flush=True)
    model = PerceptionModel("hybrid", teacher_dims={"dino": 384}); model.load_state_dict(torch.load(T.CKPT, map_location="cpu")); model = model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    head = NearGateHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    t0 = time.time(); N = len(tr[0]); log = open(OUT / "log.jsonl", "a")
    for it in range(1, ITERS + 1):
        ii = torch.randint(0, N, (BS,)); x, C, V = T.batch(*tr, ii); x, C = T.augment(x, C)
        heat, off = targets(C, V)
        with torch.autocast("cuda", torch.bfloat16):
            raw = head(x, fused_features(model, normalize(x)))
        loss = loss_fn(raw, heat, off)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if it % 1000 == 0 or it == ITERS:
            rv, fv = evaluate(model, head, *va); rt, ft = evaluate(model, head, *trs)
            rec = dict(iter=it, sec=int(time.time() - t0), loss=round(loss.item(), 4), heldout_recall=rv, heldout_false_per_frame=fv, train_recall=rt, train_false_per_frame=ft)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save(head.state_dict(), OUT / "neargate.pt")
            if it % 5000 == 0: torch.save(head.state_dict(), OUT / f"neargate_{it}.pt")


if __name__ == "__main__":
    main()
