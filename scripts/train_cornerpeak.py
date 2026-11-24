"""Train the single-corner peak head (src/pfm/cornerpeak.py) on the FROZEN main model, from DroneRacing on-policy crops (training seeds only).

env: CROPS (comma dirs), VAL_SEED (held out), CROP_STRIDE, ITERS, BS, CKPT (main model), LR
Run: env -u PYTHONPATH .venv/bin/python scripts/train_cornerpeak.py runs/cornerpeak_v1
"""
import json, math, os, random, sys, time
from pathlib import Path

for k, v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(k, v)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch, torch.nn.functional as F
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.cornerpeak import CornerPeakHead, fused_features, decode_peaks, IN, STRIDE

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/cornerpeak_v1"); OUT.mkdir(parents=True, exist_ok=True)
CROPS = [Path(os.path.expanduser(c)) for c in os.environ.get("CROPS", "~/DroneRacing_eval/runs_crops/ours,~/DroneRacing_eval/runs_crops/ours_aug").split(",")]
VAL_SEED = int(os.environ.get("VAL_SEED", "24")); CROP_STRIDE = int(os.environ.get("CROP_STRIDE", "2"))
ITERS = int(os.environ.get("ITERS", "15000")); BS = int(os.environ.get("BS", "32")); LR = float(os.environ.get("LR", "2e-3")); dev = "cuda"
CKPT = os.path.expanduser(os.environ.get("CKPT", "runs/release_v4/bigY_h_hybrid.pt"))
G = IN // STRIDE   # 112
P_ROT = float(os.environ.get("P_ROT", "0")); ROT_MAX = float(os.environ.get("ROT_MAX", "45"))
ORDER = os.environ.get("ORDER", "geo")      # geo: types = geometric tl,tr,br,bl in the image | label: types = the simulator corner identity (what the EKF matches against its map corners); no flips then


def load(seeds, stride):
    imgs, cors, viss = [], [], []
    for f in sorted(f for c in CROPS for f in c.glob("data_*.pt")):
        sd = int(f.stem.split("_")[1])
        if sd not in seeds: continue
        d = torch.load(f, map_location="cpu")
        cf = d["corners"].float()
        sane = torch.isfinite(cf).all(-1).all(-1) & (cf.abs() < 8.0).all(-1).all(-1)      # behind-camera projections are garbage (type order unusable)
        v = d["vis"] & (d["slot_valid"] & sane)[..., None]
        keep = v.any(-1).any(-1).nonzero().squeeze(1)[::stride]
        imgs.append(d["img"][keep]); cors.append(torch.nan_to_num(cf[keep], 0.0) * IN); viss.append(v[keep])     # corners in 448 px
        print(f"crops {f.parent.name} seed {sd}: {len(keep)} frames", flush=True)
    return torch.cat(imgs), torch.cat(cors), torch.cat(viss)


def geo_order(c):
    """c [...,4,2] -> permutation idx [...,4]: index of the corner that is tl, tr, br, bl (angle around centroid, start = smallest x+y)."""
    m = c.mean(-2, keepdim=True); ang = torch.atan2(c[..., 1] - m[..., 1], c[..., 0] - m[..., 0])
    idx = ang.argsort(-1); cs = torch.gather(c[..., 0] + c[..., 1], -1, idx)
    s = cs.argmin(-1, keepdim=True); r = (torch.arange(4, device=c.device) + s) % 4
    return torch.gather(idx, -1, r)


def targets(C, V, sigma=1.2):
    """C [B,Gt,4,2] px@448, V [B,Gt,4] -> heat [B,4,G,G], off target [B,2,G,G], off mask [B,G,G]."""
    B = C.shape[0]; idx = geo_order(C) if ORDER == "geo" else torch.arange(4, device=C.device).expand(*C.shape[:3])      # [B,Gt,4]
    Ct = torch.gather(C, 2, idx[..., None].expand(-1, -1, -1, 2)); Vt = torch.gather(V, 2, idx)
    b, g, t = Vt.nonzero(as_tuple=True); p = Ct[b, g, t] / STRIDE        # cells
    heat = torch.zeros(B * 4, G * G, device=C.device); off = torch.zeros(B, 2, G, G, device=C.device); om = torch.zeros(B, G, G, dtype=torch.bool, device=C.device)
    if len(b):
        ys, xs = torch.meshgrid(torch.arange(G, device=C.device) + 0.5, torch.arange(G, device=C.device) + 0.5, indexing="ij")
        gs = torch.exp(-((xs.reshape(1, -1) - p[:, :1]) ** 2 + (ys.reshape(1, -1) - p[:, 1:]) ** 2) / (2 * sigma ** 2))
        heat.index_reduce_(0, b * 4 + t, gs, "amax", include_self=True)
        ci = p.floor().long(); ok = (ci >= 0).all(-1) & (ci < G).all(-1)
        b2, ci, p2, t2 = b[ok], ci[ok], p[ok], t[ok]
        flat = heat.view(B, 4, G, G); flat[b2, t2, ci[:, 1], ci[:, 0]] = 1.0     # exact peak cell = 1 (focal positive)
        off[b2, :, ci[:, 1], ci[:, 0]] = (p2 - ci.float() - 0.5); om[b2, ci[:, 1], ci[:, 0]] = True
    return heat.view(B, 4, G, G), off, om


def focal(logit, gt):
    p = logit.float().sigmoid().clamp(1e-4, 1 - 1e-4); pos = gt.eq(1).float()
    l = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - gt) ** 4 * p ** 2 * (1 - p).log())
    return l.sum() / pos.sum().clamp(min=1)


def augment(x, C):
    """x [B,3,448,448] 0..1 on GPU; flips (corners follow, types recomputed geometrically), photometric, random blur."""
    B = x.shape[0]
    fl = torch.rand(B, device=x.device) < (0.5 if ORDER == "geo" else 0.0)
    x = torch.where(fl[:, None, None, None], x.flip(-1), x)
    C = C.clone(); C[..., 0] = torch.where(fl[:, None, None], IN - C[..., 0], C[..., 0])
    if P_ROT > 0:      # derotated loop input: image turned by the roll estimate, zero fill at the borders; label identity is kept under rotation
        ro = torch.rand(B, device=x.device) < P_ROT
        ang = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(ROT_MAX) * ro
        ca, sa = ang.cos(), ang.sin()
        th = torch.zeros(B, 2, 3, device=x.device); th[:, 0, 0] = ca; th[:, 0, 1] = -sa; th[:, 1, 0] = sa; th[:, 1, 1] = ca
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
        q = C / IN * 2 - 1      # output point o samples input th@o, so a label at input q moves to th^-1 q
        qx = ca[:, None, None] * q[..., 0] + sa[:, None, None] * q[..., 1]; qy = -sa[:, None, None] * q[..., 0] + ca[:, None, None] * q[..., 1]
        C = (torch.stack((qx, qy), -1) + 1) / 2 * IN
    g = torch.empty(B, 1, 1, 1, device=x.device)
    x = x * g.uniform_(0.6, 1.4) + torch.empty_like(g).uniform_(-0.12, 0.12)
    x = x * torch.empty(B, 3, 1, 1, device=x.device).uniform_(0.85, 1.15)
    mean = x.mean((1, 2, 3), keepdim=True); x = (x - mean) * torch.empty_like(g).uniform_(0.7, 1.3) + mean
    bl = torch.rand(B, device=x.device) < 0.3
    if bl.any():
        s = random.uniform(0.6, 2.0); r = int(math.ceil(2.5 * s)); k = torch.exp(-torch.arange(-r, r + 1, device=x.device).float() ** 2 / (2 * s * s)); k = k / k.sum()
        xb = F.conv2d(x[bl].reshape(-1, 1, IN, IN), k.view(1, 1, 1, -1), padding=(0, r)); xb = F.conv2d(xb, k.view(1, 1, -1, 1), padding=(r, 0))
        x[bl] = xb.view(-1, 3, IN, IN)
    x = x + torch.randn_like(x) * torch.empty_like(g).uniform_(0, 0.03)
    return x.clamp(0, 1), C


def batch(imgs, cors, viss, ii):
    x = F.interpolate(imgs[ii].to(dev, non_blocking=True).float().div(255), size=(IN, IN), mode="bilinear", align_corners=False)
    return x, cors[ii].to(dev), viss[ii].to(dev)


@torch.no_grad()
def validate(model, head, imgs, cors, viss, thr=0.3):
    head.eval(); st = {}; fp = 0; nf = 0
    for i0 in range(0, len(imgs), 64):
        ii = torch.arange(i0, min(len(imgs), i0 + 64)); x, C, V = batch(imgs, cors, viss, ii)
        with torch.autocast("cuda", torch.bfloat16):
            f = fused_features(model, normalize(x)); h, o = head(x, f)
        P = decode_peaks(h, o, (IN, IN), thr)
        idx = geo_order(C) if ORDER == "geo" else torch.arange(4, device=C.device).expand(*C.shape[:3]); Ct = torch.gather(C, 2, idx[..., None].expand(-1, -1, -1, 2)); Vt = torch.gather(V, 2, idx)
        size = (C.amax(2) - C.amin(2)).amax(-1) * 384 / IN                 # [B,Gt] in 384-crop px (same as the diag buckets)
        for b in range(len(ii)):
            pk = P[b]; nf += 1; used = torch.zeros(len(pk), dtype=torch.bool, device=dev)
            for g in range(C.shape[1]):
                for t in range(4):
                    if not Vt[b, g, t]: continue
                    s = float(size[b, g]); key = "0-60" if s < 60 else "60-120" if s < 120 else "120-250" if s < 250 else "250+"
                    e = st.setdefault(key, [0, 0, 0]); e[0] += 1
                    m = pk[:, 0] == t
                    if m.any():
                        dd = (pk[:, 2:] - Ct[b, g, t]).norm(dim=-1) * 384 / IN; dd[~m] = 1e9
                        if dd.min() < 4: e[1] += 1
                        if dd.min() < 8: e[2] += 1
                        used |= dd < 8
            fp += int((~used).sum())
    head.train()
    return {k: [v[0], round(v[1] / v[0], 3), round(v[2] / v[0], 3)] for k, v in sorted(st.items())}, round(fp / max(1, nf), 3)


def main():
    seeds = set(range(1, 100))
    tr = load(seeds - {VAL_SEED}, CROP_STRIDE); va = load({VAL_SEED}, 3)
    print(f"train {len(tr[0])} frames, val {len(va[0])}", flush=True)
    model = PerceptionModel("hybrid", teacher_dims={"dino": 384}); model.load_state_dict(torch.load(CKPT, map_location="cpu")); model = model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    head = CornerPeakHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    t0 = time.time(); N = len(tr[0]); log = open(OUT / "log.jsonl", "a")
    for it in range(1, ITERS + 1):
        ii = torch.randint(0, N, (BS,)); x, C, V = batch(*tr, ii); x, C = augment(x, C)
        ht, ot, om = targets(C, V)
        with torch.autocast("cuda", torch.bfloat16):
            f = fused_features(model, normalize(x)); h, o = head(x, f)
        lh = focal(h, ht); lo = (F.l1_loss(o.float().permute(0, 2, 3, 1)[om], ot.permute(0, 2, 3, 1)[om]) if om.any() else h.sum() * 0)
        loss = lh + lo
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if it % 1000 == 0 or it == ITERS:
            r, fp = validate(model, head, *va)
            rec = dict(iter=it, sec=int(time.time() - t0), loss_heat=round(lh.item(), 4), loss_off=round(lo.item(), 4), val_recall_4px_8px=r, val_false_peaks_per_frame=fp)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save(head.state_dict(), OUT / "cornerpeak.pt")
            if it % 5000 == 0: torch.save(head.state_dict(), OUT / f"cornerpeak_{it}.pt")


if __name__ == "__main__":
    main()
