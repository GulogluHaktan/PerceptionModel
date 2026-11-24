"""Feature-space condition module for the frozen core (the input-space CondAdapter hallucinated daylight pixels and hurt REAL night, Update 149).
A residual adapter on each pyramid level (p4 56x56, p3 28x28, p2 14x14, 128 ch; zero-initialised = identity): p' = p + A(p); the core's
backbone, pyramid and heads stay frozen. Two label-free signals:
  (a) synthetic pairs: clean general image x and night_real(x) (pfm.effects): heads on adapted night features -> heads on clean features
      (seg KL, depth L1, gate-map MSE) + feature MSE;   identity on clean inputs (P_CLEAN).
  (b) REAL unlabelled night (ACDC night trainB): FixMatch-style self-training - EMA teacher (core + EMA adapter) on the weak view gives
      seg pseudo-labels where confident (> CONF); the student must reproduce them on a strongly perturbed view (W_REAL).
Validation = labelled REAL dark frames (runs/selfimprove/real_dark_split.json, val half) + bright control; the test half is only for the report.
Run: env -u PYTHONPATH .venv/bin/python scripts/train_feat_adapter.py   (env ITERS BS LR W_REAL CONF OUT)"""
import os, sys, json, random, time, copy
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm import effects as E

dev = "cuda"; IN = 448
ITERS = int(os.environ.get("ITERS", "8000")); BS = int(os.environ.get("BS", "8")); LR = float(os.environ.get("LR", "3e-4")); W_REAL = float(os.environ.get("W_REAL", "0.5")); W_SYN = float(os.environ.get("W_SYN", "1.0"))
CONF = float(os.environ.get("CONF", "0.9")); P_CLEAN = float(os.environ.get("P_CLEAN", "0.15")); EMA = float(os.environ.get("EMA", "0.999"))
OUT = Path(os.environ.get("OUT", "runs/selfimprove/feat_night")); OUT.mkdir(parents=True, exist_ok=True)
REAL = os.environ.get("REAL_DIR", "data/general/raw_acdc/datasets/night/trainB")


class FeatAdapter(nn.Module):
    def __init__(self, c=128, h=96, levels=3):
        super().__init__()
        def blk():
            b = nn.Sequential(nn.Conv2d(c, h, 1), nn.GELU(), nn.Conv2d(h, h, 3, 1, 1, groups=h), nn.GELU(), nn.Conv2d(h, c, 1))
            nn.init.zeros_(b[-1].weight); nn.init.zeros_(b[-1].bias); return b
        self.a = nn.ModuleList([blk() for _ in range(levels)])
    def forward(self, pyr): return tuple(p + a(p) for p, a in zip(pyr, self.a))


def heads(core, x01, A=None):
    xn = normalize(x01); fmap, _ = core.backbone(xn); pyr = core.pyramid(fmap)
    if A is not None: pyr = A(pyr)
    return pyr, core.dense(pyr, xn.shape[-2:])


def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((IN, IN), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255


def strong(x, gen):
    b = x.shape[0]; r = lambda lo, hi: torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)
    x = (x - 0.5) * r(0.6, 1.4) + 0.5 + r(-0.1, 0.1)                        # contrast / brightness
    x = x * (1 + 0.15 * (torch.rand(b, 3, 1, 1, device=x.device) - 0.5))     # colour cast
    x = x + torch.randn(x.shape, device=x.device, generator=gen) * r(0.0, 0.06)
    if random.random() < 0.5: x = E._gauss_blur(x, random.uniform(0.5, 1.5))
    return x.clamp(0, 1)


def miou(cm):
    iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); return float(iu[cm.sum(1) > 0].mean())


@torch.no_grad()
def eval_real(core, A, frames, K=11):
    cm0 = np.zeros((K, K), np.int64); cm1 = cm0.copy()
    for p, lp in frames:
        x = load(p)[None].to(dev); lab = np.asarray(Image.open(lp).resize((IN, IN), Image.NEAREST)).astype(np.int64); v = (lab != 255) & (lab < K)
        for A_, cm in ((None, cm0), (A, cm1)):
            pr = heads(core, x, A_)[1]["seg"][0].float().argmax(0).cpu().numpy(); cm += np.bincount(lab[v] * K + pr[v], minlength=K * K).reshape(K, K)
    return round(miou(cm0), 4), round(miou(cm1), 4)


def main():
    S = json.loads(Path("runs/selfimprove/real_dark_split.json").read_text()); br = S["bright"][::4]
    val_d, val_b = S["val"], br[0::2][:60]
    core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    A = FeatAdapter().to(dev); T = copy.deepcopy(A).eval(); print(f"adapter params {sum(p.numel() for p in A.parameters())/1e3:.0f}k", flush=True)
    opt = torch.optim.AdamW(A.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(0); rng = np.random.default_rng(0)
    gfiles = [p for p in Path("data/general/train/images").rglob("*") if p.suffix.lower() in (".jpg", ".png")]; random.seed(0); random.shuffle(gfiles)
    rfiles = sorted(p for p in Path(REAL).glob("*") if p.suffix.lower() in (".jpg", ".png")); random.shuffle(rfiles)
    print(f"general {len(gfiles)}, real night {len(rfiles)}", flush=True)
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = None
    v0 = eval_real(core, A, val_d); print("val dark core/adapter (init)", v0, flush=True)
    for it in range(1, ITERS + 1):
        x = torch.stack([load(gfiles[(it * BS + i) % len(gfiles)]) for i in range(BS)]).to(dev)
        with torch.no_grad():
            pyr0, o0 = heads(core, x); inv = o0["depth"][:, 0].float()
            clean = random.random() < P_CLEAN
            xn = x if clean else torch.cat([E.night_real(x[b:b + 1], inv[b], gen, rng) for b in range(BS)])
        with torch.autocast("cuda", torch.bfloat16):
            pyr, o = heads(core, xn, A)
        lf = sum(F.mse_loss(a.float(), b.float()) for a, b in zip(pyr, pyr0)) / 3
        ls = F.kl_div(o["seg"].float().log_softmax(1), o0["seg"].float().softmax(1), reduction="batchmean") / (IN * IN)
        ld = F.l1_loss(o["depth"][:, :1].float(), o0["depth"][:, :1].float()); lg = F.mse_loss(o["gdet"].float(), o0["gdet"].float())
        loss = W_SYN * (lf + ls + ld + lg); lr_ = torch.zeros((), device=dev); frac = 0.0
        if W_REAL > 0 and rfiles:
            xr = torch.stack([load(rfiles[(it * BS + i) % len(rfiles)]) for i in range(BS // 2)]).to(dev)
            with torch.no_grad():
                pt = heads(core, xr, T)[1]["seg"].float().softmax(1); cf, pl = pt.max(1); m = cf > CONF; frac = float(m.float().mean())
            with torch.autocast("cuda", torch.bfloat16):
                so = heads(core, strong(xr, gen), A)[1]["seg"].float()
            ce = F.cross_entropy(so, pl, reduction="none"); lr_ = (ce * m).sum() / m.sum().clamp(min=1); loss = loss + W_REAL * lr_
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(A.parameters(), 1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pt_, pa in zip(T.parameters(), A.parameters()): pt_.mul_(EMA).add_(pa.detach(), alpha=1 - EMA)
        if it % 100 == 0:
            r = dict(it=it, loss=round(float(loss), 4), feat=round(float(lf), 4), seg=round(float(ls), 5), depth=round(float(ld), 4), gate=round(float(lg), 4), real=round(float(lr_), 4), conf_frac=round(frac, 3), min=round((time.time() - t0) / 60, 1))
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 2000 == 0 or it == ITERS:
            vd = eval_real(core, A, val_d); vb = eval_real(core, A, val_b); r = dict(it=it, val_dark=vd, val_bright=vb); print("VAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            torch.save(A.state_dict(), OUT / f"adapter_{it}.pt")
            score = (vd[1] - vd[0]) + min(0.0, vb[1] - vb[0])           # gain on real dark, any loss on bright counts against it
            if best is None or score > best[0]: best = (score, it); torch.save(A.state_dict(), OUT / "adapter.pt")
    print("best val checkpoint", best, flush=True)


if __name__ == "__main__":
    main()
