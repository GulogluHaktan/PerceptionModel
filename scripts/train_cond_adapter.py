"""Self-improvement loop, step 2-3: a label-free CONDITION MODULE for the frozen core.
A small image-space adapter A (residual conv net, input side) is trained so that the frozen core sees the same thing in a hard
condition as in the clean image:  core(A(cond(x))) ~= core(x)  (stride-8 fused features MSE + seg / gate-slot-score KL).
Data = unlabelled general training images; the conditions come from the simulator effects (pfm.effects), so no label is needed:
this is the "generate data -> train a module" link of the loop. The core is never modified; the shift detector (feature distance,
scripts/shift_signals.py) decides at run time whether A is plugged in.
Run: env -u PYTHONPATH .venv/bin/python scripts/train_cond_adapter.py   (env: ITERS, BS, CONDS=night,nir,fog, OUT)
"""
import os, random, sys, time, json
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.model import PerceptionModel
from pfm.train import normalize
from pfm.cornerpeak import fused_features
from pfm import effects as E

dev = "cuda"; IN = 448
ITERS = int(os.environ.get("ITERS", "6000")); BS = int(os.environ.get("BS", "12")); LR = float(os.environ.get("LR", "2e-4")); W_PIX = float(os.environ.get("W_PIX", "1.0")); P_CLEAN = float(os.environ.get("P_CLEAN", "0.2"))
GATE_DIR = os.environ.get("GATE_DIR", ""); P_GATE = float(os.environ.get("P_GATE", "0.5")); W_GATE = float(os.environ.get("W_GATE", "1.0"))
CONDS = os.environ.get("CONDS", "night,nir,fog").split(","); OUT = Path(os.environ.get("OUT", "runs/selfimprove/cond_adapter")); OUT.mkdir(parents=True, exist_ok=True)
CK = "runs/release_v4/bigY_h_hybrid.pt"


class CondAdapter(nn.Module):
    """Residual image-to-image adapter, ~0.4M params, works at 1/2 resolution with a full-res skip (identity at init)."""
    def __init__(self, c=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(c, 2 * c, 3, 2, 1), nn.GELU(), *[nn.Sequential(nn.Conv2d(2 * c, 2 * c, 3, 1, 1), nn.GELU()) for _ in range(4)])
        self.dec = nn.Sequential(nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(c, c, 4, 2, 1), nn.GELU())
        self.out = nn.Conv2d(c + 3, 3, 3, 1, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x):
        return (x + self.out(torch.cat([self.dec(self.enc(x)), x], 1))).clamp(0, 1)


_NRNG = np.random.default_rng(0)


def apply_cond(x, name, inv, gen):
    """Random-strength version of the conditions in shift_signals.py (training draws strengths, eval uses fixed ones)."""
    B = x.shape[0]; outs = []
    for b in range(B):
        xb = x[b:b + 1]; r = random.random()
        if name == "night": xb = E.sensor(xb, ev=-2.5 - 2.5 * r, gain_ev=2.0 + 1.5 * r, iso=800 + 3200 * r, gen=gen)
        elif name == "nir":
            w = [0.05 + 0.15 * random.random(), 0.2 + 0.2 * random.random()]; g = (w[0] * xb[:, 0:1] + w[1] * xb[:, 1:2] + (1 - sum(w)) * xb[:, 0:1]).clamp(0, 1)
            xb = E.lens(g.repeat(1, 3, 1, 1), 0.0, 0.0, 0.3 + 0.5 * r)
        elif name == "blur":      # random camera motion: direction and speed drawn per image
            Hh, Ww = xb.shape[-2:]; ang = random.uniform(0, 6.283); sp = 600 + 1400 * r
            flow = torch.zeros(Hh, Ww, 2, device=xb.device); flow[..., 0] = sp * np.cos(ang); flow[..., 1] = sp * np.sin(ang)
            xb = E.motion_blur(xb, flow, exposure_s=0.006 + 0.010 * random.random())
        elif name == "night_blur":   # combined condition: random motion blur then a dark noisy sensor
            Hh, Ww = xb.shape[-2:]; ang = random.uniform(0, 6.283); sp = 600 + 1400 * random.random()
            flow = torch.zeros(Hh, Ww, 2, device=xb.device); flow[..., 0] = sp * np.cos(ang); flow[..., 1] = sp * np.sin(ang)
            xb = E.motion_blur(xb, flow, exposure_s=0.006 + 0.010 * random.random())
            xb = E.sensor(xb, ev=-2.5 - 2.5 * r, gain_ev=2.0 + 1.5 * r, iso=800 + 3200 * r, gen=gen)
        elif name == "night_real":   # realistic night (light sources, headlights, dark sky, auto exposure): pfm.effects.night_real
            xb = E.night_real(xb, inv[b], gen, _NRNG)
        elif name == "fog":
            d = 1.0 / inv[b].clamp(min=1e-3); d = d * (8.0 / d.median()); xb = E.fog(xb, d, beta=0.05 + 0.15 * r, airlight=0.6 + 0.35 * random.random())
        outs.append(xb.clamp(0, 1))
    return torch.cat(outs)


def main():
    files = [p for p in Path("data/general/train/images").rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
    random.seed(0); random.shuffle(files)
    gfiles = [p for p in Path(GATE_DIR).rglob("*") if p.suffix.lower() in (".jpg", ".png")] if GATE_DIR else []
    random.shuffle(gfiles); print(f"images {len(files)}, gate images {len(gfiles)}", flush=True)
    core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load(CK, map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    A = CondAdapter().to(dev); print(f"adapter params {sum(p.numel() for p in A.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(A.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(0); log = open(OUT / "log.jsonl", "a"); t0 = time.time(); k = 0

    def load_batch():
        nonlocal k
        xs = []
        for _ in range(BS):
            src = gfiles if (gfiles and random.random() < P_GATE) else files
            im = Image.open(src[k % len(src)]).convert("RGB").resize((IN, IN), Image.BICUBIC); k += 1
            xs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
        return torch.stack(xs).to(dev).float() / 255

    def core_out(x):
        xn = normalize(x); o = core(xn, with_teacher_heads=False)
        return fused_features(core, xn), o["seg"].float(), o["gdet"].float(), o["depth"][:, :1].float()

    for it in range(1, ITERS + 1):
        x = load_batch()
        with torch.no_grad():
            f0, s0, g0, inv = core_out(x)
            c = "clean" if random.random() < P_CLEAN else random.choice(CONDS); xc = x if c == "clean" else apply_cond(x, c, inv[:, 0], gen)
        with torch.autocast("cuda", torch.bfloat16):
            xa = A(xc); xn = normalize(xa)
            fmap, _ = core.backbone(xn); p4, p3, p2 = core.pyramid(fmap)
            f = p4 + F.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False) + F.interpolate(p2, size=p4.shape[-2:], mode="bilinear", align_corners=False)
            o = core.dense((p4, p3, p2), xn.shape[-2:])
        lf = F.mse_loss(f.float(), f0); ls = F.kl_div(o["seg"].float().log_softmax(1), s0.softmax(1), reduction="batchmean") / (s0.shape[-1] * s0.shape[-2])
        lg = W_GATE * F.mse_loss(o["gdet"].float(), g0)          # raw dense gate head (heatmap + corner offsets): order-free, keeps corner precision
        lp = F.l1_loss(xa.float(), x)          # restoration anchor: the clean source image is known (and identity on clean input)
        loss = lf + ls + 0.5 * lg + W_PIX * lp
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(A.parameters(), 1.0); opt.step(); sch.step()
        if it % 100 == 0:
            r = dict(it=it, cond=c, loss=round(float(loss), 4), feat=round(float(lf), 4), seg=round(float(ls), 5), gate=round(float(lg), 4), pix=round(float(lp), 4), min=round((time.time() - t0) / 60, 1))
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 2000 == 0 or it == ITERS: torch.save(A.state_dict(), OUT / "adapter.pt")


if __name__ == "__main__":
    main()
