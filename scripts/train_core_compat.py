"""Compatible student core (Update 165): a faster backbone that reproduces release_v4's intermediate features, so the frozen pyramid,
dense heads and EVERY plug-in module (gate heads, refiner input, seg v1/v2, indoor/outdoor heads, condition modules) run on it unchanged.

Teacher: release_v4 bigY_h (FastViT-SA24 stages 0-2), frozen. Student: FastViT-SA12 stages 0-2 (same channel widths 64/128/256, half the
blocks), ImageNet init; pyramid + dense heads + teacher heads are copied from v4 and stay frozen (only the backbone trains).
Loss (no labels needed): normalised MSE on the taps s4, s8, fmap and on the pyramid sum f (what cpk / ng / seg modules read), plus the
frozen heads' outputs on the student features vs on the teacher features: seg logits (MSE), inverse depth (L1), gate head raw maps (MSE).
Images: data/mix/bigG/images (all sources), random scale crops, trained at 448 and 384 (MIXRES) so the student also serves the 384 runtime.
Usage: python scripts/train_core_compat.py [out=runs/core_sa12]   env: ITERS=40000 BS=32 LR=1e-3 STUDENT=fastvit_sa12 MIXRES=1"""
import glob, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
for k, v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(k, v)
sys.path.insert(0, "src")
import timm                                   # noqa: E402
from pfm.model import PerceptionModel         # noqa: E402
from pfm.data import MEAN, STD                # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/core_sa12"); OUT.mkdir(parents=True, exist_ok=True)
CK = os.environ.get("CKPT", "runs/release_v4/bigY_h_hybrid.pt"); STUDENT = os.environ.get("STUDENT", "fastvit_sa12")
ITERS = int(os.environ.get("ITERS", "40000")); BS = int(os.environ.get("BS", "32")); LR = float(os.environ.get("LR", "1e-3"))
MIXRES = os.environ.get("MIXRES", "1") == "1"; dev = "cuda"
W = dict(s4=0.5, s8=0.5, fmap=1.0, f=1.0, seg=1.0, depth=1.0, gdet=2.0)
GATE = os.environ.get("GATE_LOSS", "1") == "1"      # v2: gate-cell weighted gate loss + frozen cpk3 / ng2 plug-in outputs (v1 plain nMSE: seg fine, gates 0.77 / 0.52 at 10k)
if GATE: W.update(gdet=1.0, gscore=2.0, goff=1.0, cpk=1.0, ng=0.5)
INIT = os.environ.get("INIT", "")


class Images(torch.utils.data.IterableDataset):
    def __init__(self, files, side): self.files, self.side = files, side
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random((wi.id if wi else 0) * 7919 + int(time.time()))
        while True:
            f = rng.choice(self.files)
            try: im = Image.open(f).convert("RGB")
            except Exception: continue
            w, h = im.size; s = rng.uniform(0.55, 1.0); cw, ch = int(w * s), int(h * s); x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((self.side, self.side), Image.BILINEAR)
            if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
            x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
            x = (x * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)).clamp(0, 1)          # brightness / contrast jitter
            yield x


def gate_losses(st, tt):
    """Gate head raw maps (B, 9L, H, W) = per level [score logit, 8 image-normalised corner offsets]: BCE to the teacher probability with gate cells
    weighted x20, offsets L1 (x20) only where the teacher sees a gate."""
    b, c, h, w = tt.shape; L = c // 9; s, t = st.float().view(b, L, 9, h, w), tt.float().view(b, L, 9, h, w)
    tp = t[:, :, 0].sigmoid(); wgt = 1 + 20 * tp
    gscore = (F.binary_cross_entropy_with_logits(s[:, :, 0], tp, reduction="none") * wgt).sum() / wgt.sum()
    m = (tp > 0.05).float()[:, :, None]
    goff = ((s[:, :, 1:] - t[:, :, 1:]).abs() * 20 * m).sum() / (m.sum() * 8 + 1)
    return gscore, goff


def taps(m, x):
    net = m.backbone.net; s4 = net.stages_0(net.stem_2(net.stem_1(net.stem_0(x)))); s8 = net.stages_1(s4); fmap = net.stages_2(s8)
    p = m.pyramid(fmap); f = p[0] + F.interpolate(p[1], size=p[0].shape[-2:], mode="bilinear", align_corners=False) + F.interpolate(p[2], size=p[0].shape[-2:], mode="bilinear", align_corners=False)
    o = m.dense(p, x.shape[-2:]); r = dict(s4=s4, s8=s8, fmap=fmap, f=f, seg=o["seg"], depth=o["depth"][:, :1], gdet=o["gdet"])
    if GATE and PLUG:
        img01 = (x * STD_D + MEAN_D).clamp(0, 1); r["cpk"] = torch.cat(PLUG["cpk"](img01, f), 1); r["ng"] = PLUG["ng"](img01, f)
    return r


PLUG = {}; MEAN_D = STD_D = None


def nmse(a, b): return ((a.float() - b.float()) ** 2).mean() / (b.float().var() + 1e-6)


def main():
    files = [f for e in ("jpg", "png", "jpeg") for f in glob.glob(f"data/mix/bigG/images/**/*.{e}", recursive=True)]
    print("images", len(files), flush=True)
    teacher = PerceptionModel("hybrid", teacher_dims={"dino": 384}, hybrid_name="fastvit_sa24"); teacher.load_state_dict(torch.load(CK, map_location="cpu"))
    teacher = teacher.to(dev).eval().requires_grad_(False)
    student = PerceptionModel("hybrid", teacher_dims={"dino": 384}, hybrid_name=STUDENT, pretrained=True)
    sd = {k: v for k, v in torch.load(CK, map_location="cpu").items() if not k.startswith("backbone.")}
    missing, unexpected = student.load_state_dict(sd, strict=False); assert not unexpected and all(k.startswith("backbone.") for k in missing), (missing[:5], unexpected[:5])
    student = student.to(dev); student.requires_grad_(False); student.backbone.requires_grad_(True); student.eval(); student.backbone.train()
    params = [p for p in student.backbone.parameters() if p.requires_grad]
    print(f"student backbone {STUDENT}: {sum(p.numel() for p in params) / 1e6:.2f} M trainable", flush=True)
    opt = torch.optim.AdamW(params, LR, weight_decay=0.02); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.04)
    mean, std = MEAN.view(1, 3, 1, 1).to(dev), STD.view(1, 3, 1, 1).to(dev)
    global MEAN_D, STD_D; MEAN_D, STD_D = mean, std
    if GATE:                       # frozen gate plug-ins of the flown config (they read the pyramid sum f): their outputs are distillation targets too
        from pfm.cornerpeak import CornerPeakHead, NearGateHead
        cpk = CornerPeakHead(); cpk.load_state_dict(torch.load(Path.home() / "DroneRacing_eval/runs_cmp/cpk3.pt", map_location="cpu"))
        ng = NearGateHead(); ng.load_state_dict(torch.load(Path.home() / "DroneRacing_eval/runs_cmp/ng2.pt", map_location="cpu"))
        PLUG["cpk"], PLUG["ng"] = cpk.to(dev).eval().requires_grad_(False), ng.to(dev).eval().requires_grad_(False)
    if INIT: student.load_state_dict(torch.load(INIT, map_location="cpu")); print("init from", INIT, flush=True)
    loaders = {s: iter(torch.utils.data.DataLoader(Images(files, s), batch_size=BS, num_workers=int(os.environ.get("WORKERS", "12")), persistent_workers=True, prefetch_factor=4))
               for s in ((448, 384) if MIXRES else (448,))}
    val = {s: next(iter(torch.utils.data.DataLoader(Images(files[::97], s), batch_size=64, num_workers=2))) for s in loaders}
    t0 = time.time(); run = {k: [] for k in W}; best = 1e9
    for it in range(1, ITERS + 1):
        side = 448 if (not MIXRES or it % 3) else 384                                  # 2/3 at 448, 1/3 at 384
        x = (next(loaders[side]).to(dev, non_blocking=True) - mean) / std
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): tt = taps(teacher, x)
        with torch.autocast("cuda", dtype=torch.bfloat16): st = taps(student, x)
        ls = dict(s4=nmse(st["s4"], tt["s4"]), s8=nmse(st["s8"], tt["s8"]), fmap=nmse(st["fmap"], tt["fmap"]), f=nmse(st["f"], tt["f"]),
                  seg=nmse(st["seg"], tt["seg"]), depth=(st["depth"].float() - tt["depth"].float()).abs().mean() / (tt["depth"].float().abs().mean() + 1e-6),
                  gdet=nmse(st["gdet"], tt["gdet"]))
        if GATE:
            ls["gscore"], ls["goff"] = gate_losses(st["gdet"], tt["gdet"])
            ls["cpk"] = nmse(st["cpk"], tt["cpk"]); g2s, g2t = gate_losses(st["ng"], tt["ng"]); ls["ng"] = g2s + g2t
        loss = sum(W[k] * v for k, v in ls.items())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        for k, v in ls.items(): run[k].append(v.item())
        if it % 500 == 0 or it == ITERS:
            msg = {"it": it, "min": round((time.time() - t0) / 60, 1), **{k: round(float(np.mean(v)), 4) for k, v in run.items()}}; run = {k: [] for k in W}
            student.backbone.eval(); vl = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s, xv in val.items():
                    xv = (xv.to(dev) - mean) / std; a, b = taps(teacher, xv), taps(student, xv)
                    vl[s] = dict(fmap=round(nmse(b["fmap"], a["fmap"]).item(), 4), seg_agree=round((a["seg"].argmax(1) == b["seg"].argmax(1)).float().mean().item(), 4),
                                 gdet=round(nmse(b["gdet"], a["gdet"]).item(), 4))
            student.backbone.train(); msg["val"] = vl; print(json.dumps(msg), flush=True)
            score = sum(v["fmap"] + v["gdet"] for v in vl.values())
            torch.save(student.state_dict(), OUT / "last.pt")
            if score < best: best = score; torch.save(student.state_dict(), OUT / "best.pt")
    print("CORE COMPAT DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
