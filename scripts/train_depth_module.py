"""Depth module for the frozen core (same plug-in pattern as scripts/train_seg_module2.py).
Gap (scripts/eval_depth_teacher.py, AbsRel inverse fit): KITTI core 0.079 vs Depth-Anything-V2-Small 0.051, DIODE indoor 0.132 vs 0.088; DIODE outdoor core
0.260 vs teacher 0.452 (the teacher is WORSE there -> acceptance must keep outdoor).
DepthHead: backbone stages_0 / stages_1 (hooks) + fused pyramid feature + the core's inverse depth -> residual inverse depth (zero-init = identity).
Loss: scale-and-shift-invariant L1 to the teacher disparity (MiDaS: per-image median / mean-abs normalisation) + multi-scale gradient matching,
teacher computed on the fly on unlabelled images (all general training images incl. the pseudo-label driving frames; no eval image).
Selection on DIODE (in/out) + KITTI: no set may lose more than TOL, then the mean relative gain decides. Numbers are therefore selection numbers.
Run: env -u PYTHONPATH .venv/bin/python scripts/train_depth_module.py   (env ITERS BS LR OUT W_GRAD TEACHER)"""
import os, sys, json, glob, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seg_module as T
from train_seg_module2 import CoreTap, _cbr
from pfm.evaluate import absrel_both
from transformers import AutoModelForDepthEstimation

dev = "cuda"; IN = 448
ITERS = int(os.environ.get("ITERS", "16000")); BS = int(os.environ.get("BS", "12")); LR = float(os.environ.get("LR", "5e-4")); W_GRAD = float(os.environ.get("W_GRAD", "0.5"))
TOL = float(os.environ.get("TOL", "0.002")); OUT = Path(os.environ.get("OUT", "runs/modules/depth_head")); OUT.mkdir(parents=True, exist_ok=True)
TEACHER = os.environ.get("TEACHER", "depth-anything/Depth-Anything-V2-Small-hf")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DepthHead(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.l8 = nn.Sequential(_cbr(128 + 128 + 1, w * 2), _cbr(w * 2, w * 2)); self.l4 = _cbr(64, w)
        self.fuse = nn.Sequential(_cbr(w * 2 + w + 1, w), _cbr(w, w)); self.out = nn.Conv2d(w, 1, 3, 1, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, s4, s8, feat, d):
        h8, h4 = feat.shape[-2:], s4.shape[-2:]; sc = d.flatten(1).abs().mean(1).clamp(min=1e-6).view(-1, 1, 1, 1); dn = d / sc     # residual in the core's own scale
        l8 = self.l8(torch.cat((s8, feat, F.adaptive_avg_pool2d(dn, h8)), 1))
        x = torch.cat((F.interpolate(l8, size=h4, mode="bilinear", align_corners=False), self.l4(s4), F.adaptive_avg_pool2d(dn, h4)), 1)
        return d + sc * F.interpolate(self.out(self.fuse(x)), size=d.shape[-2:], mode="bilinear", align_corners=False)


def ssi_norm(d, m=None):
    b = d.shape[0]; f = d.flatten(1); med = f.median(1).values.view(b, 1, 1, 1); s = (d - med).flatten(1).abs().mean(1).clamp(min=1e-6).view(b, 1, 1, 1); return (d - med) / s


def grad_loss(a, b, scales=4):
    l = 0.0
    for k in range(scales):
        if k: a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
        r = a - b; l = l + (r[..., :, 1:] - r[..., :, :-1]).abs().mean() + (r[..., 1:, :] - r[..., :-1, :]).abs().mean()
    return l / scales


def load(p, aug):
    im = Image.open(p).convert("RGB")
    if aug:
        W, H = im.size; s = random.uniform(0.7, 1.0); cw, ch = int(W * s), int(H * s); x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch); im = im.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.asarray(im.resize((IN, IN), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255


def eval_sets():
    ind = json.load(open("data/general/eval_diode/indoor.json")); E = "data/general/eval_diode"
    return {"diode_indoor": [(f"{E}/images/{s}.jpg", f"{E}/depth/{s}.npy") for s, i in ind.items() if i],
            "diode_outdoor": [(f"{E}/images/{s}.jpg", f"{E}/depth/{s}.npy") for s, i in ind.items() if not i],
            "kitti": [(p, p.replace("/images/", "/depth/")[:-4] + ".npy") for p in sorted(glob.glob("data/general/eval_kitti/images/*.jpg"))]}


@torch.no_grad()
def evaluate(tap, M, sets):
    res = {}
    for name, items in sets.items():
        r0, r1 = [], []
        for p, g in items:
            g = np.asarray(Image.fromarray(np.load(g).astype(np.float32)).resize((IN, IN), Image.NEAREST)); v = g > 0
            if v.sum() < 500: continue
            x = load(p, False)[None].to(dev); s4, s8, f, _ = tap(x); d0 = tap.core_depth; d1 = M(s4, s8, f, d0)
            r0.append(absrel_both(d0[0, 0].float().cpu().numpy(), g, v)[1]); r1.append(absrel_both(d1[0, 0].float().cpu().numpy(), g, v)[1])
        res[name] = dict(core=round(float(np.mean(r0)), 4), module=round(float(np.mean(r1)), 4))
    return res


class DepthTap(CoreTap):
    @torch.no_grad()
    def __call__(self, x01):
        xn = T.normalize(x01); fmap, _ = self.core.backbone(xn); p4, p3, p2 = self.core.pyramid(fmap); o = self.core.dense((p4, p3, p2), xn.shape[-2:])
        f = p4 + F.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False) + F.interpolate(p2, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        self.core_depth = o["depth"][:, :1].float(); return self.t["stages_0"].float(), self.t["stages_1"].float(), f.float(), o["seg"].float()


def main():
    core = T.PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    tap = DepthTap(core); M = DepthHead().to(dev); print(f"module params {sum(p.numel() for p in M.parameters())/1e6:.2f}M", flush=True)
    da = AutoModelForDepthEstimation.from_pretrained(TEACHER).to(dev).eval(); mean, std = MEAN.to(dev), STD.to(dev)
    files = [p for p in Path("data/general/train/images").rglob("*") if p.suffix.lower() in (".jpg", ".png")]; random.seed(0); random.shuffle(files); print("images", len(files), flush=True)
    sets = eval_sets(); ev0 = evaluate(tap, M, sets); print("init", json.dumps(ev0), flush=True)
    opt = torch.optim.AdamW(M.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = None
    for it in range(1, ITERS + 1):
        x = torch.stack([load(files[(it * BS + i) % len(files)], True) for i in range(BS)]).to(dev)
        with torch.no_grad():
            t = da(pixel_values=(F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False) - mean) / std).predicted_depth[:, None]
            t = ssi_norm(F.interpolate(t.float(), size=(IN, IN), mode="bilinear", align_corners=False))
        with torch.autocast("cuda", torch.bfloat16):
            s4, s8, f, _ = tap(x); d = M(s4, s8, f, tap.core_depth)
        pn = ssi_norm(d.float()); loss = (pn - t).abs().mean() + W_GRAD * grad_loss(pn, t)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); sch.step()
        if it % 200 == 0:
            r = dict(it=it, loss=round(float(loss), 4), min=round((time.time() - t0) / 60, 1)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 2000 == 0 or it == ITERS:
            M.eval(); ev = evaluate(tap, M, sets); M.train()
            rel = {k: (v["core"] - v["module"]) / v["core"] for k, v in ev.items()}; worst = min(v["core"] - v["module"] for v in ev.values())
            r = dict(it=it, eval={k: (v["core"], v["module"]) for k, v in ev.items()}, rel_gain={k: round(x, 3) for k, x in rel.items()}); print("EVAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            torch.save(M.state_dict(), OUT / f"module_{it}.pt"); score = (worst >= -TOL, float(np.mean(list(rel.values()))))
            if best is None or score > best[0]: best = (score, it); torch.save(M.state_dict(), OUT / "module.pt")
    (OUT / "best.json").write_text(json.dumps(dict(best=best))); print("best", best, flush=True)


if __name__ == "__main__":
    main()
