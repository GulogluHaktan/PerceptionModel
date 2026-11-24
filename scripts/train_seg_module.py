"""Plug-in SEGMENTATION module for the frozen core (task-module idea of the roadmap): small objects / thin classes are the core's weak spot
(BDD v3: person 0.54, other_obstacle 0.28; not a resolution limit: 2x2 tiling makes them worse).
SegRefine: stride-4 image branch + the core's fused stride-8 features + the core's own seg logits -> residual logits (zero-initialised:
identity at start), added to the core logits. Trained supervised on the labelled general training sets (label set v5 for BDD / ADE,
v3 for RUGD = what the release core was trained on), core frozen. Acceptance on eval sets (v5 / v3): no set may lose mIoU.
Run: env -u PYTHONPATH .venv/bin/python scripts/train_seg_module.py   (env ITERS BS LR OUT CW="person:4,other_obstacle:3,car:1.5")"""
import os, sys, json, random, time
from pathlib import Path
for _k, _v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(_k, _v)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.model import PerceptionModel
from pfm.train import normalize

dev = "cuda"; IN = 448; K = 11
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]
ITERS = int(os.environ.get("ITERS", "12000")); BS = int(os.environ.get("BS", "12")); LR = float(os.environ.get("LR", "1e-3"))
OUT = Path(os.environ.get("OUT", "runs/modules/seg_refine")); OUT.mkdir(parents=True, exist_ok=True)
CW = dict(kv.split(":") for kv in os.environ.get("CW", "person:4,other_obstacle:3,car:1.5,tree_pole:1.5").split(","))
EVALS = [("eval_bdd5", "bdd"), ("eval_ade5", "ade"), ("eval_iddaw5", "iddaw"), ("eval_rugd_heldout3", "rugd")]


def _cbr(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SegRefine(nn.Module):
    def __init__(self, fdim=128, k=K):
        super().__init__()
        self.img = nn.Sequential(_cbr(3, 32, 2), _cbr(32, 48, 2), _cbr(48, 48))           # stride 4
        self.f = nn.Sequential(_cbr(fdim, 96), _cbr(96, 96))
        self.fuse = nn.Sequential(_cbr(96 + 48 + k, 96), _cbr(96, 96), _cbr(96, 64))
        self.out = nn.Conv2d(64, k, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, img01, feat, seg_logits):
        a = self.img(img01 - 0.5); b = F.interpolate(self.f(feat), size=a.shape[-2:], mode="bilinear", align_corners=False)
        k = seg_logits.shape[-1] // a.shape[-1]; s = F.avg_pool2d(seg_logits, k)      # = adaptive pool for 448 -> 112 (exact factor); ONNX-exportable
        r = self.out(self.fuse(torch.cat((b, a, s), 1)))
        return seg_logits + F.interpolate(r, size=seg_logits.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def core_pass(core, x01):
    xn = normalize(x01); fmap, _ = core.backbone(xn); p4, p3, p2 = core.pyramid(fmap); o = core.dense((p4, p3, p2), xn.shape[-2:])
    f = p4 + F.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False) + F.interpolate(p2, size=p4.shape[-2:], mode="bilinear", align_corners=False)
    return f.float(), o["seg"].float()


def train_pairs():
    L = Path("data/general/train"); out = []
    for src, lv in (("bdd", "labels_v5"), ("ade", "labels_v5"), ("rugd", "labels_v3")):
        for p in sorted((L / "images" / src).glob("*")):
            lp = L / lv / "seg" / f"{src}__{p.stem}.png"
            if lp.exists(): out.append((p, lp))
    return out


def load_pair(p, lp, aug):
    im = Image.open(p).convert("RGB"); lab = Image.open(lp)
    if aug:
        W, H = im.size; s = random.uniform(0.6, 1.0); cw, ch = int(W * s), int(H * s); x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
        im = im.crop((x0, y0, x0 + cw, y0 + ch)); lab = lab.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT); lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
    x = torch.from_numpy(np.asarray(im.resize((IN, IN), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
    y = torch.from_numpy(np.asarray(lab.resize((IN, IN), Image.NEAREST)).astype(np.int64).copy()); y[y >= K] = 255
    return x, y


@torch.no_grad()
def evaluate(core, M, n=None):
    res = {}
    for d, name in EVALS:
        D = Path("data/general") / d; ims = sorted((D / "images").glob("*"))[:n]; cm0 = np.zeros((K, K), np.int64); cm1 = cm0.copy()
        for p in ims:
            lp = D / "seg" / (p.stem + ".png")
            if not lp.exists(): continue
            x, y = load_pair(p, lp, False); x = x[None].to(dev); y = y.numpy(); v = y != 255
            f, s0 = core_pass(core, x); s1 = M(x, f, s0) if M is not None else s0
            for s, cm in ((s0, cm0), (s1, cm1)): pr = s[0].argmax(0).cpu().numpy(); cm += np.bincount(y[v] * K + pr[v], minlength=K * K).reshape(K, K)
        iu = lambda cm: np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); pres = cm0.sum(1) > 0
        res[name] = dict(core=round(float(iu(cm0)[pres].mean()), 4), module=round(float(iu(cm1)[pres].mean()), 4),
                         per_class={CL[i]: [round(float(iu(cm0)[i]), 3), round(float(iu(cm1)[i]), 3)] for i in range(K) if pres[i]})
    return res


def main():
    core = PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    M = SegRefine().to(dev); print(f"module params {sum(p.numel() for p in M.parameters())/1e6:.2f}M", flush=True)
    pairs = train_pairs(); random.seed(0); random.shuffle(pairs); print("train pairs", len(pairs), flush=True)
    w = torch.ones(K, device=dev)
    for c, v in CW.items(): w[CL.index(c)] = float(v)
    opt = torch.optim.AdamW(M.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = None
    for it in range(1, ITERS + 1):
        b = [load_pair(*pairs[(it * BS + i) % len(pairs)], True) for i in range(BS)]
        x = torch.stack([a for a, _ in b]).to(dev); y = torch.stack([c for _, c in b]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            f, s0 = core_pass(core, x)
            s = M(x, f, s0)
        loss = F.cross_entropy(s.float(), y, weight=w, ignore_index=255)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); sch.step()
        if it % 200 == 0:
            r = dict(it=it, loss=round(float(loss), 4), min=round((time.time() - t0) / 60, 1)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 3000 == 0 or it == ITERS:
            M.eval(); ev = evaluate(core, M, n=150); M.train()
            r = dict(it=it, eval={k: (v["core"], v["module"]) for k, v in ev.items()}); print("EVAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            torch.save(M.state_dict(), OUT / f"module_{it}.pt")
            gain = min(v["module"] - v["core"] for v in ev.values())          # worst set decides (acceptance: no set may lose)
            if best is None or gain > best[0]: best = (gain, it); torch.save(M.state_dict(), OUT / "module.pt")
    M.load_state_dict(torch.load(OUT / "module.pt")); M.eval(); ev = evaluate(core, M)
    (OUT / "eval_full.json").write_text(json.dumps(dict(best=best, eval=ev), indent=1)); print("FULL", json.dumps(dict(best=best, eval={k: (v["core"], v["module"]) for k, v in ev.items()})), flush=True)
    for k, v in ev.items(): print(k, {c: pc for c, pc in v["per_class"].items() if c in ("person", "other_obstacle", "car", "tree_pole")}, flush=True)


if __name__ == "__main__":
    main()
