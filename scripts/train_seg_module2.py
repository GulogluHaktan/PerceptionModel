"""Seg module v2 for the frozen core: bigger plug-in decoder + teacher pseudo-labels (v1 = scripts/train_seg_module.py, +0.01 mIoU).
Why: core+SegRefine vs a public Cityscapes Mask2Former (scripts/eval_seg_teacher.py): BDD 0.745 vs 0.812, IDD-AW 0.628 vs 0.666, person 0.51 vs 0.73.
SegHead2 reads the frozen backbone's intermediate stages (forward hooks: stages_0 64 ch stride 4, stages_1 128 ch stride 8), the fused pyramid
feature (128 ch stride 8) and the core's own seg logits; small FPN decoder at stride 4 -> residual logits (zero-init = identity at start).
Data: labelled (bdd / ade v5, rugd v3) + pseudo-labelled driving frames (scripts/make_seg_pseudo.py: acdc, mvist; low-confidence pixels ignored).
Selection = worst gain over the eval sets incl. the REAL dark val half (test half reported only).
Run: env -u PYTHONPATH .venv/bin/python scripts/train_seg_module2.py   (env ITERS BS LR OUT P_PSEUDO W=decoder width CW)"""
import os, sys, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seg_module as T          # sets the core env defaults, core_pass helpers, load_pair, class list
from pfm import effects as E

dev = "cuda"; K = T.K; CL = T.CL; IN = T.IN
ITERS = int(os.environ.get("ITERS", "20000")); BS = int(os.environ.get("BS", "12")); LR = float(os.environ.get("LR", "1e-3")); P_PSEUDO = float(os.environ.get("P_PSEUDO", "0.5"))
WD = int(os.environ.get("W", "96")); OUT = Path(os.environ.get("OUT", "runs/modules/seg_head2")); OUT.mkdir(parents=True, exist_ok=True)
CW = dict(kv.split(":") for kv in os.environ.get("CW", "person:3,other_obstacle:2,car:1.5,tree_pole:1.2").split(","))
P_NIGHT = float(os.environ.get("P_NIGHT", "0"))      # label-preserving synthetic night (pfm.effects.night_real, needs the core's inverse depth) on that share of each batch


def _cbr(i, o, k=3): return nn.Sequential(nn.Conv2d(i, o, k, 1, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SegHead2(nn.Module):
    def __init__(self, w=WD, k=K):
        super().__init__()
        self.l8 = nn.Sequential(_cbr(128 + 128 + k, w * 2), _cbr(w * 2, w * 2))                 # stage-1 feats + fused pyramid + core logits @ stride 8
        self.l4 = _cbr(64, w); self.img = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), _cbr(32, 32))   # stride 2 -> pooled to 4
        self.fuse = nn.Sequential(_cbr(w * 2 + w + 32 + k, w), _cbr(w, w), _cbr(w, w))
        self.out = nn.Conv2d(w, k, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, img01, s4, s8, feat, seg_logits):
        h8, h4 = feat.shape[-2:], s4.shape[-2:]
        l8 = self.l8(torch.cat((s8, feat, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h8[-1])), 1))
        im = F.avg_pool2d(self.img(img01 - 0.5), 2)
        x = torch.cat((F.interpolate(l8, size=h4, mode="bilinear", align_corners=False), self.l4(s4), im, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h4[-1])), 1)
        r = self.out(self.fuse(x))
        return seg_logits + F.interpolate(r, size=seg_logits.shape[-2:], mode="bilinear", align_corners=False)


class CoreTap:
    """Frozen core + hooks on the backbone's stages_0 / stages_1 -> (s4, s8, fused feature, seg logits)."""
    def __init__(self, core):
        self.core = core; self.t = {}
        net = core.backbone.net
        for n in ("stages_0", "stages_1"): getattr(net, n).register_forward_hook(lambda m, i, o, n=n: self.t.__setitem__(n, o))

    @torch.no_grad()
    def __call__(self, x01):
        f, s = T.core_pass(self.core, x01); return self.t["stages_0"].float(), self.t["stages_1"].float(), f, s


def pairs(split):
    L = Path("data/general/train"); out = []
    if split == "gt":
        return T.train_pairs()
    for p in sorted((L / "labels_pseudo/seg").glob("*.png")):
        src, sid = p.stem.split("__", 1); ip = L / "images" / src / f"{sid}.jpg"
        if ip.exists(): out.append((ip, p))
    return out


@torch.no_grad()
def evaluate(tap, M, sets, n=None):
    res = {}
    for name, prs in sets.items():
        cm0 = np.zeros((K, K), np.int64); cm1 = cm0.copy()
        for p, lp in prs[:n]:
            if not Path(lp).exists(): continue
            x, y = T.load_pair(p, lp, False); x = x[None].to(dev); y = y.numpy(); v = y != 255
            s4, s8, f, s0 = tap(x); s1 = M(x, s4, s8, f, s0)
            for s, cm in ((s0, cm0), (s1, cm1)): pr = s[0].argmax(0).cpu().numpy(); cm += np.bincount(y[v] * K + pr[v], minlength=K * K).reshape(K, K)
        iu = lambda cm: np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); pres = cm0.sum(1) > 0
        res[name] = dict(core=round(float(iu(cm0)[pres].mean()), 4), module=round(float(iu(cm1)[pres].mean()), 4),
                         per_class={CL[i]: [round(float(iu(cm0)[i]), 3), round(float(iu(cm1)[i]), 3)] for i in range(K) if pres[i]})
    return res


def eval_sets():
    S = json.loads(Path("runs/selfimprove/real_dark_split.json").read_text())
    sets = {name: [(p, Path("data/general") / d / "seg" / (p.stem + ".png")) for p in sorted((Path("data/general") / d / "images").glob("*"))] for d, name in T.EVALS}
    sets["real_dark_val"] = [(Path(a), Path(b)) for a, b in S["val"]]
    test = {"real_dark_test": [(Path(a), Path(b)) for a, b in S["test"]]}
    return sets, test


def main():
    core = T.PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    tap = CoreTap(core); M = SegHead2().to(dev); print(f"module params {sum(p.numel() for p in M.parameters())/1e6:.2f}M", flush=True)
    gt, ps = pairs("gt"), pairs("pseudo"); random.seed(0); random.shuffle(gt); random.shuffle(ps); print("gt", len(gt), "pseudo", len(ps), flush=True)
    sets, test = eval_sets()
    w = torch.ones(K, device=dev)
    for c, v in CW.items(): w[CL.index(c)] = float(v)
    opt = torch.optim.AdamW(M.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = None; gi = pi = 0
    gen = torch.Generator(device=dev).manual_seed(0); nrng = np.random.default_rng(0)
    for it in range(1, ITERS + 1):
        b = []
        for _ in range(BS):
            if ps and random.random() < P_PSEUDO: b.append(T.load_pair(*ps[pi % len(ps)], True)); pi += 1
            else: b.append(T.load_pair(*gt[gi % len(gt)], True)); gi += 1
        x = torch.stack([a for a, _ in b]).to(dev); y = torch.stack([c for _, c in b]).to(dev)
        if P_NIGHT > 0:
            sel = [i for i in range(BS) if random.random() < P_NIGHT]
            if sel:
                with torch.no_grad():
                    inv = core.dense(core.pyramid(core.backbone(T.normalize(x[sel]))[0]), (IN, IN))["depth"][:, 0].float()
                    x = x.clone()
                    for j, i in enumerate(sel): x[i:i + 1] = E.night_real(x[i:i + 1], inv[j], gen, nrng).clamp(0, 1)
        with torch.autocast("cuda", torch.bfloat16):
            s4, s8, f, s0 = tap(x); s = M(x, s4, s8, f, s0)
        loss = F.cross_entropy(s.float(), y, weight=w, ignore_index=255)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); sch.step()
        if it % 200 == 0:
            r = dict(it=it, loss=round(float(loss), 4), min=round((time.time() - t0) / 60, 1)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 4000 == 0 or it == ITERS:
            M.eval(); ev = evaluate(tap, M, sets, n=150); M.train()
            r = dict(it=it, eval={k: (v["core"], v["module"]) for k, v in ev.items()}); print("EVAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            torch.save(M.state_dict(), OUT / f"module_{it}.pt")
            gain = min(v["module"] - v["core"] for v in ev.values()); mean_gain = float(np.mean([v["module"] - v["core"] for v in ev.values()]))
            score = (gain >= 0, mean_gain if gain >= -0.005 else gain)      # no set may lose (5e-3 tolerance), then the mean gain decides
            if best is None or score > best[0]: best = (score, it); torch.save(M.state_dict(), OUT / "module.pt")
    M.load_state_dict(torch.load(OUT / "module.pt")); M.eval(); ev = evaluate(tap, M, {**sets, **test})
    (OUT / "eval_full.json").write_text(json.dumps(dict(best=best, eval=ev), indent=1)); print("FULL", json.dumps(dict(best=best, eval={k: (v["core"], v["module"]) for k, v in ev.items()})), flush=True)
    for k, v in ev.items(): print(k, {c: pc for c, pc in v["per_class"].items() if c in ("person", "other_obstacle", "car", "building")}, flush=True)


if __name__ == "__main__":
    main()
