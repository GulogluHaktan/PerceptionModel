"""Indoor segmentation module (22 classes, src/pfm/indoor_classes.py) on the frozen core: the core's 11-class seg puts desks, monitors, keyboards,
chairs, ... into other_obstacle. IndoorHead = SegHead2 design (backbone stages_0 / stages_1 taps + fused pyramid feature + the core's 11 seg logits,
stride-4 decoder) with its own 22-class output (separate output; the core's seg output is unchanged).
Data (scripts/prep_indoor_data.py): COCO panoptic val REAL labels (train part) + COCO-panoptic Mask2Former pseudo-labels on SUN / ADE / sim indoor.
Test: data/general/eval_coco_in (1000 indoor COCO images, real labels, never trained on). Checkpoint by test-free selection: a 300-image val split of the
labelled training pool (VAL); the COCO test is reported at the end.
Run: env -u PYTHONPATH .venv/bin/python scripts/train_indoor_module.py   (env ITERS BS LR OUT W P_REAL CW)"""
import os, sys, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_seg_module as T
from train_seg_module2 import CoreTap, _cbr
from pfm.indoor_classes import IN_CLASSES, K_IN
CLASSSET = os.environ.get("CLASSSET", "in")          # "out": the 19-class outdoor set (src/pfm/outdoor_classes.py), BDD raw labels
if CLASSSET == "out":
    from pfm.outdoor_classes import OUT_CLASSES as IN_CLASSES, K_OUT as K_IN
LABDIR = "labels_out" if CLASSSET == "out" else "labels_in"; TESTDIR = "eval_bdd_out" if CLASSSET == "out" else "eval_coco_in"

dev = "cuda"; IN = T.IN; K0 = T.K
ITERS = int(os.environ.get("ITERS", "24000")); BS = int(os.environ.get("BS", "12")); LR = float(os.environ.get("LR", "1e-3")); WD = int(os.environ.get("W", "96"))
P_REAL = float(os.environ.get("P_REAL", "0.4")); OUT = Path(os.environ.get("OUT", "runs/modules/indoor_head")); OUT.mkdir(parents=True, exist_ok=True)
CW = dict(kv.split(":") for kv in os.environ.get("CW", "keyboard:3,mouse:4,laptop:2,phone_remote:3,cup_bottle:2,monitor_tv:1.5,person:1.5,bag:2,book_paper:1.5").split(","))


class IndoorHead(nn.Module):
    def __init__(self, w=WD, k_core=K0, k_out=K_IN):
        super().__init__()
        self.l8 = nn.Sequential(_cbr(128 + 128 + k_core, w * 2), _cbr(w * 2, w * 2))
        self.l4 = _cbr(64, w); self.img = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), _cbr(32, 32))
        self.fuse = nn.Sequential(_cbr(w * 2 + w + 32 + k_core, w), _cbr(w, w), _cbr(w, w)); self.out = nn.Conv2d(w, k_out, 1)

    def forward(self, img01, s4, s8, feat, seg_logits):
        h8, h4 = feat.shape[-2:], s4.shape[-2:]
        l8 = self.l8(torch.cat((s8, feat, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h8[-1])), 1))
        im = F.avg_pool2d(self.img(img01 - 0.5), 2)
        x = torch.cat((F.interpolate(l8, size=h4, mode="bilinear", align_corners=False), self.l4(s4), im, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h4[-1])), 1)
        return F.interpolate(self.out(self.fuse(x)), size=seg_logits.shape[-2:], mode="bilinear", align_corners=False)


def _pool_matrix(n, k, device, dtype):
    """Row i averages input cells [floor(i n / k), ceil((i + 1) n / k)) = the bins of adaptive_avg_pool."""
    import math
    A = torch.zeros(k, n, device=device, dtype=dtype)
    for i in range(k): a, b = (i * n) // k, math.ceil((i + 1) * n / k); A[i, a:b] = 1.0 / (b - a)
    return A


def adaptive_pool_mm(x, k):
    """= F.adaptive_avg_pool2d(x, k) as two matrix products: exportable to ONNX when k does not divide the size (28 -> 3, 6)."""
    h, w = x.shape[-2:]
    return torch.einsum("kh,bchw,lw->bckl", _pool_matrix(h, k, x.device, x.dtype), x, _pool_matrix(w, k, x.device, x.dtype))


class PPM(nn.Module):
    """Pyramid pooling (global / 2x2 / 3x3 / 6x6 context) on the stride-16 feature."""
    def __init__(self, c, o):
        super().__init__(); self.bins = (1, 2, 3, 6); self.br = nn.ModuleList([nn.Sequential(nn.Conv2d(c, o // 4, 1, bias=False), nn.BatchNorm2d(o // 4), nn.ReLU(inplace=True)) for _ in self.bins])
        self.out = _cbr(c + o, o)
    def forward(self, x):
        h, w = x.shape[-2:]
        return self.out(torch.cat([x] + [F.interpolate(b(adaptive_pool_mm(x, k)), size=(h, w), mode="bilinear", align_corners=False) for k, b in zip(self.bins, self.br)], 1))


class IndoorHead2(nn.Module):
    """v2: + the backbone's stride-16 stage-2 feature (256 ch, the richest semantics) with pyramid pooling context -> stride 8 -> stride 4."""
    def __init__(self, w=WD, k_core=K0, k_out=K_IN):
        super().__init__()
        self.ppm = PPM(256, w * 2)
        self.l8 = nn.Sequential(_cbr(128 + 128 + k_core + w * 2, w * 2), _cbr(w * 2, w * 2))
        self.l4 = _cbr(64, w); self.img = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), _cbr(32, 32))
        self.fuse = nn.Sequential(_cbr(w * 2 + w + 32 + k_core, w), _cbr(w, w), _cbr(w, w)); self.out = nn.Conv2d(w, k_out, 1)

    def forward(self, img01, s4, s8, feat, seg_logits, s16):
        h8, h4 = feat.shape[-2:], s4.shape[-2:]
        ctx = F.interpolate(self.ppm(s16), size=h8, mode="bilinear", align_corners=False)
        l8 = self.l8(torch.cat((s8, feat, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h8[-1]), ctx), 1))
        im = F.avg_pool2d(self.img(img01 - 0.5), 2)
        x = torch.cat((F.interpolate(l8, size=h4, mode="bilinear", align_corners=False), self.l4(s4), im, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h4[-1])), 1)
        return F.interpolate(self.out(self.fuse(x)), size=seg_logits.shape[-2:], mode="bilinear", align_corners=False)


class IndoorHead3(nn.Module):
    """v3: own TRAINABLE copy of the backbone's last stage (stages_2, initialised from the frozen core) on the frozen stride-8 stage-1 feature, so the
    module can learn object-level indoor semantics (monitor vs laptop, chair vs person) that the frozen 11-class core features do not separate.
    The core itself stays untouched; on the vehicle this branch runs instead of seg module v2 (scene router)."""
    def __init__(self, stage2, w=WD, k_core=K0, k_out=K_IN):
        super().__init__()
        import copy; self.stage2 = copy.deepcopy(stage2)
        for p in self.stage2.parameters(): p.requires_grad_(True)
        self.ppm = PPM(256, w * 2)
        self.l8 = nn.Sequential(_cbr(128 + 128 + k_core + w * 2, w * 2), _cbr(w * 2, w * 2))
        self.l4 = _cbr(64, w); self.img = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), _cbr(32, 32))
        self.fuse = nn.Sequential(_cbr(w * 2 + w + 32 + k_core, w), _cbr(w, w), _cbr(w, w)); self.out = nn.Conv2d(w, k_out, 1)

    def forward(self, img01, s4, s8, feat, seg_logits, s16=None):
        h8, h4 = feat.shape[-2:], s4.shape[-2:]
        ctx = F.interpolate(self.ppm(self.stage2(s8)), size=h8, mode="bilinear", align_corners=False)
        l8 = self.l8(torch.cat((s8, feat, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h8[-1]), ctx), 1))
        im = F.avg_pool2d(self.img(img01 - 0.5), 2)
        x = torch.cat((F.interpolate(l8, size=h4, mode="bilinear", align_corners=False), self.l4(s4), im, F.avg_pool2d(seg_logits, seg_logits.shape[-1] // h4[-1])), 1)
        return F.interpolate(self.out(self.fuse(x)), size=seg_logits.shape[-2:], mode="bilinear", align_corners=False)


class IndoorHead4(IndoorHead3):
    """v4: trainable copies of stages_1 AND stages_2 (from the frozen stride-4 stage-0 feature); the new stride-8 feature also replaces s8 in l8."""
    def __init__(self, stage1, stage2, **kw):
        super().__init__(stage2, **kw)
        import copy; self.stage1 = copy.deepcopy(stage1)
        for p in self.stage1.parameters(): p.requires_grad_(True)

    def forward(self, img01, s4, s8, feat, seg_logits, s16=None):
        return super().forward(img01, s4, self.stage1(s4), feat, seg_logits)


class IndoorHead5(IndoorHead3):
    """v5: trainable copy of the WHOLE backbone (stem + stages 0-2, initialised from the frozen core) on the image itself; own s4 / s8 / s16.
    Max capacity of this design (on the vehicle: a second backbone pass, ~15 ms on the Orin)."""
    def __init__(self, net, **kw):
        super().__init__(net.stages_2, **kw)
        import copy; self.stem = copy.deepcopy(torch.nn.Sequential(net.stem_0, net.stem_1, net.stem_2)); self.stage0 = copy.deepcopy(net.stages_0); self.stage1 = copy.deepcopy(net.stages_1)
        for p in list(self.stem.parameters()) + list(self.stage0.parameters()) + list(self.stage1.parameters()): p.requires_grad_(True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img01, s4, s8, feat, seg_logits, s16=None):
        a4 = self.stage0(self.stem((img01 - self.mean) / self.std)); a8 = self.stage1(a4)
        return super().forward(img01, a4, a8, feat, seg_logits)


EDGE = os.environ.get("EDGE") == "1"; EDGE_W = float(os.environ.get("EDGE_W", "2.0"))


def convex_up(logits, mask, f=4):
    """RAFT convex upsampling: each output pixel = softmax-weighted mix of the 3x3 coarse neighbourhood (mask (B, 9*f*f, h, w)). Sharper edges than
    bilinear. 3x3 neighbourhood via padding + 9 shifted slices (ONNX-exportable, no unfold)."""
    B, K, h, w = logits.shape
    m = mask.view(B, 1, 9, f, f, h, w).softmax(2)
    p = F.pad(logits, (1, 1, 1, 1), mode="replicate")
    nb = torch.stack([p[:, :, i:i + h, j:j + w] for i in range(3) for j in range(3)], 2)          # (B,K,9,h,w)
    up = (m * nb.view(B, K, 9, 1, 1, h, w)).sum(2)                                              # (B,K,f,f,h,w)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(B, K, h * f, w * f)


def edge_weight(y, k=3):
    """1 + EDGE_W on pixels within k px of a label boundary (ignore pixels excluded)."""
    yf = y.float().unsqueeze(1); yf = torch.where(yf == 255, torch.full_like(yf, -1), yf)
    mx = F.max_pool2d(yf, 2 * k + 1, 1, k); mn = -F.max_pool2d(-yf, 2 * k + 1, 1, k)
    return 1.0 + EDGE_W * (mx != mn).float().squeeze(1)


class IndoorHead6(IndoorHead5):
    """v6: v5 (own whole-backbone copy) at a HIGHER input resolution than the core (env HIRES, e.g. 640): small / thin indoor objects (mouse, chair legs,
    books) are lost in the 448 squash. The core's fused feature and seg logits are resized onto the module's own grids; output = input resolution."""
    def forward(self, img_hi, s4, s8, feat, seg_logits, s16=None):
        a4 = self.stage0(self.stem((img_hi - self.mean) / self.std)); a8 = self.stage1(a4); h4, h8 = a4.shape[-2:], a8.shape[-2:]
        up = lambda t, hw: F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        ctx = up(self.ppm(self.stage2(a8)), h8)
        l8 = self.l8(torch.cat((a8, up(feat, h8), up(seg_logits, h8), ctx), 1))
        im = F.avg_pool2d(self.img(img_hi - 0.5), 2)
        x = torch.cat((up(l8, h4), self.l4(a4), im, up(seg_logits, h4)), 1); fz = self.fuse(x)
        if hasattr(self, "up_mask"): return convex_up(self.out(fz), self.up_mask(fz) * 0.25)
        return up(self.out(fz), img_hi.shape[-2:])


class CoreTap16(CoreTap):
    """CoreTap + the stride-16 stage-2 output (hook) -> (s4, s8, feat, seg_logits, s16)."""
    def __init__(self, core):
        super().__init__(core); core.backbone.net.stages_2.register_forward_hook(lambda m, i, o: self.t.__setitem__("stages_2", o))
    def __call__(self, x01):
        s4, s8, f, s = super().__call__(x01); return s4, s8, f, s, self.t["stages_2"].float()


ARCH = int(os.environ.get("ARCH", "1")); HIRES = int(os.environ.get("HIRES", "0"))      # ARCH 6: module input resolution (core stays at 448)


KD = float(os.environ.get("KD", "0")); SOFT = Path("data/general/train/soft_in")


ASPECT = os.environ.get("ASPECT") == "1"       # aspect-preserving inputs (no squash): train = square crops of a short-side-resized image, eval = whole image


def _aspect_box(W, H, aug):
    """ASPECT: short side -> S (train: random 448-640, eval: 448); train returns a 448 x 448 crop box in ORIGINAL pixel coords, eval the whole image."""
    S_ = random.randint(448, 640) if aug else 448; sc = S_ / min(W, H)
    if not aug: return (0, 0, W, H), (int(round(W * sc / 32)) * 32, int(round(H * sc / 32)) * 32)
    cw, ch = 448 / sc, 448 / sc; x0, y0 = random.uniform(0, W - cw), random.uniform(0, H - ch)
    return (int(x0), int(y0), int(x0 + cw), int(y0 + ch)), (448, 448)


def load_pair(p, lp, aug, with_soft=False):
    im = Image.open(p).convert("RGB"); lab = Image.open(lp); W, H = im.size; box = (0, 0, W, H); flip = False
    if ASPECT:
        box, size = _aspect_box(W, H, aug); im = im.crop(box); lab = lab.crop(box); flip = aug and random.random() < 0.5
        if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT); lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(im.resize(size, Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(np.asarray(lab.resize(size, Image.NEAREST)).astype(np.int64).copy()); y[(y >= K_IN) & (y != 255)] = 255
        if not with_soft: return x, y
    elif aug:
        s = random.uniform(0.6, 1.0); cw, ch = int(W * s), int(H * s); x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch); box = (x0, y0, x0 + cw, y0 + ch)
        im = im.crop(box); lab = lab.crop(box); flip = random.random() < 0.5
        if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT); lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
    if ASPECT: pass
    else:
      R_ = HIRES or IN
      x = torch.from_numpy(np.asarray(im.resize((R_, R_), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
      y = torch.from_numpy(np.asarray(lab.resize((R_, R_), Image.NEAREST)).astype(np.int64).copy()); y[(y >= K_IN) & (y != 255)] = 255
      if not with_soft: return x, y
    sp = SOFT / f"{Path(lp).stem}.npy"
    if not sp.exists(): return x, y, torch.zeros(K_IN, 112, 112), 0.0
    t = torch.from_numpy(np.load(sp)).float()[None] / 255                                      # teacher probs over the whole image (112x112 grid)
    gx = torch.linspace(box[0] / W, box[2] / W, 112) * 2 - 1; gy = torch.linspace(box[1] / H, box[3] / H, 112) * 2 - 1
    if flip: gx = gx.flip(0)
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]                  # same crop / flip as the image
    t = F.grid_sample(t, grid, mode="bilinear", align_corners=True)[0]; t = t / t.sum(0, keepdim=True).clamp(min=1e-6)
    return x, y, t, 1.0


def pool():
    L = Path("data/general/train"); real, ps = [], []
    for lp in sorted((L / LABDIR / "seg").glob("*.png")):
        src, sid = lp.stem.split("__", 1); ip = L / "images" / src / f"{sid}.jpg"
        if ip.exists(): (real if src in ("coco", "cocot", "bdd") else ps).append((ip, lp))
    return real, ps


def repeat_factor(pairs, t=float(os.environ.get("RFS_T", "0.05"))):
    """LVIS repeat-factor sampling: image repeat r = max over its classes of max(1, sqrt(t / f_c)), f_c = share of images containing class c.
    Class presence per label file is cached (data/general/train/labels_in_presence.json)."""
    cp = Path("data/general/train/labels_in_presence.json"); cache = json.loads(cp.read_text()) if cp.exists() else {}; new = 0
    for _, lp in pairs:
        k = Path(lp).stem
        if k not in cache:
            a = np.asarray(Image.open(lp)); cache[k] = [int(c) for c in np.unique(a) if c < K_IN and (a == c).sum() > 30]; new += 1
    if new: cp.write_text(json.dumps(cache))
    pres = [cache[Path(lp).stem] for _, lp in pairs]; f = np.zeros(K_IN)
    for cs in pres:
        for c in cs: f[c] += 1
    f /= max(1, len(pairs)); rc = np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-6)))
    r = np.array([max([rc[c] for c in cs], default=1.0) for cs in pres])
    print("RFS class repeat:", {IN_CLASSES[c]: round(float(rc[c]), 2) for c in range(K_IN) if rc[c] > 1.01}, "| mean image repeat", round(float(r.mean()), 2), flush=True)
    out = []
    for pr, ri in zip(pairs, r): out += [pr] * int(np.floor(ri)) + ([pr] if random.random() < ri - np.floor(ri) else [])
    random.shuffle(out); return out


@torch.no_grad()
def evaluate(tap, M, pairs):
    cm = np.zeros((K_IN, K_IN), np.int64)
    for p, lp in pairs:
        x, y = load_pair(p, lp, False); x = x[None].to(dev); y = y.numpy(); v = y != 255
        xc = F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False) if (HIRES or ASPECT) else x     # the core always sees its 448 squash
        with torch.autocast("cuda", torch.bfloat16): pr = M(x, *tap(xc))[0].float().argmax(0).cpu().numpy()
        cm += np.bincount(y[v] * K_IN + pr[v], minlength=K_IN * K_IN).reshape(K_IN, K_IN)
    iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); pres = cm.sum(1) > 0
    return dict(miou=round(float(iu[pres].mean()), 4), pix_acc=round(float(np.diag(cm).sum() / cm.sum()), 4), per_class={IN_CLASSES[i]: round(float(iu[i]), 3) for i in range(K_IN) if pres[i]}), cm


def main():
    core = T.PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    tap = CoreTap16(core) if ARCH >= 2 else CoreTap(core)
    net = core.backbone.net
    M = (IndoorHead6(net) if ARCH == 6 else IndoorHead5(net) if ARCH == 5 else IndoorHead4(net.stages_1, net.stages_2) if ARCH == 4 else IndoorHead3(net.stages_2) if ARCH == 3 else IndoorHead2() if ARCH == 2 else IndoorHead()).to(dev); print(f"module params {sum(p.numel() for p in M.parameters())/1e6:.2f}M", flush=True)
    if EDGE and ARCH == 6:      # learned convex upsampling head (stride 4 -> 1)
        M.up_mask = nn.Sequential(_cbr(M.out.in_channels, 128), nn.Conv2d(128, 9 * 16, 1)).to(dev)
    if os.environ.get("INIT"): M.load_state_dict(torch.load(os.environ["INIT"], map_location="cpu")); print("init from", os.environ["INIT"], flush=True)
    real, ps = pool(); random.seed(0); random.shuffle(real); random.shuffle(ps)
    val = real[:150] + ps[:150]; real, ps = real[150:], ps[150:]
    test = [(p, Path(f"data/general/{TESTDIR}/seg") / (p.stem + ".png")) for p in sorted(Path(f"data/general/{TESTDIR}/images").glob("*.jpg"))]
    if os.environ.get("RFS") == "1": real, ps = repeat_factor(real), repeat_factor(ps)
    print(f"train real {len(real)} pseudo {len(ps)} | val {len(val)} | test {len(test)}", flush=True)
    w = torch.ones(K_IN, device=dev)
    for c, v in CW.items():
        if c in IN_CLASSES: w[IN_CLASSES.index(c)] = float(v)
    if ARCH >= 3:            # the copied backbone stages start from good weights: 10x lower LR than the new layers
        pre = ("stage1.", "stage2.", "stage0.", "stem.")
        bb = [p for n, p in M.named_parameters() if n.startswith(pre)]; new = [p for n, p in M.named_parameters() if not n.startswith(pre)]
        opt = torch.optim.AdamW([dict(params=bb, lr=LR * 0.1), dict(params=new, lr=LR)], LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, [LR * 0.1, LR], total_steps=ITERS, pct_start=0.05)
    else:
        opt = torch.optim.AdamW(M.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = None; ri = pi = 0
    for it in range(1, ITERS + 1):
        b = []
        for _ in range(BS):
            if real and (random.random() < P_REAL or not ps): b.append(load_pair(*real[ri % len(real)], True, KD > 0)); ri += 1
            else: b.append(load_pair(*ps[pi % len(ps)], True, KD > 0)); pi += 1
        x = torch.stack([q[0] for q in b]).to(dev); y = torch.stack([q[1] for q in b]).to(dev)
        xc = F.interpolate(x, size=(IN, IN), mode="bilinear", align_corners=False) if HIRES else x
        with torch.autocast("cuda", torch.bfloat16): s = M(x, *tap(xc))
        if EDGE:
            ce = F.cross_entropy(s.float(), y, weight=w, ignore_index=255, reduction="none"); ew = edge_weight(y) * (y != 255).float()
            loss = (ce * ew).sum() / (w[y.clamp(max=K_IN - 1)] * ew).sum().clamp(min=1)
        else: loss = F.cross_entropy(s.float(), y, weight=w, ignore_index=255)
        if KD > 0:          # distillation: KL(teacher || student) on the stride-4 grid
            t = torch.stack([q[2] for q in b]).to(dev); has = torch.tensor([q[3] for q in b], device=dev)
            lq = F.log_softmax(F.adaptive_avg_pool2d(s.float(), 112), 1); kl = (t * (torch.log(t.clamp(min=1e-6)) - lq)).sum(1).mean((1, 2))
            loss = loss + KD * (kl * has).sum() / has.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); sch.step()
        if it % 200 == 0:
            r = dict(it=it, loss=round(float(loss), 4), min=round((time.time() - t0) / 60, 1)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 4000 == 0 or it == ITERS:
            M.eval(); ev, _ = evaluate(tap, M, val); M.train()
            r = dict(it=it, val=ev["miou"], val_pc=ev["per_class"]); print("VAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            if best is None or ev["miou"] > best[0]: best = (ev["miou"], it); torch.save(M.state_dict(), OUT / "module.pt")
    M.load_state_dict(torch.load(OUT / "module.pt")); M.eval(); te, cm = evaluate(tap, M, test)
    if CLASSSET == "out":
        dk = [(p, Path("data/general/eval_dark_out/seg") / (p.stem + ".png")) for p in sorted(Path("data/general/eval_dark_out/images").glob("*.jpg"))]
        te["real_dark"] = evaluate(tap, M, dk)[0]["miou"]
    (OUT / "eval_test.json").write_text(json.dumps(dict(best=best, test=te), indent=1)); np.save(OUT / "test_confusion.npy", cm)
    print("TEST", json.dumps(dict(best=best, test=te)), flush=True)


if __name__ == "__main__":
    main()
