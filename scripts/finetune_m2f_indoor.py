"""Fine-tune Mask2Former (COCO-panoptic Swin-B by default) for the 22 indoor classes (src/pfm/indoor_classes.py).
Zero-shot with the 133->22 mapping it already gets 0.730 mIoU on data/general/eval_coco_in (teacher Swin-L + TTA 0.757).
Class head 133+1 -> 22+1, initialised per indoor class from its COCO member classes (mean weight, logsumexp bias), so training starts at the
zero-shot model. Semantic training: per image, one binary mask per present class (Mask2Former set loss, HF implementation).
Data: COCO train2017 indoor REAL labels (cocot) + COCO val train part (coco) + TTA-teacher pseudo labels on SUN / ADE (office / room domain).
Selection on a 300-image val split of the training pool; final number on eval_coco_in (never trained on).
Run: env -u PYTHONPATH .venv/bin/python scripts/finetune_m2f_indoor.py  (env MODEL ITERS BS LR SIZE OUT P_PSEUDO)"""
import os, sys, json, random, time, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfm.indoor_classes import IN_CLASSES, K_IN, COCO2IN
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
sys.path.insert(0, str(Path(__file__).resolve().parent)); import train_indoor_module as B      # load_pair with the teacher soft labels (KD), same crops
if B.CLASSSET == "out":                               # outdoor: 19 Cityscapes classes, BDD raw labels, Cityscapes-trained start
    IN_CLASSES, K_IN = B.IN_CLASSES, B.K_IN
KD = float(os.environ.get("KD", "1.0")); EVAL_SQUASH = int(os.environ.get("EVAL_SQUASH", "448"))     # evaluate like the vehicle: frame squashed to 448x448

MODEL = os.environ.get("MODEL", "facebook/mask2former-swin-tiny-cityscapes-semantic" if os.environ.get("CLASSSET") == "out" else "facebook/mask2former-swin-tiny-coco-panoptic"); ITERS = int(os.environ.get("ITERS", "20000")); BS = int(os.environ.get("BS", "8"))
LR = float(os.environ.get("LR", "2e-5")); SIZE = int(os.environ.get("SIZE", "640")); P_PSEUDO = float(os.environ.get("P_PSEUDO", "0.3"))
OUT = Path(os.environ.get("OUT", "runs/modules/m2f_indoor_swinb")); OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class GroupedPredictor(torch.nn.Module):
    """Keeps the pretrained 133+1-way class head; indoor class logit = logsumexp of its COCO member logits, so softmax over the 22+1 outputs equals
    the sum of the member probabilities (exactly the zero-shot mapping: training starts at 0.730). Indoor classes with no member get a learned logit."""
    def __init__(self, lin, groups):
        super().__init__(); self.lin = lin; self.groups = groups
        self.extra = torch.nn.Parameter(torch.full((len(groups),), -20.0))
    def forward(self, x):
        z = self.lin(x); outs = []
        for c, mem in enumerate(self.groups):
            outs.append(torch.logsumexp(z[..., mem], -1) if mem else self.extra[c].expand(z.shape[:-1]))
        return torch.stack(outs + [z[..., -1]], -1)


def build():
    m = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL); id2 = m.config.id2label
    if B.CLASSSET == "out": groups = [[i] for i in range(K_IN)]                          # Cityscapes ids == our outdoor ids (same order)
    else: groups = [[int(k) for k, n in id2.items() if COCO2IN.get(n, 0) == c] for c in range(K_IN)]
    m.class_predictor = GroupedPredictor(m.class_predictor, groups)
    m.config.num_labels = K_IN; m.config.id2label = {i: c for i, c in enumerate(IN_CLASSES)}; m.config.label2id = {c: i for i, c in enumerate(IN_CLASSES)}
    crit = m.criterion; crit.num_labels = K_IN; ew = torch.ones(K_IN + 1); ew[-1] = crit.eos_coef; crit.register_buffer("empty_weight", ew)
    return m


def save(m, path):
    path.mkdir(parents=True, exist_ok=True); torch.save(m.state_dict(), path / "model.pt")


def load_best(path):
    m = build(); m.load_state_dict(torch.load(path / "model.pt", map_location="cpu")); return m


def pool():
    L = Path("data/general/train"); real, ps = [], []
    for lp in sorted((L / "labels_in/seg").glob("*.png")):
        src, sid = lp.stem.split("__", 1); ip = L / "images" / src / f"{sid}.jpg"
        if not ip.exists(): continue
        if src in ("coco", "cocot"): real.append((ip, lp))
        elif src in ("sun", "ade"): ps.append((ip, lp))
    return real, ps


def load(ip, lp, aug):
    im = Image.open(ip).convert("RGB"); lab = Image.open(lp)
    if aug:
        W, H = im.size; s = random.uniform(0.5, 1.0); cw, ch = int(W * s), int(H * s); x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
        im = im.crop((x0, y0, x0 + cw, y0 + ch)); lab = lab.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT); lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
    x = (torch.from_numpy(np.asarray(im.resize((SIZE, SIZE), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
    y = torch.from_numpy(np.asarray(lab.resize((SIZE, SIZE), Image.NEAREST)).astype(np.int64).copy())
    return x, y


def targets(y):
    cls = [c for c in torch.unique(y).tolist() if c != 255 and c < K_IN]
    if not cls: return torch.zeros(0, *y.shape), torch.zeros(0, dtype=torch.long)
    return torch.stack([(y == c).float() for c in cls]), torch.tensor(cls, dtype=torch.long)


@torch.no_grad()
def predict(m, x):
    with torch.autocast("cuda", torch.bfloat16): o = m(pixel_values=x)
    cls = o.class_queries_logits.float().softmax(-1)[..., :-1]; mk = o.masks_queries_logits.float().sigmoid()
    return torch.einsum("bqc,bqhw->bchw", cls, mk)


@torch.no_grad()
def evaluate(m, pairs, side=800):
    m.eval(); cm = np.zeros((K_IN, K_IN), np.int64)
    for ip, lp in pairs:
        im = Image.open(ip).convert("RGB"); W, H = im.size; s = side / min(W, H)
        sz = (EVAL_SQUASH, EVAL_SQUASH) if EVAL_SQUASH else (round(W * s), round(H * s))
        x = ((torch.from_numpy(np.asarray(im.resize(sz, Image.BILINEAR if EVAL_SQUASH else Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD)[None].to(dev)
        seg = F.interpolate(predict(m, x), size=(H, W), mode="bilinear", align_corners=False)[0]; pr = seg.argmax(0).cpu().numpy()
        y = np.asarray(Image.open(lp)).astype(np.int64); v = (y != 255) & (y < K_IN)
        cm += np.bincount(y[v] * K_IN + pr[v], minlength=K_IN * K_IN).reshape(K_IN, K_IN)
    m.train(); iu = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm)); pres = cm.sum(1) > 0
    return dict(miou=round(float(iu[pres].mean()), 4), per_class={IN_CLASSES[i]: round(float(iu[i]), 3) for i in range(K_IN) if pres[i]})


def main():
    m = build().to(dev); real, ps = B.pool() if B.CLASSSET == "out" else pool(); random.seed(0); random.shuffle(real); random.shuffle(ps)
    val = real[:300]; real = real[300:]
    test = [(p, Path(f"data/general/{B.TESTDIR}/seg") / (p.stem + ".png")) for p in sorted(Path(f"data/general/{B.TESTDIR}/images").glob("*.jpg"))]
    print(f"train real {len(real)} pseudo {len(ps)} val {len(val)} test {len(test)}", flush=True)
    ev0 = evaluate(m, test[:400]); vz = evaluate(m, val); print("zero-shot (22-class head init) test400", ev0["miou"], "val", vz["miou"], flush=True)
    bb = [p for n, p in m.named_parameters() if "encoder" in n and "pixel_level_module" in n]; rest = [p for n, p in m.named_parameters() if not ("encoder" in n and "pixel_level_module" in n)]
    opt = torch.optim.AdamW([dict(params=bb, lr=LR * 0.1), dict(params=rest, lr=LR)], weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, it / 500) * 0.5 * (1 + math.cos(math.pi * min(it, ITERS) / ITERS)))
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = (vz["miou"], 0); save(m, OUT / "best"); ri = pi = 0; m.train()
    for it in range(1, ITERS + 1):
        xs, ml, cl, ts, hs = [], [], [], [], []
        for _ in range(BS):
            pr = ps[pi % len(ps)] if (ps and random.random() < P_PSEUDO) else real[ri % len(real)]
            if pr in ps: pi += 1
            else: ri += 1
            if KD > 0: x01, y, t, h = B.load_pair(*pr, True, True)                  # 448 crop + teacher probs with the same crop / flip
            else: (x01, y), t, h = B.load_pair(*pr, True, False), None, 0.0
            mk, c = targets(y); xs.append((x01 - MEAN) / STD); ml.append(mk.to(dev)); cl.append(c.to(dev)); ts.append(t); hs.append(h)
        with torch.autocast("cuda", torch.bfloat16): out = m(pixel_values=torch.stack(xs).to(dev), mask_labels=ml, class_labels=cl)
        loss = out.loss
        if KD > 0:          # distillation to the Swin-L teacher's class probabilities (stride-4 grid)
            sem = torch.einsum("bqc,bqhw->bchw", out.class_queries_logits.float().softmax(-1)[..., :-1], out.masks_queries_logits.float().sigmoid())
            sem = sem / sem.sum(1, keepdim=True).clamp(min=1e-6); lq = torch.log(F.adaptive_avg_pool2d(sem, 112).clamp(min=1e-6))
            t = torch.stack(ts).to(dev); hv = torch.tensor(hs, device=dev); kl = (t * (torch.log(t.clamp(min=1e-6)) - lq)).sum(1).mean((1, 2))
            loss = loss + KD * (kl * hv).sum() / hv.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 0.01 * 100); opt.step(); sch.step()
        if it % 100 == 0:
            r = dict(it=it, loss=round(float(loss), 4), min=round((time.time() - t0) / 60, 1)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 2500 == 0 or it == ITERS:
            ev = evaluate(m, val); r = dict(it=it, val=ev["miou"]); print("VAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            if ev["miou"] > best[0]: best = (ev["miou"], it); save(m, OUT / "best")
    m = load_best(OUT / "best").to(dev); te = evaluate(m, test)
    if B.CLASSSET == "out":
        dk = [(p, Path("data/general/eval_dark_out/seg") / (p.stem + ".png")) for p in sorted(Path("data/general/eval_dark_out/images").glob("*.jpg"))]
        te["real_dark"] = evaluate(m, dk)["miou"]
    (OUT / "eval_test.json").write_text(json.dumps(dict(best=best, zero_shot_test400=ev0["miou"], test=te), indent=1)); print("TEST", json.dumps(dict(best=best, test=te)), flush=True)


if __name__ == "__main__":
    main()
