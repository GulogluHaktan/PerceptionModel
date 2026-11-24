"""ARCH 7: mask-classification indoor student (Mask2Former-style, small) on the frozen core.
Why: per-pixel students (train_indoor_module.py ARCH 1-6) classify every pixel on its own -> fragmented, ragged regions on real office frames even
when the classes are right. Mask2Former (the teacher) predicts N whole-object masks + one class per mask -> clean, coherent regions.
Model: own trainable copy of the core backbone (stem, stages 0-2, as ARCH 5) + the ARCH 3/5 pixel decoder (stride-4 per-pixel embedding) +
a 6-layer query decoder (50 queries, cross-attention round-robin over the stride-16 PPM and stride-8 features, self-attention, FFN), class + mask
heads on every layer (deep supervision). Semantic output = sum_q softmax(class_q) * sigmoid(mask_q)  (the teacher's semantic inference).
Loss: HF Mask2FormerLoss (Hungarian matching, class CE, sigmoid mask CE + dice on sampled points; per-class semantic targets) + KL distillation of the
semantic probabilities to the Swin-L teacher soft labels (stride-4 grid). Data / eval / test exactly as train_indoor_module.py.
Run: env -u PYTHONPATH .venv/bin/python scripts/train_indoor_mask.py   (env ITERS BS LR KD RFS W NQ OUT INIT)"""
import os, sys, json, math, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_indoor_module as B                      # data, KD soft labels, RFS, pools, IndoorHead5 pieces
from train_seg_module2 import _cbr
from pfm.indoor_classes import IN_CLASSES, K_IN
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

dev = "cuda"; ITERS = int(os.environ.get("ITERS", "50000")); BS = int(os.environ.get("BS", "8")); LR = float(os.environ.get("LR", "1e-4"))
NQ = int(os.environ.get("NQ", "50")); D = 256; KD = float(os.environ.get("KD", "1")); P_REAL = float(os.environ.get("P_REAL", "0.7"))
OUT = Path(os.environ.get("OUT", "runs/modules/indoor_mask7")); OUT.mkdir(parents=True, exist_ok=True)


def sine_pos(h, w, d, device):
    """2-D sine positional encoding (B-free) (h*w, d)."""
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    y = y / max(1, h - 1) * 2 * math.pi; x = x / max(1, w - 1) * 2 * math.pi; dq = d // 4
    f = 10000 ** (torch.arange(dq, device=device, dtype=torch.float32) / dq)
    py, px = y.reshape(-1, 1) / f, x.reshape(-1, 1) / f
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], 1)


class DecLayer(nn.Module):
    def __init__(self, d=D, heads=8, ff=1024):
        super().__init__()
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True); self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Linear(ff, d)); self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, q, qpos, mem, mpos):
        q = self.n1(q + self.ca(q + qpos, mem + mpos, mem)[0]); q = self.n2(q + self.sa(q + qpos, q + qpos, q)[0]); return self.n3(q + self.ff(q))


class IndoorMask(B.IndoorHead5):
    def __init__(self, net, w=128, nq=NQ, layers=6):
        super().__init__(net, w=w)
        self.mask_feat = nn.Conv2d(w, D, 1); self.p16 = nn.Conv2d(w * 2, D, 1); self.p8 = nn.Conv2d(w * 2, D, 1); self.lvl = nn.Parameter(torch.zeros(2, D))
        self.query = nn.Embedding(nq, D); self.qpos = nn.Embedding(nq, D); self.dec = nn.ModuleList([DecLayer() for _ in range(layers)]); self.qn = nn.LayerNorm(D)
        self.cls = nn.Linear(D, K_IN + 1); self.memb = nn.Sequential(nn.Linear(D, D), nn.ReLU(inplace=True), nn.Linear(D, D), nn.ReLU(inplace=True), nn.Linear(D, D))
        del self.out                                                       # per-pixel classifier replaced by mask classification

    def pred(self, q, mf):
        q = self.qn(q); return self.cls(q), torch.einsum("bqd,bdhw->bqhw", self.memb(q), mf)

    def forward_all(self, img01, s4, s8, feat, seg_logits):
        a4 = self.stage0(self.stem((img01 - self.mean) / self.std)); a8 = self.stage1(a4); h8, h4 = a8.shape[-2:], a4.shape[-2:]
        up = lambda t, hw: F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        c16 = self.ppm(self.stage2(a8)); ctx = up(c16, h8)
        l8 = self.l8(torch.cat((a8, up(feat, h8), up(seg_logits, h8), ctx), 1))
        im = F.avg_pool2d(self.img(img01 - 0.5), 2)
        fz = self.fuse(torch.cat((up(l8, h4), self.l4(a4), im, up(seg_logits, h4)), 1)); mf = self.mask_feat(fz)
        mems = []
        for k, (t, proj) in enumerate(((c16, self.p16), (l8, self.p8))):
            m = proj(t); hh, ww = m.shape[-2:]; mems.append((m.flatten(2).transpose(1, 2) + self.lvl[k], sine_pos(hh, ww, D, m.device)[None]))
        bsz = img01.shape[0]; q = self.query.weight[None].expand(bsz, -1, -1); qp = self.qpos.weight[None].expand(bsz, -1, -1); outs = []
        for i, layer in enumerate(self.dec):
            mem, mp = mems[i % 2]; q = layer(q, qp, mem, mp); outs.append(self.pred(q, mf))
        return outs                                                         # [(class (B,N,K+1), masks (B,N,h4,w4))] per layer, last = final

    @staticmethod
    def semantic(cl, mk):
        return torch.einsum("bqc,bqhw->bchw", cl.float().softmax(-1)[..., :-1], mk.float().sigmoid())

    def forward(self, img01, s4, s8, feat, seg_logits, s16=None):
        cl, mk = self.forward_all(img01, s4, s8, feat, seg_logits)[-1]
        return F.interpolate(self.semantic(cl, mk), size=img01.shape[-2:], mode="bilinear", align_corners=False)      # (B,K,H,W) class scores


def targets(y):
    out_m, out_c = [], []
    for yy in y:
        cls = [c for c in torch.unique(yy).tolist() if c != 255 and c < K_IN]
        out_m.append(torch.stack([(yy == c).float() for c in cls]) if cls else torch.zeros(0, *yy.shape, device=yy.device)); out_c.append(torch.tensor(cls, dtype=torch.long, device=yy.device))
    return out_m, out_c


def main():
    core = B.T.PerceptionModel("hybrid", teacher_dims={"dino": 384}); core.load_state_dict(torch.load("runs/release_v4/bigY_h_hybrid.pt", map_location="cpu")); core = core.to(dev).eval()
    for p in core.parameters(): p.requires_grad_(False)
    tap = B.CoreTap(core); M = IndoorMask(core.backbone.net, w=int(os.environ.get("W", "128"))).to(dev)
    if os.environ.get("INIT"):                                              # e.g. the ARCH 5 per-pixel student: backbone copy + pixel decoder
        miss = M.load_state_dict(torch.load(os.environ["INIT"], map_location="cpu"), strict=False); print("init", os.environ["INIT"], "missing", len(miss.missing_keys), flush=True)
    print(f"module params {sum(p.numel() for p in M.parameters())/1e6:.2f}M", flush=True)
    cfg = Mask2FormerConfig(num_labels=K_IN, no_object_weight=0.1, class_weight=2.0, mask_weight=5.0, dice_weight=5.0, train_num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75)
    crit = Mask2FormerLoss(cfg, weight_dict=dict(loss_cross_entropy=2.0, loss_mask=5.0, loss_dice=5.0)).to(dev); wd = dict(loss_cross_entropy=2.0, loss_mask=5.0, loss_dice=5.0)
    real, ps = B.pool(); random.seed(0); random.shuffle(real); random.shuffle(ps); val = real[:150] + ps[:150]; real, ps = real[150:], ps[150:]
    if os.environ.get("RFS") == "1": real, ps = B.repeat_factor(real), B.repeat_factor(ps)
    test = [(p, Path("data/general/eval_coco_in/seg") / (p.stem + ".png")) for p in sorted(Path("data/general/eval_coco_in/images").glob("*.jpg"))]
    print(f"train real {len(real)} pseudo {len(ps)} | val {len(val)} | test {len(test)}", flush=True)
    pre = ("stem.", "stage0.", "stage1.", "stage2.")
    bb = [p for n, p in M.named_parameters() if n.startswith(pre)]; new = [p for n, p in M.named_parameters() if not n.startswith(pre)]
    opt = torch.optim.AdamW([dict(params=bb, lr=LR * 0.1), dict(params=new, lr=LR)], weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, it / 1000) * (1 - min(it, ITERS) / ITERS) ** 0.9)
    log = open(OUT / "log.jsonl", "a"); t0 = time.time(); best = None; ri = pi = 0
    for it in range(1, ITERS + 1):
        b = []
        for _ in range(BS):
            if real and (random.random() < P_REAL or not ps): b.append(B.load_pair(*real[ri % len(real)], True, KD > 0)); ri += 1
            else: b.append(B.load_pair(*ps[pi % len(ps)], True, KD > 0)); pi += 1
        x = torch.stack([q[0] for q in b]).to(dev); y = torch.stack([q[1] for q in b]).to(dev); ml, cl = targets(y)
        with torch.autocast("cuda", torch.bfloat16): outs = M.forward_all(x, *tap(x))
        (cf, mf_) = outs[-1]; aux = [dict(masks_queries_logits=m.float(), class_queries_logits=c.float()) for c, m in outs[:-1]]
        ld = crit(mf_.float(), cf.float(), ml, cl, aux)
        loss = sum(v * wd[k.split("_aux")[0]] if not k[-1].isdigit() else v * wd["_".join(k.split("_")[:-1])] for k, v in ld.items())
        if KD > 0:
            t = torch.stack([q[2] for q in b]).to(dev); has = torch.tensor([q[3] for q in b], device=dev)
            sem = IndoorMask.semantic(cf, mf_); sem = sem / sem.sum(1, keepdim=True).clamp(min=1e-6); lq = torch.log(F.adaptive_avg_pool2d(sem, 112).clamp(min=1e-6))
            kl = (t * (torch.log(t.clamp(min=1e-6)) - lq)).sum(1).mean((1, 2)); loss = loss + KD * (kl * has).sum() / has.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 0.01 * 100); opt.step(); sch.step()
        if it % 200 == 0:
            r = dict(it=it, loss=round(float(loss), 3), min=round((time.time() - t0) / 60, 1)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if it % 5000 == 0 or it == ITERS:
            M.eval(); ev, _ = B.evaluate(tap, M, val); M.train()
            r = dict(it=it, val=ev["miou"], val_pc=ev["per_class"]); print("VAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            if best is None or ev["miou"] > best[0]: best = (ev["miou"], it); torch.save(M.state_dict(), OUT / "module.pt")
    M.load_state_dict(torch.load(OUT / "module.pt")); M.eval(); te, cm = B.evaluate(tap, M, test)
    (OUT / "eval_test.json").write_text(json.dumps(dict(best=best, test=te), indent=1)); np.save(OUT / "test_confusion.npy", cm)
    print("TEST", json.dumps(dict(best=best, test=te)), flush=True)


if __name__ == "__main__":
    main()
