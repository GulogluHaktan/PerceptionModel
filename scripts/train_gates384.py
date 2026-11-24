"""384 px gate heads by self-distillation from the 448 px system (Update 166). The backbone and pyramid stay frozen (general seg / depth unchanged);
trainable: a copy of the dense gate head (core.dense.gdet), the corner-peak head (cpk3) and the near-gate head (ng2).
Teacher: the same frozen model + original heads at 448 on the same image. Student: the image resized to 384.
Gate head targets (raw maps (B, 9L, H, W) = per level [score logit, 8 corner offsets image-normalised relative to the cell centre]): the teacher's
ABSOLUTE corner coordinates (offset + cell centre) are resampled to the student grid and re-expressed relative to the student cell centres, so the
grid change (112 -> 96 cells) adds no error; scores: BCE to the resampled teacher probability, gate cells weighted x20. cpk heat / offsets likewise
(offsets are in cells: rescaled by the grid ratio).
Usage: python scripts/train_gates384.py [out=runs/gates384]   env: ITERS=15000 BS=24 LR=3e-4 SIDE=384"""
import glob, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
for k, v in dict(PFM_GATE_HEAD="dense", PFM_GDET_STRIDE="4", PFM_GDET_K="8", PFM_GDET_LEVELS="56").items(): os.environ.setdefault(k, v)
sys.path.insert(0, "src")
from pfm.model import PerceptionModel                     # noqa: E402
from pfm.data import MEAN, STD                            # noqa: E402
from pfm.cornerpeak import CornerPeakHead, NearGateHead   # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/gates384"); OUT.mkdir(parents=True, exist_ok=True)
CK = os.environ.get("CKPT", "runs/release_v4/bigY_h_hybrid.pt"); SIDE = int(os.environ.get("SIDE", "384")); T_SIDE = 448
ITERS = int(os.environ.get("ITERS", "15000")); BS = int(os.environ.get("BS", "24")); LR = float(os.environ.get("LR", "3e-4")); dev = "cuda"
GATE_SETS = ("dr1", "sim_race", "sim_racelight", "sim_racemix", "sim_racep", "real_tii_p", "real_mrg")      # frames with gates (real_mrg = training places only)


class Images(torch.utils.data.IterableDataset):
    def __init__(self, gate_files, other_files): self.g, self.o = gate_files, other_files
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random((wi.id if wi else 0) * 7919 + int(time.time()))
        while True:
            f = rng.choice(self.g if rng.random() < 0.8 else self.o)
            try: im = Image.open(f).convert("RGB")
            except Exception: continue
            w, h = im.size; s = rng.uniform(0.6, 1.0); cw, ch = int(w * s), int(h * s); x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((T_SIDE, T_SIDE), Image.BILINEAR)
            if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
            yield torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255


def gate_maps(core, gdet, cpk, ng, img01, mean, std):
    x = (img01 - mean) / std; net = core.backbone.net
    fmap = net.stages_2(net.stages_1(net.stages_0(net.stem_2(net.stem_1(net.stem_0(x))))))
    p = core.pyramid(fmap); f = p[0] + F.interpolate(p[1], size=p[0].shape[-2:], mode="bilinear", align_corners=False) + F.interpolate(p[2], size=p[0].shape[-2:], mode="bilinear", align_corners=False)
    return dict(gdet=dense_gdet(core, gdet, p, x.shape[-2:]), cpk=cpk(img01, f), ng=ng(img01, f))


def dense_gdet(core, gdet, p, hw):
    """core.dense(...)['gdet'] with the gate head swapped for `gdet` (the dense trunk feature is computed the same way as in DenseHeads.forward)."""
    d = core.dense; saved = d.gdet; d.gdet = gdet
    try: return d(p, hw)["gdet"]
    finally: d.gdet = saved


def centres(h, w, device):
    cy = (torch.arange(h, device=device).float() + 0.5) / h; cx = (torch.arange(w, device=device).float() + 0.5) / w
    return cx.view(1, 1, 1, 1, w).expand(1, 1, 4, h, w), cy.view(1, 1, 1, h, 1).expand(1, 1, 4, h, w)


def raw_targets(t, hs, ws):
    """Teacher raw map (B, 9L, Ht, Wt) -> targets on the student grid: probability (B, L, hs, ws), offsets (B, L, 8, hs, ws)."""
    b, c, ht, wt = t.shape; L = c // 9; t = t.float().view(b, L, 9, ht, wt)
    cx, cy = centres(ht, wt, t.device); off = t[:, :, 1:].view(b, L, 4, 2, ht, wt)
    absx, absy = off[:, :, :, 0] + cx, off[:, :, :, 1] + cy
    rs = lambda a: F.interpolate(a.reshape(b, -1, ht, wt), size=(hs, ws), mode="bilinear", align_corners=False)
    ax, ay = rs(absx).view(b, L, 4, hs, ws), rs(absy).view(b, L, 4, hs, ws)
    sx, sy = centres(hs, ws, t.device)
    off_s = torch.stack((ax - sx, ay - sy), 3).view(b, L, 8, hs, ws)
    prob = rs(t[:, :, 0].sigmoid()).view(b, L, hs, ws).clamp(0, 1)
    return prob, off_s


def raw_loss(s, t):
    b, c, hs, ws = s.shape; L = c // 9; s = s.float().view(b, L, 9, hs, ws); tp, toff = raw_targets(t, hs, ws)
    w = 1 + 20 * tp; ls = (F.binary_cross_entropy_with_logits(s[:, :, 0], tp, reduction="none") * w).sum() / w.sum()
    m = (tp > 0.05).float()[:, :, None]; lo = ((s[:, :, 1:] - toff).abs() * 20 * m).sum() / (m.sum() * 8 + 1)
    return ls, lo


def cpk_loss(s, t):
    (sh, so), (th, to) = s, t; hs, ws = sh.shape[-2:]
    tp = F.interpolate(th.float().sigmoid(), size=(hs, ws), mode="bilinear", align_corners=False)
    toff = F.interpolate(to.float(), size=(hs, ws), mode="bilinear", align_corners=False)      # offsets in cells of a 112 grid -> same fraction of a cell on the 96 grid
    w = 1 + 20 * tp; lh = (F.binary_cross_entropy_with_logits(sh.float(), tp, reduction="none") * w).sum() / w.sum()
    m = (tp.amax(1, keepdim=True) > 0.1).float(); lo = ((so.float() - toff).abs() * m).sum() / (m.sum() * 2 + 1)
    return lh, lo


def main():
    root = Path("data/mix/bigG/images"); allf = [f for e in ("jpg", "png") for f in glob.glob(f"{root}/**/*.{e}", recursive=True)]
    gate = [f for f in allf if Path(f).relative_to(root).parts[0] in GATE_SETS and "red_" not in Path(f).name]; other = [f for f in allf if f not in set(gate)]
    print(f"gate frames {len(gate)}, other {len(other)}", flush=True)
    core = PerceptionModel("hybrid", teacher_dims={"dino": 384}, hybrid_name="fastvit_sa24"); core.load_state_dict(torch.load(CK, map_location="cpu")); core = core.to(dev).eval().requires_grad_(False)
    t_cpk = CornerPeakHead(); t_cpk.load_state_dict(torch.load(Path.home() / "DroneRacing_eval/runs_cmp/cpk3.pt", map_location="cpu")); t_ng = NearGateHead(); t_ng.load_state_dict(torch.load(Path.home() / "DroneRacing_eval/runs_cmp/ng2.pt", map_location="cpu"))
    t_cpk, t_ng = t_cpk.to(dev).eval().requires_grad_(False), t_ng.to(dev).eval().requires_grad_(False)
    import copy
    s_gdet, s_cpk, s_ng = copy.deepcopy(core.dense.gdet).requires_grad_(True), copy.deepcopy(t_cpk).requires_grad_(True), copy.deepcopy(t_ng).requires_grad_(True)
    t_gdet = core.dense.gdet
    params = list(s_gdet.parameters()) + list(s_cpk.parameters()) + list(s_ng.parameters())
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(params, LR, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=ITERS, pct_start=0.05)
    mean, std = MEAN.view(1, 3, 1, 1).to(dev), STD.view(1, 3, 1, 1).to(dev)
    dl = iter(torch.utils.data.DataLoader(Images(gate, other), batch_size=BS, num_workers=12, persistent_workers=True, prefetch_factor=4))
    t0 = time.time(); run = []
    for it in range(1, ITERS + 1):
        x448 = next(dl).to(dev, non_blocking=True); x384 = F.interpolate(x448, size=(SIDE, SIDE), mode="bilinear", align_corners=False, antialias=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): tt = gate_maps(core, t_gdet, t_cpk, t_ng, x448, mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16): st = gate_maps(core, s_gdet, s_cpk, s_ng, x384, mean, std)
        g1, g2 = raw_loss(st["gdet"], tt["gdet"]); c1, c2 = cpk_loss(st["cpk"], tt["cpk"]); n1, n2 = raw_loss(st["ng"], tt["ng"])
        loss = 2 * g1 + g2 + c1 + 0.5 * c2 + 0.5 * (n1 + n2)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        run.append([g1.item(), g2.item(), c1.item(), c2.item(), n1.item(), n2.item()])
        if it % 500 == 0 or it == ITERS:
            m = np.mean(run, 0); run = []
            print(json.dumps({"it": it, "min": round((time.time() - t0) / 60, 1), "gscore": round(m[0], 4), "goff": round(m[1], 4), "cpk_h": round(m[2], 4), "cpk_o": round(m[3], 4), "ng_s": round(m[4], 4), "ng_o": round(m[5], 4)}), flush=True)
            sd = core.state_dict(); sd.update({f"dense.gdet.{k}": v for k, v in s_gdet.state_dict().items()})
            torch.save(sd, OUT / "core_gdet384.pt"); torch.save(s_cpk.state_dict(), OUT / "cpk384.pt"); torch.save(s_ng.state_dict(), OUT / "ng384.pt")
    print("GATES384 DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
