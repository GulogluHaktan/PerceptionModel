"""Train the corner refiner (src/pfm/refine.py). The main PerceptionModel is not touched.

Data (true inner corners only, every corner sampled where it lies inside the image and is visible):
  * DroneRacing on-policy crops (384x384, their renderer; ~/DroneRacing_eval/scripts/collect_refine_crops.sh, training seeds 1-12 only,
    VAL_SEED held out). These are exactly what the detector sees inside their closed loop.
  * our own sim racing sets and real TII training frames (corners_full labels at 256x192 scale, images 640x480), so the refiner stays general.
  Real held-out red/TII golden frames are never used here (eval only: scripts/eval_refiner.py).
Coarse corners are simulated: true corner + error whose spread grows with gate size (as measured for the dense head, Update 103), plus a
noisy centroid and size, so the refiner learns to undo exactly that kind of error.
Usage: python scripts/train_refiner.py [out=runs/refiner_v1]   env: CROPS=~/DroneRacing_eval/runs_crops/ours ITERS=20000 VAL_SEED=12"""
import glob, io, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageFilter
sys.path.insert(0, "src")
from pfm.refine import CornerRefiner, P, sample_patches, to_patch   # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/refiner_v1"); OUT.mkdir(parents=True, exist_ok=True)
CROPS = [Path(os.path.expanduser(c)) for c in os.environ.get("CROPS", "~/DroneRacing_eval/runs_crops/ours").split(",")]      # comma list of crop folders
VAL_SEED = int(os.environ.get("VAL_SEED", "12")); CROP_STRIDE = int(os.environ.get("CROP_STRIDE", "1"))      # every n-th frame (RAM)
ITERS = int(os.environ.get("ITERS", "20000")); BS = int(os.environ.get("BS", "64")); K = 8; dev = "cuda"
OWN_SETS = os.environ.get("OWN_SETS", "sim_race,sim_racep,sim_races,sim_racelight,sim_racemix,sim_outdoor,real_tii,real_tii_p").split(",")
P_OWN = float(os.environ.get("P_OWN", "0.35"))      # share of frames from our own sets
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def load_crops(seeds):
    imgs, cors, viss = [], [], []
    for f in sorted(f for c in CROPS for f in c.glob("data_*.pt")):
        sd = int(f.stem.split("_")[1])
        if sd not in seeds: continue
        d = torch.load(f, map_location="cpu")
        cf = d["corners"].float()
        sane = torch.isfinite(cf).all(-1).all(-1) & (cf.abs() < 3.0).all(-1).all(-1)      # gates partly behind the camera project to huge values: centroid unusable
        d["slot_valid"] = d["slot_valid"] & sane
        ok = d["slot_valid"] & d["vis"].any(-1)              # [N,G] gates with at least one visible corner in the crop
        keep = ok.any(1).nonzero().squeeze(1)[::CROP_STRIDE]
        imgs.append(d["img"][keep]); cors.append(d["corners"][keep].float() * d["hw"][0]); viss.append(d["vis"][keep] & d["slot_valid"][keep][..., None])
        print(f"crops {f.parent.name} seed {sd}: {len(keep)} frames with a visible gate corner", flush=True)
    if not imgs: return None
    return torch.cat(imgs), torch.cat(cors), torch.cat(viss)


def own_index():
    roots = [Path("data/train_sim/images"), Path("data/train_real/images"), Path("data/mix/bigG/images")]
    files = {}
    for st in OWN_SETS:
        for r in roots:
            if (r / st).exists():
                for p in (r / st).iterdir(): files.setdefault(f"{st}__{p.stem}", p)
    items = []
    for lab in Path("data/mix/labels/corners_full").glob("*.json"):
        if lab.stem in files: items.append((files[lab.stem], lab))
    print(f"own frames with corner labels: {len(items)}", flush=True)
    return items


def own_frame(item):
    p, lab = item; im = Image.open(p).convert("RGB"); w, h = im.size
    if random.random() < 0.2: im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.3:
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(30, 90)); im = Image.open(b).convert("RGB")
    sx, sy = w / 256.0, h / 192.0; C, V = [], []
    for r in json.load(open(lab)):
        c = [[np.nan, np.nan] if q is None else q for q in r["corners_px"]]
        if len(c) != 4: continue
        c = np.array(c, np.float32) * [sx, sy]
        v = np.array(r.get("visible", [q is not None for q in r["corners_px"]]), bool) & ~np.isnan(c[:, 0])
        if (np.array(r.get("depths", [1] * 4)) < 0.3).any() or np.isnan(c).any(): v[:] = False      # hidden corners without a label: centroid unknown
        C.append(np.nan_to_num(c)); V.append(v)
    img = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    if not C: return img, torch.zeros(0, 4, 2), torch.zeros(0, 4, dtype=torch.bool)
    return img, torch.from_numpy(np.stack(C)).float(), torch.from_numpy(np.stack(V))


def make_samples(img, C, V, gen=random):
    """one frame -> K (patch, target offset in patch px, valid) samples with simulated coarse corners."""
    H, W = img.shape[-2:]
    inside = (C[..., 0] >= 1) & (C[..., 0] < W - 1) & (C[..., 1] >= 1) & (C[..., 1] < H - 1)
    cand = (V & inside).nonzero()
    if len(cand) == 0: return None
    pick = cand[torch.randint(len(cand), (K,))]
    tgt = C[pick[:, 0], pick[:, 1]]                                # [K,2]
    gates = C[pick[:, 0]]                                          # [K,4,2]
    size = (gates.amax(1) - gates.amin(1)).amax(-1).clamp(min=4)
    sig = (0.025 * size + 0.8) * torch.empty(K).uniform_(0.4, 2.0)
    err = torch.randn(K, 2) * sig[:, None]
    big = torch.rand(K) < 0.12
    err[big] = (torch.rand(int(big.sum()), 2) * 2 - 1) * (0.12 * size[big, None] + 4)
    coarse = tgt + err
    cen = gates.mean(1) + torch.randn(K, 2) * sig[:, None] * 0.7
    sz = size * torch.empty(K).uniform_(0.9, 1.1)
    x = img[None].float() / 255
    patches, s, R = sample_patches(x, torch.zeros(K, dtype=torch.long), coarse, cen, sz)
    off = to_patch(tgt, coarse, s, R)
    ok = (off.abs() < P / 2 - 1.5).all(-1)
    return patches, off, ok


class Frames(torch.utils.data.IterableDataset):
    def __init__(self, crops, own, p_own):
        self.crops, self.own, self.p_own = crops, own, p_own

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); sd = (wi.id if wi else 0) + int(time.time() * 1000) % 100000
        random.seed(sd); torch.manual_seed(sd)
        while True:
            if self.own and (self.crops is None or random.random() < self.p_own):
                img, C, V = own_frame(random.choice(self.own))
            else:
                i = random.randrange(len(self.crops[0])); img, C, V = self.crops[0][i], self.crops[1][i], self.crops[2][i]
            r = make_samples(img, C, V)
            if r is not None: yield r


def blur(p):
    """random Gaussian / directional blur on a share of patches (derotation resampling, motion blur in the loop camera)."""
    n = p.shape[0]; sel = torch.rand(n, device=p.device) < float(os.environ.get("P_BLUR", "0.5"))
    if not sel.any(): return p
    k = 7; t = torch.arange(k, device=p.device, dtype=torch.float32) - k // 2
    sig = torch.empty(n, device=p.device).uniform_(0.3, 1.6)
    ang = torch.rand(n, device=p.device) * math.pi; aniso = torch.empty(n, device=p.device).uniform_(1.0, 3.0)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    u = xx[None] * ang.cos()[:, None, None] + yy[None] * ang.sin()[:, None, None]; v = -xx[None] * ang.sin()[:, None, None] + yy[None] * ang.cos()[:, None, None]
    ker = torch.exp(-(u ** 2) / (2 * (sig * aniso)[:, None, None] ** 2) - (v ** 2) / (2 * sig[:, None, None] ** 2)); ker = ker / ker.sum((1, 2), keepdim=True)
    q = F.conv2d(F.pad(p[sel].reshape(1, -1, P, P), (k // 2,) * 4, mode="replicate"), ker[sel].repeat_interleave(3, 0)[:, None], groups=int(sel.sum()) * 3)
    p = p.clone(); p[sel] = q.view(-1, 3, P, P); return p


def photometric(p):
    p = blur(p)
    n = p.shape[0]
    a = torch.empty(n, 1, 1, 1, device=p.device).uniform_(0.6, 1.4); b = torch.empty(n, 1, 1, 1, device=p.device).uniform_(-0.15, 0.15)
    g = torch.empty(n, 1, 1, 1, device=p.device).uniform_(0.7, 1.4)
    p = ((p * a + b).clamp(0, 1) ** g)
    p = p + torch.randn_like(p) * torch.empty(n, 1, 1, 1, device=p.device).uniform_(0, 0.04)
    return p.clamp(0, 1)


def main():
    seeds = {int(f.stem.split("_")[1]) for c in CROPS for f in c.glob("data_*.pt")}
    crops = load_crops(seeds - {VAL_SEED}); val = load_crops({VAL_SEED})
    own = own_index()
    dl = torch.utils.data.DataLoader(Frames(crops, own, P_OWN), batch_size=BS, num_workers=int(os.environ.get("WORKERS", "12")),
                                     collate_fn=lambda b: [torch.cat(t) for t in zip(*b)], persistent_workers=True, prefetch_factor=4)
    vdata = []
    if val is not None:
        g = torch.Generator().manual_seed(1); torch.manual_seed(123)
        for i in torch.randperm(len(val[0]), generator=g)[:600].tolist():
            r = make_samples(val[0][i], val[1][i], val[2][i])
            if r is not None: vdata.append(r)
    m = CornerRefiner().to(dev); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ITERS, pct_start=0.05)
    print(f"refiner params {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M", flush=True)
    t0 = time.time(); it = 0; run = []
    for patches, off, ok in dl:
        patches, off, ok = patches.to(dev, non_blocking=True), off.to(dev), ok.to(dev)
        patches = photometric(patches)
        if torch.rand(1).item() < 0.5: patches = patches.transpose(2, 3); off = off.flip(-1)      # mirror about the canonical diagonal (keeps the centroid direction)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            po, vis, _ = m(patches)
        po = po.float(); vis = vis.float()
        l_off = (F.smooth_l1_loss(po[ok], off[ok], beta=1.0, reduction="mean")) if ok.any() else po.sum() * 0
        l_vis = F.binary_cross_entropy_with_logits(vis, ok.float())
        loss = l_off + 0.2 * l_vis
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
        run.append(l_off.item())
        if it % 500 == 0 or it == ITERS:
            msg = {"iter": it, "loss_off": round(float(np.mean(run)), 4), "sec": round(time.time() - t0)}
            run = []
            if vdata:
                m.eval(); e0, e1 = [], []
                with torch.no_grad():
                    for p_, o_, k_ in vdata:
                        p_, o_, k_ = p_.to(dev), o_.to(dev), k_.to(dev)
                        q, _, _ = m(p_)
                        e0.append(o_[k_].norm(dim=-1).cpu()); e1.append((q[k_] - o_[k_]).norm(dim=-1).cpu())
                e0, e1 = torch.cat(e0), torch.cat(e1)
                msg.update(val_coarse_median_patchpx=round(e0.median().item(), 2), val_refined_median_patchpx=round(e1.median().item(), 2),
                           val_refined_p90=round(e1.quantile(0.9).item(), 2), val_refined_lt1=round((e1 < 1).float().mean().item(), 3))
                m.train()
            print(json.dumps(msg), flush=True)
            torch.save(m.state_dict(), OUT / "refiner.pt")
            if it % 2500 == 0: torch.save(m.state_dict(), OUT / f"refiner_{it}.pt")      # snapshots: the best one on real held-out frames is picked afterwards (eval_refiner.py)
        if it >= ITERS: break
    print("REFINER DONE", OUT / "refiner.pt", flush=True)


if __name__ == "__main__":
    main()
