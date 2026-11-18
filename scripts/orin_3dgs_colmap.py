"""3DGS on REAL photos with COLMAP poses (e.g. Mip-NeRF 360 scenes): init from the SfM points, L1 + SSIM, gsplat densification.
Usage (Orin): python scripts/orin_3dgs_colmap.py <scene_dir> [iters=7000] [width=800] [sh_degree=3]
<scene_dir> must contain images/ and sparse/0/{cameras,images,points3D}.bin (PINHOLE or SIMPLE_PINHOLE cameras). Every 8th image is a test view.
Writes runs/3dgs_real/: sheet.png (GT | render | depth for test views), scene.pt, stats.json."""
import json, math, struct, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

root = Path(sys.argv[1])
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 7000
W = int(sys.argv[3]) if len(sys.argv) > 3 else 800
deg = int(sys.argv[4]) if len(sys.argv) > 4 else 3
dev = "cuda"
out = Path("runs/3dgs_real"); out.mkdir(parents=True, exist_ok=True)


def read_model(d):
    with open(d / "cameras.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; cams = {}
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            npar = {0: 3, 1: 4}[mid]                                    # SIMPLE_PINHOLE, PINHOLE only
            par = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            fx, fy, cx, cy = (par[0], par[0], par[1], par[2]) if mid == 0 else par
            cams[cid] = (w, h, fx, fy, cx, cy)
    imgs = []
    with open(d / "images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00": name += c
            npts = struct.unpack("<Q", f.read(8))[0]; f.read(24 * npts)
            imgs.append((name.decode(), np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid))
    with open(d / "points3D.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; xyz = np.zeros((n, 3), np.float32); rgb = np.zeros((n, 3), np.float32)
        for i in range(n):
            f.read(8); xyz[i] = struct.unpack("<ddd", f.read(24)); rgb[i] = struct.unpack("<BBB", f.read(3)); f.read(8)
            f.read(8 * struct.unpack("<Q", f.read(8))[0])   # track entries: image_id (I) + point2D_idx (I)
    return cams, sorted(imgs), xyz, rgb / 255


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


t0 = time.time()
cams, imgs, xyz, rgb = read_model(root / "sparse" / "0")
cw, chh = cams[imgs[0][3]][:2]
H = int(round(W * chh / cw)); s = W / cw
Ks, vms, ims, centers = [], [], [], []
for name, q, t, cid in imgs:
    _, _, fx, fy, cx, cy = cams[cid]
    Ks.append(torch.tensor([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]]))
    R = qvec2rot(q); M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    vms.append(torch.tensor(M, dtype=torch.float32)); centers.append(-R.T @ t)
    im = Image.open(root / "images" / name).convert("RGB").resize((W, H), Image.LANCZOS)
    ims.append(torch.from_numpy(np.asarray(im).copy()))
Ks, vms, ims = torch.stack(Ks).to(dev), torch.stack(vms).to(dev), torch.stack(ims).to(dev)       # images stay uint8 on the GPU
test = list(range(0, len(imgs), 8)); train = [i for i in range(len(imgs)) if i not in set(test)]
centers = np.array(centers); scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
print(json.dumps({"images": len(imgs), "train": len(train), "test": len(test), "size": f"{W}x{H}", "sfm_points": len(xyz),
                  "scene_scale": round(scene_scale, 2), "load_s": round(time.time() - t0, 1)}), flush=True)

# init from SfM points (cap 200k), scale from the mean distance to 3 neighbours
sel = np.random.default_rng(0).permutation(len(xyz))[:200_000]
P = torch.tensor(xyz[sel], device=dev); C = torch.tensor(rgb[sel], device=dev)
d3 = []
for i in range(0, len(P), 512):
    dd = torch.cdist(P[i:i + 512], P); d3.append(dd.topk(4, largest=False).values[:, 1:].mean(1))
d3 = torch.cat(d3).clamp(min=1e-6)
n0 = len(P)
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(P.clone()), "scales": torch.nn.Parameter(torch.log(d3)[:, None].repeat(1, 3)),
    "quats": torch.nn.Parameter(F.normalize(torch.randn(n0, 4, device=dev), dim=1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((n0,), 0.1, device=dev))),
    "sh0": torch.nn.Parameter(((C - 0.5) / 0.28209479)[:, None, :]),
    "shN": torch.nn.Parameter(torch.zeros(n0, (deg + 1) ** 2 - 1, 3, device=dev))})
lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
opts = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}
strategy = DefaultStrategy(verbose=False, refine_start_iter=300, refine_stop_iter=int(iters * 0.7), refine_every=100,
                           reset_every=max(1000, iters // 3), prune_opa=0.005, grow_grad2d=0.0002)
strategy.check_sanity(params, opts); state = strategy.initialize_state(scene_scale=scene_scale)


def render(vm, K, mode="RGB"):
    return rasterization(params["means"], F.normalize(params["quats"], dim=1), torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                         torch.cat([params["sh0"], params["shN"]], 1), vm, K, W, H, sh_degree=deg, packed=False, render_mode=mode)


def ssim(a, b, win=11, sg=1.5):
    g = torch.exp(-(torch.arange(win, device=dev).float() - win // 2) ** 2 / (2 * sg ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(3, 1, win, win).contiguous(); c = dict(padding=win // 2, groups=3)
    ma, mb = F.conv2d(a, w, **c), F.conv2d(b, w, **c)
    va, vb, cab = F.conv2d(a * a, w, **c) - ma ** 2, F.conv2d(b * b, w, **c) - mb ** 2, F.conv2d(a * b, w, **c) - ma * mb
    return (((2 * ma * mb + 0.01 ** 2) * (2 * cab + 0.03 ** 2)) / ((ma ** 2 + mb ** 2 + 0.01 ** 2) * (va + vb + 0.03 ** 2))).mean()


torch.cuda.reset_peak_memory_stats(); step_ms = []; t_start = time.time()
for it in range(iters):
    i = train[int(torch.randint(0, len(train), (1,)))]
    torch.cuda.synchronize(); t = time.perf_counter()
    r, alpha, info = render(vms[i:i + 1], Ks[i:i + 1])
    strategy.step_pre_backward(params, opts, state, it, info)
    gt = ims[i].float().div(255)[None].permute(0, 3, 1, 2); pr = r.permute(0, 3, 1, 2)
    loss = 0.8 * (pr - gt).abs().mean() + 0.2 * (1 - ssim(pr, gt))
    loss.backward(); strategy.step_post_backward(params, opts, state, it, info, packed=False)
    for o in opts.values():
        o.step(); o.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); step_ms.append((time.perf_counter() - t) * 1000)
    if (it + 1) % max(1, iters // 7) == 0:
        w = step_ms[-max(1, iters // 7):]
        print(json.dumps({"iter": it + 1, "gaussians": int(params["means"].shape[0]), "median_step_ms": round(float(np.median(w)), 1),
                          "loss": round(float(loss), 4), "elapsed_min": round((time.time() - t_start) / 60, 1),
                          "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)

psnr, ss, rows = [], [], []
with torch.no_grad():
    for k, i in enumerate(test):
        r, _, _ = render(vms[i:i + 1], Ks[i:i + 1], "RGB+ED")
        rgbp = r[0, ..., :3].clamp(0, 1); dep = r[0, ..., 3]
        gt = ims[i].float().div(255)
        psnr.append(float(-10 * torch.log10(F.mse_loss(rgbp, gt)))); ss.append(float(ssim(rgbp.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        if k in (0, len(test) // 3, 2 * len(test) // 3, len(test) - 1):
            dn = dep.cpu().numpy(); lo, hi = np.percentile(dn, 2), np.percentile(dn, 98); x = np.clip(1 - (dn - lo) / max(hi - lo, 1e-6), 0, 1)
            rr = np.clip(1.5 - np.abs(4 * x - 3), 0, 1); gg = np.clip(1.5 - np.abs(4 * x - 2), 0, 1); bb = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
            dc = (np.stack([rr, gg, bb], -1) * 255).astype(np.uint8)
            rows.append(np.concatenate([(gt.cpu().numpy() * 255).astype(np.uint8), (rgbp.cpu().numpy() * 255).astype(np.uint8), dc], 1))
    torch.cuda.synchronize(); t = time.perf_counter()
    for k in range(20): render(vms[test[k % len(test)]:test[k % len(test)] + 1], Ks[test[k % len(test)]:test[k % len(test)] + 1])
    torch.cuda.synchronize(); fps = 20 / (time.perf_counter() - t)
sheet = Image.fromarray(np.concatenate(rows, 0)); sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(out / "sheet.png")
torch.save({k: v.detach().half().cpu() for k, v in params.items()}, out / "scene.pt")
stats = {"final": True, "iters": iters, "size": f"{W}x{H}", "gaussians": int(params["means"].shape[0]), "train_min": round((time.time() - t_start) / 60, 1),
         "mean_it_per_s": round(1000 / float(np.mean(step_ms)), 1), "test_psnr": round(float(np.mean(psnr)), 2), "test_ssim": round(float(np.mean(ss)), 3),
         "render_fps": round(fps, 1), "scene_fp16_mb": round((out / "scene.pt").stat().st_size / 2**20, 1), "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
json.dump(stats, open(out / "stats.json", "w")); print(json.dumps(stats))
