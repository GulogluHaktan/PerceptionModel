"""Real 3DGS training test on a NeRF-Synthetic scene (known camera poses, no SfM needed) with gsplat's densification.
Usage (Orin): python scripts/orin_3dgs_train.py <scene_dir> [iters=3000] [width=400] [sh_degree=3] [init_points=100000]
Measures per-iteration time, Gaussian count growth, GPU/system memory, held-out PSNR, final render FPS, and the size of
the saved scene. L1 loss only (no SSIM), so PSNR is a little below a full 3DGS run. Dataset: NeRF Synthetic (Blender)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

root = Path(sys.argv[1])
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
W = int(sys.argv[3]) if len(sys.argv) > 3 else 400
deg = int(sys.argv[4]) if len(sys.argv) > 4 else 3
n0 = int(sys.argv[5]) if len(sys.argv) > 5 else 100_000
dev = "cuda"


def load(split, step=1):
    meta = json.load(open(root / f"transforms_{split}.json"))
    fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
    K = torch.tensor([[fx, 0, W / 2], [0, fx, W / 2], [0, 0, 1.0]], device=dev)
    ims, vms = [], []
    for fr in meta["frames"][::step]:
        im = Image.open(root / (fr["file_path"] + ".png")).convert("RGBA").resize((W, W), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255
        rgb = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])                      # white background
        ims.append(torch.from_numpy(rgb).to(dev).half())
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32, device=dev)
        c2w[:3, 1:3] *= -1                                                    # OpenGL -> OpenCV camera axes
        vms.append(torch.linalg.inv(c2w))
    return torch.stack(ims), torch.stack(vms), K


def mem_mb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) // 1024


t0 = time.time()
tr_im, tr_vm, K = load("train"); te_im, te_vm, _ = load("test", step=10)
print(json.dumps({"train_views": len(tr_im), "test_views": len(te_im), "width": W, "load_s": round(time.time() - t0, 1)}), flush=True)

g = torch.Generator(device=dev).manual_seed(0)
means = (torch.rand(n0, 3, device=dev, generator=g) * 2 - 1) * 1.3
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(means),
    "scales": torch.nn.Parameter(torch.full((n0, 3), math.log(0.03), device=dev)),
    "quats": torch.nn.Parameter(F.normalize(torch.randn(n0, 4, device=dev, generator=g), dim=1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((n0,), 0.1, device=dev))),
    "sh0": torch.nn.Parameter(torch.rand(n0, 1, 3, device=dev, generator=g)),
    "shN": torch.nn.Parameter(torch.zeros(n0, (deg + 1) ** 2 - 1, 3, device=dev)),
})
lrs = {"means": 1.6e-4 * 3, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
opts = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}
strategy = DefaultStrategy(verbose=False, refine_start_iter=300, refine_stop_iter=int(iters * 0.7), refine_every=100,
                           reset_every=max(1000, iters // 3), prune_opa=0.005, grow_grad2d=0.0002)
strategy.check_sanity(params, opts)
state = strategy.initialize_state(scene_scale=1.5)


def render(vm, p=params):
    cols = torch.cat([p["sh0"], p["shN"]], 1)
    return rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), cols,
                         vm, K[None].expand(len(vm), 3, 3), W, W, sh_degree=deg, packed=False, absgrad=False,
                         backgrounds=torch.ones(len(vm), 3, device=dev))   # targets are on a WHITE background


torch.cuda.reset_peak_memory_stats()
log, t_start = [], time.time()
step_ms = []
for it in range(iters):
    i = int(torch.randint(0, len(tr_im), (1,)))
    torch.cuda.synchronize(); t = time.perf_counter()
    out, alpha, info = render(tr_vm[i:i + 1])
    strategy.step_pre_backward(params, opts, state, it, info)
    loss = (out[0] - tr_im[i].float()).abs().mean()
    loss.backward()
    strategy.step_post_backward(params, opts, state, it, info, packed=False)
    for o in opts.values():
        o.step(); o.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); step_ms.append((time.perf_counter() - t) * 1000)
    if (it + 1) % max(1, iters // 6) == 0:
        w = step_ms[-max(1, iters // 6):]
        row = {"iter": it + 1, "gaussians": int(params["means"].shape[0]), "median_step_ms": round(float(np.median(w)), 1),
               "loss": round(float(loss), 4), "elapsed_min": round((time.time() - t_start) / 60, 1),
               "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20), "ram_avail_mb": mem_mb()}
        print(json.dumps(row), flush=True)

with torch.no_grad():
    psnrs = []
    for i in range(len(te_im)):
        o = render(te_vm[i:i + 1])[0][0].clamp(0, 1)
        mse = F.mse_loss(o, te_im[i].float()); psnrs.append(float(-10 * torch.log10(mse)))
    for _ in range(5): render(te_vm[:1])
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(30): render(te_vm[i % len(te_vm):i % len(te_vm) + 1])
    torch.cuda.synchronize(); fps = 30 / (time.perf_counter() - t)
n = int(params["means"].shape[0])
ckpt = Path("runs/3dgs_scene.pt"); ckpt.parent.mkdir(exist_ok=True)
torch.save({k: v.detach().half().cpu() for k, v in params.items()}, ckpt)
print(json.dumps({"final": True, "iters": iters, "width": W, "sh_degree": deg, "gaussians": n, "train_min": round((time.time() - t_start) / 60, 1),
                  "mean_it_per_s": round(1000 / float(np.mean(step_ms)), 1), "test_psnr": round(float(np.mean(psnrs)), 2),
                  "render_fps": round(fps, 1), "scene_fp16_mb": round(ckpt.stat().st_size / 2**20, 1),
                  "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}))
