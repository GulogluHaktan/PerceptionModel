"""3DGS timing proxy on the Orin (decision 7: measure splat render/train time first).
No real scene is needed: random Gaussians are rendered from random cameras to make targets, then a second set
of Gaussians is trained to fit them. This measures rasterisation forward speed and training-step speed/memory
versus the number of Gaussians and the image size. It is NOT a quality test and uses plain RGB colours (no
spherical harmonics, no densification), so a real run will be somewhat slower and larger.
Usage: python scripts/orin_gsplat_bench.py   (needs: gsplat, ninja, CUDA_HOME, TORCH_CUDA_ARCH_LIST=8.7 on Orin)"""
import json, math, time
import torch
from gsplat.rendering import rasterization

dev = "cuda"


def scene(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g)
    q = r(n, 4); q = q / q.norm(dim=1, keepdim=True)
    return dict(means=r(n, 3) * 0.8, quats=q, scales=torch.exp(r(n, 3) * 0.3 - 3.5), opac=torch.sigmoid(r(n) * 1.0 + 1.0),
                colors=torch.rand(n, 3, device=dev, generator=g))


def cameras(k, w, h):
    vm, Ks = [], []
    for i in range(k):
        a, b = i * 2.399, 0.5 * math.sin(i * 1.7)
        eye = torch.tensor([3 * math.cos(a) * math.cos(b), 3 * math.sin(b), 3 * math.sin(a) * math.cos(b)], device=dev)
        z = -eye / eye.norm(); up = torch.tensor([0., 1., 0.], device=dev)
        x = torch.cross(up, z, dim=0); x = x / x.norm(); y = torch.cross(z, x, dim=0)
        R = torch.stack([x, y, z]); m = torch.eye(4, device=dev); m[:3, :3] = R; m[:3, 3] = -R @ eye
        vm.append(m); Ks.append(torch.tensor([[w * 1.1, 0, w / 2], [0, w * 1.1, h / 2], [0, 0, 1.]], device=dev))
    return torch.stack(vm), torch.stack(Ks)


def render(p, vm, K, w, h):
    return rasterization(p["means"], p["quats"], p["scales"], p["opac"], p["colors"], vm, K, w, h)[0]


def run(n, w, h, views=8, iters=40):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vm, K = cameras(views, w, h)
    gt = scene(n, 0)
    with torch.no_grad():
        targets = render(gt, vm, K, w, h)
        for _ in range(3): render(gt, vm[:1], K[:1], w, h)
        torch.cuda.synchronize(); t = time.perf_counter()
        for i in range(views): render(gt, vm[i:i + 1], K[i:i + 1], w, h)
        torch.cuda.synchronize(); fwd_ms = (time.perf_counter() - t) / views * 1000
    p = {k: v.clone().requires_grad_(True) for k, v in scene(n, 1).items()}
    opt = torch.optim.Adam(list(p.values()), lr=1e-2)
    ts = []
    for it in range(iters + 5):
        i = it % views
        torch.cuda.synchronize(); t = time.perf_counter()
        out = render({**p, "quats": torch.nn.functional.normalize(p["quats"], dim=1), "opac": p["opac"].clamp(0, 1)}, vm[i:i + 1], K[i:i + 1], w, h)
        loss = (out - targets[i:i + 1]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if it >= 5: ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"gaussians": n, "size": f"{w}x{h}", "render_ms": round(fwd_ms, 2), "render_fps": round(1000 / fwd_ms, 1),
            "train_step_ms": round(ts[len(ts) // 2], 1), "train_it_per_s": round(1000 / ts[len(ts) // 2], 1),
            "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20), "loss_end": round(float(loss), 4)}


if __name__ == "__main__":
    t0 = time.time(); run(2000, 64, 64, views=2, iters=2)   # triggers the CUDA JIT build
    print(json.dumps({"jit_warmup_s": round(time.time() - t0, 1)}), flush=True)
    for n, w, h in [(100_000, 640, 480), (300_000, 640, 480), (1_000_000, 640, 480), (300_000, 1280, 720), (1_000_000, 1280, 720)]:
        try:
            print(json.dumps(run(n, w, h)), flush=True)
        except torch.OutOfMemoryError:
            print(json.dumps({"gaussians": n, "size": f"{w}x{h}", "error": "out of memory"}), flush=True)
            torch.cuda.empty_cache()
