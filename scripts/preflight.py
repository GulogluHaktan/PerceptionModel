"""Run FIRST on the 5090: python scripts/preflight.py configs/v0_5090.yaml [n_images]
Checks GPU/torch, disk vs cache estimate, and finds the largest batch that really fits by running
real training steps for BOTH students (OOM backoff). Prints a batch to put in the config."""
import shutil, sys, yaml, torch
sys.path.insert(0, "src")
from pfm.distill import distill_loss
from pfm.model import PerceptionModel

cfg = yaml.safe_load(open(sys.argv[1]))
n_img = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
assert torch.cuda.is_available(), "CUDA not available"
p = torch.cuda.get_device_properties(0)
print(f"GPU {p.name}  {p.total_memory/2**30:.1f} GB  capability {p.major}.{p.minor}  torch {torch.__version__}")
x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item(); print("sm kernels ok")

size = cfg["size"]; g = size // 14
per_img = {"dinov2": 384 * g * g * 2, "depth_da2": size * size * 2, "eradio": 1536 * (size // 16) ** 2 * 2}
mb = sum(per_img.get(t["kind"], 0) for t in cfg["teachers"]) / 2**20
free = shutil.disk_usage(".").free / 2**30
print(f"cache estimate {mb:.2f} MB/img -> {mb*n_img/1024:.0f} GB for {n_img} images; free disk {free:.0f} GB",
      "OK" if mb * n_img / 1024 < free * 0.8 else "TOO BIG: lower size / n images")

dims = {"dino": 384}
for st in ["vit", "hybrid"]:
    best = None
    for bs in [128, 96, 64, 48, 32, 24, 16, 8, 4]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = PerceptionModel(st, teacher_dims=dims, hybrid_name=cfg.get("hybrid_name", "fastvit_sa24")).cuda()
            opt = torch.optim.AdamW(m.parameters(), 1e-4)
            xb = torch.randn(bs, 3, size, size, device="cuda")
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = m(xb)
                tgt = {"dino": (torch.randn(bs, 384, g, g, device="cuda"), torch.randn(bs, 384, device="cuda"))}
                loss, _ = distill_loss({k: (a.float(), b.float()) for k, (a, b) in out["teacher"].items()}, tgt)
                loss = loss + out["seg"].float().mean() + out["corners"].float().mean() + out["slots"].float().mean() + out["depth"].float().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            best = (bs, torch.cuda.max_memory_allocated() / 2**30)
            break
        except torch.OutOfMemoryError:
            del_ = None
            continue
        finally:
            m = opt = xb = out = loss = None
    print(f"{st}: largest batch that fits = {best[0]} (peak {best[1]:.1f} GB)" if best else f"{st}: nothing fits")
print("Set `batch:` in the config to the smaller of the two (or per-student).")
