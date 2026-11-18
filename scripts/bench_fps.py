"""Speed of a trained student on this GPU, batch 1 (one camera frame at a time). Reports median / p95 milliseconds and frames per second for:
  model only (input tensor already on the GPU, 448x448) in fp32 and fp16 (autocast), and end to end from a uint8 camera frame in RAM (640x480): upload, resize, normalise, network, gate decode, results back on the CPU.
Run with NOTHING else on the GPU, otherwise the numbers are too low. Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/bigD KIND=vit python scripts/bench_fps.py [size=448]"""
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); S = int(sys.argv[1]) if len(sys.argv) > 1 else 448
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
if K == "hybrid":
    from timm.utils import reparameterize_model
    m = reparameterize_model(m)
def timeit(fn, n=300, warm=50):
    ts = []
    for i in range(n + warm):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    ts = np.sort(ts[warm:]); return {"ms_median": round(float(np.median(ts)), 2), "ms_p95": round(float(ts[int(0.95 * n)]), 2), "fps": round(1000 / float(np.median(ts)), 1)}
x = torch.rand(1, 3, S, S, device="cuda"); frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8); res = {"ckpt": CK, "kind": K, "size": S, "gpu": torch.cuda.get_device_name(0), "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}
def e2e(half):
    t = torch.from_numpy(frame).cuda().permute(2, 0, 1)[None].float() / 255; t = normalize(F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False, antialias=True))
    with torch.autocast("cuda", dtype=torch.float16, enabled=half): o = m(t, with_teacher_heads=False)
    return o["slots"].float().cpu().numpy(), o["seg"].argmax(1).to(torch.uint8).cpu().numpy(), o["depth"].float().cpu().numpy()
with torch.no_grad():
    res["model_only_fp32"] = timeit(lambda: m(x, with_teacher_heads=False))
    def f16():
        with torch.autocast("cuda", dtype=torch.float16): m(x, with_teacher_heads=False)
    res["model_only_fp16"] = timeit(f16); res["end_to_end_fp32"] = timeit(lambda: e2e(False)); res["end_to_end_fp16"] = timeit(lambda: e2e(True))
print(json.dumps(res)); json.dump(res, open(f"runs/eval/bench_fps_{os.path.basename(CK)}_{K}_{S}.json", "w"), indent=1)
