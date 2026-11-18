"""End-to-end frame pipeline on the Orin: camera-like uint8 frame -> GPU resize/normalise -> TensorRT -> argmax/depth/corner
post-processing -> result back on the CPU. Reports per-stage and total time per frame, i.e. the FPS a real loop gets,
not just the engine time. Also times JPEG encode/decode on the CPU (frame logging, data engine).
Usage (Orin): python scripts/orin_pipeline_bench.py <engine.plan> [size=384] [cam_w=1280] [cam_h=720]
Engine from scripts/orin_trt_bench.py (fp32 I/O: image, depth, seg, corners). TensorRT Python comes from the system package."""
import io, json, sys, time
sys.path.append("/usr/lib/python3.12/dist-packages")
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from PIL import Image

plan, size = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 384
cw, ch = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1280, 720)
dev = "cuda"
logger = trt.Logger(trt.Logger.WARNING)
engine = trt.Runtime(logger).deserialize_cuda_engine(open(plan, "rb").read())
ctx = engine.create_execution_context()
shapes = {n: tuple(engine.get_tensor_shape(n)) for n in [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]}
bufs = {n: torch.zeros(s, device=dev, dtype=torch.float32) for n, s in shapes.items()}
for n, b in bufs.items():
    ctx.set_tensor_address(n, b.data_ptr())
stream = torch.cuda.Stream()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
frame = torch.randint(0, 255, (ch, cw, 3), dtype=torch.uint8).pin_memory()      # camera frame in pinned memory
out_seg = torch.zeros(1, size, size, dtype=torch.uint8).pin_memory()
out_dep = torch.zeros(1, 1, size, size, dtype=torch.float16).pin_memory()
ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]


def one():
    with torch.cuda.stream(stream):
        ev[0].record(stream)
        x = frame.to(dev, non_blocking=True).permute(2, 0, 1)[None].float()          # H2D uint8 then float
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        bufs["image"].copy_((x - mean) / std)
        ev[1].record(stream)
        ctx.execute_async_v3(stream.cuda_stream)
        ev[2].record(stream)
        seg = bufs["seg"].argmax(1).to(torch.uint8)
        d = bufs["depth"]; d = ((d - d.amin()) / (d.amax() - d.amin() + 1e-6)).half()
        cor = bufs["corners"].flatten(2).argmax(2)                                  # one peak per corner map
        ev[3].record(stream)
        out_seg.copy_(seg, non_blocking=True); out_dep.copy_(d, non_blocking=True)
        ev[4].record(stream)
    stream.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(4)]


for _ in range(30):
    one()
rows, t0 = [], time.perf_counter()
n = 300
for _ in range(n):
    rows.append(one())
wall = time.perf_counter() - t0
m = np.median(np.array(rows), axis=0)
res = {"engine": plan, "size": size, "camera": f"{cw}x{ch}", "frames": n, "loop_fps": round(n / wall, 1), "loop_ms": round(wall / n * 1000, 2),
       "stage_ms": {"upload+resize+normalise": round(float(m[0]), 2), "tensorrt": round(float(m[1]), 2), "postprocess": round(float(m[2]), 2), "download": round(float(m[3]), 2)}}
# Same loop captured in ONE CUDA graph (upload + resize + TensorRT + post-processing + copies), replayed once per frame.
# The camera frame must land in the same pinned buffer every time (a real driver would DMA into it).
frame_dev = torch.empty(ch, cw, 3, dtype=torch.uint8, device=dev)
out_cor = torch.zeros(1, 4, dtype=torch.int64).pin_memory()


def ops():
    frame_dev.copy_(frame, non_blocking=True)
    x = F.interpolate(frame_dev.permute(2, 0, 1)[None].float(), size=(size, size), mode="bilinear", align_corners=False)
    bufs["image"].copy_((x - mean) / std)
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    seg = bufs["seg"].argmax(1).to(torch.uint8)
    d = bufs["depth"]; d = ((d - d.amin()) / (d.amax() - d.amin() + 1e-6)).half()
    cor = bufs["corners"].flatten(2).argmax(2)
    out_seg.copy_(seg, non_blocking=True); out_dep.copy_(d, non_blocking=True); out_cor.copy_(cor, non_blocking=True)


try:
    gs = torch.cuda.Stream()
    with torch.cuda.stream(gs):
        for _ in range(10):
            ops()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=gs):
        ops()
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay(); torch.cuda.synchronize()          # wait for every frame, like a real loop that needs the result
    gw = time.perf_counter() - t0
    res["graph_loop"] = {"loop_fps": round(n / gw, 1), "loop_ms": round(gw / n * 1000, 2),
                         "speedup_vs_eager_loop": round(res["loop_ms"] / (gw / n * 1000), 2)}
except Exception as e:
    res["graph_loop"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
# CPU JPEG cost at camera resolution (frame logging / data engine)
img = Image.fromarray(np.asarray(frame))
buf = io.BytesIO(); t = time.perf_counter()
for _ in range(20):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
enc = (time.perf_counter() - t) / 20 * 1000
data = buf.getvalue(); t = time.perf_counter()
for _ in range(20):
    Image.open(io.BytesIO(data)).convert("RGB")
dec = (time.perf_counter() - t) / 20 * 1000
res["jpeg_cpu_ms"] = {"encode_q90": round(enc, 1), "decode": round(dec, 1), "size_kb": round(len(data) / 1024)}
print(json.dumps(res))
