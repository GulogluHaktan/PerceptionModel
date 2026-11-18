"""Build a TensorRT engine from an exported ONNX file (scripts/export_onnx.py), check it against ONNX Runtime on real images and time it. Written for TensorRT 11 (pip `tensorrt`):
networks are strongly typed there, so fp16 comes from an fp16 ONNX file (HALF=1 in export_onnx.py), not from a builder flag. On the Jetson Orin (older TensorRT, trtexec --fp16) use scripts/orin_trt_bench.py with the fp32 ONNX.
Usage: python scripts/trt_build.py runs/bigD/vit/model_448_fp16.onnx [reference fp32 onnx for the check] [check images glob]
Writes <onnx>.engine and <onnx>_trt_check.json: median / p95 latency (batch 1, GPU compute only, input already on the GPU), difference to the reference on the check images
(gate set equal, gate corner difference in px at 256x192, share of pixels with the same seg class, depth difference relative to the depth range)."""
import glob, json, os, sys, time
import numpy as np, torch, tensorrt as trt
from PIL import Image
onnx_p = sys.argv[1]; ref_p = sys.argv[2] if len(sys.argv) > 2 else onnx_p; IMGS = sys.argv[3] if len(sys.argv) > 3 else "data/real_mrg/golden/images/red_*.jpg"; eng_p = onnx_p.replace(".onnx", ".engine")
log = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(log); net = builder.create_network(0); parser = trt.OnnxParser(net, log)
if not parser.parse_from_file(onnx_p):
    for i in range(parser.num_errors): print(parser.get_error(i))
    sys.exit(1)
cfg = builder.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30); t0 = time.time(); blob = builder.build_serialized_network(net, cfg)
assert blob is not None, "engine build failed"; open(eng_p, "wb").write(blob); build_s = time.time() - t0
eng = trt.Runtime(log).deserialize_cuda_engine(blob); ctx = eng.create_execution_context(); names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
S = eng.get_tensor_shape("image")[-1]; bufs = {n: torch.zeros(tuple(eng.get_tensor_shape(n)), dtype=torch.float32, device="cuda") for n in names}
for n in names:
    assert eng.get_tensor_dtype(n) == trt.DataType.FLOAT, (n, eng.get_tensor_dtype(n)); ctx.set_tensor_address(n, bufs[n].data_ptr())
stream = torch.cuda.Stream()
def run(x):
    bufs["image"].copy_(x); torch.cuda.synchronize(); ctx.execute_async_v3(stream.cuda_stream); stream.synchronize(); torch.cuda.synchronize(); return {n: bufs[n].clone() for n in names if n != "image"}
files = sorted(glob.glob(IMGS)); files = files[:: max(1, len(files) // 24)][:24]
xs = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255 for f in files]
import onnxruntime as ort
sess = ort.InferenceSession(ref_p, providers=["CPUExecutionProvider"]); rn = [o.name for o in sess.get_outputs()]; same = 0; gpx = 0.0; segs = []; dd = []
for x in xs:
    o = run(x.cuda()); r = dict(zip(rn, sess.run(None, {"image": x.numpy()})))
    a, b = o["gates"][0].cpu().numpy(), r["gates"][0]; ka, kb = a[:, 0] > 0, b[:, 0] > 0; eq = bool((ka == kb).all()); same += eq
    if eq and ka.any(): gpx = max(gpx, float(np.abs((a[ka, 1:] - b[ka, 1:]) * np.tile([256, 192], 4)).max()))
    segs.append(float((o["seg"][0].argmax(0).cpu().numpy() == r["seg"][0].argmax(0)).mean())); d = r["depth"][0, 0]; dd.append(float(np.abs(o["depth"][0, 0].cpu().numpy() - d).mean() / (d.max() - d.min() + 1e-6)))
x = xs[0].cuda(); ts = []
for i in range(250):
    torch.cuda.synchronize(); t = time.perf_counter(); bufs["image"].copy_(x); torch.cuda.synchronize(); ctx.execute_async_v3(stream.cuda_stream); stream.synchronize(); ts.append((time.perf_counter() - t) * 1000)
ts = np.sort(ts[50:]); res = {"engine": eng_p, "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0), "engine_mb": round(blob.nbytes / 2**20, 1), "build_s": round(build_s, 1), "input": S, "latency_ms_median": round(float(np.median(ts)), 2), "latency_ms_p95": round(float(ts[int(0.95 * len(ts))]), 2),
       "fps": round(1000 / float(np.median(ts)), 1), "reference": ref_p, "check_images": len(xs), "frames_with_same_gate_set": int(same), "max_gate_corner_diff_px": round(gpx, 3), "seg_same_class_share_min": round(min(segs), 4), "depth_mean_abs_diff_rel_range_max": round(max(dd), 5)}
print(json.dumps(res)); json.dump(res, open(onnx_p.replace(".onnx", "_trt_check.json"), "w"), indent=1)
