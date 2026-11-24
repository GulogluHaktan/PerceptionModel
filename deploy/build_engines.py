"""Build the TensorRT engines from the bundle's ONNX graphs ON THE TARGET DEVICE (engines are not portable between GPUs / TensorRT versions).
TensorRT 10 (the Orin has 10.16): fp16 builder flag; the decoded-gate outputs and the small heads stay accurate in fp16 (checked by check_device.py).
  system_448.plan  from system_448.onnx  (static 1x3x448x448)
  refiner.plan     from refiner.onnx     (dynamic N: profile 1 / 16 / 64 patches)
INT8 (--int8): INT8 + fp16 mixed; activation ranges from the calibration cache bundle/int8_calib.cache (made on the dev PC by
deploy/make_int8_calib.py from ~500 mixed images; the cache is a portable text file, so the Orin needs no images). Output layers stay fp16.
Run: python deploy/build_engines.py [bundle=deploy/bundle] [--fp32 | --int8]      (Orin: system python3 has tensorrt; or trtexec, see README)"""
import json, os, re, sys, time
from pathlib import Path
sys.path.append("/usr/lib/python3.12/dist-packages")
import tensorrt as trt

B = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), Path(__file__).resolve().parent / "bundle")); FP16 = "--fp32" not in sys.argv; INT8 = "--int8" in sys.argv; QDQ = "--qdq" in sys.argv      # --qdq: system_448.onnx already has Q/DQ nodes (ModelOpt, deploy/quantize_qdq.py)
lg = trt.Logger(trt.Logger.WARNING); trt.init_libnvinfer_plugins(lg, "")       # plugin ops (m2f head: MultiscaleDeformableAttnPlugin_TRT)


class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    """Cache-only calibrator (the device build never sees images); deploy/make_int8_calib.py subclasses it with real batches."""
    def __init__(self, cache):
        super().__init__(); self.cache = Path(cache)
    def get_batch_size(self): return 1
    def get_batch(self, names): return None
    def read_calibration_cache(self): return self.cache.read_bytes() if self.cache.exists() else None
    def write_calibration_cache(self, c): self.cache.write_bytes(bytes(c))


INT8_LAYERS = os.environ.get("INT8_LAYERS", r"backbone/net/(stem|stages_0|stages_1)/")     # regex on the layer name: only these may run INT8


def keep_outputs_fp16(net):
    """Mixed precision: layers whose name matches INT8_LAYERS may run INT8, every other layer (and every graph output) is pinned to fp16."""
    outs = {net.get_output(i).name for i in range(net.num_outputs)}; rx = re.compile(INT8_LAYERS); n8 = 0
    for i in range(net.num_layers):
        L = net.get_layer(i)
        if L.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.SLICE, trt.LayerType.GATHER, trt.LayerType.CONCATENATION, trt.LayerType.SHUFFLE) and not rx.search(L.name): continue
        if any(L.get_output(j).name in outs for j in range(L.num_outputs)) or not rx.search(L.name):
            if L.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE): L.precision = trt.float16
        else: n8 += 1
    print(f"INT8-eligible layers: {n8} (regex {INT8_LAYERS})", flush=True)
    return net


def build(onnx, plan, profile=None, calib=None):
    t0 = time.time(); bld = trt.Builder(lg); net = bld.create_network(0); prs = trt.OnnxParser(net, lg)
    if not prs.parse(onnx.read_bytes()):
        raise SystemExit("\n".join(str(prs.get_error(i)) for i in range(prs.num_errors)))
    cfg = bld.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if FP16 and hasattr(trt.BuilderFlag, "FP16"): cfg.set_flag(trt.BuilderFlag.FP16)
    if QDQ and calib is None and "system" in onnx.name: cfg.set_flag(trt.BuilderFlag.INT8)      # explicit quantization: scales come from the graph
    if calib is not None:
        cfg.set_flag(trt.BuilderFlag.INT8); cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS) if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS") else None
        cfg.int8_calibrator = calib; keep_outputs_fp16(net)
    if profile:
        p = bld.create_optimization_profile(); p.set_shape(*profile); cfg.add_optimization_profile(p)
    ser = bld.build_serialized_network(net, cfg); assert ser is not None, f"build failed: {onnx}"
    plan.write_bytes(bytes(ser)); print(f"{plan.name}: {plan.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f} s, fp16={FP16}, int8={calib is not None}, TensorRT {trt.__version__}", flush=True)


if __name__ == "__main__":
    if INT8: assert (B / "int8_calib.cache").exists(), "no bundle/int8_calib.cache (deploy/make_int8_calib.py on the dev PC)"
    SYS = json.loads((B / "bundle.json").read_text())["graphs"]["system"]      # system_<S>.onnx (input side from bundle.json)
    build(B / SYS, B / SYS.replace(".onnx", ".plan"), calib=CacheCalibrator(B / "int8_calib.cache") if INT8 else None)
    build(B / "refiner.onnx", B / "refiner.plan", ("patches", (1, 3, 64, 64), (16, 3, 64, 64), (64, 3, 64, 64)))
    import json
    for hname, f in (json.loads((B / "bundle.json").read_text()).get("heads") or {}).items():      # split bundles: plug-in head engines (fp16)
        if (B / f.replace(".onnx", ".plan")).exists() and "--rebuild-heads" not in sys.argv: print("keep", f.replace(".onnx", ".plan")); continue
        build(B / f, B / f.replace(".onnx", ".plan"))
