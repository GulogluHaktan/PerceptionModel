"""Does the TensorRT fp16 engine keep the accuracy of the trained student? (train -> deploy closure, run on the Orin)
Usage: python scripts/orin_trt_accuracy.py <ckpt.pt> <size> <golden_dir> [hybrid_name=fastvit_sa24]
Loads the checkpoint, reparameterises it (inference form), measures seg mIoU / depth AbsRel in PyTorch fp32 and from a TensorRT fp16 engine built from
the same weights, on the same golden-style frames, and reports the pixel agreement of the two segmentations. TensorRT Python comes from the system package."""
import json, subprocess, shutil, sys, time
from pathlib import Path
sys.path.insert(0, "src"); sys.path.append("/usr/lib/python3.12/dist-packages")
import numpy as np, tensorrt as trt, torch
from timm.utils import reparameterize_model
from pfm.data import GoldenSet, normalize
from pfm.evaluate import evaluate
from pfm.model import CLASSES, PerceptionModel

ckpt, size, gdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]; hname = sys.argv[4] if len(sys.argv) > 4 else "fastvit_sa24"; dev = "cuda"
m = PerceptionModel("hybrid", teacher_dims={"dino": 384}, hybrid_name=hname); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
m = reparameterize_model(m).to(dev).eval(); golden = GoldenSet(gdir, size)
ref = evaluate(m, golden, len(CLASSES), dev); print("pytorch fp32 (reparameterised):", ref, flush=True)

class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): o = s.m(x, with_teacher_heads=False); return o["depth"], o["seg"], o["corners"]
out = Path("runs/trt_acc") / Path(ckpt).stem; out.mkdir(parents=True, exist_ok=True); onnx = out / "student.onnx"; plan = out / "student.plan"
torch.onnx.export(Wrap(m).cpu(), torch.randn(1, 3, size, size), str(onnx), opset_version=17, input_names=["image"], output_names=["depth", "seg", "corners"], dynamo=False)
m.to(dev); t = time.time()
if not plan.exists():
    subprocess.run([shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx}", "--fp16", f"--saveEngine={plan}"], capture_output=True, check=True)
build_s = time.time() - t
rt = trt.Runtime(trt.Logger(trt.Logger.WARNING)); eng = rt.deserialize_cuda_engine(open(plan, "rb").read()); ctx = eng.create_execution_context()
bufs = {n: torch.zeros(tuple(eng.get_tensor_shape(n)), device=dev, dtype=torch.float32) for n in ("image", "depth", "seg", "corners")}
for n, b in bufs.items(): ctx.set_tensor_address(n, b.data_ptr())
stream = torch.cuda.Stream()

class TRTModel:
    def eval(self): return self
    def __call__(self, x, with_teacher_heads=False):
        outs = {"depth": [], "seg": [], "corners": []}
        for i in range(x.shape[0]):
            bufs["image"].copy_(x[i:i + 1]); ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
            for k in outs: outs[k].append(bufs[k].clone())
        return {k: torch.cat(v) for k, v in outs.items()}
trt_m = TRTModel(); trt_res = evaluate(trt_m, golden, len(CLASSES), dev); print("tensorrt fp16:", trt_res, flush=True)
same, tot = 0, 0
with torch.no_grad():
    for i in range(len(golden)):
        x, seg, _ = golden[i]; xn = normalize(x[None].to(dev)); a = m(xn, with_teacher_heads=False)["seg"].argmax(1); b = trt_m(xn)["seg"].argmax(1); same += int((a == b).sum()); tot += a.numel()
print(json.dumps({"ckpt": ckpt, "size": size, "frames": len(golden), "pytorch_fp32": ref, "tensorrt_fp16": trt_res, "seg_pixel_agreement": round(same / tot, 5), "engine_build_s": round(build_s, 1)}))
