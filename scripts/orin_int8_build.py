"""Build a calibrated INT8 TensorRT engine on the Orin, for a SPEED measurement.
Usage: python scripts/orin_int8_build.py <model.onnx> <out.plan> [size=384] [batches=32]
Calibration here uses RANDOM images, so the engine is only valid for timing (INT8 kernels are really used, unlike a
plain `trtexec --int8` without calibration data, which silently stays fp16). For a real INT8 model, calibrate with real
frames and check accuracy on the golden set. Then time it with: trtexec --loadEngine=<out.plan> ..."""
import sys
sys.path.append("/usr/lib/python3.12/dist-packages")
import tensorrt as trt
import torch

onnx_path, out_path = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 384
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 32
logger = trt.Logger(trt.Logger.WARNING)


class Calib(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        super().__init__()
        self.i, self.buf = 0, torch.empty(1, 3, size, size, device="cuda")
        self.cache = out_path + ".calib"

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= nb:
            return None
        self.buf.normal_(); self.i += 1
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)


b = trt.Builder(logger)
net = b.create_network(0)
p = trt.OnnxParser(net, logger)
if not p.parse_from_file(onnx_path):
    raise SystemExit("\n".join(str(p.get_error(i)) for i in range(p.num_errors)))
cfg = b.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_flag(trt.BuilderFlag.INT8)
cfg.int8_calibrator = Calib()
plan = b.build_serialized_network(net, cfg)
if plan is None:
    raise SystemExit("build failed")
open(out_path, "wb").write(plan)
print("built", out_path, round(plan.nbytes / 2**20, 1), "MB")
