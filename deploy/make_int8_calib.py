"""INT8 calibration cache for the system graph (run on the dev PC; ship bundle/int8_calib.cache, then `build_engines.py --int8` on the Orin).
Calibration images = the deployment mix: sim gate frames, real gate frames, driving (BDD day, ACDC night/fog/rain/snow), off-road (RUGD), indoor/outdoor (ADE),
aerial (UAVid) - N images, resized like the runtime (bilinear to 448). Also builds a local INT8 engine (bundle_int8/) for the accuracy check (check_int8.py).
Run: env -u PYTHONPATH .venv/bin/python deploy/make_int8_calib.py [N=512]"""
import sys, glob, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
import build_engines as BE
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def calib_files(n, seed=0):
    G = lambda pat: sorted(glob.glob(str(ROOT / pat), recursive=True))
    groups = {"sim": G("data/train_sim/images/**/*.png"), "real_gate": G("data/real_mrg/golden/images/*.jpg") + G("data/real_tii/golden/images/*.jpg"),
              "bdd": G("data/general/train/images/bdd/*.jpg"), "acdc": G("data/general/raw_acdc/datasets/*/trainB/*.png"), "rugd": G("data/general/train/images/rugd/*"),
              "ade": G("data/general/train/images/ade/*"), "uav": G("data/general/train/images/uav/*")}
    share = dict(sim=0.25, real_gate=0.1, bdd=0.2, acdc=0.2, rugd=0.08, ade=0.1, uav=0.07); rng = random.Random(seed); out = []
    for k, f in share.items():
        if groups[k]: out += rng.sample(groups[k], min(len(groups[k]), round(n * f)))
    rng.shuffle(out); return out


import os
import tensorrt as trt
_BASE = dict(entropy=trt.IInt8EntropyCalibrator2, minmax=trt.IInt8MinMaxCalibrator)[os.environ.get("CALIB", "entropy")]


class ImageCalibrator(_BASE):
    def __init__(self, files, cache):
        super().__init__(); self.cache = Path(cache); self.files = files; self.i = 0; self.buf = torch.zeros(1, 3, 448, 448, device="cuda")
    def get_batch_size(self): return 1
    def read_calibration_cache(self): return None                      # always recalibrate here
    def write_calibration_cache(self, c): self.cache.write_bytes(bytes(c))
    def get_batch(self, names):
        if self.i >= len(self.files): return None
        x = torch.from_numpy(np.asarray(Image.open(self.files[self.i]).convert("RGB")).copy()).cuda().permute(2, 0, 1)[None].float() / 255
        self.buf.copy_(F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)); torch.cuda.synchronize(); self.i += 1
        if self.i % 100 == 0: print("calib", self.i, flush=True)
        return [int(self.buf.data_ptr())]


if __name__ == "__main__":
    files = calib_files(N); print("calibration images", len(files), flush=True)
    B = HERE / "bundle"; tag = os.environ.get("CALIB", "entropy"); O = HERE / f"bundle_int8_{tag}_{os.environ.get('TAG', 'all')}"; O.mkdir(exist_ok=True)
    for f in ("bundle.json", "ref_frames.npz", "refiner.plan"):
        if (B / f).exists() and not (O / f).exists(): (O / f).symlink_to(B / f)
    BE.build(B / "system_448.onnx", O / "system_448.plan", calib=ImageCalibrator(files, B / f"int8_calib_{tag}.cache"))
    print("cache", (B / f"int8_calib_{tag}.cache").stat().st_size, "bytes", flush=True)
