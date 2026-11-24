"""Build one mixed-precision variant from the existing calibration cache: python deploy/int8_variant.py <name> <regex> -> deploy/bundle_int8_<name>/"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; name, rx = sys.argv[1], sys.argv[2]; os.environ["INT8_LAYERS"] = rx
sys.path.insert(0, str(HERE)); import build_engines as BE
B = HERE / "bundle"; O = HERE / f"bundle_int8_{name}"; O.mkdir(exist_ok=True)
for f in ("bundle.json", "ref_frames.npz", "refiner.plan"):
    if not (O / f).exists(): (O / f).symlink_to(B / f)
BE.build(B / "system_448.onnx", O / "system_448.plan", calib=BE.CacheCalibrator(B / "int8_calib.cache"))
