"""Vehicle package: ONNX runtime == PyTorch runtime on the shipped reference frames (skipped when the bundle is not built)."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
B = Path(__file__).resolve().parents[1] / "deploy" / "bundle"
pytestmark = pytest.mark.skipif(not (B / "system_448.onnx").exists(), reason="run deploy/export_system.py first")


def test_onnx_matches_torch_runtime():
    sys.path.insert(0, str(B.parent)); from pfm_runtime import GatePerception
    R = np.load(B / "ref_frames.npz"); a, b = GatePerception(B, "torch", device="cpu"), GatePerception(B, "onnx", device="cpu")
    for i in range(3):
        x = torch.from_numpy(R["images"][i:i + 1].astype(np.float32) / 255)
        ga, gb = a(x)["gates"], b(x)["gates"]
        assert ga.shape == gb.shape
        if len(ga): assert float((ga - gb).abs().max()) < 1.0
