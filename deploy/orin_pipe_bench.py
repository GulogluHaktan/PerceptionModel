"""Orin: sync vs pipelined throughput + equality, frames preloaded (no camera / decode cost)."""
import sys, glob, time, numpy as np, torch, cv2
sys.path.insert(0, "."); import pfm_runtime as R
fs = sorted(glob.glob("/tmp/cekim_rot/*.jpg")); frames = [cv2.resize(cv2.imread(f)[:, :, ::-1].copy(), (1640, 1232)) for f in fs] * 3
gp = R.GatePerception("bundle", "trt")
for f in frames[:6]: gp(f)
torch.cuda.synchronize(); t = time.perf_counter(); A = [gp(f) for f in frames]; torch.cuda.synchronize(); ta = time.perf_counter() - t
for f in frames[:4]: gp.step(f)
gp.flush(); B = []; torch.cuda.synchronize(); t = time.perf_counter()
for i, f in enumerate(frames):
    r = gp.step(f, i)
    if r: B.append(r)
B.append(gp.flush()); torch.cuda.synchronize(); tb = time.perf_counter() - t
mis = sum(int(a["gates"].shape != b["gates"].shape or not torch.equal(a["seg"], b["seg"])) for a, (b, _) in zip(A, B))
print(f"frames {len(frames)} | sync {len(frames)/ta:.1f} FPS ({1e3*ta/len(frames):.1f} ms) | pipelined {len(frames)/tb:.1f} FPS ({1e3*tb/len(frames):.1f} ms) | mismatching frames {mis}")
