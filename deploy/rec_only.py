"""Record-only mode (no perception, no torch, no GPU): every raw frame of the IMX219 wide mode to disk + sensor auto exposure on the CPU.
  python rec_only.py <out_dir> [--secs 600] [--mode 7] [--exp-max 8000]
Writes the same files as run.py --raw-dump (frames.raw, stamps.txt with exposure/gain per frame, meta.json), so raw_record.py conv and
scripts/car_data_round.sh work unchanged. Stop early with Ctrl+C / SIGINT (the files are closed cleanly)."""
import argparse, os, signal, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4l2raw import RawCamera
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--secs", type=float, default=600); ap.add_argument("--mode", type=int, default=7)
ap.add_argument("--exp-max", type=float, default=8000)
a = ap.parse_args()
stop = [False]; signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True)); signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
exp_us, gain = min(4000.0, a.exp_max), 32.0
PREVIEW = os.path.expanduser("~/.cache/pfm_preview.jpg"); os.makedirs(os.path.dirname(PREVIEW), exist_ok=True)
from cpu_isp import CpuISP
isp = CpuISP()


def preview(fr):
    """~3 fps CPU preview, written atomically for rec_view.py."""
    import cv2
    cv2.imwrite(PREVIEW + ".tmp.jpg", isp(fr), [cv2.IMWRITE_JPEG_QUALITY, 80]); os.replace(PREVIEW + ".tmp.jpg", PREVIEW)


cam = RawCamera(mode=a.mode, exposure_us=exp_us, gain=gain, dump=a.out)
last, n, t0, tl = 0, 0, time.time(), time.time()
print(f"recording {cam.w}x{cam.h} @ {cam.fps} fps -> {a.out} (no perception)", flush=True)
try:
    while not stop[0] and time.time() - t0 < a.secs:
        g = cam.read(last, 1.0)
        if g is None: print("WARNING: no camera frame for 1 s", flush=True); continue
        last, _, fr = g; n += 1
        if n % 15 == 0:     # same rule as run.py, grey level from the green Bayer plane (1/64 subsample), target 18 % linear grey
            lin = float(np.clip((fr[0::2, 1::2][::8, ::8].astype(np.float32) - 4096) / (65535 - 4096), 0, None).mean())
            if lin < 0.7 * 0.18 or lin > 1.4 * 0.18:
                want = exp_us * gain * min(4.0, max(0.25, 0.18 / max(lin, 1e-4)))
                exp_us = min(a.exp_max, max(50.0, want / 16.0)); gain = min(170.0, max(16.0, want / exp_us)); cam.set_exposure(exp_us, gain)
        if n % 28 == 1:
            try: preview(fr)
            except Exception as e: print("preview failed:", e, flush=True)
        if time.time() - tl > 5:
            el = time.time() - t0; print(f"{el:5.0f} s | saved {cam.ndump} frames ({cam.ndump / el:.1f} fps) | sensor drops {cam.dropped} | exp {exp_us:.0f} us gain {gain:.0f}", flush=True); tl = time.time()
finally:
    el = time.time() - t0; nd = cam.ndump; cam.release()
    print(f"done: {nd} frames in {el:.1f} s ({nd / max(el, 1e-6):.1f} fps) -> {a.out}", flush=True)
