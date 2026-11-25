"""Turn a raw recording into a viewable MP4 on the CPU (no torch / GPU):  python3 ~/rec/rec_to_mp4.py ~/car/araba1 [--every 3] [--full]
Every --every-th frame (default 3: 84 fps -> 28 fps = real time), half size (820x308) unless --full; writes <rec>/video.mp4."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_isp import CpuISP
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=3); ap.add_argument("--full", action="store_true")
a = ap.parse_args(); d = Path(a.rec).expanduser(); m = json.load(open(d / "meta.json")); W, H = m["w"], m["h"]; fsz = W * H * 2
n = (d / "frames.raw").stat().st_size // fsz; isp = CpuISP(half=not a.full); fps = m.get("fps", 84) / a.every
raw = np.memmap(d / "frames.raw", dtype=np.uint16, mode="r", shape=(n, H, W)); out = d / "video.mp4"; vw = None; t0 = time.time()
for k, i in enumerate(range(0, n, a.every)):
    img = isp(np.asarray(raw[i]))
    if vw is None: vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
    vw.write(img)
    if k % 200 == 0: print(f"{i}/{n} frames ({time.time() - t0:.0f} s)", flush=True)
if vw is not None: vw.release()
print(f"done: {out} ({len(range(0, n, a.every))} frames @ {fps:.0f} fps, {time.time() - t0:.0f} s)", flush=True)
