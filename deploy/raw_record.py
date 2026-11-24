"""Record the IMX219 raw stream at full rate (default sensor mode 7 = 1640x616 @ 84 fps, full width) to disk, then optionally convert to images
/ video with the same GPU ISP as the vehicle runtime (Update 167).
  record : python raw_record.py rec <out_dir> [--mode 7] [--secs 10] [--exp 3000] [--gain 32]      (raw uint16 frames + timestamps; ~170 MB/s)
  convert: python raw_record.py conv <out_dir> [--every 1] [--mp4] [--calib camera_imx219_cam1_1640x616.json --hfov 110]
Recording writes frames.raw (H*W uint16 per frame), meta.json (mode, size, fps) and stamps.txt (frame id, capture time, sensor sequence).
Stop the Argus daemon first (sudo systemctl stop nvargus-daemon): Argus and raw V4L2 cannot share the sensor."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["rec", "conv"]); ap.add_argument("out")
ap.add_argument("--mode", type=int, default=7); ap.add_argument("--secs", type=float, default=10); ap.add_argument("--exp", type=float, default=3000); ap.add_argument("--gain", type=float, default=32)
ap.add_argument("--every", type=int, default=1); ap.add_argument("--mp4", action="store_true"); ap.add_argument("--calib", default=""); ap.add_argument("--hfov", type=float, default=110)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

if a.cmd == "rec":
    from v4l2raw import RawCamera
    cam = RawCamera(mode=a.mode, exposure_us=a.exp, gain=a.gain); f = open(out / "frames.raw", "wb"); st = open(out / "stamps.txt", "w")
    last, n, t0 = 0, 0, time.time()
    while time.time() - t0 < a.secs:
        g = cam.read(last, 1.0)
        if g is None: print("no frame for 1 s", flush=True); continue
        last, tc, fr = g; f.write(fr.tobytes()); st.write(f"{last} {tc:.6f} {cam.seq}\n"); n += 1
    dt = time.time() - t0; f.close(); st.close()
    json.dump(dict(mode=a.mode, w=cam.w, h=cam.h, fps=cam.fps, frames=n, exposure_us=a.exp, gain=a.gain, sensor_drops=cam.dropped), open(out / "meta.json", "w"), indent=1)
    print(f"recorded {n} frames in {dt:.1f} s = {n / dt:.1f} fps, sensor drops {cam.dropped}, {os.path.getsize(out / 'frames.raw') / 2**20:.0f} MB -> {out}", flush=True)
    cam.release(); os._exit(0)

# convert: same ISP as pfm_runtime._prep_raw (black level, bilinear demosaic, grey-world WB, digital gain, gamma, optional undistort)
import torch, cv2
import pfm_runtime as R
m = json.load(open(out / "meta.json")); W, H = m["w"], m["h"]; n = m["frames"]
gp = R.GatePerception.__new__(R.GatePerception); gp.dev = "cuda"; gp.graphs = False; gp.img_bufs = [None, None]; gp._g = {}
if a.calib:
    from camera import Undistort
    gp.raw_grid = Undistort(a.calib, a.hfov).gpu("cuda").grid
raw = np.memmap(out / "frames.raw", np.uint16, "r", shape=(n, H, W)); (out / "jpg").mkdir(exist_ok=True); vw = None
for _ in range(40): gp._prep_raw(np.ascontiguousarray(raw[0]), 0)          # settle the white balance / digital gain EMAs before the first image
for i in range(0, n, a.every):
    img = gp._prep_raw(np.ascontiguousarray(raw[i]), 0)
    bgr = (img[0].permute(1, 2, 0).flip(-1).clamp(0, 1) * 255).byte().cpu().numpy()
    cv2.imwrite(str(out / "jpg" / f"{i:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if a.mp4:
        if vw is None: vw = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), m["fps"] / a.every, (bgr.shape[1], bgr.shape[0]))
        vw.write(bgr)
if vw is not None: vw.release()
print(f"converted {len(range(0, n, a.every))} frames -> {out / 'jpg'}" + (f" and {out / 'video.mp4'}" if a.mp4 else ""), flush=True)
os._exit(0)
