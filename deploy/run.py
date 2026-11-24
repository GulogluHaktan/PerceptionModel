"""Vehicle perception loop: camera -> GatePerception -> UDP JSON (one datagram per frame) + optional recording.
Message: {"id", "t_cap", "t_out", "w", "h", "gates": [[score, x0,y0, x1,y1, x2,y2, x3,y3], ...], "peaks": [[type, score, u, v], ...], "ms": {...}}
(px of the camera frame; gate corners tl,tr,br,bl). The flight stack (EKF / planner) listens on --udp.
Headless start: waits for the GPU (Orin ACR bring-up was seen to take minutes after a reboot, docs/ORIN_RESULTS.md) before loading engines.
Run: python deploy/run.py --source csi --backend trt --udp 127.0.0.1:5600 [--record /data/rec --record-every 10] [--square]"""
import argparse, json, os, socket, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="csi"); ap.add_argument("--backend", default="trt"); ap.add_argument("--bundle", default=str(HERE / "bundle"))
ap.add_argument("--udp", default="127.0.0.1:5600"); ap.add_argument("--record", default=""); ap.add_argument("--record-every", type=int, default=10)
ap.add_argument("--square", action="store_true", help="centre-crop the frame to a square before the model (the racing loop fed square derotated crops)")
ap.add_argument("--calib", default="", help="camera.json from calibrate_camera.py: undistort every frame (needed for wide / fisheye lenses)")
ap.add_argument("--hfov", type=float, default=120.0, help="horizontal field of view of the undistorted view (with --calib)")
ap.add_argument("--raw-dump", default="", help="raw camera: also write EVERY raw frame to this folder (frames.raw + stamps.txt + meta.json; raw_record.py conv turns it into images)")
ap.add_argument("--scene", default="", help="indoor|outdoor: fixed scene (only that module is loaded); empty = auto router")
ap.add_argument("--cam-fps", type=float, default=0, help="raw camera: sensor frame rate below the mode default (e.g. 72) = longer frames, longer possible exposure")
ap.add_argument("--exp-max", type=float, default=8000, help="raw camera: longest exposure in us (motion blur at speed); then analog gain rises")
ap.add_argument("--fps", type=float, default=30); ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run forever)")
ap.add_argument("--gpu-wait", type=float, default=900, help="seconds to wait for CUDA at boot")
ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="rotate frames clockwise before the model (phone held upright sends sideways frames: try 90 or 270)")
ap.add_argument("--no-pipeline", action="store_true", help="run each frame start to finish (default with TensorRT + GPU pre-processing: GPU works on frame n+1 while the host post-processes frame n)")
ap.add_argument("--cpu-prep", action="store_true", help="old path: BGR->RGB, undistort and square crop on the CPU (default: on the GPU, ~10-20 ms less CPU per frame on the Orin)")
a = ap.parse_args()
if a.scene: os.environ["PFM_SCENE"] = a.scene

import torch
t0 = time.time()
while not torch.cuda.is_available() and a.backend != "onnx":
    if time.time() - t0 > a.gpu_wait: sys.exit("CUDA not available after %.0f s" % a.gpu_wait)
    print("waiting for the GPU ...", flush=True); time.sleep(10)
from pfm_runtime import GatePerception
from camera import Camera, Undistort
gp = GatePerception(a.bundle, backend=a.backend); und = Undistort(a.calib, a.hfov) if a.calib else None
gpu_prep = not a.cpu_prep and gp.dev != "cpu"
RAW = a.source.startswith("raw")                 # raw:<sensor_mode>  custom wide modes through V4L2 (deploy/v4l2raw.py), GPU ISP in pfm_runtime._prep_raw
if RAW:
    from v4l2raw import RawCamera
    cam = RawCamera(mode=int(a.source.partition(":")[2] or 7), fps=a.cam_fps or None, exposure_us=min(4000, a.exp_max), gain=32, dump=a.raw_dump or None); exp_us, gain = min(4000, a.exp_max), 32.0
    if und is not None: gp.raw_grid = und.gpu(gp.dev).grid; und = None
    gp.raw_rotate = a.rotate
else: cam = Camera(a.source, fps=a.fps, rgb=not gpu_prep)
if gpu_prep and not RAW:
    remap = und.gpu(gp.dev) if und is not None else None; und = None; gp.bgr = True
    def _pre(x):
        if remap is not None: x = remap(x)
        if a.rotate: x = torch.rot90(x, k=-a.rotate // 90, dims=(2, 3))     # clockwise
        gp.full_hw = tuple(x.shape[-2:])                      # (undistorted) frame size before the square crop
        if a.square: H, W = x.shape[-2:]; s = min(H, W); x = x[..., (H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
        return x
    gp.pre = _pre
    if remap is not None and not a.rotate and not a.square and getattr(cam, "nv12", False):      # NV12 camera: fused NV12 + undistort on the GPU
        gp.nv12_grid = remap.grid; gp.pre = None
pipe = gpu_prep and a.backend == "trt" and not a.no_pipeline
host, port = a.udp.rsplit(":", 1); sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); dst = (host, int(port))
rec = Path(a.record) if a.record else None
if rec: rec.mkdir(parents=True, exist_ok=True); log = open(rec / "detections.jsonl", "a")
last, n, tl, stats = 0, 0, time.time(), []; t_start, all_lat = time.time(), []; ms_acc = {}
print(f"running: source={a.source} backend={a.backend} pipeline={'on' if pipe else 'off'} -> udp {a.udp}", flush=True)
try:
    while True:
        got = cam.read(last, timeout=1.0)
        if got is None: print("WARNING: no camera frame for 1 s", flush=True); continue
        last, tcap, fr = got
        if RAW and last % 15 == 0 and getattr(gp, "raw_stats", None) is not None:      # auto exposure on the sensor: exposure up to --exp-max, then gain
            lin = float(gp.raw_stats); tgt = 0.18                  # linear grey level before the digital gain
            if lin < 0.7 * tgt or lin > 1.4 * tgt:
                want = exp_us * gain * min(4.0, max(0.25, tgt / max(lin, 1e-4)))
                exp_us = min(a.exp_max, max(50.0, want / 16.0)); gain = min(170.0, max(16.0, want / exp_us))
                cam.set_exposure(exp_us, gain)
        if und is not None: fr = und(fr)          # gates / peaks are then in px of the UNDISTORTED view (send --hfov to the EKF)
        if gpu_prep:            # channel swap / undistort / square crop happen inside gp (GPU); offsets map the square crop back to the (undistorted) frame
            if pipe:
                out = gp.step(fr, (last, tcap))
                if out is None: continue
                r, (last, tcap) = out                     # result of the previous frame, with its own id / capture time
            else: r = gp(fr)
            H, W = gp.full_hw; s = gp.last_img.shape[-1]
            y0, x0 = ((H - s) // 2, (W - s) // 2) if a.square else (0, 0)
        else:
            H, W = fr.shape[:2]; x0 = y0 = 0
            if a.square: s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2; fr = fr[y0:y0 + s, x0:x0 + s]
            r = gp(np.ascontiguousarray(fr))
        g = r["gates"].cpu().numpy(); p = r["peaks"].cpu().numpy()
        if x0 or y0: g[:, 1::2] += x0; g[:, 2::2] += y0; p[:, 2] += x0; p[:, 3] += y0     # back to full-frame px
        msg = dict(id=last, t_cap=tcap, t_out=time.time(), w=W, h=H, gates=np.round(g, 2).tolist(), peaks=np.round(p, 2).tolist(), ms={k: round(v, 2) for k, v in r["ms"].items()})
        for k_, v_ in r["ms"].items(): ms_acc.setdefault(k_, []).append(v_)
        sock.sendto(json.dumps(msg).encode(), dst); n += 1; stats.append(msg["t_out"] - tcap); all_lat.append(msg["t_out"] - tcap)
        if rec and n % a.record_every == 0:          # the image the model saw (after undistort / crop), RGB
            from PIL import Image
            img_rec = (gp.last_img[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy() if gpu_prep else fr
            Image.fromarray(img_rec).save(rec / f"{last:08d}.jpg", quality=90); log.write(json.dumps(msg) + "\n"); log.flush()
            seg = r["seg"].cpu().numpy(); Image.fromarray(seg).save(rec / f"{last:08d}_seg.png")                       # 448x448 class ids (bundle.json seg_classes)
            if r.get("seg_in") is not None: Image.fromarray(r["seg_in"].cpu().numpy()).save(rec / f"{last:08d}_segin.png")   # indoor class ids (bundle.json seg_in_classes)
            if r.get("seg_fine") is not None: Image.fromarray(r["seg_fine"].cpu().numpy()).save(rec / f"{last:08d}_segfine.png")   # outdoor 19-class ids (src/pfm/outdoor_classes.py)
            dep = r["depth"].float().cpu().numpy(); d8 = (255 * (dep - dep.min()) / max(1e-6, float(dep.max() - dep.min()))).astype(np.uint8)
            Image.fromarray(d8).save(rec / f"{last:08d}_depth.png")                                                     # relative inverse depth, 0..255 per frame
        if time.time() - tl > 5:
            lat = np.array(stats) * 1e3; print(f"{n} frames | {len(stats) / (time.time() - tl):.1f} FPS | capture->output median {np.median(lat):.1f} ms p95 {np.percentile(lat, 95):.1f} ms | engine {r['ms']['engine']:.1f} ms | gates {len(g)}", flush=True)
            tl, stats = time.time(), []
        if a.frames and n >= a.frames: break
finally:
    cam.close()
    if n: print("stage ms (median): " + ", ".join(f"{k_} {np.median(v_):.2f}" for k_, v_ in ms_acc.items()), flush=True)
    if n: L = np.array(all_lat) * 1e3; print(f"done: {n} frames in {time.time() - t_start:.1f} s ({n / (time.time() - t_start):.1f} FPS) | capture->output median {np.median(L):.1f} ms p95 {np.percentile(L, 95):.1f} ms", flush=True)
    sys.stdout.flush(); os._exit(0)          # a network stream reader can block in cap.read() forever: do not wait for it at exit
