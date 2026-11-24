"""Frame sources for the vehicle. A background thread keeps only the NEWEST frame (perception must never work on a queue of old frames).
  csi[:sensor_id[:WxH@fps]]   IMX219 on the Jetson CSI port via GStreamer nvarguscamerasrc (default 1640x1232@30 = full field of view, 2x2 binned)
  usb[:index]                 V4L2 / USB camera through OpenCV
  file:<video or image glob>  replay (testing on the bench / dev PC), paced at --fps
  url:<stream url>            network camera, e.g. a phone IP-camera app: url:http://PHONE_IP:8080/video (MJPEG) or url:rtsp://PHONE_IP:8554/live
                              (OpenCV FFmpeg backend; if this OpenCV has no FFmpeg, a GStreamer pipeline is tried: souphttpsrc / rtspsrc)
read() -> (frame_id, t_capture, RGB uint8 HxWx3 (BGR with rgb=False)) or None if no new frame within timeout."""
import glob, threading, time
import numpy as np


def gst_csi(sensor_id=0, w=1640, h=1232, fps=30, flip=0, bgrx=False):
    """bgrx=True: hand the 4-channel BGRx buffer to OpenCV (supported since OpenCV 4.x; checked with 4.6 + GStreamer 1.24) and skip the CPU
    videoconvert (~0.8 ms/frame on x86, a few ms on an Orin core); the runtime drops the 4th channel on the GPU."""
    tail = "appsink drop=true max-buffers=1 sync=false" if bgrx else "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
            f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! {tail}")


class GstPipe:
    """VideoCapture stand-in for OpenCV builds without GStreamer (pip opencv in the venv): run gst-launch with the same nvarguscamerasrc pipeline
    and read raw BGRx frames from its stdout (fdsink). read() -> (ok, HxWx4 uint8 BGRx, or BGR when bgr=True)."""
    def __init__(self, sensor_id, w, h, fps, flip=0, bgr=False, nv12=False):
        import shlex
        self.w, self.h, self.bgr, self.nv12 = w, h, bgr, nv12; self.n = w * h * 3 // 2 if nv12 else w * h * 4; self.restarts = 0
        import os
        self.cmd = shlex.split(f"gst-launch-1.0 -q nvarguscamerasrc sensor-id={sensor_id} {os.environ.get('PFM_ARGUS', '')} ! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
                               f"nvvidconv flip-method={flip} ! video/x-raw,format={'NV12' if nv12 else 'BGRx'} ! fdsink fd=1 sync=false")
        self._spawn()
    def _spawn(self):
        import subprocess
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.n * 2)
    def isOpened(self): return self.p.poll() is None
    def read(self):
        buf = self.p.stdout.read(self.n)
        if buf is None or len(buf) < self.n:          # gst-launch died (seen: CSI corr_err after a few s): restart it, the caller just misses frames
            try: self.p.kill(); self.p.wait(timeout=2)
            except Exception: pass
            self.restarts += 1; print(f"camera: gst-launch stopped, restart {self.restarts}", flush=True); time.sleep(0.5); self._spawn()
            return False, None
        if self.nv12: return True, np.frombuffer(buf, np.uint8).reshape(self.h * 3 // 2, self.w)      # Y plane then interleaved UV (GPU converts)
        img = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)
        return True, (img[:, :, :3].copy() if self.bgr else img)
    def get(self, prop): return 0
    def set(self, prop, v): return False
    def release(self): self.p.kill()


def cv2_has_gstreamer(cv2):
    return any("GStreamer" in l and "YES" in l for l in cv2.getBuildInformation().splitlines())


class Undistort:
    """camera.json from calibrate_camera.py -> remap every frame to an undistorted pinhole view of `hfov` deg horizontal (out_w x out_h).
    The model was trained on an undistorted ~128 deg pinhole camera; a 120-160 deg fisheye cannot be used raw."""
    def __init__(self, calib_path, hfov=120.0, out_w=None, out_h=None):
        import cv2, json, math
        c = json.load(open(calib_path)); W, H = c["size"]; K = np.array(c["K"]); D = np.array(c["D"]); ow, oh = out_w or W, out_h or H
        f = (ow / 2) / math.tan(math.radians(hfov) / 2); P = np.array([[f, 0, ow / 2], [0, f, oh / 2], [0, 0, 1]])
        if c["model"] == "fisheye": self.m1, self.m2 = cv2.fisheye.initUndistortRectifyMap(K, D.reshape(4, 1), np.eye(3), P, (ow, oh), cv2.CV_16SC2)
        else: self.m1, self.m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), P, (ow, oh), cv2.CV_16SC2)
        self.cv2 = cv2; self.hfov = hfov
        self.calib = c; self.P = P; self.K, self.D, self.out = K, D, (ow, oh); self.size = (W, H)
    def __call__(self, rgb): return self.cv2.remap(rgb, self.m1, self.m2, self.cv2.INTER_LINEAR, borderMode=self.cv2.BORDER_CONSTANT)

    def gpu(self, dev="cuda"):
        """Same remap as __call__ on the GPU (grid_sample on the uploaded frame): on the Orin cv2.remap of a 1640x1232 frame costs ~10+ ms of
        one CPU core, the GPU does it in well under 1 ms. Returns f((B,3,H,W) float) -> (B,3,oh,ow)."""
        import torch, torch.nn.functional as F
        cv2 = self.cv2; W, H = self.size; ow, oh = self.out
        if self.calib["model"] == "fisheye": mx, my = cv2.fisheye.initUndistortRectifyMap(self.K, self.D.reshape(4, 1), np.eye(3), self.P, (ow, oh), cv2.CV_32FC1)
        else: mx, my = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.P, (ow, oh), cv2.CV_32FC1)
        g = torch.from_numpy(np.stack(((mx + 0.5) / W * 2 - 1, (my + 0.5) / H * 2 - 1), -1)).float()[None].to(dev)     # cv2 map = source px centre; grid_sample align_corners=False
        f = lambda x: F.grid_sample(x, g.expand(x.shape[0], -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=False)
        f.grid = g      # Update 166: GatePerception fuses NV12 -> RGB with this remap (pfm_runtime._prep, nv12_grid)
        return f


class Camera:
    def __init__(self, spec, fps=30, rgb=True):
        import cv2
        self.cv2 = cv2; self.rgb = rgb; self.nv12 = False; self.lock = threading.Lock(); self.frame = None; self.fid = 0; self.t = 0.0; self.stop_ = False; self.files = None; self.fps = fps
        kind, _, rest = spec.partition(":")
        if kind == "csi":
            parts = rest.split(":") if rest else []; sid = int(parts[0]) if parts and parts[0] else 0; w, h, f = 1640, 1232, 30
            if len(parts) > 1: wh, _, f = parts[1].partition("@"); w, h = map(int, wh.split("x")); f = int(f or 30)
            if not cv2_has_gstreamer(cv2):                                                              # pip OpenCV (venv): gst-launch pipe instead
                print("camera: this OpenCV has no GStreamer, reading nvarguscamerasrc through a gst-launch pipe", flush=True)
                self.cap = GstPipe(sid, w, h, f, bgr=rgb, nv12=not rgb)          # GPU path: NV12 (1.5 B/px instead of 4), converted on the GPU
                self.nv12 = not rgb
            else: self.cap = cv2.VideoCapture(gst_csi(sid, w, h, f, bgrx=not rgb), cv2.CAP_GSTREAMER)       # GPU pre-processing: BGRx straight through
            if not rgb and not self.cap.isOpened():                                                     # this OpenCV build refuses BGRx: CPU videoconvert path
                print("camera: BGRx appsink not supported, using BGR (videoconvert)", flush=True); self.cap = cv2.VideoCapture(gst_csi(sid, w, h, f), cv2.CAP_GSTREAMER)
        elif kind == "usb": self.cap = cv2.VideoCapture(int(rest or 0))
        elif kind == "url":
            self.cap = cv2.VideoCapture(rest)
            if not self.cap.isOpened():          # e.g. Jetson OpenCV built with GStreamer only
                src = f"rtspsrc location={rest} latency=0 ! decodebin" if rest.startswith("rtsp") else f"souphttpsrc location={rest} is-live=true ! multipartdemux ! jpegdec"
                self.cap = cv2.VideoCapture(f"{src} ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false", cv2.CAP_GSTREAMER)
        elif kind == "file":
            fs = sorted(glob.glob(rest))
            if len(fs) > 1 or (fs and fs[0].lower().endswith((".jpg", ".png"))): self.files, self.cap = fs, None
            else: self.cap = cv2.VideoCapture(rest)
        else: raise ValueError(spec)
        if self.cap is not None and not self.cap.isOpened(): raise RuntimeError(f"camera {spec} did not open")
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def _run(self):
        k = 0
        while not self.stop_:
            if self.files is not None:
                img = self.cv2.imread(self.files[k % len(self.files)]); k += 1; time.sleep(1.0 / self.fps)
            else:
                ok, img = self.cap.read()
                if not ok:
                    if self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT) > 0: self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)      # video file: loop (bench replay)
                    time.sleep(0.01); continue
                if self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT) > 0: time.sleep(1.0 / self.fps)      # video file: real-time pacing
            out = img if self.nv12 else (img[:, :, ::-1].copy() if self.rgb else img)          # rgb=False: hand over the BGR frame as is (run.py swaps channels on the GPU)
            with self.lock: self.frame, self.t, self.fid = out, time.time(), self.fid + 1

    def read(self, last_id=0, timeout=1.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.frame is not None and self.fid != last_id: return self.fid, self.t, self.frame
            time.sleep(0.001)
        return None

    def close(self):
        self.stop_ = True
        if self.cap is not None: self.cap.release()
