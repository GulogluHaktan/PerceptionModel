"""Raw Bayer capture from the IMX219 through V4L2 (Update 167): bypasses Argus / the ISP, which only accepts the stock sensor modes (the encrypted
imx219.nito knobsets). Used for the custom wide modes of our nv_imx219 build (~/imx219_backup on the Orin):
    sensor_mode 5: 1640x864 @ 60 fps   sensor_mode 7: 1640x616 @ 84 fps   (full sensor width, ~122 deg horizontal; x2 binning)
Frames: uint16 (H, W), RGGB, value = 10-bit << 6 (black level ~4096). Zero-copy mmap buffers; each frame carries the driver's capture timestamp
(CLOCK_MONOTONIC) and sequence number. Exposure / gain / frame rate are sensor controls (no ISP auto exposure): RawCamera.set_exposure(us, gain).

    cam = RawCamera(mode=7)                # 1640x616 @ 84
    fid, t_cap, raw = cam.read(last_id)    # same interface as camera.Camera.read (raw = numpy uint16 view, valid until the next read)
"""
import fcntl, mmap, os, struct, subprocess, threading, time
import numpy as np

MODES = {5: (1640, 864, 60), 6: (1640, 720, 72), 7: (1640, 616, 84), 4: (1280, 720, 60), 3: (1640, 1232, 30)}
CID = dict(sensor_mode=0x009A2008, gain=0x009A2009, exposure=0x009A200A, frame_rate=0x009A200B, bypass_mode=0x009A2064)
V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP = 1, 1


def _IOC(d, nr, size): return (d << 30) | (size << 16) | (ord("V") << 8) | nr


BUF_FMT = "IIIIIi qq IIBBBB4s II Q II i 4x"              # struct v4l2_buffer, 64-bit (88 bytes)
BUF_SIZE = struct.calcsize(BUF_FMT); assert BUF_SIZE == 88, BUF_SIZE
VIDIOC_REQBUFS, VIDIOC_QUERYBUF = _IOC(3, 8, 20), _IOC(3, 9, BUF_SIZE)
VIDIOC_QBUF, VIDIOC_DQBUF = _IOC(3, 15, BUF_SIZE), _IOC(3, 17, BUF_SIZE)
VIDIOC_STREAMON, VIDIOC_STREAMOFF = _IOC(1, 18, 4), _IOC(1, 19, 4)
VIDIOC_S_EXT_CTRLS = _IOC(3, 72, 32)


class RawCamera:
    def __init__(self, dev="/dev/video0", mode=7, fps=None, nbuf=4, exposure_us=None, gain=None, dump=None):
        self.w, self.h, f = MODES[mode]; self.fps = fps or f; self.dev = dev
        # format + mode with v4l2-ctl (the frame-rate control must be set AFTER the mode switch, in its own call: its range follows the mode)
        subprocess.run(["v4l2-ctl", "-d", dev, f"--set-fmt-video=width={self.w},height={self.h},pixelformat=RG10", "--set-ctrl", "bypass_mode=0",
                        "--set-ctrl", f"sensor_mode={mode}"], check=True, capture_output=True)
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"frame_rate={int(self.fps * 1e6)}"], check=True, capture_output=True)
        self.fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
        if exposure_us or gain: self.set_exposure(exposure_us, gain)
        rb = bytearray(struct.pack("IIIIB3x", nbuf, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, 0, 0)); fcntl.ioctl(self.fd, VIDIOC_REQBUFS, rb)
        n = struct.unpack("I", rb[:4])[0]; self.bufs = []
        for i in range(n):
            b = bytearray(struct.pack(BUF_FMT, i, V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b"\0" * 4, 0, V4L2_MEMORY_MMAP, 0, 0, 0, 0))
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b); f_ = struct.unpack(BUF_FMT, b); offset, length = f_[17], f_[18]
            self.bufs.append(mmap.mmap(self.fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset & 0xFFFFFFFF))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        self.stride = None
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, struct.pack("i", V4L2_BUF_TYPE_VIDEO_CAPTURE))
        self.lock = threading.Lock(); self.cond = threading.Condition(self.lock); self.frame = None; self.fid = 0; self.t = 0.0; self.seq = 0
        self.stop_ = False; self.dropped = 0; self.held = None
        self.dump = None; self.ndump = 0; self.exp_gain = (exposure_us or 0, gain or 0)
        if dump:                 # write EVERY frame in the capture thread (independent of how fast the consumer runs)
            from pathlib import Path
            import json
            Path(dump).mkdir(parents=True, exist_ok=True); self.dump_dir = Path(dump)
            self.dump = (open(self.dump_dir / "frames.raw", "wb", buffering=8 << 20), open(self.dump_dir / "stamps.txt", "w"))
            json.dump(dict(mode=mode, w=self.w, h=self.h, fps=self.fps, frames=0), open(self.dump_dir / "meta.json", "w"))
        self.clk_off = None      # camera clock -> time.time(): min over frames of (arrival - stamp) = the readout-to-arrival floor (~1-2 ms)
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def set_exposure(self, exposure_us=None, gain=None):
        """Sensor exposure (us) and analog gain (16 = 1x ... 170 = 10.6x) through VIDIOC_S_EXT_CTRLS (int64 controls)."""
        self.exp_gain = (exposure_us or self.exp_gain[0], gain or self.exp_gain[1]) if hasattr(self, "exp_gain") else (exposure_us, gain)
        items = [(CID["exposure"], int(exposure_us))] if exposure_us else []
        if gain: items.append((CID["gain"], int(gain)))
        if not items: return
        import ctypes
        arr = b"".join(struct.pack("<IIIq", cid, 0, 0, v) for cid, v in items)                 # packed struct v4l2_ext_control (20 bytes each)
        buf = ctypes.create_string_buffer(arr, len(arr))
        ctrls = bytearray(struct.pack("IIIiI4xQ", 0, len(items), 0, 0, 0, ctypes.addressof(buf)))   # struct v4l2_ext_controls (32 bytes)
        fcntl.ioctl(self.fd, VIDIOC_S_EXT_CTRLS, ctrls)

    def _run(self):
        import select
        while not self.stop_:
            r, _, _ = select.select([self.fd], [], [], 0.5)
            if not r: continue
            b = bytearray(struct.pack(BUF_FMT, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b"\0" * 4, 0, V4L2_MEMORY_MMAP, 0, 0, 0, 0))
            try: fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
            except BlockingIOError: continue
            f = struct.unpack(BUF_FMT, b); idx, used, sec, usec, seq = f[0], f[2], f[6], f[7], f[15]      # f[14] = timecode userbits, f[15] = sequence
            if self.stride is None: self.stride = used // self.h // 2
            img = np.frombuffer(self.bufs[idx], np.uint16, count=self.h * self.stride).reshape(self.h, self.stride)[:, : self.w].copy()
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)                 # give the buffer back at once (we keep our own copy: 2 MB, ~0.3 ms)
            if self.dump is not None:
                self.dump[0].write(img.tobytes()); self.dump[1].write(f"{self.ndump} {sec + usec * 1e-6 + (self.clk_off or 0):.6f} {seq} {self.exp_gain[0]:.0f} {self.exp_gain[1]:.0f}\n"); self.ndump += 1
            ts = sec + usec * 1e-6; off = time.time() - ts
            self.clk_off = off if self.clk_off is None else min(self.clk_off, off)
            with self.cond:
                if self.seq and seq != self.seq + 1: self.dropped += seq - self.seq - 1
                self.frame, self.t, self.seq = img, ts + self.clk_off, seq; self.fid += 1; self.cond.notify_all()      # t in time.time() seconds

    def read(self, last_id=0, timeout=1.0):
        """Newest frame after last_id: (frame_id, t_capture as time.time() seconds (driver stamp + estimated clock offset), uint16 (H, W)) or None."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.fid > last_id, timeout): return None
            return self.fid, self.t, self.frame

    def close(self): self.release()

    def release(self):
        self.stop_ = True; self.th.join(timeout=1)
        if self.dump is not None:
            import json
            self.dump[0].close(); self.dump[1].close(); md = json.load(open(self.dump_dir / "meta.json")); md.update(frames=self.ndump, dropped_sensor=self.dropped)
            json.dump(md, open(self.dump_dir / "meta.json", "w"), indent=1); print(f"raw dump: {self.ndump} frames -> {self.dump_dir}", flush=True); self.dump = None
        try: fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, struct.pack("i", V4L2_BUF_TYPE_VIDEO_CAPTURE))
        except OSError: pass
        for m in self.bufs: m.close()
        os.close(self.fd)


if __name__ == "__main__":       # quick check: python v4l2raw.py [mode] [seconds]
    import sys
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 7; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    cam = RawCamera(mode=m); last = 0; n = 0; t0 = time.time(); lat = []
    while time.time() - t0 < secs:
        g = cam.read(last, 1.0)
        if g: last, tc, fr = g; n += 1; lat.append(time.time() - tc)
    print(f"mode {m} {cam.w}x{cam.h}: {n / (time.time() - t0):.1f} fps read, sensor drops {cam.dropped}, frame mean {fr.mean():.0f}, "
          f"capture->read median {1e3 * np.median(lat):.1f} ms, stride {cam.stride}", flush=True)
    cam.release(); os._exit(0)
