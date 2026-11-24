"""Lens calibration for the vehicle camera (wide IMX219 120-160 deg lenses are strongly distorted; the model was trained on an
undistorted pinhole camera of ~128 deg). Shoot 20-40 frames of a printed checkerboard (inner corners COLSxROWS) from different angles,
covering the image corners, then:
  python deploy/calibrate_camera.py "calib_frames/*.jpg" --cols 9 --rows 6 --square 0.025 [--model fisheye|pinhole] --out camera.json
Writes camera.json (K, D, model, size, rms). run.py --calib camera.json --hfov 120 then undistorts every frame to a pinhole view."""
import argparse, glob, json
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("frames"); ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square", type=float, default=0.025); ap.add_argument("--model", default="fisheye"); ap.add_argument("--out", default="camera.json"); a = ap.parse_args()
obj = np.zeros((a.rows * a.cols, 1, 3), np.float32); obj[:, 0, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
O, I, size = [], [], None
for f in sorted(glob.glob(a.frames)):
    g = cv2.imread(f, cv2.IMREAD_GRAYSCALE); size = g.shape[::-1]
    ok, c = cv2.findChessboardCorners(g, (a.cols, a.rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: continue
    c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)); O.append(obj.reshape(1, -1, 3).astype(np.float64)); I.append(c.reshape(1, -1, 2).astype(np.float64))      # (1,N,C) float64: accepted by OpenCV 4 and 5
print(f"checkerboard found in {len(I)} frames"); assert len(I) >= 8, "need >= 8 good frames"
if a.model == "fisheye":
    fl = lambda n: getattr(cv2.fisheye, n, getattr(cv2, n, 0))
    flags = fl("CALIB_RECOMPUTE_EXTRINSIC") + fl("CALIB_FIX_SKEW") + fl("CALIB_USE_INTRINSIC_GUESS")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    keep = list(range(len(I)))
    while True:          # OpenCV's fisheye init fails on some near-degenerate views: drop the frame it chokes on and retry
        K = np.array([[size[0] / 3.0, 0, size[0] / 2], [0, size[0] / 3.0, size[1] / 2], [0, 0, 1]]); D = np.zeros((4, 1))
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate([O[i] for i in keep], [I[i] for i in keep], size, K, D, flags=flags, criteria=crit); break
        except cv2.error:
            bad = None
            for i in keep:           # find a frame that fails on its own
                try: cv2.fisheye.calibrate([O[i]], [I[i]], size, K.copy(), np.zeros((4, 1)), flags=flags, criteria=crit)
                except cv2.error: bad = i; break
            if bad is None: bad = keep[-1]
            keep.remove(bad); assert len(keep) >= 8, "too few usable frames"
    print(f"used {len(keep)} of {len(I)} frames")
else:
    rms, K, D, _, _ = cv2.calibrateCamera([o[0].astype(np.float32) for o in O], [i[0].astype(np.float32) for i in I], size, None, None)
hfov = float(np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0])))) if a.model != "fisheye" else None
json.dump(dict(model=a.model, size=list(size), K=K.tolist(), D=np.asarray(D).ravel().tolist(), rms=float(rms), frames=len(I), pinhole_hfov_deg=hfov), open(a.out, "w"), indent=1)
print(f"rms reprojection error {rms:.3f} px -> {a.out}  (good: < 0.5 px)")
