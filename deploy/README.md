# PerceptionModel on the vehicle (Jetson Orin NX + IMX219)

Gate perception exactly as flown in simulation (DroneRacing "min14" configuration: release v4 core + corner-peak head + near-gate
head + corner refiner + post-processing), packaged for the vehicle. The core is frozen. The condition modules / router are NOT in the
default package: none passed the real-data acceptance (handoff Update 149).

## What is in the package
| File | Purpose |
|---|---|
| `bundle/system_448.onnx` | core (FastViT-SA24 hybrid, re-parameterised) + corner-peak head + near-gate head, **one** shared feature pass (the research code ran the core 3x per frame). Outputs: depth, seg (11 classes), gate candidates (top-32 per gate head + "same gate" matrix; the tiny greedy selection runs on the host, `gdet_post.py`), corner heat/offsets, pooled feature |
| (inside system_448.onnx) | plug-in seg modules: mean of SegRefine v1 (Update 153) and SegHead2 v2 (Update 157) logits. TensorRT runtime, core -> package: BDD 0.736 -> 0.767, IDD-AW 0.611 -> 0.637, RUGD 0.505 -> 0.522, real dark test 0.607 -> 0.619; engine +15 % vs v1 only |
| `bundle/refiner.onnx` | corner refiner, dynamic batch of 64x64 patches |
| `bundle/bundle.json` | input format, class list, post-processing constants (min14), ONNX check numbers |
| `bundle/ref_frames.npz` | 24 test frames + PyTorch reference outputs (the device compares its engines to these) |
| `gdet_post.py` | gate-head selection (duplicate removal + size-level merge) in numpy, identical to the training decoder; was ~5,000 tiny ONNX ops (84 % of the engine's GPU time) |
| `pfm_runtime.py` | `GatePerception(bundle, backend="trt"/"onnx"/"torch")(frame) -> gates, peaks, seg, depth, stage times` |
| `camera.py` | IMX219 over CSI (GStreamer `nvarguscamerasrc`), USB, or file replay; keeps only the newest frame. `Undistort.gpu()` = the same remap as grid_sample on the GPU |
| `calibrate_camera.py` | checkerboard lens calibration (fisheye or pinhole) -> camera.json; checked on synthetic fisheye frames: focal 0.3 % / centre 0.5 px, rms 0.32 px |
| `run.py` | vehicle loop: camera -> perception -> one UDP JSON datagram per frame (+ optional recording); waits for the GPU at boot |
| `build_engines.py` | builds the TensorRT engines **on the device** (fp16) |
| `check_device.py` | engine-vs-reference accuracy + end-to-end latency on the device |
| `pfm-perception.service` | systemd user unit (autostart, restart on failure) |

## On the Orin (TensorRT 10.16, torch 2.15 cu132 already installed, see docs/ORIN_RESULTS.md)
```bash
tar xzf pfm_vehicle_YYYYMMDD.tar.gz -C ~ && cd ~/pfm_vehicle
sha256sum -c bundle/SHA256SUMS                      # from the package root
python3 build_engines.py bundle                     # ~2-5 min, writes bundle/*.plan
python3 check_device.py bundle trt 1640 1232        # accuracy vs reference + latency; expect gates_same_set 24/24, gates_max_px < 1
python3 run.py --source "file:bundle_test/*.jpg" --backend trt --frames 300     # bench without a camera (image folder, or a video file: loops)
python3 run.py --source url:http://PHONE_IP:8080/video --backend trt --frames 600 --record ~/pfm_rec   # phone as a stand-in camera (IP-camera app, same Wi-Fi)
python3 calibrate_camera.py "calib/*.jpg" --cols 9 --rows 6 --square 0.025 --model fisheye   # once per lens: 20-40 checkerboard shots
python3 run.py --source csi --backend trt --calib camera.json --hfov 120 --udp 127.0.0.1:5600 --record ~/pfm_rec   # live, undistorted 120 deg view
```
Python with `tensorrt` + `torch` + `opencv` (GStreamer build, the JetPack one) + `numpy` + `pillow`. No timm / project install needed.
If `python3` cannot import tensorrt, add `/usr/lib/python3.12/dist-packages` (done in the code) or use the project venv.

## Output message (UDP JSON, one per frame)
`{"id", "t_cap", "t_out", "w", "h", "gates": [[score, x0,y0, x1,y1, x2,y2, x3,y3]], "peaks": [[type, score, u, v]], "ms": {...}}`
Corners in px of the camera frame, tl,tr,br,bl. `peaks` = single corners for the EKF partial-gate update (type 0..3 = tl,tr,br,bl).
seg / depth stay on the device (`GatePerception` returns them; 448x448 class map and relative inverse depth).

## Verified on the dev PC (RTX 5090, TensorRT 10.16.1, same engine build path)
- ONNX vs PyTorch: 24/24 frames same gate set, max corner diff 0.0001 px; refiner max diff 3e-6.
- Runtime logic vs the flown research detector: identical (0.0 px on 58 frames, 76/76 gates) when the heads use the same precision as the research code (bf16); in fp32/fp16 the corner-snap step amplifies tiny head differences to a few px on ~1/4 of gates (the research code itself shows this between runs).
- TensorRT fp16 engine vs PyTorch: main gates median 0.26 px / max 0.5 px, near-gate gates max 0.43 px, seg 99.9 % same class, depth 0.6 % rel.
- Closed loop (Isaac DroneRacing, seeds 141-148, 32 flights) with the TensorRT fp16 runtime as the detector: **81.2 % success (26/32)**, gates passed 88.8 %, = research detector on the same seeds (81.2 %).
- Time per frame on the 5090 (2026-11-23 package, GPU shared with two trainings): engine 1.1 ms, total 3.1 ms at 1640x1232; packaged copy replaying a 1640x1232 video: 87 FPS (decoder-bound), capture->output median 3.9 ms.
- 2026-11-23 runtime changes (each checked for identical output): CUDA-stream race fixed (the engine could read a half-written input under GPU load);
  CUDA graphs for both engines (bit-identical outputs); gate selection moved out of the graph (graph 6,012 -> 725 ONNX nodes, engine -22 %, select 50 us/frame);
  post-processing in numpy with one device->host copy. Parity with the flown research detector: 74/76 gates matched (before: 53/76, measured with the race).
  TRT-runtime closed-loop numbers below were measured before the race fix -> re-run planned.
- Camera pre-processing on the GPU (default in run.py; `--cpu-prep` = old path): BGR->RGB, fisheye undistort (grid_sample with float maps) and square crop
  after the upload. Synthetic 160 deg fisheye calibration, 1640x1232, 61 frames: same gate sets, corners median 0.35 px apart, image 0.19/255 mean diff
  (cv2 uses fixed-point maps); pre-processing 9.7 ms (CPU path, x86 core) -> 0.8 ms. On the Orin the CPU path would be ~25 ms per frame.

## Not verified yet / risks on the real vehicle
1. **Orin timing** (estimate, run `check_device.py` first): core alone measured on the Orin at 448 px MAXN = 16.7 ms; the system engine is now
   1.15x the core-only engine on the 5090 -> ~19 ms, CUDA graph -12 % (Orin trtexec) -> ~17 ms; host work ~1.7 ms on a 5090-host core x ~3 ->
   ~5 ms => ~22 ms per frame (~45 FPS). **INT8 is not worth it**: Orin measurement for this backbone +3 % speed (docs/ORIN_RESULTS.md), and
   post-training INT8 (TensorRT implicit entropy / min-max, ModelOpt Q/DQ) breaks accuracy here (seg agreement 0.27-0.78, gates 6-22/24).
2. **Camera — use a wide lens (>= 100 deg horizontal)**. Closed loop with this runtime (seeds 141-148): HFOV 128 -> 81.2 %, HFOV 100 ->
   78.1 % (same within noise), HFOV 62 (stock IMX219 lens) -> **18.8 %**, near-gate error 4x. Use an IMX219 120 / 160 deg variant. GateNet collapses too at 62 deg (6.2 %): the narrow view breaks the whole stack (policy / EKF), not only perception. The racing loop fed square derotated crops: use `--square` to match; derotation needs the IMU (flight stack side).
3. **Headless GPU bring-up** after a power-mode reboot was seen to fail for ~15 min (docs/ORIN_RESULTS.md); `run.py` waits up to 15 min.
4. Real-world accuracy of gate detection on IMX219 images is not measured (no labelled IMX219 gate frames yet). Record with `--record`
   on the first flights; those frames are the next labelled / self-training data.
