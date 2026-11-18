# Orin NX results (2026-11-10)

Device: NVIDIA Jetson Orin NX Engineering Reference Developer Kit, 8 GB (7.4 GiB visible), NVMe Samsung 990 PRO 1 TB (836 GB free), L4T R39.2.1, Ubuntu 24.04, CUDA 13.2, TensorRT 10.16.2, Python 3.12.
Work user on the device: `<user>` (no sudo, no password, key-only SSH from the test session). Project copy in `~/PerceptionModel`, venv `.venv`.
Torch on the Orin: `torch 2.15.0+cu132` (aarch64 wheel from the cu132 index). `torch.cuda.get_arch_list()` has no sm_87 but the sm_80 kernels run on the Orin (device capability 8.7): matmul and all tests work.

## Tests
- `pytest tests`: 9 passed (29 s).

## Latency, 384x384, batch 1, fp16, random weights (latency only, no accuracy meaning)
Power mode **15W** (nvpmodel id 2, 4 CPU cores online). Clocks not locked (no jetson_clocks). Desktop session was running (idle floor includes it).

| Student | PyTorch eager | TensorRT fp16 GPU compute (median) | TRT FPS | Idle W | Active W | J/frame total | J/frame net of idle |
|---|---|---|---|---|---|---|---|
| ViT-B (87.7M) | 53.5 ms | 39.5 ms | 25.3 | 6.2 | 12.2 | 0.482 | 0.237 |
| Hybrid FastViT-SA24 (8M), plain | 62.9 ms | 44.2 ms | 22.6 | 6.3 | 10.5 | 0.465 | 0.185 |
| Hybrid FastViT-SA24, reparameterized | n/a | **26.4 ms** | **37.9** | 6.3 | 10.8 | **0.286** | **0.119** |

Notes:
- trtexec run with `--noDataTransfers`: GPU compute only, no pre/post-processing. Engine build time is excluded from the energy measurement (first attempt included it and was wrong).
- The hybrid is slower than the ViT until it is reparameterized (multi-branch blocks merged); after that it is the fastest and the cheapest per frame.
- Power is VDD_IN from `tegrastats` (first live test of `src/pfm/energy.py`; parser handles the three-value format `a/b/c`).
- Not measured yet: MAXN and 20W (6 cores; switching needs a reboot), 448 px, memory in shadow mode, accuracy, EdgeSAM-class model, 3DGS render/train time.
- Scripts: `scripts/orin_trt_bench.py <vit|hybrid> [size] [hybrid_name] [reparam]`, `scripts/bench_students.py`.

## Incident: GPU did not come up after the reboot into MAXN (2026-11-10, ~13:28)
- Switched with `sudo nvpmodel -m 0 --force` (reboot needed: 4 to 6 CPU cores; the owner approved the reboot). Plain `nvpmodel -m 0` asks for a reboot confirmation on a terminal and did not accept piped input twice; `--force` works.
- After the reboot: mode MAXN, 6 cores online, but the GPU is unusable: `nvgpu` log shows `acr-gsp.data.encrypt.bin.prod ... failed with error -40`, `ACR bootstrap failed`, `Failed initialization for acr_construct_execute`; `/dev/nvhost-gpu` was missing; `nvpmodel.service` failed (cannot read GPU devfreq). `torch.cuda.is_available()` is False and trtexec says "no CUDA-capable device".
- The firmware files exist in `/lib/firmware/nvidia/ga10b/` (package `nvidia-l4t-firmware 39.2.1`). Before this reboot the same device worked (torch cuda True, all benchmarks above). Reloading the driver (`modprobe -r nvgpu; modprobe nvgpu`) recreated the device nodes but the GPU still fails to power on.
- Unknown cause. Hypotheses: (1) a warm reboot leaves the GPU secure-boot (ACR) in a state that needs a full power cycle; (2) MAXN-specific; (3) a firmware loading problem in this early L4T R39 build (kernel cmdline has `firmware_class.path=/etc/firmware`, which does not exist). Not yet tested: full power cycle; switching back to 15W.
- Current persistent power mode is MAXN (status file). To go back: `sudo nvpmodel -m 2 --force` (reboots).
- MAXN benchmark numbers: none yet (GPU unavailable).

### Update (13:49): the GPU works again, same boot
Without any further reboot or power cycle, about 15 minutes after the driver reload, `torch.cuda.is_available()` is True, `/dev/nvhost-gpu` exists, an fp16 matmul runs (~1.9 TFLOPS) and the TensorRT engine loads. Cause still unknown (my first check came right after the reload and failed; a later check passed). If it happens again, treat the first boot after a power-mode change as suspect and verify the GPU before measuring.
A quick ViT run in MAXN gave median 33.8 ms (min 26.8, max 38.7): NOT a valid benchmark, because Firefox (about 200% CPU), the remote desktop and the system monitor were loading all six CPU cores at that moment. Clean MAXN numbers need a quiet desktop.

### Evidence: GPU came up when the RDP session started (2026-11-10)
Boot 13:27:58. nvgpu ACR bootstrap errors at boot and at every retry until 13:32:30. First successful nvgpu power-on at uptime 995 s = 13:44:33 ("enabled scaling for GPU"). The remote GDM session (RDP) started at 13:44:34. Correlation only: either the desktop session triggers a correct GPU bring-up, or the GPU was simply ready about 16 minutes after boot and nobody tried in between.
Why it matters: the drone will run headless. To settle it after the next reboot: check the GPU over SSH BEFORE connecting RDP. If it fails headless, add a boot-time service that opens CUDA once (system change, needs the owner's approval). Also unknown: whether the first boot of the day (11:43, GPU fine) already had a desktop session.

## MAXN results (clean run, 2026-11-10 ~14:15)
Same setup as the 15W table (384x384, batch 1, fp16, TensorRT, `--noDataTransfers`, random weights). Power mode MAXN (nvpmodel id 0, 6 CPU cores, CPU at 1984 MHz). Desktop idle (CPU 0-7%, one RDP session open but not streaming heavy content), no other jobs. min/max spread under 1%.

| Student | 15W median | **MAXN median** | MAXN FPS | Speed-up | MAXN idle W | MAXN active W | J/frame total (15W -> MAXN) | J/frame net of idle (15W -> MAXN) |
|---|---|---|---|---|---|---|---|---|
| ViT-B | 39.5 ms | **19.6 ms** | 50.9 | 2.0x | 6.9 | 17.7 | 0.482 -> 0.348 | 0.237 -> 0.213 |
| Hybrid, plain | 44.2 ms | **24.7 ms** | 40.5 | 1.8x | 6.9 | 15.3 | 0.465 -> 0.377 | 0.185 -> 0.208 |
| Hybrid, reparameterized | 26.4 ms | **14.6 ms** | **68.7** | 1.8x | 6.9 | 15.8 | 0.286 -> 0.230 | 0.119 -> 0.130 |

Reading:
- MAXN roughly doubles the speed (1.8-2.0x) for about +5 W of active power (hybrid reparameterized: 10.8 -> 15.8 W).
- Energy per frame, counting everything, is lower in MAXN (the idle floor is spread over more frames); net of idle it is about the same or slightly higher.
- Hybrid reparameterized at 384 px: 68.7 FPS in MAXN, 37.9 FPS in 15W.
- Still excluded: camera, pre/post-processing, any other module; the shadow-mode (two models) memory; 448 px; 20W.

## Resolution sweep, MAXN (2026-11-10 14:00-14:25, quiet device)
TensorRT fp16, batch 1, random weights, GPU compute only. Hybrid = FastViT-SA24 reparameterized.

| Size | Hybrid ms (FPS) | Hybrid active W | Hybrid J/frame net | ViT-B ms (FPS) | ViT-B active W | ViT-B J/frame net |
|---|---|---|---|---|---|---|
| 224 | 7.8 (128) | 11.2 | 0.032 | 8.0 (125) | 16.3 | 0.075 |
| 320 | 10.8 (92) | 13.7 | 0.071 | 13.7 (73) | 17.5 | 0.145 |
| 384 | 14.6 (69) | 15.8 | 0.130 | 19.6 (51) | 17.7 | 0.213 |
| 448 | 16.7 (60) | 16.8 | 0.162 | 25.2 (40) | 18.0 | 0.283 |

Reading: at 224 px the two are equal in speed (the ViT only has 196 tokens) but the ViT uses far more power; from 320 px up the hybrid is faster and cheaper per frame, and the gap widens with size. The hybrid keeps 60 FPS at 448 px in MAXN.

## Shadow mode: several engines at the same time, MAXN (decision 8)
trtexec streams run flat out (not camera-rate limited), separate processes, GPU load about 98%. Baseline RAM before: about 2.4 GB (desktop session included) of 7.4 GiB; swap already at 743 MB before the test.

| Streams | Latency per stream | Throughput per stream | RAM added | Total power |
|---|---|---|---|---|
| 1 hybrid 384 | 14.5 ms | 68.8 | +240 MB | 16.1 W |
| 2 hybrid 384 | 32.5 / 32.8 ms | 30.6 / 30.2 | +500 MB | 16.4 W |
| ViT 384 + hybrid 384 | 45.7 / 33.3 ms | 22.5 / 29.9 | +480 MB | 18.6 W |
| 3 hybrid 384 | 48.8 ms each | 20.3 each | +730 MB | 17.2 W |
| 2 ViT 384 | 45.8 / 45.7 ms | 22.3 / 22.3 | +600 MB | 17.5 W |
| 2 hybrid 448 | 37.6 / 37.4 ms | 26.3 / 26.5 | +270 MB | 17.4 W |

Reading: memory is not the limit (about 3 GB used in total, 7.4 GiB available); the GPU is. Two models running on every frame roughly double the per-frame time (2 hybrids: 32.6 ms vs 14.5 ms alone, combined throughput 60.9 vs 68.8 FPS), so there is no free capacity. To keep headroom the shadow (candidate) model must run on a subset of frames, or after the flight on recorded frames. Not tested: same-process multi-stream (could be less wasteful than separate processes), CPU pre/post-processing, camera-rate limiting.

## EdgeSAM encoder on the Orin, MAXN
`edge_sam_3x_encoder.onnx` (22 MB, from Hugging Face `chongzhou/EdgeSAM`), input 1x3x1024x1024, TensorRT fp16: median 28.6 ms (34.9 qps). The decoder (16 MB ONNX) was downloaded but not timed. So the label model of decision 2 is feasible on the Orin (about 35 encoder passes per second; offline labelling only, runs after flight). The class still has to come from somewhere (student proposals or prompts): Grounding DINO is not on the Orin.

## 3DGS timing proxy with gsplat on the Orin, MAXN (decision 7)
gsplat 1.5.3 builds its CUDA extension on first use (385 s on the Orin with 4 jobs; one-time, needs nvcc 13.2 + ninja + `CUDA_HOME` + `TORCH_CUDA_ARCH_LIST=8.7`, and `.venv/bin` on PATH). Proxy test (`scripts/orin_gsplat_bench.py`): random Gaussians, plain RGB colours, no spherical harmonics, no densification, so a real run is somewhat heavier. Quality is not tested.

| Gaussians | Image size | Render (ms / FPS) | Train step (ms / it/s) | GPU memory peak |
|---|---|---|---|---|
| 100k | 640x480 | 8.5 / 118 | 25.4 / 39.3 | 291 MB |
| 300k | 640x480 | 18.2 / 55 | 60.5 / 16.5 | 874 MB |
| 1M | 640x480 | 53.0 / 19 | 198 / 5.0 | 2.9 GB |
| 300k | 1280x720 | 42.6 / 23.5 | 123 / 8.1 | 1.9 GB |
| 1M | 1280x720 | out of memory | out of memory | (8 GB unified, desktop uses about 2.4 GB) |

Rough training time at these speeds (30k iterations, a common 3DGS length): 300k Gaussians at 640x480 about 30 min; 1M Gaussians at 640x480 about 100 min; 300k at 720p about 62 min. 1M at 720p does not fit. Rendering new views for augmentation: about 55 FPS at 300k Gaussians / 640x480.
Not tested: camera poses from real footage (COLMAP or similar on the Orin), a real scene, quality, Isaac Sim on the Orin.

## End-to-end frame loop (MAXN, 384 px, camera-like 1280x720 uint8 frame in pinned memory)
`scripts/orin_pipeline_bench.py`: upload + GPU resize/normalise -> TensorRT -> argmax / depth normalise / corner peaks -> results to CPU. 300 frames after warm-up.

| Student | Loop FPS | ms/frame | Upload+resize+norm | TensorRT | Postprocess | Download |
|---|---|---|---|---|---|---|
| Hybrid reparameterized | **57.2** | 17.5 | 0.94 | 15.7 | 0.35 | 0.04 |
| ViT-B | 46.8 | 21.4 | 0.95 | 19.6 | 0.35 | 0.04 |

Pre/post-processing costs about 1.3 ms (about 8% of the loop). The TensorRT stage is about 1 ms slower here than in trtexec (Python launch, shared stream). CPU JPEG at 720p (one core, MAXN): encode quality 90 = 12.4 ms (806 KB), decode = 15 ms: logging every frame as JPEG on one core caps at about 80 frames/s encode; fine for the data-engine "keep selected frames" path. No camera driver or capture path (V4L2/NVMM) was tested.

## Thermal soak, ViT-B 384 px, MAXN, 5.2 min at full load (dev-kit fan on)
Latency stayed 19.5-19.7 ms (50.8-51.3 FPS) for all 12 chunks: no throttling. Temperature rose from 57.8 to 66.0 C and plateaued; power 18.4-18.8 W. An enclosed drone airframe with less airflow is not covered by this test.

## Cold disk read, Orin NVMe (Samsung 990 PRO 1 TB)
`scripts/disk_bench.py`: 400 x 1 MB files, evicted from cache, random read with 8 threads: cold 3,562 MB/s (about 712 images/s at 5 files per image), warm 5,633 MB/s. The Orin disk is not a bottleneck. The laptop NVMe gave 3,161 MB/s. Use the same script on the 5090 machine's data disk.

## EdgeSAM decoder
The published `edge_sam_3x_decoder.onnx` does not parse in TensorRT 10.16 (two Cast nodes to INT16). After patching those two casts to INT32 (measurement copy only): 3.4 ms per prompt call, same for 1 or 5 points. Per image with 10 prompts: 28.6 ms encoder + 10 x 3.4 ms = about 63 ms (about 16 images/s). Not validated for mask correctness (timing only).

## What does not work / cannot be fixed here
- **DLA**: `trtexec --useDLACore=0` fails with "Cannot create DLA engine, 0 not available". `/dev/nvhost-ctrl-nvdla0` exists and the kernel driver probed, but `libnvdla_compiler.so` is not installed and the NVIDIA apt repo for r39.2 has no `nvidia-l4t-dla-compiler`. NVIDIA forum threads report the same on JetPack 7.2 (L4T R39.2): DLA is not supported in this release. Fixing it would mean waiting for an NVIDIA update or reflashing JetPack 6.x (a different software stack: not recommended now).
- **Isaac Sim on the Orin**: the official requirements list only DGX Spark for aarch64, and ask for 32 GB RAM / 16 GB VRAM. It will not run on an Orin NX 8 GB: run it on the 5090 machine only.
- **Plain `trtexec --int8` without calibration data** gave 14.35 ms for the hybrid, the same as fp16 (14.55 ms): TensorRT silently kept fp16 ("Calibrator is not being used"). That number is not an INT8 result. A calibrated build is in progress (`scripts/orin_int8_build.py`, random calibration images: speed only).

## INT8 (calibrated with random images: speed only) and CUDA graph, MAXN, 384 px unless stated
`scripts/orin_int8_build.py` builds a calibrated engine (random calibration images: the timing is valid, accuracy is not).

| Engine | fp16 median | INT8 calibrated median | Gain |
|---|---|---|---|
| Hybrid reparameterized 384 | 14.6 ms | 14.2 ms | 3% |
| ViT-B 384 | 19.7 ms | 17.8 ms | 9% |

`trtexec --useCudaGraph` (all kernels launched as one graph), fp16 unless stated:

| Engine | no graph | CUDA graph | Gain |
|---|---|---|---|
| Hybrid 384 | 14.6 ms | 12.8 ms | 12% |
| Hybrid 320 | 10.8 ms | 9.2 ms | 15% |
| Hybrid 224 | 7.8 ms | 6.2 ms | 21% |
| Hybrid 384 INT8 | 14.2 ms | 12.5 ms | 12% |
| ViT-B 384 | 19.7 ms | 19.1 ms | 3% |
| ViT-B 224 | 8.0 ms | 7.5 ms | 6% |

Reading: the hybrid is partly CPU-launch-bound (trtexec warns "may be bound by Enqueue Time"), so INT8 barely helps it and CUDA graphs do; the ViT is compute-bound, so INT8 helps a little and graphs almost nothing. For deployment use a CUDA graph around the TensorRT call (not done in `orin_pipeline_bench.py`). With a graph the hybrid at 384 px would be about 12.8 ms (about 78 FPS engine-only).

## Whole frame loop inside one CUDA graph (MAXN, 720p camera frame, per-frame wait for the result)
`scripts/orin_pipeline_bench.py` now also captures upload + resize + TensorRT + post-processing + copies into one CUDA graph and replays it once per frame (the camera frame must land in the same pinned buffer each time).

| Engine | Eager loop | Graph loop | Speed-up |
|---|---|---|---|
| Hybrid reparameterized 384 | 57.3 FPS (17.4 ms) | **70.0 FPS (14.3 ms)** | 1.22x |
| Hybrid reparameterized 224 | 109.7 FPS (9.1 ms) | **138.2 FPS (7.2 ms)** | 1.26x |
| ViT-B 384 | 46.7 FPS (21.4 ms) | 48.7 FPS (20.6 ms) | 1.04x |

So a real frame loop with a camera-sized input reaches 70 FPS with the hybrid at 384 px, including pre/post-processing (the graph loop is even a bit faster than the engine-only number from trtexec without a graph, 14.6 ms). Not tested: a real camera/capture path, CPU work in the loop (planner, logging), variable input sizes, graph re-capture on resolution change.

## Real 3DGS training on the Orin: NeRF Synthetic "lego" (MAXN)
`scripts/orin_3dgs_train.py` (gsplat 1.5.3 with its default densification, spherical harmonics degree 3, L1 loss only, 100,000 random initial points, 7,000 iterations, 100 training views, held-out PSNR on 20 test views). Dataset: NeRF Synthetic (Blender), Hugging Face mirror `XayahHina/nerf_synthetic` (tagged MIT; the scenes themselves come from third-party models: credit the original NeRF authors in any paper). Camera poses are given, so no SfM was needed.

| Image size | Train time | Gaussians (end) | Test PSNR | GPU memory peak | Scene file (fp16) | Render FPS |
|---|---|---|---|---|---|---|
| 400x400 | 5.9 min (19.9 it/s mean) | 101,812 | 31.7 dB | 297 MB | 11.5 MB | 65 |
| 800x800 | 6.3 min (18.6 it/s mean) | 81,947 | 31.0 dB | 611 MB | 9.2 MB | 124 |

Notes:
- The first 400 px run gave 15.5 dB: a bug of mine (targets are on a white background but gsplat renders empty pixels black, so the model grew white floaters). Fixed with `backgrounds=ones`; both rows above are the corrected runs. The first-run Gaussian counts and times were discarded.
- Timing is noisy: during part of both runs the step time doubled (about 60-70 ms instead of about 30 ms) and available RAM dropped from 3.9 GB to 1.6-2 GB, so something else (probably the remote desktop session) was using the device. Quiet-device step time is about 29-31 ms (about 33 it/s). The PSNR, Gaussian count and memory numbers are not affected.
- A full scene fits comfortably: 297-611 MB of GPU memory, a 9-12 MB scene file. This also supports the "compact scene as replay storage" idea (decision 5): a scene costs about 10 MB.
- Quality is below a full 3DGS recipe (no SSIM term, short schedule). Not tested: real drone footage, camera poses from SfM (COLMAP/pycolmap on the Orin), large outdoor scenes, many scenes in memory.
- **No camera is connected to the Orin** (`/dev/video*` absent, no USB camera), so the camera-capture test could not be run.

## 3DGS depth, labelling and label fusion (2026-11-10, lego scene)
**Depth from the splat.** `scripts/orin_3dgs_orbit.py` renders RGB + expected depth along a NEW 90-view orbit (not a dataset camera): 91 MB of images + 16-bit depth + poses, 800 px, generated on the Orin. Against Blender's own depth PNGs on 10 held-out test views the rendered depth correlates at |r| = 0.958 on average (range 0.93-0.98; the sign is negative because the Blender depth PNG stores near = bright). Depth is a free label, valid up to the usual relative-depth caveats (no metric check: Blender's PNG is not metric).

**Labelling rendered frames (laptop, `scripts/label_consistency.py`).** Grounding DINO + SAM2 (`src/pfm/labeler.py`, demo prompts: bulldozer / toy truck -> car, lego baseplate / flat plate -> ground) on 8 orbit frames, 20 degrees apart. Geometric check of decision 6: each labelled pixel is moved into the neighbouring labelled frames with the rendered depth and the poses; accepted if the visible target pixel has the same class, rejected if another class, ignored if occluded or unlabelled.
- Result: 52.6% of the checked pixels agree (767,822 accepted vs 691,680 rejected). The ground is consistent in all frames; the bulldozer is not: in 3 of 8 frames (0005, 0020, 0060) most of the bulldozer is labelled as ground (frame 0020: only 6 px "car"). The check works (the ground agreeing proves poses and depth are right) and it flags exactly the inconsistent labels. Real label accuracy is not measured: there is no ground-truth segmentation for the lego scene.

**Labels on the Gaussians (`scripts/gaussian_label_fusion.py fuse`).** Each Gaussian votes over the labelled views where it is visible (centre within 3% of the rendered depth).
- 81,947 Gaussians, 98.2% labelled, mean vote confidence 0.84, 57% of them with confidence >= 0.8. Classes: ground 43,437, car 37,069, unlabelled 1,441.
- Agreement of each view with the fused labels: 0010 98%, 0045 97%, 0030 97%, 0015 95%, 0000 93%, 0005 71%, 0060 64%, 0020 48%. The voting singles out the same three bad views, and the majority is right in the sense that it matches the five consistent ones.
- Rendering labels from new views (`... render`, one-hot colours, argmax) and the simulator (`scripts/orin_3dgs_sim.py`) need gsplat + CUDA, so they run on the Orin; written, not yet run at the time of this note.

**Labelling on the Orin itself (what it can do without the 5090).** Measured on the Orin: depth from the splat (free), EdgeSAM encoder 28.6 ms + decoder 3.4 ms per prompt, geometric checks (numpy, CPU). Not measured on the Orin: Grounding DINO + SAM2 per image (about 3 s on the laptop RTX 5060; transformers is installed on the Orin for the timing test, run after the training jobs). Student training stays on the 5090.
