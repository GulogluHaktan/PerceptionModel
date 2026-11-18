# Plan: 3DGS + Isaac Sim data factory (written 2026-11-10, status same day)
Idea: 3DGS = real place (background RGB + depth); Isaac Sim = inserted objects, physics, light, perfect labels; `src/pfm/effects.py` = shared camera/weather effects at the end. LLM only fills a strict scenario JSON, it never generates pixels.
Link: path A (recommended, version independent): same camera pose rendered on both sides, per-pixel depth test; object labels from Isaac, background labels from per-Gaussian votes. Path B (Isaac imports Gaussians directly): UNVERIFIED in Isaac Sim 6.1, try for half a day on the 5090.
Steps (done / not done):
1. Camera contract (OpenGL camera-to-world + camera_angle_x, same in 3DGS and Isaac scripts). Test: a known object projects within 1 px on both sides. NOT DONE
2. Metric scale and alignment (COLMAP scenes are not in metres: scale from a known distance or telemetry; scenario field meters_per_unit). NOT DONE
3. Ground/collision stand-in: RANSAC ground plane from Gaussian centres first, TSDF mesh later. NOT DONE
4. Compositor (`src/pfm/compose.py`: depth test, shadow term, label merge) with unit tests. NOT DONE
5. Scenario schema (`src/pfm/scenario.py`: scene, objects, trajectory, camera, weather, seed; merged with `effects.validate_cfg`). NOT DONE (effects schema exists)
6. Trajectory with speed limits, then Isaac drone dynamics; velocity feeds motion blur (`velocity_from_poses` exists). PARTLY
7. Dataset writer: RGB, depth, class, gate corners, pose, scenario JSON and hashes as provenance. NOT DONE
8. Validation: equal-compute real-only baseline, capped synthetic share, golden set (needs the 5090). NOT DONE
9. Replay: keep scene library (about 10 MB per scene) and scenario files; regenerate old conditions every training round. NOT DONE
Split: Orin = 3DGS train/render, depth, label voting, effects; 5090 = Isaac Sim, heavy labelling, LLM, training; laptop (8 GB, official minimum 16 GB) = smoke tests only.

## State at the usage limit
- DONE: effects library + 5 tests (`src/pfm/effects.py`, rain as 3D drops is only roughly tuned: the user said the rain looked weak and my last tuning edit was cancelled, so the rain in the repo is still too faint); label consistency + per-Gaussian voting (`scripts/label_consistency.py`, `gaussian_label_fusion.py fuse`); simulator script `scripts/orin_3dgs_sim.py` written; bonsai real-photo 3DGS on the Orin: 6000 its, 32.1 min, 1,314,981 Gaussians, test PSNR 29.26 dB, SSIM 0.928, 25.5 render FPS, scene 148 MB fp16, 2.76 GB GPU.
- FAILED, cause known: `sim/isaac_hello.py` crashed because `add_update_semantics` no longer exists in `isaacsim.core.utils.semantics` (Isaac Sim 6.1, deprecated extension). Fix: use the USD semantics schema (try `pxr.UsdSemantics.LabelsAPI`) or `rep.modify.semantics`; run with `sim/run_isaac_hello.sh` (dirs need chmod 777).
- RUNNING on the Orin (started 17:56): `chain.sh` = label render from fused labels, simulator on lego, Grounding DINO + SAM2 timing. Results in `~/PerceptionModel/runs/orin_chain.log` on the Orin.

## Update (same day, later): the pipeline now runs end to end
Done and tested (`tests/test_sim_modules.py`, 6 tests, plus 5 effects tests = 11 passing):
- Step 1 camera contract (`src/pfm/camera.py`): OpenGL camera-to-world poses, OpenCV axes inside the code, project/unproject round trip tested.
- Step 3 ground stand-in (`src/pfm/ground.py`): RANSAC plane on the Gaussian centres (found the lego baseplate: normal 0.006/-0.007/1.0, 35,968 of 81,947 Gaussians as inliers). A TSDF mesh is still the better version.
- Step 4 compositor (`src/pfm/compose.py`): depth test, soft alpha, label merge, optional contact shadow (not used yet).
- Step 5 scenario schema (`src/pfm/scenario.py`): strict, with hash; step 6 trajectory with speed limit and per-frame motion (`src/pfm/trajectory.py`); step 7 writer with manifest and provenance (`src/pfm/dataset_writer.py`).
- `scripts/make_sim_dataset.py` + `configs/scenario_lego_gate.json`, run on the Orin: 90 frames, 122 ms per frame (8.2 fps) including the 3DGS render with depth and voted labels, objects, compose, effects. A gate and a pole were inserted into the real lego splat; occlusion by the bulldozer is correct; gate corners are exported per frame with a visibility flag (visible in 90 of 90 frames; a corner outside the image is flagged hidden). Scenario hash a68bfcc718f0. Speed limit 6 m/s forced the path to an effective 16.8 fps.
- Foreground is a flat-shaded analytic stand-in (`src/pfm/proxy_fg.py`), no shadows, not photoreal: the interface (rgb, depth, label, alpha) is what Isaac Sim renders must deliver.
Measured on the Orin the same day: Grounding DINO + SAM2 (the real label pipeline) takes about 41 s per image with the full 9-class prompt set and 6.6 s with 2 classes (6 images, 640 px; model load 189 s): not a job for the Orin (8,000 images would be about 91 h), it belongs on the 5090. 3DGS simulator script: 17 frames/s on the Orin with 5-subframe motion blur and labels.
Still open: step 2 real metric scale (the scenario field exists; no real scene calibrated), step 8 validation (needs 5090 + golden set), step 9 replay scheduler, LLM scenario manager, shadows, Isaac Sim foreground.
Isaac Sim on this laptop: renders in the LunarRocket project, so the laptop is fine. My `sim/isaac_hello.py` still fails with "renderer failed to advance" even with the same image (6.0.1), renderer (RayTracedLighting), caches and a World. LunarRocket gets frames from the Isaac `Camera` sensor (`isaacsim.sensors.camera`, `cam.add_rgb_to_frame()`) driven by `world.step(render=True)`, not from Replicator annotators on a custom render product: do the same next (Camera sensor + `add_distance_to_image_plane_to_frame` + `add_semantic_segmentation_to_frame`).

## Update 3 (same day): Isaac Sim foreground is wired in, camera contract verified by numbers
- **Isaac Sim foreground pass works** (`sim/isaac_foreground.py`, launcher `sim/run_isaac.sh`, export `scripts/export_scenario_for_isaac.py`): scenario -> poses/objects JSON -> Isaac renders only the objects (RGB, depth, class labels) with the Isaac `Camera` sensor -> `scripts/make_sim_dataset.py ... <isaac_fg_dir>` composites them over the 3DGS background on the Orin. 90 frames: Isaac 0.16 s per frame after a 160 s start (120 warm-up steps); composite + effects 50 ms per frame on the Orin (19.9 fps) with the Isaac foreground.
- What was needed to make Isaac render headless on this laptop (all of it was my script, not the laptop): image `isaac-sim:6.0.1` with the shared warm caches in `~/docker/isaac-sim` and `umask 000`; `renderer: RayTracedLighting`; a `World` created first; the `Camera` sensor (`isaacsim.sensors.camera`) with `add_rgb_to_frame`, `add_distance_to_image_plane_to_frame`, `add_semantic_segmentation_to_frame`; `world.reset()` BEFORE the camera pose is set (translate + orient ops); `import omni.replicator.core` and `rep.orchestrator.step()` each step (it logs "renderer failed to advance" but it is what fills the annotator data; the working LunarRocket scripts do the same); `add_labels(prim, [class])` instead of the removed `add_update_semantics` (6.1) with a fallback for 6.0; a small own PNG writer, because the Pillow inside the container crashes on save.
- **Camera contract, measured:** Isaac foreground versus the analytic renderer on 15 frames: before the half-pixel fix mask IoU 0.968, depth difference 1.1 mm, centroid shift 0.6 px (max 1.4); after using pixel centres at +0.5 everywhere (camera.py, proxy_fg.py): mask IoU 1.000, depth difference 0, centroid shift 0 px. So Isaac and the project agree to the pixel. The 3DGS side is checked indirectly (held-out PSNR with the same pose convention, correct occlusion of Isaac objects behind and in front of splat surfaces); a direct splat-versus-Isaac pixel test of a common object was not done.
- 4-scenario set regenerated on the Orin with the corrected convention (proxy foreground): `runs/sim_set/{clear_slow,fast_blur,fog,lowlight}`, 60 frames each, 110-120 ms per frame, hashes fbba980adb2e, 261428d9bb03, 78d5f0e6e00d, 6c91e67f7509.
- 11 unit tests pass (sim modules + effects).
- **Still open:** step 2 real metric scale; step 8 validation (5090 + golden set); step 9 replay scheduler; LLM scenario manager; shadows (objects cast none onto the splat); Isaac light is only tuned by eye (sun 900, dome 200); rain still too faint (3D drop model, tuned by eye, not changed since the user said it looked weak); Isaac objects are boxes (no real gate/tree assets yet); only the lego scene was used for the full pipeline (bonsai needs an up-vector and a scale first).

## Update 4 (same day): the remaining software steps are written and tested
- **Step 2 metric scale** (`src/pfm/scale.py`): `meters_per_unit` from a known distance, or from telemetry (camera centres in scene units versus GPS/VIO metres) with Umeyama similarity alignment and an RMS residual as the trust signal. Tested on synthetic data (recovers 0.37 within 0.01). No real scene calibrated yet: it needs real telemetry or a measured distance.
- **Step 9 replay scheduler** (`src/pfm/replay.py`): scenario library with condition, round, last/best golden score; `plan_round` keeps a real-data floor, a per-condition replay floor, extra weight for regressed conditions, a share for new scenarios, and returns a held-out regression suite (first scenario of each condition) that is never trained on. Tested.
- **LLM scenario manager** (`src/pfm/llm_scenario.py`): prompt with the strict schema, JSON extraction, validation, rejected items fed back for up to 2 retries; the LLM is any callable (no network code, tested with a fake LLM). Deterministic `coverage_matrix` (speed x weather x light x objects, 36 scenarios by default) works without an LLM. No real LLM was connected.
- **Shadows** (`src/pfm/shadow.py`, wired into `generate_frame` and `make_sim_dataset.py`): convex-hull footprint along one directional light onto the ground plane, only on background pixels that lie on the plane (not on the bulldozer). Visible but subtle on the Orin run; shadows outside the splat's ground area are not drawn; no shadow on inserted objects from the splat.
- 24 tests pass in total (whole suite).
- **Orin data-factory soak** (1,200 frames, lego, proxy foreground, MAXN): 312 s wall (3.8 frames/s end to end, 8.4 fps for generation alone), 10.8 W busy (7.0 W idle), tj max 56 C, RAM max 3.8 GB, and GPU load only 9%: the factory is CPU- and disk-bound (numpy compositing, PNG encoding of 3 images per frame, device-to-host copies), not GPU-bound. Writing PNGs in parallel would be the first speed-up. With the Isaac foreground and shadows: 84 ms per frame (11.8 fps) on 90 frames.
- **Still open:** step 8 validation (needs the 5090 and a golden set); real gate/tree/building assets instead of boxes; a real LLM behind `ask`; a real metric calibration; rain tuning; bonsai/other scenes through the full pipeline (needs up-vector and scale); the 5-cycle collapse test with this data.

## Update 5 (same day): rain tuned, PNG writing parallel
- **Rain** (`src/pfm/effects.py`): more drops (26,000 x density), brighter and wider streaks, clipped alpha. Visible now: long near streaks at 1/60 s, short thin ones at 1/1000 s, tilt with wind, different pattern in forward flight. Tuned by eye, not validated against real rain footage; very short shutter times (1/1000 s) still make rain hard to see, which is physically plausible but not measured. `scripts/rain_demo.py` renders the comparison sheet.
- **Parallel PNG writing** (`src/pfm/dataset_writer.py`, 4 threads, errors re-raised on `close`): Orin end to end 3.8 -> 4.8 frames/s (300 frames, includes the start-up; the 1200-frame soak before was 3.8). Per-frame generation time rose from 119 to 178 ms because the threads share the CPU cores, so the factory is CPU-bound overall; moving the compose/effects work to the GPU is the next real speed-up (not done).
- **Not done and why:** bonsai (or any COLMAP scene) through the full pipeline needs (1) an up-vector estimate from the camera poses and (2) oriented boxes with a full rotation in the proxy/shadow code (now yaw-only, Z-up), because a COLMAP world is not Z-up; (3) a scale; Isaac also assumes Z-up. Step 8 (validation) and the 5-cycle collapse test need the 5090 and a golden set.

## Update 6 (same day): a real-photo scene (bonsai) runs through the whole pipeline
- New: `src/pfm/frames.py` (up-aligned frame: scenario geometry is built with the scene's up = +Z and rotated into the scene frame; `estimate_up_from_poses` from the camera poses), `src/pfm/colmap_io.py`, `scripts/colmap_scene_info.py` (suggests up, centre, camera distance, hfov for a COLMAP scene), oriented boxes with a full rotation in `proxy_fg.py`/`shadow.py`/`objects`, optional `scene.ground_z`, depth kept in scene units until the end of `generate_frame` (`depth_scale`), per-frame percentile depth preview. 25 tests pass (rotation invariance: camera and box moved by the same rigid motion give the identical picture).
- Bonsai run on the Orin (`configs/scenario_bonsai_gate.json`, 1.3 M Gaussians, 640x427, proxy foreground, no background labels): 90 frames, 181 ms per frame (5.5 fps), gate corners visible in 77 of 90 frames, correct occlusion (gate behind the bonsai flagged hidden, pole in front of the cloth), objects stand on the table (`ground_z` = -2.95 units from the height histogram: floor at about -8.3, table top at about -2.9).
- **Assumption, not calibration:** `meters_per_unit` = 0.14 (a table of 5.3 units guessed at 0.75 m). Object sizes and speeds in metres for this scene are only as good as that guess.
- Bug found by looking at the first bonsai sheet: the spiral's altitude wave (0.4 m) and look-at drift (0.3 m) were fixed metres, so a 0.5 m scene lost the subject; both are now fractions of the radius (`wave` 0.12 and `wander` 0.08 of the radius). This changes the lego trajectories slightly (4 m radius: 0.48 and 0.32 m instead of 0.4 and 0.3), so lego datasets made before this change differ a little for the same scenario hash.
- Isaac foreground is still Z-up-only and was not used for bonsai.

## Update 7 (same day): factory data -> trainer -> golden-style evaluation, first toy experiment (plan step 8 plumbing)
- New: `scripts/factory_to_training.py` converts factory datasets into the trainer's layout (images, seg labels, corner labels, a golden-style folder with depth in metres). The trainer got a per-image `has_corner` flag: no corner file means "corners unknown", no corner loss for that image (the earlier labeller pipeline writes a corner file for every image, so it is unchanged). Gate corners are known in only 3 of 60 frames per scenario because the gate is mostly partly outside the image; this experiment is therefore about segmentation and depth. 25 tests pass.
- Toy experiment, equal compute (308 optimiser steps), student = FastViT hybrid at 224 px with DINOv2-S + Depth Anything V2-S distillation, trained on factory frames of the lego splat, evaluated on a HELD-OUT condition (low light, 60 frames) that neither run saw: A = clear/slow only (60 frames), B = clear/slow + fast blur + fog (180 frames). Three seeds each.
  - seg mIoU: A 0.389 / 0.423 / 0.436 (mean 0.416); B 0.379 / 0.408 / 0.408 (mean 0.399).
  - depth AbsRel (scale-shift aligned, lower is better): A 0.175 / 0.173 / 0.171 (mean 0.173); B 0.168 / 0.172 / 0.168 (mean 0.169).
  - Reading: adding the blur and fog conditions did NOT help segmentation (B is lower in all 3 seeds by about 0.02, with a seed spread of about 0.03-0.05) and helped depth a little (B lower in all 3 seeds by about 0.004). Both effects are tiny and within seed noise for segmentation. Nothing here shows that the augmented conditions improve generalisation to a new condition.
  - Why this proves little: same scene and same trajectory for all splits (the "held-out" low-light frames show the same content as the training frames, only the camera effects differ), the evaluation labels come from the same pipeline (voted per-Gaussian labels, not hand labels), one student size, 60-frame test set, 3 seeds. It validates the plumbing (factory -> trainer -> evaluator) and the equal-compute protocol, not the usefulness of synthetic data. The real test needs the golden set and real flight footage on the 5090.

## Update 8 (same day): why the first experiments disappointed, and what changed
**What was wrong (all measured):**
1. The four first scenarios used IDENTICAL camera poses (max pose difference 0.0): the "different conditions" were the same 60 views with different effects, so training, test and the "held-out" low-light frames showed the same content, and the self-labelling pool was the same views as the real labelled frames. Viewpoint diversity was zero. (My design mistake: speed only changes the effective fps, not the path.)
2. The factory labels were noisy: with the first voting the bulldozer was labelled ground in about half the views (plain vote over 23 frames: 61,332 ground vs 19,641 car Gaussians, confidence >= 0.8 for only 50%). Per-class IoU showed a see-saw between ground and car (B: ground +0.10, car -0.06), so mIoU moved with label noise.
3. 224 px gives a 14x14 feature grid: thin classes suffer (pole IoU 0.22-0.28). Run-to-run noise in mIoU was +-0.03 (the Orin loop even rose in its last cycle).
4. The gate compared a model only with the previous accepted one, so a slow drift (each step -0.015) passed. Fixed: `max_total_regress` against the best-so-far (test added, 26 tests pass).
**What was changed:** label fusion with a geometric prior (a Gaussian more than 0.12 units above the ground plane cannot be ground): ground 38,828 vs car 41,526 Gaussians, confidence >= 0.8 for 98%, visibly clean in rendered views (the old per-view agreement metric stops being a quality measure, because the bad views now disagree with the good fused labels). 12 new scenarios with different viewpoints (4 train views x clear/varied conditions, 2 test views x clear/low light), equal viewpoints in both arms. Flip-consistency pseudo-labelling added to the self-evolution loop.
**Results (lego splat, FastViT hybrid, DINOv2-S + Depth Anything V2-S distillation, 420 steps each arm, test on NEW viewpoints, 3 seeds; A = 4 views all clear, B = the same 4 views with blur / fog / rain / dusk; reference labels come from the same pipeline, not hand labels):**
- 224 px: clear-test seg mIoU A 0.439 vs B 0.422 (B lower in 3/3); low-light seg A 0.419 vs B 0.415 (noise); low-light depth AbsRel A 0.1705 vs B 0.1677 (B better 3/3).
- 448 px (seed 0,1,2): clear seg A 0.509 vs B 0.481; low-light seg A 0.426 (0.453/0.386/0.440) vs B 0.472 (0.470/0.499/0.449), B better in 3/3 seeds, +0.046 mean, mostly from one bad A seed; low-light depth AbsRel A 0.235 vs B 0.223 (B better 3/3). Resolution 448 vs 224 raised seg mIoU by about 0.05-0.08, but depth AbsRel got WORSE (0.17 -> 0.22, cause not found: maybe under-trained depth head at 420 steps, or the 448 teacher maps; not investigated).
- Reading: varied-condition data costs a little on clean images (-0.02 to -0.03 seg) and helps the unseen low-light condition only at 448 px (+0.05 seg, -0.012 depth), not at 224. One scene, labels from the same pipeline, 60-frame test sets, 3 seeds.
- Collapse loop on the new data (real = 2 views, pool = 2 other views, student labels the pool, 5 cycles, gate off): confidence-only: seg mIoU 0.432 -> 0.430 (flat), pool label accuracy 98.2% -> 97.4% while the labelled share rose 84% -> 96%; with flip consistency: 0.432 -> 0.440, accuracy 98.3% -> 97.3%. No collapse here; the earlier worse drift (94.6% -> 92.4%) came from noisy labels and identical views. Flip consistency did not change the pool accuracy meaningfully.
- Orin: the 5-cycle loop (old data) took 778 s (13 min, about 6x slower than the laptop), 14.0 W mean (17.5 W max), 6.1 GB RAM, GPU load 80%; its pool label accuracy matched the laptop to the third decimal (deterministic), its mIoU wobbled (0.411, 0.376, 0.352, 0.344, 0.399).

## Update 9 — exact factory depth as a training signal + TensorRT accuracy (2026-11-10)
- Trainer: `depth_gt_weight` (masked scale-shift-invariant L1 vs `labels/depth/*.npy`, metres, 0 = no surface); `factory_to_training.py` writes the depth labels. Driver: `scripts/run_depth_gt_ablation.py`.
- Ablation (448 px, FastViT hybrid, 420 steps, 3 seeds, mean; seeds 2-3 first ran by mistake from a resumed checkpoint and were discarded, results below are the corrected runs):
  | variant | clear mIoU | clear AbsRel | low mIoU | low AbsRel |
  |---|---|---|---|---|
  | teacher only | 0.519 | 0.221 | 0.341 | 0.243 |
  | factory depth only | 0.485 | 0.176 | 0.368 | 0.222 |
  | 0.3 teacher + factory depth | 0.442 | 0.170 | 0.379 | 0.217 |
  AbsRel gain is consistent on every seed (0.221 -> 0.176/0.170, seed spread < 0.001). Seg: factory-depth-only loses 0.03 on clear (within the +-0.03 noise, seeds 0.45-0.51), the mix loses 0.08 on clear (3/3 seeds lower) but gains on low light. Recommendation: factory-depth-only as the default; the mix only if depth matters more than clear-weather seg.
- TensorRT fp16 vs PyTorch fp32 on Orin (120 clear golden frames, 448 hybrid, engine build ~252 s): baseline model seg mIoU 0.4999 / AbsRel 0.2205 (identical in both); depth-GT mix model 0.4527 vs 0.4528 mIoU, AbsRel 0.1696 vs 0.1696, pixel agreement 95.7 %. fp16 deployment loses no accuracy, and the depth gain survives it.
- Bugs found while doing this: out_dir resume silently skipped training for later seeds; trt script reused a stale engine file (now per-checkpoint dir).
- Orin thermal soak, depth-GT 448 hybrid, TensorRT fp16, 10.4 min flat out (MAXN, dev-kit fan): 15.5 ms median (~64 fps), no throttling (throughput 64.7 -> 64.6 qps), temperature 52.7 -> 60.4 C, 14.5 W. Enclosed-drone airflow still untested.

## Update 10 — new goal: Isaac Sim data factory completion + LLM selection (2026-11-10)
User direction: (1) pick an LLM that fits one 5090, (2) finish the Isaac Sim side: tree / building / car / road assets (URDF/USD, found or built), scenes from the existing photos, parkour/course design, render only what the camera sees, realistic camera physics (15 m/s motion blur, shutter, lighting), gate-interior labels like the race-gate picture, (3) LLM-free test frames labelled by the Orin and compared. LLM comes later; its job is to read the problem and the latest images, then fill the scenario schema (camera inverted / other mount, angle, speed, light) for ready-made Isaac environments.

Order (each step testable without the LLM and without the 5090):
1. Gate interior label + course: opening corners and an interior mask next to the existing outer corners; `course` block (ordered gates "10/22", slalom / circle / random generators) in the strict schema.
2. Asset registry: type -> {USD/URDF, size, class id, licence}; procedural USD tree / building / car / road first (works today), real assets swapped in later by registry entry only. Photo scenes stay COLMAP -> 3DGS; objects placed on the fitted ground.
3. Frustum-culled Isaac pass: per pose, only objects inside the view frustum (and not beyond far range) are activated; background already splat-culled by gsplat.
4. Camera model: exposure triple (shutter, ISO, EV) tied to scene light; `camera.mount` (offset, roll/pitch/yaw, inverted); blur length = speed x shutter, rolling shutter, check 15 m/s vs 2 m/s visibly different.
5. LLM-free test set: `coverage_matrix` -> render -> Orin labeller (GroundingDINO+SAM2, ~41 s/img, so ~200 frames ~2.3 h) vs exact sim labels, plus student on the same frames.
6. LLM infrastructure: local server with grammar-constrained JSON (schema from `validate_scenario`), model registry, VRAM budget (LLM unloaded while Isaac renders), vision input of recent frames, existing validate-and-repair loop.

LLM recommendation (unverified for our task, to be benchmarked on our schema): default Qwen3-VL-8B (fits next to Isaac, roughly 10 GB in FP8), upgrade Qwen3-VL-32B at 4-bit (~20 GB, time-shared when Isaac is idle); Gemma 4 26B-A4B as the alternative. JSON reliability comes mainly from constrained decoding, not the model choice; pick the final model with a 50-scenario schema-validity test on Monday.

## Update 11 — asset library, simple city, Orin labeller check (2026-11-10)
- NVIDIA public bucket (Isaac 6.0) has trees (many species), Rivermark buildings/props/signs/grass; NO passenger cars, NO road mesh. Cars: 4 Objaverse GLBs (CC BY, attribution in `assets/cars/ATTRIBUTION.json`) registered in `src/pfm/assets.py`; road/lane markings procedural. Gotchas fixed: USD references do not convert units/up-axis (trees are cm), referenced roots carry their own xform ops (3-level prim: our transform / unit fix / asset), container has no timm (classes live in `pfm/classes.py`).
- `sim/isaac_city.py` builds a seeded city (roads, sidewalks, lane markings, buildings, street trees, benches, lamps, cars), 3 lights x drone/rover/inverted-camera trajectories; `scripts/city_effects.py` makes 5 conditions per frame (clear, blur, fog, rain, low light) with exact depth: 240 frames -> 1200 images in `data/city_set`.
- Orin check `scripts/orin_city_eval.py` (Grounding DINO + SAM2 vs exact sim labels), 120 images, ~51 s/img. Partial result (12 images, one run, dusk drone): tree recall 0.89 / precision 0.42, ground recall 0.81, building recall 0.11, car 0.0, lane marking ~0. Suspects: MAX_FRAC filter drops large masks, prompts miss shop-front buildings and small cars. Final numbers pending.


## Update 12 - asset library 33 -> 212 (2026-11-11)
- NVIDIA bucket: all 44 tree species, 8 palms/ferns, 22 shrubs (class other_obstacle), 15 rocks (class rock), 18 props, 18 poles/lamps, apartment complex. URLs checked (HTTP 200); only `grass_01` returns 404 (decoration, no label class).
- Objaverse (CC0 / CC BY, listed with author and page in `assets/models/manifest.json`): 23 cars (incl. jeep, minivan, pickup, truck, ambulance, bus, taxi), 4 aircraft (2 planes, 2 helicopters), 6 two-wheelers (bicycles, scooters, motorcycle), 36 indoor items (sofa, bed, table, desk, wardrobe, fridge, TV, lamp, sink, toilet, stove, ...).
- Fine tag rides in the registry entry (`tag`); project class stays coarse (`car` or `other_obstacle`). New classes (airplane, bicycle, furniture) need a new head: not done.
- `sim/isaac_models_test.py` renders a sheet per group (`runs/fg_job/out/models_test/*.png`). Sheets show 65 of 68 models upright and visible; Gee Bee, B52 and Harley were dropped (wrong axis / base plane). The per-model pixel counter reads stale labels after the first group: unverified numbers, images are the evidence.
- Sizes: `TAG_SIZE_M` in `src/pfm/assets.py` (typical real-world longest extent per tag); the city spawner still fits GLBs to 4.5 m and must switch to `target_size_m` before using non-car GLBs.


## Update 13 - declarative indoor layout (LLM picks ids and coordinates, never code) (2026-11-11)
- `src/pfm/layout.py` (data-only, no torch): schema `{room, light, items[{id, asset, at, yaw_deg}], camera{waypoints, look_at, roll_deg, speed_mps, fps, hfov_deg}, seed}`. `at` is `[x, y]`, `{wall, along}` or `{near, side, gap}`.
  The LLM sees only `catalogue_text()` (id | tag | real size). The resolver supplies size (`assets/models/dims.json`), floor contact, wall anchoring, rotated footprint, then rejects (never silently fixes): unknown asset, extra keys (size, scale), overlap, leaving the room, item taller than the room, camera too close to walls/items, path too fast or too long, circular `near`. Each error names the item and the limit, ready for the repair loop.
- `sim/isaac_indoor.py` renders a resolved scene (`scripts/layout_to_job.py <layout.json> <name>` -> `runs/indoor_job/scene.json`): room shell, GLB furniture, 3 light modes (day, lamp, dark), camera path with roll (180 = inverted). Labels: floor ground_road, walls/ceiling building, furniture other_obstacle (coarse classes, see Update 12). Output `runs/indoor_job/out/<name>/{rgb,depth,label}` + `poses.json`.
- Verified: room_a (day, 25 frames) and room_b (lamp, roll 180, look_at, 11 frames) render with correct labels and metric depth; 5 resolver tests (`tests/test_layout.py`), 31 tests in all.
- Bug found and fixed: Isaac camera default near clip 1.0 m cut the view near walls (min depth stuck at 1.0, black pixels); indoor camera now uses 0.05 m. City generator already used 0.1 m.
- Not done yet: windows/doors/openings, stacking (item on a table), furniture orientation (which side is the front is unknown per asset, only yaw is given), effects on indoor frames (scripts/city_effects.py not yet pointed at indoor_job), labeller check indoors, only 36 indoor assets of mixed style, `dark` mode brightness not tuned, LLM wiring (prompt = `SCHEMA_TEXT` + `catalogue_text()`, repair loop = same as pfm.llm_scenario).

- Fix (2026-11-11): scripts/city_effects.py applied rain and fog to indoor rooms; indoor frames now only get clear, blur and lowlight (INDOOR_SKIP). Effect seeds now use crc32 (reproducible; hash() changed per process).

- LLM layout test (qwen3:8b, local Ollama, schema-constrained, 12 requests, one run each at temperature 0.7, so the counts are noisy): round 1 = 4 valid on the first try, 6 after two repairs (`runs/llm_layout/qwen3_8b.json`); with a nearest-free-centre hint in overlap errors and layout tips in the prompt = 5 on the first try, 9 after two repairs (`qwen3_8b_v2.json`), mean 14 s per layout. Remaining failures: chairs/lamps stacked on big furniture, a window put in `items` instead of `openings`, camera above the ceiling box.

## Update 14 - indoor closed (2026-11-11)
- Low light: effect `lowlight` for indoor now nets about 2 stops darker (`ev -3`, `gain_ev +1`; gain 3 cancelled the ev and only added noise); Isaac `dark` light raised (sphere 1500 -> 9000, the kitchen was black). No rain/fog indoors (`INDOOR_SKIP`).
- New resolver rules after looking at LLM rooms: camera pitch at most 55 deg (frames staring at the floor), and at least 40% of the frames must have an item in view (valid but empty rooms). Window/door written as items now gets a hint to use `openings`.
- `sim/run_indoor_set.sh name...` resolves and renders layouts; `scripts/city_effects.py` makes the condition images.
- LLM validity (qwen3:8b via Ollama, schema-constrained, 12 requests, 2 repair rounds, one run each, noisy): v1 (rules only) 4 first try / 6 valid; v2 (+ free-spot hint, prompt tips) 5 / 9; v3 (+ pitch rule) 2 / 7; v4 (+ visibility rule) 2 / 5. Each added rule lowers the pass rate: the 8B model is weak at geometry (items on top of each other, camera above the ceiling, path too long at high fps). Files `runs/llm_layout/qwen3_8b*.json`.
- Pictures: `data/indoor_set` (3 hand layouts), `data/indoor_llm2` (4 LLM rooms, made under rules v3).
- Next for the LLM part: an `auto` placement (the system finds a free spot, the LLM only gives the intent) and a bigger model; decide Monday on the 5090 with the 50-scenario test.

## Update 15 - outdoor declarative layout (2026-11-12)
- `src/pfm/layout_out.py` (resolver, no Isaac import), `src/pfm/layout_gen_out.py` (rule generator: drone, rover, inverted, fast_drone 30 m/s), `sim/isaac_outdoor.py` (renderer), `sim/run_outdoor_set.sh`, `scripts/layout_to_job_out.py`. Same principle as indoors: the LLM picks asset ids, roads (straight segments) and coordinates; the system sets sizes, tree/pole height, lane/kerb placement and yaw along the road, and rejects layouts that break rules (overlap, items on the carriageway, building setback, camera inside items, speed above 40 m/s, pitch above 80 deg, fewer than 40% of frames with an item in view). 40 tests in all.
- `sim/isaac_outdoor_dims.py` measured 162 assets (7 failed to load and are excluded) -> `assets/models/dims_outdoor.json`. Buildings are big (median 56 x 34 m, apartment complex 190 m).
- Four generated sets rendered (`runs/outdoor_job/out/out_{drone,rover,fast,inv}`, 95-263 frames each, 640x480, depth, labels): drone 15 m/s, rover 8 m/s, fast drone 30 m/s, inverted 12 m/s.
- FOUND: about 20 tree species (sugar_maple, red_maple, red_ash, white_ash, scarlet_oak, shumard_oak, colorado_spruce, ...) render with almost no semantic label on their leaves (labelled fraction 0.3-20%), measured on the real renders with `scripts/outdoor_label_coverage.py` (a grid test in Isaac agreed for trees but gave nonsense for buildings, so it was dropped). Un-instancing prims did not help (hypothesis rejected, code removed); cause unknown. Those 22 assets are now excluded from the outdoor catalogue (`assets/models/label_coverage.json`, threshold 0.6). The four rendered sets were made BEFORE the exclusion: their unlabelled trees are 255 (ignored in training, but false positives if a model predicts tree there). Regenerate before using them as an eval set.
- FOUND: effects assumed forward-only camera motion. `scripts/city_effects.py` now derives velocity and angular rate from consecutive poses for indoor/outdoor runs (|v| checked = 30.0 m/s on the fast run). Blur at 30 m/s is small in these scenes because the nearest surface is about 20 m away (1/250: about 2 px; blur grows as 1/depth, so the gate-close case needs close geometry).
- Not done: gates (needs the DroneRacing gate model), gate course schema, frustum culling, outdoor LLM test, per-species label fix, regeneration of the sets.

## Update 16 - slot head for partly visible gates, trainer fixes (2026-11-13)
- New `slots` output in `DenseHeads` (4 gates x [score logit + 4 corners u,v], image-normalised, values may lie outside 0..1) plus `slot_targets` / `slot_loss` in `src/pfm/train.py`; the trainer reads `labels/corners_full/*.json` (config key `slot_weight`, default 1.0). Output format matches `droneracing.estimator.det_features` input. On the 2209 racing frames rendered so far, 854 (39%) are partly-visible-gate frames that had no corner loss before; they now train the slot head (targets per frame: 4 gates 691, 3: 645, 2: 425, 1: 339, 0: 109).
- Fixed a trainer bug: the segmentation-mask variable overwrote the exact-depth mask (`has_d`), so the depth-GT loss used the wrong mask. Renamed (`hseg`). Earlier depth-GT results used factory data where both masks coincide, so they should not change; not re-run.
- `gen_sim_set.py train` now converts all three domains with the same converter (indoor/outdoor: seg + depth, no gate files). 45 tests pass. Not trained or measured: the head is wiring-checked only.
- Idea (user, 2026-11-13): photo -> 3D mesh -> USD with an image-to-3D model (TRELLIS / Hunyuan3D / TripoSR), not an LLM; class and real size set by our resolver. Not started.

## Update 17 - photo -> 3D asset prototype (2026-11-13)
- `scripts/image_to_asset.py <image> <tag> <name>`: background removal (rembg) -> TripoSR (MIT, 1 image, no LLM) -> mesh scaled to the REAL size of the class tag (`TAG_SIZE_M`, or `--size_m`), standing on y=0, Y-up -> `assets/generated/<name>.{glb,usd,json}` (meta: tag, size, faces, generator, seconds). The model gives shape and colour only; class and size come from us. Runs in its own env `<local path>` (third-party code kept out of the project; `shim/torchmcubes.py` replaces the CUDA build because there is no nvcc here).
- Measured on the laptop (RTX 5060 8 GB): 2.3 GB peak GPU, 7.5 s per object at marching-cubes 256 (first run 185 s = downloads). Two examples (chair, robot) checked by rendering views: shapes plausible from all sides, 84-161k faces, watertight, positive volume.
- Found: TripoSR's raw mesh has X as the UP axis (an unrotated export looks like a chair seen from above); fixed with a proper rotation, not an axis swap (a swap mirrors the mesh).
- Not done: only 2 clean example images, no real photos (background clutter, perspective) tested; no Isaac load test of the `.glb`/`.usd` (the registry already supports `kind: glb`, spawner rescales by tag); front direction (yaw) unknown; texture is vertex colour only (low detail); decimation for 100k+ faces; the class label must be assigned per asset by us (registry entry); licence check of source photos.
- Next: try 5-10 real photos of objects (chairs, cars, poles), load one in Isaac indoor, compare against Hunyuan3D-2 / TRELLIS on the 5090 (better texture).
- Catalogue test (2026-11-13, "from the catalogue" instead of real photos): (a) 4 cars cropped from our own Isaac renders with the class label as cut-out: poor, the label component merges neighbouring cars and the crop shows only a rear view, so the mesh grows "wings" and the 4.5 m scaling uses the junk extent; instance masks would be needed, not class masks. (b) 4 catalogue GLBs (old chair, purple armchair, desk, smart car) rendered as clean 3/4-view images (vertex colours only) and regenerated: chair/armchair/desk recognisable in shape from all sides, thin parts (legs, lamp) coarse or blobby, texture smeared; the car is tilted and blobby (bad). Verdict: usable for coarse furniture/obstacle shapes in perception data, not for cars or fine detail; a better model (Hunyuan3D-2 / TRELLIS) is worth testing on the 5090. One run each, judged by eye, no metric. Files `assets/generated/gen_*`, inputs were in the session scratchpad.
- Quality upgrade (2026-11-13): `image_to_asset.py --backend hunyuan` (default) uses Hunyuan3D-2mini-turbo (0.6B, 4.4 GB GPU, about 46 s for the first object, 5 diffusion steps, octree 256) instead of TripoSR. Same 4 catalogue inputs, judged by eye: chair legs/curved frame, armchair, desk with the globe and drawers, smart car with wheels and mirrors are all clearly correct from every side (TripoSR gave blobs; the car was unusable). Raw mesh 490k faces, decimated to 60k (`--max_faces`) with no visible loss; all four watertight, positive volume, Y-up, on y=0. Cost: SHAPE ONLY, so each asset gets one flat colour (median of the object in the photo); real texture needs the Hunyuan paint model (16+ GB GPU, try on the 5090). Size comes from the tag, so a smart car (2.7 m) tagged `car` (4.5 m) comes out too long: pass `--size_m` per asset.
- LICENCE: Hunyuan3D 2.0 Community Licence is territory-limited (not EU, UK, South Korea) and has other conditions: read `<local path>` before using generated assets in a paper or release. TripoSR (MIT) stays available with `--backend triposr`.
- Isaac load test (2026-11-13): `scripts/layout_to_job_gen.py` (offers `assets/generated/*.json` to the resolver in memory as `gen_<name>`, catalogue files untouched) + `runs/indoor_job/layouts/gen_demo.json` rendered with `sim/isaac_indoor.py` (about 3 min). Picture `runs/gen_asset_demo.png`: generated Hunyuan chair, the original textured catalogue chair and the generated armchair side by side; sizes, floor contact and orientation correct without any change to the renderer (the desk with the globe is behind). The generated ones are flat-colour (shape only), the original has its texture.
- Architecture idea (user, 2026-11-13): when the LLM sees something important in camera images that is not in the catalogue, it sends the object's picture to Hunyuan and adds the result to the catalogue. Needed pieces, not built: (1) instance mask/crop of the object (SAM-style; class masks merge neighbours, see the car test), (2) the LLM picks a tag from the closed `TAG_SIZE_M` list (or a bounded size), never free sizes/code, (3) automatic quality gate before registering: watertight, aspect ratio plausible for the tag, re-render and compare with the crop, (4) duplicate check against the catalogue, (5) licence of the source photo and of Hunyuan (territory-limited), (6) one GPU job at a time (not together with Isaac), (7) write the manifest/dims entry with provenance (source image, model, date).

## Update 18 - catalogue growth loop built (2026-11-13)
Loop (user's architecture): camera frame -> VLM names important objects not in the catalogue -> SAM2 instance crop -> Hunyuan -> automatic gate -> duplicate check -> asset registered with provenance.
- `src/pfm/asset_growth.py` (pure logic, 3 new tests, 48 in all): detection schema (tag = enum of the `TAG_SIZE_M` table, so no invented tags; `why` comes first in the schema so the model says what it sees before choosing the tag), answer validation (unknown tag, tiny box, low importance, tag already known are dropped; Qwen 0-1000 coordinates rescaled), gate, dHash duplicate check, `load_accepted`.
- `scripts/grow_catalogue.py <frame> [--known tags] [--dry]`: VLM (qwen3-vl:8b-instruct via Ollama, unloaded after use) -> SAM2.1 tiny box prompt -> `image_to_asset.py` in the Hunyuan env -> gate -> `assets/generated/<name>.json` with `status` accepted|rejected, gate metrics and provenance (source frame, licence field, VLM, detection, date, shape model). Only `accepted` is offered to the resolver (`scripts/layout_to_job_gen.py`); existing prototypes backfilled by hand (hy_cat_* accepted, TripoSR/bad-crop ones rejected). One GPU model at a time; never together with Isaac.
- Test: the Isaac demo picture (3 chairs) with `--known ""`: 3 detections with accurate boxes, 3 assets built and accepted, 3.1 min total on the laptop; rerun: all 3 flagged duplicates (VLM+SAM only, about 1 min). Output in `runs/grow_test`.
- FOUND: `qwen3-vl:8b` (thinking build) returned an empty answer with `think: false` + schema; the `-instruct` build works. First prompt asked 0..1 fractions and the tag came out as `flowerpot` for chairs; asking 0-1000 coordinates and putting `why` before `tag` fixed both on this one picture (n=1).
- LIMITATION OF THE GATE (measured, 12 meshes): silhouette IoU against the crop does NOT separate good from bad shapes (Hunyuan 0.75-0.90, TripoSR 0.67-0.88, TripoSR desk 0.88 > Hunyuan desk 0.77): a generator reproduces the input view whatever the back looks like. It only catches gross failures (empty mesh, wrong crop). The proportion check uses the catalogue's 2 examples per tag (coarse range, margin 40%): it flags only extremes (two of four bad TripoSR cars). So the gate is a floor, not a quality guarantee; accepted assets are unreviewed. Better ideas, not built: generate with 2 seeds and compare silhouettes at equal yaw (instability = low confidence), view-consistency with a second photo when the object was seen twice, render-back with CLIP/DINO similarity on the 5090, a human spot check of a sample.
- Also not done: size comes only from the tag (an armchair tagged `chair` gets 0.9 m; the VLM could output a size class but not metres), colour is flat, one source frame per object, no instance occlusion handling (a half-hidden object gives a half-object mesh), no end-to-end run on a real camera frame, the generated assets are not yet in the LLM layout catalogue (by design: opt-in until reviewed).

## Update 19 - seed-consistency gate, real-photo test, upright fix, native-Isaac scripts for the 5090 (2026-11-13)
- Seed consistency (`asset_growth.seed_consistency`, in the gate): the same photo generated with two seeds, mean silhouette IoU over 24 identical views. Calibrated on 8 inputs: 4 clean renders 0.975-0.991, 4 bad merged-car crops 0.69-0.90; threshold `CONSISTENCY_MIN = 0.93` (in-sample, n=8, so optimistic). Unlike single-view IoU (Update 18) it separates good from bad. Costs one extra generation (about 1 min on the laptop; `--no_consistency` skips).
- Real photos (Aeroscapes drone frames, `runs/grow_real`, 2 frames, 4 detections): 2 junk detections (traffic-light pole and crosswalk lines both tagged `police_car`: no matching tag exists) and a bus cut by the image edge were REJECTED by the gate (consistency 0.84/0.55/0.88, flat mesh, proportions); a silver car seen from above was ACCEPTED at 0.935 (marginal). Weakness found: the VLM must pick a tag even when none fits (no "other" tag), the gate then does the cleaning.
- FOUND and fixed: that car came out rolled (the generator copies the photo's viewpoint; height 2.2 m for a 4.5 m car) and the gate did not notice. New `src/pfm/mesh_upright.py`: for ground-standing tags (vehicles, bed, sofa, table) PCA gravity alignment (smallest axis up, longest along x, wider end down); `image_to_asset.py` applies it (`--no_upright` to skip). Result: upright sedan, 4.5 x 1.75 x 2.16 m (open door widens it); the clean smart car unchanged. Chairs, poles, planes and bikes are not aligned this way. One example each, judged by eye; the gate itself does not check uprightness yet.
- 5090 with native Isaac: `scripts/day1_5090.sh` (stages setup, isaac, hunyuan, llm, sets, assets, train), `scripts/setup_isaac_native_5090.sh`, `scripts/setup_hunyuan_5090.sh`, `configs/sim_5090.yaml`, native switch in `sim/run_isaac.sh`; the `sim/*.py` no longer hardcode `/workspace/pfm` (`PFM_ROOT`). Checked here: container render still works after the path edit (pixel difference to the earlier render: mean 0.24/255, max 11, i.e. render noise), bash syntax of all new scripts, 50 tests. NOT run: the native Isaac, the pip install route for Isaac Sim 6.0.1, the Hunyuan setup script from scratch, the `sets`/`train` stages end to end.

## Update 20 - first real 5090 runs: public-data students, simulation data, held-out comparison (2026-11-13/14)
- 13 Nov 16:14 `run_v0.sh` (8000 public images, 4 sources) on the 5090: download to 18:24, pseudo-labelling (Grounding DINO + SAM2) 19:00-21:29, teacher cache in 4 min (run in parallel while labelling), `vit` 20 epochs 49 s each (final loss 0.772, 12.2 GB), `hybrid` 54 s each (1.343). First estimate for labelling (40-80 min) was wrong: measured 1.49 s/image alone, 46 images/min with two workers (GPU saturated), 2 h 59 min in all.
- Simulation data (14 Nov 22:22-01:42): 30 layouts x 3 domains rendered with native Isaac Sim (indoor 38 s warm boot each, outdoor ~2.4 min, race ~1.5 min per layout), effects (single-threaded, ~16 images/s, the slow part) and conversion: `data/train_sim` indoor 906 + outdoor 28,550 + race 26,015 training images, 4 layouts per domain (26-29) held out in `data/heldout` (indoor 141, outdoor 5,160, race 3,075). Indoor is only 1.6% of the training set.
- Found and fixed: the outdoor generator failed for `rover` seed 1001 (a parked car or kerb tree stood on the fixed rover lane; 30 identical camera tries); now it drops the colliding items and retries; test added (51 tests).
- Baseline of the v0 students on held-out sim frames (zero-shot, `scripts/eval_students_sim.py`, one run): vit seg mIoU indoor 0.073, outdoor 0.387, race 0.301, gate IoU 0.000; depth AbsRel 0.36 / 0.72 / 1.44; hybrid clearly worse outdoors (0.264).
- Sim-trained `vit` (`configs/sim_5090.yaml`, 8 epochs of 394 s, loss 2.55 -> 1.15, 141 images/s) on the same held-out frames: seg mIoU indoor 0.809, outdoor 0.693, race 0.735, gate IoU 0.979, depth AbsRel 0.142 / 0.112 / 0.138; low light similar. Slot head (race, 513 frames, `scripts/eval_slots_sim.py`): gates fully in view 1,353, recall 95.1%, median corner error 3.9 px (256x192), mean 9.0; gates partly out of view only 46, recall 97.8%, median 5.0 px, mean 14.5; no false alarms on 36 gate-free frames.
- LIMITS: sim-to-sim (same generator, same asset catalogue, held-out layouts): shows the pipeline works end to end, NOT transfer to real images or real gates. Only 3% of the gates have a corner outside the image (46 of 1,399), so the case the slot head was built for is thin in the data; one run, no seeds.
- Next (prepared, starts after the hybrid run): close-pass racing set (`scripts/gen_race_close.py`, `scripts/race_close_pipeline.sh`): only frames within 3 m of a gate at 120 fps, offset up to 0.6 m (resolver limits), 30 layouts, 8,652 frames; then retrain and re-measure the partly-out-of-view gates.

## Update 21 - partly visible gates: why the first extra set did not help, and the tilt lever (2026-11-14)
- 14 Nov ~05:15: the close-pass racing set (`scripts/gen_race_close.py`, frames within 3 m of a gate, 120 fps, offset 0.6 m, tilt 15-35) did NOT raise the share of gates with a corner outside the image: 3.7% (normal set 2.7%). Reason (geometry, computed): the labelled corners are the INNER opening corners; with the 130 deg camera they leave the image only within ~0.6 m of the gate plane; what moves them out earlier is the camera UP-TILT t: the lower image edge is (58 - t) deg below the horizon, so a corner 0.75 m below the camera is out of view up to 0.75 / tan(58 - t) m (1.8 m at 35 deg, 5.3 m at 50 deg).
- The second training on the close set was stopped after ~5 minutes of caching (its premise did not hold). New set `scripts/gen_race_partial.py` / `scripts/race_partial_pipeline.sh`: tilt 40-50 (resolver limit 50), speed 4-12 m/s at 120 fps, frames within 3 m, offset up to 0.6 m. Measured on 3 test layouts before training: partly-visible gates 18.2% (495 of 2727) versus 2.7% (close-pass) and 1.8% (normal). 14 layouts used (22,412 training images, 4,150 held out), close-pass set parked out of the training folder; training set 77.9k images.
- `sim_v2` = same recipe as `sim_v1` (8 epochs, batch 32, ~550 s per epoch, loss 2.39 -> 1.06) plus the partial set. Held-out comparison for `vit` (one run, sim-to-sim): gates partly out of view on the high-tilt held-out frames (358): median corner error 25.4 -> 7.5 px (mean 31.2 -> 19.7); normal racing frames (81 gates): 5.7 -> 6.3 (no change); close-pass frames (112): 4.1 -> 6.5 (slightly worse, small n); fully visible gates on the high-tilt frames: median 14.2 -> 8.4 px but recall 91.2% -> 87.1% (cause not investigated); false alarms on gate-free frames 13 -> 1; segmentation and depth on the other domains unchanged (race mIoU 0.735 -> 0.736, gate IoU 0.979 -> 0.981, outdoor 0.693 -> 0.691, indoor 0.809 -> 0.810); depth AbsRel on the high-tilt frames 0.474 -> 0.356.
- CONFOUND AND LIMITS: the v1 failure on the high-tilt frames mixes two causes (partly visible gate and a camera-tilt distribution shift) that I cannot separate; the real tilt of the target drone is unknown, and tilt 40-50 may be outside it. Sim-to-sim only. The `hybrid` v2 run was still training when this was written (expected ~09:00).
- `hybrid` v2 (finished 14 Nov 09:07, loss 1.32 at epoch 6, ~606 s per epoch) on the same held-out frames, one run: gates partly out of view on the high-tilt frames (358) median corner error 25.0 -> 8.9 px (mean 35.6 -> 21.3); fully visible gates there median 16.3 -> 10.2 px, recall 91.9% -> 89.7%; normal racing frames (81 partly-out gates) 12.6 -> 14.0 px (slightly worse), close-pass frames 7.2 -> 7.6 (same); false alarms 15 -> 4; race mIoU 0.720 -> 0.729, gate IoU 0.976 -> 0.978; outdoor 0.676 -> 0.678; indoor 0.765 -> 0.763; depth AbsRel on the high-tilt frames 0.563 -> 0.381. The `hybrid` is clearly weaker than `vit` on gates cut by the image border (median 12.6-14.0 px against 5.7-6.3 px on the normal racing frames).
- Next run (14 Nov ~09:15): `vit` with 16 epochs on the same data (`configs/sim_v3_5090.yaml`, `runs/sim_v3`) to see whether training length limits the result (the loss was still falling at epoch 7).
- `sim_v3` (14 Nov 09:09-11:41): `vit` with 16 epochs (twice `sim_v2`, same data, 567 s per epoch, final loss 0.983 against 1.056) on the same held-out frames: partly-out-of-view gates on the high-tilt frames median 7.5 -> 8.5 px, normal racing frames 6.3 -> 6.6, close-pass frames 6.5 -> 5.3; fully visible gates on the high-tilt frames 8.4 -> 8.3 px; race mIoU 0.736 -> 0.759 (mostly `other_obstacle` 0.23 -> 0.30), gate IoU 0.981 -> 0.983, outdoor 0.691 -> 0.697, indoor 0.810 -> 0.813; depth AbsRel slightly worse (race 0.134 -> 0.140, indoor 0.135 -> 0.153). Twice the epochs changed the held-out numbers by about 1 px or less (within noise): the result is saturated for this data and recipe; data variety (the partial-view set), not training length, is the lever. One run, sim-to-sim.
- Housekeeping (14 Nov 10:12-10:17): disk 450 GB -> 228 GB used: effects copies, parked close-pass set (15 GB), its labels and cache, raw render outputs (38 GB), v0 cache, finished resume checkpoints, and 272,172 byte-identical label files (the 5 effect variants of a frame share one label) turned into hard links (44.8 GB). Per training image now about 1.8 MB (cache + image + label). Raw renders can be regenerated from `runs/*_job/layouts` (about 4.5 h of rendering plus effects).
- 5090 tools (14 Nov 10:20-11:08): `qwen3-vl:8b-instruct` pulled into Ollama (6.2 GB); Hunyuan3D-2mini environment installed in `~/Projects/i23d` (smoke asset ok, 4.35 GB GPU); the LLM layout test (`day1_5090.sh llm`, 50 scenarios) and the catalogue-growth loop on the 5090 are still to run.

## Update 22 - layout-LLM tests on the 5090, auto placement, unattended chain (2026-11-14)
- Layout-LLM validity (50 scenarios, 2 repair rounds, one run each, Ollama, `scripts/llm_select_test.py`): `qwen3-vl:8b-instruct` indoor 6/7, outdoor 1/1, race 4/7 (first try / after repair), 15 of 50, 2-19 s per layout; `qwen3:32b` indoor 5/5, outdoor 1/1, race 9/12, 18 of 50, 11-53 s. The big model helps only for racing tracks. 8B rerun with the outdoor `auto` schema: 16 of 50 but the model never wrote `auto` (0 of 311 outdoor items), it copied the numeric few-shot example; run-to-run noise on the unchanged domains is about +-1.
- New in the resolver: `"s": "auto"` for road items (`layout_out._auto_s`: 60 seeded candidates, same rules as the explicit path, "no free spot" error when the lane is full), JSON schema accepts it, the system prompt now says to use it and the few-shot example shows it (`_auto_example`), `ONLY_DOMAINS` filter for the test script; 52 tests. Whether the 8B/32B models now use it and how many outdoor layouts become valid is being measured by the unattended chain below.
- 14 Nov 12:58: unattended chain `scripts/overnight_5090.sh` started in tmux `overnight` (needs no Claude, deadline 22:00): eval of sim_v2 as seed 0, `vit` seeds 1 and 2 (8 epochs each) with held-out evaluation, LLM tests with the new prompt (32B outdoor, 8B all), catalogue growth on 3 real Aeroscapes photos, `hybrid` seed 1, then `~/overnight_summary.md` (mean and sd over seeds; `scripts/summarize_seeds.py`). Status: `ssh rtx5090 cat ~/overnight_status.txt`. Purpose: noise bars for every comparison made so far (all of them were single runs).
- Unattended chain, first results (14 Nov 16:30, one run each): `vit` seed 1 finished with final loss 1.058 (seed 0: 1.056); layout-LLM tests with the new prompt, outdoor: `qwen3-vl:8b-instruct` 0/15 valid (168 items written with `auto`, 95 with [x, y]; errors: lane full "no free spot" x29, camera looks more than 80 deg down x10, building on the carriageway x6), `qwen3:32b` 1/15 (296 `auto`, 46 xy; errors: `road: "none" does not exist` x42 in roadless scenes, "no free spot" x10, path gives too many frames x8); 8B all domains: indoor 5/20, outdoor 0/15, race 9/15. So the models now use `auto`, the remaining failures are new: requests for more trees than a kerb holds, and no `auto` form for scenes without roads (the prompt forces the road form: my mistake). Next: an `{"area": "auto"}` form for roadless scenes, a prompt line that says when to use which, and a capacity hint ("a kerb holds about N trees").
- Catalogue growth on the 5090, first run (3 real Aeroscapes frames): no new objects on 2, on the 3rd the VLM found a bicycle and a car, the bicycle mesh was generated (`runs/grow_5090/gen_bicycle_*`) but the gate crashed: `trimesh` was missing in the project env of the 5090 (installed now and added to `setup_5090.sh`); the chain step reported exit 0 because the loop's pipe masked the error. A rerun on that frame (`runs/grow_5090_b`) is queued to start when the chain ends.

## Update 23 - seed noise: the unattended chain finished (2026-11-14 17:22, read 15 Nov 00:00)
- Chain `scripts/overnight_5090.sh` ran 12:58-17:22 without problems (every step exit 0; the `grow` step's exit code was misleading, see below). `vit` seeds 0, 1, 2 (`sim_v2`, 8 epochs) and `hybrid` seeds 0, 1 evaluated on the held-out sim frames (`scripts/summarize_seeds.py`; seed 1 of `vit` re-evaluated under tag `s1v` because the `hybrid` seed 1 evaluation overwrote the files of tag `s1`: my tag collision, the same one as before).
- `vit`, mean +- sd over 3 seeds: partly-out gates median corner error 7.6 +- 0.4 px (high tilt), 6.8 +- 0.5 (normal racing), 6.4 +- 0.6 (close pass); fully visible gates on the high-tilt frames 8.2 +- 0.2 px and recall 0.874 +- 0.009; race gate IoU 0.981 +- 0.000, seg mIoU 0.736 +- 0.001; outdoor seg mIoU 0.692 +- 0.001; indoor 0.824 +- 0.013; depth AbsRel race 0.134, high tilt 0.350, outdoor 0.119, indoor 0.135. `hybrid` (2 seeds): partly-out gates median 8.5 / 14.5 / 8.0 px (high tilt / normal / close pass), race seg mIoU 0.725, outdoor 0.678.
- Reading with noise bars: the v1 -> v2 gain on the high-tilt partly-out gates (25.4 -> 7.6 px) is far above the noise; the drop of the fully-visible recall there (0.912 -> 0.874) is about four sd, so also real; most single-run differences of about 1 px quoted earlier are within noise (16 epochs against 8: no change, except slightly worse on the high-tilt partly-out gates, 8.5 against 7.6 +- 0.4); the possible small cost of the tilted data on the normal racing frames (5.7 for v1 against 6.8 +- 0.5 for v2) rests on a single v1 run.
- Catalogue growth on the 5090 (real Aeroscapes photos, VLM -> SAM2 -> Hunyuan -> gate), end to end after installing `trimesh`: a bicycle seen from above was generated and REJECTED by the gate (two-seed consistency 0.782 < 0.93); two other frames had no new objects. The VLM's `traffic sign pole` was tagged `police_car` in one run and absent in the next (no matching tag exists).

## Update 24 (2026-11-15): stress test in sim, first real-image check, outdoor LLM area-auto
- **Stress test on 257 held-out sim racing frames** (`scripts/eval_robustness_sim.py`, `runs/eval/robustness_sim_race.json`; effects library applied to clear frames, vit trained on `sim_v2`). Gate IoU clear 0.981; forward blur 10/30/60 m/s 0.980/0.976/0.969; turning blur up to 2 rad/s 0.980; light -1/-2/-3/-4 stops 0.979/0.974/0.969/0.960; fog 0.05 0.974 (gate recall 0.96 -> 0.87); rain 0.975 (recall 0.89); JPEG q15 0.975; **+1 stop over-exposure 0.422 (recall 0.55)**; worst mix (30 m/s + dark + fog) 0.948 with recall 0.68. **Hybrid collapses in the dark** (gate recall 0.37 at -3 stops, 0.12 at -4) and drops on over-exposure (0.775). Blur robustness is sim-blur only. Training never saw over-exposure.
- **Real-image check** (`data/real_tii/golden`, 102 frames, 206 labelled gates, from the TII Racing dataset, CC BY 4.0, labels semi-automatic then human reviewed by the authors; user looked at the overlay sheets and said they are fine; I flagged 5 frames in `review_flags.json`). Indoor hall, fisheye, camera often pointing up, purple gates. **Result: both sim-trained students find no gate at all** (max slot score 0.02 vit, 0.08 hybrid; gate pixel share about 0.1%). Preprocessing is identical to the sim evaluation, so this is a real sim-to-real failure, not a bug. Script `scripts/eval_real_golden.py`, `runs/eval/real_golden_*.json` (5090).
- Meaning: all earlier numbers (0.98 IoU, recall 0.96) are sim-to-sim only; transfer to real gates is currently zero on this set. Next: gate appearance randomisation (colour, texture, frame style), fisheye/lens randomisation, and the real-fraction fine-tune from decision 9 using TII flights split by flight (train/test on different flights).
- Outdoor LLM, `auto` area placement (15 scenarios): qwen3:32b 2/15 valid, qwen3-vl:8b 1/15; dominant error "no free spot along road for lane kerb" (models overfill kerbs with trees); also camera path/frame count and camera too close to an item.

## Update 25 (2026-11-15): first real-image transfer numbers (vit, TII Racing, equal compute)
- Setup (`scripts/real_mix_chain.py`, `scripts/real_train_prep.py`): N=3000 frames per run, 16 epochs, vit ImageNet start; real = frames from 12 TII flights NOT in the golden set (only gates with all 4 corners labelled; 942 available), sim = random `train_sim` frames. Golden = 102 real frames from 6 other flights (206 gates, 123 partly visible), same hall and track as training, different flights.
- Result (corner error under 20 px at 256x192): real 0% (300/900 real frames: 0 / 300 / 900) -> recall 0.00 / 0.23 / 0.26; under 10 px 0.00 / 0.10 / 0.16; median error about 40 px; partly visible gates under 20 px only 0.07. Gate detection score threshold 0.5; unmatched predictions 0.09 to 0.2 per frame.
- No forgetting in sim: held-out sim_racec full-view recall 0.947 / 0.956 / 0.947, partly-out 0.964 / 0.973 / 0.964.
- Meaning: a few hundred real frames turn "finds nothing" into "finds about a quarter of the gates within 20 px"; 900 is only slightly better than 300, so more of the same data is not the lever. Still far from usable. Not measured: other halls, DroneRacing's camera, fisheye randomisation in sim, gate appearance randomisation, ONNX/Orin speed. Single seed.

## Update 26 (2026-11-15): sim appearance + fisheye augmentation did not open real transfer
- `scripts/make_aug_sim.py`: existing sim racing frames (clear/blur/lowlight) with gate pixels recoloured (purple, black, orange, red, blue, green...) and a barrel/fisheye warp applied consistently to image, seg, depth and gate corners (1,500 frames, checked by eye). `scripts/real_aug_chain.py`: N=3000, 16 epochs, half the sim share replaced by these frames.
- Golden (102 real TII frames, other flights): 0 real frames + augmentation: recall<20px **0.00** (no gate found at all, same as plain sim); 300 real + augmentation: 0.223 (<10px 0.092, <40px 0.515) against 0.233 without augmentation. No gain, within noise. Sim held-out recall unchanged (0.942-0.944 full view, 0.973 partly out).
- Reading: gate colour and lens shape are not what separates sim from this real hall. Candidates left: background/scene appearance and lighting of the hall, gate texture (checker markers, logos), thin-bar geometry, real-image noise/compression statistics, partly visible gates (7% recall, no label for the hidden corners). Single seed.

## Update 27 (2026-11-15): the "real frame = gate" shortcut, and negatives
- Finding: students trained with real positives only (Update 25/26: real_mix_f10, f30, aug_r300) fire a gate on 94-95% of golden frames whose gate boxes are painted over, and on 15-33% of real Aeroscapes photos (sim-only vit: 2% and 1%). Their slot score is 1.0 on nearly every real frame, and their gate segmentation marks ceilings and walls. So the 0.23-0.26 recall of Update 25 is partly "always say gate", not recognition. Cause: every real training frame contained a gate. TII has almost no gate-free frames (1 in 18 flights), so negatives had to be made.
- Fix (`scripts/make_real_negatives.py`, `scripts/real_neg_chain.py`): 608 real gate-free crops from training-flight frames (4:3 crops that avoid every labelled gate, expanded 40%), empty slot target; eval negatives: 166 crops from the golden frames' flights. Equal compute N=3000.
- Result: golden recall<20px 0.223 (300 pos + 150 neg) and 0.257 (900 pos + 450 neg), unchanged from before; false gate on real gate-free crops 0.00 and on Aeroscapes 0.00 for both; painted-out golden frames still fire 0.12 (r300) and 0.74 (r900) (the painted grey box and leftover stands/bars are out of distribution, so unclear); sim held-out recall 0.958/0.941 full view, 0.973/0.955 partly out.
- Reading: gate-free crops are easy negatives (walls, ceiling), so 0.00 false alarms there proves little; r300 is the cleaner model. Real recall stays about a quarter within 20 px, partly visible gates 7%. Single seed. Needs: harder negatives (hall views with gate-like structures), real labels for partly visible gates, a second hall.

## Update 28 (2026-11-15): partly visible real gates, and the cross-track check that shrinks the result
- Change: real labels with hidden corners kept (hidden corner = NaN in the slot target, skipped by `slot_loss`; frames with a gate that has exactly one known corner skipped; `scripts/real_train_prep_partial.py`, 5,104 frames, 11,269 gates, 67% partly visible; unit test added, 54 tests pass). Plus real gate-free crops. N=3000, 16 epochs, vit.
- Same-track result (golden = 102 frames from other flights, SAME hall and gate layout), corner error under 20 px: 300 real frames 0.689 (partly visible gates 0.829); 900 real frames 0.816 (full 0.723, partly 0.878, under 10 px 0.597). Before this change 0.26 (partly 0.07). Sim held-out recall unchanged (0.95 full view, 0.96-0.98 partly out). False gate: 0.00 on real gate-free crops, 0.01-0.06 on Aeroscapes, 0.81-0.84 on golden frames with gate boxes painted over (cause unclear: painted box looks gate-like, or leftover structure).
- **Cross-track check** (`scripts/real_xtrack_chain.py`): train only on ellipse/lemniscate flights (02,03,04,06,07,08,10,11,12; 900 positives + 300 negatives), test on trackRATM golden frames (flights 13,16,18; 66 frames, 140 gates, other gate layout): under 20 px **0.114**, under 10 px 0.029, under 40 px 0.50, unmatched predictions 0.48 per frame. So most of the 0.82 above is the model learning this hall and track, not gates in general. Honest transfer number to a new layout in the same hall: about 0.11 (20 px).
- Open: only one hall in TII; reverse direction (train trackRATM, test ellipse) not run; single seed; painted-out false alarms unresolved. Needs DroneRacing camera frames and another environment.

## Update 29 (2026-11-15): cross-track repeat and reverse direction
- Seed 2 of ellipse/lemniscate -> trackRATM (66 frames): under 20 px 0.114 (seed 1: 0.114), under 10 px 0.036, under 40 px 0.436 (seed 1: 0.50), unmatched 0.45 per frame. The 0.11 is stable across two seeds.
- Reverse (train trackRATM flights 14,15,17; test ellipse/lemniscate golden frames of flights 01,05,09; 36 frames): under 20 px 0.212, under 10 px 0.000, under 40 px 0.803, unmatched 0.81 per frame (many extra predictions). Gates are found roughly (within 40 px) but not precisely, and the model over-predicts.
- Reading: transfer to a new gate layout within one hall is about 0.11-0.21 within 20 px and 0.44-0.80 within 40 px, with many unmatched predictions. Still not usable; 10 px precision is near zero. Small test sets (36 and 66 frames).

## Update 30 (2026-11-15): score threshold and finer slot grid on the cross-track tests
- Threshold sweep (no training; slot score 0.5 / 0.9 / 0.99 / 0.999), ellipse/lemniscate -> trackRATM: under 20 px 0.114 / 0.107 / 0.100 / 0.050, unmatched predictions per frame 0.48 / 0.32 / 0.09 / 0.00; reverse direction: 0.212 / 0.152 / 0.045 / 0.015, unmatched 0.81 / 0.56 / 0.14 / 0.00. Threshold 0.99 removes most false predictions at almost no cost in one direction; the bottleneck is corner precision, not detection.
- Finer slot head (`slot_grid` config key, env PFM_SLOT_GRID; 14x14 grid + 3x3 conv before the MLP, 6.5 M parameters; default 6 keeps old checkpoints): ellipse/lemniscate -> trackRATM under 20 px 0.221 (6x6: 0.114, 0.114 on two seeds), reverse 0.091 (6x6: 0.212). One seed each, 66 and 36 test frames: mixed, no clear gain. At threshold 0.99: 0.186 and 0.061.
- Reading: on TII alone the cross-track transfer sits at roughly 0.1-0.2 within 20 px and 0.5-0.7 within 40 px whatever the head; the test sets are too small to rank the two heads. Further runs on this single hall do not answer the open question; a second environment or DroneRacing camera frames do.

## Update 31 (2026-11-15): light augmentation closes the over-exposure and deep-dark gaps (vit, sim stress test)
- `scripts/make_light_aug_sim.py` (1,500 clear race frames through `effects.pipeline`: ev +0.5..+2 for 40%, ev -4..-1.5 with gain/iso for 60%, labels reused), `scripts/light_chain.py`. vit, N=3000, 16 epochs, sim only; light_a = 1,500 sim + 1,500 light frames; baseline = same recipe with 3,000 plain sim frames. Stress test on 257 held-out race frames (gate IoU / gate recall):
  - clear 0.955 / 0.96 -> 0.952 / 0.96 (no cost)
  - **+1 stop 0.830 / 0.57 -> 0.936 / 0.91**
  - -3 stops 0.941 / 0.94 -> 0.950 / 0.95; **-4 stops 0.891 / 0.84 -> 0.947 / 0.91** (corner error 13.4 -> 8.8 px)
  - fog 0.05 0.938 / 0.83 -> 0.941 / 0.83 (unchanged); worst mix (30 m/s + dark + fog) 0.915 / 0.56 -> 0.918 / 0.68
- Reading: the over-exposure and darkness gaps are training-coverage gaps and close with light-varied frames; fog recall (0.83) and the worst mix remain open. Absolute numbers of this N=3000 recipe are below the full sim_v2 (clear IoU 0.955 vs 0.981). Single seed. Sim-to-sim only; real-camera exposure behaviour is not covered.

## Update 32 (2026-11-15): the hybrid dark collapse is also a coverage gap
- Same comparison for the hybrid (FastViT, N=3000, 16 epochs, sim only; gate IoU / gate recall): clear 0.948 / 0.96 -> 0.949 / 0.97; **-1 stop 0.933 / 0.32 -> 0.950 / 0.97; -2 stops 0.913 / 0.14 -> 0.950 / 0.96; -3 stops 0.788 / 0.01 -> 0.948 / 0.95; -4 stops 0.412 / 0.00 -> 0.944 / 0.92; +1 stop 0.711 / 0.82 -> 0.938 / 0.95**; fog 0.05 0.935 / 0.94 -> 0.948 / 0.87; worst mix 0.898 / 0.58 -> 0.888 / 0.56. The earlier sim_v2 hybrid collapse in the dark came from missing exposure range in training, not from the architecture. The baseline hybrid here collapses already at -1 stop (recall 0.32); the earlier sim_v2 hybrid was better at -1 (0.95) because it saw 26k frames with a low-light condition.
- Open: fog recall and the worst mix (0.56-0.68) for both backbones. Next: full retrain with light frames added (sim_v4).

## Update 33 (2026-11-15): sim_v4 (full sim set + 6,000 light-varied race frames), vit and hybrid, 8 epochs
- Stress test on 257 held-out race frames (gate IoU / gate recall; vit | hybrid), `runs/eval/robustness_sim_race_sim_v4.json`: clear 0.981 / 0.96 | 0.978 / 0.97 (same as sim_v2 0.981 | 0.978); forward blur 30 m/s 0.976 / 0.95 | 0.974 / 0.97; **+1 stop 0.967 / 0.94 | 0.968 / 0.96** (sim_v2: 0.422 / 0.55 | 0.775 / 0.62); -3 stops 0.979 / 0.94 | 0.977 / 0.96 (sim_v2 hybrid 0.890 / 0.37); **-4 stops 0.976 / 0.93 | 0.974 / 0.95** (sim_v2 hybrid 0.772 / 0.12); fog 0.05 0.973 / 0.87 | 0.976 / 0.89; rain 0.976 / 0.91 | 0.977 / 0.94; worst mix 0.932 / 0.75 | 0.952 / 0.69 (sim_v2 0.948 / 0.68 | 0.943 / 0.53). The light gaps are closed on the full recipe too; the worst mix is still the weakest point.
- Held-out sim_racec slot head (1,110 frames): vit full-view recall 0.923 (median corner error 4.6 px), partly out of view 0.991 (6.4 px); hybrid 0.934 full view (sim_v2 hybrid: 0.943 / 5.4 px), partly out 0.982 (sim_v2 hybrid 0.982 / 7.6 px). Indoor seg mIoU vit 0.823, outdoor vit 0.688; sim_v2 files for the same sets were overwritten by the later hybrid run, so no like-for-like seg comparison for vit. Single seed each.
- Not covered: real-camera exposure, real gates (Update 28: about 0.1-0.2 within 20 px on a new layout). sim_v4 has no real data in it.

## Update 34 (2026-11-15): combined-degradation augmentation (fog, blur, rain, light, JPEG together), vit N=3000
- `scripts/make_mix_aug_sim.py`, `scripts/mix_chain.py`: mix_a = 1,500 sim + 1,500 race frames with random combinations (forward blur 15-50 m/s 70%, fog 60%, rain 30%, ev -4..-1 with gain 45% or +0.5..+1.5 15%, JPEG 30%). Stress test vs light_a (gate IoU / recall): clear 0.952 / 0.96 -> 0.950 / 0.95 (corner error 10.2 -> 11.1 px); fog 0.02 0.947 / 0.89 -> 0.949 / 0.93; **fog 0.05 0.941 / 0.83 -> 0.948 / 0.88**; rain 0.946 / 0.90 -> 0.949 / 0.92; **worst mix 0.918 / 0.68 -> 0.932 / 0.77**; +1 stop 0.936 / 0.91 -> 0.919 / 0.97; -4 stops 0.947 / 0.91 -> 0.947 / 0.92.
- Reading: fog and the worst mix improve moderately (recall +5 and +9 points) at a small cost in corner precision on clear frames; recall for the worst mix is still only 0.77. Single seed, N=3000 recipe, sim-to-sim. Not yet in a full retrain; sim_v4 (Update 33) has the light frames only.

## Update 35 (2026-11-15): second real golden set from other environments (MonoRaceGate) and zero-shot results
- Data: MonoRaceGate (TU Delft, MIT licence, github.com/tudelft/MonoRaceGate, release season1.zip 212 MB): real A2RLxDCL gates, undistorted images, precise inner corners TL,TR,BR,BL (same convention as our slots), 387 labelled images, 818 gates (79 with a corner outside the image), 7 sequences in 4 places (ADNEC, DHL, marina, "red"). Converted by `scripts/mrg_prepare.py` into `data/real_mrg/golden` (review sheets PDF sent to the user, label overlays checked by eye on 12 frames: precise). `scripts/eval_real_golden.py` takes `GOLDEN=` now.
- Zero-shot on these 818 gates (corner error under 10 / 20 / 40 px at 256x192): sim_v2 vit 0 / 0 / 0, sim_v2 hybrid 0 / 0 / 0.07, sim_v4 vit 0 / 0 / 0 (0.02 predictions per frame), sim_v4 hybrid 0 / 0 / 0.04: **sim-only models find essentially no real gate in new environments, light augmentation did not change that**. Models trained with TII real frames (no MonoRaceGate frame seen): neg_r300 0.038 / **0.378** / 0.776; xtrack_r900 0.034 / 0.278 / 0.752; part_r900 0.017 / 0.249 / 0.764; predictions per frame 2.2 against about 2.1 labelled, unmatched 0.5 per frame; partly visible gates under 20 px only 0.11-0.17 (n=79).
- Reading: real frames from one hall transfer to other halls and other cameras better than to a new layout inside the hall (0.25-0.38 vs 0.11-0.21 within 20 px), because gate type and look are shared, but precision is still poor (10 px about 0.03). MonoRaceGate is a much better test of transfer than TII cross-track. Single seed per model. Not yet trained on MonoRaceGate.

## Update 36 (2026-11-15): MonoRaceGate in the training mix, leave-one-place-out (vit, N=3000, 16 epochs)
- `scripts/mrg_train_prep.py`, `scripts/mrg_chain.py`. Mix: TII partly visible positives 450 + TII gate-free crops 150 + MonoRaceGate positives and gate-free crops of the training places + sim for the rest; no frame of the test place seen. Split A: train ADNEC+DHL+marina (191 positives, 90 crops, 2,119 sim), test on red (196 images, 312 gates); split B: train red+DHL+marina (263, 181, 1,956 sim), test on ADNEC (124 images, 375 gates).
- Held-out place, corner error under 10 / 20 / 40 px at 256x192: red 0.141 / **0.571** / 0.872 (TII-only model neg_r300: 0.026 / 0.420 / 0.715), median error 18 px, unmatched predictions 0.54 per frame; ADNEC 0.219 / **0.587** / 0.901 (neg_r300: 0.059 / 0.411 / 0.864), median 16 px, unmatched 0.17 per frame. Partly visible gates under 20 px: 0.06 (n=34) and 0.31 (n=16), small samples. The TII golden set (same hall as training flights) with the same models: 0.655 / 0.587 under 20 px (under 10 px 0.41 / 0.40, partly visible 0.74 / 0.68). Sim held-out recall unchanged (0.95 full view, 0.96 partly out).
- Reading: one more real source (a few hundred frames from other places and camera) lifts transfer to a place never seen from 0.41-0.42 to 0.57-0.59 within 20 px and 10 px precision from 0.03-0.06 to 0.14-0.22. More diverse real data helps more than more of the same hall. Still single seed, small place-level test sets; DroneRacing's own camera not covered.
- Next: add general indoor/outdoor sets (NYU Depth v2, DIODE, ADE20K, SUN RGB-D; all downloadable from Hugging Face without login) for the depth and segmentation heads, with a class mapping to our 11 classes and teacher labels where missing.

## Update 37 (2026-11-15): general real depth data (DIODE, indoor + outdoor) in the mix
- Data: DIODE via Hugging Face `ahd03/diode-depth-3k` (3,000 frames, 58% indoor, 16-bit depth, 0 = invalid; `scripts/diode_prepare.py`: 2,700 train, 300 held-out frames picked as every 10th row of all shards, 102 indoor / 198 outdoor; `scripts/eval_depth_diode.py`: per-image scale+shift alignment, AbsRel). A first held-out split (last shard) turned out to be all outdoor and was replaced.
- Baselines, held-out DIODE AbsRel (indoor / outdoor): sim_v4 vit 0.292 / 0.509; mrgA 0.263 / 0.517; neg_r300 0.279 / 0.469.
- mrgA recipe + 600 DIODE frames, sim reduced to keep N=3000 (`scripts/mrg_d_chain.py`, run mrgA_d): **depth 0.147 / 0.244** (almost halved). Gates (split A, red held out): under 20 px 0.571 -> 0.481, under 10 px 0.141 -> 0.058, unmatched 0.54 -> 0.68; TII golden under 20 px 0.655 -> 0.592; sim held-out slot recall 0.942 full view, 0.982 partly out. 
- Reading: general real depth data fixes the depth head's real-world weakness but cost some gate precision in this equal-compute swap (600 sim frames replaced); one seed, differences of 0.05-0.09 on 196 images are at the edge of noise. Next: add DIODE on top (N=3,600) instead of replacing sim, then segmentation sets (SUN RGB-D, ADE20K).

## Update 38 (2026-11-15): seed noise of the place-held-out gate test; DIODE costs nothing within it
- Second seed of the mrgA recipe (no DIODE) and of mrgA + 600 DIODE frames on top (mrgA_d2): red held-out gates under 20 px: no DIODE 0.571 (seed 0) / 0.446 (seed 1); with DIODE 0.401 / 0.494 (and 0.481 in the sim-replacing variant); TII golden under 20 px: no DIODE 0.655 / 0.612, with DIODE 0.612 / 0.626 (0.592). So seed noise on the 196-image test is about +-0.06 and the DIODE differences are inside it (mean red 0.51 without, 0.45 with, a possible small cost not resolved).
- DIODE depth gain is reproducible: held-out AbsRel indoor / outdoor 0.263-0.271 / 0.497-0.517 without, 0.143-0.147 / 0.244-0.245 with (both seeds).
- Reading: general real depth data is worth keeping in the mix for depth; for gates no effect that this test can resolve. Gate test sets need to be larger (or more seeds) to rank recipes at the 0.05 level.

## Update 39 (2026-11-15): general indoor data (SUN RGB-D) in the mix: indoor segmentation and depth
- Data: SUN RGB-D via Hugging Face `wyrx/SUNRGBD_seg` (`scripts/sun_prepare.py`): 3,964 train frames, 300 held-out frames (every 4th of test shard 0). Label ids are the `id2label` index + 1 (0 = unlabelled), checked by overlay (1 = wall, 2 = floor, 22 = ceiling). Map to our classes: wall, ceiling, door, window, blinds -> building; floor, floor mat -> ground_road; person -> person; other furniture/props -> other_obstacle; unlabelled -> ignore. `scripts/eval_seg_sun.py` (IoU of the four mapped classes, pixel accuracy, depth AbsRel).
- Mix run mrgA_ds (`scripts/mrg_ds_chain.py`): split A (red held out) recipe + 600 DIODE + 600 SUN RGB-D frames on top of sim 2,119, N about 4,200, 16 epochs, vit, seed 0. Held-out SUN: seg mIoU **0.636** (building 0.828, ground_road 0.861, other_obstacle 0.816, person 0.038), pixel accuracy 0.907, depth AbsRel 0.107; before: sim_v4 0.209 / 0.432 / 0.260, neg_r300 0.307 / 0.576 / 0.263, mrgA_d2 0.349 / 0.633 / 0.190. DIODE held-out AbsRel 0.135 indoor / 0.242 outdoor. Gates: red held-out under 20 px 0.474 (earlier runs 0.40-0.57, noise +-0.06), TII golden 0.621, sim held-out slot recall 0.956 full view / 0.973 partly out.
- Reading: general real indoor data nearly doubles indoor segmentation and halves indoor depth error on a held-out real indoor set; gate results unchanged within noise. Person IoU stays near 0 (few person pixels, one class among furniture-heavy scenes). Outdoor segmentation on real data is still untrained (no real outdoor seg set in the mix yet). The train/held-out split is by frame, not by scene, so SUN numbers are probably optimistic (SUN has near-duplicate frames of the same room).

## Update 40 (2026-11-15): general real outdoor data (ADE20K) in the mix
- Data: ADE20K via Hugging Face `1aurent/ADE20K` (BSD): outdoor scenes only (scene tag), 3,000 train + 300 held-out validation frames (`scripts/ade_prepare.py`). Segmentation decode class = R//10*256+G (equals the `name_ndx` of the objects, checked). Name rules map to our classes (building/wall/house -> building; road/sidewalk/grass/earth/ground/field/sand -> ground_road; tree/palm/pole/streetlight -> tree_pole; car/truck/bus/van -> car; person; rock/stone -> rock; sky/water/mountain/plant -> ignored, like sky in the sim labels; rest -> other_obstacle). The first conversion found only 459 outdoor frames in shards 0-2 (shards are ordered by scene folder, indoor first); all ten shards were scanned instead.
- Run mrgA_dsa (`scripts/mrg_dsa_chain.py`): split A recipe + 600 DIODE + 600 SUN RGB-D + 600 ADE20K on top of sim 2,119, 16 epochs, vit, seed 0. Held-out ADE20K outdoor: seg mIoU **0.514** (tree_pole 0.794, person 0.575, ground_road 0.743, building 0.499, rock 0.569, other_obstacle 0.336, car 0.079), pixel accuracy 0.753; before: sim_v4 0.153 / 0.450, neg_r300 0.194 / 0.532, mrgA_ds 0.198 / 0.539. SUN held-out mIoU 0.643, depth AbsRel 0.107 (unchanged); DIODE held-out AbsRel 0.134 indoor / 0.253 outdoor.
- Cost side: sim_outdoor (held-out sim) vit seg mIoU 0.613 (sim_v4 vit 0.688; tree_pole 0.606 vs 0.666, car 0.512 vs 0.650, lane_marking 0.070 vs 0.208) with a much smaller sim share (2,119 frames against the full set), depth AbsRel 0.068; gates: red held-out under 20 px 0.413 (runs so far 0.40-0.57, seed noise +-0.06, mean drifting down as sources are added: 0.51 / 0.45 / 0.47 / 0.41), TII golden 0.621, sim held-out slot recall 0.947 / 0.982.
- Reading: general real outdoor data lifts real outdoor segmentation from about 0.15-0.20 to 0.51 mIoU; the sim classes lane_marking and car lose precision because few sim frames remain in this small equal-size recipe, and the gate transfer on red may lose a few points as sources are added (not resolved within noise). Car IoU on ADE20K stays low (0.08), probably a label-mapping or scale issue (cars are small there).
- Next: a larger consolidated recipe (20,000 sim frames + all real sources) with red held out, to see whether the sim classes and gates recover when the sim share is not starved.

## Update 41 (2026-11-15): consolidated recipe bigA (sim + all real sources, red held out), vit, 8 epochs, seed 0
- `scripts/big_chain.py`: 20,000 sim frames + 3,000 TII partly-visible positives + 608 TII gate-free crops + MonoRaceGate ADNEC/DHL/marina (191 positives, 90 crops) + all DIODE (2,700), SUN RGB-D (3,964), ADE20K outdoor (3,000); about 33,500 frames, 228 s per epoch, 8 epochs. Red place never seen.
- Gates on the held-out place red (312 gates; corner error under 10 / 20 / 40 px): 0.147 / **0.484** / 0.846, median 20.8 px, unmatched predictions 0.41 per frame (small recipe mrgA_dsa: 0.054 / 0.413 / 0.824; earlier runs 0.40-0.57 within seed noise). TII golden set (same hall as 3,000 training frames, other flights): 0.718 / 0.835 / 0.947, median 4.4 px, partly visible gates 0.878 within 20 px (an in-domain number, not a transfer number). Sim held-out slot head: recall 0.932 full view (median 5.1 px), 1.00 partly out (112 gates, median 9.8 px).
- General real sets: DIODE depth AbsRel 0.106 indoor / 0.210 outdoor; SUN RGB-D mIoU 0.738 (building 0.865, ground_road 0.877, other_obstacle 0.850, person 0.361), pixel accuracy 0.925, depth AbsRel 0.088; ADE20K outdoor mIoU 0.574 (tree_pole 0.819, person 0.619, ground_road 0.760, building 0.586, rock 0.627, car 0.243, other_obstacle 0.366), pixel accuracy 0.783. Sim outdoor held-out: mIoU 0.653 (car 0.636, tree_pole 0.654, building 0.929, ground_road 0.975, lane_marking 0.071), depth AbsRel 0.060.
- Reading: with a large sim share the sim classes (car) recover and every real set improves again; lane_marking stays low (0.07 against 0.21 in sim_v4) because none of the real sets labels lane markings. The honest transfer number for gates on a place never seen is still about 0.48 within 20 px and 0.15 within 10 px. SUN and ADE splits are by frame, probably optimistic. Single seed.
- Not done: no model has the light/blur/fog augmentation of sim_v4/v5 together with the real sources; no ONNX/TensorRT export; no DroneRacing flight test; DroneRacing's own camera frames missing.

## Update 42 (2026-11-15): bigB = bigA + light and combined-degradation frames (real sources and sim robustness in one model)
- bigB (`scripts/bigb_chain.py`): bigA mix + 6,000 light-varied + 6,000 combined-degradation race frames, about 45,500 frames, vit, 8 epochs, seed 0, red held out.
- Sim stress test (257 held-out race frames, gate IoU / recall), bigA -> bigB: clear 0.972 / 0.96 -> 0.975 / 0.95; **+1 stop 0.614 / 0.57 -> 0.964 / 0.94**; -4 stops 0.961 / 0.92 -> 0.971 / 0.93; fog 0.05 0.964 / 0.85 -> 0.973 / 0.92; rain 0.963 / 0.90 -> 0.974 / 0.94; **worst mix (30 m/s + dark + fog) 0.940 / 0.68 -> 0.960 / 0.85** (best so far; sim_v4 was 0.932 / 0.75).
- Real sets unchanged within noise: red held-out gates under 10 / 20 / 40 px 0.103 / 0.484 / 0.792 (bigA 0.147 / 0.484 / 0.846); TII golden 0.684 / 0.816 / 0.947; DIODE AbsRel 0.113 / 0.218; SUN mIoU 0.733, depth 0.093; ADE outdoor mIoU 0.551 (car 0.137); sim outdoor mIoU 0.657 (lane_marking 0.111), depth 0.058; sim held-out slots 0.926 full view, 1.00 partly out.
- Reading: degradation augmentation and real data do not interfere; bigB is the best all-round model so far. Gate transfer to an unseen place is the open problem (0.48 within 20 px).
- Project review (code): three weaknesses found: (1) the slot head regresses all corners from a 6x6 global pool (coarse), (2) no data augmentation at all in training, (3) gate-free real sets are not used as negatives for the gate head. Implemented: dense gate head (centre heatmap + 8 corner offsets per stride-8 cell, CenterNet focal loss, decoded into the same K x 9 format; config `gate_head: dense`), augmentation (`aug: true`: zoom-in crop 1-1.6x, horizontal flip with corner-order swap, photometric jitter; labels, depth and cached teacher maps warped alike), `gate_free_prefixes`. Two new unit tests, 56 pass. Run bigC queued (`scripts/bigc_chain.py`).

## Update 43 (2026-11-15): bigC = dense gate head + augmentation + gate-free negatives: large jump on real gates
- bigC (`scripts/bigc_chain.py`): same data as bigB (red held out; Aeroscapes not yet in), `gate_head: dense`, `aug: true`, `gate_free_prefixes: [diode__, sun__, ade__]`, vit, 8 epochs, seed 0. The three changes were made together (no ablation yet).
- **Real gates on the place never seen (red, 196 images, 312 gates)**, corner error under 10 / 20 / 40 px: bigB 0.103 / 0.484 / 0.792 -> **bigC 0.846 / 0.923 / 0.968** (3x3 peak NMS: 0.859 / 0.936 / 0.974), median error 21.3 -> 5.7 px, unmatched predictions 0.49 -> 0.02 per frame, partly visible gates within 20 px 0.00 -> 0.74 (n=34). TII golden 0.684 / 0.816 -> 0.879 / 0.917, median 4.7 -> 1.9 px. False gates: 0.00 on real gate-free crops, 0.01 on golden frames with the gate boxes painted out (was 0.74-0.95 for earlier models), 0.00 on Aeroscapes photos.
- Sim gates needed a new metric: the old sim scripts matched slots by order (largest first), the dense head orders by score. New `pfm.evaluate.match_gates` (order-free, one-to-one) and `scripts/eval_gates_sim.py`; stress test patched the same way (gate found = matched within 20 px). With it (sim_racec full-view gates within 10 / 20 px, median error of matches): bigB 0.751 / 0.870, 4.3 px; bigC 0.568 / 0.615, 1.1 px (3x3 NMS at score 0.3: 0.748 / 0.783); partly out of view 0.598 / 0.812 -> 0.741 / 0.866; sim_racep full view 0.535 / 0.727 -> 0.721 / 0.738 (median 7.9 -> 0.8 px). Stress test recall (within 20 px) bigB 0.88 clear / 0.72 worst mix, bigC 0.74 / 0.60, error 3.6 -> 0.8 px.
- Diagnosis of the sim misses: the head responds at 93% of the labelled gate centres (score > 0.5); 70% of the sim gates are smaller than 8 px (far gates) and many sit within 3 grid cells of a larger gate (gates seen through gates), so peak NMS and the one-gate-per-cell limit drop them. 3x3 NMS is now the default (better on red as well). Next: stride-4 grid.
- General heads, bigB -> bigC: DIODE AbsRel 0.113 / 0.218 -> 0.125 / 0.239; SUN mIoU 0.733 -> 0.728; ADE outdoor 0.551 -> 0.541; sim outdoor 0.657 -> 0.638: slightly lower (augmentation at equal epochs; within about 0.01-0.02). Aeroscapes held-out (not trained on yet): mIoU 0.158.
- bigD started: + Aeroscapes (2,621 drone-view frames with labels), dense head at stride 4, seg class weights (person 2, car 2, lane_marking 4, opponent_drone 3), 10 epochs.

## Update 44 (2026-11-15): bigD (stride-4 dense head, Aeroscapes, class weights, 10 epochs) and what the "sim far-gate loss" really is
- bigD (`scripts/bigd_chain.py bigD "gdet_stride: 4" "epochs: 10" "seg_class_weights: {...}"`): bigC recipe + Aeroscapes (2,621 frames, also gate-free negatives) + dense head on a stride-4 grid + seg class weights (person 2, car 2, lane_marking 4, opponent_drone 3), vit, 10 epochs, seed 0, red held out. Several things changed together again; ablations are running (Update 45).
- **Real gates on the place never seen (red, 196 images, 312 gates)**, corner error under 10 / 20 / 40 px: bigB 0.103 / 0.484 / 0.792; bigC 0.859 / 0.936 / 0.974 (3x3 NMS); **bigD 0.939 / 0.978 / 0.984**, median error 3.6 px, partly visible gates within 20 px 0.88 (n=34), unmatched predictions 0.11 per frame (bigC 0.02). Single seed, seed noise on this set was +-0.06 for the old head. TII golden (same hall, not a transfer number): 0.874 / 0.913, median 1.8 px. False gates: 0.00 on real gate-free crops, 0.00 on painted-out golden frames, 0.00 on Aeroscapes photos (Aeroscapes training frames are now negatives, so this last check is no longer independent).
- The 0.11 unmatched predictions per frame were the same gate found twice (stride-4 peaks 3 cells apart). New decode step in `DenseHeads.decode_gdet` (`PFM_GDET_DEDUP`, default 0.25, 0 = old behaviour): among the 16 best peaks, one whose decoded corners lie within 0.25 x gate size (mean corner distance) of a better-scoring kept one is dropped; a small gate seen through a large one survives (unit test added, 57 pass). bigD on red with it: 0.923 / 0.978 / 0.984, **unmatched 0.01 per frame** (threshold 0.3: 0.923 / 0.984 / 0.990, unmatched 0.04). Cost: 5 gates fall from under 10 px to 10-20 px (the better-scoring duplicate is not always the more exact one), sim partly-out 0.893 -> 0.866 (3 gates).
- **Sim gates (held-out sim_racec, 1,110 frames, 2,832 fully visible gates; sim-to-sim, not a success rate)**, found within 10 / 20 px: bigB 0.751 / 0.870, bigC 0.713 / 0.738, bigD 0.778 / 0.803 (threshold 0.3: 0.859 with 0.09 unmatched per frame; bigB has 0.18 unmatched per frame); partly out of view (n=112) within 20 px 0.812 / 0.866 / 0.893; sim_racep full view 0.727 / 0.738 / 0.775, partly out (n=358) 0.894 for bigD. Median error of matches 4.3 / 1.1 / 0.7 px.
- **Diagnosis (`scripts/diag_gates_size.py`, `scripts/diag_gates_topk.py`, `runs/eval/diag_size_*.json`)**: (1) not a ranking artefact: frames have at most 4 labelled gates and top-32 peaks give the same result as top-4; (2) by gate size (larger side at 256x192), found within 20 px, bigB / bigC / bigD: under 4 px (n=993) 0.76 / 0.46 / 0.62; 4-8 px (n=1,079) 0.90 / 0.84 / 0.86; 8-16 px (n=409) 0.96 / 0.96 / 0.95; 16-32 px (n=71) 0.94 / 0.89 / 0.89; over 32 px (n=280) 1.00 / 0.99 / 1.00. So 73% of the "fully visible" sim gates are smaller than 8 px and the whole gap sits there; (3) "within 20 px" is a loose test for a 3 px gate. With a size-relative test (error below max(3 px, a quarter of the gate size)) the order flips: all fully visible gates bigB 0.35, bigC 0.67, bigD 0.76; under 4 px 0.17 / 0.37 / 0.56; 4-8 px 0.30 / 0.77 / 0.82; 8-16 px 0.47 / 0.92 / 0.93. The MLP head of bigB mostly guessed roughly where far gates are; the dense head localises them, and misses the tiny ones by low score (centre score > 0.5 at 82% of the under-4 px gates in bigD), not by NMS any more (stride 4 cut "score there but no detection" from 0.47 to 0.23 for under-4 px gates).
- Reading: the dense head is not weaker in sim; the earlier 20 px metric favoured coarse guesses. What remains is a confidence gap on gates under 8 px (far away, of little use for control) which a threshold of 0.3 mostly trades for 0.04-0.09 extra predictions per frame. No MLP+dense fusion needed.
- Stress test (257 held-out race frames, gate found within 20 px / error): clear 0.84 / 0.6 px, 30 m/s blur 0.83, +1 stop 0.76, -4 stops 0.81, fog 0.05 0.76, rain 0.79, worst mix 0.61 / 0.9 px (bigC 0.74 clear, 0.60 worst; bigB 0.88 / 0.72 with 3.6 px error). Same small-gate effect; worst mix is still the weak point.
- General heads, bigC -> bigD: DIODE AbsRel 0.125 / 0.239 -> 0.126 / 0.237; SUN mIoU 0.728 -> 0.697 (person 0.276), depth 0.101; ADE outdoor 0.541 -> 0.549 (car 0.269, was 0.137-0.243); **Aeroscapes held-out 0.158 -> 0.676** (now trained on its train split: person 0.546, car 0.756, opponent_drone 0.584, building 0.867); sim outdoor mIoU 0.638 -> 0.681 (**lane_marking 0.111 -> 0.258**, car 0.611); sim indoor 0.785. The SUN drop (0.03) is the one cost; cause not isolated (class weights or the extra outdoor share).
- Parallel training: two bigD-size runs fit on the 5090 together only with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and nothing else on the GPU (first attempt ran out of memory while an evaluation was running).

## Update 45 (2026-11-15): bigD second seed, no-augmentation ablation, speed and ONNX/TensorRT export, two label problems found (ADE car rule, no seg labels on real gate frames)
- **Second seed of bigD** (`bigD_s1`, seed 1, same recipe; all gate numbers from here on use the duplicate removal of Update 44): red held-out gates under 10 / 20 / 40 px **0.862 / 0.962 / 0.971** (seed 0: 0.923 / 0.978 / 0.984), median 3.8 px, partly visible 0.94 (n=34; seed 0: 0.88), unmatched 0.00 per frame. So two seeds give 0.96-0.98 within 20 px and 0.86-0.92 within 10 px; the spread is much smaller than the +-0.06 of the old MLP head, but it is still two seeds on one place of 196 images. TII golden 0.864 / 0.932. Sim sim_racec full view 0.780 / 0.799 (seed 0: 0.778 / 0.803), partly out within 20 px 0.830 (0.866). General heads: DIODE 0.122 / 0.234, SUN mIoU 0.752 (person 0.481 against 0.276 in seed 0), ADE 0.549, Aeroscapes 0.581 (opponent_drone 0.115 against 0.584 in seed 0), sim outdoor 0.682 (lane_marking 0.258 in both). Rare classes (person on SUN, opponent_drone on Aeroscapes) swing by 0.2-0.5 IoU between seeds: single-seed class IoUs for them mean little. False gates 0.00-0.03.
- **Ablation: no augmentation** (`bigD_noaug`, `aug: false`, rest identical, seed 0): red 0.824 / 0.901 / 0.907 (with augmentation 0.923 / 0.978 and 0.862 / 0.962 on two seeds), partly visible gates within 20 px **0.65** (0.88 and 0.94), median 4.7 px, unmatched 0.02; sim_racec full view 0.713 / 0.745, partly out 0.830; sim_racep 0.672 full, 0.804 partly out (bigD 0.775 / 0.894). General heads slightly better without augmentation: DIODE 0.114 / 0.219 (0.126 / 0.237), SUN 0.705 (0.697), sim outdoor 0.695 (0.681), ADE 0.544, Aeroscapes 0.679. Reading: augmentation (zoom-in crop, flip, photometric) is worth about 0.06-0.08 on the unseen place within 20 px and most of it on partly visible gates (the zoom-in crop makes exactly those), at a small cost for depth/segmentation at equal epochs. The dense head alone already gives 0.90 (the MLP-head ablation is running).
- **Speed, bigD vit, 448x448, batch 1, RTX 5090, GPU otherwise empty** (`scripts/bench_fps.py`, `runs/eval/bench_fps_bigD_vit_448.json`): PyTorch model only 6.2 ms fp32 / 2.64 ms fp16 (161 / 379 fps); end to end from a uint8 640x480 frame in RAM (upload, resize, normalise, network, gate decode, results on the CPU) 6.46 ms fp32 / **2.91 ms fp16 (343 fps)**. While two trainings were using the GPU the same end-to-end path took 15 ms (66 fps). 92.1 M parameters. These are desktop-GPU numbers; the Orin is not measured with this model.
- **ONNX export** (`scripts/export_onnx.py`; outputs depth, seg, gates (decoded in the graph: peak NMS + duplicate removal), gdet; ImageNet normalisation inside the graph; ViT position embedding resampled once to the export size): `runs/bigD/vit/model_448.onnx` (337 MB) against PyTorch on 12 red frames: largest output difference 5e-5, same gate set in 12/12 frames. fp16 file (`HALF=1`, 169 MB): same gate set 12/12, largest corner difference 0.03 px.
- **TensorRT 11.4** (`scripts/trt_build.py`; this version has no fp16 builder flag, networks are strongly typed, so fp16 comes from the fp16 ONNX): fp16 engine 173 MB, built in 36 s, median 1.43 ms (about 700 fps, network only, input on the GPU); fp32 engine 3.65 ms. Timed while one training was running on the same GPU (p95 5.7 / 9.9 ms), so the medians are a lower bound of what an empty GPU gives. Against the fp32 ONNX on 24 red frames: same gate set 24/24, largest corner difference 0.28 px (fp16), seg class equal on at least 99.86% of the pixels. For the Orin (older TensorRT) use the fp32 ONNX with `trtexec --fp16` (`scripts/orin_trt_bench.py`).
- **Demo videos** (`scripts/make_demo_video.py`, `runs/eval/video/`: gates, class map, depth side by side; TII flight 13 race lap at real speed, red frames). User feedback on the first clip: it looked like flying backwards; that clip was the take-off (frames 1500-2700 of flight 13), the lap is frames 2560-4760. The videos showed a real fault: **the segmentation head paints hall ceilings as "gate"**. Cause: the real gate frames (TII, MonoRaceGate) have corner labels only, no segmentation label, so the seg head never saw a real race hall. Measured with weak labels from the corners (`src/pfm/gate_seg.py`, `scripts/eval_gate_seg_real.py`): share of the surely-not-gate pixels painted as gate: bigB 0.35 on red / 0.56 on TII golden, bigD 0.17 / 0.47 (over 1% of such pixels in 89-98% of the frames); gate band recall 0.94 / 0.89. The corner (dense) head is not affected. Fix prepared for bigE: weak seg labels for the 6,315 real gate frames and gate-free crops (`scripts/make_real_gate_seg.py`: thin band outside the opening = gate, far from every gate and inside openings = "not gate" (value 254), halo = ignored) and a `not_gate_loss` in `train.py` (-log(1 - P(gate)) on the 254 pixels; `notgate_weight`, default 1). Two new unit tests, 58 pass. Also visible in the video: a gate passed very close with only one bar in view is missed until more of it enters the frame.
- **ADE20K mapping bug** (found when asking why car IoU on ADE is 0.14-0.27): the name rule `^car` also matched carousel, caravan, cart, cardboard (carousel had more pixels than real cars in the held-out set), `^bus` matched bus stop, and snow / ski slope / pitch / court went to other_obstacle. `scripts/ade_relabel_v2.py` writes corrected labels to new folders (`data/general/train/labels_v2/seg`, `data/general/eval_ade2`); old files stay. The ADE "outdoor" subset is mostly nature (cars are 1.5% of the labelled pixels in 19 of 150 frames), so ADE car IoU will stay a noisy number. All ADE numbers up to bigD were trained and scored with the v1 labels.
- **BDD100K** segmentation subset (7,000 train / 1,000 val dashcam frames, unofficial HF mirror `dronefreak/BDD100K-Semantic-Segmentation`; **BDD100K data licence: research / non-commercial**) prepared for bigE (`scripts/bdd_prepare.py`): real road scenes with car, person, road, building, pole. BDD labels lane paint as road; road pixels much brighter than their surroundings are set to "ignored" so the model is not taught that lane paint is plain road. A Mapillary Vistas mirror was looked at and dropped (lossy blended label images). Held-out negatives for the false-gate check now also include ADE held-out and BDD val frames (independent of the training negatives).
- Queue: bigD_mlp (dense head off) and bigD_noneg (no gate-free negatives) ablations; bigE = bigD + BDD + ADE v2 labels + weak gate seg labels (`scripts/bige_chain.py`), then the hybrid student with the same recipe.

## Update 46 (2026-11-15): ablation of the three bigC/bigD changes (dense head, augmentation, gate-free negatives)
- All runs: bigD data and recipe (vit, 10 epochs, seed 0 unless stated), one switch changed each; red = MonoRaceGate place never trained on (196 images, 312 gates, 34 partly visible); duplicate removal on for the dense head. Corner error under 10 / 20 px, partly visible gates within 20 px, unmatched predictions per frame:
  - full recipe (bigD seed 0 / seed 1): 0.923 / 0.978, partly 0.88, unmatched 0.01 | 0.862 / 0.962, partly 0.94, 0.00
  - **no dense head** (`bigD_mlp`: MLP slot head, augmentation and negatives kept): **0.260 / 0.702**, partly 0.24, median error 15.4 px, unmatched 0.33
  - no augmentation (`bigD_noaug`): 0.824 / 0.901, partly 0.65, unmatched 0.02
  - no gate-free negatives (`bigD_noneg`: DIODE/SUN/ADE/Aeroscapes frames not used as empty gate targets): 0.920 / 0.987, partly 0.94, unmatched 0.01
- Sim, held-out sim_racec fully visible gates (sim-to-sim), within 20 px / size-relative strict test (Update 44): full 0.803 / 0.763 and 0.799 / 0.766; no dense head 0.886 / 0.450; no augmentation 0.745 / 0.693; no negatives 0.803 / 0.766. Median error of the matches 0.6-0.7 px with the dense head, 3.3 px with the MLP head.
- False gates on gate-free real images (score > 0.5): 0.00-0.03 in all four dense-head runs, including the one WITHOUT general negatives (ADE held-out 0.00, BDD val frames never trained on 0.00, mean max score 0.05 against 0.00 with negatives); MLP head 0.06 on painted-out golden frames.
- General heads move little: DIODE AbsRel 0.114-0.126 indoor / 0.219-0.237 outdoor, SUN mIoU 0.697-0.752, ADE 0.539-0.560, sim outdoor 0.681-0.695, lane_marking 0.258-0.271 in every run (class weight 4 took it there from 0.11; nothing in the ablation changes it).
- Reading: (1) **the dense head is the main cause of the jump**: with augmentation and negatives but the old head the unseen place stays at 0.70 within 20 px and 0.26 within 10 px (the old bigB without augmentation had 0.48 / 0.10, so augmentation helps the MLP head too, by about 0.2 within 20 px); (2) augmentation adds about 0.06-0.08 within 20 px on top of the dense head and matters most for partly visible gates (0.65 without, 0.88-0.94 with); (3) the general gate-free negatives show no measurable effect, neither on recall nor on false alarms; the TII/MonoRaceGate gate-free crops that stay in every run and the dense head's focal loss seem to be enough. They cost nothing, so they stay, but they are not a reason for the result. One seed per ablation; the dense-head effect (0.28 / 0.66) is far outside seed noise, the augmentation effect is 1-2 noise widths, the negatives effect is inside noise.
- The MLP head has the higher "within 20 px" number in sim (0.886) and in the sim stress test (0.91 clear, 0.77 worst mix against 0.83-0.84 / 0.61) for the reason found in Update 44: rough guesses of tiny far gates pass a 20 px test; under the size-relative test it finds 0.45 against 0.77.
- bigE first start failed twice for plumbing reasons (BDD download ran out of retries at 84% under HTTP 429; a linked folder was not walked by the image loader); fixed, bigE (vit) and bigE_h (hybrid) restarted together at 16:40.

## Update 47 (2026-11-15): bigE (BDD100K + ADE v2 labels + weak gate seg labels + exact depth in inverse space); the depth-space conflict
- **Depth-space conflict found** (all models with `depth_gt_weight`, bigA to bigD): the depth teacher (Depth Anything v2 relative) outputs INVERSE depth (near = large), the exact labels (sim, DIODE, SUN) are metres (far = large); rank correlation of the cached teacher maps with the labels is -0.9 to -1.0 on every source. Both losses were applied to the one depth output, so they pulled in opposite directions on labelled images; the network resolved it per domain: metric-like output on images that look like labelled data, inverse depth on the rest (real race halls: the demo video showed near = large). The meaning of the output therefore changed with the image. Fix: `depth_gt_space: inverse` in the config (exact depth converted to 1/depth, clamped at 0.1 m, before the scale-shift invariant loss); old behaviour stays the default so old configs reproduce.
- Depth scoring now reports two per-image scale+shift fits (`pfm.evaluate.absrel_both`): "lin" = prediction fitted to depth (protocol of all earlier numbers), "inv" = prediction fitted to 1/depth, then inverted (right for an inverse-depth output). DIODE held-out AbsRel indoor / outdoor: bigD lin 0.126 / 0.237, inv 0.255 / 0.888 (its output is metric-like there); teacher itself lin 0.143 / 0.371, inv 0.089 / 0.452 (the inverse fit is harsh outdoors: far points dominate); sim_v4 (no exact depth) lin 0.292 / 0.509, inv 0.262 / 0.458.
- **bigE** (`scripts/bige_chain.py bigE "depth_gt_space: inverse"`): bigD recipe + 7,000 BDD100K frames (gate-free negatives too) + ADE labels v2 + weak gate / not-gate seg labels on the real gate frames with `not_gate_loss` + inverse-space exact depth; vit, 10 epochs, seed 0, about 52,500 frames, 427 s per epoch alone. Several changes at once; a twin with metric depth (bigE_m) is running as the control for the depth change.
- Gates on the place never seen (red): under 10 / 20 / 40 px **0.917 / 0.990 / 0.990**, median 3.5 px, partly visible 0.97 (n=34), unmatched 0.03 per frame (bigD 0.923 / 0.978, bigD_s1 0.862 / 0.962: same within seed noise, not a proven gain). TII golden 0.874 / 0.917. Sim sim_racec full view 0.772 / 0.790 (strict 0.761), partly out 0.839; sim_racep 0.744 / 0.758, partly out 0.874. False gates 0.00-0.01 on all five gate-free sets (BDD val and ADE held-out never trained on: 0.00).
- **Segmentation head on real gate frames** (weak check of Update 45): share of surely-not-gate pixels painted as gate: red 0.169 -> **0.0011**, TII golden 0.466 -> **0.0024**; frames with more than 1% such pixels 0.89 -> 0.005 and 0.98 -> 0.04. Gate band recall fell 0.94 -> 0.60 (red) and 0.89 -> 0.54 (TII): the head now paints the frame more thinly than the 12% band (only that thin band is a positive label, the halo is ignored). The ceiling-as-gate fault is gone in the video (`runs/eval/video/bigE_*.mp4`).
- Depth (bigE output is inverse depth everywhere; inv fit): DIODE indoor **0.108**, outdoor 0.272 (lin fit 0.156 / 0.365); SUN 0.095 (bigD lin 0.101); sim indoor 0.076 (bigD lin 0.172); sim outdoor 0.053 (bigD lin 0.060). Against bigD's best protocol: better indoors and in sim, DIODE outdoor 0.272 against 0.237, but the two numbers come from different fits; the control run bigE_m will say what the depth change itself costs or gains.
- Segmentation: BDD held-out (334 val frames) mIoU 0.672 (car **0.893**, ground_road 0.959, building 0.882, person 0.531, tree_pole 0.389, other_obstacle 0.378); ADE v2 labels 0.561 (v1 labels 0.519; bigD on v1 0.549; car 0.235, still low: few cars there); SUN 0.745 (bigD 0.697 / 0.752 on two seeds); Aeroscapes 0.623 (opponent_drone 0.392; 0.58 / 0.12 on the two bigD seeds: a noisy class); sim outdoor 0.672 (lane_marking 0.238, car 0.590), sim indoor 0.826.
- Not improved: lane_marking (0.24-0.27 in sim, no real labelled test; BDD lane paint is only "ignored"); stress-test worst mix (see robustness file `runs/eval/robustness_sim_race_bigE.json`).
- Training two bigE-size runs together no longer fits the 32 GB GPU (peak 13.2 GB each with the new loss; both attempts lost one run to out-of-memory), so runs are sequential from here: bigE_m (resumed from epoch 1), then the hybrid student.

## Update 48 (2026-11-15): control for the depth change (bigE_m = bigE with the old metric-space exact-depth loss)
- bigE_m: same data, labels and seed as bigE, only `depth_gt_space` left at the old default (metric); first epoch ran in parallel with bigE, then resumed alone after an out-of-memory stop.
- Depth, each model scored with the fit that matches its output (bigE inverse fit | bigE_m linear fit): DIODE held-out indoor **0.108 | 0.128**, outdoor 0.272 | **0.230**; SUN 0.095 | 0.100; sim indoor **0.076 | 0.142**; sim outdoor 0.053 | 0.055. With the wrong fit each model is far worse (bigE lin 0.156 / 0.365; bigE_m inv 0.243 / 0.862, sim 1.0-1.1), which confirms that bigE's output is inverse depth on labelled domains too while bigE_m's is metric-like there (and inverse on unlabelled domains, as seen in the hall video).
- Reading: putting the exact depth into inverse space removes the per-domain switch of the output's meaning and is equal or better indoors and in sim; on DIODE outdoor it scores 0.04 worse, but the inverse fit is dominated by far points there (the teacher itself gets 0.452 with it, 0.371 with the linear fit), so that comparison is not like for like. No test here measures depth on the unlabelled domains (race halls), where the old models were inconsistent. Decision: keep `depth_gt_space: inverse` (consistent output, better near-range behaviour, which is what obstacle avoidance needs); the DIODE outdoor number is the known cost.
- Everything else equal within noise (one seed each): red gates under 10 / 20 px bigE 0.917 / 0.990 (partly visible 0.97), bigE_m 0.872 / 0.955 (partly visible 0.71, n=34): three dense-head runs with augmentation now span 0.955-0.990 within 20 px and 0.86-0.92 within 10 px, and partly visible gates 0.71-0.97, so the n=34 partly-visible number is too noisy to rank recipes. False gate pixels in the seg head 0.001-0.002 in both (the weak labels work in both); BDD mIoU 0.672 | 0.667 (car 0.893 | 0.894); SUN 0.745 | 0.748; ADE v2 0.561 | 0.557; Aeroscapes 0.623 | 0.659 (opponent_drone 0.39 | 0.57); sim outdoor 0.672 | 0.680, lane_marking 0.238 | 0.254; stress-test worst mix 0.59 | 0.61.
- Hybrid (FastViT) student started with the bigE recipe (`KIND=hybrid scripts/bige_chain.py bigE_h "depth_gt_space: inverse"`), alone on the GPU.

## Update 49 (2026-11-15): hybrid (FastViT-SA24) student with the bigE recipe, export and speed; lane_marking does not transfer to real roads
- `bigE_h` (`KIND=hybrid scripts/bige_chain.py bigE_h "depth_gt_space: inverse"`): same data, labels, losses, 10 epochs, seed 0 as bigE; ImageNet-pretrained fastvit_sa24, 11.0 M parameters (vit: 92.1 M), 460 s per epoch, peak 13.6 GB.
- Gates on the place never seen (red), under 10 / 20 / 40 px: **0.901 / 0.958 / 0.987**, median 2.8 px, partly visible 0.94 (n=34), unmatched 0.00 per frame (vit bigE 0.917 / 0.990, the vit runs span 0.955-0.990 / 0.86-0.92): the small student is inside the vit range. TII golden 0.874 / 0.932. Sim sim_racec full view 0.794 / 0.812 (strict 0.778), partly out 0.902; sim_racep 0.755 / 0.767, partly out 0.852. False gates 0.00-0.01 on all gate-free sets; seg-head false gate pixels 0.0007 (red) / 0.0009 (TII), band recall 0.42 / 0.52.
- General heads, hybrid | vit bigE: DIODE AbsRel inverse fit indoor 0.140 | 0.108, outdoor 0.254 | 0.272; SUN mIoU 0.719 | 0.745, depth 0.103 | 0.095; BDD mIoU 0.661 | 0.672 (car 0.901 | 0.893); ADE v2 0.546 | 0.561; Aeroscapes 0.660 | 0.623; sim outdoor 0.657 | 0.672 (lane_marking 0.204 | 0.238), depth 0.069 | 0.053; sim indoor mIoU 0.694 | 0.826 (47 frames), depth 0.098 | 0.076. The hybrid is 0.01-0.03 behind on most segmentation and depth numbers, clearly behind only on the tiny sim-indoor set. Stress test: clear 0.84, +1 stop 0.80, -4 stops 0.83, fog 0.05 0.80, worst mix 0.63 (vit 0.83 / 0.78 / 0.83 / 0.79 / 0.59): no dark collapse.
- Export (`scripts/export_onnx.py`, FastViT re-parameterised): fp32 ONNX 31 MB, equal to PyTorch (difference 2e-5, same gates 12/12); fp16 ONNX 16 MB (same gates 12/12, corner difference 0.06 px). TensorRT 11.4 fp16 engine 20.5 MB: median **0.77 ms** network only (vit: 1.40 ms), same gates as the fp32 ONNX on 24/24 red frames, largest corner difference 0.33 px, seg class equal on at least 99.8% of the pixels. Timed while a training was running on the GPU (p95 4 ms). PyTorch timings taken under the same load are not quotable (hybrid fp16 slower than fp32, a sign of contention and of autocast overhead on a small conv net); an empty-GPU PyTorch number for the hybrid is still missing. The Orin run (`scripts/orin_trt_bench.py` with `runs/bigE_h/hybrid/model_448.onnx`) is not done.
- **lane_marking on real roads: zero.** Weak proxy on 334 BDD val frames (`scripts/eval_lane_proxy_bdd.py`: locally bright road paint): predicted as lane_marking 0.000 by bigD and bigE. Real labels: TuSimple test frames (`scripts/eval_lane_tusimple.py`, 233 frames): recall 0.00 for bigE. The 0.24-0.27 lane IoU in sim does not transfer at all. First real lane labels prepared: TuSimple via HF `dotblue/TuSimple` (Apache-2.0), but the mirror holds only part of the benchmark: 267 labelled train frames, 233 test frames (`scripts/tusimple_prepare.py`: lane lines drawn as thin bands, bright pixels inside = lane_marking, value 253 = "surely not lane_marking" on non-bright pixels outside, rest ignored; label noise: lines cross occluding cars). `train.py` handles 253 like 254 (`notlane_weight`). bigF = bigE + these frames repeated 8 times (`ADD_TUS=1 scripts/bige_chain.py bigF "depth_gt_space: inverse"`) is training.
- Model sheet for the user (`scripts/vis_model_sheet.py`, `runs/eval/vis/model_sheet_bigE_vit.jpg`): outputs on nine held-out real images; visible faults: no lane lines, sky painted as an arbitrary class (there is no sky class; sky is ignored in training), thin poles and wires coarse.

## Update 50 (2026-11-15): bigF = bigE + 267 real lane frames (TuSimple): lane_marking works on TuSimple-like highways, not yet on other real roads
- `ADD_TUS=1 scripts/bige_chain.py bigF "depth_gt_space: inverse"`: bigE recipe + the 267 TuSimple train frames repeated 8 times (2,136 samples of about 54,600), lane line pixels = lane_marking, value 253 = "not lane_marking" loss; vit, 10 epochs, seed 0, 445 s per epoch.
- **lane_marking on 233 TuSimple test frames (other clips, never trained on): recall 0.00 -> 0.826, precision 0.972, IoU 0.806** (bigE: 0.000). Precision is scored only against pixels labelled "surely not paint", unlabelled bright pixels are not scored, so it is an upper bound. Same camera, same highways as the training clips: an in-domain test of a new place, not a transfer test.
- Transfer of the lane class to other real roads: BDD val proxy (locally bright road paint) predicted as lane_marking 0.002 (bigE 0.000), plain road 0.0001. So the class is still tied to the TuSimple look; note that the 7,000 BDD training frames carry their lane paint as "ignored", which gives the model no reason to paint it there. Sim outdoor lane_marking IoU 0.258 (unchanged). Lane paint on real roads other than TuSimple-like highways remains unsolved; it needs real lane labels from more sources (the full TuSimple set, BDD lane labels, or CULane are not on a login-free mirror in full).
- Everything else inside the run-to-run range: red gates under 10 / 20 px 0.872 / 0.962, partly visible 0.94, unmatched 0.00 (five dense-head vit runs now: 0.955-0.990 within 20 px, 0.862-0.923 within 10 px); TII golden 0.850 / 0.922; sim_racec full view 0.772 / 0.794 (strict 0.759); false gates 0.00; seg-head false gate pixels 0.0017 / 0.0006; DIODE inverse fit 0.110 / 0.284; SUN mIoU 0.736, depth 0.097; BDD mIoU 0.669 (car 0.899); ADE v2 0.570; Aeroscapes 0.661; sim outdoor 0.682, depth 0.051; sim indoor 0.803; stress test clear 0.84, worst mix 0.62.
- bigF is the current best vit model (same as bigE elsewhere, plus the lane class on highways). User decision (21:30): if this run is satisfactory, pause gate work and try different environments. Next: zero-shot tests on real environments never used in training (UAVid drone-view city, RUGD off-road, KITTI road depth).

## Update 51 (2026-11-15): zero-shot tests in three environments no run has seen (UAVid, RUGD, KITTI); clean speed numbers
- User direction (21:30): gate work is paused after bigF; try different environments. `scripts/env_prepare.py` builds three evaluation-only sets from data no training run has used: UAVid (city from a drone at height, 120 frames, CC BY-NC-SA 4.0), RUGD (off-road trails / creek / park, 298 frames, research use), KITTI depth (road scenes with lidar depth, 305 validation frames, CC BY-NC-SA 3.0). Class maps in the script header; UAVid clutter and low vegetation, RUGD water and sky are ignored.
- **KITTI depth, zero-shot** (`scripts/eval_depth_dir.py`, AbsRel, per-image scale+shift, best fit of each model): bigD 0.167 (inverse fit; 0.375 linear), **bigE 0.088, bigF 0.081, hybrid bigE_h 0.085** (inverse fit; linear 0.34). On a domain with no exact-depth training data the old models fall back to the teacher's inverse depth but with half the quality; the consistent inverse-depth training of bigE halves the error. This is the transfer test that Update 48 lacked, and it supports the depth-space decision.
- **RUGD off-road, zero-shot**, mIoU over the mapped classes: bigD 0.478, bigE 0.527, **bigF 0.544**, hybrid 0.495; pixel accuracy 0.87-0.90. Good: tree_pole 0.90, ground_road 0.87, person 0.62-0.66 (hybrid 0.25), building 0.60-0.64. Weak: rock 0.04-0.13, other_obstacle 0.16-0.17 (bush, fence, log: painted as tree or ground), car 0.32-0.53 (off-road utility vehicles).
- **UAVid drone-view city, zero-shot**: mIoU bigD 0.333, bigE 0.295, bigF 0.313, hybrid 0.266; pixel accuracy 0.55-0.62. building 0.60-0.64, ground_road 0.42-0.47, tree_pole 0.17-0.34, car 0.12-0.21, **person 0.01**. The weakest environment by far: steep oblique views from height, objects a few pixels wide after shrinking the 4K frame five times; in the output sheet, roadside trees are painted as person and vegetation as other_obstacle. Differences between the models here (0.27-0.33) are within what one seed can move.
- Output sheet for these environments: `SHEET=env scripts/vis_model_sheet.py` -> `runs/eval/vis/model_sheet_bigF_vit_env.jpg`.
- Clean speed (GPU otherwise empty, batch 1, 448x448): vit PyTorch end to end 6.44 ms fp32 / 2.91 ms fp16; hybrid PyTorch end to end **2.50 ms fp32** / 3.59 ms fp16 (fp16 autocast is slower than fp32 for this small conv net); TensorRT 11.4 fp16 network only: vit **1.15 ms**, hybrid **0.68 ms** (p95 within 0.01 ms). These replace the under-load timings of Updates 45 and 49.
- Running: bigF_h (hybrid with the bigF recipe). Next: add the UAVid train split (other sequences than val/test, with tiles at higher resolution) and RUGD sequences held apart from the test sequences, then measure again; the three eval sets stay evaluation-only as they are, RUGD scoring will be restricted to sequences kept out of training.

## Update 52 (2026-11-15): bigF_h (hybrid with the lane frames); baselines on the new-environment test splits
- `bigF_h` (`KIND=hybrid ADD_TUS=1 scripts/bige_chain.py bigF_h "depth_gt_space: inverse"`, seed 0): red gates under 10 / 20 px 0.894 / 0.962, partly visible 0.97 (n=34), unmatched 0.00, median 2.6 px (bigE_h 0.901 / 0.958: the two hybrid runs agree); TuSimple test lane_marking recall 0.788, precision 0.971, IoU 0.770 (vit bigF 0.826 / 0.972 / 0.806); BDD lane proxy 0.003 (no transfer, as for the vit). DIODE inverse fit 0.130 / 0.234, SUN 0.707, BDD 0.661 (car 0.899), ADE v2 0.552, Aeroscapes 0.641, sim outdoor 0.659, sim indoor 0.712; false gates 0.00; stress test clear 0.83, worst mix 0.61. This is the current deployable small model: ONNX fp32 / fp16 and a TensorRT fp16 engine are in `runs/bigF_h/hybrid/` (checked like the others); vit bigF ONNX in `runs/bigF/vit/`.
- New-environment training data (`scripts/env_train_prepare.py`): UAVid train split, 200 frames of sequences other than val/test, each as whole view + four quadrants = 1,000 images (`uav`); RUGD every 8th frame of 14 sequences = 634 images (`rugd`); RUGD sequences creek, park-2, trail-4, village held out (95 of the 298 eval frames -> `data/general/eval_rugd_heldout`); UAVid val quadrants -> `data/general/eval_uavid_tiles` (280 images).
- Baselines of bigF (zero-shot) on the new splits: UAVid quadrant tiles mIoU 0.318 (whole frames 0.313; building 0.551, tree_pole 0.359, ground_road 0.335, car 0.269, person 0.076): **object size is not the cause of the UAVid weakness**, the tiles with objects twice as large score the same; it is the viewpoint / scene type. RUGD held-out sequences mIoU 0.433 (tree_pole 0.811, ground_road 0.815, building 0.685, car 0.527, other_obstacle 0.170, rock 0.024, person 0.000 with very few person pixels in these four sequences).
- Running: bigG = bigF + uav + rugd (`ADD_TUS=1 ADD_ENV=1 scripts/bige_chain.py bigG "depth_gt_space: inverse"`), vit.

## Update 53 (2026-11-16): bigG = bigF + UAVid and RUGD training data: the two new environments improve, nothing else moves
- `ADD_TUS=1 ADD_ENV=1 scripts/bige_chain.py bigG "depth_gt_space: inverse"`: bigF mix + 1,000 UAVid train images (whole views and quadrants of 200 frames, other sequences than the test frames) + 634 RUGD frames of 14 sequences (4 sequences held out); about 56,200 frames, vit, 10 epochs, seed 0, 456 s per epoch.
- **UAVid (drone-view city; test frames from sequences not trained on)**, mIoU bigF -> bigG: whole frames **0.313 -> 0.619** (building 0.60 -> 0.90, tree_pole 0.31 -> 0.85, ground_road 0.44 -> 0.80, car 0.21 -> 0.48, person 0.01 -> 0.07); quadrant tiles **0.318 -> 0.721** (car 0.64, person 0.30). Persons are a few pixels in whole frames and stay unsolved there; the tiles show that with training data object size matters again (0.62 whole, 0.72 tiles).
- **RUGD (off-road)**: the four held-out sequences (95 frames) 0.433 -> 0.477 (ground_road 0.815 -> 0.875, tree_pole 0.811 -> 0.858, other_obstacle 0.170 -> 0.294, rock 0.024 -> 0.048, car 0.527 -> 0.561; person 0.000 with almost no person pixels); all 298 eval frames 0.544 -> 0.607 (203 of them now come from sequences with training frames, so only the held-out number is a clean one). The gain on unseen trails is modest: rock and bush / fence / log remain weak; RUGD rock is rare and looks unlike ADE rock.
- KITTI depth still zero-shot (no KITTI in training): inverse fit 0.082 (bigF 0.081).
- Nothing else moved beyond noise: red gates under 10 / 20 px 0.891 / 0.978, partly visible 0.97, unmatched 0.02 (six dense-head vit runs: 0.955-0.990 / 0.862-0.923); TII golden 0.869 / 0.932; sim_racec full view 0.773 / 0.796 (strict 0.761); false gates 0.00; TuSimple lane recall 0.829 / IoU 0.810; BDD mIoU 0.676 (car 0.899); SUN 0.738; ADE v2 0.562; Aeroscapes 0.641 (opponent_drone 0.369); DIODE inverse fit 0.118 / 0.305 (bigF 0.110 / 0.284); sim outdoor 0.681, depth 0.045; sim indoor 0.779; stress test clear 0.83, worst mix 0.62.
- Reading: adding a few hundred to a thousand labelled frames of a new environment lifts that environment a lot (UAVid doubles) at no measurable cost elsewhere; the mix keeps absorbing sources. A zero-shot number on a new kind of scene should be expected near 0.3-0.5 mIoU (UAVid 0.31, RUGD 0.43-0.54) while depth transfers much better (KITTI 0.08). bigG is the current best vit model. Hybrid with the same recipe (bigG_h) started.

## Update 54 (2026-11-16): adverse-weather roads zero-shot (IDD-AW); the scores so far left out sky and vegetation; label set v3
- **IDD-AW** (Indian roads in rain, fog, low light, snow; 238 validation frames, evaluation only, `scripts/env2_prepare.py`; research licence; no per-condition tag in the mirror, all conditions together), zero-shot mIoU with sky and vegetation ignored as in the BDD mapping: bigF 0.478, bigG 0.479, hybrid bigF_h 0.495; ground_road 0.87, car 0.74-0.76, building 0.55-0.58, person 0.35-0.39, tree_pole (poles, lights) 0.25, other_obstacle 0.09-0.11. Roads and vehicles hold up in bad weather; thin poles and small obstacles do not. Sheet: `runs/eval/vis/model_sheet_bigG_vit_env2.jpg`.
- **What the sheets show and the scores hid**: every outdoor sheet has sky painted as tree, ground or building, and dark vegetation as building. Sky (and BDD / IDD-AW vegetation) was "ignored" in all labels, so no score counted those pixels. Scoring bigG on the same frames with sky = class 0 (background) and vegetation = tree_pole (`scripts/sky_labels_v3.py`, eval sets `eval_bdd3`, `eval_ade3`, `eval_rugd_heldout3`, `eval_iddaw3`): pixel accuracy BDD 0.946 -> **0.601**, ADE 0.809 -> 0.633, RUGD held-out 0.875 -> 0.811, IDD-AW 0.780 -> **0.429**; mIoU BDD 0.402, ADE 0.424, RUGD held-out 0.378, IDD-AW 0.290 (background IoU 0.00 by construction, tree_pole on BDD 0.194). So on road scenes about a third to a half of the pixels were outside every earlier number, and the model is wrong on most of them. For obstacle avoidance this matters: a "tree" in the sky is a false obstacle.
- Fix prepared: label set v3 for training: BDD (sky -> background, vegetation -> tree_pole), ADE (sky / cloud -> background, `SKY=1 scripts/ade_relabel_v2.py`), RUGD training frames (sky -> background); 10,634 label files in `data/general/train/labels_v3/seg`, used through `LABELS_V3=1` in `scripts/bige_chain.py` (own folder `data/mix/labels_H`; older runs keep `labels_E`). The class list is unchanged: class 0 (background) was unused. Sim frames keep sky unlabelled; UAVid has no sky label. `eval_seg_sun.py` now also reports the background class where the labels contain it (older eval sets do not, so their numbers are unchanged).
- Queue: bigG_h (hybrid, bigG recipe) is training; bigH = bigG + label set v3 follows.

## Update 55 (2026-11-16) — bigG_h: hybrid (FastViT-SA24, 11.0 M) with the bigG recipe

Same data and settings as bigG (bigF set + UAVid 1,000 + RUGD 634, `depth_gt_space: inverse`, dense gate head stride 4), `KIND=hybrid`, 10 epochs, one seed. Chain: `KIND=hybrid ADD_TUS=1 ADD_ENV=1 scripts/bige_chain.py bigG_h`.

| metric (one seed, gate numbers ±0.06) | bigF_h | bigG_h | bigG (vit) |
|---|---|---|---|
| red (unseen place) gates <10 px / <20 px | 0.894 / 0.962 | 0.885 / 0.965 | 0.891 / 0.978 |
| red partly visible <20 px (n=34, too few to rank) | – | 0.971 | – |
| red unmatched predictions per frame | – | 0.01 | – |
| false gates on BDD / ADE gate-free frames (n=300 each) | – | 0.00 / 0.00 | – |
| seg-head false gate pixel share, red / TII | – | 0.0008 / 0.0011 | – |
| UAVid full frame mIoU / tiles | – | 0.582 / 0.671 | 0.62 / 0.72 |
| RUGD held-out sequences mIoU | – | 0.491 | 0.48 |
| BDD val mIoU (old label set, sky/vegetation ignored) | – | 0.661 | – |
| TuSimple test lane_marking recall / precision | – | 0.778 / 0.972 | 0.83 / – |
| BDD lane proxy (bright road paint called lane) | – | 0.004 | 0.003 |
| DIODE held-out AbsRel, inverse fit, indoor / outdoor | – | 0.128 / 0.231 | – |
| KITTI zero-shot AbsRel, inverse fit | – | 0.081 | – |

Reference only, not generalisation: TII golden (same hall as training) <20 px 0.922; sim_racec full-view <20 px 0.808 (sim-to-sim, not a success rate); sim stress worst mix (30 m/s + dark + fog) recall 0.63.

Reading: the hybrid follows the vit within noise on gates and depth, is a little behind on aerial segmentation (UAVid 0.58 vs 0.62) and lane recall (0.78 vs 0.83). Unchanged weaknesses: person from height (UAVid person IoU 0.005 full frame, 0.13 tiles; RUGD held-out person 0.0), rock in off-road (0.13), lane marking does not transfer to BDD roads.

Export: `runs/bigG_h/hybrid/model_448.onnx` (31.2 MB, fp32, same gate set as PyTorch on 12/12 check frames), `model_448_fp16.onnx` (15.7 MB, 12/12, max corner diff 0.11 px), `model_448_fp16.engine` (TensorRT 11.4, 24/24 frames same gate set, max corner diff 0.31 px, seg same-class share ≥ 0.991). Engine latency 0.74 ms median was measured while the bigH training was using the GPU, so it is not a clean speed number (clean bigF_h number: 0.68 ms). Jetson Orin still not measured.

## Update 56 (2026-11-16) — bigH: label set v3 (sky = background, vegetation = tree_pole)

bigH = bigG data and settings, but the segmentation labels of BDD, ADE and RUGD come from label set v3 (`LABELS_V3=1`, folder `data/mix/labels_H`): sky is labelled as class 0 (background) instead of ignored, and BDD vegetation is labelled tree_pole instead of ignored. vit, 10 epochs, one seed. Chain: `ADD_TUS=1 ADD_ENV=1 LABELS_V3=1 scripts/bige_chain.py bigH 'depth_gt_space: inverse'`.

Sky-aware scoring (sky and vegetation count; these are the sets on which Update 54 showed bigG painting sky as tree / ground / building):

| set (frames never trained on) | bigG mIoU / pixel-acc | bigH mIoU / pixel-acc |
|---|---|---|
| eval_bdd3 (BDD val; same source as training data) | 0.402 / 0.601 | 0.746 / 0.934 |
| eval_ade3 (ADE held-out; same source) | 0.424 / 0.633 | 0.594 / 0.831 |
| eval_rugd_heldout3 (RUGD, 4 held-out sequences) | 0.378 / 0.811 | 0.512 / 0.867 |
| eval_iddaw3 (IDD-AW, zero-shot: no IDD data in training) | 0.290 / 0.429 | 0.590 / 0.869 |

bigH per class on IDD-AW (zero-shot): background 0.905, tree_pole 0.809, ground_road 0.851, car 0.718, building 0.460, person 0.321, other_obstacle 0.063.

Everything else, bigG -> bigH (one seed each; gate numbers ±0.06, so all of these are "no change"):

| metric | bigG | bigH |
|---|---|---|
| red (unseen place) gates <10 px / <20 px | 0.891 / 0.978 | 0.872 / 0.974 |
| red median corner error, px | 4.0 | 4.7 |
| red partly visible <20 px (n=34, too few to rank) | 0.971 | 0.882 |
| false gates BDD / ADE gate-free (n=300 each) | 0.00 / 0.00 | 0.00 / 0.00 |
| seg-head false gate pixel share red / TII | – | 0.0017 / 0.0012 |
| UAVid full / tiles mIoU | 0.619 / 0.721 | 0.616 / 0.715 |
| RUGD held-out (old labels) mIoU | 0.477 | 0.479 |
| BDD val, old labels (sky / vegetation ignored) mIoU | 0.676 | 0.632 |
| TuSimple test lane recall / IoU | 0.829 / 0.810 | 0.823 / 0.806 |
| BDD lane proxy | 0.003 | 0.002 |
| DIODE held-out AbsRel inverse fit, indoor / outdoor | 0.118 / 0.305 | 0.111 / 0.228 |
| KITTI zero-shot AbsRel inverse fit | 0.082 | 0.080 |

Reference only: TII golden (hall seen) <20 px 0.908; sim_racec full-view <20 px 0.794 (sim-to-sim); sim stress worst mix recall 0.60.

Reading:
- The sky / vegetation defect is fixed by labelling, not by more data: IDD-AW, which contributes no training frame, doubles (0.29 -> 0.59 mIoU, pixel accuracy 0.43 -> 0.87). No gate, depth, lane or aerial metric moved outside noise.
- One cost: on the old BDD labels tree_pole IoU drops 0.385 -> 0.219 (mIoU 0.676 -> 0.632). In the old set tree_pole means poles only and vegetation is ignored; bigH now paints vegetation tree_pole, which spills over thin poles' surroundings. With vegetation counted (bdd3) tree_pole is 0.801. The old BDD score is therefore no longer the right one to quote.
- The earlier segmentation scores (Updates up to 53) excluded sky and vegetation pixels and so overstated how usable the class map was outdoors; the v3 sets are the honest ones from here on.
- Still weak: other_obstacle on unseen roads (IDD-AW 0.06), person from height (UAVid 0.07 full frame), RUGD held-out person 0.0 and rock 0.11, building on IDD-AW 0.46, lane marking outside TuSimple-like highways.
- DIODE outdoor inverse-fit AbsRel 0.305 -> 0.228 is a single-seed difference; not claimed as an effect of the labels.

Export: `runs/bigH/vit/model_448.onnx` (337 MB), `model_448_fp16.onnx` (168.6 MB, 12/12 frames same gate set), `model_448_fp16.engine` (TensorRT 11.4, 1.14 ms median on an idle 5090, 24/24 frames same gate set, max corner diff 0.25 px, seg same-class share >= 0.998). Sheets: `runs/eval/vis/model_sheet_bigH_vit.jpg`, `_env.jpg`, `_env2.jpg`.

bigH is the current best general model. No hybrid with label set v3 yet.

## Update 57 (2026-11-16) — bigH_h: hybrid with label set v3

bigH_h = bigH recipe (bigG data, label set v3, `depth_gt_space: inverse`) with `KIND=hybrid` (FastViT-SA24, 11.0 M), 10 epochs, one seed.

| metric (one seed; gate numbers ±0.06) | bigG_h | bigH_h | bigH (vit) |
|---|---|---|---|
| red (unseen place) gates <10 px / <20 px | 0.885 / 0.965 | 0.907 / 0.965 | 0.872 / 0.974 |
| red median corner error, px | 2.8 | 2.5 | 4.7 |
| red partly visible <20 px (n=34) | 0.971 | 0.971 | 0.882 |
| false gates BDD / ADE gate-free | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| eval_bdd3 mIoU / pixel-acc (same source as training) | – | 0.735 / 0.931 | 0.746 / 0.934 |
| eval_ade3 (same source) | – | 0.592 / 0.833 | 0.594 / 0.831 |
| eval_rugd_heldout3 (held-out sequences) | – | 0.516 / 0.868 | 0.512 / 0.867 |
| eval_iddaw3 (zero-shot) | – | 0.614 / 0.872 | 0.590 / 0.869 |
| UAVid full / tiles mIoU | 0.582 / 0.671 | 0.580 / 0.668 | 0.616 / 0.715 |
| TuSimple test lane recall | 0.778 | 0.772 | 0.823 |
| BDD lane proxy | 0.004 | 0.004 | 0.002 |
| DIODE AbsRel inverse fit indoor / outdoor | 0.128 / 0.231 | 0.124 / 0.264 | 0.111 / 0.228 |
| KITTI zero-shot AbsRel inverse fit | 0.081 | 0.082 | 0.080 |

Reference only: TII golden (hall seen) <20 px 0.937; sim_racec full-view <20 px 0.801 (sim-to-sim); sim stress worst mix recall 0.62.

Reading: the label fix carries over to the small model in full; on the sky-aware road / off-road sets the hybrid equals the vit (differences ≤ 0.025, one seed). The vit stays ahead only on aerial views (UAVid 0.62 vs 0.58) and lane recall (0.82 vs 0.77). The hybrid's lower median corner error on red (2.5 vs 4.7 px) has now appeared in two recipe pairs (bigG_h 2.8 vs bigG 4.0), still one seed each.

Export: `runs/bigH_h/hybrid/model_448.onnx` (31.2 MB), `model_448_fp16.onnx` (15.7 MB), `model_448_fp16.engine` — TensorRT 11.4 on an idle RTX 5090: 0.68 ms median (network only), 24/24 check frames same gate set, max corner diff 0.77 px, seg same-class share ≥ 0.997. Jetson Orin not measured.

Process note: the bigH_h tmux command lacked the `; echo CHAIN_END >> log` suffix, so the GPU sat idle from about 05:10 to 07:40. A second seed of bigH (`bigH_s1`, `seed: 1`) was started at 07:45.

## Update 58 (2026-11-16) — bigH second seed (bigH_s1)

Same recipe as bigH, `seed: 1`. Two-seed ranges (seed 0 / seed 1):

| metric | seed 0 | seed 1 |
|---|---|---|
| red (unseen place) gates <10 px / <20 px | 0.872 / 0.974 | 0.897 / 0.968 |
| red median corner error, px | 4.7 | 3.9 |
| red partly visible <20 px (n=34) | 0.882 | 0.941 |
| false gates BDD / ADE gate-free | 0.00 / 0.00 | 0.00 / 0.00 |
| seg-head false gate pixel share on red | 0.0017 | 0.0029 |
| eval_bdd3 mIoU / pixel-acc | 0.746 / 0.934 | 0.745 / 0.934 |
| eval_ade3 | 0.594 / 0.831 | 0.599 / 0.838 |
| eval_rugd_heldout3 | 0.512 / 0.867 | 0.504 / 0.865 |
| eval_iddaw3 (zero-shot) | 0.590 / 0.869 | 0.595 / 0.870 |
| UAVid full / tiles mIoU | 0.616 / 0.715 | 0.624 / 0.722 |
| TuSimple test lane recall | 0.823 | 0.808 |
| BDD lane proxy | 0.002 | 0.002 |
| DIODE AbsRel inverse fit indoor / outdoor | 0.111 / 0.228 | 0.113 / 0.232 |
| KITTI zero-shot AbsRel inverse fit | 0.080 | 0.082 |

Reference only: TII golden (hall seen) <20 px 0.908 / 0.927; sim_racec full-view <20 px 0.794 / 0.785 (sim-to-sim).

Reading: the label-set-v3 gain is not a seed effect: segmentation scores of the two seeds differ by at most 0.008 mIoU, against gains of 0.13–0.34 over bigG. Segmentation and depth seed noise is about ±0.01; gate recall on red varies more (0.87–0.90 at 10 px), as before. The DIODE outdoor inverse-fit value (0.23 in both seeds vs 0.305 for bigG) repeats, so it is probably a real effect of labelling sky as background (sky pixels have no depth label; cause not verified). Per-class weak points are the same in both seeds: IDD-AW other_obstacle 0.063 / 0.065, RUGD held-out person 0.0 and rock 0.11 / 0.07, UAVid person 0.07 / 0.08.

## Update 59 (2026-11-16) — bigI: weak road-paint labels on BDD (lane_marking beyond TuSimple)

No labelled real lane data was reachable (HF mirrors checked: one BDD "lane" mirror holds images only, one CULane mirror is empty, one set is dashcam video only, one is a 40 GB lidar set). Instead: label set v4 (`scripts/lane_labels_v4.py`, chain flag `LABELS_V4=1`, folder `data/mix/labels_I`) = v3 + weak lane_marking labels on the BDD training frames: bright road paint by a white top-hat rule (top-hat > max(40, 0.35 x median road brightness), road or paint in 75% of a 41 px window, component >= 12 px, bottom 15% of the frame excluded because of the hood edge, frames with paint share > 12% skipped). 6,010 of 7,000 frames get labels, median paint share of the road 1.6%. The labels are noisy: they include crosswalks, arrows and stop lines, and some sun patches. bigI = bigH recipe with these labels, vit, 10 epochs, one seed.

| metric (one seed; gates ±0.06) | bigH (two seeds) | bigI |
|---|---|---|
| TuSimple test lane recall / precision / IoU (real labels, independent of the rule) | 0.823, 0.808 / 0.97 / 0.81, 0.79 | 0.842 / 0.967 / 0.819 |
| BDD val: bright road paint predicted lane_marking (SAME rule as the training labels: circular, not independent evidence) | 0.002 | 0.362 |
| BDD val: plain road predicted lane_marking | 0.0001 | 0.0158 |
| eval_bdd3 mIoU / ground_road IoU | 0.746, 0.745 / 0.949 | 0.745 / 0.934 |
| eval_iddaw3 (zero-shot) mIoU / ground_road IoU | 0.590, 0.595 / 0.851 | 0.595 / 0.830 |
| eval_ade3 mIoU | 0.594, 0.599 | 0.609 |
| eval_rugd_heldout3 mIoU | 0.512, 0.504 | 0.500 |
| UAVid full mIoU | 0.616, 0.624 | 0.620 |
| red (unseen place) gates <10 px / <20 px | 0.872, 0.897 / 0.974, 0.968 | 0.926 / 0.981 |
| false gates BDD / ADE gate-free | 0.00 / 0.00 | 0.00 / 0.00 |
| KITTI zero-shot AbsRel inverse fit | 0.080, 0.082 | 0.080 |
| DIODE inverse fit indoor / outdoor | 0.111, 0.113 / 0.228, 0.232 | 0.116 / 0.275 |

Reading:
- The model now marks road paint on city roads. The BDD number (0.36) is measured with the rule that made the labels, so it only shows the labels were learnt. The independent evidence is visual: on IDD-AW frames (no IDD data in training) the sheet `runs/eval/vis/model_sheet_bigI_vit_env2.jpg` shows centre lines and dashes marked in 4 of 9 frames where bigH marked none; coverage is partial and one frame marks a bright barrier base as lane. There is still no labelled real test for city-road lane marking, so no recall number can be given for it.
- TuSimple (real labels) did not get worse (0.84 vs 0.81–0.82).
- Cost: ground_road IoU drops by 0.015–0.02 on BDD and IDD-AW, because 1.6% of plain road is now called lane_marking and because true paint is "road" in those label sets. mIoU is unchanged.
- DIODE outdoor inverse fit 0.275 vs 0.23 in both bigH seeds: possibly noise, not explained.
- Gates on red 0.981 / 0.926 is the highest so far but inside the seed range of this recipe family.

Half-gate baseline for this model (see Update 60 for the test): {"crops": 479, "extra_pred_per_crop": 0.04, "corners_in_view_0": {"n": 99, "found<0.5size": 0.212, "found<0.25size": 0.051, "found<0.1size": 0.0, "no_prediction": 0.707}, "corners_in_view_1": {"n": 380, "found<0.5size": 0.379, "found<0.25size": 0.237, "found<0.1size": 0.011, "no_prediction": 0.487, "visible_corner_median_err_px": 25.9}}

## Update 60 (2026-11-16) — half gates (0–1 corners in view): test and bigJ

User request: gates of which only a part is visible should also be found.

Why they were missed. Count of gate labels by corners inside the image (all sources of the mix): sim has 267 gates with exactly one corner in view out of about 60,000 and none with zero corners in view but a bar crossing the image (the close-pass cases are dropped because a corner is behind the camera); the real labels (TII, MonoRaceGate) contain no gate with fewer than 2 labelled corners, and an unlabelled half gate in a real frame is trained as "no gate here".

Test (`scripts/eval_half_gates.py`, now part of the chain). Place never trained on (MonoRaceGate red, all 4 corners labelled). For every fully visible gate at least 0.12 of the frame wide, 2 crops per case with a fixed seed (zoom 1.6–4x, `pfm.train.half_gate_crop`): one corner in view (n=380) or no corner but a piece of the outline at least 0.3 crop widths long (n=99). The true corners outside the crop are known. Found = a prediction with score > 0.5 whose 4 corners (also the unseen ones) are within a fraction of the gate size (mean corner error / longest gate side, best cyclic order). CAUTION: a crop of a farther gate is a proxy for a real close pass (no change of wide-angle distortion, no motion blur); n=99 is small.

Training change (bigJ = bigH recipe + `half_gate_prob: 0.25`). In `CachedDS`, 25% of the frames that contain a fully visible gate with 4 known corners get a half-gate crop (same sampler; half one-corner, half no-corner) instead of the usual 1–1.6x zoom; labels and teacher maps are warped as before. `gate_list` now also accepts a gate with 4 known corners and none in view when at least 0.15 image widths of its outline cross the image. Test added (60 pass). Real frames with partly labelled gates cannot be used for these crops (unknown corners), and unlabelled half gates in real frames still act as negatives.

| half gates on red (one seed each) | bigD | bigH s0 | bigH s1 | bigH_h | bigI | bigJ |
|---|---|---|---|---|---|---|
| 1 corner in view: found < 0.25 size | 0.224 | 0.239 | 0.395 | 0.316 | 0.237 | 0.495 |
| 1 corner in view: found < 0.5 size | 0.329 | 0.426 | 0.589 | 0.437 | 0.379 | 0.724 |
| 1 corner in view: no prediction at all | 0.545 | 0.447 | 0.313 | 0.450 | 0.487 | 0.166 |
| 1 corner: visible corner median error, px (256x192) | 25.2 | 33.2 | 18.3 | 15.9 | 25.9 | 19.8 |
| 0 corners in view: found < 0.25 size | 0.000 | 0.020 | 0.020 | 0.040 | 0.051 | 0.232 |
| 0 corners in view: found < 0.5 size | 0.152 | 0.202–0.232 | | 0.212 | 0.212 | 0.455 |
| 0 corners in view: no prediction at all | 0.758 | 0.707 | 0.707 | 0.707 | 0.707 | 0.384 |
| predictions per crop matching no gate | 0.035 | 0.031 | 0.031 | 0.044 | 0.040 | 0.086 |

bigJ on the standard metrics (one seed; gates ±0.06): red full frames 0.974 / 0.913 (<20 / <10 px; bigH two seeds 0.968–0.974 / 0.872–0.897), partly visible n=34 0.971, false gates on BDD / ADE / Aeroscapes / gate-free TII crops 0.00, TII frames with gates painted out 0.01, seg-head false gate pixel share on red 0.0005. eval_bdd3 0.740, eval_iddaw3 0.592, eval_rugd_heldout3 0.502, UAVid 0.619, TuSimple lane recall 0.814, KITTI inverse fit 0.080, DIODE inverse fit 0.118 / 0.272. Reference only: TII golden (hall seen) 0.903; sim_racec full view <20 px 0.795, partly out 0.875 (sim-to-sim); sim stress worst mix 0.61.

Reading:
- The augmentation roughly doubles the one-corner case (0.24–0.40 -> 0.50 within a quarter of the gate size; a gate is reported at all in 83% instead of 51–69%) and lifts the no-corner case from about 0.03 to 0.23. The baseline varies a lot between seeds (bigH 0.24 vs 0.40), so the size of the gain is uncertain; the direction is clear for the no-corner case and for "no prediction".
- Still weak in absolute terms: half of the one-corner gates and three quarters of the bar-only gates are not placed within a quarter of their size, and the visible corner is about 20 px off (full gates: 4 px). The unseen corners have to be guessed from one corner and the bar directions.
- Cost: spurious predictions in these crops rise from 0.03–0.04 to 0.09 per crop. No cost seen on full frames or on any gate-free set.
- Next: a higher share / wider zoom range, a second seed, and the same recipe for the hybrid; real close-pass footage (own camera) is needed to validate the proxy.

## Update 61 (2026-11-16) — bigK: label set v5 (two-wheelers = vehicle) + v4 + half-gate crops

User remark on the IDD-AW sheet: motorcycles are painted as person. Cause: in the BDD / ADE / IDD mappings the rider is person and the motorcycle / bicycle itself was other_obstacle. Label set v5 (`scripts/two_wheel_labels_v5.py`, `SKY=1 TWO=1 scripts/ade_relabel_v2.py`, chain flag `LABELS_V5=1`, folder `data/mix/labels_J`): motorcycle and bicycle -> car (vehicle), rider stays person. 595 of 7,000 BDD training frames and the ADE frames with bicycle / minibike are affected; evaluation sets eval_bdd5, eval_ade5, eval_iddaw5 (119 of 238 IDD-AW frames contain a two-wheeler). Aeroscapes 'bike' and RUGD 'bicycle' are unchanged (other_obstacle). This mapping is my choice, the user was told.

bigK = bigH recipe + v4 (weak road paint) + v5 + `half_gate_prob: 0.25`, vit, 10 epochs, one seed.

Direct measurement on IDD-AW (zero-shot; `scripts/diag_two_wheelers.py`; share of the raw pixels per predicted class):

| raw IDD-AW pixels | bigH | bigI | bigK |
|---|---|---|---|
| motorcycle + bicycle -> car | 0.39 | 0.39 | 0.75 |
| motorcycle + bicycle -> person | 0.21 | 0.17 | 0.06 |
| rider -> person / car | 0.40 / 0.39 | 0.47 / 0.32 | 0.37 / 0.46 |
| pedestrian -> person / car | 0.54 / 0.20 | 0.57 / 0.20 | 0.47 / 0.29 |

Other metrics (one seed; gates ±0.06):

| metric | bigI | bigJ | bigK |
|---|---|---|---|
| red gates <10 / <20 px | 0.926 / 0.981 | 0.913 / 0.974 | 0.878 / 0.962 |
| half gate, 1 corner: found <0.25 size / no prediction | 0.237 / 0.487 | 0.495 / 0.166 | 0.545 / 0.126 |
| half gate, 0 corners: found <0.25 size / no prediction | 0.051 / 0.707 | 0.232 / 0.384 | 0.212 / 0.465 |
| spurious predictions per half-gate crop | 0.040 | 0.086 | 0.067 |
| false gates BDD / ADE | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| eval_iddaw5 mIoU / person / car IoU | 0.594 / 0.360 / 0.728 | – | 0.585 / 0.318 / 0.719 |
| eval_bdd5 mIoU | 0.745 | – | 0.744 |
| eval_ade5 mIoU / car IoU | 0.608 / 0.223 | – | 0.613 / 0.246 |
| eval_rugd_heldout3 mIoU | 0.500 | 0.502 | 0.498 |
| UAVid mIoU | 0.620 | 0.619 | 0.620 |
| TuSimple lane recall | 0.842 | 0.814 | 0.836 |
| BDD lane proxy (circular) / plain road as lane | 0.362 / 0.016 | – | 0.371 / 0.017 |
| KITTI inverse fit | 0.080 | 0.080 | 0.080 |

Reference only: TII golden (hall seen) <20 px 0.898; sim_racec full view <20 px 0.788 (sim-to-sim); sim stress worst mix 0.60.

Reading:
- The remark is fixed for the vehicle itself: motorcycles painted as person fall from 0.21 to 0.06 and 75% of their pixels are now vehicle.
- Side effect, a real regression: people next to or on vehicles are pulled into the vehicle class. Riders called car 0.39 -> 0.46 and pedestrians called car 0.20 -> 0.29; pedestrian pixels called person 0.54–0.57 -> 0.47; IDD-AW person IoU 0.36 -> 0.32. One seed, so part may be noise, but the direction matches the label change (a rider on a motorcycle is now a small person region inside a vehicle region). For obstacle avoidance both are obstacles; for anything that needs "is this a person" it is worse.
- The half-gate gain of bigJ repeats in a second, independent run (one corner 0.545, a gate reported in 87% of the crops); the no-corner case stays near 0.21–0.23.
- The lane result of bigI repeats (TuSimple 0.836, BDD proxy 0.37).
- red full-frame 0.962 / 0.878 is the low end of the seed range of this recipe family (0.962–0.990 / 0.872–0.929 over bigH x2, bigI, bigJ, bigK); no evidence that the half-gate crops cost full-gate accuracy, no evidence that they do not.

Next (bigL, started 13:15): same as bigK with person class weight 2 -> 4 (to win back people from the vehicle class) and `half_gate_prob` 0.25 -> 0.4. Two changes at once; they act on different heads and are read from different metrics, but an interaction cannot be excluded.

## Update 62 (2026-11-16) — bigL: person weight 4, half-gate share 0.4

bigL = bigK with `seg_class_weights.person` 2 -> 4 and `half_gate_prob` 0.25 -> 0.4 (label sets v3 + v4 + v5), vit, 10 epochs, one seed.

People vs two-wheelers on IDD-AW (zero-shot, share of raw pixels per predicted class):

| raw pixels | bigH | bigI | bigK | bigL |
|---|---|---|---|---|
| motorcycle + bicycle -> car / person | 0.39 / 0.21 | 0.39 / 0.17 | 0.75 / 0.06 | 0.61 / 0.19 |
| rider -> person / car | 0.40 / 0.39 | 0.47 / 0.32 | 0.37 / 0.46 | 0.47 / 0.39 |
| pedestrian -> person / car | 0.54 / 0.20 | 0.57 / 0.20 | 0.47 / 0.29 | 0.63 / 0.18 |
| eval_iddaw5 person IoU | – | 0.360 | 0.318 | 0.304 |

Half gates on red (crop proxy, one seed each):

| | before (5 models) | bigJ (0.25) | bigK (0.25) | bigL (0.4) |
|---|---|---|---|---|
| 1 corner: found < 0.25 size | 0.22–0.40 | 0.495 | 0.545 | 0.526 |
| 1 corner: no prediction | 0.31–0.55 | 0.166 | 0.126 | 0.034 |
| 0 corners: found < 0.25 size | 0.00–0.05 | 0.232 | 0.212 | 0.394 |
| 0 corners: found < 0.5 size | 0.15–0.23 | 0.455 | 0.414 | 0.616 |
| 0 corners: no prediction | 0.71–0.76 | 0.384 | 0.465 | 0.182 |
| predictions per crop matching no gate | 0.03–0.04 | 0.086 | 0.067 | 0.198 |

Other metrics of bigL (one seed; gates ±0.06): red full frames 0.974 / 0.885 (<20 / <10 px), partly visible n=34 0.912, false gates BDD / ADE 0.00 / 0.00, seg-head false gate pixel share red 0.0023; eval_bdd5 0.743, eval_ade5 0.608, eval_iddaw5 0.584, eval_rugd_heldout3 0.512, UAVid 0.624 (person 0.111, was 0.07–0.08), SUN person IoU 0.43 (bigH 0.30), Aeroscapes mIoU 0.668; TuSimple lane recall 0.834; BDD lane proxy 0.373 (circular), plain road as lane 0.017; KITTI inverse fit 0.080; DIODE inverse fit 0.114 / 0.247. Reference only: TII golden (hall seen) 0.927; sim_racec full view <20 px 0.797 (sim-to-sim); sim stress worst mix 0.62.

Reading:
- Person weight is a dial, not a fix: with weight 4 pedestrians are best so far (0.63 called person, 0.18 called car) but motorcycles painted as person go back to 0.19 (bigK 0.06, bigH 0.21), and person IoU on IDD-AW does not improve (more person false positives). bigK and bigL are two ends of the same trade-off; the labels (rider = person on top of a vehicle) make a clean separation hard at 448 px. Person on indoor frames (SUN 0.43) and from height (UAVid 0.11) improves with the higher weight.
- Half gates: the larger share helps the bar-only case clearly (0.21–0.23 -> 0.39, a gate reported in 82% of the crops) and almost removes "no prediction" for one-corner gates (0.03); placement of one-corner gates stays at about 0.5. Cost: spurious predictions in these crops double to 0.20 per crop. Gate-free real sets stay at 0.00 and full-frame red is unchanged, so the spurious ones come with partial views (other cut gates, second guesses for the same gate); not broken down yet.
- Everything else is within seed noise of bigI / bigK.

Next: the hybrid with the bigL recipe (bigL_h, started 14:36) for the Orin, then export and report update.

Addendum to Update 62 — what the spurious half-gate predictions are. `eval_half_gates.py` now also counts how many of the unmatched predictions are a rough guess of a labelled gate (4-corner error between 0.5 and 1.5 gate sizes of any labelled gate, in view or cut off): bigH 0.031 of 0.031 per crop, bigK 0.067 of 0.067, bigL 0.196 of 0.198. So nearly all of them are badly placed estimates of a real gate, not gates invented on background; this agrees with 0.00 false gates on the gate-free real sets. The practical cost is a wrong gate pose for a cut-off gate, not a phantom gate.

## Update 63 (2026-11-16) — bigL_h: hybrid with the bigL recipe (Orin candidate)

bigL_h = bigL recipe (label sets v3 + v4 + v5, `half_gate_prob: 0.4`, person weight 4, `depth_gt_space: inverse`) with `KIND=hybrid` (FastViT-SA24, 11.0 M), 10 epochs, one seed.

| metric (one seed; gates ±0.06) | bigH_h (before) | bigL_h | bigL (vit) |
|---|---|---|---|
| red (unseen place) gates <10 / <20 px | 0.907 / 0.965 | 0.913 / 0.971 | 0.885 / 0.974 |
| red median corner error, px | 2.5 | 2.5 | 4.4 |
| half gate, 1 corner: found <0.25 size / no prediction | 0.316 / 0.450 | 0.713 / 0.079 | 0.526 / 0.034 |
| half gate, 1 corner: visible corner median error, px | 15.9 | 12.1 | 21.6 |
| half gate, 0 corners: found <0.25 size / no prediction | 0.040 / 0.707 | 0.273 / 0.414 | 0.394 / 0.182 |
| unmatched predictions per half-gate crop (rough guesses of a real gate) | 0.044 | 0.102 (0.098) | 0.198 (0.196) |
| false gates BDD / ADE gate-free | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| eval_iddaw5 mIoU (zero-shot) / person IoU | – | 0.608 / 0.387 | 0.584 / 0.304 |
| eval_bdd5 / eval_ade5 mIoU | – | 0.727 / 0.583 | 0.743 / 0.608 |
| eval_rugd_heldout3 mIoU | 0.516 | 0.511 | 0.512 |
| UAVid full mIoU | 0.580 | 0.585 | 0.624 |
| SUN person IoU | 0.346 (bigG_h) | 0.491 | 0.43 |
| TuSimple lane recall | 0.772 | 0.801 | 0.834 |
| BDD lane proxy (circular) / plain road as lane | 0.004 / 0.0002 | 0.321 / 0.021 | 0.373 / 0.017 |
| KITTI zero-shot AbsRel inverse fit | 0.082 | 0.082 | 0.080 |
| DIODE inverse fit indoor / outdoor | 0.124 / 0.264 | 0.133 / 0.264 | 0.114 / 0.247 |
| IDD-AW pedestrian -> person / car | – | 0.69 / 0.19 | 0.63 / 0.18 |
| IDD-AW motorcycle + bicycle -> car / person | – | 0.63 / 0.23 | 0.61 / 0.19 |

Reference only: TII golden (hall seen) <20 px 0.922; sim_racec full view <20 px 0.813, partly out 0.884 (sim-to-sim); sim stress worst mix recall 0.62.

Reading:
- The small model takes over all three changes (half gates, road paint, two-wheelers / person weight) and loses nothing on full gates (0.971 / 0.913) or false gates (0.00).
- On one-corner half gates the hybrid is better than the vit of the same recipe (0.71 vs 0.53 within a quarter of the gate size, visible corner 12 vs 22 px); on bar-only gates the vit is better (0.39 vs 0.27). One seed each and a crop proxy, so the ranking between the two is not established; a second hybrid seed (bigL_h_s1) was started at 16:05. The lower corner error of the hybrid on full gates (2.5 vs about 4 px) has now shown in three recipe pairs.
- The vit stays ahead on aerial views, lane recall and BDD / ADE segmentation by 0.02–0.04.

Export: `runs/bigL_h/hybrid/model_448.onnx` (31.2 MB), `model_448_fp16.onnx` (15.7 MB), `model_448_fp16.engine` (TensorRT 11.4 on an idle RTX 5090: 0.71 ms median, network only; 24/24 check frames same gate set, max corner diff 0.24 px, seg same-class share >= 0.997). Also `runs/bigL/vit/` (engine 1.16 ms, 24/24). Jetson Orin still not measured (device not available). Sheets: `runs/eval/vis/model_sheet_bigL_h_hybrid{,_env2}.jpg`, `runs/eval/vis/half_gates_bigH_h_bigL_h.jpg`.

bigL_h is the current candidate for the drone; bigL is the current best large model.

## Update 64 (2026-11-16) — bigL_h second seed

Same recipe as bigL_h, `seed: 1` (bigL_h_s1). Seed 0 / seed 1:

| metric | seed 0 | seed 1 |
|---|---|---|
| red (unseen place) gates <10 / <20 px | 0.913 / 0.971 | 0.881 / 0.952 |
| red median corner error, px | 2.5 | 2.6 |
| half gate, 1 corner: found <0.25 size / <0.5 size / no prediction | 0.713 / 0.868 / 0.079 | 0.768 / 0.897 / 0.066 |
| half gate, 1 corner: visible corner median error, px | 12.1 | 11.2 |
| half gate, 0 corners: found <0.25 size / no prediction | 0.273 / 0.414 | 0.273 / 0.434 |
| unmatched predictions per half-gate crop | 0.102 | 0.086 |
| false gates BDD / ADE gate-free | 0.00 / 0.00 | 0.00 / 0.00 |
| eval_iddaw5 (zero-shot) / eval_bdd5 / eval_ade5 mIoU | 0.608 / 0.727 / 0.583 | 0.605 / 0.729 / 0.591 |
| eval_rugd_heldout3 / UAVid mIoU | 0.511 / 0.585 | 0.508 / 0.587 |
| TuSimple lane recall | 0.801 | 0.784 |
| BDD lane proxy (circular) / plain road as lane | 0.321 / 0.021 | 0.315 / 0.020 |
| KITTI inverse fit | 0.082 | 0.080 |
| IDD-AW pedestrian -> person / car | 0.69 / 0.19 | 0.70 / 0.14 |
| IDD-AW motorcycle + bicycle -> car / person | 0.63 / 0.23 | 0.59 / 0.21 |

Reading: the half-gate result of the hybrid is not a seed accident: one-corner gates 0.71 / 0.77 within a quarter of the gate size (hybrid before the change: 0.32), bar-only gates 0.27 in both seeds. Segmentation, lane, depth and the person / two-wheeler split repeat within 0.01–0.02. Full-gate recall on red is 0.952–0.971 / 0.881–0.913 for this recipe; the hybrids without half-gate crops had 0.958–0.965 / 0.885–0.907 (bigE_h, bigF_h, bigG_h, bigH_h), so no measurable cost on full gates.

Next (bigM_h, started 17:35): same recipe with `half_gate_zero_share: 0.7` (new config key: share of the half-gate crops that leave no corner in view; default 0.5) to lift the bar-only case, which is the weak one for the hybrid.

## Update 65 (2026-11-16) — direction change; video consistency on real flights; Orin correction

Direction chosen by the user (2026-11-16 evening): (1) freeze a release model, (2) measure temporal behaviour on video, (5) closed-loop simulation test; training continues after these tests. Hardware from the user's parts list: camera IMX219 CSI ("Jetson Nano camera"; lens variant / field of view not known yet, the stock one is about 62 deg horizontal, the sim data uses 130 deg; rolling shutter), 5-inch frame, MicoAir743 flight controller, TF02-Pro range sensor. Camera and Orin are not at hand at the moment.

Correction: in earlier updates and in the report I wrote that the Orin was never measured. `docs/ORIN_RESULTS.md` (2026-11-10) has Orin NX 8 GB latencies (random weights, TensorRT fp16, GPU compute only): hybrid reparameterized 448 px 16.7 ms (60 FPS) in MAXN, 384 px 14.6 ms MAXN / 26.4 ms at 15 W; ViT-B 448 px 25.2 ms MAXN. What is still not measured on the Orin: accuracy of the trained weights, the current head set, camera input and pre/post-processing.

Video consistency (`scripts/eval_video_consistency.py`). TII flights 13, 16, 18: about 120 frames/s, every frame labelled, 16,132 frames, 34,592 gate-frames (gates with >= 2 visible corners), 519 gate tracks. These flights were never trained on but the hall was (other flights), so this is frame-to-frame behaviour, NOT generalisation to a new place. A gate-frame is found when a prediction (score > 0.5) is within 20 px (256x192) on the visible corners.

| | bigH_h | bigL_h | bigL (vit) |
|---|---|---|---|
| recall per gate-frame | 0.905 | 0.903 | 0.900 |
| fully visible / partly visible (2–3 corners, n=18,065) | 0.898 / 0.911 | 0.893 / 0.913 | 0.894 / 0.906 |
| single-frame dropouts per 1000 gate-frames | 7.0 | 6.7 | 9.5 |
| share of gate-frames in miss runs >= 5 frames (40 ms) / >= 12 frames (100 ms) | 0.062 / 0.038 | 0.066 / 0.040 | 0.060 / 0.030 |
| recall if a detection is held for up to 3 frames | 0.938 | 0.937 | 0.940 |
| corner jitter px, median / p90 (motion removed) | 0.82 / 2.94 | 0.84 / 3.03 | 0.94 / 3.43 |
| recall in the last 12 frames (100 ms) of a gate track | 0.766 | 0.803 | 0.769 |
| unmatched predictions per frame (includes unlabelled gates with < 2 visible corners) | 0.172 | 0.205 | 0.174 |

By gate size (largest side of the visible corners, px at 256x192; every 2nd frame):

| size | gate-frames | bigH_h recall <20 px / within max(20 px, size/4) | bigL_h recall <20 px / within max(20 px, size/4) | bigL_h share of the long misses |
|---|---|---|---|---|
| < 8 px | 55 | 0.13 / 0.13 | 0.38 / 0.38 | 0.07 |
| 8–16 px | 813 | 0.56 / 0.56 | 0.58 / 0.58 | 0.27 |
| 16–40 px | 9,021 | 0.925 / 0.925 | 0.919 / 0.919 | 0.44 |
| 40–100 px | 6,968 | 0.955 / 0.957 | 0.959 / 0.963 | 0.11 |
| > 100 px (close pass) | 446 | 0.43 / 0.66 | 0.40 / 0.69 | 0.12 |

Reading:
- Frame to frame the detector is calm: median corner jitter under 1 px, isolated one-frame dropouts under 1%. A 3-frame hold lifts recall from 0.90 to 0.94, so a simple tracker is worth having.
- Two real weak spots: far gates under 16 px (0.4–0.6) and close gates over 100 px (0.40 within 20 px, 0.69 within a quarter of the gate size). 3–4% of all gate-frames sit in gaps of 100 ms or longer; 44% of those are mid-size gates (16–40 px), not yet explained (occlusion by nearer gates is a candidate; not checked).
- The half-gate training (bigL_h) does NOT show a gain on real close gates here: 0.66 -> 0.69 (size-relative) and 0.43 -> 0.40 (20 px), against 0.32 -> 0.71 on the crop proxy of Update 60–64. The two tests are not the same case (TII labels only contain gates with >= 2 visible corners, the proxy is 0–1 corners, and TII partly visible gates were already in the training data), but the proxy gain must not be read as "close passes are solved". The end-of-track recall moves 0.77 -> 0.80 only.
- The hybrid is at least as steady as the vit (fewer single-frame dropouts, less jitter).

## Update 66 (2026-11-16) — gate position in metres from the predicted corners (sim, step towards the closed-loop test)

`scripts/eval_gate_pose_sim.py`: for every labelled gate of the held-out sim racing set (sim_racec, 1,110 frames; SIM-TO-SIM, not a real-world number) the 4 predicted corners go through PnP (known 1.5 m inner square, calibrated pinhole camera, IPPE) and are compared with the same PnP on the true corners. A gate counts when the best prediction is within max(20 px, size/4). No 3–6 m bin (fewer than 10 gates in this set).

| distance | gates | bigL_h found | bigL_h position error median / p90 | sideways / along view (median) | gate normal error (median) | bigL (vit) position error median |
|---|---|---|---|---|---|---|
| 0–3 m | 392 | 0.977 | 0.07 m / 0.17 m (3.7%) | 0.05 / 0.04 m | 7.5 deg | 0.09 m |
| 6–12 m | 139 | 0.957 | 0.22 m / 0.60 m (2.0%) | 0.12 / 0.18 m | 18.7 deg | 0.30 m |
| 12–25 m | 895 | 0.964 | 0.83 m / 3.1 m (5.0%) | 0.45 / 0.68 m | 24.1 deg | 1.03 m |
| > 25 m | 1,518 | 0.947 | 6.0 m / 26.8 m (15%) | 3.3 / 4.7 m | 33.0 deg | 7.4 m |

Reading (for the sim camera, 130 deg at 256x192, where a 1.5 m gate at 25 m is about 4 px wide):
- Inside 3 m the position is good to 7 cm (p90 17 cm) against a 1.5 m opening; at 6–12 m to about 0.2 m. Beyond 12 m single-frame PnP is only a rough bearing (0.8 m median, 3 m at p90) and beyond 25 m it is unusable as a position; far gates should be used as direction only or fused over time.
- The gate normal from a single frame is poor at every distance (7.5 deg close, 19–33 deg farther): with a wide lens the square is small and the two IPPE solutions are close. Approach direction should come from the track layout or from filtering, not from one frame.
- The hybrid is a little more accurate than the vit at every distance (same pattern as its lower corner error on real gates).
- A narrower lens (the stock IMX219 is about 62 deg) would roughly double the pixels per gate and so the usable range, at the price of losing gates from view earlier in a close pass.

## Update 67 (2026-11-16) — bigM_h: more bar-only crops (half_gate_zero_share 0.7)

bigM_h = bigL_h recipe with the new key `half_gate_zero_share: 0.7` (70% of the half-gate crops leave no corner in view; bigL_h 50%). Hybrid, 10 epochs, one seed. Test added earlier still passes (60).

| metric | bigL_h seed 0 / seed 1 | bigM_h |
|---|---|---|
| half gate, 0 corners: found <0.25 size / <0.5 size / no prediction | 0.273, 0.273 / 0.455, 0.414 / 0.414, 0.434 | 0.343 / 0.505 / 0.293 |
| half gate, 1 corner: found <0.25 size / no prediction | 0.713, 0.768 / 0.079, 0.066 | 0.734 / 0.082 |
| unmatched predictions per half-gate crop | 0.102, 0.086 | 0.154 |
| red full frames <10 / <20 px | 0.913, 0.881 / 0.971, 0.952 | 0.904 / 0.958 |
| false gates BDD / ADE | 0.00 / 0.00 | 0.00 / 0.00 |
| eval_iddaw5 mIoU | 0.608, 0.605 | 0.607 |
| TuSimple lane recall | 0.801, 0.784 | 0.796 |

Reading: the bar-only case moves 0.27 -> 0.34 (7 of 99 crops), the one-corner case and full gates are unchanged, unmatched rough guesses rise to 0.15 per crop. That is inside what one seed and n=99 can produce; not a clear win. bigL_h stays the frozen candidate unless the video and closed-loop tests separate the two.

## Update 68 (2026-11-16) — closed-loop gate passing in simulation, first batch (bigL_h)

New: `sim/isaac_race_loop.py` (the race scene of `isaac_race.py`, rendering on request through files in `runs/race_job/loop`), `scripts/closed_loop_race.py` (drone + guidance + perception), `scripts/closed_loop_batch.sh`. Isaac note: with a remote-desktop session open, Isaac Sim hangs at start when DISPLAY / WAYLAND_DISPLAY are set; start it with `env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY` (the batch script does).

Setup. THIS IS A SIMULATION RESULT, not a real-world success rate: scenes rendered by the same renderer as the training data (the 6 track layouts c_race_0..5 are held out, 44 gates), point-mass drone at constant target speed, acceleration limit 20 m/s^2, 60 Hz, the drone knows its own pose exactly, no wind, no rolling shutter, pinhole 130 deg camera at 640x480. The drone has a rough gate map (each centre off by 1 m std per axis, exact passage direction) and flies to a point in front of the gate, then through. A gate is passed when the plane is crossed inside the 1.5 m opening with 0.25 m margin; inside the 2.7 m frame but outside that = hit; else missed. Variants: oracle (true centres), map (rough map only), percept (map corrected every frame by the model: predicted corners -> PnP -> world position, used when closer than 12 m and within 3 m of the current estimate), percept_delay (measurement used one frame late).

| variant | speed | flights | gates passed | hit | missed | gate position error at crossing (median) |
|---|---|---|---|---|---|---|
| oracle | 8 / 12 m/s | 18 / 18 | 132/132 / 132/132 | 0 | 0 | 0 |
| map only | 8 / 12 m/s | 18 / 18 | 40/132 (0.30) / 42/132 (0.32) | 76 / 75 | 16 / 15 | 0.93 m |
| percept (bigL_h) | 8 m/s | 12 | 87/88 | 1 | 0 | 0.16 m |
| percept (bigL_h) | 12 m/s | 12 | 87/88 | 1 | 0 | 0.15 m |
| percept_delay (bigL_h) | 12 m/s | 6 | 44/44 | 0 | 0 | 0.15 m |

Reading: in this easy setting the perception model closes almost the whole gap between a 1 m map (30% of the gates) and perfect knowledge (100%): 218 of 220 gates over 30 flights, two frame hits. One frame of latency makes no difference at 12 m/s. The setting is easy on purpose (clean images, known own pose); harder runs (20 and 30 m/s, 2 m map error, dark + blurred images) were started at 20:26.

## Update 69 (2026-11-16) — closed loop, harder settings (bigL_h); release v1 frozen

Same setup as Update 68 (SIMULATION, 6 held-out track layouts, 44 gates, point-mass drone that knows its own pose). Passed gates / all gates:

| setting | map only | map + model (percept) | percept, one frame late | oracle (true gate centres) |
|---|---|---|---|---|
| clean, 8 m/s, map error 1 m | 40/132 (0.30) | 87/88 (0.99) | – | 132/132 |
| clean, 12 m/s, map error 1 m | 42/132 (0.32) | 87/88 (0.99) | 44/44 | 132/132 |
| clean, 12 m/s, map error 2 m (association radius 6 m) | 10/132 (0.08) | 87/88 (0.99) | 44/44 | 132/132 |
| dark + blurred image, 12 m/s | 42/132 | 88/88 (1.00) | – | 132/132 |
| clean, 20 m/s | 39/132 (0.30) | 47/88 (0.53) | – | 66/132 (0.50) |
| dark + blurred, 20 m/s | 39/132 | 43/88 (0.49) | 24/44 (0.55) | 66/132 (0.50) |
| clean, 30 m/s | 26/132 (0.20) | 23/88 (0.26) | 9/44 (0.20) | 45/132 (0.34) |

Gate position error at the crossing (median): 0.14–0.19 m up to 20 m/s; 0.58 m at 30 m/s (1.0 m with one frame of delay). "dark + blurred" = quarter exposure + noise and a box blur that grows with speed (crude, not the `pfm.effects` model).

Reading:
- Up to 12 m/s the model in the loop passes 99–100% of the gates in every setting tried (349 of 352 over 48 flights, plus 88 of 88 with one frame of latency), including a 2 m map error where the map alone passes 8%, and dark + blurred images. Within this simulation perception is not the limiting part at these speeds.
- At 20 and 30 m/s the test says nothing about perception: the simple guidance (constant speed, 20 m/s^2, no braking before turns) passes only 50% / 34% with perfect gate positions, and the model in the loop is at the same level (53% / 26%). At 30 m/s the gate estimate at the crossing is also worse (0.6 m; 1.0 m with one frame of delay), so perception will matter there too, but a better controller is needed before that can be measured.
- The two hits of Update 68 are the same gate (c_race_4, gate 9, map-error seed 1) at both speeds; not analysed further.
- Limits of this test: same renderer as the training data, clean geometry, exact own pose, no rolling shutter, 130 deg pinhole camera. It shows that corner errors of this size do not break gate passing; it does not show real-world performance.

Release v1: `runs/release_v1/` (bigL_h weights, fp32 / fp16 ONNX with their check files, training config, SHA256SUMS, README with usage, measured numbers and what is not measured). bigM_h was not chosen (Update 67: no clear gain).

What the three tests say for the next training steps: (1) real close gates (> 100 px: 0.40 within 20 px on TII video) and far gates (< 16 px: 0.4–0.6) are the weak points of the detector on real video; (2) the half-gate crop proxy does not predict the real close-pass case; (3) frame-to-frame behaviour and closed-loop passing up to 12 m/s are fine.

## Update 70 (2026-11-16) — report second edition, videos

- `scripts/make_report_v2_tr.py` -> `docs/report/PerceptionModel_rapor_v2.pdf` (12 pages, Turkish): rewritten around release v1 with a one-page summary, gate results on the unseen place, real-video behaviour, the half-gate proxy with its caveat, the closed-loop simulation (labelled as simulation), general perception, limits and next steps. It also corrects the earlier report's "Orin never measured". The first-edition generator (`make_report_tr.py`) is kept.
- Videos in `runs/eval/video/`: `bigL_h_tii_ucus13_tur_60fps.mp4` (release model on TII flight 13, lap frames; hall seen in training, flight not) and `bigL_h_kapali_dongu_sim_c_race_3.mp4` (`scripts/make_closed_loop_video.py`: one closed-loop simulation flight at 12 m/s, camera frame with predicted gates next to the top view; 8 of 8 gates). Picture `runs/eval/vis/closed_loop_c_race_0.jpg` (`scripts/vis_closed_loop.py`).
- `closed_loop_race.py`: `TRAJ=<file>` saves the path, `SNAP=<dir>` / `SNAP_EVERY` save camera frames with the predicted gates.
- Process mistakes of this step, for the record: (1) a controller started with `&` inside a tool call kept running and a second one was started on the same renderer, so two controllers shared one request folder; (2) `pgrep -f closed_loop_race.py` matched my own shell and killed it (the rule about pkill -f applies to pgrep -f + kill too). Use tmux sessions and `tmux kill-session`, and select processes with `ps -C python`. No data was lost; the flight was repeated cleanly.

## Update 71 (2026-11-17) — bigN_h: gate centre from the completed square (`gate_centre_complete: true`)

Change against bigL_h: one flag. For a gate with 2–3 known corners the centre/size target of the dense head is taken from the completed square (`complete_square`), not from the visible corners only. Same data, labels (v5), half-gate crops (0.4), class weights, hybrid student, 10 epochs, one seed (gate numbers +-0.06).

Never-seen place (MonoRaceGate red, 196 frames, 312 gates) and the other usual checks:

| | bigL_h (two seeds) | bigN_h |
|---|---|---|
| red recall <20 px / <10 px | 0.971 / 0.913 and 0.952 / 0.881 | 0.962 / 0.904 |
| red false predictions per frame | 0.00 | 0.00 |
| half-gate crops on red, 1 corner in view, found <0.25 size | 0.71 / 0.77 | 0.68 |
| half-gate crops on red, 0 corners in view, found <0.25 size | 0.27 / 0.27 | 0.24 |
| sim_racec full view <20 px (sim, not a success rate) | 0.813 | 0.802 |
| IDD-AW (zero-shot) mIoU / pixel acc | 0.608 / 0.868 | 0.611 / 0.871 |
| TuSimple lane recall / precision | 0.80 / 0.96 | 0.80 / 0.96 |
| false gate on BDD / ADE outdoor (score > 0.5) | 0.00 / 0.00 | 0.00 / 0.00 |

All of these are inside one-seed noise: no change on the never-seen place.

Real video (TII flights 13/16/18, hall seen in training, flights not; so not a generalisation number). Both models run with STEP=2 here (8,066 frames, 17,303 gate-frames); the bigL_h column of Update 66 was STEP=1, so time-based rows differ from that table.

| | bigL_h | bigN_h |
|---|---|---|
| recall per gate-frame | 0.904 | 0.921 |
| fully / partly visible (2–3 corners, n=9,042) | 0.893 / 0.914 | 0.896 / 0.944 |
| single-frame dropouts per 1000 | 7.5 | 5.7 |
| share in miss runs >= 5 / >= 12 frames | 0.051 / 0.018 | 0.048 / 0.019 |
| recall with 3-frame hold | 0.943 | 0.946 |
| recall in last 12 frames of a track | 0.899 | 0.895 |
| corner jitter median / p90 px | 0.94 / 3.54 | 0.89 / 3.45 |
| unmatched predictions per frame | 0.203 | 0.231 |
| size < 8 px (n=55) | 0.38 | 0.29 |
| size 8–16 px (n=813) | 0.58 | 0.65 |
| size 16–40 px (n=9,021) | 0.919 | 0.955 |
| size 40–100 px (n=6,968) | 0.959 | 0.941 |
| size > 100 px (n=446), <20 px / within max(20 px, size/4) | 0.40 / 0.69 | 0.50 / 0.73 |

Reading:
- The flag does what it was meant for on partly visible gates in real video: 0.914 -> 0.944, and the close-pass bin 0.40 -> 0.50. The close-pass bin is 446 gate-frames from a few dozen passes, neighbouring frames are strongly correlated, so the real uncertainty is much larger than the count suggests.
- It is not free: 40–100 px drops 0.959 -> 0.941 and unmatched predictions rise 0.203 -> 0.231.
- Nothing moved on the never-seen place, and the half-gate crop proxy is slightly lower (inside noise).
- Verdict: small gain on seen-hall video, one seed, not confirmed. bigL_h stays the release (release_v1). bigN_h is a candidate only after a second seed repeats the partly-visible gain.
- The close-pass problem is still open: half of the > 100 px gates are still off by more than 20 px. Far gates (< 16 px) are still 0.3–0.65.

Far-gate diagnostic after Update 71 (bigL_h, TII video, STEP=4, no retraining; TII frames are 640x480, the model sees them squeezed to 448x448):

| inference | 8–16 px (n=411) | 16–40 px | 40–100 px | > 100 px | unmatched/frame |
|---|---|---|---|---|---|
| normal, 448 | 0.58 | 0.919 | 0.960 | 0.40 | 0.200 |
| + second pass on the centre half of the frame (2x zoom, `ZOOM=1`) | 0.66 | 0.920 | 0.960 | 0.40 | 0.207 |
| input 640 (`SIZE=640`, model never trained at this size) | 0.72 | 0.932 | 0.913 | 0.13 | 0.258 |

More pixels help far gates by +0.08 to +0.14, not more; in simulation the same size bin is found 0.96 of the time, so most of the real far-gate loss is not resolution but real-footage appearance (blur, light, few small real gates in training). Next runs therefore: `zoom_out_prob: 0.3` (new, shrinks gate frames 0.5–1x so real gates appear small; test added, 62 tests pass) as bigO_h, and bigN_h2 = repeat of bigN_h for the seed question. Two hybrid trainings do not fit the GPU together (15.5 GB each + 1 GB remote desktop -> out of memory), so bigO_h is queued behind bigN_h2. `bige_chain.py` now relinks labels atomically (two chains preparing the same label folder collided once).

## Update 72 (2026-11-17) — bigN_h repeated (bigN_h2): the video gain of `gate_centre_complete` holds

bigN_h2 = the bigN_h recipe run again. Caveat: neither bigN yaml sets `seed`, so both use the default seed; the two runs still differ (data-loader order / GPU non-determinism), but this is a repeat, not an independent seed like bigL_h vs bigL_h_s1 (`seed: 1`). I also ran the video check for bigL_h_s1 (STEP=2) so each recipe has two runs.

| | bigL_h | bigL_h_s1 | bigN_h | bigN_h2 |
|---|---|---|---|---|
| red (never seen) <20 / <10 px | 0.971 / 0.913 | 0.952 / 0.881 | 0.962 / 0.904 | 0.971 / 0.910 |
| half-gate crops on red: 1 corner / 0 corners, found <0.25 size | 0.71 / 0.27 | 0.77 / 0.27 | 0.68 / 0.24 | 0.73 / 0.27 |
| sim_racec full view <20 px (sim) | 0.813 | – | 0.802 | 0.803 |
| IDD-AW mIoU | 0.608 | – | 0.611 | 0.612 |
| TII video (hall seen): recall per gate-frame | 0.904 | 0.899 | 0.921 | 0.923 |
| – partly visible gates | 0.914 | 0.911 | 0.944 | 0.942 |
| – single-frame dropouts / 1000 | 7.5 | 9.2 | 5.7 | 6.8 |
| – 3-frame hold | 0.943 | 0.938 | 0.946 | 0.950 |
| – last 12 frames of a track | 0.899 | 0.890 | 0.895 | 0.906 |
| – unmatched predictions / frame | 0.203 | 0.202 | 0.231 | 0.229 |
| – 8–16 px | 0.58 | 0.65 | 0.65 | 0.64 |
| – 16–40 px | 0.919 | 0.917 | 0.955 | 0.956 |
| – 40–100 px | 0.959 | 0.947 | 0.941 | 0.942 |
| – > 100 px, <20 px / size-relative | 0.40 / 0.69 | 0.29 / 0.57 | 0.50 / 0.73 | 0.53 / 0.76 |

Reading:
- On real video both bigN runs beat both bigL_h runs on partly visible gates (+0.03), on 16–40 px gates (+0.04) and on close passes (0.29–0.40 -> 0.50–0.53). The gap is larger than the run-to-run spread, so the flag is kept in the recipe.
- Cost: about +0.03 unmatched predictions per frame; 40–100 px is equal within spread.
- Never-seen place: no difference (all four runs 0.95–0.97 / 0.88–0.91). So this is a gain in frame-to-frame behaviour in a seen hall, not a generalisation gain.
- Close passes are still only half right at 20 px. release_v1 (bigL_h) stays frozen for now; the next release decision comes after bigO_h (same recipe + `zoom_out_prob: 0.3`, training now).

## Update 73 (2026-11-17) — bigO_h: zoom-out augmentation (`zoom_out_prob: 0.3`) — no gain

bigO_h = bigN_h recipe + 30% of gate frames shrunk to 0.5–1x (edge pixels repeated around). One run.

| | bigN_h / bigN_h2 | bigO_h |
|---|---|---|
| red (never seen) <20 / <10 px | 0.962 / 0.904, 0.971 / 0.910 | 0.962 / 0.891 |
| half-gate crops on red: 1 corner / 0 corners, found <0.25 size | 0.68 / 0.24, 0.73 / 0.27 | 0.73 / 0.29 |
| TII video (hall seen): recall / partly visible | 0.921 / 0.944, 0.923 / 0.942 | 0.921 / 0.944 |
| – < 8 px (n=55) | 0.29, 0.38 | 0.20 |
| – 8–16 px (n=813) | 0.65, 0.64 | 0.65 |
| – 16–40 / 40–100 px | 0.955 / 0.941, 0.956 / 0.942 | 0.957 / 0.939 |
| – > 100 px, <20 px / size-relative | 0.50 / 0.73, 0.53 / 0.76 | 0.48 / 0.76 |
| – unmatched predictions / frame | 0.231, 0.229 | 0.225 |
| sim_racec / sim_racep full view <20 px (sim) | 0.802 / 0.789, 0.803 / 0.773 | 0.812 / 0.812 |
| sim_racec 0–4 px / 4–8 px found (thr 0.5) | 0.61 / 0.86, 0.63 / 0.87 | 0.61 / 0.87 |
| IDD-AW mIoU, TuSimple lane recall | 0.611–0.612, 0.80 | 0.612, 0.79 |

Reading: the far-gate bins did not move (8–16 px 0.65, < 8 px inside noise of n=55), nothing else moved either. Shrinking existing frames does not supply what the model lacks for real far gates. `zoom_out_prob` stays in the code (default 0) but is not part of the recipe. Current recipe = bigN_h.

## Update 74 (2026-11-17) — the "far gate" loss on real video was mostly the 4-gates-per-frame cap of the decoder

Looking at the misses instead of training more (`scripts/vis_far_gate_misses.py`, `scripts/diag_video_small_gate_matching.py`, pictures `runs/eval/vis/far_gate_misses_bigN_h2.jpg`, `far_gate_twin_cases_bigN_h2.jpg`):
- In the 8–16 px bin of the TII video only 7.5% of labels have no prediction within 20 px. Another 30% "lose" because the nearest prediction belongs to a neighbour gate about 14 px away. The pictures show two clearly visible small gates side by side, one found, one not.
- Lower score threshold does not bring them back (8–16 px: 0.64 at 0.5, 0.65 at 0.3, 0.66 at 0.15). The heat map has the peak; the decoder returns only the 4 best gates per frame (`n_slots = 4`) and TII frames often show more than 4 gates, so the weakest (usually the smallest) are cut.
- New environment key `PFM_GDET_K` (default 4 = old behaviour) sets the number of gates returned. No retraining.

TII video (hall seen, flights not; STEP=2, 17,303 gate-frames), same weights, K = 4 -> 8 (K = 12 gives the same as 8):

| | bigL_h K=4 | bigL_h K=8 | bigN_h2 K=4 | bigN_h2 K=8 |
|---|---|---|---|---|
| recall per gate-frame | 0.904 | 0.928 | 0.923 | 0.946 |
| fully / partly visible | 0.893 / 0.914 | 0.923 / 0.932 | 0.901 / 0.942 | 0.940 / 0.952 |
| single-frame dropouts / 1000 | 7.5 | 6.3 | 6.8 | 3.8 |
| share in miss runs >= 12 frames | 0.018 | 0.006 | 0.021 | 0.013 |
| 3-frame hold | 0.943 | 0.959 | 0.950 | 0.964 |
| last 12 frames of a track | 0.899 | 0.926 | 0.906 | 0.948 |
| unmatched predictions / frame | 0.203 | 0.210 | 0.229 | 0.235 |
| < 8 px (n=55) | 0.38 | 0.49 | 0.38 | 0.38 |
| 8–16 px (n=813) | 0.58 | 0.80 | 0.64 | 0.84 |
| 16–40 px | 0.919 | 0.939 | 0.956 | 0.966 |
| 40–100 px | 0.959 | 0.965 | 0.942 | 0.963 |
| > 100 px | 0.40 | 0.42 | 0.53 | 0.54 |

Other checks with K=8 (bigN_h2 / bigL_h): red (never seen) 0.974 / 0.913 and 0.971 / 0.913 (<20 / <10 px), unchanged (those frames hold 1–2 gates); red unmatched predictions per frame 0.02 / 0.00; sim_racec unchanged (0.803 / 0.813); 62 tests pass.

Reading:
- The earlier statement "gates under 16 px are found 0.4–0.6 of the time on real video" (Updates 66, 71, report v2) was mostly a decoder limit, not a perception limit: with 8 outputs it is 0.80–0.84. The far-gate diagnostic after Update 71 (zoom / 640 input) and the zoom-out run (Update 73) were chasing the wrong cause.
- The 40–100 px drop of bigN against bigL_h (0.94 vs 0.96) also disappears with K=8.
- Still weak and real: gates under 8 px (0.4, n=55) and close passes over 100 px (0.42–0.54 within 20 px).
- Export: bigN_h2 with K=8 -> `runs/bigN_h2/hybrid/model_448.onnx` (gates output 1x8x9), fp16 ONNX, TensorRT engine 0.81 ms median on the RTX 5090 (bigL_h with K=4: 0.71 ms), 24/24 check frames same gate set, corner difference 0.29 px. release_v1 files also contain the raw map (`gdet`), so a user can already decode 8 gates from it.
- Closed-loop simulation for bigN_h2 K=8 is running (`runs/closed_loop/bigN_h2_k8.jsonl`); a release v2 follows if it is not worse than bigL_h there.

## Update 75 (2026-11-17) — closed loop with bigN_h2 (8 gates per frame); release v2 frozen

Closed loop in SIMULATION (same 6 held-out layouts c_race_0..5, 44 gates, 1 m gate map error, `runs/closed_loop/bigN_h2_k8.jsonl`); gates passed:

| | bigL_h (K=4) | bigN_h2 (K=8) |
|---|---|---|
| map only, 8 / 12 m/s | 40/132, 42/132 | 40/132, 42/132 |
| true gate positions (upper bound of the controller) | 132/132, 132/132 | 132/132, 132/132 |
| perception, 8 m/s (2 seeds) | 87/88 | 88/88 |
| perception, 12 m/s (2 seeds) | 87/88 | 87/88 |
| perception with one frame of delay, 12 m/s | 44/44 | 44/44 |

Equal within one gate; the only failure is one frame hit on c_race_4 at 12 m/s. This is a simulation result, not a real-world rate; it shows that the new weights and the 8-gate decoding do not break the loop.

`runs/release_v2/` frozen: bigN_h2 weights, fp32 / fp16 ONNX with `gates` 1x8x9, check files, training config, checksums, README with v1 numbers side by side. release_v1 untouched.

## Update 76 (2026-11-17) — bigP_h: real close-pass frames shown 6 times per epoch (`close_gate_boost: 6`)

What the close-pass misses look like (`runs/eval/vis/close_gate_misses_bigN_h2.jpg`, release v2 on TII video, gates > 100 px): the gate fills the frame, the fisheye lens bends its bars, the model gives no gate or a too small square inside it. In the training set only 185 real frames hold such a gate (in-view corners spanning >= 100 px at 256x192), about 0.3% of the set. New config keys `close_gate_boost` (default 1) and `close_gate_px` (default 100): those frames are added `boost - 1` more times to each epoch (`close_gate_indices`, test added, 63 tests pass). The chain now evaluates with `PFM_GDET_K=8`.

bigP_h = bigN_h recipe + `close_gate_boost: 6`, one run (gate numbers +-0.06). All columns decoded with 8 gates per frame.

| | bigN_h2 (release v2) | bigP_h |
|---|---|---|
| red (never seen) <20 / <10 px | 0.974 / 0.913 | 0.962 / 0.894 |
| red unmatched predictions / frame | 0.02 | 0.00 |
| half-gate crops on red: 1 corner, found <0.25 size / no prediction | 0.73 / 0.07 | 0.82 / 0.03 |
| half-gate crops on red: 0 corners, found <0.25 size / no prediction | 0.27 / 0.35 | 0.27 / 0.30 |
| TII video (hall seen): recall per gate-frame | 0.946 | 0.947 |
| – fully / partly visible | 0.940 / 0.952 | 0.943 / 0.951 |
| – last 12 frames of a track | 0.948 | 0.948 |
| – share in miss runs >= 12 frames | 0.013 | 0.010 |
| – unmatched predictions / frame | 0.235 | 0.243 |
| – < 8 px (n=55) | 0.38 | 0.26 |
| – 8–16 / 16–40 / 40–100 px | 0.84 / 0.966 / 0.963 | 0.83 / 0.956 / 0.966 |
| – > 100 px (n=446), <20 px / size-relative | 0.54 / 0.76 | 0.77 / 0.89 |
| sim_racec full view <20 px (sim) | 0.803 | 0.802 |
| IDD-AW mIoU / TuSimple lane recall | 0.612 / 0.80 | 0.606 / 0.78 |
| false gates on gate-free real sets | 0.00 | 0.00–0.01 |

Reading:
- Close passes on real video: 0.54 -> 0.77 within 20 px. The rest of the video numbers are unchanged.
- Caution: the boosted frames come from the same hall and the same fisheye camera as the test flights, so part of this may be learning that hall's close views. The only close-range check in a never-seen place is the half-gate crop proxy on red, where one-corner crops go 0.73 -> 0.82 (inside the +-0.06 band taken twice, so a hint, not proof). The team camera is the real test.
- Small costs inside noise: red <10 px 0.913 -> 0.894, lane recall 0.80 -> 0.78, < 8 px gates 0.38 -> 0.26 (n=55).
- Next: bigP_h_s1 (same recipe, `seed: 1`, running) to see whether the close-pass gain repeats; release v2 stays as it is until then.

## Update 77 (2026-11-17) — bigP_h second seed (bigP_h_s1, `seed: 1`): close-pass gain repeats on seen-hall video, not on the never-seen proxy

All decoded with 8 gates per frame.

| | bigN_h2 (release v2) | bigP_h | bigP_h_s1 |
|---|---|---|---|
| red (never seen) <20 / <10 px | 0.974 / 0.913 | 0.962 / 0.894 | 0.968 / 0.897 |
| half-gate crops on red: 1 corner, found <0.25 size | 0.73 | 0.82 | 0.72 |
| half-gate crops on red: 0 corners, found <0.25 size / no prediction | 0.27 / 0.35 | 0.27 / 0.30 | 0.16 / 0.48 |
| TII video (hall seen): recall per gate-frame | 0.946 | 0.947 | 0.951 |
| – last 12 frames of a track | 0.948 | 0.948 | 0.957 |
| – share in miss runs >= 12 frames | 0.013 | 0.010 | 0.004 |
| – 8–16 / 16–40 / 40–100 px | 0.84 / 0.966 / 0.963 | 0.83 / 0.956 / 0.966 | 0.85 / 0.963 / 0.966 |
| – > 100 px (n=446), <20 px / size-relative | 0.54 / 0.76 | 0.77 / 0.89 | 0.75 / 0.88 |
| – unmatched predictions / frame | 0.235 | 0.243 | 0.245 |
| sim_racec full view <20 px (sim) | 0.803 | 0.802 | 0.809 |
| IDD-AW mIoU / TuSimple lane recall | 0.612 / 0.80 | 0.606 / 0.78 | 0.601 / 0.80 |

Reading:
- Close passes on TII video: 0.54 -> 0.75–0.77 in both seeds. Real, for this hall and this fisheye camera.
- Never-seen place: the one-corner proxy gain of bigP_h (0.82) did not repeat (0.72), and bar-only crops are lower in seed 1 (0.16; earlier runs 0.24–0.29). So there is no evidence that the boost helps close gates in a new place; it may be hall-specific learning.
- Full gates on red unchanged (0.96–0.97 / 0.89–0.90, release v2 0.97 / 0.91). Segmentation 0.60–0.61 (v2 0.61).
- Decision: `close_gate_boost` does no measurable harm and helps where we can measure close passes, but the evidence is seen-hall only, so release v2 stays the release. bigP_h is kept as the candidate for the day real close-pass footage from another place (team camera) exists.
- A fair picture of release v2 on the never-seen place (`scripts/vis_fair_gate_sheet.py`, `runs/eval/vis/fair_gates_red_v2.jpg`, 12 evenly spaced whole frames): 10 clean, 2 wrong, both frame-filling close gates (one missed with a wrong smaller square inside, one with a top corner far off).

Nested gates after Update 77 (`scripts/diag_centre_collisions.py`, 8 gates per frame). The dense head gives one gate per grid cell, so two gates with (almost) the same centre — one seen through the other — cannot both come out:

| set | gates | n | bigN_h2 found <20 px |
|---|---|---|---|
| red (never seen) | another gate's centre within 2 cells (< 4.6 px) | 20 | 0.60 (8 missed) |
| red (never seen) | no close centre | 292 | 0.993 (2 missed) |
| TII video (every 8th frame) | within 2 cells | 10 | 0.50 |
| TII video | within 2–5 cells | 214 | 0.74 |
| TII video | no close centre | 4,108 | 0.956 |

On the never-seen place 8 of the 10 remaining misses are nested gates. Fix being tried: size levels in the dense head (`gdet_levels: [24, 80]`, px of the larger gate side at 256x192): one 9-channel map per size level (3 levels here), a gate within 20% of a threshold is a target on both neighbouring levels, the decoder takes peaks from all levels and removes duplicates as before. Old checkpoints load and decode unchanged (checked: bigN_h2 red 0.974 / 0.913). Test added (64 pass). bigQ_h = bigP_h recipe + `gdet_levels: [24, 80]`, training.

## Update 78 (2026-11-17) — bigQ_h: size levels in the dense gate head (`gdet_levels: [24, 80]`) — nested gates better, half gates worse

bigQ_h = bigP_h recipe + three size levels. One run (gate numbers +-0.06). 8 gates per frame. The decoder now also combines the scores of the two peaks a gate near a size threshold leaves on neighbouring levels (`PFM_GDET_MERGE`, default on with levels; bigQ_h numbers below are with it, in brackets without).

| | bigN_h2 (release v2) | bigP_h / bigP_h_s1 | bigQ_h |
|---|---|---|---|
| red (never seen) <20 / <10 px, score > 0.5 | 0.974 / 0.913 | 0.962 / 0.894, 0.968 / 0.897 | 0.971 / 0.907 (0.965 / 0.901) |
| red <20 / <10 px, score > 0.3 | 0.974 / 0.913 | 0.965 / 0.897, 0.971 / 0.897 | 0.990 / 0.920 (without merge) |
| red: gates with another gate's centre within 2 cells (n=20) | 0.60 | 0.60 (bigP_h) | 0.85 |
| red: gates with no close centre (n=292) | 0.993 | 0.983 (bigP_h) | 0.976 (0.969) |
| half-gate crops on red: 1 corner, found <0.25 size | 0.73 | 0.82, 0.72 | 0.63 |
| half-gate crops on red: 0 corners, found <0.25 size | 0.27 | 0.27, 0.16 | 0.17 |
| TII video (hall seen): recall per gate-frame | 0.946 | 0.947, 0.951 | 0.955 (without merge) |
| – centres 2–5 cells apart (n=214, every 8th frame) | 0.74 | 0.70 (bigP_h) | 0.86 (0.85) |
| – > 100 px | 0.54 | 0.77, 0.75 | 0.79 |
| – unmatched predictions / frame | 0.235 | 0.243, 0.245 | 0.212 |
| sim_racec / sim_racep full view <20 px (sim) | 0.803 / 0.773 | 0.802, 0.809 / – | 0.808 / 0.781 |
| IDD-AW mIoU / TuSimple lane recall | 0.612 / 0.80 | 0.606 / 0.78, 0.601 / 0.80 | 0.602 / 0.78 |

Reading:
- Nested gates in the never-seen place: 0.60 -> 0.85 (n=20, so 5 more gates found). Close-but-not-identical centres on TII video 0.70–0.74 -> 0.86.
- Cost 1: gates without a close neighbour on red lose a little at score 0.5 (0.983–0.993 -> 0.976); at score 0.3 the whole set reaches 0.990 / 0.920, the highest so far. Scores are less confident, probably for gates near a size threshold.
- Cost 2: the half-gate proxy drops (one corner 0.63 against 0.72–0.82 for the four runs before). A gate of which only a piece is visible has an uncertain size, hence an uncertain level.
- sim far gates did not move (sim_racec 0.81): the sim loss is not a nested-centre problem at this grid.
- Not a release: gain and cost are of similar size and it is one run. Next: bigR_h with two levels only (`gdet_levels: [56]`), fewer thresholds for half gates to be unsure about; nested pairs on red are typically > 100 px outside and < 40 px inside. Training.

## Update 79 (2026-11-17) — bigR_h: two size levels (`gdet_levels: [56]`) — best never-seen-place numbers so far, one run

bigR_h = bigP_h recipe + two size levels (threshold 56 px of the larger gate side at 256x192). One run (gate numbers +-0.06). 8 gates per frame, score merge on.

| | bigN_h2 (release v2) | bigP_h / bigP_h_s1 | bigQ_h (3 levels) | bigR_h (2 levels) |
|---|---|---|---|---|
| red (never seen) <20 / <10 px, score > 0.5 | 0.974 / 0.913 | 0.962 / 0.894, 0.968 / 0.897 | 0.971 / 0.907 | 0.987 / 0.942 |
| red median corner error, px | 2.4 | 2.6, 2.9 | 2.7 | 3.4 |
| red: nested gates (centre within 2 cells, n=20) | 0.60 | 0.60 | 0.85 | 1.00 |
| red: gates with no close centre (n=292) | 0.993 | 0.983 | 0.976 | 0.986 |
| red unmatched predictions / frame | 0.02 | 0.00, 0.01 | 0.01 | 0.00 |
| half-gate crops on red: 1 corner, found <0.25 size | 0.73 | 0.82, 0.72 | 0.63 | 0.72 |
| half-gate crops on red: 0 corners, found <0.25 size | 0.27 | 0.27, 0.16 | 0.17 | 0.15 |
| TII video (hall seen): recall per gate-frame | 0.946 | 0.947, 0.951 | 0.955 | 0.953 |
| – last 12 frames of a track | 0.948 | 0.948, 0.957 | 0.942 | 0.955 |
| – 8–16 / 16–40 / 40–100 px | 0.84 / 0.966 / 0.963 | 0.83 / 0.956 / 0.966, 0.85 / 0.963 / 0.966 | 0.84 / 0.970 / 0.966 | 0.85 / 0.963 / 0.964 |
| – > 100 px | 0.54 | 0.77, 0.75 | 0.79 | 0.84 |
| – centres 2–5 cells apart (n=214, every 8th frame) | 0.74 | 0.70 | 0.86 | 0.68 |
| – unmatched predictions / frame | 0.235 | 0.243, 0.245 | 0.212 | 0.271 |
| sim_racec / sim_racep full view <20 px (sim) | 0.803 / 0.773 | 0.802, 0.809 | 0.808 / 0.781 | 0.799 / 0.777 |
| IDD-AW mIoU / TuSimple lane recall | 0.612 / 0.80 | 0.606 / 0.78, 0.601 / 0.80 | 0.602 / 0.78 | 0.603 / 0.79 |

Reading:
- Never-seen place: all 20 nested gates found, whole set 0.987 / 0.942. The <10 px number is above every earlier run (0.88–0.92), by about half the stated noise band, so it needs the second seed before it is believed.
- Half gates: one corner back in the usual range (0.72); bar-only crops 0.15, at the low end of earlier runs (0.16–0.29). No gain there, possibly a small loss.
- Costs seen: median corner error 3.4 px (earlier 2.4–2.9), unmatched predictions on TII video 0.27 per frame (earlier 0.21–0.25), and the TII "centres 2–5 cells apart" case is not helped by two levels (0.68; three levels 0.86) — those neighbours are of similar size.
- Segmentation, lanes, sim: unchanged.
- bigR_h_s1 (`seed: 1`) is training. If it repeats the red numbers, bigR_h becomes release v3 (with a closed-loop simulation check first).

Side views (user observation: gates seen from an angle look worse). `scripts/diag_gate_view_angle.py`, never-seen place (red), fully visible gates, view ratio = shorter / longer mean side of the labelled quad (the gates there are taller than wide: frontal is about 0.75; below 0.5 is a steep side view):

| view ratio | n | bigN_h2 (release v2) <20 / <10 / <5 px, median err | bigR_h <20 / <10 / <5 px, median err |
|---|---|---|---|
| > 0.85 | 14 | 1.00 / 1.00 / 0.86, 2.9 px | 1.00 / 0.93 / 0.50, 5.8 px |
| 0.70–0.85 | 151 | 0.95 / 0.89 / 0.87, 1.9 px | 1.00 / 1.00 / 0.89, 2.7 px |
| 0.50–0.70 | 103 | 1.00 / 0.99 / 0.93, 2.6 px | 0.99 / 0.97 / 0.83, 3.7 px |
| < 0.50 (steep) | 10 | 0.90 / 0.60 / 0.20, 9.3 px | 0.80 / 0.60 / 0.20, 7.1 px |

The observation is right for steep views: found, but corners 7–9 px off and only 0.6 within 10 px (n=10, so wide uncertainty). Training gates by the same ratio (fully in view, >= 12 px): real 22.6% / 55.8% / 14.3% / 7.3% (464 gates, almost all TII fisheye); simulation 42% / 41% / 17% / 0.1%. The simulation holds practically no steep side view.
New augmentation `side_view_prob` (default 0): a gate frame is squeezed sideways (up to 2x) with a keystone (projective warp, zoomed in so nothing outside the picture is sampled); image, labels, depth, teacher maps and corners go through the same map (test added, 65 pass). bigS_h = bigR_h recipe + `side_view_prob: 0.25`, queued behind bigR_h_s1. Also bigR_h's corner error is larger than v2's in every bin (see the median column) — the price of the size levels noted in Update 79.

## Update 80 (2026-11-17) — bigR_h second seed (bigR_h_s1): the never-seen-place gain of two size levels repeats

Both seeds with `gdet_levels: [56]`, 8 gates per frame, score merge on. Compared with release v2 (bigN_h2; its repeat bigN_h was 0.962 / 0.904 with 4 gates).

| | bigN_h2 (release v2) | bigR_h | bigR_h_s1 |
|---|---|---|---|
| red (never seen) <20 / <10 px | 0.974 / 0.913 | 0.987 / 0.942 | 0.990 / 0.942 |
| red median corner error, px | 2.4 | 3.4 | 3.1 |
| red: nested gates (n=20) / others (n=292) | 0.60 / 0.993 | 1.00 / 0.986 | 1.00 / 0.990 |
| half-gate crops on red: 1 corner, found <0.25 size | 0.73 | 0.72 | 0.64 |
| half-gate crops on red: 0 corners, found <0.25 size / no prediction | 0.27 / 0.35 | 0.15 / 0.41 | 0.22 / 0.55 |
| TII video (hall seen): recall per gate-frame | 0.946 | 0.953 | 0.951 |
| – 8–16 px / 16–40 / 40–100 | 0.84 / 0.966 / 0.963 | 0.85 / 0.963 / 0.964 | 0.87 / 0.957 / 0.969 |
| – > 100 px | 0.54 | 0.84 | 0.79 |
| – unmatched predictions / frame | 0.235 | 0.271 | 0.251 |
| – centres 2–5 cells apart (n=214) | 0.74 | 0.68 | 0.68 |
| sim_racec / sim_racep full view <20 px (sim) | 0.803 / 0.773 | 0.799 / 0.777 | 0.799 / 0.786 |
| IDD-AW mIoU / TuSimple lane recall | 0.612 / 0.80 | 0.603 / 0.79 | 0.601 / 0.79 |

View angle on red (steep side views < 0.5, n=10): bigR_h_s1 found within 10 px 0.70 (v2 0.60, bigR_h 0.60), median 5.3 px (v2 9.3). Other ratio bins: <10 px 0.86 / 0.99 / 0.99, median error 6.0 / 2.4 / 3.3 px.

Reading:
- Two seeds agree: +0.013 to +0.016 within 20 px and +0.029 within 10 px on the never-seen place, all 20 nested gates found. Each step is below the +-0.06 band, but the pattern is the same in both seeds and has a cause (nested gates 0.60 -> 1.00).
- Costs repeat too: corner error about 0.7–1.0 px larger (frontal gates 6 px against 3 px), no gain for bar-only half gates (0.15 / 0.22 against 0.27), 0.02–0.04 more unmatched predictions on video. Segmentation, lanes and sim unchanged.
- Verdict: bigR_h_s1 is a release v3 candidate (nested gates were the main remaining miss in the never-seen place). Before freezing: closed-loop simulation check. Isaac Sim needs about 16 GB VRAM and bigS_h (side-view augmentation) is training with 17 GB used, so the check runs after bigS_h. v2 stays the frozen release until then.

## Update 81 — bigS_h (bigR_h + side_view_prob 0.25): no measurable gain, not adopted

Recipe = bigR_h (levels [56], close_gate_boost 6, half_gate_prob 0.4, centre-complete) + `side_view_prob: 0.25` (projective squeeze <=2x + keystone). Single seed (default seed); red noise is about +-0.06, so only large differences count.

| metric | bigR_h_s1 | bigS_h |
|---|---|---|
| red (never seen) <10 / <20 px | 0.942 / 0.990 | 0.946 / 0.990 |
| red steep view (ratio<0.5, n=10) <10 px / median err | 0.70 / 5.3 px | 0.70 / 4.8 px |
| red view 0.7-0.85 <10 px (n=151) | 0.993 | 0.967 |
| nested gates red (n=20) <20 px | 1.00 | 0.95 (1 miss) |
| TII video recall <20 px / >100 px | see 80 (0.79 >100 px) | 0.947 / 0.805 |
| TII nested (n=10) | - | 0.30 |
| half gates, one corner visible found<0.5size | - | 0.889 (0.616 no prediction when 0 corners) |
| sim_racec / sim_racep full-view <20 px | - | 0.80 / 0.79 |
| TuSimple lane recall | - | 0.784 |

Reading: the steep bin has n=10, so the 0.70 vs 0.70 (<10 px) and 0.9 vs 0.8 (<20 px) differences are one gate each, i.e. noise. Median error 5.3 -> 4.8 px is within noise. Nested-gate recall dropped 1.00 -> 0.95 (one gate of 20). Conclusion: side-view augmentation does not demonstrably help on the never-seen red set; it stays available (default 0), not in the recipe. The real fix for steep views is data (sim steep-angle frames, own-camera footage), because red has only 10 steep gates to measure on.
Decision: release v3 candidate remains bigR_h_s1; closed-loop sim check started (runs/closed_loop/bigR_h_s1.jsonl).

## Update 82 (2026-11-17) — closed loop for bigR_h_s1 and release v3 frozen

Closed-loop SIMULATION (not a real rate), 6 held-out layouts, K=8, levels [56]: percept 175/176 gates (identical to bigN_h2), percept_delay 44/44, map alone 82/264, oracle 264/264. So no regression. Export: fp32/fp16 ONNX with 18-channel gdet match PyTorch (same gate set 12/12; TRT fp16 24/24, 0.24 px max corner diff, 0.9 ms on RTX 5090). Frozen `runs/release_v3/` (weights, ONNX, checks, train_config, README, SHA256SUMS). Orin accuracy/timing of v3 still unmeasured.

## Update 83 (2026-11-17) — steep side-view simulation set (s_race) and bigT_h

New data: `scripts/gen_race_steep.py` (30 layouts, gates 3.5–7 m apart, turns 65–100 deg, frames within 10 m, 60 fps), `scripts/race_steep_pipeline.sh`, `scripts/diag_layout_view_angle.py`, `scripts/add_sim_set.py` (adds a rendered set to the shared bigG image set, labels_J and the teacher cache; fixed a quadratic glob). Layout-level view measure (|cos| between view ray and gate normal): ratio <0.5 is 10.2 % of gate-frames (normal sets 0.0–0.3 %), 0.5–0.7 9.6 % (0.4–4 %). Isaac hung at start-up for 2 of 30 layouts (s_race_28/29, killed; 28 layouts used; s_race_26/27 held out, 561 frames). Set added with effects clear/blur/lowlight: 24,153 images (data/sim_races, data/train_sim/images/sim_races).
bigT_h = bigR_h_s1 recipe (seed 1) + this set. 10 epochs, 715 s/epoch.

| | bigR_h_s1 | bigT_h |
|---|---|---|
| red (never seen) <10 / <20 px | 0.942 / 0.990 | 0.936 / 0.994 |
| red steep view (<0.5, n=10) <10 px / median err | 0.70 / 5.3 px | 0.70 / 4.9 px |
| red nested gates (n=20) | 1.00 | 0.95 |
| red half gates 1 corner / 0 corners found <0.25 size | 0.64 / 0.22 | 0.73 / 0.17 |
| red seg: false gate pixel share / frames with >1 % false gate | 0.0004 / 0.005 (bigS_h) | 0.0032 / 0.092 |
| TII video recall (hall seen) / >100 px | 0.951 / 0.79 | 0.954 / see log |
| sim s_race held-out (SIM, in distribution) full view <20 px / partly out <20 px | 0.751 / 0.462 | 0.763 / 0.737 |
| sim_racec / sim_racep partly out <20 px (SIM) | 0.857 / 0.869 (bigS_h) | 0.911 / 0.911 |
| IDD-AW mIoU / TuSimple lane recall | 0.601 / 0.79 | 0.605 / 0.78 |

Reading: on the never-seen real place nothing moves beyond noise (steep bin n=10, identical 0.70 within 10 px). The gain is in simulation (partly out of view 0.46 → 0.74 on steep held-out layouts, a same-generator test, not real). One cost: the gate class in the segmentation map fires on non-gate pixels more often on red (>1 % false gate pixels in 9 % of frames vs 0.5 %), probably from the large share of frame-filling sim gates. Not adopted as release v4 (no real-place gain, a seg regression). To judge the real effect, the missing piece is real steep-view test data (red has 10 steep gates). Single seed, ±0.06.

## Update 84 (2026-11-17) — narrower field of view (crop-zoom proxy for the IMX219), release v3 vs v2

New `scripts/eval_fov_zoom.py`: the never-seen place (red) is cut to its centre 1/z and resized to 448; only gates with all 4 corners inside the crop are scored. A crop is NOT a real 62 deg lens (no change in distortion, less detail), only a check that gates which look larger and sit nearer the border are still found. Note: `ZOOM=` in eval_video_consistency is a different, second-pass diagnostic and gave identical numbers for 1.5 and 2 (it is a flag, not a factor).

| zoom (gates n) | bigR_h_s1 (v3) <20 / <10 px, median | bigN_h2 (v2) <20 / <10 px, median |
|---|---|---|
| 1.0 (278) | 0.993 / 0.975, 2.8 px | 0.971 / 0.921, 2.3 px |
| 1.5 (197) | 0.985 / 0.909, 3.7 px | 0.980 / 0.909, 3.0 px |
| 2.0 (130) | 0.946 / 0.808, 5.0 px | 0.962 / 0.838, 3.4 px |
| 3.0 (39) | 0.949 / 0.846, 7.0 px | 0.974 / 0.949, 6.0 px |

Reading: both models still find nearly all gates up to 2x; the 10 px precision drops (upsampled crops are blurry, and the gate gets bigger so the same px error is relatively smaller). v3 is not better than v2 under zoom (differences within noise; n small at 3x). The crop does not model the real lens or the rolling shutter, so the true IMX219 behaviour is still unmeasured; own-camera footage is the missing piece.

## Update 85 (2026-11-17) — 640 px inference with the 448-trained v3: worse

TII video (hall seen), bigR_h_s1, `SIZE=640` vs 448: recall <20 px 0.916 vs 0.951; gates 8–16 px 0.884 vs 0.872 (no real gain), >100 px 0.15 vs 0.79, 0–8 px 0.16 vs 0.26; unmatched 0.33 vs 0.25. Gate sizes change by 1.43x, which moves them across the size-level threshold and out of the trained size range. So higher input size only helps if the model is trained at it (teacher cache and training at 640 would be needed, 2x network cost on the Orin). Not pursued now. Started instead: `bigU` = big ViT with the v3 recipe (seed 1), to see how much accuracy the 11 M model leaves on the table.
NOTE (data caveat): `add_sim_set.py` put the steep set (sim_races, 24,153 images) into the SHARED `data/mix/bigG/images` and `labels_J`, so every chain started from now on trains with it. bigU therefore compares with bigT_h (same data), not with bigR_h_s1. To train without it, remove/ignore the `sim_races` folder in data/mix/bigG/images (do not delete data: move it aside) before starting a chain.

## Update 86 (2026-11-17) — bigU: big ViT-B (92 M) with the v3 recipe + steep set; the small model does not leave accuracy on the table for gates

Same data/recipe/seed as bigT_h (includes the steep sim set), `KIND=vit`, 10 epochs, 663 s/epoch. One seed, ±0.06.

| | bigT_h (hybrid 11 M) | bigU (ViT-B 92 M) |
|---|---|---|
| red <10 / <20 px (THR 0.5, chain) | 0.936 / 0.994 | 0.933 / 0.971 (K=8 eval: 0.942 / 0.981) |
| red nested gates (n=20) / others | 0.95 / 0.993 | 1.00 / 0.969 |
| red steep view (n=10) <10 px | 0.70 | 0.80 (median 6.7 px) |
| red half gates, 1 corner found <0.25 size | 0.73 | 0.47 |
| red seg false gate pixels / frames >1 % | 0.0032 / 0.092 | 0.0075 / 0.204 |
| TII video recall (hall seen) | 0.954 | 0.954 |
| sim s_race held-out full / partly out <20 px (SIM) | 0.763 / 0.737 | 0.752 / 0.766 |
| IDD-AW mIoU / TuSimple lane recall | 0.605 / 0.78 | 0.582 / 0.82 |

Reading: the 8x larger network brings no gate gain on the never-seen place (slightly lower overall, half gates clearly lower, within noise except the half-gate 1-corner column); only lane recall is better (0.82). Segmentation mIoU on IDD-AW is lower (0.58 vs 0.61), false gate pixels are higher. So for gate detection the limit is data, not network size; the hybrid stays the deliverable. The steep-set data raises false gate pixels in both architectures.

## Update 87 (2026-11-17) — bigV_h started: third seed of the v3 recipe WITHOUT the steep set
`data/mix/bigG/images/sim_races` moved (not deleted) to `data/mix/parked/bigG_sim_races`; chains started from now on train without it again. bigV_h = bigR_h_s1 recipe, `seed: 2`. Purpose: a third seed gives a better noise estimate for the red numbers (bigR_h 0.987 / 0.942, bigR_h_s1 0.990 / 0.942) before more gate claims are made.

## Update 88 (2026-11-18) — bigV_h: third seed of the v3 recipe (no steep set)

| | bigR_h (s0) | bigR_h_s1 (release v3) | bigV_h (seed 2) |
|---|---|---|---|
| red <20 / <10 px (final eval) | 0.987 / 0.942 | 0.990 / 0.942 | 0.987 / 0.946 (chain first pass 0.981 / 0.939) |
| red median error | 3.4 | 3.1 | 3.0 px |
| red nested gates (n=20) / others | 1.00 / 0.986 | 1.00 / 0.990 | 0.90 / 0.986 |
| red steep view (n=10) <10 px | 0.60 | 0.70 | 0.70 (median 5.3 px) |
| red half gates 1 corner / 0 corners, found <0.25 size | 0.72 / 0.15 | 0.64 / 0.22 | 0.72 / 0.22 |
| red seg false gate pixels / frames >1 % | – | 0.0004 / 0.005 (bigS_h) | 0.0012 / 0.020 |
| TII video recall / unmatched per frame | 0.953 / 0.271 | 0.951 / 0.251 | 0.955 / 0.230 |
| sim_racec / sim_racep full view <20 px (SIM) | 0.799 / 0.777 | 0.799 / 0.786 | 0.797 / 0.769 |
| IDD-AW mIoU / TuSimple lane recall | 0.603 / 0.79 | 0.601 / 0.79 | 0.605 / 0.79 |

Reading: three seeds of the same recipe give red <20 px 0.987–0.990 and <10 px 0.942–0.946, so the recipe is stable at this level (seed spread about 0.003–0.004, much smaller than the +-0.06 I used as a worst-case band for single runs). The nested-gate result is the one number that moved (1.00, 1.00, 0.90: 2 misses of 20), so "all nested gates found" must be stated as 18–20 of 20. Release v3 (seed 1) is a typical, not a lucky, run of this recipe. Segmentation and lane numbers are the same across seeds.

## Update 89 (2026-11-18) — bigW_h started: segmentation weak classes
Weakest classes on never-seen data: other_obstacle (IoU 0.05–0.10), person from above (0.05–0.11), rock (0.08–0.16). bigW_h = v3 recipe, seed 1 (same as bigR_h_s1, steep set parked) with `seg_class_weights` extended by `other_obstacle: 3, rock: 3`. Compare with bigR_h_s1: IDD-AW per-class IoU (other_obstacle), SUN/ADE/Aero seg, RUGD rock, and check gates (red) do not drop. Single change, same seed.

## Update 90 — bigW_h results (seg class weights other_obstacle/rock = 3), NOT adopted

Same recipe and seed as release v3 (bigR_h_s1), steep set parked; only change: `seg_class_weights` also has other_obstacle: 3, rock: 3. Single run; noise caveat applies (3-seed spread of v3 recipe on red is ~0.004, but seg metrics were not measured over seeds).

Held-out seg IoU, bigW_h vs bigR_h_s1 (real photos, never trained on):
- other_obstacle: IDD-AW 0.146 vs 0.078, RUGD-heldout 0.426 vs 0.237, ADE 0.387 vs 0.366, BDD 0.304 vs 0.263 (all up); Aeroscapes 0.283 vs 0.300 (flat/slightly down).
- rock: RUGD-heldout 0.061 vs 0.117 (DOWN), ADE 0.487 vs 0.484 (flat). The rock weight did not help.
- mIoU: IDD-AW 0.609 vs 0.599, ADE 0.587 vs 0.585, BDD 0.730 vs 0.725, SUN 0.720 vs 0.722.

Costs:
- Red (never seen) gates <20 px 0.984–0.987 vs 0.990, <10 px 0.923–0.926 vs 0.942 (small drop, larger than the 3-seed spread).
- TuSimple lane IoU 0.745 vs 0.768; half gates with 1 visible corner found<0.25 size 0.66 (v3: ~0.64–0.72, no change).
- TII 0.953 vs 0.951 (hall seen), nested/closed loop not re-run.
- seg false-gate frames on red 0.026 vs 0.005.

Verdict: other_obstacle improves (mostly one-off sets) but gate precision on red, lane and false-gate share get slightly worse and rock does not improve. Not adopted; release v3 stays bigR_h_s1. Next: nothing queued that needs the GPU; remaining gaps need own-camera footage / Orin.

## Update 91 — bigX_h start (v3 recipe, seed 1, 16 epochs instead of 10)

Only change vs release v3 (bigR_h_s1): `epochs: 16`. Question: is v3 under-trained (small gates <8 px, half gates)? ~505 s/epoch, training ends ≈ 2.3 h, eval chain after. Compare with bigR_h_s1 on red, half gates, TII, seg sets. Adopt only if red/half gates improve beyond the ~0.004 seed spread without seg/lane loss.

Update 91b — Turkish PDF 4th edition written (docs/report/PerceptionModel_rapor_v4.pdf, scripts/make_report_v4_tr.py): v3 report + section 2c summarising Updates 83–90 (three seeds, side-view aug, steep sim set, ViT-B, 640 px, seg weights, FOV crop-zoom proxy); all marked not adopted.

## Update 92 — bigX_h results (v3 recipe, seed 1, 16 epochs instead of 10)

Single run (noise caveat; v3 3-seed spread on red ~0.004, seg metrics not seed-replicated). Compared with bigR_h_s1 (10 epochs):
- Red (never seen) full gates <20/<10 px: 0.990/0.939 vs 0.990/0.942 — no gate gain on the never-seen place. Partial gates 0.941 (34 gates) vs 0.971.
- Half gates (red crops, 1 corner visible) found<0.5/<0.25 size: 0.874/0.679 vs ~0.82/0.64–0.72 (inside prior seed range). Steep side view (n=10): 0.70/0.60, unchanged.
- TII flights (hall seen, NOT generalisation): 0.960 vs 0.951; dropouts/1000 2.8 vs 5.6.
- Lane (TuSimple) IoU 0.803 vs 0.768 (recall 0.826 vs 0.790); SUN mIoU 0.761 vs 0.722; Aeroscapes mIoU 0.695 vs 0.663 (opponent_drone IoU 0.675 vs 0.536); BDD 0.735 vs 0.725; ADE 0.588 vs 0.585; IDD-AW 0.602 vs 0.599; RUGD-heldout 0.511 vs 0.512. Seg false-gate frames on red 0.026 vs 0.005 (same rise as bigW_h, worth watching).
- Sim held-out (SIM only) sim_racec full view <20 px 0.821 vs 0.799.
Reading: longer training does not improve gates on the never-seen place (the gate problem is not under-training); it improves lane, drone and SUN segmentation by amounts larger than typical seed noise but unreplicated. Cost: +60% training time, no inference cost. Not adopted for the frozen v3 release (gate result identical, seg-false-gate rise); candidate for a future v4 if a second seed confirms the seg/lane gain.

Update 92b — bigX_h nested-gate diagnostic on red: 19/20 (v3: 20/20, bigV_h 18/20: inside seed range); non-nested red gates 0.979 within 20 px. Started bigY_h = same 16-epoch recipe with seed 2 (tmux y16, log ~/bigY_h.log) to check whether the lane/drone/SUN seg gain replicates; ends ≈ +2.3 h training + 30 min eval.

## Update 93 — bigY_h (16 epochs, seed 2) replicates bigX_h; release v4 = bigY_h

Never-seen red: 0.987/0.958 within 20/10 px (v3 0.990/0.942; bigX_h 0.990/0.939), nested 20/20, half gates 1 corner found<0.5 0.895, 0 corners 0.505 (v3 0.36). TII 0.962. Lane IoU 0.803 (same as bigX_h), SUN 0.749, Aeroscapes drone 0.614, BDD 0.734, ADE 0.595, IDD-AW 0.609, RUGD 0.505. Seg false-gate frames on red 0.026 (same as bigX_h; v3 0.005) — the only cost found; gate-free real frames' gate head false alarms unaffected.
Both 16-epoch seeds agree on lane +0.035 and drone/SUN gains; red gates equal within the 10-epoch spread, so the gate gain is NOT proven (10 px: 0.939 vs 0.958 across seeds); the seg/lane gain is replicated.
Release v4 = runs/release_v4 (bigY_h, fp32+fp16 ONNX, TRT fp16 0.89 ms on 5090, TRT corner diff 1.9 px vs v3 0.24 — flagged). Not re-run: closed loop, Orin. v3 kept as fallback. pytest not affected (no code change).
Update 93b — closed loop for bigY_h started (tmux cl4, runs/closed_loop/bigY_h.jsonl; same 6 layouts as v3 run).

## Update 94 — closed loop for bigY_h (release v4 candidate)

Closed-loop SIMULATION (not a real rate), same 6 held-out layouts, K=8, levels [56]: percept 175/176 gates, percept_delay 44/44, map alone 82/264, oracle 264/264 — identical to bigR_h_s1 (v3) and v2. No regression; closed loop does not separate the models (it is saturated). release_v4 README: closed loop now measured (update the "NOT re-measured" list accordingly).

## Update 95 — bigZ_h start (v3 recipe, seed 1, 24 epochs)

Does the 10 → 16 epoch gain (lane +0.035, drone, SUN, half gates) continue? Same recipe, seed 1, epochs 24 (~3.4 h training + eval; tmux z24, log ~/bigZ_h.log; ends with "CHAIN DONE"). Compare with bigX_h (16 ep, seed 1) and bigY_h (16 ep, seed 2). Release v4 stays bigY_h unless 24 epochs clearly beats both.

## Update 96 — bigZ_h results (v3 recipe, seed 1, 24 epochs)

Single run (different seed from bigY_h; compare with bigX_h = same seed, 16 epochs). Red (never seen) gates 0.987/0.958 within 20/10 px — the numbers happen to be identical to bigY_h's (different checkpoints, sha differs; same hard gates are missed), nested 20/20, non-nested 0.983. Half gates 1 corner found<0.5 0.913 (bigX_h 0.874, bigY_h 0.895), 0 corners 0.374 (0.364 / 0.505: noisy). TII 0.965 (0.960 / 0.962). Lane IoU 0.820, recall 0.845 (0.803 / 0.803). BDD 0.746, ADE 0.604 (up ~0.01). Down: SUN mIoU 0.737 (0.761 / 0.749), Aeroscapes drone 0.595 (0.675 / 0.614), RUGD rock 0.043 (0.078 / 0.080), IDD-AW 0.597 (0.602 / 0.609). Seg false-gate frames on red 0.010 (0.026 / 0.026 — better). sim_racec full view <20 px 0.810 (SIM).
Reading: 24 epochs does not move gates on the never-seen place; lane/ADE/BDD continue up slightly, drone/SUN/rock come back down — mixed, within what a seed could do. +50% training time over 16 epochs. Release v4 stays bigY_h. Next: third seed of the 16-epoch recipe (bigQ_h, seed 3) to fix the v4 noise band.
Update 97 — bigQ_h start (16 epochs, seed 3; tmux q16, ~/bigQ_h.log).

Update 98 — MISTAKE, corrected: tag bigQ_h already existed (Update 78: 3-level gate head). My seed-3 launch reused the tag, so (1) the chain skipped training (final.pt present, runs/bigQ_h untouched, weights intact), (2) data/mix/bigQ_h.yaml was overwritten with the new config and (3) the old evaluation log ~/bigQ_h.log was overwritten by the failed eval. Fixes: yaml regenerated from runs/bigQ_h/hybrid/provenance.jsonl (the overwriting copy kept at ~/bigQ_h_overwritten_by_seed3_attempt.yaml); the Update 78 numbers in this log remain the record of that run, its raw log is lost. Rule: check `ls runs/<tag>` before starting a chain. The seed-3, 16-epoch run is relaunched as bigS3_h.

## Update 99 — DroneRacing GateNet g17 vs our v3/v4 on the never-seen real place (offline detector comparison, NOT flight success)

User asked to swap our detector into https://github.com/GulogluHaktan/DroneRacing and estimate the real-flight difference. Weights were pushed (policy soup85, estimator e4, calib, GateNet g17). Real-flight success cannot be measured (that project has no hardware flights; README: "hardware tests are still open"). The Isaac closed-loop swap needs Isaac Lab (docker image lunarsim-isaaclab:6.0.1, not on this machine) — not run.
Offline: `scripts/eval_dr_gatenet_real.py` (g17 on red, 196 frames/312 gates, input 256x192 full frame, same matching as eval_real_golden.py, scale 256x192). g17 is trained on simulator data only.
- g17 within 20 px: 0.776 at its default threshold 0.3 (0.558 at 0.5, 0.830 at 0.2 with 0.64 unmatched detections/frame); within 10 px 0.756; full gates 0.853, partial gates 0.147 (n=34); median corner error 2.0 px where it detects (corner definition agrees with our labels).
- Ours (release v3 / v4): 0.990 / 0.987 within 20 px, 0.942 / 0.958 within 10 px, unmatched 0.0/frame.
Caveats: g17 is normally run with adaptive crop around the EKF-predicted gate and the map-based EKF fusing frames, which this per-frame test does not include; our numbers are per-frame too. This is a detector comparison on one never-seen real place, not a flight success rate.
