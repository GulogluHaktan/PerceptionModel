# Monday checklist (5090)

1. Copy this folder to the 5090 (leave out `.venv`, `data`, `runs`). Then `bash scripts/setup_5090.sh`. It checks the torch build for the 5090 (sm_120), runs the tests and a fake-data pipeline, and must end with `SETUP OK`.
1b. `.venv/bin/python scripts/preflight.py configs/v0_5090.yaml 100000`: checks GPU, disk vs cache size, and runs REAL training steps for both students with OOM backoff. Put the batch it prints into the config.
2. `bash scripts/run_v0.sh 2000` runs everything, resumable (rerun to continue):
   fetch licence-tracked images -> pseudo-label (Grounding DINO + SAM2) -> cache teachers (DINOv2-S, Depth Anything V2-S) -> train `vit` -> train `hybrid` -> bench.
   Or step by step: `scripts/fetch_data.py`, `scripts/label_folder.py data/images data/labels 448`, `scripts/cache_teachers.py configs/v0_5090.yaml`, `scripts/train.py configs/v0_5090.yaml`.
3. Outputs: `runs/v0/<student>/{final.pt,last.pt,provenance.jsonl}`. `data/LICENSES.csv` lists every source and its licence tag.
4. Golden set (when the camera arrives): `data/golden/{images/<id>.jpg, seg/<id>.png (class idx, see src/pfm/model.py CLASSES, 255 = ignore), depth/<id>.npy (optional)}`. Hand-label it; never train on it. With it, `train.py` fills the gate (mIoU, AbsRel). Without it, the gate has no metrics.

## What is verified (laptop RTX 5060 8 GB, 2026-11-05)
Fake pipeline; real teachers (DINOv2-S, Depth Anything V2-S, E-RADIO); pretrained ViT-B and hybrid at 448; resume; pseudo-labelling on COCO and drone/sim frames; seg + corner heads learn the pseudo labels (94-97% labelled-pixel accuracy on 5 images, i.e. wiring check, not generalisation); licence-tracked fetch from 4 sources; full `run_v0.sh` flow at small scale.
The laptop RTX 5060 is also Blackwell (capability 12.0 = sm_120, same as the 5090), so the torch build's Blackwell kernels are exercised for real, including bf16 training. Not run on a real 5090 (speed, 32 GB) and no multi-hour run yet.

## Numbers to plan with
- Teacher cache at 448 px: about 1.2 MB/img (DINOv2-S + depth). 100k images about 120 GB. E-RADIO as an extra teacher adds about 2.4 MB/img at 448 (0.6 MB at 224). Set `size: 224` if disk is short.
- Training peak memory (bf16, 448): measured on the laptop: ViT-B 3.9 GB at batch 8 (batch 16 does not fit in 7.6 GB), hybrid 3.0 GB at batch 8. Batch 32 for ViT-B is an extrapolation (about 14-16 GB). `preflight.py` finds the real limit.
- Pseudo-labelling: about 3 s per image on the laptop GPU (about 20 Grounding DINO queries + SAM2 per image); expect much faster on the 5090, but 100k images is still hours.

## Known limits
- E-RADIO is 390M params, about 31 ms at 448 px on the laptop GPU: it is a TEACHER here. The Orin student "hybrid" is a timm hybrid (default `fastvit_sa24`, 8M params; set `hybrid_name`).
- Pseudo-label quality is unmeasured. Seen by eye: COCO frames good for person, tree, road, car; false "gate" on mirrors and bus shelters; nadir views label grass and water as ground. No real racing gates were tested. Only the golden set will tell the truth.
- Licences were read from Hugging Face dataset tags, not legally vetted. CC-BY-SA (Aeroscapes) is share-alike: whether it reaches trained weights is unsettled. UAVid (NC) is excluded by default (`--include-nc`).
- Environment pins matter: `transformers==4.57.*` (E-RADIO remote code breaks on 5.x) and `huggingface-hub<1.0`. `requirements.lock` is the working set.
- `trust_remote_code=True` runs NVIDIA's E-RADIO code from Hugging Face. torch.hub (NVlabs/RADIO) was not used: it asks for a repo trust decision that is yours to make.
- Not built yet: temporal / geometric / cross-head label checks (D6, needs video), the Orin-side labeller, metric depth per camera, the replay buffer and shift detector (D4/D5). These are Orin-phase items.

## Extra tools
- `scripts/golden_pick.py <images> <cache> <n> <out>`: picks the n most diverse frames to hand-label as the golden set (tested on fake data).
- `docs/PROTOCOL.md`: measurement protocol (decision 9), approved 2026-11-06; gate floors, Orin power mode and the equal-compute definition are still to be set from real numbers.

## Licence decision (2026-11-06, owner)
Current use is research only. So: keep all four default sources, including Aeroscapes (CC-BY-SA); credit every source in any paper or release (the ledger `data/LICENSES.csv` lists them). Non-commercial sets (UAVid, Anti-UAV) are now allowed for research with `--include-nc`, but stay off by default. Revisit before any commercial or closed-source use: then drop Aeroscapes and every NC source and retrain. This is the owner's use-case decision, not legal advice.
- `scripts/real_fraction_test.py <cfg> <real_sources> <synth_sources> <N>`: real-fraction test at equal compute (needs a golden set). `src/pfm/energy.py`: Orin energy meter (untested on an Orin).
- `scripts/self_evolve_loop.py <cfg> <real_sources> <pool_sources> <n_real> <n_pool> <cycles> <real_labels_dir> [conf]`: multi-cycle self-evolution with gate + rollback (fake-data tested only).

## Pre-flight test results (2026-11-06, laptop RTX 5060 = Blackwell sm_120)
Passed: clean-folder `setup_5090.sh` from scratch (ends SETUP OK); same pipeline on the fallback torch 2.9.1+cu128 (9 tests, smoke train, bench); unreadable/truncated/empty images (`scripts/check_images.py` writes `bad_files.txt`, loader skips them; `run_v0.sh` runs it); id collisions refused (a.jpg + a.png, a/b vs a__b); odd file names (spaces, Turkish letters, upper-case ext); cache built at another size fails early; kill mid-training then resume continues from the next epoch; labels at another resolution are resized instead of crashing.
Measured: cache loader 320-370 img/s with a warm 74 MB cache (workers 0-8 similar); ViT-B training 29 img/s on this laptop at 448 bf16. On a real 5090 training may be several times faster, so a cold disk loader could become the bottleneck: check GPU utilisation in the first epoch, and put `data/cache` on the fastest disk.
The cu128 fallback branch inside `setup_5090.sh` was run on purpose with `FORCE_CU128=1` (2026-11-06): it reinstalls torch 2.11.0+cu128, passes the 9 tests and the smoke train, ends SETUP OK. It was not triggered by a real old driver. Rented Vast.ai machines often show Max CUDA 13.0, so this branch is likely to run there. Not testable here: driver version of the 5090 machine; real 5090 speed.
Note: `/tmp` here is a 16 GB RAM disk; two extra torch environments filled it during testing (cleaned up). Keep test environments on /home.

## 5090 day 1: simulation data, LLM choice, racing (added 2026-11-12)
Order (each step is one command, resumable):
1. `bash scripts/pack_for_5090.sh` on the laptop, copy the tar to the 5090, unpack, `bash scripts/setup_5090.sh` (torch, tests, smoke train; must end `SETUP OK`).
2. Isaac: `bash scripts/setup_isaac_5090.sh` (docker + NVIDIA toolkit + `docker login nvcr.io` if the image is not cached; checks one resolve per domain). Isaac needs an idle GPU: stop Ollama/vLLM/trainers first (renders go blank otherwise).
3. LLM choice (needs GPU, so BEFORE or AFTER the renders, never together): serve candidates one at a time (vLLM or Ollama), then `python scripts/llm_select_test.py openai:<model>@http://localhost:8000/v1 runs/llm_select/<name>.json` (50 scenarios: indoor 20, outdoor 15, racing 15; schema-constrained decoding, resolver, 2 repair rounds). Compare valid-after-repair per domain and seconds per layout. Laptop reference: qwen3:8b indoor 5-9 of 12 (see `runs/llm_layout/`).
4. Data: `.venv/bin/python scripts/gen_sim_set.py all 30` (30 layouts per domain, rule-based generators, every layout resolved before rendering). Isaac boots once per layout (about 2 min), so 90 layouts take hours: run it overnight, or batch several layouts per Isaac start (not done).
5. Racing training files: `.venv/bin/python scripts/gen_sim_set.py train race data/train_sim` writes images, seg, metric depth, corner files (all-4-visible gates only) and `labels/corners_full/*.json` (all inner corners incl. outside the image, 256x192 px). Indoor/outdoor conversion for the trainer is still TODO.
6. Labeller vs exact labels, golden set, run_v0 as before.

Racing facts: gate = A2RL x DCL (2.7 m outer, 1.5 m opening, 0.6 m bar), camera 256x192 130 deg, corner order TL TR BR BL, label format compatible with `droneracing.estimator.det_features` (4 detections x score + 4 corners). Template `a2rl_2025` = the real 11-gate lap read from the KAIST plot (with the split-S pair and the 4.15 m upper gates).

**Known gap that matters for the 30 m/s goal:** the student's corner head is a 4-heatmap head, so a gate that is only partly visible has no valid target and the trainer skips its corner loss (`factory_to_training.py`, `race_to_training.py`). DroneRacing's `gatenet2.py` already uses corner targets for centre-outside gates. Before training the racing student, add an offset/extrapolation corner head that learns `corners_full` (including corners outside the image); otherwise the net is never taught the blind-spell cases.
Not tested on the 5090: everything above except the resolvers (44 tests) and the laptop renders.

## 5090 with native Isaac Sim (added 2026-11-13): one entry point
`bash scripts/day1_5090.sh [stage ...]` (default `setup isaac hunyuan`); stages `setup`, `isaac`, `hunyuan`, `llm`, `sets`, `assets`, `train`; details in the script header. One GPU job at a time (Isaac, Ollama/vLLM, Hunyuan, training never together).
- `isaac` (`scripts/setup_isaac_native_5090.sh`): uses `$ISAAC_PY` / `$ISAAC_SIM_PATH/python.sh` / `.venv_isaac`, else `pip install isaacsim[all,extscache]==6.0.1` into `.venv_isaac`; renders the hello scene; writes `.isaac_env`. `sim/run_isaac.sh` switches to native with `ISAAC_NATIVE=1` or `ISAAC_PY=...` (the `sim/*.py` read `PFM_ROOT`, container default `/workspace/pfm`). NOT RUN on a 5090 and the pip route is unverified (written on the laptop, where only the container exists): if pip fails, install Isaac Sim 6.0.1 from NVIDIA and set `ISAAC_SIM_PATH`. First start compiles shaders: 5-15 min. Needs a Blackwell-capable driver.
- `hunyuan` (`scripts/setup_hunyuan_5090.sh`): Hunyuan3D-2mini env in `$I23D` (default `~/Projects/i23d`), smoke asset. Licence is territory-limited: read it.
- `sets` renders `N` layouts per domain (default 30, hours) and converts to `data/train_sim`; `train` then runs preflight, teacher cache and `configs/sim_5090.yaml` (vit then hybrid; exact depth + seg + corners + slot head).
- `assets`: `FRAME=<png> KNOWN=tag,tag bash scripts/day1_5090.sh assets` = catalogue growth on one frame (needs Ollama with `qwen3-vl:8b-instruct`).
