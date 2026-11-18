# Measurement protocol (decision 9) - APPROVED 2026-11-06

Status: approved by the owner as written (2026-11-06). The three open questions at the bottom are still to be filled in with real numbers.

Metrics: seg mIoU and depth AbsRel on the frozen golden set; accuracy per Orin joule and per GPU-hour; latency median/p95 (batch 1, fp16, warm-up 10, fixed power mode); peak memory.
Rules (AgriJetsonBench-style): same power mode every run, report median and p95, pre/post-processing timed separately.
Baselines: every synthetic or pseudo-label experiment needs an equal-compute real-only baseline (Dream4Drive lesson).
Real-fraction test: 5 cycles with 0/10/30/50/100% real data to detect collapse (MAD, Gaps 4 and 7).
Equal compute (working definition, proposed 2026-11-06, adjust if you disagree): same number of images N, batch, resolution, epochs and therefore the same optimiser steps for every arm. GPU-seconds are logged next to the metrics but are not the matching criterion.
Gate floors live in the config (`gate:` key), not in code: set them from the first golden-set numbers.

Implemented (tested on fake data / unit tests):
- latency / memory / params: `src/pfm/bench.py`; seg mIoU + AbsRel: `src/pfm/evaluate.py`; gate: `src/pfm/gate.py`
- real-fraction test (0/10/30/50/100% real at equal compute, results CSV): `scripts/real_fraction_test.py`
- Orin energy from tegrastats (parser unit-tested, live run NOT tested on an Orin): `src/pfm/energy.py`

- multi-cycle self-evolution loop with gate and rollback (`src/pfm/selfloop.py`, `scripts/self_evolve_loop.py`): trains, lets the accepted student label the pool (confidence threshold), retrains from the same init, gates every cycle, rolls back on rejection. Tested on fake data only; a missing real label raises an error instead of being skipped.

Not implemented:
- accuracy-per-joule report that ties `energy.py` to the benchmark
- the actual numbers: gate floors, Orin power mode

Still open (numbers to fill in after the first runs): gate floors, which Orin power mode, confirmation of the equal-compute definition.
