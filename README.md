# PerceptionModel (pfm) — v0 skeleton

Self-evolving foundation model for edge perception (target: Jetson Orin NX 8 GB).
Decisions: vault `01-Projects/PerceptionModel/perceptionmodel-decisions.md`.

- `src/pfm/model.py`   two students (plain ViT-B / hybrid), teacher-specific heads, dense heads
- `src/pfm/distill.py` multi-teacher losses from cached features + self-distillation
- `src/pfm/cache.py`   teacher feature cache
- `src/pfm/gate.py`    safe-evolution gate + provenance log
- `src/pfm/bench.py`   latency / memory measurement
- `src/pfm/labeler.py` Grounding DINO + SAM2 pseudo labels (seg + gate corners)
- `src/pfm/teachers.py`, `data.py`, `train.py`, `evaluate.py`  teachers, loaders, trainer, golden-set eval
- `scripts/`  setup_5090.sh, run_v0.sh, fetch_data.py, label_folder.py, cache_teachers.py, train.py

Run: `uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e . pytest`, then `.venv/bin/python -m pytest tests`.
Status: v0 pipeline verified end to end on a laptop GPU (see docs/MONDAY.md for limits). Not yet run on the 5090.

## Pipeline (ready for the 5090)
`scripts/setup_5090.sh` -> `scripts/cache_teachers.py <cfg>` -> `scripts/train.py <cfg>`; configs in `configs/` (`smoke.yaml` fake, `v0_5090.yaml` real). See `docs/MONDAY.md`.
