---
name: project-catalogue-growth-and-native-isaac
description: "Catalogue-growth loop (VLM -> SAM2 -> Hunyuan -> gate -> registry) and the 5090 native-Isaac day-1 script; what is verified and what is not"
metadata:
  type: project
---

User architecture (2026-11-13): when the LLM sees an important object that is not in the catalogue, its picture goes to Hunyuan and the result is added to the catalogue. Built: `src/pfm/asset_growth.py`, `scripts/grow_catalogue.py`, `scripts/image_to_asset.py` (Hunyuan3D-2mini env in `~/Projects/i23d`, outside the project), `src/pfm/mesh_upright.py`. Only assets with status `accepted` reach the resolver (`scripts/layout_to_job_gen.py`); the base catalogue files are never edited by the loop. The 5090 will run NATIVE Isaac Sim (no docker): `scripts/day1_5090.sh` stages; written on the laptop, native path untested.

**Why:** the user wants the sim to cover objects the fixed catalogue lacks, without the LLM writing code or sizes (see [[project-llm-declarative-layouts]]).

**How to apply:** keep the closed tag list and the gate; the gate is a floor (seed consistency 0.93 calibrated on n=8, in-sample); Hunyuan licence is territory-limited (not EU/UK/South Korea); shape-only, flat colour (paint model needs 16+ GB, try on the 5090); never run Isaac, Ollama, Hunyuan or training together on one GPU. Details: vault `perceptionmodel-sim-plan` Updates 17-19, and [[project-work-order-indoor-outdoor-racing]].
