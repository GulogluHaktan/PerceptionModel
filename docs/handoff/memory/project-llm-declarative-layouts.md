---
name: project-llm-declarative-layouts
description: "Design rule - the LLM never writes code, it selects asset ids and coordinates; our resolver sizes, places and validates"
metadata:
  node_type: memory
  type: project
  originSessionId: d1e9c1b7-129a-4c19-bc1f-1a4f4d01a000
  modified: 2026-11-11T07:14:06.334Z
---

User decision (2026-11-11): the LLM writes no code. It chooses objects and coordinates in a JSON layout; our system sets sizes, floor contact, wall anchoring and checks the rules (safer and exact). Implemented in `src/pfm/layout.py` (resolver + rules, errors fed back for repair), `src/pfm/layout_gen.py` (rule-based generator, baseline and few-shot examples), `src/pfm/layout_llm.py` (Ollama with JSON-schema constrained decoding; asset field is an enum of catalogue ids), `sim/isaac_indoor.py` (renderer). Small local LLMs on the laptop (qwen3:8b, qwen3-vl:8b, gemma2:9b via Ollama in `~/.local/ollama`, 8 GB GPU, one model at a time, never together with Isaac) are enough for tests; the final 5090 LLM is chosen Monday with a 50-scenario validity test.

**Why:** code from a model can be unsafe or wrong; a closed schema plus resolver is exact and checkable.

**How to apply:** extend the schema (never allow free code or sizes); first measured result with qwen3:8b: 12 requests, 4 valid on the first try, 6 after two repair rounds (`runs/llm_layout/qwen3_8b.json`); failures were mostly items placed on top of each other. With the free-spot hint in overlap errors plus prompt tips (applied): 5 first try, 9 valid after two repairs (`qwen3_8b_v2.json`; n=12, one run each, noisy). See [[project-work-order-indoor-outdoor-racing]].
