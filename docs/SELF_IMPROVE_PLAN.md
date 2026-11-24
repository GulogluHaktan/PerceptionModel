# Self-improving modular perception — working plan (updated 2026-11-21 after the PaperOrchestra review)

Goal (user): a working system first, paper second — the frozen general model on a vehicle, with plug-in modules, that detects
hard conditions, generates extra data (simulation effects / 3DGS / LLM scenarios), trains a module, verifies it and plugs it in.

## Status
- Step 1 shift detection: DONE (feature distance AUROC 0.91-1.0; gate confidence predicts per-frame failure, Spearman -0.69).
- Step 2-3 condition module: v1 rejected, v2 mixed (NIR gate +0.038, night seg +0.055, fog/blur regress).

## Review-driven items (PaperOrchestra was only a TEST of the tool, user 2026-11-22: do not steer the project by it; item 2 cancelled)
1. Per-condition modules + condition-aware router (module plugged in only for the condition where it was ACCEPTED on a validation split;
   test split reported separately). -> answers "single case study" and makes the loop real. [now]
2. (CANCELLED by user) Matched post-processing for GateNet. Hook GN_MATCHED=1 stays in eval_vision.py, unused.
3. Ablation of the final gate config (each module / filter removed) on validation seeds + equivalence CI for the 128-flight result. [next]
4. Orin latency/params of every module and of GateNet. [needs the Orin]
5. Baselines: LoRA / adapter-in-backbone / fine-tuned copy of the core on the same data, scored on task gain AND general-metric loss. [after 1]
6. Genuinely new task as a module (class the core never saw) + real-data check (real night / NIR footage). [after 1]
7. Write module specs (architecture, params, data, losses) in the paper; widen related work (side-tuning, frozen-backbone extension, gate detection).
Expensive LLM pipelines (claude -p chains) only with the user's OK: the PaperOrchestra run used the account's weekly limit.
