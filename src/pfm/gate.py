"""Safe-evolution gate (D8): golden-set floors, provenance log, rollback.

A candidate model is only promoted if it beats floors on the FROZEN golden set AND does not
regress more than `max_regress` against the current model on any tracked metric.
The final promotion approval is the human's (D8): `decide` only RECOMMENDS.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateConfig:
    floors: dict = field(default_factory=lambda: {"seg_miou": 0.0, "depth_absrel_max": 1.0})
    max_regress: float = 0.02  # absolute drop allowed vs. the current (previous accepted) model
    max_total_regress: float = 0.05  # absolute drop allowed vs. the BEST model so far: catches slow drift that every single step passes


def decide(candidate: dict, current: dict | None, cfg: GateConfig, best: dict | None = None) -> dict:
    reasons = []
    for k, floor in cfg.floors.items():
        if k.endswith("_max"):
            m = k[:-4]
            if candidate.get(m, float("inf")) > floor:
                reasons.append(f"{m}={candidate.get(m)} above ceiling {floor}")
        elif candidate.get(k, -1) < floor:
            reasons.append(f"{k}={candidate.get(k)} below floor {floor}")
    if current:
        for k, v in current.items():
            c = candidate.get(k)
            if c is None:
                continue
            worse = (c > v + cfg.max_regress) if k.endswith("absrel") else (c < v - cfg.max_regress)
            if worse:
                reasons.append(f"regression on {k}: {v} -> {c}")
    if best:
        for k, v in best.items():
            c = candidate.get(k)
            if c is None:
                continue
            worse = (c > v + cfg.max_total_regress) if k.endswith("absrel") else (c < v - cfg.max_total_regress)
            if worse:
                reasons.append(f"drift vs best on {k}: best {round(v, 4)} -> {round(c, 4)}")
    return {"recommend_promote": not reasons, "reasons": reasons, "needs_human_approval": True}


def log_provenance(path: str | Path, entry: dict):
    """Append-only provenance log: which data / which parent / which metrics made this model."""
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
