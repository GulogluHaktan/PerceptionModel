"""LLM scenario manager (plan step 5 and the LLM role from the design talk): the LLM only writes scenario JSON; everything it returns is parsed,
validated against the strict schema (pfm.scenario.validate_scenario) and rejected with the error message if it does not fit. It never produces pixels.
`ask` is any callable str -> str (a local model, an API client, or a fake in tests); no network code lives here.
`coverage_matrix` builds the same kind of scenarios deterministically (no LLM) from axes: speed x weather x light x object set."""
from __future__ import annotations

import itertools
import json
import re

from .effects import DEFAULTS, RANGES
from .scenario import OBJECT_TYPES, TRAJ_KINDS, scenario_hash, validate_scenario

SCHEMA_TEXT = f"""Reply with ONLY a JSON array of scenario objects. Allowed top-level keys: scene, objects, trajectory, camera, effects, seed.
scene: id (str), meters_per_unit (float), up ([x,y,z]).
objects: list (max 50) of {{type: one of {sorted(OBJECT_TYPES)}, pos: [x, y, zoffset], yaw_deg (float), size: [w, h] for gate, [radius, height] for pole/tree, [w, d, h] for box}}.
trajectory: kind one of {sorted(TRAJ_KINDS)}, frames (2..5000), fps (1..240), max_speed_mps (0..40), center [x,y,z], radius_m (0.1..500), altitude_m.
camera: width (64..4096), height (64..4096), hfov_deg (20..140).
effects (all optional, ranges): {json.dumps({k: list(v) for k, v in RANGES.items()})}. Defaults: {json.dumps(DEFAULTS)}.
seed: int. No other keys, no comments, no text outside the JSON."""


def build_prompt(problem: str, n: int, extra_rules: str = "") -> str:
    return (f"You design synthetic test scenarios for a drone/rover perception model.\nProblem: {problem}\nWrite {n} DIFFERENT scenarios that cover the problem "
            f"(vary speed, weather, light and objects).\n{SCHEMA_TEXT}\n{extra_rules}".strip())


def extract_json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    for opener, closer in (("[", "]"), ("{", "}")):
        a, b = t.find(opener), t.rfind(closer)
        if a != -1 and b > a:
            try:
                v = json.loads(t[a:b + 1]); return v if isinstance(v, list) else [v]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in the reply")


def generate_scenarios(problem: str, n: int, ask, max_retries: int = 2):
    """Returns (valid_scenarios, rejected) where rejected is a list of {index, error}. Invalid items are fed back to the LLM (up to max_retries rounds)."""
    valid, hashes, rejected = [], set(), []
    prompt = build_prompt(problem, n)
    for attempt in range(max_retries + 1):
        try:
            items = extract_json(ask(prompt))
        except ValueError as e:
            rejected.append({"index": None, "error": str(e)}); prompt = build_prompt(problem, n, f"Your last reply was not valid JSON ({e}). Try again."); continue
        errs = []
        for i, it in enumerate(items):
            try:
                v = validate_scenario(it); h = scenario_hash(it)
                if h not in hashes:
                    hashes.add(h); valid.append(v)
            except (ValueError, TypeError) as e:
                errs.append({"index": i, "error": str(e)}); rejected.append({"index": i, "error": str(e)})
        if len(valid) >= n or not errs:
            break
        prompt = build_prompt(problem, n - len(valid), "These items were rejected, fix them: " + json.dumps(errs))
    return valid[:n], rejected


WEATHER = {"clear": {}, "fog": {"fog_beta": 0.3, "fog_airlight": 0.78}, "rain": {"rain_density": 0.7, "fog_beta": 0.05}}
LIGHT = {"day": {}, "dusk": {"ev": -2, "gain_ev": 2, "iso": 800}, "night": {"ev": -4, "gain_ev": 4, "iso": 3200}}


def coverage_matrix(speeds=(2.0, 8.0, 20.0), weathers=("clear", "fog", "rain"), lights=("day", "dusk"), object_sets=(("gate",), ("gate", "pole")), seed0=0, frames=60):
    """Deterministic scenario grid. Shutter time follows speed (fast flights: shorter shutter, but still blurred); seeds count up."""
    out, k = [], 0
    for v, w, l, objs in itertools.product(speeds, weathers, lights, object_sets):
        fx = {"exposure_s": 1 / 100 if v > 10 else 1 / 250, **WEATHER[w], **LIGHT[l]}
        sc = {"objects": [{"type": o, "pos": [1.0 + 1.5 * i, 0.5 * i, 0.0], "size": [1.4, 1.2] if o == "gate" else [0.08, 1.3]} for i, o in enumerate(objs)],
              "trajectory": {"frames": frames, "max_speed_mps": v}, "effects": fx, "seed": seed0 + k}
        out.append(validate_scenario(sc)); k += 1
    return out
