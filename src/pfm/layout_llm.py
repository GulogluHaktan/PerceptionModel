"""Local-LLM glue for layouts: the model returns a layout JSON under a JSON schema (constrained decoding: `asset` is an enum of catalogue ids, so an invented asset cannot be produced),
pfm.layout.resolve checks every rule, and rejected layouts go back to the model with the error list (repair rounds). No code from the model ever runs.
`ask` is any callable (system, user, schema) -> str; `ollama_ask` is the local one (Ollama /api/chat, `format` = schema, thinking off). Fakes work in tests."""
from __future__ import annotations

import json
import time
import urllib.request

from .layout import SCHEMA_TEXT, catalogue, catalogue_text, resolve
from .layout_gen import generate

_XY, _NUM = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, {"type": "number"}
_SIDE = {"enum": ["north", "south", "east", "west"]}


def layout_schema() -> dict:
    ids = [c["id"] for c in catalogue()]
    at = {"anyOf": [_XY, {"type": "object", "properties": {"wall": _SIDE, "along": _NUM}, "required": ["wall", "along"], "additionalProperties": False},
                    {"type": "object", "properties": {"near": {"type": "string"}, "side": _SIDE, "gap": _NUM}, "required": ["near", "side", "gap"], "additionalProperties": False},
                    {"type": "object", "properties": {"on": {"type": "string"}, "dx": _NUM, "dy": _NUM}, "required": ["on", "dx", "dy"], "additionalProperties": False}]}
    item = {"type": "object", "properties": {"id": {"type": "string"}, "asset": {"enum": ids}, "at": at, "yaw_deg": _NUM}, "required": ["id", "asset", "at"], "additionalProperties": False}
    opening = {"type": "object", "properties": {"id": {"type": "string"}, "kind": {"enum": ["door", "window"]}, "wall": _SIDE, "along": _NUM}, "required": ["id", "kind", "wall", "along"], "additionalProperties": False}
    xyz = {"type": "array", "items": _NUM, "minItems": 3, "maxItems": 3}
    cam = {"type": "object", "properties": {"waypoints": {"type": "array", "items": xyz, "minItems": 1, "maxItems": 20}, "look_at": {"anyOf": [xyz, {"type": "null"}]}, "roll_deg": _NUM,
                                           "speed_mps": _NUM, "fps": _NUM, "hfov_deg": _NUM}, "required": ["waypoints"], "additionalProperties": False}
    return {"type": "object", "properties": {"room": {"type": "object", "properties": {"w": _NUM, "d": _NUM, "h": _NUM}, "required": ["w", "d", "h"], "additionalProperties": False},
                                             "light": {"enum": ["day", "lamp", "dark"]}, "items": {"type": "array", "items": item, "maxItems": 40}, "openings": {"type": "array", "items": opening, "maxItems": 12},
                                             "camera": cam, "seed": {"type": "integer"}}, "required": ["room", "light", "items", "camera"], "additionalProperties": False}


def system_prompt(examples: int = 1) -> str:
    ex = "\n".join("Example of a valid reply:\n" + json.dumps(generate(k, 100 + i)) for i, k in enumerate(["living_room", "bedroom", "office", "kitchen"][:examples]))
    return f"You build indoor test scenes for a drone/rover perception model by choosing catalogue assets and coordinates. You never write code or sizes.\n{SCHEMA_TEXT}\nCatalogue (id | tag | size x y z):\n{catalogue_text()}\n{ex}"


def ollama_ask(model: str = "qwen3:8b", host: str = "http://localhost:11434", temperature: float = 0.7, timeout: float = 600.0):
    def ask(system: str, user: str, schema: dict) -> str:
        body = {"model": model, "stream": False, "think": False, "format": schema, "options": {"temperature": temperature, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    return ask


def make_layout(request: str, ask, max_repairs: int = 2, system: str | None = None) -> dict:
    """Returns {ok, layout, scene, rounds, errors_per_round, seconds}. Round 0 is the first answer; each repair round shows the model its own JSON and the resolver's errors."""
    system = system or system_prompt(); schema = layout_schema(); user = request; hist = []; t0 = time.time(); layout = None
    for rnd in range(max_repairs + 1):
        raw = ask(system, user, schema)
        try:
            layout = json.loads(raw)
        except json.JSONDecodeError as e:
            hist.append([f"not valid JSON: {e}"]); user = f"{request}\nYour last reply was not valid JSON ({e}). Reply with the JSON object only."; continue
        r = resolve(layout); hist.append(r["errors"])
        if r["ok"]:
            return {"ok": True, "layout": layout, "scene": r["scene"], "rounds": rnd, "errors_per_round": hist, "seconds": round(time.time() - t0, 1)}
        user = f"{request}\nYour last reply broke these rules, fix every one and answer with the full corrected JSON:\n" + "\n".join("- " + e for e in r["errors"][:12]) + "\nYour last reply was:\n" + raw
    return {"ok": False, "layout": layout, "scene": None, "rounds": max_repairs, "errors_per_round": hist, "seconds": round(time.time() - t0, 1)}
