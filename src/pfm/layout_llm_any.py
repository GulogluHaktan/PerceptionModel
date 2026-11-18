"""One LLM interface for the three declarative domains (indoor room, outdoor scene, racing track): schema for constrained decoding, system prompt, resolver, repair loop.
Backends: Ollama (laptop, local) and any OpenAI-compatible server (vLLM / llama.cpp server on the 5090; JSON-schema response_format). The model never writes code or sizes; the resolver of each domain checks everything."""
from __future__ import annotations

import json
import time
import urllib.request

from . import layout as L_in, layout_gen as G_in, layout_llm as LL_in, layout_out as L_out, layout_gen_out as G_out, layout_race as L_race

_NUM = {"type": "number"}; _XY = {"type": "array", "items": _NUM, "minItems": 2, "maxItems": 2}; _XYZ = {"type": "array", "items": _NUM, "minItems": 3, "maxItems": 3}; _SIDE = {"enum": ["north", "south", "east", "west"]}


def _outdoor_schema() -> dict:
    ids = [c["id"] for c in L_out.catalogue()]
    at = {"anyOf": [_XY, {"type": "object", "properties": {"road": {"type": "string"}, "s": {"anyOf": [_NUM, {"enum": ["auto"]}]}, "lane": {"enum": list(L_out.LANES) + list(L_out.KERBS)}}, "required": ["road", "s", "lane"], "additionalProperties": False},
                    {"type": "object", "properties": {"near": {"type": "string"}, "side": _SIDE, "gap": _NUM}, "required": ["near", "side", "gap"], "additionalProperties": False}, {"type": "object", "properties": {"area": {"enum": ["auto"]}}, "required": ["area"], "additionalProperties": False}]}
    item = {"type": "object", "properties": {"id": {"type": "string"}, "asset": {"enum": ids}, "at": at, "yaw_deg": _NUM}, "required": ["id", "asset", "at"], "additionalProperties": False}
    road = {"type": "object", "properties": {"id": {"type": "string"}, "from": _XY, "to": _XY, "width": _NUM}, "required": ["id", "from", "to", "width"], "additionalProperties": False}
    cam = {"type": "object", "properties": {"waypoints": {"type": "array", "items": _XYZ, "minItems": 1, "maxItems": 20}, "look_at": {"anyOf": [_XYZ, {"type": "null"}]}, "roll_deg": _NUM, "speed_mps": _NUM, "fps": _NUM, "hfov_deg": _NUM},
           "required": ["waypoints"], "additionalProperties": False}
    return {"type": "object", "properties": {"area": {"type": "object", "properties": {"size": _NUM}, "required": ["size"], "additionalProperties": False}, "light": {"enum": sorted(L_out.LIGHTS)},
            "roads": {"type": "array", "items": road, "maxItems": 6}, "items": {"type": "array", "items": item, "maxItems": 120}, "camera": cam, "seed": {"type": "integer"}}, "required": ["area", "light", "roads", "items", "camera"], "additionalProperties": False}


def _race_schema() -> dict:
    gate = {"type": "object", "properties": {"id": {"type": "string"}, "at": _XYZ, "yaw_deg": _NUM, "after": {"type": "string"}, "dist": _NUM, "turn_deg": _NUM, "dz": _NUM}, "required": ["id"], "additionalProperties": False}
    fl = {"type": "object", "properties": {k: _NUM for k in ("speed_mps", "fps", "tilt_deg", "offset_m", "focus_m", "hfov_deg", "roll_deg")}, "additionalProperties": False}
    return {"type": "object", "properties": {"setting": {"enum": sorted(L_race.SETTINGS)}, "light": {"enum": sorted(L_race.LIGHTS)}, "template": {"enum": sorted(L_race.TEMPLATES)},
            "gates": {"type": "array", "items": gate, "maxItems": 24}, "flight": fl, "seed": {"type": "integer"}}, "required": ["setting", "light", "gates", "flight"], "additionalProperties": False}


def _auto_example(lay: dict) -> dict:
    """The rule generator's layout with every road-relative item turned into the intent form (s = "auto"), so the few-shot example shows what the model should write."""
    import copy
    out = copy.deepcopy(lay)
    for it in out["items"]:
        if isinstance(it.get("at"), dict) and "road" in it["at"]:
            it["at"]["s"] = "auto"
        elif isinstance(it.get("at"), list) and str(it.get("id", "")).startswith("b"):         # the generator names its buildings b0, b1, ...: show them in the intent form too
            it["at"] = {"area": "auto"}
    return out


DOMAINS = {
    "indoor": dict(schema=LL_in.layout_schema, resolve=L_in.resolve, system=LL_in.system_prompt, intro=""),
    "outdoor": dict(schema=_outdoor_schema, resolve=L_out.resolve,
                    system=lambda: "You build outdoor test scenes for a drone/car perception model by choosing catalogue assets, roads and coordinates. You never write code or sizes.\n" + L_out.SCHEMA_TEXT + "\nCatalogue (id | group/tag | size):\n" + L_out.catalogue_text() + "\nIMPORTANT placement rules: (1) cars, trees, poles and props that stand along a road: at = {\"road\": <road id>, \"s\": \"auto\", \"lane\": <lane>}. (2) buildings, rocks, shrubs and EVERYTHING in a scene that has no road (a park, a forest, a field): at = {\"area\": \"auto\"} and roads = []. Never invent a road called none. (3) never write coordinates or a number for s. (4) a kerb or lane only holds a limited number of items (about one tree per 8 m of road per kerb): put the extra trees on the other kerb or use {\"area\": \"auto\"}. (5) keep the camera pitch within 80 degrees: look_at must be far enough away horizontally.\nExample of a valid reply:\n" + json.dumps(_auto_example(G_out.generate("drone", 7)))),
    "race": dict(schema=_race_schema, resolve=L_race.resolve,
                 system=lambda: "You design drone-racing tracks for a perception model by choosing gate positions and flight settings. You never write code or sizes.\n" + L_race.SCHEMA_TEXT + "\nExample of a valid reply:\n" + json.dumps(L_race.generate(3, speed=25, focus=8))),
}


def ollama_ask(model: str = "qwen3:8b", host: str = "http://localhost:11434", temperature: float = 0.7, timeout: float = 600.0, num_ctx: int = 16384):
    def ask(system, user, schema):
        body = {"model": model, "stream": False, "think": False, "format": schema, "options": {"temperature": temperature, "num_ctx": num_ctx}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}); return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    return ask


def openai_ask(model: str, host: str = "http://localhost:8000/v1", temperature: float = 0.7, timeout: float = 600.0, api_key: str = "none"):
    """vLLM / llama.cpp server / any OpenAI-compatible endpoint with structured outputs (response_format json_schema)."""
    def ask(system, user, schema):
        body = {"model": model, "temperature": temperature, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "layout", "schema": schema, "strict": True}}, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(host + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]["content"]
    return ask


def make(domain: str, request: str, ask, max_repairs: int = 2, system: str | None = None) -> dict:
    d = DOMAINS[domain]; system = system or d["system"](); schema = d["schema"](); user = request; hist = []; t0 = time.time(); layout = None
    for rnd in range(max_repairs + 1):
        raw = ask(system, user, schema)
        try:
            layout = json.loads(raw)
        except json.JSONDecodeError as e:
            hist.append([f"not valid JSON: {e}"]); user = f"{request}\nYour last reply was not valid JSON ({e}). Reply with the JSON object only."; continue
        r = d["resolve"](layout); hist.append(r["errors"])
        if r["ok"]:
            return {"ok": True, "layout": layout, "rounds": rnd, "errors_per_round": hist, "seconds": round(time.time() - t0, 1)}
        user = f"{request}\nYour last reply broke these rules, fix every one and answer with the full corrected JSON:\n" + "\n".join("- " + e for e in r["errors"][:12]) + "\nYour last reply was:\n" + raw
    return {"ok": False, "layout": layout, "rounds": max_repairs, "errors_per_round": hist, "seconds": round(time.time() - t0, 1)}
