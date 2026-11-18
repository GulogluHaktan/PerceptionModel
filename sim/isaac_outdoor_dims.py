"""Measures the real-world size (x, y, z metres, after unit/up-axis fixes and the same rescaling the renderer applies) of every outdoor asset and writes runs/fg_job/out/dims_outdoor.json (the container cannot write into assets/; copy it there) and also prints it as one DIMSJSON line.
Trees are measured at a nominal 10 m height, poles at 7 m (the renderer rescales them to the height the resolver picks, sizes scale linearly); buildings, props, rocks keep their native size; GLB cars are fitted to 4.5 m.
Run with sim/run_isaac.sh sim/isaac_outdoor_dims.py. Slow part: streaming the remote USD files (a few seconds each)."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import importlib.util, json, time
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 64, "height": 64})
import numpy as np, omni.usd
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom
from isaacsim.core.api import World
def _load(name):
    sp = importlib.util.spec_from_file_location(name, f"{PFM}/src/pfm/{name}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
REGISTRY = _load("assets").REGISTRY
world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
_U = {}
def units(uri):
    if uri not in _U:
        ly = Sdf.Layer.FindOrOpen(uri); _U[uri] = (float(ly.pseudoRoot.GetInfo("metersPerUnit") or 1.0), str(ly.pseudoRoot.GetInfo("upAxis") or "Y")) if ly else (1.0, "Y")
    return _U[uri]
def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())
GROUPS = {"tree", "shrub", "rock", "prop", "pole", "building", "car"}; NOMINAL = {"tree": 10.0, "pole": 7.0}
out = {}; t0 = time.time(); todo = [(n, a) for n, a in REGISTRY.items() if a["group"] in GROUPS and a["kind"] in ("usd", "glb")]
for i, (n, a) in enumerate(todo):
    glb = a["kind"] == "glb"; uri = (PFM + "/") + a["uri"] if glb else a["uri"]; mpu, up = (1.0, "Y") if glb else units(uri); p = f"/World/a{i}"
    prim = stage.DefinePrim(p, "Xform"); ch = stage.DefinePrim(p + "/units", "Xform"); stage.DefinePrim(p + "/units/asset", "Xform").GetReferences().AddReference(uri)
    cx = UsdGeom.Xformable(ch); cx.AddRotateXOp().Set(90.0 if up == "Y" else 0.0); cx.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))
    sc = UsdGeom.Xformable(prim).AddScaleOp(); lo, hi = extent(prim); size = hi - lo; s = 1.0
    if glb and size.max() > 1e-3: s = 4.5 / float(max(size[0], size[1]))
    elif a["group"] in NOMINAL and size[2] > 1e-3: s = NOMINAL[a["group"]] / float(size[2])
    out[n] = {"size_m": [round(float(v) * s, 3) for v in size], "scale": round(s, 5), "nominal_height": NOMINAL.get(a["group"]), "group": a["group"], "ok": bool(size.max() > 1e-3)}
    stage.RemovePrim(p)
    if i % 10 == 0: print(i, len(todo), n, out[n]["size_m"], round(time.time() - t0), "s", flush=True)
Path((PFM + "/runs/fg_job/out")).mkdir(parents=True, exist_ok=True); Path((PFM + "/runs/fg_job/out/dims_outdoor.json")).write_text(json.dumps(out, indent=1)); print("DIMSJSON", json.dumps(out), flush=True); print("bad", [k for k, v in out.items() if not v["ok"]]); print("DIMSDONE", len(out), flush=True); app.close()
