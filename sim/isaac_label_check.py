"""Which outdoor assets keep their semantic label? Every tree/palm/shrub/rock/prop/pole/building/car asset is placed in a grid with its registry class (unique per-asset labels broke the annotator: all counts were 0); a camera visits each one and counts how much of the
foreground (pixels whose depth is near the asset, not ground) carries that label. A species that renders but stays unlabelled shows up with a low fraction.
Run with sim/run_isaac.sh sim/isaac_label_check.py; output runs/fg_job/out/label_check.json (and the LABELCHECK line in the log)."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import importlib.util, json, math, time
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 320, "height": 240})
import numpy as np, omni.usd, omni.replicator.core as rep
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera
def _load(name):
    sp = importlib.util.spec_from_file_location(name, f"{PFM}/src/pfm/{name}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
REG = _load("assets").REGISTRY; DIMS = json.load(open((PFM + "/assets/models/dims_outdoor.json")))
world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
g = UsdGeom.Cube.Define(stage, "/World/ground"); g.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05)); x.AddScaleOp().Set(Gf.Vec3d(3000, 3000, 0.1)); add_labels(g.GetPrim(), ["ground_road"])
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(2500.0); UsdGeom.Xformable(stage.GetPrimAtPath("/World/sun")).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 30)); UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(500.0)
_U = {}
def units(uri):
    if uri not in _U:
        ly = Sdf.Layer.FindOrOpen(uri); _U[uri] = (float(ly.pseudoRoot.GetInfo("metersPerUnit") or 1.0), str(ly.pseudoRoot.GetInfo("upAxis") or "Y")) if ly else (1.0, "Y")
    return _U[uri]
def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())
names = [n for n, d in DIMS.items() if d["ok"] and n in REG]; info = []
for i, n in enumerate(names):
    a = REG[n]; glb = a["kind"] == "glb"; uri = (PFM + "/") + a["uri"] if glb else a["uri"]; mpu, up = (1.0, "Y") if glb else units(uri); p = f"/World/a{i}"
    prim = stage.DefinePrim(p, "Xform"); ch = stage.DefinePrim(p + "/units", "Xform"); stage.DefinePrim(p + "/units/asset", "Xform").GetReferences().AddReference(uri)
    cx = UsdGeom.Xformable(ch); cx.AddRotateXOp().Set(90.0 if up == "Y" else 0.0); cx.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))
    xf = UsdGeom.Xformable(prim); tr = xf.AddTranslateOp(); sc = xf.AddScaleOp(); sc.Set(Gf.Vec3d(1, 1, 1)); add_labels(prim, [a["cls"]])
    s = DIMS[n]["scale"]; sc.Set(Gf.Vec3d(s, s, s)); lo, hi = extent(prim); c = (lo + hi) / 2; X = (i % 12) * 60.0; Y = (i // 12) * 60.0; tr.Set(Gf.Vec3d(X - c[0], Y - c[1], -lo[2]))
    info.append({"name": n, "pos": [X, Y], "size": DIMS[n]["size_m"]})
cam = Camera(prim_path="/World/Camera", resolution=[320, 240]); cam.initialize(); cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(18.0); cg.GetHorizontalApertureAttr().Set(36.0); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 2000.0))
cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame(); world.reset()
cxf = UsdGeom.Xformable(cam.prim); cxf.ClearXformOpOrder(); cop = cxf.AddTransformOp()
def step():
    world.step(render=True); rep.orchestrator.step(); app.update()
for _ in range(300): step()
res = []; t0 = time.time()
for i, it in enumerate(info):
    sz = max(it["size"]); D = max(2.5, 1.6 * sz); h = it["size"][2] / 2; eye = np.array([it["pos"][0], it["pos"][1] - D, h + 0.3 * sz]); tgt = np.array([it["pos"][0], it["pos"][1], h])
    f = tgt - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f); m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye; cop.Set(Gf.Matrix4d(m.T.tolist()))
    for _ in range(8): step()
    fr = cam.get_current_frame(); sem = fr["semantic_segmentation"]; sd = np.squeeze(sem["data"]); dep = np.asarray(fr["distance_to_image_plane"]).astype(np.float32)
    names_ = {int(k): (list(v.values())[0] if isinstance(v, dict) else v) for k, v in sem["info"]["idToLabels"].items()}
    lab_px = int(sum((sd == k).sum() for k, nm in names_.items() if nm == REG[it["name"]]["cls"])); ground = np.isin(sd, [k for k, nm in names_.items() if nm == "ground_road"])
    fg = (np.abs(dep - D) < 0.75 * sz) & ~ground & np.isfinite(dep); unl = int((fg & ~np.isin(sd, [k for k, nm in names_.items() if nm in ("tree_pole", "building", "car", "other_obstacle", "rock", "ground_road")])).sum())
    res.append({"name": it["name"], "labelled_px": lab_px, "foreground_px": int(fg.sum()), "frac": round(lab_px / max(int(fg.sum()), 1), 3), "unlabelled_fg_px": unl})
    if i % 20 == 0: print(i, len(info), it["name"], res[-1], round(time.time() - t0), "s", flush=True)
Path((PFM + "/runs/fg_job/out/label_check.json")).write_text(json.dumps(res, indent=1)); print("LABELCHECK", json.dumps([r for r in res if r["frac"] < 0.5])); print("LABELDONE", len(res), flush=True); app.close()
