"""Can Isaac Sim load the Objaverse car GLBs (assets/cars/*.glb) directly? Lines them up, rescales each to a 4.5 m long car, labels them 'car', renders one view. Output runs/fg_job/out/cars_test/."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import json, struct, time, zlib
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 540})
import numpy as np, math
import omni.replicator.core as rep, omni.usd
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera
out = Path((PFM + "/runs/fg_job/out/cars_test")); out.mkdir(parents=True, exist_ok=True)
def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
g = UsdGeom.Cube.Define(stage, "/World/ground"); g.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05)); x.AddScaleOp().Set(Gf.Vec3d(80, 40, 0.1)); add_labels(g.GetPrim(), ["ground_road"])
files = sorted(Path((PFM + "/assets/cars")).glob("*.glb")); info = {}
def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())
for i, f in enumerate(files):
    p = f"/World/{f.stem}"; prim = stage.DefinePrim(p, "Xform"); stage.DefinePrim(p + "/up", "Xform").GetReferences().AddReference(str(f)); child = stage.GetPrimAtPath(p + "/up")
    cx = UsdGeom.Xformable(child); cx.AddRotateXOp().Set(90.0)                                   # glTF is Y-up
    xf = UsdGeom.Xformable(prim); tr = xf.AddTranslateOp(); sc = xf.AddScaleOp(); sc.Set(Gf.Vec3d(1, 1, 1)); add_labels(prim, ["car"])
    lo, hi = extent(prim); size = hi - lo; L = float(max(size[0], size[1])); s = 4.5 / L if L > 1e-3 else 1.0; sc.Set(Gf.Vec3d(s, s, s)); lo, hi = extent(prim)
    c = (lo + hi) / 2; tr.Set(Gf.Vec3d(-18 + 6 * i - c[0], -c[1], -lo[2])); info[f.stem] = {"raw_size": [round(float(v), 2) for v in size], "scale": round(s, 4)}
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(2500.0); UsdGeom.Xformable(stage.GetPrimAtPath("/World/sun")).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 30))
UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(500.0)
cam = Camera(prim_path="/World/Camera", resolution=[1280, 540]); cam.initialize(); cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(18.0); cg.GetHorizontalApertureAttr().Set(36.0)
cam.add_rgb_to_frame(); cam.add_semantic_segmentation_to_frame(); world.reset()
xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder(); op = xf.AddTransformOp()
eye, tgt = np.array([0, -22.0, 7.0]), np.array([0, 0, 1.0]); f = tgt - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye; op.Set(Gf.Matrix4d(m.T.tolist()))
for _ in range(500):
    world.step(render=True); rep.orchestrator.step(); app.update()
fr = cam.get_current_frame(); write_png(out / "cars.png", np.asarray(fr["rgb"])[..., :3].astype(np.uint8))
sd = np.squeeze(fr["semantic_segmentation"]["data"]); names = {int(k): list(v.values())[0] if isinstance(v, dict) else v for k, v in fr["semantic_segmentation"]["info"]["idToLabels"].items()}
car_px = int(sum((sd == k).sum() for k, n in names.items() if n == "car")); json.dump({"cars": info, "car_pixels": car_px}, open(out / "info.json", "w"), indent=1); print("INFO", json.dumps(info), "car_px", car_px, flush=True); print("DONE"); app.close()
