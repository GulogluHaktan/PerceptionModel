"""Smoke test for every GLB in assets/models/manifest.json: loads each one (glTF, Y-up), rescales its longest extent to the real-world size of its tag, labels it with the registry class,
renders one sheet per group (car, aircraft, two_wheeler, indoor) and reports per-model label pixels, so a model that fails to load or is invisible shows up. Output runs/fg_job/out/models_test/."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import importlib.util, json, struct, zlib
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1600, "height": 900})
import numpy as np, math
import omni.replicator.core as rep, omni.usd
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera
spec = importlib.util.spec_from_file_location("pfm_assets", (PFM + "/src/pfm/assets.py")); A = importlib.util.module_from_spec(spec); import sys; sys.modules["pfm_assets"] = A; spec.loader.exec_module(A)
out = Path((PFM + "/runs/fg_job/out/models_test")); out.mkdir(parents=True, exist_ok=True)
def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
g = UsdGeom.Cube.Define(stage, "/World/ground"); g.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05)); x.AddScaleOp().Set(Gf.Vec3d(400, 400, 0.1)); add_labels(g.GetPrim(), ["ground_road"])
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(2500.0); UsdGeom.Xformable(stage.GetPrimAtPath("/World/sun")).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 30))
UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(500.0)
cam = Camera(prim_path="/World/Camera", resolution=[1600, 900]); cam.initialize(); cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(18.0); cg.GetHorizontalApertureAttr().Set(36.0)
cam.add_rgb_to_frame(); cam.add_semantic_segmentation_to_frame(); world.reset()
cxf = UsdGeom.Xformable(cam.prim); cxf.ClearXformOpOrder(); cop = cxf.AddTransformOp()
def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())
report = {}
for group in ("car", "aircraft", "two_wheeler", "indoor"):
    items = [a for a in A.by_group(group) if a["kind"] == "glb"]
    if not items: continue
    root = stage.DefinePrim(f"/World/g_{group}", "Xform"); n = len(items); cols = math.ceil(math.sqrt(n * 1.6)); rows = math.ceil(n / cols)
    cell = max(A.target_size_m(a) for a in items) * 1.25; ids = []
    for i, a in enumerate(items):
        p = f"/World/g_{group}/m{i}"; prim = stage.DefinePrim(p, "Xform"); stage.DefinePrim(p + "/up", "Xform").GetReferences().AddReference((PFM + "/") + a["uri"])
        UsdGeom.Xformable(stage.GetPrimAtPath(p + "/up")).AddRotateXOp().Set(90.0)
        xf = UsdGeom.Xformable(prim); tr = xf.AddTranslateOp(); sc = xf.AddScaleOp(); sc.Set(Gf.Vec3d(1, 1, 1)); add_labels(prim, [f"{group}_m{i}"])
        lo, hi = extent(prim); size = hi - lo; Lmax = float(size.max()); s = A.target_size_m(a) / Lmax if Lmax > 1e-4 else 1.0; sc.Set(Gf.Vec3d(s, s, s)); lo, hi = extent(prim); c = (lo + hi) / 2
        r_, c_ = divmod(i, cols); tr.Set(Gf.Vec3d(c_ * cell - c[0], r_ * cell - c[1], -lo[2])); ids.append((f"{group}_m{i}", a["name"], [round(float(v), 2) for v in size], round(s, 4)))
    cx, cy = (cols - 1) * cell / 2, (rows - 1) * cell / 2; dist = max(cols * cell, rows * cell * 1.78) * 0.62
    eye = np.array([cx, cy - dist * 0.9, dist * 0.7]); tgt = np.array([cx, cy, 0.0]); f = tgt - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye; cop.Set(Gf.Matrix4d(m.T.tolist()))
    for _ in range(250):
        world.step(render=True); rep.orchestrator.step(); app.update()
    fr = cam.get_current_frame(); write_png(out / f"{group}.png", np.asarray(fr["rgb"])[..., :3].astype(np.uint8))
    sd = np.squeeze(fr["semantic_segmentation"]["data"]); names = {int(k): list(v.values())[0] if isinstance(v, dict) else v for k, v in fr["semantic_segmentation"]["info"]["idToLabels"].items()}
    px = {n: int((sd == k).sum()) for k, n in names.items()}
    report[group] = [{"slot": s_, "name": nm, "raw_size": sz, "scale": sc_, "label_px": px.get(s_, 0), "row": divmod(int(s_.split("_m")[1]), cols)[0], "col": divmod(int(s_.split("_m")[1]), cols)[1]} for s_, nm, sz, sc_ in ids]
    print(group, "models", n, "invisible", [x["name"] for x in report[group] if x["label_px"] == 0], flush=True)
    root.SetActive(False)
json.dump(report, open(out / "report.json", "w"), indent=1); print("MODELSDONE"); app.close()
