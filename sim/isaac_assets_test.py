"""Asset smoke test in Isaac Sim: spawns registry assets (trees, buildings, props, procedural car/road) from NVIDIA's public bucket, labels each root prim with the
class stored in the registry, renders 3 views and reports per-class pixel counts + asset bounding boxes. Output: runs/fg_job/out/assets_test/. Run with sim/run_isaac.sh sim/isaac_assets_test.py."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import json, math, struct, time, zlib
t_boot = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 960, "height": 540})
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera
import importlib.util
def _load(name):                                                                                         # load by path: importing the pfm package would pull in torch/timm, which the container does not have
    sp = importlib.util.spec_from_file_location(name, f"{PFM}/src/pfm/{name}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
REGISTRY = _load("assets").REGISTRY; CLASSES = _load("classes").CLASSES

out = Path((PFM + "/runs/fg_job/out/assets_test")); out.mkdir(parents=True, exist_ok=True); W, H = 960, 540
CLS_ID = {c: i for i, c in enumerate(CLASSES)}


def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def box(path, cls, pos, size, color, yaw=0.0):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos)); x.AddRotateZOp().Set(yaw); x.AddScaleOp().Set(Gf.Vec3d(*size))
    UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)]); add_labels(c.GetPrim(), [cls])


_UNITS = {}
def asset_units(uri):
    """(metersPerUnit, up axis) of the asset layer: USD references do NOT convert units or up-axis, so we do it on the parent transform."""
    if uri not in _UNITS:
        ly = Sdf.Layer.FindOrOpen(uri); _UNITS[uri] = (float(ly.pseudoRoot.GetInfo("metersPerUnit") or 1.0) if ly else 1.0, str(ly.pseudoRoot.GetInfo("upAxis") or "Y") if ly else "Y")
    return _UNITS[uri]


def spawn(path, name, pos, yaw=0.0, scale=1.0):
    a = REGISTRY[name]; mpu, up = asset_units(a["uri"]); prim = stage.DefinePrim(path, "Xform"); child = stage.DefinePrim(path + "/units", "Xform")
    stage.DefinePrim(path + "/units/asset", "Xform").GetReferences().AddReference(a["uri"])      # 3 levels: our transform / unit+up fix / the asset with its own ops
    cx = UsdGeom.Xformable(child); cx.AddRotateXOp().Set(90.0 if up == "Y" else 0.0); cx.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))                   # asset units/up -> metres, Z-up
    x = UsdGeom.Xformable(prim); x.AddTranslateOp().Set(Gf.Vec3d(*pos)); x.AddRotateZOp().Set(yaw); x.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    add_labels(prim, [a["cls"]]); print("spawn", name, "mpu", mpu, "up", up, flush=True); return prim


box("/World/road", "ground_road", (0, 0, -0.05), (12, 120, 0.1), (0.15, 0.15, 0.16))                    # procedural road strip
box("/World/grass", "background", (0, 0, -0.1), (200, 200, 0.1), (0.25, 0.4, 0.2))
box("/World/car", "car", (2.0, -6, 0.75), (1.8, 4.5, 1.5), (0.7, 0.1, 0.1))                             # procedural car box
placed = [("/World/t1", "tree_colorado_spruce", (-8, 4, 0)), ("/World/t2", "tree_black_oak", (8, 10, 0)), ("/World/t3", "tree_gray_birch", (-9, 18, 0)),
          ("/World/b1", "bldg_plaza_01", (-30, 30, 0)), ("/World/b2", "bldg_plaza_03", (30, 40, 0)), ("/World/p1", "bench_curved", (5, 2, 0)), ("/World/p2", "streetlamp", (6, 14, 0))]
for path, name, pos in placed:
    spawn(path, name, pos, yaw=0.0)
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(900.0); UsdGeom.Xformable(stage.GetPrimAtPath("/World/sun")).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 0))
UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(300.0)
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize()
cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(18.0); cg.GetHorizontalApertureAttr().Set(36.0); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 800.0))
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame()
world.reset()
xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder(); xf.AddTransformOp()
def look_at(eye, target):
    eye, target = np.array(eye, float), np.array(target, float); f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    xf.GetOrderedXformOps()[0].Set(Gf.Matrix4d(m.T.tolist()))                                           # USD matrices are row-vector: transpose
def step():
    world.step(render=True); rep.orchestrator.step(); app.update()
looks = [((0, -25, 3), (0, 15, 3)), ((-20, -5, 4), (0, 20, 5)), ((12, -10, 8), (-10, 25, 4))]
look_at(*looks[0]); t_load = time.time()
for _ in range(400):                                                                                    # remote USD + textures stream in
    step()
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]); sizes = {}
for path, name, _ in placed:
    r = bb.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange(); sizes[name] = [round(float(v), 2) for v in (r.GetMax() - r.GetMin())]
print("asset sizes (m, x y z):", json.dumps(sizes), flush=True)
stats = {"load_s": round(time.time() - t_load, 1), "views": [], "sizes": sizes}
for k, (e, t) in enumerate(looks):
    look_at(e, t)
    for _ in range(12):
        step()
    fr = cam.get_current_frame(); rgb = np.asarray(fr["rgb"])[..., :3].astype(np.uint8)
    sem = fr["semantic_segmentation"]; sd = np.squeeze(sem["data"]); lab = np.full(sd.shape[:2], 255, np.uint8); names = {}
    for sid, d in sem["info"]["idToLabels"].items():
        for nm in (d.values() if isinstance(d, dict) else [d]):
            names[int(sid)] = nm
            if nm in CLS_ID:
                lab[sd == int(sid)] = CLS_ID[nm]
    counts = {c: int((lab == i).sum()) for c, i in CLS_ID.items() if (lab == i).any()}
    write_png(out / f"rgb_{k}.png", rgb); write_png(out / f"label_{k}.png", (lab.astype(int) * 22 % 256).astype(np.uint8))
    stats["views"].append({"view": k, "pixels_per_class": counts, "nonblank": bool(rgb.std() > 5)}); print("view", k, counts, flush=True)
json.dump(stats, open(out / "stats.json", "w"), indent=1); print("DONE", flush=True); app.close()
