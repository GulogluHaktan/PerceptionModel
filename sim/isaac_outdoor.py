"""Renders a RESOLVED outdoor layout (pfm.layout_out.resolve -> runs/outdoor_job/scene.json) in Isaac Sim: ground, roads with sidewalks and lane markings, NVIDIA/GLB assets, camera path ->
RGB, metric depth and exact class labels. The resolver already decided every size, position, yaw and tree height; nothing is chosen here.
Run with sim/run_isaac.sh sim/isaac_outdoor.py; output runs/outdoor_job/out/<name>/{rgb,depth,label}, poses.json."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import importlib.util, json, math, struct, time, zlib
t_boot = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480})
import numpy as np
import omni.replicator.core as rep, omni.usd
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera
def _load(name):
    sp = importlib.util.spec_from_file_location(name, f"{PFM}/src/pfm/{name}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
CLASSES = _load("classes").CLASSES; CLS_ID = {c: i for i, c in enumerate(CLASSES)}
JOB = Path((PFM + "/runs/outdoor_job")); S = json.load(open(JOB / "scene.json")); name = S["name"]; sc = S["scene"]; out = JOB / "out" / name
for d in ("rgb", "depth", "label"):
    (out / d).mkdir(parents=True, exist_ok=True)
cc = sc["camera"]; W, H, APER = 640, 480, 36.0; FOCAL = APER / 2 / math.tan(math.radians(cc["hfov_deg"]) / 2); FX = W * FOCAL / APER
def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2; raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
def box(path, cls, centre, size, color, yaw=0.0):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre])); x.AddRotateZOp().Set(float(yaw)); x.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in size]))
    UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])]); add_labels(c.GetPrim(), [cls])
_U = {}
def units(uri):
    if uri not in _U:
        ly = Sdf.Layer.FindOrOpen(uri); _U[uri] = (float(ly.pseudoRoot.GetInfo("metersPerUnit") or 1.0), str(ly.pseudoRoot.GetInfo("upAxis") or "Y")) if ly else (1.0, "Y")
    return _U[uri]
def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())
def spawn(it, k):
    """three levels: our transform / unit+up fix / the asset. Scale: tree height, pole height, GLB fitted to 4.5 m; centre the bbox on the resolved xy and stand it on z=0."""
    glb = it["kind"] == "glb"; uri = (PFM + "/") + it["uri"] if glb else it["uri"]; mpu, up = (1.0, "Y") if glb else units(uri); p = f"/World/it{k}"
    prim = stage.DefinePrim(p, "Xform"); ch = stage.DefinePrim(p + "/units", "Xform"); stage.DefinePrim(p + "/units/asset", "Xform").GetReferences().AddReference(uri)
    cx = UsdGeom.Xformable(ch); cx.AddRotateXOp().Set(90.0 if up == "Y" else 0.0); cx.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))
    x = UsdGeom.Xformable(prim); tr = x.AddTranslateOp(); rz = x.AddRotateZOp(); scl = x.AddScaleOp(); scl.Set(Gf.Vec3d(1, 1, 1)); add_labels(prim, [it["cls"]])
    lo, hi = extent(prim); size = hi - lo; s = 1.0
    if glb and size.max() > 1e-3: s = 4.5 / float(max(size[0], size[1]))
    elif it.get("height_m") and size[2] > 1e-3: s = it["height_m"] / float(size[2])
    elif it.get("pole_height_m") and size[2] > 1e-3: s = it["pole_height_m"] / float(size[2])
    scl.Set(Gf.Vec3d(s, s, s)); rz.Set(float(it["yaw_deg"])); tr.Set(Gf.Vec3d(0, 0, 0)); lo, hi = extent(prim); c = (lo + hi) / 2
    tr.Set(Gf.Vec3d(it["pos"][0] - c[0], it["pos"][1] - c[1], -lo[2]))
box("/World/ground", "ground_road", (0, 0, -0.12), (2 * sc["area"] + 200, 2 * sc["area"] + 200, 0.1), (0.28, 0.42, 0.2))
SW = 3.0
for ri, r in enumerate(sc["roads"]):
    (ax, ay), (bx, by) = r["from"], r["to"]; L = r["length"]; ang = math.degrees(math.atan2(by - ay, bx - ax)); mx, my = (ax + bx) / 2, (ay + by) / 2; dx, dy = r["dir"]; nx, ny = -dy, dx
    box(f"/World/road{ri}", "ground_road", (mx, my, -0.04), (L, r["width"], 0.08), (0.14, 0.14, 0.15), ang)
    for sgn in (-1, 1):
        o = sgn * (r["width"] / 2 + SW / 2); box(f"/World/sw{ri}_{0 if sgn < 0 else 1}", "ground_road", (mx + nx * o, my + ny * o, -0.02), (L, SW, 0.1), (0.6, 0.6, 0.58), ang)
    for k, t in enumerate(np.arange(3.0, L - 3.0, 6.0)):
        box(f"/World/lm{ri}_{k}", "lane_marking", (ax + dx * t, ay + dy * t, 0.01), (3.0, 0.25, 0.02), (0.95, 0.95, 0.9), ang)
for k, it in enumerate(sc["items"]):
    spawn(it, k)
L_ = {"noon": (2200, 55, (1, 0.97, 0.9), 400, (0.7, 0.8, 1.0)), "overcast": (0, 0, (1, 1, 1), 1400, (0.85, 0.88, 0.9)), "dusk": (1500, 8, (1, 0.6, 0.35), 250, (0.5, 0.55, 0.8))}[sc["light"]]
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(float(L_[0])); sun.CreateColorAttr(Gf.Vec3f(*L_[2])); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-(90 - L_[1]) if L_[1] else -90, 0, 35))
dome = UsdLux.DomeLight.Define(stage, "/World/sky"); dome.CreateIntensityAttr(float(L_[3])); dome.CreateColorAttr(Gf.Vec3f(*L_[4]))
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize(); cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(FOCAL); cg.GetHorizontalApertureAttr().Set(APER); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame(); world.reset()
xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder(); xop = xf.AddTransformOp()
def c2w(pos, fwd, roll_deg):
    f = np.array(fwd, float); f /= np.linalg.norm(f); up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.999 else np.array([0, 1.0, 0]); a = math.radians(roll_deg)
    up = up * math.cos(a) + np.cross(f, up) * math.sin(a) + f * np.dot(f, up) * (1 - math.cos(a)); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos; return m
def step():
    world.step(render=True); rep.orchestrator.step(); app.update()
poses = [c2w(p["pos"], p["forward"], cc["roll_deg"]) for p in cc["poses"]]; xop.Set(Gf.Matrix4d(poses[0].T.tolist()))
for _ in range(500): step()
t0 = time.time(); meta = []
for k, m in enumerate(poses):
    xop.Set(Gf.Matrix4d(m.T.tolist()))
    for _ in range(6): step()
    fr = cam.get_current_frame(); rgb = np.asarray(fr["rgb"])
    if rgb.ndim != 3: print("EMPTY render at frame", k, flush=True); app.close(); raise SystemExit(2)
    rgb = rgb[..., :3].astype(np.uint8); dep = np.asarray(fr["distance_to_image_plane"]).astype(np.float32); dep = np.where(np.isfinite(dep) & (dep < 900), dep, 0).astype(np.float32)
    sem = fr["semantic_segmentation"]; sd = np.squeeze(sem["data"]); lab = np.full(sd.shape[:2], 255, np.uint8)
    for sid, d in sem["info"]["idToLabels"].items():
        for nm in (d.values() if isinstance(d, dict) else [d]):
            if nm in CLS_ID: lab[sd == int(sid)] = CLS_ID[nm]
    write_png(out / "rgb" / f"{k:04d}.png", rgb); write_png(out / "label" / f"{k:04d}.png", lab); np.save(out / "depth" / f"{k:04d}.npy", dep)
    meta.append({"transform_matrix": m.tolist(), "speed_mps": cc["speed_mps"]})
    if k % 10 == 0: print("frame", k, {CLASSES[i]: int((lab == i).sum()) for i in np.unique(lab) if i != 255}, flush=True)
json.dump({"fx": FX, "width": W, "height": H, "light": sc["light"], "traj": "outdoor", "layout_hash": sc["hash"], "frames": meta, "boot_s": round(t0 - t_boot, 1), "render_s_per_frame": round((time.time() - t0) / len(poses), 2)}, open(out / "poses.json", "w"))
print("DONE", name, flush=True); app.close()
