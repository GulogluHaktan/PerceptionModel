"""Isaac Sim hello for the data engine. Runs INSIDE the isaac-sim container (see sim/run_isaac_hello.sh).
Builds a tiny procedural town (ground, boxes = buildings, cylinders = trees/poles, an orange square gate), flies a camera along a path and saves
clean RGB, depth (metres along the optical axis), class labels (ids of src/pfm/model.py CLASSES, 255 = none) and exact poses (OpenGL camera-to-world,
same convention as the 3DGS outputs). The effects library (src/pfm/effects.py) adds blur/fog/noise afterwards.
Small on purpose (640x480, 24 frames): the laptop has 8 GB VRAM and the official minimum is 16 GB."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import json, math, time
t_boot = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480, "disable_viewport_updates": False})   # same renderer as the working LunarRocket scripts
import numpy as np
import omni.replicator.core as rep
try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World
import omni.usd
from pathlib import Path
import struct, zlib


def write_png(path, a):
    """Minimal PNG writer (uint8 HxW or HxWx3): the Pillow inside the Isaac Sim container crashes on save ('_idat' has no fileno)."""
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
from pxr import Gf, UsdGeom, UsdLux

out = Path((PFM + "/runs/isaac_hello"))
for d in ("rgb", "depth", "label"):
    (out / d).mkdir(parents=True, exist_ok=True)
CLASS = {"gate": 1, "tree_pole": 2, "ground_road": 4, "building": 5}
world = World(stage_units_in_meters=1.0)                                # the timeline/physics context the working LunarRocket scripts use; without it Replicator cannot advance the renderer
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
rng = np.random.default_rng(0)
try:
    from isaacsim.core.utils.semantics import add_labels             # Isaac Sim 6.1
except ImportError:                                                  # 6.0.x still has the old name
    from isaacsim.core.utils.semantics import add_update_semantics as _old
    add_labels = lambda prim, labels: _old(prim, labels[0])


def prim_color(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in rgb])])


def box(path, pos, size, rgb, cls):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(c); x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos])); x.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in size]))
    prim_color(c.GetPrim(), rgb); add_labels(c.GetPrim(), [cls])


def cyl(path, pos, r, h, rgb, cls):
    c = UsdGeom.Cylinder.Define(stage, path); c.GetRadiusAttr().Set(float(r)); c.GetHeightAttr().Set(float(h))
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(h) / 2)); prim_color(c.GetPrim(), rgb); add_labels(c.GetPrim(), [cls])


box("/World/ground", (0, 0, -0.05), (140, 140, 0.1), (0.35, 0.35, 0.37), "ground_road")
for i in range(20):
    x, y = rng.uniform(-38, 38, 2)
    if abs(y) < 7 and -32 < x < 20:                                     # keep the flight corridor free
        continue
    h = rng.uniform(6, 22); box(f"/World/b{i}", (x, y, h / 2), (rng.uniform(4, 9), rng.uniform(4, 9), h), rng.uniform(0.45, 0.8, 3), "building")
for i in range(16):
    x, y = rng.uniform(-30, 30, 2)
    if abs(y) < 3 and -32 < x < 20:
        continue
    cyl(f"/World/t{i}", (x, y), 0.25, rng.uniform(4, 9), (0.25, 0.45, 0.2), "tree_pole")
for n, p, s in [("L", (12, -1.6, 1.8), (0.25, 0.25, 3.6)), ("R", (12, 1.6, 1.8), (0.25, 0.25, 3.6)), ("T", (12, 0, 3.6), (0.25, 3.45, 0.25)), ("B", (12, 0, 0.0), (0.25, 3.45, 0.25))]:
    box(f"/World/gate_{n}", p, s, (1.0, 0.4, 0.0), "gate")
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(3000.0); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 20, 0))
UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(600.0)

try:
    from isaacsim.sensors.camera import Camera                         # the sensor class the working LunarRocket scripts use
except ImportError:
    from omni.isaac.sensor import Camera
W, H = 640, 480
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize()
cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(18.0); cg.GetHorizontalApertureAttr().Set(20.955); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 400.0))
fx = 18.0 / 20.955 * W                                                  # focal length in pixels
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame()
world.reset()                                                           # initialises sensors, replicator and the timeline: poses are set AFTER this


def rot_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2; q = [0.0] * 4
    q[1 + i] = 0.25 * s; q[0] = (R[k, j] - R[j, k]) / s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder()
t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble); o_op = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)


def set_pose(eye, tgt):                                                  # USD camera looks down -Z with +Y up
    eye, tgt = np.array(eye, float), np.array(tgt, float); f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1)                                          # columns: camera x, y, z in the world
    q = rot_to_quat_wxyz(R); t_op.Set(Gf.Vec3d(*[float(v) for v in eye])); o_op.Set(Gf.Quatd(*[float(v) for v in q]))
    m = np.eye(4); m[:3, :3] = R; m[:3, 3] = eye; return m               # OpenGL camera-to-world


def step():
    world.step(render=True); rep.orchestrator.step(); app.update()      # the orchestrator step logs "renderer failed to advance" but it is what fills the annotator data (the working LunarRocket scripts do it too)


N, poses = 24, []
set_pose((-30, 0, 2.2), (12, 0, 2.2))
for _ in range(50):                                                      # warm-up exactly like the working scripts
    step()
t0 = time.time()
for k in range(N):
    t = k / (N - 1); eye = (-30 + 38 * t, 3 * math.sin(2 * math.pi * t), 2.2 + 1.2 * math.sin(math.pi * t)); m = set_pose(eye, (12, 0, 2.2)); step(); step()
    fr = cam.get_current_frame(); rgb = fr.get("rgb")
    if rgb is None or np.asarray(rgb).ndim != 3:
        print("EMPTY render data at frame", k, flush=True); app.close(); raise SystemExit(2)
    rgb = np.asarray(rgb)[..., :3].astype(np.uint8); dep = np.asarray(fr["distance_to_image_plane"]).astype(np.float32)
    sem = fr["semantic_segmentation"]; sd = np.squeeze(sem["data"]); id2l = sem["info"]["idToLabels"]
    lab = np.full(sd.shape[:2], 255, np.uint8)
    for sid, d in id2l.items():
        for nm in (d.values() if isinstance(d, dict) else [d]):
            if nm in CLASS:
                lab[sd == int(sid)] = CLASS[nm]
    dep = np.where(np.isfinite(dep), dep, 0).astype(np.float32)
    write_png(out / "rgb" / f"{k:04d}.png", rgb); write_png(out / "label" / f"{k:04d}.png", lab); np.save(out / "depth" / f"{k:04d}.npy", dep)
    poses.append({"file_path": f"rgb/{k:04d}", "transform_matrix": m.tolist()})
    print("frame", k, rgb.shape, "depth max", round(float(dep.max()), 1), "labels", {int(v): int((lab == v).sum()) for v in np.unique(lab)}, flush=True)
json.dump({"fx_px": fx, "width": W, "height": H, "camera_angle_x": 2 * math.atan(0.5 * W / fx), "frames": poses}, open(out / "poses.json", "w"))
st = {"frames": N, "boot_s": round(t0 - t_boot, 1), "render_s_per_frame": round((time.time() - t0) / N, 2)}
json.dump(st, open(out / "stats.json", "w")); print("DONE", st, flush=True)
app.close()
