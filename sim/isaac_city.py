"""Simple city in Isaac Sim (NVIDIA tree/building/prop assets + procedural road/car) -> RGB, metric depth and exact class labels along a flight/drive path.
Config: runs/city_job/config.json {"name","light":noon|overcast|dusk,"traj":drone|rover|inverted,"frames","seed"}; output runs/city_job/out/<name>/{rgb,depth,label}, poses.json.
Labels come from the class registered for each asset (src/pfm/assets.py), never from a detector. Weather/blur/noise are applied afterwards by scripts/city_effects.py (needs this depth).
Run with sim/run_isaac.sh sim/isaac_city.py."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import importlib.util, json, math, struct, time, zlib
t_boot = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480})
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera


def _load(name):                                                                          # by path: the pfm package would import torch/timm, absent in the container
    sp = importlib.util.spec_from_file_location(name, f"{PFM}/src/pfm/{name}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
REGISTRY = _load("assets").REGISTRY; CLASSES = _load("classes").CLASSES; CLS_ID = {c: i for i, c in enumerate(CLASSES)}
JOB = Path((PFM + "/runs/city_job")); cfg = json.load(open(JOB / "config.json")); rng = np.random.default_rng(int(cfg["seed"]))
out = JOB / "out" / cfg["name"]
for d in ("rgb", "depth", "label"):
    (out / d).mkdir(parents=True, exist_ok=True)
W, H, FOCAL, APER = 640, 480, 18.0, 36.0; FX = W * FOCAL / APER                       # hfov 90 deg, fx = 320 px


def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def box(path, cls, pos, size, color, yaw=0.0):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos])); x.AddRotateZOp().Set(float(yaw)); x.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in size]))
    UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])]); add_labels(c.GetPrim(), [cls])


_UNITS = {}
def asset_units(uri):
    if uri not in _UNITS:                                                                  # USD references convert neither units nor up axis
        ly = Sdf.Layer.FindOrOpen(uri); _UNITS[uri] = (float(ly.pseudoRoot.GetInfo("metersPerUnit") or 1.0), str(ly.pseudoRoot.GetInfo("upAxis") or "Y")) if ly else (1.0, "Y")
    return _UNITS[uri]


def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())


def spawn(path, name, pos, yaw=0.0, height=None):
    """Three levels: our transform / unit+up fix / the asset with its own ops. height = target height in metres (rescales the whole asset)."""
    a = REGISTRY[name]; glb = a["kind"] == "glb"; uri = (PFM + "/") + a["uri"] if glb else a["uri"]; mpu, up = (1.0, "Y") if glb else asset_units(uri); prim = stage.DefinePrim(path, "Xform"); child = stage.DefinePrim(path + "/units", "Xform")
    stage.DefinePrim(path + "/units/asset", "Xform").GetReferences().AddReference(uri)
    cx = UsdGeom.Xformable(child); cx.AddRotateXOp().Set(90.0 if up == "Y" else 0.0); cx.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))
    x = UsdGeom.Xformable(prim); x.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0)); x.AddRotateZOp().Set(0.0); sc = x.AddScaleOp(); sc.Set(Gf.Vec3d(1, 1, 1))
    add_labels(prim, [a["cls"]]); lo, hi = extent(prim); s = 1.0
    if glb and hi[2] - lo[2] > 1e-3:                                                       # GLB sizes are arbitrary: fit the longest horizontal side to a 4.5 m car
        s = 4.5 / float(max(hi[0] - lo[0], hi[1] - lo[1])); sc.Set(Gf.Vec3d(s, s, s))
    elif height and hi[2] - lo[2] > 1e-3:
        s = float(height) / (hi[2] - lo[2]); sc.Set(Gf.Vec3d(s, s, s))
    lo, hi = extent(prim); ops = {o.GetOpName(): o for o in x.GetOrderedXformOps()}
    ops["xformOp:rotateZ"].Set(float(yaw)); ops["xformOp:translate"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]) - float(lo[2])))     # stand on z=0
    return prim, (hi - lo)


# ---------------- city layout ----------------
RW, SW = 10.0, 3.0                                                                          # road width, sidewalk width
box("/World/grass", "ground_road", (0, 0, -0.12), (400, 400, 0.1), (0.28, 0.42, 0.2))
box("/World/roadA", "ground_road", (0, 0, -0.04), (RW, 300, 0.08), (0.14, 0.14, 0.15)); box("/World/roadB", "ground_road", (0, 0, -0.04), (300, RW, 0.08), (0.14, 0.14, 0.15))
for i, sgn in enumerate((-1, 1)):
    box(f"/World/swA{i}", "ground_road", (sgn * (RW + SW) / 2, 0, -0.02), (SW, 300, 0.1), (0.6, 0.6, 0.58)); box(f"/World/swB{i}", "ground_road", (0, sgn * (RW + SW) / 2, -0.02), (300, SW, 0.1), (0.6, 0.6, 0.58))
for k, t in enumerate(np.arange(-145, 146, 6.0)):                                          # dashed centre lines (lane_marking)
    if abs(t) > RW:
        box(f"/World/lmA{k}", "lane_marking", (0, t, 0.01), (0.25, 3.0, 0.02), (0.95, 0.95, 0.9)); box(f"/World/lmB{k}", "lane_marking", (t, 0, 0.01), (3.0, 0.25, 0.02), (0.95, 0.95, 0.9))
rects, n_obj = [], 0                                                                        # footprints (cx, cy, hx, hy) for rejection sampling
for bi in range(int(cfg.get("buildings", 9))):
    name = str(rng.choice([k for k in REGISTRY if k.startswith("bldg_")]))
    for _try in range(80):
        yaw = float(rng.choice([0, 90, 180, 270])); prim, size = spawn(f"/World/bldg{bi}", name, (0, 0, 0), yaw=yaw); hx, hy = (size[1] / 2, size[0] / 2) if yaw in (90, 270) else (size[0] / 2, size[1] / 2)
        cx, cy = rng.uniform(-110, 110, 2)
        if abs(cx) - hx < RW / 2 + SW + 3 or abs(cy) - hy < RW / 2 + SW + 3 or any(abs(cx - r[0]) < hx + r[2] + 4 and abs(cy - r[1]) < hy + r[3] + 4 for r in rects):
            stage.RemovePrim(f"/World/bldg{bi}"); continue
        prim, size = spawn(f"/World/bldg{bi}", name, (cx, cy, 0), yaw=yaw) if False else (prim, size)
        x = UsdGeom.Xformable(prim); ops = {o.GetOpName(): o for o in x.GetOrderedXformOps()}; z0 = ops["xformOp:translate"].Get()[2]; ops["xformOp:translate"].Set(Gf.Vec3d(float(cx), float(cy), z0))
        rects.append((cx, cy, hx, hy)); break
def free(px, py, m=3.0): return not any(abs(px - r[0]) < r[2] + m and abs(py - r[1]) < r[3] + m for r in rects)
trees = [k for k in REGISTRY if k.startswith("tree_")]; ti = 0
for side in (-1, 1):
    for t in np.arange(-120, 121, 14.0):                                                   # street trees on both roads
        for axis in ("A", "B"):
            if abs(t) < RW + 4: continue
            p = (side * (RW / 2 + SW + 2.2 + rng.uniform(0, 1.5)), t + rng.uniform(-3, 3)) if axis == "A" else (t + rng.uniform(-3, 3), side * (RW / 2 + SW + 2.2 + rng.uniform(0, 1.5)))
            if free(*p, 1.5) and rng.random() < 0.8:
                spawn(f"/World/tree{ti}", str(rng.choice(trees)), (p[0], p[1], 0), yaw=float(rng.uniform(0, 360)), height=float(rng.uniform(6, 14))); ti += 1
for _ in range(int(cfg.get("park_trees", 14))):
    p = rng.uniform(-110, 110, 2)
    if abs(p[0]) > RW and abs(p[1]) > RW and free(*p, 2.0):
        spawn(f"/World/tree{ti}", str(rng.choice(trees)), (p[0], p[1], 0), yaw=float(rng.uniform(0, 360)), height=float(rng.uniform(6, 15))); ti += 1
for i in range(8):                                                                          # benches on sidewalks, street lamps on the kerb
    t = rng.uniform(-100, 100); side = rng.choice([-1, 1])
    if abs(t) > RW + 4:
        spawn(f"/World/bench{i}", "bench_curved", (side * (RW / 2 + SW / 2), t, 0), yaw=0.0)
        spawn(f"/World/lamp{i}", "streetlamp", (-side * (RW / 2 + SW - 0.5), t + 6, 0), yaw=0.0, height=7.0)
real_cars = [k for k in REGISTRY if k.startswith("car_") and REGISTRY[k]["kind"] == "glb"]
for i in range(int(cfg.get("cars", 9))):                                                    # real car models in the lanes (procedural box only if none are registered)
    t = rng.uniform(-100, 100); lane = float(rng.choice([-2.5, 2.5])); along = bool(rng.random() < 0.5)
    if abs(t) < RW: continue
    if cfg["traj"] == "rover" and not along and lane < 0: lane = 2.5                       # the rover drives in lane -2.5 of road B: keep cars out of it (camera inside a car = whole frame "car")
    if real_cars and cfg.get("real_cars", True):
        pos = (lane, t) if along else (t, lane); yaw = (90.0 if lane < 0 else -90.0) if along else (0.0 if lane < 0 else 180.0)
        spawn(f"/World/car{i}", str(rng.choice(real_cars)), (pos[0], pos[1], 0), yaw=yaw)
    else:
        col = rng.uniform(0.1, 0.9, 3); box(f"/World/car{i}", "car", (lane, t, 0.8) if along else (t, lane, 0.8), (1.8, 4.5, 1.5) if along else (4.5, 1.8, 1.5), col)
L = {"noon": (2200, 55, (1, 0.97, 0.9), 400, (0.7, 0.8, 1.0)), "overcast": (0, 0, (1, 1, 1), 1400, (0.85, 0.88, 0.9)), "dusk": (1500, 8, (1, 0.6, 0.35), 250, (0.5, 0.55, 0.8))}[cfg["light"]]
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(float(L[0])); sun.CreateColorAttr(Gf.Vec3f(*L[2])); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-90 + 90 - L[1] if False else -L[1], 0, float(rng.uniform(0, 360))))
dome = UsdLux.DomeLight.Define(stage, "/World/sky"); dome.CreateIntensityAttr(float(L[3])); dome.CreateColorAttr(Gf.Vec3f(*L[4]))
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize()
cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(FOCAL); cg.GetHorizontalApertureAttr().Set(APER); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame()
world.reset()
xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder(); xop = xf.AddTransformOp()


def pose(eye, f, roll_deg=0.0):
    f = np.array(f, float); f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg)); r, u = c * r + s * u, -s * r + c * u
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye; return m


def trajectory(kind, n):
    poses, speed = [], {"drone": 15.0, "inverted": 12.0, "rover": 8.0}[kind]
    if kind in ("drone", "inverted"):
        R, alt, a0 = float(rng.uniform(45, 75)), float(rng.uniform(15, 30)), float(rng.uniform(0, 2 * math.pi)); da = 8.0 / R
        for i in range(n):
            a = a0 + i * da; eye = np.array([R * math.cos(a), R * math.sin(a), alt + 3 * math.sin(i * 0.3)]); tang = np.array([-math.sin(a), math.cos(a), 0.0]); inward = -eye[:2] / np.linalg.norm(eye[:2])
            f = np.array([*(math.cos(0.5) * tang[:2] + math.sin(0.5) * inward), -math.tan(math.radians(rng.uniform(12, 28)))]); poses.append(pose(eye, f, 180.0 if kind == "inverted" else rng.uniform(-8, 8)))
    else:
        x0 = float(rng.uniform(-100, -40))
        for i in range(n):
            eye = np.array([x0 + i * 4.0 * (160 / max(160, n * 4)), -2.5 + rng.uniform(-0.3, 0.3), 1.3]); yawc = math.radians(rng.uniform(-6, 6)); poses.append(pose(eye, [math.cos(yawc), math.sin(yawc), -math.tan(math.radians(rng.uniform(0, 5)))]))
    return poses, speed


def step():
    world.step(render=True); rep.orchestrator.step(); app.update()


poses, speed = trajectory(cfg["traj"], int(cfg["frames"])); xop.Set(Gf.Matrix4d(poses[0].T.tolist()))
for _ in range(500):                                                                        # remote USD + textures stream in
    step()
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
    meta.append({"transform_matrix": m.tolist(), "speed_mps": speed}); 
    if k % 10 == 0: print("frame", k, {CLASSES[i]: int((lab == i).sum()) for i in np.unique(lab) if i != 255}, flush=True)
json.dump({"fx": FX, "width": W, "height": H, "light": cfg["light"], "traj": cfg["traj"], "frames": meta, "boot_s": round(t0 - t_boot, 1), "render_s_per_frame": round((time.time() - t0) / len(poses), 2)}, open(out / "poses.json", "w"))
print("DONE", cfg["name"], flush=True); app.close()
