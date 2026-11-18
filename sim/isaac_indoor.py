"""Renders a RESOLVED indoor layout (pfm.layout.resolve -> runs/indoor_job/scene.json) in Isaac Sim: room shell + GLB furniture + camera path -> RGB, metric depth and exact class labels.
Nothing here decides a size or a position: the resolver already did (src/pfm/layout.py). Labels: floor = ground_road, walls/ceiling = building, furniture = its registry class.
Run with sim/run_isaac.sh sim/isaac_indoor.py; output runs/indoor_job/out/<name>/{rgb,depth,label}, poses.json."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import importlib.util, json, math, struct, time, zlib
t_boot = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480})
import numpy as np
import omni.replicator.core as rep, omni.usd
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.sensors.camera import Camera
CLASSES = (lambda m: (m.__spec__.loader.exec_module(m), m.CLASSES)[1])(importlib.util.module_from_spec(importlib.util.spec_from_file_location("classes", (PFM + "/src/pfm/classes.py"))))
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
JOB = Path((PFM + "/runs/indoor_job")); S = json.load(open(JOB / "scene.json")); name = S["name"]; sc = S["scene"]; out = JOB / "out" / name
for d in ("rgb", "depth", "label"):
    (out / d).mkdir(parents=True, exist_ok=True)
W, H, APER = 640, 480, 36.0; cam_cfg = sc["camera"]; FOCAL = APER / 2 / math.tan(math.radians(cam_cfg["hfov_deg"]) / 2); FX = W * FOCAL / APER
RW, RD, RH = sc["room"]["w"], sc["room"]["d"], sc["room"]["h"]


def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2; raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def box(path, cls, lo, hi, color):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)])); x.AddScaleOp().Set(Gf.Vec3d(*[b - a for a, b in zip(lo, hi)]))
    UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)]); add_labels(c.GetPrim(), [cls])


def extent(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange(); return np.array(r.GetMin()), np.array(r.GetMax())


T = 0.1
box("/World/floor", "ground_road", (-T, -T, -T), (RW + T, RD + T, 0.0), (0.55, 0.45, 0.35))
box("/World/ceiling", "building", (-T, -T, RH), (RW + T, RD + T, RH + T), (0.9, 0.9, 0.88))
OPS = sc.get("openings", [])
SPAN = {"south": (RW, lambda u0, u1, z0, z1: ((u0, -T, z0), (u1, 0, z1))), "north": (RW, lambda u0, u1, z0, z1: ((u0, RD, z0), (u1, RD + T, z1))),
        "west": (RD, lambda u0, u1, z0, z1: ((-T, u0, z0), (0, u1, z1))), "east": (RD, lambda u0, u1, z0, z1: ((RW, u0, z0), (RW + T, u1, z1)))}
for nm, (L_, mk) in SPAN.items():                                                              # a wall = boxes around its openings (door, window): below, above and the solid parts between
    ops = sorted([o for o in OPS if o["wall"] == nm], key=lambda o: o["u0"]); cur = -T; k = 0; pieces = []
    for o in ops:
        pieces += [(cur, o["u0"], 0, RH)]
        if o["z0"] > 0: pieces += [(o["u0"], o["u1"], 0, o["z0"])]
        pieces += [(o["u0"], o["u1"], o["z1"], RH)]; cur = o["u1"]
    pieces += [(cur, L_ + T, 0, RH)]
    for u0, u1, z0, z1 in pieces:
        if u1 - u0 > 1e-4 and z1 - z0 > 1e-4:
            lo, hi = mk(u0, u1, z0, z1); box(f"/World/wall_{nm}_{k}", "building", lo, hi, (0.85, 0.82, 0.76)); k += 1
box("/World/outside", "ground_road", (-80, -80, -0.3), (RW + 80, RD + 80, -0.03), (0.3, 0.45, 0.25))
for it in sc["items"]:                                                                          # GLB is Y-up and has an arbitrary size: rotate, fit the longest extent to the size the resolver recorded
    p = f"/World/it_{it['id']}"; prim = stage.DefinePrim(p, "Xform"); stage.DefinePrim(p + "/up", "Xform").GetReferences().AddReference((PFM + "/") + it["uri"])
    UsdGeom.Xformable(stage.GetPrimAtPath(p + "/up")).AddRotateXOp().Set(90.0)
    x = UsdGeom.Xformable(prim); tr = x.AddTranslateOp(); rz = x.AddRotateZOp(); scl = x.AddScaleOp(); scl.Set(Gf.Vec3d(1, 1, 1)); add_labels(prim, [it["cls"]])
    lo, hi = extent(prim); s = it["target_size_m"] / float((hi - lo).max()); scl.Set(Gf.Vec3d(s, s, s)); rz.Set(float(it["yaw_deg"])); tr.Set(Gf.Vec3d(0, 0, 0))
    lo, hi = extent(prim); c = (lo + hi) / 2; tr.Set(Gf.Vec3d(it["pos"][0] - c[0], it["pos"][1] - c[1], it["pos"][2] - lo[2]))      # centre on the resolved xy, stand on the floor (or on its host)
L = sc["light"]; cx, cy = RW / 2, RD / 2
if OPS and L == "day":
    UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(700.0); sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(2500.0); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 25, 30))
if L == "day":
    for li, dx in enumerate((-0.25, 0.25)):
        r = UsdLux.RectLight.Define(stage, f"/World/lamp{li}"); r.CreateWidthAttr(RW * 0.4); r.CreateHeightAttr(RD * 0.4); r.CreateIntensityAttr(9000.0 if not OPS else 4000.0); UsdGeom.Xformable(r).AddTranslateOp().Set(Gf.Vec3d(cx + dx * RW, cy, RH - 0.02))
elif L == "lamp":
    s = UsdLux.SphereLight.Define(stage, "/World/lamp"); s.CreateRadiusAttr(0.15); s.CreateIntensityAttr(60000.0); s.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.5)); UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(cx, cy, RH - 0.5))
else:
    s = UsdLux.SphereLight.Define(stage, "/World/lamp"); s.CreateRadiusAttr(0.1); s.CreateIntensityAttr(9000.0); UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(cx, cy, RH - 0.5))
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize(); cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(FOCAL); cg.GetHorizontalApertureAttr().Set(APER); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 500.0))
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame(); world.reset()
xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder(); xop = xf.AddTransformOp()


def c2w(pos, fwd, roll_deg):
    f = np.array(fwd, float); f /= np.linalg.norm(f); up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.999 else np.array([0, 1.0, 0])
    a = math.radians(roll_deg); up = up * math.cos(a) + np.cross(f, up) * math.sin(a) + f * np.dot(f, up) * (1 - math.cos(a))       # Rodrigues: roll the up vector around the view axis
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f); m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos; return m


def step():
    world.step(render=True); rep.orchestrator.step(); app.update()


poses = [c2w(p["pos"], p["forward"], cam_cfg["roll_deg"]) for p in cam_cfg["poses"]]; xop.Set(Gf.Matrix4d(poses[0].T.tolist()))
for _ in range(300):
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
    meta.append({"transform_matrix": m.tolist(), "speed_mps": cam_cfg["speed_mps"]})
    if k % 10 == 0: print("frame", k, {CLASSES[i]: int((lab == i).sum()) for i in np.unique(lab) if i != 255}, flush=True)
json.dump({"fx": FX, "width": W, "height": H, "light": L, "traj": "indoor", "layout_hash": sc["hash"], "frames": meta, "boot_s": round(t0 - t_boot, 1), "render_s_per_frame": round((time.time() - t0) / len(poses), 2)}, open(out / "poses.json", "w"))
print("DONE", name, flush=True); app.close()
