"""Renders a RESOLVED racing track (pfm.layout_race.resolve -> runs/race_job/scene.json): A2RL-style gates (4 purple bars + white corner squares), ground, sky, camera path -> RGB, metric depth, class labels
and gate-corner labels (the resolver projected the inner corners; here corners hidden behind nearer geometry are flagged using the depth image).
Run with sim/run_isaac.sh sim/isaac_race.py; output runs/race_job/out/<name>/{rgb,depth,label}, poses.json, corners.json."""
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
JOB = Path((PFM + "/runs/race_job")); S = json.load(open(JOB / "scene.json")); name = S["name"]; sc = S["scene"]; out = JOB / "out" / name
for d in ("rgb", "depth", "label"):
    (out / d).mkdir(parents=True, exist_ok=True)
cc = sc["camera"]; W, H, APER = 640, 480, 36.0; FOCAL = APER / 2 / math.tan(math.radians(sc["hfov_deg"]) / 2); FX = W * FOCAL / APER
def write_png(path, a):
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2; raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
world = World(stage_units_in_meters=1.0); stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
def box(path, cls, centre, size, color, yaw=0.0):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0); x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre])); x.AddRotateZOp().Set(float(yaw)); x.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in size]))
    UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])]); add_labels(c.GetPrim(), [cls])
hall = sc["setting"] == "hall"
box("/World/ground", "ground_road", (0, 0, -0.12), (500, 500, 0.1), (0.35, 0.35, 0.37) if hall else (0.28, 0.42, 0.2))
GATE_C, WHITE = (0.30, 0.20, 0.70), (1.0, 1.0, 1.0); O, I, B, D = sc["outer"], sc["inner"], sc["bar"], sc["depth"]; hI = I / 2
def local(g, a, r, z):                           # gate frame: a along the passage, r right, z up -> world
    y = math.radians(g["yaw_deg"]); return (g["at"][0] + a * math.cos(y) + r * math.sin(y), g["at"][1] + a * math.sin(y) - r * math.cos(y), g["at"][2] + z)
for k, g in enumerate(sc["gates"]):
    yaw = g["yaw_deg"]
    for nm, (r, z, sy, sz) in {"top": (0, hI + B / 2, O, B), "bot": (0, -hI - B / 2, O, B), "left": (-hI - B / 2, 0, B, I), "right": (hI + B / 2, 0, B, I)}.items():
        box(f"/World/gate{k}_{nm}", "gate", local(g, 0, r, z), (D, sy, sz), GATE_C, yaw)
    zb = g["at"][2] - hI - B                      # underside of the frame: elevated gates stand on two posts (labelled other_obstacle)
    if zb > 0.1:
        for j, sr in enumerate((-1, 1)):
            px_, py_, _ = local(g, 0, sr * (hI + B - 0.06), 0); box(f"/World/gate{k}_post{j}", "other_obstacle", (px_, py_, zb / 2), (0.06, 0.06, zb), (0.55, 0.57, 0.62), yaw)
    for j, (sr, sz) in enumerate(((-1, 1), (1, 1), (1, -1), (-1, -1))):             # white squares on the inner corners = the detector keypoints
        box(f"/World/gate{k}_sq{j}", "gate", local(g, D / 2 + 0.01, sr * (hI + 0.09), sz * (hI + 0.09)), (0.02, 0.18, 0.18), WHITE, yaw)
L_ = {"noon": (2200, 55, (1, 0.97, 0.9), 400, (0.7, 0.8, 1.0)), "overcast": (0, 0, (1, 1, 1), 1400, (0.85, 0.88, 0.9)), "dusk": (1500, 8, (1, 0.6, 0.35), 250, (0.5, 0.55, 0.8))}[sc["light"]]
if hall: L_ = (0, 0, (1, 1, 1), 1800, (1.0, 1.0, 0.95))
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(float(L_[0])); sun.CreateColorAttr(Gf.Vec3f(*L_[2])); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-(90 - L_[1]) if L_[1] else -90, 0, 35))
dome = UsdLux.DomeLight.Define(stage, "/World/sky"); dome.CreateIntensityAttr(float(L_[3])); dome.CreateColorAttr(Gf.Vec3f(*L_[4]))
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize(); cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(FOCAL); cg.GetHorizontalApertureAttr().Set(APER); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame(); world.reset()
xf = UsdGeom.Xformable(cam.prim); xf.ClearXformOpOrder(); xop = xf.AddTransformOp()
def c2w(pos, fwd, up):
    f = np.array(fwd, float); f /= np.linalg.norm(f); r = np.cross(f, np.array(up, float)); r /= np.linalg.norm(r); u = np.cross(r, f)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos; return m
def step():
    world.step(render=True); rep.orchestrator.step(); app.update()
poses = [c2w(p["pos"], p["forward"], p["up"]) for p in cc["poses"]]; xop.Set(Gf.Matrix4d(poses[0].T.tolist()))
for _ in range(500): step()
t0 = time.time(); meta = []; corners = []; sx = W / sc["img"][0]
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
    rows = []
    for r_ in sc["labels"][k]:                                           # corner hidden behind nearer geometry (depth image, 0.35 m tolerance as DroneRacing) -> not visible
        vis = []
        for (u, v), z, v0 in zip(r_["corners_px"], r_["depths"], r_["visible"]):
            ui, vi = int(u * sx), int(v * sx)
            vis.append(bool(v0 and 0 <= ui < W and 0 <= vi < H and (dep[vi, ui] <= 0 or z <= dep[vi, ui] + 0.35)))
        rows.append({**r_, "visible_depth": vis, "n_visible_depth": sum(vis)})
    corners.append(rows)
    write_png(out / "rgb" / f"{k:04d}.png", rgb); write_png(out / "label" / f"{k:04d}.png", lab); np.save(out / "depth" / f"{k:04d}.npy", dep)
    meta.append({"transform_matrix": m.tolist(), "speed_mps": cc["speed_mps"]})
    if k % 20 == 0: print("frame", k, {CLASSES[i]: int((lab == i).sum()) for i in np.unique(lab) if i != 255}, flush=True)
json.dump(corners, open(out / "corners.json", "w"))
json.dump({"fx": FX, "width": W, "height": H, "light": sc["light"], "traj": "race", "setting": sc["setting"], "layout_hash": sc["hash"], "frames": meta, "boot_s": round(t0 - t_boot, 1), "render_s_per_frame": round((time.time() - t0) / len(poses), 2)}, open(out / "poses.json", "w"))
print("DONE", name, flush=True); app.close()
