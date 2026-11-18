"""CLOSED-LOOP version of sim/isaac_race.py: builds the same resolved racing scene (runs/race_job/scene.json) and then renders on request instead of along the stored camera path.
Protocol (folder runs/race_job/loop): the controller (scripts/closed_loop_race.py) writes req_<k>.json = {"pos": [x,y,z], "forward": [..], "up": [..]}; this script renders that view, writes rgb_<k>.png and then the
empty file ack_<k>; a file named stop ends the run. Run: ISAAC_PY=... bash sim/run_isaac.sh sim/isaac_race_loop.py (the scene-building part below is copied unchanged from isaac_race.py)."""
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
LOOP = JOB / "loop"; LOOP.mkdir(parents=True, exist_ok=True)
for f in LOOP.glob("*"): f.unlink()
p0 = cc["poses"][0]; xop.Set(Gf.Matrix4d(c2w(p0["pos"], p0["forward"], p0["up"]).T.tolist()))
for _ in range(500): step()
(LOOP / "ready").write_text(json.dumps({"fx": FX, "width": W, "height": H})); print("READY", flush=True); k = 0; t0 = time.time(); idle = time.time()
while not (LOOP / "stop").exists():
    if (LOOP / "reset").exists():      # a new controller run on the same scene: frame numbers start again at 0
        k = 0; (LOOP / "reset").unlink()
    rq = LOOP / f"req_{k}.json"
    if not rq.exists():
        time.sleep(0.002)
        if time.time() - idle > 600: print("TIMEOUT waiting for the controller", flush=True); break
        continue
    q = json.loads(rq.read_text()); xop.Set(Gf.Matrix4d(c2w(q["pos"], q["forward"], q["up"]).T.tolist()))
    for _ in range(int(_os.environ.get("RENDER_STEPS", "4"))): step()
    rgb = np.asarray(cam.get_current_frame()["rgb"])
    if rgb.ndim != 3: print("EMPTY render at request", k, flush=True); break
    np.save(LOOP / f"rgb_{k}.npy", rgb[..., :3].astype(np.uint8)); (LOOP / f"ack_{k}").write_text(""); rq.unlink(); k += 1; idle = time.time()
    if k % 100 == 0: print("served", k, "frames,", round((time.time() - t0) / k, 3), "s per frame", flush=True)
print("DONE", name, k, flush=True); app.close()
