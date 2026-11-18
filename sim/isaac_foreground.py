"""Isaac Sim foreground pass: renders ONLY the scenario objects (boxes from objects.json) for every pose of poses.json with the Isaac Camera sensor
(pattern of the working LunarRocket scripts). Outputs under runs/fg_job/out: rgb/NNNN.png, depth/NNNN.npy (metres along the optical axis, 0 = nothing),
label/NNNN.png (class ids, 255 = none). scripts/make_sim_dataset.py composites them over the 3DGS background. Run with sim/run_isaac.sh sim/isaac_foreground.py."""
import os as _os; PFM = _os.environ.get("PFM_ROOT", "/workspace/pfm")   # container default; native Isaac sets PFM_ROOT to the project folder
import json, math, struct, time, zlib
t_boot = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480, "disable_viewport_updates": False})
import numpy as np
import omni.replicator.core as rep                                       # importing it loads the Replicator extension the Camera sensor annotators need
import omni.usd
from pathlib import Path
from pxr import Gf, UsdGeom, UsdLux
try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World
try:
    from isaacsim.core.utils.semantics import add_labels
except ImportError:
    from isaacsim.core.utils.semantics import add_update_semantics as _old
    add_labels = lambda prim, labels: _old(prim, labels[0])
try:
    from isaacsim.sensors.camera import Camera
except ImportError:
    from omni.isaac.sensor import Camera

JOB = Path((PFM + "/runs/fg_job")); out = JOB / "out"
for d in ("rgb", "depth", "label"):
    (out / d).mkdir(parents=True, exist_ok=True)
cam_j = json.load(open(JOB / "camera.json")); poses = json.load(open(JOB / "poses.json")); objs = json.load(open(JOB / "objects.json"))
W, H, FX = int(cam_j["width"]), int(cam_j["height"]), float(cam_j["fx"])
NAME = {1: "gate", 2: "tree_pole", 4: "ground_road", 5: "building"}; CLASS = {v: k for k, v in NAME.items()}

def write_png(path, a):
    """Minimal PNG writer (uint8 HxW or HxWx3): the Pillow inside the Isaac Sim container crashes on save ('_idat' has no fileno)."""
    a = np.ascontiguousarray(a, dtype=np.uint8); h, w = a.shape[:2]; ct = 0 if a.ndim == 2 else 2
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
from pxr import Gf, UsdGeom, UsdLux


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage(); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
for i, o in enumerate(objs):
    c = UsdGeom.Cube.Define(stage, f"/World/obj{i}"); c.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(c); x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in o["center"]])); x.AddRotateZOp().Set(float(math.degrees(o["yaw"]))); x.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in o["size"]]))
    UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in o["color"]])]); add_labels(c.GetPrim(), [NAME[int(o["label"])]])
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(900.0); UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 20, 0))
UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(200.0)
cam = Camera(prim_path="/World/Camera", resolution=[W, H]); cam.initialize()
cg = UsdGeom.Camera(cam.prim); cg.GetFocalLengthAttr().Set(18.0); cg.GetHorizontalApertureAttr().Set(18.0 * W / FX); cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 400.0))   # fx matches the scenario camera
cam.add_rgb_to_frame(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame()
world.reset()
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


def set_pose(m):                                                         # m: OpenGL camera-to-world (the USD camera uses the same axes)
    m = np.asarray(m, float); q = rot_to_quat_wxyz(m[:3, :3]); t_op.Set(Gf.Vec3d(*[float(v) for v in m[:3, 3]])); o_op.Set(Gf.Quatd(*[float(v) for v in q]))


def step():
    world.step(render=True); rep.orchestrator.step(); app.update()      # the orchestrator step logs "renderer failed to advance" but it is what fills the annotator data (the working LunarRocket scripts do it too)


set_pose(poses[0]["transform_matrix"])
for _ in range(120):                                                     # long warm-up: this scene is tiny and the first frames can be empty
    step()
t0 = time.time()
for k, p in enumerate(poses):
    set_pose(p["transform_matrix"]); step(); step()
    fr = cam.get_current_frame(); rgb = fr.get("rgb")
    if k == 0:
        print("frame keys", {kk: (list(v.keys()) if isinstance(v, dict) else np.asarray(v).shape) for kk, v in fr.items()}, flush=True)
    if rgb is None or np.asarray(rgb).ndim != 3:
        print("EMPTY render data at frame", k, flush=True); app.close(); raise SystemExit(2)
    rgb = np.asarray(rgb)[..., :3].astype(np.uint8); dep = np.asarray(fr["distance_to_image_plane"]).astype(np.float32)
    sem = fr["semantic_segmentation"]; sd = np.squeeze(sem["data"]); id2l = sem["info"]["idToLabels"]
    lab = np.full(sd.shape[:2], 255, np.uint8)
    for sid, d in id2l.items():
        for nm in (d.values() if isinstance(d, dict) else [d]):
            if nm in CLASS:
                lab[sd == int(sid)] = CLASS[nm]
    dep = np.where(np.isfinite(dep), dep, 0).astype(np.float32); dep[lab == 255] = 0                       # only object pixels count as foreground
    write_png(out / "rgb" / f"{k:04d}.png", rgb); write_png(out / "label" / f"{k:04d}.png", lab); np.save(out / "depth" / f"{k:04d}.npy", dep)
    if k % 10 == 0:
        print("frame", k, "object pixels", int((lab != 255).sum()), flush=True)
st = {"frames": len(poses), "boot_s": round(t0 - t_boot, 1), "render_s_per_frame": round((time.time() - t0) / len(poses), 2)}
json.dump(st, open(out / "stats.json", "w")); print("DONE", st, flush=True)
app.close()
