"""End-to-end data factory (plan steps 1-7): scenario JSON -> trajectory -> 3DGS background (RGB, depth, voted labels) -> objects (proxy renderer now,
Isaac Sim later) -> depth-aware compose -> camera/weather effects -> dataset with exact gate corners and provenance. Needs gsplat + CUDA (run on the Orin).
Usage: python scripts/make_sim_dataset.py <scene.pt> <fused.npz|-> <scenario.json> <out_dir> [isaac_fg_dir]"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "src")
from gsplat.rendering import rasterization
from pfm.camera import Camera, to_opengl, scale_pose
from pfm.dataset_writer import DatasetWriter
from pfm.ground import fit_ground_plane, place_on_plane
from pfm.frames import aligned_to_scene, rotation_from_up, transform_boxes
from pfm.objects import build_objects
from pfm.scenario import validate_scenario
from pfm.simulate import generate_frame
from pfm.trajectory import make_trajectory, motions

ckpt, fused, sc_path, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
fg_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else None          # optional: foreground frames rendered by Isaac Sim (sim/isaac_foreground.py)
sc = validate_scenario(json.load(open(sc_path))); mpu = float(sc["scene"]["meters_per_unit"]); dev = "cuda"
cam = Camera.from_hfov_deg(sc["camera"]["width"], sc["camera"]["height"], sc["camera"]["hfov_deg"])
p = {k: v.float().to(dev) for k, v in torch.load(ckpt).items()}
NC = 11; onehot = None
if fused != "-":
    cls = torch.tensor(np.load(fused)["cls"].astype(np.int64), device=dev); onehot = F.one_hot(torch.where(cls == 255, torch.full_like(cls, NC - 1), cls), NC).float()
# everything scenario-related is built in an ALIGNED frame (scene up = +Z), then rotated into the scene frame (identity for Z-up scenes such as lego)
R_al = rotation_from_up(sc["scene"]["up"]); T = aligned_to_scene(R_al)
gz = sc["scene"].get("ground_z")
plane_a = (np.array([0.0, 0.0, 1.0]), -float(gz), 0) if gz is not None else fit_ground_plane(p["means"].cpu().numpy() @ R_al.T, up=(0, 0, 1))
print("ground plane (aligned frame) normal", np.round(plane_a[0], 3), "d", round(plane_a[1], 3), "inliers", plane_a[2], flush=True)
boxes_a, gates_a = build_objects(sc, plane_a, mpu)
boxes = transform_boxes(boxes_a, T); gates = [(T[:3, :3] @ g.T).T for g in gates_a]
plane = (T[:3, :3] @ plane_a[0], plane_a[1]); light = T[:3, :3] @ np.array([0.35, 0.25, -0.9])


poses_m, fps = make_trajectory(sc["trajectory"]); poses_unit = [T @ scale_pose(m, 1 / mpu) for m in poses_m]      # metres in the aligned frame -> scene units in the scene frame
mot = motions(poses_m, fps); print("frames", len(poses_unit), "effective fps", round(fps, 1), flush=True)
K = torch.tensor([[cam.fx, 0, cam.cx], [0, cam.fx, cam.cy], [0, 0, 1.0]], device=dev)
SH = lambda: torch.cat([p["sh0"], p["shN"]], 1)


def render_bg(c2w):
    vm = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))[None]
    with torch.no_grad():
        r, al, _ = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), SH(), vm, K[None], cam.width, cam.height,
                                 sh_degree=3, packed=False, render_mode="RGB+ED", backgrounds=torch.ones(1, 3, device=dev))
        rgb, dep, a = r[0, ..., :3].clamp(0, 1), r[0, ..., 3], al[0, ..., 0]
        dep = torch.where(a > 0.5, dep, torch.zeros_like(dep))                    # scene units; converted to metres at the end of generate_frame
        if onehot is not None:
            lo, la, _ = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), onehot, vm, K[None], cam.width, cam.height, packed=False)
            lab = lo[0].argmax(-1); lab[lab == NC - 1] = 255; lab[la[0, ..., 0] < 0.5] = 255; lab = lab.to(torch.uint8)
        else:
            lab = torch.full((cam.height, cam.width), 255, dtype=torch.uint8, device=dev)
    return {"rgb": rgb.cpu().numpy(), "depth": dep.cpu().numpy(), "label": lab.cpu().numpy()}


wr = DatasetWriter(out, json.load(open(sc_path)), extra={"scene_file": Path(ckpt).name, "effective_fps": fps}); sheet, times = [], []
pal = np.array([[60, 60, 60], [255, 120, 0], [0, 140, 0], [255, 200, 0], [120, 120, 120], [0, 0, 255], [255, 0, 255], [0, 255, 255], [160, 80, 0], [255, 255, 255], [255, 128, 128]], np.uint8)
for k, (pm, pu) in enumerate(zip(poses_m, poses_unit)):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    bg = render_bg(pu)
    fg = None
    if fg_dir is not None:
        fg_rgb = np.asarray(Image.open(fg_dir / "rgb" / f"{k:04d}.png").convert("RGB")).astype(np.float32) / 255
        fg_d = np.load(fg_dir / "depth" / f"{k:04d}.npy"); fg_l = np.asarray(Image.open(fg_dir / "label" / f"{k:04d}.png")); fg = (fg_rgb, fg_d, fg_l, (fg_d > 0).astype(np.float32))
    rgb, depth, lab, corners, vis = generate_frame(cam, pu, bg, boxes, gates, sc["effects"], mot[k], seed=sc["seed"] + k, device=dev, fg=fg, shadow=(plane, light, boxes), depth_scale=mpu)
    torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
    wr.add(k, rgb, depth, lab, to_opengl(scale_pose(pu, mpu)), {"v": [round(float(x), 3) for x in mot[k]["v"]], "w": [round(float(x), 3) for x in mot[k]["w"]]}, {"gate_corners": corners})
    if k % max(1, len(poses_m) // 6) == 0:
        lo_, hi_ = (np.percentile(depth[depth > 0], [2, 98]) if (depth > 0).any() else (0.0, 1.0)); d8 = np.clip(1 - (depth - lo_) / max(hi_ - lo_, 1e-6), 0, 1)[..., None] * np.array([255, 255, 255]); d8 = np.where((depth > 0)[..., None], d8, 255).astype(np.uint8)    # per-frame percentile range: works for a 0.5 m scene and a 4 m scene
        lc = np.where((lab == 255)[..., None], 255, pal[np.minimum(lab, 10)]).astype(np.uint8)
        im = (rgb * 255).astype(np.uint8).copy()
        for cs in corners:
            for (u, v, vi) in cs:
                if 0 <= u < cam.width and 0 <= v < cam.height:
                    c = (0, 255, 0) if vi else (255, 0, 0); x, y = int(u), int(v); im[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = c
        sheet.append(np.concatenate([im, lc, d8.astype(np.uint8)], 1))
h = wr.close()
Image.fromarray(np.concatenate(sheet, 0)).resize((960, 240 * len(sheet))).save(out / "sheet.png")
st = {"scenario_hash": h, "frames": len(poses_m), "ms_per_frame_median": round(float(np.median(times)), 1), "fps": round(1000 / float(np.median(times)), 1),
      "gate_corner_visible_frames": int(sum(1 for fr in wr.frames if any(c[2] for cs in fr["extra"]["gate_corners"] for c in cs)))}
json.dump(st, open(out / "stats.json", "w")); print(json.dumps(st))
