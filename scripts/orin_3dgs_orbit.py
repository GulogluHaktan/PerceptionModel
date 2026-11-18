"""Fly-through from a trained 3DGS scene: render RGB + expected depth along a NEW camera orbit, save them as a small
labelled image set, validate the depth against Blender's ground-truth depth on the test views, and write a GIF.
Usage (Orin): python scripts/orin_3dgs_orbit.py <scene_dir> <scene.pt> [width=800] [n_views=90] [sh_degree=3]
Outputs in runs/3dgs_orbit/: rgb/NNNN.png, depth/NNNN.png (16-bit, depth in millimetre-like units = metres*1000),
preview/NNNN.png (RGB | depth colour), transforms_orbit.json (poses), flythrough.gif, stats.json."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization

root, ckpt = Path(sys.argv[1]), sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 800
nv = int(sys.argv[4]) if len(sys.argv) > 4 else 90
deg = int(sys.argv[5]) if len(sys.argv) > 5 else 3
dev = "cuda"
p = {k: v.float().to(dev) for k, v in torch.load(ckpt).items()}
meta = json.load(open(root / "transforms_test.json"))
fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
K = torch.tensor([[fx, 0, W / 2], [0, fx, W / 2], [0, 0, 1.0]], device=dev)
out = Path("runs/3dgs_orbit")
for d in ("rgb", "depth", "preview"):
    (out / d).mkdir(parents=True, exist_ok=True)


def render(c2w_opencv):
    vm = torch.linalg.inv(c2w_opencv)[None]
    with torch.no_grad():
        r = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                          torch.cat([p["sh0"], p["shN"]], 1), vm, K[None], W, W, sh_degree=deg, packed=False,
                          render_mode="RGB+ED", backgrounds=torch.ones(1, 3, device=dev))[0][0]
    return r[..., :3].clamp(0, 1), r[..., 3]


def turbo(x):                                   # simple colour ramp: near = warm, far = cool (no matplotlib needed)
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


# 1) validate depth on the held-out test views against Blender's depth PNGs
def load_gt_depth(path):
    a = np.asarray(Image.open(path).convert("RGBA").resize((W, W), Image.NEAREST)).astype(np.float32)
    return a[..., 0] / 255, a[..., 3] > 127          # depth channel (relative), object mask


corrs = []
for fr in meta["frames"][::20]:
    dpath = root / (fr["file_path"] + "_depth_0001.png")
    if not dpath.exists():
        continue
    c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32, device=dev); c2w[:3, 1:3] *= -1
    _, d = render(c2w)
    gt, mask = load_gt_depth(dpath)
    d = d.cpu().numpy()
    m = mask & (d > 0)
    if m.sum() < 100:
        continue
    c = np.corrcoef(d[m], gt[m])[0, 1]
    corrs.append(float(c))
# the Blender depth PNG is not metric and may be inverted: take the absolute correlation and report the sign
sign = "same direction" if np.mean(corrs) > 0 else "inverted (far = bright in the GT file)"

# 2) a new orbit that is NOT one of the dataset cameras: elevation 25 deg, radius 4.0, looking at the origin
frames, ims = [], []
for i in range(nv):
    a = 2 * math.pi * i / nv; el = math.radians(25 + 10 * math.sin(2 * a))
    eye = torch.tensor([4.0 * math.cos(el) * math.cos(a), 4.0 * math.cos(el) * math.sin(a), 4.0 * math.sin(el)], device=dev)
    z = -eye / eye.norm(); up = torch.tensor([0., 0., 1.], device=dev)           # Blender scenes are Z-up
    x = torch.cross(z, up, dim=0); x = x / x.norm(); y = torch.cross(z, x, dim=0)   # OpenCV: x right, y down, z forward
    c2w = torch.eye(4, device=dev); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    rgb, d = render(c2w)
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8); d_np = d.cpu().numpy()
    obj = d_np > 0.5
    Image.fromarray(rgb_np).save(out / "rgb" / f"{i:04d}.png")
    Image.fromarray(np.clip(d_np * 1000, 0, 65535).astype(np.uint16)).save(out / "depth" / f"{i:04d}.png")
    lo, hi = (d_np[obj].min(), d_np[obj].max()) if obj.any() else (0, 1)
    dc = turbo(1 - (d_np - lo) / max(hi - lo, 1e-6)); dc[~obj] = 255
    prev = Image.fromarray(np.concatenate([rgb_np, dc], 1)); prev.save(out / "preview" / f"{i:04d}.png")
    ims.append(prev.resize((W // 2 * 2 // 2, W // 4)) if False else prev.resize((W, W // 2)))
    cc = c2w.clone(); cc[:3, 1:3] *= -1                                      # back to OpenGL for the json
    frames.append({"file_path": f"rgb/{i:04d}", "transform_matrix": cc.cpu().tolist()})
json.dump({"camera_angle_x": meta["camera_angle_x"], "frames": frames}, open(out / "transforms_orbit.json", "w"))
ims[0].save(out / "flythrough.gif", save_all=True, append_images=ims[1:], duration=60, loop=0)
stats = {"views": nv, "size": W, "depth_vs_blender_gt_correlation_mean": round(float(np.mean(corrs)), 3), "gt_depth_direction": sign,
         "correlation_per_view": [round(c, 3) for c in corrs]}
json.dump(stats, open(out / "stats.json", "w")); print(json.dumps(stats))
