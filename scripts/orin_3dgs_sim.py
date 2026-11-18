"""A small 'simulator' built from real/captured data: fly a virtual drone camera through a reconstructed 3DGS scene and record
RGB + depth + class labels + the exact camera poses, with camera effects (motion blur, sensor noise, exposure jitter).
Usage (Orin): python scripts/orin_3dgs_sim.py <scene.pt> <meta.json> <out_dir> [fused.npz] [frames=120] [width=640] [blur=5] [noise=0.02]
<meta.json>: any json with camera_angle_x (the dataset transforms file). Blender-style scene: Z up, content near the origin, radius ~4.
The flight path is a shrinking spiral with an altitude wave and a wandering look-at point (not a dataset camera). Labels come from
fused per-Gaussian labels (scripts/gaussian_label_fusion.py fuse). Outputs: rgb/ depth/ (16-bit, units*1000) label/ (class index, 255 = none),
poses.json (OpenGL camera-to-world, per frame, ground truth), preview.gif (RGB | labels | depth), stats.json (time per frame)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization

ckpt, meta_p, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
fused = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
nf = int(sys.argv[5]) if len(sys.argv) > 5 else 120
W = int(sys.argv[6]) if len(sys.argv) > 6 else 640
nblur = int(sys.argv[7]) if len(sys.argv) > 7 else 5
sigma = float(sys.argv[8]) if len(sys.argv) > 8 else 0.02
dev = "cuda"; rng = np.random.default_rng(0); deg = 3
for d in ("rgb", "depth", "label", "prev"):
    (out / d).mkdir(parents=True, exist_ok=True)
p = {k: v.float().to(dev) for k, v in torch.load(ckpt).items()}
fx = 0.5 * W / math.tan(0.5 * json.load(open(meta_p))["camera_angle_x"])
K = torch.tensor([[fx, 0, W / 2], [0, fx, W / 2], [0, 0, 1.0]], device=dev)
NC = 11; onehot = None
if fused:
    cls = torch.tensor(np.load(fused)["cls"].astype(np.int64), device=dev)
    onehot = F.one_hot(torch.where(cls == 255, torch.full_like(cls, NC - 1), cls), NC).float()
pal = np.array([[60, 60, 60], [255, 0, 0], [0, 140, 0], [255, 200, 0], [120, 120, 120], [0, 0, 255], [255, 0, 255], [0, 255, 255], [160, 80, 0], [255, 255, 255], [255, 128, 128]], np.uint8)


def pose(t):                                           # camera-to-world in OpenCV axes, Z-up world
    a = 2 * math.pi * 1.25 * t; r = 4.2 - 1.6 * t; h = 1.6 + 1.2 * math.sin(2 * math.pi * 2 * t)
    eye = np.array([r * math.cos(a), r * math.sin(a), h]); tgt = np.array([0.4 * math.sin(6 * t), 0.4 * math.cos(5 * t), 0.2])
    z = tgt - eye; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def rgb_at(c2w, mode="RGB"):
    vm = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))[None]
    with torch.no_grad():
        return rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                             torch.cat([p["sh0"], p["shN"]], 1), vm, K[None], W, W, sh_degree=deg, packed=False, render_mode=mode,
                             backgrounds=torch.ones(1, 3, device=dev))[0][0]


def turbo(x):
    x = np.clip(x, 0, 1); r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


poses, gif, times = [], [], []
shutter = 0.5 / nf                                    # exposure = half a frame interval
for k in range(nf):
    t = k / nf; torch.cuda.synchronize(); t0 = time.perf_counter()
    acc = sum(rgb_at(pose(t + shutter * (s / max(1, nblur - 1) - 0.5)))[..., :3] for s in range(nblur)) / nblur     # motion blur
    c2w = pose(t); r = rgb_at(c2w, "RGB+ED"); depth = r[..., 3]
    img = (acc * float(rng.uniform(0.9, 1.1)) + torch.randn_like(acc) * sigma).clamp(0, 1)                            # exposure + sensor noise
    lab = None
    if onehot is not None:
        vm = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))[None]
        with torch.no_grad():
            lo, al, _ = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), onehot,
                                      vm, K[None], W, W, packed=False)
        lab = lo[0].argmax(-1); lab[lab == NC - 1] = 255; lab[al[0, ..., 0] < 0.5] = 255; lab = lab.to(torch.uint8)
    torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
    i8 = (img.cpu().numpy() * 255).astype(np.uint8); dn = depth.cpu().numpy()
    Image.fromarray(i8).save(out / "rgb" / f"{k:04d}.png"); Image.fromarray(np.clip(dn * 1000, 0, 65535).astype(np.uint16)).save(out / "depth" / f"{k:04d}.png")
    obj = dn > 0.5; lo_, hi_ = (dn[obj].min(), dn[obj].max()) if obj.any() else (0, 1)
    dc = turbo(1 - (dn - lo_) / max(hi_ - lo_, 1e-6)); dc[~obj] = 255
    cols = [i8]
    if lab is not None:
        l8 = lab.cpu().numpy(); Image.fromarray(l8).save(out / "label" / f"{k:04d}.png"); cols.append(np.where((l8 == 255)[..., None], 255, pal[np.minimum(l8, 10)]).astype(np.uint8))
    cols.append(dc); fr = Image.fromarray(np.concatenate(cols, 1)); fr.save(out / "prev" / f"{k:04d}.png"); gif.append(fr.resize((fr.width // 2, fr.height // 2)))
    g = c2w.copy(); g[:3, 1:3] *= -1; poses.append({"file_path": f"rgb/{k:04d}", "t": t, "transform_matrix": g.tolist()})
json.dump({"camera_angle_x": json.load(open(meta_p))["camera_angle_x"], "frames": poses}, open(out / "poses.json", "w"))
gif[0].save(out / "preview.gif", save_all=True, append_images=gif[1:], duration=50, loop=0)
st = {"frames": nf, "size": W, "blur_subframes": nblur, "noise_sigma": sigma, "labels": fused is not None,
      "median_ms_per_frame": round(float(np.median(times)), 1), "frames_per_s": round(1000 / float(np.median(times)), 1)}
json.dump(st, open(out / "stats.json", "w")); print(json.dumps(st))
