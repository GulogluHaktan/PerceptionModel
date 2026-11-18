"""Put labels ON the Gaussians: vote per Gaussian over the labelled views, then every new render comes with a label map.
Modes:
  fuse   <scene.pt> <orbit_dir> <labelled_frame_ids,comma> <out.npz> [labels_npz_dir]
         CPU only. Needs the per-frame label maps: run scripts/label_consistency.py first (it writes labels_NNNN.npy into its out dir).
  render <scene.pt> <fused.npz> <poses.json> <out_dir> [width=800] [sh_degree=3]     (needs gsplat + CUDA: run on the Orin)
         renders a class map (one-hot colours, argmax) next to the RGB for every pose.
A Gaussian gets class c if the visible views vote for c; its confidence is votes_for_c / total_votes. Visibility = the Gaussian centre is
within 3% of the rendered depth at its projected pixel in that view."""
import json, math, sys
from pathlib import Path
import numpy as np

mode = sys.argv[1]
if mode == "fuse":
    ckpt, root, ids, out = sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(","), sys.argv[5]
    ldir = Path(sys.argv[6]) if len(sys.argv) > 6 else root
    max_gh = float(sys.argv[8]) if len(sys.argv) > 8 else float('inf')     # geometric prior: a Gaussian higher than this above the ground plane (scene units) cannot be class ground_road (4)
    min_conf = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0         # Gaussians whose winning class has less than this share of the votes become 'unlabelled' (255)
    import torch
    from PIL import Image
    means = torch.load(ckpt)["means"].float().numpy()
    meta = json.load(open(root / "transforms_orbit.json")); fr = {Path(f["file_path"]).name: f for f in meta["frames"]}
    W = np.asarray(Image.open(root / "depth" / f"{ids[0]}.png")).shape[0]
    fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"]); cx = W / 2
    NC = 11; votes = np.zeros((len(means), NC), np.int32); per_view = []
    for i in ids:
        seg = np.load(ldir / f"labels_{i}.npy"); d = np.asarray(Image.open(root / "depth" / f"{i}.png")).astype(np.float32) / 1000
        c2w = np.array(fr[i]["transform_matrix"], dtype=np.float64); c2w[:3, 1:3] *= -1
        pc = (np.linalg.inv(c2w) @ np.c_[means, np.ones(len(means))].T).T
        z = pc[:, 2]; ok = z > 0.1
        u = np.round(fx * pc[:, 0] / np.where(ok, z, 1) + cx).astype(int); v = np.round(fx * pc[:, 1] / np.where(ok, z, 1) + cx).astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < W)
        idx = np.where(ok)[0]; vis = np.abs(d[v[idx], u[idx]] - z[idx]) < 0.03 * z[idx]; idx = idx[vis]
        lab = seg[v[idx], u[idx]]; keep = lab != 255
        np.add.at(votes, (idx[keep], lab[keep]), 1); per_view.append((i, int(keep.sum())))
    if np.isfinite(max_gh):
        sys.path.insert(0, "src"); from pfm.ground import fit_ground_plane
        n_g, d_g, _ = fit_ground_plane(means); gh = means @ n_g + d_g
        votes[gh > max_gh, 4] = 0                                           # drop the 'ground' votes of Gaussians that are clearly above the ground
    tot = votes.sum(1); cls = votes.argmax(1); conf = np.where(tot > 0, votes.max(1) / np.maximum(tot, 1), 0)
    cls[tot == 0] = 255
    cls[(conf < min_conf) & (tot > 0)] = 255
    # how often does each view agree with the fused label of the Gaussians it sees?  (computed per view again)
    agree = {}
    for i in ids:
        seg = np.load(ldir / f"labels_{i}.npy"); d = np.asarray(Image.open(root / "depth" / f"{i}.png")).astype(np.float32) / 1000
        c2w = np.array(fr[i]["transform_matrix"], dtype=np.float64); c2w[:3, 1:3] *= -1
        pc = (np.linalg.inv(c2w) @ np.c_[means, np.ones(len(means))].T).T; z = pc[:, 2]; ok = z > 0.1
        u = np.round(fx * pc[:, 0] / np.where(ok, z, 1) + cx).astype(int); v = np.round(fx * pc[:, 1] / np.where(ok, z, 1) + cx).astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < W); idx = np.where(ok)[0]; idx = idx[np.abs(d[v[idx], u[idx]] - z[idx]) < 0.03 * z[idx]]
        lab = seg[v[idx], u[idx]]; k = (lab != 255) & (cls[idx] != 255)
        agree[i] = round(float((lab[k] == cls[idx][k]).mean()), 3) if k.any() else None
    np.savez(out, cls=cls, conf=conf, votes=votes)
    print(json.dumps({"gaussians": len(means), "labelled_gaussians": int((cls != 255).sum()), "mean_confidence": round(float(conf[cls != 255].mean()), 3),
                      "gaussians_with_conf_ge_0.8": round(float((conf[cls != 255] >= 0.8).mean()), 3),
                      "class_counts": {int(c): int((cls == c).sum()) for c in np.unique(cls)}, "per_view_agreement_with_fused": agree}))
elif mode == "render":
    import torch, torch.nn.functional as F
    from PIL import Image
    from gsplat.rendering import rasterization
    ckpt, fz, poses, out = sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
    W = int(sys.argv[6]) if len(sys.argv) > 6 else 800; deg = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    dev = "cuda"; p = {k: v.float().to(dev) for k, v in torch.load(ckpt).items()}
    z = np.load(fz); cls = torch.tensor(z["cls"].astype(np.int64), device=dev)
    meta = json.load(open(poses)); fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
    K = torch.tensor([[fx, 0, W / 2], [0, fx, W / 2], [0, 0, 1.0]], device=dev)
    NC = 11; onehot = F.one_hot(torch.where(cls == 255, torch.full_like(cls, NC - 1), cls), NC).float()   # unlabelled Gaussians vote for the last slot
    pal = np.array([[60, 60, 60], [255, 0, 0], [0, 140, 0], [255, 200, 0], [120, 120, 120], [0, 0, 255], [255, 0, 255], [0, 255, 255], [160, 80, 0], [255, 255, 255], [255, 128, 128]], np.uint8)
    for k, fr in enumerate(meta["frames"][:int(sys.argv[8]) if len(sys.argv) > 8 else 12]):
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32, device=dev); c2w[:3, 1:3] *= -1; vm = torch.linalg.inv(c2w)[None]
        with torch.no_grad():
            rgb = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), torch.cat([p["sh0"], p["shN"]], 1),
                                vm, K[None], W, W, sh_degree=deg, packed=False, backgrounds=torch.ones(1, 3, device=dev))[0][0].clamp(0, 1)
            lab, alpha, _ = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), onehot,
                                          vm, K[None], W, W, packed=False)
        cm = lab[0].argmax(-1).cpu().numpy(); cm[(alpha[0, ..., 0] < 0.5).cpu().numpy()] = 9      # background -> white slot
        r8 = (rgb.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([r8, pal[np.minimum(cm, 10)], (0.55 * r8 + 0.45 * pal[np.minimum(cm, 10)]).astype(np.uint8)], 1)).save(out / f"{k:04d}.png")
    print("rendered", min(len(meta["frames"]), 12), "views to", out)
