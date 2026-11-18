"""Render held-out test views from a trained 3DGS scene and put them next to the ground truth.
Usage (Orin): python scripts/orin_3dgs_render.py <scene_dir> <scene.pt> [width=800] [sh_degree=3] [n_views=6]
Writes runs/3dgs_viz/sheet.png (rows: ground truth | rendered | amplified difference) and full-size pairs."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from gsplat.rendering import rasterization

root, ckpt = Path(sys.argv[1]), sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 800
deg = int(sys.argv[4]) if len(sys.argv) > 4 else 3
nv = int(sys.argv[5]) if len(sys.argv) > 5 else 6
dev = "cuda"
p = {k: v.float().to(dev) for k, v in torch.load(ckpt).items()}
meta = json.load(open(root / "transforms_test.json"))
fx = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
K = torch.tensor([[fx, 0, W / 2], [0, fx, W / 2], [0, 0, 1.0]], device=dev)
frames = meta["frames"][::10]
pick = np.linspace(0, len(frames) - 1, nv).astype(int)
out = Path("runs/3dgs_viz"); out.mkdir(parents=True, exist_ok=True)
rows, psnrs = [], []
for i in pick:
    fr = frames[i]
    im = Image.open(root / (fr["file_path"] + ".png")).convert("RGBA").resize((W, W), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255
    gt = torch.from_numpy(a[..., :3] * a[..., 3:] + (1 - a[..., 3:])).to(dev)
    c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32, device=dev); c2w[:3, 1:3] *= -1
    with torch.no_grad():
        r = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                          torch.cat([p["sh0"], p["shN"]], 1), torch.linalg.inv(c2w)[None], K[None], W, W, sh_degree=deg, packed=False,
                          backgrounds=torch.ones(1, 3, device=dev))[0][0].clamp(0, 1)
    psnr = float(-10 * torch.log10(F.mse_loss(r, gt))); psnrs.append(psnr)
    diff = ((r - gt).abs() * 4).clamp(0, 1)
    to8 = lambda t: Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
    g, o, d = to8(gt), to8(r), to8(1 - diff)           # difference shown dark-on-white: darker = bigger error
    d_ = ImageDraw.Draw(o); d_.rectangle([0, 0, 190, 26], fill=(255, 255, 255)); d_.text((6, 6), f"PSNR {psnr:.1f} dB", fill=(0, 0, 0))
    rows.append((g, o, d))
    if len(rows) <= 2:
        pair = Image.new("RGB", (W * 2, W)); pair.paste(g, (0, 0)); pair.paste(o, (W, 0)); pair.save(out / f"pair_{len(rows)}.png")
S = 330
sheet = Image.new("RGB", (S * 3, S * len(rows) + 28), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for j, t in enumerate(["Gercek (egitimde YOK)", "Uretilen (3DGS)", "Fark (koyu = hata)"]):
    dr.text((j * S + 6, 8), t, fill=(0, 0, 0))
for r_i, row in enumerate(rows):
    for c, im in enumerate(row):
        sheet.paste(im.resize((S, S), Image.LANCZOS), (c * S, 28 + r_i * S))
sheet.save(out / "sheet.png")
print(json.dumps({"views": [int(i) for i in pick], "psnr_db": [round(x, 1) for x in psnrs], "mean_psnr_db": round(float(np.mean(psnrs)), 2), "gaussians": int(p["means"].shape[0])}))
