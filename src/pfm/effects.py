"""Camera and weather effects for synthetic frames (shared by the 3DGS renders and Isaac Sim): physics-based, parametric, deterministic given a seed.
Input: image (H,W,3) float [0,1], depth (H,W) in metres (0 or inf = no surface / sky), camera intrinsics fx, and the camera motion
(velocity v and angular rate w in the CAMERA frame: x right, y down, z forward; m/s and rad/s).
Pipeline order (like a real camera): motion -> rolling shutter -> lens -> atmosphere (fog, rain) -> exposure + sensor noise -> compression.
`pipeline(img, depth, fx, motion, cfg, seed)` applies it; `validate_cfg` is the strict schema an LLM scenario manager fills in later
(unknown keys and out-of-range values are rejected, never silently accepted)."""
from __future__ import annotations

import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEFAULTS = {
    "exposure_s": 1 / 500,     # shutter time (s): motion blur length
    "readout_s": 0.0,          # sensor readout time top to bottom (s): rolling shutter shear (0 = global shutter)
    "k1": 0.0, "k2": 0.0,      # radial lens distortion
    "vignette": 0.0,           # 0..1 corner darkening
    "fog_beta": 0.0,           # extinction coefficient (1/m): visibility about 3.9/beta
    "fog_airlight": 0.8,       # fog colour (grey level)
    "rain_density": 0.0,       # 0..1 visible drop count (1 = very heavy rain); also adds haze
    "rain_fall_speed": 8.0,    # m/s terminal velocity of the drops
    "wind_x": 0.0, "wind_z": 0.0,   # wind in the camera frame (m/s): tilts the streaks
    "ev": 0.0,                 # scene exposure change in stops (negative = darker)
    "gain_ev": 0.0,            # digital/analog gain applied AFTER the sensor noise (stops)
    "iso": 100.0,              # noise level reference (higher = noisier)
    "jpeg_quality": 0,         # 0 = none, else 1..100
}
RANGES = {"exposure_s": (0, 0.1), "readout_s": (0, 0.05), "k1": (-0.5, 0.5), "k2": (-0.5, 0.5), "vignette": (0, 1), "fog_beta": (0, 2),
          "fog_airlight": (0, 1), "rain_density": (0, 1), "rain_fall_speed": (3, 12), "wind_x": (-20, 20), "wind_z": (-20, 20), "ev": (-6, 3),
          "gain_ev": (-3, 8), "iso": (50, 12800), "jpeg_quality": (0, 100)}


def validate_cfg(cfg: dict) -> dict:
    """Strict: unknown keys or out-of-range values raise ValueError. Returns the full config with defaults filled in."""
    bad = set(cfg) - set(DEFAULTS)
    if bad:
        raise ValueError(f"unknown effect keys: {sorted(bad)}")
    out = {**DEFAULTS, **cfg}
    for k, (lo, hi) in RANGES.items():
        if not lo <= float(out[k]) <= hi:
            raise ValueError(f"{k}={out[k]} outside [{lo}, {hi}]")
    return out


def pixel_flow(depth, fx, v, w):
    """Image-plane velocity (px/s) of every pixel for a camera with translational velocity v and angular rate w (camera frame)."""
    H, W = depth.shape
    z = torch.where(torch.isfinite(depth) & (depth > 1e-3), depth, torch.full_like(depth, 1e3))      # no surface: far away
    vv, uu = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    x, y = (uu - W / 2 + 0.5) / fx * z, (vv - H / 2 + 0.5) / fx * z
    vx, vy, vz = v; wx, wy, wz = w
    xd = -vx - (wy * z - wz * y); yd = -vy - (wz * x - wx * z); zd = -vz - (wx * y - wy * x)        # point velocity in the camera frame
    return torch.stack([fx * (xd * z - x * zd) / z ** 2, fx * (yd * z - y * zd) / z ** 2], -1)         # (H,W,2) px/s


def _grid(H, W, dev):
    vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    return torch.stack([(uu + 0.5) * 2 / W - 1, (vv + 0.5) * 2 / H - 1], -1)[None].float()


def motion_blur(img, flow, exposure_s, taps=15, max_px=60.0):
    """img (1,3,H,W); flow (H,W,2) px/s. Averages `taps` samples along the pixel path during the exposure."""
    H, W = img.shape[-2:]
    d = (flow * exposure_s).clamp(-max_px, max_px)
    n = int(max(3, min(taps, 2 + float(d.abs().max()) * 1.2)))
    base, dn = _grid(H, W, img.device), d[None] * torch.tensor([2 / W, 2 / H], device=img.device)
    acc = sum(F.grid_sample(img, base + (k / (n - 1) - 0.5) * dn, mode="bilinear", padding_mode="border", align_corners=False) for k in range(n))
    return acc / n


def rolling_shutter(img, flow, readout_s):
    """Row r is exposed at t = (r/H - 0.5) * readout_s: the content moves by flow * t before that row is read."""
    H, W = img.shape[-2:]
    t = ((torch.arange(H, device=img.device) / H - 0.5) * readout_s)[:, None, None]
    d = flow * t * torch.tensor([2 / W, 2 / H], device=img.device)
    return F.grid_sample(img, _grid(H, W, img.device) - d[None], mode="bilinear", padding_mode="border", align_corners=False)


def lens(img, k1, k2, vignette):
    H, W = img.shape[-2:]
    g = _grid(H, W, img.device); r2 = (g ** 2).sum(-1, keepdim=True)
    out = F.grid_sample(img, g * (1 + k1 * r2 + k2 * r2 ** 2), mode="bilinear", padding_mode="border", align_corners=False) if (k1 or k2) else img
    return out * (1 - vignette * r2.clamp(0, 2).permute(0, 3, 1, 2) / 2) if vignette else out


def fog(img, depth, beta, airlight):
    """Koschmieder: I = J t + A (1 - t), t = exp(-beta * depth). Pixels without a surface are fully fogged."""
    d = torch.where(torch.isfinite(depth) & (depth > 1e-3), depth, torch.full_like(depth, 1e3))
    t = torch.exp(-beta * d)[None, None]
    return img * t + airlight * (1 - t)


def rain(img, depth, fx, density, exposure_s, v_cam, down, fall_speed, wind, rng, zmax=12.0):
    """Rain as 3D drops in the view volume (not 2D lines): each drop is projected at the start and end of the exposure, so the streak
    length is fx * |relative velocity| * exposure / distance and its direction tilts with the camera velocity; drops behind the scene
    surface are hidden; near drops are wide and long, far drops thin and short. Far rain is added as haze by the caller.
    density 0..1 maps to the number of visible drops (1.0 = very heavy rain). Plausible model, not validated against real footage."""
    H, W = img.shape[-2:]
    n = int(density * 26000)
    z = zmax * rng.random(n) ** (1 / 3) + 0.25                               # uniform in the frustum volume (pdf ~ z^2)
    x = (rng.random(n) * 1.4 - 0.7) * W / fx * z; y = (rng.random(n) * 1.4 - 0.7) * H / fx * z
    vr = np.asarray(down, float) * fall_speed * rng.uniform(0.8, 1.15, (n, 1)) + np.asarray(wind, float) - np.asarray(v_cam, float)
    e = max(exposure_s, 1 / 2000)
    p0 = np.stack([x, y, z], 1); p1 = p0 + vr * e
    ok = (p1[:, 2] > 0.15)
    u0, v0 = fx * p0[:, 0] / p0[:, 2] + W / 2, fx * p0[:, 1] / p0[:, 2] + H / 2
    u1, v1 = fx * p1[:, 0] / np.maximum(p1[:, 2], 0.15) + W / 2, fx * p1[:, 1] / np.maximum(p1[:, 2], 0.15) + H / 2
    um, vm = np.clip(((u0 + u1) / 2).astype(int), 0, W - 1), np.clip(((v0 + v1) / 2).astype(int), 0, H - 1)
    dz = depth.cpu().numpy(); dz = np.where(np.isfinite(dz) & (dz > 1e-3), dz, 1e9)
    ok &= (z < dz[vm, um]) & (np.minimum(u0, u1) < W) & (np.maximum(u0, u1) > 0) & (np.minimum(v0, v1) < H) & (np.maximum(v0, v1) > 0)
    wpx = fx * 0.003 / z                                                     # 3 mm drop
    alpha = np.clip(0.55 * wpx, 0.10, 0.60)                                  # tuned by eye for visibility, not measured
    SS = 2                                                                   # supersample for thin lines
    m = Image.new("L", (W * SS, H * SS), 0); dr = ImageDraw.Draw(m)
    for i in np.where(ok)[0]:
        dr.line([u0[i] * SS, v0[i] * SS, u1[i] * SS, v1[i] * SS], fill=int(alpha[i] * 255), width=max(1, int(round(wpx[i] * SS))))
    a = torch.from_numpy(np.asarray(m.resize((W, H), Image.BOX)).astype(np.float32) / 255).to(img.device)[None, None]
    a = a.clamp(0, 0.75)
    return img * (1 - a) + 0.88 * a                                          # drops scatter light: translucent bright streaks


def sensor(img, ev, gain_ev, iso, gen):
    """Noise in linear light: shot noise (photon count shrinks with ISO) + read noise; gain after the noise raises it again."""
    lin = img.clamp(0, 1) ** 2.2 * 2.0 ** ev
    n0 = 4000.0 * (100.0 / iso)                                         # photons per unit linear signal
    ph = lin * n0
    lin = (ph + ph.clamp(min=0).sqrt() * torch.randn(lin.shape, device=lin.device, generator=gen)) / n0
    lin = lin + 0.002 * math.sqrt(iso / 100.0) * torch.randn(lin.shape, device=lin.device, generator=gen)
    return (lin * 2.0 ** gain_ev).clamp(0, 1) ** (1 / 2.2)


def jpeg(img, q):
    a = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8); b = io.BytesIO()
    Image.fromarray(a).save(b, format="JPEG", quality=q); b.seek(0)
    return torch.from_numpy(np.asarray(Image.open(b)).astype(np.float32) / 255).permute(2, 0, 1)[None].to(img.device)


def pipeline(img_hwc, depth, fx, motion, cfg, seed=0, device="cuda"):
    """img_hwc (H,W,3) float [0,1] numpy/torch; depth (H,W) metres; motion = {"v": (vx,vy,vz), "w": (wx,wy,wz), "down": gravity direction in the camera frame (default +y)}. Returns (H,W,3) float numpy."""
    cfg = validate_cfg(cfg)
    dev = device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed); gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.as_tensor(np.asarray(img_hwc), dtype=torch.float32, device=dev).permute(2, 0, 1)[None]
    z = torch.as_tensor(np.asarray(depth), dtype=torch.float32, device=dev)
    v, w = motion.get("v", (0, 0, 0)), motion.get("w", (0, 0, 0))
    if cfg["exposure_s"] > 0 and (any(v) or any(w)):
        flow = pixel_flow(z, fx, v, w)
        x = motion_blur(x, flow, cfg["exposure_s"])
        if cfg["readout_s"] > 0:
            x = rolling_shutter(x, flow, cfg["readout_s"])
    x = lens(x, cfg["k1"], cfg["k2"], cfg["vignette"])
    if cfg["fog_beta"] > 0:
        x = fog(x, z, cfg["fog_beta"], cfg["fog_airlight"])
    if cfg["rain_density"] > 0:
        x = fog(x, z, 0.03 * cfg["rain_density"], cfg["fog_airlight"])       # distant rain reads as haze
        x = rain(x, z, fx, cfg["rain_density"], cfg["exposure_s"], v, motion.get("down", (0, 1, 0)), cfg["rain_fall_speed"],
                 (cfg["wind_x"], 0, cfg["wind_z"]), rng)
    x = sensor(x, cfg["ev"], cfg["gain_ev"], cfg["iso"], gen)
    if cfg["jpeg_quality"]:
        x = jpeg(x, int(cfg["jpeg_quality"]))
    return x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def velocity_from_poses(c2w_a, c2w_b, dt):
    """Camera-frame v and w from two OpenCV-axes camera-to-world poses dt seconds apart (ground-truth trajectory -> effect inputs)."""
    a, b = np.asarray(c2w_a, float), np.asarray(c2w_b, float)
    v_world = (b[:3, 3] - a[:3, 3]) / dt
    r_rel = a[:3, :3].T @ b[:3, :3]                                      # rotation from frame a to frame b, in a's axes
    ang = math.acos(max(-1.0, min(1.0, (np.trace(r_rel) - 1) / 2)))
    axis = np.array([r_rel[2, 1] - r_rel[1, 2], r_rel[0, 2] - r_rel[2, 0], r_rel[1, 0] - r_rel[0, 1]])
    w = axis / (2 * math.sin(ang)) * ang / dt if ang > 1e-6 else np.zeros(3)
    return tuple(a[:3, :3].T @ v_world), tuple(w)


def _gauss_blur(x, sigma):
    k = int(2 * round(3 * sigma) + 1); t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2; w = torch.exp(-t ** 2 / (2 * sigma ** 2)); w = w / w.sum()
    C = x.shape[1]; x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


LAMP_COLOURS = {"sodium": (1.0, 0.62, 0.28), "led": (0.92, 0.96, 1.0), "halogen": (1.0, 0.85, 0.65)}


def night_real(img, inv_depth, gen, rng, strength=1.0):
    """Realistic night from a daytime frame (single image, (1,3,H,W) in [0,1]); the uniform-darkening `sensor` night is not what a real
    night camera sees. Model, in linear light:
      ambient moonlight / city glow (very dark, blue or orange cast; sky darker still) + artificial lights that relight the scene:
      vehicle headlights (cone from the bottom centre, falloff with depth from the inverse-depth map) and street lamps (sodium / LED pools
      with their own colour, a visible light source and glare), a few distant car / tail lights near the horizon, low-light desaturation,
      then camera auto-exposure (mean brightness target as in real night frames) and high-ISO shot + read noise.
    inv_depth: (H,W) relative inverse depth (bigger = nearer), e.g. the core's depth head. rng: numpy Generator; gen: torch Generator."""
    _, _, H, W = img.shape; dev = img.device; s = float(strength)
    lin = img.clamp(0, 1) ** 2.2
    inv = inv_depth.float(); inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-6)          # 0 far .. 1 near
    yy = torch.linspace(0, 1, H, device=dev)[:, None].expand(H, W); xx = torch.linspace(0, 1, W, device=dev)[None, :].expand(H, W)
    lum = img.mean(1)[0]; sky = (((inv < torch.quantile(inv, 0.2)) | (lum > 0.55)) * (yy < 0.6)).float() * (1 - ((yy - 0.45) / 0.15).clamp(0, 1)); sky = _gauss_blur(sky[None, None], 2.0)[0, 0]
    amb_col = torch.tensor([(0.55, 0.65, 1.0), (1.0, 0.75, 0.5)][int(rng.integers(2))], device=dev).view(3, 1, 1)
    illum = amb_col * 2.0 ** (-rng.uniform(8.5, 11.0) * s) * (1 - 0.995 * sky)                    # ambient, relative to daylight
    if rng.random() < 0.85:                                                                        # own headlights: cone, ~1/d^2 on the ground
        cx = 0.5 + rng.normal(0, 0.04); cone = torch.exp(-((xx - cx) / (0.18 + 0.5 * (1 - yy))) ** 2) * (yy > 0.35).float()
        illum = illum + torch.tensor(LAMP_COLOURS["halogen"], device=dev).view(3, 1, 1) * (rng.uniform(0.3, 1.0) * inv ** 2 * cone)
    srcs = torch.zeros(3, H, W, device=dev)
    for _ in range(int(rng.integers(0, 6))):                                                       # street lamps: pool of light + the lamp itself
        col = torch.tensor(LAMP_COLOURS["sodium" if rng.random() < 0.55 else "led"], device=dev).view(3, 1, 1)
        lx, ly = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.45); r = rng.uniform(0.08, 0.3)
        pool = torch.exp(-(((xx - lx) / r) ** 2 + ((yy - ly - 0.25) / (1.4 * r)) ** 2)); illum = illum + col * rng.uniform(0.05, 0.4) * pool * (1 - 0.98 * sky)
        srcs = srcs + col * 4.0 * torch.exp(-(((xx - lx) * W) ** 2 + ((yy - ly) * H) ** 2) / (2 * (rng.uniform(0.6, 1.6) * W / 320) ** 2))
    hz = 0.4 + rng.uniform(-0.08, 0.1)
    for _ in range(int(rng.integers(0, 8))):                                                       # distant head / tail lights near the horizon
        red = rng.random() < 0.5; col = torch.tensor((1.0, 0.1, 0.05) if red else (1.0, 0.95, 0.85), device=dev).view(3, 1, 1)
        srcs = srcs + col * 3.0 * torch.exp(-(((xx - rng.uniform(0.1, 0.9)) * W) ** 2 + ((yy - hz - rng.uniform(0, 0.12)) * H) ** 2) / (2 * (rng.uniform(0.5, 1.2) * W / 320) ** 2))
    lit = lin * illum[None] + srcs[None]
    lit = lit + _gauss_blur((lit - 0.6).clamp(min=0), max(1.5, 0.012 * W)) * rng.uniform(0.15, 0.5)  # glare / bloom around bright sources
    gry = lit.mean(1, keepdim=True); lit = gry + (lit - gry) * rng.uniform(0.4, 0.85)              # low-light desaturation
    target = rng.uniform(0.012, 0.05)                                                              # auto exposure: linear mean of real night frames
    lit = lit * (target / lit.mean().clamp(min=1e-6))
    iso = rng.uniform(400, 3200); n0 = 4000.0 * (100.0 / iso); ph = lit.clamp(min=0) * n0
    lit = (ph + ph.sqrt() * torch.randn(lit.shape, device=dev, generator=gen)) / n0 + 0.002 * math.sqrt(iso / 100.0) * torch.randn(lit.shape, device=dev, generator=gen)
    out = lit.clamp(0, 1) ** (1 / 2.2)
    return _gauss_blur(out, rng.uniform(0.5, 1.2) * W / 640).clamp(0, 1)                      # camera noise reduction (phones / dashcams denoise)
