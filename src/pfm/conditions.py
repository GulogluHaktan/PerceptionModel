"""Fixed-strength hard conditions used by the self-improvement loop for evaluation (same as scripts/eval_cond_router.py) and by the
DroneRacing closed-loop condition test. img (B,3,H,W) in 0..1. `core` is needed only for fog (depth from the core's inverse depth)."""
import torch
from pfm import effects as E

CONDS = ["clean", "fog", "night", "nir", "blur"]


def apply_condition(x, name, gen, core=None):
    if name == "clean": return x
    out = []
    for b in range(x.shape[0]):
        xb = x[b:b + 1]
        if name == "fog":
            from pfm.data import normalize
            inv = core(normalize(torch.nn.functional.interpolate(xb, size=(448, 448), mode="bilinear", align_corners=False)), with_teacher_heads=False)["depth"][:, :1].float()
            inv = torch.nn.functional.interpolate(inv, size=xb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            d = 1.0 / inv.clamp(min=1e-3); d = d * (8.0 / d.median()); xb = E.fog(xb, d, beta=0.12, airlight=0.8)
        elif name == "night": xb = E.sensor(xb, ev=-4.0, gain_ev=3.0, iso=3200, gen=gen)
        elif name == "nir":
            g = (0.1 * xb[:, 0:1] + 0.3 * xb[:, 1:2] + 0.6 * xb[:, 0:1]).clamp(0, 1); xb = E.lens(g.repeat(1, 3, 1, 1), 0.0, 0.0, 0.6)
        elif name == "blur":
            Hh, Ww = xb.shape[-2:]; flow = torch.zeros(Hh, Ww, 2, device=xb.device); flow[..., 0] = 1500.0 * Ww / 448; flow[..., 1] = 400.0 * Hh / 448
            xb = E.motion_blur(xb, flow, exposure_s=0.012)
        elif name == "night_real":   # realistic night (pfm.effects.night_real); needs the core for inverse depth
            from pfm.data import normalize
            import numpy as np
            inv = core(normalize(torch.nn.functional.interpolate(xb, size=(448, 448), mode="bilinear", align_corners=False)), with_teacher_heads=False)["depth"][:, :1].float()
            inv = torch.nn.functional.interpolate(inv, size=xb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            xb = E.night_real(xb, inv, gen, np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=gen.device).item())))
        elif name == "night_blur":   # harsher combined condition (not a router class: tests routing to the nearest module)
            Hh, Ww = xb.shape[-2:]; flow = torch.zeros(Hh, Ww, 2, device=xb.device); flow[..., 0] = 1500.0 * Ww / 448; flow[..., 1] = 400.0 * Hh / 448
            xb = E.sensor(E.motion_blur(xb, flow, exposure_s=0.012), ev=-4.0, gain_ev=3.0, iso=3200, gen=gen)
        else: raise ValueError(name)
        out.append(xb.clamp(0, 1))
    return torch.cat(out)
