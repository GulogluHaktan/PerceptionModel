"""Light CPU ISP for the IMX219 wide raw modes (preview / quick videos, no torch / GPU): undo the row-class shift 8*(y%4) px of the
full-width x2-binned readout, OpenCV demosaic (RGGB), half-size, grey-world white balance (EMA), gain to 18 % grey, gamma 2.2."""
import numpy as np, cv2


class CpuISP:
    def __init__(self, half=True): self.wb = np.ones(3, np.float32); self.g = None; self.half = half

    def __call__(self, fr):
        """fr: uint16 (H, W) raw (10-bit << 6) -> uint8 BGR image (half size by default)."""
        x = fr.copy()
        for c in (1, 2, 3):
            d = 8 * c; x[c::4, d:] = fr[c::4, :-d]; x[c::4, :d] = fr[c::4, d:d + 1]
        rgb = cv2.cvtColor(x, cv2.COLOR_BayerBG2RGB)            # RGGB sensor -> OpenCV's BG naming
        if self.half: rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
        rgb = np.clip((rgb.astype(np.float32) - 4096) / (65535 - 4096), 0, None); m = rgb.reshape(-1, 3).mean(0) + 1e-6
        self.wb = 0.8 * self.wb + 0.2 * np.clip(m[1] / m, 0.4, 3.0); rgb *= self.wb
        g = 0.18 / max(float(rgb.mean()), 1e-4); self.g = g if self.g is None else 0.8 * self.g + 0.2 * g; rgb *= self.g
        return cv2.cvtColor((np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
