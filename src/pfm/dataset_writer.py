"""Writes generated frames with provenance (plan step 7): rgb png, depth (16-bit mm), label png, pose, plus a manifest with the scenario and its hash."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .scenario import scenario_hash, validate_scenario


class DatasetWriter:
    def __init__(self, out, scenario: dict, extra: dict | None = None, workers: int = 4):
        self.out = Path(out); self.scenario = validate_scenario(scenario); self.hash = scenario_hash(scenario)
        for d in ("rgb", "depth", "label"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.frames = []; self.extra = extra or {}
        self._pool = ThreadPoolExecutor(max_workers=workers); self._jobs = []       # PNG encoding releases the GIL: threads give a real speed-up

    def add(self, k, rgb, depth_m, label, c2w_opengl, motion=None, extra=None):
        a_rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8); a_d = np.clip(np.nan_to_num(depth_m) * 1000, 0, 65535).astype(np.uint16); a_l = label.astype(np.uint8)
        for arr, d in ((a_rgb, "rgb"), (a_d, "depth"), (a_l, "label")):                 # arrays are copies: safe to encode later in a worker thread
            self._jobs.append(self._pool.submit(lambda arr=arr, d=d: Image.fromarray(arr).save(self.out / d / f"{k:04d}.png")))
        self.frames.append({"file_path": f"rgb/{k:04d}", "transform_matrix": np.asarray(c2w_opengl).tolist(), "motion": motion, "extra": extra})

    def close(self):
        for j in self._jobs:
            j.result()                                                                  # re-raises any encoding error
        self._pool.shutdown()
        json.dump({"scenario_hash": self.hash, "scenario": self.scenario, "n_frames": len(self.frames), **self.extra, "frames": self.frames},
                  open(self.out / "manifest.json", "w"))
        return self.hash
