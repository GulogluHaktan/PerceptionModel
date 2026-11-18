"""Time the real label pipeline (Grounding DINO tiny + SAM2 tiny, src/pfm/labeler.py) on this machine.
Usage: python scripts/orin_label_timing.py <image_dir> [n=6]. Reports model load time, per-image seconds with the full class prompt set
(9 classes, about 20 Grounding DINO queries per image) and with a reduced 2-class set, plus peak GPU and system memory."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "src")
import pfm.labeler as L

files = sorted(Path(sys.argv[1]).glob("*.png"))[: int(sys.argv[2]) if len(sys.argv) > 2 else 6]
t = time.time(); lab = L.Labeler("cuda"); load_s = time.time() - t
full = dict(L.PROMPTS); small = {"car": ["bulldozer", "toy truck"], "ground_road": ["lego baseplate", "flat plate"]}
res = {"load_s": round(load_s, 1), "images": len(files)}
for name, pr in (("full_9_classes", full), ("reduced_2_classes", small)):
    L.PROMPTS = pr; times = []
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB").resize((640, 640))
        torch.cuda.synchronize(); t = time.perf_counter(); lab.label(img); torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    res[name] = {"first_s": round(times[0], 2), "median_s_rest": round(float(np.median(times[1:])), 2) if len(times) > 1 else None}
mem = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]
res["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20); res["ram_available_mb"] = int(mem) // 1024
print(json.dumps(res))
