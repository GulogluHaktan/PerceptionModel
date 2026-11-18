"""Per-asset label coverage measured on REAL outdoor renders: every item is projected into each frame (pose + depth for occlusion) and the 9x9 window at its mid-height is checked for label != 255.
Usage: python scripts/outdoor_label_coverage.py <out.json> <run_name>...   (needs runs/outdoor_job/layouts/<name>.json and out/<name>/). Reports, per asset, the share of in-view samples that carry a label."""
import json, sys, math, importlib.util, types
import numpy as np
from PIL import Image
from collections import defaultdict
sys.path.insert(0, "src"); pkg = types.ModuleType("pfm"); pkg.__path__ = ["src/pfm"]; sys.modules["pfm"] = pkg
for n in ("assets", "layout", "layout_out"):
    sp = importlib.util.spec_from_file_location(f"pfm.{n}", f"src/pfm/{n}.py"); m = importlib.util.module_from_spec(sp); sys.modules[f"pfm.{n}"] = m; sp.loader.exec_module(m)
resolve = sys.modules["pfm.layout_out"].resolve; acc = defaultdict(lambda: [0, 0])
for run in sys.argv[2:]:
    d = f"runs/outdoor_job/out/{run}/"; scene = resolve(json.load(open(f"runs/outdoor_job/layouts/{run}.json")))["scene"]; meta = json.load(open(d + "poses.json")); fx = meta["fx"]; W, H = meta["width"], meta["height"]
    for k in range(0, len(meta["frames"]), 4):
        m = np.array(meta["frames"][k]["transform_matrix"]); R = np.stack([m[:3, 0], -m[:3, 1], -m[:3, 2]], 1); t = m[:3, 3]; lab = np.array(Image.open(f"{d}label/{k:04d}.png")); dep = np.load(f"{d}depth/{k:04d}.npy")
        for it in scene["items"]:
            z0 = (it["height_m"] or it["footprint"][2]) * 0.6; p = R.T @ (np.array([it["pos"][0], it["pos"][1], z0]) - t)
            if p[2] < 2.0 or p[2] > 120: continue
            u, v = int(W / 2 + fx * p[0] / p[2]), int(H / 2 + fx * p[1] / p[2])
            if not (6 <= u < W - 6 and 6 <= v < H - 6): continue
            tol = 0.7 * max(it["footprint"][:2]) + 1.0; w = dep[v - 4:v + 5, u - 4:u + 5]; l = lab[v - 4:v + 5, u - 4:u + 5]; near = np.abs(w - p[2]) < tol
            if near.sum() < 10: continue                                                              # occluded or empty at this pixel
            acc[it["asset"]][0] += int((l[near] != 255).sum()); acc[it["asset"]][1] += int(near.sum())
res = {a: {"samples_px": n, "labelled_frac": round(c / n, 3)} for a, (c, n) in acc.items() if n >= 200}
json.dump(res, open(sys.argv[1], "w"), indent=1)
bad = sorted(((v["labelled_frac"], a) for a, v in res.items() if v["labelled_frac"] < 0.6)); print(len(res), "assets measured;", len(bad), "below 0.6 labelled:"); [print(" ", a, f) for f, a in bad]
