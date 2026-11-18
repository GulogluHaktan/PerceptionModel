"""Does exact factory depth as a training signal help the depth head? Trains the A arm (4 views, clear) with different depth loss mixes and evaluates on golden-style folders.
Usage: python scripts/run_depth_gt_ablation.py <base_cfg.yaml> <out.json> <seed> <images_dir> <name=golden_dir> ...   (variants are defined below)"""
import json, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, "src")
from pfm.data import GoldenSet
from pfm.evaluate import evaluate
from pfm.model import CLASSES, PerceptionModel
from pfm.train import run
base = yaml.safe_load(open(sys.argv[1])); out = sys.argv[2]; seed = int(sys.argv[3]); images = sys.argv[4]; golds = dict(a.split("=", 1) for a in sys.argv[5:])
variants = {"teacher_only": {"depth_weight": 1.0, "depth_gt_weight": 0.0}, "gt_only": {"depth_weight": 0.0, "depth_gt_weight": 1.0}, "teacher_0.3_plus_gt": {"depth_weight": 0.3, "depth_gt_weight": 1.0}}
n = sum(1 for _ in Path(images).rglob("*.png")); res = []
for name, v in variants.items():
    cfg = dict(base, images=images, seed=seed, epochs=max(1, round(420 / (n // base["batch"]))), out_dir=f"runs/dgt/s{seed}/{name}", golden=list(golds.values())[0], **v); run(cfg)
    m = PerceptionModel(cfg["student"], teacher_dims={"dino": 384}); m.load_state_dict(torch.load(Path(cfg["out_dir"]) / cfg["student"] / "final.pt", map_location="cpu")); m.cuda()
    row = {"variant": name, **v, "seed": seed}
    for gname, gdir in golds.items():
        r = evaluate(m, GoldenSet(gdir, cfg["size"]), len(CLASSES), "cuda"); row[gname] = {k: round(x, 4) for k, x in r.items()}
    res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
