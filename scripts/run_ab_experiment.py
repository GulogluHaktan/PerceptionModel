"""Equal-compute A/B on factory data: train each arm for the same number of optimiser steps, evaluate on several golden-style folders, several seeds.
Usage: python scripts/run_ab_experiment.py <base_cfg.yaml> <out.json> <seeds,comma> <armA_images_dir> <armB_images_dir> <name=golden_dir> [name=golden_dir ...]
Steps are matched by choosing epochs per arm from the image count (same batch size)."""
import json, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, "src")
from pfm.data import GoldenSet
from pfm.evaluate import evaluate
from pfm.model import CLASSES, PerceptionModel
from pfm.train import run

base = yaml.safe_load(open(sys.argv[1])); out = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]
arms = {"A": sys.argv[4], "B": sys.argv[5]}; golds = dict(a.split("=", 1) for a in sys.argv[6:]); dev = "cuda"
n_img = {k: sum(1 for _ in Path(v).rglob("*.png")) for k, v in arms.items()}; steps_target = 420
res = []
for sd in seeds:
    for arm, d in arms.items():
        steps_per_epoch = n_img[arm] // base["batch"]; epochs = max(1, round(steps_target / steps_per_epoch))
        cfg = dict(base, images=d, seed=sd, epochs=epochs, out_dir=f"runs/ab/{arm}_s{sd}", golden=list(golds.values())[0])
        m = run(cfg)
        model = PerceptionModel(cfg["student"], teacher_dims={"dino": 384}, hybrid_name=cfg.get("hybrid_name", "fastvit_sa24"))
        model.load_state_dict(torch.load(Path(cfg["out_dir"]) / cfg["student"] / "final.pt", map_location="cpu")); model.to(dev)
        row = {"arm": arm, "seed": sd, "images": n_img[arm], "epochs": epochs, "steps": steps_per_epoch * epochs}
        for gname, gdir in golds.items():
            r = evaluate(model, GoldenSet(gdir, cfg["size"]), len(CLASSES), dev); row[gname] = {k: round(v, 4) for k, v in r.items()}
        res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
