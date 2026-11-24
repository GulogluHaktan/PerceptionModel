"""Overfitting check for the corner peak head: same metric on TRAINING frames vs the held-out seed, for every snapshot.
A growing train-val gap (train recall up, val flat/down, false peaks up on val) = overfitting.
Run: env -u PYTHONPATH .venv/bin/python scripts/check_overfit_cpk.py runs/cornerpeak_v1
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import train_cornerpeak as T
from pfm.model import PerceptionModel
from pfm.cornerpeak import CornerPeakHead

D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/cornerpeak_v1")
TRAIN_SEEDS = [int(s) for s in os.environ.get("TRAIN_SEEDS", "5,18").split(",")]      # one clean + one augmented training seed
tr = T.load(set(TRAIN_SEEDS), 6); va = T.load({T.VAL_SEED}, 3)
print(f"train-subset frames {len(tr[0])} (seeds {TRAIN_SEEDS}), held-out frames {len(va[0])} (seed {T.VAL_SEED})", flush=True)
model = PerceptionModel("hybrid", teacher_dims={"dino": 384}); model.load_state_dict(torch.load(T.CKPT, map_location="cpu")); model = model.to(T.dev).eval()
snaps = sorted(D.glob("cornerpeak_*.pt"), key=lambda p: int(p.stem.split("_")[1])) + [D / "cornerpeak.pt"]
print("snapshot | recall<4px by gate size (train / held-out) | false peaks per frame (train / held-out)")
for s in snaps:
    head = CornerPeakHead().to(T.dev); head.load_state_dict(torch.load(s, map_location="cpu"))
    rt, ft = T.validate(model, head, *tr); rv, fv = T.validate(model, head, *va)
    cells = "  ".join(f"{k}: {rt[k][1]:.3f}/{rv[k][1]:.3f}" for k in rv if k in rt)
    print(f"{s.stem:16s} {cells}  | fp {ft:.2f}/{fv:.2f}", flush=True)
