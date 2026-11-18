"""Slot head check on racing frames: how well are the gate corners predicted, split into frames where every gate corner is inside the image and frames where a gate is partly OUT of view (the case this head was built for).
Matches slots by order (both sides sort the biggest gate first). Reports detection recall at score > 0.5, false alarms on frames without gates, and the mean corner error in pixels at 256x192 (the racing camera size).
Usage: [EVAL_ROOT=data/heldout CKPT=runs/sim_v1 TAG=sim_v1] python scripts/eval_slots_sim.py <set=sim_race> [step]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel      # noqa: E402
from pfm.data import normalize             # noqa: E402
from pfm.train import slot_targets         # noqa: E402
S = sys.argv[1] if len(sys.argv) > 1 else "sim_race"; step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; SIZE = 448; W, H = 256.0, 192.0
R = Path(os.environ.get("EVAL_ROOT", "data/heldout")); CK = os.environ.get("CKPT", "runs/sim_v1"); TAG = os.environ.get("TAG", "sim_v1"); dev = "cuda"
names = sorted(p.stem for p in (R / "images" / S).glob("*__clear.png"))[::step]
res = {}
for kind in os.environ.get("KINDS", "vit,hybrid").split(","):
    m = PerceptionModel(kind, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{kind}/final.pt", map_location="cpu")); m.to(dev).eval()
    acc = {"full": {"gt": 0, "hit": 0, "err": []}, "partial": {"gt": 0, "hit": 0, "err": []}}; neg_frames = 0; neg_alarm = 0
    with torch.no_grad():
        for i in range(0, len(names), 16):
            chunk = names[i:i + 16]
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(R / "images" / S / f"{n}.png").convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for n in chunk])
            pred = m(normalize(x.to(dev)), with_teacher_heads=False)["slots"].float().cpu()
            for j, n in enumerate(chunk):
                rows = json.load(open(R / "labels/corners_full" / f"{S}__{n}.json")); t = slot_targets(rows); sc = pred[j, :, 0].sigmoid()
                if t[:, 0].sum() == 0:
                    neg_frames += 1; neg_alarm += int((sc > 0.5).any()); continue
                for k in range(4):
                    if t[k, 0] < 0.5: continue
                    c = t[k, 1:].view(4, 2) * torch.tensor([W, H]); partial = bool(((c[:, 0] < 0) | (c[:, 0] >= W) | (c[:, 1] < 0) | (c[:, 1] >= H)).any()); b = acc["partial" if partial else "full"]; b["gt"] += 1
                    if sc[k] > 0.5:
                        b["hit"] += 1; b["err"].append((pred[j, k, 1:].view(4, 2) * torch.tensor([W, H]) - c).norm(dim=1).mean().item())
    def fin(b): return {"gt_gates": b["gt"], "recall@0.5": round(b["hit"] / max(b["gt"], 1), 3), "mean_corner_err_px": round(float(np.mean(b["err"])), 2) if b["err"] else None, "median_corner_err_px": round(float(np.median(b["err"])), 2) if b["err"] else None}
    res[kind] = {"frames": len(names), "full_view_gates": fin(acc["full"]), "partly_out_of_view_gates": fin(acc["partial"]), "frames_without_gate": neg_frames, "false_alarm_frames": neg_alarm}
    print(kind, json.dumps(res[kind]), flush=True)
Path("runs/eval").mkdir(parents=True, exist_ok=True); json.dump(res, open(f"runs/eval/{S}_slots_{TAG}.json", "w"), indent=1); print("saved")
