"""Order-free gate check on held-out sim racing frames (works for the MLP slot head and the dense head): labelled gates (4 largest) matched one-to-one to predictions with score > THR.
Reports, split into gates fully in view and partly out of view: share found within 10 / 20 px mean corner error (256x192), median error of the matched ones, unmatched predictions per frame, false-alarm frames.
Usage: [PFM_GATE_HEAD=dense] CKPT=runs/x KIND=vit python scripts/eval_gates_sim.py <set=sim_racec> [step]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.train import slot_targets
from pfm.evaluate import match_gates
S = sys.argv[1] if len(sys.argv) > 1 else "sim_racec"; step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; R = Path("data/heldout"); CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); THR = float(os.environ.get("THR", "0.5")); W, H = 256.0, 192.0
names = sorted(p.stem for p in (R / "images" / S).glob("*__clear.png"))[::step]
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
acc = {"full": [], "partial": []}; un = 0; negf = 0; nega = 0
with torch.no_grad():
    for i in range(0, len(names), 16):
        ch = names[i:i + 16]; x = torch.stack([torch.from_numpy(np.asarray(Image.open(R / "images" / S / f"{n}.png").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for n in ch])
        pred = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()
        for j, n in enumerate(ch):
            t = slot_targets(json.load(open(R / "labels/corners_full" / f"{S}__{n}.json")))
            if t[:, 0].sum() == 0: negf += 1; nega += int((pred[j, :, 0].sigmoid() > THR).any()); continue
            ms, u = match_gates(pred[j], t, THR); un += u
            for k, e in ms:
                c = t[k, 1:].view(4, 2) * torch.tensor([W, H]); part = bool(((c[:, 0] < 0) | (c[:, 0] >= W) | (c[:, 1] < 0) | (c[:, 1] >= H)).any()); acc["partial" if part else "full"].append(1e9 if e is None else e)
def fin(a):
    a = np.array(a); ok = a[a < 1e8]
    return {"gates": len(a), "<10px": round(float((a < 10).mean()), 3), "<20px": round(float((a < 20).mean()), 3), "median_err_px": round(float(np.median(ok)), 1) if len(ok) else None}
res = {"set": S, "frames": len(names), "full_view": fin(acc["full"]), "partly_out": fin(acc["partial"]), "unmatched_pred_per_frame": round(un / len(names), 2), "frames_without_gate": negf, "false_alarm_frames": nega}
print(CK, K, json.dumps(res)); json.dump(res, open(f"runs/eval/{S}_gates_{os.path.basename(CK)}_{K}.json", "w"))
