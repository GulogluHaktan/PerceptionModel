"""Gate class of the SEGMENTATION head on real gate frames (not the corner head): no pixel masks exist, so two weak checks from the corner labels (pfm.gate_seg):
  false gate share = share of the 'surely not gate' pixels (away from every labelled gate, and inside the openings) that the model paints as gate;
  band recall      = share of the thin band just outside each fully labelled opening that the model paints as gate.
Sets: MonoRaceGate red (place never trained on) and the TII golden frames (hall seen in training, other flights). Usage: CKPT=runs/x KIND=vit python scripts/eval_gate_seg_real.py"""
import json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES
from pfm.data import normalize
from pfm.gate_seg import weak_gate_seg, GATE, NOT_GATE
CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); dev = os.environ.get("DEVICE", "cuda"); gi = CLASSES.index("gate"); res = {}
m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.to(dev).eval()
for name, G, flt in (("red (never seen)", "data/real_mrg/golden", lambda c: c.get("seq", "").startswith("red")), ("TII golden (hall seen)", "data/real_tii/golden", lambda c: True)):
    fp = tn = tp = pos = 0; frames = 0; per = []
    for c in [c for c in json.load(open(f"{G}/candidates.json")) if flt(c)]:
        x = torch.from_numpy(np.asarray(Image.open(f"{G}/images/{c['id']}.jpg").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): pred = m(normalize(x.to(dev)), with_teacher_heads=False)["seg"][0].argmax(0).cpu().numpy() == gi
        w = weak_gate_seg([np.array([[a, b] if v > 0 else [np.nan, np.nan] for a, b, v in g["corners"]], np.float32) for g in c["gates"]], 448, 448)
        ng = w == NOT_GATE; b = w == GATE; fp += (pred & ng).sum(); tn += ng.sum(); tp += (pred & b).sum(); pos += b.sum(); frames += 1; per.append((pred & ng).sum() / max(ng.sum(), 1))
    res[name] = {"frames": frames, "false_gate_pixel_share": round(float(fp / max(tn, 1)), 4), "frames_with_over_1pct_false_gate": round(float((np.array(per) > 0.01).mean()), 3), "gate_band_recall": round(float(tp / max(pos, 1)), 3)}
    print(CK, K, name, json.dumps(res[name]), flush=True)
json.dump(res, open(f"runs/eval/gate_seg_real_{os.path.basename(CK)}_{K}.json", "w"), indent=1)
