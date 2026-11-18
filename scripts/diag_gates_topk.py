"""Diagnosis for the dense gate head on held-out sim racing frames: is the 'miss' a ranking artefact?
Labelled = the 4 largest gates (as in eval_gates_sim.py); predictions = top-K peaks by score (K = 4, 8, 16, 32) above THR, optionally re-ranked by predicted size (largest first, keep 4).
Also reports the number of labelled gates per frame. Usage: PFM_GATE_HEAD=dense [PFM_GDET_STRIDE=4] CKPT=runs/bigC python scripts/diag_gates_topk.py [set] [step]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel
from pfm.data import normalize
from pfm.train import slot_targets, gate_list
from pfm.evaluate import match_gates
S = sys.argv[1] if len(sys.argv) > 1 else "sim_racec"; step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; R = Path("data/heldout"); CK = os.environ["CKPT"]; THR = float(os.environ.get("THR", "0.5")); W, H = 256.0, 192.0
names = sorted(p.stem for p in (R / "images" / S).glob("*__clear.png"))[::step]
m = PerceptionModel("vit", teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/vit/final.pt", map_location="cpu")); m = m.cuda().eval()
def size_of(p): c = p[:, 1:].view(-1, 4, 2); return (c.max(1).values - c.min(1).values).max(1).values
KS = [(4, False), (8, False), (16, False), (32, False), (16, True), (32, True)]; acc = {k: {"full": [], "partial": []} for k in KS}; un = {k: 0 for k in KS}; ngl = []
with torch.no_grad():
    for i in range(0, len(names), 16):
        ch = names[i:i + 16]; x = torch.stack([torch.from_numpy(np.asarray(Image.open(R / "images" / S / f"{n}.png").convert("RGB").resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for n in ch])
        m.dense.n_slots = 4; raw = m(normalize(x.cuda()), with_teacher_heads=False)["gdet"]
        for key in KS:
            m.dense.n_slots = key[0]; pred = m.dense.decode_gdet(raw).cpu()
            for j, n in enumerate(ch):
                rows = json.load(open(R / "labels/corners_full" / f"{S}__{n}.json")); t = slot_targets(rows)
                if key == KS[0]: ngl.append(len(gate_list(rows)))
                if t[:, 0].sum() == 0: continue
                p = pred[j]
                if key[1]:
                    ok = p[:, 0].sigmoid() > THR; p = p[ok]; p = p[size_of(p).argsort(descending=True)[:4]] if len(p) else p
                ms, u = match_gates(p, t, THR); un[key] += u
                for k, e in ms:
                    c = t[k, 1:].view(4, 2) * torch.tensor([W, H]); part = bool(((c[:, 0] < 0) | (c[:, 0] >= W) | (c[:, 1] < 0) | (c[:, 1] >= H)).any()); acc[key]["partial" if part else "full"].append(1e9 if e is None else e)
ngl = np.array(ngl); print("labelled gates per frame: mean %.2f, share of frames with >4: %.2f, max %d" % (ngl.mean(), (ngl > 4).mean(), ngl.max()))
for key in KS:
    a = np.array(acc[key]["full"]); b = np.array(acc[key]["partial"])
    print(f"K={key[0]:2d} size-rank={key[1]!s:5}  full <10 {(a<10).mean():.3f} <20 {(a<20).mean():.3f} (n={len(a)})  partly <10 {(b<10).mean():.3f} <20 {(b<20).mean():.3f} (n={len(b)})  unmatched/frame {un[key]/len(names):.2f}")
