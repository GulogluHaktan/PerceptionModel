"""Gate accuracy of the DEPLOYED runtime (GatePerception, e.g. TensorRT on the Orin) on the real golden gate sets: per labelled gate (>= 2
visible corners) the best predicted gate under the best cyclic corner order, mean visible-corner error at 256x192 scale (as
scripts/eval_real_golden.py); recall within 10 / 20 px and false gates per frame (predictions matched to no gate within 20 px).
Run (Orin): python eval_gates_deployed.py <bundle> <golden dir with list.json> [backend=trt]
list.json: [{"set": ..., "image": <file in golden dir>, "gates": [[[x, y, v] x4], ...]} ] (normalised corners)."""
import json, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfm_runtime import GatePerception
B, G = Path(sys.argv[1]), Path(sys.argv[2]); backend = sys.argv[3] if len(sys.argv) > 3 else "trt"; W, H = 256.0, 192.0
gp = GatePerception(B, backend=backend, device="cuda"); items = json.loads((G / "list.json").read_text()); res = {}
for it in items:
    im = Image.open(G / it["image"]).convert("RGB"); iw, ih = im.size
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float().cuda() / 255
    r = gp(x); g = r["gates"].cpu().numpy(); preds = [g[k, 1:9].reshape(4, 2) / [iw, ih] * [W, H] for k in range(len(g))]
    st = res.setdefault(it["set"], dict(frames=0, gt=0, e=[], fp=0)); st["frames"] += 1; used = set()
    for gate in it["gates"]:
        gc = np.array([[a * W, b * H] for a, b, v in gate]); vis = np.array([v > 0 for _, _, v in gate])
        if vis.sum() < 2: continue
        st["gt"] += 1; best = (1e9, None)
        for i, p in enumerate(preds):
            e = min(np.linalg.norm((np.roll(p, s, 0) - gc)[vis], axis=1).mean() for s in range(4))
            if e < best[0]: best = (e, i)
        st["e"].append(best[0])
        if best[1] is not None and best[0] < 20: used.add(best[1])
    st["fp"] += len(preds) - len(used)
for k, st in res.items():
    e = np.array(st["e"]); print(k, json.dumps({"frames": st["frames"], "gates": st["gt"], "recall<10px": round(float((e < 10).mean()), 3), "recall<20px": round(float((e < 20).mean()), 3), "fp/frame": round(st["fp"] / st["frames"], 2)}), flush=True)
