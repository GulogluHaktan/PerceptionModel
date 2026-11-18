"""How good is the prompt-based labeller (Grounding DINO + SAM2) against the simulator's exact labels? Runs on the Orin.
Usage: python scripts/orin_city_eval.py <set_dir> <ids.txt> <out_dir>   (set_dir has images/<id>.png and labels/<id>.png from scripts/city_effects.py)
Per image it stores the 11x12 confusion (rows = true class, cols = predicted class, last col = unlabelled by the labeller) in <out_dir>/conf.jsonl (resumable)
and the predicted mask; `summarise` prints per-class recall / precision / IoU overall and per condition. Pixels without a true label (sky) are ignored."""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "src")
from pfm.classes import CLASSES

NC = len(CLASSES)


def confusion(truth, pred):
    m = truth != 255; t = truth[m].astype(int); p = np.where(pred[m] == 255, NC, pred[m]).astype(int)
    return np.bincount(t * (NC + 1) + p, minlength=NC * (NC + 1)).reshape(NC, NC + 1)


def summarise(rows):
    res = {}
    groups = {"all": rows}; [groups.setdefault(r["cond"], []).append(r) for r in rows]
    for g, rs in groups.items():
        C = sum(np.array(r["conf"]) for r in rs); d = {}
        for i, c in enumerate(CLASSES):
            tp = C[i, i]; true = C[i].sum(); pred = C[:, i].sum()
            if true == 0 and pred == 0: continue
            d[c] = {"recall": round(float(tp / true), 3) if true else None, "precision": round(float(tp / pred), 3) if pred else None, "iou": round(float(tp / (true + pred - tp)), 3), "true_px": int(true)}
        res[g] = {"classes": d, "unlabelled_frac": round(float(C[:, NC].sum() / C.sum()), 3), "images": len(rs)}
    return res


if __name__ == "__main__":
    set_dir, ids_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True); (out / "pred").mkdir(exist_ok=True)
    ids = ids_file.read_text().split(); done = {json.loads(l)["id"] for l in open(out / "conf.jsonl")} if (out / "conf.jsonl").exists() else set()
    import pfm.labeler as L
    lab = L.Labeler("cuda"); t0 = time.time()
    for n, sid in enumerate(ids):
        if sid in done: continue
        img = Image.open(set_dir / "images" / f"{sid}.png").convert("RGB"); truth = np.asarray(Image.open(set_dir / "labels" / f"{sid}.png"))
        t = time.time(); seg, _ = lab.label(img); dt = time.time() - t
        Image.fromarray(seg).save(out / "pred" / f"{sid}.png")
        with open(out / "conf.jsonl", "a") as f:
            f.write(json.dumps({"id": sid, "cond": sid.split("__")[-1], "conf": confusion(truth, seg).tolist(), "sec": round(dt, 1)}) + "\n")
        print(n + 1, "/", len(ids), sid, round(dt, 1), "s", flush=True)
    rows = [json.loads(l) for l in open(out / "conf.jsonl")]; s = summarise(rows); json.dump(s, open(out / "summary.json", "w"), indent=1); print(json.dumps(s)); print("CITYEVALDONE")
