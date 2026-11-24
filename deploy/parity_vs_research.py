"""Does the deployed runtime give the same gates as the flown research detector?
Reference: ~/DroneRacing_eval/ours_detector.py with the min14 environment (detect() then cpk_peaks(), as isaac/eval_vision.py calls them).
Candidate: deploy/pfm_runtime.GatePerception (backend torch / onnx / trt). Frames: sim gate images cropped to 384x384 (the loop's --crop 384
derotated input) + real golden gate frames at native size. Reports matched gates, corner differences (px) and peak counts.
Run: env -u PYTHONPATH .venv/bin/python deploy/parity_vs_research.py [backend=onnx] [n=60]"""
import glob, os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
MIN14 = dict(OURS_DEDUP_ALL="1", OURS_DEDUP_INSIDE="0", OURS_DEDUP_IOU="0.2", OURS_NG_MIN="0", OURS_NG=str(Path.home() / "DroneRacing_eval/runs_cmp/ng2.pt"), OURS_NG_THR="0.6",
             OURS_THR="0.05", OURS_CPK_CONFIRM="1", OURS_CPK_SNAP="0", OURS_MIN_SIZE="14", OURS_CREF=str(ROOT / "runs/refiner_v2/refiner.pt"), OURS_CPK=str(Path.home() / "DroneRacing_eval/runs_cmp/cpk3.pt"))
os.environ.update(MIN14)
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path.home() / "DroneRacing_eval")); sys.path.insert(0, str(HERE))
import ours_detector as od
from pfm_runtime import GatePerception

backend = sys.argv[1] if len(sys.argv) > 1 else "onnx"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60; dev = "cuda"
sim = sorted(glob.glob(str(ROOT / "data/train_sim/images/**/*.png"), recursive=True))[::997][:N // 2]
real = (sorted(glob.glob(str(ROOT / "data/real_mrg/golden/images/red_*.jpg"))) + sorted(glob.glob(str(ROOT / "data/real_tii/golden/images/*.jpg"))))[::11][:N - len(sim)]
gp = GatePerception(Path(os.environ.get("BUNDLE", HERE / "bundle")), backend=backend, device=dev)


def frame(p, crop):
    im = Image.open(p).convert("RGB")
    if crop:
        W, H = im.size; s = min(W, H, 384 * 2); im = im.crop(((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2)).resize((384, 384), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float().to(dev) / 255


stats = dict(frames=0, ref_gates=0, cand_gates=0, matched=0, same_count_frames=0, corner_err=[], ref_peaks=0, cand_peaks=0, ms=[])
for p, crop in [(p, True) for p in sim] + [(p, False) for p in real]:
    x = frame(p, crop)
    with torch.no_grad():
        rd = od.detect(x.clone()); rp = od.cpk_peaks(x.clone(), rd); rd = rd[0]
    r = gp(x.clone()); cd = r["gates"]; stats["ms"].append(r["ms"]["total"])
    stats["frames"] += 1; stats["ref_gates"] += len(rd); stats["cand_gates"] += len(cd); stats["same_count_frames"] += int(len(rd) == len(cd))
    stats["ref_peaks"] += len(rp[0]); stats["cand_peaks"] += len(r["peaks"])
    if len(rd) and len(cd):
        a = rd[:, 1:].view(-1, 4, 2); b = cd[:, 1:].view(-1, 4, 2).to(a.device)
        D = (a[:, None] - b[None]).norm(dim=-1).mean(-1)          # same corner order expected
        Dr = torch.stack([(a[:, None] - torch.roll(b, r, 1)[None]).norm(dim=-1).mean(-1) for r in range(4)]).amin(0)     # order-free (diagnostic)
        stats.setdefault("order_free_close", 0); stats["order_free_close"] += int((Dr.amin(1) < 5).sum())
        stats.setdefault("big_diff", []); stats["big_diff"] += [round(float(D[i].min()), 1) for i in range(len(a)) if D[i].min() >= 5]
        for i in range(len(a)):
            j = int(D[i].argmin())
            if D[i, j] < 5: stats["matched"] += 1; stats["corner_err"].append(float((a[i] - b[j]).norm(dim=-1).max()))
res = {k: v for k, v in stats.items() if k not in ("corner_err", "ms")}
import collections
res.update(backend=backend, max_corner_diff_px=round(max(stats["corner_err"]), 4) if stats["corner_err"] else None,
           median_corner_diff_px=round(float(np.median(stats["corner_err"])), 5) if stats["corner_err"] else None, median_ms=round(float(np.median(stats["ms"][3:])), 2))
print(json.dumps(res)); (Path(os.environ.get("BUNDLE", HERE / "bundle")) / f"parity_{backend}.json").write_text(json.dumps(res, indent=1))
