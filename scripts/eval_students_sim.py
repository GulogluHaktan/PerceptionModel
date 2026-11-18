"""Zero-shot check of the v0 students on exactly-labelled simulation frames (data/train_sim/<set>): they were trained on public photos with pseudo labels and never saw these frames.
Proper ignore handling (255 = unlabelled is skipped, unlike pfm.evaluate), per-class IoU, pixel accuracy, depth AbsRel (scale-shift aligned, metric sim depth), plus a majority-class baseline for context.
Usage: [EVAL_ROOT=data/heldout CKPT=runs/v0 TAG=v0] python scripts/eval_students_sim.py <set e.g. sim_indoor> [step] ; reads $CKPT/{vit,hybrid}/final.pt ; writes runs/eval/<set>_<TAG>.json"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from pfm.model import PerceptionModel, CLASSES      # noqa: E402
from pfm.data import normalize                      # noqa: E402
from pfm.evaluate import align_ss                   # noqa: E402
S = sys.argv[1]; step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; SIZE = 448; R = Path(os.environ.get("EVAL_ROOT", "data/train_sim")); CK = os.environ.get("CKPT", "runs/v0"); TAG = os.environ.get("TAG", "v0"); NC = len(CLASSES); dev = "cuda"
def ids(cond): return sorted(p.stem for p in (R / "images" / S).glob(f"*__{cond}.png"))[::step]
def batch(names):
    xs, segs, deps = [], [], []
    for n in names:
        xs.append(torch.from_numpy(np.asarray(Image.open(R / "images" / S / f"{n}.png").convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255)
        segs.append(torch.from_numpy(np.asarray(Image.open(R / "labels/seg" / f"{S}__{n}.png").resize((SIZE, SIZE), Image.NEAREST)).copy()).long())
        d = np.load(R / "labels/depth" / f"{S}__{n}.npy").astype(np.float32); deps.append(torch.nn.functional.interpolate(torch.from_numpy(d)[None, None], size=(SIZE, SIZE), mode="nearest")[0, 0])
    return torch.stack(xs), torch.stack(segs), torch.stack(deps)
def run(kind, cond):
    m = PerceptionModel(kind, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{kind}/final.pt", map_location="cpu")); m.to(dev).eval()
    conf = torch.zeros(NC, NC, dtype=torch.long); rels = []; rinv = []; names = ids(cond)
    with torch.no_grad():
        for i in range(0, len(names), 16):
            x, seg, dep = batch(names[i:i + 16]); dep = dep[:, None]; out = m(normalize(x.to(dev)), with_teacher_heads=False); pred = out["seg"].argmax(1).cpu(); v = seg != 255
            conf += torch.bincount(seg[v] * NC + pred[v], minlength=NC * NC).view(NC, NC)
            mask = dep > 1e-3; a = align_ss(out["depth"].float().cpu(), dep, mask)
            rels += [((a[j] - dep[j]).abs() / dep[j].clamp(min=1e-3))[mask[j]].mean().item() for j in range(len(dep)) if mask[j].sum() > 100]
            from pfm.evaluate import absrel_both
            rinv += [absrel_both(out["depth"][j, 0].float().cpu().numpy(), dep[j, 0].numpy(), mask[j, 0].numpy())[1] for j in range(len(dep)) if mask[j].sum() > 100]
    inter = conf.diag().float(); gt = conf.sum(1).float(); pr = conf.sum(0).float(); union = gt + pr - inter; pres = gt > 0
    return {"frames": len(names), "seg_miou": round((inter[pres] / union[pres]).mean().item(), 4), "pixel_acc": round((inter.sum() / conf.sum()).item(), 4), "depth_absrel": round(float(np.mean(rels)), 4), "depth_absrel_invfit": round(float(np.mean(rinv)), 4),
            "per_class_iou": {CLASSES[c]: round((inter[c] / union[c]).item(), 3) for c in range(NC) if pres[c]}, "gt_share": {CLASSES[c]: round((gt[c] / gt.sum()).item(), 3) for c in range(NC) if pres[c]}}, conf
res = {}
for cond in ("clear", "lowlight"):
    for kind in os.environ.get("KINDS", "vit,hybrid").split(","):
        r, conf = run(kind, cond); res[f"{kind}_{cond}"] = r; print(kind, cond, json.dumps({k: v for k, v in r.items() if k not in ("gt_share",)}), flush=True)
    g = conf.sum(1).float(); res[f"majority_baseline_{cond}"] = {"class": CLASSES[int(g.argmax())], "seg_miou": round((g[g.argmax()] / (g.sum() + 0 + 1e-9) / (len([c for c in g if c > 0]))).item(), 4), "pixel_acc": round((g.max() / g.sum()).item(), 4)}
Path("runs/eval").mkdir(parents=True, exist_ok=True); json.dump(res, open(f"runs/eval/{S}_{TAG}.json", "w"), indent=1); print("saved runs/eval/%s_%s.json" % (S, TAG))
