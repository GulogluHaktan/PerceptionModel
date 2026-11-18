"""Multi-cycle self-evolution loop (the collapse-risk mechanism, decision 9 / gaps 4 and 7).

cycle k: train student on [real labelled + student-labelled pool from cycle k-1]
         -> evaluate on the frozen golden set -> gate vs. previous accepted model
         -> if accepted: model becomes the labeller for the pool; else STOP and keep the previous model (rollback)
The student labels the pool itself (no heavy teacher): pixels below `conf` are 255 (ignored).
Every cycle trains from the SAME init with the SAME steps (equal compute), so cycles are comparable.
Real labelled data is never dropped (minimum real share), the golden set is never used for training.
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .cache import FeatureCache
from .data import ImageFolder, normalize
from .gate import decide, log_provenance
from .model import PerceptionModel
from .train import gate_cfg_from, run


def load_student(cfg, ckpt, device):
    cache = FeatureCache(cfg["cache"])
    imgs = ImageFolder(cfg["images"], cfg["size"])
    dims = {}
    for t in cfg["teachers"]:
        if t["kind"].endswith("depth") or t["kind"] == "depth_da2":
            continue
        dims[t["name"]] = cache.get(t["name"], imgs[0][0])[0].shape[0]
    m = PerceptionModel(cfg["student"], teacher_dims=dims, hybrid_name=cfg.get("hybrid_name", "fastvit_sa24"))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def student_label(model, folder, out_dir, size, conf=0.8, device="cuda", bs=16, flip_consistency=False):
    """Write seg PNGs for every image in `folder` (255 where max softmax < conf). Returns mean labelled fraction."""
    ds = ImageFolder(folder, size)
    out = Path(out_dir) / "seg"
    out.mkdir(parents=True, exist_ok=True)
    fracs = []
    for ids, x in DataLoader(ds, batch_size=bs):
        xd = normalize(x.to(device)); prob = model(xd, with_teacher_heads=False)["seg"].float().softmax(1)
        agree = None
        if flip_consistency:   # test-time augmentation: the image and its mirror must give the same class, else the pixel stays unlabelled
            p2 = model(xd.flip(-1), with_teacher_heads=False)["seg"].float().softmax(1).flip(-1)
            agree = prob.argmax(1) == p2.argmax(1); prob = (prob + p2) / 2
        c, a = prob.max(1)
        bad = c < conf if agree is None else (c < conf) | ~agree
        a = a.masked_fill(bad, 255).byte().cpu().numpy()
        for sid, lab in zip(ids, a):
            Image.fromarray(lab).save(out / f"{sid}.png")
            fracs.append(float((lab != 255).mean()))
    return float(np.mean(fracs)) if fracs else 0.0


def pool_label_accuracy(student_dir, truth_dir, size):
    """Pixel accuracy of the student's pool labels (only pixels the student labelled) against reference labels, and the labelled fraction."""
    ok = tot = lab = n = 0
    for p in sorted((Path(student_dir) / "seg").glob("*.png")):
        t = Path(truth_dir) / "seg" / p.name
        if not t.exists():
            continue
        a = np.asarray(Image.open(p)); b = np.asarray(Image.open(t).resize((size, size), Image.NEAREST))
        if a.shape != b.shape:
            a = np.asarray(Image.open(p).resize((size, size), Image.NEAREST))
        m = (a != 255) & (b != 255); ok += int((a[m] == b[m]).sum()); tot += int(m.sum()); lab += int((a != 255).sum()); n += a.size
    return (round(ok / tot, 4) if tot else None), round(lab / max(1, n), 4)


def run_loop(cfg, real_sources, pool_sources, n_real, n_pool, cycles, real_labels, work="runs/selfloop", conf=0.8):
    """cfg: loaded config dict with `images` root, `golden`, `cache`. Returns list of per-cycle dicts."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["images"])
    from .data import EXT
    def files(srcs):
        return sorted(p.relative_to(root) for s in srcs for p in (root / s).rglob("*") if p.suffix.lower() in EXT)
    real, pool = files(real_sources)[:n_real], files(pool_sources)[:n_pool]
    assert len(real) == n_real and len(pool) == n_pool, "not enough images"
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    mix = work / "mix"
    if mix.exists():
        shutil.rmtree(mix)
    for rel in real + pool:
        dst = mix / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink((root / rel).resolve(), dst)

    rows, prev_metrics, prev_ckpt, best = [], None, None, {}
    gcfg = gate_cfg_from(cfg)
    for k in range(cycles):
        labels = work / f"labels_{k}"
        if labels.exists():
            shutil.rmtree(labels)
        for sub in ("seg", "corners"):
            (labels / sub).mkdir(parents=True)
        missing = []
        for rel in real:      # real labels are always kept; a missing one is an error, never skipped silently
            sid = str(rel.with_suffix("")).replace("/", "__")
            src = Path(real_labels) / "seg" / f"{sid}.png"
            if not src.exists():
                missing.append(sid)
                continue
            os.symlink(src.resolve(), labels / "seg" / f"{sid}.png")
            cs = Path(real_labels) / "corners" / f"{sid}.npy"
            if cs.exists():
                os.symlink(cs.resolve(), labels / "corners" / f"{sid}.npy")
        if missing:
            raise FileNotFoundError(f"{len(missing)} real images have no label in {real_labels}, e.g. {missing[:3]}")
        pool_frac = pool_acc = None
        if prev_ckpt is not None:  # student labels the pool with the last ACCEPTED model
            model = load_student(cfg, prev_ckpt, dev)
            tmp = work / f"pool_view_{k}"
            if tmp.exists():
                shutil.rmtree(tmp)
            for rel in pool:
                dst = tmp / rel; dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink((root / rel).resolve(), dst)
            pool_frac = student_label(model, tmp, work / f"pool_labels_{k}", cfg["size"], conf, dev, flip_consistency=bool(cfg.get("flip_consistency", False)))
            pool_acc, _ = pool_label_accuracy(work / f"pool_labels_{k}", real_labels, cfg["size"])        # vs the reference labels in real_labels (toy worlds only)
            for p in (work / f"pool_labels_{k}" / "seg").glob("*.png"):
                os.symlink(p.resolve(), labels / "seg" / p.name)
            del model
        c = dict(cfg, images=str(mix), labels=str(labels), out_dir=str(work / f"cycle_{k}"))
        metrics = run(c)
        ckpt = Path(c["out_dir"]) / cfg["student"] / "final.pt"
        verdict = decide(metrics, prev_metrics, gcfg, best or None)
        row = {"cycle": k, "n_real": n_real, "n_pool": n_pool, "pool_labelled_frac": pool_frac, "pool_label_acc_vs_reference": pool_acc,
               "accepted": verdict["recommend_promote"], **metrics}
        log_provenance(work / "provenance.jsonl", {"cycle": k, "metrics": metrics, "gate": verdict,
                                                  "parent": str(prev_ckpt), "pool_conf": conf})
        rows.append(row)
        print("CYCLE", row, "|", verdict["reasons"], flush=True)
        if not verdict["recommend_promote"]:
            print(f"gate rejected cycle {k}: rollback to {prev_ckpt}, stopping")
            break
        prev_metrics, prev_ckpt = metrics, ckpt
        for mk, mv in metrics.items():                                   # best-so-far anchor (max for scores, min for AbsRel)
            best[mk] = min(best.get(mk, mv), mv) if mk.endswith("absrel") else max(best.get(mk, mv), mv)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("cycle", "n_real", "n_pool"), k))
    with open(work / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows
