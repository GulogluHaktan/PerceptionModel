"""Seed summary: reads runs/eval/*_<TAG>.json written by eval_students_sim.py / eval_slots_sim.py and prints mean and sd over the available seeds (tags s0, s1, s2 ...) for the key held-out metrics, per student.
Usage: python scripts/summarize_seeds.py > summary.md"""
import json, os, statistics as st
from pathlib import Path
E = Path("runs/eval")
def ld(name):
    p = E / name
    return json.load(open(p)) if p.exists() else None
def metrics(kind, tag):
    out = {}
    for s, label in (("sim_race", "race"), ("sim_racep", "racep"), ("sim_outdoor", "outdoor"), ("sim_indoor", "indoor")):
        d = ld(f"{s}_{tag}.json"); r = (d or {}).get(f"{kind}_clear")
        if r:
            out[f"{label} seg mIoU"] = r["seg_miou"]; out[f"{label} depth AbsRel"] = r["depth_absrel"]
            if label == "race": out["race gate IoU"] = r["per_class_iou"].get("gate")
    for s, label in (("sim_racep", "racep (high tilt)"), ("sim_race", "race (normal)"), ("sim_racec", "racec (close pass)")):
        d = ld(f"{s}_slots_{tag}.json"); r = (d or {}).get(kind)
        if r:
            out[f"{label} partly-out gates median err px"] = r["partly_out_of_view_gates"]["median_corner_err_px"]
            out[f"{label} full-view gates median err px"] = r["full_view_gates"]["median_corner_err_px"]
            out[f"{label} full-view recall"] = r["full_view_gates"]["recall@0.5"]
    return out
for kind in ("vit", "hybrid"):
    tags = ("s0", "s1v", "s2") if kind == "vit" else ("s0", "s1", "s2")                                    # seed 1 of vit lives under s1v: the hybrid run of seed 1 overwrote the files of tag s1
    per = {t: metrics(kind, t) for t in tags if metrics(kind, t)}
    if not per: continue
    print(f"## {kind}: seeds {', '.join(per)} (held-out sim frames, sim-to-sim)\n\n| metric | " + " | ".join(per) + " | mean | sd |\n|---|" + "---|" * (len(per) + 2))
    for k in sorted({k for m in per.values() for k in m}):
        v = [per[t].get(k) for t in per]; vv = [x for x in v if x is not None]
        print(f"| {k} | " + " | ".join("" if x is None else f"{x:.3f}" for x in v) + f" | {st.mean(vv):.3f} | {(st.stdev(vv) if len(vv) > 1 else float('nan')):.3f} |")
    print()
