"""On-device check of the TensorRT engines against the PyTorch reference outputs shipped in bundle/ref_frames.npz (made on the dev PC):
per-output error, same gate set, corner error in px; then end-to-end latency of the full runtime (camera-size frame -> gates) on this device.
Run: python deploy/check_device.py [bundle=deploy/bundle] [backend=trt] [cam_w=1280 cam_h=720]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from pfm_runtime import GatePerception
B = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bundle"; backend = sys.argv[2] if len(sys.argv) > 2 else "trt"
cw, ch = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1280, 720)
R = np.load(B / "ref_frames.npz"); gp = GatePerception(B, backend=backend); S = gp.S; LG = np.log(0.05 / 0.95)
err = dict(depth_rel=[], seg_agree=[], cpk_heat=[], cpk_off_at_peaks=[], pooled_rel=[]); gate_same = ng_same = 0; gate_px = []; ng_px = []
heads = [None] + list((gp.cfg.get("heads") or {}).keys()) if gp.cfg.get("heads") else [None]
for head in heads:
  for i in range(len(R["images"])):
    x = torch.from_numpy(R["images"][i:i + 1].astype(np.float32) / 255).to(gp.dev)
    o = gp.be.system(x, head) if gp.backend == "trt" else gp.be.system(x); gp._select(o); o = {k: v.cpu().numpy() for k, v in o.items() if torch.is_tensor(v)}      # _select: candidate-format bundles -> (1,K,9) like the reference
    d0 = R["depth"][i, 0].astype(np.float32); err["depth_rel"].append(float(np.abs(o["depth"][0, 0] - d0).mean() / (np.abs(d0).mean() + 1e-6)))
    err["seg_agree"].append(float((o["seg"][0].argmax(0) == (R["seg_outdoor"][i] if head == "seg2" else R["seg"][i])).mean()))   # seg2 head: v1+v2 reference
    if "seg_in" in R and "seg_in" in o and head != "m2f": err.setdefault("seg_in_agree", []).append(float((o["seg_in"][0] == R["seg_in"][i]).mean()))   # m2f: no torch reference in ref_frames
    err["cpk_heat"].append(float(np.abs(o["cpk_heat"] - R["cpk_heat"][i:i + 1]).max()))
    pk = R["cpk_heat"][i] > 0; err["cpk_off_at_peaks"].append(float(np.abs(o["cpk_off"][0][:, pk.any(0)] - R["cpk_off"][i][:, pk.any(0)]).max()) if pk.any() else 0.0)
    err["pooled_rel"].append(float(np.abs(o["pooled"] - R["pooled"][i]).max() / np.abs(R["pooled"][i]).max()))
    for key, same, px, thr in (("gates", "g", gate_px, LG), ("ng_gates", "n", ng_px, np.log(0.6 / 0.4))):    # the score each head is used at
        a, b = R[key][i], o[key][0]; A, Bk = a[a[:, 0] > thr, 1:], b[b[:, 0] > thr, 1:]          # order-free: near-tied candidates may swap rank
        if len(A) == len(Bk):
            if same == "g": gate_same += 1
            else: ng_same += 1
            for g in A: px.append(float(np.abs(Bk - g).max(1).min() * S))
res = dict(backend=backend, frames=len(R["images"]) * len(heads), heads=[str(h) for h in heads], **{k: round(float(np.max(v) if k not in ("seg_agree", "seg_in_agree") else np.min(v)), 5) for k, v in err.items()},
           seg_agree_mean=round(float(np.mean(err["seg_agree"])), 5), gates_same_set=gate_same, gates_max_px=round(max(gate_px), 3) if gate_px else 0, gates_median_px=round(float(np.median(gate_px)), 3) if gate_px else 0,
           ng_same_set=ng_same, ng_max_px=round(max(ng_px), 3) if ng_px else 0)
# end-to-end latency at camera size (synthetic frame with the first reference image upscaled: gates present)
fr = torch.nn.functional.interpolate(torch.from_numpy(R["images"][:1].astype(np.float32) / 255), size=(ch, cw), mode="bilinear")[0].permute(1, 2, 0).numpy()
fr = (fr * 255).astype(np.uint8); ms = []
for k in range(60):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    r = gp(fr); ms.append(r["ms"])
res["latency_ms_median"] = {k: round(float(np.median([m[k] for m in ms[10:]])), 2) for k in ms[0]}; res["camera"] = f"{cw}x{ch}"
print(json.dumps(res, indent=1)); (B / f"check_{backend}.json").write_text(json.dumps(res, indent=1))
