"""Offline per-frame corner accuracy on DroneRacing-renderer frames (data.pt of isaac/collect_percep.py), by gate size and number of visible corners.
Full frame, no crop/EKF: separates 'the head cannot regress it' from 'the loop pipeline loses it'. NOTE dr1 frames are TRAINING data for bigDR_h (seen domain, upper bound); for bigY_h they are unseen.
Usage: OURS_CKPT=<ckpt> python scripts/eval_dr_offline_by_size.py <data.pt> [n_frames=600] [stride=20]"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/DroneRacing_eval"))
import ours_detector as od
d = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False); N = int(sys.argv[2]) if len(sys.argv) > 2 else 600; S = int(sys.argv[3]) if len(sys.argv) > 3 else 20
dev = "cuda"; H, W = 616, 820; bins = [(0, 60), (60, 120), (120, 250), (250, 2000)]; res = {b: [] for b in bins}
for i in range(0, min(N * S, d["img"].shape[0]), S):
    img = d["img"][i].float().div(255)[None].to(dev); det = od.detect_once(img)[0].cpu()
    cor = d["corners"][i].float().numpy() * np.array([W, H]); vis = d["vis"][i].numpy(); sv = d["slot_valid"][i].numpy()
    for k in range(cor.shape[0]):
        if not sv[k] or vis[k].sum() < 4: continue          # fully visible gates only
        size = float(np.ptp(cor[k], 0).max()); b = next(b for b in bins if b[0] <= size < b[1])
        if len(det) == 0: res[b].append((np.nan, 1)); continue
        pr = det[:, 1:].view(-1, 4, 2).numpy(); e = np.min([np.linalg.norm(pr - np.roll(cor[k], r, 0)[None], axis=2).mean(1) for r in range(4)], 0); j = e.argmin()      # corner order of the dataset is not assumed: best cyclic shift
        res[b].append((float(e[j]) if e[j] < 40 else np.nan, 0 if e[j] < 40 else 1))
for b in bins:
    r = np.array(res[b]); 
    if len(r): print(f"size {b[0]}-{b[1]} px: n {len(r)} missed(>40px) {r[:,1].mean():.2f} median corner err {np.nanmedian(r[:,0]):.2f} px")
