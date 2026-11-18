"""Figures for the README (architecture block diagram, plots/sonuc_1..3.png). All numbers come from docs/experiment-log.md (Updates 35-43); this script only draws them.
Usage: python scripts/make_readme_figures.py   (matplotlib only)"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.makedirs("plots", exist_ok=True); os.makedirs("assets", exist_ok=True)
C = {"sim": "#8da0cb", "real": "#66c2a5", "teach": "#fc8d62", "stud": "#e5c494", "head": "#a6d854", "ink": "#222222"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---- architecture
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(-0.6, 4.6)
def box(x, y, w, h, t, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=col, ec=C["ink"], lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9.5)
def arrow(x0, y0, x1, y1): ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.2))
box(0.1, 3.0, 2.2, 1.1, "Simulation (Isaac Sim)\nlayouts, gates, light,\nblur, fog, rain\nexact labels + depth", C["sim"])
box(0.1, 1.5, 2.2, 1.1, "Real data\nTII, MonoRaceGate,\nDIODE, SUN RGB-D,\nADE20K, Aeroscapes", C["real"])
box(0.1, 0.1, 2.2, 1.0, "Teachers (frozen)\nDINOv2-S,\nDepth Anything V2-S\n(features cached)", C["teach"])
box(3.2, 1.5, 2.4, 1.6, "Student backbone\nViT-B/16 or FastViT\n(448 px) + pyramid", C["stud"])
box(6.6, 3.45, 3.0, 0.8, "Depth head", C["head"]); box(6.6, 2.55, 3.0, 0.8, "Segmentation head (11 classes)", C["head"])
box(6.6, 1.65, 3.0, 0.8, "Corner heatmaps (4)", C["head"]); box(6.6, 0.45, 3.0, 1.1, "Dense gate head\ncentre + 8 corner offsets\n-> 4 slots x [score, 4 corners]", C["head"])
box(9.9, 1.2, 1.0, 1.9, "to\nDroneRacing\n(4 x [score,\n4 corners])", "#ffffff")
for y in (3.85, 2.95, 2.05, 1.0): arrow(5.6, 2.3, 6.55, y)
arrow(9.6, 1.0, 9.85, 1.8); arrow(2.35, 3.5, 3.15, 2.8); arrow(2.35, 2.05, 3.15, 2.3); arrow(1.2, 1.15, 3.15, 1.7)
ax.text(5.5, -0.35, "losses: feature distillation + depth (relative and exact) + segmentation + corner heatmap + dense gate focal/L1", ha="center", fontsize=9, style="italic")
fig.savefig("assets/architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- result 1: real gates on a place never seen (milestones; docs/experiment-log.md Updates 35-43, 82, 93)
names = ["sim only", "TII real\nframes only", "+ MonoRaceGate\n(2 seeds)", "bigB\nMLP slot head", "bigC\ndense head", "release v3\n(FastViT)", "release v4\n(FastViT)"]
p10 = [0.00, 0.026, 0.10, 0.103, 0.846, 0.942, 0.958]; p20 = [0.00, 0.420, 0.51, 0.484, 0.923, 0.990, 0.987]
fig, ax = plt.subplots(figsize=(10, 4.7)); x = range(len(names)); w = 0.38
b1 = ax.bar([i - w / 2 for i in x], p20, w, color=C["real"], label="corner error < 20 px"); b2 = ax.bar([i + w / 2 for i in x], p10, w, color=C["sim"], label="corner error < 10 px")
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=8.5)
ax.axvline(4.5, color="#999999", lw=0.8, ls="--"); ax.text(4.55, 1.04, "later evaluation protocol of Updates 82/93", fontsize=8, color="#555555")
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 1.1); ax.set_ylabel("share of labelled gates found"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Real gates on a place never seen in training (MonoRaceGate 'red', 312 gates)\nsingle seed per bar, seed noise about +-0.06 (v4 vs v3: inside the noise)", fontsize=10.5)
fig.savefig("plots/sonuc_1.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- result 2: simulated stress test, gate recall (bigA vs bigB milestone)
conds = ["clear", "blur 30 m/s", "+1 stop\n(bright)", "-4 stops\n(dark)", "fog 0.05", "rain", "worst mix\n30 m/s+dark+fog"]
a = [0.96, 0.96, 0.57, 0.92, 0.85, 0.90, 0.68]; b = [0.95, 0.95, 0.94, 0.93, 0.92, 0.94, 0.85]
fig, ax = plt.subplots(figsize=(9.5, 4.4)); x = range(len(conds)); w = 0.38
ax.bar([i - w / 2 for i in x], a, w, color=C["sim"], label="bigA (no degradation frames in training)"); ax.bar([i + w / 2 for i in x], b, w, color=C["real"], label="bigB (+ light and combined-degradation frames)")
for i, (u, v) in enumerate(zip(a, b)): ax.text(i - w / 2, u + 0.012, f"{u:.2f}", ha="center", fontsize=8.5); ax.text(i + w / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(list(x)); ax.set_xticklabels(conds, fontsize=9); ax.set_ylim(0, 1.12); ax.set_ylabel("gate recall (held-out simulated frames)"); ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.set_title("Simulated stress test (milestone bigA vs bigB): degradations on 257 held-out race frames", fontsize=10.5)
fig.savefig("plots/sonuc_2.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- result 3: general real held-out sets (segmentation: sim-only vs release v4; depth: sim-only vs milestone bigB)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
n1 = ["SUN RGB-D\n(indoor)", "ADE20K\n(outdoor)"]; s0 = [0.209, 0.153]; s1 = [0.749, 0.595]; x = range(2); w = 0.38
a1.bar([i - w / 2 for i in x], s0, w, color=C["sim"], label="sim-only model"); a1.bar([i + w / 2 for i in x], s1, w, color=C["real"], label="release v4 (real general data)")
for i in x: a1.text(i - w / 2, s0[i] + 0.012, f"{s0[i]:.2f}", ha="center", fontsize=9); a1.text(i + w / 2, s1[i] + 0.012, f"{s1[i]:.2f}", ha="center", fontsize=9)
a1.set_xticks(list(x)); a1.set_xticklabels(n1); a1.set_ylim(0, 0.95); a1.set_ylabel("mIoU (higher is better)"); a1.legend(frameon=False, fontsize=8.5, loc="upper left")
n2 = ["DIODE depth\nindoor", "DIODE depth\noutdoor"]; d0 = [0.292, 0.509]; d1 = [0.113, 0.218]
a2.bar([i - w / 2 for i in x], d0, w, color=C["sim"], label="sim-only model"); a2.bar([i + w / 2 for i in x], d1, w, color=C["real"], label="milestone bigB")
for i in x: a2.text(i - w / 2, d0[i] + 0.01, f"{d0[i]:.2f}", ha="center", fontsize=9); a2.text(i + w / 2, d1[i] + 0.01, f"{d1[i]:.2f}", ha="center", fontsize=9)
a2.set_xticks(list(x)); a2.set_xticklabels(n2); a2.set_ylim(0, 0.62); a2.set_ylabel("AbsRel (lower is better)"); a2.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Held-out real indoor and outdoor frames (SUN RGB-D split by frame, so probably optimistic)", fontsize=10.5)
fig.savefig("plots/sonuc_3.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- result 4: DroneRacing GateNet g17 vs release v4, same never-seen place, per frame (Update 99)
fig, ax = plt.subplots(figsize=(8.2, 4.4)); labs = ["GateNet g17\n(its default score 0.3)", "GateNet g17\n(score 0.5)", "release v4"]
v20 = [0.776, 0.558, 0.987]; v10 = [0.756, None, 0.958]; w = 0.38; x = range(3)
r1 = ax.bar([i - w / 2 for i in x], v20, w, color=C["real"], label="corner error < 20 px"); r2 = ax.bar([i + w / 2 for i, v in zip(x, v10) if v is not None], [v for v in v10 if v is not None], w, color=C["sim"], label="corner error < 10 px")
for bar in list(r1) + list(r2): ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labs, fontsize=9); ax.set_ylim(0, 1.12); ax.set_ylabel("share of labelled gates found"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Offline per-frame detector comparison on the never-seen place (312 gates)\nNOT a flight result: GateNet normally runs with a tracking crop and EKF fusion", fontsize=10)
fig.savefig("plots/sonuc_4.png", dpi=150, bbox_inches="tight"); plt.close(fig); print("figures written")
