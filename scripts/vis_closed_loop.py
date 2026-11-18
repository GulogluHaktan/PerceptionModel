"""Picture of one closed-loop track (scripts/closed_loop_race.py with TRAJ / SNAP): top view of the flown paths (true gate positions, rough map only, map + model), the summary of all runs, and camera frames
of the model flight with the predicted gates. SIMULATION. Usage: python scripts/vis_closed_loop.py <track> -> runs/eval/vis/closed_loop_<track>.jpg"""
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
T = sys.argv[1]; D = "runs/closed_loop/traj"; R = {v: json.load(open(f"{D}/{T}_{v}.json")) for v in ("oracle", "map", "percept")}
COL = {"oracle": "#8a8f98", "map": "#d9822b", "percept": "#2f6fde"}; NAME = {"oracle": "Gerçek kapı konumları (üst sınır)", "map": "Yalnızca kaba harita (±1 m)", "percept": "Kaba harita + model (bigL_h)"}
fig = plt.figure(figsize=(15, 10.5), dpi=110, facecolor="white"); gs = fig.add_gridspec(3, 4, height_ratios=[2.6, 1, 1], hspace=0.3, wspace=0.12)
ax = fig.add_subplot(gs[0, :3])
for g in R["oracle"]["gates"]:
    c = np.array(g["c"]); a = np.array(g["a"]); r = np.array([a[1], -a[0]])
    for half, lw, col in ((R["oracle"]["outer"] / 2, 5, "#c9b8ee"), (R["oracle"]["inner"] / 2, 5, "white")): ax.plot([c[0] - r[0] * half, c[0] + r[0] * half], [c[1] - r[1] * half, c[1] + r[1] * half], color=col, lw=lw, solid_capstyle="butt", zorder=2)
    ax.plot([c[0] - r[0] * R["oracle"]["outer"] / 2, c[0] + r[0] * R["oracle"]["outer"] / 2], [c[1] - r[1] * R["oracle"]["outer"] / 2, c[1] + r[1] * R["oracle"]["outer"] / 2], color="#6b46c1", lw=1, zorder=3)
for v in ("oracle", "map", "percept"):
    t = np.array(R[v]["traj"]); ax.plot(t[:, 0], t[:, 1], color=COL[v], lw=2.2 if v != "oracle" else 1.6, ls="--" if v == "oracle" else "-", zorder=4 if v != "percept" else 5, label=f"{NAME[v]}: {R[v]['per_gate'].count('passed')}/{len(R[v]['per_gate'])} kapı")
    for g, res in zip(R[v]["gates"], R[v]["per_gate"]):
        if res != "passed" and v != "oracle": c = np.array(g["c"]); k = int(np.argmin(np.linalg.norm(t[:, :2] - c[:2], axis=1))); ax.plot(t[k, 0], t[k, 1], "x", color=COL[v], ms=11, mew=3, zorder=6)
ax.set_aspect("equal"); ax.set_title(f"Kapalı döngü simülasyon, parkur {T}, 12 m/s, kuşbakışı\nmor çizgi: kapı, ×: çarpma ya da kaçırma; alttaki karelerde yeşil: modelin bulduğu kapılar", fontsize=11, loc="left")
ax.legend(loc="best", frameon=False, fontsize=10); ax.set_xlabel("m"); ax.set_ylabel("m"); ax.grid(color="#e6e8eb", lw=0.6); [s.set_visible(False) for s in ax.spines.values()]
# summary of all runs (Update 68 / 69)
ax2 = fig.add_subplot(gs[0, 3]); rows = [("Temiz, 8 m/s", 40 / 132, 87 / 88, 1.0), ("Temiz, 12 m/s", 42 / 132, 87 / 88, 1.0), ("Harita hatası 2 m,\n12 m/s", 10 / 132, 87 / 88, 1.0), ("Karanlık + bulanık,\n12 m/s", 42 / 132, 88 / 88, 1.0), ("Temiz, 20 m/s", 39 / 132, 47 / 88, 66 / 132), ("Temiz, 30 m/s", 26 / 132, 23 / 88, 45 / 132)]
y = np.arange(len(rows))[::-1] * 1.0
for i, (k, v) in enumerate((("map", 1), ("percept", 2), ("oracle", 3))): ax2.barh(y + (1 - i) * 0.26, [r[v] for r in rows], height=0.24, color=COL[k])
for yy, r in zip(y, rows):
    for i, v in enumerate(r[1:]): ax2.text(v + 0.02, yy + (1 - i) * 0.26, f"{v * 100:.0f}%", va="center", fontsize=8, color="#333")
ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=9); ax2.yaxis.tick_right(); ax2.set_xlim(0, 1.18); ax2.set_xticks([]); ax2.set_title("Geçilen kapı oranı, 6 parkur\nturuncu: harita, mavi: + model,\ngri: üst sınır", fontsize=10, loc="left"); [s.set_visible(False) for s in ax2.spines.values()]
fs = sorted(glob.glob(f"{D}/snap_{T}/*.jpg")); pick = [fs[int(i)] for i in np.linspace(0, len(fs) - 1, 8)]
for i, f in enumerate(pick):
    a = fig.add_subplot(gs[1 + i // 4, i % 4]); a.imshow(Image.open(f)); a.axis("off"); a.set_title(f"kamera karesi {f.split('/f')[-1][:4]}", fontsize=9, loc="left")
fig.text(0.01, 0.005, "Simülasyon sonucudur, gerçek başarı oranı değildir: sahneler eğitim verisiyle aynı çiziciden, drone nokta kütle ve kendi konumunu tam biliyor. 20 m/s ve üstünde sınırlayan basit kontrolcüdür.", fontsize=9, color="#555")
out = f"runs/eval/vis/closed_loop_{T}.jpg"; fig.savefig(out, bbox_inches="tight", pil_kwargs={"quality": 90}); print("wrote", out)
