---
name: project-real-golden-set-tii
description: Real-image golden candidates from TII Racing dataset; sim-trained students detect zero gates on them
metadata:
  node_type: memory
  type: project
  originSessionId: 8b3a1328-c85a-4425-b944-3667c45da259
  modified: 2026-11-14T21:43:41.461Z
---

TII Racing dataset (CC BY 4.0, autonomous flights downloaded to data/real_tii on laptop, 102 candidate frames in data/real_tii/golden, copy on 5090). User reviewed the overlays on 2026-11-15 and called them fine. Sim students (sim_v2 vit and hybrid) find 0 gates on them (max slot score <=0.08): sim-to-real transfer is currently zero here; indoor hall, fisheye, purple gates, camera often up.

**Why:** This is the first real measurement; all 0.98 IoU / 0.96 recall numbers are sim-to-sim only.

**How to apply:** Do not quote sim numbers as success rates. Next work: gate appearance and fisheye randomisation, real-fraction fine-tune split by flight (decision 9). See [[project-work-order-indoor-outdoor-racing]].

**Update 2026-11-15 (later):** training with real positives only taught a shortcut (real frame = gate: 94% false alarms on painted-out frames). Real gate-free crops fixed crop/Aeroscapes false alarms (0%) but recall stays 0.22-0.26 within 20 px; sim colour/fisheye augmentation gave no gain. Always evaluate false alarms alongside recall; TII has almost no gate-free frames.

**Update 2026-11-15 (end):** partial-corner labels (NaN masked) lifted same-track golden recall to 0.82, but cross-track (train ellipse/lemniscate, test trackRATM) is only 0.11 within 20 px: track memorisation. Never quote the 0.82 as generalisation. Next needs a second environment or DroneRacing camera frames.
