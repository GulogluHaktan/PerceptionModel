---
name: project-general-real-data-mix
description: Real data sources now in the training mix (TII, MonoRaceGate, DIODE, SUN RGB-D, ADE20K) and honest transfer numbers
metadata:
  type: project
---

Real sources (all downloadable without login): TII Racing (GitHub releases), MonoRaceGate (github.com/tudelft/MonoRaceGate, MIT, 4 places, A2RL gates), DIODE depth (HF ahd03/diode-depth-3k), SUN RGB-D (HF wyrx/SUNRGBD_seg, label = id+1), ADE20K outdoor (HF 1aurent/ADE20K, class = R//10*256+G). Roboflow Universe gate sets need an account/API key (403 to automation). Scripts: scripts/*_prepare.py, chains mrg_chain.py / big_chain.py; results in the vault plan note Updates 35-41.

Consolidated recipe bigA (sim 20k + all real, MonoRaceGate place "red" held out): red gates 0.48 within 20 px / 0.15 within 10 px (the honest transfer number), SUN seg mIoU 0.74, ADE outdoor 0.57, DIODE depth AbsRel 0.11/0.21. Gate test seed noise is about +-0.06.

**Why:** user asked to mix sim with real data from other domains (indoor/outdoor, not only racing) and wants honest numbers.

**How to apply:** Keep one place held out when quoting gate transfer; never quote TII golden (same hall) as transfer. See [[project-real-golden-set-tii]].
