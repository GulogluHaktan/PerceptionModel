---
name: project-work-order-indoor-outdoor-racing
description: "User's stated work order for the sim/perception work - indoor, then outdoor, then drone-racing gates up to 30 m/s, tested by swapping our student into the DroneRacing perception"
metadata:
  node_type: memory
  type: project
  originSessionId: d1e9c1b7-129a-4c19-bc1f-1a4f4d01a000
  modified: 2026-11-11T07:14:01.298Z
---

Order given by the user on 2026-11-11: (1) finish the indoor scene system, (2) outdoor, (3) drone-racing track: fly through gates; when the gate is so close that it is only partly visible the perception must not lose its position; must perceive correctly at speeds up to 30 m/s. The goal is to replace the perception net of the simulator in `/home/haktan/Projects/DroneRacing` (folder `droneracing/`, GateNet + PnP + EKF, camera 192x256, 130 deg, output 4 detections x [score, 4 corners]) with our student net and test it there. The student net does not exist yet, so this is preparation only. Gate information and the gate model already exist in DroneRacing (do not modify its frozen `releases/v1_camera_ekf_t10/`, `ekf.py`, `gatenet.py`, `estimator.py`, `camera.py`; new variants go in new files; its handover.md has the hard rules).

**Why:** the user wants the indoor and outdoor features reused for the racing track, and a real downstream test (flight success) instead of only label metrics.

**How to apply:** when choosing the next task, follow this order; keep our gate head's output compatible with `droneracing.estimator.det_features`; include close/partly visible gates (`droneracing/gatenet2.py` already has corner targets for centre-outside gates) and speeds to 30 m/s in sim data (motion blur from `src/pfm/effects.py`). The LLM only picks asset ids and coordinates (declarative layout, `src/pfm/layout.py`), it never writes code; see [[project-llm-declarative-layouts]].
