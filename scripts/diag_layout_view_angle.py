"""Share of gate-frames per view ratio (|cos| of the angle between the camera-to-gate direction and the gate normal; about the width/height ratio of the gate in the picture) for layouts without rendering.
A gate counts when it is within `focus` m and inside the horizontal half field of view. Usage: python scripts/diag_layout_view_angle.py name1 name2 ... (layouts in runs/race_job/layouts)"""
import json, sys, math
import numpy as np
sys.path.insert(0, "src")
from pfm.layout_race import resolve
edges = [0.5, 0.7, 0.85]; cnt = np.zeros(4); 
for nm in sys.argv[1:]:
    lay = json.load(open(f"runs/race_job/layouts/{nm}.json")); sc = resolve(lay)["scene"]; foc = lay["flight"].get("focus_m", 12.0); half = math.radians(sc["hfov_deg"] / 2)
    for p in sc["camera"]["poses"]:
        c = np.array(p["pos"]); f = np.array(p["forward"]); f /= np.linalg.norm(f)
        for g in sc["gates"]:
            d = np.array(g["at"]) - c; dist = np.linalg.norm(d)
            if dist > foc or dist < 1.0: continue
            if math.acos(max(-1, min(1, float(d @ f) / dist))) > half: continue
            y = math.radians(g["yaw_deg"]); n = np.array([math.cos(y), math.sin(y), 0.0]); r = abs(float(d @ n) / dist)
            cnt[int(np.searchsorted(edges, r, side="right"))] += 1
print("gate-frames", int(cnt.sum()), "ratio <0.5 / 0.5-0.7 / 0.7-0.85 / >0.85:", [round(x, 3) for x in cnt / cnt.sum()])
