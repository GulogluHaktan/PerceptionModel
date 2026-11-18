"""Scenario -> the files the Isaac foreground pass needs: camera.json, poses.json (OpenGL camera-to-world, metres), objects.json (boxes).
Usage: python scripts/export_scenario_for_isaac.py <scenario.json> <scene.pt> <out_dir>   (runs on any machine: numpy/torch CPU only)"""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "src")
from pfm.camera import Camera, to_opengl
from pfm.ground import fit_ground_plane
from pfm.objects import build_objects
from pfm.scenario import validate_scenario
from pfm.trajectory import make_trajectory

sc = validate_scenario(json.load(open(sys.argv[1]))); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
means = torch.load(sys.argv[2], map_location="cpu")["means"].float().numpy(); mpu = float(sc["scene"]["meters_per_unit"])
plane = fit_ground_plane(means, up=sc["scene"]["up"]); boxes, gates = build_objects(sc, plane, mpu)
cam = Camera.from_hfov_deg(sc["camera"]["width"], sc["camera"]["height"], sc["camera"]["hfov_deg"]); poses, fps = make_trajectory(sc["trajectory"])
json.dump({"width": cam.width, "height": cam.height, "fx": cam.fx}, open(out / "camera.json", "w"))
json.dump([{"transform_matrix": to_opengl(p).tolist()} for p in poses], open(out / "poses.json", "w"))
json.dump([{"center": np.asarray(b["center"]).tolist(), "size": np.asarray(b["size"]).tolist(), "yaw": b["yaw"], "color": list(b["color"]), "label": b["label"]} for b in boxes], open(out / "objects.json", "w"))
print("exported", len(poses), "poses,", len(boxes), "boxes, ground z at origin", round(float(plane[1]), 3), "->", out)
