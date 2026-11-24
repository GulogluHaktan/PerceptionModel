"""Recorded frames (run.py --record) -> contact sheet: camera | 11-class seg overlay | indoor 22-class overlay (if recorded) | relative depth.
Usage: python deploy/rec_sheet.py <rec_dir> [out.jpg] [max_rows=6]"""
import sys, glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
R = Path(sys.argv[1]); OUT = sys.argv[2] if len(sys.argv) > 2 else str(R / "sheet.jpg"); N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]
PAL = np.array([[70, 70, 70], [255, 0, 255], [0, 160, 0], [220, 20, 60], [128, 64, 128], [180, 120, 60], [255, 255, 0], [0, 0, 230], [120, 100, 80], [255, 255, 255], [255, 140, 0]], np.uint8)
IN = ["other", "wall", "floor", "ceiling", "door", "window", "person", "chair", "sofa", "table", "cabinet_shelf", "bed", "monitor_tv", "laptop", "keyboard", "mouse",
      "phone_remote", "cup_bottle", "book_paper", "plant", "bag", "appliance"]
PIN = np.array([(60, 60, 60), (170, 130, 90), (128, 64, 128), (200, 200, 230), (140, 80, 20), (90, 190, 230), (220, 20, 60), (255, 160, 0), (190, 90, 200), (120, 70, 30),
                (100, 140, 60), (240, 120, 160), (0, 80, 230), (0, 200, 200), (255, 255, 0), (255, 0, 255), (0, 255, 120), (0, 120, 255), (250, 250, 250), (0, 160, 0),
                (150, 40, 40), (110, 110, 160)], np.uint8)
fs = sorted(f for f in glob.glob(str(R / "*.jpg")) if "sheet" not in f)
OUTN = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
POUT = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                 (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)], np.uint8)
if not any(Path(f[:-4] + "_segin.png").exists() for f in fs) and any(Path(f[:-4] + "_segfine.png").exists() for f in fs):   # outdoor run: show the 19-class map
    for f in fs:
        if Path(f[:-4] + "_segfine.png").exists(): Path(f[:-4] + "_segin.png").write_bytes(Path(f[:-4] + "_segfine.png").read_bytes())
    IN, PIN = OUTN, POUT
fin = [f for f in fs if Path(f[:-4] + "_segin.png").exists()]
if fin: fs = fin                      # scene router: indoor maps exist only after the switch to the indoor head -> use those frames
fs = fs[:: max(1, len(fs) // N)][:N]; S = 320; rows = []
has_in = all(Path(f[:-4] + "_segin.png").exists() for f in fs)
for f in fs:
    im = Image.open(f).convert("RGB").resize((S, S)); seg = np.asarray(Image.open(f[:-4] + "_seg.png").resize((S, S), Image.NEAREST)); dep = Image.open(f[:-4] + "_depth.png").convert("L").resize((S, S))
    cols = [np.asarray(im), (0.45 * np.asarray(im) + 0.55 * PAL[np.minimum(seg, 10)]).astype(np.uint8)]
    if has_in: si = np.asarray(Image.open(f[:-4] + "_segin.png").resize((S, S), Image.NEAREST)); cols.append((0.35 * np.asarray(im) + 0.65 * PIN[np.minimum(si, 21)]).astype(np.uint8))
    dc = np.asarray(dep.convert("RGB")).copy(); dc[..., 0] = np.asarray(dep); dc[..., 2] = 255 - np.asarray(dep); cols.append(dc)
    rows.append(np.concatenate(cols, 1))
sheet = Image.fromarray(np.concatenate(rows, 0)); d = ImageDraw.Draw(sheet)
present = sorted({int(c) for f in fs for c in np.unique(np.asarray(Image.open(f[:-4] + "_seg.png")))})
for k, c in enumerate(present): d.rectangle([S + 6, 6 + 16 * k, S + 18, 18 + 16 * k], fill=tuple(int(v) for v in PAL[c])); d.text((S + 22, 6 + 16 * k), CL[c], fill=(255, 255, 255))
if has_in:
    pin = sorted({int(c) for f in fs for c in np.unique(np.asarray(Image.open(f[:-4] + "_segin.png")))})
    for k, c in enumerate(pin): d.rectangle([2 * S + 6, 6 + 16 * k, 2 * S + 18, 18 + 16 * k], fill=tuple(int(v) for v in PIN[c])); d.text((2 * S + 22, 6 + 16 * k), IN[c], fill=(255, 255, 255))
sheet.save(OUT, quality=88); print(OUT, len(rows), "rows; classes", [CL[c] for c in present], "indoor", [IN[c] for c in pin] if has_in else "-")
