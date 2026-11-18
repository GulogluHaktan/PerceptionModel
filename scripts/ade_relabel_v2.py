"""ADE20K label mapping v2 (fixes the name rules of ade_prepare.py; images are untouched, old labels stay where they are).
v1 bug found in Update 45: the car rule `^car` also matched carousel, caravan, cart, cardboard; `^bus` matched bus stop; snow / ski slope / pitch / court went to other_obstacle, etc.
v2 rules use the FIRST synonym of the ADE name: car = car, truck, bus, van, minivan, jeep, taxi, pickup, ambulance, police car (4-wheeled road vehicles; two-wheelers, trailers, caravans, trains -> other_obstacle,
as in the Aeroscapes and BDD mappings); ground_road adds snow, ski slope, pitch, court, ice rink, golf green, embankment, track, land, soil, parking, playground, plaza;
building adds window, door, roof, chimney, balcony, ceiling, shop, hut, facade; tree_pole adds forest, trunk; rock adds cliff, boulder, canyon, crevasse.
Writes data/general/train/labels_v2/seg/ade__<id>.png for the existing training frames and data/general/eval_ade2/{images -> symlink, seg} for the held-out frames."""
import io, os, re
import numpy as np, pyarrow.parquet as pq
from PIL import Image
from collections import Counter
RAW = "data/general/ade_raw"; TR = "data/general/train"; EV = "data/general/eval_ade"; EV2 = "data/general/eval_ade3" if os.environ.get("SKY") == "1" else "data/general/eval_ade2"; LV = "labels_v3" if os.environ.get("SKY") == "1" else "labels_v2"
if os.environ.get("TWO") == "1":    # label set v5 (use together with SKY=1): two-wheelers count as vehicles (class car); writes labels_v5 and eval_ade5
    EV2, LV = "data/general/eval_ade5", "labels_v5"
CL = ["background", "gate", "tree_pole", "person", "ground_road", "building", "opponent_drone", "car", "rock", "lane_marking", "other_obstacle"]; ix = {c: i for i, c in enumerate(CL)}
CAR = {"car", "truck", "bus", "van", "minivan", "jeep", "taxi", "pickup", "ambulance", "police car", "sports car", "race car", "suv", "minibus", "school bus", "lorry", "fire engine", "garbage truck", "tow truck", "trolleybus", "limousine", "convertible"}
RULES = [(r"\b(sky|water|sea|river|lake|mountain|plant|plants|flora|waterfall|pond|swimming|ice|marsh|fog|cloud|clouds|iceberg|whitewater|smoke)\b", 255), (r"\bperson\b|\bpeople\b", ix["person"]),
         (r"\b(tree|trees|palm|pole|poles|streetlight|street lamp|forest|trunk|lamppost|post)\b", ix["tree_pole"]),
         (r"\b(building|buildings|house|houses|skyscraper|tower|church|castle|wall|hovel|bridge|edifice|garage|shed|window|windowpane|door|roof|chimney|balcony|ceiling|shop|hut|facade|temple|mosque|cathedral|factory)\b", ix["building"]),
         (r"\b(road|sidewalk|grass|earth|ground|field|path|sand|pavement|dirt|hill|lawn|runway|floor|crosswalk|terrain|snow|ski slope|pitch|court|ice rink|artificial golf green|golf green|embankment|track|land|soil|parking|parking lot|playground|plaza|street|trail|beach|desert|square|badlands|savanna|valley|salt plain|grassland|slope|dune|dunes|plateau|tableland|plain|meadow)\b", ix["ground_road"]),
         (r"\b(rock|rocks|stone|stones|cliff|boulder|canyon|crevasse)\b", ix["rock"])]
if os.environ.get("TWO") == "1": CAR |= {"bicycle", "motorbike", "motorcycle", "minibike", "scooter", "moped", "tricycle", "bike"}
SKY = os.environ.get("SKY") == "1"      # label set v3: sky / cloud -> class 0 (background) instead of ignored; writes labels_v3 and eval_ade3
def name2cls(n):
    first = n.lower().split(",")[0].strip()
    if SKY and re.search(r"\b(sky|cloud|clouds)\b", first): return 0
    if first in CAR: return ix["car"]
    for pat, c in RULES:
        if re.fullmatch(".*" + pat + ".*", first) and not re.search(r"bus stop|car wash|water tower|recycling plant|plant pot", first): return c
    if re.search(r"water tower", first): return ix["building"]
    return ix["other_obstacle"]
def relabel(pf, want, out_dir, prefix, seen, pxc):
    n = 0
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g, columns=["segmentations", "filename", "scene", "objects"]).to_pylist():
            sid = os.path.splitext(r["filename"])[0].replace("ADE_", "")
            if sid not in want or not r["segmentations"]: continue
            seg = np.asarray(Image.open(io.BytesIO(r["segmentations"][0]["bytes"])).convert("RGB")); cls = seg[..., 0].astype(np.int32) // 10 * 256 + seg[..., 1].astype(np.int32)
            names = {o["name_ndx"]: o["name"] for o in r["objects"] if o["parts"]["part_level"] == 0}; out = np.full(cls.shape, 255, np.uint8)
            for k, cnt in zip(*np.unique(cls, return_counts=True)):
                if k == 0 or int(k) not in names: continue
                c = name2cls(names[int(k)]); out[cls == k] = c; seen[names[int(k)]] = c; pxc[names[int(k)]] += cnt / cls.size
            Image.fromarray(out).resize((640, 480), Image.NEAREST).save(f"{out_dir}/{prefix}{sid}.png"); n += 1; want.discard(sid)
    return n
os.makedirs(f"{TR}/{LV}/seg", exist_ok=True); os.makedirs(f"{EV2}/seg", exist_ok=True)
if not os.path.exists(f"{EV2}/images"): os.symlink(os.path.abspath(f"{EV}/images"), f"{EV2}/images")
seen = {}; pxc = Counter(); want = {f[:-4] for f in os.listdir(f"{TR}/images/ade")}; n = 0
for k in range(10):
    p = f"{RAW}/train-{k:05d}-of-00010.parquet"
    if not want or not os.path.exists(p): continue
    n += relabel(pq.ParquetFile(p), want, f"{TR}/{LV}/seg", "ade__", seen, pxc); print("train shard", k, n, "left", len(want), flush=True)
wv = {f[:-4] for f in os.listdir(f"{EV}/images")}; m = relabel(pq.ParquetFile(f"{RAW}/validation-00000-of-00001.parquet"), wv, f"{EV2}/seg", "", seen, pxc); print("train", n, "missing", len(want), "| eval", m, "missing", len(wv))
for c in ("car", "ground_road", "building", "tree_pole", "rock", "person"):
    print("==", c, [(k.split(",")[0][:24], round(v, 1)) for k, v in pxc.most_common() if seen[k] == ix[c]][:40])
print("== other_obstacle (top)", [(k.split(",")[0][:24], round(v, 1)) for k, v in pxc.most_common() if seen[k] == ix["other_obstacle"]][:70])
print("== ignore", [(k.split(",")[0][:24], round(v, 1)) for k, v in pxc.most_common() if seen[k] == 255][:30])
