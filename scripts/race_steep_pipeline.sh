#!/usr/bin/env bash
# Partly-visible-gate racing set, end to end (native Isaac): layouts s_race_0..N-1 (scripts/gen_race_partial.py) -> render -> effects (data/sim_races) -> trainer layout (data/train_sim, set sim_races) -> last two layouts to data/heldout.
# Usage: N=14 bash scripts/race_partial_pipeline.sh   (idle GPU needed)
set -euo pipefail
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH; set -a; . ./.isaac_env; set +a; export ISAAC_NATIVE=1 PFM_ROOT=$PWD; N=${N:-14}; PY=.venv/bin/python
[ -f runs/race_job/layouts/s_race_0.json ] || $PY scripts/gen_race_steep.py 30
todo=$(for i in $(seq 0 $((N-1))); do n=s_race_$i; [ -f runs/race_job/out/$n/poses.json ] || echo $n; done)
[ -z "$todo" ] || bash sim/run_race_set.sh $todo
runs=$(for i in $(seq 0 $((N-1))); do echo runs/race_job/out/s_race_$i; done); $PY scripts/city_effects.py data/sim_races $runs
$PY scripts/race_to_training.py data/train_sim data/sim_races runs/race_job/out sim_races
python3 - <<PY
import re, shutil
from pathlib import Path
R, H, s, N = Path("data/train_sim"), Path("data/heldout"), "sim_races", int("$N"); pat = re.compile(rf"s_race_({N-2}|{N-1})__"); n = 0
for sub in ("images/" + s, "labels/seg", "labels/depth", "labels/corners", "labels/corners_full"):
    src = R / sub
    if not src.exists(): continue
    dst = H / sub; dst.mkdir(parents=True, exist_ok=True)
    for f in list(src.iterdir()):
        if pat.search(f.name): shutil.move(str(f), str(dst / f.name)); n += 1
print("moved to heldout:", n)
PY
echo "RACE_STEEP_DONE: train $(ls data/train_sim/images/sim_races | wc -l)  heldout $(ls data/heldout/images/sim_races | wc -l)"
