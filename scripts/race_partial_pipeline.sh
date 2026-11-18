#!/usr/bin/env bash
# Partly-visible-gate racing set, end to end (native Isaac): layouts p_race_0..N-1 (scripts/gen_race_partial.py) -> render -> effects (data/sim_racep) -> trainer layout (data/train_sim, set sim_racep) -> last two layouts to data/heldout.
# The close-pass set (sim_racec) is parked outside the training folder: its partly-visible share (2.7%) is no better than the normal set's. Then the second training (scripts/retrain_sim_v2.sh -> runs/sim_v2).
# Usage: N=14 bash scripts/race_partial_pipeline.sh   (idle GPU needed)
set -euo pipefail
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH; set -a; . ./.isaac_env; set +a; export ISAAC_NATIVE=1 PFM_ROOT=$PWD; N=${N:-14}; PY=.venv/bin/python
[ -f runs/race_job/layouts/p_race_0.json ] || $PY scripts/gen_race_partial.py 30
todo=$(for i in $(seq 0 $((N-1))); do n=p_race_$i; [ -f runs/race_job/out/$n/poses.json ] || echo $n; done)
[ -z "$todo" ] || bash sim/run_race_set.sh $todo
runs=$(for i in $(seq 0 $((N-1))); do echo runs/race_job/out/p_race_$i; done); $PY scripts/city_effects.py data/sim_racep $runs
$PY scripts/race_to_training.py data/train_sim data/sim_racep runs/race_job/out sim_racep
python3 - <<PY
import re, shutil
from pathlib import Path
R, H, s, N = Path("data/train_sim"), Path("data/heldout"), "sim_racep", int("$N"); pat = re.compile(rf"p_race_({N-2}|{N-1})__"); n = 0
for sub in ("images/" + s, "labels/seg", "labels/depth", "labels/corners", "labels/corners_full"):
    src = R / sub
    if not src.exists(): continue
    dst = H / sub; dst.mkdir(parents=True, exist_ok=True)
    for f in list(src.iterdir()):
        if pat.search(f.name): shutil.move(str(f), str(dst / f.name)); n += 1
print("moved to heldout:", n)
P = Path("data/parked/images"); P.mkdir(parents=True, exist_ok=True)
if (R / "images/sim_racec").exists(): shutil.move(str(R / "images/sim_racec"), str(P / "sim_racec")); print("sim_racec parked")
PY
echo "RACE_PARTIAL_DONE: train $(ls data/train_sim/images/sim_racep | wc -l)  heldout $(ls data/heldout/images/sim_racep | wc -l)"
bash scripts/retrain_sim_v2.sh
