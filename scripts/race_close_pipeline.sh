#!/usr/bin/env bash
# Close-pass racing set, end to end (native Isaac): layouts c_race_* -> render -> effects (data/sim_racec) -> trainer layout (data/train_sim, set name sim_racec) -> layouts 26-29 moved to data/heldout.
# Needs an idle GPU. Usage: bash scripts/race_close_pipeline.sh   (resumable: already rendered layouts are skipped by run_race_set.sh only if you delete nothing: it re-renders names it is given, so the list is filtered here)
set -euo pipefail
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH; set -a; . ./.isaac_env; set +a; export ISAAC_NATIVE=1 PFM_ROOT=$PWD
PY=.venv/bin/python; [ -f runs/race_job/layouts/c_race_0.json ] || $PY scripts/gen_race_close.py 30
todo=$(for f in runs/race_job/layouts/c_race_*.json; do n=$(basename $f .json); [ -f runs/race_job/out/$n/poses.json ] || echo $n; done)
[ -z "$todo" ] || bash sim/run_race_set.sh $todo
runs=$(ls -d runs/race_job/out/c_race_*); $PY scripts/city_effects.py data/sim_racec $runs
$PY scripts/race_to_training.py data/train_sim data/sim_racec runs/race_job/out sim_racec
python3 - <<'PY'
import re, shutil
from pathlib import Path
R, H, s = Path("data/train_sim"), Path("data/heldout"), "sim_racec"; pat = re.compile(r"c_race_(26|27|28|29)__"); n = 0
for sub in ("images/" + s, "labels/seg", "labels/depth", "labels/corners", "labels/corners_full"):
    src = R / sub
    if not src.exists(): continue
    dst = H / sub; dst.mkdir(parents=True, exist_ok=True)
    for f in list(src.iterdir()):
        if pat.search(f.name) and f.name.startswith(("sim_racec", "c_race_")): shutil.move(str(f), str(dst / f.name)); n += 1
print("moved to heldout:", n)
PY
echo "RACE_CLOSE_DONE: train $(ls data/train_sim/images/sim_racec | wc -l)  heldout $(ls data/heldout/images/sim_racec 2>/dev/null | wc -l)"
