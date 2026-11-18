#!/bin/bash
# Renders the given indoor layouts: sim/run_indoor_set.sh name1 name2 ... (each runs/indoor_job/layouts/<name>.json is resolved, then rendered)
cd "${PFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; mkdir -p runs/indoor_job/out; chmod -R ugo+rwx runs/indoor_job 2>/dev/null
for n in "$@"; do
  python3 scripts/layout_to_job.py runs/indoor_job/layouts/$n.json $n || { echo "$n RESOLVE FAIL"; continue; }
  bash sim/run_isaac.sh sim/isaac_indoor.py > runs/indoor_$n.log 2>&1; if grep -q "^DONE" runs/indoor_$n.log; then echo "$n OK"; else echo "$n FAILED (see runs/indoor_$n.log)"; fi
done
echo ALLDONE
