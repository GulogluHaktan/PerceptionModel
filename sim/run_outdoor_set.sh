#!/bin/bash
# Renders the given outdoor layouts: sim/run_indoor_set.sh name1 name2 ... (each runs/outdoor_job/layouts/<name>.json is resolved, then rendered)
cd "${PFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; mkdir -p runs/outdoor_job/out; chmod -R ugo+rwx runs/outdoor_job 2>/dev/null
for n in "$@"; do
  python3 scripts/layout_to_job_out.py runs/outdoor_job/layouts/$n.json $n || { echo "$n RESOLVE FAIL"; continue; }
  bash sim/run_isaac.sh sim/isaac_outdoor.py > runs/outdoor_$n.log 2>&1; if grep -q "^DONE" runs/outdoor_$n.log; then echo "$n OK"; else echo "$n FAILED (see runs/outdoor_$n.log)"; fi
done
echo ALLDONE
