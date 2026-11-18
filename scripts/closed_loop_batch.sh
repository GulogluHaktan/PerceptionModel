#!/bin/bash
# Closed-loop gate passing over several held-out tracks (see scripts/closed_loop_race.py; SIMULATION result). Usage: CKPT=runs/x KIND=hybrid scripts/closed_loop_batch.sh <out.jsonl> track1 track2 ...
# Per track: oracle and map (no rendering; seeds 0-2, speeds $SPEEDS), then percept (seeds $SEEDS x speeds $SPEEDS) and percept_delay (seed 0, fastest speed) against one renderer.
cd "$(dirname "$0")/.."; OUT=$1; shift; mkdir -p "$(dirname "$OUT")"; PY="env -u PYTHONPATH .venv/bin/python"; SPEEDS=${SPEEDS:-"8 12"}; SEEDS=${SEEDS:-"0 1"}; export PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4
for t in "$@"; do
  $PY scripts/layout_to_job_race.py runs/race_job/layouts/$t.json $t || continue
  for sp in $SPEEDS; do for s in 0 1 2; do for v in oracle map; do SPEED=$sp $PY scripts/closed_loop_race.py $v $s >> "$OUT"; done; done; done
  tmux kill-session -t loop 2>/dev/null; rm -f ~/loop_render.log
  tmux new -d -s loop "cd $(pwd) && env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY ISAAC_PY=/home/shannon/.local/bin/isaac-py bash sim/run_isaac.sh sim/isaac_race_loop.py > ~/loop_render.log 2>&1"
  for i in $(seq 200); do grep -qE "^READY|Traceback" ~/loop_render.log 2>/dev/null && break; sleep 3; done
  grep -q "^READY" ~/loop_render.log || { echo "{\"track\": \"$t\", \"error\": \"renderer did not start\"}" >> "$OUT"; continue; }
  for sp in $SPEEDS; do for s in $SEEDS; do KEEP_RENDERER=1 SPEED=$sp $PY scripts/closed_loop_race.py percept $s 2>/dev/null | tail -1 >> "$OUT"; done; done
  KEEP_RENDERER=1 SPEED=${SPEEDS##* } $PY scripts/closed_loop_race.py percept_delay 0 2>/dev/null | tail -1 >> "$OUT"
  touch runs/race_job/loop/stop; sleep 8; tmux kill-session -t loop 2>/dev/null
done
echo BATCH_END >> "$OUT"
