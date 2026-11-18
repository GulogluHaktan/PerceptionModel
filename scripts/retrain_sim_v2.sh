#!/usr/bin/env bash
# Second simulation training: same recipe as sim_v1 (configs/sim_5090.yaml) plus the close-pass racing set (data/train_sim/images/sim_racec), output runs/sim_v2. Teacher cache is resumable: only the new images are computed.
# Usage: bash scripts/retrain_sim_v2.sh   (idle GPU needed; vit first, then hybrid)
set -euo pipefail
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH; PY=.venv/bin/python; CFG=configs/sim_v2_5090.yaml
$PY scripts/cache_teachers.py $CFG
for ST in vit hybrid; do sed "s/^student: .*/student: $ST/" $CFG > /tmp/simv2_$ST.yaml; $PY scripts/train.py /tmp/simv2_$ST.yaml; done
echo "RETRAIN_V2_DONE"
