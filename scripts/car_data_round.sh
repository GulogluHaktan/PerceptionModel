#!/bin/bash
# One round of own-camera self-training after a recording session (Update 167).
#   bash scripts/car_data_round.sh araba1 araba2 ...      (recording names under /home/haktan/car/ on the Orin)
# 1 Orin: every 6th raw frame of each recording -> .npy   2 copy to the 5090   3 ISP + Swin-L teacher labels (hard + soft), appended to
# the existing camera set   4 outdoor / indoor split (blocked val)   5 outdoor fine-tune from the current best module (soft KD, rare-class
# sampling, ClassMix, MIC). Afterwards: compare with scripts/cmp + export_m2f_indoor.py + trtexec on the Orin (see docs/handoff).
set -e
cd "$(dirname "$0")/.."
KEY=~/.ssh/id_ed25519_orin; ORIN=lovelace@10.9.25.123; RAW=${RAW:-data/general/camraw}; STEP=${STEP:-6}
INIT=${INIT:-runs/modules/m2f_out_cam9/best/model.pt}; OUT=${OUT:-runs/modules/m2f_out_car_$(date +%m%d_%H%M)}
mkdir -p "$RAW"
scp -q -i $KEY scripts/orin_sub_raw.py $ORIN:/tmp/sub_raw.py
for n in "$@"; do
  ssh -i $KEY $ORIN "mkdir -p /tmp/sub && sudo -n python3 /tmp/sub_raw.py /home/haktan/car/$n /tmp/sub/$n.npy $STEP && sudo -n chmod 644 /tmp/sub/$n.npy"
  scp -q -i $KEY $ORIN:/tmp/sub/$n.npy "$RAW/"
done
env -u PYTHONPATH .venv/bin/python scripts/camera_pseudo_labels.py "$RAW"
env -u PYTHONPATH .venv/bin/python scripts/camera_soft_labels.py
python3 scripts/camera_split.py
mkdir -p "$OUT"
env -u PYTHONPATH CAMWIDE=1 CLASSSET=out INIT="$INIT" OUT="$OUT" ITERS=${ITERS:-8000} LR=1e-5 P_CAM=${P_CAM:-0.5} P_PSEUDO=0 P_MIX=0.5 P_MIC=0.5 \
  .venv/bin/python scripts/finetune_m2f_indoor.py 2>&1 | grep --line-buffered -v -i warn | tee "$OUT/train.log"
