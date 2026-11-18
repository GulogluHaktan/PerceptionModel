#!/usr/bin/env bash
# Whole v0 flow on the 5090 after setup_5090.sh. Resumable: rerun to continue.
# Usage: bash scripts/run_v0.sh [images_per_source]
set -euo pipefail
N=${1:-2000}
PY=.venv/bin/python
[ -d data/images ] && [ "$(ls data/images 2>/dev/null | wc -l)" -gt 0 ] || $PY scripts/fetch_data.py --n "$N"
$PY scripts/check_images.py data/images
[ -d data/labels/seg ] || $PY scripts/label_folder.py data/images data/labels 448
$PY scripts/cache_teachers.py configs/v0_5090.yaml
CFG=configs/v0_5090.yaml
[ -d data/golden ] || { sed '/^golden:/d' $CFG > /tmp/v0_nogolden.yaml; CFG=/tmp/v0_nogolden.yaml; echo "no golden set yet: gate has no metrics"; }
for ST in vit hybrid; do
  sed "s/^student: .*/student: $ST/" $CFG > /tmp/v0_$ST.yaml
  $PY scripts/train.py /tmp/v0_$ST.yaml
done
$PY scripts/bench_students.py
