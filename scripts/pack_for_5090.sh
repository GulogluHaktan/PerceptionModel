#!/usr/bin/env bash
# Packs what the 5090 needs (code, configs, docs, assets 330 MB, sim scripts, tests). NOT: .venv, data, runs. Usage: bash scripts/pack_for_5090.sh [out.tar.gz]
set -euo pipefail
cd "$(dirname "$0")/.."; OUT=${1:-/tmp/pfm_5090.tar.gz}
tar --exclude='__pycache__' --exclude='.pytest_cache' -czf "$OUT" src sim scripts configs tests docs assets pyproject.toml README.md 2>/dev/null || tar --exclude='__pycache__' -czf "$OUT" src sim scripts configs tests docs assets
ls -lh "$OUT"
