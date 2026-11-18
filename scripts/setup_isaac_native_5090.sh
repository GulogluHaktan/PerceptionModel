#!/usr/bin/env bash
# Native Isaac Sim for the 5090 machine (no docker). Finds or installs it, then runs the hello scene through sim/run_isaac.sh with ISAAC_NATIVE=1.
# Order of detection: $ISAAC_PY (a python or isaac-sim/python.sh you already have) -> $ISAAC_SIM_PATH/python.sh -> existing .venv_isaac -> pip install isaacsim==$ISAAC_VERSION into .venv_isaac (Python 3.12, same version as the container used on the laptop: 6.0.1).
# Needs: NVIDIA driver for Blackwell (580+), glibc >= 2.35, ~25 GB disk. The first start compiles shaders and downloads extensions: 5-15 minutes, later starts are fast. Stop Ollama/vLLM/trainers first (Isaac needs an idle GPU).
# NOT run on a 5090 yet (written on the laptop where only the container exists): if the pip step fails, install Isaac Sim 6.0.1 from NVIDIA's zip/workstation and rerun with ISAAC_SIM_PATH=<folder>.
set -euo pipefail
cd "$(dirname "$0")/.."; V=${ISAAC_VERSION:-6.0.1}
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] || echo "WARNING: GPU busy (>2 GB used). Stop other GPU jobs before rendering."
PY=""
if [ -n "${ISAAC_PY:-}" ]; then PY=$ISAAC_PY
elif [ -n "${ISAAC_SIM_PATH:-}" ] && [ -x "$ISAAC_SIM_PATH/python.sh" ]; then PY=$ISAAC_SIM_PATH/python.sh
elif [ -x .venv_isaac/bin/python ] && .venv_isaac/bin/python -c "import isaacsim" 2>/dev/null; then PY=.venv_isaac/bin/python
else
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv_isaac
  uv pip install -p .venv_isaac/bin/python "isaacsim[all,extscache]==$V" --extra-index-url https://pypi.nvidia.com || { echo "pip install of isaacsim==$V failed: install Isaac Sim $V from NVIDIA and rerun with ISAAC_SIM_PATH=<folder with python.sh>"; exit 1; }
  uv pip install -p .venv_isaac/bin/python numpy pillow
  PY=.venv_isaac/bin/python
fi
echo "Isaac python: $PY"; export ISAAC_PY=$PY PFM_ROOT=$PWD
rm -rf runs/isaac_hello; mkdir -p runs
bash sim/run_isaac.sh sim/isaac_hello.py > runs/isaac_hello_native.log 2>&1 || { echo "hello scene failed, see runs/isaac_hello_native.log"; tail -20 runs/isaac_hello_native.log; exit 1; }
grep -q "^DONE" runs/isaac_hello_native.log || { echo "hello scene did not finish: see runs/isaac_hello_native.log"; exit 1; }
echo "ISAAC_PY=$PY" > .isaac_env                                        # day1_5090.sh and run scripts source this
.venv/bin/python scripts/gen_sim_set.py check 2>&1 | tail -3
echo "NATIVE ISAAC OK (use: source .isaac_env; bash sim/run_indoor_set.sh <layout names>)"
