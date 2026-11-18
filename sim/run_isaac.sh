#!/bin/bash
# Generic launcher: sim/run_isaac.sh sim/<script>.py  (official Isaac Sim 6.0.1 container, shared warm caches, same setup as LunarRocket)
# Native Isaac Sim (5090 machine): ISAAC_NATIVE=1 (pip install in .venv_isaac, see scripts/setup_isaac_native_5090.sh) or ISAAC_PY=<python or isaac-sim/python.sh>. Otherwise the container below.
set -e
PROJ=${PFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; SCRIPT=${1:?script path relative to the project}
if [ "${ISAAC_NATIVE:-0}" = 1 ] || [ -n "${ISAAC_PY:-}" ]; then
  export PFM_ROOT=$PROJ PYTHONPATH=$PROJ/src OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
  mkdir -p $PROJ/runs/fg_job/out; cd $PROJ; exec "${ISAAC_PY:-$PROJ/.venv_isaac/bin/python}" "$PROJ/$SCRIPT"
fi
CR=${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}
mkdir -p $PROJ/runs/fg_job/out; chmod -R ugo+rwx $PROJ/runs/fg_job 2>/dev/null || true
docker run --rm --entrypoint /bin/bash --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONPATH=/workspace/pfm/src -v "$PROJ:/workspace/pfm" \
  -v $CR/cache/ov:/root/.cache/ov -v $CR/cache/pip:/root/.cache/pip -v $CR/cache/glcache:/root/.cache/nvidia -v $CR/cache/kit:/root/.cache/kit \
  -v $CR/data:/root/.local/share/ov -v $CR/documents:/root/Documents -w /workspace/pfm \
  --name isaac-pfm-run "${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}" \
  -lc "umask 000 && cd /isaac-sim && ./python.sh /workspace/pfm/$SCRIPT"
