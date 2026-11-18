#!/usr/bin/env bash
# Isaac Sim 6.0.1 container for sim/run_isaac.sh on the 5090 machine (docker + NVIDIA container toolkit must exist). Creates the cache dirs, pulls the image and renders one indoor frame set as a check.
set -euo pipefail
CR=${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}; mkdir -p $CR/cache/{ov,pip,glcache,kit} $CR/{data,documents}; chmod -R ugo+rwx $CR
docker info >/dev/null; docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L || { echo "docker cannot see the GPU: install nvidia-container-toolkit"; exit 1; }
docker pull "${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"          # needs 'docker login nvcr.io' (user $oauthtoken, NGC API key) if not cached
cd "$(dirname "$0")/.."; ls assets/models/manifest.json >/dev/null
# Isaac renders go blank if another GPU job runs: stop Ollama / vLLM / trainers first (check nvidia-smi)
.venv/bin/python scripts/gen_sim_set.py check 2>&1 | tail -3
echo "ISAAC SETUP OK (now: .venv/bin/python scripts/gen_sim_set.py all 30)"
