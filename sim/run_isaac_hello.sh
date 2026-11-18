#!/bin/bash
# Isaac Sim hello in the container, set up like the working LunarRocket scripts (image 6.0.1, shared warm caches in ~/docker/isaac-sim, umask 000).
set -e
PROJ=/home/haktan/Projects/PerceptionModel; CR=${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}
mkdir -p $PROJ/runs/isaac_hello; chmod -R ugo+rwx $PROJ/runs/isaac_hello 2>/dev/null || true   # folders made by earlier container runs belong to root
docker run --rm --entrypoint /bin/bash --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONPATH=/workspace/pfm/src -v "$PROJ:/workspace/pfm" \
  -v $CR/cache/ov:/root/.cache/ov -v $CR/cache/pip:/root/.cache/pip -v $CR/cache/glcache:/root/.cache/nvidia -v $CR/cache/kit:/root/.cache/kit \
  -v $CR/data:/root/.local/share/ov -v $CR/documents:/root/Documents -w /workspace/pfm \
  --name isaac-pfm-hello "${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}" \
  -lc "umask 000 && cd /isaac-sim && ./python.sh /workspace/pfm/sim/isaac_hello.py"
