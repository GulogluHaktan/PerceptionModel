#!/usr/bin/env bash
# One-shot setup for the 5090 machine. Usage: bash scripts/setup_5090.sh
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python torch torchvision timm "transformers==4.57.*" "huggingface-hub>=0.34,<1.0" trimesh accelerate einops open_clip_torch opencv-python-headless numpy pyyaml pillow tqdm pytest
# transformers must stay <5: the E-RADIO remote code breaks on 5.x. huggingface-hub must stay <1.0 for 4.57.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# If the default torch build (newest CUDA) does not match an older NVIDIA driver, fall back to the cu128 build
# (first CUDA version with Blackwell/sm_120 support). Driver too old for cu132 shows up as cuda=False here.
# FORCE_CU128=1 runs this branch on purpose (to test it on a machine whose driver is new enough).
if [ "${FORCE_CU128:-0}" = 1 ] || ! .venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA not usable with the default torch build: trying the cu128 build"
  uv pip install -p .venv/bin/python --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
# RTX 5090 is Blackwell (sm_120): needs a CUDA 12.8+ torch build.
print("capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "CUDA not available: check driver / torch build for sm_120"
x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item(); print("matmul ok")
PY
.venv/bin/python -m pytest -q tests
python3 scripts/make_fake.py 2>/dev/null || .venv/bin/python scripts/make_fake.py
.venv/bin/python scripts/cache_teachers.py configs/smoke.yaml && .venv/bin/python scripts/train.py configs/smoke.yaml
echo "SETUP OK"
