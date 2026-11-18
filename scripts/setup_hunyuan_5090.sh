#!/usr/bin/env bash
# Hunyuan3D-2mini env for scripts/image_to_asset.py and scripts/grow_catalogue.py (photo -> 3D asset). Third-party code stays OUT of the project, in $I23D (default ~/Projects/i23d).
# Licence: Tencent Hunyuan 3D 2.0 Community Licence (territory limits: not EU / UK / South Korea): read $I23D/hy3d/LICENSE before using generated assets in a paper or release.
# Versions are the ones that worked on the laptop (2026-11-13): huggingface-hub<1, diffusers 0.35.2, transformers<5. Weights (~5 GB) download on first use.
set -euo pipefail
I23D=${I23D:-$HOME/Projects/i23d}; mkdir -p "$I23D"; cd "$I23D"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d hy3d ] || git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git hy3d
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv/bin/python numpy pillow trimesh "transformers>=4.40,<5" "huggingface-hub<1.0" "diffusers==0.35.2" accelerate einops omegaconf pymeshlab pygltflib opencv-python-headless tqdm pyyaml safetensors scikit-image rembg onnxruntime usd-core scipy
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
cd - >/dev/null
# smoke test: one catalogue render -> asset (needs the input picture used on the laptop? no: any picture works, here a generated one)
.venv/bin/python - <<'PY'
from PIL import Image, ImageDraw
im = Image.new("RGB", (512, 512), "white"); d = ImageDraw.Draw(im); d.rectangle([150, 250, 360, 330], fill=(120, 80, 40)); d.rectangle([160, 330, 185, 440], fill=(90, 60, 30)); d.rectangle([325, 330, 350, 440], fill=(90, 60, 30)); d.rectangle([150, 130, 175, 250], fill=(120, 80, 40)); im.save("/tmp/pfm_hy_smoke.png")
PY
I23D=$I23D "$I23D/.venv/bin/python" scripts/image_to_asset.py /tmp/pfm_hy_smoke.png chair hy_smoke --out /tmp/pfm_hy_smoke_out | tail -1 | cut -c1-200
echo "HUNYUAN OK (I23D=$I23D)"
