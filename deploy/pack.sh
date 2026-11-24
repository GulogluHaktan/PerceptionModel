#!/usr/bin/env bash
# Build the device package: deploy/dist/pfm_vehicle_<date>.tar.gz (code + ONNX + reference frames; TensorRT engines are built ON the device).
set -euo pipefail
cd "$(dirname "$0")"
D=dist/pfm_vehicle; rm -rf "$D"; mkdir -p "$D/bundle"
cp pfm_runtime.py gdet_post.py camera.py run.py build_engines.py check_device.py calibrate_camera.py rec_sheet.py README.md pfm-perception.service "$D/"
cp ../src/pfm/refine.py "$D/pfm_refine.py"            # vendored: the runtime needs only patch sampling + read-out
cp bundle/system_448.onnx bundle/refiner.onnx bundle/bundle.json bundle/ref_frames.npz "$D/bundle/"
cp bundle/parity_*.json bundle/check_*.json "$D/bundle/" 2>/dev/null || true
( cd "$D" && sha256sum bundle/*.onnx bundle/ref_frames.npz > bundle/SHA256SUMS )
T=dist/pfm_vehicle_$(date +%Y%m%d).tar.gz; tar -C dist -czf "$T" pfm_vehicle; ls -la "$T"
