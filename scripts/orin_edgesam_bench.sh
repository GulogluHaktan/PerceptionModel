#!/usr/bin/env bash
# EdgeSAM encoder latency on the Orin (decision 2: label model on Orin, first candidate EdgeSAM).
# Downloads the ready-made ONNX files from Hugging Face (chongzhou/EdgeSAM, 3x variant), builds a TensorRT fp16 engine, times it.
set -e
D=runs/edgesam; mkdir -p $D
for f in edge_sam_3x_encoder.onnx edge_sam_3x_decoder.onnx; do
  [ -f $D/$f ] || curl -L -s -o $D/$f https://huggingface.co/chongzhou/EdgeSAM/resolve/main/$f
done
ls -l $D | awk '{print $5, $9}'
TRT=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
[ -f $D/encoder.plan ] || $TRT --onnx=$D/edge_sam_3x_encoder.onnx --fp16 --saveEngine=$D/encoder.plan > $D/build.log 2>&1
$TRT --loadEngine=$D/encoder.plan --warmUp=2000 --duration=15 --avgRuns=100 --noDataTransfers 2>&1 | grep -E "Throughput|GPU Compute Time: min|Input|Inputs"
