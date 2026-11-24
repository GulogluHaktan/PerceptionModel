#!/usr/bin/env bash
# One-shot Orin bring-up from the dev PC: copy the package, build engines, accuracy check, latency, optional live run on a stream camera.
# Usage: deploy/orin_bringup.sh <user@orin_ip> [stream_url e.g. http://PHONE_IP:8080/video]
# Results land in deploy/orin_results/<date>/ (device info, build log, check_trt.json, run log).
set -euo pipefail
cd "$(dirname "$0")"
H=$1; URL=${2:-}; KEY=${ORIN_KEY:-$HOME/.ssh/id_ed25519_orin}
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 $H"
PKG=$(ls -t dist/pfm_vehicle_*.tar.gz | head -1); OUT=orin_results/$(date +%Y%m%d_%H%M); mkdir -p "$OUT"
echo "== device"; $SSH 'cat /etc/nv_tegra_release 2>/dev/null | head -1; uname -r; nvpmodel -q 2>/dev/null | head -2; free -h | head -2; df -h ~ | tail -1; python3 -c "import tensorrt, torch, cv2; print(tensorrt.__version__, torch.__version__, torch.cuda.is_available(), cv2.__version__)"' | tee "$OUT/device.txt"
echo "== copy $PKG"; scp -i "$KEY" "$PKG" "$H:~/"; $SSH "rm -rf ~/pfm_vehicle && tar xzf ~/$(basename "$PKG") -C ~ && cd ~/pfm_vehicle && sha256sum -c bundle/SHA256SUMS"
echo "== build engines (2-5 min)"; $SSH 'cd ~/pfm_vehicle && python3 build_engines.py bundle 2>&1 | grep -v "\[W\]"' | tee "$OUT/build.txt"
echo "== accuracy + latency"; $SSH 'cd ~/pfm_vehicle && python3 check_device.py bundle trt 1640 1232 2>&1 | grep -v "\[W\]" | tail -40' | tee "$OUT/check.txt"
scp -i "$KEY" "$H:~/pfm_vehicle/bundle/check_trt.json" "$OUT/" || true
echo "== replay (no camera)"; $SSH 'cd ~/pfm_vehicle && python3 -c "
import numpy as np, cv2; R=np.load(\"bundle/ref_frames.npz\"); w=cv2.VideoWriter(\"/tmp/pfm_t.mp4\", cv2.VideoWriter_fourcc(*\"mp4v\"), 30, (1640,1232))
[w.write(cv2.resize(cv2.cvtColor(im.transpose(1,2,0), cv2.COLOR_RGB2BGR), (1640,1232))) for _ in range(10) for im in R[\"images\"]]; w.release()" && timeout 300 python3 run.py --source file:/tmp/pfm_t.mp4 --backend trt --frames 300 --fps 30 2>&1 | grep -v "\[W\]" | tail -4' | tee "$OUT/replay.txt"
if [ -n "$URL" ]; then
  echo "== live stream camera $URL"; $SSH "cd ~/pfm_vehicle && timeout 300 python3 run.py --source url:$URL --backend trt --frames 600 --record ~/pfm_rec --record-every 20 2>&1 | grep -v '\[W\]' | tail -6" | tee "$OUT/live.txt"
  mkdir -p "$OUT/rec"; scp -i "$KEY" -r "$H:~/pfm_rec/*" "$OUT/rec/" 2>/dev/null || true
fi
echo "== done -> $OUT"
