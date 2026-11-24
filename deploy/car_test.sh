#!/bin/bash
# Car test (Update 167): live perception on the wide raw camera + every raw frame saved, one command.
#   bash car_test.sh [name] [seconds] [rotate] [scene]   e.g.  bash car_test.sh run1 300 0 outdoor   (scene: outdoor (default) | indoor; only that module runs)
# Writes ~/car/<name>/ : frames.raw (+stamps, meta: every camera frame, 84 fps, ~170 MB/s), rec/ (model outputs every 20th frame:
# image, 11-class seg, indoor/outdoor map, depth, detections.jsonl), log.txt. Stop early with Ctrl+C (once).
# Afterwards: python raw_record.py conv ~/car/<name> --every 4 --mp4 --calib camera_imx219_cam1_1640x616.json --hfov 100
NAME=${1:-run_$(date +%H%M%S)}; SECS=${2:-600}; ROT=${3:-0}; SCENE=${4:-outdoor}; OUT=/home/haktan/car/$NAME      # ROT: 0 / 90 / 180 / 270 (camera mounted rotated, clockwise)
sudo systemctl stop nvargus-daemon 2>/dev/null
sudo jetson_clocks 2>/dev/null       # lock CPU/GPU/EMC clocks at max (reset by reboot)
sudo -u haktan -H bash -c "mkdir -p $OUT && cd ~/pfm_drone384 && timeout -s INT $SECS /home/haktan/PerceptionModel/.venv/bin/python run.py --bundle bundle --source raw:7 --backend trt \
  --calib camera_imx219_cam1_1640x616.json --hfov 100 --rotate $ROT --scene $SCENE --exp-max ${EXPMAX:-8000} ${CAMFPS:+--cam-fps $CAMFPS} --raw-dump $OUT --record $OUT/rec --record-every 20 2>&1 \
  | grep --line-buffered -v '\[W\]' | tee $OUT/log.txt"
echo "saved in $OUT"; df -h /home | tail -1
