#!/bin/bash
# Record-only (no perception, low power), detached from the terminal: survives ssh disconnects.
#   bash rec_bg.sh <name> [seconds=600]      follow: tail -f ~/rec_last.log      stop early: bash ~/rec_stop.sh
# Raw frames go to /home/haktan/car/<name>/ (same format as car_test.sh). Does not change the power mode (switching it reboots the Orin).
NAME=${1:-rec_$(date +%H%M%S)}; SECS=${2:-600}; OUT=/home/haktan/car/$NAME
if sudo test -e "$OUT"; then NAME=${NAME}_$(date +%H%M%S); OUT=/home/haktan/car/$NAME; echo "name already used: recording to $OUT instead"; fi
FREE=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); NEED=$((SECS * 17 / 100 + 5))
if [ "$FREE" -lt "$NEED" ]; then echo "only ${FREE} GB free, need ~${NEED} GB"; exit 1; fi
sudo systemctl stop nvargus-daemon 2>/dev/null
echo "power mode: $(sudo nvpmodel -q | head -1)  (set once with: sudo nvpmodel -m 1  -> 10W, needs a reboot; -m 0 = MAXN for perception)"
setsid nohup sudo -u haktan -H bash -c "mkdir -p $OUT && cd ~/pfm_drone384 && /usr/bin/python3 rec_only.py $OUT --secs $SECS" > ~/rec_last.log 2>&1 < /dev/null &
sleep 4; tail -3 ~/rec_last.log
