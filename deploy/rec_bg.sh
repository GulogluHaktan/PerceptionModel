#!/bin/bash
# Record-only (no perception, low power) as the logged-in user, detached from the terminal: survives ssh / RDP disconnects.
#   bash ~/rec_bg.sh <name> [seconds=600]      follow: tail -f ~/rec_last.log      stop early: bash ~/rec_stop.sh
#   preview window: python3 ~/rec_view.py      video afterwards: python3 ~/rec/rec_to_mp4.py ~/car/<name>   (-> ~/car/<name>/video.mp4)
# Raw frames go to ~/car/<name>/ (frames.raw + stamps.txt + meta.json). Does not change the power mode (switching it reboots the Orin).
NAME=${1:-rec_$(date +%H%M%S)}; SECS=${2:-600}; OUT=$HOME/car/$NAME
if [ -e "$OUT" ]; then NAME=${NAME}_$(date +%H%M%S); OUT=$HOME/car/$NAME; echo "name already used: recording to $OUT instead"; fi
FREE=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); NEED=$((SECS * 17 / 100 + 5))
if [ "$FREE" -lt "$NEED" ]; then echo "only ${FREE} GB free, need ~${NEED} GB"; exit 1; fi
sudo systemctl stop nvargus-daemon 2>/dev/null
echo "power mode: $(nvpmodel -q 2>/dev/null | head -1)"
mkdir -p "$OUT"; cd ~/rec
setsid nohup /usr/bin/python3 rec_only.py "$OUT" --secs "$SECS" > ~/rec_last.log 2>&1 < /dev/null &
sleep 4; tail -3 ~/rec_last.log
