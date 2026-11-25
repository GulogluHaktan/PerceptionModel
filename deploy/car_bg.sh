#!/bin/bash
# Start car_test.sh detached from the terminal: the recording keeps running if ssh / the laptop disconnects.
#   bash car_bg.sh araba1 1800 0 outdoor      follow: tail -f ~/car_last.log      stop early: bash ~/car_stop.sh
setsid nohup bash ~/car_test.sh "$@" > ~/car_last.log 2>&1 < /dev/null &
sleep 3; echo "recording started (log ~/car_last.log)"; tail -3 ~/car_last.log
