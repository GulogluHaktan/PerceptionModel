#!/bin/bash
# Stop a car_bg.sh recording cleanly (SIGINT = Ctrl+C: run.py finishes the raw dump and prints the summary).
sudo pkill -INT -f "^timeout -s INT" && echo "stop sent, finishing..." || echo "no recording running"
sleep 4; tail -4 ~/car_last.log
