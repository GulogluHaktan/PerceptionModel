#!/bin/bash
# Stop a rec_bg.sh recording cleanly (SIGINT: files closed, meta.json written).
sudo pkill -INT -f "^/usr/bin/python3 rec_only.py" && echo "stop sent, finishing..." || echo "no recording running"
sleep 3; tail -3 ~/rec_last.log
