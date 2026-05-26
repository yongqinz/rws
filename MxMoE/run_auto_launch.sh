#!/bin/bash
cd /data/zengyq/paper/baselines/MxMoE

# Wait for both GPU scripts to finish
echo "[$(date)] Monitoring GPU 0 (PID $1) and GPU 1 (PID $2)..."
while kill -0 $1 2>/dev/null || kill -0 $2 2>/dev/null; do
  sleep 30
done
echo "[$(date)] Both scripts finished. Starting Mixtral experiments..."
bash run_mixtral.sh
