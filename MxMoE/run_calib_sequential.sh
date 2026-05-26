#!/bin/bash
# Sequential calibration: W4A16 first (already running), then W8A8
set -e
cd "$(dirname "$0")"

GPU=0
NSAMPLES=128
SEED=42

# Check if W4A16 is still running
W4A16_FILE="calib/qwen2_moe-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json"

while true; do
    if [ ! -f "$W4A16_FILE" ]; then
        echo "[$(date)] W4A16 file not found yet, waiting..."
        sleep 60
        continue
    fi

    LAYERS=$(python3 -c "import json; print(len(json.load(open('$W4A16_FILE'))))")
    echo "[$(date)] W4A16 progress: $LAYERS/24 layers"

    if [ "$LAYERS" -ge 24 ]; then
        echo "[$(date)] W4A16 complete! Starting W8A8..."
        break
    fi

    # Check if W4A16 process is still running
    if ! pgrep -f "qcfg w4a16_g-1_asym" > /dev/null 2>&1; then
        echo "[$(date)] W4A16 process not running but only $LAYERS layers. May have crashed."
        break
    fi

    sleep 120
done

# Now run W8A8
W8A8_FILE="calib/qwen2_moe-MOE-rtn-W8A8_g-1_sym-wiki2-128-4096-layer_out_norm.json"
if [ -f "$W8A8_FILE" ]; then
    LAYERS=$(python3 -c "import json; print(len(json.load(open('$W8A8_FILE'))))")
    if [ "$LAYERS" -ge 24 ]; then
        echo "[$(date)] W8A8 already complete ($LAYERS layers). Skipping."
        exit 0
    fi
fi

echo "[$(date)] Starting W8A8 calibration on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU /data/zengyq/miniconda3/envs/arpo/bin/python3 -m mxmoe.quant.quant calib \
    --model qwen2_moe \
    --method rtn \
    --metric layer_out_norm \
    --qcfg w8a8_g-1_sym \
    --nsamples $NSAMPLES \
    --seed $SEED

echo "[$(date)] W8A8 calibration done!"
ls -la "$W8A8_FILE"
