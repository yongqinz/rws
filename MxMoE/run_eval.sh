#!/bin/bash
# RouteQEP Evaluation Pipeline
# Runs after calibration and ILP solving are complete
#
# Usage: bash run_eval.sh [model] [qtype]
set -e
cd "$(dirname "$0")"

MODEL=${1:-qwen2_moe}
QTYPE=${2:-rtn}
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================"
echo "RouteQEP Evaluation"
echo "============================================"
echo "Model: $MODEL"
echo "Quant type: $QTYPE"
echo "GPU: $GPU"
echo "============================================"

# Define the qconfig files to evaluate
QCONFIGS=(
    # Baseline (prop_hop=0)
    "qconfigs/w4a16_g-1_asym+w8a8_g-1_sym/${MODEL}_${QTYPE}_Slayer_bs512_wbits3.25_r1.0.json"
    # RouteQEP (prop_hop=1)
    "qconfigs/w4a16_g-1_asym+w8a8_g-1_sym/${MODEL}_${QTYPE}_Slayer_bs512_wbits3.25_r1.0_prop1.json"
)

RESULTS_DIR="results/routeqep_${MODEL}"
mkdir -p "$RESULTS_DIR"

for QCFG in "${QCONFIGS[@]}"; do
    if [ ! -f "$QCFG" ]; then
        echo "[SKIP] Config not found: $QCFG"
        continue
    fi

    # Extract a name from the path
    NAME=$(basename "$QCFG" .json)
    echo ""
    echo "============================================"
    echo "Evaluating: $NAME"
    echo "============================================"

    # PPL evaluation
    echo "[PPL] Evaluating perplexity on WikiText2..."
    /data/zengyq/miniconda3/envs/arpo/bin/python3 -m mxmoe.quant.quant eval \
        --model "$MODEL" \
        --method "$QTYPE" \
        --qconfig "$QCFG" \
        --tasks ppl \
        --save "$RESULTS_DIR/${NAME}_ppl.json"

    # Downstream tasks
    echo "[Tasks] Evaluating downstream tasks..."
    /data/zengyq/miniconda3/envs/arpo/bin/python3 -m mxmoe.quant.quant eval \
        --model "$MODEL" \
        --method "$QTYPE" \
        --qconfig "$QCFG" \
        --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
        --save "$RESULTS_DIR/${NAME}_tasks.json"
done

echo ""
echo "============================================"
echo "Evaluation complete! Results in $RESULTS_DIR/"
echo "============================================"
