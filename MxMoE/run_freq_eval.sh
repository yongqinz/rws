#!/bin/bash
# Frequency-only allocation evaluation
# 3 models × 3 budgets = 9 configs, each: PPL + 6 downstream tasks
set -e
cd "$(dirname "$0")"

MODELS=(qwen2_moe deepseek_moe mixtral)
BUDGETS=(2.5 2.75 3.0)
QTYPE=rtn
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python

RESULTS_DIR="results/freq_only"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Frequency-Only Evaluation (3 models × 3 budgets)"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    for BUDGET in "${BUDGETS[@]}"; do
        QCFG="qconfigs/freq_only/${MODEL}_rtn_freq_wbits${BUDGET}.json"
        NAME="${MODEL}_freq_${BUDGET}b"

        if [ ! -f "$QCFG" ]; then
            echo "[SKIP] Config not found: $QCFG"
            continue
        fi

        echo ""
        echo "============================================"
        echo "Evaluating: $NAME"
        echo "============================================"

        # PPL
        echo "[PPL] $NAME ..."
        $PYTHON -m mxmoe.quant.quant eval \
            --model "$MODEL" \
            --method "$QTYPE" \
            --qconfig "$QCFG" \
            --tasks ppl \
            --save "$RESULTS_DIR/${NAME}_ppl.json"

        # Downstream
        echo "[Tasks] $NAME ..."
        $PYTHON -m mxmoe.quant.quant eval \
            --model "$MODEL" \
            --method "$QTYPE" \
            --qconfig "$QCFG" \
            --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
            --save "$RESULTS_DIR/${NAME}_tasks.json"
    done
done

echo ""
echo "============================================"
echo "All done! Results in $RESULTS_DIR/"
echo "============================================"
