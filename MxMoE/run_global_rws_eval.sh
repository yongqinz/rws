#!/bin/bash
# Global RWS (Route-Weighted Global Allocation) Evaluation
# 2 models × 3 budgets = 6 configs, each: PPL + downstream tasks
set -e
cd "$(dirname "$0")"

MODELS=(qwen2_moe deepseek_moe)
BUDGETS=(2.5 2.75 3.0)
QTYPE=rtn
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python

RESULTS_DIR="results/global_rws"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Global RWS Evaluation (2 models × 3 budgets)"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    for BUDGET in "${BUDGETS[@]}"; do
        QCFG="qconfigs/global_rws/${MODEL}_rtn_grws_rws_wbits${BUDGET}.json"
        NAME="${MODEL}_grws_${BUDGET}b"

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
