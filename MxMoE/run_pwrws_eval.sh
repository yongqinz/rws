#!/bin/bash
# Per-Weight Route-Weighted Sensitivity (PW-RWS) Evaluation
set -e
cd "$(dirname "$0")"

MODELS=(qwen2_moe deepseek_moe)
BUDGETS=(2.5 2.75 3.0)
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS_DIR="results/pwrws"
QCFG_DIR="qconfigs/global_rws"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "PW-RWS Evaluation (2 models × 3 budgets)"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    for BUDGET in "${BUDGETS[@]}"; do
        QCFG="$QCFG_DIR/${MODEL}_rtn_pwrws_rws_wbits${BUDGET}.json"
        NAME="${MODEL}_pwrws_${BUDGET}b"

        if [ ! -f "$QCFG" ]; then
            echo "[SKIP] Config not found: $QCFG"
            continue
        fi

        echo ""
        echo "============================================"
        echo "Evaluating: $NAME"
        echo "============================================"

        echo "[PPL] $NAME ..."
        $PYTHON -m mxmoe.quant.quant eval \
            --model "$MODEL" \
            --method rtn \
            --qconfig "$QCFG" \
            --tasks ppl \
            --save "$RESULTS_DIR/${NAME}_ppl.json"

        echo "[Tasks] $NAME ..."
        $PYTHON -m mxmoe.quant.quant eval \
            --model "$MODEL" \
            --method rtn \
            --qconfig "$QCFG" \
            --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
            --save "$RESULTS_DIR/${NAME}_tasks.json"
    done
done

echo ""
echo "============================================"
echo "All done! Results in $RESULTS_DIR/"
echo "============================================"
