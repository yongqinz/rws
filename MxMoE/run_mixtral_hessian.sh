#!/bin/bash
# Mixtral Hessian-based allocation: compute hessian → gen qconfigs → eval
# Outputs: calib/hessian/mixtral_hessian.json, qconfigs/hessian/mixtral_*.json, results/hessian/mixtral_*
set -e
cd "$(dirname "$0")"

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
MODEL=mixtral
BUDGETS=(2.5 2.75 3.0)
RESULTS_DIR="results/hessian"
QCFG_DIR="qconfigs/hessian"
HESSIAN_FILE="calib/hessian/mixtral_hessian.json"

echo "============================================"
echo "Mixtral Hessian Pipeline"
echo "============================================"

# ---- Step 0: Compute Hessian trace (GPU required, ~30min) ----
if [ ! -f "$HESSIAN_FILE" ]; then
    echo ""
    echo "[Step 0] Computing Hessian trace for Mixtral..."
    $PYTHON compute_hessian.py --model mixtral --num_samples 50 --save "$HESSIAN_FILE"
else
    echo "[Step 0] Hessian data exists: $HESSIAN_FILE"
fi

# ---- Step 1: Generate qconfigs ----
echo ""
echo "[Step 1] Generating Hessian qconfigs..."
mkdir -p "$QCFG_DIR"

for BUDGET in "${BUDGETS[@]}"; do
    # Hess-Only (no RWS)
    QCFG="$QCFG_DIR/${MODEL}_rtn_hess_hess_only_wbits${BUDGET}.json"
    if [ ! -f "$QCFG" ]; then
        echo "  Generating Hess-Only ${BUDGET}b..."
        $PYTHON gen_hessian_qconfig.py --model "$MODEL" --wbits "$BUDGET" --no_rws --save "$QCFG"
    else
        echo "  [SKIP] Hess-Only ${BUDGET}b qconfig exists"
    fi

    # Hess-RWS
    QCFG="$QCFG_DIR/${MODEL}_rtn_hess_rws_wbits${BUDGET}.json"
    if [ ! -f "$QCFG" ]; then
        echo "  Generating Hess-RWS ${BUDGET}b..."
        $PYTHON gen_hessian_qconfig.py --model "$MODEL" --wbits "$BUDGET" --use_rws --save "$QCFG"
    else
        echo "  [SKIP] Hess-RWS ${BUDGET}b qconfig exists"
    fi
done

# ---- Step 2: Evaluate ----
echo ""
echo "[Step 2] Evaluating Hessian configs..."
mkdir -p "$RESULTS_DIR"

for BUDGET in "${BUDGETS[@]}"; do
    for MODE in hess_only rws; do
        QCFG="$QCFG_DIR/${MODEL}_rtn_hess_${MODE}_wbits${BUDGET}.json"
        NAME="${MODEL}_${MODE}_${BUDGET}b"

        if [ ! -f "$QCFG" ]; then
            echo "  [SKIP] $NAME qconfig not found"
            continue
        fi

        # PPL
        PPL_FILE="$RESULTS_DIR/${NAME}_ppl.json"
        if [ ! -f "$PPL_FILE" ]; then
            echo "  [PPL] $NAME ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method rtn --qconfig "$QCFG" \
                --tasks ppl --save "$PPL_FILE"
        else
            echo "  [SKIP] $NAME PPL exists"
        fi

        # Tasks
        TASKS_FILE="$RESULTS_DIR/${NAME}_tasks.json"
        if [ ! -f "$TASKS_FILE" ]; then
            echo "  [Tasks] $NAME ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method rtn --qconfig "$QCFG" \
                --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
                --bs 64 --save "$TASKS_FILE"
        else
            echo "  [SKIP] $NAME tasks exist"
        fi
    done
done

echo ""
echo "============================================"
echo "Mixtral Hessian pipeline DONE!"
echo "============================================"
