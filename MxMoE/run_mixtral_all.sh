#!/bin/bash
# Mixtral-8x7B All-in-One Evaluation Script
# Methods: FP16, RTN uniform, GPTQ, MxMoE Baseline, RWS, Freq-Only, Hessian
set -e
cd "$(dirname "$0")"

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
MODEL=mixtral
RESULTS=results
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"
BUDGETS=(2.5 2.75 3.0)
BS=64

# ========================================
echo "============================================"
echo "Mixtral-8x7B All Methods Evaluation"
echo "============================================"

# ---- 1. FP16 ----
echo ""
echo "=== [1] FP16 ==="
for TYPE in ppl tasks; do
    SAVE="${RESULTS}/${MODEL}_fp16_${TYPE}.json"
    if [ ! -f "$SAVE" ]; then
        echo "  [FP16] $TYPE ..."
        $PYTHON -m mxmoe.quant.quant eval \
            --model "$MODEL" --method rtn \
            --tasks ${TYPE} \
            $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
            --save "$SAVE"
    else
        echo "  [SKIP] FP16 $TYPE"
    fi
done

# ---- 2. RTN uniform (W2/W3/W4) ----
echo ""
echo "=== [2] RTN Uniform ==="
for WBITS in 2 3 4; do
    for TYPE in ppl tasks; do
        SAVE="${RESULTS}/${MODEL}_rtn_w${WBITS}_${TYPE}.json"
        if [ ! -f "$SAVE" ]; then
            echo "  [RTN-W${WBITS}] $TYPE ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method rtn --wbits "$WBITS" \
                --tasks ${TYPE} \
                $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
                --save "$SAVE"
        else
            echo "  [SKIP] RTN-W${WBITS} $TYPE"
        fi
    done
done

# ---- 3. GPTQ uniform (W3/W4) ----
echo ""
echo "=== [3] GPTQ Uniform ==="
for WBITS in 3 4; do
    for TYPE in ppl tasks; do
        SAVE="${RESULTS}/${MODEL}_gptq_w${WBITS}_${TYPE}.json"
        if [ ! -f "$SAVE" ]; then
            echo "  [GPTQ-W${WBITS}] $TYPE ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method gptq --wbits "$WBITS" \
                --tasks ${TYPE} \
                $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
                --save "$SAVE"
        else
            echo "  [SKIP] GPTQ-W${WBITS} $TYPE"
        fi
    done
done

# ---- 4. MxMoE Baseline ----
echo ""
echo "=== [4] MxMoE Baseline ==="
QCONFIGS="qconfigs/w2a16_g128_asym+w4a16_g-1_asym"
for BUDGET in 2.5 2.75 3.0; do
    QCFG="${QCONFIGS}/${MODEL}_rtn_Slayer_bs512_wbits${BUDGET}_r1.0.json"
    for TYPE in ppl tasks; do
        SAVE="${RESULTS}/${MODEL}_baseline_${BUDGET}b_${TYPE}.json"
        if [ ! -f "$SAVE" ]; then
            echo "  [Baseline ${BUDGET}b] $TYPE ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method rtn --qconfig "$QCFG" \
                --tasks ${TYPE} \
                $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
                --save "$SAVE"
        else
            echo "  [SKIP] Baseline ${BUDGET}b $TYPE"
        fi
    done
done

# ---- 5. RWS (Route-Weighted) ----
echo ""
echo "=== [5] RWS (Route-Weighted) ==="
for BUDGET in 2.5 2.75 3.0; do
    QCFG="${QCONFIGS}/${MODEL}_rtn_Slayer_bs512_wbits${BUDGET}_r1.0_prop1.json"
    for TYPE in ppl tasks; do
        SAVE="${RESULTS}/${MODEL}_routewtd_${BUDGET}b_${TYPE}.json"
        if [ ! -f "$SAVE" ]; then
            echo "  [RWS ${BUDGET}b] $TYPE ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method rtn --qconfig "$QCFG" \
                --tasks ${TYPE} \
                $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
                --save "$SAVE"
        else
            echo "  [SKIP] RWS ${BUDGET}b $TYPE"
        fi
    done
done

# ---- 6. Freq-Only ----
echo ""
echo "=== [6] Freq-Only ==="
FREQ_DIR="${RESULTS}/freq_only"
mkdir -p "$FREQ_DIR"
for BUDGET in "${BUDGETS[@]}"; do
    QCFG="qconfigs/freq_only/${MODEL}_rtn_freq_wbits${BUDGET}.json"
    if [ ! -f "$QCFG" ]; then
        echo "  [SKIP] Freq ${BUDGET}b qconfig not found"
        continue
    fi
    for TYPE in ppl tasks; do
        SAVE="${FREQ_DIR}/${MODEL}_freq_${BUDGET}b_${TYPE}.json"
        if [ ! -f "$SAVE" ]; then
            echo "  [Freq ${BUDGET}b] $TYPE ..."
            $PYTHON -m mxmoe.quant.quant eval \
                --model "$MODEL" --method rtn --qconfig "$QCFG" \
                --tasks ${TYPE} \
                $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
                --save "$SAVE"
        else
            echo "  [SKIP] Freq ${BUDGET}b $TYPE"
        fi
    done
done

# ---- 7. Hessian (Hess-Only + Hess-RWS) ----
echo ""
echo "=== [7] Hessian (Hess-Only + Hess-RWS) ==="

# Step 7a: Compute Hessian trace if not done
HESSIAN_FILE="calib/hessian/${MODEL}_hessian.json"
if [ ! -f "$HESSIAN_FILE" ]; then
    echo "  Computing Hessian trace (GPU required, ~30min)..."
    $PYTHON compute_hessian.py --model "$MODEL" --num_samples 50 --save "$HESSIAN_FILE"
else
    echo "  [SKIP] Hessian data exists"
fi

# Step 7b: Generate qconfigs
HESS_CFG_DIR="qconfigs/hessian"
mkdir -p "$HESS_CFG_DIR"
for BUDGET in "${BUDGETS[@]}"; do
    for MODE in hess_only rws; do
        QCFG="${HESS_CFG_DIR}/${MODEL}_rtn_hess_${MODE}_wbits${BUDGET}.json"
        if [ ! -f "$QCFG" ]; then
            USE_RWS=$([ "$MODE" = "rws" ] && echo "--use_rws" || echo "--no_rws")
            echo "  Generating ${MODE} ${BUDGET}b qconfig..."
            $PYTHON gen_hessian_qconfig.py --model "$MODEL" --wbits "$BUDGET" $USE_RWS --save "$QCFG"
        fi
    done
done

# Step 7c: Evaluate
HESS_RES_DIR="${RESULTS}/hessian"
mkdir -p "$HESS_RES_DIR"
for BUDGET in "${BUDGETS[@]}"; do
    for MODE in hess_only rws; do
        QCFG="${HESS_CFG_DIR}/${MODEL}_rtn_hess_${MODE}_wbits${BUDGET}.json"
        NAME="${MODEL}_${MODE}_${BUDGET}b"
        if [ ! -f "$QCFG" ]; then continue; fi

        for TYPE in ppl tasks; do
            SAVE="${HESS_RES_DIR}/${NAME}_${TYPE}.json"
            if [ ! -f "$SAVE" ]; then
                echo "  [Hess-${MODE} ${BUDGET}b] $TYPE ..."
                $PYTHON -m mxmoe.quant.quant eval \
                    --model "$MODEL" --method rtn --qconfig "$QCFG" \
                    --tasks ${TYPE} \
                    $([ "$TYPE" = "tasks" ] && echo "--bs $BS --tasks $TASKS") \
                    --save "$SAVE"
            else
                echo "  [SKIP] Hess-${MODE} ${BUDGET}b $TYPE"
            fi
        done
    done
done

# ========================================
echo ""
echo "============================================"
echo "Mixtral-8x7B All Methods DONE!"
echo "============================================"
