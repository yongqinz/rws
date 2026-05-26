#!/bin/bash
# 2.3b extreme compression: Qwen1.5-MoE + DeepSeek-MoE, baseline + route-weighted
set -e
cd /data/zengyq/paper/baselines/MxMoE
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

MODEL=$1     # qwen2_moe or deepseek_moe
METHOD=$2    # baseline or routewtd
GPU=$3       # CUDA_VISIBLE_DEVICES

PREFIX=""
if [ "$MODEL" = "deepseek_moe" ]; then
    PREFIX="deepseek_moe_"
fi

if [ "$METHOD" = "baseline" ]; then
    TAG=""
    QCFG_TAG=""
else
    TAG="routewtd"
    QCFG_TAG="_prop1"
fi

QCFG="${QCONFIGS}/${MODEL}_rtn_Slayer_bs512_wbits2.3_r1.0${QCFG_TAG}.json"

# PPL
ppl_save="${RESULTS}/${PREFIX}${METHOD}_2.3b_ppl.json"
echo "[$(date)] ${MODEL} ${METHOD} 2.3b PPL..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mxmoe.quant.quant eval \
    --model $MODEL --method rtn \
    --qconfig "$QCFG" \
    --tasks ppl --save "$ppl_save"

# Tasks
tasks_save="${RESULTS}/${PREFIX}${METHOD}_2.3b_tasks.json"
echo "[$(date)] ${MODEL} ${METHOD} 2.3b Tasks..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mxmoe.quant.quant eval \
    --model $MODEL --method rtn \
    --qconfig "$QCFG" \
    --tasks $TASKS --bs 64 --save "$tasks_save"

echo "[$(date)] DONE: ${MODEL} ${METHOD}"
