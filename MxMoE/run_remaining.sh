#!/bin/bash
set -e

cd /data/zengyq/paper/baselines/MxMoE
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
QCONFIG=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
RESULTS=results

# Already done: baseline_3.25b_ppl, baseline_3.25b_tasks (running)

# 1. Qwen 3.25b route-weighted PPL
if [ ! -f "$RESULTS/routewtd_3.25b_ppl.json" ]; then
    echo "[$(date)] Running Qwen 3.25b route-weighted PPL..."
    $PYTHON -m mxmoe.quant.quant eval \
        --model qwen2_moe --method rtn \
        --qconfig $QCONFIG/qwen2_moe_rtn_Slayer_bs512_wbits3.25_r1.0_prop1.json \
        --tasks ppl \
        --save $RESULTS/routewtd_3.25b_ppl.json
fi

# 2. Qwen 3.25b route-weighted tasks
if [ ! -f "$RESULTS/routewtd_3.25b_tasks.json" ]; then
    echo "[$(date)] Running Qwen 3.25b route-weighted tasks..."
    $PYTHON -m mxmoe.quant.quant eval \
        --model qwen2_moe --method rtn \
        --qconfig $QCONFIG/qwen2_moe_rtn_Slayer_bs512_wbits3.25_r1.0_prop1.json \
        --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard \
        --bs 64 \
        --save $RESULTS/routewtd_3.25b_tasks.json
fi

# 3. DeepSeek 3.25b baseline tasks
if [ ! -f "$RESULTS/deepseek_moe_baseline_3.25b_tasks.json" ]; then
    echo "[$(date)] Running DeepSeek 3.25b baseline tasks..."
    $PYTHON -m mxmoe.quant.quant eval \
        --model deepseek_moe --method rtn \
        --qconfig $QCONFIG/deepseek_moe_rtn_Slayer_bs512_wbits3.25_r1.0.json \
        --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard \
        --bs 64 \
        --save $RESULTS/deepseek_moe_baseline_3.25b_tasks.json
fi

# 4. DeepSeek 3.25b route-weighted tasks
if [ ! -f "$RESULTS/deepseek_moe_routewtd_3.25b_tasks.json" ]; then
    echo "[$(date)] Running DeepSeek 3.25b route-weighted tasks..."
    $PYTHON -m mxmoe.quant.quant eval \
        --model deepseek_moe --method rtn \
        --qconfig $QCONFIG/deepseek_moe_rtn_Slayer_bs512_wbits3.25_r1.0_prop1.json \
        --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard \
        --bs 64 \
        --save $RESULTS/deepseek_moe_routewtd_3.25b_tasks.json
fi

echo "[$(date)] All remaining experiments done!"
