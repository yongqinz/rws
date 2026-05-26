#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
OUT=out/gptq
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== GPU 0: Qwen GPTQ eval + DeepSeek 3.0b tasks ==="

# 1. Qwen GPTQ W3 PPL
ppl_save="${RESULTS}/qwen2_moe_gptq_w3_ppl.json"
qweight="${OUT}/qwen2_moe-w3_g-1_n128.pt"
if [ ! -f "$ppl_save" ] && [ -f "$qweight" ]; then
  echo "[$(date)] Qwen GPTQ W3 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method gptq --qweight "$qweight" \
    --qstr w3a16_g-1_asym --tasks ppl --save "$ppl_save"
fi

# 2. Qwen GPTQ W3 Tasks
tasks_save="${RESULTS}/qwen2_moe_gptq_w3_tasks.json"
if [ ! -f "$tasks_save" ] && [ -f "$qweight" ]; then
  echo "[$(date)] Qwen GPTQ W3 Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method gptq --qweight "$qweight" \
    --qstr w3a16_g-1_asym --tasks $TASKS --bs 64 --save "$tasks_save"
fi

# 3. Qwen GPTQ W4 PPL
ppl_save="${RESULTS}/qwen2_moe_gptq_w4_ppl.json"
qweight="${OUT}/qwen2_moe-w4_g-1_n128.pt"
if [ ! -f "$ppl_save" ] && [ -f "$qweight" ]; then
  echo "[$(date)] Qwen GPTQ W4 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method gptq --qweight "$qweight" \
    --qstr w4a16_g-1_asym --tasks ppl --save "$ppl_save"
fi

# 4. Qwen GPTQ W4 Tasks
tasks_save="${RESULTS}/qwen2_moe_gptq_w4_tasks.json"
if [ ! -f "$tasks_save" ] && [ -f "$qweight" ]; then
  echo "[$(date)] Qwen GPTQ W4 Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method gptq --qweight "$qweight" \
    --qstr w4a16_g-1_asym --tasks $TASKS --bs 64 --save "$tasks_save"
fi

# 5. DeepSeek baseline 3.0b tasks
tasks_save="${RESULTS}/deepseek_moe_baseline_3.0b_tasks.json"
if [ ! -f "$tasks_save" ]; then
  echo "[$(date)] DeepSeek baseline 3.0b Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method rtn \
    --qconfig "${QCONFIGS}/deepseek_moe_rtn_Slayer_bs512_wbits3.0_r1.0.json" \
    --tasks $TASKS --bs 64 --save "$tasks_save"
fi

# 6. DeepSeek route-weighted 3.0b tasks
tasks_save="${RESULTS}/deepseek_moe_routewtd_3.0b_tasks.json"
if [ ! -f "$tasks_save" ]; then
  echo "[$(date)] DeepSeek route-weighted 3.0b Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method rtn \
    --qconfig "${QCONFIGS}/deepseek_moe_rtn_Slayer_bs512_wbits3.0_r1.0_prop1.json" \
    --tasks $TASKS --bs 64 --save "$tasks_save"
fi

echo "[$(date)] === GPU 0 GPTQ eval + DeepSeek 3.0b DONE ==="
