#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
OUT=out/gptq
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== GPU 0: DeepSeek GPTQ W4 + Missing Qwen experiments ==="

# 1. GPTQ DeepSeek W4 quantize
outfile="${OUT}/deepseek_moe-w4_g-1_n128.pt"
if [ ! -f "$outfile" ]; then
  echo "[$(date)] GPTQ DeepSeek W4 quantize..."
  $PYTHON -m mxmoe.quant.gptq deepseek_moe wikitext2 \
    --wbits 4 --groupsize -1 --nsamples 128 --save "$OUT"
else
  echo "[$(date)] SKIP ${outfile}"
fi

# 2. GPTQ DeepSeek W4 eval (PPL + tasks)
ppl_save="${RESULTS}/deepseek_moe_gptq_w4_ppl.json"
if [ ! -f "$ppl_save" ] && [ -f "$outfile" ]; then
  echo "[$(date)] GPTQ DeepSeek W4 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method gptq --qweight "$outfile" \
    --qstr w4a16_g-1_asym --tasks ppl --save "$ppl_save"
fi

tasks_save="${RESULTS}/deepseek_moe_gptq_w4_tasks.json"
if [ ! -f "$tasks_save" ] && [ -f "$outfile" ]; then
  echo "[$(date)] GPTQ DeepSeek W4 Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method gptq --qweight "$outfile" \
    --qstr w4a16_g-1_asym --tasks $TASKS --bs 64 --save "$tasks_save"
fi

# 3. Qwen FP16 PPL
ppl_save="${RESULTS}/qwen2_moe_fp16_ppl.json"
if [ ! -f "$ppl_save" ]; then
  echo "[$(date)] Qwen FP16 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method rtn --tasks ppl \
    --save "$ppl_save"
else
  echo "[$(date)] SKIP Qwen FP16 PPL"
fi

# 4. Qwen baseline 3.0b tasks
tasks_save="${RESULTS}/baseline_3.0b_tasks.json"
if [ ! -f "$tasks_save" ]; then
  echo "[$(date)] Qwen baseline 3.0b Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method rtn \
    --qconfig "${QCONFIGS}/qwen2_moe_rtn_Slayer_bs512_wbits3.0_r1.0.json" \
    --tasks $TASKS --bs 64 --save "$tasks_save"
else
  echo "[$(date)] SKIP Qwen baseline 3.0b tasks"
fi

# 5. Qwen route-weighted 3.0b tasks
tasks_save="${RESULTS}/routewtd_3.0b_tasks.json"
if [ ! -f "$tasks_save" ]; then
  echo "[$(date)] Qwen route-weighted 3.0b Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model qwen2_moe --method rtn \
    --qconfig "${QCONFIGS}/qwen2_moe_rtn_Slayer_bs512_wbits3.0_r1.0_prop1.json" \
    --tasks $TASKS --bs 64 --save "$tasks_save"
else
  echo "[$(date)] SKIP Qwen route-weighted 3.0b tasks"
fi

echo "[$(date)] === GPU 0 fill experiments DONE ==="
