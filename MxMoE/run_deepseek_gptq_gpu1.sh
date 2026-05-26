#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

export CUDA_VISIBLE_DEVICES=1
PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
OUT=out/gptq
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== DeepSeek GPTQ on GPU 1 ==="

# Quantize W3
outfile="${OUT}/deepseek_moe-w3_g-1_n128.pt"
if [ ! -f "$outfile" ]; then
  echo "[$(date)] GPTQ DeepSeek W3..."
  $PYTHON -m mxmoe.quant.gptq deepseek_moe wikitext2 \
    --wbits 3 --groupsize -1 --nsamples 128 --save "$OUT"
fi

# Quantize W4
outfile="${OUT}/deepseek_moe-w4_g-1_n128.pt"
if [ ! -f "$outfile" ]; then
  echo "[$(date)] GPTQ DeepSeek W4..."
  $PYTHON -m mxmoe.quant.gptq deepseek_moe wikitext2 \
    --wbits 4 --groupsize -1 --nsamples 128 --save "$OUT"
fi

# Eval GPTQ W3
ppl_save="${RESULTS}/deepseek_moe_gptq_w3_ppl.json"
if [ ! -f "$ppl_save" ] && [ -f "${OUT}/deepseek_moe-w3_g-1_n128.pt" ]; then
  echo "[$(date)] GPTQ DeepSeek W3 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method gptq --qweight "${OUT}/deepseek_moe-w3_g-1_n128.pt" \
    --qstr w3a16_g-1_asym --tasks ppl --save "$ppl_save"
fi

tasks_save="${RESULTS}/deepseek_moe_gptq_w3_tasks.json"
if [ ! -f "$tasks_save" ] && [ -f "${OUT}/deepseek_moe-w3_g-1_n128.pt" ]; then
  echo "[$(date)] GPTQ DeepSeek W3 Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method gptq --qweight "${OUT}/deepseek_moe-w3_g-1_n128.pt" \
    --qstr w3a16_g-1_asym --tasks $TASKS --bs 64 --save "$tasks_save"
fi

# Eval GPTQ W4
ppl_save="${RESULTS}/deepseek_moe_gptq_w4_ppl.json"
if [ ! -f "$ppl_save" ] && [ -f "${OUT}/deepseek_moe-w4_g-1_n128.pt" ]; then
  echo "[$(date)] GPTQ DeepSeek W4 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method gptq --qweight "${OUT}/deepseek_moe-w4_g-1_n128.pt" \
    --qstr w4a16_g-1_asym --tasks ppl --save "$ppl_save"
fi

tasks_save="${RESULTS}/deepseek_moe_gptq_w4_tasks.json"
if [ ! -f "$tasks_save" ] && [ -f "${OUT}/deepseek_moe-w4_g-1_n128.pt" ]; then
  echo "[$(date)] GPTQ DeepSeek W4 Tasks..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model deepseek_moe --method gptq --qweight "${OUT}/deepseek_moe-w4_g-1_n128.pt" \
    --qstr w4a16_g-1_asym --tasks $TASKS --bs 64 --save "$tasks_save"
fi

echo "[$(date)] === DeepSeek GPTQ DONE ==="
