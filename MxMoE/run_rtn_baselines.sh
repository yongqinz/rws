#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results

TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== Step 1: RTN Uniform Baselines ==="

for model in qwen2_moe deepseek_moe; do
  for wbits in 2 3 4; do
    qstr="w${wbits}a16_g-1_asym"

    # PPL
    ppl_save="${RESULTS}/${model}_rtn_w${wbits}_ppl.json"
    if [ ! -f "$ppl_save" ]; then
      echo "[$(date)] RTN ${model} W${wbits} PPL..."
      $PYTHON -m mxmoe.quant.quant eval \
        --model $model --method rtn --qstr $qstr \
        --tasks ppl \
        --save "$ppl_save"
    else
      echo "[$(date)] SKIP $ppl_save (exists)"
    fi

    # Tasks
    tasks_save="${RESULTS}/${model}_rtn_w${wbits}_tasks.json"
    if [ ! -f "$tasks_save" ]; then
      echo "[$(date)] RTN ${model} W${wbits} Tasks..."
      $PYTHON -m mxmoe.quant.quant eval \
        --model $model --method rtn --qstr $qstr \
        --tasks $TASKS --bs 64 \
        --save "$tasks_save"
    else
      echo "[$(date)] SKIP $tasks_save (exists)"
    fi
  done
done

echo "[$(date)] === RTN baselines DONE ==="
