#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== Mixtral-8x7B Experiments ==="

# Wait for both GPUs to have < 30GB used
echo "[$(date)] Waiting for GPUs to be free..."
while true; do
  used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print int($1)}')
  used1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | awk '{print int($1)}')
  if [ "$used0" -lt 30000 ] && [ "$used1" -lt 30000 ]; then
    echo "[$(date)] Both GPUs free! GPU0=${used0}MiB, GPU1=${used1}MiB"
    break
  fi
  echo "[$(date)] Waiting... GPU0=${used0}MiB, GPU1=${used1}MiB"
  sleep 60
done

# FP16 baseline PPL
ppl_save="${RESULTS}/mixtral_fp16_ppl.json"
if [ ! -f "$ppl_save" ] || grep -q "NaN" "$ppl_save"; then
  echo "[$(date)] Mixtral FP16 PPL..."
  $PYTHON -m mxmoe.quant.quant eval \
    --model mixtral --method rtn --tasks ppl \
    --save "$ppl_save"
fi

# Quantized PPL + downstream for each budget
for budget in 2.5 2.75 3.0 3.25; do
  # Baseline PPL
  ppl_save="${RESULTS}/mixtral_baseline_${budget}b_ppl.json"
  if [ ! -f "$ppl_save" ]; then
    echo "[$(date)] Mixtral baseline ${budget}b PPL..."
    $PYTHON -m mxmoe.quant.quant eval \
      --model mixtral --method rtn \
      --qconfig "${QCONFIGS}/mixtral_rtn_Slayer_bs512_wbits${budget}_r1.0.json" \
      --tasks ppl --save "$ppl_save"
  fi

  # Route-weighted PPL
  ppl_save="${RESULTS}/mixtral_routewtd_${budget}b_ppl.json"
  if [ ! -f "$ppl_save" ]; then
    echo "[$(date)] Mixtral route-weighted ${budget}b PPL..."
    $PYTHON -m mxmoe.quant.quant eval \
      --model mixtral --method rtn \
      --qconfig "${QCONFIGS}/mixtral_rtn_Slayer_bs512_wbits${budget}_r1.0_prop1.json" \
      --tasks ppl --save "$ppl_save"
  fi
done

# Downstream tasks for 2.5b and 2.75b budgets
for budget in 2.5 2.75; do
  # Baseline tasks
  tasks_save="${RESULTS}/mixtral_baseline_${budget}b_tasks.json"
  if [ ! -f "$tasks_save" ]; then
    echo "[$(date)] Mixtral baseline ${budget}b Tasks..."
    $PYTHON -m mxmoe.quant.quant eval \
      --model mixtral --method rtn \
      --qconfig "${QCONFIGS}/mixtral_rtn_Slayer_bs512_wbits${budget}_r1.0.json" \
      --tasks $TASKS --bs 64 --save "$tasks_save"
  fi

  # Route-weighted tasks
  tasks_save="${RESULTS}/mixtral_routewtd_${budget}b_tasks.json"
  if [ ! -f "$tasks_save" ]; then
    echo "[$(date)] Mixtral route-weighted ${budget}b Tasks..."
    $PYTHON -m mxmoe.quant.quant eval \
      --model mixtral --method rtn \
      --qconfig "${QCONFIGS}/mixtral_rtn_Slayer_bs512_wbits${budget}_r1.0_prop1.json" \
      --tasks $TASKS --bs 64 --save "$tasks_save"
  fi
done

echo "[$(date)] === Mixtral experiments DONE ==="
