#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
OUT=out/gptq
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== Step 1: Generate 2.25b ILP qconfigs ==="
echo "NOTE: 2.25b budget = uniform w2a16_g128_asym (same as lowest strategy), ILP not needed."
echo "Using --qstr w2a16_g128_asym directly for 2.25b uniform eval."

echo "=== Step 2: GPTQ Pre-quantization (W3, W4 only, Qwen only) ==="
echo "DeepSeek GPTQ runs in parallel on GPU 1"
for model in qwen2_moe; do
  for wbits in 3 4; do
    # GPTQ without rotation
    outfile="${OUT}/${model}-w${wbits}_g-1_n128.pt"
    if [ ! -f "$outfile" ]; then
      echo "[$(date)] GPTQ ${model} W${wbits}..."
      $PYTHON -m mxmoe.quant.gptq $model wikitext2 \
        --wbits $wbits --groupsize -1 --nsamples 128 --save "$OUT"
    else
      echo "[$(date)] SKIP ${outfile}"
    fi

    # GPTQ-Had skipped (requires fast_hadamard_transform package)
    echo "[$(date)] SKIP GPTQ-Had (missing fast_hadamard_transform package)"
  done
done

echo "=== Step 3: RTN W2.25 SKIPPED (identical to MxMoE 2.25b) ==="

echo "=== Step 4: MxMoE 2.25b = uniform w2a16_g128_asym (skipped) ==="

echo "=== Step 5: GPTQ W3/W4 eval (Qwen only) ==="
echo "DeepSeek GPTQ eval runs in parallel on GPU 1"
for model in qwen2_moe; do
  for wbits in 3 4; do
    qweight="${OUT}/${model}-w${wbits}_g-1_n128.pt"
    if [ -f "$qweight" ]; then
      ppl_save="${RESULTS}/${model}_gptq_w${wbits}_ppl.json"
      if [ ! -f "$ppl_save" ]; then
        echo "[$(date)] GPTQ ${model} W${wbits} PPL..."
        $PYTHON -m mxmoe.quant.quant eval \
          --model $model --method gptq --qweight "$qweight" \
          --qstr "w${wbits}a16_g-1_asym" --tasks ppl --save "$ppl_save"
      fi

      tasks_save="${RESULTS}/${model}_gptq_w${wbits}_tasks.json"
      if [ ! -f "$tasks_save" ]; then
        echo "[$(date)] GPTQ ${model} W${wbits} Tasks..."
        $PYTHON -m mxmoe.quant.quant eval \
          --model $model --method gptq --qweight "$qweight" \
          --qstr "w${wbits}a16_g-1_asym" --tasks $TASKS --bs 64 --save "$tasks_save"
      fi
    fi
  done
done

echo "=== Step 6: GPTQ-Had SKIPPED (missing fast_hadamard_transform) ==="

echo "[$(date)] === ALL BASELINE EXPERIMENTS DONE ==="
