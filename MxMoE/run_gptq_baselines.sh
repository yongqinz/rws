#!/bin/bash
set -e
cd /data/zengyq/paper/baselines/MxMoE

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS=results
OUT=out/gptq
TASKS="piqa hellaswag arc_easy arc_challenge winogrande lambada_openai lambada_standard"

echo "=== Step 2: GPTQ Pre-quantization ==="

for model in qwen2_moe deepseek_moe; do
  for wbits in 2 3 4; do
    # GPTQ without rotation
    outfile="${OUT}/${model}-w${wbits}_g-1_n128.pt"
    if [ ! -f "$outfile" ]; then
      echo "[$(date)] GPTQ ${model} W${wbits}..."
      $PYTHON -m mxmoe.quant.gptq $model wikitext2 \
        --wbits $wbits --groupsize -1 \
        --nsamples 128 --save "$OUT"
    else
      echo "[$(date)] SKIP GPTQ ${outfile} (exists)"
    fi

    # GPTQ with Hadamard rotation
    outfile_had="${OUT}/${model}-w${wbits}_g-1_had_n128.pt"
    if [ ! -f "$outfile_had" ]; then
      echo "[$(date)] GPTQ-Had ${model} W${wbits}..."
      $PYTHON -m mxmoe.quant.gptq $model wikitext2 \
        --wbits $wbits --groupsize -1 \
        --nsamples 128 --rotation --save "$OUT"
    else
      echo "[$(date)] SKIP GPTQ-Had ${outfile_had} (exists)"
    fi
  done
done

echo "[$(date)] === GPTQ Pre-quantization DONE ==="

echo "=== Step 3: GPTQ Eval ==="

for model in qwen2_moe deepseek_moe; do
  for wbits in 2 3 4; do
    qweight="${OUT}/${model}-w${wbits}_g-1_n128.pt"

    # PPL
    ppl_save="${RESULTS}/${model}_gptq_w${wbits}_ppl.json"
    if [ ! -f "$ppl_save" ] && [ -f "$qweight" ]; then
      echo "[$(date)] GPTQ ${model} W${wbits} PPL..."
      $PYTHON -m mxmoe.quant.quant eval \
        --model $model --method gptq --qweight "$qweight" \
        --qstr "w${wbits}a16_g-1_asym" \
        --tasks ppl \
        --save "$ppl_save"
    else
      echo "[$(date)] SKIP $ppl_save"
    fi

    # Tasks
    tasks_save="${RESULTS}/${model}_gptq_w${wbits}_tasks.json"
    if [ ! -f "$tasks_save" ] && [ -f "$qweight" ]; then
      echo "[$(date)] GPTQ ${model} W${wbits} Tasks..."
      $PYTHON -m mxmoe.quant.quant eval \
        --model $model --method gptq --qweight "$qweight" \
        --qstr "w${wbits}a16_g-1_asym" \
        --tasks $TASKS --bs 64 \
        --save "$tasks_save"
    else
      echo "[$(date)] SKIP $tasks_save"
    fi
  done
done

echo "[$(date)] === GPTQ Eval DONE ==="

echo "=== Step 4: GPTQ-Had Eval ==="

for model in qwen2_moe deepseek_moe; do
  for wbits in 2 3 4; do
    qweight_had="${OUT}/${model}-w${wbits}_g-1_had_n128.pt"

    # PPL
    ppl_save="${RESULTS}/${model}_gptqhad_w${wbits}_ppl.json"
    if [ ! -f "$ppl_save" ] && [ -f "$qweight_had" ]; then
      echo "[$(date)] GPTQ-Had ${model} W${wbits} PPL..."
      $PYTHON -m mxmoe.quant.quant eval \
        --model $model --method gptq-had --qweight "$qweight_had" \
        --qstr "w${wbits}a16_g-1_asym" \
        --tasks ppl \
        --save "$ppl_save"
    else
      echo "[$(date)] SKIP $ppl_save"
    fi

    # Tasks
    tasks_save="${RESULTS}/${model}_gptqhad_w${wbits}_tasks.json"
    if [ ! -f "$tasks_save" ] && [ -f "$qweight_had" ]; then
      echo "[$(date)] GPTQ-Had ${model} W${wbits} Tasks..."
      $PYTHON -m mxmoe.quant.quant eval \
        --model $model --method gptq-had --qweight "$qweight_had" \
        --qstr "w${wbits}a16_g-1_asym" \
        --tasks $TASKS --bs 64 \
        --save "$tasks_save"
    else
      echo "[$(date)] SKIP $tasks_save"
    fi
  done
done

echo "[$(date)] === ALL GPTQ baselines DONE ==="
