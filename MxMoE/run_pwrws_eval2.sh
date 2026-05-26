#!/bin/bash
# PW-RWS tasks-only eval with smaller batch size
set -e
cd "$(dirname "$0")"

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
RESULTS_DIR="results/pwrws"
QCFG_DIR="qconfigs/global_rws"

# Check which PPL results exist and run only tasks
for MODEL in qwen2_moe deepseek_moe; do
  for BUDGET in 2.5 2.75 3.0; do
    QCFG="$QCFG_DIR/${MODEL}_rtn_pwrws_rws_wbits${BUDGET}.json"
    NAME="${MODEL}_pwrws_${BUDGET}b"
    TASKS_FILE="$RESULTS_DIR/${NAME}_tasks.json"

    if [ ! -f "$QCFG" ]; then continue; fi
    if [ -f "$TASKS_FILE" ]; then
      echo "[SKIP] $NAME tasks already exists"
      continue
    fi

    echo "============================================"
    echo "Evaluating tasks: $NAME"
    echo "============================================"

    # PPL if not done
    PPL_FILE="$RESULTS_DIR/${NAME}_ppl.json"
    if [ ! -f "$PPL_FILE" ]; then
      echo "[PPL] $NAME ..."
      $PYTHON -m mxmoe.quant.quant eval \
          --model "$MODEL" --method rtn --qconfig "$QCFG" \
          --tasks ppl --save "$PPL_FILE"
    fi

    echo "[Tasks] $NAME (bs=8) ..."
    $PYTHON -m mxmoe.quant.quant eval \
        --model "$MODEL" --method rtn --qconfig "$QCFG" \
        --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada \
        --bs 8 \
        --save "$TASKS_FILE"
  done
done
echo "All done!"
