#!/bin/bash
# Run calibration-seed sensitivity for Qwen 2.5b
# For each seed: calibration → ILP → PPL evaluation
set -e
cd /data/zengyq/paper/baselines/MxMoE

PYTHON=/data/zengyq/miniconda3/envs/arpo/bin/python
QCONFIGS=qconfigs/w2a16_g128_asym+w4a16_g-1_asym
FILTER="w2a16_g128_asym w4a16_g-1_asym"
GATE=calib/gate/qwen2_moe/wiki2/4096/moe-gate.json
RESULTS=results

SEEDS=${1:-"0 1 2"}

# Standard calib paths (what the quant module outputs)
STD_W2="calib/qwen2_moe-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json"
STD_W4="calib/qwen2_moe-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json"

# Back up originals
BACKUP_DIR="calib/seed_backup"
mkdir -p "$BACKUP_DIR"
if [ ! -f "$BACKUP_DIR/orig_W2.json" ]; then
  cp "$STD_W2" "$BACKUP_DIR/orig_W2.json"
  cp "$STD_W4" "$BACKUP_DIR/orig_W4.json"
  echo "Backed up original calib files"
fi

for SEED in $SEEDS; do
  echo "============================================"
  echo "[$(date)] SEED $SEED — Start"
  echo "============================================"

  SEED_W2="calib/qwen2_moe_seed${SEED}-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json"
  SEED_W4="calib/qwen2_moe_seed${SEED}-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json"

  # Step 1: Calibration — for each strategy, run with seed then rename output
  for QCFG in w2a16_g128_asym w4a16_g-1_asym; do
    # qcfg key (lowercase) → output filename (uppercase first letter pair)
    case "$QCFG" in
      w2a16_g128_asym) QCFG_FILE="W2A16_g128_asym" ;;
      w4a16_g-1_asym)  QCFG_FILE="W4A16_g-1_asym" ;;
    esac
    SEED_OUT="calib/qwen2_moe_seed${SEED}-MOE-rtn-${QCFG_FILE}-wiki2-128-4096-layer_out_norm.json"
    if [ -f "$SEED_OUT" ] && [ $(python3 -c "import json; print(len(json.load(open('$SEED_OUT'))))" 2>/dev/null || echo 0) -ge 24 ]; then
      echo "[$(date)] Seed $SEED $QCFG calib already done, skipping"
    else
      echo "[$(date)] Seed $SEED calibrating $QCFG..."
      CUDA_VISIBLE_DEVICES=0 $PYTHON -m mxmoe.quant.quant calib \
        --model qwen2_moe --method rtn --metric layer_out_norm \
        --qcfg "$QCFG" --nsamples 128 --seed $SEED
      # quant module saves to standard path — rename to seed-specific
      if [ "$QCFG" = "w2a16_g128_asym" ]; then
        cp "$STD_W2" "$SEED_OUT"
      else
        cp "$STD_W4" "$SEED_OUT"
      fi
      echo "  Saved to $SEED_OUT"
    fi
  done

  # Step 2: ILP solve (baseline + RWS) using seed-specific calibration
  echo "[$(date)] Seed $SEED: setting up ILP..."
  # Replace standard files with seed-specific calib
  cp "$SEED_W2" "$STD_W2"
  cp "$SEED_W4" "$STD_W4"

  for PROP in 0 1; do
    TAG=""
    [ "$PROP" = "1" ] && TAG="_prop1"
    SAVE="${QCONFIGS}/qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0_seed${SEED}${TAG}.json"
    if [ -f "$SAVE" ]; then
      echo "[$(date)] Seed $SEED prop=$PROP ILP already done, skipping"
    else
      echo "[$(date)] Seed $SEED solving ILP prop_hop=$PROP..."
      $PYTHON -m mxmoe.quant.bits_solver \
        --model qwen2_moe --qtype rtn --solve_mode layer \
        --wbits 2.5 --batch 512 --r 1.0 --prop_hop $PROP \
        --filter_list $FILTER \
        --trace_file "$GATE" 2>&1 | tail -3
      # Rename the output to seed-specific name
      ORIG_QCFG="${QCONFIGS}/qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0${TAG}.json"
      if [ -f "$ORIG_QCFG" ] && [ ! -f "$SAVE" ]; then
        cp "$ORIG_QCFG" "$SAVE"
        echo "  Saved to $SAVE"
      fi
    fi
  done

  # Restore original calib files
  cp "$BACKUP_DIR/orig_W2.json" "$STD_W2"
  cp "$BACKUP_DIR/orig_W4.json" "$STD_W4"

  # Step 3: PPL evaluation with seed-specific allocations
  for PROP in 0 1; do
    TAG=""
    METHOD="baseline"
    [ "$PROP" = "1" ] && TAG="_prop1" && METHOD="routewtd"
    QCFG="${QCONFIGS}/qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0_seed${SEED}${TAG}.json"
    SAVE="${RESULTS}/qwen2_moe_${METHOD}_2.5b_seed${SEED}_ppl.json"
    if [ -f "$SAVE" ]; then
      echo "[$(date)] Seed $SEED $METHOD PPL already done"
    else
      echo "[$(date)] Seed $SEED evaluating $METHOD PPL..."
      CUDA_VISIBLE_DEVICES=0 $PYTHON -m mxmoe.quant.quant eval \
        --model qwen2_moe --method rtn \
        --qconfig "$QCFG" --tasks ppl --save "$SAVE"
    fi
  done

  echo "[$(date)] SEED $SEED — Done"
done

# Restore originals one final time
cp "$BACKUP_DIR/orig_W2.json" "$STD_W2"
cp "$BACKUP_DIR/orig_W4.json" "$STD_W4"

echo "============================================"
echo "[$(date)] ALL SEEDS COMPLETE"
echo "============================================"
