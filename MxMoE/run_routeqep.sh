#!/bin/bash
# RouteQEP: Full experiment pipeline
# Stage 1: Trace routing + co-routing mass
# Stage 2: Calibrate quantization loss
# Stage 3: Solve ILP with propagated delta
# Stage 4: Evaluate accuracy
#
# Usage: bash run_routeqep.sh [model] [prop_hop] [bits]
# Example: bash run_routeqep.sh qwen2_moe 1 3.25

set -e
cd "$(dirname "$0")"

MODEL=${1:-qwen2_moe}
PROP_HOP=${2:-1}
WBITS=${3:-3.25}
QTYPE=${4:-rtn}
BATCH=${5:-512}
R=${6:-1.0}
SEED=42
NSAMPLES=128
SEQLEN=4096
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================"
echo "RouteQEP Experiment Pipeline"
echo "============================================"
echo "Model: $MODEL"
echo "Propagation hops: $PROP_HOP"
echo "Weight bits: $WBITS"
echo "Quant type: $QTYPE"
echo "Batch: $BATCH"
echo "r: $R"
echo "GPU: $GPU"
echo "============================================"

CALIB_DIR="calib"
GATE_DIR="$CALIB_DIR/gate/$MODEL/wiki2/$SEQLEN"
LOSS_DIR="$CALIB_DIR/$MODEL"

mkdir -p "$GATE_DIR" "$LOSS_DIR" qconfigs

# ============================================================
# Step 1: Trace MoE Gate (expert activation frequencies)
# ============================================================
if [ ! -f "$GATE_DIR/moe-gate.json" ]; then
    echo "[Step 1a] Tracing MoE gate for $MODEL..."
    python3 -m mxmoe.quant.moe_tracer \
        --model "$MODEL" \
        --trace_gate \
        --dataset wiki2 \
        --seqlen "$SEQLEN" \
        --nsamples "$NSAMPLES" \
        --seed "$SEED"
    echo "[Step 1a] Gate trace done."
else
    echo "[Step 1a] Gate trace already exists, skipping."
fi

# ============================================================
# Step 1b: Trace co-routing mass for RouteQEP
# ============================================================
if [ ! -f "$GATE_DIR/corouting.json" ]; then
    echo "[Step 1b] Tracing co-routing mass for RouteQEP..."
    python3 -m mxmoe.quant.moe_tracer \
        --model "$MODEL" \
        --trace_corouting \
        --dataset wiki2 \
        --seqlen "$SEQLEN" \
        --nsamples "$NSAMPLES" \
        --seed "$SEED"
    echo "[Step 1b] Co-routing trace done."
else
    echo "[Step 1b] Co-routing trace already exists, skipping."
fi

# ============================================================
# Step 2: Calibrate quantization loss per block per strategy
# ============================================================
STRATEGIES=("w4a16_g-1_asym" "w8a8_g-1_sym")
for QCFG in "${STRATEGIES[@]}"; do
    LOSS_FILE="$LOSS_DIR/${MODEL}-MOE-${QTYPE^^}-${QCFG}-wiki2-${NSAMPLES}-${SEQLEN}-layer_out_norm.json"
    if [ ! -f "$LOSS_FILE" ]; then
        echo "[Step 2] Calibrating loss for $QCFG..."
        python3 -m mxmoe.quant.quant calib \
            --model "$MODEL" \
            --method "$QTYPE" \
            --metric layer_out_norm \
            --qcfg "$QCFG" \
            --nsamples "$NSAMPLES" \
            --seqlen "$SEQLEN" \
            --seed "$SEED"
    else
        echo "[Step 2] Loss for $QCFG already exists, skipping."
    fi
done

# ============================================================
# Step 3: Solve ILP with propagated delta
# ============================================================
echo "[Step 3] Solving ILP with prop_hop=$PROP_HOP..."
python3 -m mxmoe.quant.bits_solver \
    --model "$MODEL" \
    --qtype "$QTYPE" \
    --solve_mode layer \
    --wbits "$WBITS" \
    --batch "$BATCH" \
    --r "$R" \
    --prop_hop "$PROP_HOP" \
    --prop_lam 1.0 \
    --trace_file "$GATE_DIR/moe-gate.json"
echo "[Step 3] ILP solved. QConfig saved."

# Also solve without propagation for baseline comparison
echo "[Step 3b] Solving baseline ILP (prop_hop=0)..."
python3 -m mxmoe.quant.bits_solver \
    --model "$MODEL" \
    --qtype "$QTYPE" \
    --solve_mode layer \
    --wbits "$WBITS" \
    --batch "$BATCH" \
    --r "$R" \
    --prop_hop 0 \
    --trace_file "$GATE_DIR/moe-gate.json"
echo "[Step 3b] Baseline ILP solved."

echo "============================================"
echo "Pipeline complete. Check qconfigs/ for results."
echo "============================================"
