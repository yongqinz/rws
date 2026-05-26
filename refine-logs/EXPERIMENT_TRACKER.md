# Experiment Tracker

## Pipeline Progress

### Pre-requisites (Calibration)
- [x] Gate tracing (moe-gate.json) — 74KB, complete
- [x] Co-routing tracing (corouting.json) — 3.5MB, RHI mean=1.84
- [ ] W4A16 calibration — **IN PROGRESS** (6/24 layers, ~3h remaining on GPU 0)
- [ ] W8A8 calibration — **QUEUED** (will start after W4A16 completes)
- [x] ILP solver integration verified (both prop_hop=0 and prop_hop=1 work)
- [x] Propagation module verified (gain amplification 17-28x across layers)

### Step 3: ILP Solving (~5 min)
- [ ] Baseline (prop_hop=0) with W4A16 + W8A8 strategies
- [ ] RouteQEP (prop_hop=1) with W4A16 + W8A8 strategies

### Step 4: Evaluation (~2h per config)
- [ ] Baseline: PPL on WikiText2
- [ ] RouteQEP: PPL on WikiText2
- [ ] Baseline: Downstream tasks (PIQA, HellaSwag, ARC, Winogrande, Lambada)
- [ ] RouteQEP: Downstream tasks

## Key Findings So Far
- Route Heterogeneity Index (RHI) for Qwen1.5-MoE: mean=1.84, max=2.12
- Propagation gain varies by layer: 17-28x amplification across 4 layers
- Deeper layers get more amplification (28x at layer 3 vs 17x at layer 0)

## Estimated Timeline
| Task | Duration | Status |
|------|----------|--------|
| W4A16 calibration | ~3h more | IN PROGRESS |
| W8A8 calibration | ~3h | QUEUED (after W4A16) |
| ILP solving | ~5 min | READY (waiting for data) |
| Evaluation (PPL + tasks) | ~4h | READY (waiting for ILP) |
| **Total remaining** | **~10h** | |

## Files Ready
- `run_calib_sequential.sh` — auto-runs W8A8 after W4A16
- `run_eval.sh` — evaluates baseline vs RouteQEP
- `calib/calib_sequential.log` — monitoring log
