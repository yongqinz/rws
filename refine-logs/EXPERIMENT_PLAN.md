# Experiment Plan: RouteQEP — Route-Conditioned Error Propagation for MoE Mixed-Precision

**Date**: 2026-04-17
**Based on**: FINAL_PROPOSAL.md + pilot diagnostic results

## Claim-Driven Experiment Blocks

### Claim 1: MoE quantization error propagates through routed expert paths, and this propagation is route-dependent (not just sequential)

**Block 1A: Extended Dependency Horizon Diagnostic**
- Run the pilot diagnostic on ALL MoE layers (not just 3) for Qwen1.5-MoE
- Perturb multiple experts (not just expert 0) to see if propagation depends on which expert is perturbed
- Measure route-conditioned error: does error follow the routing graph?
- Compute RHI (Route Heterogeneity Index) per layer
- Models: Qwen1.5-MoE, Mixtral-8x7B
- GPU: 1x A100, ~4 hours
- Success: RHI > 0 (route-dependent propagation exists); error extends beyond 1 layer

**Block 1B: Route-Conditional Propagation Validation**
- For expert pairs with matched local Δ but different P·a (propagated sensitivity), measure actual downstream error
- Dense QEP predicts similar damage; route-conditioned predicts the gap
- Models: Qwen1.5-MoE, Mixtral-8x7B
- GPU: 1x A100, ~2 hours
- Success: Actual downstream error correlates with route-conditioned prediction better than local Δ

### Claim 2: Route-conditioned propagation improves bit allocation over per-layer sensitivity

**Block 2A: Main Ablation Ladder (WikiText-2 PPL)**
| Config | Sensitivity | Bits | Expected |
|--------|------------|------|----------|
| MxMoE (baseline) | Local Δ | 3.25 | 7.02 |
| Dense QEP on MoE | Layer factor c_l | 3.25 | ~6.95 |
| RouteQEP 1-hop (expert) | P·a | 3.25 | ~6.85 |
| RouteQEP 2-hop (block) | Full Π | 3.25 | ~6.80 |
| MxMoE (baseline) | Local Δ | 2.25 | 8.79 |
| Dense QEP on MoE | Layer factor c_l | 2.25 | ~8.50 |
| RouteQEP 1-hop | P·a | 2.25 | ~8.20 |
| RouteQEP 2-hop | Full Π | 2.25 | ~8.00 |

- Models: Qwen1.5-MoE, Mixtral-8x7B (2 models)
- Bits: 3.25 (weight-only), 2.25 (weight-only), 5-5 (weight-activation)
- GPU: 1x A100, ~8 hours per model = 16 hours total
- Success: RouteQEP > Dense QEP > MxMoE at same bits

**Block 2B: Downstream Task Evaluation**
- Evaluate best config from 2A on: ARC-Challenge, HellaSwag, PIQA, WinoGrande, LAMBADA
- Compare against MxMoE baselines from the paper
- GPU: 1x A100, ~4 hours per model = 8 hours
- Success: Consistent improvement across tasks

### Claim 3: Route heterogeneity predicts the benefit of route-conditioned propagation

**Block 3A: RHI Correlation Study**
- Compute RHI for each MoE layer across all models
- Measure improvement (PPL reduction) of RouteQEP vs Dense QEP per layer
- Correlate RHI with improvement magnitude
- Compute cost: negligible (reuse data from 1A)
- Success: Positive correlation (higher RHI → larger gain)

### Claim 4: (Secondary) Robust allocation improves domain stability

**Block 4A: Multi-Domain Routing Trace Collection**
- Trace expert activations on: WikiText-2, HumanEval-X, GSM8K
- Compute per-domain expert frequency distributions
- GPU: 1x A100, ~3 hours

**Block 4B: Robust vs Average Allocation Comparison**
- Compare single-domain, pooled, and CVaR-robust ILP solutions
- Evaluate on held-out domain (leave-one-domain-out)
- GPU: 1x A100, ~4 hours
- Success: Robust allocation improves worst-domain PPL by >0.1 over pooled

### Claim 5: RouteQEP maintains real hardware performance

**Block 5A: End-to-End Latency/Throughput Benchmark**
- Use MxMoE's existing `run_mxmoe_gg.py` to benchmark RouteQEP allocations
- Compare against MxMoE baseline at same average bits
- GPU: 1x A100, ~2 hours
- Success: Comparable or better throughput (allocation-level change shouldn't hurt)

## Run Order

1. **Block 1A** (dependency horizon) → validates hypothesis, ~4h
2. **Block 2A** (main ablation) → core result, ~16h
3. **Block 3A** (RHI correlation) → analysis, negligible
4. **Block 2B** (downstream tasks) → strengthens paper, ~8h
5. **Block 4A+B** (robustness) → secondary, ~7h
6. **Block 5A** (latency benchmark) → systems credibility, ~2h
7. **Block 1B** (route-conditional validation) → deep validation, ~2h

Total: ~43 GPU-hours (within budget)

## Block Priority

- **Must-run**: 1A, 2A, 3A — these establish the core claim
- **Strongly recommended**: 2B, 5A — needed for paper completeness
- **Nice-to-have**: 4A, 4B, 1B — strengthens secondary claims

## First 3 Runs to Launch

1. **Block 1A**: Extended dependency horizon diagnostic on Qwen1.5-MoE (4h)
2. **Block 2A-1**: Main ablation ladder for Qwen1.5-MoE at 3.25-bit (4h)
3. **Block 2A-2**: Main ablation ladder for Mixtral-8x7B at 3.25-bit (4h)
