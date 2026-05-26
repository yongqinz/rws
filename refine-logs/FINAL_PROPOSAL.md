# Final Proposal: RouteQEP — Route-Conditioned Error Propagation for Mixed-Precision MoE Quantization

**Problem Anchor**: MxMoE's per-layer sensitivity estimation ignores cross-layer quantization error propagation through routed expert paths, causing suboptimal bit allocation, especially on models with strong inter-layer dependencies.

## Method Thesis

MoE quantization error follows **conditional expert routes**, not sequential layers. Per-block local loss and dense layer-wise propagation both miss the true dependency structure. We introduce a **Route-Conditioned Error Propagation** operator that models how quantization error propagates through the routing network, and use it to improve mixed-precision bit allocation in MxMoE's ILP framework.

## Core Technical Contribution

### 1. Expert Transition Matrix

During calibration tracing (extends `moe_tracer.py`), collect the adjacent-layer co-routing mass:

```
C_{i,i'}^{(l)} = Σ_t ρ_{t,i}^{(l)} · ρ_{t,i'}^{(l+1)}
```

Normalize to get the conditional expert transition matrix:

```
P_{i,i'}^{(l)} := Pr(e_{l+1}=i' | e_l=i) ≈ C_{i,i'}^{(l)} / Σ_{i'} C_{i,i'}^{(l)}
```

Cost: O(TLk²) — negligible overhead on top of existing tracing.

### 2. Route-Conditioned Propagation

Define downstream block gain:
```
a_{i',j'}^{(l+1)} ≈ Δ_{l+1,i',j'}^{probe} / (π_{i'}^{(l+1)} + ε)
```

Block-to-block propagation operator:
```
Π_{(i,j)→(i',j')}^{(l)} = P_{i,i'}^{(l)} · a_{i',j'}^{(l+1)}
```

Multi-hop propagation (H hops):
```
g^{(L)} = 1,   g^{(l)} = 1 + λ·M^{(l)}·g^{(l+1)}
```

Propagated block cost:
```
Δ̂_{l,i,j,k} = g^{(l)}_{(i,j)} · Δ_{l,i,j,k}
```

### 3. ILP Update

Replace MxMoE's local Δ with propagated Δ̂ in the objective:

```
L_RA = Σ_{l,i,j,k} x_{l,i,j,k} · Δ̂_{l,i,j,k}
```

Keep the same latency term T, budget constraints, and log-objective form. **No change to the ILP structure** — still linear, still solvable by Gurobi.

### 4. Why NOT Dense QEP

Dense QEP uses a layer-only factor c_l, so every block in layer l gets c_l · Δ. This is equivalent to the rank-1 approximation:

```
P^{(l)} ≈ 1 · π^{(l+1)T}    (dense QEP)
```

Our method keeps the **full conditional routing operator** P^{(l)}. These are equal only when all source experts have identical downstream propagation patterns — which is degenerate for MoE models with heterogeneous routing.

**Route Heterogeneity Index (RHI)**: RHI_l = ||P^{(l)} - 1·π^{(l+1)T}||_F

We show that higher RHI correlates with larger gains over dense QEP.

## Dominant Contribution

**Route-Conditioned Error Propagation** — a MoE-specific error propagation model that uses the conditional expert transition matrix to estimate how quantization error propagates through routed paths. Dense QEP is a special case (rank-1 approximation).

## Evaluation Plan

### Ablation Ladder (must-run)
| Config | Sensitivity | Propagation |
|--------|------------|-------------|
| MxMoE (baseline) | Local Δ | None |
| Dense QEP on MoE | Layer factor c_l | Sequential |
| RouteQEP (1-hop, expert-level) | P_{i,i'} · a_{i'} | 1-hop routed |
| RouteQEP (2-hop, block-level) | Full Π matrix | 2-hop routed |

### Key Diagnostics
1. **Dependency horizon**: How many hops before propagated error becomes noise?
2. **Route Heterogeneity Index (RHI)**: Correlation between RHI and gain over dense QEP
3. **Route agreement**: Measure router drift after quantization (address EAC-MoE concern)

### Models
- Qwen1.5-MoE-A2.7B (where MxMoE shows cross-layer issues)
- Mixtral-8x7B-v0.1
- DeepSeek-MoE-16B (if available)

### Metrics
- WikiText-2 PPL, downstream tasks (ARC, HellaSwag, PIQA, WinoGrande)
- Average bits matched to MxMoE for fair comparison
- End-to-end throughput/latency using MxMoE's kernel infrastructure

### Robustness (secondary)
- Single-domain calibration vs pooled traces vs CVaR robust optimization
- Leave-one-domain-out on code (HumanEval), math (GSM8K), general (WikiText)

## Implementation Plan

### Files to modify (in `/data/zengyq/paper/baselines/MxMoE`)

1. **`mxmoe/quant/moe_tracer.py`**: Add co-routing mass collection (C_{i,i'})
   - Extend trace output to include adjacent-layer expert pair counts
   - ~50 lines of new code

2. **`mxmoe/quant/bits_solver.py`**: Replace Δ with Δ̂ in ILP objective
   - Add propagation computation (P matrix, gain vector, g recursion)
   - Add RHI diagnostic output
   - ~100 lines of new code

3. **`mxmoe/quant/quant.py`**: Update calibration to support propagated sensitivity
   - Add `--prop-hop` argument (0=local, 1=1-hop, 2=2-hop)
   - ~30 lines of new code

4. **New file: `mxmoe/quant/propagation.py`**: Core propagation logic
   - Expert transition matrix computation
   - Multi-hop propagation operator
   - RHI computation
   - ~150 lines

### No changes needed:
- `mxmoe/kernels/` — kernel generation unchanged (allocation-level improvement only)
- `mxmoe/quant/gptq.py` — quantization algorithm unchanged

### Compute budget
- Calibration + tracing: ~2-4 hours per model
- ILP solving: minutes per config
- Evaluation: ~4-6 hours per model per config
- Total: ~50-80 GPU-hours (A100)

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Propagated Δ̂ highly correlated with local Δ | 30% | High (method doesn't help) | Diagnostic pilot validates first |
| RHI is small for all MoE models | 20% | High (no route structure) | Show RHI on diverse MoE architectures |
| Router drift dominates error (EAC-MoE concern) | 15% | Medium | Measure and report route agreement |
| 1-hop already captures most benefit | 40% | Low | Still publishable with ablation |
| Dense QEP adaptation gets similar results | 25% | Critical | Key ablation: show route-conditioning matters via RHI correlation |

## Venue Target

- **Primary**: ICML 2026 or NeurIPS 2026
- **Backup**: MLSys 2026 (if systems contribution stronger)
- **Timeline**: 4-6 weeks for full experiments + paper
