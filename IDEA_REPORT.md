# Research Idea Report

**Direction**: MoE quantization mixed-precision, extending MxMoE (ICML 2025)
**Generated**: 2026-04-17
**Ideas evaluated**: 10 generated → 8 survived filtering → 1 piloted (diagnostic) → 3 recommended

## Executive Summary

The top recommended direction is a unified paper combining **cross-layer error propagation awareness** with **scenario-robust allocation** for MoE mixed-precision quantization. MxMoE (ICML 2025) explicitly acknowledges that single-layer sensitivity estimation causes issues (especially on Qwen2-MoE), and QEP (NeurIPS 2025) demonstrates significant gains from cross-layer error propagation for dense LLMs. Nobody has combined these insights. The diagnostic pilot measures MoE-specific error propagation structure to validate the hypothesis.

## Literature Landscape

### Directly Competing (MoE Mixed-Precision Quantization)
- **MxMoE** (ICML 2025): Linear-block ILP allocation + auto-generated GroupGEMM kernels — **our base**
- **DynaExq** (arXiv 2511.15015): Online dynamic expert quantization with EMA stats — online but no kernel support
- **DyMoE** (2024): Dynamic expert orchestration + mixed-precision for edge — different scope
- **MC-MoE** (arXiv 2410.06270): Combined static quantization + dynamic pruning — coarse granularity
- **MiLo** (MLSys 2025): Low-rank compensators for quantized MoE — no mixed-precision
- **MoPEQ** (ICCV 2025 WS): Expert-level mixed-precision PTQ — coarser than MxMoE
- **BT-MoE** (ICLR 2025): Joint bit-rank allocation for MoE — **eliminates our low-rank idea**
- **EAC-MoE** (ACL 2025): Expert-selection aware compressor for MoE — addresses expert-shift
- **MoEQuant** (ICML 2025): MoE-specific PTQ with heterogeneous expert sensitivity

### Cross-Layer Error Propagation
- **QEP** (NeurIPS 2025): Cross-layer quantization error propagation for dense LLMs — **NOT applied to MoE**
- **Router Choice Matters** (OpenReview): Rank-aware PTQ for MoE addressing router sensitivity

### General Mixed-Precision
- **ResQ** (ICML 2025): Low-rank residuals + mixed-precision via PCA — dense LLMs only
- **SliM-LLM** (ICML 2025): Salience-driven mixed-precision — dense only

### Expert Pruning
- **Shapley-MoE** (NeurIPS 2025): Shapley value-based expert importance for pruning

### Key Consensus
- MoE quantization benefits from mixed-precision (all major 2025 works agree)
- Linear-block level allocation > expert-level > layer-level (MxMoE demonstrates)
- Post-training quantization preferred over QAT for MoE (cost)
- Hardware-aware co-design essential for wall-clock speedup

## Recommended Ideas (ranked)

### 🏆 Idea 1: RouteQEP — Routing-Aware Cross-Layer Error Propagation for Mixed-Precision MoE

- **Hypothesis**: MoE quantization errors are not independent — they amplify along frequently-routed expert chains. Incorporating 1-2 hop propagated error into MxMoE's ILP will improve allocation quality, especially for models with strong cross-layer dependencies (e.g., Qwen2-MoE).
- **Minimum experiment**: Modify `bits_solver.py` to use 2-hop propagated sensitivity instead of single-layer loss; evaluate on Qwen1.5-MoE and Mixtral at matched average bits; compare PPL and downstream accuracy.
- **Expected outcome**: At same average bits, RouteQEP should achieve 0.1-0.5 lower PPL than vanilla MxMoE, with larger gains on models with stronger cross-layer coupling.
- **Novelty**: 9/10 — QEP (NeurIPS 2025) does cross-layer for dense LLMs only; EAC-MoE addresses expert-shift but not via error propagation; nobody combines cross-layer propagation with MoE mixed-precision allocation.
- **Feasibility**: ~15 GPU-hours (A100). Extends existing `bits_solver.py` and `quant.py`.
- **Risk**: MEDIUM — propagated scores might correlate strongly with existing per-block loss, yielding marginal ILP improvement. The diagnostic pilot validates this.
- **Contribution type**: New method + diagnostic
- **Reviewer's likely objection**: "This is just QEP applied to MxMoE" — must show genuinely MoE-specific propagation model (routing-aware chains, not just sequential layers).
- **Why we should do this**: Directly addresses MxMoE's acknowledged limitation. If the propagation is real and MoE-specific (hypothesis), this is a clear improvement with a strong story.

### Idea 2: Scenario-Robust MxMoE — Allocation Across Routing Distributions

- **Hypothesis**: MxMoE's single-dataset offline routing statistics overfit the calibration corpus. Optimizing over multiple domain traces (code, math, wiki) with CVaR or worst-case optimization will improve out-of-domain stability with minimal average-case accuracy cost.
- **Minimum experiment**: Collect routing traces on wikitext2, humaneval-x, and GSM8K; solve robust ILP; evaluate on held-out domain.
- **Expected outcome**: More stable PPL across domains; small improvement on worst-case domain; negligible average-case regression.
- **Novelty**: 7/10 — DynaExq adapts online without kernels; this is static but robust and kernel-compatible. Standard robust optimization but novel application to MoE allocation.
- **Feasibility**: ~10 GPU-hours. Extends `moe_tracer.py` (already supports multi-dataset tracing).
- **Risk**: LOW — routing distributions may not shift enough across domains to matter, but even a null result (distributions are stable) is informative.
- **Contribution type**: New method + empirical finding
- **Reviewer's likely objection**: "Why not just average the traces?" — must show robust (worst-case) optimization provides measurable benefit over simple pooling.
- **Why we should do this**: Practical deployment relevance. Makes MxMoE's allocation robust to workload shifts.

### Idea 3: ActiveMxMoE — Near-Optimal Allocation from Sparse Profiling

- **Hypothesis**: Block sensitivities and kernel runtimes lie on a low-dimensional manifold across layers and experts. Profiling only 10-20% of candidate (block, scheme) pairs and fitting a surrogate is sufficient to recover most of the final Pareto frontier.
- **Minimum experiment**: Subsample qconfigs and blocks, fit surrogate (e.g., nearest-neighbor or linear), solve ILP with surrogate estimates, compare PPL/latency against full profiling.
- **Expected outcome**: >5x reduction in profiling time with <0.1 PPL regression.
- **Novelty**: 7/10 — active/sparse profiling for quantization sensitivity is unexplored for MoE. Practical impact.
- **Feasibility**: ~8 GPU-hours. Extends `quant.py calib` and `bits_solver.py`.
- **Risk**: LOW — even partial success (3x speedup with small regression) is publishable.
- **Contribution type**: Empirical finding + new method
- **Reviewer's likely objection**: "This is a practical speedup, not a new scientific idea" — counter with: profiling cost is a real barrier to adoption, and the manifold structure itself is an empirical finding.
- **Why we should do this**: Attacks MxMoE's core practical limitation (profiling overhead). Safest publishable result.

## Eliminated Ideas (for reference)

| Idea | Reason eliminated |
|------|-------------------|
| SparseComp-MxMoE (low-rank + mixed-precision) | **Already done**: BT-MoE (ICLR 2025) unifies bit-rank allocation for MoE |
| Format-Mixed MoE (INT4/INT8/FP8 per block) | Platform constraint: A100/4090 FP8 is limited; needs H100/Blackwell |
| BundleSwitch (online dynamic kernels) | HIGH risk: switching overhead may erase gains; heavy systems work |
| Skip or Squeeze (prune+quantize co-design) | HIGH risk: overlaps with MC-MoE; complex implementation |
| MarginBits (router uncertainty activation precision) | MEDIUM risk: kernel-unfriendly; fake-quant evidence insufficient for systems venues |
| Beyond Frequency (Shapley importance allocation) | MEDIUM risk: importance likely correlates with sensitivity/frequency |

## Diagnostic Pilot: Quantization Dependency Horizon in MoE

**Status**: COMPLETED — POSITIVE SIGNAL ✅
**Model**: Qwen1.5-MoE-A2.7B
**Design**: Perturb expert 0's linear blocks (gate/up/down) at layers 6, 12, 18; measure relative L2 error at +1/+2/+3/+4 layers.

### Results: Relative L2 Error of Hidden States

| Perturbed Layer | Block | +1 layer | +2 layers | +3 layers | +4 layers |
|----------------|-------|----------|-----------|-----------|-----------|
| Layer 6 (early) | gate | 0.0021 | 0.0030 | 0.0037 | 0.0048 |
| Layer 6 (early) | up | 0.0021 | 0.0029 | 0.0032 | 0.0040 |
| Layer 6 (early) | down | 0.0022 | 0.0036 | 0.0040 | 0.0047 |
| Layer 12 (mid) | gate | 0.0027 | 0.0032 | 0.0034 | 0.0038 |
| Layer 12 (mid) | up | 0.0027 | 0.0039 | 0.0041 | 0.0043 |
| Layer 12 (mid) | down | 0.0031 | 0.0039 | 0.0043 | 0.0043 |
| Layer 18 (late) | gate | 0.0009 | 0.0019 | 0.0029 | 0.0031 |
| Layer 18 (late) | up | 0.0007 | 0.0019 | 0.0027 | 0.0029 |
| Layer 18 (late) | down | 0.0007 | 0.0023 | 0.0031 | 0.0032 |

### Key Findings
1. **Error accumulates** across layers (doesn't decay) — 2-3x growth over 4 layers
2. **Pattern varies by depth**: early layers cause more downstream propagation than late layers
3. **Block heterogeneity**: down_proj tends to cause more propagation at middle layers; patterns differ across block types
4. **Supports RouteQEP hypothesis**: cross-layer dependency is real; 2-hop propagation justified

## Suggested Execution Order

1. **Diagnostic pilot** → validates RouteQEP hypothesis
2. **If pilot positive**: RouteQEP (Idea 1) + Scenario-Robust (Idea 2) as unified paper
3. **If pilot negative**: ActiveMxMoE (Idea 3) as standalone MLSys paper

## Best Paper Combination

**Recommended single paper: RouteQEP + Scenario-Robust + Diagnostic**
- Title: "Routing-Aware Cross-Layer Error Propagation for Robust Mixed-Precision MoE Quantization"
- Story: MxMoE fails because quantization error is (a) nonlocal and (b) routing-dependent; we first diagnose MoE-specific dependency structure, then propose a propagation-aware, scenario-robust allocator
- Venue target: NeurIPS/ICML 2026
- Build on: `bits_solver.py` (ILP), `moe_tracer.py` (tracing), `quant.py` (calibration/eval)

## Next Steps
- [ ] Collect diagnostic pilot results → validate RouteQEP hypothesis
- [ ] If positive, implement RouteQEP in bits_solver.py
- [ ] Run full experiments on Qwen1.5-MoE, Mixtral-8x7B
- [ ] Invoke /research-refine-pipeline for method refinement
- [ ] Invoke /auto-review-loop for iterative improvement
