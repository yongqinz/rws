# Literature Survey: MoE Quantization & Mixed-Precision

**Date**: 2026-04-17
**Topic**: MoE quantization, mixed-precision allocation, extending MxMoE

## Landscape Summary

### Tier 1: Directly Competing — MoE Mixed-Precision Quantization

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| **MxMoE** (Duanmu et al.) | ICML 2025 | Linear-block ILP allocation + auto-generated GroupGEMM kernels | 2.4 PPL improvement over GPTQ at 2.25-bit, 3.4x speedup | **Reference paper** |
| **MoPEQ** (Chitty-Venkata et al.) | ICCV 2025 Workshop | Expert-level bit-width assignment via PTQ | Balances accuracy/efficiency per-expert | Expert-level only (coarser than MxMoE) |
| **DynaExq** | arXiv 2511.15015 (2025) | Online dynamic expert quantization with EMA-smoothed stats | Runtime-aware mixed-precision under HBM constraints | **Direct competitor**: online adaptation, but no kernel support |
| **DyMoE** | 2024 | Dynamic expert orchestration + mixed-precision for edge | Edge-optimized MoE inference | Edge-focused, different scope |
| **MC-MoE** (Huang et al.) | arXiv 2410.06270 | Static quantization + dynamic pruning combined | Training-free extreme compression for MoE | **Complementary**: combined pruning+quantization |
| **MiLo** | MLSys 2025 | Low-rank compensators for quantized MoE | INT3 MoE with minimal accuracy loss | **Complementary**: low-rank recovery, no mixed-precision |
| **Automated Fine-Grained MoE Quantization** | ACL 2025 Findings | Automated fine-grained quantization for MoE | QDrop-based extremely low-bit PTQ | Finer granularity but no kernel co-design |
| **Examining PTQ for MoE** (Li et al.) | arXiv 2406.08155 | Benchmark of MoE structure-aware quantization heuristics | Comprehensive MoE quantization benchmark | Baseline/benchmark reference |

### Tier 2: Cross-Layer Error Propagation

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| **QEP** (Ichikawa et al.) | NeurIPS 2025 | Cross-layer quantization error propagation for PTQ | Significant improvement over independent layer-wise PTQ | **Critical gap**: MxMoE acknowledges this issue; QEP solves it for dense LLMs |

### Tier 3: General Mixed-Precision for LLMs

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| **ResQ** | ICML 2025 | Low-rank residuals + mixed-precision via PCA | 33% lower PPL, 3x speedup | **Gap**: low-rank residual idea not applied to MoE |
| **SliM-LLM** | ICML 2025 | Salience-driven mixed-precision for LLMs | Salience-based bit allocation | Dense LLM only, not MoE-aware |

### Tier 4: FP8/FP4 Hardware Support

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| **FP4 Training** | ICML 2025 | First FP4 training framework | FP4 via FP8 tensor cores on H100 | **Gap**: No MoE-specific FP4/FP8 kernels |
| **Blackwell FP4 Benchmarks** | arXiv 2512.02189 | Microbenchmarking NVFP4 on Blackwell | MoE sees amplified benefits from quantization | **Gap**: MxMoE only supports INT on RTX 4090 |

### Tier 5: Expert Pruning & Compression

| Paper | Venue | Method | Key Result | Relevance |
|-------|-------|--------|------------|-----------|
| **Shapley-MoE** | NeurIPS 2025 | Shapley value-based expert importance for pruning | Principled expert selection | **Gap**: pruning importance not used for bit allocation |
| **Sub-MoE** | AAAI | Expert merging for MoE compression | Parameter reduction via merging | Different approach (merging vs quantization) |
| **µ-MoE** | MERL 2025 | Test-time micro-grained pruning | Finest-grained adaptation at test time | Test-time adaptation concept |

## Key Gaps Identified

1. **Cross-layer error propagation + MoE mixed-precision**: QEP (NeurIPS 2025) solves cross-layer error for dense LLMs, but nobody has combined this with MoE-specific mixed-precision allocation. MxMoE explicitly acknowledges this as a limitation causing issues on Qwen2-MoE.

2. **Low-rank compensators for MoE mixed-precision**: ResQ (ICML 2025) uses low-rank residuals for dense LLMs; MiLo (MLSys 2025) uses them for MoE. But nobody combines low-rank compensation with mixed-precision allocation at the linear-block level.

3. **FP8/FP4 native MoE kernels with mixed-precision**: All current MoE mixed-precision work uses INT. No native FP8/FP4 MoE kernel generation exists despite Blackwell's native support.

4. **Expert importance-aware bit allocation**: Shapley-MoE quantifies expert importance via cooperative game theory, but this information is not used for mixed-precision bit allocation. Combining expert importance scores with MxMoE's ILP could improve allocation.

5. **Online/dynamic allocation with kernel support**: DynaExq does online adaptation but lacks optimized kernels. MxMoE has kernels but is static. Nobody has both.

6. **Pruning + quantization co-design**: MC-MoE combines pruning+quantization but is coarse. A fine-grained approach combining expert pruning with linear-block level mixed-precision allocation would be novel.

## Field Consensus
- MoE quantization benefits from mixed-precision (all major 2025 works agree)
- Linear-block level allocation > expert-level > layer-level (MxMoE shows this)
- Post-training quantization is preferred over QAT for MoE (cost reasons)
- Hardware-aware co-design is essential for wall-clock speedup
- Expert heterogeneity (sensitivity, activation frequency) is a key differentiator vs dense LLMs
