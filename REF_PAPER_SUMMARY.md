# Reference Paper Summary

**Title**: MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
**Authors**: Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, Dahua Lin
**Venue**: ICML 2025

## What They Did
MxMoE introduces a mixed-precision quantization framework for MoE models that jointly optimizes accuracy and hardware performance. Key innovations: (1) linear-block level bitwidth allocation (finer than expert-level), (2) ILP-based hardware-aware allocation considering sensitivity, activation patterns, and roofline model, (3) auto-generated mixed-precision GroupGEMM kernels via micro-kernel specialization with tile scheduling.

## Key Results
- 2.4 lower Wikitext-2 perplexity than GPTQ at 2.25-bit
- Up to 3.4x speedup over full precision
- Up to 29.4% speedup over uniform quantization at equivalent accuracy (5-bit W-A)
- Tested on DeepSeekV2-Lite, Qwen1.5/2-MoE, Mixtral-8x7B on RTX 4090
- Linear-block allocation outperforms expert-level allocation consistently

## Limitations & Open Questions
1. **Hardware limited**: Only RTX 4090 (Ada Lovelace); no H100/Blackwell FP8/FP4 support
2. **Cross-layer dependency**: Sensitivity estimated per-layer; inter-layer error propagation acknowledged but not addressed (Qwen2-MoE shows degraded results from this)
3. **Offline profiling**: Expert activation patterns collected offline; no online/adaptive mechanism
4. **No training-aware**: Only post-training quantization; no QAT or mixed-precision training
5. **Model scope**: Only tested on small-to-medium MoE (up to Mixtral 8x7B); no 100B+ models
6. **ILP solver**: Uses commercial Gurobi; no open-source fallback
7. **Profiling overhead**: Calibration process takes minutes to hours per model
8. **No KV-cache**: Only weight and activation quantization; no KV-cache quantization
9. **Static allocation**: Once computed, bit-width allocation is fixed for all inputs

## Potential Improvement Directions
1. **Cross-layer sensitivity**: Propagate quantization error across layers to improve allocation
2. **Dynamic/online allocation**: Adapt precision based on runtime token distribution
3. **Extend to larger MoE**: DeepSeek-V3 671B, Qwen3-MoE, etc.
4. **FP8/FP4 native kernels**: Leverage H100/Blackwell tensor cores
5. **Lightweight calibration**: Reduce profiling overhead with approximation or transfer learning
6. **Joint MoE quantization**: Combine weight/activation/KV-cache quantization
7. **Open-source solver**: Replace Gurobi with greedy/RL-based allocation
8. **Expert pruning + quantization**: Combine structured expert pruning with mixed-precision

## Codebase
- **Repo**: https://github.com/cat538/MxMoE
- **Local**: `/data/zengyq/paper/baselines/MxMoE`
- Key modules: `mxmoe/quant/bits_solver.py` (ILP), `mxmoe/kernels/compose_kernel.py` (kernel gen), `mxmoe/quant/moe_tracer.py` (profiling)
