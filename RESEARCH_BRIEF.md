# Research Brief: MoE Quantization

## Direction
Mixed-precision quantization for Mixture-of-Experts (MoE) models, extending the MxMoE framework.

## Reference Paper
- **MxMoE** (ICML 2025, arXiv:2505.05799): Mixed-precision quantization for MoE with accuracy and performance co-design
  - Key contributions: linear-block level bitwidth allocation, hardware-aware ILP optimization, auto-generated mixed-precision GroupGEMM kernels
  - Tested on: DeepSeekV2-Lite, Qwen1.5/2-MoE, Mixtral-8x7B (RTX 4090)
  - Limitations noted: mainly tested on RTX-4090, profiling step time-consuming, inter-layer dependency issue in sensitivity estimation

## Base Codebase
- **Location**: `/data/zengyq/paper/baselines/MxMoE`
- **Structure**: `mxmoe/quant/` (quantization, GPTQ, ILP solver, tracer), `mxmoe/kernels/` (CUDA kernel generation)
- **Dependencies**: PyTorch 2.6, Transformers 4.52, Gurobi (ILP), CUDA/CUTLASS, flash-attn
- **Models supported**: DeepSeekV2-Lite, Qwen1.5-MoE, Qwen2-MoE, Mixtral-8x7B
- **Quant methods**: RTN, GPTQ, GPTQ-HAD, SmoothQuant, AWQ
- **Bit configs**: W2A16 to W8A8, group sizes -1/128, sym/asym

## Known Gaps in MxMoE
1. Only tested on RTX 4090 (Ampere); no support for H100/Blackwell (FP8, FP4)
2. Sensitivity estimation uses single-layer loss; cross-layer dependencies cause issues (acknowledged in paper)
3. No support for newer/larger MoE models (DeepSeek-V3 671B, Qwen3-MoE, etc.)
4. Weight-only and weight-activation only; no KV-cache quantization integration
5. ILP solver uses Gurobi (commercial); no open-source alternative
6. No training-aware quantization (only post-training)
7. Profile/tuning step is time-consuming (noted in README)
8. Expert activation pattern collected offline; no online adaptation

## Research Goals
Find novel, publishable extensions to MxMoE that address its limitations and advance MoE quantization.
