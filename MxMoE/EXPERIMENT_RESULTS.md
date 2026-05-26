# Route-Weighted Sensitivity Experiment Results

## Model: Qwen1.5-MoE-A2.7B (24 layers, 60 experts + 1 shared)

### Calibration Data (all 24/24 layers)
- W2A16_g128_asym (2.25-bit weight, 16-bit act, group-128)
- W4A16_g-1_asym (4-bit weight, 16-bit act)

### Routing Analysis
- **Route Heterogeneity Index (RHI)**: mean=1.84, max=2.12
- Co-routing transition matrices: 23 matrices (60x60)

### PPL Results (WikiText2)

| # | Budget | Method | PPL | Δ PPL |
|---|--------|--------|-----|-------|
| 1 | 3.25b | **MxMoE Baseline** | **7.644** | — |
| 2 | 3.25b | Route-weighted | 7.652 | +0.1% |
| 3 | 3.0b | **MxMoE Baseline** | **7.953** | — |
| 4 | 3.0b | Route-weighted | 7.955 | +0.03% |
| 5 | 2.75b | MxMoE Baseline | 8.525 | — |
| 6 | 2.75b | **Route-weighted** | **8.428** | **-1.1%** |
| 7 | 2.5b | MxMoE Baseline | 9.619 | — |
| 8 | 2.5b | **Route-weighted** | **9.351** | **-2.8%** |

### Downstream Task Results (7 tasks)

#### 3.25-bit Budget

| Task | Baseline | Route-Weighted | Δ |
|------|----------|---------------|-------|
| PIQA | 0.7742 | 0.7780 | +0.38 |
| HellaSwag | 0.7471 | 0.7481 | +0.10 |
| ARC-Easy | 0.6511 | 0.6431 | -0.80 |
| ARC-Challenge | 0.4164 | 0.4147 | -0.17 |
| Winogrande | 0.6606 | 0.6567 | -0.39 |
| Lambada OpenAI | 0.6765 | 0.6831 | +0.66 |
| Lambada Standard | 0.6103 | 0.6160 | +0.57 |
| **Average** | **0.6480** | **0.6485** | **+0.05** |

### Downstream Task Results (7 tasks)

#### 2.5-bit Budget

| Task | Baseline | Route-Weighted | Δ |
|------|----------|---------------|-------|
| PIQA | 0.7541 | 0.7584 | +0.43 |
| HellaSwag | 0.6955 | 0.7013 | +0.58 |
| ARC-Easy | 0.5812 | 0.5926 | +1.14 |
| ARC-Challenge | 0.3771 | 0.3686 | -0.85 |
| Winogrande | 0.6338 | 0.6298 | -0.40 |
| Lambada OpenAI | 0.5927 | 0.6134 | +2.07 |
| Lambada Standard | 0.5321 | 0.5461 | +1.40 |
| **Average** | **0.5952** | **0.6015** | **+0.63** |

#### 2.75-bit Budget

| Task | Baseline | Route-Weighted | Δ |
|------|----------|---------------|-------|
| PIQA | 0.7622 | 0.7655 | +0.33 |
| HellaSwag | 0.7215 | 0.7180 | -0.35 |
| ARC-Easy | 0.6263 | 0.6385 | +1.22 |
| ARC-Challenge | 0.3874 | 0.3916 | +0.42 |
| Winogrande | 0.6472 | 0.6567 | +0.95 |
| Lambada OpenAI | 0.6369 | 0.6400 | +0.31 |
| Lambada Standard | 0.5768 | 0.5698 | -0.70 |
| **Average** | **0.6226** | **0.6257** | **+0.31** |

---

## Model: DeepSeek-MoE-16B (28 layers, 64 routed + 2 shared experts)

### Calibration Data (all 28/28 layers)
- W2A16_g128_asym (2.25-bit weight, 16-bit act, group-128)
- W4A16_g-1_asym (4-bit weight, 16-bit act)

### Routing Analysis
- **Route Heterogeneity Index (RHI)**: mean=1.56, max=1.98
- Top-6 routing with 64 experts

### PPL Results (WikiText2)

| # | Budget | Method | PPL | Δ vs FP16 | Δ vs Baseline |
|---|--------|--------|-----|-----------|---------------|
| - | FP16 | — | **6.1155** | — | — |
| 1 | 3.25b | MxMoE Baseline | 6.7299 | +10.1% | — |
| 2 | 3.25b | Route-weighted | 6.7341 | +10.1% | +0.06% |
| 3 | 3.0b | MxMoE Baseline | 6.9341 | +13.4% | — |
| 4 | 3.0b | Route-weighted | 6.9385 | +13.5% | +0.06% |
| 5 | 2.75b | MxMoE Baseline | 7.3201 | +19.7% | — |
| 6 | 2.75b | Route-weighted | 7.3178 | +19.6% | **-0.03%** |
| 7 | 2.5b | MxMoE Baseline | 8.0053 | +30.9% | — |
| 8 | 2.5b | Route-weighted | 8.0412 | +31.5% | +0.45% |

### Downstream Task Results (7 tasks)

#### 2.5-bit Budget

| Task | Baseline | Route-Weighted | Δ |
|------|----------|---------------|-------|
| PIQA | 0.7486 | 0.7524 | +0.38 |
| HellaSwag | 0.7022 | 0.6994 | -0.28 |
| ARC-Easy | 0.6368 | 0.6338 | -0.30 |
| ARC-Challenge | 0.3908 | 0.3891 | -0.17 |
| Winogrande | 0.6322 | 0.6393 | +0.71 |
| Lambada OpenAI | 0.6109 | 0.6128 | +0.19 |
| Lambada Standard | 0.5478 | 0.5506 | +0.28 |
| **Average** | **0.6099** | **0.6111** | **+0.12** |

#### 2.75-bit Budget

| Task | Baseline | Route-Weighted | Δ |
|------|----------|---------------|-------|
| PIQA | 0.7633 | 0.7633 | 0.00 |
| HellaSwag | 0.7200 | 0.7198 | -0.02 |
| ARC-Easy | 0.6448 | 0.6448 | 0.00 |
| ARC-Challenge | 0.3968 | 0.4027 | +0.59 |
| Winogrande | 0.6543 | 0.6519 | -0.24 |
| Lambada OpenAI | 0.6431 | 0.6404 | -0.27 |
| Lambada Standard | 0.5824 | 0.5859 | +0.35 |
| **Average** | **0.6292** | **0.6298** | **+0.06** |

#### 3.25-bit Budget

| Task | Baseline | Route-Weighted | Δ |
|------|----------|---------------|-------|
| PIQA | 0.7813 | 0.7840 | +0.27 |
| HellaSwag | 0.7412 | 0.7396 | -0.16 |
| ARC-Easy | 0.6599 | 0.6528 | -0.71 |
| ARC-Challenge | 0.4241 | 0.4189 | -0.52 |
| Winogrande | 0.6748 | 0.6661 | -0.87 |
| Lambada OpenAI | 0.6856 | 0.6825 | -0.31 |
| Lambada Standard | 0.6280 | 0.6187 | -0.93 |
| **Average** | **0.6564** | **0.6518** | **-0.46** |
- W2A16_g128_asym (2.25-bit weight, 16-bit act, group-128)
- W4A16_g-1_asym (4-bit weight, 16-bit act)

### Routing Analysis
- **Route Heterogeneity Index (RHI)**: mean=1.56, max=1.98
- Top-6 routing with 64 experts

### PPL Results (WikiText2)

| # | Budget | Method | PPL | Δ vs FP16 | Δ vs Baseline |
|---|--------|--------|-----|-----------|---------------|
| - | FP16 | — | **6.1155** | — | — |
| 1 | 3.25b | MxMoE Baseline | 6.7299 | +10.1% | — |
| 2 | 3.25b | Route-weighted | 6.7341 | +10.1% | +0.06% |
| 3 | 3.0b | MxMoE Baseline | 6.9341 | +13.4% | — |
| 4 | 3.0b | Route-weighted | 6.9385 | +13.5% | +0.06% |
| 5 | 2.75b | MxMoE Baseline | 7.3201 | +19.7% | — |
| 6 | 2.75b | Route-weighted | 7.3178 | +19.6% | **-0.03%** |
| 7 | 2.5b | MxMoE Baseline | 8.0053 | +30.9% | — |
| 8 | 2.5b | Route-weighted | 8.0412 | +31.5% | +0.45% |

### Cross-Model Comparison

| Model | RHI (mean) | Top-k | 2.5b Δ (Route-Weighted) | 2.75b Δ (Route-Weighted) |
|-------|-----------|-------|------------------------|-------------------------|
| Qwen1.5-MoE-A2.7B | 1.84 | top-4 | **-2.8%** | **-1.1%** |
| DeepSeek-MoE-16B | 1.56 | top-6 | +0.45% | -0.03% |

### Key Finding

**Route-weighted sensitivity improvement correlates with Route Heterogeneity Index (RHI).**

- Qwen1.5-MoE (RHI=1.84): Route-weighting provides significant improvement (-2.8% at 2.5b)
- DeepSeek-MoE (RHI=1.56): Route-weighting provides negligible/mixed results

This suggests that **route heterogeneity is a necessary condition** for route-weighted sensitivity to be effective. When routing is more uniform (lower RHI), there's less signal to exploit, and the weighting approach doesn't improve allocation.

### Files Generated

**Code changes:**
- `mxmoe/quant/propagation.py` — Route-conditioned error propagation module
- `mxmoe/quant/bits_solver.py` — ILP solver with route-weighted + deepseek_moe support
- `mxmoe/quant/moe_tracer.py` — Co-routing trace + deepseek model_type support
- `mxmoe/quant/moe_utils.py` — deepseek_moe model support
- `mxmoe/quant/data_utils.py` — deepseek_moe cache entries
- `mxmoe/quant/quant.py` — position_embeddings fix for DeepSeek
- `mxmoe/kernels/gen_workload.py` — deepseek_moe trace file support

**Calibration files:**
- `calib/deepseek_moe-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json` (116KB)
- `calib/deepseek_moe-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json` (116KB)
- `calib/gate/deepseek_moe/wiki2/4096/moe-gate.json`
- `calib/gate/deepseek_moe/wiki2/4096/corouting.json`

**QConfig files:**
- `qconfigs/.../deepseek_moe_rtn_Slayer_bs512_wbits{2.5,2.75,3.0,3.25}_r1.0.json` (baseline)
- `qconfigs/.../deepseek_moe_rtn_Slayer_bs512_wbits{2.5,2.75,3.0,3.25}_r1.0_prop1.json` (route-weighted)

**PPL results:**
- `results/deepseek_moe_fp16_ppl.json`
- `results/deepseek_moe_baseline_{2.5b,2.75b,3.0b,3.25b}_ppl.json`
- `results/deepseek_moe_routewtd_{2.5b,2.75b,3.0b,3.25b}_ppl.json`
