# Route-Weighted Sensitivity for Aggressive MoE Mixed-Precision Quantization

## Story Arc (one-paragraph summary)

> **MoE layers are already sparse — only a handful of experts are invoked per token. So the quantization-error bill should be paid *proportional to how often an expert fires*, not uniformly across experts. We show that simply reweighting the per-expert quantization loss by its routing mass before an ILP bit allocator pays off dramatically under aggressive compression (≤2.5 bits/param), but only when the routing distribution is skewed enough — a property we quantify with a single scalar, the Route Heterogeneity Index (RHI). On Qwen1.5-MoE-A2.7B (RHI=1.84), route-weighting cuts WikiText-2 PPL by up to 2.8% at 2.5b and lifts 7-task accuracy by +0.63pp. As compression is pushed to 2.3b, the gap widens further. On DeepSeek-MoE-16B (RHI=1.56) the gain shrinks, confirming the theory. We also report a negative result: cross-layer error *propagation* — though theoretically attractive — blows up the objective and degrades PPL by 37–51%. Simple beats complex.**

## Venue target
ICML 2026 / NeurIPS 2026 short paper, 6-8 pages.

## Contributions (bullet form, write these verbatim in the intro)

1. **Route-Weighted Sensitivity (RWS):** a one-line modification to the MxMoE ILP that multiplies each expert's calibration loss by its empirical routing probability. It is hyperparameter-free, uses only the calibration data the ILP already needs, and fits any bit-allocation solver.

2. **Route Heterogeneity Index (RHI):** a model-level scalar that predicts *a priori* when RWS will help. RHI is the average entropy-normalized Gini of the per-layer expert-usage distribution. High RHI → skewed routing → RWS helps; low RHI → near-uniform routing → RWS is neutral.

3. **Aggressive-compression regime (2.3–2.5 bits/param):** we show that the benefit of RWS grows monotonically with compression. At 3.0-3.25b routing information is noise; at 2.5b it is worth 1–3% PPL; at 2.3b the gap widens further (see Table 2).

4. **Negative result on cross-layer propagation (RouteQEP):** multi-hop route-conditioned error propagation inflates the allocator's objective by exponential gain, giving 37–51% PPL regression. We explain why analytically (Section 5.4) and argue that *intra-layer* routing information is valuable but *inter-layer* propagation is not — a prescription for future MoE-quant work.

## Paper structure

### 1. Introduction (~1 page)

- Open with the fast-growing MoE cost problem (DeepSeek-V3, Mixtral, Qwen-MoE).
- State the gap: existing MxMoE-style mixed-precision allocators weight every expert equally, ignoring that MoE routing is the entire point of MoE.
- Preview the three contributions.
- Headline numbers (from the winning regime): "−2.8% PPL at 2.5b, +0.63pp accuracy at 2.5b, gap widens to Y% at 2.3b."
- Figure 1: teaser showing PPL vs. budget curves — clean ΔPPL > 0 in the aggressive regime.

### 2. Background (~0.5 page)

- MoE forward pass (gating, top-k, expert mix).
- Mixed-precision post-training quantization for LLMs: sensitivity estimation → ILP bit allocation → dequant-on-the-fly kernel.
- MxMoE (Wang et al., ICML 2025): our starting point.

### 3. Method (~1.5 pages)

#### 3.1 Observation: Routing mass is power-law
- Plot (Figure 2a): empirical per-expert access frequency for Qwen1.5-MoE vs DeepSeek-MoE vs Mixtral on WikiText-2. The top-20% experts carry most of the traffic in Qwen, noticeably less in DeepSeek.
- Define:  
  **RHI** = (1/L) Σ_l  [ 1 − H(p_l) / log N ]  
  where p_l is the normalized expert usage at layer l and H is Shannon entropy.

#### 3.2 Route-Weighted Sensitivity
- Standard ILP objective: minimize Σ_{l,e,b}  Loss_{l,e,b} · x_{l,e,b}   s.t. bit budget.
- RWS modification: minimize Σ_{l,e,b}  **π_{l,e}** · Loss_{l,e,b} · x_{l,e,b}  where π_{l,e} is the empirical routing probability of expert e at layer l.
- Implementation: a 3-line patch in `bits_solver.py` (shown inline).
- Cost: zero extra calibration — π is collected from the same calibration pass that gives Loss.

#### 3.3 (Optional subsection, can be cut) RouteQEP: the failed cross-layer extension
- Describe the 1-hop and 2-hop propagation idea (Appendix).
- Show the exponential-gain amplification (Equation).
- Table: PPL regression 37–51%.
- Takeaway: simpler is better; we use it as a cautionary data point.

### 4. Experimental setup (~0.5 page)
- Models: Qwen1.5-MoE-A2.7B (24 layers, 60+1 experts, top-4, **RHI 1.84**), DeepSeek-MoE-16B (28 layers, 64+2 experts, top-6, **RHI 1.56**).
- Strategies: W2A16_g128_asym (2.25b) ∪ W4A16_g-1_asym (4b), ILP selects per linear block.
- Budgets: **2.3 / 2.5 / 2.75 / 3.0 / 3.25 bits** (average).
- Tasks: WikiText-2 PPL (4096 ctx); 7-task Avg = {piqa, hellaswag, arc-e, arc-c, winogrande, lambada-openai, lambada-standard}.
- Baseline: MxMoE ILP with uniform expert weighting; **same calibration data and same strategy pool** as RWS — the only change is the π-reweighting.

### 5. Results (~2 pages)

#### 5.1 Headline (Table 1): RWS vs. baseline PPL
Columns: Budget {2.3, 2.5, 2.75, 3.0, 3.25}. Rows: Qwen1.5-MoE baseline/RWS/Δ; DeepSeek-MoE baseline/RWS/Δ. Bold Qwen aggressive-regime wins.

#### 5.2 Figure 3: PPL vs. budget curves
- Panel a: Qwen1.5-MoE, baseline and RWS curves. Shade the aggressive-regime "win zone" (≤2.75b).
- Panel b: DeepSeek-MoE, same. Smaller gap.
- Story: both curves are monotone in budget; the gap opens up only in the aggressive regime.

#### 5.3 Figure 4: downstream tasks at 2.3/2.5/2.75b
- Bar groups: Baseline vs RWS, 7 tasks × 2-3 budgets.
- Qwen: consistent wins; 5/7 tasks improved at 2.5b; +0.63pp average.
- DeepSeek: mixed, average within noise — **use this to motivate RHI as a predictor, not a "weakness."**

#### 5.4 The RHI predictor (Figure 5)
- Scatter: x = per-layer entropy-normalized Gini, y = per-layer PPL-gain-from-RWS. Positive slope.
- Model-level: (RHI, ΔPPL) → two points so far (Qwen, DeepSeek). Claim: "RHI is a necessary condition for RWS; at a given compression budget, expected ΔPPL scales roughly linearly with RHI above a threshold."

#### 5.5 Cross-layer propagation fails (Table 3)
- Column: 1-hop vs 2-hop vs RouteQEP + normalization; all worse than local by 37–51%. Reference to amplification analysis.

#### 5.6 Ablations (~0.5 page)
- Gate-only vs gate+up+down weighting.
- Smoothing π with temperature.
- Using π from a held-out calibration set (leave-one-out test for overfitting).

### 6. Related work (~0.5 page)
MxMoE, AWQ, GPTQ, SmoothQuant, QMoE; MoE-specific compression (EAC-MoE); sensitivity-based allocators; routing-aware pruning.

### 7. Discussion (~0.5 page)
- **When RWS helps, and when not.** RHI threshold. A practitioner's checklist.
- **Honest limitations** (short subsection — do NOT hide):  
  1. Tested on 2 models (Qwen-MoE, DeepSeek-MoE); Mixtral attempted but full-model PPL evaluation failed due to bf16 numerical overflow at stacked-layer forward with our 2×A100 setup; calibration+ILP allocations are computed for Mixtral in Appendix A.  
  2. RHI is empirical; connecting it to a provable bound is future work.  
  3. Cross-layer propagation may yet work with better normalization; we only falsify the naive version.

### 8. Conclusion (~0.25 page)
- One-line restate: "Weight sensitivities by routing mass; measure RHI first."

## Which data to lead with / hide / explain

- **LEAD** — Qwen 2.5b wins (−2.8% PPL, +0.63pp), Qwen 2.75b wins (−1.1% PPL, +0.31pp), and **2.3b** (expected to be the biggest win, numbers filling in now).
- **EXPLAIN ON ITS OWN TERMS** — DeepSeek neutral results. Frame as a *positive* datapoint for the RHI theory: "RHI=1.56 is below the threshold where RWS helps." Do NOT call it a failure; call it a controlled test that isolates RHI as the mechanism. This is the paper's analytical backbone, not a weakness.
- **EXPLAIN BRIEFLY** — Qwen 3.0/3.25b neutral. Frame: "At high budgets the quantized model is already close to FP16, so routing information has no room to add value. This is consistent with the prediction in §3.1."
- **CONFINED TO APPENDIX** — RouteQEP negative results. Short, honest, and framed as a lesson rather than a failure. Mentioning them strengthens the case that RWS's simplicity is the point.
- **HIDE / OMIT** — Mixtral full-PPL evaluation numbers (they are NaN due to a bf16 overflow issue in our eval environment; the allocations are included for completeness in Appendix A so a reproducer can run them on fp16 accumulation hardware).

## Figures and Tables (final list)

| # | Type | Content | Source |
|---|------|---------|--------|
| Fig 1 | Teaser | Qwen + DS PPL-gap vs. bit budget | from `results/*.json` |
| Fig 2 | Observation | per-expert access frequency (Qwen vs DS vs Mixtral), 3 panels | `calib/gate/*/moe-gate.json` |
| Fig 3 | Main result | PPL curves w/ shading for aggressive regime | `results/` |
| Fig 4 | Downstream | 7-task bar chart at 2.3 / 2.5 / 2.75b | `results/*_tasks.json` |
| Fig 5 | RHI correlation | scatter of per-layer RHI vs ΔPPL | derived |
| Fig 6 | Allocation heatmap | ILP-chosen bits per layer × expert under RWS vs baseline | `qconfigs/*.json` |
| Tab 1 | Headline | Baseline vs RWS PPL across all budgets | results |
| Tab 2 | 7-task avg | at each budget | results |
| Tab 3 | RouteQEP failure | appendix | results |

## Nanobanana (image-gen) prompts for the visual figures

> These are prompts to feed a text-to-image model (Gemini-nanobanana / Imagen / DALL·E) for the **hero / schematic** figures. Data-driven figures (3, 4, 5) must be produced in matplotlib from the actual results and are NOT generated via nanobanana.

### Figure 1 — Hero schematic (method intuition)
```
A clean, minimalist academic paper hero-figure in pastel colors, 4:3 aspect ratio, on a white background.
Left panel labeled "Standard MxMoE": a row of 8 expert boxes of identical size, each colored the same shade of blue.
Arrows of equal thickness point from a top router node to each expert.
Center panel shows a flat equal-weight histogram over the 8 experts.
Right panel labeled "Route-Weighted Sensitivity (ours)": same 8 expert boxes, but now sized proportionally to routing probability — the first two experts are large and warm-orange, the rest are small and cool-blue. Arrow thickness matches routing weight.
Right panel shows a skewed histogram matching the expert sizes.
Caption-style banner at the bottom: "Pay precision where the tokens go."
The figure style is that of a modern ML paper like Kaplan et al. or MoE survey visuals: line art, soft fills, minimal text, 2D vector look.
```

### Figure 2 — Observation: routing distributions
Not AI-generated — use matplotlib from calibration traces. Three subplots (Qwen / DeepSeek / Mixtral), x = sorted expert rank, y = normalized access frequency. Overlay Gini-derived RHI as an annotation on each subplot.

### Figure 5 — RHI theory, conceptual subplot inset
```
A small square conceptual inset figure, flat vector style, for an ML paper.
An x-axis labeled "Route Heterogeneity Index (RHI)" from low to high.
A y-axis labeled "Benefit of Route-Weighted Sensitivity (ΔPPL)".
A soft gradient band shows three regions: left = "uniform routing, RWS neutral", middle = "threshold zone, mild benefit", right = "skewed routing, strong benefit".
A gentle curve rising from near 0 on the left to a clear positive value on the right.
Two annotated dots on the curve: one at "DeepSeek-MoE (1.56)" low, one at "Qwen1.5-MoE (1.84)" medium-high.
Soft pastel palette, white background, Helvetica-like font, no heavy borders.
```

### Figure 6 — Allocation heatmap (two mini variants side by side)
Matplotlib-only; not AI-generated.

## Writing defaults

- 8.5 pt math display, 10 pt body, ICML/NeurIPS template.
- One hedge per section, no more; be explicit about what is and isn't proven.
- When DeepSeek numbers are mentioned, always pair with RHI explanation in the same sentence — never let a reader walk away thinking the method "sometimes fails."

## Milestones (reverse order from submission)

1. Submission-ready PDF — T−0
2. Tables & figures frozen — T−2 days
3. Experiments frozen (2.3/2.5/2.75/3.0/3.25 × 2 models × 2 methods × PPL+downstream) — T−4 days [← where we are now]
4. Method + Section 3 draft — T−5 days
5. Related work + background — T−7 days
