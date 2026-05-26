# Auto Review Log

## Paper: Route-Weighted Sensitivity for Mixed-Precision MoE Quantization
## Date: 2026-04-22
## Target: EMNLP

---

## Round 1 (23:00)
- Score: 5/10
- Verdict: Borderline reject
- Key criticisms: too obvious/incremental, modest gains, ablation gap, unconditional loss, RHI underdeveloped, narrow scope

### Actions Taken
- Reframed cross-layer as "Cautionary Ablation"
- Expanded gate-weight ablation note
- Rewrote conclusion with principled justification
- Corrected shared expert handling

## Round 2 (23:05)
- Score: 5.5/10
- Verdict: Borderline, trending positive
- Key criticisms: missing frequency-only baseline, missing conditional sensitivity, narrow scope

### Actions Taken
- Added "Why Not Route-Only Allocation?" section
- Reframed intro as "combining two orthogonal information sources"
- Reframed gate-weight as supporting design principle

## Round 3 (23:10)
- Score: 5.8/10
- Verdict: Almost, still borderline
- Key criticism: argues route-only fails but doesn't empirically show it

### Actions Taken
- Softened route-only claims ("expected to be suboptimal", not "fails")
- Added explicit "Scope" paragraph
- Narrowed conclusion claims

## Round 4 - FINAL (23:15)
- Score: 6/10
- Verdict: Almost / submission-ready with risk
- Assessment: "Submit if the deadline is now, but expect borderline reviews. Sympathetic reviewer may score 6-7, skeptical reviewer may score 5."

### Remaining Weaknesses (cannot fix without new experiments)
1. Missing frequency-only baseline
2. Missing conditional sensitivity comparison
3. Narrow experimental scope (binary pool, RTN only)
4. Small gains on Mixtral/DeepSeek
5. RHI remains descriptive

### Score Progression
Round 1: 5.0 → Round 2: 5.5 → Round 3: 5.8 → Round 4: 6.0
