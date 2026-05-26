# Auto Review Log

## Round 1 (2026-04-22)

### Assessment (Summary)
- Score: **5/10**
- Verdict: Almost, not ready
- Key criticisms:
  1. RHI predictor not credible with 2 data points
  2. Model coverage too thin (1 positive + 1 neutral model)
  3. Baseline universe too small (only same-ILP-without-π)
  4. Objective normalization unclear — may double-count routing frequency
  5. Gains modest and localized, no variance/confidence intervals
  6. Non-monotone compression claim underexplained
  7. RouteQEP negative result needs stronger evidence or move to appendix

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 5/10**

**Verdict: Almost, but not ready for a top venue yet.**

The core idea is clean and plausible, and the Qwen result is real enough to be interesting. But the current evidence supports a narrow empirical observation more than a top-venue paper claim. I would not reject it as weak engineering, but I would reject it in its current form for overclaiming relative to the evidence.

**Critical Weaknesses**

1. **RHI predictor claim is not credible with only two effective data points.** A scalar predictor cannot be established from Qwen high-RHI positive and DeepSeek moderate-RHI neutral. Fig. 5 with two points will look unserious at NeurIPS/ICML. **Minimum fix:** Reframe RHI as a diagnostic hypothesis, not a validated predictor. Or add at least 3-5 more MoE checkpoints/configurations, even smaller ones, with measured RHI and RWS gain.

2. **Model coverage is too thin, and Mixtral failure hurts.** The paper claims a structural MoE quantization mismatch, but full evidence is only one positive model and one neutral model. Mixtral failing at layer 1 leaves an obvious reviewer objection: maybe the method only works on Qwen. **Minimum fix:** Either fix Mixtral evaluation, or add another working MoE model. If impossible, explicitly narrow the claim.

3. **The baseline universe is too small.** The comparison is only "same ILP without π-reweighting." That isolates the contribution, but does not show competitiveness against practical MoE quantization baselines. **Minimum fix:** Add at least cheap baselines: routing-frequency-only allocation, sensitivity-only allocation, random/tie-break allocation, and possibly AWQ/GPTQ-style uniform mixed precision.

4. **The objective needs sharper justification.** RWS is only correct if the expert reconstruction loss is normalized per expert rather than already summed over routed tokens. Otherwise multiplying by π may double-count routing frequency. **Minimum fix:** State exactly how L_{b,s} is normalized. Add a short derivation showing RWS estimates expected routed-token loss, plus an ablation comparing unweighted, π-weighted, and raw token-count-weighted objectives.

5. **Magnitude of gains is modest and localized.** The headline result is 2.8% relative PPL improvement at one budget on one model. Downstream gains are +0.63pp and +0.31pp, which are useful but not decisive without variance. **Minimum fix:** Add calibration-seed sensitivity or bootstrap confidence intervals.

6. **Non-monotone compression claim is plausible but underexplained.** The 2.5-2.75b peak is interesting, but currently descriptive. **Minimum fix:** Analyze allocation changes by budget: which experts/layers switch from 2-bit to 4-bit, and whether rare experts collapse at 2.3b.

7. **RouteQEP negative result needs stronger evidence.** "37-51% PPL regression" sounds like either a real insight or a fragile failed implementation. **Minimum fix:** Move to appendix unless you can show a controlled diagnostic.

</details>

### Actions Planned (priority order)

1. **Clarify L_{b,s} normalization** — audit code, add derivation to paper (cheap, high impact)
2. **Add per-layer allocation-change analysis** — show which experts gain/lose bits under RWS vs baseline (cheap, addresses #6)
3. **Reframe RHI as hypothesis** — tone down predictor language throughout paper (cheap, addresses #1)
4. **Add cheap baselines** — uniform W2A16, uniform W4A16, random allocation at 2.5b (medium effort, addresses #3)
5. **Move RouteQEP to appendix** — keep as a brief mention in main text (cheap, addresses #7)
6. **Add seed-sensitivity** — rerun 2.5b with different calibration seeds (GPU time needed, addresses #5)

### Status
- Continuing to Round 2
- Difficulty: medium

## Round 2 (2026-04-22)

### Assessment (Summary)
- Score: **6/10**
- Verdict: Almost, borderline
- Key criticisms addressed:
  1. Added L_{b,s} normalization justification (no double-counting)
  2. RHI reframed as "hypothesis" not "validated predictor"
  3. Added broader baseline table (uniform W2/W3/W4, GPTQ W3/W4)
  4. Added per-task downstream breakdown in Appendix B
  5. Added allocation-change analysis (why 2.5-2.75b is sweet spot)
  6. RouteQEP moved to Appendix C with full diagnostics
- Remaining: expected-loss independence assumption, gate-weight variant, seed sensitivity

## Round 3 (2026-04-22)

### Assessment (Summary)
- Score: **6/10**
- Verdict: Almost, borderline
- Changes:
  1. Added approximation caveat (RWS is first-order, independence may not hold)
  2. Added gate-weight vs selection-frequency justification
  3. Title "predictor" → "diagnostic" (partially)
  4. Baseline table caveat strengthened
- Remaining: incomplete terminology sweep, gate-weight not empirically shown, GPTQ wording too strong

## Round 4 (2026-04-22, FINAL)

### Assessment (Summary)
- Score: **6/10**
- Verdict: Almost, submit-able if deadline is now
- Changes:
  1. All "predictor" → "diagnostic/hypothesis" terminology sweep completed
  2. Determinism and variance paragraph added (explains eval-seed = deterministic, calibration-subset variance unmeasured)
  3. GPTQ comparison wording softened
  4. Title finalized: "Heterogeneity Diagnostic"
- Remaining unaddressed:
  - No third model (Mixtral unevaluable, no other MoE available)
  - No calibration-subset sensitivity (~2 hrs per seed, not run)
  - Gate-weight ablation asserted (correlation >0.99) not empirically shown with PPL

## Final Summary

**Score progression**: 5 → 6 → 6 → 6 (over 4 rounds)

**Strengths (confirmed by reviewer)**:
- Clean, simple core idea
- Honest framing after revision
- Good negative-result reporting
- Appropriate appendices

**Blockers for top venue (7/10)**:
- Need ≥1 more evaluable MoE model OR calibration-subset sensitivity
- Gate-weight ablation needs empirical PPL confirmation

**Recommendation**: Submit-able as-is for a workshop or short paper. For NeurIPS/ICML main venue, add one more model or run calibration-subset sensitivity.

## Method Description

RWS (Route-Weighted Sensitivity) modifies the ILP objective for MoE mixed-precision bit allocation by multiplying each expert's per-block calibration loss by its empirical routing probability. The method is hyperparameter-free and uses only data already collected during calibration. It is combined with RHI (Route Heterogeneity Index), a diagnostic scalar that measures routing skewness and helps practitioners decide whether RWS will help. Experiments on Qwen1.5-MoE (RHI=1.84) show −2.8% PPL at 2.5 bits/param, with gains concentrated in the 2.5–2.75 bit budget range.
