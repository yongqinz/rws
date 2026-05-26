# Auto Review Log

## Round 1 (2026-04-23)

### Assessment (Summary)
- Score: 4/10
- Verdict: Not ready
- Key criticisms:
  1. Novelty is incremental
  2. Empirical gains are small and inconsistent
  3. Baselines are too limited
  4. Freq-Only Mixtral was estimated (not actually run)
  5. RHI diagnostic is under-supported
  6. Calibration generality not proven
  7. Budget and allocation behavior need clearer evidence

### Actions Taken
- **Fix 4**: Ran actual Mixtral freq-only experiments. Real results: PPL=4.48/4.62/6.19 (3.0b/2.75b/2.5b). Estimated values for 3.0b and 2.75b were exact; 2.5b was overestimated (8.50 vs actual 6.19).
- **Fix 5**: Downgraded RHI language from "preliminary diagnostic" to "qualitative observation". Removed predictive language.
- **Fix 7**: Added allocation behavior analysis: 32/1464 blocks changed on Qwen (2.2%), traffic-weighted bits +0.034, upgraded blocks have higher frequency (364 vs 258).
- **Fix 1**: Added routing entropy analysis: Qwen 0.995 (nearly uniform), DeepSeek 0.955, Mixtral 0.595.

### Updated Results (Mixtral Freq-Only - now real data)
| Budget | MxMoE | Freq-Only | RWS |
|--------|-------|-----------|-----|
| 3.0b | 4.36 | 4.48 | 4.34 |
| 2.75b | 4.49 | 4.62 | 4.46 |
| 2.5b | 4.76 | 6.19 | 4.71 |

### Status
- Continuing to Round 2
- Difficulty: medium

## Round 2 (2026-04-23)

### Assessment (Summary)
- Score: 5/10 (up from 4)
- Verdict: Almost
- Key remaining criticisms:
  1. Core contribution incremental (32/1464 blocks is tiny)
  2. Gains still modest (need confidence intervals)
  3. Baselines not strong enough
  4. Qwen near-uniform routing but largest gain - reconcile with entropy
  5. Calibration generality untested
  6. Tone down freq-only narrative (Mixtral not catastrophic)

### Actions Planned
- Fix 1: Explain why tiny allocation change matters (block sensitivity + reversion study)
- Fix 4: Reconcile Qwen entropy paradox
- Fix 6: Tone down freq-only language
- Fix 2: Per-task results already in appendix, add to text

### Round 2 Fixes Implemented
- Fix 4 (Qwen entropy paradox): Added "Why small routing differences matter" paragraph explaining ILP threshold effect. Even near-uniform routing (entropy 0.995) has small deviations (CV=0.21); RWS uses these to break ties among near-equal sensitivities. Changed blocks are 1.4σ above mean freq.
- Fix 6 (Tone down freq-only): Changed "collapses catastrophically" to "consistently worse", added Mixtral data, removed "25×" emphasis.
- Fix 7 (Allocation): Added quantitative analysis with traffic-weighted bits.

## Round 3 (2026-04-23)

### Assessment (Summary)
- Score: 5.5/10 (up from 5)
- Verdict: Almost
- Remaining:
  1. Baseline limitation (main blocker) — hard to fix without new engineering
  2. Calibration-domain robustness — can test sample sizes
  3. Contribution framing — position as minimal correction
  4. Per-task downstream transparency
  5. Binary pool cherry-picking concern

### Actions Planned
- Add per-task win/loss/tie counts to text
- Reframe contribution in intro as "minimal correction, zero overhead"
- Add explicit limitation about strategy pool scope

## Round 4 — FINAL (2026-04-23)

### Assessment (Summary)
- Score: 6/10 (up from 5.5)
- Verdict: Almost / borderline submission-ready
- Remaining weaknesses acknowledged as limitations

### Score Progression
| Round | Score | Verdict |
|-------|-------|---------|
| 1     | 4/10  | Not ready |
| 2     | 5/10  | Almost |
| 3     | 5.5/10| Almost |
| 4     | 6/10  | Almost / borderline accept |

### Key Improvements Made
1. Mixtral freq-only: estimated → real data
2. RHI: predictive diagnostic → qualitative observation
3. Allocation analysis: added block-level, traffic-weighted, z-score analysis
4. Qwen entropy paradox: explained via ILP threshold effect
5. Freq-only narrative: toned down from "catastrophic" to "consistently worse"
6. Per-task transparency: 16/21 wins (76%)
7. Scope limitation: explicit about binary pool constraint

### Remaining Known Limitations
- Only RTN + binary W2/W4 ILP validated
- No calibration-domain robustness test
- Simple method with modest gains (strength is cost-benefit, not novelty)
- Binary pool limits generality

### Final Recommendation
Submit to EMNLP. Expect borderline reviews (weak reject to borderline accept). Key selling point: zero-cost correction with consistent improvements.
