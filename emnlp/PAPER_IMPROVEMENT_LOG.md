# Paper Improvement Log

## Round 1 — Score: 4/10

### Issues and Fixes

#### CRITICAL 1: Method definition vs implementation mismatch
- **Issue**: Eq. 5 says π·L but implementation uses L·(1+w). These are different.
- **Fix**: Make the implementation the official method. Rewrite Eq. 5 to show (1+π/max(π))·L. Add note that raw π·L is equivalent in ranking.

#### CRITICAL 2: Table inconsistencies
- **Issue**: 7-task average but only 6 task columns shown; DeepSeek downstream claims may not match
- **Fix**: Add LAMBADA-Standard column or clarify 6-task average. Verify all numbers.

#### CRITICAL 3: RHI overclaimed
- **Issue**: "predicts" language too strong for 2 data points
- **Fix**: Replace all "predicts" with "is consistent with" or "may indicate"

#### MAJOR 4: Only 2 models
- **Status**: User says third model to be added later. Strengthen Mixtral diagnosis.

#### MAJOR 5: No variance estimates
- **Status**: Seed sensitivity experiments are running (background). Add note about pending results.

#### MAJOR 6: Baseline comparison scope
- **Fix**: Limit "drop-in" claim to sensitivity-based ILP allocators

#### MAJOR 7: Strategy pool limited
- **Fix**: Add justification for {W2,W4} only (binary decision is where routing matters most)

#### MAJOR 8: Expected-loss untested
- **Fix**: Add conditional-loss ranking check

#### MAJOR 9: RouteQEP overinterpreted
- **Fix**: Soften "confirming" → "suggests"

#### MINOR 10: Assertive language
- **Fix**: Soften throughout
