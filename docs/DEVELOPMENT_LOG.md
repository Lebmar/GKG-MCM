# Development audit

- 14 October: source discovery and data inventory. Pooled descriptive market
  statistics across all 180 paths were printed at this initial stage, without
  policy evaluation. This exposure is disclosed in the protocol and report.
- 17 October: formal development began; final-return parsing was gated. Financial
  identities, initial costs, inflation and annual drawdown were unit tested.
- A fitting-only experiment on 1–80 used stated utility coefficients 10/4/1.
  It showed that higher utility is not equivalent to questionnaire suitability.
  No client was forced into a desired category. A 5% lower bound on each asset
  was then adopted as an explicit three-asset product rule before validation.
- The validation design used fixed epsilon .002 and a fixed joint five-year
  block forecast. No epsilon/forecast/risk coefficient search was performed on
  81–120. All six policy variants and the architecture rule were specified.
- Independent review found the first implementation did not gate initial
  allocation by estimated benefit; it was corrected before the first formal
  validation output was generated.
- Independent review then identified a measurement mismatch: projected drawdown
  included an instantaneous fee point whereas reported drawdown was annual.
  The implementation was corrected to annual observations everywhere and the
  development stage was rerun. This was a consistency correction before any
  final policy evaluation, not a parameter change to improve validation scores.
- Validation and freeze both retain source, protocol and data fingerprints.
  Refit-on-120 questionnaire estimates are explicitly in-sample diagnostics.
  Frozen client assignments are not revised using final-scenario outcomes.

Exact results and frozen fingerprints accompany the computation outputs. Human
author review is still required before submitting the competition report.
