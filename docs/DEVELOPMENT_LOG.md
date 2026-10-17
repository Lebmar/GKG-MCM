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

## 17 October: foundation and execution research after V1

The entries above describe the archived V1 sequence. The subsequent foundation
review retired unelicited utility coefficients from the new research objective.
It distinguishes the proposed service reference alpha=.10, technical training
screens, customer H/L, and evidence strength. Only original complete scenarios
1–120 are parsed in this phase. V1 final results were already inspected and
cannot become a fresh blind evaluation for later methods.

The strict172 study preserves whole-path nested menu fitting and explicit
Review. Its 5% positive-weight grid floor is a numerical resolution, not an
elicited asset minimum. B/C/E annual OOF exceedance counts are 9/9/9 of 120;
precise targets remain unstable across training fits.

The autonomous execution stage records seven separate, source-frozen studies:

1. Post-crossing pending hold preserves first crossing and the ever-exceeded
   event pathwise; costs, later severity and recovery may change. Recovery uses
   the fixed pre-crossing peak and right censoring at the original H.
2. Unconditional calendar-fan growth and mean-Pareto gates can worsen actual
   full-path risk. A two-path counterexample identifies the conditional-law
   limitation; exact averaging alone does not provide a policy guarantee.
3. A pathwise support guard has a restricted fixed-fan argument but adds no hold
   decisions on these development paths.
4. The same guard remains identical to its reference at five known fee rates.
5. A training-selected risk-adjusted price gate trades forecast growth against
   whole-history exceedance probability. It improves growth/fees but produces
   B/C/E OOF exceedances 10/14/10 of 120. C fails the .10 reference frequency.
6. Adding target updates among the three fitted products is not selected in
   any of six training contexts; the selected C paths equal the fixed learner.
7. Freezing those default-cost selected C controllers and varying known costs
   from 0 to 1% leaves OOF exceedances 14/120 throughout. Prices and admission are
   not relearned under stress; the annual-pending and exact-thirds references
   remain separate comparisons.

Each has a separate numerical audit. These are AI-assisted independent
implementations, not external human peer review. Unselected training candidates
have saved metrics and selector checks, but not every candidate's full state
tree; audit descriptions retain this distinction.

`reproduce_execution_stage.py` then replays all seven frozen runners in a
temporary copy and restores frozen parent metadata between stages. All 73 saved
output files match byte for byte, and original source/artifact hashes match
before and after. This is same-source reproducibility, not new market evidence.
The executed notebook, source verification and 353 tests are recorded in the
research checkpoint. The V1 `src/`, `report/` and `output/` remain unchanged
relative to 7bca10b; no new submission paper or CMT upload was produced.

The separately reviewed next-stage JSON is a design proposal only. It calls
for complete inner refitting of products and controllers, whole-H calibration,
explicit coverage and Review. It does not treat a new technical screen as a
guarantee or tune a threshold to force the observed 14/120 down to a target.

The final requirement crosswalk preserves another distinction for future work:
Q2 concerns a customer assigned Balanced. An inner fit assigning a different
product must be recorded outside that case, rather than counted as financial
Review or mixed into Balanced results. All six current C contexts are Balanced.
The crosswalk also leaves liquidity policy, a coherent final recommendation,
new frozen evaluation and the final report/package explicitly open.

An archive check found that ignored cache files and replay logs were required
by frozen file-hash contracts. Four existing cache pairs and the sanitized
replay logs are now retained in Git, without changing their bytes. In-place
regeneration would change historical metadata timestamps. New disposable
caches remain excluded; the pinned inputs are development-only.

A tracked-files-only clone at financial-source revision 1f7018e then passed
353 tests, its saved-source checkpoint, and all seven replay stages with all
73 artifacts again byte-exact. Its receipt and completed test log are under
`results/model_review/execution_stage_clean_checkout_v1/`; the temporary clone
was removed. Later notebook scope clarifications and the direct dependency
version record do not alter those frozen financial sources. The current
checkpoint checks 46 saved snapshots and 407 distinct files. This extra
checkout test supplies reproducibility evidence, not a second market sample.
