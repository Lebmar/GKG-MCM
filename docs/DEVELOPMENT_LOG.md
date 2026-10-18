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

## 2026-10-18 — Complete-process dynamic calibration

The next-stage proposal was implemented as a separate prospective protocol and
30 fixed tasks on development scenarios 1–120. Every dynamic inner training
set reselects the strict172 static menu and gamma before learning the nine-price
fixed-target controller. Beta is selected on whole inner validation paths;
only an effective Balanced case enters Q2, with no financial imputation or
hidden annual fallback for Review. The old six static refits reproduce exactly.

The pooled outer learner automatically executes 48/120 paths and records 72
execution Reviews. Its 2/48 exceedances must be read with that 40% coverage.
On the same 48 paths, the pending and exact-thirds references each have one
exceedance. Compared with pending, mean annual log growth increases by about
0.359 percentage points, mean MDD increases by 2.076 points, and fees/trading
fall. The served subgroup has the same financial results as the old price
learner. The full 120 descriptive fit remains fully automatic; this does not
replace the outer evidence or establish stable coverage.

Outer folds 2/4 choose beta .05 but have no admissible price on final refitting;
fold 3 has no eligible inner beta. In fold 2, .05/.075/.10 have identical saved
inner outcomes, so the smaller-beta tie convention is not unique empirical
identification. These findings are retained without changing the grid, alpha,
tie rule or Review outcomes after seeing OOF results.

401 unit tests pass. An AI-assisted independent implementation passes 445,295
selection/accounting checks, replays all 1,248 selected actual/inner accounts,
and verifies 79 prespecified Q states. The same-source isolated replay completes
all 30 tasks: 15 output artifacts are byte-exact; progress differs only in
declared runtime fields. Source/input/original-output fingerprints remain
unchanged. These are computational checks, not new market samples or external
human peer review.

The executed development notebook and new figures explain coverage and matched
financial comparisons. A separate liquidity design lists missing customer
facts and normative alternatives without inventing withdrawals or adopting a
cash floor in this study. The stage review records remaining service-design
and prediction questions. No final policy, new final evaluation, submission
paper or CMT upload was produced; V1 source/report/output remain intact.


## 2026-10-18 — Parameter/refit mechanisms, service units and joint liquidity

The old complete-process study remains frozen. A training-only descriptive
ledger separates finite beta/controller decision platforms from statistical
identification or elicited customer preferences. Outer 2's .05/.075/.10 inner
outcomes are identical; the minimum-beta tie selects .05, whose full 96 training
cap 4 lies below all nine controllers' minimum 6 events. The unrounded 4.8 is
also insufficient. Outer 4 instead changes its admitted validation cohort:
common 7/72 plus newly included 5/24. Outer 3 has no eligible inner beta. Exact
shared/added-event bookkeeping is not a causal decomposition of target, fan
and endogenous-state effects.

Coverage is now reported in separate units: representative questionnaire cases,
completed training-fit admission indicators, market-path-weighted admission,
conditional market risk and within-horizon trading. The five overlapping
training fits do not identify customer-population coverage or independent
future deployment success probability. Non-admitted financial outcomes stay
NULL, including zero-auto conditional risk. Annual and equal-thirds references
are hypothetical comparisons, never a customer fallback.

A new short-questionnaire convention sets target bill floors High 30%,
Medium 10%, Low no additional floor. These are stated normative choices, not
identified cash needs, drifted live balances or withdrawal guarantees. The
competition's given questionnaire is sufficient for that explicit convention;
extra real-deployment facts must not make all provided examples automatically
unanswerable. The pure governance prototype validates versioned scope and
fit lineage, distinguishes notices from actions, and retains historical peaks,
CPI, full horizon and absorbing breach state. Its proposal ranking is a
separate prototype; the numerical stage's actual static source/protocol is the
authoritative rule, and a reserve-version reference alone is not a complete
service-admission receipt or completed human review.

The new joint stage refits the floor-constrained menu/oracle at every static
subfold in all 30 tasks. Canonical and retained-top-inner-set/refit-aware final
selectors share all training and inner evidence. Primary retains only the
original risk-eligible maximum-admission/global-growth-tied beta set and takes
its smallest refit-feasible beta under the unchanged 9-price cap. It does not
search all eligible betas, relax alpha or choose a method from OOF outcomes.
The first implementation attempt stopped at Q1 diagnostic export because
`floor` was supplied twice to a dictionary constructor. Its source, protocol,
progress and failure log are preserved separately. A dictionary-merge repair
and realistic diagnostic fixtures preceded the same-rule rerun; the data had
already been processed and the rerun is not first exposure.

Both methods admit the full 120 descriptive fit. Outer canonical admits two
fits/48 paths with 2 events; primary admits three/72 with 5 events, keeping 48
NULLs. Shared 48 states/actions/financials are exactly equal; the added 24 have3
events. On primary's 72, the annual/pending reference has 2 events; primary
mean annual log growth is higher by .00378616, mean MDD by .0240213, nominal fees
lower by .348468 per initial 100, and trade share lower by .746759. These tradeoffs
are reported together;5/72 is not a fixed-policy certificate or a per-fit10%
promise. Full-fit products are 25/30/45,45/5/50,65/5/30; E remains Growth at H6,
with no inheritance of C-H30 dynamic evidence. A/D have no strict172 risk
candidate under this selected gamma=.075 and their own H/L, a scoped empirical
statement rather than impossibility at every service reference or portfolio.

An independent audit passes 538,236 checks, including 541,800 joint feasibility
cells, Q1 own-H records, both selectors and their complete attempts, 1440 saved
strategy paths and 91 active Q states. Same-source isolated replay completes
all 30 tasks;18 outputs are byte-exact and progress differs only in explicit
runtime metadata. A separate fixed-state diagnostic replays 158 branches at 79
active predetermined states. Its six outer points provide no conditional
probability calibration or population forecast-accuracy estimate; current-action
then B remains distinct from learned adaptive continuation.

496 tests pass. Read-only summaries add admission partitions, matching state
receipts, own-horizon P10/volatility/tail metrics, paired financial comparisons
and visually checked attributed figures. The updated executed notebook is a
research explanation. No new final scenarios, final-policy approval, submission
paper, CMT upload or modifications to V1 source/report/output were produced.


## 2026-10-18 — New-admission risk, repeated complete fits and forecast assumptions

The three follow-ups were specified separately before their numerical access.
The incremental study reads only locked old traces and keeps all24 added market
IDs under the old second outer fit's40/25/35 target. Relative to same-target
annual/pending B, all24 have higher MDD,18 higher G and22 lower fees; events
are3 versus1. Mean MDD is27.97% versus24.92% and empirical fractional-tail
CVaR90 is47.69% versus43.51%. Primary-only events35/112 have matching
crossing-year end weights but different wealth and whole-history peaks. All72
three-arm histories reconcile; no peak reset or skipped initial fee was found.
The identity relating wealth/peak ratios to current drawdown is bookkeeping,
not a causal attribution or a cash-weight repair claim. The archive passes
20,172 consistency checks and a byte-exact read-only rerun.

The stability study completes all125 jobs in five fixed repeats,25 outer fits,
with unchanged joint Q1, both selectors and the original9-price family. Outer
training n96 and dynamic-inner n72 never vary. Primary automatic counts are
0,48,120,72,48 of the same original120 per repeat; eventsNULL,8,12,7,3.
Canonical counts are0,24,48,48,48. All25 final C labels areBalanced. Primary
admits12 fits, rejects7 at inner eligibility and6 at final top-E feasibility;
canonical admits7. Common168 scenario-repeat financial records are exact;
added120 records have14 events and reuse92 original market IDs. Initial
admission A(S,C) precedes evaluation returns and applies uniformly to each
fit's24 held-out paths. No look-ahead path screening,600 independent samples,
customer-population coverage, binomial fit confidence interval or best-seed
selection is inferred. Of12 primary admitted fits,8 have zero selected-training
event slack and4 have one event. This is in-sample integer feasibility only.

The independent stability audit passes1,801,082 checks and independently
replays1656 actual plus2208 inner saved accounts,115920 annual decisions and
316 active prespecified Q states. A further17 Q points are pending skips.
The 2624 actual raw-menu static fits provide2,256,640 joint candidate/profile
mask cells; c505_inner4 has no selected gamma and therefore no unsupported
final raw menu. All125 jobs are complete. The audit does not rerun global menu
search or unrecorded nine-price training state trees, and no new full isolated
numeric replay is claimed. Read-only summaries reconcile3095 saved cells;
all five new plots were rendered and actually viewed.

The fixed lag-state ridge comparator uses train-only calendar centering,
96x29 residual channel scales and penalty1 without a search. Original-path
averaged standardized MSE falls0.938926% for the next-year three-real-asset
vector and0.109113% for remaining-period averages. Inflation is separate.
The independent augmented-least-squares audit passes423182 checks. Equal-fold
OOF mean-bias cancellation and finite empirical interval coverage are not
conditional calibration evidence. This mean predictor supplies neither joint
suffix distributions nor verified action/tail probabilities and was not
inserted into trading. Independent claim review corrected a scale-description
error before the focused notebook was executed.

The new companion notebooks/admission_stability_forecasts.ipynb executes all
five code cells successfully; its HTML embeds the inspected plots. The old
notebook and V1 source/report/output remain unchanged. The in-app browser's
local-file security restriction prevents a rendered HTML browser check; no
workaround was attempted. Actual PNG inspection and execution are the verified
scope.540 tests pass. Current judgments retain the learned controllers as
research comparators. A separately reviewed simplification diagnostic is being
designed after these results; it is not an already approved final Q2 strategy.
No new final evaluation, submission paper, final-policy approval or CMT upload
was produced.

The phase Git-index receipt confirms168 exact source/input/output files. This
is not a fresh-clone or full numerical replay. Seven frozen forecast CSVs
retain their original CRLF endings; staged diff-check notices are limited to
those line endings, with no trailing spaces/tabs or source/Markdown/SVG issue.
The checkpoint checks94 saved snapshots and698 distinct hashed files.


## 2026-10-18 — Simplified execution controls, full gates and exact replay

After the earlier development results were known, a separately reviewed
protocol froze two diagnostic architectures before their new numerical run.
The ready source/protocol was committed asba111c4. Both reuse125 exact Q1
component bundles from the parent study, whose raw-menu audit scope is2624
fits; this stage doeszero new static refits. It performs125 new Q2 training
jobs on the same five fixed partitions. Candidate bands0/5%/10% use pre-return
TV drift, mandatory charged initial restoration and full-history pending hold.
Annual control has only band0. Fixed alpha10%, complete-training cap no worse
than B, one global G tie band and exact severity/actual-fee/order ties precede
an independent same-U inner risk gate. Review outcomes stayNULL; references
never replace service. Each outer fit gives one A(S,C) decision to24 returns.

Explicit source guards validate every cached target/name/local/global index
against its Q1 raw menu. New pure controls and125-job boundary fixtures pass,
including independent100-step transfer-fee accounts and altered-target rejection.
The analytic destination-transfer bound is d/(1+tau)<=m<=d, fee fractiontau*m.
Inside5/10% bands this bounds an avoided hypothetical restoration fee by
1.25/2.5bp at the stated rate. Actual triggered trades can cost more; a same
state/fee example with opposite future equity returns proves drift alone does
not establish benefit>cost. These are technical diagnostic thresholds, not
identified preferences or optimal transaction rules.

All125 Q2 jobs complete in6.11s. Annual control admits120,120,120,96,120 paths
with9,8,7,10,7 events; TV admits96,96,72,72,72 with8,7,6,5,4. Training feasible
families include B, so absence of final candidate failure is imposed rather
than evidence of safety. Annual24/25 andTV17/25 fit admission have separate
inner gates; annual's one rejection exceeds the absolute inner cap, TV's eight
reject by one extra same-U B event. Common-U membership agrees for both
architectures. Fourth-repeat annual automatic10/96 differs from hypothetical
full B10/120; the rejected24 havezero B events, a cohort-denominator effect
which cannot justify outer-based gate changes or financial imputation.

TV versus B on matching automatic groups saves fees and increases mean G
slightly, but meanMDD rises in all five repeats and events are8vs7,7vs7,6vs5,
5vs5,4vs2. Versus the old learner on common automated paths, TV instead lowers
both G and MDD, with fees changing sign. No method, seed or threshold is chosen
from these observed tradeoffs. Neither simple architecture estimates per-trade
benefit versus cost, so both remain diagnostic controls rather than a complete
final Q2 proposal.

The independently frozen audit passes5,100,120 checks, covers28800 training
metrics,4800 inner method rows,2400 inner B rows,2400 outer rows and9240 saved
physical accounts. Its exact-key scalar cache replays4280 unique accounts,
128400 transitions; maximum discrepancy is7.96e-12. This caches identical
physical calculations, not independent market observations. No numerical or
audit implementation failure occurs. The source/input closure replay also
passes on its first attempt:45 copied files,125 tasks,15 deterministic
artifacts byteexact and18540 named NPZ arrays exact. Only declared runtime
metadata is normalized; original20 study/audit files remain byte-identical.
This is a same-source closure copy, not a fresh Git clone or new market test.

603 tests pass in42.975s. The separately executed simple_execution_diagnostic
notebook retains all repeats, NULLs and construction facts. The actual PNG was
inspected after fixing bar-label collisions. Browser HTML review remains
unavailable; executed output and static HTML checks are recorded without
claiming a full-page rendered review. All V1 source/report/output and prior
research remain intact. No new final market evaluation, final architecture
approval, submission paper or CMT upload is produced. The next modelling
question is an explicit, inspectable benefit-cost estimate with the simple
controls retained as benchmarks, rather than a tuned alpha or best seed.

The final companion executes9 code cells withzero errors; static HTML checks
cover13 tables andone embedded inspected PNG. The final checkpoint checks109
saved snapshots/755 distinct files, and the phase Git-index proof binds93
source/input/output files exactly. One whitespace-only blank line in the
reproduction test is retained because its bytes are already bound by the
successful replay receipt; it is the sole staged diff-check exception.
