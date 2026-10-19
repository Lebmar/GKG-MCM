# SafeInvest: cost-aware advice with explicit limits

2026 HKUST(GZ) MCM Senior Group Problem A. Team **Lebmar**, number **87**.
Author: **Chengxuan YUAN**.

**Submission closure (2026-10-19):** the task-facing numbers and metric definitions
are in [TASK_ANSWER_SHEET.md](submission/TASK_ANSWER_SHEET.md), the Q3 sensitivity
note in [Q3_SENSITIVITY.md](submission/Q3_SENSITIVITY.md), and the authoritative
vs superseded document map in [AUTHORITATIVE_INDEX.md](docs/AUTHORITATIVE_INDEX.md).
Read the latter before quoting an older report or registry. The one-page
onboarding flow and 5×3 product screen are in
[Q1_ONBOARDING_FLOW.md](submission/Q1_ONBOARDING_FLOW.md) and
[Q1_PRODUCT_COMPARISON.md](submission/Q1_PRODUCT_COMPARISON.md); the V2b report
draft is [main.pdf](report/v2b_submission/main.pdf) with
[LaTeX source](report/v2b_submission/main.tex).

**Current technical version: the simpler V2b mitigation model is locked, and its
separate nonblind final evaluation, independent audit and isolated replay are complete.**
Main C initially funds 60/60 paths; 2 breach the original 40% annual real-MDD
limit. All 60 trigger human referral by decision year 7 and then follow the
modeled hold assumption. Ordinary autonomous service covers 19.44% of
account-years. This is a risk–growth–service tradeoff, not a demonstrated
autonomous 30-year service.

Start with the [new model registry](docs/mitigation_main_model_registry.json),
[technical judgment](docs/mitigation_technical_completion_judgment.json),
[development reading copy](notebooks/technical_rebuild_2026_10_19.executed.html),
[separate final reading copy](notebooks/mitigation_locked_final_2026_10_19.html),
and [new final evidence](results/mitigation_locked_final_evaluation_v1/).
The development notebook retains9 source-backed figures and runnable cells.
No paper has been regenerated in this phase; the existing report remains an
archived prototype. Historical final exposure remains disclosed: this is not a
fresh blind confirmation sample, and no parameter was adjusted after this run.

| Fixed Q2 object (same60 IDs/H30) | Median real CAGR | Median real MDD | P10 real CAGR | P90 real MDD | Breaches | Trade periods | Ordinary service |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2b protective trade then review/hold | 2.35% | 20.02% | 0.87% | 28.69% | 2/60 | 21.22% | 19.44% |
| Same-target annual/pending B | 3.56% | 27.49% | 1.50% | 43.54% | 8/60 | 94.56% | 94.56% |
| Same-target continuous annual | 3.56% | 27.49% | 1.48% | 43.54% | 8/60 | 100% | 100% |
| Equal-weight baseline (exact thirds, annual) | 3.38% | 22.29% | 1.47% | 39.36% | 6/60 | 100% | 100% |

The equal-weight baseline is the required 1/3-each annual-restoration strategy.
Trade periods = 382/1,800 = 21.22% of all decision periods, including the mandatory
initial allocation; excluding the initial allocation it is 322/1,740 = 18.51%.
Ordinary autonomous service = 350/1,800 = 19.44% and is a different metric. The
V2b policy buys lower tail risk and fees at a cost of median and P10 growth; it is
not a claim of all-metric dominance.

All financial paths keep original CPI, horizon, peak and absorbing breach history
after referral. Initial Review is financialNULL; funded review/hold is fully
accounted. Fees use the original destination-buy cost once. The final severe
example is synthetic ID161, selected by maximum main full-horizon MDD with
smallest-ID ties. A one-year supported action is not a recursive or real-market
safety guarantee. Q1 assignments are **A Review, B Conservative, C Balanced,
D Review, E Growth**. Static B/C/E product references retain adverse tails:
B 11/60, C 8/60, E 5/60 with P10 −1.58%; they are not autonomous execution
approvals, and the 10% reference is a declared service convention, not a contest
mandate or customer guarantee.

The final audit independently checks720 complete accounts and every main active
decision; an isolated copy reproduces all11 output artifacts exactly. The source
and release hashes preceded execution. A mistaken test filename prevented the
intended pre-run Git commit; its late completion is explicitly recorded in the
[engineering incident](docs/mitigation_final_precommit_incident.json). These
implementation checks do not establish predictive validity.

For compact technical reproduction see [SUBMISSION_README.md](SUBMISSION_README.md).
The additive adapter defaults to constructed synthetic data; only explicit
`--development` reads IDs1–120, reproduces the locked four Q2 objects and checks
canonical array fingerprints. The final-table adapter and the fixed
`--reproduce-fixed-nonblind-final` command are documented in
[FINAL_REPRODUCTION.md](submission/FINAL_REPRODUCTION.md). Historical runners keep
their full evidence gates.

The final closure package is
[Team87_Lebmar_Technical_Source_v4.zip](dist/Team87_Lebmar_Technical_Source_v4.zip):
it keeps the V2 compact core unchanged and adds the Q1 5×3 screen, the one-page
onboarding flow, the answer sheet, the Q3 note, the report draft/source and the
authoritative index, still far below the 50 MB CMT limit. The report is uploaded
separately; the ZIP is the supplementary/source package.

Earlier authoritative registry/final records below are retained historical
evidence and are superseded for the new V2b evaluation only.

This repository records a reproducible investigation of purchasing-power risk,
customer suitability, selective rebalancing and bounded target updates. It does
not assume that a more complicated policy improves investment outcomes.

## Historical final evaluation of the earlier model

The separate [execution release](docs/final_evaluation_release.json), audit,
reproduction and paired-analysis plans were fixed before this run. All 60
original IDs121–180 are retained. No new fitting, parameter selection, target
change or admission decision occurred. The main C service has0 automatic
paths and 60 pre-investment Review decisions; all its financial metrics are NULL.

The following are research comparisons at C's 30-year horizon and 40% annual
real-MDD limit. The shadow is the fixed candidate that failed the development
execution gate; its returns never substitute for the main service'sNULLs.

| Object | Median real CAGR | P10 real CAGR | P90 real MDD | Breaches |
| --- | ---: | ---: | ---: | ---: |
| Main C service: Review | NULL | NULL | NULL | NULL |
| Unapproved fixed shadow | 3.73% | 1.28% | 45.03% | 11/60 |
| Same-target annual/pending B | 3.56% | 1.50% | 43.54% | 8/60 |
| Exact-thirds annual reference | 3.38% | 1.47% | 39.36% | 6/60 |

The shadow trades less and pays lower average nominal fees, with higher
central growth but worse lower-tail growth and drawdown tails. Against B on
the same 60 IDs, growth is higher on 45 paths but MDD is larger on 53; three
additional paths breach and no B-only breach disappears. Both references and
all adverse outcomes remain in the [paired analysis](results/final_locked_analysis_v1/)
and [reading figure](research/figures/locked_final_evaluation/paired_growth_drawdown_and_service_status.png).
P10/P90 use the empirical inverse CDF; medians use the ordinary median. MDD
uses annual endpoints and does not measure intrayear losses.

Q1's own-horizon static annual references also retain adverse findings:
B (15 years, L=25%) has 11/60 breaches, C (30 years, L=40%) has 8/60, and
E (6 years, L=35%) has 5/60 with P10 real CAGR −1.58%. B/C exceed the 10%
empirical service reference; E's lower breach frequency does not ensure positive
growth. A/D remain Review.
These are suitability/product references, not automatic execution approvals;
their different horizons and limits do not permit an equal-horizon risk ranking.

The [independent final audit](results/final_locked_evaluation_v1/independent_audit.json)
passes 81,911 checks, reconstructing all 360 available physical accounts and
independently recomputing nine predeclared scalar-fan Q witnesses. The
[isolated replay](results/final_locked_reproduction_v1/) reproduces all six
producer artifacts byte for byte and all 234 NPZ fields exactly. A separate
paired arithmetic review passes 1,030 checks with zero differences. The new
[final checkpoint](results/final_locked_checkpoint_v1/verification.json)
passes 942 unit tests and verifies 1,055 distinct files; the previous development
checkpoint remains unchanged. These AI-assisted numerical checks establish
implementation/reproduction evidence, not calibrated future risk.

Old V1 had already exposed these final scenarios before later development.
This unchanged, released final reuse cannot be described as a fresh blind
confirmation sample. Synthetic scenario frequencies are not a guarantee that
future losses exceed a customer's limit at most 10% of the time. Paper,
submission packaging and CMT upload remain separate, unfinished delivery tasks.

## Development history

The October19 mechanism study reconstructs11 fixed model/reference pairs
for original IDs35/52/98/112. All22 complete accounts retain the original
30-year clock, CPI, real peak and absorbing breach history. Higher terminal
wealth can coexist with greater drawdown because the historical peak also
changes. Independent accounting checks pass; crossing-year target equality
does not remove the effect of the earlier path.

Replacing only the first model/reference action disagreement, then letting
the unchanged policy replan from each branch's own state, removes none of
the11 breach/first-crossing outcomes. Nine pairs attain proportional holdings
and peaks before their eventual worst drawdown; all later normalized states,
actions and Q values agree. This post-hoc positive-homogeneity check explains
their equal MDD despite different wealth. The other two pairs remain visible.

One metadata-matched full112/outer112 pair isolates a particular forecast-fan
change at the same original year5 state, target45/5/50 and eta.03. Full120
and a fixed-state119 fan deleting actual112 restore; outer96 holds. All335
paired members/670 action branches pass independent scalar checks. The result
does not attribute the complete future loss to that decision or establish
conditional forecasting accuracy; deleting self alone does not flip it.
A [post-hoc member identity review](docs/matched_fan_tail_vote_identity_review.json)
shows that original95 supplies the sole full120 hold-versus-restore event
difference at this state. The original outer fan omits95; all24 omitted
members and their growth/event outcomes remain visible. No95-only deletion,
new subset choice or recommendation to drop an influential market is made.

An exact baseline bridge reuses five known-fee-rate experiments for the
current rejected fixed candidate, after checking all48 candidate NPZ fields
and both reference accounts. Candidate breaches across0/.125/.25/.5/1 percent
fees are10/9/9/10/9 out of120. Fees can change actions, so risk is not monotone.
These are archival shadow outcomes, not admitted service returns or approval
robustness. Current C still requires Review with financial outputsNULL.

The new [executed mechanism companion](notebooks/failure_path_development.ipynb)
and [reading copy](notebooks/failure_path_development.html) collect four figures,
all fixed cases and the exact scope of each audit. The [model contract](docs/current_model_contract_2026_10_19.json)
also reconciles Q1/Q2/Q3 requirements, a severe originally admitted outer case,
and a44-word customer explanation. This is model specification, not a paper.

The [locked evaluation preparation](docs/final_evaluation_protocol.json)
now fixes all primary, shadow and reference objects before final data. Its
[development reproduction](results/model_review/final_evaluation_development_gate_v1/)
matches all120 candidate/reference fields exactly. Independent saved-array
review checks369,371 values,480 Q2 rows,600 Q1 rows and nine summaries; all
unserved financials stayNULL. The [preparation judgment](docs/final_evaluation_development_gate_judgment.json)
states the scope and preserves the first auditor's metadata-adapter failure.
Only original1–120 were parsed at that preparation checkpoint. The separate
final release did not yet exist; the development gate itself never opens121–180
or changes admission. The later final release and results are indexed above.
Read the adapter's fixed
objects without parsing market values:

```sh
.venv/bin/python scripts/prepare_final_evaluation.py --describe
```

The prior evening session localizes the admission differences by original
market ID: full-training9 to inner10 consists of8 shared events, ID98 leaving,
and IDs35/112 entering. On the same inner paths B has8 events, with the two
model-only events35/112. Targets and training fans also change, so this is
not an isolated test of eta. The [720-row identity ledger](results/model_review/admission_failure_identities_v1/)
preserves all120 full/inner and600 repeated outer records, including504
non-common automatic records with NULL model financials and differences.

A separately frozen fixed-state diagnostic checks39 original witness slots
from13 archived fits and8 distinct markets. Of35 active states,3 have at least
one action flip after deleting a single fan member;21 of3072 OOF deletions
flip, while3 full-training self deletions do not. Two sensitive states are in
year30 and involve near-break-even net growth rather than future continuation.
Same-current-action B and saved adaptive continuations differ in growth/MDD,
but none of these35 witnesses changes its terminal breach flag. All3432 fan
members/deletions and relevant account states pass245925 independent arithmetic
checks. These witnesses exclude the failure IDs and are neither learner LOOCV,
conditional calibration nor an explanation of those failures. Current C stays
Review. See the [diagnostic judgment](docs/fan_continuation_diagnostic_judgment.json)
and [next-session handoff](docs/overnight_handoff_2026_10_18.json).

The benefit-cost investigation retained two failed simplifications openly.
Strictly positive predicted financial-growth trading produced31/120 complete
training events; a single local constrained repair produced25/120, against a
same-target annual/pending reference cap of9. Both were prospectively frozen,
independently audited and replayed in isolated source/input copies. Neither
was relabelled as a successful strategy or rescued with a new alpha/price.

The main framework therefore retains the previously developed complete-policy
constrained learner, followed by one prospectively fixed additional guard.
Its existing target, selected beta/eta and top-E/refit rule are immutable.
Only the old selected beta's inner predictions are checked against B on the
identical automatic set U; failure means Review without another candidate.
This is an openly subsequent development choice, not a retrospectively
prespecified winner. Eta is a technical finite-controller index, not an
identified customer preference or a KKT multiplier. The action score uses
already-net Q values, including current and future continuation fees, and
does not subtract another transaction cost. Risk-reducing trades may sacrifice
predicted financial growth and are explicitly labelled as such.

The final full120 fit passes the absolute inner count cap12 but has10 events
against B's8 on the same U, so its automatic execution is Review. That two-event
difference is not claimed statistically significant: the rejection implements
a conservative empirical norm, stricter than the problem explicitly requires.
Across five dependent repeats the added guard retains0,24,0,24,48 automatic
path records, withNULL,5,NULL,1,3 events. On common automatic groups it saves
fees and increases mean growth but has more events and larger meanMDD than B;
one repeat has5/24=20.83%. The96 admitted scenario-repeat records involve74
original IDs. Neither a risk guarantee nor independent96/600 observations
is claimed. See `results/model_review/global_policy_admission_v1/`.

The development work below uses scenarios **1–120 only** and keeps the V1 archives intact.
Start with `notebooks/objective_risk_foundations.ipynb`, then the
[executed development notebook](notebooks/risk_menu_development.ipynb).
A [reading copy](notebooks/risk_menu_development.html) hides code cells;
it is a research note, not a submission
report. The notebook separates the growth objective, risk-budget policy,
sampling evidence, liquidity assumptions and the actual execution rule.

The 2026-10-18 follow-up freezes three separate diagnostics. The added24 paths
from the old second outer fold use the same40/25/35 equity/bond/bill target
for the learned controller and annual/pending reference. All24 have larger
MDD under the controller;18 have higher annual log growth and22 lower fees.
Events are3 versus1, mean MDD27.97% versus24.92%, and empirical CVaR90 of
MDD47.69% versus43.51%. All24 remain visible, including the two primary-only
crossings. Their crossing-year end weights agree, but wealth and historical
peaks differ. Complete state reconstruction confirms the original peak and
absorbing event history; this is not a causal attribution to any one trade.
See `results/model_review/incremental_admission_risk_v1/`.

Five newly fixed complete-refit partitions use125 training tasks. Primary
automatic path counts are0,48,120,72,48 out of the same original120 in each
repeat; conditional event counts areNULL,8,12,7,3. All25 final fits keep C
Balanced, but only12 admit execution:7 have no top eligible inner budget,
6 have no feasible final controller within that set. Of12 admitted fits,
8 select a controller exactly at its training event cap and4 have only one
event of slack. The canonical rule admits7 fits; the common168 scenario-repeat
financial records are identical. These are overlapping fits and repeated
market records, not independent deployment trials or600 new scenarios.
Admission is determined from training data and C before evaluation returns;
each fit applies one initial decision to all24 held-out paths. The new rules
remain research comparators, with no stable10% guarantee. See
`docs/refit_stability_protocol.json`,
`results/model_review/refit_stability_v1/` and
`results/model_review/refit_stability_summary_v1/`.

The fixed previous-year-state predictor uses the original five whole-path
folds, train-only calendar centering and pooled residual scales, and fixed
ridge penalty1. Three-real-asset path-averaged standardized MSE falls by0.939%
for next-year quantities and0.109% for remaining-horizon averages. Inflation
is a separate channel. These small prediction-loss differences do not validate
conditional trading values or tail probabilities and do not select a new
controller. The diagnostic and independent augmented-least-squares audit
are under `results/model_review/forecast_assumptions_v2/`; judgments are in
`docs/forecast_assumption_diagnostic_judgment.json`.

The follow-up simplification diagnostic is frozen before its own numerical
run, after the prior development results were known. It reuses125 matching,
audited Q1 components and runs125 new Q2 tasks. Annual positive-control
admission counts are120,120,120,96,120 with9,8,7,10,7 events; TV-band counts
are96,96,72,72,72 with8,7,6,5,4 events. The families include annual B, making
supported training feasibility an imposed property. An independent inner gate
still rejects one annual and eight TV fits. On each TV automatic cohort,
growth rises modestly and fees fall relative to B, but mean MDD rises and
events are8vs7,7vs7,6vs5,5vs5,4vs2. No risk dominance, best architecture or
future10% certificate is inferred. The TV rule also lacks the per-trade
benefit-cost estimate required of a complete final Q2 solution.

`docs/simple_drift_cost_interpretation.json` proves the immediate fee bounds
and gives a same-drift/same-fee counterexample to a benefit claim. The study,
independent scalar-account audit and same-source isolated replay are under
`simple_execution_comparison_v1/`, `simple_execution_summary_v1/` and
`simple_execution_reproduction_v1/` inside `results/model_review/`. Replay
matches15 deterministic artifacts and18,540 named NPZ arrays, with only
declared timing normalization. This is a source/input closure copy, not a
fresh Git clone or another market sample. Replay without overwriting evidence:

```sh
python scripts/reproduce_simple_execution_stage.py --output results/model_review/simple_execution_reproduction_new
```

The static study declares its procedure in
`docs/risk_menu_development_protocol.json`: 232 fixed targets, three risk-budget
cases, three reserve-policy cases, five whole-scenario folds and 1,000 complete
bootstrap reselections. Results, diagnostics and independent accounting replay
are under `results/model_review/risk_menu_v1/`. The primary 10% sample-screen
menu is a relaxed research comparator: its B/C/E assignments all sit on the
12/120 training boundary, while precise weights vary substantially on resampling.
The 232-point universe permits zero weights. Since the statement says each
product uses all three assets, a separate 172-point strictly positive subset
is examined under the same nested procedure. Its 5% minimum positive weight
comes from the fixed grid resolution, not an elicited risk or liquidity floor.

Nested calibration keeps the service reference alpha at 10% and selects a
technical training screen gamma inside four inner whole-path folds. It does
not select gamma using outer evaluation results or turn empirical calibration
into a probability guarantee. Raw and effective assignments, admission evidence
and every denominator are saved separately.
Agreement on gamma does not identify exact weights: the strict Balanced target
holds 15–25% bonds across the five outer training fits, versus 5% in the full
120-path descriptive fit. These overlapping-fold ranges are not confidence
intervals and do not support freezing one precise allocation yet.

Complementary diagnostics are recorded separately. The finite-family
Clopper–Pearson study requires an additional iid scenario assumption and is
deliberately conservative; it does not certify real markets or dynamically
learned policies. The fixed-target annual-versus-buy-and-hold ablation shows why
a suitability result cannot be inherited after changing the execution rule.
Using frozen targets and assignments from the relaxed232 primary study,
the finite-horizon prototype corrects a one-period cost-gate counterexample,
but its growth-only decisions increase B/C/E out-of-fold exceedances to
26/40/14 out of 120, versus 12/13/11 under annual restoration. It therefore has
not passed risk suitability. Fee-instant observations and fixed within-path
year-order stresses examine additional measurement and modelling assumptions.
None of these diagnostics selects a new final service policy.

A separate interval-product argument bounds unavoidable annual drawdown even
under zero-cost perfect foresight. It shows why failure within a finite set of
fixed targets is not proof that every dynamic annual policy is infeasible.
The ideal asset-switching path is a mathematical relaxation, not advice.
The Q2 study also records the exact-thirds annual baseline, pooled path
quantiles, actual trade fractions and one explicitly selected severe episode.
The original proposed referral contract remains unchanged. Its executed
post-crossing hold study has a separate protocol and archive: it preserves the
first crossing and full-horizon exceedance event exactly, while subsequent loss
severity and recovery can change. Missing adviser-response data are not replaced
with invented returns or response times.

The strict172 execution studies then compare unconditional calendar-fan growth,
mean-risk and pathwise guards. A synthetic counterexample shows why a local
Pareto comparison using an unconditional fan need not improve full-path risk.
The stronger pathwise guard has a narrower support-dependent argument, but makes
no additional hold decisions at any of five known fee rates. These are separate
mathematical, numerical and development-data results.

A finite-family learner selects a risk-adjusted trading price using training
paths, with both a 10% empirical cap and a declared no-worse-than-reference
training constraint. It improves growth and fees in the pooled outer-fold
comparison, but B/C/E exceedances are **10/14/10 of 120**, versus 9/9/9 for annual
restoration followed by post-crossing hold. Customer C therefore exceeds the
10% reference frequency. Training admission does not establish future safety.
Allowing moves among the three frozen targets does not help the predeclared
training selector: all six contexts select the fixed-target controller. This
does not show that every possible weight-update method is valueless.

Current C comparisons, the exact-thirds annual benchmark and a severe episode
selected by the original annual rule are under
`results/model_review/q2_execution_development_v2/`. Known-fee sensitivity keeps
the default-cost selected controllers frozen; it is recorded separately under
`results/model_review/selected_controller_cost_stress_v1/`. None is a final
service recommendation or submission report.

The first complete-process study nests the entire fixed-target dynamic learner, including the
static menu and gamma selection, within four inner folds per outer training
set. Thirty fixed training tasks evaluate five dynamic training budgets and
the same nine-price family; outer paths select neither budgets nor prices.
It preserves risk Review, outside-Balanced cases and execution Review separately,
with NULL financial results and the original 120-path denominator. Same-target
annual/pending and exact-thirds annual policies remain hypothetical references;
they never become a hidden fallback for rejected customer paths.
The protocol and results are in `docs/nested_dynamic_calibration_protocol.json`
and `results/model_review/nested_dynamic_calibration_v1/`; readable comparisons
and selection stability are under
`results/model_review/nested_dynamic_calibration_summary_v1/`.
For the same client C, two of five overlapping outer training fits admit automatic
execution, corresponding to 48/120 evaluation market paths; three abstain,
leaving 72 financial NULLs. This is not 40% customer-population coverage.
There are 2 breaches among the 48 observed automatic paths. On those same 48 paths each reference has
1 breach. Growth and fees improve relative to same-target annual/pending,
while drawdown rises; reduced coverage must not be presented as universal risk
improvement. The full 120 descriptive fit serves all paths, so parameter and
coverage stability remain open. See `docs/nested_dynamic_stage_review.json`.

The historical risk-only study did not incorporate liquidity. Its design record
`docs/liquidity_service_design.json` distinguishes ordinal questionnaire labels,
customer facts that are missing, and normative reserve assumptions. It does
not invent withdrawal amounts or turn a target bill share into a guarantee of
cash availability. No new liquidity gate or reserve floor was inserted into
the dynamic experiment.

The current refinement distinguishes finite technical-parameter decision ties
from statistical identification or elicited customer preferences. The saved
training-only ledger under `results/model_review/refit_admission_diagnostics_v1/`
explains why inner eligibility can disappear after a target, training fan and
sample are refitted. The second outer fit has objective-tied beta values;
the fourth changes the admitted validation cohort. The old results remain
unchanged, including all rejected-path NULLs.

A separately frozen joint-suitability prototype uses normative target-bill
floors High 30%, Medium 10%, Low0 in **every** static inner menu/oracle and refit.
The short questionnaire is sufficient for that stated convention; a real
service would separately clarify cash amounts and dates. These floors are
neither data-identified needs nor live-balance or withdrawal guarantees.
`docs/liquidity_policy_refinement_review.json` corrects the earlier overbroad
missing-information design. The pure `service_governance.py` prototype checks
versioned evidence/fit lineage and preserves historical risk state; a version
fingerprint alone is not a completed governance admission receipt.

The two final selectors in `docs/joint_liquidity_refit_protocol.json` share all
30 training tasks and inner evidence. The original canonical minimum-beta rule
is compared with a predeclared primary retaining the top inner objective tie
set and choosing its smallest refit-feasible beta. It never uses a beta outside
that set, relaxes the original risk cap, or imputes reference returns. Outputs
are in `results/model_review/joint_liquidity_refit_v1/`; read-only comparisons
are in `results/model_review/joint_liquidity_summary_v1/`. Only original C,
H30/L40%, receives Q2 dynamic evaluation; the other profiles retain their own
Q1 horizons. The first implementation attempt stopped on a Q1 diagnostic-export
dictionary error; its source, protocol and failure log are preserved under
`results/model_review/joint_liquidity_refit_attempt1_incomplete/`. The rerun
repairs that export and its realistic diagnostic fixtures, with no economic or
selection-rule change. This is not first exposure to the data.

In the joint stage, the canonical selector admits two of five overlapping
outer fits (48/120 evaluation paths,2 events); the retained-set selector admits
three (72/120,5 events). Their shared 48 paths have exactly equal saved states
and actions; the newly admitted 24 have 3 events. This is an admission change,
not an investment improvement on the common subgroup. On its own 72 common
paths, the primary has 5 events versus 2 for the annual/pending reference;
higher mean growth and lower trading/fees coexist with higher mean drawdown.
The full 120 fit is descriptive. Full-fit products are 25/30/45 Conservative,
45/5/50 Balanced and 65/5/30 Growth (equity/bond/bills); E remains Growth after
joint refitting rather than the Balanced assignment obtained by postfiltering
the old fixed catalogue. A/D Review is relative to the selected gamma=.075,
strict172 and own horizons/loss limits, not an impossibility theorem.

The fixed-state diagnostic under
`results/model_review/fixed_state_forecast_diagnostic_v1/` separately compares
hold/restore now followed by the same annual/pending continuation. Its six
outer states are nonrandom points from two fits, not a probability-calibration
test or population accuracy estimate. No new prediction model is selected.
The new joint stage has its own guarded, isolated replay:

```sh
python scripts/reproduce_joint_liquidity_stage.py --output results/model_review/joint_liquidity_reproduction_new
```

To replay the seven execution studies without overwriting their evidence:

```sh
python -m pip install -r requirements-research-pinned.txt
python scripts/reproduce_execution_stage.py --output results/model_review/execution_stage_reproduction_new
```

The pinned file records the exact direct dependency versions used here with
Python 3.12. The checkpoint also records the Python version and architecture;
indirect dependencies and byte equality on other platforms are not locked.

This command runs each frozen runner in a disposable copy, compares its saved
outputs exactly (named-array equality for NPZ ZIP metadata), then restores its
original frozen archive in the copy before replaying its child. Parent manifest
and audit timestamps are part of downstream hash contracts; do not rerun all
stages consecutively over the original archives. A fresh receipt needs a new
output directory. Same-source reproduction is distinct from the separately
implemented accounting audits and from statistical validation.

The newer 30-task stage has its own isolated reproduction command:

```sh
python scripts/reproduce_nested_dynamic_stage.py --output results/model_review/nested_dynamic_reproduction_new
```

This command records its explicit file closure and guarded loader calls. It
compares deterministic artifacts byte-for-byte, NPZ arrays exactly, and timing
metadata using the documented normalization. It does not overwrite the study
or replay the earlier seven stages.

Earlier source stages are indexed in their manifests: static menus and
bootstrap (`run_risk_menu_research.py`, `analyze_risk_menu_diagnostics.py`),
nested/strict calibration (`run_nested_static_calibration.py`,
`run_positive_asset_specification.py`), interval and simultaneous-risk studies,
and the finite-horizon, fee and temporal-order diagnostic runners. The numerical
cache contains only allowed development paths. Four frozen cache pairs are
retained in Git because later contracts include their original metadata hashes;
regenerating a timestamped cache in place would invalidate those contracts.
Other caches remain disposable. Source, data-array and output fingerprints
remain in the manifests. A new fit
requires a new protocol/archive rather than replacement of frozen parents.
Do not use V1 `final` or `reproduce` to select a revised model.

Notebook authoring tools are optional and listed in `requirements-review.txt`.
The readable HTML hides code cells; the notebook retains the code, outputs,
formulas and links to the recorded methods. Execution prototypes live separately
under `research/experimental_execution/`, with their own protocols and numerical
audits; they do not replace the primary procedure. The research preserves full
path folds, records referrals as null outcomes, and distinguishes empirical
calibration, numerical correctness and statistical evidence.

The preserved development checkpoint passes **897 tests** and checks 189 saved manifests,
audits and receipts against 1009 distinct source, input, result and execution
evidence files. The preceding benefit-cost phase's Git-index proof binds172
frozen package files. The
previous joint stage passes 538,236 independent selection/accounting checks and its
30-task isolated replay matches 18 deterministic output files byte for byte;
progress differs only in declared runtime fields. The first implementation
attempt's export failure is explicitly preserved. These are computational
checks, not independent market observations or risk guarantees.

The historical seven-study reproduction matches all 73 saved outputs. Its
tracked-files-only clone at financial-source revision 1f7018e passed 353 tests
and matched those 73 outputs; that old receipt does not claim to test the newer
joint stage. Current evidence is indexed in
`results/model_review/research_checkpoint_v1/verification.json`.
AI assistance, including separately implemented numerical audits, is disclosed
in `docs/DATA_NOTICE.md`.

The historical next-stage proposal `docs/dynamic_calibration_next_stage.json`
was implemented in the complete-process stage above: calibrate the complete
learning procedure in inner whole-path folds, refitting its menu and controller.
The proposed primary family keeps nine fixed-target price candidates; the
18-controller update family would be a separately declared incremental study.
More cautious admission may lead to more Review cases and still cannot turn
reused development results into a new blind test.
The historical requirement inventory is
`docs/problem_A_deliverable_crosswalk.json`. Its earlier liquidity and policy
gaps must be read alongside the current model contract and its independent
review. The preparation judgment records the remaining steps at that earlier
phase; the final evaluation and current unfinished paper/delivery work are
indexed above. The Balanced-only Q2 case stays distinct from Q1 product references.

Historical notebook-authoring recipe for an isolated copy, preserving the
frozen research artifacts in the working repository:

```sh
python -m pip install -r requirements-review.txt
python -m ipykernel install --prefix .venv --name safeinvest-review --display-name "SafeInvest review"
python scripts/build_risk_menu_notebook.py
JUPYTER_PATH="$PWD/.venv/share/jupyter" python -m nbconvert --execute --to notebook --inplace notebooks/risk_menu_development.ipynb
python -m nbconvert --to html --no-input notebooks/risk_menu_development.ipynb --output risk_menu_development.html
mkdir -p tmp
PYTHONPATH=src:research python -m unittest discover -s tests -q > tmp/research_checkpoint_tests.log 2>&1
python scripts/verify_research_checkpoint.py
```

## Archived V1 prototype

The 9-page report and 2.76 MB supplementary ZIP are in `output/`. They have not
been submitted to CMT. There are 31 passing unit/policy tests, 901 passing
independent numerical/provenance checks, and an exact 13-comparison reproduction
of the unchanged protocol in a temporary directory.

The frozen policy retains a fixed Balanced target. Its 30-year-horizon median
annual real CAGR is 3.66% versus 3.38% for equal weights, while P90 maximum real drawdown
is higher: 43.44% versus 39.52%. Dynamic target updates are not enabled because
the predeclared incremental utility criterion was not met. Customer C's
borderline loss-exceedance screen does not retain pointwise calibration on the
final scenarios. This is a qualified result and a reason for human confirmation,
not a claim of universal dominance or guaranteed suitability.

## Archived V1 reproduction

Use Python 3.12 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/reproduce.py
```

The reproduction command runs the unchanged study in a disposable directory and
compares numerical outputs against committed archives. It does not overwrite
the original evidence or change a choice after seeing final results. The three
individual study stages are also available for a fresh, archive-free copy:

```sh
python scripts/run_study.py develop
python scripts/run_study.py freeze
python scripts/run_study.py final
```

The committed result files are the archived study record. `freeze` and `final`
deliberately refuse to overwrite existing archives. For clean reproduction,
copy the repository to a separate directory and move its archived
`results/frozen` and `results/final` directories outside that copy first.
Do not remove or replace the original study record to tune against final data.

## Archived V1 design

- Three-asset long-only portfolios; 100 units initially in bills.
- Explicit self-financing fees of 0.25% per amount actually transferred once.
- Growth, volatility and maximum drawdown measured in purchasing power.
- 1–80 fit, 81–120 architecture validation, 121–180 final evaluation.
- Four unguarded comparisons isolate target choice, execution and updates;
  two additional variants quantify the effect of review/pending behaviour.
- Scenario-level paired bootstrap; no annual-row pseudo-replication.
- Final access requires frozen source/configuration/data fingerprints.

The initial inventory exposed pooled market descriptive statistics across all
180 scenarios. No final strategy comparison or parameter selection was done
then. This limitation is retained in the protocol; we do not describe the
holdout as completely pristine. The formal modelling pipeline gates all final
returns until the model is frozen.

## Archived V1 files

- `docs/protocol.json`: assumptions and selection rule fixed before validation.
- `src/safeinvest/`: accounting, loading, portfolios and causal policies.
- `tests/`: financial identities and timing/safeguard tests.
- `results/development/`: fitting and validation evidence.
- `results/frozen/`: final model, customer assignments and hashes.
- `results/final/`: held-out evaluation and sensitivity diagnostics.
- `docs/references.md`: verified primary citations and their claim boundaries.
- `report/`: English report source, figures and compiled review PDF.
- `output/pdf/SafeInvest_Team87_Report.pdf`: 9-page English review draft.
- `output/submission/Team87_Lebmar_Code.zip`: source and supplementary evidence.

To regenerate figures and the report (requires a local TeX installation):

```sh
python scripts/make_figures.py
python scripts/build_report.py
cd report/latex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ../..
python scripts/package_submission.py
```

`scripts/verify_results.py` independently replays archived actions and checks
the numerical tables and hashes; it does not call the policy simulator.

## Data and AI use

The supplied data and problem statement are synthetic educational material,
distributed under **CC BY-NC-SA 4.0**. See `docs/DATA_NOTICE.md` and the retained
source attribution. Raw historical JST data are not substituted for the contest
scenarios. The questionnaire's loss tolerance is not a guarantee.

OpenAI Codex assisted with methods, code, testing, analysis and English drafting.
Independent automated review is recorded; the human author remains responsible
for checking the report and submission. No CMT upload is performed by this repo.
