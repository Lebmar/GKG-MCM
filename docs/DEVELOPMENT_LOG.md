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


## 2026-10-18 — Benefit-cost estimates and main-framework admission convergence

The user requested that transaction benefits/costs be made explicit and the
main model/admission rule converge before paper writing. All new financial
work uses development scenarios1–120 only. Original joint-liquidity Q1 caches
are reused after exact context/target/local-original-candidate verification;
no static menu is refitted in this phase. The original full120 static context
has outer_test_ids=[], while its separate dynamic descriptive evaluation has
120 training IDs. A local validation wrapper preserves this distinction.

First, a prospectively frozen one-year positive-net-growth controller and
remaining-original-H sensitivity compare one trade then hold with hold. Gross
paired log advantage minus exact current log fee equals the direct paid net
ratio. Both apply a common full-H buy-and-hold risk-direction veto; deployment
requires positive initial economics, then mandatory initial execution. They
complete130 tasks but both reject all26 final fits: full120 training has31
events versus the same-target B cap9. Independent audit checks2,775,963
identities, all20160 training rows and260 fixed complete-policy witness slots;
256 witnesses replay and4 Q1/outside slots remain unavailable without replacement.
All3912 saved reference accounts replay; no learner has been admitted, so an
empty learner forecast subset is not relabelled as safety evidence.

The overly narrow financial-growth interpretation would prohibit some
risk-reducing restorations. One additional controller is frozen prospectively:
full-H current-action-then-B netQ; local alpha=.1 feasible actions maximize
netQg, with an explicit min-event-count/net-growth/hold least-violation rule
when both are infeasible. No price grid, alpha change or new architecture is
selected after that experiment. All130 tasks complete; full120 events improve
to25 but remain above9, and all26 fits stayReview. Independent audit checks
1,925,729 identities,10080 training rows,130 summaries,26 gates and3912 complete
reference accounts. Its272 planned learner Q slots are unavailable, not
substituted. A frozen experimental U40 initial diagnostic label clips one
character from its41-character literal; actions, fees and Qs are unaffected,
and the independent audit records it without changing frozen source.

These failures reject the two simplifications, not every feasible policy.
A public subsequent-development judgment retains the pre-existing
joint_tie_set_refit_learner mathematical framework. Within each beta it
maximizes complete-training G over feasible9 controllers; beta selection first
maximizes inner automatic coverage then conditional growth, with unchanged
top-E/refit feasibility. Eta is a technical finite-controller index, not an
elicited risk preference, uniquely identified structural parameter or KKT
multiplier. Finite scores Delta_netQg-H*eta*Delta_model_probability already
contain current and future branch fees; no extra fee is subtracted. Negative
or tied net-growth restorations must strictly reduce predicted event count;
risk_first has a separate integer lexicographic check. Growth sacrifices are
model opportunity costs, not realized losses or summable overlapping Q gaps.

A separately frozen final postfit guard makes no new fit, price/target/beta
choice or rejected-fit rescue. Only an old already-automated fit's exact final
beta original inner predictions define U; require positiveU, k<=floor(.1U),
and k<=same-U B with matching job/target/IDs. The26 fits retain13 old refusals,
add9 refusals and retain4 automated OOF fits. Full120 training has9 events,
but the selected-beta inner procedure has10/120 versus B8/120: absolute cap12
passes, comparative guard fails, so current C is executionReview with NULL
financial outputs. This two-event difference is not declared statistically
significant; the relative guard is an explicit conservative norm stricter
than the problem requires, not an inferred law or true-risk certificate.

The five dependent repeated OOF populations admit0,24,0,24,48 records with
NULL,5,NULL,1,3 events; same-outer-A B counts areNULL,4,NULL,0,2. One admitted
cohort has5/24=20.83%. Every nonempty same-A comparison has higher growth and
lower fees but higher meanMDD and more events.96 admitted scenario-repeat
records involve74 original market IDs.678 optional restorations comprise376
net-growth improvements and302 risk reductions with strict predicted growth
cost. All408 old selected accounts/12240 decisions remain separately labelled
archived development evidence; rejected financial results never fill new
service outcomes. The independent overlay audit checks605004 identities and
all26/2160/12240 records, maximum numeric difference2.22e-16, with parent
manifest and source/input/artifact hashes bound.

Two isolated source/input-copy numerical replays match12 saved artifacts each;
the final saved-data postprocess replay matches5 artifacts byteexact and its
manifest exactly without copying or reading a raw market CSV. These are
reproduction, not new market validation or fresh Git clones.725 tests pass in
156.254s. Research source/report/output remain unchanged from7bca10b.

The final research companion executes11/11 code cells withzero errors and
exports10 tables/3 embedded plots. Root actually inspects all three PNGs,
requests legend/title and NULL-label layout fixes, then reinspects/confirms.
The whole HTML browser view remains unverified; standard MathJax/require.js
CDN rendering is not claimed fully offline. An early checkpoint correctly
blocks an unexecuted companion; the notebook is completed before rerunning
the check, rather than weakening that guard.

The conclusion is convergence of estimands, mathematical framework and
admission rules, not proof of a successful autonomous C investment service.
Current C remainsReview. Forecast calibration and service stability remain
material limits; no paper rewrite, final121–180 reopening, CMT upload or
regenerated submission ZIP is performed in this phase.

Final checkpoint passes143 saved snapshots and877 distinct hashed files; the
phase Git-index proof binds172 frozen source/input/output/package-helper files.
Its sole new Git-style exceptions are446 Matplotlib SVG path line-end spaces
across three frozen figures. The builder strips EOF whitespace only; the
already-hashed figures and executed notebook are preserved, with a separately
fingerprinted explanatory note. All non-SVG staged files pass diff checks.


## 2026-10-18 evening — Fixed-state forecast diagnosis and overnight handoff

The user requested the last work session of the day. This bounded session
adds diagnosis only: no new controller, price/target/beta/alpha selection,
changed admission, paper rewrite or final-scenario evaluation.

An openly post-hoc ledger organizes previously observed audited results into
720 identity rows:120 full-training versus selected-beta inner records and600
repeated outer records. The original selected full controller has9 events;
8 remain in the inner procedure,98 leaves, and35/112 enter, yielding10.
Same-own-inner-target B has8 events and neither model event is offset by a
B-only event. The changing targets and training fans confound attribution to
eta or prediction. Outer model-only events are52 in repeat2 and112 in both
repeats4/5; repeated112 is not independent market evidence. All504 outer
non-common automatic records preserve NULL model financials and differences.
Root separately checks the original source-table event sets and ledger NULLs.
No influential scenario is deleted. An early ledger script draft had a
repeated keyword before output creation; its post-hoc review records the fix.

A new finite witness diagnostic freezes the exact13 originally automated fits
including those later rejected, each original audit scenario at years2/16/30.
The39 slots involve only8 original marketIDs. Source review and22 synthetic
checks precede numerical execution. The first root freeze attempt incorrectly
listed a nonexistent namespace-package initializer and aborted before protocol
mutation or market execution. The draft was committed as52d002f; the corrected
ready source freeze is6f12f49/protocol622cefcdaeabc7bdbaf0c54e02ce629bebe861448677aaeeec52549a21786066.
This history is preserved rather than retroactively calling the draft ready.

All39slots remain:35 active and4 pending skips without replacement. Each active
state retains original holdings/CPI/peak/MDD/H/target/eta and its exact original
training fan. All3432 paired fan-member outcomes and all3432 fixed-state N-1
deletions are saved. Three of32 active OOF witnesses have at least one action
flip:21/3072 deletions; the360 full-training deletions and3 actual self deletions
have none. Two OOF sensitive states are terminal-year near-break-even growth
comparisons with zero paired risk-probability change. The other is a year16
risk/growth comparison. These local deletions are not a full learner refit,
LOOCV or proof that resubstitution optimism is absent.

Saved adaptive continuation is compared only with the same actually chosen
current action followed by B on the identical realized suffix and inherited
state. Total remaining net real log-growth gaps have17 positive,7 negative
and11 zero values; MDD gaps16 positive,2 negative and17 zero. All35 terminal
event flags agree in this selected inventory. All11 active year30 controls
have zero continuation gaps because no future decision remains. These are
pathwise descriptive differences, not unchosen-action adaptive Q, conditional
calibration or a policy-improvement theorem. Witnesses exclude35/52/98/112,
so they do not identify those admission-failure causes.

A separately implemented independent audit uses its own scalar fan, explicit
member slicing and100-step fee reference. It passes245925 checks, all3432
members/deletions,1106 current-action/B suffix periods and553 saved adaptive
suffix periods, with maximum numeric absolute error4.55e-13. Source/input/output
hashes agree at start and end. Audit is computational AI-assisted review,
not external human peer review or new market evidence.747 tests pass in
152.217s. All numerical work parses only original development IDs1–120;
legacy src/report/output remain unchanged from7bca10b.

Current full-development C remains executionReview with financial outputsNULL.
The next-session handoff prioritizes exact first-divergence reconstruction for
the named failure IDs, a clear separation of training/inner/outer estimands,
and reconciliation with the original deliverables before final model closure.
There is no automatic fallback, outlier deletion or forced-admission search.

Final evening checkpoint passes151 saved snapshots and898 distinct hashed
files, including the completed747-test log and executed earlier companions.
The new evidence is additive; existing frozen sources/results and all
previous numeric/visual archives retain their original hashes.


## 2026-10-19: failure paths, controlled interventions and model closure

The user's daytime work session continues development deeply while paper
writing remains deferred. All numerical work below uses original IDs1–120;
final121–180 is not reopened. The unchanged main registry and legacy
src/report/output remain authoritative. New studies are additive and never
turn current C executionReview into a simulated investment return.

The first mechanism protocol is frozen before its numerical execution
(7249682). Eleven fixed M/B cases from the original failure-ID inventory
reconstruct22 original30-year accounts,682 endpoints and660 account periods.
Four missing inner B trace arrays are reconstructed with the original own-target
annual/pending reference and checked against saved financial metrics. Independent
fee/account arithmetic passes72210 checks, maximum3.67e-13. The auditor initially
encountered absent optional loss_tolerances metadata in old B arrays; its explicit
adapter uses the frozen group L. Producer and old evidence are unchanged. The
first action difference is not necessarily the first exposure difference, and
crossing-year equal targets do not imply equal wealth or historical real peaks.
Current-DD accounting components are not additive lifetime-MDD causal shares.

The next question was fixed before the new mechanism outputs (a2d559f), then
its implementation/protocol frozen in27b2393. Only the original first M/B action
disagreement is replaced; both branches subsequently re-evaluate the unchanged
policy from their own original histories. All11 control paths reproduce the
original model. Independent review passes12036 checks,22 complete accounts and
54 active fixed Q witness slots (43 distinct); the finite witness scalar-fan
checks are not described as an independent recomputation of every new Q.
None of11 breach/first-crossing labels changes. Total net-log wealth changes
are5 positive/6 negative. Nine MDD changes are rounding scale; full98 improves
by0.548826pp and outer_r4_112 worsens by1.994169pp. No intervention becomes an
approved repair, and no unfavorable case is replaced.

The9 equal-MDD observations motivate an explicitly post-hoc positive-homogeneity
question. Its source, plan and10 synthetic tests are frozen in04262b7 before
new saved-array matches. Every original case is retained. Exactly9 reach a
sufficient proportional holdings/real-peak state at an interior endpoint before
both arms attain finalMDD; all subsequent actions, normalized physical states,
Q/counts and future fee scaling pass, maximum8.88e-16. The two materially
changed-MDD cases have no such witness. This derivative diagnosis uses the
already independently audited intervention arrays; it is not a fifth independent
market or all-policy audit, a probability estimate or proof of necessity.

A separate outcome-free metadata match within the already selected11 cases
finds exactly one same-ID/target/eta pair among55 possibilities. The protocol
and source freeze in dafda15 precedes new local Q computation. Full112 and
outer_r5_112 first act differently in original year5 after15 categories of
exactly equal prior state/history. Full120 and delete-self119 recommend restore;
original outer96 holds. Independent scalar fan reconstructs every335 paired
member and670 action branches, maximum1.22e-15. The risk reduction9→8 in120/119
compensates negative predicted financial growth, while96 has7→7 and no risk
compensation. Removing112 alone does not explain this local flip. This is a
particular fixed-state fan replacement, not a full119 learner, calibration or
attribution of the entire later MDD difference to year5.

The known-cost evidence bridge (89632b8) requires exact base-cost identity of
all48 candidate fields and36 fields for each reference before copying old cost
rows. Old reference target metadata is the sole explicitly supplied missing
field. Full120 fixed-candidate/reference subsets at five original rates yield
1800 shadow rows; no old outer fit, menu, selection or admission transfers.
Root's independently authored bridge review compares120 arrays/369371 scalar
values and19800 saved financial scalars, and separately recomputes15 aggregate
groups; maximum2.22e-16. Its first metadata adapter assumed plural start/end
keys, while the old cost receipt uses singular source_and_artifact keys; this
was corrected explicitly without changing old producer/results. Candidate
breach counts10/9/9/10/9 are nonmonotone because known fees also change actions.
Fixed Review under all rates is not empirical admission robustness.

A new current model contract is independently reconciled with the original
three-page problem and current artifacts (ec6c0db).193 checks cover18 summary
rows,96 records/74 IDs, same-auto B comparisons,678=376+302 restore classes,
Q1 own-H labels/annual reference,112 severe-case fields and44 customer words.
The static Q1 annual reference is distinct from Q2 B annual-then-pending.
Training resubstitution is distinct from the causal information set of heldout
execution. Eight finite technical prices are separate from risk_first. The
historical registry's growth_only31 trial label is clarified as the positive
financial-benefit simplification; retained eta0 training is37/120. Frozen
registry/history are not edited. Source/accounting correctness does not solve
conditional prediction or autonomous-service stability; no new search is
undertaken merely to obtain automatic approval.

The new companion builder is frozen in2de2c14 after7 synthetic artifact-gate
tests and actual inspection of all four final PNGs. Its initial source freeze
attempt was stopped by gitdiff whitespace checking before commit; the draft EOF
was fixed and presentation fingerprints refreshed before the successful freeze.
The executed notebook has13 sequential code cells,9 HTML tables and4 embedded
figures, zero errors. All PNG/SVG bytes remain those actually inspected; SVGs
have no trailing whitespace. The receipt binds source/evidence/figure manifest,
notebook andHTML. Entire-HTML browser preview and fully offline math rendering
are explicitly not claimed.860 unit tests pass in154.139s for this checkpoint.
A premature checkpoint attempt correctly rejects the still-running test log;
no rule is weakened, and the completed log is used on the successful rerun.

The readiness judgment supports closing model definition and preparing a
locked transparent final evaluation, not declaring C deployable. Current C
remainsReview/financialNULL because selected-beta inner10 exceeds same-U B8.
The remaining evaluation adapter must separate primary service, nonadmitted
shadow and references, lock metrics/NULL/information timing before final data,
and disclose that the earlier V1 already saw121–180. Paper/ZIP/CMT work remains
outside this session's current modelling checkpoint.


A later explicitly post-hoc regrouping of all335 already independently audited
matched-fan member rows identifies original95 as the only full120 paired event
vote (hold true, restore false). It is among the original24 members absent in
outer96. All common96 member growth/event fields are exact; removed24 has2/1
events and paired mean-growth advantage+.00212114, versus−.00677266 in common96.
All24 omitted identities and the full set of nonzero votes are preserved;
no95-only deletion or new fan simulation occurs. The review is descriptive
algebra of audited results, not another independent market audit or a reason
to discard95. Its maximum mean-Q aggregate roundoff is3.68e-16.

Archival whitespace checking initially flags the standard CRLF bytes of new
CSV evidence. Those frozen bytes/hashes are preserved; a one-command Git
cr-at-eol text convention confirms no actual trailing spaces or blank-EOF
errors. No audit or model rule is changed. The completed mechanism checkpoint
passes179 saved snapshots and985 distinct hashed files, including860 tests.


## 2026-10-19: freeze evaluation objects and validate the adapter only

The same session advances one concrete preparation step without reopening
finaldata. The new adapter fixes4Q2objects and5own-H Q1objects in source before
final values: the complete guardedC service, its separately unapprovedfixed
shadow, own-targetB/pending, exactthirds annual, and the existing Q1 staticannual
product references. MainC/A/DReview never obtain physical account trees or
substituted finance. OriginalH/L/liq, case/original/financial/auto/Review
populations and all growth/tail/trade metrics are explicit. Both source
reviewers and root check these scopes;27 synthetic producer tests and10
auditor tests precede the numerical reproduction.

The adapter source/protocol/auditplan freeze in3953565. Root enables only
preparation developmentready; preparation finalready staysFalse permanently.
A later NEW engineering release must bind the unchanged source, objects,
passed developmentmanifest and exact independentaudit plus their current
closures before a final parser can be invoked. It is not an additional
human-permission flow. Default--describe reads metadata only. No release is
created and no final output directory exists in this session.

The source review corrects the initial draft's numerical-excess semantics:
when MDD-L lies inside the strict1e-12 equality tolerance, excess iszero.
Terminalasset fields are explicitly weights, not currency values. Both
possiblefuture final execution and current developmentgate write a unique
incomplete attempt before any valueparser, retainphase/errorlogs/failed
receipts on exception, and prohibit overwriting/retry. Finalpossibleaccess
is marked before its parser and completeload separately. These changes are
made before sourcefreeze and any new numerical gate, not after seeingfinal.

Only the guarded1–120loader is called in the actual run. The original joint
parent arrayfingerprint matches exactly. Single-fixedprice shadow48fields,
B36 andthirds36 match everyold dtype/shape/value/NaNmask without tolerance.
All360 availableQ1annualproduct metricpaths match savedCSV serialization;
240Q1Reviewpaths remainNULL. The new artifact includes the six actually
simulated physicaltrees so an independent reader can check them without
another simulation; there is no invented mainReview physicalstate.

Independent audit compares120fields/369371scalarvalues, reconstructs480Q2
and600Q1rows and9summaries, and checks720 savedphysicalaccount artifacts.
It passes52037 checks, maximum2.27e-13, with0newmarketparsers,0simulations,
0newfullfanQevaluations. This is implementation/serialization evidence;
conditionalriskcalibration and serviceefficacy remain unproven.

The first independentauditor attempt stops after exactarray checks because
its receipt fieldlabels omit the producer'sobject/ prefix. Oldauditor and
originalfreeze bytes are preserved underresearch/audit_attempts, along with
an explicit failure explanation. The historicalfreeze resolves its hashes
to archivedsource and records the oldcurrentpath separately. V2 changes
only the labeladapter, retaining allarrayfields/values/dtypes/shapes/NULL
checks/tolerances and untouchedproducer. The successfulreceipt doesnot
claim a successfulfirstattempt or a newindependentmarketengine.

CurrentC still failsinner model10vsB8 and remainsReview. Final121–180,
report, ZIP andCMTare not run. The unchangedmainmodel and newmodelcontract
can now be evaluated by the fixedadapter after a separately declaredrelease,
with oldV1final exposure disclosed and no furtheroutcome-driven selection.


Final complete suite passes897 tests in153.393s. The final checkpoint passes
189 saved snapshots and1009 distinct hashedfiles, including the executed
13-cell mechanism companion and development-only evaluation gate plus its
independent audit/failure history. Legacy src/report/output are unchanged
from7bca10b. Default evaluation description is checked without marketvalue
parsing; the separate finalrelease and finaloutput directory remain absent.


## 2026-10-19: execute the unchanged locked final evaluation

The user's next instruction explicitly authorizes final evaluation while
keeping the established model and rules unchanged. A separate execution
release binds the preparation's eight source files, twenty inputs, object
specification and passed development audit. The preparation's own finalready
flag staysFalse; the new release is the sole engineering activation. Fixed
independent-audit, isolated-reproduction and paired-analysis plans accompany
the release in41da208 before this run. Source checks and45 new synthetic tests
precede execution. Old V1's prior final121–180 exposure is disclosed throughout;
this is locked reuse, not a newly blind confirmation sample.

The first released producer execution completes without an exception. It uses
the unchanged120-path training fan and all sixty final IDs121–180, with no
new fit, parameter or target selection, admission change, fallback or outcome
selection. Four Q2 objects and five Q1 own-horizon objects produce240 and300
path records plus nine summaries. C's primary service remains pre-investment
executionReview on all sixty paths, with financial outcomesNULL. The rejected
fixed candidate is separately named unapproved shadow. Same-target B/pending
and exact-thirds annual are hypothetical references, not replacement service.

Final Q2 shadow/B/thirds breach counts are11/8/6 of60. Median real CAGR is
3.7335/3.5616/3.3761 percent, while P10 is1.2772/1.4955/1.4664 percent and
P90 annual real MDD is45.0275/43.5416/39.3573 percent. Shadow trading and mean
nominal fees fall, but lower-tail growth and drawdown tails worsen. Against B
on identical IDs, growth is higher on45/60 paths and MDD larger on53/60;
three extra breaches are124/152/167 with no B-only event. Against thirds,
five additional breaches are124/152/156/167/170 with no reference-only event.
Both reference groups and all sixty paired paths remain visible. There is
no post-result rescue, alternative winner or significance claim.

Q1 B/C/E static annual references retain own horizons15/30/6 and loss limits
25/40/35 percent. Breaches are11/8/5 of60; B/C exceed the10% empirical reference,
while E's P10 real CAGR is−1.5800%. A/D remainReview/financeNULL. These findings
limit prospective risk claims and are not screened out. Different horizons
and limits prevent a simple equal-horizon product-risk ranking. Q1 C continues
annual restoration after crossing, unlike Q2 B's pending hold; matching event
counts do not make the subsequent accounts identical.

The first independent final audit passes81,911 arithmetic checks with maximum
absolute error4.55e-13. It reconstructs all360 physical accounts,8460 periods,
228 physical-array fields and all tables/summaries, preserving180 Review NULL
rows. Nine prospectively fixed IDs121/150/180 by originalyears2/16/30 slots
independently recompute both action branches over all120 training fan members.
All nine are active; other active Q decisions receive saved-Q algebra and
accounting checks, not a claimed independent complete Q recomputation. The
audit is AI-assisted numerical checking, not external human peer review or
conditional forecasting calibration. Original producer bytes remain unchanged.

The first isolated same-source replay copies46 allowlisted dependencies to a
fresh temporary workspace and reproduces all six producer artifacts byte for
byte. All234 named NPZ fields and281540 scalar values match exactly; only
the declared completed_utc key is normalized in the manifest. Copied files
and original producer outputs are unchanged. This uses the same scenarios and
runtime, so it is computational reproduction rather than another market test.
The separate interpretation review independently reconstructs600 paired
deltas,20 mean/median summaries,30 sign counts and all event-ID partitions:
1030 checks, maximum difference0. The final reading figure retains all60
same-B points and all four Q2 objects, including main financialNULL, and passes
both author and root actual-image review before its manifest is frozen.

The full suite passes942 tests in153.892s. A new final checkpoint checks1055
distinct files and10632 hash/map relations, including release, producer,
audit/attempt, replay, paired analysis, interpretation and figure lineage.
Its verifier and saved receipt also receive separate read-only review. The
old development verification and its897-test log remain byte-identical to
the pre-final Git checkpoint; its189 snapshots are preserved. Legacy
src/report/output remain unchanged from7bca10b. The final evaluation is
procedurally complete; reliable autonomous service and future10% risk control
are not established. No paper, codeZIP or CMT submission is made in this phase.

## 2026-10-20 Q1 guard generalization addendum (development-only)

After submission closure, one additive question remained: does the locked V2b
protective rule extend to the other two assigned profiles? A new protocol
(`docs/q1_guard_generalization_protocol.json`) and source freeze were fixed
before any numeric run. The locked `release_0.5_uniform` candidate was reused
unchanged with each profile's own target, horizon and loss limit on development
scenarios1-120 only; the loader's allow_final gate stayed closed. No refit,
selection or threshold change occurred, and no locked artifact was modified.

Both guard arms end in pre-investment Review: the declared one-year support
floor ((1-L/2) of current real wealth; B 87.5, E 82.5, vs C's passing 80.0)
admits no feasible initial allocation for the Conservative25/30/45 or
Growth65/5/30 targets on the calendar development support. The safeguard
therefore generalizes as a stricter funding gate, not as an executed policy;
no funded B/E guard paths exist and no breach-reduction claim is made for
them. The static references independently reproduce the previously audited
development5x3 screen exactly (B Conservative9/120, E Growth6/120 breaches).

An independent audit (`results/q1_guard_generalization_v1/independent_audit.json`)
recomputes every metric and summary from the saved arrays without importing
the producer's metric/policy modules, checks the self-financing identity and
fee convention on all960 path records, and passes10993 checks. Its first two
attempts failed on auditor-side convention bugs (running-MDD reconstruction;
first-crossing off-by-one); only the auditor changed. An isolated same-source
replay (`results/q1_guard_generalization_reproduction_v3/`) reruns the producer
in a fresh temporary workspace: all six deterministic artifacts match byte for
byte and all210 NPZ array fields match exactly, with only declared manifest
runtime fields normalized. Two earlier replay attempts failed on replay-script
dependency/zip handling and are preserved unchanged. Eleven new focused unit
tests pin the freeze gate, summary conventions, NULL handling and partitions;
the full suite then passes1545 tests. Report-facing wording rules are recorded
in `submission/Q1_GUARD_GENERALIZATION.md` and
`docs/q1_guard_generalization_judgment.json`. This phase adds no market
observations and no final-data access.
