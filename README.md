# SafeInvest: cost-aware advice with explicit limits

2026 HKUST(GZ) MCM Senior Group Problem A. Team **Lebmar**, number **87**.
Author: **Chengxuan YUAN**.

**Current status: model foundations under review.** The V1 code and report below
are an archived, auditable prototype; they are not the accepted final competition
model or a submission recommendation. Work has returned to the objective,
parameter meanings, assumptions, internal validation and decision rules.
See `docs/model_foundation_review.json`. No report is being regenerated in this
phase. The old evaluation scenarios have already been seen and cannot be
represented as a new blind test for subsequent changes.

This repository records a reproducible investigation of purchasing-power risk,
customer suitability, selective rebalancing and bounded target updates. It does
not assume that a more complicated policy improves investment outcomes.

## Current development research

The new work uses scenarios **1–120 only** and keeps the V1 archives intact.
Start with `notebooks/objective_risk_foundations.ipynb`, then the
[executed development notebook](notebooks/risk_menu_development.ipynb).
A [reading copy](notebooks/risk_menu_development.html) hides code cells;
it is a research note, not a submission
report. The notebook separates the growth objective, risk-budget policy,
sampling evidence, liquidity assumptions and the actual execution rule.

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

The latest checkpoint passes **353 tests** and checks 46 saved manifests/audits
against 407 distinct source, input, result and execution-evidence files. All 73
saved artifacts from seven execution studies also reproduce byte for byte in
an isolated copy, without changing the frozen archives. A second replay using
only Git-tracked files from a fresh clone passes the same tests and matches
all 73 outputs; its context records the tested financial-source revision. See
`results/model_review/research_checkpoint_v1/verification.json`; these are
computational checks, not independent market observations or risk guarantees.
AI assistance, including separately implemented numerical audits, is disclosed
in `docs/DATA_NOTICE.md`.

The next-stage proposal is `docs/dynamic_calibration_next_stage.json`. It has
been reviewed but not executed: calibrate the complete learning procedure in
inner whole-path folds, refitting its product menu as well as its controller.
The proposed primary family keeps nine fixed-target price candidates; the
18-controller update family would be a separately declared incremental study.
More cautious admission may lead to more Review cases and still cannot turn
reused development results into a new blind test.
The requirement-by-requirement inventory is
`docs/problem_A_deliverable_crosswalk.json`. It also separates the Balanced-only
Q2 case from other future training assignments and records the unresolved
liquidity, final-policy, evaluation and submission steps.

To rebuild the reading notebook in an activated project virtual environment:

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
