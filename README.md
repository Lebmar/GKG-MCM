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

## Archived V1 prototype

The 9-page report and 2.76 MB supplementary ZIP are in `output/`. They have not
been submitted to CMT. There are 31 passing unit/policy tests, 901 passing
independent numerical/provenance checks, and an exact 13-comparison reproduction
of the unchanged protocol in a temporary directory.

The frozen policy retains a fixed Balanced target. Its 30-year median real
growth is 3.66% versus 3.38% for equal weights, while P90 maximum real drawdown
is higher: 43.44% versus 39.52%. Dynamic target updates are not enabled because
the predeclared incremental utility criterion was not met. Customer C's
borderline loss-exceedance screen does not retain pointwise calibration on the
final scenarios. This is a qualified result and a reason for human confirmation,
not a claim of universal dominance or guaranteed suitability.

## Reproduce

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

## Design

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

## Files

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
