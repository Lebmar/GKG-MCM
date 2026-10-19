# Q3 sensitivity and safeguard note

Team **Lebmar (87)** — Chengxuan YUAN. Internal technical note for the final report.

**Scope:** frozen V2b selected arm; development scenarios **1–120** only. This is
not a new final evaluation and does not read 121–180. The final base evaluation
(121–180) remains the one in
[summary.csv.gz](../results/mitigation_locked_final_evaluation_v1/summary.csv.gz).

Source: [stress summary.csv.gz](../results/state_budget_mitigation_stress_v1/summary.csv.gz),
[stress protocol](../docs/state_budget_mitigation_stress_protocol.json).

## 1. Transaction-fee stress (primary Q3 test)

Setting: all actors know the same fee; actual trade accounting and references use
that fee. No refit, no reselection, no new final read. Rows are outer-OOF selected
learner arm, 120 original paths.

| Fee | Breaches | Median CAGR | P10 CAGR | Mean MDD | P90 MDD | Trade fraction | Main recommendation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0% | 0/120 | 2.865% | 0.767% | 18.997% | 28.979% | 24.44% | unchanged |
| 0.125% | 0/120 | 2.856% | 0.764% | 19.015% | 28.979% | 24.36% | unchanged |
| **0.25% (base)** | **0/120** | **2.847%** | **0.761%** | **19.030%** | **28.979%** | **24.22%** | **unchanged** |
| 0.5% | 0/120 | 2.831% | 0.757% | 19.016% | 28.979% | 23.78% | unchanged |
| 1% | 0/120 | 2.802% | 0.748% | 19.079% | 28.979% | 23.31% | unchanged |

**Conclusion:** on development data, the known-fee rate does not change the main
recommendation. Growth falls modestly and trading/fees adjust; the breach count
stays zero. This tests cost magnitude, not cost misestimation, refitting,
re-approval or real-market safety.

## 2. Modeling-assumption stresses

| Setting | V2b result (1–120) | What changes |
| --- | --- | --- |
| Calendar support (base) | 0/120 breaches; median CAGR 2.847% | reference |
| **Pooled training-year support** | 0 funded; initial **Review**; financial **NULL** | **Main recommendation changes to Review** |
| **Actual inflation +2pp** | 21/120 breaches; median CAGR 0.745%; P10 −1.163%; mean MDD 27.655%; P90 MDD 43.540%; trade 23.06% | Risk materially worsens; **suspend/review** rather than claim unchanged safety |

- **Pooled support** replaces the calendar-year support with all chronological
  training-year joint real-gross rows. Broader support is more conservative and
  refuses initial investment; the NULL is not zero risk and not a bill portfolio.
- **+2pp inflation** adds 0.02 to every actual inflation decimal, with nominal
  returns, training fan and support unchanged. Later actual observations can
  legitimately change later decisions; no future shock is supplied early.
- These are descriptive stresses of an already selected frozen candidate, not
  refit/admission robustness or a calibrated future risk bound.

## 3. The one important limitation and the safeguard

**Limitation:** the one-year support/forecast condition is empirical and finite.
It is not a conditional return distribution, a calibrated loss probability, an
intrayear stop-loss or a future guarantee. The final 121–180 sample was exposed
earlier and is not fresh blind validation.

**Primary human-review/suspension condition:** a funded support-family failure or
no feasible supported action. The policy may take one protective trade only if
the modeled worst next-real-wealth outcome improves after exact costs; then it
enters absorbing human review/hold. An annual real MDD above the customer's L
also triggers review. Model/input-integrity failure or a material
questionnaire/cash-needs change triggers pre-investment review.

**While review is pending:** hold existing holdings; no new automatic trades;
continue actual market exposure, CPI, original peak/MDD/H and financial history;
notify the customer and human adviser. Do not assume an adviser response, a
recovery reset or a liquidation. The 41-word customer text is in
[mitigation_customer_explanation_contract.json](../docs/mitigation_customer_explanation_contract.json).

## 4. How to state this in the report

Recommended one-sentence answer:

> A known transaction fee of 0–1% does not change the main recommendation on the
> development scenarios, but the policy is not robust to the support assumption:
> pooled support changes the initial decision to Review, and +2pp inflation
> raises breaches to 21/120 and lowers P10 growth to −1.16%. We therefore treat
> support/forecast failure as a suspension trigger and hold existing exposures
> under human review.

Do **not** say that the final 121–180 policy was fee-stressed; it was not. Do not
present the 1–120 stress rows as new independent market observations or as a
replacement for the final evaluation. Do not use the pooled-support NULL as a
zero-risk outcome.
