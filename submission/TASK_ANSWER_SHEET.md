# SafeInvest — submission answer sheet

Team **Lebmar (87)** — Chengxuan YUAN. 2026 HKUST(GZ) MCM Senior Group Problem A.

**Status:** internal closure document, not the competition paper. It maps the frozen
V2b evidence to the three questions. No model, parameter, target or final-output
change is made here. Final data are the previously exposed scenarios 121–180.

## 0. Bottom line

- **Q1:** three fixed products are defined: Conservative **25/30/45**, Balanced
  **45/5/50**, Growth **65/5/30** (equity/bond/bill). The onboarding rule assigns
  **A Review, B Conservative, C Balanced, D Review, E Growth**.
- **Q2:** the V2b policy reduces tail risk and fees relative to the equal-weight
  baseline, but it gives up median and lower-tail growth. It is a risk-control /
  assisted-service design, not a return-enhancement rule.
- **Q3:** the clean transaction-fee stress leaves the main recommendation
  unchanged on development data; a pooled-support change flips it to **Review**,
  and a +2pp inflation stress materially worsens the risk. The main safeguard is
  an absorbing human review/hold after funded-support failure, with no new
  automatic trades while pending.

## 1. Q1 — suitability, products and A–E

### 1.1 Measures

| Object | Definition | Larger means |
| --- | --- | --- |
| Growth | Net real annualised log growth `G_H`; CAGR = exp(G_H) − 1; net real wealth = nominal wealth / cumulative CPI | higher growth |
| Ordinary fluctuation | Sample standard deviation of annual net real returns (ddof=1) | more ordinary volatility |
| Large loss | Maximum real drawdown from the original historical peak, using annual endpoints; the complete original peak/history is retained | **larger drawdown = larger loss** |
| Tail severity | Empirical CVaR90 of the MDD distribution; also P90 MDD | larger tail loss |
| Purchasing power | All growth, MDD, volatility and tail measures are computed on real wealth, not nominal wealth | — |

The large-loss measure is intentionally positive and increasing in loss. Annual
endpoint MDD does **not** observe intrayear highs/lows. Customer loss tolerance
`L` is a questionnaire response and a review trigger, never a guaranteed limit.

### 1.2 Products

| Product | Equity | Bond | Bill | Target bill floor by liquidity |
| --- | ---: | ---: | ---: | --- |
| Conservative | 25% | 30% | 45% | high 30%, medium 10%, low 0% |
| Balanced | 45% | 5% | 50% | high 30%, medium 10%, low 0% |
| Growth | 65% | 5% | 30% | high 30%, medium 10%, low 0% |

The floors are normative target conventions. They are not live cash reserves,
withdrawal guarantees or evidence that the customer's actual cash needs are met.

### 1.3 A–E assignments and final own-horizon evidence

Final evaluation uses original scenarios 121–180 (60 paths), each profile's own
horizon and loss tolerance. A/D remain unfunded Review with **NULL** financials.

| Profile | H | L | Liquidity | Assignment | Target | Median CAGR | P10 CAGR | Median MDD | P90 MDD | Breaches |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | 5 | 10% | High | **Review** | — | NULL | NULL | NULL | NULL | NULL |
| B | 15 | 25% | Medium | **Conservative** | 25/30/45 | 3.21% | 0.35% | 14.09% | 29.53% | 11/60 |
| C | 30 | 40% | Low | **Balanced** | 45/5/50 | 3.56% | 1.48% | 27.49% | 43.54% | 8/60 |
| D | 25 | 15% | Medium | **Review** | — | NULL | NULL | NULL | NULL | NULL |
| E | 6 | 35% | High | **Growth** | 65/5/30 | 4.51% | −1.58% | 14.17% | 31.34% | 5/60 |

Source: [summary.csv.gz](../results/mitigation_locked_final_evaluation_v1/summary.csv.gz).
These are **static annual product references**, not the Q2 guarded service. Q1
does not authorize the dynamic execution program for every product.

### 1.4 Why A/D are Review, and why B/C/E are assigned

Full-development feasibility counts (original 1–120; exact joint-menu procedure):

| Profile | Risk-feasible | Liquidity-feasible | Joint-feasible | Menu candidates | Assignment |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0 | 92 | 0 | 0 | Review |
| B | 52 | 154 | 52 | 1 | Conservative |
| C | 93 | 172 | 93 | 2 | Balanced |
| D | 0 | 154 | 0 | 0 | Review |
| E | 157 | 92 | 92 | 3 | Growth |

A/D have **zero risk-feasible candidates** under their own H/L/liquidity, so
Review is an honest output, not a missing number. B/C/E have feasible common-menu
candidates. The exact joint-menu/tie-break procedure is in
[service_refinement_readiness_review.json](../docs/service_refinement_readiness_review.json)
and [joint_liquidity_refit_protocol.json](../docs/joint_liquidity_refit_protocol.json).

A compact statement of the onboarding rule for the report is:

1. Set the three named products and the high/medium/low bill floors.
2. For each profile's own horizon, evaluate the finite common product menu on
   training data with exact fees; require the declared empirical loss-event screen
   and the liquidity floor.
3. Retain feasible common products; among them, retain `G >= global max G − 1e−12`,
   then apply the declared `(excess, CVaR90, candidate index)` tie-break.
4. If no product is feasible, output **Review** (A/D).
5. Do not reinterpret the empirical 10% service reference as a customer guarantee.

**Material caveat:** B/C static final frequencies are **18.3% / 13.3%**, above the
10% empirical service reference; E has **−1.58% P10 growth** despite a positive
median. These adverse tails are real and must stay visible. The 10% reference is a
declared service convention, not a contest-mandated cap; Q1 is a static-product
stress, while C's Q2 V2b guard is the execution safeguard (2/60 breaches). All
assigned products should use the same loss-trigger human review; do not call the
static Q1 numbers autonomous service performance.

Optional same-horizon equal-thirds comparison (not the selected product):

| Profile | Selected product median CAGR / P10 / MDD / P90 / breaches | Equal-thirds median CAGR / P10 / MDD / P90 / breaches |
| --- | --- | --- |
| B | 3.21% / 0.35% / 14.09% / 29.53% / 11 | 3.76% / 0.30% / 16.51% / 34.00% / 14 |
| C | 3.56% / 1.48% / 27.49% / 43.54% / 8 | 3.38% / 1.47% / 22.29% / 39.36% / 6 |
| E | 4.51% / −1.58% / 14.17% / 31.34% / 5 | 2.79% / −0.08% / 8.41% / 22.75% / 1 |

This is a trade-off table, not a dominance claim: B's Conservative lowers breaches
and MDD versus equal thirds; C's Balanced buys growth with worse tails, which is
why the Q2 guard matters; E's Growth buys median growth with a negative P10.

### 1.5 Development 5×3 product screen

The exact 15-cell table is in
[Q1_PRODUCT_COMPARISON.md](Q1_PRODUCT_COMPARISON.md), with the one-page decision
flow in [Q1_ONBOARDING_FLOW.md](Q1_ONBOARDING_FLOW.md). The compact matrix below
uses the **development 1–120** selection screen; each cell is
`median CAGR / P10; median MDD / P90; breaches/120`.

| Profile | Conservative 25/30/45 | Balanced 45/5/50 | Growth 65/5/30 | Assignment |
| --- | --- | --- | --- | --- |
| A (H5, L10%, high) | 3.17 / −1.59; 5.56 / 14.90; 34 | 4.36 / −1.85; 8.59 / 20.53; 49 | 5.10 / −2.73; 11.94 / 27.46; 66 | **Review** |
| B (H15, L25%, medium) | **3.04 / 0.10; 13.25 / 24.22; 9** | 4.08 / 0.10; 17.35 / 32.65; 27 | 5.26 / 0.02; 23.18 / 42.07; 49 | **Conservative** |
| C (H30, L40%, low) | 2.97 / 1.02; 17.37 / 31.74; 1 | **4.03 / 1.44; 22.60 / 38.04; 9** | 5.12 / 1.48; 32.26 / 48.84; 35 | **Balanced** |
| D (H25, L15%, medium) | 3.19 / 0.90; 16.71 / 29.37; 67 | 4.31 / 1.34; 20.90 / 35.91; 92 | 5.33 / 1.41; 29.18 / 47.90; 104 | **Review** |
| E (H6, L35%, high) | 2.99 / −1.74; 8.08 / 18.35; 0 | 3.75 / −2.33; 10.29 / 23.11; 2 | **5.00 / −3.69; 13.89 / 31.18; 6** | **Growth** |

All selected B/C/E rows pass the declared 10% training screen (9/120, 9/120,
6/120). A/D have no risk-feasible product. The final selected-product table in
§1.3 is the evaluation table; the final 5×3 descriptive table is retained in the
comparison file but was not used to change any assignment. The final drift is
adverse and disclosed: B 11/60, C 8/60, E 5/60, E P10 −1.58%.

## 2. Q2 — Balanced policy vs the equal-weight baseline

### 2.1 Frozen policy

- Profile C: H = 30, L = 40%, Balanced target **45/5/50**, fee 0.25%.
- V2b: release fraction 0.5, uniform one-year forecast, calendar own-training
  support; 13 primary finite proposals and 9 first-mitigation proposals.
- Decisions: hold; restore/rebalance; temporary protective exposure proposal;
  first mitigation; then absorbing human review/hold. The strategic target is
  **not** persistently relearned. V3 target-update experiments were retained as a
  development ablation but not selected; no clear all-purpose improvement.
- Growth trade condition: strictly positive fee-net paired predicted log
  advantage. Protection may sacrifice predicted growth only when hold fails the
  normative one-year support floor and the modeled worst outcome improves after
  exact costs; this is labelled risk reduction with growth cost.
- Fees: 0.25% of each destination amount moved once, post-dollar, deducted before
  returns; initial allocation charged; no terminal liquidation or taxes.

### 2.2 Required comparison

Equal-weight baseline = **exact-thirds annual**: 1/3 equity, 1/3 bond, 1/3 bill;
restore every review; charge the same 0.25% fee. Final data: original 121–180,
60 paths, H = 30.

| Strategy | Median CAGR | Median MDD | P10 CAGR | P90 MDD | Breaches | Trade periods | Ordinary service | Mean nominal fees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **V2b protective then review/hold** | **2.35%** | **20.02%** | **0.87%** | **28.69%** | **2/60** | **21.22%** | **19.44%** | **0.353** |
| Same-target annual/pending B | 3.56% | 27.49% | 1.50% | 43.54% | 8/60 | 94.56% | 94.56% | 1.018 |
| Same-target continuous annual | 3.56% | 27.49% | 1.48% | 43.54% | 8/60 | 100% | 100% | 1.067 |
| Equal-weight baseline (exact thirds) | 3.38% | 22.29% | 1.47% | 39.36% | 6/60 | 100% | 100% | 0.979 |

**Trade-period definition:** 382 / 1,800 = **21.22%** counts every decision period,
including the mandatory initial allocation and pending periods. Excluding the
initial allocation: 322 / 1,740 = **18.51%**. Ordinary autonomous service is
350 / 1,800 = **19.44%** and is a different metric; do not call it the trade rate.

**Decision on automated weight updates:** the evidence supports **risk-control
value, not growth value**. Against equal thirds, V2b lowers median MDD by 2.27pp,
P90 MDD by 10.68pp, breaches from 6 to 2, fees to about 36%, and trading from 100%
to 21.22%; it also lowers median CAGR by 1.03pp and P10 by 0.60pp. Persistent
strategic target updates are not enabled.

### 2.3 Severe-loss case

The preselected severe case is synthetic **ID 161**, chosen by maximum main full-H
MDD with smallest-ID tie-break. It is a purposeful failure illustration, not a
random or representative customer.

| Arm | MDD | CAGR | First crossing | Trade fraction | Nominal fees |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2b main | 47.68% | −0.17% | 14 | 20.0% | 0.236 |
| Same-target B | 55.56% | −0.26% | 12 | 40.0% | 0.274 |
| Equal thirds | 54.84% | +0.32% | 12 | 100% | 0.971 |

V2b reduces the drawdown relative to both references, but it still crosses the
40% limit. All 60 paths enter human review by decision year 7 (54 in year 7); the
two crossings are original IDs 132 and 161, at endpoints 23 and 14. A one-year
supported action is not a recursive or real-market safety guarantee.

### 2.4 Customer explanation (≤50 words)

Use the final 41-word text from
[mitigation_customer_explanation_contract.json](../docs/mitigation_customer_explanation_contract.json):

> Your portfolio’s risk buffer no longer supports our usual trading options. We
> make a protective trade only if the modeled worst outcome improves after costs,
> then pause automated trading for human review. Existing holdings remain invested
> and may continue losing value.

This covers a protective trade and a human-review request. It does not promise a
recovery or an adviser response. The 44-word text in the earlier
[model contract](../docs/current_model_contract_2026_10_19.json) is **superseded**
for the final Q2 explanation; do not use it in the report.

## 3. Q3 — reliability, sensitivity and safeguards

Full detail and exact rows: [Q3_SENSITIVITY.md](Q3_SENSITIVITY.md). All stress
results use development scenarios 1–120 and the frozen selected V2b arm, not the
final 121–180 sample.

| Setting | V2b result (1–120) | Does the main recommendation change? |
| --- | --- | --- |
| Known fee 0%, 0.125%, 0.25%, 0.5%, 1% | 0/120 breaches; median CAGR 2.80–2.86%; P90 MDD ≈28.98% | **No** (actions/fees move slightly) |
| Pooled training-year support | 0 funded; initial **Review**, financial NULL | **Yes** — changes to Review |
| Actual inflation +2pp | 21/120 breaches; median CAGR 0.75%; P10 −1.16%; P90 MDD 43.54% | **Not robust** — suspend/review |

The clean transaction-fee stress is the Q3 primary test. The support change is a
stronger assumption change; the inflation shift is an adverse stress. Do not
extrapolate the 1–120 fee results to 121–180.

**One important limitation:** the one-year support/forecast check is a finite
empirical condition, not a calibrated conditional distribution or a future risk
guarantee. The final 121–180 sample was exposed earlier; it is not fresh blind
validation.

**Primary human-review/suspension condition:** a funded support-floor failure or
no feasible supported action. The policy may make one protective trade only if the
modeled worst next-real-wealth outcome improves after exact costs; then it enters
absorbing human review. An annual real MDD above the customer's L also triggers
review. Model/input-integrity failure or material questionnaire/cash-needs change
triggers pre-investment review rather than an invented fallback.

**While pending:** hold existing holdings; make no new automatic trades; continue
actual market exposure, CPI, original peak/MDD/H and financial history; notify the
customer/adviser; do not assume an adviser response, recovery reset or liquidation.

## 4. Conclusion (task-facing)

- Products: Conservative **25/30/45**, Balanced **45/5/50**, Growth **65/5/30**.
- Profiles: **A Review, B Conservative, C Balanced, D Review, E Growth**.
- Balanced: use the V2b protective/review rule; **do not enable persistent
  strategic target updates**. The rule trades risk and fees against growth.
- Main safeguard: funded-support failure → protective trade (only if modeled
  worst outcome improves after costs) → absorbing human review/hold; MDD > L also
  triggers review; no new automatic trades while pending.
- Service reality: 19.44% ordinary account-years; all 60 paths referred by year 7;
  2/60 final breaches; this is a risk-aware assisted service, not a demonstrated
  30-year autonomous robo-advisor.

## 5. Evidence map

| Purpose | File |
| --- | --- |
| Current model registry | [mitigation_main_model_registry.json](../docs/mitigation_main_model_registry.json) |
| Technical judgment | [mitigation_technical_completion_judgment.json](../docs/mitigation_technical_completion_judgment.json) |
| Final summary (14 objects) | [summary.csv.gz](../results/mitigation_locked_final_evaluation_v1/summary.csv.gz) |
| Final object specification | [object_specification.json](../results/mitigation_locked_final_evaluation_v1/object_specification.json) |
| Final audit | [independent_audit.json](../results/mitigation_locked_final_evaluation_v1/independent_audit.json) |
| Severe case | [severe_example.json](../results/mitigation_locked_final_evaluation_v1/severe_example.json) |
| Q1 5×3 product screen | [Q1_PRODUCT_COMPARISON.md](Q1_PRODUCT_COMPARISON.md) and [manifest](../results/q1_product_comparison_v1/manifest.json) |
| Q1 one-page onboarding flow | [Q1_ONBOARDING_FLOW.md](Q1_ONBOARDING_FLOW.md) |
| V2b report draft | [main.pdf](../report/v2b_submission/main.pdf) and [main.tex](../report/v2b_submission/main.tex) |
| Q1 menu/feasibility | [q1_context_assignments.csv](../results/model_review/joint_liquidity_refit_v1/q1_context_assignments.csv) |
| Q3 stress summary | [summary.csv.gz](../results/state_budget_mitigation_stress_v1/summary.csv.gz) |
| Customer text | [mitigation_customer_explanation_contract.json](../docs/mitigation_customer_explanation_contract.json) |
| Requirements closure | [technical_evidence_closure_2026_10_19.json](../docs/technical_evidence_closure_2026_10_19.json) |

## 6. Remaining limitations to disclose

1. Final 121–180 was previously exposed; this is nonblind, fixed-policy repetition,
   not fresh blind validation.
2. All 60 paths enter human review by year 7; ordinary autonomous service is
   19.44% of account-years. No adviser response/restart model is estimated.
3. Q1 static B/C final breach frequencies exceed the declared 10% service
   reference; E's P10 growth is negative. These are product-reference results, not
   autonomous approvals.
4. The one-year support/forecast rule is empirical and finite; it is not a
   calibrated probability of future loss, nor an intrayear stop-loss guarantee.
5. The data are synthetic and based on historical U.S. series; they are not
   historical events or forecasts. No real-market predictive validity is claimed.
