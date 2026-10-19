# Independent cached-result verification

The audit script is independent of the strategy and metric implementation: it imports neither `safeinvest` nor `scripts/run_study.py`. It uses cached outputs and immutable provenance records. With explicit authorization, it also reads the original final-period market rows to verify the arithmetic of *already archived actions*. No policy decisions are generated or rerun, and no model is refitted, selected, or modified by the audit.

Run from the repository root:

```sh
.venv/bin/python scripts/verify_results.py
```

The command prints its complete JSON audit and writes `results/verification.json`. A failed numerical or provenance check produces a nonzero exit code. The JSON records the audit script hash, runtime, audited artifact hashes, every individual check, and any failures. These hashes bind this audit to the cached artifacts checked on that run.

## Recorded result

**PASS within the arithmetic, archived-action replay, and provenance scope:** all 901 checks pass. The largest absolute numerical residual is approximately `2.05e-12`, within the declared absolute and relative tolerances of `2e-11`. This count is an inventory of assertions, not an overall statistical accuracy percentage.

| Cached results | Coverage |
|---|---|
| Final evaluation | All six policies, scenarios 121–180, 30 years |
| Final market/action accounting | All 10,800 archived annual transitions, using original final returns and inflation |
| Development validation | All six policies, scenarios 81–120, 30 years |
| Final paired comparisons | All eight comparison groups, all four estimands, point estimates and both confidence bounds |
| Validation architecture comparison | Frozen S3-review versus S2-review comparison, all four estimands and confidence bounds |
| Onboarding | All 15 frozen client/portfolio diagnostic rows and all 15 final assignment labels |

## Independent formulas and accounting checks

For nominal net wealth `W[t]` and inflation `pi[t]`, the independent implementation accumulates a logarithmic price index:

```text
P[0] = 1
P[t] = exp(sum(log(1 + pi[u]), u = 1..t))
R[t] = W[t] / P[t]
CAGR = exp((log(R[H]) - log(R[0])) / H) - 1
MDD = max_t((running_peak(R)[t] - R[t]) / running_peak(R)[t])
```

An additional direct-product deflator checks the logarithmic route. The audit retains the original 100-unit pre-allocation point. Annual real net returns therefore include initial allocation fees. Their sample standard deviation, with denominator `H - 1`, gives the volatility statistic. The audit computes maximum drawdown using an explicit annual peak recursion rather than calling the production metric helper.

For every cached policy, it checks:

- Expected shapes, scenario IDs, finite values, positive wealth, and positive inflation gross factors.
- Nonnegative costs, transfer amounts, holdings weights, and target weights; both weight vectors sum to one.
- `fees = 0.0025 × actual transferred amount`, fees are funded from wealth, and transfers do not exceed investable wealth.
- Trade flags agree with actual transfers, pending-review status persists, and pending years have neither trades nor fees.
- Every numeric field in `summary.csv`, including all percentiles, volatility, terminal wealth, utility, fees, trade frequencies, and review frequency.

The original final CSV paths have unique, complete scenario-period keys and valid gross factors. Their inflation exactly reconciles to the cache within numerical tolerance. Starting from `(0, 0, 100)` asset holdings, the audit reconstructs each saved post-trade allocation and applies original market returns:

```text
post[t] = (cached_wealth[t] - cached_cost[t]) × cached_posttrade_weights[t]
end[t] = post[t] × (1 + original_asset_returns[t])
pre[t + 1] = end[t]
```

Every reconstructed end wealth agrees with cached next-year wealth. The positive increases from pre-trade to post-trade asset amounts reproduce the actual transfer amounts. Reconstructed sales fund the transfers plus fees, and the fee equals 0.25% of the reconstructed transfers. No-trade and pending-review years preserve asset amounts before returns; their weights may drift with market prices. Executed trades achieve the saved target weights. This validates accounting for the saved actions without computing a new forecast or action.

Quantiles are recomputed directly from sorted order statistics using linear interpolation. Paired uncertainty intervals reuse the *declared* seed and 4,000 whole-scenario bootstrap resamples, without calling the production bootstrap helper. The audited score is the frozen definition `log(real terminal wealth / 100) - 4 × MDD`. Recomputing these statistics is distinct from rerunning a policy.

The base-fee rows in `cost_sensitivity.csv` also reconcile to the independently verified main results. The worst-case scenario rule identifies scenario **155** from the largest cached S0 real maximum drawdown, matching the saved evaluation manifest.

## Freeze and assignment integrity

All eight frozen source/configuration/data fingerprints match the current bytes and the final evaluation manifest. The complete model source-file set matches the frozen set. The forecast bank SHA-256 also matches. The saved architecture decision equals the decision embedded in the frozen manifest, and the freeze timestamp precedes final completion.

The final manifest continues to select **S2_review**, as did the frozen validation decision. The audit reconstructs onboarding eligibility and the resulting assignments from the frozen diagnostic rows:

| Client | Frozen assignment |
|---|---|
| A | Human review |
| B | Conservative |
| C | Balanced |
| D | Human review |
| E | Conservative |

Every final diagnostic row retains the corresponding frozen assignment. In particular, **C's final Balanced diagnostics record 10 breaches out of 60 paths (16.7%), exceeding the 10% empirical screen**. This is a consequential generalization limitation, not a reason to revise a recommendation after examining final results. The frozen Balanced recommendation remains recorded unchanged. Passing an empirical screen does not establish a probabilistic loss guarantee.

## Limits of this verification

- Original final market returns and reconstructed holdings independently validate the market-to-wealth transition and each saved trade's asset flows. The forecast scores, admissibility rules, and action-selection logic are not independently re-executed; accounting agreement does not prove that the chosen actions were optimal or free from methodological limitations. Economic accounting unit tests provide separate evidence for the execution helper.
- Non-base-fee sensitivity paths and the separately recomputed 15-year paths are not archived. Their aggregate CSVs cannot be independently reconstructed from the retained path caches. This audit does not claim to have verified those calculations.
- Client-specific wealth paths are not archived. The audit reconstructs assignments from frozen eligibility evidence and verifies final assignment preservation, but does not independently recompute each client's reported loss metrics.
- Fingerprints establish consistency with the frozen model, not the economic validity of its assumptions. Correct arithmetic does not establish stationarity, future performance, sufficient tail coverage, or suitability guarantees.
- The frozen prior-exposure disclosure remains applicable: before formal development, pooled descriptive statistics over all 180 scenarios were printed during initial source inspection. Final strategy results were not used to revise the frozen policy or assignments. This should not be described as a completely pristine holdout.

The numerical checks above support use of the cached main tables and paired comparisons within their defined scope. Narrative, references, plots, document layout, and broader modelling claims require their own review.
