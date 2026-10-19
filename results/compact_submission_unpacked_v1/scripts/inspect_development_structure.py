"""Audit market structure and liquidity assumptions using scenarios 1--120 only.

This is an economic diagnostic, not policy selection or a risk certificate.
The only market reader is load_paths(..., range(1, 121), allow_final=False).
Annual tables describe the equally weighted scenario/year mixture; bootstrap
resamples whole original scenarios, never individual annual rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from safeinvest.core import actual_wealth, execute_target, metric_arrays
from safeinvest.data import load_paths


ASSETS = ("equity", "government_bond", "bill_proxy")
HORIZONS = (5, 6, 15, 25, 30)
LOSS_LEVELS = (.10, .15, .25, .35, .40)
BOOTSTRAP_REPLICATES = 2000
SEED = 2026101707


def stats(values):
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite values in a diagnostic")
    qs = (0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1)
    names = ("min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            **dict(zip(names, map(float, np.quantile(x, qs))))}


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return None
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def cvar(values, level=.90):
    eta = np.quantile(values, level)
    return float(eta + np.mean(np.maximum(values - eta, 0)) / (1 - level))


def boot_mean_interval(scenario_values, indices):
    means = np.asarray(scenario_values)[indices].mean(axis=1)
    return np.quantile(means, [.025, .975], axis=0).T.tolist()


def fixed_annual_paths(returns, target, cost=.0025):
    """Execute a predeclared diagnostic target; no model fitting or forecasts."""
    n, years, _ = returns.shape
    x = np.zeros((n, 3)); x[:, 2] = 100.
    wealth = np.full((n, years + 1), 100.)
    costs = np.zeros((n, years)); moved = np.zeros_like(costs)
    for t in range(years):
        x, costs[:, t], moved[:, t] = execute_target(x, target, cost)
        x *= 1 + returns[:, t]
        wealth[:, t + 1] = x.sum(axis=1)
    return wealth, costs, moved


def longest_true_run(values):
    best = running = 0
    for value in values:
        running = running + 1 if value else 0
        best = max(best, running)
    return best


def market_diagnostics(returns, inflation, indices):
    real_gross = (1 + returns) / (1 + inflation[..., None])
    np.testing.assert_allclose(real_gross * (1 + inflation[..., None]), 1 + returns)
    real = real_gross - 1
    log_real = np.log(real_gross)
    n, years, _ = returns.shape
    asset_summary = {}
    for a, name in enumerate(ASSETS):
        loss_share = (real[:, :, a] < 0).mean(axis=1)
        nominal_loss = (returns[:, :, a] < 0).mean(axis=1)
        mean_real = real[:, :, a].mean(axis=1)
        mean_log = log_real[:, :, a].mean(axis=1)
        # Equal path lengths make the two-stage mean equal to the flat mean;
        # retain the two-stage calculation to make the sampling unit explicit.
        np.testing.assert_allclose(mean_real.mean(), real[:, :, a].mean())
        asset_summary[name] = {
            "real_gross_annual_distribution": stats(real_gross[:, :, a]),
            "real_return_annual_distribution": stats(real[:, :, a]),
            "nominal_return_annual_distribution": stats(returns[:, :, a]),
            "mean_annual_real_return_scenario_equal": float(mean_real.mean()),
            "mean_annual_log_real_gross_scenario_equal": float(mean_log.mean()),
            "nominal_annual_loss_share_scenario_equal": float(nominal_loss.mean()),
            "real_annual_loss_share_scenario_equal": float(loss_share.mean()),
            "negative_real_annual_count": int((real[:, :, a] < 0).sum()),
            "nonnegative_nominal_but_negative_real_share": float(
                ((returns[:, :, a] >= 0) & (real[:, :, a] < 0)).mean(axis=1).mean()),
            "mean_real_return_cluster_bootstrap95": boot_mean_interval(mean_real, indices),
            "loss_share_cluster_bootstrap95": boot_mean_interval(loss_share, indices),
            "scenario_mean_real_return_distribution": stats(mean_real),
            "scenario_real_loss_share_distribution": stats(loss_share),
            "longest_negative_real_year_run_distribution": stats(
                [longest_true_run(row) for row in real[:, :, a] < 0]),
            "nominal_return_vs_inflation_annual_correlation": corr(returns[:, :, a], inflation),
            "real_return_vs_inflation_annual_correlation": corr(real[:, :, a], inflation),
        }
    joint = np.column_stack([real.reshape(-1, 3), inflation.ravel()])
    loss_code = ((real < 0) * np.array([4, 2, 1])).sum(axis=2)
    patterns = {}
    for code in range(8):
        patterns[f"{code:03b}"] = {
            "count": int(np.count_nonzero(loss_code == code)),
            "share": float((loss_code == code).mean(axis=1).mean()),
        }
    high_cut = float(np.quantile(inflation, .75))
    high = inflation >= high_cut
    high_inflation = {
        "definition": "contemporaneous pooled annual inflation >= development P75; descriptive, not a forecast",
        "inflation_cutoff": high_cut,
        "annual_rows": int(high.sum()),
        "scenarios_with_at_least_one_row": int(high.any(axis=1).sum()),
        "asset_loss_shares": {name: float((real[:, :, a][high] < 0).mean()) for a, name in enumerate(ASSETS)},
    }
    serial = {}
    for a, name in enumerate((*ASSETS, "inflation")):
        values = log_real[:, :, a] if a < 3 else inflation
        serial[name] = {}
        demeaned = values - values.mean(axis=1, keepdims=True)
        for lag in (1, 2, 3, 5):
            per_scene = np.array([corr(row[:-lag], row[lag:]) for row in values], dtype=float)
            if not np.all(np.isfinite(per_scene)):
                raise ValueError("undefined serial correlation")
            serial[name][str(lag)] = {
                "mean_within_scenario_correlation": float(per_scene.mean()),
                "median_within_scenario_correlation": float(np.median(per_scene)),
                "p10_p90_within_scenario_correlation": np.quantile(per_scene, [.10, .90]).tolist(),
                "mean_cluster_bootstrap95": boot_mean_interval(per_scene, indices),
                "pooled_within_scenario_demeaned_correlation": corr(demeaned[:, :-lag], demeaned[:, lag:]),
            }
    blocks = {}
    for start, stop in ((0, 10), (10, 20), (20, 30)):
        blocks[f"years_{start + 1}_{stop}"] = {
            "mean_inflation_scenario_equal": float(inflation[:, start:stop].mean(axis=1).mean()),
            "mean_real_return_scenario_equal": {
                name: float(real[:, start:stop, a].mean(axis=1).mean()) for a, name in enumerate(ASSETS)},
            "real_loss_share_scenario_equal": {
                name: float((real[:, start:stop, a] < 0).mean(axis=1).mean()) for a, name in enumerate(ASSETS)},
        }
    path_features = np.column_stack([real.mean(axis=1), inflation.mean(axis=1)])
    ordered_corr = {name: corr(path_features[:-1, a], path_features[1:, a])
                    for a, name in enumerate((*ASSETS, "inflation"))}
    return {
        "annual_sample_description": {
            "scenario_count": n, "annual_rows": n * years,
            "mixture_weight": "1/120 for a scenario, then 1/30 for a year within it",
            "not_independent_annual_trials": True,
            "annual_quantiles_and_correlations": "descriptive weighted mixture; no binomial or iid-row certification",
            "bootstrap": "percentile intervals from 2000 whole-scenario resamples; descriptive sampling variability under the supplied independent-scenario model, not predictive calibration",
        },
        "assets": asset_summary,
        "inflation_distribution": stats(inflation),
        "joint_annual_correlation": {"order": [*ASSETS, "inflation"], "matrix": np.corrcoef(joint, rowvar=False).tolist()},
        "joint_negative_real_return_patterns": {
            "bits": "equity,bond,bill; 1 means real gross < 1", "patterns": patterns},
        "contemporaneous_high_inflation": high_inflation,
        "serial_dependence": {
            "variables": "asset log real gross and raw inflation; no pairs cross scenario boundaries",
            "small_sample_caveat": "30-year per-path correlations are noisy and demeaning can bias short-path autocorrelations; these summaries do not establish exploitable prediction",
            "by_asset_and_lag": serial,
        },
        "early_middle_late_annual_descriptions": blocks,
        "between_scenario_structure": {
            "independence": "asserted by the problem statement, not proved by descriptive correlations",
            "scenario_specific_summary_distributions": "see per-asset means and loss fractions above",
            "adjacent_scenario_id_correlation_of_path_means": ordered_corr,
            "interpretation": "scenario IDs are labels, not market dates; this is an order diagnostic, not evidence for independence or a cross-market forecast",
        },
    }


def path_risk_diagnostics(returns, inflation):
    output = {}
    for name, target in {"all_bills": [0., 0., 1.], "equal_weight_annual": [1 / 3] * 3}.items():
        wealth, fees, moved = fixed_annual_paths(returns, target)
        if name == "all_bills":
            np.testing.assert_array_equal(fees, np.zeros_like(fees))
            np.testing.assert_allclose(wealth[:, 1:], 100 * np.cumprod(1 + returns[:, :, 2], axis=1))
        by_horizon = {}
        previous_mdd = np.zeros(len(returns))
        for h in HORIZONS:
            metrics = metric_arrays(wealth[:, :h + 1], inflation[:, :h])
            if np.any(metrics["mdd"] < previous_mdd - 1e-12):
                raise AssertionError("prefix maximum drawdown must not decrease with horizon")
            previous_mdd = metrics["mdd"]
            real = actual_wealth(wealth[:, :h + 1], inflation[:, :h])
            np.testing.assert_allclose(real[:, -1], metrics["terminal_real"])
            log_g = float(np.mean(np.log(real[:, -1] / 100)) / h)
            by_horizon[str(h)] = {
                "n_scenarios": len(returns), "annualized_expected_log_growth": log_g,
                "geometric_growth_expG_minus1": float(np.expm1(log_g)),
                "cagr_median": float(np.median(metrics["cagr"])),
                "cagr_p10": float(np.quantile(metrics["cagr"], .10)),
                "cagr_p10_inverted_cdf": float(np.quantile(metrics["cagr"], .10, method="inverted_cdf")),
                "mdd_mean": float(np.mean(metrics["mdd"])),
                "mdd_median": float(np.median(metrics["mdd"])),
                "mdd_p90": float(np.quantile(metrics["mdd"], .90)),
                "mdd_p90_inverted_cdf": float(np.quantile(metrics["mdd"], .90, method="inverted_cdf")),
                "mdd_cvar90": cvar(metrics["mdd"]),
                "terminal_real_median": float(np.median(real[:, -1])),
                "terminal_real_p10": float(np.quantile(real[:, -1], .10)),
                "terminal_purchasing_power_loss_count": int((real[:, -1] < 100).sum()),
                "terminal_purchasing_power_loss_share": float((real[:, -1] < 100).mean()),
                "mdd_exceedance": {
                    f"{level:.2f}": {"count": int((metrics["mdd"] > level).sum()),
                                       "share": float((metrics["mdd"] > level).mean()),
                                       "expected_excess": float(np.maximum(metrics["mdd"] - level, 0).mean())}
                    for level in LOSS_LEVELS},
                "mean_cumulative_nominal_fees": float(fees[:, :h].sum(axis=1).mean()),
                "trade_check_share": float((moved[:, :h] > 1e-9).mean()),
            }
        output[name] = {"target": target, "execution": "annually restored fixed target, fee after target weights; initial allocation charged, no terminal liquidation", "horizons": by_horizon}
    return output


def liquidity_review(returns, inflation):
    # This is a separate, explicitly hypothetical liability-planning identity.
    # No withdrawals are actually inserted into the contest wealth simulations.
    bill_gross = (1 + returns[:, :2, 2]) / (1 + inflation[:, :2])
    bill_accumulation = np.cumprod(bill_gross, axis=1)
    examples = {}
    for k in (1, 2):
        pv_multiplier = np.sum(1 / bill_accumulation[:, :k], axis=1)
        # Validate the funding identity without changing contest wealth paths.
        reserve = .10 * pv_multiplier
        for t in range(k):
            reserve = reserve * bill_gross[:, t] - .10
            if np.any(reserve < -1e-12):
                raise AssertionError("computed reserve cannot fund the hypothetical payment")
        np.testing.assert_allclose(reserve, 0, atol=1e-12)
        examples[str(k)] = {
            "reserve_multiplier_for_annual_real_need_q": stats(pv_multiplier),
            "interpretation": "initial reserve fraction b=q*sum_{j=1}^K(1/B_j), where B_j is cumulative real bill gross and each year-end payment is q times initial real wealth",
            "scenarios": len(returns),
            "illustrative_annual_need_fractions": {
                f"{q:.2f}": {"zero_real_return_reserve": q * k,
                               "empirical_p90_required_reserve": float(q * np.quantile(pv_multiplier, .90)),
                               "empirical_p95_required_reserve": float(q * np.quantile(pv_multiplier, .95))}
                for q in (.05, .10, .15)},
        }
    return {
        "identifiability": {
            "fixed_bill_floors_0.30_0.10_0_identified": False,
            "reason": "High/Medium/Low are ordinal. The questionnaire supplies no required amount, access date, withdrawal schedule, external reserve, or permitted liquidation loss. Market returns cannot identify those preferences.",
            "market_liquidity_vs_reserve": "The contest supplies no asset-specific settlement delay, sale-capacity or liquidity penalty. Assets can be transferred at the common fee at annual reviews. A bill floor is a declared reserve/market-risk buffer proxy, not a measured marketability requirement.",
            "main_experiment": "No deposits or withdrawals, as required by the problem. Potential-spending scenarios below are supplemental interpretation only.",
            "low_does_not_imply_zero": "A Low label does not prove that required reserves equal zero.",
        },
        "questions_to_clarify": ["How much purchasing power could be needed?", "When and how quickly could it be needed?", "Are there reserves outside this account?", "What liquidation loss or delay is acceptable?"],
        "possible_predeclared_policy_cases_not_recommendations": {
            "ordinal_floor_approach": "Declare several monotone floor triplets before testing; treat .30/.10/0 as one unsupported policy case, not a calibrated baseline or a regulatory rule.",
            "reserve_coverage_approach": "Declare annual potential need q and coverage years K=1 or2; derive b from an explicit bill-accumulation/liability convention. Do not assign q to High/Medium/Low without elicitation or an openly labelled assumption.",
            "zero_real_return_examples": ".30 corresponds to one year at q=.30 or two years at q=.15; .10 corresponds to one year at q=.10 or two years at q=.05. These interpretations are assumptions, not facts about the five clients.",
            "immediate_need": "A known immediate need fraction q requires at least q available before earning returns; it is different from the year-end payment examples.",
            "no_costless_safety_claim": "Higher bill reserves can sacrifice growth and still lose purchasing power; they do not solve inflation risk.",
        },
        "hypothetical_year_end_real_payment_examples": examples,
        "example_limitations": "Uses first1or2 years of 120 original scenarios. q, K, payment timing and P90/P95 reserve quantile are extra assumptions. Empirical reserve quantiles are not independently certified coverage and are not client preferences. No spending, fees on withdrawals, income or external resources are supplied; these examples must not replace the contest no-withdrawal paths.",
        "primary_sources": [
            {"title": "2026 MCM Senior Group Problem A", "path": "docs/problem_A.pdf", "pages": [1, 2], "supports": "Only horizon, loss tolerance and ordinal liquidity are supplied; no later deposits or withdrawals are allowed."},
            {"title": "SEC Investor.gov: Investment Products", "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products", "accessed": "2026-10-17", "supports": "Liquidity concerns ease and cost of converting investments into usable money; product choice also depends on goals, timeframe and risk tolerance.", "does_not_support": "Any .30/.10/0 floor, liability fraction or probability guarantee."},
            {"title": "FINRA: Suitability Considerations and Sales Practices for Senior Investors, 2016 conference materials, reproduction of Notice11-25", "url": "https://www.finra.org/sites/default/files/2016_SIPC_Suitability.pdf", "accessed": "2026-10-17", "supports": "Conceptual liquidity needs include obligations, speed, amount and conversion losses/costs.", "does_not_support": "A current regulatory conclusion or a mandated allocation for this academic contest."},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/model_review/development_structure.json")
    args = parser.parse_args()
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if ids.tolist() != list(range(1, 121)) or returns.shape != (120, 30, 3):
        raise AssertionError("unexpected development scope")
    digest = hashlib.sha256()
    for array in (ids.astype("<i8"), returns.astype("<f8"), inflation.astype("<f8")):
        digest.update(array.tobytes(order="C"))
    rng = np.random.default_rng(SEED)
    indices = rng.integers(120, size=(BOOTSTRAP_REPLICATES, 120))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"scenario_ids": ids.tolist(), "final_scenarios_loaded": False,
                  "parsed_development_arrays_sha256": digest.hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "units": "decimal returns/risk fractions, wealth starts at100",
                  "quantiles": "linear interpolation for descriptive quantiles; explicitly suffixed inverted_cdf P10/P90 are empirical inverse-CDF alternatives for chance-constraint consistency; median uses np.median",
                  "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                  "purpose": "development-only structural diagnostics; no new menu or learned policy selected; V1 unchanged"},
        "market": market_diagnostics(returns, inflation, indices),
        "predeclared_diagnostic_paths": path_risk_diagnostics(returns, inflation),
        "liquidity_assumptions": liquidity_review(returns, inflation),
        "interpretation_limits": [
            "All probabilities describe the supplied synthetic development distribution, not guaranteed future markets.",
            "3600 annual rows are not independent trials; the original scenario is the sampling unit.",
            "Nominal annual returns and real gross/return summaries exclude portfolio fees; diagnostic portfolio paths include fees consistently.",
            "Serial structure and contemporaneous correlation do not prove forecast usefulness or identify a causal inflation mechanism.",
            "Annual observations omit intrayear drawdowns and transaction timing detail.",
            "No parameters or products may be justified by old final outcomes or a desired client mapping.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenario_count": len(ids),
                      "scenario_id_range": [int(ids[0]), int(ids[-1])],
                      "final_scenarios_loaded": False,
                      "annual_real_loss_shares": {a: result["market"]["assets"][a]["real_annual_loss_share_scenario_equal"] for a in ASSETS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
