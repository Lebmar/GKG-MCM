"""Reproduce the objective/risk foundation audit without reading market data.

Inputs are mathematical constants and explicitly constructed counterexamples.
This script does not import the strategy engine, select weights, or access any
development/final scenario. It leaves the archived V1 experiment unchanged.
"""
from __future__ import annotations

import hashlib
import json
from math import ceil, expm1, log
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import beta, binom


ROOT = Path(__file__).resolve().parents[1]


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """Pointwise one-sided exact upper bound for a fixed iid Bernoulli rule."""
    if (isinstance(k, bool) or isinstance(n, bool)
            or not isinstance(k, (int, np.integer))
            or not isinstance(n, (int, np.integer))
            or not (n > 0 and 0 <= k <= n and 0 < delta < 1)):
        raise ValueError("Require integer counts n>0, 0<=k<=n and 0<delta<1")
    return 1.0 if k == n else float(beta.ppf(1 - delta, k + 1, n - k))


def relative_mdd(wealth: list[float]) -> float:
    values = np.asarray(wealth, dtype=float)
    if (values.ndim != 1 or len(values) < 2
            or not np.all(np.isfinite(values)) or np.any(values <= 0)):
        raise ValueError("Require a positive annual wealth path including time zero")
    peaks = np.maximum.accumulate(values)
    return float(np.max((peaks - values) / peaks))


def discrete_cvar(losses: list[float], probabilities: list[float], level=0.90) -> float:
    """General-distribution CVaR via its scalar variational definition.

    The minimizer can be chosen at a support point for this finite distribution;
    this handles atoms, unlike averaging all observations strictly above VaR.
    """
    losses = np.asarray(losses, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (losses.shape != probabilities.shape or losses.ndim != 1
            or len(losses) == 0 or not np.all(np.isfinite(losses))
            or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-12)
            or not 0 < level < 1):
        raise ValueError("Require a finite probability distribution and 0<level<1")
    probabilities = probabilities / probabilities.sum()
    values = [eta + np.dot(probabilities, np.maximum(losses - eta, 0)) / (1 - level)
              for eta in np.unique(losses)]
    return float(min(values))


def distribution_metrics(paths, probabilities, tolerance):
    losses = np.array([relative_mdd(path) for path in paths])
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Require a loss tolerance between 0 and 1")
    # Validate the finite distribution before using it in any mean.
    tail_mean = discrete_cvar(losses, probabilities)
    probabilities = probabilities / probabilities.sum()
    # A floating-point guard for these constructed examples, not a risk budget.
    numerical_tolerance = 1e-12
    # The hand-built examples keep every terminal value at 100.
    return {
        "wealth_paths": paths,
        "probabilities": probabilities.tolist(),
        "mdd_by_outcome": losses.tolist(),
        "expected_log_terminal_ratio": float(np.dot(
            probabilities, np.log([path[-1] / path[0] for path in paths]))),
        "mean_mdd": float(np.dot(probabilities, losses)),
        "tolerance": tolerance,
        "exceedance_probability": float(np.dot(
            probabilities, losses > tolerance + numerical_tolerance)),
        "expected_excess_mdd": float(np.dot(probabilities, np.maximum(losses - tolerance, 0))),
        "cvar_90_mdd": tail_mean,
    }


def compute_audit() -> dict:
    if not __debug__:
        raise RuntimeError("Run this audit without -O so its verification assertions execute")
    lambda_rows = []
    for penalty in [10, 4, 1]:
        row = {"lambda": penalty, "mdd_increase": 0.01,
               "terminal_geometric_wealth_compensation": expm1(0.01 * penalty)}
        for horizon in [5, 15, 30]:
            row[f"annual_factor_minus_one_H{horizon}"] = expm1(0.01 * penalty / horizon)
        lambda_rows.append(row)

    confidence_rows = []
    for k, n in [(0, 24), (0, 30), (0, 120), (8, 80), (12, 120), (6, 120), (7, 120)]:
        upper = cp_upper(k, n)
        # Independent numerical route: invert the binomial CDF directly.
        inverted = brentq(lambda p: binom.cdf(k, n, p) - 0.05,
                          0.0, 1.0, xtol=1e-14)
        assert abs(upper - inverted) < 2e-12
        confidence_rows.append({"exceedances": k, "independent_paths": n,
                                "point_estimate": k / n,
                                "one_sided_95_upper": upper,
                                "binomial_cdf_at_upper": float(binom.cdf(k, n, upper))})

    max_k = max(k for k in range(121) if cp_upper(k, 120) <= 0.10)
    zero_n = ceil(log(0.05) / log(1 - 0.10))
    assert max_k == 6 and zero_n == 29
    assert cp_upper(0, zero_n) <= 0.10 < cp_upper(0, zero_n - 1)
    simultaneous = {str(n): 1 - (0.05 / 232) ** (1 / n) for n in [80, 120]}

    same_mean = {
        "steady_loss": distribution_metrics([[100, 90, 100]], [1], 0.25),
        "rare_deep_loss": distribution_metrics(
            [[100, 100, 100], [100, 50, 100]], [0.8, 0.2], 0.25),
    }
    same_probability = {
        "small_overshoot": distribution_metrics(
            [[100, 90, 100], [100, 79, 100]], [0.9, 0.1], 0.20),
        "large_overshoot": distribution_metrics(
            [[100, 90, 100], [100, 20, 100]], [0.9, 0.1], 0.20),
    }
    assert np.isclose(same_mean["steady_loss"]["mean_mdd"],
                      same_mean["rare_deep_loss"]["mean_mdd"])
    assert same_mean["steady_loss"]["exceedance_probability"] == 0
    assert same_mean["rare_deep_loss"]["exceedance_probability"] == 0.2
    assert np.isclose(same_probability["large_overshoot"]["expected_excess_mdd"] /
                      same_probability["small_overshoot"]["expected_excess_mdd"], 60)
    # Atom-sensitive check: 90% point mass at .10, 10% at .21 has CVaR_90=.21.
    assert np.isclose(same_probability["small_overshoot"]["cvar_90_mdd"], 0.21)
    # The worst 10% contains all 1% at .9 and only 9% of the .2 atom.
    assert np.isclose(discrete_cvar([0.1, 0.2, 0.9], [0.89, 0.10, 0.01]), 0.27)
    assert distribution_metrics([[100, 70, 100]], [1], 0.30)["exceedance_probability"] == 0
    for function, arguments in [
        (cp_upper, (0.5, 10)), (cp_upper, (0, 10.5)),
        (relative_mdd, ([100, float("nan"), 100],)),
        (discrete_cvar, ([0.1, float("inf")], [0.9, 0.1])),
    ]:
        try:
            function(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite or non-integer input was accepted")

    terminal_a = np.array([110.0, 160.0, 92.0])
    terminal_b = np.array([106.0, 140.0, 99.0])
    price_index = np.array([1.10, 1.50, 0.97])
    nominal_gap = float(np.mean(np.log(terminal_a / 100) - np.log(terminal_b / 100)))
    real_gap = float(np.mean(np.log(terminal_a / price_index / 100)
                             - np.log(terminal_b / price_index / 100)))
    assert abs(nominal_gap - real_gap) < 1e-14

    return {
        "scope": "Pure mathematical diagnostics; no market scenario or archived result is read.",
        "date_beijing": "2026-10-17",
        "scipy_version": scipy.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lambda_tradeoffs": lambda_rows,
        "lambda_interpretation": "Multiplicative compensation for a one-percentage-point increase in expected MDD, holding expected log terminal wealth minus lambda times expected MDD fixed. Annual entries are factors minus one, not additive CAGR percentage points.",
        "fixed_rule_binomial_examples": confidence_rows,
        "binomial_scope": "One-sided pointwise exact coverage requires a fixed strategy evaluated on independent identically distributed complete paths, without selection on these evaluation outcomes. These examples do not certify V1 or a new policy.",
        "maximum_k_at_n120_for_upper_le_10pct": max_k,
        "minimum_zero_exceedance_n_for_upper_le_10pct": zero_n,
        "zero_exceedance_bonferroni_M232_upper": simultaneous,
        "bonferroni_scope": "Illustration for one H,L and 232 finite strategies fixed independently of the evaluation sample. Not valid for the full learned forecast/policy family without further coverage work.",
        "same_mean_mdd_counterexample": same_mean,
        "same_exceedance_probability_counterexample": same_probability,
        "constructed_example_numerical_comparison_tolerance": 1e-12,
        "inflation_log_cancellation_example": {
            "nominal_terminal_a": terminal_a.tolist(),
            "nominal_terminal_b": terminal_b.tolist(),
            "common_price_index": price_index.tolist(),
            "nominal_expected_log_gap": nominal_gap,
            "real_expected_log_gap": real_gap,
            "scope": "Constructed outcomes verify a pathwise algebraic identity, not an empirical independence assumption.",
        },
        "sunk_mdd_example": {
            "past_mdd": 0.35, "tolerance": 0.25,
            "future_mdd_including_past_peak": [0.10, 0.30],
            "full_mdd_for_both_actions": [0.35, 0.35],
            "scope": "Two possible continuation-path MDD values; historical loss cannot be undone. Full-path exceedance is already certain after a past breach.",
        },
        "repeated_annual_hazard_example": {
            "conditional_new_breach_probability_each_year": 0.10,
            "years": 30,
            "full_horizon_breach_probability": 1 - 0.90 ** 30,
            "scope": "Illustration with a constant conditional hazard given survival. A 10% annual/new-decision budget is not a 10% full-horizon budget. This is not a fitted market model.",
        },
        "checks": {
            "beta_quantile_reconciles_with_binomial_cdf_inversion": True,
            "zero_exceedance_sample_size_boundary": True,
            "counterexamples_recomputed_from_positive_wealth_paths": True,
            "discrete_cvar_variational_definition_handles_atoms": True,
            "partial_boundary_atom_cvar_27pct": True,
            "equality_boundary_and_invalid_input_guards": True,
            "inflation_term_cancels_without_independence_assumption": True,
        },
    }


def main():
    result = compute_audit()
    destination = ROOT / "results/model_review/objective_risk_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(destination.relative_to(ROOT)),
                      "max_k_n120": result["maximum_k_at_n120_for_upper_le_10pct"],
                      "min_zero_breach_n": result["minimum_zero_exceedance_n_for_upper_le_10pct"],
                      "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
