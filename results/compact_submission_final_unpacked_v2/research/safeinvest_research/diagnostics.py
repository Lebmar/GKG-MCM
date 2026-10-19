"""Array-only catalogue comparators and development-consistency diagnostics.

These functions never open market files, choose a final model, or certify a
population loss probability. Growth means expected annualized log growth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .metrics import NUMERIC_TOL, discrete_cvar
from .selection import pareto_frontier


def _vectors(growth, risk):
    growth, risk = np.asarray(growth, float), np.asarray(risk, float)
    if growth.ndim != 1 or risk.shape != growth.shape or not len(growth):
        raise ValueError("growth and risk must be nonempty equally sized vectors")
    if not np.all(np.isfinite(growth)) or not np.all(np.isfinite(risk)):
        raise ValueError("growth and risk must be finite")
    return growth, risk


def reference_menu(reference_growth, reference_cvar):
    """Predeclared H15 frontier endpoints and distinct risk-midpoint product.

    Return indices in increasing reference-risk order, plus explicit degeneracy
    flags. The middle point is closest to half the endpoint CVaR range; ties
    prefer higher growth, lower risk, then lower candidate ID. Equivalent
    frontier outcomes do not manufacture additional economic risk roles.
    """
    growth, risk = _vectors(reference_growth, reference_cvar)
    frontier = pareto_frontier(growth, risk)
    low = min(frontier.tolist(), key=lambda i: (risk[i], -growth[i], i))
    high = min(frontier.tolist(), key=lambda i: (-growth[i], risk[i], i))
    selected = [int(low)]
    if high != low:
        selected.append(int(high))
    midpoint = float((risk[low] + risk[high]) / 2)
    interior = [int(i) for i in frontier if i not in selected
                and risk[i] > risk[low] + NUMERIC_TOL
                and risk[i] < risk[high] - NUMERIC_TOL]
    if interior:
        distance = min(abs(risk[i] - midpoint) for i in interior)
        tied = [i for i in interior if abs(risk[i] - midpoint) <= distance + NUMERIC_TOL]
        selected.append(min(tied, key=lambda i: (-growth[i], risk[i], i)))
    selected.sort(key=lambda i: (risk[i], growth[i], i))
    distinct_risks = len(selected) == 3 and all(
        risk[b] - risk[a] > NUMERIC_TOL for a, b in zip(selected[:-1], selected[1:]))
    reasons = []
    if low == high:
        reasons.append("one_frontier_endpoint_serves_both_extremes")
    if not interior:
        reasons.append("no_distinct_interior_risk_point")
    if not distinct_risks:
        reasons.append("fewer_than_three_distinct_reference_risk_levels")
    return {"indices": selected, "reference_frontier": frontier.tolist(),
            "endpoint_indices": [int(low), int(high)], "risk_midpoint": midpoint,
            "degenerate": not distinct_risks, "degeneracy_reasons": reasons,
            "complete": len(selected) == 3,
            "interpretation": "Reference-H15 catalogue comparator, not an inferred customer preference."}


def _integer_indices(values, name, *, unique):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values)
            or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0)):
        raise ValueError(f"{name} must be a nonempty nonnegative integer vector")
    if unique and len(np.unique(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates")
    return values.astype(int, copy=False)


def _order_record(values, candidate_ids, reference_values=None):
    values = np.asarray(values, float)
    ties, reversals, crossings = [], [], []
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            pair = [int(candidate_ids[a]), int(candidate_ids[b])]
            delta = float(values[b] - values[a])
            if abs(delta) <= NUMERIC_TOL:
                ties.append(pair)
            elif delta < 0:
                reversals.append(pair)
            if reference_values is not None:
                old_delta = float(reference_values[b] - reference_values[a])
                if abs(old_delta) > NUMERIC_TOL and abs(delta) > NUMERIC_TOL and old_delta * delta < 0:
                    crossings.append(pair)
    order = sorted(range(len(values)), key=lambda i: (values[i], candidate_ids[i]))
    return {"values_in_reference_label_order": values.tolist(),
            "ascending_candidate_indices": [int(candidate_ids[i]) for i in order],
            "tied_pairs": ties, "against_reference_label_order": reversals,
            "crossings_against_same_metric_at_reference_horizon": crossings}


def label_order_diagnostics(metrics_by_H: Mapping, menuindices, trainindices,
                            reference_horizon=15):
    """Check expected-log growth, CVaR90 and P90 order using nominated rows.

    Input arrays have shape (scenario, candidate). No held-out row enters a
    summary. Repeated training rows support bootstrap samples. Labels are
    determined by H15 CVaR90; a growth reversal is not itself a risk reversal.
    """
    menu = _integer_indices(menuindices, "menuindices", unique=True)
    rows = _integer_indices(trainindices, "trainindices", unique=False)
    if reference_horizon not in metrics_by_H:
        raise ValueError("the reference horizon must exist in metrics_by_H")
    summaries = {}
    for horizon, fields in metrics_by_H.items():
        growth, mdd = np.asarray(fields["log_growth"], float), np.asarray(fields["mdd"], float)
        if (growth.ndim != 2 or mdd.shape != growth.shape
                or np.max(rows) >= growth.shape[0] or np.max(menu) >= growth.shape[1]):
            raise ValueError("metric arrays must share valid scenario-by-candidate shapes")
        selected_growth = growth[np.ix_(rows, menu)]
        selected_mdd = mdd[np.ix_(rows, menu)]
        if (not np.all(np.isfinite(selected_growth)) or not np.all(np.isfinite(selected_mdd))
                or np.any(selected_mdd < 0) or np.any(selected_mdd > 1)):
            raise ValueError("nominated metric rows must contain finite growth and valid drawdown")
        summaries[horizon] = {"mean_log_growth": selected_growth.mean(axis=0),
                              "cvar90_mdd": discrete_cvar(selected_mdd),
                              "p90_mdd": np.quantile(selected_mdd, .9, axis=0, method="inverted_cdf")}
    ref = summaries[reference_horizon]
    positions = sorted(range(len(menu)), key=lambda i: (ref["cvar90_mdd"][i],
                                                        ref["mean_log_growth"][i], menu[i]))
    ordered = menu[positions]
    reference = {key: values[positions] for key, values in ref.items()}
    by_horizon = {}
    for horizon, fields in sorted(summaries.items()):
        by_horizon[str(horizon)] = {
            key: _order_record(values[positions], ordered, reference[key])
            for key, values in fields.items()}
    risk_metrics = ("cvar90_mdd", "p90_mdd")
    return {
        "reference_horizon": int(reference_horizon), "n_training": len(rows),
        "reference_label_candidate_indices": ordered.tolist(),
        "input_menu_matches_reference_order": bool(np.array_equal(menu, ordered)),
        "by_horizon": by_horizon,
        "any_risk_label_reversal": any(
            bool(fields[key]["against_reference_label_order"])
            for fields in by_horizon.values() for key in risk_metrics),
        "any_cross_horizon_risk_reversal": any(
            bool(fields[key]["crossings_against_same_metric_at_reference_horizon"])
            for fields in by_horizon.values() for key in risk_metrics),
        "reference_risk_ties": by_horizon[str(reference_horizon)]["cvar90_mdd"]["tied_pairs"],
        "interpretation": "Descriptive order diagnostics; no universal hierarchy or population certificate."}


def _count(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def empirical_screen_flags(breach_count, n, alpha=.1):
    """Apply the empirical count budget and identify one-path boundary cases."""
    n, k = _count(n, "n", minimum=1), _count(breach_count, "breach_count")
    if k > n:
        raise ValueError("breach_count cannot exceed n")
    if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must be finite and in [0,1]")
    maximum = int(np.floor(n * float(alpha) + NUMERIC_TOL))
    passed = k <= maximum
    return {"breach_count": k, "n": n, "alpha": float(alpha),
            "breach_probability": k / n, "maximum_empirical_count": maximum,
            "empirical_pass": passed,
            "one_path_boundary_pass": bool(passed and k == maximum and maximum < n),
            "one_path_boundary_fail": bool(not passed and k == maximum + 1),
            "population_certified": False}


def _automatic(assignment):
    if assignment is None:
        return False
    if not isinstance(assignment, str):
        raise ValueError("assignments must be product names, a referral string, or None")
    return assignment.strip().casefold() not in {"review", "human review", "refer", "referral"}


def sampling_flags(full_fit: Mapping, folds: Sequence[Mapping], oof_counts: Mapping,
                   alpha=None, expected_folds=5):
    """Flag per-client development consistency from root-supplied fit records.

    ``full_fit`` and each fold use ``fit_menu``'s assignments/details format.
    ``oof_counts[client]`` supplies breach_count and n, counting held-out paths
    only where that fold froze an automatic assignment. Referrals are not
    fabricated as zero-risk investments. Returned grades are diagnostics.
    """
    if alpha is None:
        alpha = full_fit["alpha"]
    empirical_screen_flags(0, 1, alpha)
    expected_folds = _count(expected_folds, "expected_folds", minimum=1)
    assignments = full_fit["assignments"]
    fold_assignments = [item["assignments"] for item in folds]
    if any(set(item) != set(assignments) for item in fold_assignments):
        raise ValueError("all fold assignments must cover the same named clients")
    complete = len(folds) == expected_folds
    results = {}
    for client, assignment in assignments.items():
        auto = _automatic(assignment)
        statuses = [_automatic(item[client]) for item in fold_assignments]
        chosen = [item for item in full_fit.get("client_product_details", [])
                  if item["client"] == client and item.get("chosen", False)]
        if auto and len(chosen) != 1:
            raise ValueError("an automatic full-fit assignment needs exactly one chosen detail")
        screen = empirical_screen_flags(chosen[0]["breach_count"], chosen[0]["n"], alpha) if auto else None
        counts = oof_counts.get(client, {"breach_count": 0, "n": 0})
        n = _count(counts["n"], "OOF n")
        k = _count(counts["breach_count"], "OOF breach_count")
        if k > n:
            raise ValueError("OOF breach_count cannot exceed n")
        p = k / n if n else None
        warning = p is not None and p > float(alpha) + NUMERIC_TOL
        boundary = bool(screen and screen["one_path_boundary_pass"])
        disagreement = any(status != auto for status in statuses)
        mixed_folds = bool(statuses and any(statuses) and not all(statuses))
        if not complete:
            grade = "incomplete_internal_diagnostics"
        elif screen is not None and not screen["empirical_pass"]:
            grade = "full_fit_screen_failure"
        elif not auto and not any(statuses) and n == 0:
            grade = "consistent_referral"
        elif warning:
            grade = "internal_calibration_warning"
        elif disagreement or boundary:
            grade = "sampling_sensitive"
        elif auto and all(statuses) and n > 0:
            grade = "development_consistent_auto"
        else:
            grade = "insufficient_evidence"
        results[client] = {
            "grade": grade, "full_assignment": assignment, "full_screen": screen,
            "automatic_training_folds": int(sum(statuses)), "n_folds": len(folds),
            "fold_diagnostics_complete": complete, "mixed_fold_coverage": mixed_folds,
            "full_vs_fold_coverage_disagreement": disagreement,
            "role_assignment_varies": len({item[client] for item in fold_assignments}) > 1,
            "oof_breach_count": k, "oof_n_automatically_assigned": n,
            "oof_breach_probability": p, "internal_calibration_warning": warning,
            "population_certified": False}
    return results


def policy_flags(case_fits: Mapping, primary_case):
    """Separate declared policy dependence from sampling uncertainty.

    Values may be full ``fit_menu`` records or simple assignment mappings. With
    plain mappings only role names can be compared; unchanged names do not
    establish unchanged weights. The function does not rank policy desirability.
    """
    if primary_case not in case_fits or not case_fits:
        raise ValueError("primary_case must exist")
    assignments = {case: fit.get("assignments", fit) for case, fit in case_fits.items()}
    clients = set(assignments[primary_case])
    if any(set(values) != clients for values in assignments.values()):
        raise ValueError("policy cases must contain the same clients")
    results = {}
    for client in sorted(clients):
        values = {str(case): mapping[client] for case, mapping in assignments.items()}
        automatic = [_automatic(value) for value in values.values()]
        coverage_sensitive = len(set(automatic)) > 1
        role_sensitive = len(set(values.values())) > 1
        targets = []
        target_available = True
        for case, fit in case_fits.items():
            label = assignments[case][client]
            if not _automatic(label):
                continue
            matches = [item for item in fit.get("menu", []) if item["name"] == label]
            if len(matches) != 1:
                target_available = False
                break
            target = np.asarray(matches[0]["weights"], float)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError("menu target weights must be a finite length-three vector")
            targets.append(target)
        target_sensitive = (any(np.any(np.abs(target - targets[0]) > NUMERIC_TOL)
                                for target in targets[1:]) if target_available and targets else None)
        grade = "P2" if coverage_sensitive else "P1" if role_sensitive or target_sensitive else "P0"
        results[client] = {"grade": grade, "assignments_by_case": values,
                           "coverage_policy_sensitive": coverage_sensitive,
                           "role_policy_sensitive": role_sensitive,
                           "target_comparison_available": bool(target_available and targets),
                           "target_policy_sensitive": target_sensitive,
                           "meaning": "Normative policy dependence, not statistical failure or risk certification."}
    return results
