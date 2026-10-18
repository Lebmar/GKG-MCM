"""Pure selection and case accounting for complete-process calibration.

No market loading, menu fitting or simulation occurs here. Static assignments
are classified before selecting an execution candidate; non-Balanced cases are
distinct from a request for human review.
"""
from __future__ import annotations

import math
import numpy as np

BETAS = (0., .025, .05, .075, .10)
ALPHA = .10
GROWTH_TOL = 1e-12
STATUSES = ("automated", "execution_review", "risk_review", "not_balanced_case")


def classify_assignment(assignment):
    if assignment == "Balanced": return "balanced_case"
    if assignment == "Review": return "risk_review"
    if assignment in ("Conservative", "Growth"): return "not_balanced_case"
    raise ValueError("Unknown effective product assignment")


def price_selection(summaries, reference_count, n, beta):
    """Select a declared price from full-policy training counts and growth."""
    if isinstance(beta, (bool, np.bool_)) or beta not in BETAS:
        raise ValueError("Require a predeclared dynamic training beta")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Require a positive original training-path count")
    if isinstance(reference_count, (bool, np.bool_)) or not isinstance(reference_count, (int, np.integer)) or not 0 <= reference_count <= n:
        raise ValueError("Reference count must be an integer within original n")
    if len(summaries) != 9 or [x["execution_candidate_order"] for x in summaries] != list(range(9)):
        raise ValueError("Require all nine candidates in the declared order")
    cap = min(int(reference_count), math.floor(beta * n + 1e-12))
    classified = []
    for row in summaries:
        k, g = row["breach_count"], row["mean_log_growth"]
        if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                or not 0 <= k <= n or row["n_training"] != n or not np.isfinite(g)):
            raise ValueError("All candidates must describe every full training path")
        classified.append({**row, "training_beta": float(beta), "training_breach_cap": cap,
                           "training_eligible": int(k) <= cap})
    eligible = [x for x in classified if x["training_eligible"]]
    maximum = max((x["mean_log_growth"] for x in eligible), default=None)
    chosen = next((x for x in eligible if x["mean_log_growth"] >= maximum - GROWTH_TOL), None)
    return {"training_beta": float(beta), "service_alpha": ALPHA,
            "n_training": int(n), "reference_training_breach_count": int(reference_count),
            "training_breach_cap": cap, "candidate_summaries": classified,
            "maximum_eligible_mean_log_growth": maximum,
            "selected_candidate_order": chosen["execution_candidate_order"] if chosen else None,
            "execution_status": "automated" if chosen else "execution_review",
            "execution_review_reason": None if chosen else "no_admissible_training_price"}


def final_status(case, selected_beta, selected_order):
    if case in ("risk_review", "not_balanced_case"): return case, case
    if case != "balanced_case": raise ValueError("Unknown case")
    if selected_beta is None: return "execution_review", "no_eligible_inner_beta"
    if selected_order is None: return "execution_review", "no_admissible_training_price"
    return "automated", None


def denominator_counts(rows):
    if len({int(x["scenario_id"]) for x in rows}) != len(rows):
        raise ValueError("Each original path must occur exactly once")
    counts = dict.fromkeys(("original_n", "balanced_case_n", "risk_review_n", "not_balanced_case_n",
                           "case_auto_n", "case_execReview_n"), 0)
    for row in rows:
        status = row["execution_status"]
        if status not in STATUSES: raise ValueError("Unknown service status")
        case = classify_assignment(row["q1_effective_assignment"])
        if status in ("automated", "execution_review") and case != "balanced_case":
            raise ValueError("Non-Balanced product entered the Q2 case")
        if status in ("risk_review", "not_balanced_case") and status != case:
            raise ValueError("Risk review and outside-case status were confused")
        if bool(row["assigned"]) != (status == "automated"):
            raise ValueError("Financial admission and execution status disagree")
        counts["original_n"] += 1
        if case == "balanced_case":
            counts["balanced_case_n"] += 1
            counts["case_auto_n" if status == "automated" else "case_execReview_n"] += 1
        else: counts[case + "_n"] += 1
        if status != "automated":
            for field in ("log_growth", "mdd", "breach"):
                value = row.get(field)
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    raise ValueError("Nonautomated financial outcomes must remain NULL")
    assert counts["original_n"] == counts["balanced_case_n"] + counts["risk_review_n"] + counts["not_balanced_case_n"]
    assert counts["balanced_case_n"] == counts["case_auto_n"] + counts["case_execReview_n"]
    return counts


def beta_summary(rows, beta, expected_ids):
    if any(x["training_beta"] != beta for x in rows): raise ValueError("Mixed beta records")
    if sorted(x["scenario_id"] for x in rows) != sorted(expected_ids):
        raise ValueError("Missing, duplicate or foreign inner validation paths")
    counts = denominator_counts(rows)
    auto = [x for x in rows if x["assigned"]]
    for row in auto:
        if not np.isfinite(row["log_growth"]) or not np.isfinite(row["mdd"]) or not isinstance(row["breach"], (bool, np.bool_)):
            raise ValueError("Automated growth, loss and event must be observed")
        if bool(row["breach"]) != (row["mdd"] - .4 > 1e-12):
            raise ValueError("Event and whole-H drawdown disagree")
    k = sum(bool(x["breach"]) for x in auto) if auto else None
    eligible = bool(auto) and k <= math.floor(ALPHA * len(auto) + 1e-12)
    return {"training_beta": float(beta), **counts,
            "breach_count": k, "conditional_breach_probability": k / len(auto) if auto else None,
            "conditional_mean_log_growth": float(np.mean([x["log_growth"] for x in auto])) if auto else None,
            "auto_inner_folds": len({x["dynamic_inner_fold"] for x in auto}),
            "eligible": eligible, "reason": "passed_empirical_inner_screen" if eligible else
                "no_inner_auto_evidence" if not auto else "inner_conditional_risk_exceeds_reference"}


def choose_beta(summaries):
    if len(summaries) != len(BETAS) or [x["training_beta"] for x in summaries] != list(BETAS):
        raise ValueError("Require every predeclared beta in order")
    eligible = [x for x in summaries if x["eligible"]]
    if not eligible:
        return {"selected_beta": None, "status": "no_eligible_inner_beta",
                "maximum_auto_coverage": None, "maximum_mean_log_growth_at_coverage": None}
    coverage = max(x["case_auto_n"] for x in eligible)
    covered = [x for x in eligible if x["case_auto_n"] == coverage]
    growth = max(x["conditional_mean_log_growth"] for x in covered)
    chosen = min(x["training_beta"] for x in covered if x["conditional_mean_log_growth"] >= growth - GROWTH_TOL)
    return {"selected_beta": chosen, "status": "inner_beta_selected",
            "maximum_auto_coverage": coverage, "maximum_mean_log_growth_at_coverage": growth,
            "eligible_betas": [x["training_beta"] for x in eligible]}
