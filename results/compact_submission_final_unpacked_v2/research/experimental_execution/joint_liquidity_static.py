"""Training-only joint risk/liquidity version of the frozen static learner.

The sole economic specification change is the predeclared bills-weight floors
High=.30, Medium=.10, Low=0. Every inner menu and feasible oracle is refitted
with these floors. This is not a filter applied to an old risk-only menu.
These normative allocation floors are not a calibrated withdrawal guarantee.
No function reads data or saved results. Original scenarios1..120 only may be
supplied, with fitting confined to the explicitly nominated training IDs.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from experimental_execution import nested_static_adapter as base
from safeinvest_research.selection import fit_menu
from safeinvest_research.static import simplex_grid

GAMMAS = base.GAMMAS
SERVICE_ALPHA = base.SERVICE_ALPHA
LIQUIDITY_FLOORS = {"high": .30, "medium": .10, "low": 0.}
PROFILES = [dict(name=n, horizon=h, loss_tolerance=l, liquidity=q)
            for n, h, l, q in [("A", 5, .1, "high"), ("B", 15, .25, "medium"),
                               ("C", 30, .4, "low"), ("D", 25, .15, "medium"),
                               ("E", 6, .35, "high")]]
NUMERIC_TOL = 1e-12
make_context = base.make_context
validate_context = base.validate_context
prepare_strict_family = base.prepare_strict_family


def _family(weights, candidate_original_indices):
    all_weights = simplex_grid(.05)
    allowed = np.flatnonzero(np.all(all_weights >= .05 - NUMERIC_TOL, axis=1))
    supplied = np.asarray(weights, dtype=float)
    if len(allowed) != 172 or not np.array_equal(supplied, all_weights[allowed]):
        raise ValueError("Require the exact unchanged strict172 weights and ordering")
    if candidate_original_indices is not None:
        mapping = np.asarray(candidate_original_indices)
        if (not np.issubdtype(mapping.dtype, np.integer)
                or np.issubdtype(mapping.dtype, np.bool_)
                or not np.array_equal(mapping, allowed)):
            raise ValueError("Require the exact strict172 local-to-original232 mapping")
    return supplied, allowed


def fit_gamma(metrics, weights, profiles, indices, gamma):
    """Refit the common menu and feasible oracles jointly at one original gamma.

    All five questionnaire profiles enter the unchanged coverage/regret menu
    objective. Liquidity does not change service alpha, loss limits or labels.
    Diagnostics count the full supplied family; they never change selection.
    """
    fitted = fit_menu(metrics, weights, profiles, indices, alpha=gamma,
                      liquidity_floors=dict(LIQUIDITY_FLOORS))
    fitted["training_screen_gamma"] = fitted.pop("alpha")
    fitted["solver_gamma"] = float(gamma)
    fitted["service_alpha"] = SERVICE_ALPHA
    fitted["liquidity_case"] = "joint_fixed_30_10_0"
    fitted["parameter_interpretation"] = (
        "gamma is the unchanged technical empirical screen; service alpha=.10; "
        "bills floors30/10/0 are a predeclared normative liquidity proxy")
    for detail in fitted["client_product_details"]:
        detail["training_screen_feasible"] = detail.pop("sample_feasible")
    diagnostics = {}
    indices = np.asarray(indices, dtype=int)
    weights = np.asarray(weights, dtype=float)
    for p in profiles:
        # Match empirical_risk's subtraction-before-comparison convention,
        # including possible one-ULP differences from D > L + tolerance.
        risk = np.mean(np.asarray(metrics[p["horizon"]]["mdd"])[indices]
                       - p["loss_tolerance"] > NUMERIC_TOL, axis=0) <= gamma + NUMERIC_TOL
        liquid = weights[:, 2] >= LIQUIDITY_FLOORS[p["liquidity"]] - NUMERIC_TOL
        joint = risk & liquid
        diagnostics[p["name"]] = {
            "risk_feasible_n": int(risk.sum()), "liquidity_feasible_n": int(liquid.sum()),
            "joint_feasible_n": int(joint.sum()),
            "menu_joint_feasible_n": sum(bool(joint[x["candidate_index"]]) for x in fitted["menu"]),
            "floor": LIQUIDITY_FLOORS[p["liquidity"]], "training_n": len(indices),
            "training_screen_gamma": float(gamma)}
    fitted["feasibility_diagnostics"] = diagnostics
    return fitted


def _review_details(refit, profiles):
    result = {}
    admission = {row["client"]: row for row in refit["admission_details"]}
    fit = refit["raw_fit"]
    for p in profiles:
        name = p["name"]
        reasons = []
        if refit["effective_assignments"][name] == "Review":
            if fit is None:
                reasons = ["calibration_unsupported"]
            elif admission[name]["reason"] == "no_inner_auto_evidence":
                reasons = ["no_inner_auto_evidence"]
            else:
                diag = fit["feasibility_diagnostics"][name]
                if diag["risk_feasible_n"] == 0:
                    reasons.append("no_risk_feasible_candidates")
                if diag["liquidity_feasible_n"] == 0:
                    reasons.append("no_liquidity_feasible_candidates")
                if not reasons and diag["joint_feasible_n"] == 0:
                    reasons.append("risk_liquidity_intersection_empty")
                if not reasons:
                    reasons.append("joint_menu_referral")
        result[name] = {"q1_review_kind": reasons[0] if reasons else None,
                        "q1_review_reasons": reasons,
                        "effective_assignment": refit["effective_assignments"][name],
                        "feasibility_diagnostics": deepcopy(fit["feasibility_diagnostics"][name]) if fit else None}
    return result


def calibrate_liquidity_static(metrics_by_horizon, weights, profiles, context, *,
                               candidate_original_indices=None):
    """Relearn gamma/menu/effective assignments on T with fixed joint floors.

    Same four pre-recorded static folds, original gammas and raw all-profile
    objective, pooled original-path validation and unchanged final admission as
    nested_static_adapter.calibrate_static. Only liquidity floors change.
    T may be120/96/90/72. The family is exactly strict172, including exact
    thirds. Return the old adapter schema plus descriptive review diagnostics;
    candidate_index stays local and original232 identifiers are always added.
    No outside-T value participates, including in explanatory diagnostics.
    """
    canonical = validate_context(context)
    weights, mapping = _family(weights, candidate_original_indices)
    if profiles != PROFILES:
        raise ValueError("Require the unchanged five questionnaire profiles, horizons and loss limits")
    weights = base._validate_inputs(metrics_by_horizon, weights, profiles, canonical["training_ids"])
    frozen = base.frozen_nested
    summaries, records, folds_by_gamma = [], [], {}
    for gamma in GAMMAS:
        rows, fits = [], []
        for fold in canonical["inner_folds"]:
            fitted = fit_gamma(metrics_by_horizon, weights, profiles,
                               np.asarray(fold["train_ids"]) - 1, gamma)
            fitted.update(inner_fold=fold["inner_fold"], train_ids=fold["train_ids"], test_ids=fold["test_ids"])
            fits.append(fitted)
            rows.extend(frozen.path_records(metrics_by_horizon, profiles, fitted, fold["test_ids"],
                        context=canonical["name"], gamma=gamma, inner_fold=fold["inner_fold"]))
        summaries.append(frozen.inner_summary(rows, gamma))
        records.extend(rows)
        folds_by_gamma[str(gamma)] = fits
    selection = frozen.select_gamma(summaries)
    gamma = selection["selected_gamma"]
    chosen = next((s for s in summaries if s["training_screen_gamma"] == gamma), None)
    raw_fit = None
    if gamma is not None:
        raw_fit = fit_gamma(metrics_by_horizon, weights, profiles,
                            np.asarray(canonical["training_ids"]) - 1, gamma)
        counts = np.asarray([r["breach_count"] for r in raw_fit["client_product_details"]], dtype=int)
        upper = frozen.simultaneous_upper(counts, len(canonical["training_ids"]), 1160, .05)
        for detail, bound in zip(raw_fit["client_product_details"], upper, strict=True):
            detail.update(postfit_static_family_risk_upper=float(bound),
                          postfit_bound_within_service_alpha=bool(bound <= SERVICE_ALPHA),
                          postfit_bound_used_in_selection=False)
    refit = {"context": canonical["name"], "context_id": canonical["context_id"],
             "training_ids": canonical["training_ids"], "outer_test_ids": canonical["outer_test_ids"],
             "selection": selection, "selected_gamma_inner_evidence": chosen, "raw_fit": raw_fit,
             **frozen.apply_admission(raw_fit, profiles, chosen)}
    refit["q1_review_details"] = _review_details(refit, profiles)
    output = base.frozen_strict.annotate_indices({"refit": refit, "inner_summaries": summaries,
        "inner_fold_fits": folds_by_gamma, "inner_oof_paths": records}, mapping)
    output["q2_case"] = base.classify_balanced_case(output["refit"])
    case = output["q2_case"]
    case.update(deepcopy(output["refit"]["q1_review_details"]["C"]))
    case["public_status"] = "q1_review" if case["status"] == "risk_review" else case["status"]
    output["context"] = deepcopy(canonical)
    output["metadata"] = {"training_only": True, "market_data_loaded": False,
        "scenario_row_convention": "original scenario i at row i-1, rows1..120 only",
        "candidate_count": 172, "candidate_index_space": "local plus original232 fields",
        "service_alpha": SERVICE_ALPHA, "gamma_candidates": list(GAMMAS),
        "liquidity_case": "joint_fixed_30_10_0", "liquidity_floors": dict(LIQUIDITY_FLOORS),
        "liquidity_interpretation": "fixed normative bills-weight proxy; not a withdrawal-sufficiency guarantee",
        "static_gamma_objective": "unchanged raw all-five-profile inner OOF",
        "balanced_only_applied_after_Q1": True,
        "review_semantics": "internal risk_review case is public q1_review; consult per-profile reasons",
        "inner_fold_aggregation": "pool each original path once, not an unweighted mean of fold statistics",
        "postfit_bound_used_in_selection": False, "risk_certificate": False}
    return output
