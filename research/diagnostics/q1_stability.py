"""Training-only individual static feasible-set diagnostics, not a new menu.

The five profiles are representative design cases, not IID customer samples.
An individual optimum omits the shared-three-product restriction and never
replaces the frozen common menu or its effective suitability decisions.
"""
from __future__ import annotations

import math
import numpy as np

TOL = 1e-12
EPSILON_ANNUAL_LOG_GROWTH = .001
PROFILES = (
    dict(name="A",horizon=5,loss_tolerance=.10,liquidity="high"),
    dict(name="B",horizon=15,loss_tolerance=.25,liquidity="medium"),
    dict(name="C",horizon=30,loss_tolerance=.40,liquidity="low"),
    dict(name="D",horizon=25,loss_tolerance=.15,liquidity="medium"),
    dict(name="E",horizon=6,loss_tolerance=.35,liquidity="high"),
)
FLOORS = dict(high=.30,medium=.10,low=0.)


def development_ids(values, *, maximum=120):
    values = list(values)
    if (not values or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer))
            or not 1 <= i <= min(maximum,120) for i in values)
            or len(set(values)) != len(values)):
        raise ValueError("Require unique original integer development IDs1-120")
    return [int(i) for i in values]


def individual_set(log_growth, mdd, weights, training_ids, *, L, liquidity_floor,
                   gamma, raw_menu_index=None):
    """Use nominated training rows only, retaining the full epsilon-near-best set.

    The fixed epsilon=.001 is a tolerance in annual *log growth*, not a claim
    that a particular risk parameter is identified or CAGR differs by0.1pp.
    Gamma=None means no selected technical screen; its set is unavailable,
    distinct from a valid screen with an empty feasible set.
    """
    w = np.asarray(weights,float); g = np.asarray(log_growth,float); d = np.asarray(mdd,float)
    if (w.ndim != 2 or w.shape[1] != 3 or len(w) == 0 or not np.isfinite(w).all()
            or np.any(w < 0) or not np.allclose(w.sum(axis=1),1,atol=TOL,rtol=0)
            or g.ndim != 2 or g.shape != d.shape or g.shape[1] != len(w)):
        raise ValueError("Require simplex targets and aligned scenario-by-candidate fields")
    ids = development_ids(training_ids,maximum=len(g)); index = np.array(ids)-1
    if not 0 <= L < 1 or not 0 <= liquidity_floor <= 1:
        raise ValueError("Require valid profile loss tolerance and normative bill floor")
    result = dict(training_n=len(ids),training_ids=ids,L=float(L),liquidity_floor=float(liquidity_floor),
        gamma=None if gamma is None else float(gamma),epsilon_annual_log_growth=EPSILON_ANNUAL_LOG_GROWTH,
        status="no_selected_gamma" if gamma is None else "pending",event_count_cap=None,
        feasible_n=None,feasible_indices=None,near_optimal_n=None,near_optimal_indices=None,
        best_mean_log_growth=None,weight_min=None,weight_max=None,total_variation_diameter=None,
        raw_menu_index=raw_menu_index,raw_menu_feasible=None,raw_menu_mean_log_growth=None,
        raw_menu_regret_if_feasible=None,raw_menu_in_near_optimal_set=None)
    if gamma is None:
        return result,[]
    if isinstance(gamma,(bool,np.bool_)) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Gamma must be a valid explicitly labeled training screen")
    training_g, training_d = g[index], d[index]
    if (not np.isfinite(training_g).all() or not np.isfinite(training_d).all()
            or np.any(training_d < 0) or np.any(training_d > 1)):
        raise ValueError("Training metrics must be finite with MDD in[0,1]")
    counts = np.count_nonzero(training_d-L > TOL,axis=0)
    cap = math.floor(float(gamma)*len(ids)+TOL)
    feasible = (counts <= cap) & (w[:,2] >= liquidity_floor-TOL)
    available = np.flatnonzero(feasible)
    growth = training_g.mean(axis=0)
    result.update(event_count_cap=cap,feasible_n=len(available),feasible_indices=available.tolist(),
        status="empty_feasible_set" if len(available)==0 else "available")
    if raw_menu_index is not None:
        if (isinstance(raw_menu_index,(bool,np.bool_)) or not isinstance(raw_menu_index,(int,np.integer))
                or not 0 <= raw_menu_index < len(w)):
            raise ValueError("Raw menu index must address the supplied target universe")
        result.update(raw_menu_feasible=bool(feasible[raw_menu_index]),
            raw_menu_mean_log_growth=float(growth[raw_menu_index]))
    if not len(available):
        result.update(near_optimal_n=0,near_optimal_indices=[])
        return result,[]
    best = float(growth[available].max())
    near = available[growth[available] >= best-EPSILON_ANNUAL_LOG_GROWTH-TOL]
    near_weights = w[near]
    diameter = float(np.max(np.abs(near_weights[:,None,:]-near_weights[None,:,:]).sum(axis=2))/2)
    result.update(best_mean_log_growth=best,near_optimal_n=len(near),near_optimal_indices=near.tolist(),
        weight_min=near_weights.min(axis=0).tolist(),weight_max=near_weights.max(axis=0).tolist(),
        total_variation_diameter=diameter)
    if raw_menu_index is not None:
        result["raw_menu_in_near_optimal_set"] = bool(raw_menu_index in near)
        if feasible[raw_menu_index]:
            result["raw_menu_regret_if_feasible"] = float(best-growth[raw_menu_index])
    members = [dict(candidate_local_index=int(k),weights=w[k].tolist(),
        training_event_count=int(counts[k]),mean_training_log_growth=float(growth[k]),
        gap_to_individual_best=float(best-growth[k])) for k in near]
    return result,members


def gamma_objective_band(bundle):
    """Recover the declared near-equal inner objective band from saved metadata."""
    rows = [r for r in bundle["inner_summaries"] if r["eligible"]]
    if not rows:
        return []
    coverage = max(r["raw_auto_path_profile_units"] for r in rows)
    rows = [r for r in rows if r["raw_auto_path_profile_units"] == coverage]
    best = max(r["raw_auto_unit_mean_log_growth"] for r in rows)
    return sorted(float(r["training_screen_gamma"]) for r in rows
        if r["raw_auto_unit_mean_log_growth"] >= best-TOL)


def profile_metadata(job, bundle, profile):
    name = profile["name"]; refit = bundle["refit"]; raw_fit = refit["raw_fit"]
    if refit["training_ids"] != job["training_ids"]:
        raise ValueError("Saved Q1 fit does not match this job's training identities")
    gamma = refit["selection"]["selected_gamma"]
    raw = refit["raw_assignments"][name]; effective = refit["effective_assignments"][name]
    product = next((r for r in raw_fit["menu"] if r["name"] == raw),None) if raw_fit else None
    admission = next(r for r in refit["admission_details"] if r["client"] == name)
    details = refit.get("q1_review_details",{}).get(name,{})
    result = dict(job_id=job["job_id"],context_id=job["parent_context_id"],stage=job["stage"],
        profile=name,horizon=profile["horizon"],L=profile["loss_tolerance"],liquidity=profile["liquidity"],
        training_n=len(job["training_ids"]),selected_gamma=gamma,raw_assignment=raw,effective_assignment=effective,
        admission_reason=admission["reason"],review_reasons=details.get("q1_review_reasons",[]),
        raw_menu_index=None if product is None else product["candidate_index"],
        raw_original232_index=None if product is None else product["candidate_original232_index"],
        raw_weights=None if product is None else product["weights"],
        raw_identified_by_profiles=None if product is None else product["identified_by_profiles"],
        raw_reference15_CVaR=None if product is None else product["reference_cvar90_mdd"])
    if effective != "Review" and (raw_fit is None or effective not in {r["name"] for r in raw_fit["menu"]}):
        raise ValueError("Effective assignment does not identify a saved fitted product")
    return result


def own_horizon_rows(job, bundle, profile, fields, *, thirds_index):
    """Evaluate the already chosen effective product; no heldout weight search."""
    meta = profile_metadata(job,bundle,profile)
    train = development_ids(job["training_ids"]); actual = development_ids(job["evaluation_ids"])
    if set(train) & set(actual) or job["stage"] != "refit" or job["parent_context_id"] == 0:
        raise ValueError("Primary Q1 OOF requires a disjoint outer refit")
    raw_fit = bundle["refit"]["raw_fit"]
    product = next((r for r in raw_fit["menu"] if r["name"] == meta["effective_assignment"]),None) if raw_fit else None
    chosen = None if product is None else product["candidate_index"]
    rows, references = [],[]
    metric_keys = ("log_growth","cagr","mdd","volatility","terminal_real","nominal_fees","trade_fraction","breach")
    def financial(sid,k):
        return dict(log_growth=float(fields["log_growth"][sid-1,k]),cagr=float(fields["cagr"][sid-1,k]),
            mdd=float(fields["mdd"][sid-1,k]),volatility=float(fields["volatility"][sid-1,k]),
            terminal_real=float(fields["terminal_real"][sid-1,k]),nominal_fees=float(fields["fees"][sid-1,k]),
            trade_fraction=float(fields["trade_fraction"][sid-1,k]),
            breach=bool(fields["mdd"][sid-1,k]-profile["loss_tolerance"] > TOL))
    for sid in actual:
        common = dict(job_id=job["job_id"],scenario_id=sid,profile=profile["name"],
            horizon=profile["horizon"],L=profile["loss_tolerance"],raw_assignment=meta["raw_assignment"],
            effective_assignment=meta["effective_assignment"],admission_reason=meta["admission_reason"])
        rows.append(dict(common,assigned=chosen is not None,candidate_local_index=chosen,
            **(financial(sid,chosen) if chosen is not None else dict.fromkeys(metric_keys))))
        references.append(dict(common,assigned=True,candidate_local_index=thirds_index,**financial(sid,thirds_index)))
    return rows,references
