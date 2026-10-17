"""Common-menu research: empirical suitability, coverage and growth regret.

This is an explicit product-design policy, not an elicited investor utility.
The five representative profiles do not represent population frequencies.
No function loads data or provides a population-risk certificate.
"""
from __future__ import annotations

from itertools import combinations, islice
from typing import Mapping

import numpy as np

from .metrics import NUMERIC_TOL, discrete_cvar, empirical_risk


def pareto_frontier(growth, risk) -> np.ndarray:
    """Indices not strictly dominated in higher growth / lower scalar risk.

    Numerically equal outcomes remain as distinct admissible targets, so this
    function does not pretend that the data identify one of equivalent weights.
    """
    growth, risk = np.asarray(growth, float), np.asarray(risk, float)
    if growth.ndim != 1 or risk.shape != growth.shape or len(growth) == 0:
        raise ValueError("growth and risk must be equally sized nonempty vectors")
    if not np.all(np.isfinite(growth)) or not np.all(np.isfinite(risk)):
        raise ValueError("frontier inputs must be finite")
    result = []
    for i in range(len(growth)):
        weak = (growth >= growth[i] - NUMERIC_TOL) & (risk <= risk[i] + NUMERIC_TOL)
        strict = (growth > growth[i] + NUMERIC_TOL) | (risk < risk[i] - NUMERIC_TOL)
        if not np.any(weak & strict):
            result.append(i)
    return np.asarray(result, dtype=int)


def _score_tuple(values, oracle):
    covered = np.isfinite(values)
    regret = np.where(covered, oracle - values, 0.0)
    count = int(covered.sum())
    return (count, float(regret.max(initial=0)),
            float(regret.sum() / count) if count else 0.0)


def _same_objective(a, b):
    return (a[0] == b[0] and abs(a[1] - b[1]) <= NUMERIC_TOL
            and abs(a[2] - b[2]) <= NUMERIC_TOL)


def _better(a, b):
    if a[0] != b[0]:
        return a[0] > b[0]
    if abs(a[1] - b[1]) > NUMERIC_TOL:
        return a[1] < b[1]
    return a[2] < b[2] - NUMERIC_TOL


def _useful_candidates(scores):
    """Prune profile-growth dominated targets, preserving deterministic ties."""
    useful = np.flatnonzero(np.any(np.isfinite(scores), axis=0))
    kept = []
    for i in useful:
        current = scores[:, i, None]
        other = scores[:, useful]
        # Exact weak dominance is transitive. Allowing a tolerance on the
        # weak comparison can form cycles and incorrectly delete every target.
        weak = np.all(other >= current, axis=0)
        strict = np.any(other > current + NUMERIC_TOL, axis=0)
        # Equal profile scores identify no extra function; retain lowest ID.
        equivalent = np.all(other == current, axis=0)
        earlier_equal = equivalent & (useful < i)
        # A higher-ID dominator can leave the whole-menu objective tied when
        # another product already supplies its improvement. Prune only using
        # earlier IDs, preserving the declared full-universe canonical tie.
        earlier_dominator = weak & strict & (useful < i)
        if not np.any(earlier_dominator) and not np.any(earlier_equal):
            kept.append(int(i))
    return np.asarray(kept, dtype=int)


def optimize_menu(scores, max_products=3, *, prune=True):
    """Exhaustively optimize common menus with up to three products.

    scores[j,w] is client j's training mean annualized log growth for a
    feasible target, and -inf otherwise. Priority is coverage, minimax regret
    relative to each client's feasible oracle, then mean covered-client regret.
    Candidate-ID order breaks exact numerical ties. Return the smallest menu
    matching the best up-to-three-product objective, plus each size's optimum.
    A menu need not cover all clients, and an empty feasible set is explicit.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or 0 in scores.shape or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("scores must be a nonempty client-by-target matrix of finite values or -inf")
    if isinstance(max_products, bool) or max_products not in (1, 2, 3):
        raise ValueError("max_products must be 1, 2 or 3")
    any_feasible = np.any(np.isfinite(scores), axis=1)
    oracle = np.where(any_feasible, np.max(scores, axis=1), 0.0)
    pool = _useful_candidates(scores) if prune else np.flatnonzero(np.any(np.isfinite(scores), axis=0))
    best_by_size = []
    for size in range(1, min(max_products, len(pool)) + 1):
        best_objective = (-1, float("inf"), float("inf"))
        best_ids = None
        iterator = combinations(pool.tolist(), size)
        while True:
            batch = list(islice(iterator, 50_000))
            if not batch:
                break
            batch = np.asarray(batch, dtype=int)
            values = np.max(scores[:, batch], axis=2)
            covered = np.isfinite(values)
            counts = covered.sum(axis=0)
            regret = np.where(covered, oracle[:, None] - values, 0.0)
            maximum = regret.max(axis=0)
            mean = np.divide(regret.sum(axis=0), counts,
                             out=np.zeros(len(batch), dtype=float), where=counts > 0)
            mask = counts == counts.max()
            low_maximum = maximum[mask].min()
            mask &= maximum <= low_maximum + NUMERIC_TOL
            low_mean = mean[mask].min()
            mask &= mean <= low_mean + NUMERIC_TOL
            index = int(np.flatnonzero(mask)[0])
            objective = (int(counts[index]), float(maximum[index]), float(mean[index]))
            ids = tuple(int(x) for x in batch[index])
            if (_better(objective, best_objective)
                    or (_same_objective(objective, best_objective)
                        and (best_ids is None or ids < best_ids))):
                best_objective, best_ids = objective, ids
        best_by_size.append({"size": size, "indices": list(best_ids),
                             "objective": list(best_objective)})
    if not best_by_size:
        return {"identified_indices": [], "objective": [0, 0.0, 0.0],
                "oracle_growth": oracle.tolist(), "oracle_feasible": any_feasible.tolist(),
                "useful_candidate_count": 0, "optima_by_size": []}
    best = best_by_size[0]
    for item in best_by_size[1:]:
        if _better(item["objective"], best["objective"]):
            best = item
    identified = next(item for item in best_by_size
                      if _same_objective(item["objective"], best["objective"]))
    return {"identified_indices": identified["indices"],
            "objective": identified["objective"],
            "oracle_growth": oracle.tolist(), "oracle_feasible": any_feasible.tolist(),
            "useful_candidate_count": len(pool), "optima_by_size": best_by_size}


def complete_menu(identified, reference_growth, reference_risk, count=3):
    """Extend demand-identified targets by a stated reference-risk breadth rule."""
    growth, risk = np.asarray(reference_growth, float), np.asarray(reference_risk, float)
    if len(growth) < count or count < len(identified):
        raise ValueError("not enough targets or too many identified products")
    frontier = pareto_frontier(growth, risk)
    selected = list(identified)
    extension = []
    while len(selected) < count:
        available = [int(i) for i in frontier if i not in selected]
        if not available:
            # A degenerate frontier cannot identify three efficient risk roles.
            # Expose that condition instead of silently inserting a dominated item.
            break
        if not selected:
            chosen = min(available, key=lambda i: (risk[i], -growth[i], i))
        else:
            distance = {i: float(np.min(np.abs(risk[i] - risk[selected]))) for i in available}
            largest = max(distance.values())
            if largest <= NUMERIC_TOL:
                # Distinct weights with identical reference risk do not
                # establish another economically distinct risk category.
                break
            spaced = [i for i in available if distance[i] >= largest - NUMERIC_TOL]
            chosen = min(spaced, key=lambda i: (-growth[i], i))
        selected.append(chosen)
        extension.append(chosen)
    ordered = sorted(selected, key=lambda i: (risk[i], growth[i], i))
    distinct = []
    for i in ordered:
        if not distinct or risk[i] - distinct[-1] > NUMERIC_TOL:
            distinct.append(float(risk[i]))
    return {"indices": ordered, "extension_indices": extension,
            "reference_frontier": frontier.tolist(), "complete": len(ordered) == count,
            "distinct_reference_risk_count": len(distinct),
            "three_distinct_reference_risks": len(distinct) >= count}


def fit_menu(metrics_by_horizon: Mapping, weights, profiles, indices,
             alpha=0.10, liquidity_floors=None):
    """Fit a single common menu on the nominated scenario indices only.

    Complete-path arrays may be precomputed for the pre-fixed static universe;
    only these indices enter growth, risk, oracle, menu and assignment fitting.
    Repeated indices deliberately support whole-scenario bootstrap resampling.
    The return value is JSON-serializable and contains no population guarantee.
    """
    weights = np.asarray(weights, float)
    indices = np.asarray(indices)
    if (weights.ndim != 2 or weights.shape[1] != 3 or len(weights) < 3
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.all(np.abs(weights.sum(axis=1) - 1) <= NUMERIC_TOL)):
        raise ValueError("require at least three long-only fully-invested targets")
    if indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a nonempty integer vector")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    floors = liquidity_floors or {"high": 0.0, "medium": 0.0, "low": 0.0}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in floors.values()):
        raise ValueError("liquidity floors must lie in [0,1]")
    all_growth, all_risk, all_excess, all_cvar, feasible = [], [], [], [], []
    for profile in profiles:
        horizon, loss = profile["horizon"], profile["loss_tolerance"]
        fields = metrics_by_horizon[horizon]
        if np.any(indices < 0) or np.any(indices >= len(fields["mdd"])):
            raise ValueError("training index outside supplied scenarios")
        growth = np.mean(np.asarray(fields["log_growth"])[indices], axis=0)
        mdd = np.asarray(fields["mdd"])[indices]
        risk = empirical_risk(mdd, loss)
        permitted = ((risk["breach_probability"] <= alpha + NUMERIC_TOL)
                     & (weights[:, 2] >= floors[profile["liquidity"]] - NUMERIC_TOL))
        all_growth.append(growth)
        all_risk.append(risk["breach_probability"])
        all_excess.append(risk["expected_excess"])
        all_cvar.append(discrete_cvar(mdd))
        feasible.append(permitted)
    growth, risk, excess, cvar = map(np.asarray, (all_growth, all_risk, all_excess, all_cvar))
    feasible = np.asarray(feasible)
    scores = np.where(feasible, growth, -np.inf)
    optimum = optimize_menu(scores)
    reference = metrics_by_horizon[15]
    reference_growth = np.mean(np.asarray(reference["log_growth"])[indices], axis=0)
    reference_cvar = discrete_cvar(np.asarray(reference["mdd"])[indices])
    completion = complete_menu(optimum["identified_indices"], reference_growth, reference_cvar)
    names = ["Conservative", "Balanced", "Growth"]
    menu = [{"name": names[position], "candidate_index": int(i),
             "weights": weights[i].tolist(),
             "identified_by_profiles": i in optimum["identified_indices"],
             "reference_mean_log_growth": float(reference_growth[i]),
             "reference_cvar90_mdd": float(reference_cvar[i])}
            for position, i in enumerate(completion["indices"])]
    assignments, details = {}, []
    for j, profile in enumerate(profiles):
        allowed = [item for item in menu if feasible[j, item["candidate_index"]]]
        chosen = None
        if allowed:
            highest = max(growth[j, item["candidate_index"]] for item in allowed)
            tied = [item for item in allowed if growth[j, item["candidate_index"]] >= highest - NUMERIC_TOL]
            chosen = min(tied, key=lambda item: (excess[j, item["candidate_index"]],
                                                cvar[j, item["candidate_index"]],
                                                item["candidate_index"]))
        assignments[profile["name"]] = chosen["name"] if chosen else "Review"
        for item in menu:
            i = item["candidate_index"]
            count = int(np.count_nonzero(np.asarray(metrics_by_horizon[profile["horizon"]]["mdd"])[indices, i]
                                         > profile["loss_tolerance"] + NUMERIC_TOL))
            details.append({"client": profile["name"], "portfolio": item["name"],
                            "candidate_index": i, "horizon": profile["horizon"],
                            "loss_tolerance": profile["loss_tolerance"], "n": len(indices),
                            "breach_count": count, "breach_probability": float(risk[j, i]),
                            "expected_excess": float(excess[j, i]), "cvar90_mdd": float(cvar[j, i]),
                            "mean_log_growth": float(growth[j, i]),
                            "liquidity_pass": bool(weights[i, 2] >= floors[profile["liquidity"]] - NUMERIC_TOL),
                            "sample_feasible": bool(feasible[j, i]),
                            "one_path_boundary": bool(count == int(np.floor(len(indices) * alpha + NUMERIC_TOL))),
                            "chosen": chosen is not None and i == chosen["candidate_index"]})
    return {"alpha": float(alpha), "liquidity_floors": dict(floors), "n_training": len(indices),
            "menu": menu, "assignments": assignments, "client_product_details": details,
            "selection": optimum, "completion": completion,
            "oracle_candidate_count": {profile["name"]: int(feasible[j].sum())
                                       for j, profile in enumerate(profiles)}}
