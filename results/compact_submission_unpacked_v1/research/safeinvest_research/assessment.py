"""Assess already specified menus without changing product definitions.

Used for the fixed H15 comparator and fixed-menu policy/cost counterfactuals.
Assessment can change onboarding choices, so callers must distinguish it from
performance of an already frozen customer assignment.
"""
from __future__ import annotations

import numpy as np

from .metrics import NUMERIC_TOL, discrete_cvar, empirical_risk


def assess_fixed_menu(metrics_by_horizon, weights, menu, profiles, indices,
                      alpha=0.10, liquidity_floors=None):
    """Screen a fixed named menu using only the specified training rows.

    ``menu`` contains name/candidate_index records referring to ``weights``.
    Product names remain fixed, even if a sensitivity changes risk ordering.
    Return assignment/details in the same shape as ``fit_menu``, without
    selecting new targets, computing an oracle, or making confidence claims.
    """
    weights = np.asarray(weights, float)
    rows = np.asarray(indices)
    if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("indices must be a nonempty integer vector")
    if (weights.ndim != 2 or weights.shape[1] != 3 or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.all(np.abs(weights.sum(axis=1) - 1) <= NUMERIC_TOL)):
        raise ValueError("weights must be long-only fully-invested three-asset targets")
    if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must lie in [0,1]")
    floors = liquidity_floors or {"high": 0.0, "medium": 0.0, "low": 0.0}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in floors.values()):
        raise ValueError("liquidity floors must lie in [0,1]")
    if (not menu or len({p["name"] for p in menu}) != len(menu)
            or len({p["candidate_index"] for p in menu}) != len(menu)):
        raise ValueError("require distinct named menu products")
    product_ids = np.asarray([p["candidate_index"] for p in menu])
    if (not np.issubdtype(product_ids.dtype, np.integer)
            or np.any(product_ids < 0) or np.any(product_ids >= len(weights))):
        raise ValueError("menu candidate indices outside target array")
    details, assignments = [], {}
    for profile in profiles:
        fields = metrics_by_horizon[profile["horizon"]]
        values = np.asarray(fields["mdd"])
        if (values.ndim != 2 or values.shape[1] != len(weights)
                or np.any(rows < 0) or np.any(rows >= len(values))):
            raise ValueError("metric rows or target dimensions do not match")
        losses = values[np.ix_(rows, product_ids)]
        growth = np.asarray(fields["log_growth"])[np.ix_(rows, product_ids)].mean(axis=0)
        if not np.all(np.isfinite(growth)):
            raise ValueError("growth must be finite")
        risk = empirical_risk(losses, profile["loss_tolerance"])
        tail = discrete_cvar(losses)
        current = []
        for j, item in enumerate(menu):
            candidate = int(item["candidate_index"])
            liquidity_ok = weights[candidate, 2] >= floors[profile["liquidity"]] - NUMERIC_TOL
            count = int(np.count_nonzero(losses[:, j] > profile["loss_tolerance"] + NUMERIC_TOL))
            maximum = int(np.floor(len(rows) * alpha + NUMERIC_TOL))
            current.append({"client": profile["name"], "portfolio": item["name"],
                            "candidate_index": candidate, "horizon": profile["horizon"],
                            "loss_tolerance": profile["loss_tolerance"], "n": len(rows),
                            "breach_count": count, "breach_probability": float(risk["breach_probability"][j]),
                            "expected_excess": float(risk["expected_excess"][j]),
                            "cvar90_mdd": float(tail[j]), "mean_log_growth": float(growth[j]),
                            "liquidity_pass": bool(liquidity_ok),
                            "sample_feasible": bool(count <= maximum and liquidity_ok),
                            "one_path_boundary": bool(count == maximum and maximum < len(rows)),
                            "chosen": False})
        allowed = [d for d in current if d["sample_feasible"]]
        chosen = None
        if allowed:
            highest = max(d["mean_log_growth"] for d in allowed)
            tied = [d for d in allowed if d["mean_log_growth"] >= highest - NUMERIC_TOL]
            chosen = min(tied, key=lambda d: (d["expected_excess"], d["cvar90_mdd"], d["candidate_index"]))
            chosen["chosen"] = True
        assignments[profile["name"]] = chosen["portfolio"] if chosen else "Review"
        details.extend(current)
    return {"alpha": float(alpha), "liquidity_floors": dict(floors),
            "n_training": len(rows), "assignments": assignments,
            "menu": [{**item, "weights": weights[item["candidate_index"]].tolist()} for item in menu],
            "client_product_details": details,
            "scope": "Fixed-menu empirical assessment; no target selection or population-risk certification."}
