"""Calendar-path fan comparisons of hold versus one fixed-target restore.

This is an empirical, repeatedly replanned research heuristic.  Each current
action is followed by annual restoration until that forecast continuation's
own first annual real-MDD breach, then hold.  Only the fixed training fan is
an input to a comparison; realized evaluation returns are not.  A training
suffix keeps its dates, joint assets/inflation and within-path chronology.
No conditional-law, Bellman improvement or risk-certificate claim is made.

All fees use the exact self-financing executor once.  Holdings mean nominal
asset-value balances, which grow during hold.  Original peak, running MDD,
elapsed calendar time and absorbing breach are never reset.  The evaluator
is for active, pre-breach states; actual already-breached states bypass it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target

NUMERIC_TOL = 1e-12
POLICIES = ("growth_only_rollout", "pareto_hold_rollout")


def _market(returns, inflation, name):
    r, pi = np.asarray(returns, float), np.asarray(inflation, float)
    if (r.ndim != 3 or r.shape[-1] != 3 or min(r.shape[:2]) < 1
            or pi.shape != r.shape[:2] or not np.all(np.isfinite(r))
            or not np.all(np.isfinite(pi)) or np.any(r <= -1) or np.any(pi <= -1)):
        raise ValueError(f"{name} must be finite nonempty(n,T,3)/(n,T), with positive gross")
    return r, pi


def _parameters(target, L, cost_rate, horizon, max_h):
    w = np.asarray(target, float)
    if (w.shape != (3,) or not np.all(np.isfinite(w)) or np.any(w < 0)
            or not np.isclose(w.sum(), 1, atol=NUMERIC_TOL, rtol=0)):
        raise ValueError("target must be one finite nonnegative simplex point(3,)")
    if isinstance(L, (bool, np.bool_)) or not np.isscalar(L):
        raise ValueError("L must be a scalar fraction in[0,1]")
    loss = float(L)
    if not np.isfinite(loss) or not 0 <= loss <= 1:
        raise ValueError("L must be a finite scalar fraction in[0,1]")
    if isinstance(cost_rate, (bool, np.bool_)) or not np.isscalar(cost_rate):
        raise ValueError("cost_rate must be a scalar in[0,1)")
    rate = float(cost_rate)
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("cost_rate must be finite and in[0,1)")
    h = max_h if horizon is None else horizon
    if (isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
            or not 1 <= h <= max_h):
        raise ValueError("horizon must be an integer within both supplied path lengths")
    return w / w.sum(), loss, rate, int(h)


def _state(holdings, cpi, real_peak, running_mdd, ever_breached, loss):
    x = np.asarray(holdings, float)
    if x.shape == (3,):
        x = x[None, :]
    if (x.ndim != 2 or x.shape[-1] != 3 or len(x) < 1
            or not np.all(np.isfinite(x)) or np.any(x < 0)):
        raise ValueError("holdings must have shape(3,) or(n,3), finite and nonnegative")
    n = len(x)
    try:
        price, peak, worst = [np.broadcast_to(np.asarray(v, float), (n,)).copy()
                              for v in (cpi, real_peak, running_mdd)]
        flag_input = np.asarray(ever_breached)
        if flag_input.dtype.kind != "b":
            raise ValueError("ever_breached must be boolean")
        flag = np.broadcast_to(flag_input, (n,))
    except ValueError as exc:
        raise ValueError("state scalars/vectors must broadcast to holdings rows") from exc
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        total = x.sum(axis=-1)
        real = total / price
    if (not np.all(np.isfinite(price)) or np.any(price <= 0)
            or not np.all(np.isfinite(peak)) or np.any(peak <= 0)
            or not np.all(np.isfinite(real)) or np.any(real <= 0)
            or not np.all(np.isfinite(worst)) or np.any(worst < 0) or np.any(worst > 1)):
        raise ValueError("state requires positive finite CPI/wealth/peak and MDD in[0,1]")
    # Relative scale: a tiny account must not accept a huge relative peak error.
    if np.any((real > peak) & ((real - peak) > NUMERIC_TOL * np.maximum(real, peak))):
        raise ValueError("real_peak cannot lie below current real wealth")
    peak = np.maximum(peak, real)
    current = np.maximum(1 - real / peak, 0)
    if np.any(current - worst > NUMERIC_TOL):
        raise ValueError("running_mdd cannot be below current drawdown")
    worst = np.maximum(worst, current)
    if np.any(flag) or np.any(worst - loss > NUMERIC_TOL):
        raise ValueError("evaluate_actions is only for consistent active pre-breach states")
    return x, price, peak, worst, real


def _advance(post, gross, cpi, inflation_gross, peak, worst, ever, loss):
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        after = post * gross
        next_cpi = cpi * inflation_gross
        nominal = after.sum(axis=-1)
        real = nominal / next_cpi
        weights = after / nominal[..., None]
    if (not np.all(np.isfinite(after)) or not np.all(np.isfinite(next_cpi))
            or np.any(next_cpi <= 0) or not np.all(np.isfinite(real)) or np.any(real <= 0)
            or not np.all(np.isfinite(weights))):
        raise ValueError("Account/forecast values exceed the numerical range")
    next_peak = np.maximum(peak, real)
    drawdown = np.maximum(1 - real / next_peak, 0)
    next_worst = np.maximum(worst, drawdown)
    next_ever = ever | (next_worst - loss > NUMERIC_TOL)
    return after, next_cpi, nominal, real, weights, next_peak, drawdown, next_worst, next_ever


def _recommendations(hold_log, restore_log, hold_bad, restore_bad):
    hold_mean, restore_mean = hold_log.mean(axis=-1), restore_log.mean(axis=-1)
    hold_count, restore_count = hold_bad.sum(axis=-1), restore_bad.sum(axis=-1)
    growth = restore_mean - hold_mean > NUMERIC_TOL
    pareto = ~((hold_mean >= restore_mean - NUMERIC_TOL) & (hold_count <= restore_count))
    n = hold_log.shape[-1]
    if n == 1:
        growth_flip = pareto_flip = np.full(len(hold_log), -1, dtype=np.int64)
    else:
        # Same state, deleting one whole forecast path.  This is NOT a policy
        # refit or a leave-one-out financial replay.
        hm = (hold_log.sum(axis=-1, keepdims=True) - hold_log) / (n - 1)
        rm = (restore_log.sum(axis=-1, keepdims=True) - restore_log) / (n - 1)
        hc = hold_count[:, None] - hold_bad.astype(np.int64)
        rc = restore_count[:, None] - restore_bad.astype(np.int64)
        growth_loo = rm - hm > NUMERIC_TOL
        pareto_loo = ~((hm >= rm - NUMERIC_TOL) & (hc <= rc))
        growth_flip = (growth_loo != growth[:, None]).sum(axis=-1)
        pareto_flip = (pareto_loo != pareto[:, None]).sum(axis=-1)
    return {"growth_recommend_restore": growth, "pareto_recommend_restore": pareto,
            "growth_delete_one_flip_count": growth_flip,
            "pareto_delete_one_flip_count": pareto_flip}


def evaluate_actions(holdings: ArrayLike, cpi: ArrayLike, real_peak: ArrayLike,
                     running_mdd: ArrayLike, ever_breached: ArrayLike,
                     training_returns: ArrayLike, training_inflation: ArrayLike, *,
                     year_index: int, target: ArrayLike, L: float,
                     cost_rate: float = .0025, horizon: int | None = None) -> dict[str, Any]:
    """Evaluate current hold/restore, then baseline B, on complete training paths.

    ``year_index`` is the zero-based next return column; ``horizon`` is the
    original H, never a restarted remaining horizon.  Holdings(3,) or(n,3)
    and scalar/(n,) history inputs are accepted.  Already-breached/inconsistent
    states are rejected: actual pending clients have no automatic gate.

    Each action's dictionary has total_log_growth(n,N), ever_breached(n,N),
    mean_log_growth(n), breach_count(n).  Log scores are net REAL TOTAL growth
    from current wealth to original H, not annualized.  Recommendations and
    local delete-one flip counts are(n,), fan_n is the scalar N.  For N=1 flip
    counts are-1, to be serialized as null.  No returned score includes a
    second fee or any realized evaluation-period return.
    """
    train, infl = _market(training_returns, training_inflation, "training fan")
    w, loss, rate, h = _parameters(target, L, cost_rate, horizon, train.shape[1])
    if (isinstance(year_index, (bool, np.bool_)) or not isinstance(year_index, (int, np.integer))
            or not 0 <= year_index < h):
        raise ValueError("year_index must be an integer in[0,original H)")
    x, price, peak, worst, initial_real = _state(holdings, cpi, real_peak, running_mdd, ever_breached, loss)
    n, fan_n = len(x), len(train)
    result: dict[str, Any] = {"fan_n": fan_n, "year_index": int(year_index), "horizon": h}
    for action in ("hold", "restore"):
        current = np.broadcast_to(x[:, None, :], (n, fan_n, 3)).copy()
        cp = np.broadcast_to(price[:, None], (n, fan_n)).copy()
        pk = np.broadcast_to(peak[:, None], (n, fan_n)).copy()
        dd = np.broadcast_to(worst[:, None], (n, fan_n)).copy()
        bad = np.zeros((n, fan_n), dtype=bool)
        for year in range(int(year_index), h):
            restore = np.full((n, fan_n), action == "restore") if year == year_index else ~bad
            post = current.copy()
            if np.any(restore):
                post[restore] = execute_target(current[restore], w, rate)[0]
            current, cp, _, real, _, pk, _, dd, bad = _advance(
                post, 1 + train[None, :, year, :], cp, 1 + infl[None, :, year], pk, dd, bad, loss)
        total_log = np.log(real) - np.log(initial_real[:, None])
        result[action] = {"total_log_growth": total_log, "ever_breached": bad,
                          "mean_log_growth": total_log.mean(axis=-1),
                          "breach_count": bad.sum(axis=-1)}
    result.update(_recommendations(result["hold"]["total_log_growth"], result["restore"]["total_log_growth"],
                                   result["hold"]["ever_breached"], result["restore"]["ever_breached"]))
    for value in result.values():
        if isinstance(value, dict):
            for array in value.values():
                array.setflags(write=False)
        elif isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def _recovery(arm, loss, horizon):
    triggered = arm["first_crossing"] >= 0
    times = np.arange(horizon + 1)[None, :]
    possible = (triggered[:, None] & (times > arm["first_crossing"][:, None])
                & (arm["real_wealth"] >= arm["peak_at_trigger"][:, None]))
    recovered = possible.any(axis=-1)
    endpoint = possible.argmax(axis=-1)
    arm["triggered"], arm["recovered"] = triggered, recovered
    arm["recovery_endpoint"] = np.where(recovered, endpoint, -1).astype(np.int64)
    arm["recovery_time"] = np.where(recovered, endpoint - arm["first_crossing"], -1).astype(np.int64)
    arm["right_censored"] = triggered & ~recovered
    arm["available_followup"] = np.where(triggered, horizon - arm["first_crossing"], -1).astype(np.int64)
    arm["observed_followup"] = np.where(recovered, arm["recovery_time"], arm["available_followup"])
    arm["added_severity"] = np.where(triggered,
        np.maximum(arm["running_mdd"][:, -1] - arm["drawdown_at_trigger"], 0), np.nan)
    arm["terminal_excess"] = np.maximum(arm["running_mdd"][:, -1] - loss, 0)


def simulate_policy(actual_returns: ArrayLike, actual_inflation: ArrayLike,
                    training_returns: ArrayLike, training_inflation: ArrayLike,
                    target: ArrayLike, *, policy: str, L: float,
                    cost_rate: float = .0025, horizon: int | None = None) -> dict[str, Any]:
    """Replay one declared rule from100/all bills, with exact action accounting.

    Actual and training paths are(n,T,3)/(n,T) and(N,Ttrain,3)/(N,Ttrain).
    The original H defaults to actual T; explicit H must fit both arrays.
    Policy is growth_only_rollout or pareto_hold_rollout.  Inception restores
    mandatorily; already-breached accounts hold, without a forecast decision.

    The returned arm has the pending-core schema with its m axis removed.
    Qg/counts/forecast_n/delete_one_flip_count/gate_reason and predicted restore
    advantages are(n,H); inactive float/int diagnostics are NaN/-1.  Scores
    use total log units.  ``trades`` counts moved>1e-12 whereas target_execution
    counts restore decisions, including identities.  Input arrays are never
    mutated and output arrays are read-only.  No-trigger recovery/censoring
    placeholders MUST become null in a CSV/JSON presentation.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of{POLICIES}")
    market, infl = _market(actual_returns, actual_inflation, "actual paths")
    train, train_infl = _market(training_returns, training_inflation, "training fan")
    default_h = market.shape[1] if horizon is None else horizon
    w, loss, rate, h = _parameters(target, L, cost_rate, default_h, min(market.shape[1], train.shape[1]))
    n = len(market)
    periods, endpoints = (n, h), (n, h + 1)
    arm: dict[str, Any] = {key: np.zeros(periods) for key in (
        "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
        "pretrade_tv_drift", "end_year_tv_drift")}
    arm.update({key: np.zeros(endpoints) for key in (
        "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd")})
    arm.update({key: np.full(periods, np.nan) for key in (
        "qg_hold", "qg_restore", "predicted_growth_advantage_restore")})
    arm.update({key: np.full(periods, -1, dtype=np.int64) for key in (
        "qbad_hold_count", "qbad_restore_count", "forecast_n", "delete_one_flip_count",
        "predicted_breach_count_delta_restore")})
    arm["gate_reason"] = np.full(periods, "", dtype="U48")
    arm["restoration_predicted_growth_sacrifice"] = np.zeros(periods, bool)
    arm["holdings"], arm["weights"] = np.zeros(endpoints + (3,)), np.zeros(endpoints + (3,))
    arm["posttrade_holdings"] = np.zeros(periods + (3,))
    arm["ever_breached"] = np.zeros(endpoints, bool)
    arm["target_execution"], arm["pending_at_decision"] = np.zeros(periods, bool), np.zeros(periods, bool)
    arm["actions"] = np.full(periods, "hold", dtype="U7")
    arm["first_crossing"] = np.full(n, -1, dtype=np.int64)
    for key in ("drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger"):
        arm[key] = np.full(n, np.nan)
    arm["cpi"] = np.ones(endpoints)
    arm["holdings"][:, 0, 2], arm["weights"][:, 0, 2] = 100., 1.
    for key in ("nominal_wealth", "real_wealth", "real_peak"):
        arm[key][:, 0] = 100.

    for year in range(h):
        current, wealth, u = arm["holdings"][:, year], arm["nominal_wealth"][:, year], arm["weights"][:, year]
        pending = arm["ever_breached"][:, year]
        restore = np.full(n, year == 0, dtype=bool)
        arm["gate_reason"][pending, year] = "already_breached_pending_hold"
        if year == 0:
            arm["gate_reason"][:, year] = "mandatory_initial_restore"
        else:
            active = ~pending
            if np.any(active):
                values = evaluate_actions(current[active], arm["cpi"][active, year], arm["real_peak"][active, year],
                    arm["running_mdd"][active, year], pending[active], train, train_infl,
                    year_index=year, target=w, L=loss, cost_rate=rate, horizon=h)
                prefix = "growth" if policy == "growth_only_rollout" else "pareto"
                chosen = values[prefix + "_recommend_restore"]
                restore[active] = chosen
                qh, qr = values["hold"]["mean_log_growth"], values["restore"]["mean_log_growth"]
                ch, cr = values["hold"]["breach_count"], values["restore"]["breach_count"]
                for key, value in (("qg_hold", qh), ("qg_restore", qr), ("qbad_hold_count", ch),
                    ("qbad_restore_count", cr), ("forecast_n", values["fan_n"]),
                    ("delete_one_flip_count", values[prefix + "_delete_one_flip_count"]),
                    ("predicted_growth_advantage_restore", qr - qh),
                    ("predicted_breach_count_delta_restore", cr - ch),
                    ("restoration_predicted_growth_sacrifice", chosen & (qr - qh < -NUMERIC_TOL))):
                    arm[key][active, year] = value
                reasons = (("growth_strict_advantage", "growth_tie_or_hold_advantage") if prefix == "growth" else
                           ("pareto_growth_or_risk_requires_restore", "pareto_hold"))
                arm["gate_reason"][active, year] = np.where(chosen, *reasons)
        post, fee, moved = current.copy(), np.zeros(n), np.zeros(n)
        if np.any(restore):
            post[restore], fee[restore], moved[restore] = execute_target(current[restore], w, rate)
        arm["posttrade_holdings"][:, year] = post
        arm["fees"][:, year], arm["moved"][:, year] = fee, moved
        arm["pretrade_wealth"][:, year] = wealth
        arm["fee_fraction"][:, year], arm["moved_fraction"][:, year] = fee / wealth, moved / wealth
        arm["pretrade_tv_drift"][:, year] = .5 * np.abs(u - w).sum(axis=-1)
        arm["target_execution"][:, year], arm["pending_at_decision"][:, year] = restore, pending
        arm["actions"][:, year] = np.where(restore, "restore", "hold")
        after, cp, nominal, real, weights, peak, drawdown, worst, ever = _advance(
            post, 1 + market[:, year], arm["cpi"][:, year], 1 + infl[:, year],
            arm["real_peak"][:, year], arm["running_mdd"][:, year], pending, loss)
        first = (arm["first_crossing"] < 0) & ever
        arm["first_crossing"][first] = year + 1
        for key, value in (("drawdown_at_trigger", worst), ("peak_at_trigger", peak), ("real_wealth_at_trigger", real)):
            arm[key][first] = value[first]
        for key, value in (("holdings", after), ("weights", weights), ("nominal_wealth", nominal),
            ("real_wealth", real), ("real_peak", peak), ("current_drawdown", drawdown),
            ("running_mdd", worst), ("ever_breached", ever), ("cpi", cp)):
            arm[key][:, year + 1] = value
        arm["end_year_tv_drift"][:, year] = .5 * np.abs(weights - w).sum(axis=-1)
    _recovery(arm, loss, h)
    arm["trades"] = arm["moved"] > NUMERIC_TOL
    arm["end_weights"] = arm["weights"][:, 1:, :]
    arm["targets"], arm["loss_tolerances"] = w.copy(), np.asarray([loss])
    arm["policy"] = policy
    for value in arm.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return arm
