"""Incremental target-state controllers on three already frozen products.

The actual hold alternative keeps the current target identity for its future
baseline.  A restore/retarget alternative executes its selected target now
and uses that SAME target in future annual-until-breach-then-hold forecasts.
Other products' hypothetical hold branches are never substituted for hold.
Technical risk prices and this recombined training fan are heuristic model
comparisons, not calibrated utility, conditional laws or risk certificates.
No files, market data, menu fitting or admission selection occur here.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target
from .fan_rollout import NUMERIC_TOL, _advance, _market, _parameters, _recovery, evaluate_actions as _fixed_actions
from .risk_priced_fan_rollout import DEFAULT_ANNUAL_PRICES, _prices, priced_choices


def _targets(values):
    w = np.asarray(values, float)
    if (w.shape != (3, 3) or not np.all(np.isfinite(w)) or np.any(w < 0)
            or not np.allclose(w.sum(axis=1), 1, atol=NUMERIC_TOL, rtol=0)):
        raise ValueError("target_matrix must be three finite nonnegative simplex rows in C/B/G order")
    return w / w.sum(axis=1, keepdims=True)


def _indices(values, n):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise ValueError("current target indices must be integers0..2")
    try:
        result = np.broadcast_to(raw, (n,))
    except ValueError as exc:
        raise ValueError("current_target_index must be scalar or(n,)") from exc
    if np.any(result < 0) or np.any(result > 2):
        raise ValueError("current target indices must be integers0..2")
    return result.astype(np.int64)


def evaluate_actions(holdings: ArrayLike, cpi: ArrayLike, real_peak: ArrayLike,
                     running_mdd: ArrayLike, ever_breached: ArrayLike,
                     training_returns: ArrayLike, training_inflation: ArrayLike, *,
                     year_index: int, target_matrix: ArrayLike, current_target_index: ArrayLike,
                     L: float, cost_rate: float = .0025, horizon: int | None = None,
                     updates_enabled: bool = True):
    """Return action Q arrays for active states, indexed hold/C/B/G=0/1/2/3.

    Holdings(3,) or(n,3); target index scalar/(n,).  Returned total_log_actions
    and ever_breached_actions are(n,4,N), qg_actions/qbad_actions and
    available_actions are(n,4).  Qg is net REAL TOTAL future log growth.
    Unavailable actions are NaN/-1/false and must be excluded from selection.
    Forecast states retain current actual CPI/peak/MDD and each branch's own
    absorbing breach.  Already-breached accounts have no automatic gate.
    """
    if not isinstance(updates_enabled, (bool, np.bool_)):
        raise ValueError("updates_enabled must be boolean")
    raw_targets = np.asarray(target_matrix, float).copy()
    w = _targets(raw_targets)
    x = np.asarray(holdings, float)
    if x.shape == (3,): x = x[None, :]
    if x.ndim != 2 or x.shape[-1] != 3 or len(x) == 0:
        raise ValueError("holdings must be(3,) or nonempty(n,3)")
    n = len(x); current = _indices(current_target_index, n)
    train, train_pi = _market(training_returns, training_inflation, "training fan")
    count = len(train)
    logs = np.full((n, 4, count), np.nan)
    bad = np.zeros((n, 4, count), dtype=bool)
    available = np.zeros((n, 4), dtype=bool)
    available[:, 0] = True
    available[np.arange(n), current + 1] = True
    if updates_enabled: available[:] = True
    for m in range(3):
        rows = np.ones(n, bool) if updates_enabled else current == m
        if not rows.any(): continue
        # Broadcast histories before selecting rows, preserving scalar inputs.
        history = [np.broadcast_to(np.asarray(v), (n,))[rows] for v in (cpi, real_peak, running_mdd, ever_breached)]
        values = _fixed_actions(x[rows], *history, train, train_pi,
            year_index=year_index, target=w[m], L=L, cost_rate=cost_rate, horizon=horizon)
        logs[rows, m + 1] = values["restore"]["total_log_growth"]
        bad[rows, m + 1] = values["restore"]["ever_breached"]
        # ONLY current-m hold has the baseline of actually doing nothing.
        local_current = current[rows] == m
        global_rows = np.flatnonzero(rows)[local_current]
        logs[global_rows, 0] = values["hold"]["total_log_growth"][local_current]
        bad[global_rows, 0] = values["hold"]["ever_breached"][local_current]
    qg = logs.mean(axis=-1)
    qbad = np.where(available, bad.sum(axis=-1), -1)
    result = {"total_log_actions": logs, "ever_breached_actions": bad, "qg_actions": qg,
              "qbad_actions": qbad, "available_actions": available, "fan_n": count}
    for value in result.values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    return result


def choose_actions(qg_actions, qbad_actions, available_actions, fan_n, annual_prices,
                   original_horizon, current_target_index):
    """Choose0/1/2/3, with strictly valid gains before global-tolerance ties.

    Finite scores are relative to the single actual hold baseline, and use
    original H*annual price*probability change.  Risk-first compares integer
    counts before Qg, never infinity arithmetic.  Priority is hold, current
    restore, other restores in C/B/G order.  Hold is selected when no strict
    benefit exists.  Returns(choice(n), scores(n,4), reason(n)).
    """
    qg, qb, available = np.asarray(qg_actions, float), np.asarray(qbad_actions), np.asarray(available_actions)
    if (qg.ndim != 2 or qg.shape[1] != 4 or len(qg) < 1 or qb.shape != qg.shape
            or available.shape != qg.shape or available.dtype.kind != "b" or qb.dtype.kind not in "iu"
            or isinstance(fan_n, (bool, np.bool_)) or not isinstance(fan_n, (int, np.integer)) or fan_n < 1
            or isinstance(original_horizon, (bool, np.bool_)) or not isinstance(original_horizon, (int, np.integer)) or original_horizon < 1):
        raise ValueError("Require(n,4) finite available Qg/integer counts, boolean availability and positive integer N/H")
    n = len(qg); current = _indices(current_target_index, n)
    if (not available[:, 0].all() or not available[np.arange(n), current + 1].all()
            or not np.isfinite(qg[available]).all() or np.any(qb[available] < 0) or np.any(qb[available] > fan_n)):
        raise ValueError("Hold/current restore must be available, with finite Qg and bounded counts")
    raw_price = np.asarray(annual_prices)
    if raw_price.dtype.kind == "b": raise ValueError("Technical prices cannot be boolean")
    try:
        prices = np.broadcast_to(np.asarray(annual_prices, float), (n,))
    except ValueError as exc:
        raise ValueError("annual_prices must broadcast to active accounts") from exc
    if not np.all((np.isfinite(prices) & (prices >= 0)) | np.isposinf(prices)):
        raise ValueError("Prices must be nonnegative finite values or risk-first +infinity label")
    choices, scores = np.zeros(n, dtype=np.int64), np.full((n, 4), np.nan)
    reasons = np.full(n, "", dtype="U48")
    for i in range(n):
        ci = int(current[i]) + 1
        # Preserve the parent's exact arithmetic and reason labels when the
        # menu is restricted to the same two fixed-target alternatives.
        if available[i].sum() == 2:
            chosen, score, reason = priced_choices(qg[i, 0], qg[i, ci], qb[i, 0], qb[i, ci],
                fan_n, prices[i], original_horizon)
            choices[i] = ci if bool(chosen) else 0
            scores[i, ci] = float(score)
            if np.isfinite(prices[i]): scores[i, 0] = 0.
            reasons[i] = str(reason)
            continue
        order = [0, ci, *[j for j in range(1, 4) if j != ci]]
        if np.isfinite(prices[i]):
            indexes = available[i]
            delta_q = qb[i, indexes].astype(float) - float(qb[i, 0])
            with np.errstate(over="ignore", invalid="ignore"):
                scores[i, indexes] = qg[i, indexes] - qg[i, 0] - original_horizon * prices[i] * (delta_q / fan_n)
            if not np.isfinite(scores[i, indexes]).all():
                raise ValueError("Finite action scores exceed numerical range")
            positive = available[i] & (scores[i] > NUMERIC_TOL)
            if positive.any():
                maximum = scores[i, positive].max()
                choices[i] = next(j for j in order if positive[j] and scores[i, j] >= maximum - NUMERIC_TOL)
            reasons[i] = "finite_score_positive" if choices[i] else "finite_score_nonpositive"
        else:
            minimum = qb[i, available[i]].min()
            valid = available[i] & (qb[i] == minimum)
            if qb[i, 0] == minimum:
                valid &= qg[i] - qg[i, 0] > NUMERIC_TOL
            if valid.any():
                maximum = qg[i, valid].max()
                choices[i] = next(j for j in order if valid[j] and qg[i, j] >= maximum - NUMERIC_TOL)
            if choices[i]:
                reasons[i] = "risk_first_lower_predicted_risk" if minimum < qb[i, 0] else "risk_first_equal_risk_growth_positive"
            else:
                reasons[i] = "risk_first_equal_risk_hold_or_tie"
    return choices, scores, reasons


def simulate_controllers(actual_returns: ArrayLike, actual_inflation: ArrayLike,
                         training_returns: ArrayLike, training_inflation: ArrayLike,
                         target_matrix: ArrayLike, annual_prices: ArrayLike = DEFAULT_ANNUAL_PRICES, *,
                         initial_target_index: int = 1, updates_enabled: bool = True,
                         L: float = .4, cost_rate: float = .0025, horizon: int | None = None):
    """Replay n accounts x p prices with current-target state, from100/all bills.

    Main schema matches risk_priced_fan_rollout.simulate_prices.  CPI alone is
    (n,H+1), state(n,p,H+1), periods(n,p,H), holdings add assets.  Extra fields:
    current_target_index(n,p,H+1), current_target_before/after(n,p,H),
    target_updated(n,p,H), action_index(n,p,H;hold0/restoreC1/B2/G3),
    qg_actions/qbad_actions/available_actions/action_scores(n,p,H,4),
    selected_score(n,p,H).  Initial/pending Q is null/NaN/-1, unavailable.

    Compatibility qg_restore/predicted_score describe CURRENT-target restore;
    selected_score describes the chosen action (finite hold=0, risk-firstNaN).
    Updating target metadata can have moved=0 and thus be a non-trade.  Initial
    Balanced allocation does not count as an update.  Pending never updates or
    re-enters.  Arrays are read-only, no inputs mutate and no data are loaded.
    """
    if not isinstance(updates_enabled, (bool, np.bool_)):
        raise ValueError("updates_enabled must be boolean")
    w = _targets(target_matrix)
    initial = int(_indices(initial_target_index, 1)[0])
    market, infl = _market(actual_returns, actual_inflation, "actual paths")
    train, train_pi = _market(training_returns, training_inflation, "training fan")
    requested_h = market.shape[1] if horizon is None else horizon
    _, loss, rate, h = _parameters(w[initial], L, cost_rate, requested_h, min(market.shape[1], train.shape[1]))
    prices = _prices(annual_prices); n, p = len(market), len(prices)
    periods, endpoints = (n, p, h), (n, p, h + 1)
    arm = {key: np.zeros(periods) for key in ("fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
                                            "pretrade_tv_drift", "end_year_tv_drift")}
    arm.update({key: np.zeros(endpoints) for key in ("nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd")})
    arm.update({key: np.full(periods, np.nan) for key in ("qg_hold", "qg_restore", "predicted_score", "selected_score", "predicted_growth_advantage_restore")})
    arm.update({key: np.full(periods, -1, dtype=np.int64) for key in ("qbad_hold_count", "qbad_restore_count", "forecast_n", "predicted_breach_count_delta_restore")})
    arm["qg_actions"], arm["action_scores"] = np.full(periods + (4,), np.nan), np.full(periods + (4,), np.nan)
    arm["qbad_actions"] = np.full(periods + (4,), -1, dtype=np.int64)
    arm["available_actions"] = np.zeros(periods + (4,), bool)
    arm["current_target_index"] = np.full(endpoints, initial, dtype=np.int64)
    for key in ("current_target_before", "current_target_after"): arm[key] = np.full(periods, initial, dtype=np.int64)
    arm["action_index"] = np.zeros(periods, dtype=np.int64)
    arm["target_updated"] = np.zeros(periods, bool)
    arm["gate_reason"] = np.full(periods, "", dtype="U48")
    arm["restoration_predicted_growth_sacrifice"] = np.zeros(periods, bool)
    arm["holdings"], arm["weights"] = np.zeros(endpoints + (3,)), np.zeros(endpoints + (3,))
    arm["posttrade_holdings"] = np.zeros(periods + (3,))
    arm["ever_breached"] = np.zeros(endpoints, bool)
    arm["target_execution"], arm["pending_at_decision"] = np.zeros(periods, bool), np.zeros(periods, bool)
    arm["actions"] = np.full(periods, "hold", dtype="U7")
    arm["first_crossing"] = np.full((n, p), -1, dtype=np.int64)
    for key in ("drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger"): arm[key] = np.full((n, p), np.nan)
    arm["cpi"] = np.ones((n, h + 1))
    arm["holdings"][..., 0, 2], arm["weights"][..., 0, 2] = 100., 1.
    for key in ("nominal_wealth", "real_wealth", "real_peak"): arm[key][..., 0] = 100.
    grid_prices = np.broadcast_to(prices, (n, p))
    for t in range(h):
        current, wealth, u = arm["holdings"][..., t, :], arm["nominal_wealth"][..., t], arm["weights"][..., t, :]
        before = arm["current_target_index"][..., t]
        after_target = before.copy()
        pending = arm["ever_breached"][..., t]
        choice = np.full((n, p), initial + 1 if t == 0 else 0, dtype=np.int64)
        arm["gate_reason"][..., t] = np.where(pending, "already_breached_pending_hold", "")
        if t == 0:
            arm["gate_reason"][..., t] = "mandatory_initial_restore"
        else:
            active = ~pending
            if active.any():
                cp = np.broadcast_to(arm["cpi"][:, None, t], (n, p))
                values = evaluate_actions(current[active], cp[active], arm["real_peak"][..., t][active],
                    arm["running_mdd"][..., t][active], pending[active], train, train_pi,
                    year_index=t, target_matrix=target_matrix, current_target_index=before[active], L=loss,
                    cost_rate=rate, horizon=h, updates_enabled=updates_enabled)
                chosen, scores, reasons = choose_actions(values["qg_actions"], values["qbad_actions"], values["available_actions"],
                    values["fan_n"], grid_prices[active], h, before[active])
                choice[active] = chosen
                count = len(chosen); ci = before[active] + 1; row = np.arange(count)
                qg, qb = values["qg_actions"], values["qbad_actions"]
                current_restore_scores = scores[row, ci]
                for key, value in (("qg_actions", qg), ("qbad_actions", qb), ("available_actions", values["available_actions"]),
                    ("action_scores", scores), ("qg_hold", qg[:, 0]), ("qg_restore", qg[row, ci]),
                    ("qbad_hold_count", qb[:, 0]), ("qbad_restore_count", qb[row, ci]), ("forecast_n", values["fan_n"]),
                    ("predicted_score", current_restore_scores), ("selected_score", scores[row, chosen]), ("gate_reason", reasons),
                    ("predicted_growth_advantage_restore", qg[row, ci] - qg[:, 0]),
                    ("predicted_breach_count_delta_restore", qb[row, ci] - qb[:, 0]),
                    ("restoration_predicted_growth_sacrifice", (chosen > 0) & (qg[row, chosen] - qg[:, 0] < -NUMERIC_TOL))):
                    if key in ("qg_actions", "qbad_actions", "available_actions", "action_scores"):
                        arm[key][..., t, :][active] = value
                    else:
                        arm[key][..., t][active] = value
        restore = choice > 0
        after_target[restore] = choice[restore] - 1
        arm["current_target_before"][..., t], arm["current_target_after"][..., t] = before, after_target
        arm["current_target_index"][..., t + 1] = after_target
        arm["target_updated"][..., t] = (after_target != before) & (t > 0)
        arm["action_index"][..., t] = choice
        post, fee, moved = current.copy(), np.zeros((n, p)), np.zeros((n, p))
        if restore.any(): post[restore], fee[restore], moved[restore] = execute_target(current[restore], w[after_target[restore]], rate)
        arm["posttrade_holdings"][..., t, :] = post
        arm["fees"][..., t], arm["moved"][..., t] = fee, moved
        arm["pretrade_wealth"][..., t] = wealth
        arm["fee_fraction"][..., t], arm["moved_fraction"][..., t] = fee / wealth, moved / wealth
        arm["pretrade_tv_drift"][..., t] = .5 * np.abs(u - w[before]).sum(axis=-1)
        arm["target_execution"][..., t], arm["pending_at_decision"][..., t] = restore, pending
        arm["actions"][..., t] = np.where(restore, "restore", "hold")
        after, cp_next, nominal, real, weights, peak, drawdown, worst, ever = _advance(
            post, 1 + market[:, None, t, :], arm["cpi"][:, None, t], 1 + infl[:, None, t],
            arm["real_peak"][..., t], arm["running_mdd"][..., t], pending, loss)
        arm["cpi"][:, t + 1] = cp_next[:, 0]
        first = (arm["first_crossing"] < 0) & ever
        arm["first_crossing"][first] = t + 1
        for key, value in (("drawdown_at_trigger", worst), ("peak_at_trigger", peak), ("real_wealth_at_trigger", real)):
            arm[key][first] = value[first]
        for key, value in (("holdings", after), ("weights", weights), ("nominal_wealth", nominal), ("real_wealth", real),
                           ("real_peak", peak), ("current_drawdown", drawdown), ("running_mdd", worst), ("ever_breached", ever)):
            if key in ("holdings", "weights"): arm[key][..., t + 1, :] = value
            else: arm[key][..., t + 1] = value
        arm["end_year_tv_drift"][..., t] = .5 * np.abs(weights - w[after_target]).sum(axis=-1)
    small = {key: arm[key].reshape(n * p, -1) for key in ("real_wealth", "running_mdd")}
    for key in ("first_crossing", "peak_at_trigger", "drawdown_at_trigger"): small[key] = arm[key].reshape(-1)
    _recovery(small, loss, h)
    for key in ("triggered", "recovered", "recovery_endpoint", "recovery_time", "right_censored", "available_followup", "observed_followup", "added_severity", "terminal_excess"):
        arm[key] = small[key].reshape(n, p)
    arm["trades"] = arm["moved"] > NUMERIC_TOL
    arm["end_weights"] = arm["weights"][..., 1:, :]
    arm["annual_prices"], arm["targets"], arm["loss_tolerances"] = prices, w[initial].copy(), np.asarray([loss])
    arm["target_matrix"], arm["updates_enabled"] = w, bool(updates_enabled)
    for value in arm.values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    return arm
