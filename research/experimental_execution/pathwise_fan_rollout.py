"""Pathwise hold shield on one fixed, complete calendar-path training fan.

Hold is allowed only if BOTH total terminal log growth and the absorbing
everbreach indicator weakly dominate restore on EVERY supplied continuation.
There is no growth epsilon, probability margin, risk price or average-score
tradeoff.  Fees and base-policy B continuations reuse the frozen v1 evaluator.

For an actual path retained in this same fixed fan at every date, backward
induction shows terminal growth >= B and everbreach <= B: at any state the
selected current-action-plus-B dominates current-restore-plus-B on that
path's suffix; induction replaces subsequent B by the shield without loss.
B holds after breach, when both policies do too.  Both share initial restore.
This is a finite-support, pathwise model statement.  It is NOT an OOF or
population risk guarantee; changing/dropping the actual continuation from
the fan invalidates the argument.  Comparisons use computed floating values;
test roundoff tolerances are separate from the gate's exact-zero threshold.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target
from .fan_rollout import (NUMERIC_TOL, _advance, _market, _parameters, _recovery,
                         evaluate_actions as _base_evaluate)

POLICY = "pathwise_hold_rollout"


def evaluate_actions(holdings: ArrayLike, cpi: ArrayLike, real_peak: ArrayLike,
                     running_mdd: ArrayLike, ever_breached: ArrayLike,
                     training_returns: ArrayLike, training_inflation: ArrayLike, *,
                     year_index: int, target: ArrayLike, L: float,
                     cost_rate: float = .0025, horizon: int | None = None) -> dict[str, Any]:
    """Return frozen-v1 pathwise Q values plus exact-zero shield diagnostics.

    Shapes, active-state validation, history and total-log units are exactly
    those of v1.evaluate_actions.  New(n,) arrays are pathwise_recommend_restore,
    min_log_advantage_hold, growth_violation_count, breach_violation_count,
    union_violation_count and delete_one_flip_count.  A growth violation is
    hold_log < restore_log, WITHOUT1e-12 slack.  A risk violation is b_hold and
    not b_restore.  A nonempty union forces restoration.  Deleting one path
    changes the local recommendation iff it removes the only violating path;
    this diagnostic is-1 for a single-path fan and never changes the action.
    """
    values = _base_evaluate(holdings, cpi, real_peak, running_mdd, ever_breached,
        training_returns, training_inflation, year_index=year_index, target=target,
        L=L, cost_rate=cost_rate, horizon=horizon)
    hold, restore = values["hold"], values["restore"]
    growth_violation = hold["total_log_growth"] < restore["total_log_growth"]
    breach_violation = hold["ever_breached"] & ~restore["ever_breached"]
    union_count = (growth_violation | breach_violation).sum(axis=-1)
    values.update({"pathwise_recommend_restore": union_count > 0,
        "min_log_advantage_hold": (hold["total_log_growth"] - restore["total_log_growth"]).min(axis=-1),
        "growth_violation_count": growth_violation.sum(axis=-1),
        "breach_violation_count": breach_violation.sum(axis=-1),
        "union_violation_count": union_count,
        "delete_one_flip_count": (union_count == 1).astype(np.int64) if values["fan_n"] > 1 else
                                 np.full(len(union_count), -1, dtype=np.int64)})
    for value in values.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return values


def simulate_policy(actual_returns: ArrayLike, actual_inflation: ArrayLike,
                    training_returns: ArrayLike, training_inflation: ArrayLike,
                    target: ArrayLike, *, L: float, cost_rate: float = .0025,
                    horizon: int | None = None, policy: str = POLICY) -> dict[str, Any]:
    """Replay the pathwise shield, preserving the v1 arm schema without m axis.

    Actual(n,T,3)/(n,T) and training(N,Ttrain,3)/(N,Ttrain) inputs are separate.
    Every account starts at pre-fee100/all bills, CPI1/peak100/MDD0.  Initial
    restore is mandatory.  Actual breach locks in no-response pending hold.
    Active post-inception gates use only training same-calendar t:H suffixes.

    New period(n,H) diagnostics have the names documented by evaluate_actions;
    inactive floats are NaN, integer counts-1.  Growth threshold is EXACT0.
    Output arrays are read-only.  No-trigger recovery/censor placeholders must
    be null in presentations, as for the frozen pending and v1 kernels.
    """
    if policy != POLICY:
        raise ValueError(f"policy must be{POLICY}")
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
        "qg_hold", "qg_restore", "predicted_growth_advantage_restore", "min_log_advantage_hold")})
    arm.update({key: np.full(periods, -1, dtype=np.int64) for key in (
        "qbad_hold_count", "qbad_restore_count", "forecast_n", "delete_one_flip_count",
        "predicted_breach_count_delta_restore", "growth_violation_count", "breach_violation_count", "union_violation_count")})
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
                chosen = values["pathwise_recommend_restore"]
                restore[active] = chosen
                qh, qr = values["hold"]["mean_log_growth"], values["restore"]["mean_log_growth"]
                ch, cr = values["hold"]["breach_count"], values["restore"]["breach_count"]
                for key, value in (("qg_hold", qh), ("qg_restore", qr), ("qbad_hold_count", ch),
                    ("qbad_restore_count", cr), ("forecast_n", values["fan_n"]),
                    ("predicted_growth_advantage_restore", qr - qh),
                    ("predicted_breach_count_delta_restore", cr - ch),
                    ("restoration_predicted_growth_sacrifice", chosen & (qr - qh < -NUMERIC_TOL))):
                    arm[key][active, year] = value
                for key in ("min_log_advantage_hold", "growth_violation_count", "breach_violation_count",
                            "union_violation_count", "delete_one_flip_count"):
                    arm[key][active, year] = values[key]
                arm["gate_reason"][active, year] = np.where(chosen, "pathwise_requires_restore", "pathwise_hold")
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
