"""Pure positive estimated-benefit execution with an explicit current fee.

The two declared estimands compare trade-once-then-hold to hold, either for
one calendar year or through the original client horizon. Both use the same
complete training-path suffixes, and both apply a common original-H
buy-and-hold risk-direction veto to OPTIONAL restorations. Neither the
unconditional empirical fan nor this veto is a conditional risk guarantee.
Annual replanning does not implement the forecast's no-later-trade promise.

Inception is mandatory only after its own estimated economic gain passes;
it does not use the risk-direction veto against the all-bill alternative.
Its actual fee is still paid once. Nonpositive inception estimates raise a
specific ValueError for the caller to retain a Review outcome. Original
peak/CPI/MDD/H survive, actual breach permanently holds, and there is no
terminal liquidation. This module performs no I/O, fitting or selection.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target
from .fan_rollout import (NUMERIC_TOL, _advance, _market, _parameters,
                          _recovery, _state)

BENEFIT_HORIZONS = ("one_year", "remaining_to_original_H")


def _benefit_horizon(value):
    if not isinstance(value, str) or value not in BENEFIT_HORIZONS:
        raise ValueError(f"benefit_horizon must be one of {BENEFIT_HORIZONS}")
    return value


def _readonly(value):
    if isinstance(value, dict):
        for entry in value.values():
            _readonly(entry)
    elif isinstance(value, np.ndarray):
        value.setflags(write=False)
    return value


class InitialEconomicRejection(ValueError):
    """A specified candidate is undeployable, rather than a hold/B fallback.

    ``estimate`` is the read-only initial estimate, including quote and
    original-H risk counts; ``failed_indices`` identifies nonpassing rows.
    No actual return or actual executed account has yet been advanced.
    """
    reason = "mandatory_initial_restore_economic_nonpositive"

    def __init__(self, estimate):
        self.estimate = estimate
        self.failed_indices = np.flatnonzero(~estimate["economic_pass"])
        self.failed_indices.setflags(write=False)
        super().__init__(self.reason)


def _hold_branches(x, post, price, peak, worst, train, infl, year, h, loss):
    """Current hold/paid restore followed ONLY by hold; retain actual history."""
    n, fan_n = len(x), len(train)
    current = np.broadcast_to(np.stack((x, post), axis=1)[:, :, None, :],
                              (n, 2, fan_n, 3)).copy()
    shape = (n, 2, fan_n)
    cp = np.broadcast_to(price[:, None, None], shape).copy()
    pk = np.broadcast_to(peak[:, None, None], shape).copy()
    dd = np.broadcast_to(worst[:, None, None], shape).copy()
    bad = np.zeros(shape, dtype=bool)
    start = x.sum(axis=-1) / price
    one_year = None
    for t in range(year, h):
        current, cp, _, real, _, pk, _, dd, bad = _advance(
            current, 1 + train[None, None, :, t, :], cp,
            1 + infl[None, None, :, t], pk, dd, bad, loss)
        if t == year:
            one_year = np.log(real) - np.log(start[:, None, None])
    total = np.log(real) - np.log(start[:, None, None])
    return {name: {"one_year_log_growth": one_year[:, j, :].copy(),
                   "total_log_growth": total[:, j, :].copy(),
                   "ever_breached": bad[:, j, :].copy(),
                   "running_mdd": dd[:, j, :].copy()}
            for j, name in enumerate(("hold", "restore"))}


def estimate_trade(holdings: ArrayLike, cpi: ArrayLike, real_peak: ArrayLike,
                   running_mdd: ArrayLike, ever_breached: ArrayLike,
                   training_returns: ArrayLike, training_inflation: ArrayLike, *,
                   year_index: int, target: ArrayLike, L: float,
                   cost_rate: float = .0025, horizon: int | None = None,
                   benefit_horizon: str = "one_year") -> dict:
    """Quote the current trade and compare paired empirical gross/net logs.

    State inputs accept (3,) or (n,3), with matching scalar/(n,) history.
    Only active pre-breach states are accepted. Training is (N,T,3)/(N,T),
    ``year_index`` is the next zero-based return column, and ``horizon`` is
    ORIGINAL H. The function has no actual-future-return argument.

    per-fan arrays gross_log_gain/net_log_gain are (n,N); their means and
    exact current quote/log-fee are (n,). ``hold``/``restore`` contain (n,N)
    one-year/remaining total real logs, ever-breach and historical MDD under
    no future trades. Risk counts always use the whole remaining original-H
    suffix, regardless of the economic-benefit horizon. Recommendations are
    OPTIONAL decisions: mean_net>1e-12 AND k_restore<=k_hold. The caller
    handles mandatory inception separately using economic_pass alone.

    If rho=C/V, v=post/(V-C), u=x/V, and Q is a joint asset gross vector
    (or asset-wise cumulative suffix gross), then gross=log(v.Q/u.Q),
    fee_log_penalty=-log1p(-rho), net=gross-fee_log_penalty. No fee is added
    again to a direct paid-wealth comparison. The quoted post/C may be used
    directly for actual execution, without a second fee call.
    """
    mode = _benefit_horizon(benefit_horizon)
    train, infl = _market(training_returns, training_inflation, "training fan")
    w, loss, rate, h = _parameters(target, L, cost_rate, horizon, train.shape[1])
    if (isinstance(year_index, (bool, np.bool_))
            or not isinstance(year_index, (int, np.integer))
            or not 0 <= year_index < h):
        raise ValueError("year_index must be an integer within the original horizon")
    year = int(year_index)
    x, price, peak, worst, _ = _state(holdings, cpi, real_peak, running_mdd,
                                     ever_breached, loss)
    post, fee, moved = execute_target(x, w, rate)
    wealth = x.sum(axis=-1)
    rho = fee / wealth
    penalty = -np.log1p(-rho)
    u, v = x / wealth[:, None], post / (wealth - fee)[:, None]
    end = year + 1 if mode == "one_year" else h
    log_factors = np.log1p(train[:, year:end, :]).sum(axis=1)
    # Stable weighted sums, including exactly zero long-only weights.
    with np.errstate(divide="ignore"):
        log_u, log_v = np.log(u), np.log(v)
    hold_log = np.logaddexp.reduce(log_u[:, None, :] + log_factors[None, :, :], axis=-1)
    restore_log = np.logaddexp.reduce(log_v[:, None, :] + log_factors[None, :, :], axis=-1)
    gross = restore_log - hold_log
    net = gross - penalty[:, None]
    if not np.isfinite(gross).all() or not np.isfinite(net).all():
        raise ValueError("Estimated log advantage exceeds the finite numeric range")
    branches = _hold_branches(x, post, price, peak, worst, train, infl, year, h, loss)
    kh = branches["hold"]["ever_breached"].sum(axis=-1)
    kr = branches["restore"]["ever_breached"].sum(axis=-1)
    mean_net = net.mean(axis=-1)
    economic = mean_net > NUMERIC_TOL
    direction = kr <= kh
    recommend = economic & direction
    reason = np.where(~economic, "estimated_net_nonpositive_hold",
                      np.where(~direction, "full_H_risk_direction_veto_hold",
                               "positive_net_and_risk_direction_restore"))
    result = {**branches, "posttrade_holdings": post, "quoted_nominal_fee": fee,
              "quoted_nominal_moved": moved, "quoted_fee_fraction": rho,
              "fee_log_penalty": penalty, "gross_log_gain": gross,
              "net_log_gain": net, "mean_gross_log_gain": gross.mean(axis=-1),
              "mean_net_log_gain": mean_net, "risk_hold_breach_count": kh,
              "risk_restore_breach_count": kr, "economic_pass": economic,
              "risk_direction_pass": direction, "recommend_restore": recommend,
              "gate_reason": reason, "fan_n": len(train), "year_index": year,
              "horizon": h, "benefit_horizon": mode,
              "benefit_periods": end - year,
              "benefit_continuation": "trade_once_then_hold",
              "risk_continuation": "trade_once_then_hold_to_original_H"}
    return _readonly(result)


def simulate_benefit_cost(actual_returns: ArrayLike, actual_inflation: ArrayLike,
                          training_returns: ArrayLike, training_inflation: ArrayLike,
                          target: ArrayLike, *, L: float,
                          benefit_horizon: str = "one_year",
                          cost_rate: float = .0025,
                          horizon: int | None = None) -> dict:
    """Replay one causal fixed-target benefit-cost candidate; no family fitting.

    Returns the squeezed simple/fan physical arm schema. Additional (n,H)
    fields are mean_gross_log_gain, mean_net_log_gain, fee_log_penalty,
    quoted_nominal_fee, quoted_nominal_moved, quoted_fee_fraction,
    risk_hold_breach_count/risk_restore_breach_count, forecast_n,benefit_periods,
    economic_pass/risk_direction_pass/estimate_available, gate_reason.
    Active inception also has an estimate, but risk_direction_pass does not
    veto its mandatory restore. Pending estimates are absent (NaN/-1/False),
    never fabricated zero predictions. No full (n,H,N) estimate tree is kept.

    Nonpositive initial mean_net raises InitialEconomicRejection BEFORE any
    actual return is used, so the caller can preserve a candidate Review.
    Every OPTIONAL restore has positive estimated net gain and passes the
    common full-H hold-branch direction veto. Neither property is a safety
    theorem; actual full-H risk/admission must be assessed externally.
    """
    mode = _benefit_horizon(benefit_horizon)
    market, pi = _market(actual_returns, actual_inflation, "actual paths")
    train, train_pi = _market(training_returns, training_inflation, "training fan")
    requested = market.shape[1] if horizon is None else horizon
    w, loss, rate, h = _parameters(target, L, cost_rate, requested,
                                  min(market.shape[1], train.shape[1]))
    n = len(market)
    periods, endpoints = (n, h), (n, h + 1)
    arm = {key: np.zeros(periods) for key in (
        "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
        "pretrade_tv_drift", "end_year_tv_drift")}
    arm.update({key: np.zeros(endpoints) for key in (
        "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd")})
    arm.update({key: np.full(periods, np.nan) for key in (
        "mean_gross_log_gain", "mean_net_log_gain", "fee_log_penalty",
        "quoted_nominal_fee", "quoted_nominal_moved", "quoted_fee_fraction")})
    arm.update({key: np.full(periods, -1, dtype=np.int64) for key in (
        "risk_hold_breach_count", "risk_restore_breach_count", "forecast_n", "benefit_periods")})
    arm.update({key: np.zeros(periods, dtype=bool) for key in (
        "economic_pass", "risk_direction_pass", "estimate_available",
        "target_execution", "pending_at_decision")})
    arm["holdings"], arm["weights"] = np.zeros(endpoints + (3,)), np.zeros(endpoints + (3,))
    arm["posttrade_holdings"] = np.zeros(periods + (3,))
    arm["ever_breached"] = np.zeros(endpoints, dtype=bool)
    arm["cpi"] = np.ones(endpoints)
    arm["actions"] = np.full(periods, "hold", dtype="U7")
    arm["gate_reason"] = np.full(periods, "", dtype="U56")
    arm["first_crossing"] = np.full(n, -1, dtype=np.int64)
    for key in ("drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger"):
        arm[key] = np.full(n, np.nan)
    arm["holdings"][:, 0, 2], arm["weights"][:, 0, 2] = 100., 1.
    for key in ("nominal_wealth", "real_wealth", "real_peak"):
        arm[key][:, 0] = 100.

    float_fields = ("mean_gross_log_gain", "mean_net_log_gain", "fee_log_penalty",
                    "quoted_nominal_fee", "quoted_nominal_moved", "quoted_fee_fraction")
    count_fields = ("risk_hold_breach_count", "risk_restore_breach_count")
    for year in range(h):
        current, wealth = arm["holdings"][:, year, :], arm["nominal_wealth"][:, year]
        pending = arm["ever_breached"][:, year]
        active = ~pending
        restore = np.zeros(n, dtype=bool)
        post, fee, moved = current.copy(), np.zeros(n), np.zeros(n)
        arm["gate_reason"][pending, year] = "already_breached_pending_hold"
        if np.any(active):
            values = estimate_trade(current[active], arm["cpi"][active, year],
                arm["real_peak"][active, year], arm["running_mdd"][active, year], pending[active],
                train, train_pi, year_index=year, target=w, L=loss,
                cost_rate=rate, horizon=h, benefit_horizon=mode)
            if year == 0 and not np.all(values["economic_pass"]):
                raise InitialEconomicRejection(values)
            chosen = np.ones(np.sum(active), dtype=bool) if year == 0 else values["recommend_restore"]
            restore[active] = chosen
            for key in float_fields + count_fields + ("economic_pass", "risk_direction_pass"):
                arm[key][active, year] = values[key]
            arm["forecast_n"][active, year] = values["fan_n"]
            arm["benefit_periods"][active, year] = values["benefit_periods"]
            arm["estimate_available"][active, year] = True
            arm["gate_reason"][active, year] = ("mandatory_initial_restore_economic_positive"
                if year == 0 else values["gate_reason"])
            chosen_ids = np.flatnonzero(active)[chosen]
            post[chosen_ids] = values["posttrade_holdings"][chosen]
            fee[chosen_ids] = values["quoted_nominal_fee"][chosen]
            moved[chosen_ids] = values["quoted_nominal_moved"][chosen]
        arm["posttrade_holdings"][:, year] = post
        arm["fees"][:, year], arm["moved"][:, year] = fee, moved
        arm["pretrade_wealth"][:, year] = wealth
        arm["fee_fraction"][:, year], arm["moved_fraction"][:, year] = fee / wealth, moved / wealth
        arm["pretrade_tv_drift"][:, year] = .5 * np.abs(arm["weights"][:, year] - w).sum(axis=-1)
        arm["target_execution"][:, year], arm["pending_at_decision"][:, year] = restore, pending
        arm["actions"][:, year] = np.where(restore, "restore", "hold")
        # The current ACTUAL return is read only after the estimate and action.
        after, cp, nominal, real, weights, peak, drawdown, worst, ever = _advance(
            post, 1 + market[:, year, :], arm["cpi"][:, year], 1 + pi[:, year],
            arm["real_peak"][:, year], arm["running_mdd"][:, year], pending, loss)
        first = (arm["first_crossing"] < 0) & ever
        arm["first_crossing"][first] = year + 1
        for key, value in (("drawdown_at_trigger", worst), ("peak_at_trigger", peak),
                           ("real_wealth_at_trigger", real)):
            arm[key][first] = value[first]
        for key, value in (("holdings", after), ("cpi", cp), ("nominal_wealth", nominal),
            ("real_wealth", real), ("weights", weights), ("real_peak", peak),
            ("current_drawdown", drawdown), ("running_mdd", worst), ("ever_breached", ever)):
            arm[key][:, year + 1] = value
        arm["end_year_tv_drift"][:, year] = .5 * np.abs(weights - w).sum(axis=-1)

    _recovery(arm, loss, h)
    arm["trades"] = arm["moved"] > NUMERIC_TOL
    arm["end_weights"] = arm["weights"][:, 1:, :]
    arm["decision_reason"] = arm["gate_reason"]
    arm["targets"], arm["loss_tolerances"] = w.copy(), np.asarray([loss])
    arm["original_horizon"], arm["cost_rate"] = h, rate
    arm["benefit_horizon"] = mode
    arm["policy"] = "benefit_cost_" + mode
    arm["initial_deployable"] = True
    arm["trade_count"] = arm["trades"].sum(axis=1).astype(np.int64)
    arm["target_execution_count"] = arm["target_execution"].sum(axis=1).astype(np.int64)
    arm["nominal_fees"], arm["nominal_moved"] = arm["fees"].sum(axis=1), arm["moved"].sum(axis=1)
    real = arm["real_wealth"]
    peaks = np.maximum.accumulate(real, axis=1)
    logg = (np.log(real[:, -1]) - np.log(100.)) / h
    annual = real[:, 1:] / real[:, :-1] - 1
    arm.update(log_growth=logg, cagr=np.expm1(logg), mdd=np.max((peaks - real) / peaks, axis=1),
        volatility=np.std(annual, axis=1, ddof=1) if h > 1 else np.zeros(n),
        terminal_real=real[:, -1].copy(), breach=arm["ever_breached"][:, -1].copy())
    arm["excess"] = np.maximum(arm["mdd"] - loss, 0)
    return _readonly(arm)
