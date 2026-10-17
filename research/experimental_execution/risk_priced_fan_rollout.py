"""Finite technical-price family on the frozen calendar-fan evaluator.

This is an execution heuristic, not exact constrained DP or a calibrated
customer utility.  A caller must select prices by full-policy training
feasibility, retain Review outcomes, and evaluate the selected learner on
whole held-out paths.  Neither local score nor price search certifies risk.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target
from .fan_rollout import NUMERIC_TOL, _advance, _market, _parameters, evaluate_actions

DEFAULT_ANNUAL_PRICES = np.array([0., .001, .003, .01, .03, .1, .3, 1., np.inf])


def priced_choices(qg_hold, qg_restore, bad_hold, bad_restore, fan_n, annual_prices, original_horizon):
    """Return restore flags, finite total-log scores and explicit reasons.

    Inputs broadcast over active accounts.  Bad values are INTEGER counts;
    divide by the common fan denominator exactly once.  Positive infinity is
    a label for the separately defined risk-first candidate, never infinity
    arithmetic.  The multiplier uses the original horizon at every date.
    """
    if np.asarray(annual_prices).dtype.kind == "b": raise ValueError("Prices are numeric technical parameters, not booleans")
    qh, qr, bh, br, prices = np.broadcast_arrays(np.asarray(qg_hold, float), np.asarray(qg_restore, float),
        np.asarray(bad_hold), np.asarray(bad_restore), np.asarray(annual_prices, float))
    if (not np.all(np.isfinite(qh)) or not np.all(np.isfinite(qr))
            or bh.dtype.kind not in "iu" or br.dtype.kind not in "iu"
            or isinstance(fan_n, (bool, np.bool_)) or not isinstance(fan_n, (int, np.integer)) or fan_n < 1
            or np.any(bh < 0) or np.any(br < 0) or np.any(bh > fan_n) or np.any(br > fan_n)
            or not np.all((np.isfinite(prices) & (prices >= 0)) | np.isposinf(prices))
            or isinstance(original_horizon, (bool, np.bool_)) or not isinstance(original_horizon, (int, np.integer))
            or original_horizon < 1):
        raise ValueError("Require finite Qg, integer bounded bad counts, N/H positive integers and nonnegative prices")
    finite = np.isfinite(prices); advantage = qr - qh
    scores = np.full(qh.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        # Counts may be unsigned; subtract after conversion to avoid wrapping
        # a risk reduction into a huge positive count.  Empirical fan counts
        # in this study are at most120 and are exact in float64.
        count_delta = br[finite].astype(np.float64) - bh[finite].astype(np.float64)
        scores[finite] = advantage[finite] - original_horizon * prices[finite] * (count_delta / fan_n)
    if not np.all(np.isfinite(scores[finite])): raise ValueError("Finite-price total-log score exceeds numerical range")
    restore = np.zeros(qh.shape, dtype=bool)
    restore[finite] = scores[finite] > NUMERIC_TOL
    restore[~finite] = ((br[~finite] < bh[~finite]) |
                        ((br[~finite] == bh[~finite]) & (advantage[~finite] > NUMERIC_TOL)))
    reasons = np.where(restore, "finite_score_positive", "finite_score_nonpositive").astype("U48")
    lex = ~finite
    reasons[lex & (br < bh)] = "risk_first_lower_predicted_risk"
    reasons[lex & (br > bh)] = "risk_first_higher_predicted_risk"
    reasons[lex & (br == bh) & restore] = "risk_first_equal_risk_growth_positive"
    reasons[lex & (br == bh) & ~restore] = "risk_first_equal_risk_hold_or_tie"
    return restore, scores, reasons


def _prices(values):
    raw = np.asarray(values)
    prices = np.asarray(values, float)
    if (raw.dtype.kind == "b" or prices.ndim != 1 or len(prices) < 1
            or not np.all((np.isfinite(prices) & (prices >= 0)) | np.isposinf(prices))
            or len(np.unique(prices)) != len(prices)):
        raise ValueError("annual_prices must be a nonempty unique vector of nonnegative finite values/risk-first label")
    return prices.copy()


def simulate_prices(actual_returns: ArrayLike, actual_inflation: ArrayLike,
                    training_returns: ArrayLike, training_inflation: ArrayLike,
                    target: ArrayLike, annual_prices: ArrayLike = DEFAULT_ANNUAL_PRICES, *,
                    L: float, cost_rate: float = .0025, horizon: int | None = None):
    """Execute all declared prices, preserving a separate price axis.

    Actual arrays are(n,T,3)/(n,T), forecast arrays(N,Ttrain,3)/(N,Ttrain),
    target(3,), annual_prices(p,).  H is the original client horizon.  State
    arrays are(n,p,H+1), period arrays(n,p,H), holdings add a final asset axis.
    CPI is(n,H+1), common to prices on the same actual path.  annual_prices(p)
    is metadata.  Inactive Q/score values are NaN and integer counts are-1;
    risk-first active scores are NaN with explicit lexicographic reasons.

    Inception restores mandatorily and actual already-breached accounts hold.
    Only frozen training suffixes enter the active evaluator; actual year's
    return is read after its action.  Exact fee is paid once.  Outputs are
    read-only.  Recovery sentinels must be serialized null when inapplicable.
    """
    market, infl = _market(actual_returns, actual_inflation, "actual paths")
    train, train_infl = _market(training_returns, training_inflation, "training fan")
    requested_h = market.shape[1] if horizon is None else horizon
    w, loss, rate, h = _parameters(target, L, cost_rate, requested_h, min(market.shape[1], train.shape[1]))
    prices = _prices(annual_prices); n, p = len(market), len(prices)
    periods, endpoints = (n, p, h), (n, p, h + 1)
    arm = {key: np.zeros(periods) for key in ("fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
                                             "pretrade_tv_drift", "end_year_tv_drift")}
    arm.update({key: np.zeros(endpoints) for key in ("nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd")})
    arm.update({key: np.full(periods, np.nan) for key in ("qg_hold", "qg_restore", "predicted_score", "predicted_growth_advantage_restore")})
    arm.update({key: np.full(periods, -1, dtype=np.int64) for key in ("qbad_hold_count", "qbad_restore_count", "forecast_n", "predicted_breach_count_delta_restore")})
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
        pending = arm["ever_breached"][..., t]
        restore = np.full((n, p), t == 0, dtype=bool)
        arm["gate_reason"][..., t] = np.where(pending, "already_breached_pending_hold", "")
        if t == 0:
            arm["gate_reason"][..., t] = "mandatory_initial_restore"
        else:
            active = ~pending
            if np.any(active):
                cp = np.broadcast_to(arm["cpi"][:, None, t], (n, p))
                values = evaluate_actions(current[active], cp[active], arm["real_peak"][..., t][active],
                    arm["running_mdd"][..., t][active], pending[active], train, train_infl,
                    year_index=t, target=w, L=loss, cost_rate=rate, horizon=h)
                qh, qr = values["hold"]["mean_log_growth"], values["restore"]["mean_log_growth"]
                bh, br = values["hold"]["breach_count"], values["restore"]["breach_count"]
                chosen, score, reason = priced_choices(qh, qr, bh, br, values["fan_n"], grid_prices[active], h)
                restore[active] = chosen
                for key, value in (("qg_hold", qh), ("qg_restore", qr), ("qbad_hold_count", bh), ("qbad_restore_count", br),
                    ("forecast_n", values["fan_n"]), ("predicted_score", score), ("gate_reason", reason),
                    ("predicted_growth_advantage_restore", qr - qh), ("predicted_breach_count_delta_restore", br - bh),
                    ("restoration_predicted_growth_sacrifice", chosen & (qr - qh < -NUMERIC_TOL))):
                    arm[key][..., t][active] = value
        post, fee, moved = current.copy(), np.zeros((n, p)), np.zeros((n, p))
        if np.any(restore): post[restore], fee[restore], moved[restore] = execute_target(current[restore], w, rate)
        arm["posttrade_holdings"][..., t, :] = post
        arm["fees"][..., t], arm["moved"][..., t] = fee, moved
        arm["pretrade_wealth"][..., t] = wealth
        arm["fee_fraction"][..., t], arm["moved_fraction"][..., t] = fee / wealth, moved / wealth
        arm["pretrade_tv_drift"][..., t] = .5 * np.abs(u - w).sum(axis=-1)
        arm["target_execution"][..., t], arm["pending_at_decision"][..., t] = restore, pending
        arm["actions"][..., t] = np.where(restore, "restore", "hold")
        after, cp_next, nominal, real, weights, peak, drawdown, worst, ever = _advance(
            post, 1 + market[:, None, t, :], arm["cpi"][:, None, t], 1 + infl[:, None, t],
            arm["real_peak"][..., t], arm["running_mdd"][..., t], pending, loss)
        if not np.all(cp_next == cp_next[:, :1]): raise RuntimeError("CPI unexpectedly differs by price")
        arm["cpi"][:, t + 1] = cp_next[:, 0]
        first = (arm["first_crossing"] < 0) & ever
        arm["first_crossing"][first] = t + 1
        for key, value in (("drawdown_at_trigger", worst), ("peak_at_trigger", peak), ("real_wealth_at_trigger", real)):
            arm[key][first] = value[first]
        for key, value in (("holdings", after), ("weights", weights), ("nominal_wealth", nominal), ("real_wealth", real),
                           ("real_peak", peak), ("current_drawdown", drawdown), ("running_mdd", worst), ("ever_breached", ever)):
            if key in ("holdings", "weights"): arm[key][..., t + 1, :] = value
            else: arm[key][..., t + 1] = value
        arm["end_year_tv_drift"][..., t] = .5 * np.abs(weights - w).sum(axis=-1)
    triggered = arm["first_crossing"] >= 0
    times = np.arange(h + 1)[None, None, :]
    possible = triggered[..., None] & (times > arm["first_crossing"][..., None]) & (arm["real_wealth"] >= arm["peak_at_trigger"][..., None])
    recovered, endpoint = possible.any(axis=-1), possible.argmax(axis=-1)
    arm["triggered"], arm["recovered"] = triggered, recovered
    arm["recovery_endpoint"] = np.where(recovered, endpoint, -1).astype(np.int64)
    arm["recovery_time"] = np.where(recovered, endpoint - arm["first_crossing"], -1).astype(np.int64)
    arm["right_censored"] = triggered & ~recovered
    arm["available_followup"] = np.where(triggered, h - arm["first_crossing"], -1).astype(np.int64)
    arm["observed_followup"] = np.where(recovered, arm["recovery_time"], arm["available_followup"])
    arm["added_severity"] = np.where(triggered, np.maximum(arm["running_mdd"][..., -1] - arm["drawdown_at_trigger"], 0), np.nan)
    arm["terminal_excess"] = np.maximum(arm["running_mdd"][..., -1] - loss, 0)
    arm["trades"] = arm["moved"] > NUMERIC_TOL
    arm["end_weights"] = arm["weights"][..., 1:, :]
    arm["annual_prices"], arm["targets"], arm["loss_tolerances"] = prices, w.copy(), np.asarray([loss])
    for value in arm.values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    return arm
