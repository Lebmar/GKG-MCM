"""Exploratory V3 pure primary: state-cushion-scaled temporary proposals.

Copied from frozen V2 primary8480, changing only its four proposal destinations
and adding witnesses. The13point support filter, paired forecast scores,
fees, budget release fraction and ties stay unchanged. This is not a hard
cap on holdings, compulsory cash, CPPI insurance or recursive feasibility.
The initial full original target is untouched; fallback is outside this file.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target

TOL = 1e-12
TILTS = np.array([1., .75, .5, .25])
CANDIDATES = 13
MAX_FORECASTS = 5
FORECAST_LABELS = ("uniform", "shrink_conditional", "id_block_1", "id_block_2", "id_block_3")


def _scalar(value, name, lower, upper, *, lower_open=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + " must be a numeric scalar")
    result = float(value)
    if (not np.isfinite(result) or result > upper or result < lower or (lower_open and result == lower)):
        raise ValueError(name + " outside its declared range")
    return result


def _target(target):
    w = np.asarray(target, float)
    if (w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1., atol=TOL, rtol=0)):
        raise ValueError("target must be one finite nonnegative three-asset simplex")
    return w / w.sum()


def _real_support(value, name):
    a = np.asarray(value, float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 1 or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError(name + " must be finite positive (members,3) real gross")
    return a


def cushion_scale(real_wealth, real_peak, L):
    """m=clip((R-(1-L)P)/(L R),0,1); no fitted parameter.

    At peak the scale is1. For L=0 this is an explicit boundary convention;
    a tolerated R<P state instead has scale0. Neither convention weakens
    the existing support floor or absorbing actual-breach rule.
    """
    real = _scalar(real_wealth, "real_wealth", 0, np.inf, lower_open=True)
    peak = _scalar(real_peak, "real_peak", 0, np.inf, lower_open=True)
    loss = _scalar(L, "L", 0, 1)
    if real > peak+TOL*max(real, peak):
        raise ValueError("Historical peak cannot be below real wealth")
    if real >= peak:
        return 1.
    if loss == 0:
        return 0.
    headroom = real-(1-loss)*peak
    if headroom <= 0:
        return 0.
    return float(np.clip(headroom/(loss*real), 0., 1.))


def segment_interval(base_margin: ArrayLike, slope: ArrayLike):
    """Exact closed [0,1] intersection of A_s+q B_s>=0; no tolerance budget.

    A common positive scale may be divided out by the caller. Binding member
    indices are zero-based; -1 means the [0,1] box supplies that bound.
    Empty intersections retain NaN endpoints and a contradictory-row witness.
    """
    a, b = np.asarray(base_margin, float), np.asarray(slope, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require matching finite nonempty margin/slope vectors")
    lo, hi, lower_member, upper_member = 0., 1., -1, -1
    for index, (margin, delta) in enumerate(zip(a, b)):
        if delta == 0:
            if margin < 0:
                return dict(feasible=False, lo=np.nan, hi=np.nan, lower_member=index, upper_member=index)
        elif delta > 0:
            bound = -margin / delta
            if bound > lo:
                lo, lower_member = float(bound), index
        else:
            bound = -margin / delta
            if bound < hi:
                hi, upper_member = float(bound), index
    if lo > hi or lo > 1 or hi < 0:
        return dict(feasible=False, lo=np.nan, hi=np.nan, lower_member=lower_member, upper_member=upper_member)
    return dict(feasible=True, lo=lo, hi=hi, lower_member=lower_member, upper_member=upper_member)


def decide_state_action(holdings, cpi, real_peak, running_mdd, support_real_gross,
                        forecast_real_gross, forecast_weights, *, target, L, cost_rate=.0025,
                        release_fraction=.5, forecast_rule="uniform", initial=False):
    """Pure scalar-account action and complete finite-menu witnesses.

    No actual current-year or future market is an argument. Forecast weights
    are (members-of-ensemble,N); support and growth members can differ (pooled
    support stress). Candidate0 is hold; each proposal then has lo/mid/hi.
    Relative support roundoff within1e-12 is recorded; interval construction
    itself never adds slack to the financial floor.
    """
    x = np.asarray(holdings, float); w = _target(target)
    loss = _scalar(L, "L", 0, 1); rate = _scalar(cost_rate, "cost_rate", 0, 1)
    rho = _scalar(release_fraction, "release_fraction", 0, 1, lower_open=True)
    if rate == 1 or forecast_rule not in ("uniform", "ensemble_min"):
        raise ValueError("Require cost_rate<1 and a declared forecast rule")
    cp = _scalar(cpi, "cpi", 0, np.inf, lower_open=True)
    peak = _scalar(real_peak, "real_peak", 0, np.inf, lower_open=True)
    worst = _scalar(running_mdd, "running_mdd", 0, 1)
    if x.shape != (3,) or not np.isfinite(x).all() or np.any(x < 0) or x.sum() <= 0:
        raise ValueError("holdings must be positive-total finite nonnegative (3,)")
    wealth = float(x.sum()); real = wealth / cp
    if not np.isfinite(wealth) or not np.isfinite(real) or real <= 0:
        raise ValueError("Positive current nominal/real wealth must be within numerical range")
    if real > peak + TOL * max(real, peak) or worst + TOL < max(1 - real / peak, 0) or worst - loss > TOL:
        raise ValueError("Only consistent unbreached current/history states enter the active helper")
    peak = max(peak, real)
    support = _real_support(support_real_gross, "support")
    growth = _real_support(forecast_real_gross, "forecast")
    weights = np.asarray(forecast_weights, float)
    if (weights.ndim != 2 or weights.shape[1] != len(growth) or not 1 <= len(weights) <= MAX_FORECASTS
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1., atol=TOL, rtol=0)
            or not np.allclose(weights[0], 1./len(growth), atol=TOL, rtol=0)):
        raise ValueError("forecast weights must be normalized nonnegative (K,N),1<=K<=5")
    floor = (1 - loss) * peak
    # A strict-not-yet-breached state can sit marginally below F within the
    # separate MDD roundoff convention. Never release a negative headroom.
    budget_floor = max(floor, real - rho * (real - floor))
    hold_support = support @ x / cp
    hold_forecast = growth @ x / cp
    if not np.isfinite(hold_support).all() or not np.isfinite(hold_forecast).all() or np.any(hold_forecast <= 0):
        raise ValueError("Support/forecast wealth exceeds numerical range")
    hold_log = np.log(hold_forecast / real)
    scale = 1. if initial else cushion_scale(real, peak, loss)
    sleeve = TILTS*scale
    proposals = sleeve[:, None]*w + (1-sleeve[:, None])*np.array([0., 0., 1.])
    endpoints, endpoint_fees, endpoint_moved = execute_target(x[None, :], proposals, rate)
    delta = endpoints - x
    margin = (hold_support - budget_floor) / real
    slopes = support @ delta.T / (cp * real)
    intervals = [segment_interval(margin, slopes[:, j]) for j in range(4)]
    post = np.full((CANDIDATES, 3), np.nan)
    q = np.full(CANDIDATES, np.nan); fees = np.full(CANDIDATES, np.nan); moved = np.full(CANDIDATES, np.nan)
    proposal_index = np.full(CANDIDATES, -1, dtype=np.int64)
    post[0], q[0], fees[0], moved[0] = x, 0., 0., 0.
    eligible = np.zeros(CANDIDATES, bool)
    hold_feasible = bool(margin.min() >= -TOL)
    eligible[0] = hold_feasible and not initial
    for j, interval in enumerate(intervals):
        if initial:
            values = (1.,) if j == 0 else ()
        elif interval["feasible"]:
            values = (interval["lo"], (interval["lo"] + interval["hi"]) / 2, interval["hi"])
        else:
            values = ()
        for point, value in enumerate(values):
            index = 1 + 3 * j + point
            q[index], fees[index], moved[index] = value, value * endpoint_fees[j], value * endpoint_moved[j]
            post[index] = x + value * delta[j]; proposal_index[index] = j
            slack = support @ post[index] / cp - budget_floor
            eligible[index] = bool(slack.min() / real >= -TOL)
    finite = np.isfinite(q)
    support_slack = np.full(CANDIDATES, np.nan); support_worst_member = np.full(CANDIDATES, -1, dtype=np.int64)
    member_advantage = np.full((CANDIDATES, len(weights)), np.nan)
    member_log_growth = np.full_like(member_advantage, np.nan)
    member_gross_advantage = np.full_like(member_advantage, np.nan)
    log_fee = np.full(CANDIDATES, np.nan); score = np.full(CANDIDATES, np.nan)
    shortfall = np.full(CANDIDATES, np.nan)
    next_breach = np.full_like(member_advantage, np.nan)
    for index in np.flatnonzero(finite):
        next_real = growth @ post[index] / cp
        ratio_log = np.log(next_real / hold_forecast)
        member_advantage[index] = weights @ ratio_log
        member_log_growth[index] = weights @ np.log(next_real / real)
        log_fee[index] = np.log1p(-fees[index] / wealth)
        new_weights = post[index] / post[index].sum()
        member_gross_advantage[index] = weights @ np.log((growth @ new_weights) / (growth @ (x / wealth)))
        score[index] = member_advantage[index, 0] if forecast_rule == "uniform" else member_advantage[index].min()
        slacks = support @ post[index] / cp - budget_floor
        support_worst_member[index] = int(np.argmin(slacks)); support_slack[index] = float(slacks.min())
        shortfall[index] = max(-support_slack[index], 0.)
        next_peak = np.maximum(peak, next_real)
        predicted_mdd = np.maximum(worst, 1 - next_real / next_peak)
        next_breach[index] = weights @ (predicted_mdd - loss > TOL)
    choice = -1
    available = np.flatnonzero(eligible)
    if len(available):
        maximum = float(score[available].max())
        if not initial and hold_feasible and maximum <= TOL:
            choice = 0
        else:
            band = [int(i) for i in available if score[i] >= maximum-TOL
                    and (initial or not hold_feasible or i == 0 or score[i] > TOL)]
            choice = min(band, key=lambda k: (k != 0, fees[k], proposal_index[k], k))
    selected = choice if choice >= 0 else 0
    reason = ("initial_target_support_infeasible" if initial else "no_feasible_declared_proposal_pending") if choice < 0 else (
        "mandatory_initial_full_target_support_pass" if initial else (
        "risk_budget_protection" if not hold_feasible else ("support_safe_hold" if selected == 0 else "support_safe_growth")))
    action = "hold" if selected == 0 or moved[selected] <= TOL else ("restore" if q[selected] == 1 else "partial")
    return dict(status="selected" if choice >= 0 else "initial_review" if initial else "no_feasible_action",
        selected_index=choice, posttrade_holdings=post[selected].copy(), fee=float(fees[selected]), moved=float(moved[selected]),
        action=action, gate_reason=reason, hold_feasible=hold_feasible, support_feasible=bool(choice >= 0),
        guard_triggered=choice < 0, floor=floor, budget_floor=budget_floor, real_wealth=real,
        cushion_scale=scale, proposal_target_weights=proposals.copy(),
        selected_score=float(score[selected]), pred_log_growth_hold=float(member_log_growth[0, 0]),
        pred_log_growth_selected=float(member_log_growth[selected, 0]),
        uniform_predicted_advantage=float(member_advantage[selected, 0]),
        conditional_predicted_advantage=float(member_advantage[selected, 1]) if len(weights) > 1 else np.nan,
        min_predicted_advantage=float(member_advantage[selected].min()),
        current_log_fee=float(log_fee[selected]), gross_log_advantage_uniform=float(member_gross_advantage[selected, 0]),
        gross_log_advantage_min=float(member_gross_advantage[selected].min()),
        worst_real_shortfall_hold=float(shortfall[0]), worst_real_shortfall_selected=float(shortfall[selected]),
        worst_real_shortfall_improvement=float(shortfall[0] - shortfall[selected]),
        growth_sacrifice=bool(choice >= 0 and score[selected] < -TOL),
        interval_lo=np.array([v["lo"] for v in intervals]), interval_hi=np.array([v["hi"] for v in intervals]),
        interval_feasible=np.array([v["feasible"] for v in intervals]),
        interval_lower_member=np.array([v["lower_member"] for v in intervals], dtype=np.int64),
        interval_upper_member=np.array([v["upper_member"] for v in intervals], dtype=np.int64),
        candidate_posttrade=post, candidate_lambda=q, candidate_fees=fees, candidate_moved=moved,
        candidate_proposal=proposal_index, candidate_feasible=eligible, candidate_score=score,
        candidate_min_support_slack=support_slack, candidate_worst_support_member=support_worst_member,
        candidate_member_advantage=member_advantage, candidate_member_log_growth=member_log_growth,
        candidate_member_gross_advantage=member_gross_advantage, candidate_log_fee=log_fee,
        candidate_predicted_next_breach=next_breach)


def _market(returns, inflation, name):
    r, pi = np.asarray(returns, float), np.asarray(inflation, float)
    if (r.ndim != 3 or r.shape[-1] != 3 or min(r.shape[:2]) < 1 or pi.shape != r.shape[:2]
            or not np.isfinite(r).all() or not np.isfinite(pi).all() or np.any(r <= -1) or np.any(pi <= -1)):
        raise ValueError(name + " must be finite (n,T,3)/(n,T) with positive gross")
    return r, pi


def _previous(r, pi, t):
    return np.column_stack(((1 + r[:, t-1]) / (1 + pi[:, t-1, None]) - 1, pi[:, t-1]))
