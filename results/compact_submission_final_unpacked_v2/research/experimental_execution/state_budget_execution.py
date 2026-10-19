"""One-year paired-growth policy with a state-dependent joint-support filter.

This new research engine leaves every earlier controller unchanged. Safety is
conditional on a declared next-year joint support, not a probability/risk
certificate or recursively feasible full-horizon guarantee. Exact post-dollar
segments pay the original transaction fee once. The finite action menu uses
only each segment's feasible low/mid/high points, never a claimed continuous
optimum. The base target stays fixed; cash tilts are temporary proposals.

At inception only the full original target is allowed. A failed support test
returns Review with arm=None before actual values are validated/used. Once
funded, no feasible declared proposal (not proof of physical impossibility)
or an actual breach causes absorbing pending hold,
retaining physical finance, original CPI/peak/MDD and original horizon.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target
from .forecast_ensemble import build_forecast_ensemble

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
    proposals = TILTS[:, None] * w + (1 - TILTS[:, None]) * np.array([0., 0., 1.])
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


def simulate_state_budget(actual_returns, actual_inflation, training_returns, training_inflation, target, *,
                          L, horizon, cost_rate=.0025, release_fraction=.5, forecast_rule="uniform",
                          training_ids, actual_ids=None, support_mode="calendar"):
    """Return a funded standard arm or an unfunded Review wrapper, never both.

    Candidate execution uses only original state and frozen training arrays.
    Actual current-year returns are advanced only after the action is fixed.
    Array validation after admission does not feed realized future outcomes
    into the rule. Pending holds preserve all subsequent financial endpoints.
    """
    train, train_pi = _market(training_returns, training_inflation, "training")
    raw_r, raw_pi = np.asarray(actual_returns), np.asarray(actual_inflation)
    if raw_r.ndim != 3 or raw_r.shape[-1] != 3 or raw_r.shape[0] < 1 or raw_pi.shape != raw_r.shape[:2]:
        raise ValueError("actual shapes must be nonempty(n,T,3)/(n,T)")
    n = len(raw_r)
    if (isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer))
            or not 1 <= horizon <= min(30, raw_r.shape[1], train.shape[1])):
        raise ValueError("original horizon must be a positive integer within both arrays")
    h = int(horizon); w = _target(target)
    if support_mode not in ("calendar", "pooled"):
        raise ValueError("support_mode must be calendar or separately labelled pooled stress")
    ids = np.asarray(actual_ids) if actual_ids is not None else np.arange(1, n + 1)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != n:
        raise ValueError("actual_ids are unique integer labels, never action inputs")
    initial_ensemble = build_forecast_ensemble(training_ids, period_index=0, n_actual=1)
    if initial_ensemble.weights.shape[-1] != len(train):
        raise ValueError("training IDs must align exactly with training members")
    if h > 1 and len(train) < 3:
        raise ValueError("Positive-period ensemble needs at least three training members")
    real_gross = (1 + train[:, :h]) / (1 + train_pi[:, :h, None])
    pooled = real_gross.reshape(-1, 3)
    support = real_gross[:, 0] if support_mode == "calendar" else pooled
    initial = decide_state_action([0., 0., 100.], 1., 100., 0., support, real_gross[:, 0], initial_ensemble.weights[0],
        target=w, L=L, cost_rate=cost_rate, release_fraction=release_fraction, forecast_rule=forecast_rule, initial=True)
    metadata = dict(original_horizon=h, L=float(L), cost_rate=float(cost_rate), release_fraction=float(release_fraction),
        forecast_rule=forecast_rule, support_mode=support_mode, certificate_scope="one_step_declared_support_only",
        full_path_or_future_probability_guarantee=False, base_target_persistent=True, temporary_cash_tilt_proposals=TILTS.tolist(),
        initial_only_full_original_target=True, initial_decision_uses_actual_market=False,
        training_ids=np.asarray(training_ids).tolist(), actual_ids=ids.tolist(), forecast_labels=list(FORECAST_LABELS),
        growth_objective="uniform mean or ensemble minimum of paired candidate/hold one-year log wealth ratios",
        finite_segment_points=["lo", "mid", "hi"], support_numerical_relative_tolerance=TOL,
        no_feasible_scope="declared hold and four post-dollar proposal segments only; not all physical portfolios",
        pending_at_decision_includes_current_guard_trigger=True,
        budget_floor_clamped_to_original_client_floor=True,
        tie="global maximum score band1e-12, hold, actualfee, less cash tilt, proposal/point order")
    if initial["status"] != "selected":
        return dict(status="preinvestment_review", financial_available=False, arm=None, original_n=n,
            preinvestment_review_n=n, initial_witness=initial, metadata=metadata)
    market, actual_pi = _market(raw_r, raw_pi, "actual")
    periods, endpoints = (n, h), (n, h + 1)
    float_period = ("fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction", "pretrade_tv_drift", "end_year_tv_drift",
        "floor", "budget_floor", "selected_score", "pred_log_growth_hold", "pred_log_growth_selected", "uniform_predicted_advantage",
        "conditional_predicted_advantage", "min_predicted_advantage", "current_log_fee", "gross_log_advantage_uniform",
        "gross_log_advantage_min", "worst_real_shortfall_hold", "worst_real_shortfall_selected", "worst_real_shortfall_improvement")
    arm = {name: np.full(periods, np.nan) for name in float_period}
    for name in ("nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd"):
        arm[name] = np.zeros(endpoints)
    for name in ("target_execution", "pending_at_decision", "guard_pending_at_decision", "was_pending_input", "was_guard_pending_input", "automatic_service", "support_feasible", "hold_feasible", "growth_sacrifice", "trades", "guard_triggered", "gate_evaluated"):
        arm[name] = np.zeros(periods, bool)
    arm.update(holdings=np.zeros(endpoints + (3,)), weights=np.zeros(endpoints + (3,)), posttrade_holdings=np.zeros(periods + (3,)),
        cpi=np.ones(endpoints), ever_breached=np.zeros(endpoints, bool), guard_pending=np.zeros(endpoints, bool),
        actions=np.full(periods, "hold", dtype="U8"), gate_reason=np.full(periods, "", dtype="U64"),
        first_crossing=np.full(n, -1, dtype=np.int64), first_guard_trigger=np.full(n, -1, dtype=np.int64),
        selected_index=np.full(periods, -1, dtype=np.int64), selected_proposal=np.full(periods, -1, dtype=np.int64),
        selected_lambda=np.full(periods, np.nan), forecast_n=np.zeros(periods, dtype=np.int64), forecast_member_count=np.zeros(periods, dtype=np.int64),
        actual_previous_state=np.full(periods + (4,), np.nan), training_previous_state=np.full((h, len(train), 4), np.nan),
        training_real_gross=np.transpose(real_gross, (1, 0, 2)).copy(), realized_real_gross=np.full(periods + (3,), np.nan))
    for name in ("drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger"):
        arm[name] = np.full(n, np.nan)
    for name in ("interval_lo", "interval_hi"):
        arm[name] = np.full(periods + (4,), np.nan)
    arm["interval_feasible"] = np.zeros(periods + (4,), bool)
    for name in ("interval_lower_member", "interval_upper_member"):
        arm[name] = np.full(periods + (4,), -1, dtype=np.int64)
    for name in ("candidate_lambda", "candidate_fees", "candidate_moved", "candidate_score", "candidate_min_support_slack", "candidate_log_fee"):
        arm[name] = np.full(periods + (CANDIDATES,), np.nan)
    arm["candidate_feasible"] = np.zeros(periods + (CANDIDATES,), bool)
    for name in ("candidate_proposal", "candidate_worst_support_member"):
        arm[name] = np.full(periods + (CANDIDATES,), -1, dtype=np.int64)
    arm["candidate_posttrade"] = np.full(periods + (CANDIDATES, 3), np.nan)
    for name in ("candidate_member_advantage", "candidate_member_log_growth", "candidate_member_gross_advantage", "candidate_predicted_next_breach"):
        arm[name] = np.full(periods + (CANDIDATES, MAX_FORECASTS), np.nan)
    arm["holdings"][:, 0, 2] = 100.; arm["weights"][:, 0, 2] = 1.
    for name in ("nominal_wealth", "real_wealth", "real_peak"):
        arm[name][:, 0] = 100.
    decisions = tuple(name for name in initial if name in arm and name not in ("real_wealth", "posttrade_holdings", "fees", "moved", "selected_index"))
    for t in range(h):
        x = arm["holdings"][:, t]; nominal = x.sum(axis=1)
        pending = arm["guard_pending"][:, t] | arm["ever_breached"][:, t]
        arm["was_pending_input"][:, t] = pending
        arm["was_guard_pending_input"][:, t] = arm["guard_pending"][:, t]
        arm["pending_at_decision"][:, t] = pending; arm["guard_pending_at_decision"][:, t] = arm["guard_pending"][:, t]
        arm["gate_reason"][:, t] = np.where(arm["guard_pending"][:, t], "guard_pending_hold", "already_breached_pending_hold")
        arm["pretrade_wealth"][:, t] = nominal; arm["fees"][:, t] = 0.; arm["moved"][:, t] = 0.
        post = x.copy(); active = np.flatnonzero(~pending); new_guard = np.zeros(n, bool)
        if t:
            training_previous = _previous(train, train_pi, t); actual_previous = _previous(market, actual_pi, t)
            arm["training_previous_state"][t] = training_previous; arm["actual_previous_state"][:, t] = actual_previous
        else:
            training_previous = actual_previous = None
        if len(active):
            ensemble = build_forecast_ensemble(training_ids, period_index=t, n_actual=len(active),
                training_previous_state=training_previous, actual_previous_state=None if t == 0 else actual_previous[active])
            for order, i in enumerate(active):
                result = initial if t == 0 else decide_state_action(x[i], arm["cpi"][i, t], arm["real_peak"][i, t],
                    arm["running_mdd"][i, t], real_gross[:, t] if support_mode == "calendar" else pooled,
                    real_gross[:, t], ensemble.weights[order], target=w, L=L, cost_rate=cost_rate,
                    release_fraction=release_fraction, forecast_rule=forecast_rule)
                for name in decisions:
                    value = result[name]
                    if name.startswith("candidate_member") or name == "candidate_predicted_next_breach":
                        arm[name][i, t, :, :len(ensemble.labels)] = value
                    else:
                        arm[name][i, t] = value
                post[i] = result["posttrade_holdings"]; arm["fees"][i, t] = result["fee"]; arm["moved"][i, t] = result["moved"]
                arm["gate_evaluated"][i, t] = True
                arm["actions"][i, t] = result["action"]; arm["selected_index"][i, t] = result["selected_index"]
                arm["gate_reason"][i, t] = result["gate_reason"]; arm["forecast_n"][i, t] = len(train)
                arm["forecast_member_count"][i, t] = len(ensemble.labels)
                selected = result["selected_index"]
                if selected >= 0:
                    arm["selected_proposal"][i, t] = result["candidate_proposal"][selected]
                    arm["selected_lambda"][i, t] = result["candidate_lambda"][selected]
                    arm["target_execution"][i, t] = selected > 0
                if result["guard_triggered"]:
                    new_guard[i] = True; arm["first_guard_trigger"][i] = t + 1
        arm["pending_at_decision"][:, t] |= new_guard
        arm["guard_pending_at_decision"][:, t] |= new_guard
        arm["automatic_service"][:, t] = ~(pending | new_guard)
        arm["posttrade_holdings"][:, t] = post
        arm["fee_fraction"][:, t] = arm["fees"][:, t] / nominal
        arm["moved_fraction"][:, t] = arm["moved"][:, t] / nominal
        arm["trades"][:, t] = arm["moved"][:, t] > TOL
        arm["pretrade_tv_drift"][:, t] = .5 * np.abs(arm["weights"][:, t] - w).sum(axis=1)
        # Post-decision trace only: no current actual realization enters choice.
        arm["realized_real_gross"][:, t] = (1 + market[:, t]) / (1 + actual_pi[:, t, None])
        after = post * (1 + market[:, t]); cp = arm["cpi"][:, t] * (1 + actual_pi[:, t])
        end_nominal = after.sum(axis=1); real = end_nominal / cp
        if not np.isfinite(after).all() or not np.isfinite(cp).all() or not np.isfinite(real).all() or np.any(cp <= 0) or np.any(real <= 0):
            raise ValueError("Physical account exceeds numerical range")
        peak = np.maximum(arm["real_peak"][:, t], real); dd = np.maximum(1 - real / peak, 0)
        mdd = np.maximum(arm["running_mdd"][:, t], dd); ever = arm["ever_breached"][:, t] | (mdd - L > TOL)
        first = (arm["first_crossing"] < 0) & ever
        arm["first_crossing"][first] = t + 1
        for name, values in (("drawdown_at_trigger", mdd), ("peak_at_trigger", peak), ("real_wealth_at_trigger", real)):
            arm[name][first] = values[first]
        for name, values in (("holdings", after), ("weights", after/end_nominal[:, None]), ("nominal_wealth", end_nominal),
                ("real_wealth", real), ("real_peak", peak), ("current_drawdown", dd), ("running_mdd", mdd), ("ever_breached", ever), ("cpi", cp)):
            arm[name][:, t+1] = values
        arm["guard_pending"][:, t+1] = arm["guard_pending"][:, t] | new_guard
        arm["end_year_tv_drift"][:, t] = .5 * np.abs(arm["weights"][:, t+1] - w).sum(axis=1)
    triggered = arm["first_crossing"] >= 0; endpoint = np.arange(h+1)[None, :]
    recovery = triggered[:, None] & (endpoint > arm["first_crossing"][:, None]) & (arm["real_wealth"] >= arm["peak_at_trigger"][:, None])
    recovered = recovery.any(axis=1); recovery_endpoint = np.where(recovered, recovery.argmax(axis=1), -1)
    arm.update(triggered=triggered, recovered=recovered, recovery_endpoint=recovery_endpoint,
        recovery_time=np.where(recovered, recovery_endpoint-arm["first_crossing"], -1), right_censored=triggered & ~recovered,
        available_followup=np.where(triggered, h-arm["first_crossing"], -1),
        observed_followup=np.where(recovered, recovery_endpoint-arm["first_crossing"], np.where(triggered, h-arm["first_crossing"], -1)),
        added_severity=np.where(triggered, np.maximum(arm["running_mdd"][:, -1]-arm["drawdown_at_trigger"], 0), np.nan),
        terminal_excess=np.where(arm["ever_breached"][:, -1], arm["running_mdd"][:, -1]-L, 0),
        end_weights=arm["weights"][:, 1:].copy(), targets=w.copy(), loss_tolerances=np.array([L]))
    for value in arm.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return dict(status="funded", financial_available=True, arm=arm, original_n=n, preinvestment_review_n=0,
        initial_witness=initial, metadata=metadata)
