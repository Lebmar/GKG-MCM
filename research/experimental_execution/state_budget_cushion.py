"""Exploratory V3: cushion-scaled primary proposals with frozen V2b fallback.

This separate module is proposed after V2/V2b development. Only the primary
four destinations are state-scaled. The frozen V2b9point mitigation helper,
forecast family, initial Review and account loop remain unchanged. No old
module global is assigned or monkeypatched; all original artifacts survive.
"""
from __future__ import annotations

import numpy as np

from .forecast_ensemble import build_forecast_ensemble
from .state_budget_cushion_primary import (
    TOL, TILTS, CANDIDATES, MAX_FORECASTS, FORECAST_LABELS,
    _scalar, _target, _real_support, _market, _previous,
    decide_state_action as primary_decide_state_action,
)
from .state_budget_mitigation import mitigate_support_failure

MITIGATION_CANDIDATES = 9
MITIGATION_Q = (.5, 1.)
BASELINE_SOURCE_SHA256 = "8480dc1c2d54175a9fafed2471991a0ea8fc6a4519a24140044d056dbafafb1f"

def _blank_mitigation(members):
    result = dict(mitigation_evaluated=False, mitigation_trade=False, mitigation_selected_index=-1,
        mitigation_selected_proposal=-1, mitigation_selected_q=np.nan,
        mitigation_worst_next_real_hold=np.nan, mitigation_worst_next_real_selected=np.nan,
        mitigation_worst_next_real_improvement=np.nan,
        actual_decision_source="primary", primary_failed_hold_score=np.nan,
        primary_failed_hold_pred_log_growth=np.nan, primary_failed_hold_fee=np.nan,
        primary_failed_hold_shortfall=np.nan, primary_failed_hold_posttrade=np.full(3, np.nan))
    for name in ("q", "fees", "moved", "worst_next_real", "shortfall"):
        result["mitigation_candidate_"+name] = np.full(MITIGATION_CANDIDATES, np.nan)
    for name in ("proposal", "worst_support_member"):
        result["mitigation_candidate_"+name] = np.full(MITIGATION_CANDIDATES, -1, dtype=np.int64)
    result["mitigation_candidate_posttrade"] = np.full((MITIGATION_CANDIDATES, 3), np.nan)
    for name in ("advantage", "log_growth", "gross_advantage", "predicted_next_breach"):
        result["mitigation_candidate_member_"+name] = np.full((MITIGATION_CANDIDATES, members), np.nan)
    return result


def _add_mitigation_fields(arm, periods):
    for key, value in _blank_mitigation(MAX_FORECASTS).items():
        if isinstance(value, np.ndarray):
            arm[key] = np.broadcast_to(value, periods+value.shape).copy()
        else:
            arm[key] = np.full(periods, value)
    arm["actual_decision_source"] = np.full(periods, "pending", dtype="U10")
    arm["primary_gate_reason"] = np.full(periods, "", dtype="U64")
    arm["cushion_scale"] = np.full(periods, np.nan)
    arm["proposal_target_weights"] = np.full(periods+(4, 3), np.nan)
    for key in ("primary_selected_score", "primary_pred_log_growth_selected"):
        arm[key] = np.full(periods, np.nan)


def decide_state_action(holdings, cpi, real_peak, running_mdd, support_real_gross,
                        forecast_real_gross, forecast_weights, *, target, L, cost_rate=.0025,
                        release_fraction=.5, forecast_rule="uniform", initial=False):
    """Cushion-scaled primary first, with the frozen best-effort fallback.

    All13 primary candidate witnesses stay intact. Physical-action scalar
    predictions describe mitigation when used; primary_* fields preserve the
    failed primary hold comparison. Primary selected_index remains-1 and
    support_feasible remainsFalse at mitigation, even if its shortfall shrinks.
    """
    result = primary_decide_state_action(holdings, cpi, real_peak, running_mdd, support_real_gross,
        forecast_real_gross, forecast_weights, target=target, L=L, cost_rate=cost_rate,
        release_fraction=release_fraction, forecast_rule=forecast_rule, initial=initial)
    result.update(_blank_mitigation(len(forecast_weights)))
    result.update(primary_gate_reason=result["gate_reason"], primary_selected_score=result["selected_score"],
        primary_pred_log_growth_selected=result["pred_log_growth_selected"])
    if initial or result["status"] != "no_feasible_action":
        return result
    result.update(primary_failed_hold_score=result["selected_score"],
        primary_failed_hold_pred_log_growth=result["pred_log_growth_selected"], primary_failed_hold_fee=result["fee"],
        primary_failed_hold_shortfall=result["worst_real_shortfall_selected"],
        primary_failed_hold_posttrade=result["posttrade_holdings"].copy(), actual_decision_source="mitigation")
    mitigation = mitigate_support_failure(holdings, cpi, support_real_gross,
        target=target, cost_rate=cost_rate, budget_floor=result["budget_floor"],
        forecast_real_gross=forecast_real_gross, forecast_weights=forecast_weights,
        real_peak=real_peak, running_mdd=running_mdd, L=L)
    x = np.asarray(holdings, float)
    weights = np.asarray(forecast_weights, float)
    member_advantage = mitigation["candidate_member_advantage"]
    member_growth = mitigation["candidate_member_log_growth"]
    member_gross = mitigation["candidate_member_gross_advantage"]
    next_breach = mitigation["candidate_member_predicted_next_breach"]
    selected = mitigation["selected_index"]
    selected_score = member_advantage[selected, 0] if forecast_rule == "uniform" else member_advantage[selected].min()
    result.update(mitigation_evaluated=True, mitigation_trade=mitigation["trade"],
        mitigation_selected_index=selected, mitigation_selected_proposal=mitigation["selected_proposal"],
        mitigation_selected_q=mitigation["selected_q"],
        mitigation_worst_next_real_hold=mitigation["worst_next_real_hold"],
        mitigation_worst_next_real_selected=mitigation["worst_next_real_selected"],
        mitigation_worst_next_real_improvement=mitigation["worst_next_real_improvement"])
    for name in ("posttrade", "q", "fees", "moved", "proposal", "worst_next_real", "worst_support_member", "shortfall"):
        result["mitigation_candidate_"+name] = mitigation["candidate_"+name]
    for name, value in (("advantage", member_advantage), ("log_growth", member_growth),
                        ("gross_advantage", member_gross), ("predicted_next_breach", next_breach)):
        result["mitigation_candidate_member_"+name] = value
    result.update(posttrade_holdings=mitigation["posttrade_holdings"], fee=mitigation["fee"], moved=mitigation["moved"],
        action="mitigate" if mitigation["trade"] else "hold", gate_reason="risk_mitigation_then_review",
        selected_score=float(selected_score), pred_log_growth_selected=float(member_growth[selected, 0]),
        uniform_predicted_advantage=float(member_advantage[selected, 0]),
        conditional_predicted_advantage=float(member_advantage[selected, 1]) if len(weights)>1 else np.nan,
        min_predicted_advantage=float(member_advantage[selected].min()),
        current_log_fee=float(np.log1p(-mitigation["fee"]/x.sum())),
        gross_log_advantage_uniform=float(member_gross[selected, 0]), gross_log_advantage_min=float(member_gross[selected].min()),
        worst_real_shortfall_selected=mitigation["worst_real_shortfall_selected"],
        worst_real_shortfall_improvement=mitigation["worst_real_shortfall_improvement"],
        growth_sacrifice=bool(selected_score < -TOL))
    return result


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
    metadata.update(research_version="V3_cushion_primary_with_frozen_V2b_mitigation",
        proposed_after_adverse_V2_development_results=True, baseline_source_sha256=BASELINE_SOURCE_SHA256,
        mitigation_menu="hold plus four same cash-tilt endpoints at q=.5/1", mitigation_floor_restoration_guarantee=False,
        mitigation_selection="maximum supported worst next realwealth, net exact fees; strict improvement over hold",
        mitigation_tie="globalmax worstwealth band1e-12*currentreal, hold, lower fee, proposal order, q order",
        mitigation_forecast_scores_are_annotations_not_ranking=True,
        unchanged_six_tuple_family_forecasts_and_selector=True,
        proposed_after_V2_and_V2b_development=True, primary_cushion_scaled=True,
        primary_proposal_formula="a*m*w+(1-a*m)*e_bill; a=1/.75/.5/.25",
        cushion_zero_loss_convention="L0 at peak:m1; tolerated R<peak:m0",
        primary_cushion_is_hold_hard_cap=False,
        primary_cushion_is_a_fitted_parameter=False,
        proposal_replacement_is_not_a_superset_of_previous_family=True,
        mitigation_cushion_scaled=False,
        frozen_mitigation_source_sha256="fe3904062f92a42f7fc37d664b53ee3ec6d336b2498448c1a707cb2edfda89ff")
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
    _add_mitigation_fields(arm, periods)
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
                    if name.startswith(("candidate_member", "mitigation_candidate_member")) or name == "candidate_predicted_next_breach":
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
                if result["mitigation_evaluated"]:
                    arm["target_execution"][i, t] = result["mitigation_selected_index"] > 0
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


simulate_state_budget_cushion = simulate_state_budget
