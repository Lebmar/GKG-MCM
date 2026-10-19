"""One forced action followed by the unchanged, replanned empirical policy.

This is a fixed-market controlled diagnostic, not a candidate policy family.
Original time, horizon, CPI, historical peak and absorbing breach are retained.
"""
from __future__ import annotations

import math
import numpy as np

from safeinvest.core import execute_target
from .fan_rollout import _market, _parameters, _state, evaluate_actions, NUMERIC_TOL
from .risk_priced_fan_rollout import priced_choices


def replay_intervention(actual_returns, actual_inflation, training_returns,
                        training_inflation, target, eta, *, holdings, cpi,
                        real_peak, running_mdd, year_index, forced_first_action,
                        horizon=30, loss=.4, cost_rate=.0025):
    """Start at one active original state; force only its current action.

    Every subsequent active decision recomputes both current-action-then-B
    values on the original training fan. Actual future returns never enter Q.
    First-step policy values are also recorded, although the action is forced.
    Arrays start at ``year_index`` and end at the original horizon.
    """
    actual, inflation = _market(np.asarray(actual_returns)[None, ...],
                               np.asarray(actual_inflation)[None, ...], "actual")
    train, train_inflation = _market(training_returns, training_inflation, "fan")
    w, L, rate, H = _parameters(target, loss, cost_rate, horizon,
                               min(actual.shape[1], train.shape[1]))
    if (isinstance(year_index, (bool, np.bool_))
            or not isinstance(year_index, (int, np.integer))
            or not 1 <= year_index < H):
        raise ValueError("Intervention must be after mandatory inception, before original H")
    if forced_first_action not in ("hold", "restore"):
        raise ValueError("Exactly one declared current action is required")
    if eta is not None and (isinstance(eta, (bool, np.bool_))
                            or not np.isfinite(eta) or eta < 0):
        raise ValueError("Finite nonnegative fixed eta or separate risk_first None")
    x, cp, pk, worst, initial_real = _state(holdings, cpi, real_peak,
                                           running_mdd, False, L)
    if len(x) != 1:
        raise ValueError("One original account state per intervention")
    x = x[0].copy(); cp = float(cp[0]); pk = float(pk[0]); worst = float(worst[0])
    initial_real = float(initial_real[0]); bad = False
    states = {k: [] for k in ("holdings", "weights", "nominal_wealth", "real_wealth",
                              "cpi", "real_peak", "current_drawdown", "running_mdd", "ever_breached")}
    periods = {k: [] for k in ("original_year", "actions", "policy_actions", "forced",
                               "pending_at_decision", "posttrade_holdings", "fees", "moved",
                               "qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count",
                               "forecast_n", "predicted_score", "gate_reason")}

    def save_state():
        V = float(x.sum()); real = V / cp
        if not np.isfinite([V, real, cp, pk, worst]).all() or real <= 0:
            raise ValueError("Nonfinite or nonpositive physical state")
        for k, value in (("holdings", x.copy()), ("weights", x / V),
                         ("nominal_wealth", V), ("real_wealth", real), ("cpi", cp),
                         ("real_peak", pk), ("current_drawdown", max(1-real/pk, 0.)),
                         ("running_mdd", worst), ("ever_breached", bad)):
            states[k].append(value)

    save_state(); first_crossing = None
    for t in range(int(year_index), H):
        forced = t == year_index
        if bad:
            qh = qr = score = math.nan; kh = kr = fan_n = -1
            policy_action = action = "hold"; reason = "already_breached_pending_hold"
        else:
            values = evaluate_actions(x, cp, pk, worst, False, train, train_inflation,
                year_index=t, target=w, L=L, cost_rate=rate, horizon=H)
            qh, qr = (float(values[a]["mean_log_growth"][0]) for a in ("hold", "restore"))
            kh, kr = (int(values[a]["breach_count"][0]) for a in ("hold", "restore"))
            fan_n = int(values["fan_n"])
            chosen, numeric_score, reasons = priced_choices(qh, qr, kh, kr, fan_n,
                np.inf if eta is None else float(eta), H)
            policy_action = "restore" if bool(chosen) else "hold"
            score = float(numeric_score); reason = str(reasons.item())
            action = forced_first_action if forced else policy_action
            if forced:
                reason = "diagnostic_forced_first_action"
        if forced and bad:
            raise AssertionError("Intervention is restricted to active original states")
        post, fee, moved = execute_target(x, w, rate) if action == "restore" else (x.copy(), 0., 0.)
        for k, value in (("original_year", t+1), ("actions", action),
                         ("policy_actions", policy_action), ("forced", forced),
                         ("pending_at_decision", bad), ("posttrade_holdings", post.copy()),
                         ("fees", float(fee)), ("moved", float(moved)), ("qg_hold", qh),
                         ("qg_restore", qr), ("qbad_hold_count", kh), ("qbad_restore_count", kr),
                         ("forecast_n", fan_n), ("predicted_score", score), ("gate_reason", reason)):
            periods[k].append(value)
        # The current actual return is consumed only after the action and fee.
        x = post * (1 + actual[0, t]); cp *= 1 + inflation[0, t]
        real = float(x.sum()) / cp; pk = max(pk, real)
        worst = max(worst, max(1-real/pk, 0.))
        newly_bad = worst - L > NUMERIC_TOL
        if not bad and newly_bad:
            first_crossing = t + 1
        bad = bool(bad or newly_bad)
        save_state()
    out = {k: np.asarray(v) for k, v in {**states, **periods}.items()}
    out.update(original_horizon=H, original_year_index=int(year_index),
        first_crossing_original_year=first_crossing,
        terminal_real=float(states["real_wealth"][-1]), terminal_mdd=float(worst),
        terminal_breach=bool(bad), total_remaining_log_growth=math.log(states["real_wealth"][-1]/initial_real),
        nominal_fees=float(sum(periods["fees"])), nominal_moved=float(sum(periods["moved"])),
        suffix_restore_decisions_including_forced=int(sum(v == "restore" for v in periods["actions"])),
        forced_action=forced_first_action, eta_technical=eta)
    for value in out.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return out
