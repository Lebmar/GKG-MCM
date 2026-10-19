"""Two fixed current actions followed by the same annual/pending continuation.

This does not simulate a new adaptive learner. History and the original clock
are carried into each branch; no current or terminal wealth is reset.
"""
from __future__ import annotations

import math
import numpy as np

from safeinvest.core import execute_target

TOL = 1e-12


def continue_from_state(holdings, cpi, peak, worst, target, loss, returns,
                        inflation, *, year_index, original_horizon,
                        current_action, ever_breached=False, cost_rate=.0025):
    """Return an inspectable single realized suffix under current action then B."""
    x, w = np.asarray(holdings, float).copy(), np.asarray(target, float).copy()
    r, inf = np.asarray(returns, float), np.asarray(inflation, float)
    if (x.shape != (3,) or w.shape != (3,) or not np.isfinite(x).all()
            or np.any(x < 0) or x.sum() <= 0 or not np.isfinite(w).all()
            or np.any(w < 0) or abs(float(w.sum()) - 1) > TOL):
        raise ValueError("Require finite three-asset holdings and a normalized target")
    if (isinstance(original_horizon, (bool, np.bool_))
            or not isinstance(original_horizon, (int, np.integer)) or original_horizon < 1
            or isinstance(year_index, (bool, np.bool_))
            or not isinstance(year_index, (int, np.integer))
            or not 0 <= year_index < original_horizon):
        raise ValueError("Require a consistent original horizon and zero-based decision year")
    if (r.shape != (original_horizon, 3) or inf.shape != (original_horizon,)
            or not np.isfinite(r).all() or not np.isfinite(inf).all()
            or np.any(r <= -1) or np.any(inf <= -1)):
        raise ValueError("Require a complete joint market path on the original time axis")
    if (isinstance(ever_breached, (bool, np.bool_)) is False or ever_breached
            or not all(np.isfinite(v) for v in (cpi, peak, worst, loss, cost_rate))
            or cpi <= 0 or peak <= 0 or not 0 <= worst <= 1 or not 0 < loss < 1
            or not 0 <= cost_rate < 1 or current_action not in ("hold", "restore")):
        raise ValueError("Only finite active pre-breach states and declared actions are valid")
    current_real = float(x.sum()) / cpi
    current_dd = max(1 - current_real / peak, 0.)
    if (current_real - peak > TOL * max(current_real, peak)
            or current_dd - worst > TOL or worst - loss > TOL):
        raise ValueError("Saved peak, drawdown and active status must be consistent")
    w /= w.sum()
    cp, pk, dd, bad = float(cpi), max(float(peak), current_real), max(float(worst), current_dd), False
    states = {k: [] for k in ("holdings", "cpi", "real_wealth", "real_peak", "running_mdd", "ever_breached")}
    periods = {k: [] for k in ("actions", "fees", "moved", "posttrade_holdings")}
    first = None

    def save():
        for key, value in (("holdings", x.copy()), ("cpi", cp), ("real_wealth", float(x.sum()) / cp),
                           ("real_peak", pk), ("running_mdd", dd), ("ever_breached", bad)):
            states[key].append(value)

    save()
    for t in range(int(year_index), int(original_horizon)):
        restore = current_action == "restore" if t == year_index else not bad
        post, fee, moved = execute_target(x, w, cost_rate) if restore else (x.copy(), 0., 0.)
        for key, value in (("actions", "restore" if restore else "hold"), ("fees", float(fee)),
                           ("moved", float(moved)), ("posttrade_holdings", post.copy())):
            periods[key].append(value)
        x = post * (1 + r[t]); cp *= 1 + inf[t]
        real = float(x.sum()) / cp
        if not np.isfinite(real) or real <= 0 or not np.isfinite(x).all() or cp <= 0:
            raise ValueError("Counterfactual account exceeded the finite numeric range")
        pk = max(pk, real); dd = max(dd, max(1 - real / pk, 0.))
        bad = bad or dd - loss > TOL
        if bad and first is None: first = t + 1
        save()
    result = {k: np.asarray(v) for k, v in {**states, **periods}.items()}
    result.update({"total_log_growth": math.log((float(x.sum()) / cp) / current_real),
                   "terminal_real": float(x.sum()) / cp, "history_mdd": dd,
                   "breach": bool(bad), "first_crossing_original_year": first,
                   "year_index": int(year_index), "original_horizon": int(original_horizon),
                   "nominal_fees": float(np.sum(result["fees"])),
                   "nominal_moved": float(np.sum(result["moved"]))})
    return result


def sign_with_band(value):
    if not np.isfinite(value): raise ValueError("A comparison score must be finite")
    return 1 if value > TOL else -1 if value < -TOL else 0


def compare_actions(predicted, realized, *, annual_price, original_horizon, chosen_action):
    """Descriptive forecast/single-suffix contrast, never a probability test."""
    if (not np.isfinite(annual_price) or annual_price < 0
            or isinstance(original_horizon, (bool, np.bool_))
            or not isinstance(original_horizon, (int, np.integer)) or original_horizon < 1
            or chosen_action not in ("hold", "restore")):
        raise ValueError("This finite-price diagnostic needs the original finite price/action")
    for action in ("hold", "restore"):
        if (not np.isfinite(predicted[action]["log"])
                or not np.isfinite(predicted[action]["bad_fraction"])
                or not 0 <= predicted[action]["bad_fraction"] <= 1
                or not np.isfinite(realized[action]["total_log_growth"])
                or not isinstance(realized[action]["breach"], (bool, np.bool_))):
            raise ValueError("Require finite predictions, fractions and boolean realized events")
    delta_q = predicted["restore"]["log"] - predicted["hold"]["log"]
    delta_p = predicted["restore"]["bad_fraction"] - predicted["hold"]["bad_fraction"]
    delta_g = realized["restore"]["total_log_growth"] - realized["hold"]["total_log_growth"]
    delta_b = int(realized["restore"]["breach"]) - int(realized["hold"]["breach"])
    predicted_score = delta_q - original_horizon * annual_price * delta_p
    suffix_score = delta_g - original_horizon * annual_price * delta_b
    predicted_action = "restore" if predicted_score > TOL else "hold"
    suffix_better_action = "restore" if suffix_score > TOL else "hold"
    values = {a: realized[a]["total_log_growth"] - original_horizon * annual_price * int(realized[a]["breach"])
              for a in ("hold", "restore")}
    shortfall = max(values.values()) - values[chosen_action]
    return {"predicted_delta_log": float(delta_q), "predicted_delta_bad_fraction": float(delta_p),
            "realized_suffix_delta_log": float(delta_g), "realized_suffix_delta_event": delta_b,
            "delta_log_residual": float(delta_g - delta_q),
            "delta_event_residual": float(delta_b - delta_p),
            "predicted_score": float(predicted_score), "realized_suffix_score": float(suffix_score),
            "score_residual": float(suffix_score - predicted_score),
            "predicted_growth_sign": sign_with_band(delta_q), "realized_suffix_growth_sign": sign_with_band(delta_g),
            "growth_sign_agreement": sign_with_band(delta_q) == sign_with_band(delta_g),
            "predicted_action": predicted_action, "realized_suffix_better_current_action": suffix_better_action,
            "score_action_agreement": predicted_action == suffix_better_action,
            "two_branch_realized_score_shortfall": max(float(shortfall), 0.)}
