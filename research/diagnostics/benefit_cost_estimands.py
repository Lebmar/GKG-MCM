"""Algebra and two-period synthetic witnesses for benefit/cost definitions.

No market file, learned model, scenario loader, or market-policy runner exists
in this module. Gross factors are supplied explicitly by the synthetic tests.
The fee operation is the unchanged contest-accounting implementation.
"""
from __future__ import annotations

import math
import numpy as np

from safeinvest.core import execute_target


def _state(holdings, target):
    x, w = np.asarray(holdings, float), np.asarray(target, float)
    if (x.shape != (3,) or w.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(x < 0) or np.any(w < 0) or x.sum() <= 0 or not np.isclose(w.sum(), 1., atol=1e-13, rtol=0)):
        raise ValueError("Finite positive account and three-asset simplex target required")
    return x, w / w.sum()


def one_step_parts(holdings, target, gross, *, cost_rate=.0025, inflation_gross=1.):
    """Exact one-step terminal-log comparison; fee charged exactly once."""
    x, w = _state(holdings, target); g = np.asarray(gross, float)
    if g.shape != (3,) or not np.isfinite(g).all() or np.any(g <= 0) or not np.isfinite(inflation_gross) or inflation_gross <= 0:
        raise ValueError("Positive joint gross factors and inflation gross required")
    post, fee, moved = execute_target(x, w, cost_rate)
    W = float(x.sum()); rho = float(fee) / W; u = x / W
    benefit = math.log(float(w @ g) / float(u @ g))
    log_cost = -math.log1p(-rho)
    hold_real = float(x @ g) / inflation_gross
    restore_real = float(post @ g) / inflation_gross
    direct = math.log(restore_real / hold_real)
    return dict(wealth=W, fee=float(fee), moved=float(moved), fee_fraction=rho,
                gross_log_benefit=benefit, current_log_cost=log_cost,
                net_log_advantage=benefit - log_cost, direct_net_log_advantage=direct,
                hold_real=hold_real, restore_real=restore_real)


def hold_to_horizon_parts(holdings, target, joint_gross_path, *, cost_rate=.0025, inflation_gross_path=None):
    """Both branches make no further trades; quantities, not weights, persist."""
    g = np.asarray(joint_gross_path, float)
    if g.ndim != 2 or g.shape[1] != 3 or len(g) < 1 or not np.isfinite(g).all() or np.any(g <= 0):
        raise ValueError("A nonempty joint positive gross path required")
    pi = np.ones(len(g)) if inflation_gross_path is None else np.asarray(inflation_gross_path, float)
    if pi.shape != (len(g),) or not np.isfinite(pi).all() or np.any(pi <= 0):
        raise ValueError("Common positive inflation gross path required")
    return one_step_parts(holdings, target, np.prod(g, axis=0), cost_rate=cost_rate, inflation_gross=float(np.prod(pi)))


def two_period_annual_witness(holdings, target, joint_gross_path, *, initial_action, charge_current_fee=True,
                              loss=.4, cost_rate=.0025, real_peak=100., current_cpi=1.,
                              running_mdd=0., inflation_gross_path=(1., 1.), pending_enabled=True):
    """Exactly two synthetic periods: specified action now, annual restore later.

    All branches preserve the same supplied inherited peak and CPI. Turning
    off only the current restore fee is a decomposition counterfactual; future
    transfers still pay the original rate. This is not a fitted/deployed policy.
    """
    x, w = _state(holdings, target); g = np.asarray(joint_gross_path, float)
    pi = np.asarray(inflation_gross_path, float)
    if (g.shape != (2, 3) or pi.shape != (2,) or not np.isfinite(g).all() or not np.isfinite(pi).all()
            or np.any(g <= 0) or np.any(pi <= 0) or initial_action not in ("hold", "restore")):
        raise ValueError("Exactly two explicit positive joint synthetic periods required")
    if initial_action == "hold" and not charge_current_fee:
        raise ValueError("Hold incurs no current transfer fee to waive")
    W = float(x.sum()); cp = float(current_cpi); peak = float(real_peak); worst = float(running_mdd)
    if cp <= 0 or peak < W / cp or worst < max(1 - W / cp / peak, 0.) or worst > loss:
        raise ValueError("Consistent unbreached common inherited state required")
    bad = False; wealth = [W / cp]; peaks = [peak]; actions = []; fees = []; breaches = [False]; drifts = []
    for t in range(2):
        action = initial_action if t == 0 else "hold" if bad and pending_enabled else "restore"
        drift = .5 * float(np.abs(x / x.sum() - w).sum()); drifts.append(drift)
        if action == "restore":
            if t == 0 and not charge_current_fee: post, fee = x.sum() * w, 0.
            else: post, fee, _ = execute_target(x, w, cost_rate)
        else: post, fee = x.copy(), 0.
        actions.append(action); fees.append(float(fee))
        x = post * g[t]; cp *= pi[t]; real = float(x.sum()) / cp
        peak = max(peak, real); worst = max(worst, 1 - real / peak)
        bad = bool(bad or worst - loss > 1e-12)
        wealth.append(real); peaks.append(peak); breaches.append(bad)
    return dict(terminal_real=wealth[-1], real_wealth=wealth, real_peak=peaks, actions=actions,
                fees=fees, breach=bad, ever_breached=breaches, mdd=worst,
                pretrade_tv_drift=drifts, terminal_holdings=x.tolist())


def continuation_decomposition(hold_branch, restore_net_branch, restore_no_current_fee_branch, fee_fraction):
    """Exact identity with an explicit response residual, never assumed zero."""
    h, rn, r0 = [float(x["terminal_real"]) for x in (hold_branch, restore_net_branch, restore_no_current_fee_branch)]
    if min(h, rn, r0) <= 0 or not 0 <= fee_fraction < 1: raise ValueError("Positive terminal values and valid fee fraction required")
    benefit = math.log(r0 / h); cost = -math.log1p(-fee_fraction)
    residual = math.log(rn / ((1 - fee_fraction) * r0))
    return dict(current_fee_free_action_benefit=benefit, current_log_cost=cost,
                continuation_response_residual=residual, direct_net_log_advantage=math.log(rn / h),
                decomposed_net_log_advantage=benefit - cost + residual)


def zero_preference_local_gate(training_gross, holdings, target, *, cost_rate=.0025, numerical_tolerance=1e-12):
    """Only a fixed local estimand example, not a calibrated full-H controller.

    Each row must be an explicitly declared equal-weight training observation.
    No selection, independence claim, uncertainty bound or risk guarantee is
    supplied by the economic sign test. Ties hold.
    """
    g = np.asarray(training_gross, float)
    if g.ndim != 2 or g.shape[1] != 3 or not len(g): raise ValueError("Nonempty joint forecast support required")
    parts = [one_step_parts(holdings, target, row, cost_rate=cost_rate) for row in g]
    benefit = float(np.mean([p["gross_log_benefit"] for p in parts])); cost = parts[0]["current_log_cost"]
    net = benefit - cost
    return dict(estimated_gross_log_benefit=benefit, exact_current_log_cost=cost,
                estimated_net_log_advantage=net, action="restore" if net > numerical_tolerance else "hold",
                forecast_support_rows=len(g), risk_preference_parameter=None,
                interpretation="One-step point-estimate break-even only; full-H risk admission and forecast adequacy remain separate")
