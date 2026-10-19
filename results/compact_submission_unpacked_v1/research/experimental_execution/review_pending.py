"""Paired annual restoration and first-breach-then-pending-hold accounting.

Both arms begin as new accounts at100, entirely in the bill proxy, before
initial fees.  An annual endpoint whose running real MDD exceeds the client's
L by more than1e-12 records the first crossing.  The pending arm starts holding
at the *next* decision; the crossing year's action and fee are not undone.
Pending has no simulated human response, target/model update or re-entry.

The paired-prefix identity is structural.  Until the first crossing both
arms take the same actions on the same path and therefore have the same
state.  If no crossing occurs they never diverge.  Otherwise both flags are
already true before divergence and remain absorbing.  Thus first crossing
and full-H everbreach must match on every path, although severity, wealth,
fees and recovery can differ.  This module checks that coupling explicitly.

No files, forecasts, candidate catalogue, client admission or market data are
loaded.  A caller must supply already frozen targets and preserve any Review
clients as null outcomes outside this numerical kernel.  It is a research
ablation, not an adopted service or a completed Q2 benefit-gated policy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from safeinvest.core import execute_target

NUMERIC_TOL = 1e-12


def _inputs(returns, inflation, targets, loss_tolerances, horizon, cost_rate):
    market, infl = np.asarray(returns, float), np.asarray(inflation, float)
    desired = np.asarray(targets, float)
    if (market.ndim != 3 or market.shape[-1] != 3 or min(market.shape[:2]) < 1
            or infl.shape != market.shape[:2] or not np.all(np.isfinite(market))
            or not np.all(np.isfinite(infl)) or np.any(market <= -1) or np.any(infl <= -1)):
        raise ValueError("Require nonempty finite(n,T,3) returns and(n,T) inflation with positive gross")
    if desired.shape == (3,):
        desired = desired[None, :]
    if (desired.ndim != 2 or desired.shape[1] != 3 or len(desired) < 1
            or not np.all(np.isfinite(desired)) or np.any(desired < 0)
            or not np.allclose(desired.sum(axis=1), 1, atol=NUMERIC_TOL, rtol=0)):
        raise ValueError("targets must be one simplex point or a nonempty(m,3) simplex matrix")
    desired = desired / desired.sum(axis=1, keepdims=True)
    loss_raw = np.asarray(loss_tolerances)
    if np.issubdtype(loss_raw.dtype, np.bool_):
        raise ValueError("loss_tolerances must be numeric fractions, not booleans")
    loss = np.asarray(loss_tolerances, float)
    if loss.ndim == 0:
        loss = np.full(len(desired), float(loss))
    if loss.shape != (len(desired),) or not np.all(np.isfinite(loss)) or np.any(loss < 0) or np.any(loss > 1):
        raise ValueError("loss_tolerances must be a scalar or(m,) vector in[0,1]")
    if horizon is None:
        horizon = market.shape[1]
    if (isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer))
            or not 1 <= horizon <= market.shape[1]):
        raise ValueError("horizon must be an integer in[1,T]")
    if isinstance(cost_rate, (bool, np.bool_)) or not np.isscalar(cost_rate):
        raise ValueError("cost_rate must be a scalar in[0,1)")
    rate = float(cost_rate)
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("cost_rate must be finite and in[0,1)")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        cpi = np.column_stack((np.ones(len(market)), np.cumprod(1 + infl[:, :horizon], axis=1)))
    if not np.all(np.isfinite(cpi)) or np.any(cpi <= 0):
        raise ValueError("Cumulative CPI exceeds the numerical range")
    return market[:, :horizon], desired, loss, int(horizon), rate, cpi


def _simulate(market, targets, losses, horizon, rate, cpi, pending_hold):
    n, m = len(market), len(targets)
    periods, endpoints = (n, m, horizon), (n, m, horizon + 1)
    arm: dict[str, Any] = {
        name: np.zeros(periods, dtype=float) for name in (
            "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
            "pretrade_tv_drift", "end_year_tv_drift")}
    arm.update({name: np.zeros(endpoints, dtype=float) for name in (
        "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd")})
    arm["holdings"] = np.zeros(endpoints + (3,))
    arm["weights"] = np.zeros(endpoints + (3,))
    arm["posttrade_holdings"] = np.zeros(periods + (3,))
    arm["ever_breached"] = np.zeros(endpoints, dtype=bool)
    arm["target_execution"] = np.zeros(periods, dtype=bool)
    arm["pending_at_decision"] = np.zeros(periods, dtype=bool)
    arm["actions"] = np.full(periods, "restore", dtype="U7")
    arm["first_crossing"] = np.full((n, m), -1, dtype=np.int64)
    for key in ("drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger"):
        arm[key] = np.full((n, m), np.nan)
    arm["holdings"][..., 0, 2] = 100.0
    arm["weights"][..., 0, 2] = 1.0
    for key in ("nominal_wealth", "real_wealth", "real_peak"):
        arm[key][..., 0] = 100.0
    all_targets = np.broadcast_to(targets[None, :, :], (n, m, 3))

    for year in range(horizon):
        current = arm["holdings"][..., year, :]
        wealth = arm["nominal_wealth"][..., year]
        u = arm["weights"][..., year, :]
        pending = arm["first_crossing"] >= 0 if pending_hold else np.zeros((n, m), bool)
        restore = ~pending
        post = current.copy()
        fee, moved = np.zeros((n, m)), np.zeros((n, m))
        # Only selected restores execute.  There is no hypothetical fee
        # assigned to pending holdings and no second fee applied to wealth.
        if np.any(restore):
            selected_post, selected_fee, selected_moved = execute_target(current[restore], all_targets[restore], rate)
            post[restore] = selected_post
            fee[restore], moved[restore] = selected_fee, selected_moved
        arm["posttrade_holdings"][..., year, :] = post
        arm["fees"][..., year], arm["moved"][..., year] = fee, moved
        arm["pretrade_wealth"][..., year] = wealth
        arm["fee_fraction"][..., year] = fee / wealth
        arm["moved_fraction"][..., year] = moved / wealth
        arm["pretrade_tv_drift"][..., year] = .5 * np.abs(u - targets).sum(axis=-1)
        arm["pending_at_decision"][..., year] = pending
        arm["target_execution"][..., year] = restore
        arm["actions"][..., year] = np.where(restore, "restore", "hold")
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            after = post * (1 + market[:, None, year, :])
            nominal = after.sum(axis=-1)
            real = nominal / cpi[:, None, year + 1]
            end_weights = after / nominal[..., None]
        if (not np.all(np.isfinite(after)) or not np.all(np.isfinite(real))
                or np.any(nominal <= 0) or np.any(real <= 0) or not np.all(np.isfinite(end_weights))):
            raise ValueError("Executed holdings/wealth exceed the numerical range")
        peak = np.maximum(arm["real_peak"][..., year], real)
        drawdown = np.maximum(1 - real / peak, 0.0)
        running = np.maximum(arm["running_mdd"][..., year], drawdown)
        ever = arm["ever_breached"][..., year] | (running - losses[None, :] > NUMERIC_TOL)
        first = (arm["first_crossing"] < 0) & ever
        arm["first_crossing"][first] = year + 1
        for key, value in (("drawdown_at_trigger", running), ("peak_at_trigger", peak),
                           ("real_wealth_at_trigger", real)):
            arm[key][first] = value[first]
        arm["holdings"][..., year + 1, :] = after
        arm["weights"][..., year + 1, :] = end_weights
        arm["nominal_wealth"][..., year + 1] = nominal
        arm["real_wealth"][..., year + 1] = real
        arm["real_peak"][..., year + 1] = peak
        arm["current_drawdown"][..., year + 1] = drawdown
        arm["running_mdd"][..., year + 1] = running
        arm["ever_breached"][..., year + 1] = ever
        arm["end_year_tv_drift"][..., year] = .5 * np.abs(end_weights - targets).sum(axis=-1)

    triggered = arm["first_crossing"] >= 0
    endpoint = np.arange(horizon + 1)[None, None, :]
    recovery_candidates = (triggered[..., None] & (endpoint > arm["first_crossing"][..., None])
                           & (arm["real_wealth"] >= arm["peak_at_trigger"][..., None]))
    recovered = np.any(recovery_candidates, axis=-1)
    recovered_at = np.argmax(recovery_candidates, axis=-1)
    arm["triggered"] = triggered
    arm["recovered"] = recovered
    arm["recovery_endpoint"] = np.where(recovered, recovered_at, -1).astype(np.int64)
    arm["recovery_time"] = np.where(recovered, recovered_at - arm["first_crossing"], -1).astype(np.int64)
    arm["right_censored"] = triggered & ~recovered
    arm["available_followup"] = np.where(triggered, horizon - arm["first_crossing"], -1).astype(np.int64)
    arm["observed_followup"] = np.where(recovered, arm["recovery_time"], arm["available_followup"]).astype(np.int64)
    arm["added_severity"] = np.where(triggered,
                                       np.maximum(arm["running_mdd"][..., -1] - arm["drawdown_at_trigger"], 0), np.nan)
    arm["terminal_excess"] = np.maximum(arm["running_mdd"][..., -1] - losses[None, :], 0)
    arm["trades"] = arm["moved"] > NUMERIC_TOL
    arm["end_weights"] = arm["weights"][..., 1:, :]
    arm["cpi"] = cpi.copy()
    arm["targets"] = targets.copy()
    arm["loss_tolerances"] = losses.copy()
    return arm


def _coupling(annual, pending):
    n, m, points = annual["nominal_wealth"].shape
    horizon = points - 1
    same_first = np.array_equal(annual["first_crossing"], pending["first_crossing"])
    same_flag = np.array_equal(annual["ever_breached"][..., -1], pending["ever_breached"][..., -1])
    cutoff = np.where(annual["triggered"], annual["first_crossing"], horizon)
    state_mask = np.arange(horizon + 1)[None, None, :] <= cutoff[..., None]
    decision_mask = np.arange(1, horizon + 1)[None, None, :] <= cutoff[..., None]
    max_balance_error = 0.0
    same_prefix = np.array_equal(annual["cpi"], pending["cpi"])
    for key in ("holdings", "weights", "nominal_wealth", "real_wealth", "real_peak",
                "current_drawdown", "running_mdd", "ever_breached"):
        left, right = annual[key][state_mask], pending[key][state_mask]
        same_prefix &= np.array_equal(left, right)
        if key != "ever_breached":
            max_balance_error = max(max_balance_error, float(np.max(np.abs(left - right), initial=0)))
    for key in ("posttrade_holdings", "fees", "moved", "actions", "target_execution", "trades", "pending_at_decision"):
        same_prefix &= np.array_equal(annual[key][decision_mask], pending[key][decision_mask])
    if not (same_first and same_flag and same_prefix):
        raise RuntimeError("First-crossing/prefix/everbreach coupling invariant failed")
    return {"path_target_pairs": n * m, "triggered_pairs": int(annual["triggered"].sum()),
            "never_triggered_pairs": int((~annual["triggered"]).sum()),
            "same_first_crossing": bool(same_first), "same_terminal_ever_breached": bool(same_flag),
            "identical_prefixes_through_crossing": bool(same_prefix),
            "max_prefix_balance_error": max_balance_error,
            "interpretation": "Identical full-H everbreach is structural; only subsequent outcomes can differ"}


def simulate_paired(returns: ArrayLike, inflation: ArrayLike, targets: ArrayLike,
                     loss_tolerances: ArrayLike, *, horizon: int | None = None,
                     cost_rate: float = .0025) -> dict[str, Any]:
    """Return annual/pending_hold arrays and a checked coupling receipt.

    Returns/inflation have shape(n,T,3)/(n,T).  Targets are(3,) or(m,3), and
    L is scalar or(m,).  Each invocation starts a *new* original-H account at
    pre-fee100/all bills; it is not an interface for resetting existing client
    history.  Strict172 target eligibility/IDs and Review rows are the caller's
    frozen context, never selected or inferred here.

    Endpoint fields have shape(n,m,H+1): nominal_wealth, real_wealth,
    real_peak, current_drawdown, running_mdd, ever_breached.  holdings and
    weights add an asset axis.  ``holdings`` are nominal asset-value balances,
    not literal share counts: during pending these balances still grow with
    returns even though no asset quantities are exchanged.  CPI is(n,H+1).

    Period fields(n,m,H) include actual fees/moved, actions('restore'/'hold'),
    pending_at_decision, target_execution and trades.  target_execution counts
    scheduled restoration, while trades uses actual moved>1e-12.  Posttrade
    holdings are(n,m,H,3).  The usual fixed.horizon_values accounting/drift
    keys are retained for read-only compatibility.

    Summary fields(n,m) include first_crossing=1..H or-1, triggered,
    drawdown_at_trigger/peak_at_trigger/real_wealth_at_trigger(NaN if absent),
    recovery_endpoint/time(-1 if not recovered), recovered, right_censored,
    available_followup(H-tau), observed_followup(time to recovery or censor),
    added_severity, and terminal_excess.  Recovery is the first t>tau with
    actual R_t>=the *fixed* P_tau; it does not cause re-entry.  A trigger atH
    is right-censored with zero follow-up.  For no trigger the follow-up fields
    are-1 and boolean recovery/censor placeholders are false; callers MUST
    serialize all recovery/censor/trigger-value fields as null when triggered
    is false.  NPZ sentinels are not zero-risk or zero-time observations.

    All returned arm arrays are read-only.  No response delay, forecast, adviser
    return, bill conversion or extra fee-instant observation is invented.
    """
    market, desired, losses, h, rate, cpi = _inputs(returns, inflation, targets, loss_tolerances, horizon, cost_rate)
    annual = _simulate(market, desired, losses, h, rate, cpi, False)
    pending = _simulate(market, desired, losses, h, rate, cpi, True)
    receipt = _coupling(annual, pending)
    for arm in (annual, pending):
        for value in arm.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    return {"annual": annual, "pending_hold": pending, "coupling": receipt}
