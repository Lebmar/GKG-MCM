"""Pure saved-trace diagnostics; no market loader, simulator, or selector."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOL = 1e-12


@dataclass
class ConsistencyChecks:
    """Count accounting identities and retain their largest absolute errors."""
    count: int = 0
    maximum_absolute_errors: dict = field(default_factory=dict)

    def require(self, condition, name):
        self.count += 1
        if not bool(condition):
            raise ValueError(name)

    def close(self, observed, expected, name, atol=2e-10, rtol=2e-13):
        a, b = np.asarray(observed), np.asarray(expected)
        self.require(a.shape == b.shape, name + "/shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), name + "/finite")
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
        self.maximum_absolute_errors[name] = max(error, self.maximum_absolute_errors.get(name, 0.))
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), name)


def fractional_upper_cvar(values, tail_fraction=.1):
    """Mean of exactly the upper stated empirical probability mass.

    Fractional boundary weight matters with 24 observations: the upper 10%
    contains the two largest values and 0.4 of the third, not three whole paths.
    """
    a = np.asarray(values, float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Finite nonempty one-dimensional sample required")
    if isinstance(tail_fraction, (bool, np.bool_)) or not 0 < tail_fraction <= 1:
        raise ValueError("Tail fraction must be in (0,1]")
    mass = tail_fraction * len(a)
    ordered = np.sort(a)[::-1]
    whole = int(math.floor(mass))
    weighted = float(ordered[:whole].sum())
    if whole < len(a):
        weighted += float(ordered[whole]) * (mass - whole)
    return weighted / mass


def peak_trough_diagnostic(real_wealth, loss=.4):
    """Recover drawdowns from saved positive annual real-wealth checkpoints."""
    w = np.asarray(real_wealth, float)
    if w.ndim != 1 or len(w) < 2 or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("At least two positive finite checkpoints required")
    if isinstance(loss, (bool, np.bool_)) or not np.isfinite(loss) or not 0 < loss < 1:
        raise ValueError("Loss tolerance must be in (0,1)")
    # All peak-trough intervals are independent of the simulator's sequential
    # peak implementation. argmax ties retain the earliest original endpoint.
    interval = np.asarray([max([0.] + [1 - w[t] / w[s] for s in range(t)]) for t in range(len(w))])
    running = np.maximum.accumulate(interval)
    peak = np.maximum.accumulate(w)
    current = np.maximum(1 - w / peak, 0.)
    crossed = np.flatnonzero(running - loss > TOL)
    first = int(crossed[0]) if len(crossed) else None
    trough = int(np.argmax(current))
    peak_year = int(np.argmax(w[:trough + 1]))
    recovery = None
    if first is not None:
        later = np.flatnonzero(w[first + 1:] >= peak[first])
        recovery = int(first + 1 + later[0]) if len(later) else None
    return dict(peak=peak, current_drawdown=current, running_mdd=running,
                ever_breached=running - loss > TOL, first_crossing=first,
                mdd=float(running[-1]), mdd_peak_year=peak_year,
                mdd_trough_year=trough, recovery_endpoint=recovery,
                post_crossing_endpoints_below_original_peak_n=None if first is None else int(np.sum(w[first + 1:] < peak[first])),
                added_mdd_after_crossing=None if first is None else float(running[-1] - running[first]))


def _finite_arm(arm, name, shape):
    value = np.asarray(arm[name])
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Invalid finite trace field: " + name)
    return value


def summarize_saved_path(arm, index, row, *, policy, checks=None, loss=.4, cost=.0025):
    """Reconstruct one saved account and export every annual decision.

    This validates historical accounting, not unobserved annual returns or a
    continuation forecast. References remain explicitly hypothetical.
    """
    checks = checks if checks is not None else ConsistencyChecks()
    real = np.asarray(arm["real_wealth"])[index].astype(float)
    h = len(real) - 1
    if h < 1:
        raise ValueError("Positive original horizon required")
    p = {name: np.asarray(value)[index] for name, value in arm.items()
         if name not in ("targets", "loss_tolerances", "annual_prices")}
    target = np.asarray(arm["targets"], float)
    checks.require(target.shape == (3,) and np.all(target >= 0), "target/simplex")
    checks.close(float(sum(target)), 1., "target/sum", atol=1e-13, rtol=0)
    target = target / float(sum(target))
    for name in ("real_wealth", "nominal_wealth", "real_peak", "current_drawdown", "running_mdd", "cpi"):
        _finite_arm(p, name, (h + 1,))
    for name in ("holdings", "weights"):
        _finite_arm(p, name, (h + 1, 3))
    for name in ("fees", "moved", "pretrade_wealth", "pretrade_tv_drift", "end_year_tv_drift"):
        _finite_arm(p, name, (h,))
    _finite_arm(p, "posttrade_holdings", (h, 3))
    checks.require(np.all(p["cpi"] > 0) and np.all(p["nominal_wealth"] > 0), "account/positive")
    checks.close(p["holdings"].sum(axis=1), p["nominal_wealth"], "account/holdings_sum")
    checks.close(p["weights"], p["holdings"] / p["nominal_wealth"][:, None], "account/weights", atol=2e-13, rtol=0)
    checks.close(p["nominal_wealth"] / p["cpi"], real, "account/real_wealth")
    checks.close(p["pretrade_wealth"], p["nominal_wealth"][:-1], "account/pretrade_wealth")
    checks.close(p["posttrade_holdings"].sum(axis=1), p["nominal_wealth"][:-1] - p["fees"], "account/self_financing")
    buys = np.maximum(p["posttrade_holdings"] - p["holdings"][:-1], 0.).sum(axis=1)
    checks.close(p["moved"], buys, "account/one_way_buys")
    checks.close(p["fees"], cost * buys, "account/cost_identity")
    checks.require(np.array_equal(np.asarray(p["trades"], bool), buys > TOL), "account/trade_count")
    actions = np.asarray(p["actions"])
    checks.require(actions.shape == (h,) and np.isin(actions, ["hold", "restore"]).all(), "account/actions")
    restore = actions == "restore"
    checks.require(np.array_equal(p["target_execution"], restore), "account/target_execution")
    if np.any(restore):
        post_w = p["posttrade_holdings"][restore] / p["posttrade_holdings"][restore].sum(axis=1)[:, None]
        checks.close(post_w, np.broadcast_to(target, post_w.shape), "account/restored_target", atol=2e-13, rtol=0)
    if np.any(~restore):
        checks.close(p["posttrade_holdings"][~restore], p["holdings"][:-1][~restore], "account/hold_state")
        checks.close(p["fees"][~restore], np.zeros(np.sum(~restore)), "account/hold_fee")
    predrift = .5 * np.abs(p["weights"][:-1] - target).sum(axis=1)
    enddrift = .5 * np.abs(p["weights"][1:] - target).sum(axis=1)
    checks.close(p["pretrade_tv_drift"], predrift, "account/pretrade_drift", atol=2e-13, rtol=0)
    checks.close(p["end_year_tv_drift"], enddrift, "account/end_year_drift", atol=2e-13, rtol=0)
    dd = peak_trough_diagnostic(real, loss)
    checks.close(p["real_peak"], dd["peak"], "drawdown/peak")
    checks.close(p["current_drawdown"], dd["current_drawdown"], "drawdown/current", atol=2e-13, rtol=0)
    checks.close(p["running_mdd"], dd["running_mdd"], "drawdown/all_intervals", atol=2e-13, rtol=0)
    checks.require(np.array_equal(p["ever_breached"], dd["ever_breached"]), "drawdown/absorbing_breach")
    saved_crossing = int(p["first_crossing"])
    checks.require(saved_crossing == (-1 if dd["first_crossing"] is None else dd["first_crossing"]), "drawdown/first_crossing")
    pending = np.asarray(p["pending_at_decision"], bool)
    expected_pending = np.zeros(h, bool) if policy == "equal_thirds_annual" else dd["ever_breached"][:-1]
    checks.require(np.array_equal(pending, expected_pending), "drawdown/pending_next_decision")
    checks.require(not np.any(restore & pending), "drawdown/no_restore_while_pending")
    if policy == "equal_thirds_annual":
        checks.require(restore.all(), "reference/exact_thirds_always_annual")
    growth = math.log(real[-1] / real[0]) / h
    derived = dict(log_growth=growth, cagr=math.expm1(growth), terminal_real=float(real[-1]), mdd=dd["mdd"],
                   excess=max(dd["mdd"] - loss, 0.), breach=dd["first_crossing"] is not None,
                   nominal_fees=float(p["fees"].sum()), nominal_moved=float(p["moved"].sum()),
                   trade_fraction=float(np.mean(p["trades"])), target_execution_fraction=float(np.mean(restore)))
    for field, value in derived.items():
        if field == "breach":
            checks.require(isinstance(row[field], (bool, np.bool_)) and bool(row[field]) == value, "CSV/" + field)
        else:
            checks.close(float(row[field]), value, "CSV/" + field)
    observed_crossing = row["first_crossing"]
    checks.require((observed_crossing is None or (isinstance(observed_crossing, float) and math.isnan(observed_crossing)))
                   if dd["first_crossing"] is None else observed_crossing == dd["first_crossing"], "CSV/first_crossing")
    for j, name in enumerate(("equity", "bond", "bill")):
        checks.close(row["terminal_" + name], p["weights"][-1, j], "CSV/terminal_" + name, atol=2e-13, rtol=0)
    tau = dd["first_crossing"]
    path = {**derived, "first_crossing": tau, "actual_trade_count": int(np.sum(p["trades"])),
            "target_execution_count": int(np.sum(restore)), "pending_decision_count": int(np.sum(pending)),
            "mdd_peak_year": dd["mdd_peak_year"], "mdd_trough_year": dd["mdd_trough_year"],
            "max_pretrade_tv_drift": float(max(predrift)), "max_end_year_tv_drift": float(max(enddrift)),
            "max_end_year_equity": float(max(p["weights"][1:, 0])),
            "min_end_year_bill": float(min(p["weights"][1:, 2])),
            "added_mdd_after_crossing": dd["added_mdd_after_crossing"],
            "recovery_endpoint": dd["recovery_endpoint"],
            "right_censored_recovery": None if tau is None else dd["recovery_endpoint"] is None,
            "post_crossing_endpoints_below_original_peak_n": dd["post_crossing_endpoints_below_original_peak_n"],
            **{"terminal_" + name: float(p["weights"][-1, j]) for j, name in enumerate(("equity", "bond", "bill"))}}
    annual = []
    for t in range(h):
        postweights = p["posttrade_holdings"][t] / p["posttrade_holdings"][t].sum()
        annual.append({"year": t + 1, "action": str(actions[t]), "trade": bool(p["trades"][t]),
                       "pending_at_decision": bool(pending[t]), "nominal_fee": float(p["fees"][t]),
                       "real_wealth_start": float(real[t]), "real_wealth_end": float(real[t + 1]),
                       "real_peak_end": float(dd["peak"][t + 1]), "current_drawdown_end": float(dd["current_drawdown"][t + 1]),
                       "running_mdd_end": float(dd["running_mdd"][t + 1]), "ever_breached_end": bool(dd["ever_breached"][t + 1]),
                       "first_crossing_this_year": tau == t + 1,
                       "pretrade_tv_drift": float(predrift[t]), "end_year_tv_drift": float(enddrift[t]),
                       **{"pretrade_" + n: float(p["weights"][t, j]) for j, n in enumerate(("equity", "bond", "bill"))},
                       **{"posttrade_" + n: float(postweights[j]) for j, n in enumerate(("equity", "bond", "bill"))},
                       **{"end_year_" + n: float(p["weights"][t + 1, j]) for j, n in enumerate(("equity", "bond", "bill"))}})
    return path, annual


def distribution_summary(records):
    """Conditional descriptive distribution; caller provides one fixed cohort."""
    if not records:
        raise ValueError("No undefined financial distributions for unadmitted paths")
    mdd = np.asarray([x["mdd"] for x in records], float)
    return dict(n=len(records), breach_n=sum(bool(x["breach"]) for x in records),
                mean_mdd=float(np.mean(mdd)), p90_mdd=float(np.quantile(mdd, .9, method="linear")),
                cvar90_mdd=fractional_upper_cvar(mdd),
                **{"mean_" + field: float(np.mean([x[field] for x in records]))
                   for field in ("excess", "log_growth", "cagr", "nominal_fees", "actual_trade_count", "pending_decision_count", "max_end_year_tv_drift", "max_end_year_equity", "min_end_year_bill")})


def select_event_witnesses(records):
    """All any-arm breaches and minimum all-arm nonbreach original ID."""
    if not records:
        raise ValueError("Nonempty arm inventory required")
    by_id = {}
    policies = {row["policy"] for row in records}
    for row in records:
        sid = row["scenario_id"]
        key = (sid, row["policy"])
        if not isinstance(sid, (int, np.integer)) or isinstance(sid, bool) or sid < 1:
            raise ValueError("Positive original scenario ID required")
        arm = by_id.setdefault(sid, {})
        if row["policy"] in arm:
            raise ValueError("Duplicate original-ID/policy")
        if not isinstance(row["breach"], (bool, np.bool_)):
            raise ValueError("Observed boolean breach required")
        arm[row["policy"]] = bool(row["breach"])
    if any(set(arm) != policies for arm in by_id.values()):
        raise ValueError("Every policy must have the same original IDs")
    breach_ids = sorted(sid for sid, arms in by_id.items() if any(arms.values()))
    nonbreach_ids = sorted(set(by_id) - set(breach_ids))
    return dict(any_arm_breach_ids=breach_ids,
                minimum_all_arm_nonbreach_id=nonbreach_ids[0] if nonbreach_ids else None,
                selected_scenario_ids=sorted(breach_ids + nonbreach_ids[:1]))
