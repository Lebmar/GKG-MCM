"""Descriptive accounting of fixed paired paths, never strategy selection.

The decompositions here are exact identities on two already fixed accounts
under the same market. They are not controlled causal effects or a new
conditional forecast. First-difference dates are numerical descriptions;
the original strict drawdown breach rule is kept separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

TOL = 1e-12
STATE_DIFFERENCE_ATOL = 1e-10
RATIO_DIFFERENCE_ATOL = 1e-12
ASSETS = ("equity", "bond", "bill")
ENDPOINT_FIELDS = ("nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd", "cpi", "ever_breached")
PERIOD_FIELDS = ("fees", "moved", "actions", "pending_at_decision", "target_execution", "trades")
Q_FLOAT_FIELDS = ("qg_hold", "qg_restore", "predicted_score", "predicted_growth_advantage_restore")
Q_COUNT_FIELDS = ("qbad_hold_count", "qbad_restore_count", "forecast_n", "predicted_breach_count_delta_restore")
FIRST_DIFFERENCE_FIELDS = {
    "first_pretrade_holdings_divergence_year": ("pretrade_holdings", STATE_DIFFERENCE_ATOL, 1),
    "first_pretrade_weights_divergence_year": ("pretrade_weights", RATIO_DIFFERENCE_ATOL, 1),
    "first_nominal_fee_divergence_year": ("fees", STATE_DIFFERENCE_ATOL, 1),
    "first_posttrade_holdings_divergence_year": ("posttrade_holdings", STATE_DIFFERENCE_ATOL, 1),
    "first_posttrade_weights_divergence_year": ("posttrade_weights", RATIO_DIFFERENCE_ATOL, 1),
    "first_endpoint_holdings_divergence_endpoint": ("holdings", STATE_DIFFERENCE_ATOL, 0),
    "first_nominal_wealth_divergence_endpoint": ("nominal_wealth", STATE_DIFFERENCE_ATOL, 0),
    "first_real_wealth_divergence_endpoint": ("real_wealth", STATE_DIFFERENCE_ATOL, 0),
    "first_real_peak_divergence_endpoint": ("real_peak", STATE_DIFFERENCE_ATOL, 0),
    "first_current_drawdown_divergence_endpoint": ("current_drawdown", RATIO_DIFFERENCE_ATOL, 0),
    "first_running_mdd_divergence_endpoint": ("running_mdd", RATIO_DIFFERENCE_ATOL, 0),
}


@dataclass
class Checks:
    count: int = 0
    maximum_absolute_errors: dict = field(default_factory=dict)

    def require(self, condition, name):
        self.count += 1
        if not bool(condition):
            raise ValueError(name)

    def close(self, actual, expected, name, *, atol=2e-10, rtol=3e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, name + "/shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), name + "/finite")
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
        self.maximum_absolute_errors[name] = max(error, self.maximum_absolute_errors.get(name, 0.))
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), name)


def first_difference(left, right, *, atol, offset=0):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.ndim < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite matching time arrays required")
    different = np.abs(a - b) > atol
    if different.ndim > 1:
        different = different.any(axis=tuple(range(1, different.ndim)))
    found = np.flatnonzero(different)
    return int(found[0] + offset) if len(found) else None


def _finite(arm, name, shape):
    a = np.asarray(arm[name])
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("Invalid finite path field: " + name)
    return a


def prepare_account(arm, returns, inflation, target, *, loss=.4, cost=.0025,
                    is_model=False, annual_price=None, fan_n=None, checks=None):
    """Validate a single fixed original account and add derived arrays.

    No forecast is recomputed. Copied model Qs are checked for availability,
    recorded count/score arithmetic, and consistency with recorded actions.
    B has no Q: all its exported Q columns remain NULL.
    """
    c = checks if checks is not None else Checks()
    r, z, w = np.asarray(returns, float), np.asarray(inflation, float), np.asarray(target, float)
    if r.ndim != 2 or r.shape[1] != 3 or not len(r) or z.shape != (len(r),):
        raise ValueError("One original complete market path required")
    h = len(r)
    c.require(np.isfinite(r).all() and np.isfinite(z).all() and (r > -1).all() and (z > -1).all(), "market/positive_factors")
    c.require(w.shape == (3,) and np.isfinite(w).all() and (w >= 0).all(), "target/simplex")
    c.close(w.sum(), 1., "target/sum", atol=1e-13, rtol=0)
    if not 0 < loss < 1 or not 0 <= cost < 1:
        raise ValueError("Valid original loss and cost required")
    a = {name: np.asarray(value).copy() for name, value in arm.items()}
    for name in ENDPOINT_FIELDS:
        _finite(a, name, (h + 1,))
    for name in ("holdings", "weights"):
        _finite(a, name, (h + 1, 3))
    _finite(a, "posttrade_holdings", (h, 3))
    for name in ("fees", "moved"):
        _finite(a, name, (h,))
    for name in ("actions", "pending_at_decision", "target_execution", "trades"):
        c.require(np.asarray(a[name]).shape == (h,), "shape/" + name)
    x, post = a["holdings"], a["posttrade_holdings"]
    V, fee = a["nominal_wealth"], a["fees"]
    cp, real = a["cpi"], a["real_wealth"]
    c.require((x >= 0).all() and (post >= 0).all() and (V > 0).all() and (cp > 0).all(), "account/nonnegative")
    c.close(x[0], [0., 0., 100.], "account/original_all_bills")
    c.close(cp[0], 1., "account/initial_cpi")
    c.close(x.sum(axis=1), V, "account/holdings_sum")
    c.close(a["weights"], x / V[:, None], "account/weights", atol=2e-13, rtol=0)
    c.close(post.sum(axis=1), V[:-1] - fee, "account/self_finance")
    buys = np.maximum(post - x[:-1], 0.).sum(axis=1)
    c.close(a["moved"], buys, "account/destination_once_buys")
    c.close(fee, cost * buys, "account/exact_fee")
    c.close(x[1:], post * (1 + r), "account/common_market_growth")
    c.close(cp[1:], cp[:-1] * (1 + z), "account/cpi_history")
    c.close(real, V / cp, "account/real_wealth")
    peak = np.maximum.accumulate(real)
    dd = np.maximum(1 - real / peak, 0.)
    worst = np.maximum.accumulate(dd)
    bad = worst - loss > TOL
    c.close(a["real_peak"], peak, "account/whole_historical_peak")
    c.close(a["current_drawdown"], dd, "account/current_drawdown", atol=2e-13, rtol=0)
    c.close(a["running_mdd"], worst, "account/running_mdd", atol=2e-13, rtol=0)
    c.require(np.array_equal(a["ever_breached"], bad), "account/absorbing_breach")
    crossed = np.flatnonzero(bad)
    tau = int(crossed[0]) if len(crossed) else None
    c.require(int(a["first_crossing"]) == (-1 if tau is None else tau), "account/first_crossing")
    actions = a["actions"]
    c.require(np.isin(actions, ("hold", "restore")).all(), "actions/vocabulary")
    restore = actions == "restore"
    c.require(bool(restore[0]), "actions/mandatory_initial_restore")
    c.require(np.array_equal(a["pending_at_decision"], bad[:-1]), "actions/absorbing_pending")
    c.require(not np.any(restore & bad[:-1]), "actions/no_restore_in_pending")
    c.require(np.array_equal(a["target_execution"], restore), "actions/target_execution")
    c.require(np.array_equal(a["trades"], buys > TOL), "actions/actual_trade")
    if restore.any():
        c.close(post[restore] / post[restore].sum(axis=1)[:, None], np.broadcast_to(w, post[restore].shape), "actions/restore_weights", atol=2e-13, rtol=0)
    if (~restore).any():
        c.close(post[~restore], x[:-1][~restore], "actions/hold_preserves_holdings")
    active = ~bad[:-1]
    active[0] = False
    if is_model:
        if annual_price is None or not np.isfinite(annual_price) or annual_price < 0 or not isinstance(fan_n, int) or isinstance(fan_n, bool) or fan_n < 1:
            raise ValueError("These fixed witnesses require finite technical price and positive integer fan")
        for name in Q_FLOAT_FIELDS:
            q = np.asarray(a[name], float)
            c.require(q.shape == (h,) and np.isfinite(q[active]).all() and np.isnan(q[~active]).all(), "Q/availability/" + name)
        for name in Q_COUNT_FIELDS:
            q = np.asarray(a[name])
            c.require(q.shape == (h,) and q.dtype.kind in "iu" and (q[~active] == -1).all(), "Q/count_availability/" + name)
        c.require((a["forecast_n"][active] == fan_n).all(), "Q/own_fan_size")
        for name in ("qbad_hold_count", "qbad_restore_count"):
            c.require(((a[name][active] >= 0) & (a[name][active] <= fan_n)).all(), "Q/bounded/" + name)
        delta_g = a["qg_restore"][active] - a["qg_hold"][active]
        delta_k = a["qbad_restore_count"][active] - a["qbad_hold_count"][active]
        score = delta_g - h * annual_price * delta_k / fan_n
        c.close(a["predicted_growth_advantage_restore"][active], delta_g, "Q/net_growth_difference", atol=2e-13, rtol=0)
        c.close(a["predicted_breach_count_delta_restore"][active], delta_k, "Q/count_difference", atol=0, rtol=0)
        c.close(a["predicted_score"][active], score, "Q/original_H_score", atol=2e-13, rtol=0)
        c.require(np.array_equal(restore[active], score > TOL), "Q/fixed_action_score")
    else:
        c.require(np.array_equal(restore, ~bad[:-1]), "B/annual_then_absorbing_pending")
    a["posttrade_weights"] = post / post.sum(axis=1)[:, None]
    a["pretrade_holdings"] = x[:-1].copy()
    a["pretrade_weights"] = a["weights"][:-1].copy()
    a["gross_portfolio_factor"] = np.sum(a["posttrade_weights"] * (1 + r), axis=1)
    a["fee_fraction"] = fee / V[:-1]
    a["real_log_growth"] = np.log(real[1:] / real[:-1])
    a["first_crossing_value"] = tau
    a["model_active"] = active if is_model else np.zeros(h, bool)
    return a


def endpoint_records(a, *, loss=.4):
    rows = []
    for t in range(len(a["real_wealth"])):
        row = {"endpoint": t, **{name: bool(a[name][t]) if name == "ever_breached" else float(a[name][t]) for name in ENDPOINT_FIELDS},
               "remaining_threshold_margin": float(loss - a["running_mdd"][t])}
        for j, asset in enumerate(ASSETS):
            row["holding_" + asset] = float(a["holdings"][t, j])
            row["weight_" + asset] = float(a["weights"][t, j])
        rows.append(row)
    return rows


def annual_records(a):
    rows = []
    h = len(a["fees"])
    for t in range(h):
        row = dict(year=t + 1, action=str(a["actions"][t]), nominal_fee=float(a["fees"][t]), nominal_moved=float(a["moved"][t]),
                   fee_fraction=float(a["fee_fraction"][t]), pending_at_decision=bool(a["pending_at_decision"][t]),
                   actual_trade=bool(a["trades"][t]), target_execution=bool(a["target_execution"][t]),
                   gross_portfolio_factor=float(a["gross_portfolio_factor"][t]), real_log_growth=float(a["real_log_growth"][t]),
                   first_crossing_this_year=a["first_crossing_value"] == t + 1,
                   q_available=bool(a["model_active"][t]), gate_reason=str(a["gate_reason"][t]) if "gate_reason" in a else None)
        for field in Q_FLOAT_FIELDS:
            row[field] = float(a[field][t]) if a["model_active"][t] else None
        for field in Q_COUNT_FIELDS:
            row[field] = int(a[field][t]) if a["model_active"][t] else None
        for j, asset in enumerate(ASSETS):
            row["posttrade_holding_" + asset] = float(a["posttrade_holdings"][t, j])
            row["posttrade_weight_" + asset] = float(a["posttrade_weights"][t, j])
        rows.append(row)
    return rows


def paired_records(model, baseline, returns, inflation, *, loss=.4, checks=None):
    """Two fixed paths' exact cost/exposure/wealth/peak accounting.

    The nominal three-term decomposition uses B's growth factor as an
    explicit ordering convention. The separate direct P&L decomposition is
    another identity, not additional terms to add to the first one.
    """
    c = checks if checks is not None else Checks()
    m, b, r, z = model, baseline, np.asarray(returns, float), np.asarray(inflation, float)
    h = len(m["fees"])
    if r.shape != (h, 3) or z.shape != (h,) or len(b["fees"]) != h:
        raise ValueError("Matching complete paired markets required")
    c.close(m["cpi"], b["cpi"], "pair/common_original_CPI", atol=0, rtol=0)
    rows = []
    for t in range(h):
        Vm, Vb = float(m["nominal_wealth"][t]), float(b["nominal_wealth"][t])
        fm, fb = float(m["fees"][t]), float(b["fees"][t])
        gm, gb = float(m["gross_portfolio_factor"][t]), float(b["gross_portfolio_factor"][t])
        prior, end = Vm - Vb, float(m["nominal_wealth"][t + 1] - b["nominal_wealth"][t + 1])
        fee_gap = fm - fb
        pnl_gap = float(np.sum(r[t] * (m["posttrade_holdings"][t] - b["posttrade_holdings"][t])))
        inherited, fee_effect, exposure = prior * gb, -fee_gap * gb, (Vm - fm) * (gm - gb)
        c.close(end, prior - fee_gap + pnl_gap, "pair/direct_nominal_identity")
        c.close(end, inherited + fee_effect + exposure, "pair/ordered_nominal_identity")
        log_start = math.log(float(m["real_wealth"][t] / b["real_wealth"][t]))
        log_end = math.log(float(m["real_wealth"][t + 1] / b["real_wealth"][t + 1]))
        log_fees = math.log1p(-fm / Vm) - math.log1p(-fb / Vb)
        log_exposure = math.log(gm) - math.log(gb)
        growth_gap = float(m["real_log_growth"][t] - b["real_log_growth"][t])
        c.close(log_end - log_start, log_fees + log_exposure, "pair/log_gap_increment", atol=2e-13, rtol=0)
        c.close(growth_gap, log_fees + log_exposure, "pair/common_inflation_cancels", atol=2e-13, rtol=0)
        Rm, Rb, Pm, Pb = (float(a[k][t + 1]) for a, k in ((m, "real_wealth"), (b, "real_wealth"), (m, "real_peak"), (b, "real_peak")))
        # Accounts include their own historical peak, so current DD is 1-R/P.
        dd_gap = float(m["current_drawdown"][t + 1] - b["current_drawdown"][t + 1])
        wealth_ratio_part, peak_ratio_part = (Rb - Rm) / Pb, Rm * (1 / Pb - 1 / Pm)
        c.close(dd_gap, wealth_ratio_part + peak_ratio_part, "pair/current_DD_ratio_identity", atol=2e-13, rtol=0)
        rows.append(dict(year=t + 1, model_action=str(m["actions"][t]), B_action=str(b["actions"][t]),
            action_differs=bool(m["actions"][t] != b["actions"][t]),
            nominal_wealth_gap_start=prior, nominal_wealth_gap_end=end, nominal_fee_gap=fee_gap,
            nominal_return_pnl_gap=pnl_gap,
            inherited_wealth_gap_at_B_growth=inherited, fee_gap_at_B_growth=fee_effect,
            exposure_gap_at_model_postcapital=exposure,
            real_wealth_gap_end=Rm - Rb, real_peak_gap_end=Pm - Pb,
            log_real_wealth_ratio_start=log_start, log_real_wealth_ratio_end=log_end,
            annual_real_log_growth_gap=growth_gap, log_fee_drag_gap=log_fees,
            log_posttrade_gross_gap=log_exposure,
            current_drawdown_gap_end=dd_gap, current_DD_wealth_ratio_part=wealth_ratio_part,
            current_DD_peak_ratio_part=peak_ratio_part,
            running_mdd_gap_end=float(m["running_mdd"][t + 1] - b["running_mdd"][t + 1]),
            model_first_crossing_this_year=m["first_crossing_value"] == t + 1,
            B_first_crossing_this_year=b["first_crossing_value"] == t + 1,
            inflation=float(z[t]), **{"market_return_" + n: float(r[t, j]) for j, n in enumerate(ASSETS)}))
    c.close(sum(x["log_fee_drag_gap"] + x["log_posttrade_gross_gap"] for x in rows),
            math.log(float(m["real_wealth"][-1] / b["real_wealth"][-1])), "pair/cumulative_log_identity", atol=2e-12, rtol=0)
    different = np.flatnonzero(m["actions"] != b["actions"])
    summary = dict(first_action_divergence_year=int(different[0] + 1) if len(different) else None,
        model_first_crossing=m["first_crossing_value"], B_first_crossing=b["first_crossing_value"],
        model_mdd=float(m["running_mdd"][-1]), B_mdd=float(b["running_mdd"][-1]),
        model_breach=bool(m["ever_breached"][-1]), B_breach=bool(b["ever_breached"][-1]),
        terminal_real_wealth_gap=float(m["real_wealth"][-1] - b["real_wealth"][-1]),
        terminal_log_real_wealth_ratio=math.log(float(m["real_wealth"][-1] / b["real_wealth"][-1])),
        mean_annual_log_growth_gap=float(np.mean(m["real_log_growth"] - b["real_log_growth"])),
        total_nominal_fee_gap=float(sum(m["fees"]) - sum(b["fees"])),
        total_log_fee_drag_gap=float(sum(x["log_fee_drag_gap"] for x in rows)),
        total_log_posttrade_gross_gap=float(sum(x["log_posttrade_gross_gap"] for x in rows)))
    for name, (field, atol, offset) in FIRST_DIFFERENCE_FIELDS.items():
        summary[name] = first_difference(m[field], b[field], atol=atol, offset=offset)
    # Fixed union of both crossings and their pre-crossing endpoints, without
    # replacement when an account never crosses. Initial endpoint is valid.
    witness_endpoints = sorted({e for a in (m, b) if a["first_crossing_value"] is not None
                                for e in (a["first_crossing_value"] - 1, a["first_crossing_value"])})
    witnesses = []
    for endpoint in witness_endpoints:
        for arm_name, a in (("model", m), ("B", b)):
            witnesses.append(dict(arm=arm_name,
                relation_to_own_crossing="no_crossing" if a["first_crossing_value"] is None else
                    "crossing" if endpoint == a["first_crossing_value"] else "pre_crossing" if endpoint == a["first_crossing_value"] - 1 else "other_arm_witness",
                **endpoint_records(a, loss=loss)[endpoint],
                preceding_action=None if endpoint == 0 else str(a["actions"][endpoint - 1]),
                next_action=None if endpoint == h else str(a["actions"][endpoint]),
                next_pending=None if endpoint == h else bool(a["pending_at_decision"][endpoint])))
    return rows, summary, witnesses


def cross_stage_summary(full, inner):
    """Describe full versus own-inner model trajectories, with no attribution.

    Callers must attach both original target/fan/price metadata. This is not
    the same-target M/B contrast and cannot identify one changed component.
    """
    different = np.flatnonzero(full["actions"] != inner["actions"])
    out = {"first_action_divergence_year": int(different[0] + 1) if len(different) else None}
    for name, (field, atol, offset) in FIRST_DIFFERENCE_FIELDS.items():
        out[name] = first_difference(full[field], inner[field], atol=atol, offset=offset)
    for stage, a in (("full", full), ("inner", inner)):
        h = len(a["fees"])
        out.update({stage + "_first_crossing": a["first_crossing_value"],
                    stage + "_mdd": float(a["running_mdd"][-1]),
                    stage + "_terminal_real": float(a["real_wealth"][-1]),
                    stage + "_mean_annual_log_growth": math.log(float(a["real_wealth"][-1] / a["real_wealth"][0])) / h,
                    stage + "_nominal_fees": float(sum(a["fees"]))})
    return out
