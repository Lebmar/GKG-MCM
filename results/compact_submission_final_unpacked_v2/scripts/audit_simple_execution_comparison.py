"""Independent arithmetic framework for the prospective simple-control study.

This module does not import the drift simulator, fitters, or selectors. Pure
helpers accept synthetic arrays or explicitly supplied saved records. The CLI
uses the guarded development loader only after a ready frozen protocol,
predeclared audit plan and complete numerical archive are verified. Importing
this module opens no data files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

DRIFTS = (0., .05, .1)
EVENT_TOL = 1e-12
GROWTH_TOL = 1e-12
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/simple_execution_comparison_protocol.json"
OUT = ROOT / "results/model_review/simple_execution_comparison_v1"
AUDIT_PLAN = ROOT / "docs/simple_execution_audit_plan.json"
CACHE = ROOT / "results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz"
LEARNERS = ("annual_gated_diagnostic", "drift_band_gated_diagnostic")
REFERENCES = ("annual_then_pending", "equal_thirds_annual")
FAMILIES = {LEARNERS[0]: (0,), LEARNERS[1]: (0, 1, 2)}
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real", "breach", "excess", "nominal_fees", "nominal_moved",
           "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill", "first_crossing", "mean_fee_fraction")
PAIR_FIELDS = ("log_growth", "cagr", "mdd", "excess", "nominal_fees", "trade_fraction", "breach", "mean_fee_fraction")
PAIRINGS = tuple((a, b) for a in LEARNERS for b in REFERENCES) + ((LEARNERS[1], LEARNERS[0]),)


@dataclass
class AuditLedger:
    checks_passed: int = 0
    maximum_absolute_errors: dict = field(default_factory=dict)

    def require(self, condition, name):
        if not bool(condition):
            raise ValueError(name)
        self.checks_passed += 1

    def close(self, actual, expected, name, *, atol=2e-10, rtol=3e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, name + "/shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), name + "/finite")
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
        self.maximum_absolute_errors[name] = max(self.maximum_absolute_errors.get(name, 0.), error)
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), name)


def _real(value, name, *, lower=0., upper=None):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or value < lower or (upper is not None and value > upper)):
        raise ValueError(name)
    return float(value)


def _integer(value, name, *, lower=0, upper=None):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < lower or (upper is not None and value > upper)):
        raise ValueError(name)
    return int(value)


def normalized_target(target):
    w = np.asarray(target, float)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0) or sum(w) <= 0:
        raise ValueError("Require a finite nonnegative three-asset target")
    if not math.isclose(float(sum(w)), 1., abs_tol=1e-12, rel_tol=0):
        raise ValueError("Target must sum to one before normalization")
    return w / float(sum(w))


def independent_fee_trade(holdings, target, *, cost_rate=.0025):
    """Solve fee = rate * total purchases using 100 bisection steps.

    Purchases use post-fee target values. Selling does not create a second
    transfer charge. This is independent of any simulator fee implementation.
    """
    x, w = np.asarray(holdings, float), normalized_target(target)
    rate = _real(cost_rate, "Nonnegative fee rate", upper=1.)
    if x.shape != (3,) or not np.isfinite(x).all() or np.any(x < 0) or sum(x) <= 0:
        raise ValueError("Positive finite account required")
    wealth = float(sum(x))
    if rate == 0:
        post = wealth * w
        return post, 0., float(sum(max(float(post[j] - x[j]), 0.) for j in range(3)))
    left, right = 0., rate * wealth
    for _ in range(100):
        mid = (left + right) / 2
        purchases = sum(max((wealth - mid) * float(w[j]) - float(x[j]), 0.) for j in range(3))
        if mid > rate * purchases:
            right = mid
        else:
            left = mid
    fee = (left + right) / 2
    post = (wealth - fee) * w
    moved = float(sum(max(float(post[j] - x[j]), 0.) for j in range(3)))
    return post, fee, moved


def independent_drift_action(holdings, target, delta, year_index, ever_breached, *, decision_tolerance=0.):
    """Initial restore; absorbing pending hold; delta0 annual; else strict TV gate.

    The explicit tolerance defaults to zero. The frozen study protocol must
    determine whether any numerical tolerance is added to the drift threshold.
    """
    d = _real(delta, "One fixed drift threshold required", upper=1.)
    if d not in DRIFTS:
        raise ValueError("Threshold must belong to the fixed three-candidate family")
    t = _integer(year_index, "Original zero-based decision year required")
    if not isinstance(ever_breached, (bool, np.bool_)):
        raise ValueError("Observed absorbing breach flag must be boolean")
    tol = _real(decision_tolerance, "Nonnegative declared drift comparison tolerance")
    x, w = np.asarray(holdings, float), normalized_target(target)
    if x.shape != (3,) or not np.isfinite(x).all() or np.any(x < 0) or sum(x) <= 0:
        raise ValueError("Positive finite account required")
    drift = .5 * float(sum(abs(float(x[j] / sum(x)) - float(w[j])) for j in range(3)))
    if t == 0:
        if ever_breached:
            raise ValueError("Original initial account cannot start previously breached")
        return "restore", drift
    if ever_breached:
        return "hold", drift
    return ("restore" if d == 0 or drift > d + tol else "hold"), drift


def independent_account(returns, inflation, target, delta, *, loss=.4, cost_rate=.0025,
                        initial_wealth=100., decision_tolerance=0., hold_after_breach=True):
    """Scalar annual account replay with its own actions and historical state.

    Passed arrays can be synthetic. This function never loads a market file.
    The real account starts before its initial purchase, preserves CPI/peak,
    observes drawdown at annual endpoints, and charges no terminal liquidation.
    """
    r, pi = np.asarray(returns, float), np.asarray(inflation, float)
    if (r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or pi.shape != (len(r),)
            or not np.isfinite(r).all() or not np.isfinite(pi).all()
            or np.any(r <= -1) or np.any(pi <= -1)):
        raise ValueError("One nonempty finite joint annual path required")
    initial = _real(initial_wealth, "Positive initial wealth")
    if initial <= 0: raise ValueError("Positive initial wealth")
    L = _real(loss, "Loss tolerance in (0,1)", upper=1.)
    if not 0 < L < 1: raise ValueError("Loss tolerance in (0,1)")
    if not isinstance(hold_after_breach, (bool, np.bool_)):
        raise ValueError("Reference pending behavior must be explicit boolean")
    if not hold_after_breach and delta != 0.:
        raise ValueError("Only the fixed always-annual reference may bypass pending hold")
    w = normalized_target(target)
    x = np.array([0., 0., initial]); cp = 1.; peak = initial; worst = 0.; bad = False; first = None
    fields = ("holdings", "weights", "nominal_wealth", "real_wealth", "cpi", "real_peak", "current_drawdown", "running_mdd", "ever_breached",
              "posttrade_holdings", "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction", "pretrade_tv_drift", "end_year_tv_drift",
              "actions", "trades", "target_execution", "pending_at_decision")
    out = {name: [] for name in fields}

    def save_state(current):
        nominal = float(sum(x))
        values = dict(holdings=x.copy(), weights=x / nominal, nominal_wealth=nominal,
                      real_wealth=nominal / cp, cpi=cp, real_peak=peak,
                      current_drawdown=current, running_mdd=worst, ever_breached=bad)
        for name, value in values.items(): out[name].append(value)

    save_state(0.)
    for t in range(len(r)):
        pending = bool(bad and hold_after_breach)
        action, drift = independent_drift_action(x, w, delta, t, pending, decision_tolerance=decision_tolerance)
        nominal = float(sum(x))
        post, fee, moved = independent_fee_trade(x, w, cost_rate=cost_rate) if action == "restore" else (x.copy(), 0., 0.)
        decisions = dict(posttrade_holdings=post.copy(), fees=fee, moved=moved, pretrade_wealth=nominal,
                         fee_fraction=fee / nominal, moved_fraction=moved / nominal,
                         pretrade_tv_drift=drift, actions=action, trades=moved > EVENT_TOL,
                         target_execution=action == "restore", pending_at_decision=pending)
        for name, value in decisions.items(): out[name].append(value)
        # Multiplication is performed scalar-by-scalar to avoid sharing the
        # simulator's vectorized transition implementation.
        x = np.array([float(post[j]) * (1 + float(r[t, j])) for j in range(3)])
        cp = cp * (1 + float(pi[t]))
        real = float(sum(x)) / cp
        peak = max(peak, real)
        current = max(1 - real / peak, 0.)
        worst = max(worst, current)
        bad = bool(bad or worst - L > EVENT_TOL)
        if bad and first is None: first = t + 1
        out["end_year_tv_drift"].append(.5 * float(sum(abs(float(x[j] / sum(x)) - float(w[j])) for j in range(3))))
        save_state(current)
    result = {name: np.asarray(values) for name, values in out.items()}
    result.update(targets=w, first_crossing=-1 if first is None else first, loss_tolerance=L, delta=float(delta))
    # A second all-interval computation verifies the sequential running state.
    real = result["real_wealth"]
    intervals = max([0.] + [1 - float(real[t]) / float(real[s]) for s in range(len(real) - 1) for t in range(s + 1, len(real))])
    if not math.isclose(intervals, worst, abs_tol=2e-14, rel_tol=0):
        raise ValueError("Independent interval and running-state drawdown disagree")
    recovered_at = None
    if first is not None:
        recovered_at = next((t for t in range(first + 1, len(real)) if real[t] >= result["real_peak"][first]), None)
    result.update(triggered=first is not None, recovered=recovered_at is not None,
                  drawdown_at_trigger=np.nan if first is None else float(result["running_mdd"][first]),
                  peak_at_trigger=np.nan if first is None else float(result["real_peak"][first]),
                  real_wealth_at_trigger=np.nan if first is None else float(real[first]),
                  recovery_endpoint=-1 if recovered_at is None else recovered_at,
                  recovery_time=-1 if recovered_at is None else recovered_at - first,
                  right_censored=first is not None and recovered_at is None,
                  available_followup=-1 if first is None else len(r) - first,
                  observed_followup=-1 if first is None else (recovered_at - first if recovered_at is not None else len(r) - first),
                  added_severity=np.nan if first is None else max(worst - float(result["running_mdd"][first]), 0.),
                  terminal_excess=max(worst - L, 0.), end_weights=result["weights"][1:],
                  trade_count=int(np.sum(result["trades"])), target_execution_count=int(np.sum(result["target_execution"])))
    reasons = []
    for t, action in enumerate(result["actions"]):
        reasons.append("already_breached_pending_hold" if result["pending_at_decision"][t] else
                       "mandatory_initial_restore" if t == 0 else
                       "annual_restore_threshold_zero" if delta == 0 else
                       "tv_exceeds_threshold_restore" if action == "restore" else "tv_within_threshold_hold")
    result["decision_reason"] = np.asarray(reasons)
    metric_values = independent_metrics(result, loss=L)
    result.update({name: value for name, value in metric_values.items() if name != "first_crossing"})
    return result


def independent_metrics(account, *, loss=.4):
    real = np.asarray(account["real_wealth"], float)
    h = len(real) - 1
    if h < 1 or np.any(real <= 0): raise ValueError("Positive original-H account required")
    g = math.log(float(real[-1]) / float(real[0])) / h
    mdd = max([0.] + [1 - float(real[t]) / float(real[s]) for s in range(h) for t in range(s + 1, h + 1)])
    weights = np.asarray(account["weights"])
    return dict(log_growth=g, cagr=math.expm1(g), mdd=mdd,
                volatility=float(np.std(real[1:] / real[:-1] - 1, ddof=1)) if h > 1 else 0.,
                terminal_real=float(real[-1]), breach=mdd - loss > EVENT_TOL,
                excess=max(mdd - loss, 0.), nominal_fees=float(sum(account["fees"])),
                nominal_moved=float(sum(account["moved"])), trade_fraction=float(np.mean(account["trades"])),
                target_execution_fraction=float(np.mean(account["target_execution"])),
                terminal_equity=float(weights[-1, 0]), terminal_bond=float(weights[-1, 1]), terminal_bill=float(weights[-1, 2]),
                first_crossing=None if account["first_crossing"] < 0 else int(account["first_crossing"]))


def compare_saved_account(saved, expected, ledger=None):
    """Compare complete physical fields; never trust saved actions as inputs."""
    ledger = ledger if ledger is not None else AuditLedger()
    for field in ("holdings", "weights", "nominal_wealth", "real_wealth", "cpi", "real_peak", "current_drawdown", "running_mdd",
                  "posttrade_holdings", "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction", "pretrade_tv_drift", "end_year_tv_drift", "targets"):
        ledger.require(field in saved, "saved/required/" + field)
        ledger.close(saved[field], expected[field], "saved/" + field)
    for field in ("ever_breached", "actions", "trades", "target_execution", "pending_at_decision"):
        ledger.require(field in saved and np.array_equal(saved[field], expected[field]), "saved/" + field)
        if field != "actions":
            ledger.require(np.asarray(saved[field]).dtype.kind == "b", "saved/boolean_dtype/" + field)
    crossing = _integer(saved["first_crossing"], "Saved first crossing must be an integer sentinel or endpoint", lower=-1, upper=len(expected["fees"]))
    ledger.require(crossing == expected["first_crossing"], "saved/first_crossing")
    optional = ("triggered", "recovered", "drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger", "recovery_endpoint", "recovery_time",
                "right_censored", "available_followup", "observed_followup", "added_severity", "terminal_excess", "end_weights", "trade_count", "target_execution_count",
                "decision_reason", "nominal_fees", "nominal_moved", "log_growth", "cagr", "mdd", "volatility", "terminal_real", "breach", "excess")
    for field in optional:
        if field not in saved: continue
        a, b = np.asarray(saved[field]), np.asarray(expected[field])
        if b.dtype.kind in "bUS":
            ledger.require(a.dtype.kind == b.dtype.kind and np.array_equal(a, b), "saved/optional/" + field)
        else:
            ledger.require(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)), "saved/optional_NULL/" + field)
            if np.isfinite(b).any(): ledger.close(a[np.isfinite(b)], b[np.isfinite(b)], "saved/optional/" + field)
    return ledger


def independent_training_choice(records, n, reference_breach_count, *, alpha=.1, candidate_orders=(0, 1, 2)):
    """Independent three-candidate growth/cap arithmetic, no market IO.

    Required records use candidate_order, delta, n, breach_count,
    mean_log_growth, mean_excess, mean_nominal_fees. Complete family,
    integer count and the global growth band are checked; secondary keys use
    exact numeric ordering, never sequential pairwise tolerance comparisons.
    """
    n = _integer(n, "Positive original training-path count", lower=1)
    k_b = _integer(reference_breach_count, "Original B breach count", upper=n)
    a = _real(alpha, "Fixed empirical risk reference", upper=1.)
    orders = tuple(candidate_orders)
    if orders not in ((0,), (0, 1, 2)):
        raise ValueError("Only annual positive control or fixed three-control family permitted")
    if len(records) != 3: raise ValueError("Exactly three fixed drift controls required")
    cap = min(k_b, math.floor(a * n + EVENT_TOL))
    candidates = []
    for order, (row, delta) in enumerate(zip(records, DRIFTS)):
        if (_integer(row["candidate_order"], "Candidate order", upper=2) != order
                or row["delta"] != delta or _integer(row["n"], "Original training n", lower=1) != n):
            raise ValueError("Fixed ordered complete family and original n required")
        k = _integer(row["breach_count"], "Candidate breach count", upper=n)
        g = row["mean_log_growth"]
        if isinstance(g, (bool, np.bool_)) or not np.isfinite(g): raise ValueError("Finite candidate growth required")
        excess = _real(row["mean_excess"], "Finite nonnegative mean excess")
        fees = _real(row["mean_nominal_fees"], "Finite nonnegative mean total nominal fees")
        if order in orders and k <= cap: candidates.append((order, float(g), excess, fees))
    maximum = max((g for _, g, _, _ in candidates), default=None)
    band = [x for x in candidates if x[1] >= maximum - GROWTH_TOL]
    selected = min(band, key=lambda x: (x[2], x[3], x[0]))[0] if band else None
    return dict(training_n=n, effective_cap=cap, family_candidate_orders=list(orders), eligible_candidate_orders=[x[0] for x in candidates],
                selected_candidate_order=selected, selected_delta=None if selected is None else DRIFTS[selected],
                maximum_eligible_mean_log_growth=maximum)


def independent_four_inner_receipt(records, original_ids, *, alpha=.1):
    """Reconstruct denominators for four inner validations before a policy gate.

    The gate requires positive automatic coverage, empirical selected risk at
    most alpha, and no more events than annual B on exactly the same automatic
    original IDs. All-four admission is reported descriptively, not required.
    Required fold fields: fold, evaluation_ids, automated, breach_count,
    mean_log_growth, reference_breach_count (NULL when unavailable).
    """
    ids = list(original_ids)
    if len(ids) == 0 or len(set(ids)) != len(ids) or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in ids):
        raise ValueError("Unique positive original training IDs required")
    if len(records) != 4: raise ValueError("Exactly four predeclared inner folds required")
    a = _real(alpha, "Fixed empirical validation risk reference", upper=1.)
    seen = []; auto_n = 0; breach_n = 0; reference_k = 0; weighted_growth = 0.; auto_folds = []
    for i, row in enumerate(records, 1):
        if row["fold"] != i or isinstance(row["fold"], bool): raise ValueError("Canonical original inner-fold order required")
        fold_ids = list(row["evaluation_ids"])
        if not fold_ids or len(set(fold_ids)) != len(fold_ids): raise ValueError("Unique nonempty original inner evaluation IDs required")
        seen.extend(fold_ids)
        if not isinstance(row["automated"], (bool, np.bool_)): raise ValueError("Observed inner admission boolean required")
        if row["automated"]:
            k = _integer(row["breach_count"], "Inner original-path breach count", upper=len(fold_ids))
            kb = _integer(row["reference_breach_count"], "Same automatic-ID annual B breach count", upper=len(fold_ids))
            g = row["mean_log_growth"]
            if isinstance(g, (bool, np.bool_)) or not np.isfinite(g): raise ValueError("Finite automatic inner growth required")
            auto_n += len(fold_ids); breach_n += k; reference_k += kb; weighted_growth += len(fold_ids) * float(g); auto_folds.append(i)
        elif row["breach_count"] is not None or row["mean_log_growth"] is not None or row["reference_breach_count"] is not None:
            raise ValueError("Unavailable inner outcomes must remain NULL")
    if len(seen) != len(ids) or len(set(seen)) != len(seen) or set(seen) != set(ids):
        raise ValueError("Four inner folds must partition the original training IDs exactly once")
    return dict(original_n=len(ids), automatic_n=auto_n, review_n=len(ids) - auto_n,
                automatic_inner_folds=auto_folds, all_four_automated=len(auto_folds) == 4,
                breach_count=None if not auto_n else breach_n,
                reference_breach_count_same_automatic_ids=None if not auto_n else reference_k,
                conditional_mean_log_growth=None if not auto_n else weighted_growth / auto_n,
                effective_gate_cap=None if not auto_n else math.floor(a * auto_n + EVENT_TOL),
                empirical_risk_eligible=bool(auto_n and breach_n <= math.floor(a * auto_n + EVENT_TOL)),
                paired_B_dominance_eligible=bool(auto_n and breach_n <= reference_k),
                inner_gate_eligible=bool(auto_n and breach_n <= math.floor(a * auto_n + EVENT_TOL) and breach_n <= reference_k))


def validate_original_membership(job, *, allowed_ids=tuple(range(1, 121))):
    """Check a declared disjoint train/evaluation job, without opening data."""
    train, evaluation = list(job["training_ids"]), list(job["evaluation_ids"])
    allowed = set(allowed_ids)
    for group in (train, evaluation):
        if not group or len(set(group)) != len(group) or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in group):
            raise ValueError("Original unique integer train/evaluation IDs required")
    if set(train) & set(evaluation) or not (set(train) | set(evaluation)) <= allowed:
        raise ValueError("Train/evaluation overlap or out-of-scope original IDs")
    return dict(training_n=len(train), evaluation_n=len(evaluation))


def validate_protocol_membership(protocol):
    """Verify all declared 25 parents, 125 jobs and 500 static subfolds.

    This is pure protocol arithmetic. It does not read cached Q1 outcomes,
    market observations, or the prospective experiment's numerical results.
    """
    allowed = list(range(1, 121))
    if protocol["data"]["scenario_ids"] != allowed or protocol["data"]["allow_final"] is not False:
        raise ValueError("Original development-only data scope required")
    if (protocol["original_H"] != 30 or protocol["loss_tolerance"] != .4 or protocol["service_alpha"] != .1
            or protocol["cost_rate"] != .0025 or protocol["liquidity_floors"] != {"high": .3, "medium": .1, "low": 0.}):
        raise ValueError("Fixed original client and policy constants required")
    if protocol["architectures"] != {"annual_gated_diagnostic": [0.], "drift_band_gated_diagnostic": [0., .05, .1]}:
        raise ValueError("Only the two predeclared diagnostic architectures permitted")
    contexts = protocol["contexts"]; jobs = protocol["jobs"]; static = protocol["static_contexts"]
    if len(contexts) != 25 or len(jobs) != 125 or len(static) != 125:
        raise ValueError("Complete declared 25-parent/125-job inventory required")
    by_job = {j["job_id"]: j for j in jobs}
    if len(by_job) != 125 or set(by_job) != set(static): raise ValueError("Unique matching dynamic/static job IDs required")
    by_parent = {(c["repeat_id"], c["outer_fold"]): c for c in contexts}
    if set(by_parent) != {(r, f) for r in range(1, 6) for f in range(1, 6)}:
        raise ValueError("Exactly all five folds of all five repeats required")
    names = [c["name"] for c in contexts]; cids = [c["context_id"] for c in contexts]
    if len(set(names)) != 25 or len(set(cids)) != 25: raise ValueError("Unique parent context identities required")
    seen_jobs = []; static_subfold_n = 0
    for repeat in range(1, 6):
        outer_seen = []
        for fold in range(1, 6):
            c = by_parent[(repeat, fold)]
            parent_membership = validate_original_membership({"training_ids": c["training_ids"], "evaluation_ids": c["outer_test_ids"]})
            if parent_membership != {"training_n": 96, "evaluation_n": 24} or set(c["training_ids"]) | set(c["outer_test_ids"]) != set(allowed):
                raise ValueError("Each parent must retain a complete original 96/24 partition")
            outer_seen.extend(c["outer_test_ids"])
            if len(c["inner_folds"]) != 4: raise ValueError("Each parent must declare four dynamic inner folds")
            inner_seen = []
            for i, inner in enumerate(c["inner_folds"], 1):
                if inner["inner_fold"] != i: raise ValueError("Canonical dynamic inner-fold order required")
                ji = f"c{c['context_id']}_inner{i}"
                if ji not in by_job: raise ValueError("Missing declared dynamic inner job")
                j = by_job[ji]
                membership = validate_original_membership(j)
                if membership != {"training_n": 72, "evaluation_n": 24}:
                    raise ValueError("Dynamic inner training/evaluation n must be 72/24")
                if j["training_ids"] != inner["train_ids"] or j["evaluation_ids"] != inner["test_ids"]:
                    raise ValueError("Dynamic job membership must equal its own parent inner fold")
                if set(j["training_ids"]) | set(j["evaluation_ids"]) != set(c["training_ids"]):
                    raise ValueError("Dynamic inner data must exactly partition parent training IDs")
                if j["dynamic_inner_fold"] != i or j["stage"] != "inner": raise ValueError("Dynamic inner stage identity changed")
                inner_seen.extend(j["evaluation_ids"]); seen_jobs.append(ji)
            if len(inner_seen) != 96 or len(set(inner_seen)) != 96 or set(inner_seen) != set(c["training_ids"]):
                raise ValueError("Four dynamic validations must cover each parent training ID once")
            refit_id = f"c{c['context_id']}_refit"
            if refit_id not in by_job: raise ValueError("Missing final parent refit")
            final = by_job[refit_id]
            if final["training_ids"] != c["training_ids"] or final["evaluation_ids"] != c["outer_test_ids"] or final["stage"] != "refit" or final["dynamic_inner_fold"] is not None:
                raise ValueError("Final refit must preserve original parent train/outer memberships")
            seen_jobs.append(refit_id)
            for j in [by_job[f"c{c['context_id']}_inner{i}"] for i in range(1, 5)] + [final]:
                if (j["parent_context"] != c["name"] or j["parent_context_id"] != c["context_id"]
                        or j["repeat_id"] != repeat or j["outer_fold"] != fold):
                    raise ValueError("Job parent/repeat/fold lineage mismatch")
                sc = static[j["job_id"]]
                expected_static_name = c["name"] if j["stage"] == "refit" else j["job_id"]
                if (sc["name"] != expected_static_name or sc["training_ids"] != j["training_ids"]
                        or sc["outer_test_ids"] != j["evaluation_ids"] or sc["inner_seed"] != j["static_inner_seed"]):
                    raise ValueError("Static component cache context/job identity mismatch")
                if len(sc["inner_folds"]) != 4: raise ValueError("Four declared static calibration subfolds required")
                static_seen = []
                for i, sf in enumerate(sc["inner_folds"], 1):
                    m = validate_original_membership({"training_ids": sf["train_ids"], "evaluation_ids": sf["test_ids"]})
                    n = len(j["training_ids"])
                    if sf["inner_fold"] != i or m != {"training_n": n * 3 // 4, "evaluation_n": n // 4}:
                        raise ValueError("Static subfold size/order mismatch")
                    if set(sf["train_ids"]) | set(sf["test_ids"]) != set(j["training_ids"]):
                        raise ValueError("Static calibration may use only its declared job training paths")
                    static_seen.extend(sf["test_ids"]); static_subfold_n += 1
                if len(static_seen) != len(j["training_ids"]) or len(set(static_seen)) != len(static_seen) or set(static_seen) != set(j["training_ids"]):
                    raise ValueError("Static validations must cover each own job training ID once")
        if len(outer_seen) != 120 or len(set(outer_seen)) != 120 or set(outer_seen) != set(allowed):
            raise ValueError("Every repeat must evaluate all original120 paths exactly once")
    if len(seen_jobs) != 125 or set(seen_jobs) != set(by_job): raise ValueError("Orphan/missing dynamic training job")
    return dict(parent_contexts=25, dynamic_training_jobs=125, static_contexts=125,
                static_inner_folds=static_subfold_n, repeated_outer_original_ID_records=600)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as handle: return json.load(handle)
    return json.loads(p.read_text())


def maybe(actual, expected, ledger, name):
    """Preserve NULL/boolean/string identities rather than coercing them."""
    if expected is None:
        ledger.require(actual is None or (isinstance(actual, (float, np.floating)) and np.isnan(actual)), name + "/NULL")
    elif isinstance(expected, (bool, np.bool_)):
        ledger.require(isinstance(actual, (bool, np.bool_)) and bool(actual) == bool(expected), name + "/boolean")
    elif isinstance(expected, str): ledger.require(actual == expected, name + "/string")
    else: ledger.close(actual, expected, name)


def index_rows(frame, keys):
    result = {}
    for row in frame.to_dict("records"):
        key = tuple(row[k] for k in keys)
        if key in result: raise ValueError("Duplicate saved identity: " + str(keys))
        result[key] = row
    return result


def observed_bools(frame):
    result = frame.copy()
    for field in ("assigned", "breach", "reference_only", "hypothetical_reference", "initial_admitted", "inner_gate_passed",
                  "reference_is_hypothetical", "learner_auto", "reference_available", "common_auto", "eligible"):
        if field not in result: continue
        def convert(x):
            if isinstance(x, (bool, np.bool_)) or pd.isna(x): return x
            if isinstance(x, str) and x in ("True", "False"): return x == "True"
            raise ValueError("Literal observed boolean required: " + field)
        result[field] = result[field].map(convert)
    return result


def independent_cvar90(values):
    a = sorted(map(float, values), reverse=True)
    if not a or not all(math.isfinite(x) for x in a): raise ValueError("Finite nonempty empirical losses required")
    mass = len(a) / 10; whole = math.floor(mass)
    return (sum(a[:whole]) + (mass - whole) * a[whole] if whole < len(a) else sum(a)) / mass


def compare_metric_row(row, expected, ledger, *, available, status, reason=None):
    maybe(row["assigned"], available, ledger, "path/assigned")
    maybe(row["execution_status"], status, ledger, "path/execution_status")
    maybe(row["execution_reason"], reason, ledger, "path/execution_reason")
    if available:
        for field in METRICS: maybe(row[field], expected[field], ledger, "metric/" + field)
    else:
        for field in METRICS: maybe(row[field], None, ledger, "unavailable/" + field)


def expected_performance(rows, policy):
    """Independently aggregate saved, individually replayed records."""
    if not rows: raise ValueError("Nonempty original summary denominator required")
    if len({r["scenario_id"] for r in rows}) != len(rows): raise ValueError("One original ID per performance summary")
    reference = policy in REFERENCES
    auto = [r for r in rows if r["assigned"]]
    cases = {name: sum(r["case_status"] == name for r in rows) for name in ("balanced_case", "q1_review", "not_balanced_case")}
    n, k = len(auto), sum(bool(r["breach"]) for r in auto) if auto else None
    result = dict(original_n=len(rows), balanced_case_n=cases["balanced_case"], q1_review_n=cases["q1_review"], not_balanced_case_n=cases["not_balanced_case"],
                  case_auto_n=None if reference else sum(r["execution_status"] == "automated" for r in rows),
                  case_execReview_n=None if reference else sum(r["execution_status"] == "execution_review" for r in rows),
                  assigned_paths=n, financial_null_paths=len(rows) - n, reference_only=reference,
                  reference_available_n=n if reference else None,
                  case_reference_available_n=sum(r["assigned"] and r["case_status"] == "balanced_case" for r in rows) if reference else None,
                  breach_count=k, conditional_breach_probability=k / n if n else None, breaches_per_original_n=k / len(rows) if n else None)
    if not reference and result["balanced_case_n"] != result["case_auto_n"] + result["case_execReview_n"]:
        raise ValueError("Balanced service partition differs")
    mapping = {"mean_log_growth": ("log_growth", np.mean), "median_cagr": ("cagr", np.median),
               "p10_cagr": ("cagr", lambda x: np.quantile(x, .1, method="inverted_cdf")),
               "mean_mdd": ("mdd", np.mean), "median_mdd": ("mdd", np.median),
               "p90_mdd": ("mdd", lambda x: np.quantile(x, .9, method="inverted_cdf")),
               "cvar90_mdd": ("mdd", independent_cvar90), "mean_nominal_fees": ("nominal_fees", np.mean),
               "mean_trade_fraction": ("trade_fraction", np.mean), "mean_excess": ("excess", np.mean),
               "mean_fee_fraction": ("mean_fee_fraction", np.mean)}
    for field, (source, function) in mapping.items(): result[field] = float(function([r[source] for r in auto])) if n else None
    return result


def audit_pipeline(protocol, bundles, returns, inflation, ids, tables, training_choices, context_selections,
                   trace_groups, archives, cache, *, data_digest, ledger=None):
    """Audit a complete supplied experiment; this function performs no file IO.

    Identical deterministic accounts may be reused only through an exact key
    containing all physical inputs and the full guarded data-array digest.
    Every saved record and selected state tree is still compared separately.
    """
    ledger = ledger if ledger is not None else AuditLedger()
    membership = validate_protocol_membership(protocol)
    ledger.require(list(map(int, ids)) == list(range(1, 121)) and np.shape(returns) == (120, 30, 3) and np.shape(inflation) == (120, 30), "data/original120")
    if not isinstance(data_digest, str) or len(data_digest) != 64: raise ValueError("Full guarded input-array SHA required for replay cache")
    for name, expected in protocol["expected_rows"].items(): ledger.require(len(tables[name]) == expected, "inventory/" + name)
    jobs = protocol["jobs"]; contexts = {c["name"]: c for c in protocol["contexts"]}
    ledger.require(set(bundles) == {j["job_id"] for j in jobs} == set(training_choices), "inventory/Q1_and_choice_jobs")
    ledger.require(set(context_selections) == set(contexts), "inventory/context_choices")
    weights = np.asarray(cache["weights"]); allowed = np.flatnonzero(np.all(weights >= .05 - EVENT_TOL, axis=1))
    ledger.require(len(weights) == 232 and len(allowed) == 172, "Q1/strict_candidate_mapping")
    train_rows = index_rows(tables["training_paths"], ("job_id", "execution_candidate_order", "scenario_id"))
    train_summaries = index_rows(tables["training_summary"], ("job_id", "execution_candidate_order"))
    inner_rows = index_rows(tables["inner_validation_paths"], ("job_id", "policy", "scenario_id"))
    inner_B = index_rows(tables["inner_reference_paths"], ("job_id", "scenario_id"))
    actual_rows = index_rows(tables["client_paths"], ("context", "policy", "scenario_id"))
    gate_rows = index_rows(tables["inner_gate_summaries"], ("context", "policy"))
    admission_rows = index_rows(tables["fit_admission"], ("context", "policy"))
    replay_cache = {}; expected_groups = {"actual": {}, "validation": {}}
    expected_inner = {name: {policy: [] for policy in LEARNERS} for name in contexts}
    expected_gate = {}; checked_training = 0; checked_inner = 0; checked_actual = 0; checked_trace_paths = 0

    def physical(sid, target, delta, pending=True):
        normalized = normalized_target(target)
        key = (data_digest, tuple(map(float, normalized)), float(delta), int(sid), .4, .0025, 30, bool(pending), 100.)
        if key not in replay_cache:
            arm = independent_account(returns[sid - 1], inflation[sid - 1], normalized, delta, hold_after_breach=pending)
            metrics = independent_metrics(arm)
            metrics["mean_fee_fraction"] = float(np.mean(arm["fee_fraction"]))
            replay_cache[key] = (arm, metrics)
        return replay_cache[key]

    def identity(row, job, bundle, *, policy=None, order=None, equal_reference=False):
        case = bundle["q2_case"]; product = case["product"]
        expected = dict(job_id=job["job_id"], context=job["parent_context"], context_id=job["parent_context_id"],
                        repeat_id=job["repeat_id"], outer_fold=job["outer_fold"], dynamic_inner_fold=job["dynamic_inner_fold"],
                        client="C", horizon=30, loss_tolerance=.4, q1_effective_assignment=case["effective_assignment"],
                        case_status=case["public_status"], static_gamma=bundle["refit"]["selection"]["selected_gamma"],
                        selection_training_n=len(job["training_ids"]))
        if policy is not None: expected["policy"] = policy
        for field, value in expected.items(): maybe(row[field], value, ledger, "identity/" + field)
        if not equal_reference:
            for field, key in (("candidate_local_index", "candidate_index"), ("candidate_original232_index", "candidate_original232_index")):
                maybe(row[field], product[key] if product else None, ledger, "identity/" + field)
        else:
            maybe(row["candidate_local_index"], None, ledger, "reference/exact_thirds_has_no_catalogue_index")
            maybe(row["candidate_original232_index"], None, ledger, "reference/exact_thirds_has_no_global_index")
        if order is not None:
            for field in ("candidate_order", "delta", "tv_threshold"):
                maybe(row[field], order if field == "candidate_order" else DRIFTS[order], ledger, "identity/" + field)

    for job in jobs:
        jid, name = job["job_id"], job["parent_context"]
        b = bundles[jid]; case = b["q2_case"]; refit = b["refit"]
        ledger.require(b["context"] == protocol["static_contexts"][jid], "Q1/component_context")
        ledger.require(refit["training_ids"] == job["training_ids"] and refit["outer_test_ids"] == job["evaluation_ids"], "Q1/original_membership")
        assignment = refit["effective_assignments"]["C"]
        expected_case = {"Balanced": "balanced_case", "Review": "q1_review", "Conservative": "not_balanced_case", "Growth": "not_balanced_case"}[assignment]
        ledger.require(case["effective_assignment"] == assignment and case["public_status"] == expected_case, "Q1/effective_case")
        menu = refit["raw_fit"]["menu"] if refit.get("raw_fit") else []
        menu_product = next((item for item in menu if item["name"] == assignment), None)
        if assignment == "Review":
            ledger.require(case.get("product") is None and case.get("q1_product") is None, "Q1/Review_no_product")
        else:
            ledger.require(menu_product is not None and case.get("q1_product") is not None, "Q1/effective_named_menu_product")
            for key in ("name", "candidate_index", "candidate_original232_index"):
                ledger.require(case["q1_product"][key] == menu_product[key], "Q1/q1_product_semantics/" + key)
            ledger.close(case["q1_product"]["weights"], menu_product["weights"], "Q1/q1_product_semantics/weights", atol=0, rtol=0)
            if assignment == "Balanced":
                ledger.require(case.get("product") is not None, "Q1/Balanced_product_present")
                for key in ("name", "candidate_index", "candidate_original232_index"):
                    ledger.require(case["product"][key] == menu_product[key], "Q1/q2_product_semantics/" + key)
                ledger.close(case["product"]["weights"], menu_product["weights"], "Q1/q2_product_semantics/weights", atol=0, rtol=0)
            else: ledger.require(case.get("product") is None, "Q1/outside_Balanced_no_Q2_product")
        if "q1_review_details" in refit:
            detail = refit["q1_review_details"]["C"]
            ledger.require(case.get("q1_review_kind") == detail.get("q1_review_kind")
                           and case.get("q1_review_reasons", []) == detail.get("q1_review_reasons", []), "Q1/original_review_reasons")
        gamma = refit["selection"]["selected_gamma"]
        ledger.require(gamma is None or gamma in (0., .025, .05, .075, .1), "Q1/frozen_gamma_family")
        target = None; k_b = None
        if expected_case == "balanced_case":
            item = case["product"]; local, original = item["candidate_index"], item["candidate_original232_index"]
            ledger.require(isinstance(local, int) and not isinstance(local, bool) and 0 <= local < 172 and original == int(allowed[local]), "Q1/candidate_identity")
            ledger.close(item["weights"], weights[original], "Q1/exact_saved_target", atol=2e-15, rtol=0)
            target = item["weights"]
            k_b = int(np.sum(cache["H30__mdd"][np.asarray(job["training_ids"]) - 1, original] - .4 > EVENT_TOL))
        tc = training_choices[jid]
        ledger.require(tc["training_ids"] == job["training_ids"] and tc["evaluation_ids"] == job["evaluation_ids"], "choices/original_membership")
        ledger.require(tc["q1_effective_assignment"] == assignment and tc["case_status"] == expected_case, "choices/Q1_case")
        stats = []
        for order, delta in enumerate(DRIFTS):
            metrics = []
            for sid in job["training_ids"]:
                row = train_rows[(jid, order, sid)]; identity(row, job, b, order=order)
                maybe(row["execution_candidate_order"], order, ledger, "training/candidate_order")
                maybe(row["scope"], "training_support", ledger, "training/scope")
                metric = physical(sid, target, delta)[1] if target is not None else None
                compare_metric_row(row, metric, ledger, available=target is not None, status="automated" if target is not None else expected_case,
                                   reason=None if target is not None else expected_case)
                if metric is not None: metrics.append(metric)
                checked_training += 1
            summary = train_summaries[(jid, order)]
            expected = dict(execution_candidate_order=order, candidate_order=order, tv_threshold=delta, delta=delta,
                            n_training=len(job["training_ids"]), case_status=expected_case,
                            simulation_status="evaluated" if metrics else expected_case,
                            breach_count=sum(bool(m["breach"]) for m in metrics) if metrics else None)
            for field, source in (("mean_log_growth", "log_growth"), ("mean_excess", "excess"), ("mean_nominal_fees", "nominal_fees"),
                                  ("mean_nominal_moved", "nominal_moved"), ("mean_trade_fraction", "trade_fraction"), ("mean_fee_fraction", "mean_fee_fraction")):
                expected[field] = float(np.mean([m[source] for m in metrics])) if metrics else None
            for field, value in expected.items(): maybe(summary[field], value, ledger, "training_summary/" + field)
            stats.append(dict(candidate_order=order, delta=delta, n=len(job["training_ids"]), breach_count=expected["breach_count"],
                              mean_log_growth=expected["mean_log_growth"], mean_excess=expected["mean_excess"], mean_nominal_fees=expected["mean_nominal_fees"]))
        if target is not None: ledger.require(stats[0]["breach_count"] == k_b, "training/independent_B_first_event_equals_Q1_cache")
        maybe(tc["reference_training_breach_count"], k_b, ledger, "choices/recomputed_B_count")
        if job["stage"] == "refit":
            for policy in LEARNERS:
                receipt = independent_four_inner_receipt(expected_inner[name][policy], contexts[name]["training_ids"])
                expected_gate[(name, policy)] = receipt
                saved = gate_rows[(name, policy)]
                expected_values = dict(original_n=receipt["original_n"], auto_inner_n=receipt["automatic_n"],
                                       selected_breach_count=receipt["breach_count"], same_U_B_breach_count=receipt["reference_breach_count_same_automatic_ids"],
                                       absolute_alpha_count_cap=receipt["effective_gate_cap"], eligible=receipt["inner_gate_eligible"],
                                       conditional_mean_log_growth=receipt["conditional_mean_log_growth"],
                                       conditional_breach_probability=receipt["breach_count"] / receipt["automatic_n"] if receipt["automatic_n"] else None,
                                       conditional_B_breach_probability=receipt["reference_breach_count_same_automatic_ids"] / receipt["automatic_n"] if receipt["automatic_n"] else None)
                for field, value in expected_values.items(): maybe(saved[field], value, ledger, "inner_gate/" + field)
                u = sorted(sid for entry in expected_inner[name][policy] if entry["automated"] for sid in entry["evaluation_ids"])
                if isinstance(saved["auto_set_scenario_ids"], str): saved_u = json.loads(saved["auto_set_scenario_ids"])
                else: saved_u = saved["auto_set_scenario_ids"]
                ledger.require(saved_u == u, "inner_gate/exact_U")
                reasons = []
                if not receipt["automatic_n"]: reasons.append("no_inner_auto_evidence")
                else:
                    if not receipt["empirical_risk_eligible"]: reasons.append("inner_absolute_risk_exceeds_reference")
                    if not receipt["paired_B_dominance_eligible"]: reasons.append("inner_risk_exceeds_same_U_B")
                reason = reasons[0] if reasons else "passed_inner_gate"
                maybe(saved["reason"], reason, ledger, "inner_gate/priority_reason")
                receipt["reason"] = reason
                nested = context_selections[name]["inner_gates"][policy]
                for field, value in expected_values.items(): maybe(nested[field], value, ledger, "context_gate/" + field)
                ledger.require(nested["auto_set_scenario_ids"] == u and nested["reasons"] == reasons, "context_gate/U_and_reasons")
        for policy, family in FAMILIES.items():
            performed = target is not None and (job["stage"] == "inner" or expected_gate[(name, policy)]["inner_gate_eligible"])
            selection = independent_training_choice(stats, len(job["training_ids"]), k_b, candidate_orders=family) if performed else None
            order = selection["selected_candidate_order"] if selection else None
            if target is None: status, reason = expected_case, expected_case
            elif not performed: status, reason = "execution_review", expected_gate[(name, policy)]["reason"]
            elif order is None: status, reason = "execution_review", "no_admissible_training_candidate"
            else: status, reason = "automated", None
            saved_choice = tc["methods"][policy]
            for field, value in dict(selection_performed=performed, selected_candidate_order=order,
                                     selected_threshold=None if order is None else DRIFTS[order], execution_status=status, execution_reason=reason).items():
                maybe(saved_choice[field], value, ledger, "selection/" + field)
            ledger.require(saved_choice["family_candidate_orders"] == list(family), "selection/fixed_architecture_family")
            if performed:
                for field, value in dict(n_training=len(job["training_ids"]), reference_training_breach_count=k_b,
                                         training_breach_cap=selection["effective_cap"], maximum_eligible_mean_log_growth=selection["maximum_eligible_mean_log_growth"]).items():
                    maybe(saved_choice[field], value, ledger, "selection/" + field)
                band = [s["candidate_order"] for s in stats if s["candidate_order"] in family and s["breach_count"] <= selection["effective_cap"]
                        and s["mean_log_growth"] >= selection["maximum_eligible_mean_log_growth"] - GROWTH_TOL] if selection["maximum_eligible_mean_log_growth"] is not None else []
                ledger.require(saved_choice["global_growth_tied_candidate_orders"] == band, "selection/global_growth_band")
            selected_metrics = []
            for sid in job["evaluation_ids"]:
                row = inner_rows[(jid, policy, sid)] if job["stage"] == "inner" else actual_rows[(name, policy, sid)]
                identity(row, job, b, policy=policy)
                maybe(row["selected_candidate_order"], order, ledger, "heldout/training_only_choice")
                for field in ("candidate_order", "delta", "tv_threshold"):
                    maybe(row[field], None if order is None else order if field == "candidate_order" else DRIFTS[order], ledger, "heldout/" + field)
                metric = physical(sid, target, DRIFTS[order])[1] if status == "automated" else None
                compare_metric_row(row, metric, ledger, available=status == "automated", status=status, reason=reason)
                if metric is not None: selected_metrics.append(metric)
                if job["stage"] == "inner": checked_inner += 1
                else: checked_actual += 1
            if job["stage"] == "inner":
                expected_inner[name][policy].append(dict(fold=job["dynamic_inner_fold"], evaluation_ids=job["evaluation_ids"], automated=status == "automated",
                    breach_count=sum(bool(m["breach"]) for m in selected_metrics) if selected_metrics else None,
                    reference_breach_count=sum(bool(physical(sid, target, 0.)[1]["breach"]) for sid in job["evaluation_ids"]) if selected_metrics else None,
                    mean_log_growth=float(np.mean([m["log_growth"] for m in selected_metrics])) if selected_metrics else None))
            else:
                cr = context_selections[name]
                ledger.require(cr["training_ids"] == job["training_ids"] and cr["evaluation_ids"] == job["evaluation_ids"] and cr["case"] == case, "context/final_Q1_and_membership")
                ledger.require(cr["methods"][policy] == saved_choice, "context/same_training_only_choice")
                ar = admission_rows[(name, policy)]
                for field, value in dict(training_n=96, evaluation_n=24, initial_admitted=status == "automated", inner_gate_passed=expected_gate[(name, policy)]["inner_gate_eligible"],
                                         selected_candidate_order=order, selected_threshold=None if order is None else DRIFTS[order], execution_status=status, execution_reason=reason).items():
                    maybe(ar[field], value, ledger, "fit_admission/" + field)
            if status == "automated":
                scope = "validation" if job["stage"] == "inner" else "actual"
                expected_groups[scope][jid + "__" + policy] = (job, target, DRIFTS[order], True, order, policy)
        # References are audited separately from service admission and never
        # fill the learner's unavailable financial records.
        for sid in job["evaluation_ids"]:
            row = inner_B[(jid, sid)] if job["stage"] == "inner" else actual_rows[(name, REFERENCES[0], sid)]
            identity(row, job, b, policy=REFERENCES[0], order=0)
            metric = physical(sid, target, 0.)[1] if target is not None else None
            compare_metric_row(row, metric, ledger, available=target is not None,
                               status="hypothetical_reference" if target is not None else "reference_unavailable", reason=None if target is not None else expected_case)
            maybe(row["reference_only"], True, ledger, "reference/B_hypothetical_only")
            if job["stage"] == "refit": checked_actual += 1
            if job["stage"] == "refit":
                equal = actual_rows[(name, REFERENCES[1], sid)]; identity(equal, job, b, policy=REFERENCES[1], equal_reference=True)
                metric = physical(sid, np.ones(3) / 3, 0., False)[1]
                compare_metric_row(equal, metric, ledger, available=True, status="hypothetical_reference")
                maybe(equal["reference_only"], True, ledger, "reference/EQ_hypothetical_only"); checked_actual += 1
        if target is not None:
            scope = "validation" if job["stage"] == "inner" else "actual"
            suffix = "__inner_B" if job["stage"] == "inner" else "__B"
            expected_groups[scope][jid + suffix] = (job, target, 0., True, 0, REFERENCES[0])
        if job["stage"] == "refit": expected_groups["actual"][jid + "__equal"] = (job, np.ones(3) / 3, 0., False, None, REFERENCES[1])

    for scope in ("actual", "validation"):
        ledger.require(set(trace_groups[scope]) == set(expected_groups[scope]), "trace/complete_selected_group_inventory")
        archive = archives[scope]
        ledger.require(all(any(k.startswith(g + "__") for g in expected_groups[scope]) for k in archive), "trace/no_orphan_arrays")
        ledger.require(not any(any(bad in k.lower() for bad in ("qg_", "qbad", "predicted_score", "annual_prices", "forecast_n")) for k in archive), "trace/no_fictional_Q_or_price")
        for group, (job, target, delta, pending, order, policy) in expected_groups[scope].items():
            metadata = trace_groups[scope][group]
            ledger.require(metadata["scenario_ids"] == job["evaluation_ids"], "trace/original_ID_order")
            ledger.require(metadata["metadata"]["job_id"] == job["job_id"] and metadata["metadata"]["policy"] == policy, "trace/job_policy_identity")
            maybe(metadata["selected_candidate_order"], order, ledger, "trace/physical_candidate_order")
            # Filter names before accessing compressed NPZ values. Iterating
            # items() first would decompress every other group's arrays again.
            arm = {name[len(group) + 2:]: archive[name] for name in archive if name.startswith(group + "__")}
            if order is not None:
                ledger.require("decision_reason" in arm and "loss_tolerances" in arm, "trace/simple_control_reason_and_loss_present")
            if "loss_tolerances" in arm: ledger.close(arm["loss_tolerances"], [.4], "trace/original_loss_tolerance", atol=0, rtol=0)
            for i, sid in enumerate(job["evaluation_ids"]):
                expected, _ = physical(sid, target, delta, pending)
                saved = {field: arm[field] if field == "targets" else arm[field][i] for field in expected
                         if field not in ("loss_tolerance", "delta") and field in arm}
                compare_saved_account(saved, expected, ledger)
                checked_trace_paths += 1

    # Every financial record used below has been checked against independent
    # physical replay or checked for unavailable NULLs above.
    fit_summary = index_rows(tables["fit_summary"], ("context", "policy"))
    summary = index_rows(tables["summary"], ("repeat_id", "policy"))
    actual_list = list(actual_rows.values())
    for name in contexts:
        for policy in (*LEARNERS, *REFERENCES):
            rows = [r for r in actual_list if r["context"] == name and r["policy"] == policy]
            for field, value in expected_performance(rows, policy).items(): maybe(fit_summary[(name, policy)][field], value, ledger, "fit_summary/" + field)
    for repeat in range(1, 6):
        for policy in (*LEARNERS, *REFERENCES):
            rows = [r for r in actual_list if r["repeat_id"] == repeat and r["policy"] == policy]
            for field, value in expected_performance(rows, policy).items(): maybe(summary[(repeat, policy)][field], value, ledger, "repeat_summary/" + field)
    pairs = index_rows(tables["paired_paths"], ("repeat_id", "learner_policy", "reference_policy", "scenario_id"))
    paired = index_rows(tables["paired_summary"], ("repeat_id", "learner_policy", "reference_policy"))
    for repeat in range(1, 6):
        for left, right in PAIRINGS:
            expected_deltas = []; left_n = 0; right_n = 0
            for sid in range(1, 121):
                left_rows = [r for r in actual_list if r["repeat_id"] == repeat and r["policy"] == left and r["scenario_id"] == sid]
                right_rows = [r for r in actual_list if r["repeat_id"] == repeat and r["policy"] == right and r["scenario_id"] == sid]
                ledger.require(len(left_rows) == 1 and len(right_rows) == 1, "pair/exact_original_IDs")
                a, b = left_rows[0], right_rows[0]; both = bool(a["assigned"] and b["assigned"])
                row = pairs[(repeat, left, right, sid)]; left_n += int(a["assigned"]); right_n += int(b["assigned"])
                for field, value in dict(learner_auto=bool(a["assigned"]), reference_available=bool(b["assigned"]), common_auto=both,
                                         reference_is_hypothetical=right in REFERENCES, context=a["context"], outer_fold=a["outer_fold"]).items():
                    maybe(row[field], value, ledger, "pair/" + field)
                deltas = {field: float(a[field]) - float(b[field]) if both else None for field in PAIR_FIELDS}
                for field, value in deltas.items(): maybe(row["delta_" + field], value, ledger, "pair/delta_" + field)
                if both: expected_deltas.append(deltas)
            s = paired[(repeat, left, right)]
            for field, value in dict(original_pairs=120, learner_auto_n=left_n, reference_available_n=right_n, common_auto_n=len(expected_deltas), reference_is_hypothetical=right in REFERENCES).items():
                maybe(s[field], value, ledger, "paired_summary/" + field)
            for field in PAIR_FIELDS:
                value = float(np.mean([r[field] for r in expected_deltas])) if expected_deltas else None
                maybe(s["mean_delta_" + field], value, ledger, "paired_summary/delta_" + field)
    per_id = index_rows(tables["per_scenario_stability"], ("policy", "scenario_id"))
    for policy in LEARNERS:
        for sid in range(1, 121):
            rows = [r for r in actual_list if r["policy"] == policy and r["scenario_id"] == sid]
            ledger.require(len(rows) == 5 and {r["repeat_id"] for r in rows} == set(range(1, 6)), "per_ID/same_original_ID_five_refits")
            auto = [r for r in rows if r["assigned"]]; saved = per_id[(policy, sid)]
            for field, value in dict(repeat_records_n=5, admitted_refits_n=len(auto), breached_among_admitted_refits_n=sum(bool(r["breach"]) for r in auto) if auto else None,
                                     min_log_growth=min(r["log_growth"] for r in auto) if auto else None, max_log_growth=max(r["log_growth"] for r in auto) if auto else None).items():
                maybe(saved[field], value, ledger, "per_ID/" + field)
    ledger.require(checked_training == 28800 and checked_inner == 4800 and checked_actual == 2400, "inventory/all_saved_path_rows_checked")
    return dict(status="passed", checks_passed=ledger.checks_passed, maximum_absolute_errors=ledger.maximum_absolute_errors,
                membership=membership, checked_training_metric_rows=checked_training, checked_inner_method_rows=checked_inner,
                checked_inner_B_reference_rows=len(inner_B), checked_outer_all_policy_rows=checked_actual, checked_selected_trace_accounts=checked_trace_paths,
                unique_scalar_physical_accounts_replayed=len(replay_cache), unique_scalar_transitions=len(replay_cache) * 30,
                cache_key="Unrounded full normalized target tuple, delta, original SID, L, cost, H, pending mode, initial wealth and guarded data-array SHA",
                q1_audit_scope="Immutable parent component audit referenced; own context/membership/effective case/target/candidate/cache B first-event count checked; no new all172 static-mask optimization audit",
                interpretation="Independent numerical algorithm checks; no independent human review, new samples, forecast-Q validation, or future risk certificate")


def prepare_audit_plan():
    """Freeze the audit's own scope/sources before parsing new market values."""
    if (OUT / "independent_audit.json").exists(): raise RuntimeError("Never overwrite a completed independent audit")
    if (OUT / "audit_failure_receipt.json").exists(): raise RuntimeError("Preserved failed audit requires an explicit revised audit plan")
    if not AUDIT_PLAN.exists() and OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Audit plan must precede the new numerical study; do not backfill preregistration")
    p = read_json(PROTOCOL)
    if p.get("execution_ready") is not True: raise RuntimeError("Frozen ready protocol required before preparing the numerical audit")
    source = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
              "tests/test_audit_simple_execution_comparison.py": sha(ROOT / "tests/test_audit_simple_execution_comparison.py")}
    content = dict(version="simple-execution-independent-audit-plan-v1", protocol_sha256=sha(PROTOCOL),
                   prepared_before_new_numeric_study=True, audit_sources_sha256=source,
                   allowed_scenario_ids=list(range(1, 121)), allow_final=False,
                   intended_scope=dict(training_metric_rows=28800, training_summaries=375, selected_inner_method_rows=4800,
                                       inner_B_reference_rows=2400, parent_architecture_gates=50, outer_all_policy_rows=2400,
                                       parent_contexts=25, dynamic_jobs=125, static_membership_subfolds=500,
                                       all_selected_physical_trace_arrays=True, all_paired_and_original_ID_aggregates=True),
                   independent_methods=["Own100-step scalar fee bisection", "Own strict TV gate using pretrade holdings", "Full original-H CPI/peak/absorbing event replay", "All-interval maximum drawdown", "Independent global growth-band and exact severity/actual-fee/order selection", "Same automatic-ID U risk and B gate"],
                   physical_cache_key="Guarded data-array SHA, full unrounded normalized target float tuple, delta, original SID, L, cost, H, pending mode, initial wealth",
                   q1_scope="Reference the completed parent all172-mask audit; independently recheck context/membership/case/raw menu/target/local-global candidate and annual B first-event cache identity",
                   no_new_policy_selection=True, no_forecast_or_Q_validation=True,
                   failure_policy="No successful audit overwrite. Failed attempts preserve audit_failure_receipt.json and require an explicit revised plan before another numerical audit.")
    target = AUDIT_PLAN
    if target.exists():
        existing = read_json(target)
        if {k: existing[k] for k in content} != content: raise RuntimeError("Existing audit plan differs; preserve it and explicitly revise before new numerical work")
    else:
        content["prepared_utc"] = datetime.now(timezone.utc).isoformat()
        target.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return target


def main():
    if (OUT / "independent_audit.json").exists(): raise RuntimeError("Never overwrite a completed independent audit")
    if (OUT / "audit_failure_receipt.json").exists(): raise RuntimeError("Preserved failed audit requires an explicit revised audit plan")
    p = read_json(PROTOCOL)
    if p.get("execution_ready") is not True: raise RuntimeError("Actual-study audit requires the frozen ready protocol")
    manifest = read_json(OUT / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("training_tasks") != 125: raise RuntimeError("Complete frozen125-job archive required")
    if manifest["protocol_sha256"] != sha(PROTOCOL): raise RuntimeError("Executed protocol snapshot changed")
    parent_protocol = read_json(ROOT / "docs/refit_stability_protocol.json")
    for key in ("contexts", "jobs", "static_contexts"):
        if p[key] != parent_protocol[key]: raise RuntimeError("Immutable parent membership changed")
    parent_audit = ROOT / "results/model_review/refit_stability_v1/independent_audit.json"
    if read_json(parent_audit)["status"] != "passed": raise RuntimeError("Parent Q1 component audit required")
    for name, expected in p["execution_sources_sha256"].items():
        if sha(ROOT / name) != expected: raise RuntimeError("Frozen execution source changed: " + name)
    for name, expected in p["frozen_input_sha256"].items():
        if sha(ROOT / name) != expected: raise RuntimeError("Frozen source/input changed: " + name)
    for name, expected in manifest["outputs_sha256"].items():
        if sha(OUT / name) != expected: raise RuntimeError("Original executed output changed: " + name)
    plan = prepare_audit_plan()
    snapshot_paths = [PROTOCOL, OUT / "manifest.json", plan, Path(__file__).resolve(), ROOT / "tests/test_audit_simple_execution_comparison.py", CACHE, parent_audit]
    snapshot_paths += [ROOT / name for name in p["execution_sources_sha256"]]
    snapshot_paths += [ROOT / name for name in p["frozen_input_sha256"]]
    snapshot_paths += [OUT / name for name in manifest["outputs_sha256"]]
    snapshot = {str(path.relative_to(ROOT)): sha(path) for path in snapshot_paths}
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / p["data"]["path"], p["data"]["scenario_ids"], allow_final=False)
    digest = hashlib.sha256()
    for a in (ids, returns, inflation): digest.update(np.ascontiguousarray(a).tobytes())
    if ids.tolist() != list(range(1, 121)) or manifest["final_scenarios_loaded"] or digest.hexdigest() != manifest["development_arrays_sha256"]:
        raise RuntimeError("Guarded development-array identity changed")
    tables = {name: observed_bools(pd.read_csv(OUT / (name + ".csv.gz"))) for name in manifest["row_counts"]}
    bundles = read_json(ROOT / p["q1_component_cache"]["path"])
    with np.load(CACHE, allow_pickle=False) as raw_cache: cache = {name: raw_cache[name] for name in ("weights", "H30__mdd")}
    with np.load(OUT / "actual_traces.npz", allow_pickle=False) as actual, np.load(OUT / "validation_traces.npz", allow_pickle=False) as validation:
        result = audit_pipeline(p, bundles, returns, inflation, ids, tables, read_json(OUT / "training_choices.json.gz"),
                                read_json(OUT / "context_selections.json"), read_json(OUT / "trace_groups.json"),
                                {"actual": actual, "validation": validation}, cache, data_digest=digest.hexdigest())
    end = {str(path.relative_to(ROOT)): sha(path) for path in snapshot_paths}
    if snapshot != end: raise RuntimeError("Frozen source/input/archive changed during audit")
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(Path(__file__)),
                  parent_manifest_sha256=sha(OUT / "manifest.json"), protocol_sha256=sha(PROTOCOL),
                  sources_and_artifacts_at_start=snapshot, sources_and_artifacts_at_end=end,
                  scenario_ids_loaded=ids.tolist(), final_scenarios_loaded=False)
    target = OUT / "independent_audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "checks_passed": result["checks_passed"], "audit_sha256": sha(target),
                      "unique_scalar_physical_accounts_replayed": result["unique_scalar_physical_accounts_replayed"]}))


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--prepare-plan"]:
            path = prepare_audit_plan(); print(json.dumps({"status": "prepared_before_numeric_audit", "audit_plan_sha256": sha(path)}))
        elif sys.argv[1:]: raise SystemExit("Use no argument for audit, or --prepare-plan for the source/scope snapshot")
        else: main()
    except Exception as error:
        if OUT.exists() and AUDIT_PLAN.exists() and not (OUT / "independent_audit.json").exists() and not (OUT / "audit_failure_receipt.json").exists():
            receipt = dict(status="failed_audit_attempt_preserved", completed_utc=datetime.now(timezone.utc).isoformat(),
                           audit_plan_sha256=sha(AUDIT_PLAN), auditor_sha256=sha(Path(__file__)),
                           error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc(),
                           successful_audit_written=False, original_numeric_archive_modified_by_auditor=False)
            (OUT / "audit_failure_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        raise
