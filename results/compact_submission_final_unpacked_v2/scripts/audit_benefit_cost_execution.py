"""Independent scalar audit primitives for the prospective benefit-cost model.

No new execution kernel, selector, market loader or study artifact is imported.
Only the previously frozen independent fee bisection is reused. Forecast
branches below retain actual inherited CPI, peak, MDD and original calendar.
The bounded actual-study reader follows the pre-execution locked audit plan.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_simple_execution_comparison import independent_fee_trade, AuditLedger
import audit_simple_execution_comparison as accounting

TOL = 1e-12
PROTOCOL = ROOT / "docs/benefit_cost_execution_protocol.json"
AUDIT_PLAN = ROOT / "docs/benefit_cost_execution_audit_plan.json"
OUT = ROOT / "results/model_review/benefit_cost_execution_v1"
CACHE = ROOT / "results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz"
METHODS = {"benefit_cost_one_year": "one_year", "benefit_cost_remaining_H": "remaining_to_original_H"}
REFERENCES = ("annual_then_pending", "equal_thirds_annual")
QFLOATS = ("mean_gross_log_gain", "mean_net_log_gain", "fee_log_penalty", "quoted_nominal_fee", "quoted_nominal_moved", "quoted_fee_fraction")
QCOUNTS = ("risk_hold_breach_count", "risk_restore_breach_count", "forecast_n", "benefit_periods")


def _integer(value, name, lower=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
        raise ValueError(name)
    return int(value)


def _input_state(holdings, cpi, peak, worst, target, loss, year_index, horizon):
    x, w = np.asarray(holdings, float), np.asarray(target, float)
    t = _integer(year_index, "Original zero-based decision year required")
    h = _integer(horizon, "Positive original horizon required", 1)
    values = [cpi, peak, worst, loss]
    if (x.shape != (3,) or w.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(x < 0) or x.sum() <= 0 or np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-13, rtol=0)
            or any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) for v in values)
            or cpi <= 0 or peak <= 0 or not 0 < loss < 1 or not 0 <= worst <= 1 or not t < h):
        raise ValueError("Finite consistent actual state, target and original time required")
    real = float(x.sum()) / float(cpi)
    if peak < real - 1e-10 or worst < max(1 - real / peak, 0.) - 1e-12:
        raise ValueError("Inherited real peak and historical MDD cannot be reset")
    return x, w / w.sum(), t, h


def scalar_buy_hold_branch(post_holdings, original_current_wealth, cpi, peak, worst, remaining_returns, remaining_inflation,
                           *, loss, original_year_index, economic_periods):
    """One current action already executed, then fixed quantities to original H.

    Current fee is represented only in post_holdings. Annual endpoints observe
    loss with the unchanged historical peak, never a newly normalized account.
    """
    r, pi = np.asarray(remaining_returns, float), np.asarray(remaining_inflation, float)
    if (r.ndim != 2 or r.shape[1] != 3 or not len(r) or pi.shape != (len(r),)
            or not np.isfinite(r).all() or not np.isfinite(pi).all() or np.any(r <= -1) or np.any(pi <= -1)):
        raise ValueError("Finite positive-gross joint suffix required")
    k = _integer(economic_periods, "Positive predeclared economic horizon required", 1)
    if k > len(r): raise ValueError("Economic horizon cannot exceed the original remaining H")
    x = list(map(float, post_holdings)); cp = float(cpi); p = float(peak); d = float(worst)
    if len(x) != 3 or min(x) < 0 or sum(x) <= 0: raise ValueError("Positive post-action account required")
    real_values = [float(original_current_wealth) / cp]; peaks = [p]; worst_values = [d]
    bad = d - loss > TOL; first = None
    for j in range(len(r)):
        x = [x[a] * (1 + float(r[j, a])) for a in range(3)]
        cp = cp * (1 + float(pi[j]))
        real = sum(x) / cp; p = max(p, real); d = max(d, 1 - real / p)
        if not bad and d - loss > TOL: first = original_year_index + j + 1
        bad = bool(bad or d - loss > TOL)
        real_values.append(real); peaks.append(p); worst_values.append(d)
    return dict(real_wealth=np.asarray(real_values), real_peak=np.asarray(peaks), running_mdd=np.asarray(worst_values),
                ever_breached=bad, first_crossing_original_year=first, economic_real=real_values[k], terminal_real=real_values[-1],
                economic_total_log_growth=math.log(real_values[k] / real_values[0]),
                full_remaining_total_log_growth=math.log(real_values[-1] / real_values[0]))


def independent_train_comparison(holdings, cpi, peak, worst, target, training_returns, training_inflation, *,
                                 year_index, horizon, economic_periods, loss=.4, cost_rate=.0025):
    """Evaluate all supplied training suffixes from exactly one current state.

    The economic horizon and full original-H risk horizon are separate. This
    is an empirical-law arithmetic audit, not true conditional calibration.
    """
    x, w, t, h = _input_state(holdings, cpi, peak, worst, target, loss, year_index, horizon)
    if worst - loss > TOL: raise ValueError("Already breached actual states must be pending, not re-estimated")
    r, pi = np.asarray(training_returns, float), np.asarray(training_inflation, float)
    if r.ndim != 3 or r.shape[1:] != (h, 3) or not len(r) or pi.shape != r.shape[:2]:
        raise ValueError("Exactly the original-H complete joint training paths required")
    if not np.isfinite(r).all() or not np.isfinite(pi).all() or np.any(r <= -1) or np.any(pi <= -1):
        raise ValueError("Finite positive-gross training paths required")
    k = _integer(economic_periods, "Positive economic horizon required", 1)
    if k > h - t: raise ValueError("Economic horizon exceeds actual remaining original H")
    post, fee, moved = independent_fee_trade(x, w, cost_rate=cost_rate)
    W = float(x.sum()); rho = float(fee) / W; log_cost = -math.log1p(-rho)
    gross_values = []; net_values = []; hold_values = []; restore_values = []; hold_branches = []; restore_branches = []
    for i in range(len(r)):
        common = dict(loss=loss, original_year_index=t, economic_periods=k)
        hold = scalar_buy_hold_branch(x, W, cpi, peak, worst, r[i, t:], pi[i, t:], **common)
        restore = scalar_buy_hold_branch(post, W, cpi, peak, worst, r[i, t:], pi[i, t:], **common)
        cum = [math.prod(1 + float(r[i, j, a]) for j in range(t, t + k)) for a in range(3)]
        gross = math.log(sum(float(w[a]) * cum[a] for a in range(3)) / sum(float(x[a] / W) * cum[a] for a in range(3)))
        net = math.log(restore["economic_real"] / hold["economic_real"])
        if not math.isclose(net, gross - log_cost, rel_tol=0, abs_tol=4e-14):
            raise ValueError("Exact one-fee buy-hold decomposition failed")
        gross_values.append(gross); net_values.append(net)
        hold_values.append(hold["economic_total_log_growth"]); restore_values.append(restore["economic_total_log_growth"])
        hold_branches.append(hold); restore_branches.append(restore)
    kh = sum(b["ever_breached"] for b in hold_branches); kr = sum(b["ever_breached"] for b in restore_branches)
    gross = float(np.mean(gross_values)); net = float(np.mean(net_values))
    return dict(mean_gross_log_gain=gross, mean_net_log_gain=net, fee_log_penalty=log_cost,
                quoted_nominal_fee=float(fee), quoted_nominal_moved=float(moved), quoted_fee_fraction=rho,
                forecast_n=len(r), benefit_periods=k, risk_hold_breach_count=kh, risk_restore_breach_count=kr,
                economic_pass=net > TOL, risk_direction_pass=kr <= kh,
                sample_gross_log_gain=np.asarray(gross_values), sample_net_log_gain=np.asarray(net_values),
                sample_hold_economic_log_growth=np.asarray(hold_values), sample_restore_economic_log_growth=np.asarray(restore_values),
                hold_branches=hold_branches, restore_branches=restore_branches)


def independent_benefit_action(*, year_index, ever_breached, mean_net_log_gain=None, hold_bad=None, restore_bad=None, forecast_n=None):
    """Initial positive-gain contract, then positive gain plus full-H risk veto."""
    t = _integer(year_index, "Original zero-based decision year required")
    if not isinstance(ever_breached, (bool, np.bool_)): raise ValueError("Observed absorbing breach flag required")
    if ever_breached: return "hold", "already_breached_pending_hold"
    if isinstance(mean_net_log_gain, (bool, np.bool_)) or mean_net_log_gain is None or not np.isfinite(mean_net_log_gain):
        raise ValueError("Active economic estimate must be finite")
    n = _integer(forecast_n, "Positive original training-scenario denominator required", 1)
    kh = _integer(hold_bad, "Integer hold risk count required"); kr = _integer(restore_bad, "Integer restore risk count required")
    if max(kh, kr) > n: raise ValueError("Risk counts cannot exceed common training denominator")
    economic = mean_net_log_gain > TOL
    if t == 0:
        if not economic: raise ValueError("Initial economic rejection must prevent model deployment")
        return "restore", "mandatory_initial_restore_economic_positive"
    if economic and kr <= kh: return "restore", "positive_net_and_risk_direction_restore"
    return "hold", "estimated_net_nonpositive_hold" if not economic else "full_H_risk_direction_veto_hold"


def _finish_account(account, target, *, loss=.4):
    result = {name: np.asarray(values) for name, values in account.items()}
    result["targets"] = np.asarray(target, float)
    crossed = np.flatnonzero(result["ever_breached"])
    tau = int(crossed[0]) if len(crossed) else None
    result["first_crossing"] = -1 if tau is None else tau
    h = len(result["fees"]); recovered_at = None
    if tau is not None:
        recovered_at = next((t for t in range(tau + 1, h + 1) if result["real_wealth"][t] >= result["real_peak"][tau]), None)
    result.update(triggered=tau is not None, recovered=recovered_at is not None,
                  drawdown_at_trigger=np.nan if tau is None else float(result["running_mdd"][tau]),
                  peak_at_trigger=np.nan if tau is None else float(result["real_peak"][tau]),
                  real_wealth_at_trigger=np.nan if tau is None else float(result["real_wealth"][tau]),
                  recovery_endpoint=-1 if recovered_at is None else recovered_at,
                  recovery_time=-1 if recovered_at is None else recovered_at - tau,
                  right_censored=tau is not None and recovered_at is None,
                  available_followup=-1 if tau is None else h - tau,
                  observed_followup=-1 if tau is None else recovered_at - tau if recovered_at is not None else h - tau,
                  added_severity=np.nan if tau is None else max(float(result["running_mdd"][-1] - result["running_mdd"][tau]), 0.),
                  terminal_excess=max(float(result["running_mdd"][-1]) - loss, 0.), end_weights=result["weights"][1:],
                  trade_count=int(np.sum(result["trades"])), target_execution_count=int(np.sum(result["target_execution"])))
    metric = accounting.independent_metrics(result, loss=loss)
    result.update({name: value for name, value in metric.items() if name != "first_crossing"})
    metric["mean_fee_fraction"] = float(np.mean(result["fee_fraction"]))
    return result, metric


def independent_model_replay(actual_returns, actual_inflation, target, train_returns, train_inflation, *, mode,
                              saved_estimates=None, ledger=None):
    """Replay from100; use independent forecasts or audit every saved estimate.

    With saved_estimates=None every active decision is independently forecast.
    With saved estimates, actions, quotes and algebra are independently checked;
    the caller separately reconstructs forecasts at the predeclared subset.
    """
    r, pi = np.asarray(actual_returns, float), np.asarray(actual_inflation, float)
    if r.shape != (30, 3) or pi.shape != (30,) or mode not in METHODS.values():
        raise ValueError("Original complete30-year account and fixed method required")
    w = np.asarray(target, float); w = w / w.sum()
    x = np.array([0., 0., 100.]); cp = 1.; peak = 100.; worst = 0.; bad = False
    names = ("holdings", "weights", "nominal_wealth", "real_wealth", "cpi", "real_peak", "current_drawdown", "running_mdd", "ever_breached",
             "posttrade_holdings", "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction", "pretrade_tv_drift", "end_year_tv_drift",
             "actions", "trades", "target_execution", "pending_at_decision", "decision_reason")
    out = {name: [] for name in names}; estimates = []; independent_forecasts = 0
    ledger = ledger if ledger is not None else AuditLedger()

    def state(current):
        nominal = float(sum(x))
        for key, value in dict(holdings=x.copy(), weights=x / nominal, nominal_wealth=nominal,
                               real_wealth=nominal / cp, cpi=cp, real_peak=peak, current_drawdown=current,
                               running_mdd=worst, ever_breached=bad).items(): out[key].append(value)
    state(0.)
    for t in range(30):
        K = 1 if mode == "one_year" else 30 - t
        available = not bad; e = None; W = float(sum(x))
        if available:
            if saved_estimates is None:
                raw = independent_train_comparison(x, cp, peak, worst, w, train_returns, train_inflation,
                                                   year_index=t, horizon=30, economic_periods=K)
                e = {key: raw[key] for key in (*QFLOATS, *QCOUNTS, "economic_pass", "risk_direction_pass")}
                independent_forecasts += 1
            else:
                e = {key: saved_estimates[key][t].item() for key in (*QFLOATS, *QCOUNTS, "economic_pass", "risk_direction_pass")}
                postquote, feequote, movedquote = independent_fee_trade(x, w)
                for key, value in dict(quoted_nominal_fee=feequote, quoted_nominal_moved=movedquote,
                                       quoted_fee_fraction=feequote / W, fee_log_penalty=-math.log1p(-feequote / W)).items():
                    ledger.close(e[key], value, "decision/" + key)
                ledger.close(e["mean_net_log_gain"], e["mean_gross_log_gain"] - e["fee_log_penalty"], "decision/one_fee_log_decomposition", atol=2e-12, rtol=0)
                ledger.require(e["forecast_n"] == len(train_returns) and e["benefit_periods"] == K, "decision/original_training_denominator_and_K")
                ledger.require(isinstance(e["economic_pass"], (bool, np.bool_)) and e["economic_pass"] == (e["mean_net_log_gain"] > TOL), "decision/economic_boolean")
                ledger.require(isinstance(e["risk_direction_pass"], (bool, np.bool_)) and e["risk_direction_pass"] == (e["risk_restore_breach_count"] <= e["risk_hold_breach_count"]), "decision/risk_direction_boolean")
            if t == 0 and not e["economic_pass"]:
                return dict(status="initial_economic_rejection", estimate=e, independent_forecasts=independent_forecasts)
            action, reason = independent_benefit_action(year_index=t, ever_breached=False,
                    mean_net_log_gain=e["mean_net_log_gain"], hold_bad=e["risk_hold_breach_count"],
                    restore_bad=e["risk_restore_breach_count"], forecast_n=e["forecast_n"])
        else:
            action, reason = independent_benefit_action(year_index=t, ever_breached=True)
        if saved_estimates is not None:
            ledger.require(bool(saved_estimates["estimate_available"][t]) == available, "decision/estimate_available")
            ledger.require(str(saved_estimates["gate_reason"][t]) == reason and str(saved_estimates["decision_reason"][t]) == reason, "decision/reason")
            if not available:
                ledger.require(all(np.isnan(saved_estimates[key][t]) for key in QFLOATS), "decision/pending_NULL_float")
                ledger.require(all(saved_estimates[key][t] == -1 for key in QCOUNTS), "decision/pending_integer_sentinel")
                ledger.require(not saved_estimates["economic_pass"][t] and not saved_estimates["risk_direction_pass"][t], "decision/pending_boolean_sentinel")
        post, fee, moved = independent_fee_trade(x, w) if action == "restore" else (x.copy(), 0., 0.)
        for key, value in dict(posttrade_holdings=post.copy(), fees=fee, moved=moved, pretrade_wealth=W,
                               fee_fraction=fee / W, moved_fraction=moved / W, pretrade_tv_drift=.5 * float(np.abs(x / W - w).sum()),
                               actions=action, trades=moved > TOL, target_execution=action == "restore", pending_at_decision=bad,
                               decision_reason=reason).items(): out[key].append(value)
        estimates.append(e)
        x = np.array([float(post[a]) * (1 + float(r[t, a])) for a in range(3)])
        cp *= 1 + float(pi[t]); real = float(sum(x)) / cp
        peak = max(peak, real); current = max(1 - real / peak, 0.); worst = max(worst, current)
        bad = bool(bad or worst - .4 > TOL)
        out["end_year_tv_drift"].append(.5 * float(np.abs(x / x.sum() - w).sum())); state(current)
    account, metric = _finish_account(out, w)
    return dict(status="complete", account=account, metrics=metric, estimates=estimates,
                independent_forecasts=independent_forecasts)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path): return accounting.read_json(path)
def writej(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_membership(p, parent, full):
    """Original125 disjoint jobs plus the separately named full120 descriptor."""
    if (p["data"]["scenario_ids"] != list(range(1, 121)) or p["data"]["allow_final"] is not False or p["methods"] != METHODS
            or p["primary_method"] != "benefit_cost_one_year" or p["target_updates_enabled"] is not False
            or p["original_H"] != 30 or p["loss_tolerance"] != .4 or p["cost_rate"] != .0025 or p["service_alpha"] != .1):
        raise ValueError("Fixed original-H benefit/cost experiment required")
    if p["contexts"][:25] != parent["contexts"] or p["jobs"][:125] != parent["jobs"]:
        raise ValueError("Original repeated whole-path partitions changed")
    old = {**parent, **{key: p[key] for key in ("original_H", "loss_tolerance", "service_alpha", "cost_rate", "liquidity_floors")},
           "architectures": {"annual_gated_diagnostic": [0.], "drift_band_gated_diagnostic": [0., .05, .1]}}
    accounting.validate_protocol_membership(old)
    if len(p["contexts"]) != 26 or len(p["jobs"]) != 130 or len(p["static_contexts"]) != 130:
        raise ValueError("Complete130-job inventory required")
    for j, original in zip(p["jobs"][125:], full["jobs"][:5]):
        if {k: v for k, v in j.items() if k not in ("repeat_id", "outer_fold")} != original or j["repeat_id"] != 0 or j["outer_fold"] != 0:
            raise ValueError("Full-development job identity changed")
    expected_static = {**parent["static_contexts"], **{k: v for k, v in full["static_contexts"].items() if k.startswith("c0_")}}
    if p["static_contexts"] != expected_static: raise ValueError("Static component memberships changed")
    c = p["contexts"][-1]
    if c != {**p["static_contexts"]["c0_refit"], "repeat_id": 0, "outer_fold": 0, "outer_test_ids": []}:
        raise ValueError("Full120 context identity changed")
    if c["training_ids"] != list(range(1, 121)) or p["jobs"][-1]["evaluation_ids"] != list(range(1, 121)):
        raise ValueError("Full120 in-sample identities changed")
    return dict(repeated_outer_contexts=25, full_descriptive_contexts=1, dynamic_jobs=130,
                repeated_original_records=600, descriptive_in_sample_records=120)


def check_row_arithmetic(row, ledger, *, available):
    if not available:
        for key in accounting.METRICS: accounting.maybe(row[key], None, ledger, "unavailable/" + key)
        return
    for key in accounting.METRICS:
        if key not in ("first_crossing", "breach"): ledger.require(np.isfinite(row[key]), "path/finite/" + key)
    ledger.require(isinstance(row["breach"], (bool, np.bool_)), "path/observed_breach_boolean")
    ledger.close(row["log_growth"], math.log(row["terminal_real"] / 100) / 30, "path/original_H_growth")
    ledger.close(row["cagr"], math.expm1(row["log_growth"]), "path/CAGR")
    ledger.close(row["nominal_fees"], .0025 * row["nominal_moved"], "path/one_way_fee")
    ledger.require(bool(row["breach"]) == (row["mdd"] - .4 > TOL), "path/event")
    ledger.close(row["excess"], max(row["mdd"] - .4, 0.), "path/excess")
    ledger.close(row["terminal_equity"] + row["terminal_bond"] + row["terminal_bill"], 1., "path/terminal_simplex")
    if row["breach"]: ledger.require(row["first_crossing"] in range(1, 31), "path/crossing_original_year")
    else: accounting.maybe(row["first_crossing"], None, ledger, "path/no_crossing_NULL")


def audit_artifacts(p, bundles, r, pi, ids, tables, choices, groups, archives, cache, *, parent, full, clock_start=None):
    """Bounded, predeclared independent forecast and complete saved-tree audit."""
    ledger = AuditLedger(); membership = validate_membership(p, parent, full)
    start = time.monotonic() if clock_start is None else clock_start
    if list(map(int, ids)) != list(range(1, 121)) or r.shape != (120, 30, 3) or pi.shape != (120, 30): raise ValueError("Guarded original120 arrays required")
    for name, count in p["expected_rows"].items(): ledger.require(len(tables[name]) == count, "inventory/" + name)
    if set(bundles) != set(j["job_id"] for j in p["jobs"]) or set(choices) != set(bundles): raise ValueError("Complete130 Q1/choice jobs required")
    weights = cache["weights"]; allowed = np.flatnonzero(np.all(weights >= .05 - TOL, axis=1))
    ledger.require(len(weights) == 232 and len(allowed) == 172, "Q1/strict172_mapping")
    train = accounting.index_rows(tables["training_paths"], ("job_id", "policy", "scenario_id"))
    summ = accounting.index_rows(tables["training_summary"], ("job_id", "policy"))
    inner = accounting.index_rows(tables["inner_validation_paths"], ("job_id", "policy", "scenario_id"))
    inner_B = accounting.index_rows(tables["inner_reference_paths"], ("job_id", "scenario_id"))
    actual = accounting.index_rows(tables["client_paths"], ("job_id", "policy", "scenario_id"))
    gates = accounting.index_rows(tables["inner_gate_summaries"], ("context", "policy"))
    admissions = accounting.index_rows(tables["fit_admission"], ("context", "policy"))
    job_by_id = {j["job_id"]: j for j in p["jobs"]}; contexts = {c["name"]: c for c in p["contexts"]}
    expected_groups = {"actual": {}, "validation": {}}; witnesses = []; q_slots = []; q1_cases = {}; init_estimates = {}
    counters = dict(training_rows=0, initial_forecasts=0, training_policy_witnesses=0, witness_forecast_states=0,
                    selected_trace_accounts=0, selected_trace_decisions=0, independently_recomputed_subset_states=0)

    def budget():
        if time.monotonic() - start > 20 * 60: raise TimeoutError("Fixed20-minute audit budget exceeded; do not shrink predetermined scope")

    def identity(row, job, b, policy, *, equal=False):
        expected = dict(job_id=job["job_id"], context=job["parent_context"], context_id=job["parent_context_id"],
                        repeat_id=job["repeat_id"], outer_fold=job["outer_fold"], client="C", horizon=30, loss_tolerance=.4,
                        case_status=b["q2_case"]["public_status"], q1_effective_assignment=b["q2_case"]["effective_assignment"],
                        policy=policy, selection_training_n=len(job["training_ids"]), static_gamma=b["refit"]["selection"]["selected_gamma"])
        for key, value in expected.items(): accounting.maybe(row[key], value, ledger, "identity/" + key)
        for field in ("reference_only", "hypothetical_reference"):
            accounting.maybe(row[field], policy in REFERENCES, ledger, "identity/" + field)
        product = b["q2_case"]["product"]
        for key, source in (("candidate_local_index", "candidate_index"), ("candidate_original232_index", "candidate_original232_index")):
            accounting.maybe(row[key], None if equal or product is None else product[source], ledger, "identity/" + key)
        if policy in METHODS: accounting.maybe(row["benefit_horizon"], METHODS[policy], ledger, "identity/fixed_benefit_horizon")

    for job in p["jobs"]:
        budget(); jid = job["job_id"]; b = bundles[jid]; case = b["q2_case"]; refit = b["refit"]
        ledger.require(b["context"] == p["static_contexts"][jid] and refit["training_ids"] == job["training_ids"], "Q1/context_and_training_identity")
        expected_static_eval = [] if jid == "c0_refit" else job["evaluation_ids"]
        ledger.require(refit["outer_test_ids"] == expected_static_eval, "Q1/static_evaluation_identity_including_full120_exception")
        label = refit["effective_assignments"]["C"]
        expected_case = {"Balanced": "balanced_case", "Review": "q1_review", "Conservative": "not_balanced_case", "Growth": "not_balanced_case"}[label]
        ledger.require(case["effective_assignment"] == label and case["public_status"] == expected_case, "Q1/effective_case")
        raw_menu = refit["raw_fit"]["menu"] if refit.get("raw_fit") else []
        item = next((x for x in raw_menu if x["name"] == label), None)
        if label == "Review": ledger.require(case["product"] is None and case["q1_product"] is None, "Q1/Review_no_product")
        else:
            ledger.require(item is not None and case["q1_product"] is not None, "Q1/effective_product")
            for field in ("name", "candidate_index", "candidate_original232_index", "weights"):
                ledger.require(case["q1_product"][field] == item[field], "Q1/raw_menu_identity/" + field)
                if label == "Balanced": ledger.require(case["product"][field] == item[field], "Q1/Balanced_identity/" + field)
        if "q1_review_details" in refit:
            detail = refit["q1_review_details"]["C"]
            ledger.require(case.get("q1_review_kind") == detail.get("q1_review_kind") and case.get("q1_review_reasons", []) == detail.get("q1_review_reasons", []), "Q1/original_review_reason")
        target = None; kb = None; tx = np.asarray(job["training_ids"]) - 1
        if expected_case == "balanced_case":
            local, original = item["candidate_index"], item["candidate_original232_index"]
            ledger.require(original == int(allowed[local]), "Q1/local_original_index")
            ledger.close(item["weights"], weights[original], "Q1/exact_target", atol=2e-15, rtol=0)
            target = item["weights"]; kb = int(np.sum(cache["H30__mdd"][tx, original] - .4 > TOL))
        q1_cases[jid] = (b, target)
        choice = choices[jid]
        ledger.require(choice["training_ids"] == job["training_ids"] and choice["evaluation_ids"] == job["evaluation_ids"] and choice["case_status"] == expected_case, "choices/original_identity")
        accounting.maybe(choice["reference_training_breach_count"], kb, ledger, "choices/cache_B_count")
        ledger.require(choice["target"] == target, "choices/exact_target")
        for policy, mode in METHODS.items():
            s = summ[(jid, policy)]; initial = None
            if target is not None:
                raw = independent_train_comparison([0., 0., 100.], 1., 100., 0., target, r[tx], pi[tx],
                         year_index=0, horizon=30, economic_periods=1 if mode == "one_year" else 30)
                initial = {key: raw[key] for key in (*QFLOATS, *QCOUNTS, "economic_pass", "risk_direction_pass")}
                counters["initial_forecasts"] += 1
            init_estimates[(jid, policy)] = initial
            available = initial is not None and initial["economic_pass"]
            accounting.maybe(s["initial_economic_eligible"], bool(available), ledger, "training/initial_economy")
            for field, key in (("initial_mean_net_log_gain", "mean_net_log_gain"), ("initial_mean_gross_log_gain", "mean_gross_log_gain"), ("initial_log_cost", "fee_log_penalty")):
                accounting.maybe(s[field], initial[key] if initial is not None else None, ledger, "training/" + field)
            rows = []
            for sid in job["training_ids"]:
                row = train[(jid, policy, sid)]; identity(row, job, b, policy)
                accounting.maybe(row["assigned"], bool(available), ledger, "training/availability")
                check_row_arithmetic(row, ledger, available=available); rows.append(row); counters["training_rows"] += 1
            kval = sum(bool(row["breach"]) for row in rows) if available else None
            accounting.maybe(s["breach_count"], kval, ledger, "training/breach_aggregate")
            for field, source in (("mean_log_growth", "log_growth"), ("mean_excess", "excess"), ("mean_nominal_fees", "nominal_fees")):
                accounting.maybe(s[field], float(np.mean([row[source] for row in rows])) if available else None, ledger, "training/" + field)
            optional = int(round(sum(row["target_execution_fraction"] * 30 for row in rows) - len(rows))) if available else None
            accounting.maybe(s["optional_restores"], optional, ledger, "training/optional_restores")
            accounting.maybe(s["positive_net_optional_restores"], optional, ledger, "training/positive_optional_restore_count")
            witness = dict(job_id=jid, policy=policy, scenario_id=min(job["training_ids"]), status=None)
            if target is None: witness["status"] = expected_case
            elif not available: witness["status"] = "initial_economic_rejection"
            else:
                replay = independent_model_replay(r[witness["scenario_id"] - 1], pi[witness["scenario_id"] - 1], target, r[tx], pi[tx], mode=mode)
                ledger.require(replay["status"] == "complete", "training_witness/initial_positive_consistency")
                observed = train[(jid, policy, witness["scenario_id"])]
                for field in accounting.METRICS: accounting.maybe(observed[field], replay["metrics"][field], ledger, "training_witness/" + field)
                counters["witness_forecast_states"] += replay["independent_forecasts"]; witness["status"] = "complete_independent_policy_replay"
            witnesses.append(witness); counters["training_policy_witnesses"] += 1
            training_eligible = bool(available and kval <= min(kb, math.floor(.1 * len(tx) + TOL))) if target is not None else False
            saved_choice = choice["methods"][policy]
            accounting.maybe(saved_choice["training_eligible"], training_eligible, ledger, "admission/training_eligible")
            if target is not None:
                for field, value in dict(reference_training_breach_count=kb, training_breach_cap=min(kb, math.floor(.1 * len(tx) + TOL)), n_training=len(tx)).items():
                    accounting.maybe(saved_choice[field], value, ledger, "admission/" + field)
            if expected_case != "balanced_case": status, reason = expected_case, expected_case
            elif not available: status, reason = "execution_review", "mandatory_initial_restore_economic_nonpositive"
            elif not training_eligible: status, reason = "execution_review", "complete_training_risk_exceeds_cap"
            else: status, reason = "automated", None
            if job["stage"] == "refit":
                entries = []
                for inner_job in [j for j in p["jobs"] if j["parent_context_id"] == job["parent_context_id"] and j["stage"] == "inner"]:
                    selected = [inner[(inner_job["job_id"], policy, sid)] for sid in inner_job["evaluation_ids"]]
                    autos = [row for row in selected if row["assigned"]]
                    if len(autos) not in (0, len(selected)): raise ValueError("Inner admission must be uniform across its heldout paths")
                    entries.append(dict(fold=inner_job["dynamic_inner_fold"], evaluation_ids=inner_job["evaluation_ids"], automated=bool(autos),
                        breach_count=sum(bool(row["breach"]) for row in autos) if autos else None,
                        reference_breach_count=sum(bool(inner_B[(inner_job["job_id"], row["scenario_id"])]["breach"]) for row in autos) if autos else None,
                        mean_log_growth=float(np.mean([row["log_growth"] for row in autos])) if autos else None))
                g = accounting.independent_four_inner_receipt(entries, contexts[job["parent_context"]]["training_ids"])
                saved_gate = gates[(job["parent_context"], policy)]; nested = saved_choice["inner_gate"]
                values = dict(original_n=g["original_n"], auto_inner_n=g["automatic_n"], selected_breach_count=g["breach_count"],
                    same_U_B_breach_count=g["reference_breach_count_same_automatic_ids"], absolute_alpha_count_cap=g["effective_gate_cap"],
                    conditional_mean_log_growth=g["conditional_mean_log_growth"], eligible=g["inner_gate_eligible"],
                    conditional_breach_probability=g["breach_count"] / g["automatic_n"] if g["automatic_n"] else None,
                    conditional_B_breach_probability=g["reference_breach_count_same_automatic_ids"] / g["automatic_n"] if g["automatic_n"] else None)
                for field, value in values.items():
                    accounting.maybe(saved_gate[field], value, ledger, "inner_gate/" + field); accounting.maybe(nested[field], value, ledger, "nested_gate/" + field)
                U = sorted(sid for entry in entries if entry["automated"] for sid in entry["evaluation_ids"])
                encoded = saved_gate["auto_set_scenario_ids"]; decoded = json.loads(encoded) if isinstance(encoded, str) else encoded
                ledger.require(decoded == U and nested["auto_set_scenario_ids"] == U, "inner_gate/exact_same_U")
                reasons = []
                if not g["automatic_n"]: reasons.append("no_inner_auto_evidence")
                else:
                    if not g["empirical_risk_eligible"]: reasons.append("inner_absolute_risk_exceeds_reference")
                    if not g["paired_B_dominance_eligible"]: reasons.append("inner_risk_exceeds_same_U_B")
                gate_reason = reasons[0] if reasons else "passed_inner_gate"
                ledger.require(nested["reasons"] == reasons and nested["reason"] == gate_reason, "inner_gate/failure_reason")
                if target is not None and not g["inner_gate_eligible"]: status, reason = "execution_review", gate_reason
                ar = admissions[(job["parent_context"], policy)]
                for field, value in dict(training_n=len(tx), evaluation_n=len(job["evaluation_ids"]), initial_admitted=status == "automated", inner_gate_passed=g["inner_gate_eligible"], execution_status=status, execution_reason=reason).items():
                    accounting.maybe(ar[field], value, ledger, "fit_admission/" + field)
            accounting.maybe(saved_choice["execution_status"], status, ledger, "choices/final_status")
            accounting.maybe(saved_choice["execution_reason"], reason, ledger, "choices/final_reason")
            category = "validation" if job["stage"] == "inner" else "actual"
            source_rows = inner if job["stage"] == "inner" else actual
            expected_scope = "inner_validation" if job["stage"] == "inner" else "in_sample_full120" if job["parent_context_id"] == 0 else "outer_oof"
            for sid in job["evaluation_ids"]:
                row = source_rows[(jid, policy, sid)]; identity(row, job, b, policy)
                accounting.maybe(row["scope"], expected_scope, ledger, "heldout/scope")
                accounting.maybe(row["assigned"], status == "automated", ledger, "heldout/uniform_admission")
                accounting.maybe(row["execution_status"], status, ledger, "heldout/status"); accounting.maybe(row["execution_reason"], reason, ledger, "heldout/reason")
                check_row_arithmetic(row, ledger, available=status == "automated")
            if status == "automated": expected_groups[category][jid + "__" + policy] = (job, target, policy)
        category = "validation" if job["stage"] == "inner" else "actual"
        for sid in job["evaluation_ids"]:
            row = inner_B[(jid, sid)] if job["stage"] == "inner" else actual[(jid, REFERENCES[0], sid)]
            identity(row, job, b, REFERENCES[0]); accounting.maybe(row["assigned"], target is not None, ledger, "reference/B_availability")
            check_row_arithmetic(row, ledger, available=target is not None)
            if job["stage"] == "refit":
                eq = actual[(jid, REFERENCES[1], sid)]; identity(eq, job, b, REFERENCES[1], equal=True)
                accounting.maybe(eq["assigned"], True, ledger, "reference/EQ_availability"); check_row_arithmetic(eq, ledger, available=True)
        if target is not None: expected_groups[category][jid + "__B"] = (job, target, REFERENCES[0])
        if job["stage"] == "refit": expected_groups["actual"][jid + "__equal"] = (job, np.ones(3) / 3, REFERENCES[1])

    for category in ("actual", "validation"):
        archive = archives[category]
        ledger.require(set(groups[category]) == set(expected_groups[category]), "trace/complete_selected_inventory")
        ledger.require(all(any(name.startswith(group + "__") for group in expected_groups[category]) for name in archive), "trace/no_orphans")
        for key, (job, target, policy) in expected_groups[category].items():
            budget(); meta = groups[category][key]
            ledger.require(meta["scenario_ids"] == job["evaluation_ids"] and meta["training_ids"] == job["training_ids"], "trace/original_train_evaluation_identity")
            ledger.require(meta["metadata"]["job_id"] == job["job_id"] and meta["metadata"]["policy"] == policy, "trace/job_policy")
            arm = {name[len(key) + 2:]: archive[name] for name in archive if name.startswith(key + "__")}
            tridx = np.asarray(job["training_ids"]) - 1; smallest = min(job["evaluation_ids"])
            for i, sid in enumerate(job["evaluation_ids"]):
                if policy in METHODS:
                    e = {field: arm[field][i] for field in (*QFLOATS, *QCOUNTS, "economic_pass", "risk_direction_pass", "estimate_available", "gate_reason", "decision_reason")}
                    replay = independent_model_replay(r[sid - 1], pi[sid - 1], target, r[tridx], pi[tridx], mode=METHODS[policy], saved_estimates=e, ledger=ledger)
                    ledger.require(replay["status"] == "complete", "trace/initial_positive_admission")
                    expected, metric = replay["account"], replay["metrics"]
                else:
                    expected = accounting.independent_account(r[sid - 1], pi[sid - 1], target, 0., hold_after_breach=policy != REFERENCES[1])
                    metric = accounting.independent_metrics(expected); metric["mean_fee_fraction"] = float(np.mean(expected["fee_fraction"]))
                saved = {field: arm[field] if field == "targets" else arm[field][i] for field in expected if field in arm and field not in ("delta", "loss_tolerance")}
                accounting.compare_saved_account(saved, expected, ledger)
                table = inner_B if category == "validation" and policy == REFERENCES[0] else inner if category == "validation" else actual
                row = table[(job["job_id"], sid)] if table is inner_B else table[(job["job_id"], policy, sid)]
                for field in accounting.METRICS: accounting.maybe(row[field], metric[field], ledger, "replayed_metric/" + field)
                counters["selected_trace_accounts"] += 1; counters["selected_trace_decisions"] += 30
                if policy in METHODS and sid == smallest:
                    for year in (1, 2, 16, 30):
                        t = year - 1; slot = dict(category=category, group=key, job_id=job["job_id"], policy=policy, scenario_id=sid, original_year=year)
                        if expected["pending_at_decision"][t]: slot["status"] = "pending_skip_no_replacement"
                        else:
                            estimate = independent_train_comparison(expected["holdings"][t], expected["cpi"][t], expected["real_peak"][t], expected["running_mdd"][t], target,
                                       r[tridx], pi[tridx], year_index=t, horizon=30, economic_periods=1 if METHODS[policy] == "one_year" else 30 - t)
                            for field in (*QFLOATS, *QCOUNTS, "economic_pass", "risk_direction_pass"):
                                accounting.maybe(arm[field][i, t], estimate[field], ledger, "independent_forecast_subset/" + field)
                            counters["independently_recomputed_subset_states"] += 1; slot["status"] = "independently_recomputed"
                        q_slots.append(slot)
    ledger.require(counters["training_rows"] == 20160 and counters["training_policy_witnesses"] == 260, "inventory/all_training_records_and_witness_slots")
    return dict(status="passed", checks_passed=ledger.checks_passed, maximum_absolute_errors=ledger.maximum_absolute_errors,
                membership=membership, counts=counters, training_witness_inventory=witnesses, fixed_forecast_state_inventory=q_slots,
                unrecorded_training_trees_all_replayed=False, every_saved_actual_estimate_independently_recomputed=False,
                scope_limits=readj(AUDIT_PLAN)["limitations"], elapsed_seconds=time.monotonic() - start)


def main():
    if (OUT / "independent_audit.json").exists(): raise RuntimeError("Never overwrite a successful frozen audit")
    if (OUT / "audit_failure_receipt.json").exists(): raise RuntimeError("Preserved failed attempt requires an explicit plan addendum")
    p = readj(PROTOCOL); manifest = readj(OUT / "manifest.json"); plan = readj(AUDIT_PLAN)
    if p.get("execution_ready") is not True or manifest.get("status") != "complete" or manifest["training_tasks"] != 130:
        raise RuntimeError("Complete frozen130-job archive required before any market-value parsing")
    if sha(PROTOCOL) != manifest["protocol_sha256"]: raise RuntimeError("Executed protocol changed")
    for mapping in (p["execution_sources_sha256"], p["frozen_input_sha256"]):
        for name, expected in mapping.items():
            if sha(ROOT / name) != expected: raise RuntimeError("Frozen source/input changed: " + name)
    for name, expected in manifest["outputs_sha256"].items():
        if sha(OUT / name) != expected: raise RuntimeError("Executed numeric artifact changed: " + name)
    paths = [PROTOCOL, AUDIT_PLAN, OUT / "manifest.json", Path(__file__), ROOT / "tests/test_audit_benefit_cost_execution.py", ROOT / "scripts/audit_simple_execution_comparison.py", CACHE]
    paths += [ROOT / name for name in p["execution_sources_sha256"]] + [ROOT / name for name in p["frozen_input_sha256"]] + [OUT / name for name in manifest["outputs_sha256"]]
    source_snapshot = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    r, pi, ids = load_paths(ROOT / p["data"]["path"], p["data"]["scenario_ids"], allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, r, pi): digest.update(np.ascontiguousarray(value).tobytes())
    if digest.hexdigest() != manifest["development_arrays_sha256"] or ids.tolist() != list(range(1, 121)) or manifest["final_scenarios_loaded"]:
        raise RuntimeError("Guarded development arrays changed")
    parent = readj(ROOT / "docs/refit_stability_protocol.json"); full = readj(ROOT / "docs/joint_liquidity_refit_protocol.json")
    bundles = readj(ROOT / "results/model_review/refit_stability_v1/static_refits.json.gz")
    bundles.update({k: v for k, v in readj(ROOT / "results/model_review/joint_liquidity_refit_v1/static_refits.json").items() if k.startswith("c0_")})
    for name in ("refit_stability_v1", "joint_liquidity_refit_v1"):
        if readj(ROOT / f"results/model_review/{name}/independent_audit.json")["status"] != "passed": raise RuntimeError("Completed parent Q1 component audit required")
    tables = {name: accounting.observed_bools(pd.read_csv(OUT / (name + ".csv.gz"), low_memory=False)) for name in p["expected_rows"]}
    with np.load(CACHE, allow_pickle=False) as f: cache = {name: f[name] for name in ("weights", "H30__mdd")}
    with np.load(OUT / "actual_traces.npz", allow_pickle=False) as a, np.load(OUT / "validation_traces.npz", allow_pickle=False) as v:
        result = audit_artifacts(p, bundles, r, pi, ids, tables, readj(OUT / "job_choices.json.gz"), readj(OUT / "trace_groups.json"), {"actual": a, "validation": v}, cache, parent=parent, full=full)
    end = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    if source_snapshot != end: raise RuntimeError("Frozen source/input/artifact changed during audit")
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(), parent_manifest_sha256=sha(OUT / "manifest.json"),
                  protocol_sha256=sha(PROTOCOL), audit_plan_sha256=sha(AUDIT_PLAN), auditor_sha256=sha(Path(__file__)),
                  sources_and_artifacts_at_start=source_snapshot, sources_and_artifacts_at_end=end,
                  scenario_ids_loaded=ids.tolist(), final_scenarios_loaded=False,
                  q1_component_audit_scope="Both completed parent Q1 component audits referenced; own130 context/target/candidate/Review-reason identity and B cache counts checked; no new all172-mask optimization audit",
                  interpretation="Independent numerical arithmetic and fixed-subset audit; not human peer review, true conditional calibration, new blind evidence or a risk guarantee")
    writej(OUT / "independent_audit.json", result)
    print(json.dumps({"status": "passed", "checks_passed": result["checks_passed"], "counts": result["counts"], "audit_sha256": sha(OUT / "independent_audit.json")}))


if __name__ == "__main__":
    try: main()
    except Exception as error:
        if OUT.exists() and not (OUT / "independent_audit.json").exists() and not (OUT / "audit_failure_receipt.json").exists():
            writej(OUT / "audit_failure_receipt.json", dict(status="failed_audit_attempt_preserved", completed_utc=datetime.now(timezone.utc).isoformat(),
                auditor_sha256=sha(Path(__file__)), audit_plan_sha256=sha(AUDIT_PLAN), error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc(),
                numeric_archive_modified_by_auditor=False, successful_audit_written=False))
        raise
