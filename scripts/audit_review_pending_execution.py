"""Independent accounting and censoring audit of frozen pending-hold replay.

Only the guarded data loader is imported from the project.  Fees use an
independent scalar bisection; states, crossing, recovery and metrics are rebuilt
path by path.  The reviewed kernel, runner, simulator and metric helpers are
never imported.  No fitting, policy selection or final data access is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/model_review/review_pending_execution_v1"
PROTOCOL = ROOT / "docs/review_pending_execution_protocol.json"
FROZEN = ROOT / "results/model_review/positive_asset_specification_v1"
TOL = 1e-12
ARMS = ("annual", "pending_hold")
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
RECOVERY_FIELDS = ("first_crossing", "drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger",
                   "recovery_endpoint", "recovery_time", "recovered", "right_censored", "available_followup", "observed_followup", "added_severity")
PAIR_FIELDS = (*METRICS, "nominal_fees", "nominal_moved", "mean_annual_fee_fraction", "mean_annual_moved_fraction",
               "actual_trade_fraction", "max_end_year_tv_drift", "excess", "added_severity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def independent_trade(holdings, target, cost_rate):
    """Independently solve C=c*sum positive((W-C)w-x) by scalar bisection."""
    wealth = float(sum(holdings))
    weights = np.asarray(target, float)
    weights = weights / sum(weights)
    if not cost_rate:
        post = wealth * weights
        return post, 0., float(np.maximum(post - holdings, 0).sum())
    low, high = 0., wealth * cost_rate
    for _ in range(100):
        fee = (low + high) / 2
        buys = sum(max((wealth - fee) * weights[j] - holdings[j], 0.) for j in range(3))
        if fee > cost_rate * buys:
            high = fee
        else:
            low = fee
    fee = (low + high) / 2
    post = (wealth - fee) * weights
    return post, fee, float(np.maximum(post - holdings, 0).sum())


def independent_path(returns, inflation, target, loss_tolerance, *, arm, cost_rate=.0025):
    """Reconstruct one original-H account without calling the reviewed kernel."""
    market, infl = np.asarray(returns, float), np.asarray(inflation, float)
    horizon = len(market)
    if market.shape != (horizon, 3) or infl.shape != (horizon,) or horizon < 1 or arm not in ARMS:
        raise ValueError("Require one nonempty joint annual path and a stated arm")
    target = np.asarray(target, float)
    target = target / sum(target)
    states = {key: [] for key in ["holdings", "weights", "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd", "ever_breached", "cpi"]}
    decisions = {key: [] for key in ["posttrade_holdings", "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
                                    "pretrade_tv_drift", "end_year_tv_drift", "target_execution", "pending_at_decision", "actions", "trades"]}
    holdings = np.array([0., 0., 100.])
    cpi = peak = 1.
    peak = 100.
    running = 0.
    breached = False
    tau = None

    def append_state(nominal, real, current):
        states["holdings"].append(holdings.copy())
        states["weights"].append(holdings / nominal)
        for key, value in [("nominal_wealth", nominal), ("real_wealth", real), ("real_peak", peak),
                           ("current_drawdown", current), ("running_mdd", running), ("ever_breached", breached), ("cpi", cpi)]:
            states[key].append(value)

    append_state(100., 100., 0.)
    for year in range(horizon):
        nominal = float(sum(holdings))
        current_weights = holdings / nominal
        pending = arm == "pending_hold" and tau is not None
        if pending:
            post, fee, moved = holdings.copy(), 0., 0.
        else:
            post, fee, moved = independent_trade(holdings, target, cost_rate)
        for key, value in [("posttrade_holdings", post.copy()), ("fees", fee), ("moved", moved),
                           ("pretrade_wealth", nominal), ("fee_fraction", fee / nominal), ("moved_fraction", moved / nominal),
                           ("pretrade_tv_drift", float(.5 * np.abs(current_weights - target).sum())),
                           ("target_execution", not pending), ("pending_at_decision", pending),
                           ("actions", "hold" if pending else "restore"), ("trades", moved > TOL)]:
            decisions[key].append(value)
        holdings = post * (1 + market[year])
        cpi *= 1 + infl[year]
        nominal = float(sum(holdings))
        real = nominal / cpi
        peak = max(peak, real)
        current = max(1 - real / peak, 0.)
        running = max(running, current)
        breached = breached or running - loss_tolerance > TOL
        if tau is None and breached:
            tau = year + 1
        decisions["end_year_tv_drift"].append(float(.5 * np.abs(holdings / nominal - target).sum()))
        append_state(nominal, real, current)
    result = {key: np.asarray(value) for key, value in {**states, **decisions}.items()}
    result["end_weights"] = result["weights"][1:]
    result["first_crossing"] = -1 if tau is None else tau
    result["triggered"] = tau is not None
    recovery = None
    if tau is not None:
        for endpoint in range(tau + 1, horizon + 1):
            if result["real_wealth"][endpoint] >= result["real_peak"][tau]:
                recovery = endpoint
                break
    result.update({"drawdown_at_trigger": np.nan if tau is None else result["running_mdd"][tau],
                   "peak_at_trigger": np.nan if tau is None else result["real_peak"][tau],
                   "real_wealth_at_trigger": np.nan if tau is None else result["real_wealth"][tau],
                   "recovery_endpoint": -1 if recovery is None else recovery,
                   "recovery_time": -1 if recovery is None else recovery - tau,
                   "recovered": recovery is not None,
                   "right_censored": tau is not None and recovery is None,
                   "available_followup": -1 if tau is None else horizon - tau,
                   "observed_followup": -1 if tau is None else horizon - tau if recovery is None else recovery - tau,
                   "added_severity": np.nan if tau is None else max(result["running_mdd"][-1] - result["running_mdd"][tau], 0.),
                   "terminal_excess": max(result["running_mdd"][-1] - loss_tolerance, 0.),
                   "targets": target.copy(), "loss_tolerances": float(loss_tolerance)})
    # A second formula checks the full MDD using all intervals, independent
    # of the sequential peak calculation used to make the pending decision.
    real = result["real_wealth"]
    interval_mdd = max([0.] + [1 - real[t] / real[s] for s in range(horizon) for t in range(s + 1, horizon + 1)])
    if abs(interval_mdd - result["running_mdd"][-1]) > 2e-14:
        raise AssertionError("Independent interval and state MDD differ")
    return result


def independent_metrics(path, loss):
    real, horizon = path["real_wealth"], len(path["fees"])
    log_growth = math.log(real[-1] / real[0]) / horizon
    current = real[1:] / real[:-1] - 1
    triggered = bool(path["triggered"])
    fields = {"log_growth": log_growth, "cagr": math.expm1(log_growth), "mdd": float(path["running_mdd"][-1]),
              "volatility": float(np.std(current, ddof=1)) if horizon > 1 else 0., "terminal_real": float(real[-1]),
              "breach": bool(path["ever_breached"][-1]), "excess": float(max(path["running_mdd"][-1] - loss, 0)),
              "nominal_fees": float(sum(path["fees"])), "nominal_moved": float(sum(path["moved"])),
              "mean_annual_fee_fraction": float(np.mean(path["fee_fraction"])),
              "mean_annual_moved_fraction": float(np.mean(path["moved_fraction"])),
              "actual_trade_count": int(sum(path["trades"])), "review_periods": horizon,
              "actual_trade_fraction": float(np.mean(path["trades"])),
              "max_end_year_tv_drift": float(max(path["end_year_tv_drift"])),
              "terminal_equity": float(path["weights"][-1, 0]), "terminal_bond": float(path["weights"][-1, 1]),
              "terminal_bill": float(path["weights"][-1, 2]), "triggered": triggered}
    fields.update({key: path[key] if triggered else None for key in RECOVERY_FIELDS})
    if triggered and not path["recovered"]:
        fields["recovery_endpoint"] = fields["recovery_time"] = None
    return fields


def self_checks():
    """Closed-form synthetic identities, without the reviewed implementation."""
    post, fee, moved = independent_trade(np.array([0., 0., 100.]), [1., 0., 0.], .0025)
    assert abs(fee - 100 * .0025 / 1.0025) < 1e-14 and abs(fee - .0025 * moved) < 1e-14
    market = np.array([[-.5, 0, 0], [1., 0, 0], [1., 0, 0]])
    annual = independent_path(market, np.zeros(3), [.5, 0, .5], .2, arm="annual", cost_rate=0)
    pending = independent_path(market, np.zeros(3), [.5, 0, .5], .2, arm="pending_hold", cost_rate=0)
    assert np.array_equal(annual["real_wealth"], [100., 75., 112.5, 168.75])
    assert np.array_equal(pending["real_wealth"], [100., 75., 100., 150.])
    assert annual["first_crossing"] == pending["first_crossing"] == 1
    assert pending["recovery_endpoint"] == 2 and pending["recovery_time"] == 1
    assert list(pending["actions"]) == ["restore", "hold", "hold"]
    last = independent_path(market[:1], np.zeros(1), [.5, 0, .5], .2, arm="pending_hold", cost_rate=0)
    assert last["right_censored"] and last["available_followup"] == last["observed_followup"] == 0
    last_fields = independent_metrics(last, .2)
    assert last_fields["recovery_endpoint"] is None and last_fields["recovery_time"] is None
    no_event = independent_path(np.zeros((2, 3)), np.zeros(2), [.5, 0, .5], .2, arm="pending_hold", cost_rate=0)
    assert not no_event["triggered"] and all(independent_metrics(no_event, .2)[key] is None for key in RECOVERY_FIELDS)
    return {"full_switch_fee_closed_form": True, "known_paired_wealth": True, "next_decision_hold": True,
            "fixed_peak_recovery": True, "crossing_at_H_zero_followup_censoring": True, "untriggered_serialization_null": True}


def independent_cvar(values):
    ordered = np.sort(np.asarray(values, float))[::-1]
    mass = len(ordered) / 10
    whole = int(math.floor(mass))
    numerator = float(sum(ordered[:whole]))
    if whole < len(ordered):
        numerator += (mass - whole) * ordered[whole]
    return numerator / mass


def evidence_snapshot(directory):
    manifest = readj(directory / "manifest.json")
    paths = [directory / name for name in manifest["outputs_sha256"]]
    paths += [directory / "manifest.json", Path(__file__).resolve()]
    paths += [ROOT / path for path in manifest["sources"]]
    paths += [ROOT / path for path in manifest["frozen_dependency_sha256"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def run_audit(directory):
    synthetic = self_checks()
    before = evidence_snapshot(directory)
    manifest, protocol = readj(directory / "manifest.json"), readj(PROTOCOL)
    groups = readj(directory / "execution_groups.json")
    frozen = readj(FROZEN / "nested_context_refits.json")
    strict = readj(ROOT / "docs/positive_asset_specification_protocol.json")
    mapping = readj(FROZEN / "candidate_mapping.json")
    rows = pd.read_csv(directory / "client_path_metrics.csv")
    paired = pd.read_csv(directory / "paired_path_differences.csv")
    trace = pd.read_csv(directory / "yearly_trace.csv")
    checks = 0
    maximum_errors = {}

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)

    def equal_numeric(actual, expected, label, *, atol=2e-11, rtol=2e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape, label + "/shape")
        if a.dtype.kind in "bUiS" or b.dtype.kind in "bUiS":
            check(np.array_equal(a, b), label + "/exact")
        else:
            difference = np.abs(a.astype(float) - b.astype(float))
            finite = np.isfinite(difference)
            name = label.split("/")[-1]
            maximum_errors[name] = max(maximum_errors.get(name, 0.), float(difference[finite].max(initial=0)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + "/numeric")

    for path, expected in protocol["frozen_dependencies_sha256"].items():
        check(sha(ROOT / path) == expected, "frozen_hash/" + path)
    check(protocol["execution_kernel"]["ready"] and sha(ROOT / protocol["execution_kernel"]["path"]) == protocol["execution_kernel"]["sha256"], "kernel_frozen")
    check(protocol["data"]["allowed_scenario_ids"] == list(range(1, 121)) and protocol["data"]["allow_final"] is False, "development_only_contract")
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths  # Guarded loader only; no reviewed accounting code is imported.
    returns, inflation, ids = load_paths(ROOT / protocol["data"]["path"], range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in [ids, returns, inflation]: digest.update(np.ascontiguousarray(value).tobytes())
    check(np.array_equal(ids, np.arange(1, 121)) and digest.hexdigest() == manifest["development_arrays_sha256"], "guarded_array_fingerprint")
    check(len(rows) == 2400 and len(paired) == 1200, "all_profile_path_denominators")
    check(not rows.duplicated(["context", "client", "scenario_id", "arm"]).any(), "unique_profile_paths")
    check(not paired.duplicated(["context", "client", "scenario_id"]).any(), "unique_paired_paths")
    check(len(groups) == 18, "auto_execution_groups")
    review = rows.loc[~rows.assigned]
    financial_fields = [*METRICS, "breach", "triggered", *RECOVERY_FIELDS, "nominal_fees", "nominal_moved", "actual_trade_count", "actual_trade_fraction",
                        "terminal_equity", "terminal_bond", "terminal_bill", "candidate_local_index", "candidate_original232_index"]
    check(review[financial_fields].isna().all().all(), "Review_outcomes_null")
    check(len(review) == 960, "Review_complete_denominator")
    paired_review_fields = ["delta_breach", "triggered", "first_crossing"]
    paired_review_fields += [prefix + field for prefix in ["annual_", "pending_", "delta_"] for field in PAIR_FIELDS]
    check(paired.loc[~paired.assigned, paired_review_fields].isna().all().all(), "paired_Review_outcomes_null")
    check(rows.loc[rows.assigned, [*METRICS, "breach", "triggered", "candidate_local_index", "candidate_original232_index"]].notna().all().all(), "auto_base_outcomes_present")
    record_lookup = rows.set_index(["context", "client", "scenario_id", "arm"])
    pair_lookup = paired.set_index(["context", "client", "scenario_id"])
    context_lookup = {context["name"]: context for context in strict["contexts"]}
    saved_annual = pd.concat([pd.read_csv(FROZEN / "nested_full_paths.csv"), pd.read_csv(FROZEN / "nested_outer_oof_paths.csv")])
    saved_annual = saved_annual.loc[saved_annual.view.eq("effective")].set_index(["context", "client", "scenario_id"])
    verified_paths = transitions = 0
    expected_trace = []
    with np.load(directory / "executed_paths.npz", allow_pickle=False) as archive:
        for group in groups:
            context = context_lookup[group["context"]]
            expected_ids = context["training_ids"] if context["context_id"] == 0 else context["outer_test_ids"]
            check(group["scenario_ids"] == expected_ids, group["prefix"] + "/scenario_membership")
            check(not set(context["training_ids"]) & set(context["outer_test_ids"]), group["prefix"] + "/train_test_disjoint")
            fit = frozen[group["context"]]
            check(fit["effective_assignments"][group["client"]] == group["assignment"], group["prefix"] + "/frozen_assignment")
            product = next(item for item in fit["raw_fit"]["menu"] if item["name"] == group["assignment"])
            check(group["candidate_local_index"] == product["candidate_index"]
                  and mapping[group["candidate_local_index"]]["candidate_original232_index"] == group["candidate_original232_index"], group["prefix"] + "/local_global_identity")
            equal_numeric(group["target"], product["weights"], group["prefix"] + "/target", atol=0, rtol=0)
            profile = next(item for item in strict["clients"] if item["name"] == group["client"])
            check(group["horizon"] == profile["horizon"] and group["loss_tolerance"] == profile["loss_tolerance"], group["prefix"] + "/original_H_L")
            references = {arm: [] for arm in ARMS}
            for path_index, scenario in enumerate(expected_ids):
                pair_ref = {}
                for arm in ARMS:
                    path = independent_path(returns[scenario - 1, :group["horizon"]], inflation[scenario - 1, :group["horizon"]],
                                            group["target"], group["loss_tolerance"], arm=arm, cost_rate=.0025)
                    references[arm].append(path)
                    pair_ref[arm] = path
                    verified_paths += 1
                    transitions += group["horizon"]
                    expected_fields = independent_metrics(path, group["loss_tolerance"])
                    record = record_lookup.loc[(group["context"], group["client"], scenario, arm)]
                    check(record.assigned and record.assignment == group["assignment"]
                          and record.candidate_local_index == group["candidate_local_index"]
                          and record.candidate_original232_index == group["candidate_original232_index"], group["prefix"] + "/row_identity")
                    for field, expected in expected_fields.items():
                        value = record[field]
                        if expected is None:
                            check(pd.isna(value), group["prefix"] + "/" + field + "/null")
                        else:
                            equal_numeric(value, expected, group["prefix"] + "/row/" + field)
                    check(record.financial_referral_reason == "financial_drawdown_crossing" if path["triggered"] else pd.isna(record.financial_referral_reason),
                          group["prefix"] + "/reason")
                    if arm == "annual":
                        saved = saved_annual.loc[(group["context"], group["client"], scenario)]
                        for field in METRICS: equal_numeric(record[field], saved[field], group["prefix"] + "/original_saved/" + field)
                    for year in range(group["horizon"] + 1):
                        expected_trace.append({"context": group["context"], "client": group["client"], "scenario_id": scenario, "arm": arm,
                                               "year": year, "reference": path})
                a, b = pair_ref["annual"], pair_ref["pending_hold"]
                check(a["first_crossing"] == b["first_crossing"] and a["ever_breached"][-1] == b["ever_breached"][-1], group["prefix"] + "/structural_coupling")
                stop = group["horizon"] if a["first_crossing"] < 0 else a["first_crossing"]
                for key in ["holdings", "weights", "real_wealth", "real_peak", "running_mdd", "ever_breached"]:
                    equal_numeric(a[key][:stop + 1], b[key][:stop + 1], group["prefix"] + "/prefix/" + key, atol=0, rtol=0)
                    actual_a = archive[group["prefix"] + "__annual__" + key][path_index, 0, :stop + 1]
                    actual_b = archive[group["prefix"] + "__pending_hold__" + key][path_index, 0, :stop + 1]
                    check(np.array_equal(actual_a, actual_b), group["prefix"] + "/saved_exact_prefix/" + key)
                for key in ["posttrade_holdings", "fees", "moved", "actions", "target_execution", "pending_at_decision"]:
                    actual_a = archive[group["prefix"] + "__annual__" + key][path_index, 0, :stop]
                    actual_b = archive[group["prefix"] + "__pending_hold__" + key][path_index, 0, :stop]
                    check(np.array_equal(actual_a, actual_b), group["prefix"] + "/saved_exact_decision_prefix/" + key)
                if a["triggered"]:
                    check(np.all(b["fees"][stop:] == 0) and np.all(b["moved"][stop:] == 0)
                          and np.all(b["actions"][stop:] == "hold"), group["prefix"] + "/no_post_trigger_trade")
                pair_row = pair_lookup.loc[(group["context"], group["client"], scenario)]
                metrics_a, metrics_b = independent_metrics(a, group["loss_tolerance"]), independent_metrics(b, group["loss_tolerance"])
                check(pair_row.delta_breach == 0, group["prefix"] + "/paired_breach_invariant")
                for field in PAIR_FIELDS:
                    expected = None if metrics_a[field] is None else metrics_b[field] - metrics_a[field]
                    if expected is None:
                        check(pd.isna(pair_row["delta_" + field]), group["prefix"] + "/paired_" + field + "/null")
                    else:
                        equal_numeric(pair_row["delta_" + field], expected, group["prefix"] + "/paired/" + field)
            for arm, paths in references.items():
                prefix = group["prefix"] + "__" + arm + "__"
                for key in paths[0]:
                    expected = np.stack([path[key] for path in paths])
                    if key == "targets": expected = expected[:1]
                    elif key == "loss_tolerances": expected = expected[:1]
                    elif key != "cpi": expected = expected[:, None, ...]
                    equal_numeric(archive[prefix + key], expected, group["prefix"] + "/" + arm + "/array/" + key)
    # The human-readable trace is independently tied to the same exact states.
    check(len(trace) == len(expected_trace) and not trace.duplicated(["context", "client", "scenario_id", "arm", "year"]).any(), "trace_complete_unique")
    trace_lookup = trace.set_index(["context", "client", "scenario_id", "arm", "year"])
    for item in expected_trace:
        year, path = item["year"], item["reference"]
        observed = trace_lookup.loc[tuple(item[key] for key in ["context", "client", "scenario_id", "arm", "year"])]
        for field in ["nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd", "cpi", "ever_breached"]:
            equal_numeric(observed[field], path[field][year], "trace/" + field)
        for index, asset in enumerate(["equity", "bond", "bill"]):
            equal_numeric(observed["holding_" + asset], path["holdings"][year, index], "trace/holding_" + asset)
            equal_numeric(observed["weight_" + asset], path["weights"][year, index], "trace/weight_" + asset)
        for field in ["fees", "moved", "trades", "actions", "pending_at_decision"]:
            if year == 0:
                check(pd.isna(observed[field + "_before_return_year"]), "trace/initial_no_decision")
            else:
                equal_numeric(observed[field + "_before_return_year"], path[field][year - 1], "trace/" + field)
    for scope in ["full", "oof"]:
        for arm in ARMS:
            for profile in strict["clients"]:
                selected = rows.loc[rows.scope.eq(scope) & rows.arm.eq(arm) & rows.client.eq(profile["name"])]
                check(len(selected) == 120 and set(selected.scenario_id) == set(range(1, 121)), "complete_client_scope_denominator")
    # Censoring summaries must describe the same selected subgroup and must not
    # treat nonrecoveries as finite recovered delays.
    for filename, by_context in [("profile_summary.csv", False), ("context_profile_summary.csv", True)]:
        summary = pd.read_csv(directory / filename)
        for record in summary.itertuples(index=False):
            mask = rows.scope.eq(record.scope) & rows.arm.eq(record.arm) & rows.client.eq(record.client)
            if by_context: mask &= rows.context.eq(record.context)
            group = rows.loc[mask]
            auto = group.loc[group.assigned]
            triggered = auto.loc[auto.triggered.astype(bool)]
            selected = auto if record.subset == "all_assigned" else triggered
            denominator = 24 if by_context and record.scope == "oof" else 120
            check(record.total_profile_paths == denominator and record.auto_n == len(auto) and record.review_n == denominator - len(auto)
                  and record.triggered_n == len(triggered) and record.summary_n == len(selected), filename + "/denominators")
            if len(selected):
                recovered = triggered.loc[triggered.recovered.astype(bool)]
                check(record.recovery_risk_set_n == len(triggered) and record.recovered_n == len(recovered)
                      and record.right_censored_n == len(triggered) - len(recovered), filename + "/recovery_censor_counts")
                if len(recovered): equal_numeric(record.median_recovery_delay_among_recovered_only, recovered.recovery_time.median(), "summary/recovered_only_median")
                else: check(pd.isna(record.median_recovery_delay_among_recovered_only), "summary/no_imputed_recovery")
                if len(triggered): equal_numeric(record.mean_added_severity_among_triggered, triggered.added_severity.mean(), "summary/added_severity_among_triggered")
                else: check(pd.isna(record.mean_added_severity_among_triggered), "summary/no_untriggered_severity_imputation")
                check(record.breach_count == int(selected.breach.astype(bool).sum()), filename + "/breach_count")
                expected_statistics = {"mean_log_growth": selected.log_growth.mean(), "median_cagr": selected.cagr.median(),
                    "p10_cagr": np.quantile(selected.cagr, .1, method="inverted_cdf"), "median_mdd": selected.mdd.median(),
                    "p90_mdd": np.quantile(selected.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(selected.mdd),
                    "mean_terminal_real": selected.terminal_real.mean(), "mean_excess": selected.excess.mean(),
                    "mean_nominal_fees": selected.nominal_fees.mean(), "mean_nominal_moved": selected.nominal_moved.mean(),
                    "actual_trade_fraction": selected.actual_trade_count.sum() / selected.review_periods.sum(),
                    "breach_probability": selected.breach.astype(bool).mean()}
                for field, expected in expected_statistics.items(): equal_numeric(getattr(record, field), expected, "summary/" + field)
            else:
                check(all(pd.isna(getattr(record, field)) for field in ["breach_probability", "mean_log_growth", "mean_terminal_real",
                          "mean_nominal_fees", "median_recovery_delay_among_recovered_only"]), filename + "/null_financial_outcomes")
    for filename, by_context in [("paired_profile_summary.csv", False), ("paired_context_profile_summary.csv", True)]:
        summary = pd.read_csv(directory / filename)
        for record in summary.itertuples(index=False):
            mask = paired.scope.eq(record.scope) & paired.client.eq(record.client)
            if by_context: mask &= paired.context.eq(record.context)
            group = paired.loc[mask]
            auto = group.loc[group.assigned]
            selected = auto if record.subset == "all_assigned" else auto.loc[auto.triggered.astype(bool)]
            denominator = 24 if by_context and record.scope == "oof" else 120
            check(record.total_profile_paths == denominator and record.auto_n == len(auto) and record.summary_n == len(selected), filename + "/denominators")
            check(record.added_severity_pair_n == int(selected.delta_added_severity.notna().sum()), filename + "/severity_pair_denominator")
            if len(selected): check(record.sum_delta_breach == 0, filename + "/coupling_breach_invariant")
            else: check(pd.isna(record.sum_delta_breach), filename + "/no_imputed_breach_delta")
            for field in PAIR_FIELDS:
                expected = selected["delta_" + field].mean()
                actual = getattr(record, "mean_delta_" + field)
                if pd.isna(expected): check(pd.isna(actual), filename + "/null_delta_" + field)
                else: equal_numeric(actual, expected, filename + "/mean_delta_" + field)
            for field in ["mdd", "terminal_real"]:
                change = selected["delta_" + field]
                for name, expected in [("increased", (change > TOL).sum()), ("decreased", (change < -TOL).sum()), ("equal", (change.abs() <= TOL).sum())]:
                    check(getattr(record, field + "_" + name + "_n") == expected, filename + "/direction_counts")
    for path, expected in manifest["sources"].items(): check(sha(ROOT / path) == expected, "source_hash/" + path)
    for path, expected in manifest["frozen_dependency_sha256"].items(): check(sha(ROOT / path) == expected, "dependency_hash/" + path)
    for path, expected in manifest["outputs_sha256"].items(): check(sha(directory / path) == expected, "output_hash/" + path)
    check(manifest["sources"] == manifest["sources_at_end"]
          and manifest["frozen_dependency_sha256"] == manifest["frozen_dependency_sha256_at_end"], "original_start_end_hashes")
    after = evidence_snapshot(directory)
    check(before == after, "audit_sources_and_artifacts_stable")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks_passed": checks,
            "synthetic_reference_checks": synthetic, "maximum_errors": maximum_errors,
            "verified_auto_arm_paths": verified_paths, "verified_annual_transitions": transitions,
            "client_path_rows": len(rows), "paired_rows": len(paired), "yearly_trace_rows": len(trace),
            "execution_groups": len(groups), "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
            "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(),
            "final_scenarios_read": False, "new_model_fit_or_policy_selection": False,
            "independence": "Only guarded loader reused. Own scalar100-step bisection fee, yearly holdings/CPI/peak/flag recursion, all-interval terminal MDD, recovery and metrics; no reviewed kernel/runner/simulator/metric helper imported",
            "scope": "Frozen strict172 effective targets, complete original-H annual observations, first individual-L crossing then next-decision hold, no response through H",
            "limitations": ["Structural coupling proves the same first-crossing/full-H everbreach event in this specified ablation; it does not prove a risk-reduction benefit",
                            "Right-censored recovery is retained; recovered-only delay is a selected subgroup statistic",
                            "No adviser benefits, optimal permanent inaction, true conditional forecast, population risk certificate or final service adoption claim",
                            "OOF training sets overlap and their frozen policies differ; these checks are computational, not independent market samples"],
            "original_manifest_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(self_checks()))
        return
    result = run_audit(arguments.directory)
    destination = arguments.out or arguments.directory / "independent_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "verified_auto_arm_paths", "verified_annual_transitions", "yearly_trace_rows", "maximum_errors"]}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
