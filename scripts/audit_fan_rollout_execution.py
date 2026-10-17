"""Independent frozen-fan execution replay and predeclared scalar-Q audit.

No reviewed rollout/kernel/runner/metric helper is imported.  Accounting reuses
only our independent pending-audit bisection and metric reference.  The Q audit
identities are fixed below and in the prospective experiment protocol; skipped
Review/pending states are disclosed and never replaced by other examples.
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
sys.path.insert(0, str(ROOT / "scripts"))
from audit_review_pending_execution import independent_trade, independent_metrics, independent_cvar

DIRECTORY = ROOT / "results/model_review/fan_rollout_execution_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PROTOCOL = ROOT / "docs/fan_rollout_execution_protocol.json"
POLICIES = ("annual_then_pending", "growth_only_rollout", "pareto_hold_rollout")
CANDIDATES = POLICIES[1:]
TOL = 1e-12
Q_SUBSET_RULE = {"scenario": "minimum evaluation scenario ID in each frozen context",
                 "clients": "all automatically served original profiles; auxiliary onboarding Review skips are logged separately",
                 "policies": list(CANDIDATES), "years": "sorted distinct {2,floor(original H/2)+1,original H}",
                 "missing": "Only actually active pre-breach gates are evaluated; log missing/pending skips without replacement"}
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def snapshot(directory):
    manifest = readj(directory / "manifest.json")
    paths = [directory / name for name in manifest["outputs_sha256"]]
    paths += [directory / "manifest.json", Path(__file__).resolve(), ROOT / "scripts/audit_review_pending_execution.py"]
    paths += [ROOT / name for name in manifest["sources"]]
    paths += [ROOT / name for name in manifest["input_hashes"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def recommendation(qh, qr, kh, kr, policy):
    if policy == "growth_only_rollout":
        return "restore" if qr - qh > TOL else "hold"
    if policy == "pareto_hold_rollout":
        return "hold" if qh >= qr - TOL and kh <= kr else "restore"
    raise ValueError("Only the two declared gates have recommendations")


def reference_actual(returns, inflation, target, loss, action_at_year, *, cost_rate=.0025):
    """Execute supplied already-fixed actions with independent accounting."""
    h = len(returns)
    x, cpi, peak, worst, bad, tau = np.array([0., 0., 100.]), 1., 100., 0., False, -1
    arrays = {key: [] for key in ["holdings", "weights", "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd", "ever_breached", "cpi",
        "posttrade_holdings", "fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction", "pretrade_tv_drift", "end_year_tv_drift", "target_execution", "pending_at_decision", "actions", "trades"]}

    def state(current_dd):
        wealth = float(sum(x)); real = wealth / cpi
        for key, value in [("holdings", x.copy()), ("weights", x / wealth), ("nominal_wealth", wealth), ("real_wealth", real),
                           ("real_peak", peak), ("current_drawdown", current_dd), ("running_mdd", worst), ("ever_breached", bad), ("cpi", cpi)]:
            arrays[key].append(value)
    state(0.)
    for t in range(h):
        action = action_at_year(t, x.copy(), cpi, peak, worst, bad)
        if action not in ("hold", "restore"):
            raise AssertionError("Unknown fixed action")
        wealth = float(sum(x))
        post, fee, moved = independent_trade(x, target, cost_rate) if action == "restore" else (x.copy(), 0., 0.)
        for key, value in [("posttrade_holdings", post.copy()), ("fees", fee), ("moved", moved), ("pretrade_wealth", wealth),
                           ("fee_fraction", fee / wealth), ("moved_fraction", moved / wealth), ("pretrade_tv_drift", .5 * np.abs(x / wealth - target).sum()),
                           ("target_execution", action == "restore"), ("pending_at_decision", bad), ("actions", action), ("trades", moved > TOL)]:
            arrays[key].append(value)
        x = post * (1 + returns[t]); cpi *= 1 + inflation[t]
        real = float(sum(x)) / cpi
        peak = max(peak, real); current = max(1 - real / peak, 0.)
        worst = max(worst, current); bad = bad or worst - loss > TOL
        if tau < 0 and bad: tau = t + 1
        arrays["end_year_tv_drift"].append(.5 * np.abs(x / sum(x) - target).sum())
        state(current)
    result = {key: np.asarray(value) for key, value in arrays.items()}
    recovered_at = -1
    if tau >= 0:
        for endpoint in range(tau + 1, h + 1):
            if result["real_wealth"][endpoint] >= result["real_peak"][tau]:
                recovered_at = endpoint
                break
    result.update({"first_crossing": tau, "triggered": tau >= 0,
        "drawdown_at_trigger": np.nan if tau < 0 else result["running_mdd"][tau],
        "peak_at_trigger": np.nan if tau < 0 else result["real_peak"][tau],
        "real_wealth_at_trigger": np.nan if tau < 0 else result["real_wealth"][tau],
        "recovery_endpoint": recovered_at, "recovery_time": recovered_at - tau if recovered_at >= 0 else -1,
        "recovered": recovered_at >= 0, "right_censored": tau >= 0 and recovered_at < 0,
        "available_followup": h - tau if tau >= 0 else -1,
        "observed_followup": recovered_at - tau if recovered_at >= 0 else h - tau if tau >= 0 else -1,
        "added_severity": np.nan if tau < 0 else max(worst - result["running_mdd"][tau], 0.),
        "terminal_excess": max(worst - loss, 0.), "end_weights": result["weights"][1:]})
    interval_mdd = max([0.] + [1 - result["real_wealth"][t] / result["real_wealth"][s] for s in range(h) for t in range(s + 1, h + 1)])
    if abs(interval_mdd - worst) > 2e-14: raise AssertionError("Independent interval/state MDD mismatch")
    return result


def scalar_fan(holdings, cpi, peak, worst, target, loss, training_returns, training_inflation, year, *, cost_rate=.0025):
    """Current action then each branch's own annual-until-breach/pending B."""
    h, n = training_returns.shape[1], len(training_returns)
    current_real = float(sum(holdings)) / cpi
    outcome = {}
    for action in ("hold", "restore"):
        logs, bads = [], []
        for j in range(n):
            x, cp, pk, dd, bad = np.asarray(holdings, float).copy(), float(cpi), float(peak), float(worst), False
            for t in range(year, h):
                restore = action == "restore" if t == year else not bad
                if restore: x = independent_trade(x, target, cost_rate)[0]
                x = x * (1 + training_returns[j, t]); cp *= 1 + training_inflation[j, t]
                real = float(sum(x)) / cp; pk = max(pk, real)
                dd = max(dd, max(1 - real / pk, 0.)); bad = bad or dd - loss > TOL
            logs.append(math.log((float(sum(x)) / cp) / current_real)); bads.append(bad)
        outcome[action] = {"logs": np.asarray(logs), "bad": np.asarray(bads), "mean": float(np.mean(logs)), "count": int(sum(bads))}
    return outcome


def delete_one_count(values, policy):
    n = len(values["hold"]["logs"])
    if n <= 1: return None
    a, b = values["hold"], values["restore"]
    original = recommendation(a["mean"], b["mean"], a["count"], b["count"], policy)
    changed = 0
    for j in range(n):
        # Explicitly delete whole branches, independently of the kernel's
        # sum-minus-observation implementation.
        keep = np.arange(n) != j
        chosen = recommendation(float(np.mean(a["logs"][keep])), float(np.mean(b["logs"][keep])),
                                int(sum(a["bad"][keep])), int(sum(b["bad"][keep])), policy)
        changed += chosen != original
    return int(changed)


def synthetic_reference_check():
    gross = np.array([[[2., 1., 1.], [.8, 1., 1.]], [[.5, 1., 1.], [1., .68, 1.]]])
    target = np.array([.5, .4, .1])
    values = scalar_fan([100., 40., 10.], 1., 150., 0., target, .12, gross - 1, np.zeros((2, 2)), 1, cost_rate=0)
    assert values["hold"]["count"] == values["restore"]["count"] == 1
    assert abs(values["hold"]["mean"] - .5 * math.log((130 / 150) * (137.2 / 150))) < 1e-14
    assert abs(values["restore"]["mean"] - .5 * math.log(.9 * .872)) < 1e-14
    assert recommendation(values["hold"]["mean"], values["restore"]["mean"], 1, 1, "pareto_hold_rollout") == "hold"
    baseline, pareto = [], []
    for j in range(2):
        base_rule = lambda t, x, cp, pk, dd, bad: "hold" if bad else "restore"
        def pareto_rule(t, x, cp, pk, dd, bad):
            if t == 0: return "restore"
            if bad: return "hold"
            q = scalar_fan(x, cp, pk, dd, target, .12, gross - 1, np.zeros((2, 2)), t, cost_rate=0)
            return recommendation(q["hold"]["mean"], q["restore"]["mean"], q["hold"]["count"], q["restore"]["count"], "pareto_hold_rollout")
        baseline.append(reference_actual(gross[j] - 1, np.zeros(2), target, .12, base_rule, cost_rate=0))
        pareto.append(reference_actual(gross[j] - 1, np.zeros(2), target, .12, pareto_rule, cost_rate=0))
    assert np.allclose([p["real_wealth"][-1] for p in baseline], [135., 62.2], atol=1e-13, rtol=0)
    assert np.allclose([p["real_wealth"][-1] for p in pareto], [130., 62.2], atol=1e-13, rtol=0)
    assert sum(p["ever_breached"][-1] for p in baseline) == 1 and sum(p["ever_breached"][-1] for p in pareto) == 2
    return {"two_path_exact_Q_reference": True, "unconditional_recombination_counterexample": True,
            "baseline_terminals": [135., 62.2], "pareto_terminals": [130., 62.2], "baseline_breaches": 1, "pareto_breaches": 2}


def run_audit(directory):
    synthetic = synthetic_reference_check()
    before = snapshot(directory)
    manifest, protocol = readj(directory / "manifest.json"), readj(PROTOCOL)
    contexts = readj(directory / "contexts.json")
    frozen = readj(STRICT / "nested_context_refits.json")
    profiles = readj(ROOT / "docs/positive_asset_specification_protocol.json")["clients"]
    rows = pd.read_csv(directory / "client_paths.csv")
    gates = pd.read_csv(directory / "decision_gates.csv")
    checks, max_errors = 0, {}
    q_checks, q_results = 0, []

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition: raise AssertionError(label)

    def close(actual, expected, field, *, atol=3e-11, rtol=3e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape, field + "/shape")
        if a.dtype.kind in "bUiS" or b.dtype.kind in "bUiS":
            check(np.array_equal(a, b), field + "/exact")
        else:
            difference = np.abs(a - b); finite = np.isfinite(difference)
            name = field.split("/")[-1]
            max_errors[name] = max(max_errors.get(name, 0.), float(difference[finite].max(initial=0)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), field + "/numeric")

    for name, expected in protocol["frozen_input_sha256"].items(): check(sha(ROOT / name) == expected, "frozen_input/" + name)
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for array in [ids, returns, inflation]: digest.update(np.ascontiguousarray(array).tobytes())
    check(np.array_equal(ids, np.arange(1, 121)) and digest.hexdigest() == manifest["development_array_sha256"], "development_arrays")
    check(len(rows) == 3600 and not rows.duplicated(["context", "client", "scenario_id", "policy"]).any(), "client_denominator")
    check(len(gates) == 36720 and not gates.duplicated(["context", "client", "scenario_id", "policy", "year"]).any(), "decision_denominator")
    check(rows.loc[~rows.assigned, [*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing", "candidate_local_index", "candidate_original232_index"]].isna().all().all(), "Review_null")
    check(int((~rows.assigned).sum()) == 1440, "Review_denominator")
    row_lookup = rows.set_index(["context", "client", "scenario_id", "policy"]).sort_index()
    gate_lookup = gates.set_index(["context", "client", "scenario_id", "policy", "year"]).sort_index()
    all_references = {}
    verified_paths = transitions = 0
    with np.load(directory / "actual_traces.npz", allow_pickle=False) as archive:
        for context_name, context in contexts.items():
            fit = frozen[context_name]
            expected_ids = fit["training_ids"] if context_name == "full_development" else fit["outer_test_ids"]
            check(context["training_ids"] == fit["training_ids"] and context["evaluation_ids"] == expected_ids, context_name + "/identities")
            check(context["products"] == fit["raw_fit"]["menu"] and context["effective_assignments"] == fit["effective_assignments"], context_name + "/frozen_products")
            check(context_name == "full_development" or not set(context["training_ids"]) & set(expected_ids), context_name + "/train_test_disjoint")
            training = np.asarray(context["training_ids"]) - 1
            for profile in profiles:
                client, h, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
                assignment = fit["effective_assignments"][client]
                years = sorted({2, h // 2 + 1, h})
                subset_scenario = min(expected_ids)
                if assignment == "Review":
                    for policy in CANDIDATES:
                        for year in years:
                            q_results.append({"context": context_name, "client": client, "policy": policy, "scenario_id": subset_scenario,
                                              "year": year, "status": "skipped_frozen_onboarding_Review"})
                    for policy in POLICIES:
                        group = rows.loc[rows.context.eq(context_name) & rows.client.eq(client) & rows.policy.eq(policy)]
                        check(len(group) == len(expected_ids) and set(group.scenario_id) == set(expected_ids) and not group.assigned.any(), "Review_frozen_membership")
                    continue
                product = next(item for item in fit["raw_fit"]["menu"] if item["name"] == assignment)
                target = np.asarray(product["weights"])
                check(np.min(target) >= .05 - TOL, "strict_positive_target")
                for policy in POLICIES:
                    prefix = f"{context_name}__{client}__{policy}__"
                    check(np.array_equal(archive[prefix + "scenario_ids"], expected_ids), prefix + "/scenario_order")
                    reference_paths = []
                    for scenario in expected_ids:
                        current_gates = gate_lookup.loc[(context_name, client, scenario, policy)].sort_index()
                        check(list(current_gates.index) == list(range(1, h + 1)), prefix + "/all_decisions")

                        def action(t, x, cpi, peak, worst, bad):
                            record = current_gates.loc[t + 1]
                            close([record.holding_equity, record.holding_bond, record.holding_bill], x, "gate/holdings")
                            close(record.cpi, cpi, "gate/cpi"); close(record.real_peak, peak, "gate/peak"); close(record.running_mdd, worst, "gate/MDD")
                            check(bool(record.ever_breached) == bad and record.available_training_fan_paths == len(training), "gate/history_and_available_fan_N")
                            active = t > 0 and not bad and policy in CANDIDATES
                            if t == 0: expected = "restore"
                            elif bad: expected = "hold"
                            elif policy == "annual_then_pending": expected = "restore"
                            else:
                                check(record.forecast_paths == len(training), "gate/active_fan_N")
                                check(np.isfinite([record.qg_hold, record.qg_restore, record.qbad_count_hold, record.qbad_count_restore]).all(), "active_Q_present")
                                check(0 <= record.qbad_count_hold <= len(training) and 0 <= record.qbad_count_restore <= len(training)
                                      and float(record.qbad_count_hold).is_integer() and float(record.qbad_count_restore).is_integer(), "integer_risk_counts")
                                expected = recommendation(record.qg_hold, record.qg_restore, record.qbad_count_hold, record.qbad_count_restore, policy)
                            if not active:
                                check(pd.isna(record.forecast_paths), "inactive_forecast_fan_null")
                                check(record[["qg_hold", "qg_restore", "qbad_count_hold", "qbad_count_restore", "delete_one_flip_count"]].isna().all(), "inactive_Q_null")
                            check(record.action == expected, "gate_rule_and_timing")
                            if t == 0: reason = "mandatory_initial" if policy == "annual_then_pending" else "mandatory_initial_restore"
                            elif bad: reason = "pending_hold" if policy == "annual_then_pending" else "already_breached_pending_hold"
                            elif policy == "annual_then_pending": reason = "annual_reference"
                            elif policy == "growth_only_rollout": reason = "growth_strict_advantage" if expected == "restore" else "growth_tie_or_hold_advantage"
                            else: reason = "pareto_growth_or_risk_requires_restore" if expected == "restore" else "pareto_hold"
                            check(record.gate_reason == reason, "gate_reason")
                            return expected

                        reference = reference_actual(returns[scenario - 1, :h], inflation[scenario - 1, :h], target, loss, action)
                        reference_paths.append(reference); all_references[(context_name, client, scenario, policy)] = reference
                        verified_paths += 1; transitions += h
                        observed = row_lookup.loc[(context_name, client, scenario, policy)]
                        check(observed.assigned and observed.assignment == assignment and observed.horizon == h and observed.loss_tolerance == loss
                              and observed.candidate_local_index == product["candidate_local_index"]
                              and observed.candidate_original232_index == product["candidate_original232_index"], "frozen_client_path_identity")
                        fields = independent_metrics(reference, loss)
                        fields["trade_fraction"] = fields.pop("actual_trade_fraction")
                        fields["target_execution_fraction"] = float(np.mean(reference["target_execution"]))
                        for field in [*METRICS, *ACCOUNTS, "breach", "excess"]: close(observed[field], fields[field], "row/" + field)
                        if reference["first_crossing"] < 0: check(pd.isna(observed.first_crossing), "row/no_crossing_null")
                        else: check(observed.first_crossing == reference["first_crossing"], "row/first_crossing")
                        for t in range(h):
                            record = current_gates.loc[t + 1]
                            close(record.fee, reference["fees"][t], "gate/fee"); close(record.moved, reference["moved"][t], "gate/moved")
                            check(bool(record.actual_trade) == bool(reference["trades"][t]), "gate/actual_trade")
                    for key in reference_paths[0]:
                        expected = np.stack([path[key] for path in reference_paths])
                        close(archive[prefix + key], expected, "arrays/" + key)
                    if policy in CANDIDATES:
                        reference = all_references[(context_name, client, subset_scenario, policy)]
                        for year in years:
                            identity = {"context": context_name, "client": client, "policy": policy, "scenario_id": subset_scenario, "year": year}
                            record = gate_lookup.loc[(context_name, client, subset_scenario, policy, year)]
                            if pd.isna(record.qg_hold):
                                check(bool(reference["ever_breached"][year - 1]), "subset_skip_is_pending")
                                q_results.append({**identity, "status": "skipped_actual_already_pending_no_Q"})
                                continue
                            values = scalar_fan(reference["holdings"][year - 1], reference["cpi"][year - 1], reference["real_peak"][year - 1],
                                reference["running_mdd"][year - 1], target, loss, returns[training, :h], inflation[training, :h], year - 1)
                            q_checks += 1
                            close(record.qg_hold, values["hold"]["mean"], "Qsubset/qg_hold", atol=3e-12, rtol=0)
                            close(record.qg_restore, values["restore"]["mean"], "Qsubset/qg_restore", atol=3e-12, rtol=0)
                            check(record.qbad_count_hold == values["hold"]["count"] and record.qbad_count_restore == values["restore"]["count"], "Qsubset/breach_counts")
                            flip = delete_one_count(values, policy)
                            check(record.delete_one_flip_count == flip, "Qsubset/delete_one_flip")
                            expected_action = recommendation(values["hold"]["mean"], values["restore"]["mean"], values["hold"]["count"], values["restore"]["count"], policy)
                            check(record.action == expected_action, "Qsubset/recommendation")
                            q_results.append({**identity, "status": "audited", "training_paths": len(training),
                                "qg_hold": values["hold"]["mean"], "qg_restore": values["restore"]["mean"],
                                "qbad_count_hold": values["hold"]["count"], "qbad_count_restore": values["restore"]["count"],
                                "delete_one_flip_count": flip, "recommended_action": expected_action})
    check(verified_paths == 2160 and transitions == 36720, "all_actual_replay_counts")
    check(len(q_results) == 180, "complete_Q_subset_and_auxiliary_Review_coverage")
    for (scope, client, policy), group in rows.groupby(["scope", "client", "policy"]):
        check(len(group) == 120 and set(group.scenario_id) == set(range(1, 121)), "full_OOF120_denominators")
    summary = pd.read_csv(directory / "summary.csv")
    for record in summary.itertuples(index=False):
        group = rows.loc[rows.scope.eq(record.scope) & rows.client.eq(record.client) & rows.policy.eq(record.policy)]
        auto = group.loc[group.assigned]
        check(record.total_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "summary_denominator")
        if not len(auto):
            check(pd.isna(record.mean_log_growth) and pd.isna(record.breach_probability), "summary_Review_null")
            continue
        expected = {"mean_log_growth": auto.log_growth.mean(), "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
                    "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(auto.mdd),
                    "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": auto.breach.astype(bool).mean(), "mean_excess": auto.excess.mean(),
                    **{"mean_" + field: auto[field].mean() for field in ACCOUNTS}}
        for field, value in expected.items(): close(getattr(record, field), value, "summary/" + field)
    for category in ["sources", "input_hashes"]:
        for name, expected in manifest[category].items(): check(sha(ROOT / name) == expected, category + "/" + name)
    for name, expected in manifest["outputs_sha256"].items(): check(sha(directory / name) == expected, "output_hash/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_hashes"] == manifest["input_hashes_at_end"], "original_start_end_hashes")
    after = snapshot(directory); check(before == after, "audit_source_input_stable")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks_passed": checks,
            "maximum_errors": max_errors, "verified_actual_paths": verified_paths, "verified_actual_decisions": transitions,
            "client_rows": len(rows), "review_null_rows": int((~rows.assigned).sum()),
            "synthetic_reference_checks": synthetic, "Q_subset_rule": Q_SUBSET_RULE,
            "Q_subset_requested_identities": sum(item["status"] != "skipped_frozen_onboarding_Review" for item in q_results), "Q_subset_audited": q_checks,
            "Q_subset_skipped": sum(item["status"] == "skipped_actual_already_pending_no_Q" for item in q_results),
            "auxiliary_onboarding_Review_skips": sum(item["status"] == "skipped_frozen_onboarding_Review" for item in q_results), "Q_subset_results": q_results,
            "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
            "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(),
            "final_scenarios_read": False, "policy_fit_selection_or_modification": False,
            "independence": "Own100-step fee bisection from prior audit; own actual path state replay and scalar common-fan current-action/B simulation. No reviewed rollout/kernel/runner/metric helper imported",
            "limitations": ["Only the prospectively fixed deterministic Q subset is independently recomputed; all actual actions are checked against their saved score/count rule and all financial transitions are replayed",
                            "Source and membership audit do not prove hidden execution provenance beyond saved artifacts",
                            "Local delete-one influence keeps the actual state fixed; it is not full-policy leave-one-out validation",
                            "Recombined calendar fans are unconditional reference continuations; no model-consistent rollout improvement, global chance budget or population guarantee follows",
                            "Reused full/OOF development paths and overlapping fitted policies do not restore a fresh blind holdout"],
            "original_manifests_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(synthetic_reference_check()))
        return
    result = run_audit(arguments.directory)
    destination = arguments.out or arguments.directory / "independent_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "verified_actual_paths", "verified_actual_decisions", "Q_subset_requested_identities", "Q_subset_audited", "Q_subset_skipped", "maximum_errors"]}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
