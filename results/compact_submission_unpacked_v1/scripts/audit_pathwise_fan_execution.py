"""Independent actual/training replay and scalar-Q audit for zero-epsilon shield.

Only our previously independent audit references and guarded loader are reused.
No reviewed execution, rollout, runner or metric implementation is imported.
Zero-threshold classification differences are retained for review, never hidden
by changing the gate or by the separate1e-10 training-property audit guard.
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
from audit_fan_rollout_execution import reference_actual, scalar_fan
from audit_review_pending_execution import independent_metrics, independent_cvar

DIRECTORY = ROOT / "results/model_review/pathwise_fan_execution_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PROTOCOL = ROOT / "docs/pathwise_fan_execution_protocol.json"
POLICIES = ("annual_then_pending", "pathwise_hold_rollout")
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill")
TOL = 1e-12
PROPERTY_GUARD = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def snapshot(directory):
    manifest = readj(directory / "manifest.json")
    paths = [directory / name for name in manifest["outputs_sha256"]]
    paths += [directory / "manifest.json", Path(__file__).resolve(), ROOT / "scripts/audit_fan_rollout_execution.py", ROOT / "scripts/audit_review_pending_execution.py"]
    paths += [ROOT / name for name in manifest["sources"]]
    paths += [ROOT / name for name in manifest["input_hashes"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def run_audit(directory):
    before = snapshot(directory)
    manifest, protocol = readj(directory / "manifest.json"), readj(PROTOCOL)
    contexts = readj(directory / "contexts.json")
    frozen = readj(STRICT / "nested_context_refits.json")
    profiles = readj(ROOT / "docs/positive_asset_specification_protocol.json")["clients"]
    rows, gates = pd.read_csv(directory / "client_paths.csv"), pd.read_csv(directory / "decision_gates.csv")
    support = pd.read_csv(directory / "training_support_paths.csv")
    support_summary = pd.read_csv(directory / "training_support_summary.csv")
    checks, maximum_errors = 0, {}
    q_results, q_disagreements = [], []
    replay_counts = {"actual_paths": 0, "training_paths": 0, "actual_decisions": 0, "training_decisions": 0}
    own_property_deltas = []

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
            error = np.abs(a - b); finite = np.isfinite(error)
            name = field.split("/")[-1]
            maximum_errors[name] = max(maximum_errors.get(name, 0.), float(error[finite].max(initial=0)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), field + "/numeric")

    for path, expected in protocol["frozen_input_sha256"].items(): check(sha(ROOT / path) == expected, "frozen_hash/" + path)
    check(manifest["gate_growth_allowance"] == 0 and manifest["numerical_property_check_guard"] == PROPERTY_GUARD, "zero_gate_vs_property_guard")
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for array in [ids, returns, inflation]: digest.update(np.ascontiguousarray(array).tobytes())
    check(np.array_equal(ids, np.arange(1, 121)) and digest.hexdigest() == manifest["development_array_sha256"], "guarded_data_digest")
    check(len(rows) == 2400 and not rows.duplicated(["context", "client", "scenario_id", "policy"]).any(), "actual_all_clients")
    check(len(support) == 3000 and not support.duplicated(["context", "client", "scenario_id"]).any(), "training_all_clients")
    check(len(gates) == 12240 and not gates.duplicated(["context", "client", "scenario_id", "year"]).any(), "actual_decision_rows")
    check(rows.loc[~rows.assigned, [*METRICS, *ACCOUNTS, "first_crossing", "breach", "excess", "candidate_local_index", "candidate_original232_index"]].isna().all().all(), "actual_Review_null")
    check(support.loc[~support.assigned, ["reference_total_log_growth", "pathwise_total_log_growth", "total_log_growth_delta", "reference_breach", "pathwise_breach"]].isna().all().all(), "training_Review_null")
    row_lookup = rows.set_index(["context", "client", "scenario_id", "policy"]).sort_index()
    gate_lookup = gates.set_index(["context", "client", "scenario_id", "year"]).sort_index()
    support_lookup = support.set_index(["context", "client", "scenario_id"]).sort_index()
    with np.load(directory / "actual_traces.npz", allow_pickle=False) as actual_archive, np.load(directory / "training_traces.npz", allow_pickle=False) as train_archive:
        for context_name, context in contexts.items():
            fit = frozen[context_name]
            eval_ids = fit["training_ids"] if context_name == "full_development" else fit["outer_test_ids"]
            check(context["training_ids"] == fit["training_ids"] and context["evaluation_ids"] == eval_ids
                  and context["products"] == fit["raw_fit"]["menu"] and context["effective_assignments"] == fit["effective_assignments"], context_name + "/frozen_identity")
            check(context_name == "full_development" or not set(fit["training_ids"]) & set(eval_ids), context_name + "/OOF_disjoint")
            train_ix = np.asarray(fit["training_ids"]) - 1
            for profile in profiles:
                client, h, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
                assignment = fit["effective_assignments"][client]
                years = sorted({2, h // 2 + 1, h})
                if assignment == "Review":
                    for year in years: q_results.append({"context": context_name, "client": client, "scenario_id": min(eval_ids), "year": year, "status": "auxiliary_onboarding_Review_skip"})
                    for kind, identity_ids, frame in [("actual", eval_ids, rows), ("training", fit["training_ids"], support)]:
                        group = frame.loc[frame.context.eq(context_name) & frame.client.eq(client)]
                        check(len(group) == len(identity_ids) * (2 if kind == "actual" else 1)
                              and set(group.scenario_id) == set(identity_ids) and not group.assigned.any(), "Review_frozen_denominators")
                    continue
                product = next(item for item in fit["raw_fit"]["menu"] if item["name"] == assignment)
                target = np.asarray(product["weights"])
                check(np.min(target) >= .05 - TOL, "strict_positive")
                actual_references = {}
                for kind, identity_ids, archive in [("actual", eval_ids, actual_archive), ("training", fit["training_ids"], train_archive)]:
                    reference_by_policy = {}
                    for policy in POLICIES:
                        prefix = f"{context_name}__{client}__{policy}__"
                        saved_arrays = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                        check(np.array_equal(saved_arrays["scenario_ids"], identity_ids), kind + "/scenario_order")
                        paths = []
                        for index, scenario in enumerate(identity_ids):
                            def action(t, x, cpi, peak, worst, bad):
                                if t == 0: expected = "restore"
                                elif bad: expected = "hold"
                                elif policy == "annual_then_pending": expected = "restore"
                                else:
                                    counts = {field: int(saved_arrays[field][index, t]) for field in ["growth_violation_count", "breach_violation_count", "union_violation_count"]}
                                    n = len(fit["training_ids"])
                                    check(all(0 <= count <= n for count in counts.values()), kind + "/integer_violation_range")
                                    check(max(counts["growth_violation_count"], counts["breach_violation_count"]) <= counts["union_violation_count"]
                                          <= min(n, counts["growth_violation_count"] + counts["breach_violation_count"]), kind + "/union_logic")
                                    minimum = saved_arrays["min_log_advantage_hold"][index, t]
                                    check((minimum < 0) == (counts["growth_violation_count"] > 0), kind + "/zero_growth_sign")
                                    expected = "restore" if counts["union_violation_count"] else "hold"
                                    check(saved_arrays["forecast_n"][index, t] == n, kind + "/fan_N")
                                    check(saved_arrays["delete_one_flip_count"][index, t] == int(counts["union_violation_count"] == 1), kind + "/local_deleteone_logic")
                                check(saved_arrays["actions"][index, t] == expected, kind + "/action_rule_timing")
                                if policy == "pathwise_hold_rollout":
                                    active = t > 0 and not bad
                                    for field in ["qg_hold", "qg_restore", "min_log_advantage_hold"]:
                                        check(np.isfinite(saved_arrays[field][index, t]) if active else np.isnan(saved_arrays[field][index, t]), kind + "/active_score_mask")
                                    for field in ["qbad_hold_count", "qbad_restore_count", "forecast_n", "growth_violation_count", "breach_violation_count", "union_violation_count", "delete_one_flip_count"]:
                                        check(saved_arrays[field][index, t] >= 0 if active else saved_arrays[field][index, t] == -1, kind + "/active_integer_mask")
                                return expected
                            path = reference_actual(returns[scenario - 1, :h], inflation[scenario - 1, :h], target, loss, action)
                            paths.append(path); replay_counts[kind + "_paths"] += 1; replay_counts[kind + "_decisions"] += h
                            for field in path:
                                if field in saved_arrays: close(saved_arrays[field][index], path[field], kind + "/trace/" + field)
                            if kind == "actual":
                                actual_references[(scenario, policy)] = path
                                observed = row_lookup.loc[(context_name, client, scenario, policy)]
                                check(observed.assigned and observed.assignment == assignment and observed.horizon == h and observed.loss_tolerance == loss
                                      and observed.candidate_local_index == product["candidate_local_index"]
                                      and observed.candidate_original232_index == product["candidate_original232_index"], "actual_client_identity")
                                metrics = independent_metrics(path, loss)
                                metrics["trade_fraction"] = metrics.pop("actual_trade_fraction")
                                metrics["target_execution_fraction"] = float(np.mean(path["target_execution"]))
                                for field in [*METRICS, *ACCOUNTS, "breach", "excess"]: close(observed[field], metrics[field], "actual_metrics/" + field)
                                if path["first_crossing"] < 0: check(pd.isna(observed.first_crossing), "actual_first_crossing_null")
                                else: check(observed.first_crossing == path["first_crossing"], "actual_first_crossing")
                                if policy == "pathwise_hold_rollout":
                                    decision_rows = gate_lookup.loc[(context_name, client, scenario)]
                                    check(list(decision_rows.index) == list(range(1, h + 1)), "actual_decision_membership")
                                    for t in range(h):
                                        gate = decision_rows.loc[t + 1]
                                        close([gate.holding_equity, gate.holding_bond, gate.holding_bill], path["holdings"][t], "CSVgate/holdings")
                                        for field, reference_field in [("cpi", "cpi"), ("real_peak", "real_peak"), ("running_mdd", "running_mdd")]: close(gate[field], path[reference_field][t], "CSVgate/" + field)
                                        check(gate.action == path["actions"][t] and gate.ever_breached == path["ever_breached"][t]
                                              and gate.actual_trade == path["trades"][t], "CSVgate/action_flags")
                                        close(gate.fee, path["fees"][t], "CSVgate/fee"); close(gate.moved, path["moved"][t], "CSVgate/moved")
                                        for field in ["qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "forecast_n", "min_log_advantage_hold", "growth_violation_count", "breach_violation_count", "union_violation_count", "delete_one_flip_count"]:
                                            value = saved_arrays[field][index, t]
                                            if (field not in ["qg_hold", "qg_restore", "min_log_advantage_hold"] and value < 0) or not np.isfinite(value): check(pd.isna(gate[field]), "CSVgate/inactive_null")
                                            else: close(gate[field], value, "CSVgate/" + field)
                        reference_by_policy[policy] = paths
                    if kind == "training":
                        for index, scenario in enumerate(identity_ids):
                            a, b = reference_by_policy[POLICIES[0]][index], reference_by_policy[POLICIES[1]][index]
                            delta = math.log(b["real_wealth"][-1] / 100) - math.log(a["real_wealth"][-1] / 100)
                            own_property_deltas.append(delta)
                            check(delta >= -PROPERTY_GUARD and (not b["ever_breached"][-1] or a["ever_breached"][-1]), "independent_training_support_property")
                            observed = support_lookup.loc[(context_name, client, scenario)]
                            check(observed.assigned and observed.horizon == h and observed.loss_tolerance == loss
                                  and observed.candidate_local_index == product["candidate_local_index"]
                                  and observed.candidate_original232_index == product["candidate_original232_index"], "training_support_identity")
                            close(observed.total_log_growth_delta, delta, "training_support/total_log_growth_delta", atol=3e-12, rtol=0)
                            check(observed.reference_breach == a["ever_breached"][-1] and observed.pathwise_breach == b["ever_breached"][-1], "training_support_breach")
                            check(observed.computed_growth_delta_strictly_negative == (observed.total_log_growth_delta < 0)
                                  and observed.growth_within_numerical_audit_guard == (observed.total_log_growth_delta >= -PROPERTY_GUARD)
                                  and observed.breach_order_preserved == (not observed.pathwise_breach or observed.reference_breach), "training_reported_property_flags")
                scenario = min(eval_ids)
                path = actual_references[(scenario, "pathwise_hold_rollout")]
                for year in years:
                    identity = {"context": context_name, "client": client, "scenario_id": scenario, "year": year}
                    gate = gate_lookup.loc[(context_name, client, scenario, year)]
                    if pd.isna(gate.qg_hold):
                        check(path["ever_breached"][year - 1], "Qsubset_skip_is_pending")
                        q_results.append({**identity, "status": "skipped_already_pending_no_replacement"}); continue
                    values = scalar_fan(path["holdings"][year - 1], path["cpi"][year - 1], path["real_peak"][year - 1], path["running_mdd"][year - 1],
                        target, loss, returns[train_ix, :h], inflation[train_ix, :h], year - 1)
                    hold, restore = values["hold"], values["restore"]
                    difference = hold["logs"] - restore["logs"]
                    growth_bad = difference < 0
                    risk_bad = hold["bad"] & ~restore["bad"]
                    union = growth_bad | risk_bad
                    independent_counts = {"growth_violation_count": int(sum(growth_bad)), "breach_violation_count": int(sum(risk_bad)),
                                          "union_violation_count": int(sum(union)), "delete_one_flip_count": int(sum(union) == 1)}
                    close(gate.qg_hold, hold["mean"], "Qsubset/qg_hold", atol=3e-12, rtol=0)
                    close(gate.qg_restore, restore["mean"], "Qsubset/qg_restore", atol=3e-12, rtol=0)
                    close(gate.min_log_advantage_hold, min(difference), "Qsubset/min_log_advantage_hold", atol=3e-12, rtol=0)
                    check(gate.qbad_hold_count == hold["count"] and gate.qbad_restore_count == restore["count"], "Qsubset/risk_counts")
                    disagreement = {field: {"saved": int(gate[field]), "independent": expected} for field, expected in independent_counts.items() if gate[field] != expected}
                    independent_action = "restore" if union.any() else "hold"
                    if gate.action != independent_action: disagreement["action"] = {"saved": gate.action, "independent": independent_action}
                    if disagreement: q_disagreements.append({**identity, "differences": disagreement,
                        "minimum_absolute_branch_log_advantage": float(np.min(np.abs(difference))),
                        "independent_smallest_absolute_advantages": sorted(map(float, difference), key=abs)[:10]})
                    q_results.append({**identity, "status": "audited" if not disagreement else "audited_numeric_classification_review_needed",
                        "training_paths": len(train_ix), "qg_hold": hold["mean"], "qg_restore": restore["mean"],
                        "qbad_hold_count": hold["count"], "qbad_restore_count": restore["count"], "min_log_advantage_hold": float(min(difference)),
                        **independent_counts, "independent_action": independent_action})
    check(replay_counts == {"actual_paths": 1440, "training_paths": 3600, "actual_decisions": 24480, "training_decisions": 61200}, "complete_main_and_training_replays")
    check(len(q_results) == 90, "all_Q_identities_and_Review_auxiliary_skips")
    for (scope, client, policy), group in rows.groupby(["scope", "client", "policy"]): check(len(group) == 120 and set(group.scenario_id) == set(range(1, 121)), "full_OOF120_denominators")
    for record in support_summary.itertuples(index=False):
        group = support.loc[support.context.eq(record.context) & support.client.eq(record.client)]
        auto = group.loc[group.assigned]
        check(record.training_support_paths == len(group) and record.assigned_paths == len(auto) and record.review_paths == len(group) - len(auto), "training_summary_denominators")
        check(record.gate_growth_allowance == 0 and record.numerical_audit_guard_total_log_units == PROPERTY_GUARD, "training_guard_not_gate")
        if len(auto):
            close(record.minimum_total_log_growth_delta, auto.total_log_growth_delta.min(), "training_summary/min_delta")
            close(record.mean_total_log_growth_delta, auto.total_log_growth_delta.mean(), "training_summary/mean_delta")
            check(record.strictly_negative_computed_growth_delta_count == int((auto.total_log_growth_delta < 0).sum())
                  and record.negative_beyond_audit_guard_count == int((auto.total_log_growth_delta < -PROPERTY_GUARD).sum())
                  and record.breach_order_violations == int((auto.pathwise_breach.astype(bool) & ~auto.reference_breach.astype(bool)).sum())
                  and record.reference_breach_count == int(auto.reference_breach.astype(bool).sum())
                  and record.pathwise_breach_count == int(auto.pathwise_breach.astype(bool).sum()), "training_summary_property_counts")
        else: check(pd.isna(record.minimum_total_log_growth_delta) and pd.isna(record.reference_breach_count), "training_summary_Review_null")
    for record in pd.read_csv(directory / "summary.csv").itertuples(index=False):
        group = rows.loc[rows.scope.eq(record.scope) & rows.client.eq(record.client) & rows.policy.eq(record.policy)]
        auto = group.loc[group.assigned]
        check(record.total_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "summary_denominators")
        if not len(auto): check(pd.isna(record.mean_log_growth) and pd.isna(record.breach_probability), "summary_Review_null"); continue
        stats = {"mean_log_growth": auto.log_growth.mean(), "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
            "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(auto.mdd),
            "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": auto.breach.astype(bool).mean(), "mean_excess": auto.excess.mean(),
            **{"mean_" + field: auto[field].mean() for field in ACCOUNTS}}
        for field, value in stats.items(): close(getattr(record, field), value, "summary/" + field)
    for record in pd.read_csv(directory / "paired_summary.csv").itertuples(index=False):
        a = rows.loc[rows.scope.eq(record.scope) & rows.client.eq(record.client) & rows.policy.eq("annual_then_pending")].set_index(["context", "scenario_id"])
        b = rows.loc[rows.scope.eq(record.scope) & rows.client.eq(record.client) & rows.policy.eq("pathwise_hold_rollout")].set_index(["context", "scenario_id"]).loc[a.index]
        auto = a.assigned & b.assigned
        check(record.total_pairs == 120 and record.assigned_pairs == int(auto.sum()) and record.review_pairs == int((~auto).sum()), "paired_summary_denominators")
        for field in [*METRICS, *ACCOUNTS, "excess"]:
            expected = float((b.loc[auto, field] - a.loc[auto, field]).mean()) if auto.any() else None
            if expected is None: check(pd.isna(getattr(record, "mean_delta_" + field)), "paired_summary_null")
            else: close(getattr(record, "mean_delta_" + field), expected, "paired_summary/" + field)
        if auto.any(): check(record.breach_count_delta == int(b.loc[auto, "breach"].astype(bool).sum() - a.loc[auto, "breach"].astype(bool).sum()), "paired_summary_breach_delta")
        else: check(pd.isna(record.breach_count_delta), "paired_summary_Review_breach_null")
    for kind in ["sources", "input_hashes"]:
        for name, expected in manifest[kind].items(): check(sha(ROOT / name) == expected, kind + "/" + name)
    for name, expected in manifest["outputs_sha256"].items(): check(sha(directory / name) == expected, "output_hash/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_hashes"] == manifest["input_hashes_at_end"], "original_source_input_start_end")
    after = snapshot(directory); check(before == after, "audit_source_input_stable")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not q_disagreements else "numerical_classification_review_needed",
        "checks_passed": checks, "maximum_errors": maximum_errors, "replay_counts": replay_counts,
        "client_rows": len(rows), "training_support_rows": len(support), "training_support_assigned_pairs": int(support.assigned.sum()),
        "independent_training_min_total_log_delta": min(own_property_deltas), "independent_training_strict_negative_count": sum(value < 0 for value in own_property_deltas),
        "independent_training_negative_beyond_1e10_count": sum(value < -PROPERTY_GUARD for value in own_property_deltas),
        "Q_subset_results": q_results, "Q_subset_requested_served_identities": sum(item["status"] != "auxiliary_onboarding_Review_skip" for item in q_results),
        "Q_subset_audited": sum(item["status"].startswith("audited") for item in q_results),
        "Q_subset_pending_skipped": sum(item["status"] == "skipped_already_pending_no_replacement" for item in q_results),
        "Q_subset_classification_disagreements": q_disagreements,
        "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
        "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(), "final_scenarios_read": False,
        "gate_growth_allowance": 0, "training_property_audit_guard": PROPERTY_GUARD,
        "independence": "Own prior-audit100-step bisection, actual path state reconstruction, scalar fan current-action then B, metrics/tails; no reviewed kernel/runner/metrics helper imported",
        "limitations": ["Any zero-threshold numerical classification discrepancy is explicit and requires review, not an extra gate epsilon",
                        "Support induction concerns fixed training calendar paths, including overlapping contexts; no OOF or population improvement/certification follows",
                        "The1e-10 log-unit property guard is only a disclosed audit convention, not an action parameter or proved numerical error bound",
                        "All actual/training financial transitions are replayed, while independently recomputed Q arrays cover only the prospective deterministic subset",
                        "No menu/assignment/threshold retuning, adviser effect, final service choice or fresh blind-test claim"],
        "original_manifests_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    result = run_audit(arguments.directory)
    destination = arguments.out or arguments.directory / "independent_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "replay_counts", "Q_subset_requested_served_identities", "Q_subset_audited", "Q_subset_pending_skipped", "Q_subset_classification_disagreements", "independent_training_min_total_log_delta", "independent_training_strict_negative_count", "maximum_errors"]}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
