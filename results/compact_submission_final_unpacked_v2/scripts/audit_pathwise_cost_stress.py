"""Independent replay of the predeclared five known-cost experiments.

Only our independent scalar fee/state/fan references and the guarded loader
are reused. No reviewed execution kernel, runner, selector or metric helper
is imported. A zero-epsilon Q classification difference is disclosed rather
than repaired with a new tolerance. Original experiment artifacts are read-only.
"""
from __future__ import annotations

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

DIRECTORY = ROOT / "results/model_review/pathwise_cost_stress_v1"
BASE = ROOT / "results/model_review/pathwise_fan_execution_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PROTOCOL = ROOT / "docs/pathwise_cost_stress_protocol.json"
RATES = (("zero", 0.), ("half", .00125), ("base", .0025), ("double", .005), ("quadruple", .01))
POLICIES = ("annual_then_pending", "pathwise_hold_rollout")
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill")
QFLOAT = ("qg_hold", "qg_restore", "min_log_advantage_hold")
QINT = ("qbad_hold_count", "qbad_restore_count", "forecast_n", "growth_violation_count", "breach_violation_count", "union_violation_count", "delete_one_flip_count")
GUARD = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def snapshot():
    manifest = readj(DIRECTORY / "manifest.json")
    paths = [DIRECTORY / name for name in manifest["outputs_sha256"]]
    paths += [DIRECTORY / "manifest.json", Path(__file__).resolve(),
              ROOT / "scripts/audit_fan_rollout_execution.py", ROOT / "scripts/audit_review_pending_execution.py"]
    paths += [ROOT / name for name in manifest["sources"]]
    paths += [ROOT / name for name in manifest["input_hashes"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def main():
    before = snapshot()
    manifest, protocol = readj(DIRECTORY / "manifest.json"), readj(PROTOCOL)
    frozen, contexts = readj(STRICT / "nested_context_refits.json"), readj(DIRECTORY / "contexts.json")
    profiles = readj(ROOT / "docs/positive_asset_specification_protocol.json")["clients"]
    rows, gates = pd.read_csv(DIRECTORY / "client_paths.csv"), pd.read_csv(DIRECTORY / "decision_gates.csv")
    support = pd.read_csv(DIRECTORY / "training_support_paths.csv")
    support_summary = pd.read_csv(DIRECTORY / "training_support_summary.csv")
    checks, errors, q_results, q_disagreements = 0, {}, [], []
    counts = {"actual_paths": 0, "training_paths": 0, "actual_decisions": 0, "training_decisions": 0}
    property_deltas = {label: [] for label, _ in RATES}

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)

    def close(actual, expected, label, *, atol=3e-11, rtol=3e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape, label + "/shape")
        if a.dtype.kind in "bUiS" or b.dtype.kind in "bUiS":
            check(np.array_equal(a, b), label + "/exact")
        else:
            difference = np.abs(a - b)
            finite = np.isfinite(difference)
            field = label.split("/")[-1]
            errors[field] = max(errors.get(field, 0.), float(difference[finite].max(initial=0)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + "/numeric")

    check(contexts == frozen, "all_frozen_contexts_unchanged")
    check(manifest["actual_forecast_costs_identical"] is True, "known_cost_convention")
    check(manifest["gate_growth_allowance"] == 0 and manifest["numerical_property_audit_guard"] == GUARD,
          "zero_gate_distinct_from_property_audit_guard")
    check([(item["label"], item["cost_rate"]) for item in protocol["rates"]] == list(RATES), "prospective_rates")
    for name, expected in protocol["frozen_input_sha256"].items():
        check(sha(ROOT / name) == expected, "frozen_input/" + name)
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation):
        digest.update(np.ascontiguousarray(value).tobytes())
    check(ids.tolist() == list(range(1, 121)) and digest.hexdigest() == manifest["development_array_sha256"], "guarded_development_arrays")
    rkeys = ["cost_label", "context", "client", "scenario_id", "policy"]
    gkeys = rkeys + ["year"]
    skeys = rkeys[:-1]
    check(len(rows) == 12000 and not rows.duplicated(rkeys).any(), "all_actual_client_identities")
    check(len(support) == 15000 and not support.duplicated(skeys).any(), "all_training_client_identities")
    check(len(gates) == 122400 and not gates.duplicated(gkeys).any(), "all_actual_decisions")
    check(rows.loc[~rows.assigned, [*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing", "candidate_local_index", "candidate_original232_index"]].isna().all().all(), "actual_Review_null")
    check(support.loc[~support.assigned, ["reference_total_log_growth", "pathwise_total_log_growth", "total_log_growth_delta", "reference_breach", "pathwise_breach"]].isna().all().all(), "training_Review_null")
    rl, gl, sl = rows.set_index(rkeys).sort_index(), gates.set_index(gkeys).sort_index(), support.set_index(skeys).sort_index()
    with np.load(DIRECTORY / "actual_traces.npz", allow_pickle=False) as actual, np.load(DIRECTORY / "training_traces.npz", allow_pickle=False) as training:
        for label, rate in RATES:
            for context_name, fit in contexts.items():
                eval_ids = fit["training_ids"] if context_name == "full_development" else fit["outer_test_ids"]
                train_ids = fit["training_ids"]
                train_ix = np.asarray(train_ids) - 1
                check(context_name == "full_development" or not set(eval_ids) & set(train_ids), "outer_train_test_disjoint")
                for profile in profiles:
                    client, h, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
                    assignment = fit["effective_assignments"][client]
                    years = sorted({2, h // 2 + 1, h})
                    if assignment == "Review":
                        for year in years:
                            q_results.append({"cost_label": label, "context": context_name, "client": client,
                                              "scenario_id": min(eval_ids), "year": year, "status": "auxiliary_onboarding_Review_skip"})
                        for kind, identity_ids, frame in (("actual", eval_ids, rows), ("training", train_ids, support)):
                            group = frame.loc[frame.cost_label.eq(label) & frame.context.eq(context_name) & frame.client.eq(client)]
                            check(len(group) == len(identity_ids) * (2 if kind == "actual" else 1)
                                  and set(group.scenario_id) == set(identity_ids) and not group.assigned.any(), "Review_complete_denominators")
                        continue
                    item = next(product for product in fit["raw_fit"]["menu"] if product["name"] == assignment)
                    target = np.asarray(item["weights"])
                    check(np.min(target) >= .05 - 1e-12, "three_assets_positive")
                    actual_paths = {}
                    for kind, identity_ids, archive in (("actual", eval_ids, actual), ("training", train_ids, training)):
                        by_policy = {}
                        for policy in POLICIES:
                            prefix = f"{label}__{context_name}__{client}__{policy}__"
                            saved = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                            check(np.array_equal(saved["scenario_ids"], identity_ids), kind + "/scenario_order")
                            paths = []
                            for i, sid in enumerate(identity_ids):
                                def action(t, x, cpi, peak, worst, bad):
                                    if t == 0:
                                        expected = "restore"
                                    elif bad:
                                        expected = "hold"
                                    elif policy == POLICIES[0]:
                                        expected = "restore"
                                    else:
                                        n = len(train_ids)
                                        v = {field: int(saved[field][i, t]) for field in ("growth_violation_count", "breach_violation_count", "union_violation_count")}
                                        check(all(0 <= value <= n for value in v.values()), kind + "/violation_count_range")
                                        check(max(v["growth_violation_count"], v["breach_violation_count"]) <= v["union_violation_count"]
                                              <= min(n, v["growth_violation_count"] + v["breach_violation_count"]), kind + "/union_logic")
                                        check((saved["min_log_advantage_hold"][i, t] < 0) == (v["growth_violation_count"] > 0), kind + "/strict_zero_growth_sign")
                                        check(saved["forecast_n"][i, t] == n and saved["delete_one_flip_count"][i, t] == int(v["union_violation_count"] == 1), kind + "/fan_denominator_influence")
                                        expected = "restore" if v["union_violation_count"] else "hold"
                                    check(saved["actions"][i, t] == expected, kind + "/action_timing")
                                    if policy == POLICIES[1]:
                                        active = t > 0 and not bad
                                        for field in QFLOAT:
                                            check(np.isfinite(saved[field][i, t]) if active else np.isnan(saved[field][i, t]), kind + "/active_score_mask")
                                        for field in QINT:
                                            check(saved[field][i, t] >= 0 if active else saved[field][i, t] == -1, kind + "/active_integer_mask")
                                    return expected
                                path = reference_actual(returns[sid - 1, :h], inflation[sid - 1, :h], target, loss, action, cost_rate=rate)
                                paths.append(path)
                                counts[kind + "_paths"] += 1
                                counts[kind + "_decisions"] += h
                                for field in path:
                                    if field in saved:
                                        close(saved[field][i], path[field], kind + "/trace/" + field)
                                if kind == "actual":
                                    actual_paths[(sid, policy)] = path
                                    record = rl.loc[(label, context_name, client, sid, policy)]
                                    check(record.assigned and record.assignment == assignment and record.horizon == h and record.loss_tolerance == loss
                                          and record.cost_rate == rate and record.candidate_local_index == item["candidate_local_index"]
                                          and record.candidate_original232_index == item["candidate_original232_index"], "actual_client_product_identity")
                                    metrics = independent_metrics(path, loss)
                                    metrics["trade_fraction"] = metrics.pop("actual_trade_fraction")
                                    metrics["target_execution_fraction"] = float(np.mean(path["target_execution"]))
                                    for field in (*METRICS, *ACCOUNTS, "breach", "excess"):
                                        close(record[field], metrics[field], "path_metrics/" + field)
                                    check(pd.isna(record.first_crossing) if path["first_crossing"] < 0 else record.first_crossing == path["first_crossing"], "first_crossing_or_null")
                                    group = gl.loc[(label, context_name, client, sid, policy)]
                                    check(list(group.index) == list(range(1, h + 1)), "CSV_decision_original_H")
                                    for t in range(h):
                                        gate = group.loc[t + 1]
                                        close([gate.holding_equity, gate.holding_bond, gate.holding_bill], path["holdings"][t], "CSV_gate/holdings")
                                        for field in ("cpi", "real_peak", "running_mdd"):
                                            close(gate[field], path[field][t], "CSV_gate/" + field)
                                        check(gate.action == path["actions"][t] and gate.ever_breached == path["ever_breached"][t]
                                              and gate.actual_trade == path["trades"][t] and gate.pending_at_decision == path["pending_at_decision"][t], "CSV_gate_action_state")
                                        for field in ("fees", "moved"):
                                            close(gate["fee" if field == "fees" else field], path[field][t], "CSV_gate/" + field)
                                        for field in (*QFLOAT, *QINT):
                                            value = saved[field][i, t] if field in saved else np.nan
                                            if not np.isfinite(value) or (field in QINT and value < 0):
                                                check(pd.isna(gate[field]), "CSV_gate_inactive_null")
                                            else:
                                                close(gate[field], value, "CSV_gate/" + field)
                            by_policy[policy] = paths
                        if kind == "training":
                            for i, sid in enumerate(identity_ids):
                                a, b = by_policy[POLICIES[0]][i], by_policy[POLICIES[1]][i]
                                ga, gb = math.log(a["real_wealth"][-1] / 100), math.log(b["real_wealth"][-1] / 100)
                                delta = gb - ga
                                property_deltas[label].append(delta)
                                check(delta >= -GUARD and (not b["ever_breached"][-1] or a["ever_breached"][-1]), "same_cost_training_support_property")
                                record = sl.loc[(label, context_name, client, sid)]
                                check(record.assigned and record.cost_rate == rate and record.horizon == h and record.loss_tolerance == loss
                                      and record.candidate_local_index == item["candidate_local_index"]
                                      and record.candidate_original232_index == item["candidate_original232_index"], "training_support_product_identity")
                                for field, value in (("reference_total_log_growth", ga), ("pathwise_total_log_growth", gb), ("total_log_growth_delta", delta)):
                                    close(record[field], value, "training_support/" + field, atol=3e-12, rtol=0)
                                check(record.reference_breach == a["ever_breached"][-1] and record.pathwise_breach == b["ever_breached"][-1], "training_support_breach")
                                check(record.computed_growth_delta_strictly_negative == (record.total_log_growth_delta < 0)
                                      and record.growth_within_numerical_audit_guard == (record.total_log_growth_delta >= -GUARD)
                                      and record.breach_order_preserved == (not record.pathwise_breach or record.reference_breach), "training_reported_property_flags")
                    sid = min(eval_ids)
                    path = actual_paths[(sid, POLICIES[1])]
                    for year in years:
                        identity = {"cost_label": label, "cost_rate": rate, "context": context_name, "client": client, "scenario_id": sid, "year": year}
                        gate = gl.loc[(label, context_name, client, sid, POLICIES[1], year)]
                        if pd.isna(gate.qg_hold):
                            check(path["ever_breached"][year - 1], "Q_skip_is_pending")
                            q_results.append({**identity, "status": "skipped_already_pending_no_replacement"})
                            continue
                        values = scalar_fan(path["holdings"][year - 1], path["cpi"][year - 1], path["real_peak"][year - 1], path["running_mdd"][year - 1],
                            target, loss, returns[train_ix, :h], inflation[train_ix, :h], year - 1, cost_rate=rate)
                        hold, restore = values["hold"], values["restore"]
                        difference = hold["logs"] - restore["logs"]
                        gbad, rbad = difference < 0, hold["bad"] & ~restore["bad"]
                        union = gbad | rbad
                        own = {"growth_violation_count": int(gbad.sum()), "breach_violation_count": int(rbad.sum()),
                               "union_violation_count": int(union.sum()), "delete_one_flip_count": int(union.sum() == 1)}
                        close(gate.qg_hold, hold["mean"], "Q_subset/qg_hold", atol=3e-12, rtol=0)
                        close(gate.qg_restore, restore["mean"], "Q_subset/qg_restore", atol=3e-12, rtol=0)
                        close(gate.min_log_advantage_hold, min(difference), "Q_subset/min_log_advantage_hold", atol=3e-12, rtol=0)
                        check(gate.qbad_hold_count == hold["count"] and gate.qbad_restore_count == restore["count"], "Q_subset_terminal_bad_counts")
                        differing = {field: {"saved": int(gate[field]), "independent": expected} for field, expected in own.items() if gate[field] != expected}
                        own_action = "restore" if union.any() else "hold"
                        if gate.action != own_action:
                            differing["action"] = {"saved": gate.action, "independent": own_action}
                        if differing:
                            q_disagreements.append({**identity, "differences": differing, "minimum_absolute_branch_advantage": float(np.min(np.abs(difference))),
                                                    "independent_nearest_zero_advantages": sorted(map(float, difference), key=abs)[:10]})
                        q_results.append({**identity, "status": "audited" if not differing else "audited_numeric_classification_review_needed",
                            "training_paths": len(train_ids), "qg_hold": hold["mean"], "qg_restore": restore["mean"],
                            "qbad_hold_count": hold["count"], "qbad_restore_count": restore["count"], "min_log_advantage_hold": float(min(difference)),
                            **own, "independent_action": own_action})
            print(json.dumps({"completed_cost": label, "checks_passed_so_far": checks}), flush=True)
    check(counts == {"actual_paths": 7200, "training_paths": 18000, "actual_decisions": 122400, "training_decisions": 306000}, "all_actual_training_replays")
    check(len(q_results) == 450, "all_prospective_Q_and_auxiliary_Review_identities")
    for (label, scope, client, policy), group in rows.groupby(["cost_label", "scope", "client", "policy"]):
        check(len(group) == 120 and set(group.scenario_id) == set(range(1, 121)), "full_OOF_120_denominators")
    for record in support_summary.itertuples(index=False):
        group = support.loc[support.cost_label.eq(record.cost_label) & support.context.eq(record.context) & support.client.eq(record.client)]
        auto = group.loc[group.assigned]
        check(record.training_support_paths == len(group) and record.assigned_paths == len(auto) and record.review_paths == len(group) - len(auto), "training_summary_denominators")
        check(record.gate_growth_allowance == 0 and record.numerical_audit_guard_total_log_units == GUARD, "training_summary_guard_not_gate")
        if len(auto):
            close(record.minimum_total_log_growth_delta, auto.total_log_growth_delta.min(), "training_summary/min_delta")
            close(record.mean_total_log_growth_delta, auto.total_log_growth_delta.mean(), "training_summary/mean_delta")
            check(record.strictly_negative_computed_growth_delta_count == int((auto.total_log_growth_delta < 0).sum())
                  and record.negative_beyond_audit_guard_count == int((auto.total_log_growth_delta < -GUARD).sum())
                  and record.breach_order_violations == int((auto.pathwise_breach.astype(bool) & ~auto.reference_breach.astype(bool)).sum())
                  and record.reference_breach_count == int(auto.reference_breach.astype(bool).sum())
                  and record.pathwise_breach_count == int(auto.pathwise_breach.astype(bool).sum()), "training_summary_counts")
        else:
            check(pd.isna(record.minimum_total_log_growth_delta) and pd.isna(record.reference_breach_count), "training_summary_Review_null")
    for record in pd.read_csv(DIRECTORY / "summary.csv").itertuples(index=False):
        group = rows.loc[rows.cost_label.eq(record.cost_label) & rows.scope.eq(record.scope) & rows.client.eq(record.client) & rows.policy.eq(record.policy)]
        auto = group.loc[group.assigned]
        check(record.total_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "summary_denominators")
        if not len(auto):
            check(pd.isna(record.mean_log_growth) and pd.isna(record.breach_probability), "summary_Review_null")
            continue
        expected = {"mean_log_growth": auto.log_growth.mean(), "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
                    "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(auto.mdd),
                    "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": auto.breach.astype(bool).mean(), "mean_excess": auto.excess.mean(),
                    **{"mean_" + field: auto[field].mean() for field in ACCOUNTS}}
        for field, value in expected.items():
            close(getattr(record, field), value, "summary/" + field)
    for record in pd.read_csv(DIRECTORY / "paired_summary.csv").itertuples(index=False):
        part = rows.loc[rows.cost_label.eq(record.cost_label) & rows.scope.eq(record.scope) & rows.client.eq(record.client)]
        a, b = [part.loc[part.policy.eq(policy)].set_index(["context", "scenario_id"]) for policy in POLICIES]
        b = b.loc[a.index]
        auto = a.assigned & b.assigned
        check(record.total_pairs == 120 and record.assigned_pairs == int(auto.sum()) and record.review_pairs == int((~auto).sum()), "paired_denominators")
        for field in (*METRICS, *ACCOUNTS, "excess"):
            if auto.any():
                close(getattr(record, "mean_delta_" + field), (b.loc[auto, field] - a.loc[auto, field]).mean(), "paired/" + field)
            else:
                check(pd.isna(getattr(record, "mean_delta_" + field)), "paired_Review_null")
        check(record.breach_count_delta == int(b.loc[auto, "breach"].astype(bool).sum() - a.loc[auto, "breach"].astype(bool).sum()) if auto.any() else pd.isna(record.breach_count_delta), "paired_breach_delta_or_null")

    # Reconstruct whole-trajectory disagreements from CSV identities; these
    # pair different account states, not a same-state marginal cost effect.
    changes = pd.read_csv(DIRECTORY / "decision_changes_by_path.csv")
    change_summary = pd.read_csv(DIRECTORY / "decision_changes_summary.csv")
    cl = changes.set_index(rkeys).sort_index()
    check(len(changes) == len(rows) and not changes.duplicated(rkeys).any(), "change_all_client_rows")
    baseline_gates = gates.loc[gates.cost_label.eq("base")].set_index(gkeys[1:]).sort_index()
    for key, group in gates.groupby(rkeys, sort=False):
        label, context, client, sid, policy = key
        group = group.set_index("year").sort_index()
        baseline = baseline_gates.loc[(context, client, sid, policy)].loc[group.index]
        row = cl.loc[key]
        changed = group.action != baseline.action
        both = (group.index > 1) & ~group.pending_at_decision & ~baseline.pending_at_decision
        check(row.assigned and row.all_decisions == len(group) and row.both_active_postinitial_decisions == int(both.sum()), "change_denominators")
        for kind, difference in (("action", changed), ("trade_flag", group.actual_trade != baseline.actual_trade), ("pending_flag", group.pending_at_decision != baseline.pending_at_decision)):
            check(row[kind + "_changed_count"] == int(difference.sum()), "whole_trajectory_change_count")
            close(row[kind + "_changed_fraction"], difference.mean(), "whole_trajectory_change/fraction")
        check(row.both_active_action_changed_count == int((changed & both).sum()), "both_active_change_count")
    check(changes.loc[~changes.assigned, ["all_decisions", "action_changed_count", "trade_flag_changed_count", "pending_flag_changed_count", "both_active_postinitial_decisions"]].isna().all().all(), "Review_changes_null")
    for record in change_summary.itertuples(index=False):
        group = changes.loc[changes.cost_label.eq(record.cost_label) & changes.scope.eq(record.scope) & changes.client.eq(record.client) & changes.policy.eq(record.policy)]
        auto = group.loc[group.assigned]
        check(record.original_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "change_summary_complete_paths")
        if not len(auto):
            check(pd.isna(record.all_decisions) and pd.isna(record.action_changed_fraction), "change_summary_Review_null")
            continue
        n, nb = int(auto.all_decisions.sum()), int(auto.both_active_postinitial_decisions.sum())
        check(record.all_decisions == n and record.both_active_postinitial_decisions == nb, "change_summary_decision_denominators")
        for kind in ("action", "trade_flag", "pending_flag"):
            expected = int(auto[kind + "_changed_count"].sum())
            check(getattr(record, kind + "_changed_count") == expected, "change_summary_counts")
            close(getattr(record, kind + "_changed_fraction"), expected / n, "change_summary/fraction")
        if nb:
            close(record.both_active_action_changed_fraction, auto.both_active_action_changed_count.sum() / nb, "change_summary/both_active_fraction")
        else:
            check(pd.isna(record.both_active_action_changed_fraction), "no_both_active_null")

    # Independently compare the complete base-rate CSV and NPZ values to v2.
    base_receipts = {}
    frame_specs = (("client_paths.csv", rows, rkeys[1:]), ("summary.csv", pd.read_csv(DIRECTORY / "summary.csv"), ["scope", "client", "policy"]),
                   ("paired_summary.csv", pd.read_csv(DIRECTORY / "paired_summary.csv"), ["scope", "client", "policy"]),
                   ("training_support_paths.csv", support, skeys[1:]), ("training_support_summary.csv", support_summary, ["context", "client"]))
    for name, table, keys in frame_specs:
        original = pd.read_csv(BASE / name).set_index(keys).sort_index()
        new = table.loc[table.cost_label.eq("base")].set_index(keys).sort_index()
        check(new.index.equals(original.index), "base_CSV_row_identity/" + name)
        for field in original:
            check(np.array_equal(new[field].isna(), original[field].isna()), "base_CSV_nullmask/" + field)
            keep = original[field].notna()
            if pd.api.types.is_numeric_dtype(original[field].dtype):
                close(new.loc[keep, field].to_numpy(), original.loc[keep, field].to_numpy(), "base_CSV/" + field)
            else:
                check(np.array_equal(new.loc[keep, field].to_numpy(), original.loc[keep, field].to_numpy()), "base_CSV_text/" + field)
        base_receipts[name] = {"rows": len(original), "columns_compared": len(original.columns)}
    for name in ("actual_traces.npz", "training_traces.npz"):
        with np.load(BASE / name, allow_pickle=False) as original, np.load(DIRECTORY / name, allow_pickle=False) as new:
            selected_keys = [key.removeprefix("base__") for key in new.files if key.startswith("base__")]
            check(set(selected_keys) == set(original.files), "base_NPZ_key_identity")
            for key in original.files:
                close(new["base__" + key], original[key], "base_NPZ/" + key.rsplit("__", 1)[-1])
            base_receipts[name] = {"array_fields_compared": len(original.files)}
    for kind in ("sources", "input_hashes"):
        for name, expected in manifest[kind].items():
            check(sha(ROOT / name) == expected, kind + "/" + name)
    for name, expected in manifest["outputs_sha256"].items():
        check(sha(DIRECTORY / name) == expected, "output_hash/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_hashes"] == manifest["input_hashes_at_end"], "original_start_end_hashes")
    after = snapshot()
    check(before == after, "independent_audit_source_input_stability")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not q_disagreements else "numerical_classification_review_needed", "checks_passed": checks,
        "replay_counts": counts, "maximum_errors": errors, "client_rows": len(rows), "training_support_rows": len(support),
        "training_property_by_cost": {label: {"assigned_pairs": len(values), "minimum_independent_total_log_delta": min(values),
            "strict_negative_computed_count": sum(value < 0 for value in values), "negative_beyond_audit_guard_count": sum(value < -GUARD for value in values)} for label, values in property_deltas.items()},
        "Q_subset_results": q_results, "Q_subset_requested_served_identities": sum(item["status"] != "auxiliary_onboarding_Review_skip" for item in q_results),
        "Q_subset_audited": sum(item["status"].startswith("audited") for item in q_results), "Q_subset_pending_skipped": sum(item["status"] == "skipped_already_pending_no_replacement" for item in q_results),
        "Q_subset_classification_disagreements": q_disagreements, "base_reproduction_independent_receipts": base_receipts,
        "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
        "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(), "final_scenarios_read": False,
        "gate_growth_allowance": 0, "training_property_audit_guard": GUARD,
        "independence": "Own prior-audit100-step bisection, actual state replay, scalar current-action then B fan, path metrics/tails; no reviewed kernel/runner/metric imported",
        "limitations": ["All five rates are hypothetical known costs shared by actual and forecast execution, not estimated costs or cost uncertainty",
            "Each complete training support check is conditional on its frozen empirical fan; overlapping contexts and rates are not independent observations",
            "Any exact-zero gate classification discrepancy is explicit; the1e-10 support check is only an audit convention, not a new gate margin or general error bound",
            "Every saved real transition is replayed, while independent future Q arrays cover the prospective deterministic subset only",
            "Cost/base action changes compare entire trajectories with different states, not a same-state causal marginal fee effect",
            "No cost winner, target/assignment/gamma retuning, OOF risk certificate or final-set access"],
        "original_manifests_modified": False}
    destination = DIRECTORY / "independent_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_passed", "replay_counts", "Q_subset_requested_served_identities", "Q_subset_audited", "Q_subset_pending_skipped", "Q_subset_classification_disagreements", "training_property_by_cost", "maximum_errors")}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
