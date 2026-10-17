"""Independent physical/Q/provenance audit of frozen-controller cost stress.

Only our own earlier independent references are imported. Controller choices
are read from frozen parents, never selected again at a stress cost. Every
actual transition is replayed; future Q values cover the declared subset.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_bounded_target_updates import independent_choice, reference_target_path
from audit_fan_rollout_execution import reference_actual, scalar_fan
from audit_review_pending_execution import independent_metrics, independent_cvar

OUT = ROOT / "results/model_review/selected_controller_cost_stress_v1"
PRICE = ROOT / "results/model_review/risk_priced_fan_execution_v1"
BOUNDED = ROOT / "results/model_review/bounded_target_updates_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PROTOCOL = ROOT / "docs/selected_controller_cost_stress_protocol.json"
RATES = (("zero", 0.), ("half", .00125), ("base", .0025), ("double", .005), ("quadruple", .01))
POLICIES = ("fixed_target_learner", "expanded_target_learner", "annual_then_pending", "equal_thirds_annual")
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill")
DELTAS = (*METRICS, *ACCOUNTS, "breach", "excess", "target_update_fraction", "target_update_count")
H, LOSS, TOL = 30, .4, 1e-12


def readj(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    manifest = readj(OUT / "manifest.json")
    paths = [OUT / name for name in manifest["outputs_sha256"]] + [OUT / "manifest.json", Path(__file__).resolve()]
    paths += [ROOT / ("scripts/" + name) for name in ("audit_bounded_target_updates.py", "audit_fan_rollout_execution.py", "audit_review_pending_execution.py")]
    paths += [ROOT / name for name in manifest["sources"]] + [ROOT / name for name in manifest["input_hashes"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def main():
    before = snapshot()
    manifest, protocol = readj(OUT / "manifest.json"), readj(PROTOCOL)
    plan = readj(OUT / "frozen_controller_plan.json")
    fits = readj(STRICT / "nested_context_refits.json")
    parent_price, parent_update = readj(PRICE / "context_selections.json"), readj(BOUNDED / "context_selections.json")
    rows, gates = pd.read_csv(OUT / "client_paths.csv"), pd.read_csv(OUT / "decision_gates.csv")
    checks, errors, q_results, q_disagreements = 0, {}, [], []
    arm_paths = periods = 0

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition: raise AssertionError(label)

    def close(actual, expected, label, *, atol=3e-11, rtol=3e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape, label + "/shape")
        if a.dtype.kind in "bUiS" or b.dtype.kind in "bUiS":
            check(np.array_equal(a, b), label + "/exact")
        else:
            with np.errstate(invalid="ignore"):
                difference = np.abs(a - b)
            finite = np.isfinite(difference); field = label.split("/")[-1]
            errors[field] = max(errors.get(field, 0.), float(difference[finite].max(initial=0)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + "/numeric")

    check(plan == protocol["frozen_controller_plan"], "exact_predeclared_frozen_plan")
    check(manifest["controller_parameters_frozen_at_default_cost"] is True
          and manifest["same_actual_forecast_known_cost"] is True
          and manifest["training_risk_recalibrated_at_stress_costs"] is False, "no_reselection_or_risk_recalibration")
    check([(item["label"], item["cost_rate"]) for item in protocol["rates"]] == list(RATES), "all_declared_costs")
    for name, expected in protocol["frozen_input_sha256"].items():
        check(sha(ROOT / name) == expected, "frozen_input/" + name)
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    check(ids.tolist() == list(range(1, 121)) and digest.hexdigest() == manifest["development_array_sha256"], "guarded_parent_data_identity")
    rkeys = ["cost_label", "context", "scenario_id", "policy"]
    gkeys = rkeys + ["year"]
    check(len(rows) == 4800 and not rows.duplicated(rkeys).any(), "all_original_path_cost_policy_rows")
    check(not gates.duplicated(gkeys).any(), "unique_all_actual_decisions")
    check(rows.loc[~rows.assigned, [*DELTAS, "terminal_target_index", "first_crossing"]].isna().all().all(), "inherited_Review_null")
    rl, gl = rows.set_index(rkeys).sort_index(), gates.set_index(gkeys).sort_index()
    with np.load(OUT / "actual_traces.npz", allow_pickle=False) as archive:
        for label, rate in RATES:
            for context, entry in plan.items():
                fit = fits[context]; training_ids, evaluation_ids = entry["training_ids"], entry["evaluation_ids"]
                expected_eval = fit["training_ids"] if context == "full_development" else fit["outer_test_ids"]
                check(training_ids == fit["training_ids"] and evaluation_ids == expected_eval, "frozen_fold_identities")
                check(context == "full_development" or not set(training_ids) & set(evaluation_ids), "outer_no_test_in_training")
                menu = {item["name"]: item for item in fit["raw_fit"]["menu"]}
                check(entry["menu"] == [menu[role] for role in ("Conservative", "Balanced", "Growth")], "frozen_menu_and_mapping")
                matrix = np.asarray([item["weights"] for item in entry["menu"]])
                check(np.min(matrix) >= .05 - TOL, "all_three_assets_positive")
                tx = np.asarray(training_ids) - 1
                for policy in POLICIES:
                    learner = policy in POLICIES[:2]
                    chosen = entry["learners"].get(policy)
                    if learner:
                        expected_order = parent_price[context]["profiles"]["C"]["selected_candidate_order"] if policy == POLICIES[0] else parent_update[context]["selected_controller_order"]
                        check(chosen["selected_controller_order"] == expected_order, "same_default_parent_controller")
                        served = expected_order is not None
                        price_order = chosen["price_order"] if served else None
                        updates = chosen["updates_enabled"] if served else None
                        if served:
                            check(expected_order == price_order + (9 if updates else 0), "frozen_price_mode_identity")
                    else:
                        served, price_order, updates = True, None, False
                    group = rows.loc[rows.cost_label.eq(label) & rows.context.eq(context) & rows.policy.eq(policy)]
                    check(len(group) == len(evaluation_ids) and set(group.scenario_id) == set(evaluation_ids)
                          and group.assigned.eq(served).all() and group.cost_rate.eq(rate).all(), "original_service_coverage_at_every_cost")
                    prefix = f"{label}__{context}__C__{policy}__"
                    if not served:
                        check(not any(key.startswith(prefix) for key in archive.files), "no_fabricated_Review_trace")
                        for year in (2, 16, 30):
                            q_results.append({"cost_label": label, "context": context, "policy": policy, "scenario_id": min(evaluation_ids), "year": year, "status": "execution_Review_skip"})
                        continue
                    saved = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                    check(np.array_equal(saved["scenario_ids"], evaluation_ids), "actual_scenario_order")
                    if learner:
                        check(saved["annual_prices"].shape == (1,), "single_actually_executed_price_metadata")
                        if price_order < 8:
                            close(saved["annual_prices"], np.array([PRICES[price_order]]), "frozen_metadata/annual_price")
                        else:
                            check(np.isposinf(saved["annual_prices"][0]), "risk_first_metadata_label")
                    actual_paths = {}
                    for i, sid in enumerate(evaluation_ids):
                        if policy == POLICIES[3]:
                            check(np.all(saved["actions"][i] == "restore") and not saved["pending_at_decision"][i].any(), "equal_thirds_remains_annual_after_breach")
                            path = reference_actual(returns[sid - 1, :H], inflation[sid - 1, :H], np.ones(3) / 3, LOSS, lambda *args: "restore", cost_rate=rate)
                            path["pending_at_decision"] = np.zeros(H, bool)
                            for field in ("current_target_index", "current_target_before", "current_target_after"):
                                path[field] = np.full(H + 1 if field == "current_target_index" else H, -1)
                            path["target_updated"] = np.zeros(H, bool); path["action_index"] = np.full(H, -1)
                        else:
                            def decide(t, x, cpi, peak, worst, bad, current):
                                if t == 0: action = 2
                                elif bad: action = 0
                                elif not learner: action = 2
                                else:
                                    available = saved["available_actions"][i, t]
                                    expected = np.ones(4, bool) if updates else (np.arange(4) == 0) | (np.arange(4) == current + 1)
                                    check(np.array_equal(available, expected), "available_actions_respect_frozen_mode")
                                    n = int(saved["forecast_n"][i, t])
                                    check(n == len(training_ids), "frozen_train_fan_denominator")
                                    action, score = independent_choice(saved["qg_actions"][i, t], saved["qbad_actions"][i, t], available, n, PRICES[price_order], current)
                                    close(saved["action_scores"][i, t], score, "actual_gate/action_scores", atol=3e-12, rtol=0)
                                    if price_order < 8:
                                        close(saved["selected_score"][i, t], score[action], "actual_gate/selected_score", atol=3e-12, rtol=0)
                                        check(action == 0 or score[action] > TOL, "positive_recorded_finite_net_benefit")
                                check(saved["action_index"][i, t] == action, "actual_action_and_pending_timing")
                                after = current if action == 0 else action - 1
                                check(saved["current_target_before"][i, t] == current and saved["current_target_after"][i, t] == after,
                                      "target_before_after_identity")
                                return int(action)
                            path = reference_target_path(returns[sid - 1, :H], inflation[sid - 1, :H], matrix, decide, cost_rate=rate)
                        actual_paths[sid] = path; arm_paths += 1; periods += H
                        for field in path:
                            if field in saved: close(saved[field][i], path[field], "actual_trace/" + field)
                        row = rl.loc[(label, context, sid, policy)]
                        metrics = independent_metrics(path, LOSS)
                        metrics["trade_fraction"] = metrics.pop("actual_trade_fraction")
                        metrics["target_execution_fraction"] = float(path["target_execution"].mean())
                        for field in (*METRICS, *ACCOUNTS, "breach", "excess"):
                            close(row[field], metrics[field], "actual_metrics/" + field)
                        check(pd.isna(row.first_crossing) if path["first_crossing"] < 0 else row.first_crossing == path["first_crossing"], "actual_first_crossing_or_null")
                        check(row.target_update_count == int(path["target_updated"].sum()) and row.terminal_target_index == path["current_target_index"][-1], "actual_target_count_terminal")
                        close(row.target_update_fraction, path["target_updated"].mean(), "actual_metrics/target_update_fraction")
                        gg = gl.loc[(label, context, sid, policy)]
                        check(list(gg.index) == list(range(1, H + 1)), "all_original_decisions")
                        for t in range(H):
                            gate = gg.loc[t + 1]
                            check(gate.action == path["actions"][t] and gate.action_index == path["action_index"][t]
                                  and gate.actual_trade == path["trades"][t] and gate.target_before == path["current_target_before"][t]
                                  and gate.target_after == path["current_target_after"][t] and gate.target_updated == path["target_updated"][t]
                                  and gate.pending_at_decision == path["pending_at_decision"][t] and gate.ever_breached == path["ever_breached"][t], "CSV_actual_action_target_flags")
                            close(gate.nominal_fee, path["fees"][t], "CSV_gate/fee"); close(gate.nominal_moved, path["moved"][t], "CSV_gate/moved")
                            active = learner and t > 0 and not path["ever_breached"][t]
                            check(gate.automatic_gate_evaluated == active, "CSV_automatic_gate_active")
                            for a in range(4):
                                available = active and bool(saved["available_actions"][i, t, a])
                                check(gate[f"action_{a}_available"] == available, "CSV_action_available")
                                for stem, key in (("qg", "qg_actions"), ("bad_count", "qbad_actions"), ("score", "action_scores")):
                                    if available and (stem != "score" or price_order < 8):
                                        close(gate[f"action_{a}_{stem}"], saved[key][i, t, a], "CSV_gate/" + stem)
                                    else: check(pd.isna(gate[f"action_{a}_{stem}"]), "CSV_inactive_Q_null")
                    if learner:
                        sid = min(evaluation_ids); path = actual_paths[sid]
                        for year in (2, 16, 30):
                            identity = {"cost_label": label, "context": context, "policy": policy, "scenario_id": sid, "year": year}
                            gate = gl.loc[(label, context, sid, policy, year)]
                            if not gate.automatic_gate_evaluated:
                                check(path["ever_breached"][year - 1], "Q_subset_pending_skip")
                                q_results.append({**identity, "status": "pending_skip_no_replacement"}); continue
                            current = int(path["current_target_index"][year - 1]); n = len(training_ids)
                            mvalues = range(3) if updates else [current]
                            fan = {m: scalar_fan(path["holdings"][year - 1], path["cpi"][year - 1], path["real_peak"][year - 1], path["running_mdd"][year - 1],
                                matrix[m], LOSS, returns[tx, :H], inflation[tx, :H], year - 1, cost_rate=rate) for m in mvalues}
                            qg, qb, available = np.full(4, np.nan), np.full(4, -1, int), np.zeros(4, bool)
                            qg[0], qb[0], available[0] = fan[current]["hold"]["mean"], fan[current]["hold"]["count"], True
                            for m in mvalues:
                                qg[m + 1], qb[m + 1], available[m + 1] = fan[m]["restore"]["mean"], fan[m]["restore"]["count"], True
                            for a in np.flatnonzero(available):
                                close(gate[f"action_{a}_qg"], qg[a], "Q_subset/qg", atol=3e-12, rtol=0)
                                check(gate[f"action_{a}_bad_count"] == qb[a], "Q_subset_integer_risk")
                            action, score = independent_choice(qg, qb, available, n, PRICES[price_order], current)
                            if gate.action_index != action:
                                q_disagreements.append({**identity, "saved": int(gate.action_index), "independent": int(action)})
                            q_results.append({**identity, "status": "audited" if gate.action_index == action else "numeric_action_review_needed", "training_paths": n,
                                "available_actions": available.tolist(), "independent_qg": [float(value) if np.isfinite(value) else None for value in qg], "independent_bad": qb.tolist(), "independent_action": int(action)})
            print(json.dumps({"completed_cost": label, "checks_passed_so_far": checks}), flush=True)
    check(arm_paths == int(rows.assigned.sum()) and periods == len(gates) == int(rows.loc[rows.assigned, "horizon"].sum()), "all_saved_real_transitions_replayed")
    check(len(q_results) == 180, "all_prospective_learner_Q_identities")
    for (_, _, _), group in rows.groupby(["cost_label", "scope", "policy"]):
        check(len(group) == 120 and set(group.scenario_id) == set(range(1, 121)), "all_original120_denominators")
    for (_, _), group in rows.groupby(["context", "policy"]):
        check(group.groupby("scenario_id").assigned.nunique().eq(1).all(), "admission_never_relearned_at_stress_cost")
    for record in pd.read_csv(OUT / "summary.csv").itertuples(index=False):
        group = rows.loc[rows.cost_label.eq(record.cost_label) & rows.scope.eq(record.scope) & rows.policy.eq(record.policy)]; auto = group.loc[group.assigned]
        check(record.total_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "summary_coverage")
        if not len(auto): check(pd.isna(record.mean_log_growth) and pd.isna(record.breach_probability), "summary_Review_null"); continue
        values = {"mean_log_growth": auto.log_growth.mean(), "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
            "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(auto.mdd),
            "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": auto.breach.astype(bool).mean(), "mean_excess": auto.excess.mean(),
            "mean_target_update_fraction": auto.target_update_fraction.mean(), "mean_target_update_count": auto.target_update_count.mean(),
            **{"mean_" + field: auto[field].mean() for field in ACCOUNTS}}
        for field, value in values.items(): close(getattr(record, field), value, "summary/" + field)
    for stem in ("paired_vs_default", "policy_contrasts"):
        pairs = pd.read_csv(OUT / (stem + "_paths.csv"))
        check(len(pairs) == 4800 and not pairs.duplicated(["cost_label", "context", "scenario_id", "policy", "comparison_cost_label", "comparison_policy"]).any(), stem + "/all_original_pairs")
        for record in pairs.itertuples(index=False):
            a = rl.loc[(record.cost_label, record.context, record.scenario_id, record.policy)]
            b = rl.loc[(record.comparison_cost_label, record.context, record.scenario_id, record.comparison_policy)]
            common = bool(a.assigned and b.assigned)
            check(record.common_auto == common and record.assigned == a.assigned and record.comparison_assigned == b.assigned, stem + "/paired_admission")
            for field in DELTAS:
                if common: close(getattr(record, "delta_" + field), float(a[field]) - float(b[field]), stem + "/" + field)
                else: check(pd.isna(getattr(record, "delta_" + field)), stem + "/unserved_pair_null")
        for record in pd.read_csv(OUT / (stem + "_summary.csv")).itertuples(index=False):
            group = pairs.loc[pairs.scope.eq(record.scope) & pairs.cost_label.eq(record.cost_label) & pairs.policy.eq(record.policy)
                & pairs.comparison_cost_label.eq(record.comparison_cost_label) & pairs.comparison_policy.eq(record.comparison_policy)]
            auto = group.loc[group.common_auto]
            check(record.original_pairs == 120 and record.common_auto_pairs == len(auto) and record.assigned_paths == int(group.assigned.sum())
                  and record.comparison_assigned_paths == int(group.comparison_assigned.sum()), stem + "/summary_denominators")
            for field in DELTAS:
                if len(auto): close(getattr(record, "mean_delta_" + field), auto["delta_" + field].mean(), stem + "/summary_" + field)
                else: check(pd.isna(getattr(record, "mean_delta_" + field)), stem + "/summary_null")
    changes = pd.read_csv(OUT / "decision_changes_by_path.csv"); cl = changes.set_index(rkeys).sort_index()
    check(len(changes) == 4800 and not changes.duplicated(rkeys).any(), "all_change_original_path_rows")
    base_gates = gates.loc[gates.cost_label.eq("base")].set_index(gkeys[1:]).sort_index()
    for key, group in gates.groupby(rkeys, sort=False):
        label, context, sid, policy = key; group = group.set_index("year").sort_index()
        base = base_gates.loc[(context, sid, policy)].loc[group.index]; result = cl.loc[key]
        both = (group.index > 1) & ~group.ever_breached & ~base.ever_breached
        check(result.all_decisions == H and result.both_prebreach_postinitial_decisions == int(both.sum()), "change_decision_denominators")
        for field in ("action", "actual_trade", "pending_at_decision", "target_after", "target_updated"):
            delta = group[field] != base[field]
            check(result[field + "_changed_count"] == int(delta.sum()), "whole_trajectory_change_count")
            close(result[field + "_changed_fraction"], delta.mean(), "whole_trajectory_change/fraction")
        check(result.both_prebreach_action_changed_count == int(((group.action != base.action) & both).sum()), "both_prebreach_action_change")
    for record in pd.read_csv(OUT / "decision_changes_summary.csv").itertuples(index=False):
        group = changes.loc[changes.cost_label.eq(record.cost_label) & changes.scope.eq(record.scope) & changes.policy.eq(record.policy)]; auto = group.loc[group.assigned]
        check(record.original_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "change_summary_coverage")
        if not len(auto): check(pd.isna(record.all_decisions), "change_summary_Review_null"); continue
        n, both = int(auto.all_decisions.sum()), int(auto.both_prebreach_postinitial_decisions.sum())
        check(record.all_decisions == n and record.both_prebreach_postinitial_decisions == both, "change_summary_decisions")
        for field in ("action", "actual_trade", "pending_at_decision", "target_after", "target_updated"):
            k = int(auto[field + "_changed_count"].sum()); check(getattr(record, field + "_changed_count") == k, "change_summary_counts")
            close(getattr(record, field + "_changed_fraction"), k / n, "change_summary/fraction")
        if both: close(record.both_prebreach_action_changed_fraction, auto.both_prebreach_action_changed_count.sum() / both, "change_summary/both_fraction")
        else: check(pd.isna(record.both_prebreach_action_changed_fraction), "change_summary_zero_both_null")

    # Base-cost parent columns and every recorded parent array, except the
    # explicitly preserved historical price-catalog metadata.
    base_receipts = []
    parent_specs = ((BOUNDED, POLICIES[0], POLICIES[0]), (BOUNDED, POLICIES[1], POLICIES[1]),
                    (PRICE, "risk_priced_learner", POLICIES[0]), (PRICE, POLICIES[2], POLICIES[2]))
    with np.load(OUT / "actual_traces.npz", allow_pickle=False) as archive:
        for directory, original_policy, policy in parent_specs:
            expected = pd.read_csv(directory / "client_paths.csv")
            expected = expected.loc[expected.client.eq("C") & expected.policy.eq(original_policy)].copy(); expected["policy"] = policy
            expected = expected.set_index(["context", "scenario_id"]).sort_index()
            current = rows.loc[rows.cost_label.eq("base") & rows.policy.eq(policy)].set_index(["context", "scenario_id"]).sort_index()
            check(current.index.equals(expected.index), "base_CSV_parent_identity")
            for field in expected:
                check(np.array_equal(current[field].isna(), expected[field].isna()), "base_CSV_parent_nullmask")
                keep = expected[field].notna()
                if pd.api.types.is_numeric_dtype(expected[field].dtype): close(current.loc[keep, field].to_numpy(), expected.loc[keep, field].to_numpy(), "base_CSV/" + field)
                else: check(np.array_equal(current.loc[keep, field].to_numpy(), expected.loc[keep, field].to_numpy()), "base_CSV_text")
            arrays_checked = 0
            with np.load(directory / "actual_traces.npz", allow_pickle=False) as parent_archive:
                for context in plan:
                    prefix = f"{context}__C__{original_policy}__"
                    for key in parent_archive.files:
                        if not key.startswith(prefix): continue
                        field = key[len(prefix):]
                        if field == "annual_prices": continue
                        newkey = f"base__{context}__C__{policy}__{field}"
                        check(newkey in archive.files, "base_parent_array_exists")
                        close(archive[newkey], parent_archive[key], "base_parent_array/" + field)
                        arrays_checked += 1
            base_receipts.append({"parent": directory.name, "source_policy": original_policy, "cost_policy": policy,
                                  "CSV_rows": len(expected), "CSV_fields": len(expected.columns), "parent_arrays_checked": arrays_checked,
                                  "excluded_metadata": "annual_prices only; actual single-price metadata independently checked"})
    for kind in ("sources", "input_hashes"):
        for name, expected in manifest[kind].items(): check(sha(ROOT / name) == expected, kind + "/" + name)
    for name, expected in manifest["outputs_sha256"].items(): check(sha(OUT / name) == expected, "original_output/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_hashes"] == manifest["input_hashes_at_end"], "original_start_end_hashes")
    after = snapshot(); check(before == after, "independent_source_artifact_stability")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not q_disagreements else "numerical_action_review_needed",
        "checks_passed": checks, "actual_arm_paths_replayed": arm_paths, "actual_decisions_replayed": periods, "maximum_errors": errors,
        "Q_subset_results": q_results, "Q_subset_audited": sum(item["status"] == "audited" for item in q_results),
        "Q_subset_pending_skipped": sum(item["status"] == "pending_skip_no_replacement" for item in q_results), "Q_subset_action_disagreements": q_disagreements,
        "base_reproduction_independent_receipts": base_receipts, "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
        "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(), "final_scenarios_read": False,
        "independence": "Own100-step scalar fee/state reconstruction, independently carried target identity and scalar fan/current-action comparison; no reviewed execution/runner/selection implementation imported",
        "limitations": ["All actual transitions and published summaries/pairs/changes are independently replayed/reaggregated; future Q values cover only the fixed prospective subset",
            "Controller parameters and admission are imposed from default-cost parents; no stress-cost empirical chance feasibility or optimal recalibration is claimed",
            "Hypothetical known cost enters both actual and forecasts; this is not cost estimation-error robustness",
            "Whole-trajectory disagreements involve changed states/history, not same-state marginal fee effects",
            "Repeated paths, overlapping folds, duplicate chosen learners and rates are not independent market evidence",
            "No fee winner, target/price/gamma/admission retuning, probability certificate or final-set access"], "original_manifests_modified": False}
    destination = OUT / "independent_audit.json"; destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_passed", "actual_arm_paths_replayed", "actual_decisions_replayed", "Q_subset_audited", "Q_subset_pending_skipped", "Q_subset_action_disagreements", "maximum_errors")}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__": main()
