"""Prospective known-cost sensitivity, with every frozen product held fixed.

Rates are declared hypothetical values, never selected or estimated here.
Both actual and forecast execution know the same rate.  Review rows remain
null and training support checks are distinct from held-out evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
import run_pathwise_fan_execution as frozen
from experimental_execution.pathwise_fan_rollout import simulate_policy
from experimental_execution.review_pending import simulate_paired
from safeinvest.data import load_paths

PROTOCOL = ROOT / "docs/pathwise_cost_stress_protocol.json"
OUT = ROOT / "results/model_review/pathwise_cost_stress_v1"
BASE = ROOT / "results/model_review/pathwise_fan_execution_v1"
RATES = (("zero", 0.), ("half", .00125), ("base", .0025), ("double", .005), ("quadruple", .01))
POLICIES = ("annual_then_pending", "pathwise_hold_rollout")
base = frozen.frozen_runner
sha, write_json = frozen.sha, frozen.write_json


def validate_protocol(protocol, *, require_ready=True):
    if any(isinstance(item["cost_rate"], bool) for item in protocol["rates"]):
        raise ValueError("A rate must be a numeric fraction, not boolean")
    pairs = [(item["label"], item["cost_rate"]) for item in protocol["rates"]]
    if (pairs != list(RATES) or protocol["data"]["allowed_scenario_ids"] != [1, 120]
            or protocol["data"]["forbidden_scenario_ids"] != [121, 180]
            or protocol["frozen_design"]["policies"] != list(POLICIES)
            or protocol["training_support_check"]["audit_guard_total_log_units"] != frozen.NUMERICAL_PROPERTY_GUARD):
        raise ValueError("Cost rates, policies, numerical audit guard or development scope changed")
    if require_ready and protocol.get("execution_ready") is not True:
        raise RuntimeError("Freeze implementation/test source before any market replay")


def verify_inputs(protocol):
    inputs = {key: sha(ROOT / key) for key in protocol["frozen_input_sha256"]}
    if inputs != protocol["frozen_input_sha256"]:
        raise RuntimeError("Prospective frozen source/product/base artifact changed")
    code = {key: sha(ROOT / key) for key in protocol["execution_source_sha256"]}
    required = {"scripts/run_pathwise_cost_stress.py", "tests/test_pathwise_cost_stress.py"}
    if set(code) != required or code != protocol["execution_source_sha256"]:
        raise RuntimeError("Cost experiment source/test differs from prospective freeze")
    inputs.update(code)
    inputs[str(frozen.DATA.relative_to(ROOT))] = sha(frozen.DATA)
    return inputs


def source_manifest():
    names = [*frozen.source_manifest(), "scripts/run_pathwise_cost_stress.py",
             "tests/test_pathwise_cost_stress.py", "docs/pathwise_cost_stress_protocol.json"]
    return {name: sha(ROOT / name) for name in dict.fromkeys(names)}


def compare_frames(actual, expected, keys):
    """Check exact identities/null masks and numeric serialization tolerance."""
    if (len(actual) != len(expected) or actual.duplicated(keys).any() or expected.duplicated(keys).any()
            or not set(expected).issubset(actual)):
        raise RuntimeError("Base reproduction row/column identity changed")
    left, right = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise RuntimeError("Base reproduction scenario/client/policy keys changed")
    maximum, numeric_columns = 0., 0
    for key in right:
        a, e = left[key], right[key]
        if not np.array_equal(a.isna().to_numpy(), e.isna().to_numpy()):
            raise RuntimeError(f"Base reproduction null mask changed:{key}")
        keep = e.notna()
        if not keep.any():
            continue
        if pd.api.types.is_bool_dtype(e.dtype):
            np.testing.assert_array_equal(a[keep].to_numpy(), e[keep].to_numpy())
        elif pd.api.types.is_numeric_dtype(e.dtype):
            av, ev = a[keep].to_numpy(float), e[keep].to_numpy(float)
            np.testing.assert_allclose(av, ev, atol=2e-10, rtol=5e-13)
            maximum = max(maximum, float(np.max(np.abs(av - ev), initial=0)))
            numeric_columns += 1
        else:
            np.testing.assert_array_equal(a[keep].to_numpy(), e[keep].to_numpy())
    return {"rows": len(actual), "numeric_columns": numeric_columns, "max_abs_error": maximum,
            "identities_and_null_masks_match": True}


def compare_trace(actual, path):
    """The base rate reproduces every available frozen trace array."""
    maximum = 0.
    with np.load(path, allow_pickle=False) as expected:
        if set(actual) != set(expected.files):
            raise RuntimeError("Base trace field identities changed")
        for key in expected.files:
            av, ev = actual[key], expected[key]
            if av.shape != ev.shape:
                raise RuntimeError(f"Base trace shape changed:{key}")
            if ev.dtype.kind == "f":
                np.testing.assert_allclose(av, ev, atol=2e-10, rtol=5e-13, equal_nan=True)
                finite = np.isfinite(ev)
                maximum = max(maximum, float(np.max(np.abs(av[finite] - ev[finite]), initial=0)))
            else:
                np.testing.assert_array_equal(av, ev)
    return {"array_fields": len(actual), "max_abs_error": maximum, "matches": True}


def decision_rows(arm, common, ids, policy):
    rows = []
    for i, sid in enumerate(ids):
        for t in range(common["horizon"]):
            row = {**common, "scenario_id": sid, "policy": policy, "year": t + 1,
                   "action": str(arm["actions"][i, t]), "actual_trade": bool(arm["trades"][i, t]),
                   "target_execution": bool(arm["target_execution"][i, t]),
                   "pending_at_decision": bool(arm["pending_at_decision"][i, t]),
                   "fee": float(arm["fees"][i, t]), "moved": float(arm["moved"][i, t]),
                   "cpi": float(arm["cpi"][i, t]), "real_peak": float(arm["real_peak"][i, t]),
                   "running_mdd": float(arm["running_mdd"][i, t]),
                   "ever_breached": bool(arm["ever_breached"][i, t])}
            for asset, value in zip(("equity", "bond", "bill"), arm["holdings"][i, t]):
                row["holding_" + asset] = float(value)
            row["gate_reason"] = str(arm["gate_reason"][i, t]) if "gate_reason" in arm else (
                "mandatory_initial_restore" if t == 0 else "already_breached_pending_hold" if arm["pending_at_decision"][i, t] else "annual_reference")
            for key in ("qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "forecast_n",
                        "min_log_advantage_hold", "growth_violation_count", "breach_violation_count",
                        "union_violation_count", "delete_one_flip_count"):
                value = arm[key][i, t] if key in arm else np.nan
                score = key in ("qg_hold", "qg_restore", "min_log_advantage_hold")
                row[key] = float(value) if np.isfinite(value) and (score or value >= 0) else None
            rows.append(row)
    return rows


def decision_changes(gates, paths):
    """Whole-trajectory action disagreements versus base, never same-state effects."""
    keys = ["context", "client", "scenario_id", "policy", "year"]
    baseline = gates.loc[gates.cost_label == "base", keys + ["action", "actual_trade", "pending_at_decision"]]
    if baseline.duplicated(keys).any():
        raise ValueError("Duplicated base decision identity")
    items = []
    for label, rate in RATES:
        subset = gates.loc[gates.cost_label == label]
        if len(subset) != len(baseline):
            raise ValueError("Every cost must retain the same complete decision identities")
        paired = subset.merge(baseline, on=keys, suffixes=("", "_base"), validate="one_to_one")
        if len(paired) != len(subset):
            raise ValueError("Cost/base decision identities do not pair")
        for (context, client, sid, policy), group in paired.groupby(keys[:-1], sort=False):
            both = (group.year > 1) & ~group.pending_at_decision & ~group.pending_at_decision_base
            changed = group.action != group.action_base
            items.append({"cost_label": label, "cost_rate": rate, "context": context, "client": client,
                "scenario_id": sid, "policy": policy, "all_decisions": len(group),
                "action_changed_count": int(changed.sum()), "action_changed_fraction": float(changed.mean()),
                "trade_flag_changed_count": int((group.actual_trade != group.actual_trade_base).sum()),
                "trade_flag_changed_fraction": float((group.actual_trade != group.actual_trade_base).mean()),
                "pending_flag_changed_count": int((group.pending_at_decision != group.pending_at_decision_base).sum()),
                "pending_flag_changed_fraction": float((group.pending_at_decision != group.pending_at_decision_base).mean()),
                "both_active_postinitial_decisions": int(both.sum()),
                "both_active_action_changed_count": int((changed & both).sum())})
    changes = pd.DataFrame(items)
    idkeys = ["cost_label", "cost_rate", "context", "client", "scenario_id", "policy"]
    identities = paths[[*idkeys, "scope", "assigned", "horizon"]]
    full = identities.merge(changes, on=idkeys, how="left", validate="one_to_one")
    if (full.loc[full.assigned, "all_decisions"].isna().any()
            or not full.loc[full.assigned, "all_decisions"].eq(full.loc[full.assigned, "horizon"]).all()
            or full.loc[~full.assigned, "all_decisions"].notna().any()):
        raise ValueError("Incomplete original-H decision trace or imputed Review decision")
    result = []
    for (label, rate, scope, client, policy), group in full.groupby(["cost_label", "cost_rate", "scope", "client", "policy"], sort=False):
        auto = group.loc[group.assigned]
        row = {"cost_label": label, "cost_rate": rate, "scope": scope, "client": client, "policy": policy,
               "original_paths": len(group), "assigned_paths": len(auto), "review_paths": len(group) - len(auto),
               "all_decisions": int(auto.all_decisions.sum()) if len(auto) else None,
               "both_active_postinitial_decisions": int(auto.both_active_postinitial_decisions.sum()) if len(auto) else None}
        for kind in ("action", "trade_flag", "pending_flag"):
            count = float(auto[kind + "_changed_count"].sum()) if len(auto) else None
            row[kind + "_changed_count"] = int(count) if count is not None else None
            row[kind + "_changed_fraction"] = count / row["all_decisions"] if row["all_decisions"] else None
        n_both = row["both_active_postinitial_decisions"]
        row["both_active_action_changed_fraction"] = float(auto.both_active_action_changed_count.sum()) / n_both if n_both else None
        result.append(row)
    return full, pd.DataFrame(result)


def add_traces(archive, label, context, client, policy, ids, arm):
    temporary = {}
    frozen.add_trace(temporary, context, client, policy, ids, arm)
    archive.update({label + "__" + key: value for key, value in temporary.items()})


def review_training(common, ids):
    return [{"context": common["context"], "client": common["client"], "scenario_id": sid,
        "horizon": common["horizon"], "loss_tolerance": common["loss_tolerance"], "assigned": False,
        "assignment": "Review", "cost_label": common["cost_label"], "cost_rate": common["cost_rate"],
        **dict.fromkeys(["reference_total_log_growth", "pathwise_total_log_growth", "total_log_growth_delta", "reference_breach", "pathwise_breach"])} for sid in ids]


def main():
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text()); validate_protocol(protocol)
    inputs, sources = verify_inputs(protocol), source_manifest()
    base_manifest = json.loads((BASE / "manifest.json").read_text())
    if (base_manifest["sources"] != frozen.source_manifest() or base_manifest["sources"] != base_manifest["sources_at_end"]
            or base_manifest["input_hashes"] != base_manifest["input_hashes_at_end"]
            or base_manifest["scenario_ids_loaded"] != list(range(1, 121)) or base_manifest["final_scenarios_loaded"] is not False):
        raise RuntimeError("Completed base pilot source/data scope is stale")
    for filename, digest in base_manifest["outputs_sha256"].items():
        if sha(BASE / filename) != digest:
            raise RuntimeError("Completed base pilot output digest changed")
    contexts = json.loads((frozen.STRICT / "nested_context_refits.json").read_text()); base.validate_contexts(contexts)
    mapping = json.loads((frozen.STRICT / "candidate_mapping.json").read_text())
    clients = json.loads((ROOT / "docs/positive_asset_specification_protocol.json").read_text())["clients"]
    returns, inflation, ids = load_paths(frozen.DATA, range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)):
        raise RuntimeError("Guarded development identities changed")
    data_digest = hashlib.sha256()
    for array in (ids, returns, inflation): data_digest.update(np.ascontiguousarray(array).tobytes())
    if data_digest.hexdigest() != base_manifest["development_array_sha256"]:
        raise RuntimeError("Guarded development arrays differ from frozen base replay")
    records, training_records, support_summaries, gates, checks = [], [], [], [], {}
    actual_traces, training_traces = {}, {}
    for label, rate in RATES:
        for context, fit in contexts.items():
            scope = "full_development" if context == "full_development" else "outer_oof"
            train_ids = fit["training_ids"]
            eval_ids = train_ids if scope == "full_development" else fit["outer_test_ids"]
            tx, ex = np.asarray(train_ids) - 1, np.asarray(eval_ids) - 1
            for client in clients:
                name, h, loss = client["name"], client["horizon"], client["loss_tolerance"]
                assignment = fit["effective_assignments"][name]
                item = next((p for p in fit["raw_fit"]["menu"] if p["name"] == assignment), None)
                common = {"cost_label": label, "cost_rate": rate, "scope": scope, "context": context,
                    "outer_fold": fit["context_id"] or None, "client": name, "horizon": h, "loss_tolerance": loss,
                    "assignment": assignment, "assigned": item is not None,
                    "candidate_local_index": item["candidate_local_index"] if item else None,
                    "candidate_original232_index": item["candidate_original232_index"] if item else None}
                if item is None:
                    if assignment != "Review": raise RuntimeError("Invalid frozen effective assignment")
                    for policy in POLICIES:
                        records.extend({**common, "scenario_id": sid, "policy": policy,
                            **dict.fromkeys([*base.METRICS, *base.ACCOUNTS, "breach", "excess", "first_crossing"])} for sid in eval_ids)
                    training_records.extend(review_training(common, train_ids))
                    support_summaries.append({"cost_label": label, "cost_rate": rate, "context": context, "client": name,
                        "training_support_paths": len(train_ids), "assigned_paths": 0, "review_paths": len(train_ids),
                        **dict.fromkeys(["minimum_total_log_growth_delta", "mean_total_log_growth_delta", "strictly_negative_computed_growth_delta_count", "negative_beyond_audit_guard_count", "breach_order_violations", "reference_breach_count", "pathwise_breach_count"]),
                        "gate_growth_allowance": 0, "numerical_audit_guard_total_log_units": frozen.NUMERICAL_PROPERTY_GUARD})
                    continue
                target = base.validate_product(item, mapping)
                pair = simulate_paired(returns[ex], inflation[ex], target, loss, horizon=h, cost_rate=rate)
                reference = base.squeeze_pending(pair["pending_hold"])
                proposed = simulate_policy(returns[ex, :h], inflation[ex, :h], returns[tx, :h], inflation[tx, :h],
                                           target, L=loss, cost_rate=rate)
                checks[f"{label}__{context}__{name}"] = pair["coupling"]
                for policy, arm in ((POLICIES[0], reference), (POLICIES[1], proposed)):
                    records.extend(frozen.make_rows(arm, common, eval_ids, policy))
                    gates.extend(decision_rows(arm, common, eval_ids, policy))
                    add_traces(actual_traces, label, context, name, policy, eval_ids, arm)
                if scope == "full_development":
                    train_reference, train_proposed = reference, proposed
                else:
                    train_reference = base.squeeze_pending(simulate_paired(returns[tx], inflation[tx], target, loss,
                                                                          horizon=h, cost_rate=rate)["pending_hold"])
                    train_proposed = simulate_policy(returns[tx, :h], inflation[tx, :h], returns[tx, :h], inflation[tx, :h],
                                                     target, L=loss, cost_rate=rate)
                rows, summary = frozen.verify_training_property(train_reference, train_proposed, context=context,
                    client=name, ids=train_ids, horizon=h, tolerance=loss, item=item)
                training_records.extend({**row, "cost_label": label, "cost_rate": rate} for row in rows)
                support_summaries.append({**summary, "cost_label": label, "cost_rate": rate})
                for policy, arm in ((POLICIES[0], train_reference), (POLICIES[1], train_proposed)):
                    add_traces(training_traces, label, context, name, policy, train_ids, arm)
            print(json.dumps({"cost_label": label, "context": context, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    frame, training, gate_frame = pd.DataFrame(records), pd.DataFrame(training_records), pd.DataFrame(gates)
    summaries, paired = [], []
    for label, rate in RATES:
        part, train_part = frame.loc[frame.cost_label == label], training.loc[training.cost_label == label]
        frozen.check_coverage(part, train_part)
        summaries.extend({**row, "cost_label": label, "cost_rate": rate} for row in base.summarize(part))
        paired.extend({**row, "cost_label": label, "cost_rate": rate} for row in frozen.paired_summary(part))
    summary_frame, paired_frame, support_frame = pd.DataFrame(summaries), pd.DataFrame(paired), pd.DataFrame(support_summaries)
    receipts = {}
    for filename, current, keys in (
        ("client_paths.csv", frame, ["context", "client", "scenario_id", "policy"]),
        ("summary.csv", summary_frame, ["scope", "client", "policy"]),
        ("paired_summary.csv", paired_frame, ["scope", "client", "policy"]),
        ("training_support_paths.csv", training, ["context", "client", "scenario_id"]),
        ("training_support_summary.csv", support_frame, ["context", "client"])):
        receipts[filename] = compare_frames(current.loc[current.cost_label == "base"], pd.read_csv(BASE / filename), keys)
    for filename, archive in (("actual_traces.npz", actual_traces), ("training_traces.npz", training_traces)):
        base_arrays = {key.removeprefix("base__"): value for key, value in archive.items() if key.startswith("base__")}
        receipts[filename] = compare_trace(base_arrays, BASE / filename)
    changes, change_summary = decision_changes(gate_frame, frame)
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, table in (("client_paths.csv", frame), ("summary.csv", summary_frame), ("paired_summary.csv", paired_frame),
        ("decision_gates.csv", gate_frame), ("training_support_paths.csv", training),
        ("training_support_summary.csv", support_frame), ("decision_changes_by_path.csv", changes),
        ("decision_changes_summary.csv", change_summary)):
        table.to_csv(OUT / filename, index=False)
    np.savez_compressed(OUT / "actual_traces.npz", **actual_traces)
    np.savez_compressed(OUT / "training_traces.npz", **training_traces)
    write_json(OUT / "contexts.json", contexts)
    write_json(OUT / "accounting_checks.json", checks)
    write_json(OUT / "base_reproduction.json", receipts)
    end_sources, end_inputs = source_manifest(), {key: sha(ROOT / key) for key in inputs}
    if sources != end_sources or inputs != end_inputs:
        raise RuntimeError("Source/input snapshot changed during cost stress")
    outputs = {p.name: sha(p) for p in OUT.iterdir() if p.name not in ("manifest.json", "independent_audit.json")}
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "sources": sources, "sources_at_end": end_sources,
        "input_hashes": inputs, "input_hashes_at_end": end_inputs, "outputs_sha256": outputs,
        "development_array_sha256": data_digest.hexdigest(), "scenario_ids_loaded": ids.tolist(), "final_scenarios_loaded": False,
        "cost_rates": [{"label": label, "cost_rate": rate} for label, rate in RATES], "actual_forecast_costs_identical": True,
        "client_path_rows": len(frame), "training_support_rows": len(training), "decision_rows": len(gate_frame),
        "base_reproduction": receipts, "training_support_overlap": "600 context identities per client/rate; not independent",
        "gate_growth_allowance": 0, "numerical_property_audit_guard": frozen.NUMERICAL_PROPERTY_GUARD,
        "scope": "Hypothetical known-cost sensitivity with frozen products/assignments, no recalibration, no winner or OOF risk certificate"})
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "client_path_rows": len(frame), "training_support_rows": len(training)}))


if __name__ == "__main__":
    main()
