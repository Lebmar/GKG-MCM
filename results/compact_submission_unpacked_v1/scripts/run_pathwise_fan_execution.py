"""Separate pointwise calendar-fan pilot; never select an OOF winner.

The fixed-support induction is verified on every context's training paths.
Held-out paths are evaluated separately and are outside that property.
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
import run_fan_rollout_execution as frozen_runner
from safeinvest.data import load_paths
from experimental_execution.review_pending import simulate_paired

PROTOCOL = ROOT / "docs/pathwise_fan_execution_protocol.json"
OUT = ROOT / "results/model_review/pathwise_fan_execution_v1"
PENDING = ROOT / "results/model_review/review_pending_execution_v1"
STRICT = frozen_runner.STRICT
DATA = frozen_runner.DATA
POLICIES = ("annual_then_pending", "pathwise_hold_rollout")
TRACE_FIELDS = ("holdings", "weights", "posttrade_holdings", "nominal_wealth", "real_wealth", "real_peak",
                "current_drawdown", "running_mdd", "ever_breached", "cpi", "fees", "moved", "actions",
                "trades", "target_execution", "pending_at_decision", "first_crossing", "qg_hold", "qg_restore",
                "qbad_hold_count", "qbad_restore_count", "forecast_n", "gate_reason", "min_log_advantage_hold",
                "growth_violation_count", "breach_violation_count", "union_violation_count", "delete_one_flip_count")
NUMERICAL_PROPERTY_GUARD = 1e-10
sha, write_json = frozen_runner.sha, frozen_runner.write_json


def source_manifest():
    names = ["src/safeinvest/core.py", "src/safeinvest/data.py", "research/safeinvest_research/metrics.py",
             "research/experimental_execution/review_pending.py", "research/experimental_execution/fan_rollout.py",
             "research/experimental_execution/pathwise_fan_rollout.py", "scripts/run_fan_rollout_execution.py",
             "scripts/run_pathwise_fan_execution.py", "docs/pathwise_fan_execution_protocol.json"]
    return {name: sha(ROOT / name) for name in names}


def verify_inputs(protocol):
    actual = {key: sha(ROOT / key) for key in protocol["frozen_input_sha256"]}
    if actual != protocol["frozen_input_sha256"]: raise RuntimeError("Prospective pathwise dependency snapshot changed")
    return actual


def verify_training_property(reference, proposed, *, context, client, ids, horizon, tolerance, item):
    """Empirical support check, never a pooled statistical confidence claim."""
    base_log = np.log(reference["real_wealth"][:, -1] / reference["real_wealth"][:, 0])
    new_log = np.log(proposed["real_wealth"][:, -1] / proposed["real_wealth"][:, 0])
    growth_delta = new_log - base_log
    base_bad, new_bad = reference["ever_breached"][:, -1], proposed["ever_breached"][:, -1]
    risk_violation = new_bad & ~base_bad
    if np.any(growth_delta < -NUMERICAL_PROPERTY_GUARD) or np.any(risk_violation):
        raise RuntimeError("Fixed training-support pathwise induction property failed; investigate, do not hide")
    result = []
    for i, sid in enumerate(ids):
        result.append({"context": context, "client": client, "scenario_id": sid, "horizon": horizon,
                       "loss_tolerance": tolerance, "assigned": True, "assignment": item["name"],
                       "candidate_local_index": item["candidate_local_index"],
                       "candidate_original232_index": item["candidate_original232_index"],
                       "reference_total_log_growth": float(base_log[i]), "pathwise_total_log_growth": float(new_log[i]),
                       "total_log_growth_delta": float(growth_delta[i]),
                       "reference_breach": bool(base_bad[i]), "pathwise_breach": bool(new_bad[i]),
                       "computed_growth_delta_strictly_negative": bool(growth_delta[i] < 0),
                       "growth_within_numerical_audit_guard": bool(growth_delta[i] >= -NUMERICAL_PROPERTY_GUARD),
                       "breach_order_preserved": not bool(risk_violation[i])})
    summary = {"context": context, "client": client, "training_support_paths": len(ids), "assigned_paths": len(ids),
               "review_paths": 0, "minimum_total_log_growth_delta": float(growth_delta.min()),
               "mean_total_log_growth_delta": float(growth_delta.mean()),
               "strictly_negative_computed_growth_delta_count": int((growth_delta < 0).sum()),
               "negative_beyond_audit_guard_count": int((growth_delta < -NUMERICAL_PROPERTY_GUARD).sum()),
               "breach_order_violations": int(risk_violation.sum()), "reference_breach_count": int(base_bad.sum()),
               "pathwise_breach_count": int(new_bad.sum()), "gate_growth_allowance": 0,
               "numerical_audit_guard_total_log_units": NUMERICAL_PROPERTY_GUARD}
    return result, summary


def make_rows(arm, common, ids, policy):
    values = frozen_runner.path_values(arm, common["horizon"])
    rows = []
    for i, sid in enumerate(ids):
        first = int(arm["first_crossing"][i]); loss = common["loss_tolerance"]
        rows.append({**common, "scenario_id": sid, "policy": policy,
                     **{key: float(values[key][i]) for key in (*frozen_runner.METRICS, *frozen_runner.ACCOUNTS)},
                     "breach": bool(arm["ever_breached"][i, -1]), "excess": float(max(values["mdd"][i] - loss, 0)),
                     "first_crossing": first if first >= 0 else None})
    return rows


def add_trace(archive, context, client, policy, ids, arm):
    prefix = f"{context}__{client}__{policy}"
    archive[prefix + "__scenario_ids"] = np.asarray(ids)
    for key in TRACE_FIELDS:
        if key in arm: archive[prefix + "__" + key] = arm[key]


def decision_rows(arm, common, ids):
    rows = []
    for i, sid in enumerate(ids):
        for t in range(common["horizon"]):
            record = {**common, "scenario_id": sid, "policy": "pathwise_hold_rollout", "year": t + 1,
                      "action": str(arm["actions"][i, t]), "gate_reason": str(arm["gate_reason"][i, t]),
                      "cpi": float(arm["cpi"][i, t]), "real_peak": float(arm["real_peak"][i, t]),
                      "running_mdd": float(arm["running_mdd"][i, t]), "ever_breached": bool(arm["ever_breached"][i, t]),
                      "fee": float(arm["fees"][i, t]), "moved": float(arm["moved"][i, t]), "actual_trade": bool(arm["trades"][i, t])}
            for asset, value in zip(("equity", "bond", "bill"), arm["holdings"][i, t]): record["holding_" + asset] = float(value)
            for key in ("qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "forecast_n",
                        "min_log_advantage_hold", "growth_violation_count", "breach_violation_count",
                        "union_violation_count", "delete_one_flip_count"):
                value = arm[key][i, t]
                score = key in ("qg_hold", "qg_restore", "min_log_advantage_hold")
                record[key] = float(value) if np.isfinite(value) and (score or value >= 0) else None
            rows.append(record)
    return rows


def check_coverage(frame, training):
    if len(frame) != 2400 or frame.duplicated(["context", "client", "scenario_id", "policy"]).any():
        raise RuntimeError("Expected2400 unique full/OOF client rows")
    for (scope, client, policy), group in frame.groupby(["scope", "client", "policy"]):
        if len(group) != 120 or set(group.scenario_id) != set(range(1, 121)):
            raise RuntimeError("Original120 denominator changed")
    if not frame.loc[~frame.assigned, [*frozen_runner.METRICS, *frozen_runner.ACCOUNTS, "breach", "excess", "first_crossing"]].isna().all().all():
        raise RuntimeError("Review result imputed")
    if len(training) != 3000 or training.duplicated(["context", "client", "scenario_id"]).any():
        raise RuntimeError("Expected3000 context-labeled training profile units, with overlap disclosed")


def paired_summary(frame):
    keys = ["context", "client", "scenario_id"]
    joined = frame.loc[frame.policy == "pathwise_hold_rollout"].merge(
        frame.loc[frame.policy == "annual_then_pending"], on=keys, suffixes=("", "_reference"), validate="one_to_one")
    result = []
    for (scope, client), group in joined.groupby(["scope", "client"], sort=False):
        auto = group.loc[group.assigned & group.assigned_reference]
        row = {"scope": scope, "client": client, "policy": "pathwise_hold_rollout", "reference": "annual_then_pending",
               "total_pairs": len(group), "assigned_pairs": len(auto), "review_pairs": len(group) - len(auto)}
        for field in (*frozen_runner.METRICS, *frozen_runner.ACCOUNTS, "excess"):
            row[f"mean_delta_{field}"] = float((auto[field] - auto[field + "_reference"]).mean()) if len(auto) else None
        row["breach_count_delta"] = int(auto.breach.astype(bool).sum() - auto.breach_reference.astype(bool).sum()) if len(auto) else None
        result.append(row)
    return result


def main():
    from experimental_execution.pathwise_fan_rollout import simulate_policy
    started = time.perf_counter(); protocol = json.loads(PROTOCOL.read_text())
    sources, inputs = source_manifest(), verify_inputs(protocol)
    for extra in (DATA, PENDING / "manifest.json", PENDING / "client_path_metrics.csv", PENDING / "independent_audit.json"):
        inputs[str(extra.relative_to(ROOT))] = sha(extra)
    if json.loads((PENDING / "independent_audit.json").read_text())["status"] != "passed":
        raise RuntimeError("Pending independent audit must pass first")
    contexts = json.loads((STRICT / "nested_context_refits.json").read_text()); frozen_runner.validate_contexts(contexts)
    mapping = json.loads((STRICT / "candidate_mapping.json").read_text())
    clients = json.loads((ROOT / "docs/positive_asset_specification_protocol.json").read_text())["clients"]
    originals = pd.concat([pd.read_csv(STRICT / "nested_full_paths.csv"), pd.read_csv(STRICT / "nested_outer_oof_paths.csv")], ignore_index=True)
    pending_saved = pd.read_csv(PENDING / "client_path_metrics.csv")
    returns, inflation, ids = load_paths(DATA, range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for array in (ids, returns, inflation): digest.update(np.ascontiguousarray(array).tobytes())
    records, training_records, property_summaries, gates, meta, checks = [], [], [], [], {}, {}
    traces, train_traces = {}, {}
    for context, fit in contexts.items():
        scope = "full_development" if context == "full_development" else "outer_oof"
        eval_ids = fit["training_ids"] if scope == "full_development" else fit["outer_test_ids"]
        train_ids = fit["training_ids"]; eval_ix, train_ix = np.asarray(eval_ids) - 1, np.asarray(train_ids) - 1
        meta[context] = {"training_ids": train_ids, "evaluation_ids": eval_ids, "products": fit["raw_fit"]["menu"],
                         "effective_assignments": fit["effective_assignments"]}
        for client in clients:
            name, h, loss = client["name"], client["horizon"], client["loss_tolerance"]
            assignment = fit["effective_assignments"][name]
            item = next((p for p in fit["raw_fit"]["menu"] if p["name"] == assignment), None)
            common = {"scope": scope, "context": context, "outer_fold": fit["context_id"] or None, "client": name,
                      "horizon": h, "loss_tolerance": loss, "assignment": assignment, "assigned": item is not None,
                      "candidate_local_index": item["candidate_local_index"] if item else None,
                      "candidate_original232_index": item["candidate_original232_index"] if item else None}
            if item is None:
                if assignment != "Review": raise RuntimeError("Invalid frozen assignment")
                frozen_runner.verify_review(originals, context, name, eval_ids)
                for policy in POLICIES:
                    records.extend({**common, "scenario_id": sid, "policy": policy,
                        **dict.fromkeys([*frozen_runner.METRICS, *frozen_runner.ACCOUNTS, "breach", "excess", "first_crossing"])} for sid in eval_ids)
                training_records.extend({"context": context, "client": name, "scenario_id": sid, "horizon": h,
                    "loss_tolerance": loss, "assigned": False, "assignment": "Review",
                    **dict.fromkeys(["reference_total_log_growth", "pathwise_total_log_growth", "total_log_growth_delta", "reference_breach", "pathwise_breach"])} for sid in train_ids)
                property_summaries.append({"context": context, "client": name, "training_support_paths": len(train_ids),
                    "assigned_paths": 0, "review_paths": len(train_ids), "minimum_total_log_growth_delta": None,
                    "mean_total_log_growth_delta": None, "strictly_negative_computed_growth_delta_count": None,
                    "negative_beyond_audit_guard_count": None, "breach_order_violations": None,
                    "reference_breach_count": None, "pathwise_breach_count": None, "gate_growth_allowance": 0,
                    "numerical_audit_guard_total_log_units": NUMERICAL_PROPERTY_GUARD})
                continue
            target = frozen_runner.validate_product(item, mapping)
            pair = simulate_paired(returns[eval_ix], inflation[eval_ix], target, loss, horizon=h)
            annual, reference = frozen_runner.squeeze_pending(pair["annual"]), frozen_runner.squeeze_pending(pair["pending_hold"])
            checks[f"{context}__{name}"] = {"annual_original_errors": frozen_runner.verify_original(annual, originals, context, name, eval_ids, item, h),
                                           "annual_pending_coupling": pair["coupling"]}
            frozen_runner.verify_pending_reference(reference, pending_saved, context, name, eval_ids, h)
            proposed = simulate_policy(returns[eval_ix, :h], inflation[eval_ix, :h], returns[train_ix, :h], inflation[train_ix, :h], target, L=loss)
            for policy, arm in (("annual_then_pending", reference), ("pathwise_hold_rollout", proposed)):
                records.extend(make_rows(arm, common, eval_ids, policy)); add_trace(traces, context, name, policy, eval_ids, arm)
            gates.extend(decision_rows(proposed, common, eval_ids))
            if scope == "full_development":
                train_reference, train_proposed = reference, proposed
            else:
                train_reference = frozen_runner.squeeze_pending(simulate_paired(returns[train_ix], inflation[train_ix], target, loss, horizon=h)["pending_hold"])
                train_proposed = simulate_policy(returns[train_ix, :h], inflation[train_ix, :h], returns[train_ix, :h], inflation[train_ix, :h], target, L=loss)
            invariant, summary = verify_training_property(train_reference, train_proposed, context=context, client=name,
                ids=train_ids, horizon=h, tolerance=loss, item=item)
            training_records.extend(invariant); property_summaries.append(summary)
            for policy, arm in (("annual_then_pending", train_reference), ("pathwise_hold_rollout", train_proposed)):
                add_trace(train_traces, context, name, policy, train_ids, arm)
        print(json.dumps({"context": context, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    frame, training = pd.DataFrame(records), pd.DataFrame(training_records); check_coverage(frame, training)
    end_sources, end_inputs = source_manifest(), {key: sha(ROOT / key) for key in inputs}
    if sources != end_sources or inputs != end_inputs: raise RuntimeError("Sources/inputs changed during the pilot")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "client_paths.csv", index=False)
    pd.DataFrame(frozen_runner.summarize(frame)).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(paired_summary(frame)).to_csv(OUT / "paired_summary.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "decision_gates.csv", index=False)
    training.to_csv(OUT / "training_support_paths.csv", index=False)
    pd.DataFrame(property_summaries).to_csv(OUT / "training_support_summary.csv", index=False)
    np.savez_compressed(OUT / "actual_traces.npz", **traces)
    np.savez_compressed(OUT / "training_traces.npz", **train_traces)
    write_json(OUT / "contexts.json", meta); write_json(OUT / "accounting_checks.json", checks)
    outputs = {p.name: sha(p) for p in OUT.iterdir() if p.name not in ("manifest.json", "independent_audit.json")}
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
        "sources": sources, "sources_at_end": end_sources, "input_hashes": inputs, "input_hashes_at_end": end_inputs,
        "outputs_sha256": outputs, "development_array_sha256": digest.hexdigest(), "scenario_ids_loaded": ids.tolist(),
        "final_scenarios_loaded": False, "client_path_rows": len(frame), "training_support_rows": len(training),
        "training_support_assigned_rows": int(training.assigned.sum()), "decision_rows": len(gates),
        "gate_growth_allowance": 0, "numerical_property_check_guard": NUMERICAL_PROPERTY_GUARD,
        "scope": "Separate fixed-fan pointwise pilot, no refit or OOF policy selection. In-sample support induction is not a risk certificate or OOF guarantee;600 overlapping training path identities per client are not600 independent scenarios."})
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "client_paths": len(frame), "training_support_rows": len(training)}))


if __name__ == "__main__": main()
