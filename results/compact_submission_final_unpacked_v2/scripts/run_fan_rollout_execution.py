"""Two predeclared local rollout gates on frozen strict172 products.

No product/assignment/risk-price fitting occurs here.  The calendar fan is
training-only, unconditional and recombined each year; no dynamic-programming
optimality or whole-policy risk certificate is claimed.  Review rows remain
financially null.  Original annual and pending paths are verified first.
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from safeinvest.data import load_paths
from safeinvest_research.metrics import NUMERIC_TOL, discrete_cvar, path_metrics
from experimental_execution.review_pending import simulate_paired

PROTOCOL = ROOT / "docs/fan_rollout_execution_protocol.json"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PENDING = ROOT / "results/model_review/review_pending_execution_v1"
OUT = ROOT / "results/model_review/fan_rollout_execution_v1"
DATA = ROOT / "data/SafeInvest_Synthetic_Data.csv"
POLICIES = ("annual_then_pending", "growth_only_rollout", "pareto_hold_rollout")
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction",
            "terminal_equity", "terminal_bond", "terminal_bill")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_manifest():
    names = ["src/safeinvest/core.py", "src/safeinvest/data.py", "research/safeinvest_research/metrics.py",
             "research/experimental_execution/review_pending.py", "research/experimental_execution/fan_rollout.py",
             "scripts/run_fan_rollout_execution.py", "docs/fan_rollout_execution_protocol.json"]
    return {name: sha(ROOT / name) for name in names}


def frozen_inputs(protocol):
    hashes = {name: sha(ROOT / name) for name in protocol["frozen_input_sha256"]}
    if hashes != protocol["frozen_input_sha256"]:
        raise RuntimeError("Frozen strict172 artifact differs from the prospective protocol")
    return hashes


def validate_protocol(protocol):
    if (protocol["data"]["allowed_scenario_ids"] != [1, 120]
            or protocol["data"]["forbidden_scenario_ids"] != [121, 180]
            or protocol["state_and_accounting"]["cost_rate"] != .0025
            or protocol["evaluation"]["policies"] != list(POLICIES)):
        raise ValueError("Declared execution/data scope differs from implementation")


def validate_contexts(contexts):
    if list(contexts) != ["full_development", *[f"outer_fold_{i}" for i in range(1, 6)]]:
        raise ValueError("Require exactly the frozen full and five outer contexts")
    total = set(range(1, 121)); all_test = []
    for name, item in contexts.items():
        train, test = item["training_ids"], item["outer_test_ids"]
        if any(isinstance(x, bool) or not isinstance(x, int) for x in train + test):
            raise ValueError("Scenario identities must be integers")
        if len(train) != len(set(train)) or len(test) != len(set(test)):
            raise ValueError("Duplicate scenario identity")
        if name == "full_development":
            if set(train) != total or test: raise ValueError("Invalid full development context")
        else:
            if len(train) != 96 or len(test) != 24 or set(train) & set(test) or set(train) | set(test) != total:
                raise ValueError("Outer train/test are not complementary96/24 paths")
            all_test.extend(test)
    if len(all_test) != 120 or set(all_test) != total:
        raise ValueError("Outer tests must partition all120 original paths once")


def validate_product(item, mapping):
    local = item["candidate_local_index"]
    if isinstance(local, bool) or not isinstance(local, int) or not 0 <= local < 172:
        raise ValueError("Invalid strict172 local index")
    row = mapping[local]
    weights = np.asarray(item["weights"], float)
    expected = np.array([row["w_equity"], row["w_bond"], row["w_bill"]])
    if (row["candidate_local_index"] != local or item["candidate_index"] != local
            or item["candidate_original232_index"] != row["candidate_original232_index"]
            or not np.array_equal(weights, expected) or np.any(weights < .05 - NUMERIC_TOL)):
        raise ValueError("Strict172 local/global target mapping changed")
    return weights


def path_values(arm, horizon):
    real = np.asarray(arm["real_wealth"])
    if real.ndim != 2 or real.shape[1] != horizon + 1:
        raise ValueError("Require one actual original-H path per scenario")
    values = path_metrics(real)
    weights = np.asarray(arm["weights"])
    values.update({"nominal_fees": arm["fees"].sum(axis=1), "nominal_moved": arm["moved"].sum(axis=1),
                   "trade_fraction": arm["trades"].mean(axis=1),
                   "target_execution_fraction": arm["target_execution"].mean(axis=1),
                   "terminal_equity": weights[:, -1, 0], "terminal_bond": weights[:, -1, 1],
                   "terminal_bill": weights[:, -1, 2]})
    return values


def squeeze_pending(arm):
    """Remove only the explicitly singleton target axis, never scenario/time."""
    result = {}
    for key, value in arm.items():
        if not isinstance(value, np.ndarray): continue
        if key == "cpi": result[key] = value.copy()
        elif key in ("targets", "loss_tolerances"): continue
        elif value.ndim >= 2 and value.shape[1] == 1: result[key] = value[:, 0].copy()
        else: raise ValueError(f"Unexpected paired target axis for{key}")
    return result


def verify_original(annual, saved, context, client, ids, item, horizon):
    rows = saved.loc[(saved.context == context) & (saved.client == client) & (saved.view == "effective")]
    if len(rows) != len(ids) or set(rows.scenario_id) != set(ids) or not rows.assigned.all():
        raise RuntimeError("Saved original assigned denominator changed")
    rows = rows.set_index("scenario_id").loc[ids]
    if not rows.candidate_local_index.eq(item["candidate_local_index"]).all():
        raise RuntimeError("Original annual metrics use a different strict local target")
    values = path_values(annual, horizon)
    errors = {}
    for field in METRICS:
        expected = rows[field].to_numpy(float)
        np.testing.assert_allclose(values[field], expected, atol=2e-10, rtol=5e-13)
        errors[field] = float(np.max(np.abs(values[field] - expected)))
    np.testing.assert_array_equal(annual["ever_breached"][:, -1], rows.breach.to_numpy(bool))
    return errors


def verify_pending_reference(arm, saved, context, client, ids, horizon):
    if saved is None: raise RuntimeError("Complete independent pending study before rollout comparison")
    rows = saved.loc[(saved.context == context) & (saved.client == client) & (saved.arm == "pending_hold")]
    if len(rows) != len(ids): raise RuntimeError("Saved pending reference denominator changed")
    rows = rows.set_index("scenario_id").loc[ids]
    values = path_values(arm, horizon)
    for field in METRICS:
        np.testing.assert_allclose(values[field], rows[field].to_numpy(float), atol=2e-10, rtol=5e-13)
    for field, saved_field in (("nominal_fees", "nominal_fees"), ("nominal_moved", "nominal_moved"),
                               ("trade_fraction", "actual_trade_fraction")):
        np.testing.assert_allclose(values[field], rows[saved_field].to_numpy(float), atol=2e-10, rtol=5e-13)


def verify_review(saved, context, client, ids):
    rows = saved.loc[(saved.context == context) & (saved.client == client) & (saved.view == "effective")]
    if (len(rows) != len(ids) or set(rows.scenario_id) != set(ids) or rows.assigned.any()
            or not rows.assignment.eq("Review").all()
            or not rows[[*METRICS, "breach", "excess", "candidate_local_index"]].isna().all().all()):
        raise RuntimeError("Frozen onboarding Review/null denominator changed")


def summarize(frame):
    summaries = []
    for (scope, client, policy), group in frame.groupby(["scope", "client", "policy"], sort=False):
        auto = group.loc[group.assigned]
        row = {"scope": scope, "client": client, "policy": policy, "total_paths": len(group),
               "assigned_paths": len(auto), "review_paths": len(group) - len(auto),
               "horizon": int(group.horizon.iloc[0]), "loss_tolerance": float(group.loss_tolerance.iloc[0])}
        fields = ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd",
                  "breach_count", "breach_probability", "mean_excess", *[f"mean_{x}" for x in ACCOUNTS]]
        row.update(dict.fromkeys(fields))
        if len(auto):
            row.update({"mean_log_growth": float(auto.log_growth.mean()), "median_cagr": float(auto.cagr.median()),
                        "p10_cagr": float(np.quantile(auto.cagr, .10, method="inverted_cdf")),
                        "median_mdd": float(auto.mdd.median()),
                        "p90_mdd": float(np.quantile(auto.mdd, .90, method="inverted_cdf")),
                        "cvar90_mdd": float(discrete_cvar(auto.mdd.to_numpy())),
                        "breach_count": int(auto.breach.astype(bool).sum()),
                        "breach_probability": float(auto.breach.astype(bool).mean()), "mean_excess": float(auto.excess.mean()),
                        **{f"mean_{x}": float(auto[x].mean()) for x in ACCOUNTS}})
        summaries.append(row)
    return summaries


def paired_summary(frame):
    keys = ["context", "client", "scenario_id"]
    base = frame.loc[frame.policy == "annual_then_pending"]
    result = []
    for policy in POLICIES[1:]:
        joined = frame.loc[frame.policy == policy].merge(base, on=keys, suffixes=("", "_reference"), validate="one_to_one")
        for (scope, client), group in joined.groupby(["scope", "client"], sort=False):
            auto = group.loc[group.assigned & group.assigned_reference]
            row = {"scope": scope, "client": client, "policy": policy, "reference": "annual_then_pending",
                   "total_pairs": len(group), "assigned_pairs": len(auto), "review_pairs": len(group) - len(auto)}
            for field in (*METRICS, *ACCOUNTS, "excess"):
                row[f"mean_delta_{field}"] = float((auto[field] - auto[field + "_reference"]).mean()) if len(auto) else None
            row["breach_count_delta"] = int(auto.breach.astype(bool).sum() - auto.breach_reference.astype(bool).sum()) if len(auto) else None
            result.append(row)
    return result


def check_coverage(frame):
    keys = ["scope", "context", "client", "scenario_id", "policy"]
    if frame.duplicated(keys).any(): raise RuntimeError("Duplicated client/path/policy result")
    if len(frame) != 3600: raise RuntimeError("Require full and pooled OOF120×5×3 original rows")
    for (scope, client, policy), group in frame.groupby(["scope", "client", "policy"]):
        if len(group) != 120 or set(group.scenario_id) != set(range(1, 121)):
            raise RuntimeError("Original full/OOF denominator changed")
    review = frame.loc[~frame.assigned]
    if not review[[*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing"]].isna().all().all():
        raise RuntimeError("Review has imputed financial outcomes")


def main():
    from experimental_execution.fan_rollout import simulate_policy
    started = time.perf_counter(); protocol = json.loads(PROTOCOL.read_text()); validate_protocol(protocol)
    sources, inputs = source_manifest(), frozen_inputs(protocol)
    for extra in [ROOT / "docs/risk_menu_development_protocol.json", DATA,
                  PENDING / "manifest.json", PENDING / "client_path_metrics.csv", PENDING / "independent_audit.json"]:
        if not extra.exists(): raise RuntimeError(f"Required completed dependency missing:{extra}")
        inputs[str(extra.relative_to(ROOT))] = sha(extra)
    pending_audit = json.loads((PENDING / "independent_audit.json").read_text())
    if pending_audit.get("status") != "passed": raise RuntimeError("Pending baseline independent audit did not pass")
    contexts = json.loads((STRICT / "nested_context_refits.json").read_text()); validate_contexts(contexts)
    mapping = json.loads((STRICT / "candidate_mapping.json").read_text())
    if not isinstance(mapping, list) or len(mapping) != 172: raise ValueError("Require explicit172 mapping rows")
    clients = json.loads((ROOT / "docs/risk_menu_development_protocol.json").read_text())["clients"]
    strict_clients = json.loads((ROOT / "docs/positive_asset_specification_protocol.json").read_text())["clients"]
    if clients != strict_clients: raise RuntimeError("Original client H/L/attributes changed")
    originals = pd.concat([pd.read_csv(STRICT / "nested_full_paths.csv"), pd.read_csv(STRICT / "nested_outer_oof_paths.csv")], ignore_index=True)
    pending_saved = pd.read_csv(PENDING / "client_path_metrics.csv")
    returns, inflation, ids = load_paths(DATA, range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)): raise RuntimeError("Development identities changed")
    data_digest = hashlib.sha256()
    for array in (ids, returns, inflation): data_digest.update(np.ascontiguousarray(array).tobytes())
    records, gates, checks, meta, traces = [], [], {}, {}, {}
    for context, fit in contexts.items():
        scope = "full_development" if context == "full_development" else "outer_oof"
        evaluation_ids = fit["training_ids"] if scope == "full_development" else fit["outer_test_ids"]
        eval_ix, train_ix = np.asarray(evaluation_ids) - 1, np.asarray(fit["training_ids"]) - 1
        meta[context] = {"training_ids": fit["training_ids"], "evaluation_ids": evaluation_ids,
                         "effective_assignments": fit["effective_assignments"], "products": fit["raw_fit"]["menu"]}
        for client in clients:
            name, horizon, loss = client["name"], client["horizon"], client["loss_tolerance"]
            assigned = fit["effective_assignments"][name]
            product = next((item for item in fit["raw_fit"]["menu"] if item["name"] == assigned), None)
            if assigned != "Review" and product is None: raise RuntimeError("Frozen assignment does not identify a product")
            common = {"scope": scope, "context": context, "outer_fold": fit["context_id"] or None, "client": name,
                      "horizon": horizon, "loss_tolerance": loss, "assignment": assigned, "assigned": product is not None,
                      "candidate_local_index": product["candidate_local_index"] if product else None,
                      "candidate_original232_index": product["candidate_original232_index"] if product else None}
            if product is None:
                verify_review(originals, context, name, evaluation_ids)
                for policy in POLICIES:
                    for sid in evaluation_ids:
                        records.append({**common, "scenario_id": sid, "policy": policy,
                                        **dict.fromkeys([*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing"])})
                continue
            target = validate_product(product, mapping)
            pair = simulate_paired(returns[eval_ix], inflation[eval_ix], target, loss, horizon=horizon)
            annual, reference = squeeze_pending(pair["annual"]), squeeze_pending(pair["pending_hold"])
            checks[f"{context}__{name}"] = {"annual_replay_max_abs_errors": verify_original(annual, originals, context, name, evaluation_ids, product, horizon),
                                           "paired_prefix_identity": pair["coupling"]}
            verify_pending_reference(reference, pending_saved, context, name, evaluation_ids, horizon)
            simulations = {"annual_then_pending": reference}
            for policy in POLICIES[1:]:
                simulations[policy] = simulate_policy(returns[eval_ix, :horizon], inflation[eval_ix, :horizon],
                    returns[train_ix, :horizon], inflation[train_ix, :horizon], target,
                    policy=policy, L=loss, cost_rate=.0025)
            for policy, arm in simulations.items():
                fields = path_values(arm, horizon)
                prefix = f"{context}__{name}__{policy}"
                traces[prefix + "__scenario_ids"] = np.asarray(evaluation_ids)
                for key, value in arm.items():
                    if isinstance(value, np.ndarray): traces[prefix + "__" + key] = value
                for i, sid in enumerate(evaluation_ids):
                    first = int(arm["first_crossing"][i])
                    records.append({**common, "scenario_id": sid, "policy": policy,
                                    **{field: float(fields[field][i]) for field in (*METRICS, *ACCOUNTS)},
                                    "breach": bool(arm["ever_breached"][i, -1]),
                                    "excess": float(max(fields["mdd"][i] - loss, 0)), "first_crossing": first if first >= 0 else None})
                    for t in range(horizon):
                        gate = {**common, "scenario_id": sid, "policy": policy, "year": t + 1,
                                "available_training_fan_paths": len(train_ix),
                                "forecast_paths": int(arm["forecast_n"][i, t]) if "forecast_n" in arm and arm["forecast_n"][i, t] >= 0 else None,
                                "cpi": float(arm["cpi"][i, t]),
                                "real_peak": float(arm["real_peak"][i, t]), "running_mdd": float(arm["running_mdd"][i, t]),
                                "ever_breached": bool(arm["ever_breached"][i, t]),
                                "action": str(arm["actions"][i, t]), "fee": float(arm["fees"][i, t]),
                                "moved": float(arm["moved"][i, t]), "actual_trade": bool(arm["trades"][i, t])}
                        for asset, value in zip(("equity", "bond", "bill"), arm["holdings"][i, t]): gate["holding_" + asset] = float(value)
                        for key in ("qg_hold", "qg_restore", "qbad_count_hold", "qbad_count_restore", "delete_one_flip_count"):
                            arm_key = {"qbad_count_hold": "qbad_hold_count", "qbad_count_restore": "qbad_restore_count"}.get(key, key)
                            value = arm[arm_key][i, t] if arm_key in arm else np.nan
                            is_count = key not in ("qg_hold", "qg_restore")
                            gate[key] = float(value) if np.isfinite(value) and (not is_count or value >= 0) else None
                        gate["gate_reason"] = str(arm["gate_reason"][i, t]) if "gate_reason" in arm else ("mandatory_initial" if t == 0 else "pending_hold" if arm["pending_at_decision"][i, t] else "annual_reference")
                        if gate["qg_hold"] is not None:
                            advantage = gate["qg_restore"] - gate["qg_hold"]
                            gate["predicted_growth_advantage_restore"] = advantage
                            gate["predicted_breach_count_delta_restore"] = gate["qbad_count_restore"] - gate["qbad_count_hold"]
                            gate["restoration_predicted_growth_sacrifice"] = gate["action"] == "restore" and advantage < -NUMERIC_TOL
                        else:
                            for field in ("predicted_growth_advantage_restore", "predicted_breach_count_delta_restore", "restoration_predicted_growth_sacrifice"):
                                gate[field] = None
                        gates.append(gate)
        print(json.dumps({"context": context, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    frame = pd.DataFrame(records); check_coverage(frame)
    gate_frame = pd.DataFrame(gates)
    sources_end, inputs_end = source_manifest(), {key: sha(ROOT / key) for key in inputs}
    if sources != sources_end or inputs != inputs_end: raise RuntimeError("Source/input changed during experiment")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "client_paths.csv", index=False)
    pd.DataFrame(summarize(frame)).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(paired_summary(frame)).to_csv(OUT / "paired_summary.csv", index=False)
    gate_frame.to_csv(OUT / "decision_gates.csv", index=False)
    np.savez_compressed(OUT / "actual_traces.npz", **traces)
    write_json(OUT / "contexts.json", meta); write_json(OUT / "accounting_checks.json", checks)
    outputs = {path.name: sha(path) for path in OUT.iterdir() if path.name != "manifest.json" and path.name != "independent_audit.json"}
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "sources": sources, "sources_at_end": sources_end,
        "input_hashes": inputs, "input_hashes_at_end": inputs_end, "outputs_sha256": outputs,
        "development_array_sha256": data_digest.hexdigest(), "scenario_ids_loaded": ids.tolist(), "final_scenarios_loaded": False,
        "policies": list(POLICIES), "client_path_rows": len(frame), "decision_rows": len(gate_frame),
        "assigned_rows": int(frame.assigned.sum()), "review_null_rows": int((~frame.assigned).sum()),
        "scope": "Frozen strict172 product execution comparison with training-only recombined calendar fans; no refit, policy selection, fresh holdout, dynamic-programming guarantee or risk certificate"})
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "client_path_rows": len(frame), "decision_rows": len(gate_frame)}))


if __name__ == "__main__":
    main()
