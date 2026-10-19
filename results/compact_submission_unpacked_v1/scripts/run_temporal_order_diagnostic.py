"""Frozen H30 temporal-order counterfactuals on development paths only.

All three assets and inflation move together within each original scenario.
The forecast, Bellman tables, targets and assignments remain the original
ones. These transformed paths are neither independent data nor an IID test.
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
import run_fixed_execution_ablation as fixed
import run_finite_horizon_execution_prototype as prototype
import run_risk_menu_research as primary
from safeinvest.data import load_paths

PROTOCOL = ROOT / "docs/temporal_order_diagnostic_protocol.json"
OUT = ROOT / "results/model_review/temporal_order_v1"
PRIOR = ROOT / "results/model_review/finite_horizon_execution_v1"
CASE = "alpha_10__risk_only"
POLICIES = ("annual_restore", "initial_then_hold", "finite_horizon_gate")


def make_orders(n_scenarios, years, seed):
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (n_scenarios, years, seed)) or min(n_scenarios, years) < 1 or seed < 0:
        raise ValueError("Positive integer dimensions and nonnegative integer seed required")
    rng = np.random.default_rng(seed)
    original = np.tile(np.arange(years), (n_scenarios, 1))
    return {"original": original, "reverse": original[:, ::-1].copy(),
            "permuted": np.stack([rng.permutation(years) for _ in range(n_scenarios)])}


def apply_order(returns, inflation, indices):
    market, infl, order = np.asarray(returns), np.asarray(inflation), np.asarray(indices)
    if (market.ndim != 3 or market.shape[-1] != 3 or infl.shape != market.shape[:2]
            or order.shape != market.shape[:2] or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order, axis=1), np.tile(np.arange(market.shape[1]), (len(market), 1)))):
        raise ValueError("Each scenario requires a complete within-path annual permutation")
    return np.take_along_axis(market, order[..., None], axis=1), np.take_along_axis(infl, order, axis=1)


def client_c_rows(fields, fitted, scenario_ids, *, context="toy", scope="full", fold=None,
                  order="original", policy="annual_restore"):
    rows = fixed.client_rows({30: fields}, fitted, [{"name": "C", "horizon": 30, "loss_tolerance": .40}],
                             scenario_ids, context=context, scope=scope, fold=fold, policy=policy)
    for row in rows:
        row["order"] = order
    return rows


def summarize(frame, keys, *, clients=False):
    result = fixed.summarize(frame, keys, clients=clients)
    extra = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple): values = (values,)
        selected = group.loc[group.assigned] if clients else group
        extra.append({**dict(zip(keys, values)), "mean_mdd": float(selected.mdd.mean()) if len(selected) else None,
                      "distinct_frozen_candidate_ids": int(selected.candidate_index.nunique())})
    return result.merge(pd.DataFrame(extra), on=keys, validate="one_to_one")


def paired_order_differences(frame, keys, *, clients=False):
    reference = frame.loc[frame.order == "original"]
    output = []
    for order in ("reverse", "permuted"):
        changed = frame.loc[frame.order == order]
        merged = reference.merge(changed, on=keys, suffixes=("_original", "_changed"), validate="one_to_one")
        if len(merged) != len(reference) or len(merged) != len(changed):
            raise RuntimeError("Counterfactuals do not contain identical paired identities")
        data = merged[keys].copy()
        data["changed_order"] = order
        assigned = merged.assigned_original.to_numpy() if clients else np.ones(len(merged), bool)
        if clients and not np.array_equal(assigned, merged.assigned_changed):
            raise RuntimeError("Reordering changed a frozen client assignment")
        data["assigned"] = assigned
        for field in fixed.VALUE_FIELDS + (("excess",) if clients else ()):
            data["delta_" + field] = merged[field + "_changed"] - merged[field + "_original"]
        if clients:
            before, after = merged.breach_original.eq(True), merged.breach_changed.eq(True)
            data["delta_breach"] = np.where(assigned, after.astype(int) - before.astype(int), np.nan)
            data["new_breach"] = np.where(assigned, after & ~before, np.nan)
            data["averted_breach"] = np.where(assigned, before & ~after, np.nan)
        output.append(data)
    return pd.concat(output, ignore_index=True)


def tail_differences(summary, keys, entity):
    original = summary.loc[summary.order == "original"]
    output = []
    for order in ("reverse", "permuted"):
        merged = original.merge(summary.loc[summary.order == order], on=keys,
                                suffixes=("_original", "_changed"), validate="one_to_one")
        for _, row in merged.iterrows():
            item = {"entity": entity, "changed_order": order, **{key: row[key] for key in keys},
                    "interpretation": "Difference of group statistics, not quantile/CVaR of paired path differences."}
            for field in ("mean_log_growth", "mean_mdd", "p90_mdd", "cvar90_mdd", "mean_nominal_fees", "mean_trade_fraction", "breach_probability"):
                if field + "_original" in merged:
                    a, b = row[field + "_original"], row[field + "_changed"]
                    item["original_" + field], item["changed_" + field] = a, b
                    item["delta_group_" + field] = b - a if pd.notna(a) and pd.notna(b) else None
            output.append(item)
    return pd.DataFrame(output)


def source_manifest():
    sources = prototype.source_manifest()
    for path in (Path(__file__).resolve(), PROTOCOL, ROOT / "tests/test_temporal_order_diagnostic.py"):
        sources[str(path.relative_to(ROOT))] = fixed.sha(path)
    return sources


def input_manifest():
    names = [primary.OUT / "full_development_menus.json", primary.OUT / "validation_fold_menus.json",
             primary.OUT / "fit_manifest.json", primary.OUT / "validation_manifest.json",
             PRIOR / "manifest.json", PRIOR / "solution_tables.npz", PRIOR / "target_path_metrics.csv", PRIOR / "client_path_metrics.csv"]
    return {str(path.relative_to(ROOT)): fixed.sha(path) for path in names}


def verify_original(frame, previous, *, clients=False):
    reference = previous.loc[(previous.horizon == 30) &
                             (previous.policy.isin(POLICIES + (() if clients else ("equal_weight_annual",)))) &
                             ((previous.mesh_step == .05) | previous.mesh_step.isna())]
    if clients:
        reference = reference.loc[reference.client == "C"]
    actual = frame.loc[frame.order == "original"]
    keys = ["context", "scope", "fold", "scenario_id", "policy", "horizon",
            *( ["client", "assignment", "candidate_index"] if clients else ["portfolio", "candidate_index"])]
    merged = actual.merge(reference, on=keys, suffixes=("_new", "_prior"), validate="one_to_one")
    if len(merged) != len(actual) or len(merged) != len(reference):
        raise RuntimeError("Original-order execution identities differ from prior H30 prototype")
    if clients and not np.array_equal(merged.assigned_new, merged.assigned_prior):
        raise RuntimeError("Original C assignment differs from frozen artifact")
    selected = merged.loc[merged.assigned_new] if clients else merged
    errors = {}
    for field in fixed.VALUE_FIELDS:
        values, expected = selected[field + "_new"].to_numpy(float), selected[field + "_prior"].to_numpy(float)
        np.testing.assert_allclose(values, expected, atol=1e-10, rtol=1e-12)
        errors[field] = float(np.max(np.abs(values - expected), initial=0))
    return {"paired_rows": len(merged), "maximum_absolute_errors": errors}


def main():
    started = time.perf_counter()
    sources, inputs = source_manifest(), input_manifest()
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol["case"] != CASE or protocol["horizon"] != 30 or protocol["cost_rate"] != .0025
            or protocol["primary_mesh_step"] != .05 or protocol["permutation_seed"] != 87020261019):
        raise ValueError("Temporal diagnostic protocol differs from the declared fixed design")
    prior_manifest = json.loads((PRIOR / "manifest.json").read_text())
    for name, digest in prior_manifest["sources"].items():
        if fixed.sha(ROOT / name) != digest:
            raise RuntimeError("Prototype source snapshot changed")
    for name in ("solution_tables.npz", "target_path_metrics.csv", "client_path_metrics.csv"):
        if fixed.sha(PRIOR / name) != prior_manifest["outputs_sha256"][name]:
            raise RuntimeError("Prior prototype output hash mismatch")
    research = json.loads(primary.PROTOCOL.read_text())
    primary.validate_folds(research)
    full = primary.compatible_full_fit()[CASE]
    validation_manifest = json.loads((primary.OUT / "validation_manifest.json").read_text())
    if validation_manifest["sources"] != primary.source_manifest():
        raise RuntimeError("Frozen fold selection sources changed")
    folds = json.loads((primary.OUT / "validation_fold_menus.json").read_text())[CASE]
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)):
        raise RuntimeError("Only original development IDs1..120 permitted")
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    if digest.hexdigest() != prior_manifest["development_arrays_sha256"]:
        raise RuntimeError("Development market inputs differ from the frozen prototype")
    orders = make_orders(120, 30, protocol["permutation_seed"])
    for name, order in orders.items():
        changed_r, changed_i = apply_order(returns, inflation, order)
        undo_r, undo_i = apply_order(changed_r, changed_i, np.argsort(order, axis=1))
        if not np.array_equal(undo_r, returns) or not np.array_equal(undo_i, inflation):
            raise RuntimeError("Joint multiset preservation failed: " + name)
    contexts = [("full_development", "full", None, np.arange(120), np.arange(120), full)]
    for fitted, declared in zip(folds, research["validation"]["folds"], strict=True):
        if (fitted["fold"] != declared["fold"] or fitted["train_ids"] != declared["train_ids"]
                or fitted["test_ids"] != declared["test_ids"]):
            raise RuntimeError("Frozen fold differs from predeclared96/24 split")
        contexts.append((f"fold_{fitted['fold']}_test", "oof", fitted["fold"], np.asarray(fitted["train_ids"]) - 1,
                         np.asarray(fitted["test_ids"]) - 1, fitted))
    target_records, client_records, reconstruction, invariants = [], [], [], []
    with np.load(PRIOR / "solution_tables.npz", allow_pickle=False) as tables:
        for context, scope, fold_number, train, test, fitted in contexts:
            if scope == "oof" and np.intersect1d(train, test).size:
                raise RuntimeError("Train/test overlap")
            targets = np.asarray([p["weights"] for p in fitted["menu"]])
            solution = prototype.solve_finite_horizon(1 + returns[train], targets, horizon=30, step=.05, cost_rate=.0025)
            prefix = context + "__mesh_050"
            component_errors = {}
            for name, actual in {"V": solution.V, "tradebase": solution.tradebase,
                                 "L_grid": solution.L_grid, "L_target": solution.L_target,
                                 "targets": solution.targets, "gross_samples": solution.gross_samples,
                                 "mesh_states": solution.mesh.states}.items():
                expected = tables[prefix + "__" + name]
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
                component_errors[name] = float(np.max(np.abs(actual - expected), initial=0))
            reconstruction.append({"context": context, "train_ids": ids[train].tolist(), "test_ids": ids[test].tolist(),
                                   "original_training_gross_sha256": hashlib.sha256(np.ascontiguousarray(1 + returns[train]).tobytes()).hexdigest(),
                                   "component_maximum_absolute_errors": component_errors,
                                   "forecast_retrained_on_counterfactual": False, "frozen_C_assignment": fitted["assignments"]["C"]})
            original_hold = None
            for order_name, all_indices in orders.items():
                market, infl = apply_order(returns[test], inflation[test], all_indices[test])
                simulations = {policy: fixed.simulate_execution(market, infl, targets, policy, .0025)
                               for policy in ("annual_restore", "initial_then_hold")}
                simulations["finite_horizon_gate"], _ = prototype.simulate_gate(market, infl, solution, 30, "finite_horizon_gate")
                fixed.verify_hold_closed_form(simulations["initial_then_hold"], market, infl, .0025)
                hold_fields = fixed.horizon_values(simulations["initial_then_hold"], 30)
                if order_name == "original":
                    original_hold = hold_fields
                else:
                    errors = {}
                    for name in ("terminal_real", "log_growth", "cagr"):
                        np.testing.assert_allclose(hold_fields[name], original_hold[name], atol=1e-10, rtol=3e-13)
                        errors[name] = float(np.max(np.abs(hold_fields[name] - original_hold[name]), initial=0))
                    invariants.append({"context": context, "order": order_name, "buy_hold_terminal_invariance_max_abs": errors})
                for policy, simulation in simulations.items():
                    fields = fixed.horizon_values(simulation, 30)
                    rows = fixed.target_rows(fields, fitted["menu"], ids[test], context=context, scope=scope,
                                             fold=fold_number, policy=policy, horizon=30)
                    for row in rows: row["order"] = order_name
                    target_records.extend(rows)
                    client_records.extend(client_c_rows(fields, fitted, ids[test], context=context, scope=scope,
                                                         fold=fold_number, order=order_name, policy=policy))
                equal = np.full((1, 3), 1 / 3)
                equal_sim = fixed.simulate_execution(market, infl, equal, "annual_restore", .0025)
                equal_id = None
                # This comparator is exact thirds, independent of candidate ID
                # ordering; retain the known main-grid identity when available.
                weights = np.asarray([[i / 20, j / 20, (20 - i - j) / 20] for i in range(21) for j in range(21 - i)] + [[1 / 3] * 3])
                weights = weights[np.lexsort((weights[:, 2], weights[:, 1], weights[:, 0]))]
                equal_id = int(np.flatnonzero(np.all(np.abs(weights - equal[0]) <= 1e-12, axis=1))[0])
                menu = [{"name": "EqualWeightReference", "candidate_index": equal_id, "weights": equal[0].tolist()}]
                equal_rows = fixed.target_rows(fixed.horizon_values(equal_sim, 30), menu, ids[test], context=context,
                                               scope=scope, fold=fold_number, policy="equal_weight_annual", horizon=30)
                for row in equal_rows: row["order"] = order_name
                target_records.extend(equal_rows)
                print(json.dumps({"context": context, "order": order_name, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    target, client = pd.DataFrame(target_records), pd.DataFrame(client_records)
    original_checks = {"targets": verify_original(target, pd.read_csv(PRIOR / "target_path_metrics.csv")),
                       "client_C": verify_original(client, pd.read_csv(PRIOR / "client_path_metrics.csv"), clients=True)}
    target_keys = ["context", "scope", "fold", "scenario_id", "portfolio", "candidate_index", "policy", "horizon"]
    client_keys = ["context", "scope", "fold", "scenario_id", "client", "assignment", "candidate_index", "policy", "horizon", "loss_tolerance"]
    target_pairs = paired_order_differences(target, target_keys)
    client_pairs = paired_order_differences(client, client_keys, clients=True)
    target_groups = ["scope", "order", "policy", "portfolio", "horizon"]
    client_groups = ["scope", "order", "policy", "client", "horizon", "loss_tolerance"]
    target_summary, client_summary = summarize(target, target_groups), summarize(client, client_groups, clients=True)
    for frame, groups in ((target, target_groups), (client, client_groups)):
        for _, group in frame.groupby(groups, dropna=False):
            if len(group) != 120 or group.scenario_id.nunique() != 120:
                raise RuntimeError("Full/OOF summary lost the complete120 original-path denominator")
    outputs = {"target_path_metrics.csv": target, "client_C_path_metrics.csv": client,
               "target_summary.csv": target_summary, "client_C_summary.csv": client_summary,
               "target_context_summary.csv": summarize(target, ["context", "scope", "fold", "order", "policy", "portfolio", "candidate_index", "horizon"]),
               "client_C_context_summary.csv": summarize(client, ["context", "scope", "fold", "order", "policy", "client", "assignment", "candidate_index", "horizon", "loss_tolerance"], clients=True),
               "paired_target_path_differences.csv": target_pairs, "paired_client_C_path_differences.csv": client_pairs,
               "paired_target_summary.csv": fixed.summarize_paired(target_pairs, ["scope", "changed_order", "policy", "portfolio", "horizon"]),
               "paired_client_C_summary.csv": fixed.summarize_paired(client_pairs, ["scope", "changed_order", "policy", "client", "horizon", "loss_tolerance"], clients=True),
               "group_tail_differences.csv": pd.concat([tail_differences(target_summary, ["scope", "policy", "portfolio", "horizon"], "target"),
                                                        tail_differences(client_summary, ["scope", "policy", "client", "horizon", "loss_tolerance"], "client_C")], ignore_index=True)}
    sources_end, inputs_end = source_manifest(), input_manifest()
    if sources_end != sources or inputs_end != inputs:
        raise RuntimeError("Source/frozen input drift; no temporal results written")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items(): frame.to_csv(OUT / name, index=False)
    primary.write_json(OUT / "permutations.json", {"seed": protocol["permutation_seed"],
                                                  "construction": protocol["orders"]["permuted"],
                                                  "scenario_permutations": [{"scenario_id": int(sid),
                                                                             "zero_based_indices": orders["permuted"][i].tolist(),
                                                                             "original_periods_in_new_order": (orders["permuted"][i] + 1).tolist()}
                                                                            for i, sid in enumerate(ids)]})
    primary.write_json(OUT / "checks.json", {"joint_multisets_preserved": True, "forecast_table_reconstruction": reconstruction,
                                            "buy_hold_terminal_invariance": invariants, "original_H30_reproduction": original_checks,
                                            "target_rows": len(target), "client_C_rows": len(client),
                                            "horizon": 30, "no_counterfactual_target_or_assignment_refit": True})
    files = [*outputs, "permutations.json", "checks.json"]
    if source_manifest() != sources or input_manifest() != inputs:
        raise RuntimeError("Provenance drift during output writing; no completion manifest")
    primary.write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                              "elapsed_seconds": time.perf_counter() - started,
                                              "sources": sources, "sources_at_end": sources_end,
                                              "input_hashes": inputs, "input_hashes_at_end": inputs_end,
                                              "outputs_sha256": {name: fixed.sha(OUT / name) for name in files},
                                              "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(),
                                              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                                              "protocol": protocol, "final_scenarios_loaded": False,
                                              "forecast_refitted_on_reordered_paths": False,
                                              "new_policy_or_winner_selected": False,
                                              "scope": protocol["scope"], "limitations": protocol["interpretation_limits"],
                                              "OOF_roles": "Pooled role summaries contain different frozen target weights/forecast laws by fold; distinct candidate counts are disclosed. These are internal procedure stress diagnostics.",
                                              "fees": "Nominal cumulative fees are accounting units; annual fee fractions use current pretrade wealth and are not CAGR differences."})
    print(json.dumps({"output": str(OUT), "target_rows": len(target), "client_C_rows": len(client),
                      "forecast_reconstruction_contexts": len(reconstruction), "elapsed_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
