"""Frozen-target execution prototype on guarded development paths only.

Every finite-horizon policy is simulated afresh at its true H. Forecasts are
learned from the full training context or each recorded 96-path training fold;
targets and assignments are read, never refitted. Current Q uses direct
empirical current-state transitions, not interpolation of grid Qhold.
"""
from __future__ import annotations

import argparse
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
import run_risk_menu_research as primary
from experimental_execution.finite_horizon import evaluate_action, solve_finite_horizon
from safeinvest.core import execute_target
from safeinvest.data import load_paths

PROTOCOL = ROOT / "docs/finite_horizon_execution_prototype_protocol.json"
OUT = ROOT / "results/model_review/finite_horizon_execution_v1"
DYNAMIC = ("myopic_gate", "finite_horizon_gate")
MESHES = (.05, .025)
HORIZONS = (5, 6, 15, 25, 30)
BASE = ROOT / "results/model_review/fixed_execution_ablation_v1"
TREE_AUDIT = OUT / "synthetic_tree_audit.json"


def source_manifest():
    sources = fixed.source_manifest()
    for path in [Path(__file__).resolve(), PROTOCOL, ROOT / "research/experimental_execution/finite_horizon.py"]:
        sources[str(path.relative_to(ROOT))] = fixed.sha(path)
    return sources


def input_manifest():
    result = fixed.input_manifest()
    for name in ["manifest.json", "target_path_metrics.csv", "executed_paths.npz"]:
        result[str((BASE / name).relative_to(ROOT))] = fixed.sha(BASE / name)
    result[str(TREE_AUDIT.relative_to(ROOT))] = fixed.sha(TREE_AUDIT)
    return result


def evaluate_unique(solution, current_weights, remaining, target_id, *, chunk_size=32):
    """Deduplicate exactly identical states; never round, project or quantize u."""
    u = np.asarray(current_weights, float)
    shape = np.broadcast_shapes(u.shape[:-1], np.shape(remaining), np.shape(target_id))
    weights = np.broadcast_to(u, shape + (3,)).reshape(-1, 3)
    h = np.broadcast_to(remaining, shape).reshape(-1)
    k = np.broadcast_to(target_id, shape).reshape(-1)
    if not np.issubdtype(h.dtype, np.integer) or not np.issubdtype(k.dtype, np.integer):
        raise ValueError("Remaining horizon and target IDs must be integers")
    unique, inverse = np.unique(np.column_stack((weights, h, k)), axis=0, return_inverse=True)
    result = evaluate_action(solution, unique[:, :3], unique[:, 3].astype(int), unique[:, 4].astype(int), chunk_size=chunk_size)
    return {key: np.asarray(value)[inverse].reshape(shape) for key, value in result.items()}


def simulate_gate(returns, inflation, solution, horizon, policy, *, chunk_size=32):
    """Execute causal actions; realized paired gain is calculated afterwards.

    The action API receives only frozen solution, actual u, h and target ID.
    Actual evaluation gross and inflation are never arguments to that API.
    Counterfactual restore-vs-hold gain is retained even when hold is selected;
    selected_action_realized_gain is zero for a selected hold.
    """
    market, infl = np.asarray(returns, float), np.asarray(inflation, float)
    if policy not in DYNAMIC or isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)):
        raise ValueError("Require a dynamic prototype policy and integer H")
    if (market.ndim != 3 or market.shape[-1] != 3 or not 1 <= horizon <= market.shape[1]
            or infl.shape != market.shape[:2] or not np.all(np.isfinite(market))
            or not np.all(np.isfinite(infl)) or np.any(market <= -1) or np.any(infl <= -1)
            or horizon > solution.horizon):
        raise ValueError("Invalid evaluated market or horizon")
    targets = solution.targets
    n, m = len(market), len(targets)
    shape = (n, m, horizon)
    sim = {key: np.zeros(shape) for key in ["fees", "moved", "pretrade_wealth", "fee_fraction", "moved_fraction",
                                            "pretrade_tv_drift", "end_year_tv_drift"]}
    sim["nominal_wealth"] = np.full((n, m, horizon + 1), 100.)
    sim["end_weights"] = np.zeros(shape + (3,))
    sim["target_execution"] = np.zeros(shape, dtype=bool)
    trace = {key: np.zeros(shape) for key in ["hold_q", "trade_q", "gain", "one_period_gain", "current_log_return",
                                              "hypothetical_fee_fraction", "restore_realized_paired_gain",
                                              "selected_action_realized_gain"]}
    trace["pretrade_weights"] = np.zeros(shape + (3,))
    trace["remaining_horizon"] = np.zeros(shape, dtype=int)
    trace["decision_h"] = np.zeros(shape, dtype=int)
    trace["trade"] = np.zeros(shape, dtype=bool)
    trace["identity"] = np.zeros(shape, dtype=bool)
    holdings = np.zeros((n, m, 3)); holdings[..., 2] = 100.
    target_ids = np.arange(m)[None, :]
    for year in range(horizon):
        wealth = holdings.sum(axis=-1)
        actual_u = holdings / wealth[..., None]
        decision_h = 1 if policy == "myopic_gate" else horizon - year
        # This call has no access to evaluated future returns or inflation.
        decision = evaluate_unique(solution, actual_u, decision_h, target_ids, chunk_size=chunk_size)
        trade = decision["trade"]
        post_restore, potential_fee, potential_moved = execute_target(holdings, targets[None], solution.cost_rate)
        np.testing.assert_allclose(potential_fee / wealth, decision["fee_fraction"], atol=5e-15, rtol=1e-12)
        post = np.where(trade[..., None], post_restore, holdings)
        sim["pretrade_wealth"][..., year] = wealth
        sim["pretrade_tv_drift"][..., year] = .5 * np.abs(actual_u - targets).sum(axis=-1)
        sim["fees"][..., year] = np.where(trade, potential_fee, 0.)
        sim["moved"][..., year] = np.where(trade, potential_moved, 0.)
        sim["fee_fraction"][..., year] = sim["fees"][..., year] / wealth
        sim["moved_fraction"][..., year] = sim["moved"][..., year] / wealth
        sim["target_execution"][..., year] = trade
        trace["pretrade_weights"][..., year, :] = actual_u
        trace["remaining_horizon"][..., year] = horizon - year
        trace["decision_h"][..., year] = decision_h
        trace["trade"][..., year] = trade
        trace["identity"][..., year] = decision["identity"]
        for key in ["hold_q", "trade_q", "gain", "one_period_gain", "current_log_return"]:
            trace[key][..., year] = decision[key]
        trace["hypothetical_fee_fraction"][..., year] = decision["fee_fraction"]
        # Only after selecting the action is the realized year-t gross used.
        gross = 1 + market[:, None, year, :]
        hold_next = (holdings * gross).sum(axis=-1)
        restore_next = (post_restore * gross).sum(axis=-1)
        realized_delta = np.log(restore_next / hold_next)
        trace["restore_realized_paired_gain"][..., year] = realized_delta
        trace["selected_action_realized_gain"][..., year] = np.where(trade, realized_delta, 0.)
        holdings = post * gross
        nominal = holdings.sum(axis=-1)
        sim["nominal_wealth"][..., year + 1] = nominal
        end_u = holdings / nominal[..., None]
        sim["end_weights"][..., year, :] = end_u
        sim["end_year_tv_drift"][..., year] = .5 * np.abs(end_u - targets).sum(axis=-1)
    price = np.column_stack((np.ones(n), np.cumprod(1 + infl[:, :horizon], axis=1)))
    sim["real_wealth"] = sim["nominal_wealth"] / price[:, None, :]
    sim["trades"] = sim["moved"] > fixed.NUMERIC_TOL
    sim["targets"] = targets
    return sim, trace


def add_rows(target_records, client_records, simulation, fitted, profiles, ids, *, context, scope, fold, policy, mesh, horizon):
    fields = fixed.horizon_values(simulation, horizon)
    rows = fixed.target_rows(fields, fitted["menu"], ids, context=context, scope=scope, fold=fold, policy=policy, horizon=horizon)
    for i, row in enumerate(rows):
        row["mesh_step"] = mesh
        row["initial_trade"] = bool(simulation["trades"][i // len(fitted["menu"]), i % len(fitted["menu"]), 0])
        row["reference_only"] = policy == "equal_weight_annual"
    target_records.extend(rows)
    if policy != "equal_weight_annual":
        own = [p for p in profiles if p["horizon"] == horizon]
        rows = fixed.client_rows({horizon: fields}, fitted, own, ids, context=context, scope=scope, fold=fold, policy=policy)
        lookup = {p["name"]: col for col, p in enumerate(fitted["menu"])}
        for i, row in enumerate(rows):
            row["mesh_step"] = mesh
            row["initial_trade"] = bool(simulation["trades"][i % len(ids), lookup[row["assignment"]], 0]) if row["assigned"] else None
        client_records.extend(rows)


def comparisons_to_annual(frame, keys, *, clients=False):
    reference = frame.loc[frame.policy == "annual_restore"].copy()
    alternatives = frame.loc[~frame.policy.isin(["annual_restore", "equal_weight_annual"])]
    rows = []
    for (policy, mesh), group in alternatives.groupby(["policy", "mesh_step"], sort=False, dropna=False):
        pair = pd.concat([reference, group.assign(policy="initial_then_hold")], ignore_index=True)
        diff = fixed.paired_differences(pair, keys, clients=clients)
        diff["comparison_policy"] = policy
        diff["mesh_step"] = mesh
        rows.append(diff)
    return pd.concat(rows, ignore_index=True)


def summarize_with_initial(frame, group_keys, *, clients=False):
    summary = fixed.summarize(frame, group_keys, clients=clients)
    rows = []
    for keys, group in frame.groupby(group_keys, sort=False, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        selected = group.loc[group.assigned] if clients else group
        rows.append({**dict(zip(group_keys, keys)),
                     "initial_trade_count": int(selected.initial_trade.sum()) if len(selected) else None,
                     "initial_trade_fraction": float(selected.initial_trade.mean()) if len(selected) else None})
    return summary.merge(pd.DataFrame(rows), on=group_keys, validate="one_to_one")


def resolution_diagnostic(target_frame, trace_archive, trace_index, initial_values):
    coarse = target_frame.loc[(target_frame.mesh_step == .05) & target_frame.policy.isin(DYNAMIC)]
    fine = target_frame.loc[(target_frame.mesh_step == .025) & target_frame.policy.isin(DYNAMIC)]
    keys = ["context", "scope", "fold", "scenario_id", "portfolio", "candidate_index", "policy", "horizon"]
    joined = coarse.merge(fine, on=keys, suffixes=("_coarse", "_fine"), validate="one_to_one")
    for field in fixed.VALUE_FIELDS:
        joined[f"fine_minus_coarse_{field}"] = joined[f"{field}_fine"] - joined[f"{field}_coarse"]
    summaries = []
    for (context, policy, horizon), _ in joined.groupby(["context", "policy", "horizon"], sort=False):
        left = trace_index[(context, .05, horizon, policy)]
        right = trace_index[(context, .025, horizon, policy)]
        u_left = trace_archive[f"{left}__pretrade_weights"]
        u_right = trace_archive[f"{right}__pretrade_weights"]
        common = np.all(u_left == u_right, axis=-1)
        disagreement = trace_archive[f"{left}__trade"] != trace_archive[f"{right}__trade"]
        for column in range(u_left.shape[1]):
            mask = common[:, column]
            summaries.append({"context": context, "policy": policy, "horizon": horizon, "target_column": column,
                              "path_year_decisions": int(mask.size), "exact_common_visited_states": int(mask.sum()),
                              "action_disagreements_on_exact_common_states": int((disagreement[:, column] & mask).sum()),
                              "action_disagreements_all_paired_path_years": int(disagreement[:, column].sum()),
                              "initial_decision_h": 1 if policy == "myopic_gate" else horizon,
                              "initial_approx_decision_nominal_log_factor_coarse": initial_values[(context, .05, horizon, policy)][column],
                              "initial_approx_decision_nominal_log_factor_fine": initial_values[(context, .025, horizon, policy)][column]})
        if policy == "myopic_gate":
            if np.any(disagreement):
                raise RuntimeError("Exact H=1 myopic actions changed with future-value mesh")
            group = joined.loc[(joined.context == context) & (joined.policy == policy) & (joined.horizon == horizon)]
            for field in fixed.VALUE_FIELDS:
                np.testing.assert_allclose(group[f"{field}_fine"], group[f"{field}_coarse"], atol=1e-10, rtol=1e-12)
    return joined, pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    sources, inputs = source_manifest(), input_manifest()
    protocol = json.loads(PROTOCOL.read_text())
    research_protocol = json.loads(primary.PROTOCOL.read_text())
    tree_audit = json.loads(TREE_AUDIT.read_text())
    if not tree_audit["passed"] or tree_audit["market_data_loaded"]:
        raise RuntimeError("A passed independent synthetic tree audit is required")
    for name in ["research/experimental_execution/finite_horizon.py", "src/safeinvest/core.py"]:
        if (tree_audit["source_sha256_start"][name] != fixed.sha(ROOT / name)
                or tree_audit["source_sha256_end"][name] != fixed.sha(ROOT / name)):
            raise RuntimeError("Production accounting/solver differs from the independently audited version")
    if (protocol["initial_fee_rate"] != .0025 or protocol["numerics"]["state_mesh_primary_step"] != .05
            or protocol["numerics"]["state_mesh_check_step"] != .025 or args.chunk_size < 1):
        raise ValueError("Prototype must retain the declared cost and both state resolutions")
    primary.validate_folds(research_protocol)
    arrays, cache = primary.get_cache(research_protocol, step=.05, cost=.0025)
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)) or not np.array_equal(ids, arrays["scenario_ids"]):
        raise RuntimeError("Only the guarded development IDs 1--120 are allowed")
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    ablation_manifest = json.loads((BASE / "manifest.json").read_text())
    if digest.hexdigest() != ablation_manifest["development_arrays_sha256"]:
        raise RuntimeError("Executed market arrays differ from the frozen ablation inputs")
    full = primary.compatible_full_fit()[fixed.CASE]
    folds = json.loads((primary.OUT / "validation_fold_menus.json").read_text())[fixed.CASE]
    validation_manifest = json.loads((primary.OUT / "validation_manifest.json").read_text())
    if validation_manifest["sources"] != primary.source_manifest():
        raise RuntimeError("Frozen fold fits have a different source snapshot")
    contexts = [("full_development", "full", None, np.arange(120), np.arange(120), full)]
    for fitted, declared in zip(folds, research_protocol["validation"]["folds"], strict=True):
        if fitted["fold"] != declared["fold"] or fitted["train_ids"] != declared["train_ids"] or fitted["test_ids"] != declared["test_ids"]:
            raise RuntimeError("Frozen fold does not match the declared training/evaluation split")
        contexts.append((f"fold_{fitted['fold']}_test", "oof", fitted["fold"], np.asarray(fitted["train_ids"]) - 1,
                         np.asarray(fitted["test_ids"]) - 1, fitted))
    target_records, client_records, trace_summaries, checks, models = [], [], [], [], []
    traces, trace_index, wealth_paths, tables, initial_values = {}, {}, {}, {}, {}
    for context, scope, fold_number, train, test, fitted in contexts:
        if (fitted["alpha"] != .10 or fitted["liquidity_floors"] != research_protocol["liquidity_cases"]["risk_only"]
                or len(fitted["menu"]) != 3 or (scope == "oof" and np.intersect1d(train, test).size)):
            raise ValueError("Require the frozen primary alpha10/risk_only three-target context")
        targets = np.asarray([p["weights"] for p in fitted["menu"]]); candidates = [p["candidate_index"] for p in fitted["menu"]]
        np.testing.assert_allclose(targets, arrays["weights"][candidates], atol=1e-14, rtol=0)
        baselines = {policy: fixed.simulate_execution(returns[test], inflation[test], targets, policy) for policy in fixed.POLICIES}
        fixed.verify_hold_closed_form(baselines["initial_then_hold"], returns[test], inflation[test], .0025)
        with np.load(BASE / "executed_paths.npz") as archive:
            for policy, simulation in baselines.items():
                for field in ["nominal_wealth", "real_wealth", "fees", "moved", "trades", "end_weights", "end_year_tv_drift"]:
                    np.testing.assert_allclose(simulation[field], archive[f"{context}__{policy}__{field}"], atol=0, rtol=0)
        equal = np.full((1, 3), 1 / 3)
        equal_id = int(np.flatnonzero(np.all(np.abs(arrays["weights"] - equal) <= 1e-12, axis=1))[0])
        equal_fit = {"menu": [{"name": "EqualWeightReference", "candidate_index": equal_id, "weights": equal[0].tolist()}]}
        baselines["equal_weight_annual"] = fixed.simulate_execution(returns[test], inflation[test], equal, "annual_restore")
        for policy, simulation in baselines.items():
            menu_fit = equal_fit if policy == "equal_weight_annual" else fitted
            for horizon in HORIZONS:
                if policy in ["annual_restore", "equal_weight_annual"]:
                    errors = fixed.verify_annual_cache(fixed.horizon_values(simulation, horizon), arrays, cache, test,
                                                       [equal_id] if policy == "equal_weight_annual" else candidates, horizon)
                    checks.append({"context": context, "policy": policy, "horizon": horizon, "cache_max_abs_error": errors})
                add_rows(target_records, client_records, simulation, menu_fit, research_protocol["clients"], ids[test],
                         context=context, scope=scope, fold=fold_number, policy=policy, mesh=None, horizon=horizon)
            for field in ["real_wealth", "nominal_wealth", "end_weights", "fees", "moved", "trades"]:
                wealth_paths[f"{context}__{policy}__{field}"] = simulation[field]
        for mesh in MESHES:
            model_started = time.perf_counter()
            solution = solve_finite_horizon(1 + returns[train], targets, horizon=30, step=mesh, cost_rate=.0025, chunk_size=args.chunk_size)
            prefix = f"{context}__mesh_{int(round(mesh * 1000)):03d}"
            for field in ["V", "tradebase", "L_grid", "L_target", "targets", "gross_samples"]:
                tables[f"{prefix}__{field}"] = getattr(solution, field)
            tables[f"{prefix}__mesh_states"] = solution.mesh.states
            train_hash = hashlib.sha256(np.ascontiguousarray(1 + returns[train]).tobytes()).hexdigest()
            model_metadata = {"context": context, "scope": scope, "mesh_step": mesh, "train_ids": ids[train].tolist(),
                              "evaluation_ids": ids[test].tolist(), "training_joint_gross_sha256": train_hash,
                              "targets": fitted["menu"], "frozen_assignments": fitted["assignments"], **solution.metadata}
            for horizon in HORIZONS:
                for policy in DYNAMIC:
                    simulation, trace = simulate_gate(returns[test], inflation[test], solution, horizon, policy, chunk_size=args.chunk_size)
                    trace_prefix = f"{prefix}__H{horizon}__{policy}"
                    trace_index[(context, mesh, horizon, policy)] = trace_prefix
                    for key, value in trace.items(): traces[f"{trace_prefix}__{key}"] = value
                    for key in ["fees", "moved", "fee_fraction", "moved_fraction", "trades", "end_year_tv_drift"]:
                        traces[f"{trace_prefix}__{key}"] = simulation[key]
                    traces[f"{trace_prefix}__scenario_ids"] = ids[test]
                    for field in ["real_wealth", "nominal_wealth", "end_weights"]:
                        wealth_paths[f"{trace_prefix}__{field}"] = simulation[field]
                    initial = np.maximum(trace["hold_q"][0, :, 0], trace["trade_q"][0, :, 0])
                    initial_values[(context, mesh, horizon, policy)] = initial.tolist()
                    add_rows(target_records, client_records, simulation, fitted, research_protocol["clients"], ids[test],
                             context=context, scope=scope, fold=fold_number, policy=policy, mesh=mesh, horizon=horizon)
                    for column, item in enumerate(fitted["menu"]):
                        trade = trace["trade"][:, column]
                        trace_summaries.append({"context": context, "mesh_step": mesh, "horizon": horizon, "policy": policy,
                                                "portfolio": item["name"], "candidate_index": item["candidate_index"],
                                                "path_count": len(test), "decision_count": int(trade.size),
                                                "initial_trade_count": int(trade[:, 0].sum()), "trade_count": int(trade.sum()),
                                                "mean_modeled_continuation_gain": float(trace["gain"][:, column].mean()),
                                                "mean_one_period_restore_gain": float(trace["one_period_gain"][:, column].mean()),
                                                "mean_restore_realized_paired_gain": float(trace["restore_realized_paired_gain"][:, column].mean()),
                                                "mean_selected_action_realized_gain": float(trace["selected_action_realized_gain"][:, column].mean()),
                                                "initial_decision_h": 1 if policy == "myopic_gate" else horizon,
                                                "initial_approx_decision_nominal_log_factor": float(initial[column])})
            model_metadata["solve_and_simulate_seconds"] = time.perf_counter() - model_started
            models.append(model_metadata)
            print(json.dumps({"context": context, "mesh_step": mesh, "seconds": model_metadata["solve_and_simulate_seconds"],
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    target_frame, client_frame = pd.DataFrame(target_records), pd.DataFrame(client_records)
    original = pd.read_csv(primary.OUT / "validation_oof_paths.csv")
    oof_check = fixed.verify_original_oof(client_frame, original)
    target_keys = ["context", "scope", "fold", "scenario_id", "portfolio", "candidate_index", "horizon"]
    client_keys = ["context", "scope", "fold", "scenario_id", "client", "assignment", "candidate_index", "horizon", "loss_tolerance"]
    target_paired = comparisons_to_annual(target_frame, target_keys)
    client_paired = comparisons_to_annual(client_frame, client_keys, clients=True)
    resolution_paths, resolution_actions = resolution_diagnostic(target_frame, traces, trace_index, initial_values)
    outputs = {"target_path_metrics.csv": target_frame, "client_path_metrics.csv": client_frame,
               "target_summary.csv": summarize_with_initial(target_frame, ["context", "scope", "fold", "mesh_step", "policy", "portfolio", "candidate_index", "horizon"]),
               "client_summary.csv": summarize_with_initial(client_frame, ["scope", "mesh_step", "policy", "client", "horizon", "loss_tolerance"], clients=True),
               "paired_target_path_differences.csv": target_paired,
               "paired_target_summary.csv": fixed.summarize_paired(target_paired, ["context", "scope", "fold", "comparison_policy", "mesh_step", "portfolio", "candidate_index", "horizon"]),
               "paired_client_path_differences.csv": client_paired,
               "paired_client_summary.csv": fixed.summarize_paired(client_paired, ["scope", "comparison_policy", "mesh_step", "client", "horizon", "loss_tolerance"], clients=True),
               "action_summary.csv": pd.DataFrame(trace_summaries),
               "resolution_paired_paths.csv": resolution_paths, "resolution_action_comparison.csv": resolution_actions}
    if source_manifest() != sources or input_manifest() != inputs:
        raise RuntimeError("Sources or frozen inputs changed during the run; no result was written")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items(): frame.to_csv(args.out / name, index=False)
    np.savez_compressed(args.out / "action_traces.npz", **traces)
    np.savez_compressed(args.out / "executed_wealth.npz", **wealth_paths)
    np.savez_compressed(args.out / "solution_tables.npz", **tables)
    primary.write_json(args.out / "model_metadata.json", models)
    primary.write_json(args.out / "accounting_checks.json", {"cache_checks": checks, "original_oof": oof_check,
                                                               "annual_and_hold_match_fixed_ablation": True,
                                                               "myopic_mesh_invariance": True})
    files = [*outputs, "action_traces.npz", "executed_wealth.npz", "solution_tables.npz", "model_metadata.json", "accounting_checks.json"]
    sources_end, inputs_end = source_manifest(), input_manifest()
    if sources_end != sources or inputs_end != inputs:
        raise RuntimeError("Sources or frozen inputs changed during output writing; no completion manifest was written")
    primary.write_json(args.out / "manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "frozen_input_hashes": inputs, "development_arrays_sha256": digest.hexdigest(),
        "sources_at_end": sources_end, "frozen_input_hashes_at_end": inputs_end,
        "scenario_ids": ids.tolist(), "final_scenarios_loaded": False,
        "outputs_sha256": {name: fixed.sha(args.out / name) for name in files}, "protocol": protocol,
        "independent_synthetic_tree_audit": {"path": str(TREE_AUDIT.relative_to(ROOT)), "sha256": fixed.sha(TREE_AUDIT),
                                              "passed": True, "scope": "Small synthetic numerical audit; not a global error bound, lower bound or market-risk certificate"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "scope": "Exploratory frozen-target execution comparison; no menu/assignment/risk-budget refitting, winner selection or static finite-family CP guarantee for the learned dynamic rule",
        "referrals": "Original Review retains null financial outcomes and coverage denominator; no equal-weight fallback. Equal-weight annual is target-level reference only.",
        "forecast": "Full 120 or fold 96 training scenarios, every original joint annual gross; equal original-scenario and year weights. Annual support points are not independent inferential observations.",
        "true_horizon": "Each H simulated afresh from initial all-bill 100; finite decision_h=H-t, myopic decision_h=1. Initial gate is optional, fixed annual/hold initial allocation mandatory.",
        "deduplication": "Only exactly repeated (actual u,h,target ID) rows are deduplicated. No state projection, quantization or grid-Q interpolation in online actions.",
        "trace_axes": "Each dynamic trace=(scenario,target,year1..H), pretrade_weights adds asset axis. IDs identify each evaluation context. fee_fraction is actual, hypothetical_fee_fraction is the cost of restoring whether selected or not.",
        "trace_gains": "gain=net modeled Qtrade-Qhold; one_period_gain=exact empirical restore-vs-hold expected net log increment. restore_realized_paired_gain is computed after the action using realized gross, with fee once. selected_action_realized_gain is zero if hold is selected. No gain is a guaranteed future benefit.",
        "model_value_units": "Initial approximate decision V/Q factors are nominal log-wealth factors under the annual mixture, not exact realized gains or lower bounds. initial_decision_h=H for finite horizon, but 1 for myopic: the latter is not an H-period policy forecast. Realized performance uses net real wealth and client H. Common inflation cancels from paired action differences, not realized risk.",
        "resolution": "Both meshes retained. Common visited states mean exactly identical numerical u on paired trajectories; all-path-year action disagreement also includes trajectory drift. Neither statistic proves a continuous-state error bound.",
        "pairing": "Alternative minus annual at each original path and frozen target before aggregation; no bootstrap confidence or OOF pooled certificate.",
        "cost_units": "Cumulative nominal fees/moved are accounting amounts. Annual fee/moved fractions use pretrade wealth; moved_to_initial_wealth uses fixed100. These are not return percentage points.",
    })
    print(f"Wrote finite-horizon prototype to {args.out}", flush=True)


if __name__ == "__main__":
    main()
