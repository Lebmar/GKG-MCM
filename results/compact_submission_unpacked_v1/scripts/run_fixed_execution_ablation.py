"""Paired execution ablation of frozen targets; no selection or reassignment.

Only the primary alpha10/risk_only full and recorded fold targets are used.
The annual branch must reproduce the guarded static cache and assigned OOF
records before any output is written. No final paths or forecast models enter.
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
import run_risk_menu_research as primary
from safeinvest.core import execute_target
from safeinvest.data import load_paths
from safeinvest_research.metrics import NUMERIC_TOL, SUMMARY_METADATA, discrete_cvar, path_metrics

PROTOCOL = ROOT / "docs/fixed_execution_ablation_protocol.json"
OUT = ROOT / "results/model_review/fixed_execution_ablation_v1"
CASE = "alpha_10__risk_only"
POLICIES = ("annual_restore", "initial_then_hold")
METRIC_FIELDS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNT_FIELDS = ("nominal_fees", "nominal_moved", "moved_to_initial_wealth", "mean_annual_fee_fraction",
                  "mean_annual_moved_fraction", "trade_fraction", "target_execution_fraction",
                  "max_end_year_tv_drift", "terminal_equity", "terminal_bond", "terminal_bill")
VALUE_FIELDS = METRIC_FIELDS + ACCOUNT_FIELDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    names = ["src/safeinvest/core.py", "src/safeinvest/data.py",
             "research/safeinvest_research/metrics.py", "research/safeinvest_research/static.py",
             "research/safeinvest_research/selection.py", "scripts/run_risk_menu_research.py",
             "docs/risk_menu_development_protocol.json", str(PROTOCOL.relative_to(ROOT)),
             str(Path(__file__).resolve().relative_to(ROOT))]
    return {name: sha(ROOT / name) for name in names}


def input_manifest():
    files = ["full_development_menus.json", "fit_manifest.json", "validation_fold_menus.json",
             "validation_manifest.json", "validation_oof_paths.csv"]
    return {str((primary.OUT / file).relative_to(ROOT)): sha(primary.OUT / file) for file in files}


def simulate_execution(returns, inflation, targets, policy, cost_rate=.0025):
    """Actual annual holdings with a mandatory common initial allocation.

    Time-zero wealth is the original pre-trade 100. Fraction denominators are
    the actual wealth immediately before that year's execution. End-year TV
    drift is distinct from the next pre-trade decision's drift.
    """
    market, infl, weights = map(lambda x: np.asarray(x, float), (returns, inflation, targets))
    if weights.shape == (3,):
        weights = weights[None]
    if (market.ndim != 3 or market.shape[-1] != 3 or not market.shape[0] or not market.shape[1]
            or infl.shape != market.shape[:2] or not np.all(np.isfinite(market))
            or not np.all(np.isfinite(infl)) or np.any(market <= -1) or np.any(infl <= -1)):
        raise ValueError("Require finite (n,T,3) returns and matching inflation with positive gross")
    if (weights.ndim != 2 or weights.shape[1] != 3 or not len(weights)
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=NUMERIC_TOL)):
        raise ValueError("Require fully invested nonnegative target weights")
    if policy not in POLICIES:
        raise ValueError("Unknown fixed execution policy")
    if isinstance(cost_rate, (bool, np.bool_)) or not np.isscalar(cost_rate) or not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("cost_rate must lie in [0,1)")
    weights = weights / weights.sum(axis=1, keepdims=True)
    n, years = market.shape[:2]
    shape = (n, len(weights), years)
    values = {key: np.zeros(shape) for key in ["fees", "moved", "pretrade_wealth", "fee_fraction",
                                              "moved_fraction", "pretrade_tv_drift", "end_year_tv_drift"]}
    values["target_execution"] = np.zeros(shape, dtype=bool)
    values["end_weights"] = np.zeros(shape + (3,))
    values["nominal_wealth"] = np.full(shape[:2] + (years + 1,), 100.)
    holdings = np.zeros(shape[:2] + (3,))
    holdings[..., 2] = 100.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for year in range(years):
            wealth = holdings.sum(axis=-1)
            values["pretrade_wealth"][..., year] = wealth
            values["pretrade_tv_drift"][..., year] = .5 * np.abs(holdings / wealth[..., None] - weights).sum(axis=-1)
            if year == 0 or policy == "annual_restore":
                holdings, fee, moved = execute_target(holdings, weights[None], cost_rate)
                values["fees"][..., year] = fee
                values["moved"][..., year] = moved
                values["fee_fraction"][..., year] = fee / wealth
                values["moved_fraction"][..., year] = moved / wealth
                values["target_execution"][..., year] = True
            holdings = holdings * (1 + market[:, None, year, :])
            current = holdings.sum(axis=-1)
            values["nominal_wealth"][..., year + 1] = current
            end_weights = holdings / current[..., None]
            values["end_weights"][..., year, :] = end_weights
            values["end_year_tv_drift"][..., year] = .5 * np.abs(end_weights - weights).sum(axis=-1)
        prices = np.column_stack((np.ones(n), np.cumprod(1 + infl, axis=1)))
        values["real_wealth"] = values["nominal_wealth"] / prices[:, None, :]
    if not np.all(np.isfinite(values["real_wealth"])) or np.any(values["real_wealth"] <= 0):
        raise ValueError("Executed wealth exceeds numerical range")
    values["trades"] = values["moved"] > NUMERIC_TOL
    values["targets"] = weights
    return values


def verify_hold_closed_form(simulation, returns, inflation, cost_rate):
    weights = simulation["targets"]
    initial = np.array([0., 0., 100.])
    post, fees, moved = execute_target(initial, weights, cost_rate)
    gross = np.cumprod(1 + np.asarray(returns), axis=1)
    expected = np.sum(post[None, :, None, :] * gross[:, None, :, :], axis=-1)
    nominal = np.concatenate((np.full(expected.shape[:2] + (1,), 100.), expected), axis=-1)
    prices = np.column_stack((np.ones(len(inflation)), np.cumprod(1 + inflation, axis=1)))
    np.testing.assert_allclose(simulation["nominal_wealth"], nominal, atol=1e-10, rtol=3e-14)
    np.testing.assert_allclose(simulation["real_wealth"], nominal / prices[:, None, :], atol=1e-10, rtol=3e-14)
    np.testing.assert_allclose(simulation["fees"][..., 0], np.broadcast_to(fees, simulation["fees"][..., 0].shape), atol=1e-12, rtol=0)
    np.testing.assert_allclose(simulation["moved"][..., 0], np.broadcast_to(moved, simulation["moved"][..., 0].shape), atol=1e-12, rtol=0)
    if np.any(simulation["fees"][..., 1:]) or np.any(simulation["moved"][..., 1:]):
        raise RuntimeError("Hold branch transferred wealth after initial allocation")
    return float(np.max(np.abs(simulation["nominal_wealth"] - nominal)))


def horizon_values(simulation, horizon):
    fields = path_metrics(simulation["real_wealth"][..., :horizon + 1])
    end = simulation["end_weights"][..., horizon - 1, :]
    fields.update({"nominal_fees": simulation["fees"][..., :horizon].sum(axis=-1),
                   "nominal_moved": simulation["moved"][..., :horizon].sum(axis=-1),
                   "moved_to_initial_wealth": simulation["moved"][..., :horizon].sum(axis=-1) / 100.,
                   "mean_annual_fee_fraction": simulation["fee_fraction"][..., :horizon].mean(axis=-1),
                   "mean_annual_moved_fraction": simulation["moved_fraction"][..., :horizon].mean(axis=-1),
                   "trade_fraction": simulation["trades"][..., :horizon].mean(axis=-1),
                   "target_execution_fraction": simulation["target_execution"][..., :horizon].mean(axis=-1),
                   "max_end_year_tv_drift": simulation["end_year_tv_drift"][..., :horizon].max(axis=-1),
                   "terminal_equity": end[..., 0], "terminal_bond": end[..., 1], "terminal_bill": end[..., 2]})
    return fields


def verify_annual_cache(fields, arrays, cached_metrics, row_indices, candidate_indices, horizon):
    ix = np.ix_(row_indices, candidate_indices)
    error = {}
    for name in METRIC_FIELDS:
        reference = cached_metrics[horizon][name][ix]
        np.testing.assert_allclose(fields[name], reference, atol=1e-10, rtol=3e-13)
        error[name] = float(np.max(np.abs(fields[name] - reference)))
    for name, cache_name in [("nominal_fees", "fees"), ("nominal_moved", "moved"), ("trade_fraction", "trade_fraction")]:
        reference = arrays[f"H{horizon}__{cache_name}"][ix]
        np.testing.assert_allclose(fields[name], reference, atol=1e-10, rtol=3e-13)
        error[name] = float(np.max(np.abs(fields[name] - reference)))
    return error


def target_rows(fields, menu, scenario_ids, *, context, scope, fold, policy, horizon):
    result = []
    for row, scenario in enumerate(scenario_ids):
        for column, item in enumerate(menu):
            result.append({"context": context, "scope": scope, "fold": fold, "scenario_id": int(scenario),
                           "portfolio": item["name"], "candidate_index": item["candidate_index"],
                           "w_equity": item["weights"][0], "w_bond": item["weights"][1], "w_bill": item["weights"][2],
                           "policy": policy, "horizon": horizon,
                           **{key: float(fields[key][row, column]) for key in VALUE_FIELDS}})
    return result


def client_rows(by_horizon, fit, profiles, scenario_ids, *, context, scope, fold, policy):
    result = []
    for profile in profiles:
        assigned = fit["assignments"][profile["name"]]
        matches = [(i, item) for i, item in enumerate(fit["menu"]) if item["name"] == assigned]
        if assigned != "Review" and len(matches) != 1:
            raise ValueError("Frozen assignment does not identify exactly one target")
        selected = matches[0] if matches else None
        fields = by_horizon[profile["horizon"]]
        for row, scenario in enumerate(scenario_ids):
            item = {"context": context, "scope": scope, "fold": fold, "scenario_id": int(scenario),
                    "client": profile["name"], "assignment": assigned, "assigned": selected is not None,
                    "candidate_index": selected[1]["candidate_index"] if selected else None,
                    "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"], "policy": policy,
                    **{key: None for key in VALUE_FIELDS}, "breach": None, "excess": None}
            if selected:
                column = selected[0]
                item.update({key: float(fields[key][row, column]) for key in VALUE_FIELDS})
                breached = item["mdd"] > profile["loss_tolerance"] + NUMERIC_TOL
                item["breach"] = bool(breached)
                item["excess"] = item["mdd"] - profile["loss_tolerance"] if breached else 0.
            result.append(item)
    return result


def summarize(frame, group_keys, *, clients=False):
    rows = []
    for keys, group in frame.groupby(group_keys, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        selected = group.loc[group.assigned] if clients else group
        row.update({"total_paths": len(group), "assigned_paths": len(selected),
                    "review_paths": len(group) - len(selected)})
        names = ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd", "median_volatility"]
        row.update({key: None for key in names})
        row.update({f"mean_{key}": None for key in ACCOUNT_FIELDS})
        if clients:
            row.update({"breach_count": None, "breach_probability": None, "expected_excess": None})
        if len(selected):
            row.update({"mean_log_growth": float(selected.log_growth.mean()),
                        "median_cagr": float(selected.cagr.median()),
                        "p10_cagr": float(np.quantile(selected.cagr, .10, method="inverted_cdf")),
                        "median_mdd": float(selected.mdd.median()),
                        "p90_mdd": float(np.quantile(selected.mdd, .90, method="inverted_cdf")),
                        "cvar90_mdd": float(discrete_cvar(selected.mdd.to_numpy())),
                        "median_volatility": float(selected.volatility.median()),
                        **{f"mean_{key}": float(selected[key].mean()) for key in ACCOUNT_FIELDS}})
            if clients:
                count = int(selected.breach.astype(bool).sum())
                row.update({"breach_count": count, "breach_probability": count / len(selected),
                            "expected_excess": float(selected.excess.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(frame, keys, *, clients=False):
    annual = frame.loc[frame.policy == "annual_restore"]
    hold = frame.loc[frame.policy == "initial_then_hold"]
    merged = annual.merge(hold, on=keys, suffixes=("_annual", "_hold"), validate="one_to_one")
    if len(merged) != len(annual) or len(annual) != len(hold):
        raise ValueError("Execution branches do not contain the same paired paths")
    result = merged[keys].copy()
    assigned = merged.assigned_annual.to_numpy() if clients else np.ones(len(merged), dtype=bool)
    if clients and not np.array_equal(merged.assigned_annual, merged.assigned_hold):
        raise ValueError("Execution ablation changed a frozen assignment")
    result["assigned"] = assigned
    for field in VALUE_FIELDS + (("excess",) if clients else ()):
        result[f"delta_{field}"] = merged[f"{field}_hold"] - merged[f"{field}_annual"]
    if clients:
        annual_breach = merged.breach_annual.eq(True)
        hold_breach = merged.breach_hold.eq(True)
        result["delta_breach"] = np.where(assigned, hold_breach.astype(int) - annual_breach.astype(int), np.nan)
        result["new_breach"] = np.where(assigned, hold_breach & ~annual_breach, np.nan)
        result["averted_breach"] = np.where(assigned, annual_breach & ~hold_breach, np.nan)
    return result


def summarize_paired(frame, group_keys, *, clients=False):
    result = []
    fields = [key for key in frame if key.startswith("delta_")]
    for keys, group in frame.groupby(group_keys, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        assigned = group.loc[group.assigned]
        row = {**dict(zip(group_keys, keys)), "total_pairs": len(group), "assigned_pairs": len(assigned),
               "review_pairs": len(group) - len(assigned)}
        for key in fields:
            values = assigned[key].to_numpy(float)
            row.update({f"mean_{key}": float(values.mean()) if len(values) else None,
                        f"median_{key}": float(np.median(values)) if len(values) else None,
                        f"p10_{key}": float(np.quantile(values, .10, method="inverted_cdf")) if len(values) else None,
                        f"p90_{key}": float(np.quantile(values, .90, method="inverted_cdf")) if len(values) else None})
        if clients:
            row["new_breach_count"] = int(assigned.new_breach.sum()) if len(assigned) else None
            row["averted_breach_count"] = int(assigned.averted_breach.sum()) if len(assigned) else None
        result.append(row)
    return pd.DataFrame(result)


def verify_original_oof(client_frame, original):
    selected = client_frame.loc[(client_frame.scope == "oof") & (client_frame.policy == "annual_restore")]
    reference = original.loc[original.case == CASE]
    keys = ["fold", "scenario_id", "client"]
    merged = selected.merge(reference, on=keys, suffixes=("_actual", "_reference"), validate="one_to_one")
    if len(merged) != 600 or len(selected) != 600 or len(reference) != 600:
        raise RuntimeError("Original OOF records do not cover all frozen client-path pairs")
    if not np.array_equal(merged.assigned_actual, merged.assigned_reference) or not np.array_equal(merged.assignment_actual, merged.assignment_reference):
        raise RuntimeError("A frozen OOF assignment changed")
    assigned = merged.loc[merged.assigned_actual]
    for field in ["candidate_index", "log_growth", "cagr", "mdd", "excess"]:
        np.testing.assert_allclose(assigned[f"{field}_actual"].astype(float), assigned[f"{field}_reference"].astype(float), atol=1e-12, rtol=1e-12)
    if not np.array_equal(assigned.breach_actual.astype(bool), assigned.breach_reference.astype(bool)):
        raise RuntimeError("Annual branch OOF breach event changed")
    return {"original_client_path_records": len(merged), "assigned_records": len(assigned),
            "review_records": int((~merged.assigned_actual).sum()), "annual_oof_matches": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    sources, inputs = source_manifest(), input_manifest()
    protocol = json.loads(PROTOCOL.read_text())
    development_protocol = json.loads(primary.PROTOCOL.read_text())
    if (protocol["data_scope"]["allowed_ids"] != list(range(1, 121))
            or protocol["data_scope"]["final_access"] is not False
            or protocol["accounting"]["cost_rate"] != .0025
            or protocol["accounting"]["initial_holdings"] != [0, 0, 100]
            or set(protocol["policies"]) != set(POLICIES)):
        raise ValueError("Ablation protocol differs from the declared accounting/data scope")
    primary.validate_folds(development_protocol)
    arrays, cache = primary.get_cache(development_protocol, step=.05, cost=.0025)
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if not np.array_equal(ids, arrays["scenario_ids"]) or not np.array_equal(ids, np.arange(1, 121)):
        raise RuntimeError("Guarded development inputs disagree")
    market_digest = hashlib.sha256()
    for value in (ids, returns, inflation):
        market_digest.update(np.ascontiguousarray(value).tobytes())
    cache_path = primary.OUT / "cache" / f"{primary.cache_key(.05, .0025)}.npz"
    cache_metadata = json.loads(cache_path.with_suffix(".json").read_text())
    if market_digest.hexdigest() != cache_metadata["development_arrays_sha256"]:
        raise RuntimeError("Cache and executed market arrays differ")
    full = primary.compatible_full_fit()[CASE]
    folds = json.loads((primary.OUT / "validation_fold_menus.json").read_text())[CASE]
    validation_manifest = json.loads((primary.OUT / "validation_manifest.json").read_text())
    if validation_manifest["sources"] != primary.source_manifest():
        raise RuntimeError("Frozen folds were produced by different research sources")
    contexts = [("full_development", "full", None, np.arange(120), full)]
    for fold, expected in zip(folds, development_protocol["validation"]["folds"], strict=True):
        if (fold["fold"] != expected["fold"] or fold["train_ids"] != expected["train_ids"]
                or fold["test_ids"] != expected["test_ids"] or fold["n_training"] != 96):
            raise RuntimeError("Recorded fold menu does not match the fixed 96/24 design")
        contexts.append((f"fold_{fold['fold']}_test", "oof", fold["fold"], np.asarray(fold["test_ids"]) - 1, fold))
    target_records, client_records, paths, checks, frozen = [], [], {}, [], []
    for context, scope, fold_number, indices, fitted in contexts:
        if (fitted["alpha"] != .10 or fitted.get("liquidity_case", "risk_only") != "risk_only"
                or fitted["liquidity_floors"] != development_protocol["liquidity_cases"]["risk_only"]
                or len(fitted["menu"]) != 3):
            raise ValueError("Require the three frozen primary alpha10/risk_only targets")
        menu = fitted["menu"]
        targets = np.asarray([item["weights"] for item in menu])
        candidates = [item["candidate_index"] for item in menu]
        np.testing.assert_allclose(targets, arrays["weights"][candidates], atol=1e-14, rtol=0)
        frozen.append({"context": context, "scenario_ids": ids[indices].tolist(), "menu": menu,
                       "assignments": fitted["assignments"]})
        simulations = {policy: simulate_execution(returns[indices], inflation[indices], targets, policy)
                       for policy in POLICIES}
        annual, hold = [simulations[p] for p in POLICIES]
        np.testing.assert_allclose(annual["fees"][..., 0], hold["fees"][..., 0], atol=0, rtol=0)
        np.testing.assert_allclose(annual["moved"][..., 0], hold["moved"][..., 0], atol=0, rtol=0)
        closed_error = verify_hold_closed_form(hold, returns[indices], inflation[indices], .0025)
        for policy, simulation in simulations.items():
            prefix = f"{context}__{policy}"
            paths[f"{prefix}__scenario_ids"] = ids[indices]
            for field, value in simulation.items():
                paths[f"{prefix}__{field}"] = value
            by_horizon = {h: horizon_values(simulation, h) for h in development_protocol["horizons"]}
            for horizon, fields in by_horizon.items():
                if policy == "annual_restore":
                    errors = verify_annual_cache(fields, arrays, cache, indices, candidates, horizon)
                    checks.append({"context": context, "horizon": horizon,
                                   "annual_vs_cache_max_abs_error": errors,
                                   "hold_closed_form_max_nominal_abs_error": closed_error})
                target_records.extend(target_rows(fields, menu, ids[indices], context=context, scope=scope,
                                                  fold=fold_number, policy=policy, horizon=horizon))
            client_records.extend(client_rows(by_horizon, fitted, development_protocol["clients"], ids[indices],
                                              context=context, scope=scope, fold=fold_number, policy=policy))
    target_frame, client_frame = pd.DataFrame(target_records), pd.DataFrame(client_records)
    oof_check = verify_original_oof(client_frame, pd.read_csv(primary.OUT / "validation_oof_paths.csv"))
    target_keys = ["context", "scope", "fold", "scenario_id", "portfolio", "candidate_index", "horizon"]
    client_keys = ["context", "scope", "fold", "scenario_id", "client", "assignment", "candidate_index", "horizon", "loss_tolerance"]
    target_paired = paired_differences(target_frame, target_keys)
    client_paired = paired_differences(client_frame, client_keys, clients=True)
    outputs = {
        "target_path_metrics.csv": target_frame,
        "target_summary.csv": summarize(target_frame, ["context", "scope", "fold", "portfolio", "candidate_index", "horizon", "policy"]),
        "client_path_metrics.csv": client_frame,
        "client_summary.csv": summarize(client_frame, ["scope", "client", "horizon", "loss_tolerance", "policy"], clients=True),
        "client_context_summary.csv": summarize(client_frame, ["context", "scope", "fold", "client", "horizon", "loss_tolerance", "policy"], clients=True),
        "paired_target_path_differences.csv": target_paired,
        "paired_target_summary.csv": summarize_paired(target_paired, ["context", "scope", "fold", "portfolio", "candidate_index", "horizon"]),
        "paired_client_path_differences.csv": client_paired,
        "paired_client_summary.csv": summarize_paired(client_paired, ["scope", "client", "horizon", "loss_tolerance"], clients=True),
    }
    if source_manifest() != sources or input_manifest() != inputs:
        raise RuntimeError("Sources or frozen fit inputs changed; no output was written")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out / name, index=False)
    np.savez_compressed(args.out / "executed_paths.npz", **paths)
    primary.write_json(args.out / "frozen_targets_assignments.json", frozen)
    primary.write_json(args.out / "accounting_checks.json", {"contexts": checks, "original_oof": oof_check,
                                                               "initial_fee_and_moved_equal": True,
                                                               "hold_has_no_subsequent_transfers": True})
    all_outputs = [*outputs, "executed_paths.npz", "frozen_targets_assignments.json", "accounting_checks.json"]
    primary.write_json(args.out / "manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "frozen_input_hashes": inputs,
        "development_arrays_sha256": market_digest.hexdigest(), "cache_sha256": sha(cache_path),
        "scenario_ids": ids.tolist(), "final_scenarios_loaded": False,
        "outputs_sha256": {name: sha(args.out / name) for name in all_outputs},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "protocol": protocol, "summary_metadata": SUMMARY_METADATA,
        "pairing": "initial_then_hold minus annual_restore at the individual original scenario/frozen target, then summarized; quantiles of differences are not differences of quantiles",
        "null_referrals": "Frozen Review is retained with null outcomes and counted in coverage, without invented adviser returns",
        "cost_units": "nominal_fees and nominal_moved are cumulative accounting amounts; moved_to_initial_wealth divides by fixed initial 100 and is not return. fee_fraction and moved_fraction divide each annual amount by its own pre-trade wealth. No terminal liquidation fee.",
        "npz_axes": "wealth=(scenario,target,year0..30); annual accounting/drift=(scenario,target,year1..30); end_weights adds final asset axis equity,bond,bill. scenario_ids and targets identify each context.",
        "decision_fraction": "target_execution_fraction counts scheduled execute_target calls, even if nothing is moved; trade_fraction counts actual positive transfers. Neither models human review frequency.",
        "drift": "max_end_year_tv_drift=max_t .5*sum_i abs(end_weight_t_i-target_i), using annual post-return weights through H; pretrade_tv_drift is stored separately",
        "scope": "Prespecified fixed-target execution ablation; full-development is descriptive/in-sample, OOF is paired internal procedure comparison with overlapping training fits; no winner, reassignment, gate, bootstrap confidence or risk certificate selected",
    })
    print(f"Wrote fixed execution ablation to {args.out}", flush=True)


if __name__ == "__main__":
    main()
