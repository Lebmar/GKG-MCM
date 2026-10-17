"""Development-only static frontiers, common menus and internal validation.

The experiment is declared in docs/risk_menu_development_protocol.json. All
market access uses load_paths(..., range(1,121), allow_final=False). Static
candidate paths may be cached, but only training indices enter a fold's menu
or assignment. This script never opens the V1 final-result directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from safeinvest.data import load_paths
from safeinvest_research.metrics import SUMMARY_METADATA, NUMERIC_TOL, path_metrics, summarize_paths, empirical_risk
from safeinvest_research.selection import fit_menu
from safeinvest_research.static import simplex_grid, simulate_annual_targets

PROTOCOL = ROOT / "docs/risk_menu_development_protocol.json"
OUT = ROOT / "results/model_review/risk_menu_v1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest():
    sources = sorted((ROOT / "research/safeinvest_research").glob("*.py"))
    sources += [Path(__file__).resolve(), PROTOCOL]
    return {str(p.relative_to(ROOT)): sha(p) for p in sources}


def cache_key(step, cost):
    return f"step_{round(step * 10000):04d}__cost_{round(cost * 1000000):06d}"


def get_cache(protocol, step=0.05, cost=0.0025):
    """Cache pre-fixed candidate outcomes, never learned forecasts or menus."""
    key = cache_key(step, cost)
    file = OUT / "cache" / f"{key}.npz"
    metadata = file.with_suffix(".json")
    required_sources = {name: sha(ROOT / name) for name in [
        "research/safeinvest_research/static.py", "research/safeinvest_research/metrics.py",
        "src/safeinvest/core.py", "src/safeinvest/data.py"]}
    if file.exists() and metadata.exists():
        manifest = json.loads(metadata.read_text())
        if manifest["source_hashes"] != required_sources:
            raise RuntimeError("Cached numerical sources changed; preserve old cache and use a new research directory")
        if manifest["npz_sha256"] != sha(file):
            raise RuntimeError("Research cache hash mismatch")
        with np.load(file) as archive:
            arrays = {k: archive[k] for k in archive.files}
    else:
        returns, inflation, ids = load_paths(
            ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
        weights = simplex_grid(step)
        simulation = simulate_annual_targets(returns, inflation, weights, cost_rate=cost)
        arrays = {"scenario_ids": ids, "weights": weights}
        for horizon in protocol["horizons"]:
            fields = path_metrics(simulation["real_wealth"][..., :horizon + 1])
            arrays.update({f"H{horizon}__{field}": value for field, value in fields.items()})
            arrays[f"H{horizon}__fees"] = simulation["fees"][..., :horizon].sum(axis=-1)
            arrays[f"H{horizon}__moved"] = simulation["moved"][..., :horizon].sum(axis=-1)
            arrays[f"H{horizon}__trade_fraction"] = simulation["trades"][..., :horizon].mean(axis=-1)
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(file, **arrays)
        development_hash = hashlib.sha256()
        for value in (ids, returns, inflation):
            development_hash.update(np.ascontiguousarray(value).tobytes())
        write_json(metadata, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_ids": ids.tolist(), "step": step, "cost_rate": cost,
            "candidate_count": len(weights), "source_hashes": required_sources,
            "development_arrays_sha256": development_hash.hexdigest(),
            "npz_sha256": sha(file), "summary_metadata": SUMMARY_METADATA,
            "scope": "Pre-fixed static candidate outcomes on development paths only; no fitted menu or final data.",
        })
    if not np.array_equal(arrays["scenario_ids"], np.arange(1, 121)):
        raise RuntimeError("Unexpected scenario IDs in development cache")
    metrics = {h: {name: arrays[f"H{h}__{name}"] for name in
                  ["log_growth", "cagr", "mdd", "volatility", "terminal_real"]}
               for h in protocol["horizons"]}
    return arrays, metrics


def case_id(alpha, liquidity):
    return f"alpha_{round(100 * alpha):02d}__{liquidity}"


def policy_cases(protocol):
    for alpha in protocol["alpha_cases"]:
        for liquidity, floors in protocol["liquidity_cases"].items():
            yield case_id(alpha, liquidity), alpha, liquidity, floors


def summarize_grid(protocol, arrays, metrics):
    rows, client_rows = [], []
    weights = arrays["weights"]
    for horizon, fields in metrics.items():
        summary = summarize_paths(fields)
        for i, target in enumerate(weights):
            rows.append({"candidate_index": i, "horizon": horizon,
                         "w_equity": target[0], "w_bond": target[1], "w_bill": target[2],
                         **{k: float(value[i]) for k, value in summary.items()},
                         "mean_fee_units": float(arrays[f"H{horizon}__fees"][:, i].mean()),
                         "mean_trade_fraction": float(arrays[f"H{horizon}__trade_fraction"][:, i].mean())})
    for profile in protocol["clients"]:
        summary = summarize_paths(metrics[profile["horizon"]], profile["loss_tolerance"])
        for i, target in enumerate(weights):
            client_rows.append({"client": profile["name"], "candidate_index": i,
                                "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                                "w_equity": target[0], "w_bond": target[1], "w_bill": target[2],
                                **{k: float(value[i]) for k, value in summary.items()}})
    pd.DataFrame(rows).to_csv(OUT / "grid_by_horizon.csv", index=False)
    pd.DataFrame(client_rows).to_csv(OUT / "grid_by_client.csv", index=False)


def run_fit(protocol, arrays, metrics):
    summarize_grid(protocol, arrays, metrics)
    results = {}
    start = time.perf_counter()
    for name, alpha, liquidity, floors in policy_cases(protocol):
        fit = fit_menu(metrics, arrays["weights"], protocol["clients"],
                       np.arange(120), alpha=alpha, liquidity_floors=floors)
        fit["liquidity_case"] = liquidity
        results[name] = fit
        print(json.dumps({"case": name, "assignments": fit["assignments"],
                          "objective": fit["selection"]["objective"],
                          "identified_products": len(fit["selection"]["identified_indices"]),
                          "useful_candidates": fit["selection"]["useful_candidate_count"],
                          "weights": [p["weights"] for p in fit["menu"]]}, ensure_ascii=False), flush=True)
    write_json(OUT / "full_development_menus.json", results)
    write_json(OUT / "fit_manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "sources": source_manifest(), "scenario_ids": arrays["scenario_ids"].tolist(),
        "scope": "Training/descriptive fit on all 120 development paths; not out-of-sample validation.",
    })
    return results


def run_validation(protocol, arrays, metrics):
    fold_results, records = {}, []
    start = time.perf_counter()
    for name, alpha, liquidity, floors in policy_cases(protocol):
        fold_results[name] = []
        for fold in protocol["validation"]["folds"]:
            train = np.asarray(fold["train_ids"]) - 1
            test = np.asarray(fold["test_ids"]) - 1
            if np.intersect1d(train, test).size or len(train) != 96 or len(test) != 24:
                raise RuntimeError("Invalid scenario-level folds")
            fitted = fit_menu(metrics, arrays["weights"], protocol["clients"], train,
                              alpha=alpha, liquidity_floors=floors)
            fitted["fold"] = fold["fold"]
            fitted["train_ids"] = fold["train_ids"]
            fitted["test_ids"] = fold["test_ids"]
            fold_results[name].append(fitted)
            for profile in protocol["clients"]:
                assigned = fitted["assignments"][profile["name"]]
                item = next((p for p in fitted["menu"] if p["name"] == assigned), None)
                for index in test:
                    row = {"case": name, "alpha": alpha, "liquidity_case": liquidity,
                           "fold": fold["fold"], "scenario_id": int(index + 1),
                           "client": profile["name"], "horizon": profile["horizon"],
                           "loss_tolerance": profile["loss_tolerance"], "assignment": assigned,
                           "assigned": item is not None, "candidate_index": None,
                           "cagr": None, "log_growth": None, "mdd": None,
                           "breach": None, "excess": None}
                    if item is not None:
                        candidate = item["candidate_index"]
                        fields = metrics[profile["horizon"]]
                        loss = float(fields["mdd"][index, candidate])
                        row.update({"candidate_index": candidate,
                                    "cagr": float(fields["cagr"][index, candidate]),
                                    "log_growth": float(fields["log_growth"][index, candidate]),
                                    "mdd": loss,
                                    "breach": bool(loss > profile["loss_tolerance"] + NUMERIC_TOL),
                                    "excess": max(loss - profile["loss_tolerance"], 0.0)})
                    records.append(row)
        print(json.dumps({"stage": "validation", "case": name,
                          "fold_assignments": [f["assignments"] for f in fold_results[name]]},
                         ensure_ascii=False), flush=True)
    write_json(OUT / "validation_fold_menus.json", fold_results)
    data = pd.DataFrame(records)
    data.to_csv(OUT / "validation_oof_paths.csv", index=False)
    summaries = []
    for (name, client), group in data.groupby(["case", "client"], sort=False):
        assigned = group.loc[group.assigned]
        alpha = float(group.alpha.iloc[0])
        summary = {"case": name, "client": client, "alpha": alpha,
                   "total_test_paths": len(group), "assigned_test_paths": len(assigned),
                   "review_test_paths": int((~group.assigned).sum()),
                   "auto_folds": int(group.groupby("fold").assigned.first().sum()),
                   "review_folds": int((~group.groupby("fold").assigned.first()).sum()),
                   "breach_count": None, "breach_probability": None,
                   "mean_log_growth": None, "median_cagr": None, "p10_cagr": None,
                   "median_mdd": None, "p90_mdd": None, "expected_excess": None,
                   "internal_calibration_warning": None}
        if len(assigned):
            count = int(assigned.breach.astype(bool).sum())
            summary.update({"breach_count": count, "breach_probability": count / len(assigned),
                            "mean_log_growth": float(assigned.log_growth.mean()),
                            "median_cagr": float(assigned.cagr.median()),
                            "p10_cagr": float(np.quantile(assigned.cagr, .10, method="inverted_cdf")),
                            "median_mdd": float(assigned.mdd.median()),
                            "p90_mdd": float(np.quantile(assigned.mdd, .90, method="inverted_cdf")),
                            "expected_excess": float(assigned.excess.mean()),
                            "internal_calibration_warning": count / len(assigned) > alpha + NUMERIC_TOL})
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(OUT / "validation_oof_summary.csv", index=False)
    write_json(OUT / "validation_manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "sources": source_manifest(), "split_unit": "complete scenario",
        "interpretation": "Overlapping training folds and different fitted policies; OOF counts are internal procedure diagnostics, not an iid fixed-rule certificate.",
        "referral_handling": "Referrals retained in coverage counts; investment outcomes are null for unassigned profiles. No adviser performance or bill-only fallback is invented.",
    })
    return fold_results, summaries


def run_bootstrap(protocol, arrays, metrics, full_fits, name=None, replicates=None):
    name = name or case_id(.10, "risk_only")
    base = full_fits[name]
    replicates = replicates or protocol["stability"]["bootstrap_replicates"]
    rng = np.random.default_rng(protocol["stability"]["bootstrap_seed"])
    samples = rng.integers(0, 120, size=(replicates, 120))
    rows, menus, assignments = [], [], []
    reference_weights = np.asarray([p["weights"] for p in base["menu"]])
    aligned = []
    start = time.perf_counter()
    for replicate, indices in enumerate(samples):
        fitted = fit_menu(metrics, arrays["weights"], protocol["clients"], indices,
                          alpha=base["alpha"], liquidity_floors=base["liquidity_floors"])
        rows.append({"replicate": replicate, "covered_clients": fitted["selection"]["objective"][0],
                     "max_growth_regret": fitted["selection"]["objective"][1],
                     "mean_growth_regret": fitted["selection"]["objective"][2],
                     "identified_cardinality": len(fitted["selection"]["identified_indices"]),
                     "menu_complete": fitted["completion"]["complete"],
                     "three_distinct_reference_risks": fitted["completion"]["three_distinct_reference_risks"]})
        weights = np.asarray([p["weights"] for p in fitted["menu"]])
        if len(weights) == len(reference_weights):
            order = min(permutations(range(len(weights))), key=lambda p:
                        np.abs(weights[list(p)] - reference_weights).sum())
            aligned.append(weights[list(order)].tolist())
        menus.append({"replicate": replicate,
                      "candidate_indices_by_risk_label": [p["candidate_index"] for p in fitted["menu"]],
                      "weights_by_risk_label": weights.tolist()})
        assignment = {"replicate": replicate, **fitted["assignments"]}
        for profile in protocol["clients"]:
            chosen = next((d for d in fitted["client_product_details"]
                           if d["client"] == profile["name"] and d["chosen"]), None)
            assignment[f"{profile['name']}_candidate_index"] = chosen["candidate_index"] if chosen else None
        assignments.append(assignment)
        if (replicate + 1) % 100 == 0:
            print(json.dumps({"stage": "bootstrap", "case": name, "completed": replicate + 1,
                              "replicates": replicates, "elapsed_seconds": time.perf_counter() - start}), flush=True)
    folder = OUT / "bootstrap" / name
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "selection_diagnostics.csv", index=False)
    pd.DataFrame(assignments).to_csv(folder / "assignments.csv", index=False)
    write_json(folder / "menus.json", menus)
    summary = {"replicates": replicates, "seed": protocol["stability"]["bootstrap_seed"],
               "case": name, "sources": source_manifest(),
               "elapsed_seconds": time.perf_counter() - start,
               "interpretation": "Complete-reselection stability under whole-scenario empirical resampling, not independent market evidence or a risk confidence certificate.",
               "client_assignment_counts": {p["name"]: dict(Counter(a[p["name"]] for a in assignments))
                                            for p in protocol["clients"]},
               "automatic_frequency": {p["name"]: float(np.mean([a[p["name"]] != "Review" for a in assignments]))
                                       for p in protocol["clients"]},
               "identified_cardinality_counts": {str(k): int(v) for k, v in Counter(frame.identified_cardinality).items()},
               "coverage_counts": {str(k): int(v) for k, v in Counter(frame.covered_clients).items()},
               "weight_alignment": "Minimum total L1 target distance to full-development reference, over all product permutations; original risk labels retained in menus.json.",
               "aligned_replicates": len(aligned),
               "aligned_weight_quantiles_025_50_975": np.quantile(aligned, [.025, .50, .975], axis=0).tolist() if aligned else None,
               "max_growth_regret_quantiles_025_50_975": np.quantile(frame.max_growth_regret, [.025, .50, .975]).tolist()}
    write_json(folder / "summary.json", summary)
    print(json.dumps({"stage": "bootstrap_complete", "case": name,
                      "automatic_frequency": summary["automatic_frequency"]}, ensure_ascii=False), flush=True)
    return summary


def run_neighborhoods(protocol, arrays, metrics, full_fits):
    base = full_fits[case_id(.10, "risk_only")]
    rows = []
    for product in base["menu"]:
        target = np.asarray(product["weights"])
        distance = np.abs(arrays["weights"] - target).sum(axis=1)
        neighborhood = np.flatnonzero((distance > NUMERIC_TOL) & (distance <= .10 + NUMERIC_TOL))
        for profile in protocol["clients"]:
            fields = metrics[profile["horizon"]]
            risk = empirical_risk(fields["mdd"][:, neighborhood], profile["loss_tolerance"])
            growth = fields["log_growth"][:, neighborhood].mean(axis=0)
            for k, candidate in enumerate(neighborhood):
                rows.append({"product": product["name"], "client": profile["name"],
                             "candidate_index": int(candidate), "distance_l1": float(distance[candidate]),
                             "w_equity": float(arrays["weights"][candidate, 0]),
                             "w_bond": float(arrays["weights"][candidate, 1]),
                             "w_bill": float(arrays["weights"][candidate, 2]),
                             "mean_log_growth": float(growth[k]),
                             "breach_probability": float(risk["breach_probability"][k]),
                             "expected_excess": float(risk["expected_excess"][k]),
                             "sample_risk_feasible_at_alpha10": bool(risk["breach_probability"][k] <= .10 + NUMERIC_TOL)})
    pd.DataFrame(rows).to_csv(OUT / "local_neighborhoods.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "validate", "bootstrap", "neighbors", "all"])
    parser.add_argument("--bootstrap-case")
    parser.add_argument("--replicates", type=int)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, metrics = get_cache(protocol)
    if args.stage in {"fit", "all"}:
        full = run_fit(protocol, arrays, metrics)
    else:
        full = json.loads((OUT / "full_development_menus.json").read_text())
    if args.stage in {"validate", "all"}:
        run_validation(protocol, arrays, metrics)
    if args.stage in {"neighbors", "all"}:
        run_neighborhoods(protocol, arrays, metrics, full)
    if args.stage in {"bootstrap", "all"}:
        run_bootstrap(protocol, arrays, metrics, full, args.bootstrap_case, args.replicates)


if __name__ == "__main__":
    main()
