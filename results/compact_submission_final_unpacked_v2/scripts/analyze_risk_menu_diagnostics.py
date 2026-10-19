"""Predeclared development-only comparator and sensitivity diagnostics.

CLI stages read only the runner's guarded 1--120 development cache. They never
replace the primary fit, access final scenarios, or write a competition report.
Importing this module performs no market access. Pure helpers accept arrays so
their leakage, referral and fixed-assignment semantics can be tested in isolation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
SPEC = importlib.util.spec_from_file_location("risk_menu_root_runner", ROOT / "scripts/run_risk_menu_research.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

from safeinvest_research.assessment import assess_fixed_menu
from safeinvest_research.diagnostics import label_order_diagnostics, policy_flags, reference_menu, sampling_flags
from safeinvest_research.metrics import NUMERIC_TOL, SUMMARY_METADATA, discrete_cvar, empirical_risk, summarize_paths
from safeinvest_research.selection import fit_menu


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sources():
    return {**RUNNER.source_manifest(),
            str(Path(__file__).resolve().relative_to(ROOT)): RUNNER.sha(Path(__file__).resolve())}


def _development_guard(arrays):
    if not np.array_equal(arrays["scenario_ids"], np.arange(1, 121)):
        raise RuntimeError("Diagnostics require exactly development scenario IDs 1..120")


def _rows(indices, total):
    values = np.asarray(indices)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= total)):
        raise ValueError("indices must be valid nonempty integer scenario rows")
    return values


def comparator_catalogue(metrics, weights, indices):
    """Construct the fixed H15 comparator solely from nominated training rows."""
    fields = metrics[15]
    rows = _rows(indices, len(fields["mdd"]))
    growth = np.asarray(fields["log_growth"])[rows].mean(axis=0)
    risk = discrete_cvar(np.asarray(fields["mdd"])[rows])
    construction = reference_menu(growth, risk)
    names = ["Conservative", "Balanced", "Growth"]
    menu = [{"name": names[position], "candidate_index": int(candidate),
             "weights": np.asarray(weights[candidate]).tolist()}
            for position, candidate in enumerate(construction["indices"])]
    return {"construction": construction, "menu": menu,
            "training_scenario_ids": (rows + 1).tolist()}


def profile_oracles(metrics, weights, profiles, indices, alpha, floors):
    """Training-only feasible growth benchmarks, never extra offered products."""
    result = {}
    for profile in profiles:
        fields = metrics[profile["horizon"]]
        rows = _rows(indices, len(fields["mdd"]))
        growth = np.asarray(fields["log_growth"])[rows].mean(axis=0)
        risk = empirical_risk(np.asarray(fields["mdd"])[rows], profile["loss_tolerance"])
        allowed = ((risk["breach_probability"] <= alpha + NUMERIC_TOL)
                   & (weights[:, 2] >= floors[profile["liquidity"]] - NUMERIC_TOL))
        result[profile["name"]] = {
            "feasible_candidate_count": int(np.count_nonzero(allowed)),
            "best_mean_log_growth": float(np.max(growth[allowed])) if np.any(allowed) else None,
        }
    return result


def assessment_objective(assessment, oracle):
    """Coverage and regret of a frozen menu relative to training-only oracles."""
    per_client, regrets = {}, []
    for client, assignment in assessment["assignments"].items():
        chosen = [item for item in assessment["client_product_details"]
                  if item["client"] == client and item.get("chosen", False)]
        if assignment == "Review":
            if chosen:
                raise ValueError("A referred client cannot have a chosen product")
            per_client[client] = None
            continue
        if len(chosen) != 1 or oracle[client]["best_mean_log_growth"] is None:
            raise ValueError("Automatic assignment needs one chosen detail and a feasible oracle")
        regret = oracle[client]["best_mean_log_growth"] - chosen[0]["mean_log_growth"]
        if regret < -NUMERIC_TOL:
            raise ValueError("Assigned growth exceeds the nominated oracle universe")
        regret = max(float(regret), 0.0)
        per_client[client] = regret
        regrets.append(regret)
    return {"covered_clients": len(regrets), "maximum_growth_regret": max(regrets, default=0.0),
            "mean_growth_regret": float(np.mean(regrets)) if regrets else 0.0,
            "growth_regret_by_client": per_client,
            "oracle": oracle, "regret_denominator": "covered representative clients only"}


def frozen_path_records(metrics, fitted, profiles, indices, **labels):
    """Score saved assignments on nominated rows; referrals stay null.

    This helper never screens a held-out row or changes an assignment after
    observing performance. It is shared by OOF and fixed-customer cost analyses.
    """
    records = []
    by_name = {item["name"]: item for item in fitted["menu"]}
    for profile in profiles:
        fields = metrics[profile["horizon"]]
        rows = _rows(indices, len(fields["mdd"]))
        assignment = fitted["assignments"][profile["name"]]
        item = None if assignment == "Review" else by_name.get(assignment)
        if assignment != "Review" and item is None:
            raise ValueError("Frozen assignment is absent from the fixed menu")
        for index in rows:
            row = {**labels, "scenario_id": int(index + 1), "client": profile["name"],
                   "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                   "assignment": assignment, "assigned": item is not None,
                   "candidate_index": None, "log_growth": None, "cagr": None,
                   "mdd": None, "volatility": None, "terminal_real": None,
                   "breach": None, "excess": None}
            if item is not None:
                candidate = item["candidate_index"]
                loss = float(fields["mdd"][index, candidate])
                breach = loss - profile["loss_tolerance"] > NUMERIC_TOL
                row.update({"candidate_index": int(candidate),
                            **{field: float(fields[field][index, candidate]) for field in
                               ("log_growth", "cagr", "mdd", "volatility", "terminal_real")},
                            "breach": bool(breach),
                            "excess": loss - profile["loss_tolerance"] if breach else 0.0})
            records.append(row)
    return records


def summarize_records(records, group_fields=("case", "client")):
    """Keep total/referral denominators separate from automatic-outcome rates."""
    frame = pd.DataFrame(records)
    summaries = []
    for key, group in frame.groupby(list(group_fields), sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        assigned = group.loc[group.assigned]
        row = {**dict(zip(group_fields, key)), "horizon": int(group.horizon.iloc[0]),
               "total_paths": len(group), "assigned_paths": len(assigned),
               "review_paths": int((~group.assigned).sum()), "breach_count": None,
               "breach_probability": None, "expected_excess": None, "mean_log_growth": None,
               "median_cagr": None, "p10_cagr": None, "median_mdd": None, "p90_mdd": None,
               "mean_mdd": None, "cvar90_mdd": None, "median_volatility": None,
               "median_terminal_real": None}
        if len(assigned):
            fields = {name: assigned[name].to_numpy(dtype=float) for name in
                      ("log_growth", "cagr", "mdd", "volatility", "terminal_real")}
            summary = summarize_paths(fields, loss_tolerance=float(group.loss_tolerance.iloc[0]))
            count = int(assigned.breach.astype(bool).sum())
            row.update({name: float(value) for name, value in summary.items()})
            row["breach_count"] = count
            if not np.isclose(row["breach_probability"], count / len(assigned), atol=NUMERIC_TOL, rtol=0):
                raise RuntimeError("Path-level breaches and summary denominator disagree")
        summaries.append(row)
    return summaries


def comparator_analysis(protocol, arrays, metrics):
    """Predeclared full/train-fold H15 comparator; primary results are untouched."""
    _development_guard(arrays)
    RUNNER.validate_folds(protocol)
    full_rows = np.arange(120)
    full_catalogue = comparator_catalogue(metrics, arrays["weights"], full_rows)
    fold_catalogues = {}
    for fold in protocol["validation"]["folds"]:
        train = np.asarray(fold["train_ids"], dtype=int) - 1
        fold_catalogues[str(fold["fold"])] = comparator_catalogue(metrics, arrays["weights"], train)
    full, folded, paths = {}, {}, []
    for case, alpha, liquidity, floors in RUNNER.policy_cases(protocol):
        assessment = assess_fixed_menu(metrics, arrays["weights"], full_catalogue["menu"],
                                       protocol["clients"], full_rows, alpha, floors)
        assessment["objective_diagnostics"] = assessment_objective(
            assessment, profile_oracles(metrics, arrays["weights"], protocol["clients"], full_rows, alpha, floors))
        full[case] = assessment
        folded[case] = []
        for fold in protocol["validation"]["folds"]:
            train = np.asarray(fold["train_ids"], dtype=int) - 1
            test = np.asarray(fold["test_ids"], dtype=int) - 1
            catalogue = fold_catalogues[str(fold["fold"])]
            fitted = assess_fixed_menu(metrics, arrays["weights"], catalogue["menu"],
                                      protocol["clients"], train, alpha, floors)
            fitted["objective_diagnostics"] = assessment_objective(
                fitted, profile_oracles(metrics, arrays["weights"], protocol["clients"], train, alpha, floors))
            fitted.update({"fold": fold["fold"], "train_ids": fold["train_ids"], "test_ids": fold["test_ids"]})
            folded[case].append(fitted)
            paths.extend(frozen_path_records(metrics, fitted, protocol["clients"], test,
                                             case=case, alpha=alpha, liquidity_case=liquidity, fold=fold["fold"]))
    return {"full_catalogue": full_catalogue, "fold_catalogues": fold_catalogues,
            "full_assessments": full, "fold_assessments": folded,
            "oof_paths": paths, "oof_summary": summarize_records(paths),
            "interpretation": "Predeclared descriptive comparator; no post-hoc replacement of the primary method. OOF outcomes condition on fold-frozen automatic assignments."}


def flags_analysis(protocol, metrics, full_fits, fold_fits, oof_summary):
    """Separate policy dependence, label ordering and sampling diagnostics."""
    primary = RUNNER.case_id(protocol["alpha_reference"], "risk_only")
    frame = pd.DataFrame(oof_summary)
    results = {}
    for case, full in full_fits.items():
        counts = {}
        selected = frame.loc[frame.case == case]
        if set(selected.client) != set(full["assignments"]) or selected.client.duplicated().any():
            raise ValueError("OOF summary needs one row per named case/client")
        for row in selected.to_dict("records"):
            n = int(row["assigned_test_paths"])
            k = int(row["breach_count"]) if n else 0
            counts[row["client"]] = {"n": n, "breach_count": k}
        indices = [item["candidate_index"] for item in full["menu"]]
        results[case] = {
            "label_order": label_order_diagnostics(metrics, indices, np.arange(120)),
            "sampling": sampling_flags(full, fold_fits[case], counts, alpha=full["alpha"]),
            "oof_automatic_counts": counts,
        }
    return {"primary_case": primary, "policy_dependence": policy_flags(full_fits, primary),
            "by_case": results, "interpretation": "Policy dependence, parameter/label ordering and sampling diagnostics are separate; none is a population risk certificate."}


def cost_analysis(protocol, base, metrics_by_cost, weights_by_cost):
    """Fixed original customers and separately re-screened fixed menu by cost."""
    paths, rescreened = [], {}
    reference_weights = None
    for cost, metrics in metrics_by_cost.items():
        weights = np.asarray(weights_by_cost[cost])
        if reference_weights is None:
            reference_weights = weights
        elif not np.array_equal(reference_weights, weights):
            raise ValueError("Cost caches must share exactly the same candidate weights and IDs")
        for item in base["menu"]:
            if not np.allclose(weights[item["candidate_index"]], item["weights"], rtol=0, atol=NUMERIC_TOL):
                raise ValueError("A cost cache changed an original product definition")
        paths.extend(frozen_path_records(metrics, base, protocol["clients"], np.arange(120), cost_rate=float(cost)))
        screened = assess_fixed_menu(metrics, weights, base["menu"], protocol["clients"], np.arange(120),
                                     alpha=base["alpha"], liquidity_floors=base["liquidity_floors"])
        rescreened[str(cost)] = screened
    return {"frozen_assignment_paths": paths,
            "frozen_assignment_summary": summarize_records(paths, group_fields=("cost_rate", "client")),
            "fixed_menu_rescreened": rescreened,
            "original_assignments": base["assignments"],
            "interpretation": "The original menu weights/names and original customer assignments are frozen for performance. Re-screening is a separate counterfactual and never replaces those outcomes."}


def local_resolution_analysis(protocol, coarse, fine_arrays, fine_metrics):
    """Refit only within the declared union of neighborhoods, preserving primary."""
    _development_guard(fine_arrays)
    step = protocol["predeclared_sensitivity"]["grid_step"]
    if not np.isclose(step, .025, rtol=0, atol=NUMERIC_TOL):
        raise ValueError("This diagnostic implements the declared .025 local refinement")
    fine_weights = np.asarray(fine_arrays["weights"])
    coarse_targets = np.asarray([item["weights"] for item in coarse["menu"]])
    distances = np.abs(fine_weights[:, None, :] - coarse_targets[None, :, :]).sum(axis=2)
    selected = np.flatnonzero(np.any(distances <= .10 + NUMERIC_TOL, axis=1))
    if len(selected) < 3:
        raise RuntimeError("Declared local fine universe has fewer than three targets")
    weights = fine_weights[selected]
    metrics = {horizon: {name: np.asarray(values)[:, selected] for name, values in fields.items()}
               for horizon, fields in fine_metrics.items()}
    fitted = fit_menu(metrics, weights, protocol["clients"], np.arange(120), alpha=coarse["alpha"],
                      liquidity_floors=coarse["liquidity_floors"])
    for item in fitted["menu"]:
        item["global_fine_candidate_index"] = int(selected[item["candidate_index"]])
    for detail in fitted["client_product_details"]:
        detail["global_fine_candidate_index"] = int(selected[detail["candidate_index"]])
    mapped_menu = []
    for item in coarse["menu"]:
        matches = np.flatnonzero(np.all(np.abs(weights - np.asarray(item["weights"])) <= NUMERIC_TOL, axis=1))
        if len(matches) != 1:
            raise RuntimeError("Each coarse primary target must have one exact local fine-grid counterpart")
        mapped_menu.append({**item, "candidate_index": int(matches[0])})
    coarse_assessment = assess_fixed_menu(metrics, weights, mapped_menu, protocol["clients"], np.arange(120),
                                           alpha=coarse["alpha"], liquidity_floors=coarse["liquidity_floors"])
    oracle = profile_oracles(metrics, weights, protocol["clients"], np.arange(120), coarse["alpha"], coarse["liquidity_floors"])
    return {"fine_grid_step": float(step), "neighborhood_l1_radius": .10,
            "full_fine_grid_count": len(fine_weights), "local_candidate_count": len(selected),
            "local_to_global_fine_indices": selected.tolist(),
            "primary_original_assignments": coarse["assignments"],
            "coarse_fixed_menu_reassessment": coarse_assessment,
            "coarse_reassessment_matches_original_assignments": coarse_assessment["assignments"] == coarse["assignments"],
            "fine_local_fit": fitted,
            "coarse_common_universe_objective": assessment_objective(coarse_assessment, oracle),
            "fine_common_universe_objective": assessment_objective(fitted, oracle),
            "interpretation": "Local resolution diagnostic only; both regrets use the same local fine-grid oracle universe. The primary .05 fit is preserved even if the fine result improves."}


def _save_stage(stage, payload, sources, inputs):
    if sources != _sources():
        raise RuntimeError("Analysis sources changed during execution")
    if any(RUNNER.sha(ROOT / relative) != expected for relative, expected in inputs.items()):
        raise RuntimeError("An input artifact changed during diagnostic computation")
    folder = RUNNER.OUT / "analysis" / stage
    folder.mkdir(parents=True, exist_ok=True)
    if stage == "comparator":
        RUNNER.write_json(folder / "full_and_fold_menus.json", json_ready({k: v for k, v in payload.items() if k not in ("oof_paths", "oof_summary")}))
        pd.DataFrame(payload["oof_paths"]).to_csv(folder / "oof_paths.csv", index=False)
        pd.DataFrame(payload["oof_summary"]).to_csv(folder / "oof_summary.csv", index=False)
    elif stage == "costs":
        RUNNER.write_json(folder / "fixed_menu_rescreened.json", json_ready(payload["fixed_menu_rescreened"]))
        pd.DataFrame(payload["frozen_assignment_paths"]).to_csv(folder / "frozen_assignment_paths.csv", index=False)
        pd.DataFrame(payload["frozen_assignment_summary"]).to_csv(folder / "frozen_assignment_summary.csv", index=False)
        RUNNER.write_json(folder / "interpretation.json", {"original_assignments": payload["original_assignments"], "interpretation": payload["interpretation"]})
    else:
        RUNNER.write_json(folder / "diagnostics.json", json_ready(payload))
    artifacts = {str(path.relative_to(ROOT)): RUNNER.sha(path) for path in sorted(folder.iterdir()) if path.is_file() and path.name != "manifest.json"}
    RUNNER.write_json(folder / "manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "sources": sources, "input_artifact_sha256": inputs, "output_artifact_sha256": artifacts,
        "scenario_ids": list(range(1, 121)), "summary_metadata": SUMMARY_METADATA,
        "scope": "Predeclared exploratory development diagnostics; no final access or primary model replacement.",
    })


def _input_hashes(paths):
    return {str(path.relative_to(ROOT)): RUNNER.sha(path) for path in paths}


def _verified_validation_inputs():
    manifest_path = RUNNER.OUT / "validation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["sources"] != RUNNER.source_manifest():
        raise RuntimeError("Validation sources differ from the current declared procedure")
    folds_path = RUNNER.OUT / "validation_fold_menus.json"
    summary_path = RUNNER.OUT / "validation_oof_summary.csv"
    return json.loads(folds_path.read_text()), pd.read_csv(summary_path), [manifest_path, folds_path, summary_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["comparator", "flags", "costs", "resolution", "all"])
    args = parser.parse_args()
    protocol = json.loads(RUNNER.PROTOCOL.read_text())
    full = RUNNER.compatible_full_fit()
    arrays, metrics = RUNNER.get_cache(protocol)
    _development_guard(arrays)
    primary = full[RUNNER.case_id(protocol["alpha_reference"], "risk_only")]
    base_inputs = [RUNNER.PROTOCOL, RUNNER.OUT / "fit_manifest.json", RUNNER.OUT / "full_development_menus.json"]
    main_key = RUNNER.cache_key(.05, protocol["candidate_family"]["cost_rate"])
    base_inputs.extend([RUNNER.OUT / "cache" / f"{main_key}.npz", RUNNER.OUT / "cache" / f"{main_key}.json"])
    stages = ["comparator", "flags", "costs", "resolution"] if args.stage == "all" else [args.stage]
    for stage in stages:
        sources = _sources()
        inputs = list(base_inputs)
        if stage == "comparator":
            work = lambda: comparator_analysis(protocol, arrays, metrics)
        elif stage == "flags":
            folds, summary, validation_inputs = _verified_validation_inputs()
            inputs.extend(validation_inputs)
            work = lambda: flags_analysis(protocol, metrics, full, folds, summary)
        elif stage == "costs":
            by_cost, weights_by_cost = {}, {}
            for cost in protocol["predeclared_sensitivity"]["cost_rates"]:
                cached_arrays, cached_metrics = RUNNER.get_cache(protocol, step=.05, cost=cost)
                _development_guard(cached_arrays)
                by_cost[cost], weights_by_cost[cost] = cached_metrics, cached_arrays["weights"]
                key = RUNNER.cache_key(.05, cost)
                inputs.extend([RUNNER.OUT / "cache" / f"{key}.npz", RUNNER.OUT / "cache" / f"{key}.json"])
            work = lambda: cost_analysis(protocol, primary, by_cost, weights_by_cost)
        else:
            fine_arrays, fine_metrics = RUNNER.get_cache(protocol, step=.025, cost=protocol["candidate_family"]["cost_rate"])
            key = RUNNER.cache_key(.025, protocol["candidate_family"]["cost_rate"])
            inputs.extend([RUNNER.OUT / "cache" / f"{key}.npz", RUNNER.OUT / "cache" / f"{key}.json"])
            work = lambda: local_resolution_analysis(protocol, primary, fine_arrays, fine_metrics)
        input_hashes = _input_hashes(inputs)
        payload = work()
        _save_stage(stage, payload, sources, input_hashes)
        print(json.dumps({"stage": stage, "saved": str((RUNNER.OUT / "analysis" / stage).relative_to(ROOT)), "primary_replaced": False}))


if __name__ == "__main__":
    main()
