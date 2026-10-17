"""Nested technical-screen calibration of the original static menu procedure.

Service alpha remains .10. Gamma is an inner-selected training screen only.
Inner scoring uses raw menu assignments; the declared admission guard applies
after the context refit and its effective service is reported separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
import run_risk_menu_research as primary
from safeinvest_research.evidence import simultaneous_upper
from safeinvest_research.metrics import NUMERIC_TOL, discrete_cvar
from safeinvest_research.selection import fit_menu

PROTOCOL = ROOT / "docs/nested_static_calibration_protocol.json"
OUT = ROOT / "results/model_review/nested_static_calibration_v1"
CASE = "alpha_10__risk_only"
SERVICE_ALPHA = .10
GAMMAS = (0., .025, .05, .075, .10)
FLOORS = {"high": 0., "medium": 0., "low": 0.}


def source_manifest():
    result = primary.source_manifest()
    for path in [Path(__file__).resolve(), PROTOCOL, ROOT / "src/safeinvest/core.py", ROOT / "src/safeinvest/data.py"]:
        result[str(path.relative_to(ROOT))] = primary.sha(path)
    return result


def input_manifest():
    files = [primary.OUT / name for name in ["full_development_menus.json", "fit_manifest.json", "validation_fold_menus.json",
                                            "validation_manifest.json", "validation_oof_paths.csv"]]
    cache = primary.OUT / "cache" / f"{primary.cache_key(.05,.0025)}.npz"
    files += [cache, cache.with_suffix(".json")]
    return {str(path.relative_to(ROOT)): primary.sha(path) for path in files}


def validate_context(context):
    """Check saved IDs, original-path complements and reproducible seed splits."""
    train, outer = context["training_ids"], context["outer_test_ids"]
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 1 <= i <= 120 for i in train + outer):
        raise ValueError("Require original integer scenario IDs 1..120")
    if len(train) not in (96, 120) or len(set(train)) != len(train) or len(set(outer)) != len(outer) or set(train) & set(outer):
        raise ValueError("Training and outer evaluation scenarios must be unique and disjoint")
    if (len(train) == 120 and outer) or (len(train) == 96 and (len(outer) != 24 or set(train) | set(outer) != set(range(1, 121)))):
        raise ValueError("Require full120 or a complete 96/24 outer partition")
    if context["inner_seed"] != 87020261020 + context["context_id"] or len(context["inner_folds"]) != 4:
        raise ValueError("Use the declared context seed and four inner folds")
    expected = np.array_split(np.random.default_rng(context["inner_seed"]).permutation(sorted(train)), 4)
    tests = []
    for number, (fold, expected_test) in enumerate(zip(context["inner_folds"], expected, strict=True), 1):
        inner_train, test = fold["train_ids"], fold["test_ids"]
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 1 <= i <= 120 for i in inner_train + test):
            raise ValueError("Inner IDs must be original integers in 1..120")
        if (fold["inner_fold"] != number or sorted(test) != sorted(map(int, expected_test))
                or len(inner_train) != 3 * len(train) // 4 or len(test) != len(train) // 4
                or len(set(inner_train)) != len(inner_train) or len(set(test)) != len(test)
                or set(inner_train) & set(test) or set(inner_train) | set(test) != set(train)):
            raise ValueError("Saved inner train/test IDs differ from deterministic complementary folds")
        tests.extend(test)
    if sorted(tests) != sorted(train):
        raise ValueError("Each context training scenario must be validated exactly once")


def fit_gamma(metrics, weights, profiles, indices, gamma):
    fitted = fit_menu(metrics, weights, profiles, indices, alpha=gamma, liquidity_floors=FLOORS)
    fitted["training_screen_gamma"] = fitted.pop("alpha")
    fitted["solver_gamma"] = float(gamma)
    fitted["service_alpha"] = SERVICE_ALPHA
    fitted["liquidity_case"] = "risk_only"
    fitted["parameter_interpretation"] = "gamma is a technical empirical training screen; service alpha=.10 is unchanged"
    for detail in fitted["client_product_details"]:
        detail["training_screen_feasible"] = detail.pop("sample_feasible")
    return fitted


def path_records(metrics, profiles, fitted, scenario_ids, *, context, gamma, inner_fold=None,
                 outer_fold=None, view="raw", assignments=None):
    menu = {item["name"]: item for item in fitted["menu"]} if fitted else {}
    assignment = assignments if assignments is not None else fitted["assignments"]
    records = []
    for profile in profiles:
        name = assignment[profile["name"]]
        product = menu.get(name)
        if name != "Review" and product is None:
            raise ValueError("Assignment does not identify a recorded menu product")
        for scenario in scenario_ids:
            index = int(scenario) - 1
            row = {"context": context, "training_screen_gamma": gamma, "service_alpha": SERVICE_ALPHA,
                   "solver_gamma": gamma,
                   "inner_fold": inner_fold, "outer_fold": outer_fold, "scenario_id": int(scenario), "view": view,
                   "client": profile["name"], "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                   "assignment": name, "assigned": product is not None, "candidate_index": None,
                   "log_growth": None, "cagr": None, "mdd": None, "volatility": None, "terminal_real": None,
                   "breach": None, "excess": None}
            if product is not None:
                candidate = product["candidate_index"]
                fields = metrics[profile["horizon"]]
                row.update({key: float(fields[key][index, candidate]) for key in ["log_growth", "cagr", "mdd", "volatility", "terminal_real"]})
                breached = row["mdd"] > profile["loss_tolerance"] + NUMERIC_TOL
                row.update({"candidate_index": candidate, "breach": bool(breached),
                            "excess": row["mdd"] - profile["loss_tolerance"] if breached else 0.})
            records.append(row)
    return records


def inner_summary(records, gamma):
    frame = pd.DataFrame(records)
    selected = frame.loc[frame.assigned]
    profiles = {}
    failing = []
    for client, group in frame.groupby("client", sort=False):
        auto = group.loc[group.assigned]
        count = int(auto.breach.astype(bool).sum()) if len(auto) else None
        probability = count / len(auto) if len(auto) else None
        passed = probability <= SERVICE_ALPHA + NUMERIC_TOL if probability is not None else None
        profiles[client] = {"total_paths": len(group), "auto_n": len(auto), "review_n": len(group) - len(auto),
                            "auto_folds": int(group.groupby("inner_fold").assigned.any().sum()),
                            "breach_count": count, "conditional_breach_probability": probability,
                            "conditional_mean_log_growth": float(auto.log_growth.mean()) if len(auto) else None,
                            "empirical_service_risk_pass": passed,
                            "evidence_note": "No automatic outcomes; risk is null" if not len(auto) else "Conditional auto subgroup, possibly small; dependent-fold empirical heuristic"}
        if passed is False:
            failing.append(client)
    reasons = (["no_inner_auto_outcomes"] if not len(selected) else [])
    reasons += [f"empirical_service_risk_exceeded:{client}" for client in failing]
    return {"training_screen_gamma": float(gamma), "service_alpha": SERVICE_ALPHA,
            "solver_gamma": float(gamma),
            "total_path_profile_units": len(frame), "raw_auto_path_profile_units": len(selected),
            "raw_review_path_profile_units": len(frame) - len(selected),
            "raw_auto_unit_mean_log_growth": float(selected.log_growth.mean()) if len(selected) else None,
            "profiles": profiles, "eligible": len(selected) > 0 and not failing, "eligibility_reasons": reasons}


def select_gamma(summaries, tolerance=1e-12):
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        return {"selected_gamma": None, "status": "calibration_unsupported", "eligible_gammas": [],
                "maximum_raw_inner_coverage": None, "maximum_auto_unit_growth_at_coverage": None}
    coverage = max(row["raw_auto_path_profile_units"] for row in eligible)
    covered = [row for row in eligible if row["raw_auto_path_profile_units"] == coverage]
    growth = max(row["raw_auto_unit_mean_log_growth"] for row in covered)
    tied = [row for row in covered if row["raw_auto_unit_mean_log_growth"] >= growth - tolerance]
    gamma = min(row["training_screen_gamma"] for row in tied)
    return {"selected_gamma": gamma, "status": "heuristic_gamma_selected",
            "eligible_gammas": [row["training_screen_gamma"] for row in eligible],
            "maximum_raw_inner_coverage": coverage, "maximum_auto_unit_growth_at_coverage": growth,
            "growth_tolerance": tolerance,
            "objective_scope": "Raw inner menu process; not a globally optimized post-admission effective service"}


def apply_admission(raw_fit, profiles, selected_summary):
    raw = dict(raw_fit["assignments"]) if raw_fit else {p["name"]: "Review" for p in profiles}
    effective = dict(raw)
    details = []
    for profile in profiles:
        name = profile["name"]
        evidence = selected_summary["profiles"][name] if selected_summary else None
        reason = "calibration_unsupported" if raw_fit is None else "raw_menu_referral" if raw[name] == "Review" else "inner_auto_evidence_present"
        if raw[name] != "Review" and evidence["auto_n"] == 0:
            effective[name] = "Review"
            reason = "no_inner_auto_evidence"
        details.append({"client": name, "raw_assignment": raw[name], "effective_assignment": effective[name],
                        "reason": reason, "inner_auto_n": evidence["auto_n"] if evidence else None,
                        "inner_auto_folds": evidence["auto_folds"] if evidence else None,
                        "inner_breach_count": evidence["breach_count"] if evidence else None,
                        "inner_conditional_breach_probability": evidence["conditional_breach_probability"] if evidence else None})
    return {"raw_assignments": raw, "effective_assignments": effective, "admission_details": details,
            "raw_covered_profiles": sum(name != "Review" for name in raw.values()),
            "effective_covered_profiles": sum(name != "Review" for name in effective.values()),
            "objective_scope": "Raw menu objective is preserved; admission does not refit products or reoptimize the effective objective"}


def calibrate_context(metrics, weights, profiles, context):
    validate_context(context)
    summaries, records, folds_by_gamma = [], [], {}
    for gamma in GAMMAS:
        rows, fits = [], []
        for fold in context["inner_folds"]:
            fitted = fit_gamma(metrics, weights, profiles, np.asarray(fold["train_ids"]) - 1, gamma)
            fitted["inner_fold"] = fold["inner_fold"]
            fitted["train_ids"] = fold["train_ids"]
            fitted["test_ids"] = fold["test_ids"]
            fits.append(fitted)
            rows.extend(path_records(metrics, profiles, fitted, fold["test_ids"], context=context["name"], gamma=gamma,
                                     inner_fold=fold["inner_fold"]))
        summaries.append(inner_summary(rows, gamma)); records.extend(rows)
        folds_by_gamma[str(gamma)] = fits
    selection = select_gamma(summaries)
    gamma = selection["selected_gamma"]
    chosen_summary = next((row for row in summaries if row["training_screen_gamma"] == gamma), None)
    raw_fit = None
    if gamma is not None:
        raw_fit = fit_gamma(metrics, weights, profiles, np.asarray(context["training_ids"]) - 1, gamma)
        # A separate fixed-family diagnostic; not consulted by the procedure.
        counts = np.array([row["breach_count"] for row in raw_fit["client_product_details"]], dtype=int)
        upper = simultaneous_upper(counts, len(context["training_ids"]), 1160, .05)
        for detail, bound in zip(raw_fit["client_product_details"], upper, strict=True):
            detail["postfit_static_family_risk_upper"] = float(bound)
            detail["postfit_bound_within_service_alpha"] = bool(bound <= SERVICE_ALPHA)
            detail["postfit_bound_used_in_selection"] = False
    admission = apply_admission(raw_fit, profiles, chosen_summary)
    refit = {"context": context["name"], "context_id": context["context_id"], "training_ids": context["training_ids"],
             "outer_test_ids": context["outer_test_ids"], "selection": selection,
             "selected_gamma_inner_evidence": chosen_summary, "raw_fit": raw_fit, **admission}
    return refit, summaries, folds_by_gamma, records


def performance_summary(records, keys):
    frame = pd.DataFrame(records); result = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple): values = (values,)
        auto = group.loc[group.assigned]
        row = {**dict(zip(keys, values)), "total_paths": len(group), "auto_n": len(auto),
               "review_n": len(group) - len(auto), "breach_count": None, "breach_probability": None,
               "mean_log_growth": None, "median_cagr": None, "p10_cagr": None, "median_mdd": None,
               "p90_mdd": None, "cvar90_mdd": None, "expected_excess": None}
        if len(auto):
            count = int(auto.breach.astype(bool).sum())
            row.update({"breach_count": count, "breach_probability": count / len(auto),
                        "mean_log_growth": float(auto.log_growth.mean()), "median_cagr": float(auto.cagr.median()),
                        "p10_cagr": float(np.quantile(auto.cagr, .10, method="inverted_cdf")),
                        "median_mdd": float(auto.mdd.median()), "p90_mdd": float(np.quantile(auto.mdd, .90, method="inverted_cdf")),
                        "cvar90_mdd": float(discrete_cvar(auto.mdd.to_numpy())), "expected_excess": float(auto.excess.mean())})
        result.append(row)
    return result


def compare_common_auto(effective, reference):
    rows = pd.DataFrame(effective)
    rows = rows.loc[rows.view == "effective"]
    keys = ["outer_fold", "scenario_id", "client"]
    old = reference.rename(columns={"fold": "outer_fold"})
    merged = rows.merge(old, on=keys, suffixes=("_nested", "_primary"), validate="one_to_one")
    result = []
    for client, group in merged.groupby("client", sort=False):
        common = group.loc[group.assigned_nested & group.assigned_primary]
        result.append({"client": client, "total_paths": len(group), "nested_auto_n": int(group.assigned_nested.sum()),
                       "primary_auto_n": int(group.assigned_primary.sum()), "common_auto_n": len(common),
                       "nested_common_breaches": int(common.breach_nested.astype(bool).sum()) if len(common) else None,
                       "primary_common_breaches": int(common.breach_primary.astype(bool).sum()) if len(common) else None,
                       "paired_common_mean_log_growth_delta": float((common.log_growth_nested - common.log_growth_primary).mean()) if len(common) else None,
                       "interpretation": "Paired common-auto subgroup only; unconditional coverage and conditional outcomes are reported separately, no pooled risk certificate"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); started = time.perf_counter()
    sources = source_manifest()
    protocol = json.loads(PROTOCOL.read_text()); research = json.loads(primary.PROTOCOL.read_text())
    if (protocol["service_alpha"] != SERVICE_ALPHA or tuple(protocol["training_gamma_candidates"]) != GAMMAS
            or protocol["static_family"]["cost_rate"] != .0025 or protocol["static_family"]["candidate_count"] != 232
            or protocol["static_family"]["liquidity_floors"] != FLOORS or protocol["clients"] != research["clients"]):
        raise ValueError("The predeclared calibration parameters/family changed")
    primary.validate_folds(research)
    contexts = protocol["contexts"]
    if len(contexts) != 6 or contexts[0]["training_ids"] != list(range(1, 121)) or contexts[0]["outer_test_ids"]:
        raise ValueError("Require full-development plus the five original outer contexts")
    for context, expected in zip(contexts[1:], research["validation"]["folds"], strict=True):
        if (context["context_id"] != expected["fold"] or context["training_ids"] != expected["train_ids"]
                or context["outer_test_ids"] != expected["test_ids"]):
            raise ValueError("Nested contexts differ from the original outer partition")
    for context in contexts: validate_context(context)
    arrays, metrics = primary.get_cache(research, step=.05, cost=.0025)
    inputs = input_manifest()
    if not np.array_equal(arrays["scenario_ids"], np.arange(1, 121)) or not np.allclose(arrays["weights"], primary.simplex_grid(.05), atol=1e-14, rtol=0):
        raise RuntimeError("Require guarded 1--120 and the original static target family")
    full_primary = primary.compatible_full_fit()[CASE]
    all_refits, inner_summaries, all_folds, inner_paths, full_paths, outer_paths = {}, {}, {}, [], [], []
    for context in contexts:
        refit, summaries, fits, rows = calibrate_context(metrics, arrays["weights"], research["clients"], context)
        all_refits[context["name"]] = refit; inner_summaries[context["name"]] = summaries
        all_folds[context["name"]] = fits; inner_paths.extend(rows)
        evaluation_ids = context["training_ids"] if context["context_id"] == 0 else context["outer_test_ids"]
        destination = full_paths if context["context_id"] == 0 else outer_paths
        for view in ["raw", "effective"]:
            assignments = refit[f"{view}_assignments"]
            destination.extend(path_records(metrics, research["clients"], refit["raw_fit"], evaluation_ids,
                                             context=context["name"], gamma=refit["selection"]["selected_gamma"],
                                             outer_fold=context["context_id"] or None, view=view, assignments=assignments))
        print(json.dumps({"context": context["name"], "selected_gamma": refit["selection"]["selected_gamma"],
                          "status": refit["selection"]["status"], "raw_assignments": refit["raw_assignments"],
                          "effective_assignments": refit["effective_assignments"], "elapsed_seconds": time.perf_counter() - started}), flush=True)
    original = pd.read_csv(primary.OUT / "validation_oof_paths.csv")
    original = original.loc[original.case == CASE].copy()
    if len(original) != 600: raise RuntimeError("Original primary OOF must retain all 600 profile-path rows")
    baseline = original.rename(columns={"fold": "outer_fold", "alpha": "service_alpha"}).copy()
    baseline["view"] = "primary_gamma10"; baseline["training_screen_gamma"] = .10
    comparison = performance_summary([*outer_paths, *baseline.to_dict("records")], ["view", "client"])
    full_reference = path_records(metrics, research["clients"], full_primary, list(range(1, 121)), context="full_primary",
                                  gamma=.10, view="primary_gamma10")
    full_summary = performance_summary([*full_paths, *full_reference], ["view", "client"])
    frames = {"inner_oof_paths.csv": pd.DataFrame(inner_paths), "full_refit_paths.csv": pd.DataFrame(full_paths),
              "outer_oof_paths.csv": pd.DataFrame(outer_paths), "outer_vs_primary_summary.csv": pd.DataFrame(comparison),
              "full_vs_primary_summary.csv": pd.DataFrame(full_summary),
              "outer_common_auto_paired_comparison.csv": pd.DataFrame(compare_common_auto(outer_paths, original))}
    jsons = {"inner_gamma_diagnostics.json": inner_summaries, "inner_fold_fits.json": all_folds,
             "context_refits_raw_effective.json": all_refits}
    if source_manifest() != sources or input_manifest() != inputs:
        raise RuntimeError("Sources/frozen inputs changed; no output was written")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items(): frame.to_csv(args.out / name, index=False)
    for name, value in jsons.items(): primary.write_json(args.out / name, value)
    end_sources, end_inputs = source_manifest(), input_manifest()
    if end_sources != sources or end_inputs != inputs: raise RuntimeError("Sources/frozen inputs changed during output writing")
    cache_meta = primary.OUT / "cache" / f"{primary.cache_key(.05,.0025)}.json"
    primary.write_json(args.out / "manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "sources_at_end": end_sources, "frozen_input_hashes": inputs, "frozen_input_hashes_at_end": end_inputs,
        "development_arrays_sha256": json.loads(cache_meta.read_text())["development_arrays_sha256"],
        "scenario_ids": arrays["scenario_ids"].tolist(), "final_scenarios_loaded": False,
        "outputs_sha256": {name: primary.sha(args.out / name) for name in [*frames, *jsons]}, "protocol": protocol,
        "scope": "Exploratory procedure on reused development paths. Service alpha=.10; gamma is a learned technical training screen. No final winner or certified calibration.",
        "postprocessing": "Inner raw scores select gamma. Admission guard follows raw context refit, so the reported raw objective is not a globally optimized effective-service objective. Raw/effective coverage and outcomes remain separate.",
        "inference": "Dependent inner/outer folds, possible small conditional auto groups and different learned policies. No pooled binomial certificate; null referrals are retained. Postfit1160-event upper bounds apply only conditionally to fixed static target/client risks and are not used by calibration.",
    })
    print(f"Wrote nested static calibration diagnostic to {args.out}", flush=True)


if __name__ == "__main__":
    main()
