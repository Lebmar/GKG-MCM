"""Strict-positive, finite-grid specification sensitivity; no final policy choice.

Fitting and calibration are delegated unchanged to the original modules.  Solver
indices remain local; original232 identifiers are additional explicit fields.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
import run_nested_static_calibration as nested
import run_risk_menu_research as primary

PROTOCOL = ROOT / "docs/positive_asset_specification_protocol.json"
OUT = ROOT / "results/model_review/positive_asset_specification_v1"
INDEX_LISTS = {"identified_indices", "indices", "search_pool_indices", "extension_indices", "reference_frontier"}


def subset_family(weights, metrics):
    weights = np.asarray(weights)
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError("Require (candidate,3) weights")
    allowed = np.flatnonzero(np.all(weights >= .05 - 1e-12, axis=1))
    if len(allowed) == 0: raise ValueError("Strict-positive subset is empty")
    selected = {}
    for horizon, fields in metrics.items():
        selected[horizon] = {}
        for field, values in fields.items():
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[1] != len(weights):
                raise ValueError("Every path metric must have scenario axis0 and candidate axis1")
            selected[horizon][field] = values[:, allowed].copy()
    return allowed, weights[allowed].copy(), selected


def annotate_indices(value, allowed):
    """Preserve local solver fields and add global references, without mutation."""
    if isinstance(value, list): return [annotate_indices(item, allowed) for item in value]
    if not isinstance(value, dict): return deepcopy(value)
    result = {}
    for key, item in value.items():
        result[key] = annotate_indices(item, allowed)
        if key == "candidate_index":
            if item is None:
                result["candidate_local_index"] = None
                result["candidate_original232_index"] = None
            else:
                local = int(item)
                if local != item or not 0 <= local < len(allowed): raise ValueError("Invalid local candidate index")
                result["candidate_local_index"] = local
                result["candidate_original232_index"] = int(allowed[local])
        elif key in INDEX_LISTS:
            if not isinstance(item, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(allowed) for i in item):
                raise ValueError(f"Invalid local index list: {key}")
            result[key + "_original232"] = [int(allowed[i]) for i in item]
    return result


def verify_fit(fitted, weights, global_weights, allowed):
    if fitted is None: return
    for item in fitted["menu"]:
        local = item["candidate_index"]
        if not np.array_equal(item["weights"], weights[local]) or not np.array_equal(weights[local], global_weights[allowed[local]]):
            raise RuntimeError("Fitted menu weights do not match their local/global candidate")
        if np.min(item["weights"]) < .05 - 1e-12: raise RuntimeError("A product lacks one of the three positive assets")
    names = {item["name"] for item in fitted["menu"]}
    if any(name != "Review" and name not in names for name in fitted["assignments"].values()):
        raise RuntimeError("An assignment does not identify its fitted product")
    for detail in fitted["client_product_details"]:
        item = next(item for item in fitted["menu"] if item["name"] == detail["portfolio"])
        if detail["candidate_index"] != item["candidate_index"]: raise RuntimeError("Detail/menu candidate index mismatch")
    annotate_indices(fitted, allowed)


def verify_path_records(records, global_metrics, allowed):
    for row in records:
        if not row["assigned"]:
            if any(row[name] is not None for name in ["candidate_index", "mdd", "cagr", "log_growth", "breach", "excess"]):
                raise RuntimeError("Review outcomes must remain null")
            continue
        global_index = allowed[row["candidate_index"]]
        for field in ["log_growth", "cagr", "mdd", "volatility", "terminal_real"]:
            if row[field] != global_metrics[row["horizon"]][field][row["scenario_id"] - 1, global_index]:
                raise RuntimeError("Local path lookup differs from the mapped original cache column")


def source_manifest():
    result = nested.source_manifest()
    for path in [Path(__file__).resolve(), PROTOCOL, ROOT / "docs/problem_A.pdf"]:
        result[str(path.relative_to(ROOT))] = primary.sha(path)
    return result


def input_manifest():
    result = nested.input_manifest()
    for name in ["context_refits_raw_effective.json", "outer_oof_paths.csv", "full_refit_paths.csv", "manifest.json"]:
        path = nested.OUT / name
        result[str(path.relative_to(ROOT))] = primary.sha(path)
    return result


def marked(records, specification, procedure):
    return [{**row, "specification": specification, "procedure": procedure} for row in records]


def paired_comparison(strict_rows, relaxed_rows, procedure):
    a = pd.DataFrame(strict_rows); b = pd.DataFrame(relaxed_rows)
    keys = ["outer_fold", "scenario_id", "client"]
    merged = a.merge(b, on=keys, suffixes=("_strict", "_relaxed"), validate="one_to_one")
    result = []
    for client, group in merged.groupby("client", sort=False):
        common = group.loc[group.assigned_strict & group.assigned_relaxed]
        result.append({"procedure": procedure, "client": client, "total_paths": len(group),
                       "strict_auto_n": int(group.assigned_strict.sum()), "relaxed_auto_n": int(group.assigned_relaxed.sum()),
                       "common_auto_n": len(common), "strict_common_breaches": int(common.breach_strict.astype(bool).sum()) if len(common) else None,
                       "relaxed_common_breaches": int(common.breach_relaxed.astype(bool).sum()) if len(common) else None,
                       "common_mean_log_growth_delta_strict_minus_relaxed": float((common.log_growth_strict - common.log_growth_relaxed).mean()) if len(common) else None})
    return result


def main():
    started = time.perf_counter(); sources = source_manifest()
    protocol = json.loads(PROTOCOL.read_text()); original_protocol = json.loads(nested.PROTOCOL.read_text())
    research = json.loads(primary.PROTOCOL.read_text())
    old_nested_manifest = json.loads((nested.OUT / "manifest.json").read_text())
    nested_source = Path(nested.__file__).resolve()
    nested_source_key = str(nested_source.relative_to(ROOT))
    if (protocol["reused_nested_protocol_sha256"] != primary.sha(nested.PROTOCOL)
            or primary.sha(nested_source) != old_nested_manifest["sources"][nested_source_key]):
        raise RuntimeError("Strict sensitivity must reuse the exact saved232 nested protocol and calibration source")
    if (protocol["contexts"] != original_protocol["contexts"] or protocol["clients"] != research["clients"]
            or protocol["unchanged_parameters"]["gamma_candidates"] != list(nested.GAMMAS)
            or protocol["unchanged_parameters"]["service_alpha"] != nested.SERVICE_ALPHA):
        raise ValueError("Frozen contexts or calibration parameters changed")
    primary.validate_folds(research)
    for context in protocol["contexts"]: nested.validate_context(context)
    arrays, global_metrics = primary.get_cache(research, step=.05, cost=.0025)
    if not np.array_equal(arrays["scenario_ids"], np.arange(1, 121)): raise RuntimeError("Only guarded development IDs allowed")
    frozen_inputs = input_manifest()
    allowed, weights, metrics = subset_family(arrays["weights"], global_metrics)
    if len(arrays["weights"]) != 232 or len(allowed) != 172 or not np.all(np.diff(allowed) > 0):
        raise RuntimeError("Require the declared172 candidates in original ascending order")
    if np.sum(np.all(np.isclose(weights, 1 / 3, atol=1e-15, rtol=0), axis=1)) != 1:
        raise RuntimeError("Exact-thirds candidate was lost or duplicated")
    mapping = [{"candidate_local_index": int(local), "candidate_original232_index": int(global_index),
                "w_equity": float(weights[local, 0]), "w_bond": float(weights[local, 1]), "w_bill": float(weights[local, 2])}
               for local, global_index in enumerate(allowed)]
    refits, diagnostics, inner_fits, fixed_fits = {}, {}, {}, {}
    inner_paths, full_paths, outer_paths, fixed_full, fixed_outer = [], [], [], [], []
    for context in protocol["contexts"]:
        refit, summaries, fits, rows = nested.calibrate_context(metrics, weights, research["clients"], context)
        verify_fit(refit["raw_fit"], weights, arrays["weights"], allowed)
        for gamma_fits in fits.values():
            for fitted in gamma_fits: verify_fit(fitted, weights, arrays["weights"], allowed)
        inner_paths.extend(rows)
        refits[context["name"]] = refit; diagnostics[context["name"]] = summaries; inner_fits[context["name"]] = fits
        ids = context["training_ids"] if context["context_id"] == 0 else context["outer_test_ids"]
        destination = full_paths if context["context_id"] == 0 else outer_paths
        for view in ["raw", "effective"]:
            destination.extend(nested.path_records(metrics, research["clients"], refit["raw_fit"], ids,
                               context=context["name"], gamma=refit["selection"]["selected_gamma"], outer_fold=context["context_id"] or None,
                               view=view, assignments=refit[view + "_assignments"]))
        fixed = nested.fit_gamma(metrics, weights, research["clients"], np.asarray(context["training_ids"]) - 1, .10)
        verify_fit(fixed, weights, arrays["weights"], allowed)
        fixed["training_ids"], fixed["outer_test_ids"] = context["training_ids"], context["outer_test_ids"]
        fixed_fits[context["name"]] = fixed
        fixed_destination = fixed_full if context["context_id"] == 0 else fixed_outer
        fixed_destination.extend(nested.path_records(metrics, research["clients"], fixed, ids,
                                 context=context["name"], gamma=.10, outer_fold=context["context_id"] or None, view="raw"))
        print(json.dumps({"context": context["name"], "strict_nested_gamma": refit["selection"]["selected_gamma"],
                          "raw": refit["raw_assignments"], "effective": refit["effective_assignments"],
                          "strict_fixed10": fixed["assignments"], "elapsed_seconds": time.perf_counter() - started}), flush=True)
    for rows in [inner_paths, full_paths, outer_paths, fixed_full, fixed_outer]: verify_path_records(rows, global_metrics, allowed)
    relaxed_nested_full = pd.read_csv(nested.OUT / "full_refit_paths.csv").to_dict("records")
    relaxed_nested_outer = pd.read_csv(nested.OUT / "outer_oof_paths.csv").to_dict("records")
    relaxed_fixed_outer_frame = pd.read_csv(primary.OUT / "validation_oof_paths.csv")
    relaxed_fixed_outer_frame = relaxed_fixed_outer_frame.loc[relaxed_fixed_outer_frame.case == nested.CASE].rename(columns={"fold": "outer_fold"})
    relaxed_fixed_outer_frame["view"] = "raw"
    relaxed_fixed_outer = relaxed_fixed_outer_frame.to_dict("records")
    relaxed_full_fit = primary.compatible_full_fit()[nested.CASE]
    relaxed_fixed_full = nested.path_records(global_metrics, research["clients"], relaxed_full_fit, list(range(1, 121)),
                                             context="full_relaxed_primary", gamma=.10, view="raw")
    full_comparison = [*marked(full_paths, "strict172", "nested"), *marked(fixed_full, "strict172", "fixed_gamma10"),
                       *marked(relaxed_nested_full, "relaxed232", "nested"), *marked(relaxed_fixed_full, "relaxed232", "fixed_gamma10")]
    outer_comparison = [*marked(outer_paths, "strict172", "nested"), *marked(fixed_outer, "strict172", "fixed_gamma10"),
                        *marked(relaxed_nested_outer, "relaxed232", "nested"), *marked(relaxed_fixed_outer, "relaxed232", "fixed_gamma10")]
    summaries_full = nested.performance_summary(full_comparison, ["specification", "procedure", "view", "client"])
    summaries_outer = nested.performance_summary(outer_comparison, ["specification", "procedure", "view", "client"])
    strict_effective = [row for row in outer_paths if row["view"] == "effective"]
    relaxed_effective = [row for row in relaxed_nested_outer if row["view"] == "effective"]
    paired = [*paired_comparison(fixed_outer, relaxed_fixed_outer, "fixed_gamma10"), *paired_comparison(strict_effective, relaxed_effective, "nested_effective")]
    frames = {"candidate_mapping.csv": pd.DataFrame(mapping), "inner_oof_paths.csv": pd.DataFrame(annotate_indices(inner_paths, allowed)),
              "nested_full_paths.csv": pd.DataFrame(annotate_indices(full_paths, allowed)), "nested_outer_oof_paths.csv": pd.DataFrame(annotate_indices(outer_paths, allowed)),
              "fixed_gamma10_full_paths.csv": pd.DataFrame(annotate_indices(fixed_full, allowed)), "fixed_gamma10_outer_oof_paths.csv": pd.DataFrame(annotate_indices(fixed_outer, allowed)),
              "full_strict_vs_relaxed_summary.csv": pd.DataFrame(summaries_full), "outer_strict_vs_relaxed_summary.csv": pd.DataFrame(summaries_outer),
              "outer_common_auto_paired_comparison.csv": pd.DataFrame(paired)}
    jsons = {"candidate_mapping.json": mapping, "nested_context_refits.json": annotate_indices(refits, allowed),
             "inner_gamma_diagnostics.json": diagnostics, "inner_fold_fits.json": annotate_indices(inner_fits, allowed),
             "fixed_gamma10_context_fits.json": annotate_indices(fixed_fits, allowed)}
    if source_manifest() != sources or input_manifest() != frozen_inputs: raise RuntimeError("Sources or inputs changed before output")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items(): frame.to_csv(OUT / name, index=False)
    for name, value in jsons.items(): primary.write_json(OUT / name, value)
    end_sources, end_inputs = source_manifest(), input_manifest()
    if end_sources != sources or end_inputs != frozen_inputs: raise RuntimeError("Sources or inputs changed during output writing")
    cache_meta = primary.OUT / "cache" / f"{primary.cache_key(.05,.0025)}.json"
    primary.write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "sources_at_end": end_sources, "input_sha256": frozen_inputs, "input_sha256_at_end": end_inputs,
        "outputs_sha256": {name: primary.sha(OUT / name) for name in [*frames, *jsons]}, "protocol": protocol,
        "development_arrays_sha256": json.loads(cache_meta.read_text())["development_arrays_sha256"],
        "scenario_ids": arrays["scenario_ids"].tolist(), "final_scenarios_loaded": False,
        "candidate_count": len(allowed), "candidate_index_space": "LOCAL strict172 throughout candidate_index/indices; added original232 fields and candidate_mapping give global cache addresses",
        "checks": {"all_menu_assets_positive_at_least_grid_minimum": True, "all_assigned_paths_match_global_cache_via_map": True,
                   "all_referrals_null": True, "original_contexts_unchanged": True, "same_nested_implementation": True,
                   "nested_source_sha_matches_old232_run": True, "nested_protocol_sha_matches_predeclared_reference": True},
        "scope": "Finite-grid strict-positive specification sensitivity on reused development data. Old232 family is a relaxation. No final winner or service is adopted; 5% is grid resolution, not identified client risk/liquidity preference or continuous optimality.",
        "inference": "Service alpha=.10 differs from technical gamma. Inner eligibility is a dependent-fold heuristic. Postfit1160 bounds conservatively retain the old232x5 superset; no pooled OOF certificate or guarantee for dynamic policies."})
    print(f"Wrote strict-positive specification diagnostic to {OUT}", flush=True)


if __name__ == "__main__": main()
