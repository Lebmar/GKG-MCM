"""Independent saved-artifact audit of strict-positive finite-grid research.

No optimizer, simulator or calibration helper is imported or rerun.  Original
guarded cache columns are used to verify every saved strict path lookup and
training risk/growth statistic; inner gamma selection is recalculated directly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/model_review/positive_asset_specification_v1"
PROTOCOL = ROOT / "docs/positive_asset_specification_protocol.json"
INDEX_LISTS = {"identified_indices", "indices", "search_pool_indices", "extension_indices", "reference_frontier"}
FIELDS = ["log_growth", "cagr", "mdd", "volatility", "terminal_real"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def snapshots(directory):
    manifest = readj(directory / "manifest.json")
    files = [directory / name for name in manifest["outputs_sha256"]]
    files += [directory / "manifest.json", PROTOCOL, Path(__file__).resolve()]
    files += [ROOT / name for name in manifest["sources"]]
    files += [ROOT / name for name in manifest["input_sha256"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def run_audit(directory):
    before = snapshots(directory)
    manifest, protocol = readj(directory / "manifest.json"), readj(PROTOCOL)
    mapping = pd.read_csv(directory / "candidate_mapping.csv")
    refits = readj(directory / "nested_context_refits.json")
    fixed = readj(directory / "fixed_gamma10_context_fits.json")
    inner_fits = readj(directory / "inner_fold_fits.json")
    diagnostics = readj(directory / "inner_gamma_diagnostics.json")
    frames = {name: pd.read_csv(directory / name) for name in ["inner_oof_paths.csv", "nested_full_paths.csv",
              "nested_outer_oof_paths.csv", "fixed_gamma10_full_paths.csv", "fixed_gamma10_outer_oof_paths.csv"]}
    checks, max_errors = [], {field: 0. for field in FIELDS}
    maximum_training_growth_error = 0.
    fit_count = annotated_candidate_refs = annotated_lists = 0

    def check(condition, label):
        checks.append({"name": label, "passed": bool(condition)})
        if not condition:
            raise AssertionError(label)

    cache_relative = next(name for name in manifest["input_sha256"] if name.endswith(".npz"))
    with np.load(ROOT / cache_relative, allow_pickle=False) as archive:
        check(np.array_equal(archive["scenario_ids"], np.arange(1, 121)), "cache_scenario_ids")
        global_weights = archive["weights"]
        allowed = np.flatnonzero(np.all(global_weights >= .05 - 1e-12, axis=1))
        check(len(global_weights) == 232 and len(allowed) == 172, "exact_family_sizes")
        check(np.array_equal(mapping.candidate_local_index, np.arange(172)), "local_index_order")
        check(np.array_equal(mapping.candidate_original232_index, allowed), "global_subset_and_order")
        check(np.max(np.abs(mapping[["w_equity", "w_bond", "w_bill"]].to_numpy() - global_weights[allowed])) < 2e-15,
              "mapping_weights")
        check(np.all(global_weights[allowed] >= .05 - 1e-12)
              and np.count_nonzero(np.all(np.isclose(global_weights[allowed], 1 / 3, atol=1e-15, rtol=0), axis=1)) == 1,
              "all_three_assets_positive_and_exact_thirds")

        def annotation(value, tag):
            nonlocal annotated_candidate_refs, annotated_lists
            if isinstance(value, list):
                for index, item in enumerate(value): annotation(item, tag + f"/{index}")
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key == "candidate_index":
                        annotated_candidate_refs += 1
                        if item is None:
                            check(value["candidate_local_index"] is None and value["candidate_original232_index"] is None,
                                  tag + "/null_candidate_map")
                        else:
                            check(isinstance(item, int) and not isinstance(item, bool) and 0 <= item < 172,
                                  tag + "/local_candidate_range")
                            check(value["candidate_local_index"] == item and value["candidate_original232_index"] == int(allowed[item]),
                                  tag + "/global_candidate_map")
                    elif key in INDEX_LISTS:
                        annotated_lists += 1
                        check(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 172 for i in item),
                              tag + "/" + key + "/local_range")
                        check(value[key + "_original232"] == [int(allowed[i]) for i in item], tag + "/" + key + "/global_map")
                    annotation(item, tag + "/" + key)

        for name, value in [("refits", refits), ("fixed", fixed), ("inner_fits", inner_fits)]:
            annotation(value, name)

        def fit_check(fitted, training_ids, gamma, tag):
            nonlocal fit_count, maximum_training_growth_error
            if fitted is None:
                return
            fit_count += 1
            check(fitted["n_training"] == len(training_ids) and fitted["training_screen_gamma"] == gamma
                  and fitted["service_alpha"] == .1, tag + "/configuration")
            train = np.asarray(training_ids) - 1
            menu = {item["name"]: item for item in fitted["menu"]}
            for item in menu.values():
                check(np.max(np.abs(np.asarray(item["weights"]) - global_weights[allowed[item["candidate_index"]]])) < 2e-15,
                      tag + "/menu_weights")
            for detail in fitted["client_product_details"]:
                column = int(allowed[detail["candidate_index"]])
                horizon = detail["horizon"]
                values = archive[f"H{horizon}__mdd"][train, column]
                count = int(np.count_nonzero(values > detail["loss_tolerance"] + 1e-12))
                growth = float(np.mean(archive[f"H{horizon}__log_growth"][train, column]))
                error = abs(growth - detail["mean_log_growth"])
                maximum_training_growth_error = max(maximum_training_growth_error, error)
                check(count == detail["breach_count"] and detail["n"] == len(train), tag + "/training_only_breach_count")
                check(error < 2e-15, tag + "/training_only_growth")
                check(detail["training_screen_feasible"] == (count / len(train) <= gamma + 1e-12), tag + "/training_screen")
                check(menu[detail["portfolio"]]["candidate_index"] == detail["candidate_index"], tag + "/detail_menu_identity")

        outcome_fields = ["candidate_index", "candidate_local_index", "candidate_original232_index", *FIELDS, "breach", "excess"]
        for name, frame in frames.items():
            check(frame.loc[~frame.assigned, outcome_fields].isna().all().all(), name + "/Review_null")
            check(frame.loc[frame.assigned, outcome_fields].notna().all().all(), name + "/auto_nonnull")
            check(frame.service_alpha.eq(.1).all(), name + "/service_alpha")
            auto = frame.loc[frame.assigned]
            local = auto.candidate_index.to_numpy(dtype=int)
            check(np.all(local >= 0) and np.all(local < 172), name + "/local_range")
            check(np.array_equal(auto.candidate_local_index.to_numpy(), local)
                  and np.array_equal(auto.candidate_original232_index.to_numpy(), allowed[local]), name + "/path_mapping")
            for horizon, rows in auto.groupby("horizon"):
                scene = rows.scenario_id.to_numpy(dtype=int) - 1
                columns = rows.candidate_original232_index.to_numpy(dtype=int)
                for field in FIELDS:
                    cached = archive[f"H{horizon}__{field}"][scene, columns]
                    difference = np.abs(rows[field].to_numpy() - cached)
                    max_errors[field] = max(max_errors[field], float(difference.max(initial=0)))
                    check(np.all(difference <= 2e-14 * np.maximum(1, np.abs(cached))), name + f"/H{horizon}/{field}/global_cache")
            check(np.array_equal((auto.mdd > auto.loss_tolerance + 1e-12).to_numpy(), auto.breach.astype(bool).to_numpy()),
                  name + "/breach_definition")

        for context in protocol["contexts"]:
            name, training = context["name"], set(context["training_ids"])
            refit = refits[name]
            check(set(refit["training_ids"]) == training and set(refit["outer_test_ids"]) == set(context["outer_test_ids"])
                  and not training & set(context["outer_test_ids"]), name + "/outer_partition")
            rows = frames["inner_oof_paths.csv"].loc[lambda frame: frame.context.eq(name)]
            scores = []
            for gamma_diagnostic in diagnostics[name]:
                gamma = gamma_diagnostic["training_screen_gamma"]
                current = rows.loc[rows.training_screen_gamma.eq(gamma)]
                check(len(current) == len(training) * 5 and not current.duplicated(["scenario_id", "client"]).any(),
                      name + f"/gamma={gamma}/denominator")
                selected = current.loc[current.assigned]
                eligible = len(selected) > 0
                for client, evidence in gamma_diagnostic["profiles"].items():
                    group = current.loc[current.client.eq(client)]
                    auto = group.loc[group.assigned]
                    count = int(auto.breach.astype(bool).sum())
                    risk = count / len(auto) if len(auto) else None
                    check(len(group) == len(training) and len(auto) == evidence["auto_n"]
                          and risk == evidence["conditional_breach_probability"], name + f"/{gamma}/{client}/inner_risk")
                    eligible = eligible and (risk is None or risk <= .1 + 1e-12)
                check(eligible == gamma_diagnostic["eligible"] and len(selected) == gamma_diagnostic["raw_auto_path_profile_units"],
                      name + f"/{gamma}/eligibility_and_coverage")
                growth = float(selected.log_growth.mean()) if len(selected) else None
                scores.append((eligible, len(selected), growth, gamma))
                for fold, fitted in zip(context["inner_folds"], inner_fits[name][str(gamma)], strict=True):
                    check(fitted["train_ids"] == fold["train_ids"] and fitted["test_ids"] == fold["test_ids"], name + "/inner_ids")
                    check(not set(fold["train_ids"]) & set(fold["test_ids"])
                          and set(fold["train_ids"]) | set(fold["test_ids"]) == training, name + "/inner_partition")
                    fit_check(fitted, fold["train_ids"], gamma, name + f"/{gamma}/{fold['inner_fold']}")
                    fold_rows = current.loc[current.inner_fold.eq(fold["inner_fold"])]
                    menu = {item["name"]: item["candidate_index"] for item in fitted["menu"]}
                    for client, assignment in fitted["assignments"].items():
                        group = fold_rows.loc[fold_rows.client.eq(client)]
                        check(set(group.scenario_id) == set(fold["test_ids"]) and group.assignment.eq(assignment).all(),
                              name + "/inner_frozen_assignment")
                        check(group.candidate_index.isna().all() if assignment == "Review" else group.candidate_index.eq(menu[assignment]).all(),
                              name + "/inner_frozen_target")
            eligible = [score for score in scores if score[0]]
            chosen_gamma = None
            if eligible:
                coverage = max(score[1] for score in eligible)
                covered = [score for score in eligible if score[1] == coverage]
                growth = max(score[2] for score in covered)
                chosen_gamma = min(score[3] for score in covered if score[2] >= growth - 1e-12)
            check(chosen_gamma == refit["selection"]["selected_gamma"], name + "/independent_gamma_selection")
            fit_check(refit["raw_fit"], context["training_ids"], chosen_gamma, name + "/nested_refit")
            fit_check(fixed[name], context["training_ids"], .1, name + "/fixed10_refit")
            evidence = next((row for row in diagnostics[name] if row["training_screen_gamma"] == chosen_gamma), None)
            for client, raw in refit["raw_assignments"].items():
                expected = raw if evidence is not None and (raw == "Review" or evidence["profiles"][client]["auto_n"] > 0) else "Review"
                check(refit["effective_assignments"][client] == expected, name + f"/{client}/admission")
            expected_ids = context["training_ids"] if context["context_id"] == 0 else context["outer_test_ids"]
            for filename, fitted, views in [
                ("nested_full_paths.csv" if context["context_id"] == 0 else "nested_outer_oof_paths.csv", refit, ["raw", "effective"]),
                ("fixed_gamma10_full_paths.csv" if context["context_id"] == 0 else "fixed_gamma10_outer_oof_paths.csv", fixed[name], ["raw"])]:
                frame = frames[filename].loc[lambda data: data.context.eq(name)]
                for view in views:
                    data = frame.loc[frame.view.eq(view)]
                    assignments = fitted[view + "_assignments"] if "raw_assignments" in fitted else fitted["assignments"]
                    fit_data = fitted["raw_fit"] if "raw_fit" in fitted else fitted
                    menu = {item["name"]: item["candidate_index"] for item in fit_data["menu"]} if fit_data else {}
                    check(len(data) == 5 * len(expected_ids), name + f"/{filename}/{view}/denominator")
                    for client, assignment in assignments.items():
                        group = data.loc[data.client.eq(client)]
                        check(set(group.scenario_id) == set(expected_ids) and group.assignment.eq(assignment).all(),
                              name + f"/{filename}/{view}/{client}/membership_and_assignment")
                        check(group.candidate_index.isna().all() if assignment == "Review" else group.candidate_index.eq(menu[assignment]).all(),
                              name + f"/{filename}/{view}/{client}/frozen_target")

    for scope, nested_file, fixed_file, summary_file in [
        ("full", "nested_full_paths.csv", "fixed_gamma10_full_paths.csv", "full_strict_vs_relaxed_summary.csv"),
        ("outer", "nested_outer_oof_paths.csv", "fixed_gamma10_outer_oof_paths.csv", "outer_strict_vs_relaxed_summary.csv")]:
        combined = pd.concat([frames[nested_file].assign(procedure="nested"),
                              frames[fixed_file].assign(procedure="fixed_gamma10")], ignore_index=True)
        saved = pd.read_csv(directory / summary_file)
        for (procedure, view, client), group in combined.groupby(["procedure", "view", "client"]):
            saved_rows = saved.loc[saved.specification.eq("strict172") & saved.procedure.eq(procedure)
                                  & saved.view.eq(view) & saved.client.eq(client)]
            tag = f"{scope}/{procedure}/{view}/{client}/summary"
            check(len(saved_rows) == 1, tag + "/unique")
            row, auto = saved_rows.iloc[0], group.loc[group.assigned]
            check(len(group) == 120 and row.total_paths == 120 and row.auto_n == len(auto)
                  and row.review_n == 120 - len(auto), tag + "/denominator")
            if len(auto):
                check(row.breach_count == int(auto.breach.astype(bool).sum())
                      and abs(row.breach_probability - auto.breach.astype(bool).mean()) < 2e-16, tag + "/risk")
                check(abs(row.mean_log_growth - auto.log_growth.mean()) < 2e-16, tag + "/growth")
                check(abs(row.expected_excess - auto.excess.mean()) < 2e-16, tag + "/expected_excess")
                check(abs(row.median_mdd - auto.mdd.median()) < 2e-16
                      and abs(row.p90_mdd - np.quantile(auto.mdd, .9, method="inverted_cdf")) < 2e-16, tag + "/MDD_summary")
            else:
                check(pd.isna(row.breach_probability) and pd.isna(row.mean_log_growth), tag + "/Review_null")
    check(protocol["reused_nested_protocol_sha256"] == sha(ROOT / "docs/nested_static_calibration_protocol.json"),
          "old_nested_protocol_unchanged")
    old = readj(ROOT / "results/model_review/nested_static_calibration_v1/manifest.json")
    check(sha(ROOT / "scripts/run_nested_static_calibration.py") == old["sources"]["scripts/run_nested_static_calibration.py"],
          "old_nested_implementation_unchanged")
    for name, digest in manifest["sources"].items(): check(sha(ROOT / name) == digest, "source_hash/" + name)
    for name, digest in manifest["input_sha256"].items(): check(sha(ROOT / name) == digest, "input_hash/" + name)
    for name, digest in manifest["outputs_sha256"].items(): check(sha(directory / name) == digest, "output_hash/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_sha256"] == manifest["input_sha256_at_end"],
          "original_source_and_input_start_end")
    after = snapshots(directory)
    check(before == after, "audit_source_and_evidence_stable")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks_passed": len(checks),
            "checks": checks, "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
            "source_and_evidence_sha256_start": before, "source_and_evidence_sha256_end": after,
            "verified_fit_count": fit_count, "annotated_candidate_references": annotated_candidate_refs,
            "annotated_index_lists": annotated_lists, "maximum_path_metric_errors": max_errors,
            "maximum_training_mean_growth_error": maximum_training_growth_error,
            "selected_gamma_by_context": {name: value["selection"]["selected_gamma"] for name, value in refits.items()},
            "path_row_counts": {name: len(frame) for name, frame in frames.items()},
            "raw_market_or_final_data_read": False, "cache_scope": "Original guarded1..120 NPZ only",
            "fit_or_policy_rerun": False, "original_manifests_modified": False,
            "scope": "Saved local/global identity, positive finite-grid specification, cache lookup, training-only statistics, calibration arithmetic and membership consistency",
            "limitations": ["Does not independently rerun global menu optimization or annual financial accounting",
                            "Saved memberships and hashes do not by themselves prove actual execution provenance",
                            "Five percent is the original grid's smallest positive point, not an identified economic requirement",
                            "No continuous strict-positive optimum, population risk certificate or final service selection claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    result = run_audit(arguments.directory)
    destination = arguments.out or arguments.directory / "independent_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "verified_fit_count", "maximum_path_metric_errors",
                       "maximum_training_mean_growth_error", "selected_gamma_by_context"]}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
