"""Read-only audit of saved nested static-screen calibration artifacts.

This does not fit a menu, choose a new criterion, simulate a policy or read raw
market data.  It independently reconstructs saved score/selection/admission
arithmetic, verifies scenario memberships, and records its own evidence hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/model_review/nested_static_calibration_v1"
PROTOCOL = ROOT / "docs/nested_static_calibration_protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def artifact_hashes(directory):
    manifest = readj(directory / "manifest.json")
    paths = [directory / name for name in manifest["outputs_sha256"]]
    paths += [directory / "manifest.json", PROTOCOL,
              ROOT / "results/model_review/risk_menu_v1/validation_oof_paths.csv"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "scripts/run_nested_static_calibration.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def run_audit(directory):
    before_sources, before_artifacts = source_hashes(), artifact_hashes(directory)
    protocol = readj(PROTOCOL)
    manifest = readj(directory / "manifest.json")
    refits = readj(directory / "context_refits_raw_effective.json")
    diagnostics = readj(directory / "inner_gamma_diagnostics.json")
    fits = readj(directory / "inner_fold_fits.json")
    inner = pd.read_csv(directory / "inner_oof_paths.csv")
    outer = pd.read_csv(directory / "outer_oof_paths.csv")
    full = pd.read_csv(directory / "full_refit_paths.csv")
    summary = pd.read_csv(directory / "outer_vs_primary_summary.csv")
    common = pd.read_csv(directory / "outer_common_auto_paired_comparison.csv")
    primary = pd.read_csv(ROOT / "results/model_review/risk_menu_v1/validation_oof_paths.csv")
    primary = primary.loc[primary.case.eq("alpha_10__risk_only")].rename(columns={"fold": "outer_fold"})
    checks = []

    def check(condition, label):
        checks.append({"name": label, "passed": bool(condition)})
        if not condition:
            raise AssertionError(label)

    outcome_fields = ["candidate_index", "log_growth", "cagr", "mdd", "volatility", "terminal_real", "breach", "excess"]
    for name, frame in [("inner", inner), ("outer", outer), ("full", full)]:
        check(frame.loc[~frame.assigned, outcome_fields].isna().all().all(), name + "/Review_null")
        check(frame.loc[frame.assigned, outcome_fields].notna().all().all(), name + "/auto_nonnull")
        check(frame.service_alpha.eq(.1).all(), name + "/fixed_service_alpha")
        auto = frame.loc[frame.assigned]
        check(((auto.mdd > auto.loss_tolerance + 1e-12).to_numpy() == auto.breach.astype(bool).to_numpy()).all(),
              name + "/breach_definition")
    for context in protocol["contexts"]:
        name, training = context["name"], set(context["training_ids"])
        refit = refits[name]
        rows = inner.loc[inner.context.eq(name)]
        recomputed = []
        for diagnostic in diagnostics[name]:
            gamma = diagnostic["training_screen_gamma"]
            selected_rows = rows.loc[rows.training_screen_gamma.eq(gamma)]
            tag = f"{name}/gamma={gamma}"
            check(len(selected_rows) == len(training) * 5, tag + "/complete_denominator")
            check(not selected_rows.duplicated(["client", "scenario_id"]).any(), tag + "/unique_client_path")
            check(set(selected_rows.scenario_id) == training, tag + "/inner_membership")
            auto_rows = selected_rows.loc[selected_rows.assigned]
            check(len(auto_rows) == diagnostic["raw_auto_path_profile_units"], tag + "/raw_coverage")
            eligible = len(auto_rows) > 0
            for client, evidence in diagnostic["profiles"].items():
                group = selected_rows.loc[selected_rows.client.eq(client)]
                auto = group.loc[group.assigned]
                n, count = len(auto), int(auto.breach.astype(bool).sum())
                check(len(group) == len(training) and n == evidence["auto_n"]
                      and evidence["review_n"] == len(training) - n, tag + f"/{client}/denominator")
                check(evidence["auto_folds"] == int(group.groupby("inner_fold").assigned.any().sum()),
                      tag + f"/{client}/auto_folds")
                check(evidence["breach_count"] == (count if n else None), tag + f"/{client}/breach_count")
                check(evidence["conditional_breach_probability"] == (count / n if n else None), tag + f"/{client}/risk")
                eligible = eligible and (n == 0 or count / n <= .1 + 1e-12)
            check(eligible == diagnostic["eligible"], tag + "/eligibility")
            growth = float(auto_rows.log_growth.mean()) if len(auto_rows) else None
            check(growth is None or abs(growth - diagnostic["raw_auto_unit_mean_log_growth"]) < 2e-16,
                  tag + "/growth_score")
            recomputed.append((eligible, len(auto_rows), growth, gamma))
            for fold, fitted in zip(context["inner_folds"], fits[name][str(gamma)], strict=True):
                tag_fold = tag + f"/inner={fold['inner_fold']}"
                check(set(fitted["train_ids"]) == set(fold["train_ids"])
                      and set(fitted["test_ids"]) == set(fold["test_ids"]), tag_fold + "/saved_membership")
                check(not set(fitted["train_ids"]) & set(fitted["test_ids"])
                      and set(fitted["train_ids"]) | set(fitted["test_ids"]) == training, tag_fold + "/disjoint_partition")
                fold_rows = selected_rows.loc[selected_rows.inner_fold.eq(fold["inner_fold"])]
                menu = {product["name"]: product["candidate_index"] for product in fitted["menu"]}
                for client, assignment in fitted["assignments"].items():
                    group = fold_rows.loc[fold_rows.client.eq(client)]
                    check(set(group.scenario_id) == set(fold["test_ids"]), tag_fold + f"/{client}/test_membership")
                    check(group.assignment.eq(assignment).all(), tag_fold + f"/{client}/assignment")
                    check(group.candidate_index.isna().all() if assignment == "Review"
                          else group.candidate_index.eq(menu[assignment]).all(), tag_fold + f"/{client}/candidate")
        eligible = [row for row in recomputed if row[0]]
        if eligible:
            coverage = max(row[1] for row in eligible)
            covered = [row for row in eligible if row[1] == coverage]
            growth = max(row[2] for row in covered)
            chosen_gamma = min(row[3] for row in covered if row[2] >= growth - 1e-12)
        else:
            chosen_gamma = None
        check(refit["selection"]["selected_gamma"] == chosen_gamma, name + "/independently_selected_gamma")
        chosen = next((row for row in diagnostics[name] if row["training_screen_gamma"] == chosen_gamma), None)
        for client, raw in refit["raw_assignments"].items():
            expected = raw
            if chosen is None or raw != "Review" and chosen["profiles"][client]["auto_n"] == 0:
                expected = "Review"
            check(refit["effective_assignments"][client] == expected, name + f"/{client}/admission")
        frame = full if context["context_id"] == 0 else outer.loc[outer.outer_fold.eq(context["context_id"])]
        expected_ids = context["training_ids"] if context["context_id"] == 0 else context["outer_test_ids"]
        check(not training & set(context["outer_test_ids"]), name + "/outer_disjoint")
        for view in ["raw", "effective"]:
            data = frame.loc[frame.view.eq(view)]
            check(len(data) == len(expected_ids) * 5, name + f"/{view}/complete_denominator")
            check(not data.duplicated(["client", "scenario_id"]).any(), name + f"/{view}/unique_client_path")
            for client, assignment in refit[view + "_assignments"].items():
                group = data.loc[data.client.eq(client)]
                check(set(group.scenario_id) == set(expected_ids), name + f"/{view}/{client}/membership")
                check(group.assignment.eq(assignment).all(), name + f"/{view}/{client}/assignment")
                if assignment != "Review":
                    candidate = next(product["candidate_index"] for product in refit["raw_fit"]["menu"]
                                     if product["name"] == assignment)
                    check(group.candidate_index.eq(candidate).all(), name + f"/{view}/{client}/candidate")
    for (view, client), group in outer.groupby(["view", "client"]):
        row = summary.loc[summary.view.eq(view) & summary.client.eq(client)].iloc[0]
        auto = group.loc[group.assigned]
        tag = f"outer_summary/{view}/{client}"
        check(len(group) == 120 and row.total_paths == 120 and row.auto_n == len(auto), tag + "/denominator")
        check(row.review_n == 120 - len(auto), tag + "/Review_count")
        if len(auto):
            check(int(auto.breach.astype(bool).sum()) == row.breach_count, tag + "/breach_count")
            check(abs(auto.log_growth.mean() - row.mean_log_growth) < 2e-16, tag + "/growth")
        else:
            check(pd.isna(row.breach_count) and pd.isna(row.breach_probability), tag + "/no_risk_imputation")
    paired = outer.loc[outer.view.eq("effective")].merge(primary, on=["outer_fold", "scenario_id", "client"],
                                                         suffixes=("_nested", "_primary"), validate="one_to_one")
    for client, group in paired.groupby("client"):
        shared = group.loc[group.assigned_nested & group.assigned_primary]
        row = common.loc[common.client.eq(client)].iloc[0]
        check(row.total_paths == 120 and row.common_auto_n == len(shared), f"common/{client}/denominator")
        if len(shared):
            check(row.nested_common_breaches == int(shared.breach_nested.astype(bool).sum())
                  and row.primary_common_breaches == int(shared.breach_primary.astype(bool).sum()), f"common/{client}/breaches")
            check(abs(row.paired_common_mean_log_growth_delta - (shared.log_growth_nested - shared.log_growth_primary).mean()) < 2e-16,
                  f"common/{client}/paired_growth")
        else:
            check(pd.isna(row.paired_common_mean_log_growth_delta), f"common/{client}/null_growth")
    for name, digest in manifest["outputs_sha256"].items():
        check(sha(directory / name) == digest, "original_manifest_output_hash/" + name)
    check(manifest["sources"] == manifest["sources_at_end"]
          and manifest["frozen_input_hashes"] == manifest["frozen_input_hashes_at_end"], "original_start_end_hashes")
    after_sources, after_artifacts = source_hashes(), artifact_hashes(directory)
    check(before_sources == after_sources and before_artifacts == after_artifacts, "audit_sources_and_inputs_stable")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks_passed": len(checks),
            "runtime": {"python": sys.version, "pandas": pd.__version__}, "checks": checks,
            "sources_at_start": before_sources, "sources_at_end": after_sources,
            "artifacts_at_start": before_artifacts, "artifacts_at_end": after_artifacts,
            "row_counts": {"inner": len(inner), "outer_raw_and_effective": len(outer), "full_raw_and_effective": len(full)},
            "selected_gamma_by_context": {name: refit["selection"]["selected_gamma"] for name, refit in refits.items()},
            "service_alpha": .1,
            "outer_effective_by_client": json.loads(summary.loc[summary.view.eq("effective")].to_json(orient="records")),
            "market_data_read": False, "final_scenarios_read": False, "fit_or_policy_rerun": False,
            "scope": "Arithmetic and saved-membership/frozen-assignment consistency audit of reused-development nested heuristic",
            "limitations": ["Does not independently rerun fitted menu global optimization or market accounting",
                            "Does not establish execution provenance beyond saved memberships and artifact hashes",
                            "Inner empirical selection is not a risk-probability certificate",
                            "Outer folds use overlapping training data and different learned policies; no pooled binomial claim",
                            "Raw objective is not a globally optimized post-admission effective-service objective",
                            "Gamma selection stability across these contexts is not population or distribution-shift stability"],
            "original_manifest_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    result = run_audit(arguments.directory)
    destination = arguments.out or arguments.directory / "independent_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks_passed"], "row_counts": result["row_counts"],
                      "output": str(destination), "output_sha256": sha(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
