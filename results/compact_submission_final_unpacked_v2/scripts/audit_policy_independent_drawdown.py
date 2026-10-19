"""Audit saved annual MDD envelopes by independent all-interval products.

Reads saved development evidence and its hashes only.  It neither imports the
reviewed envelope routine nor reloads original market arrays or fits a policy.
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
DIRECTORY = ROOT / "results/model_review/policy_independent_drawdown_bound_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_hashes(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    paths = [directory / name for name in manifest["outputs_sha256"]]
    paths.append(directory / "manifest.json")
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def run_audit(directory):
    script_hash_before = sha(Path(__file__).resolve())
    before = evidence_hashes(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    evidence = pd.read_csv(directory / "clairvoyant_asset_path.csv")
    bounds = pd.read_csv(directory / "path_bounds.csv")
    counts = pd.read_csv(directory / "client_counts.csv")
    grid = pd.read_csv(directory / "grid_client_counts.csv")
    checks = []
    max_error = max_witness_error = max_envelope_error = 0.0

    def check(condition, label):
        checks.append({"name": label, "passed": bool(condition)})
        if not condition:
            raise AssertionError(label)

    check(len(evidence) == 3600 and not evidence.duplicated(["scenario_id", "year"]).any(), "evidence_identities")
    check(len(bounds) == 600 and not bounds.duplicated(["client", "scenario_id"]).any(), "bound_identities")
    check(evidence.uses_future_year_return.all() and not evidence.investable_recommendation.any(), "clairvoyant_labels")
    lookup = {}
    for scenario, rows in evidence.groupby("scenario_id"):
        check(sorted(rows.year) == list(range(1, 31)), f"{scenario}/all_years")
        rows = rows.sort_values("year")
        gross = rows[["equity_gross", "government_bond_gross", "bill_proxy_gross"]].to_numpy()
        inflation_gross = rows.inflation_gross.to_numpy()
        envelope = np.max(gross, axis=1) / inflation_gross
        lookup[scenario] = envelope
        error = float(np.abs(envelope - rows.zero_fee_real_gross_envelope.to_numpy()).max())
        max_envelope_error = max(max_envelope_error, error)
        check(error < 2e-14, f"{scenario}/envelope")
        check((np.argmax(gross, axis=1) == rows.clairvoyant_best_asset_index.to_numpy()).all(), f"{scenario}/best_asset")
        wealth_factor = rows.ideal_real_wealth_after.to_numpy() / rows.ideal_real_wealth_before.to_numpy()
        check(np.max(np.abs(wealth_factor - envelope)) < 2e-14, f"{scenario}/wealth_factor_identity")
    independent = []
    for row in bounds.itertuples(index=False):
        envelope = lookup[row.scenario_id][:row.horizon]
        # Enumerate all intervals directly; do not reuse the reviewed
        # cumulative-log/running-peak implementation.
        brute = max([0.0] + [1 - float(np.prod(envelope[start:end]))
                             for start in range(row.horizon) for end in range(start + 1, row.horizon + 1)])
        error = abs(brute - row.mdd_lower_bound)
        max_error = max(max_error, error)
        tag = f"{row.client}/{row.scenario_id}"
        check(error < 2e-14, tag + "/all_interval_bound")
        check((brute - row.loss_tolerance > 1e-12) == row.forced_breach, tag + "/forced_event")
        start, end = row.witness_peak_time, row.witness_trough_time
        check(0 <= start <= end <= row.horizon, tag + "/witness_interval")
        witness = 1 - float(np.prod(envelope[start:end]))
        max_witness_error = max(max_witness_error, abs(witness - brute))
        check(abs(witness - brute) < 2e-14, tag + "/witness")
        independent.append({"client": row.client, "scenario_id": row.scenario_id,
                            "forced": brute - row.loss_tolerance > 1e-12})
    independent_frame = pd.DataFrame(independent)
    for client, rows in bounds.groupby("client"):
        check(len(rows) == 120 and set(rows.scenario_id) == set(range(1, 121)), client + "/denominator")
        count_row = counts.loc[counts.client.eq(client)].iloc[0]
        forced = int(independent_frame.loc[independent_frame.client.eq(client), "forced"].sum())
        candidates = grid.loc[grid.client.eq(client)]
        check(forced == count_row.forced_breach_count, client + "/independent_forced_count")
        check(len(candidates) == 232 and candidates.candidate_index.nunique() == 232, client + "/grid_catalogue")
        check(candidates.breach_count.min() == count_row.grid_min_breach_count
              and (candidates.breach_count >= forced).all(), client + "/cached_grid_count_lower_bound")
        check(bool(count_row.empirical_budget_impossible_for_entire_declared_class) == (forced > 12),
              client + "/empirical_impossibility_flag")
        expected_ids = sorted(independent_frame.loc[independent_frame.client.eq(client) & independent_frame.forced,
                                                     "scenario_id"].tolist())
        check(summary["forced_original_scenario_ids"][client] == expected_ids, client + "/forced_ids")
    for field in ["source_sha256_start", "input_artifact_sha256_start"]:
        for name, digest in manifest[field].items():
            check(sha(ROOT / name) == digest, "source_or_input_hash/" + name)
    for name, digest in manifest["outputs_sha256"].items():
        check(sha(directory / name) == digest, "output_hash/" + name)
    check(manifest["source_sha256_start"] == manifest["source_sha256_end"]
          and manifest["input_artifact_sha256_start"] == manifest["input_artifact_sha256_end"], "original_start_end_hashes")
    after = evidence_hashes(directory)
    script_hash_after = sha(Path(__file__).resolve())
    check(before == after and script_hash_before == script_hash_after, "audit_source_and_evidence_stable")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks_passed": len(checks),
            "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
            "checks": checks, "independent_path_bounds": len(bounds),
            "maximum_errors": {"peak_vs_independent_interval": max_error, "witness": max_witness_error,
                               "envelope_identity": max_envelope_error},
            "forced_counts": independent_frame.groupby("client").forced.sum().astype(int).to_dict(),
            "source_sha256_start": script_hash_before, "source_sha256_end": script_hash_after,
            "evidence_sha256_start": before, "evidence_sha256_end": after,
            "raw_market_data_read": False, "final_scenarios_read": False, "new_policy_fitted_or_selected": False,
            "proof": "Every annual interval has product(actual nonnegative-cost long-only real factors) <= product(annual best-asset real factors); taking maximum declines over all intervals plus zero proves MDD_policy >= D_star.",
            "scope": "Initial plus annual endpoints; annual trading only; fixed three-asset set; no external funding, leverage, shorts or negative fees; positive gross factors/CPI/wealth",
            "limitations": ["Evidence arithmetic verification, not independent provenance of raw data loading",
                            "Clairvoyant zero-fee best-asset path is an optimistic relaxation, not an investment recommendation",
                            "No forced count above12 observed; no empirical impossibility of the whole declared policy class follows",
                            "A lower bound at or below the budget does not prove a causal safe policy exists",
                            "No population probability, new statistical certificate, model validity or final-policy selection claim"],
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
    print(json.dumps({"status": result["status"], "checks": result["checks_passed"], "maximum_errors": result["maximum_errors"],
                      "forced_counts": result["forced_counts"], "output": str(destination), "output_sha256": sha(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
