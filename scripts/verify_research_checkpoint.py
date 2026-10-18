"""Check saved research fingerprints and test evidence without fitting a policy."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
import platform
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/model_review/research_checkpoint_v1/verification.json"
HASH_KEYS = {
    "sources", "sources_at_end", "source_sha256_start", "source_sha256_end",
    "source_sha256", "cache_source_sha256", "sources_sha256", "sources_sha256_at_end",
    "source_sha256_at_start", "source_sha256_at_end", "sources_at_start",
    "frozen_input_hashes", "frozen_input_hashes_at_end", "input_sha256", "input_sha256_at_end",
    "input_hashes", "input_hashes_at_end", "inputs_sha256", "inputs_sha256_at_end", "input_artifact_sha256",
    "frozen_dependency_sha256", "frozen_dependency_sha256_at_end",
    "source_and_artifact_sha256_start", "source_and_artifact_sha256_end",
    "source_and_evidence_sha256_start", "source_and_evidence_sha256_end",
    "source_artifact_sha256_start", "source_artifact_sha256_end",
    "evidence_sha256_start", "evidence_sha256_end",
    "input_artifact_sha256_start", "input_artifact_sha256_end", "artifacts_at_start", "artifacts_at_end",
    "closure_sha256", "source_input_artifact_sha256_start", "source_input_artifact_sha256_end",
    "sources_and_artifacts_at_start", "sources_and_artifacts_at_end",
    "evidence_sha256", "source_evidence_sha256_start", "source_evidence_sha256_end", "sources_sha256_at_start",
    "frozen_input_sha256", "execution_sources_sha256",
    "reviewed_sources_sha256", "audit_source_sha256", "audit_sources_sha256", "source_input_artifact_sha256",
    "outputs_sha256_before", "outputs_sha256_after",
    "copied_files_sha256_before", "copied_files_sha256_after", "wrapper_test_sources_sha256",
    "original_archive_sha256_start", "original_archive_sha256_end",
    "outputs", "outputs_sha256", "output_artifact_sha256", "output_file_sha256_after_write",
}
PAIRS = [
    ("sources", "sources_at_end"), ("sources_at_start", "sources_at_end"),
    ("sources_sha256", "sources_sha256_at_end"),
    ("sources_sha256_at_start", "sources_sha256_at_end"),
    ("source_sha256_start", "source_sha256_end"),
    ("source_sha256_at_start", "source_sha256_at_end"),
    ("frozen_input_hashes", "frozen_input_hashes_at_end"),
    ("input_sha256", "input_sha256_at_end"), ("input_hashes", "input_hashes_at_end"),
    ("frozen_dependency_sha256", "frozen_dependency_sha256_at_end"),
    ("source_and_artifact_sha256_start", "source_and_artifact_sha256_end"),
    ("source_and_evidence_sha256_start", "source_and_evidence_sha256_end"),
    ("source_artifact_sha256_start", "source_artifact_sha256_end"),
    ("evidence_sha256_start", "evidence_sha256_end"),
    ("inputs_sha256", "inputs_sha256_at_end"),
    ("input_artifact_sha256_start", "input_artifact_sha256_end"),
    ("artifacts_at_start", "artifacts_at_end"),
    ("source_input_artifact_sha256_start", "source_input_artifact_sha256_end"),
    ("sources_and_artifacts_at_start", "sources_and_artifacts_at_end"),
    ("source_evidence_sha256_start", "source_evidence_sha256_end"),
    ("outputs_sha256_before", "outputs_sha256_after"),
    ("copied_files_sha256_before", "copied_files_sha256_after"),
    ("original_archive_sha256_start", "original_archive_sha256_end"),
]


def evidence_records(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from evidence_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_records(child)


def main():
    paths = list((ROOT / "results/model_review").rglob("*manifest*.json"))
    paths += list((ROOT / "results/model_review").rglob("independent_audit.json"))
    paths += list((ROOT / "research/figures").rglob("manifest.json"))
    paths += [p for p in [ROOT / "docs/nested_dynamic_stage_review.json",
                          ROOT / "docs/liquidity_service_design.json",
                          ROOT / "docs/coverage_admission_policy_review.json",
                          ROOT / "docs/incremental_admission_risk_diagnostic_protocol.json",
                          ROOT / "docs/refit_stability_method_review.json",
                          ROOT / "docs/refit_stability_preexecution_review.json",
                          ROOT / "docs/refit_stability_protocol.json",
                          ROOT / "docs/refit_stability_audit_plan.json",
                          ROOT / "docs/forecast_assumption_diagnostic_protocol.json",
                          ROOT / "docs/forecast_assumption_diagnostic_judgment.json",
                          ROOT / "docs/current_diagnostic_claims_review.json",
                          ROOT / "docs/admission_stability_judgment.json",
                          ROOT / "docs/refit_stability_independent_judgment.json",
                          ROOT / "docs/simple_drift_cost_interpretation.json",
                          ROOT / "docs/simple_drift_execution_implementation_review.json",
                          ROOT / "docs/simple_execution_comparison_protocol.json",
                          ROOT / "docs/simple_execution_design_review.json",
                          ROOT / "docs/simple_execution_final_source_review.json",
                          ROOT / "docs/simple_execution_preexecution_method_review.json",
                          ROOT / "docs/simple_execution_freeze_receipt.json",
                          ROOT / "docs/simple_execution_audit_plan.json",
                          ROOT / "docs/simple_execution_diagnostic_judgment.json",
                          ROOT / "docs/simple_execution_independent_judgment.json",
                          ROOT / "results/model_review/incremental_admission_risk_v1/repeatability_receipt.json",
                          ROOT / "docs/service_refinement_readiness_review.json",
                          ROOT / "docs/liquidity_policy_refinement_review.json",
                          ROOT / "docs/service_governance_independent_review.json",
                          ROOT / "docs/service_policy_refinement_contract.json",
                          ROOT / "docs/refit_aware_selection_contract.json",
                          ROOT / "docs/joint_liquidity_refit_protocol.json",
                          ROOT / "docs/fixed_state_forecast_diagnostic_protocol.json",
                          ROOT / "results/model_review/joint_liquidity_refit_attempt1_incomplete/failure_receipt.json"] if p.is_file()]
    paths = sorted(set(p for p in paths if "cache" not in p.parts and p != OUT))
    digests, checks, snapshots = {}, 0, {}

    def sha(path):
        nonlocal checks
        resolved = path.resolve()
        if resolved not in digests:
            digests[resolved] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        checks += 1
        return digests[resolved]

    for path in paths:
        item = json.loads(path.read_text())
        if path.name == "independent_audit.json":
            assert item.get("status") == "passed" or item.get("passed") is True, path
            checks += 1
        assert not item.get("final_scenarios_loaded", False), path
        assert not item.get("final_scenarios_read", False), path
        for record in evidence_records(item):
            for before, after in PAIRS:
                if before in record and after in record:
                    assert record[before] == record[after], (path, before, after)
                    checks += 1
            for key in HASH_KEYS:
                values = record.get(key)
                if not isinstance(values, dict):
                    continue
                for relative, expected in values.items():
                    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                        continue
                    candidates = [ROOT / relative, path.parent / relative]
                    if key.startswith("output"):
                        candidates.reverse()
                    actual = next((candidate for candidate in candidates if candidate.is_file()), None)
                    assert actual is not None, (path, key, relative)
                    assert sha(actual) == expected, (path, key, relative)
        snapshots[str(path.relative_to(ROOT))] = sha(path)

    test_log = ROOT / "tmp/research_checkpoint_tests.log"
    tests = test_log.read_text()
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s", tests)
    assert match and tests.rstrip().endswith("OK"), "Need completed passing test evidence"
    notebook_path = ROOT / "notebooks/risk_menu_development.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    legacy = subprocess.run(["git", "diff", "--exit-code", "7bca10b", "--", "src", "report", "output"],
                            cwd=ROOT, capture_output=True, text=True)
    assert legacy.returncode == 0, "V1 source/report/output changed"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    test_log_snapshot = OUT.parent / "unit_tests.txt"
    test_log_snapshot.write_bytes(test_log.read_bytes())
    test_log_sha = sha(test_log)
    assert sha(test_log_snapshot) == test_log_sha
    notebook_sha = sha(notebook_path)
    reading_copy_sha = sha(ROOT / "notebooks/risk_menu_development.html")
    additional_notebooks = []
    for companion in [ROOT / "notebooks/admission_stability_forecasts.ipynb",
                      ROOT / "notebooks/simple_execution_diagnostic.ipynb"]:
        if not companion.exists():
            continue
        saved = nbformat.read(companion, as_version=4)
        companion_code = [cell for cell in saved.cells if cell.cell_type == "code"]
        assert companion_code and all(cell.execution_count is not None for cell in companion_code)
        assert not any(output.output_type == "error" for cell in companion_code for output in cell.outputs)
        reading_copy = companion.with_suffix(".html")
        additional_notebooks.append({"path": str(companion.relative_to(ROOT)),
            "executed_code_cells": len(companion_code), "sha256": sha(companion),
            "reading_copy_path": str(reading_copy.relative_to(ROOT)), "reading_copy_sha256": sha(reading_copy)})

    environment_file = ROOT / "requirements-research-pinned.txt"
    environment_sha = sha(environment_file)
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "scope": "Saved-file fingerprints, completed test log and executed research notebook; no model fit or market return parsing",
        "hash_comparisons_and_consistency_checks": checks,
        "distinct_hashed_files": len(digests), "saved_snapshots_checked": len(snapshots),
        "snapshots_sha256": snapshots,
        "unit_tests": {"passed": int(match.group(1)), "seconds": float(match.group(2)),
                       "command": "PYTHONPATH=src:research .venv/bin/python -m unittest discover -s tests -v",
                       "log_path": str(test_log_snapshot.relative_to(ROOT)),
                       "log_sha256": test_log_sha, "result": "OK"},
        "notebook": {"path": str(notebook_path.relative_to(ROOT)), "executed_code_cells": len(code_cells),
                     "sha256": notebook_sha,
                     "reading_copy_sha256": reading_copy_sha},
        "additional_notebooks": additional_notebooks,
        "legacy": {"baseline_revision": "7bca10b", "unchanged_paths": ["src", "report", "output"]},
        "runtime": {"python": sys.version.split()[0], "system": platform.system(),
                    "architecture": platform.machine(),
                    "direct_dependencies": {name: version(name) for name in
                        ["numpy", "pandas", "matplotlib", "scipy", "pypdf", "nbformat", "nbclient", "ipykernel", "nbconvert"]},
                    "version_file": str(environment_file.relative_to(ROOT)), "version_file_sha256": environment_sha,
                    "scope": "Tested direct dependency versions; indirect dependencies and cross-platform byte equality are not locked"},
        "limitations": ["Numerical and provenance checks do not certify risk or forecast validity",
                        "Independent algorithms were separately implemented with AI assistance, not external human peer review",
                        "Development data were reused and earlier final-set results were already inspected"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ["status", "saved_snapshots_checked", "distinct_hashed_files", "unit_tests"]}))


if __name__ == "__main__":
    main()
