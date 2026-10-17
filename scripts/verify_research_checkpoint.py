"""Check saved research fingerprints and test evidence without fitting a policy."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

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
    "outputs", "outputs_sha256", "output_artifact_sha256", "output_file_sha256_after_write",
}
PAIRS = [
    ("sources", "sources_at_end"), ("sources_at_start", "sources_at_end"),
    ("sources_sha256", "sources_sha256_at_end"),
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
]


def main():
    paths = list((ROOT / "results/model_review").rglob("*manifest*.json"))
    paths += list((ROOT / "results/model_review").rglob("independent_audit.json"))
    paths += list((ROOT / "research/figures").rglob("manifest.json"))
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
        for before, after in PAIRS:
            if before in item and after in item:
                assert item[before] == item[after], (path, before, after)
                checks += 1
        assert not item.get("final_scenarios_loaded", False), path
        assert not item.get("final_scenarios_read", False), path
        for key in HASH_KEYS:
            values = item.get(key)
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
    test_log_sha = sha(test_log)
    notebook_sha = sha(notebook_path)
    reading_copy_sha = sha(ROOT / "notebooks/risk_menu_development.html")
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "scope": "Saved-file fingerprints, completed test log and executed research notebook; no model fit or market return parsing",
        "hash_comparisons_and_consistency_checks": checks,
        "distinct_hashed_files": len(digests), "saved_snapshots_checked": len(snapshots),
        "snapshots_sha256": snapshots,
        "unit_tests": {"passed": int(match.group(1)), "seconds": float(match.group(2)),
                       "command": "PYTHONPATH=src:research .venv/bin/python -m unittest discover -s tests -q",
                       "log_sha256": test_log_sha, "result": "OK"},
        "notebook": {"path": str(notebook_path.relative_to(ROOT)), "executed_code_cells": len(code_cells),
                     "sha256": notebook_sha,
                     "reading_copy_sha256": reading_copy_sha},
        "legacy": {"baseline_revision": "7bca10b", "unchanged_paths": ["src", "report", "output"]},
        "limitations": ["Numerical and provenance checks do not certify risk or forecast validity",
                        "Independent algorithms were separately implemented with AI assistance, not external human peer review",
                        "Development data were reused and earlier final-set results were already inspected"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ["status", "saved_snapshots_checked", "distinct_hashed_files", "unit_tests"]}))


if __name__ == "__main__":
    main()
