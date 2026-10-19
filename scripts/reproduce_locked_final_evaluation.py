"""One isolated same-source replay of an already complete locked evaluation.

This does not select a policy, add observations or independently validate
investment risk. A predeclared plan and separately frozen reproducer sources
are required. All original outputs remain read-only; every replica output,
log and difference is retained. Final values are consumed only by the copied
unchanged adapter after its original release checks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("docs/final_evaluation_reproduction_plan.json")
SOURCE_FREEZE = Path("docs/final_evaluation_reproduction_source_freeze.json")
PREPARATION = Path("docs/final_evaluation_protocol.json")
RELEASE = Path("docs/final_evaluation_release.json")
DEVELOPMENT = Path("results/model_review/final_evaluation_development_gate_v1")
ORIGINAL = Path("results/final_locked_evaluation_v1")
OUT = Path("results/final_locked_reproduction_v1")
DEV_OUTPUTS = {"Q1_paths.csv.gz", "Q1_reference_reproduction.json", "Q2_paths.csv.gz", "development_traces.npz",
               "exact_array_reproduction.json", "object_specification.json", "summary.csv.gz"}
FINAL_OUTPUTS = {"Q1_paths.csv.gz", "Q2_paths.csv.gz", "summary.csv.gz", "evaluation_traces.npz",
                 "object_specification.json", "existing_development_context.json"}
REPRO_SOURCES = {"scripts/reproduce_locked_final_evaluation.py", "tests/test_reproduce_locked_final_evaluation.py"}
ALLOWED_MANIFEST_NORMALIZATION = ("completed_utc",)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def canonical_path(name):
    if not isinstance(name, str):
        raise ValueError("Canonical repository-relative copied path required")
    path = Path(name)
    if (not name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or name in (".", "..")):
        raise ValueError("Canonical repository-relative copied path required")
    if path == ORIGINAL or ORIGINAL in path.parents or path == OUT or OUT in path.parents:
        raise ValueError("Original final outputs and replay archive may never enter the copied dependency closure")
    return path


def merge_hashes(target, additions):
    for name, digest in additions.items():
        canonical_path(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Lowercase SHA256 digests required")
        if name in target and target[name] != digest:
            raise ValueError("Conflicting frozen dependency hash: " + name)
        target[name] = digest


def dependency_allowlist(prep, dev, audit, identity_hashes):
    """Pure transitive closure, excluding the not-yet-created final release."""
    if (len(prep["execution_sources_sha256"]) != 8 or len(prep["frozen_input_sha256"]) != 20
            or prep["development_execution_ready"] is not True or prep["final_execution_ready"] is not False
            or dev["status"] != "passed" or dev["final_scenarios_loaded"] is not False
            or audit["status"] != "passed"
            or audit["parent_manifest_sha256"] != identity_hashes[str(DEVELOPMENT / "manifest.json")]
            or dev["preparation_protocol_sha256"] != identity_hashes[str(PREPARATION)]
            or dev["object_specification_sha256"] != prep["object_specification_sha256"]):
        raise RuntimeError("Complete frozen preparation and passed bound development audit required")
    if (dev["sources"] != dev["sources_at_end"] or dev["sources"] != prep["execution_sources_sha256"]
            or dev["input_sha256"] != dev["input_sha256_at_end"] or dev["input_sha256"] != prep["frozen_input_sha256"]
            or audit["sources_and_artifacts_at_start"] != audit["sources_and_artifacts_at_end"]
            or set(dev["outputs_sha256"]) != DEV_OUTPUTS):
        raise RuntimeError("Development source/input/output inventory or independent audit closure changed")
    result = {}
    for mapping in (prep["execution_sources_sha256"], prep["frozen_input_sha256"],
                    audit["sources_and_artifacts_at_start"], identity_hashes,
                    {str(DEVELOPMENT / name): digest for name, digest in dev["outputs_sha256"].items()}):
        merge_hashes(result, mapping)
    return dict(sorted(result.items()))


def check_current(root, mapping):
    root = Path(root)
    for name, digest in mapping.items():
        canonical_path(name)
        if sha(root / name) != digest:
            raise RuntimeError("Current dependency differs from immutable allowlist: " + name)


def copy_dependencies(original_root, replica_root, mapping):
    original_root, replica_root = Path(original_root), Path(replica_root)
    check_current(original_root, mapping)
    for name, digest in mapping.items():
        path = replica_root / canonical_path(name)
        if path.exists():
            raise RuntimeError("Do not overwrite a replica dependency")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / name, path)
        if sha(path) != digest:
            raise RuntimeError("Copied dependency hash differs: " + name)
    if (replica_root / ORIGINAL).exists():
        raise RuntimeError("A final output directory must not be copied into the fresh replica")


def normalized_manifest(value):
    """Only the single predeclared completion timestamp may differ."""
    result = copy.deepcopy(value)
    for name in ALLOWED_MANIFEST_NORMALIZATION:
        stamp = result.get(name)
        if not isinstance(stamp, str):
            raise ValueError("Declared runtime timestamp must exist as a string")
        parsed = datetime.fromisoformat(stamp)
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Completion timestamp must be UTC")
        result.pop(name)
    return result


def compare_npz(left_path, right_path):
    rows = []; differences = []
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        left_names, right_names = set(left.files), set(right.files)
        if left_names != right_names:
            differences.append(dict(kind="npz_field_inventory", left_only=sorted(left_names - right_names), right_only=sorted(right_names - left_names)))
        for name in sorted(left_names & right_names):
            a, b = left[name], right[name]
            same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
            same_values = (bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))) if same_shape and same_dtype else False
            row = dict(field=name, shape_left=list(a.shape), shape_right=list(b.shape), dtype_left=str(a.dtype), dtype_right=str(b.dtype),
                       scalar_values=int(a.size), shape_equal=same_shape, dtype_equal=same_dtype, values_and_paired_NULL_equal=same_values)
            rows.append(row)
            if not (same_shape and same_dtype and same_values):
                differences.append(dict(kind="npz_array", **row))
    return dict(exact=not differences, fields_checked=len(rows), scalar_values_checked=sum(r["scalar_values"] for r in rows),
                container_bytes_equal=sha(left_path) == sha(right_path), arrays=rows, differences=differences)


def compare_outputs(original, replica):
    """Inspect every planned output; never stop on the first difference."""
    original, replica = Path(original), Path(replica)
    differences = []; files = {}; npz = None
    required = FINAL_OUTPUTS | {"manifest.json"}
    for folder, label in ((original, "original"), (replica, "replica")):
        names = {p.name for p in folder.iterdir() if p.is_file()} if folder.is_dir() else set()
        missing = required - names
        if missing:
            differences.append(dict(kind="missing_outputs", side=label, missing=sorted(missing)))
        entries = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
        if label == "replica" and entries - required:
            differences.append(dict(kind="unexpected_replica_outputs", names=sorted(entries - required)))
    for name in sorted(FINAL_OUTPUTS):
        a, b = original / name, replica / name
        if not a.is_file() or not b.is_file():
            continue
        exact_bytes = sha(a) == sha(b)
        row = dict(original_sha256=sha(a), replica_sha256=sha(b), byte_equal=exact_bytes)
        if name.endswith(".npz"):
            try:
                npz = compare_npz(a, b); row["array_exact"] = npz["exact"]
                differences.extend(npz["differences"])
            except Exception as exc:
                differences.append(dict(kind="artifact_comparison_error", output=name, error_type=type(exc).__name__, error=str(exc)))
        elif not exact_bytes:
            # Semantic/decoded comparison is explanatory only and does not
            # relax the preregistered byte criterion for CSV/JSON.
            try:
                if name.endswith(".json"):
                    row["JSON_semantic_equal"] = readj(a) == readj(b)
                elif name.endswith(".csv.gz"):
                    with gzip.open(a, "rb") as x, gzip.open(b, "rb") as y:
                        row["decoded_CSV_bytes_equal"] = x.read() == y.read()
            except Exception as exc:
                row["explanatory_comparison_error"] = dict(error_type=type(exc).__name__, error=str(exc))
            differences.append(dict(kind="nonmanifest_artifact_bytes", output=name, **row))
        files[name] = row
    manifest_equal = False
    a, b = original / "manifest.json", replica / "manifest.json"
    if a.is_file() and b.is_file():
        try:
            left, right = readj(a), readj(b)
            if set(left.get("outputs_sha256", {})) != FINAL_OUTPUTS or set(right.get("outputs_sha256", {})) != FINAL_OUTPUTS:
                differences.append(dict(kind="manifest_output_inventory", original=sorted(left.get("outputs_sha256", {})), replica=sorted(right.get("outputs_sha256", {}))))
            l, r = normalized_manifest(left), normalized_manifest(right)
            manifest_equal = l == r
            if not manifest_equal:
                fields = sorted(key for key in set(l) | set(r) if l.get(key) != r.get(key) or (key in l) != (key in r))
                differences.append(dict(kind="normalized_manifest", unequal_top_level_fields=fields))
        except Exception as exc:
            differences.append(dict(kind="manifest_comparison_error", error_type=type(exc).__name__, error=str(exc)))
    return dict(status="passed" if not differences else "differences", files=files, npz=npz,
        manifest_equal_after_declared_normalization=manifest_equal,
        normalized_manifest_keys=list(ALLOWED_MANIFEST_NORMALIZATION), differences=differences)


def isolated_command(python_executable, workspace):
    executable = Path(python_executable)
    if not executable.is_absolute():
        raise ValueError("Use the original venv's absolute Python executable, without resolving away its venv symlink")
    return [str(executable), "-I", str(Path(workspace) / "scripts/prepare_final_evaluation.py"), "--execute-final"]


def require_ready(root=ROOT):
    root = Path(root); plan, frozen = readj(root / PLAN), readj(root / SOURCE_FREEZE)
    if (plan.get("version") != "isolated-locked-final-reproduction-plan-v1"
            or plan.get("execution_requires_separate_source_freeze") is not True
            or frozen.get("version") != "isolated-locked-final-reproduction-source-freeze-v1"
            or frozen.get("execution_ready") is not True or frozen.get("plan_sha256") != sha(root / PLAN)
            or set(frozen.get("source_sha256", {})) != REPRO_SOURCES):
        raise RuntimeError("Predeclared plan and a separately bound frozen reproducer are required")
    check_current(root, frozen["source_sha256"])
    if (plan["original_final_directory"] != str(ORIGINAL) or plan["replay_archive_directory"] != str(OUT)
            or set(plan["predeclared_final_output_names"]) != FINAL_OUTPUTS
            or plan["manifest_normalization_allowed_top_keys"] != list(ALLOWED_MANIFEST_NORMALIZATION)
            or plan["python_executable"] != sys.executable):
        raise RuntimeError("Frozen replay paths, output inventory, normalization or original runtime changed")
    prep, dev, audit = readj(root / PREPARATION), readj(root / DEVELOPMENT / "manifest.json"), readj(root / DEVELOPMENT / "independent_audit.json")
    identity = {str(PREPARATION): sha(root / PREPARATION), str(DEVELOPMENT / "manifest.json"): sha(root / DEVELOPMENT / "manifest.json"),
                str(DEVELOPMENT / "independent_audit.json"): sha(root / DEVELOPMENT / "independent_audit.json")}
    allowlist = dependency_allowlist(prep, dev, audit, identity)
    if allowlist != plan["frozen_copy_allowlist_without_final_release_sha256"]:
        raise RuntimeError("Frozen transitive copy allowlist changed")
    check_current(root, allowlist)
    if not (root / RELEASE).is_file() or not (root / ORIGINAL / "manifest.json").is_file():
        raise RuntimeError("First locked final release and complete original final archive must exist first")
    release, original = readj(root / RELEASE), readj(root / ORIGINAL / "manifest.json")
    if (release.get("final_execution_ready") is not True or original.get("status") != "complete"
            or release.get("reproduction_plan_sha256") != sha(root / PLAN)
            or original.get("release_sha256") != sha(root / RELEASE)
            or original.get("preparation_protocol_sha256") != identity[str(PREPARATION)]
            or original.get("object_specification_sha256") != prep["object_specification_sha256"]
            or original.get("sources") != prep["execution_sources_sha256"] or original.get("sources_at_end") != original["sources"]
            or original.get("input_sha256") != prep["frozen_input_sha256"] or original.get("input_sha256_at_end") != original["input_sha256"]
            or original.get("training_ids_loaded") != list(range(1, 121)) or original.get("evaluation_ids_loaded") != list(range(121, 181))
            or original.get("final_scenarios_loaded") is not True or original.get("completed_final_array_load") is not True
            or original.get("training_development_arrays_sha256") != dev["development_arrays_sha256"]
            or (original.get("Q2_path_rows"), original.get("Q1_path_rows"), original.get("summary_rows")) != (240, 300, 9)
            or original.get("new_fits") != 0 or original.get("selection_changed") is not False or original.get("admission_changed") is not False
            or original.get("current_C_service") != "execution_review" or original.get("current_C_service_financials") is not None
            or set(original.get("outputs_sha256", {})) != FINAL_OUTPUTS):
        raise RuntimeError("Original complete final evaluation differs from the frozen release, source or unchanged decision contract")
    for name, digest in original["outputs_sha256"].items():
        if sha(root / ORIGINAL / name) != digest:
            raise RuntimeError("Original producer output changed")
    # The unchanged copied adapter also checks every release/dev/audit gate.
    merge_hashes(allowlist, {str(RELEASE): sha(root / RELEASE)})
    return plan, frozen, dict(sorted(allowlist.items())), original


def run(root=ROOT):
    root = Path(root); out = root / OUT
    if out.exists():
        raise RuntimeError("Never overwrite a replay archive, including a failed one")
    plan, frozen, allowlist, original = require_ready(root)
    original_before = {name: sha(root / ORIGINAL / name) for name in sorted(FINAL_OUTPUTS | {"manifest.json"})}
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status="incomplete", started_utc=datetime.now(timezone.utc).isoformat(),
        phase="all_original_complete_and_release_guards_passed", plan_sha256=sha(root / PLAN),
        source_freeze_sha256=sha(root / SOURCE_FREEZE), reproducer_sources=frozen["source_sha256"],
        original_final_producer_sha256_start=original_before, final_scenarios_loaded=False,
        same_source_same_runtime=True, independent_new_risk_validation=False)
    writej(out / "manifest.json", state); workspace = None
    try:
        workspace = Path(tempfile.mkdtemp(prefix="safeinvest-locked-final-replay-"))
        state.update(temporary_workspace=str(workspace), phase="copying_frozen_dependency_allowlist")
        writej(out / "manifest.json", state)
        copy_dependencies(root, workspace, allowlist)
        writej(out / "copy_allowlist.json", allowlist)
        command = isolated_command(sys.executable, workspace)
        env = dict(os.environ); env.update(PYTHONPATH="", PYTHONNOUSERSITE="1")
        writej(out / "launch.json", dict(command=command, cwd=str(workspace), python_executable=sys.executable,
            python_isolated_mode=True, PYTHONPATH="", PYTHONNOUSERSITE="1", original_final_outputs_copied=False))
        state.update(phase="running_unchanged_adapter_in_fresh_isolated_workspace", replica_execution_started=True,
            final_values_may_have_been_parsed=True)
        writej(out / "manifest.json", state)
        child = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True, timeout=plan["subprocess_timeout_seconds"])
        (out / "child_stdout.txt").write_text(child.stdout); (out / "child_stderr.txt").write_text(child.stderr)
        replica_output = workspace / ORIGINAL
        if replica_output.exists():
            shutil.copytree(replica_output, out / "replayed_outputs")
        if (replica_output / "manifest.json").is_file():
            try:
                state["final_scenarios_loaded"] = readj(replica_output / "manifest.json").get("final_scenarios_loaded", False)
            except (ValueError, OSError):
                state["child_manifest_parse_error"] = True
        state.update(child_returncode=child.returncode, phase="checking_all_artifacts_and_preserving_differences")
        copied_after = {name: sha(workspace / name) for name in allowlist}
        writej(out / "copied_dependencies_at_end.json", copied_after)
        comparison = compare_outputs(root / ORIGINAL, replica_output) if replica_output.is_dir() else dict(status="differences", differences=[dict(kind="no_replica_outputs")])
        if child.returncode:
            comparison["status"] = "differences"; comparison["differences"].append(dict(kind="child_exit", returncode=child.returncode))
        if copied_after != allowlist:
            comparison["status"] = "differences"; comparison["differences"].append(dict(kind="copied_dependency_changed", paths=sorted(name for name in allowlist if copied_after.get(name) != allowlist[name])))
        original_after = {name: sha(root / ORIGINAL / name) for name in original_before}
        if original_before != original_after:
            comparison["status"] = "differences"; comparison["differences"].append(dict(kind="original_producer_archive_changed"))
        writej(out / "comparison.json", comparison)
        check_current(root, allowlist); check_current(root, frozen["source_sha256"])
        if sha(root / PLAN) != state["plan_sha256"] or sha(root / SOURCE_FREEZE) != state["source_freeze_sha256"]:
            raise RuntimeError("Frozen replay plan/source receipt changed during execution")
        if comparison["status"] == "passed":
            shutil.rmtree(workspace); removed = True
        else:
            removed = False
        state.update(status=comparison["status"], completed_utc=datetime.now(timezone.utc).isoformat(), phase="finished",
            copied_files=len(allowlist), copied_sources_and_inputs_unchanged=copied_after == allowlist,
            original_final_producer_sha256_end=original_after, original_producer_archive_unchanged=original_before == original_after,
            temporary_workspace_removed=removed, differences=len(comparison["differences"]),
            normalization_allowed=list(ALLOWED_MANIFEST_NORMALIZATION),
            new_fits=0, selection_changed=False, admission_changed=False,
            scope="Isolated replay of the same already complete frozen final evaluation; no new market sample, independent numerical method or risk certification")
        state["outputs_sha256"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file() and path != out / "manifest.json"}
        writej(out / "manifest.json", state)
        print(json.dumps(dict(status=state["status"], copied_files=len(allowlist), differences=state["differences"],
            manifest_sha256=sha(out / "manifest.json"), temporary_workspace_removed=removed)))
    except BaseException as exc:
        (out / "error.log").write_text(traceback.format_exc())
        if isinstance(exc, subprocess.TimeoutExpired):
            for name, content in (("child_stdout.txt", exc.stdout), ("child_stderr.txt", exc.stderr)):
                if content is not None:
                    (out / name).write_bytes(content if isinstance(content, bytes) else content.encode())
        if workspace is not None and (workspace / ORIGINAL).exists() and not (out / "replayed_outputs").exists():
            shutil.copytree(workspace / ORIGINAL, out / "replayed_outputs")
        replica_manifest = workspace / ORIGINAL / "manifest.json" if workspace is not None else None
        if replica_manifest is not None and replica_manifest.is_file():
            try:
                state["final_scenarios_loaded"] = readj(replica_manifest).get("final_scenarios_loaded", False)
            except (ValueError, OSError):
                state["child_manifest_parse_error"] = True
        state.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(), exception_type=type(exc).__name__,
            exception_message=str(exc), temporary_workspace_removed=False, automatic_retry=False)
        writej(out / "manifest.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run only after frozen reproducer sources, original final release and complete original final evaluation")
    run()
