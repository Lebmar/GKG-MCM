"""Synthetic isolation, exact-comparison and archival gates; no market data."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("locked_final_reproducer", ROOT / "scripts/reproduce_locked_final_evaluation.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value)
    else:
        m.writej(path, value)
    return m.sha(path)


def output_fixture(folder, stamp="2026-10-19T01:00:00+00:00"):
    folder.mkdir(parents=True, exist_ok=True)
    for name in m.FINAL_OUTPUTS:
        if name.endswith(".csv.gz"):
            (folder / name).write_bytes(gzip.compress(b"id,value\n121,2\n180,3\n", mtime=0))
        elif name.endswith(".json"):
            m.writej(folder / name, {"synthetic": True, "financials": None})
        else:
            np.savez_compressed(folder / name, ids=np.array([121, 180]), state=np.array([[1., np.nan], [2., 3.]]))
    manifest = dict(status="complete", completed_utc=stamp, outputs_sha256={n: m.sha(folder / n) for n in sorted(m.FINAL_OUTPUTS)})
    m.writej(folder / "manifest.json", manifest)
    return manifest


def gate_fixture(root):
    """Complete tiny fake closure; paths and IDs are synthetic labels only."""
    sources = {f"synthetic/s{i}.py": put(root, f"synthetic/s{i}.py", f"# source {i}\n") for i in range(8)}
    inputs = {f"synthetic/i{i}.txt": put(root, f"synthetic/i{i}.txt", f"input {i}\n") for i in range(20)}
    prep = dict(development_execution_ready=True, final_execution_ready=False, execution_sources_sha256=sources,
                frozen_input_sha256=inputs, object_specification_sha256="f" * 64)
    prep_sha = put(root, m.PREPARATION, prep)
    outputs = {name: put(root, m.DEVELOPMENT / name, b"synthetic evidence\n") for name in m.DEV_OUTPUTS}
    dev = dict(status="passed", final_scenarios_loaded=False, preparation_protocol_sha256=prep_sha,
        object_specification_sha256=prep["object_specification_sha256"], sources=sources, sources_at_end=sources,
        input_sha256=inputs, input_sha256_at_end=inputs, outputs_sha256=outputs, development_arrays_sha256="e" * 64)
    dev_sha = put(root, m.DEVELOPMENT / "manifest.json", dev)
    audit_map = {**sources, **inputs, **{str(m.DEVELOPMENT / n): d for n, d in outputs.items()},
                 str(m.DEVELOPMENT / "manifest.json"): dev_sha}
    audit = dict(status="passed", parent_manifest_sha256=dev_sha,
        sources_and_artifacts_at_start=audit_map, sources_and_artifacts_at_end=audit_map)
    audit_sha = put(root, m.DEVELOPMENT / "independent_audit.json", audit)
    identity = {str(m.PREPARATION): prep_sha, str(m.DEVELOPMENT / "manifest.json"): dev_sha,
                str(m.DEVELOPMENT / "independent_audit.json"): audit_sha}
    allowlist = m.dependency_allowlist(prep, dev, audit, identity)
    plan = dict(version="isolated-locked-final-reproduction-plan-v1", execution_requires_separate_source_freeze=True,
        original_final_directory=str(m.ORIGINAL), replay_archive_directory=str(m.OUT),
        predeclared_final_output_names=sorted(m.FINAL_OUTPUTS), manifest_normalization_allowed_top_keys=["completed_utc"],
        python_executable=sys.executable, subprocess_timeout_seconds=1,
        frozen_copy_allowlist_without_final_release_sha256=allowlist)
    plan_sha = put(root, m.PLAN, plan)
    source_hashes = {name: put(root, name, "# synthetic reproducer source\n") for name in m.REPRO_SOURCES}
    put(root, m.SOURCE_FREEZE, dict(version="isolated-locked-final-reproduction-source-freeze-v1",
        execution_ready=True, plan_sha256=plan_sha, source_sha256=source_hashes))
    release_sha = put(root, m.RELEASE, dict(final_execution_ready=True, reproduction_plan_sha256=plan_sha))
    original = output_fixture(root / m.ORIGINAL)
    original.update(release_sha256=release_sha, preparation_protocol_sha256=prep_sha,
        object_specification_sha256=prep["object_specification_sha256"], sources=sources, sources_at_end=sources,
        input_sha256=inputs, input_sha256_at_end=inputs, training_ids_loaded=list(range(1, 121)),
        evaluation_ids_loaded=list(range(121, 181)), final_scenarios_loaded=True, completed_final_array_load=True,
        training_development_arrays_sha256=dev["development_arrays_sha256"], Q2_path_rows=240, Q1_path_rows=300,
        summary_rows=9, new_fits=0, selection_changed=False, admission_changed=False,
        current_C_service="execution_review", current_C_service_financials=None)
    put(root, m.ORIGINAL / "manifest.json", original)
    return prep, dev, audit, identity


class LockedFinalReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_copy_paths_reject_escape_original_and_replay_outputs(self):
        for value in ("../outside", "/absolute", "docs/../x", "./x", 12,
                      str(m.ORIGINAL / "evaluation_traces.npz"), str(m.OUT / "manifest.json")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.canonical_path(value)

    def test_hash_merge_rejects_conflict_and_malformed_digest(self):
        mapping = {}; m.merge_hashes(mapping, {"docs/x": "a" * 64})
        for additions in ({"docs/x": "b" * 64}, {"docs/z": "A" * 64}):
            with self.assertRaises(ValueError):
                m.merge_hashes(mapping, additions)

    def test_transitive_closure_includes_every_development_output_and_audit_artifact(self):
        prep, dev, audit, identity = gate_fixture(self.root)
        closure = m.dependency_allowlist(prep, dev, audit, identity)
        self.assertTrue(set(audit["sources_and_artifacts_at_start"]) <= set(closure))
        self.assertTrue({str(m.DEVELOPMENT / n) for n in m.DEV_OUTPUTS} <= set(closure))
        for key in ("final_scenarios_loaded", "status"):
            bad = copy.deepcopy(dev); bad[key] = True if key.startswith("final") else "incomplete"
            with self.assertRaises(RuntimeError):
                m.dependency_allowlist(prep, bad, audit, identity)

    def test_conflicting_audit_start_end_or_parent_binding_is_rejected(self):
        prep, dev, audit, identity = gate_fixture(self.root)
        for key in ("parent_manifest_sha256", "sources_and_artifacts_at_end"):
            bad = copy.deepcopy(audit); bad[key] = "bad" if key.startswith("parent") else {}
            with self.assertRaises(RuntimeError):
                m.dependency_allowlist(prep, dev, bad, identity)

    def test_copy_checks_bytes_and_never_overwrites(self):
        mapping = {"docs/fake.txt": put(self.root, "docs/fake.txt", "source\n")}
        replica = self.root / "replica"; m.copy_dependencies(self.root, replica, mapping)
        self.assertEqual((replica / "docs/fake.txt").read_text(), "source\n")
        with self.assertRaises(RuntimeError):
            m.copy_dependencies(self.root, replica, mapping)
        (self.root / "docs/fake.txt").write_text("changed\n")
        with self.assertRaises(RuntimeError):
            m.check_current(self.root, mapping)

    def test_only_UTC_completion_timestamp_is_normalized(self):
        a = dict(completed_utc="2026-10-19T00:00:00+00:00", elapsed_seconds=1, result=None)
        b = dict(a, completed_utc="2026-10-19T02:00:00Z")
        self.assertEqual(m.normalized_manifest(a), m.normalized_manifest(b))
        b["elapsed_seconds"] = 2
        self.assertNotEqual(m.normalized_manifest(a), m.normalized_manifest(b))
        for stamp in (None, "2026-10-19T01:00:00", "2026-10-19T01:00:00+08:00"):
            with self.assertRaises(ValueError):
                m.normalized_manifest(dict(a, completed_utc=stamp))

    def test_npz_checks_dtype_shape_values_NULL_and_inventory(self):
        a, b = self.root / "a.npz", self.root / "b.npz"
        np.savez(a, x=np.array([1., np.nan]), i=np.array([121, 180], dtype=np.int64))
        np.savez(b, x=np.array([1., np.nan]), i=np.array([121, 180], dtype=np.int64))
        self.assertTrue(m.compare_npz(a, b)["exact"])
        np.savez(b, x=np.array([[1., 2.]]), i=np.array([121, 180], dtype=np.int32), extra=np.zeros(1))
        report = m.compare_npz(a, b)
        self.assertFalse(report["exact"]); self.assertEqual(len(report["differences"]), 3)
        np.savez(b, x=np.array([1., 0.]), i=np.array([121, 180], dtype=np.int64))
        self.assertFalse(m.compare_npz(a, b)["exact"])

    def test_all_planned_artifacts_match_with_only_timestamp_difference(self):
        a, b = self.root / "a", self.root / "b"
        output_fixture(a); shutil.copytree(a, b)
        manifest = m.readj(b / "manifest.json"); manifest["completed_utc"] = "2026-10-19T03:00:00+00:00"
        m.writej(b / "manifest.json", manifest)
        report = m.compare_outputs(a, b)
        self.assertEqual(report["status"], "passed"); self.assertEqual(set(report["files"]), m.FINAL_OUTPUTS)

    def test_missing_unexpected_corrupt_and_semantically_equal_bytes_all_stay_differences(self):
        a, b = self.root / "a", self.root / "b"
        output_fixture(a); shutil.copytree(a, b)
        (b / "Q1_paths.csv.gz").unlink(); (b / "extra_folder").mkdir()
        (b / "evaluation_traces.npz").write_bytes(b"not npz")
        value = m.readj(b / "object_specification.json")
        (b / "object_specification.json").write_text(json.dumps(value))
        (b / "manifest.json").write_text("broken JSON")
        report = m.compare_outputs(a, b); kinds = {x["kind"] for x in report["differences"]}
        self.assertTrue({"missing_outputs", "unexpected_replica_outputs", "artifact_comparison_error",
                         "nonmanifest_artifact_bytes", "manifest_comparison_error"} <= kinds)
        self.assertTrue(report["files"]["object_specification.json"]["JSON_semantic_equal"])
        self.assertIn("summary.csv.gz", report["files"])

    def test_isolated_command_preserves_absolute_venv_interpreter(self):
        command = m.isolated_command("/synthetic/.venv/bin/python", "/synthetic/replica")
        self.assertEqual(command, ["/synthetic/.venv/bin/python", "-I", "/synthetic/replica/scripts/prepare_final_evaluation.py", "--execute-final"])
        with self.assertRaises(ValueError):
            m.isolated_command("python", "/synthetic/replica")

    def test_source_freeze_and_original_complete_are_required_before_child(self):
        gate_fixture(self.root); m.require_ready(self.root)
        frozen = m.readj(self.root / m.SOURCE_FREEZE); frozen["execution_ready"] = False
        m.writej(self.root / m.SOURCE_FREEZE, frozen)
        with mock.patch.object(m.subprocess, "run") as child, self.assertRaises(RuntimeError):
            m.run(self.root)
        child.assert_not_called(); self.assertFalse((self.root / m.OUT).exists())

    def test_changed_release_binding_or_original_metadata_is_rejected(self):
        gate_fixture(self.root)
        original_path = self.root / m.ORIGINAL / "manifest.json"
        original = m.readj(original_path)
        for changes in (dict(status="failed"), dict(selection_changed=True), dict(current_C_service_financials=0),
                        dict(training_development_arrays_sha256="wrong"), dict(Q2_path_rows=239)):
            m.writej(original_path, dict(original, **changes))
            with self.assertRaises(RuntimeError):
                m.require_ready(self.root)
        m.writej(original_path, original)
        release = m.readj(self.root / m.RELEASE); release["reproduction_plan_sha256"] = "wrong"
        m.writej(self.root / m.RELEASE, release)
        with self.assertRaises(RuntimeError):
            m.require_ready(self.root)

    def test_existing_replay_archive_rejects_even_before_gates(self):
        (self.root / m.OUT).mkdir(parents=True)
        with mock.patch.object(m, "require_ready") as ready, self.assertRaises(RuntimeError):
            m.run(self.root)
        ready.assert_not_called()

    def test_timeout_keeps_logs_failed_attempt_and_disallows_retry(self):
        gate_fixture(self.root)
        workspace = self.root / "isolated"; workspace.mkdir()
        timeout = subprocess.TimeoutExpired(["synthetic"], 1, output=b"partial stdout", stderr=b"partial stderr")
        with mock.patch.object(m.tempfile, "mkdtemp", return_value=str(workspace)), \
             mock.patch.object(m.subprocess, "run", side_effect=timeout), self.assertRaises(subprocess.TimeoutExpired):
            m.run(self.root)
        out = self.root / m.OUT; state = m.readj(out / "manifest.json")
        self.assertEqual(state["status"], "failed"); self.assertFalse(state["final_scenarios_loaded"])
        self.assertTrue(state["final_values_may_have_been_parsed"])
        self.assertEqual((out / "child_stdout.txt").read_bytes(), b"partial stdout")
        self.assertEqual((out / "child_stderr.txt").read_bytes(), b"partial stderr")
        self.assertTrue(workspace.exists()); self.assertTrue((out / "error.log").exists())
        with self.assertRaises(RuntimeError):
            m.run(self.root)

    def test_synthetic_success_copies_only_closure_and_preserves_outputs(self):
        gate_fixture(self.root); original_manifest_sha = m.sha(self.root / m.ORIGINAL / "manifest.json")
        workspace = self.root / "isolated"; workspace.mkdir()
        def fake_child(command, **kwargs):
            self.assertEqual(kwargs["cwd"], workspace); self.assertEqual(kwargs["env"]["PYTHONPATH"], "")
            self.assertFalse((workspace / m.ORIGINAL).exists())
            shutil.copytree(self.root / m.ORIGINAL, workspace / m.ORIGINAL)
            return subprocess.CompletedProcess(command, 0, "synthetic complete\n", "")
        with mock.patch.object(m.tempfile, "mkdtemp", return_value=str(workspace)), \
             mock.patch.object(m.subprocess, "run", side_effect=fake_child):
            m.run(self.root)
        state = m.readj(self.root / m.OUT / "manifest.json")
        self.assertEqual(state["status"], "passed"); self.assertTrue(state["final_scenarios_loaded"])
        self.assertFalse(workspace.exists()); self.assertEqual(m.sha(self.root / m.ORIGINAL / "manifest.json"), original_manifest_sha)
        self.assertEqual(set(m.readj(self.root / m.OUT / "copy_allowlist.json")) & {str(m.ORIGINAL)}, set())
        self.assertTrue((self.root / m.OUT / "replayed_outputs/evaluation_traces.npz").exists())


if __name__ == "__main__":
    unittest.main()
