"""Replay frozen development execution studies in an isolated temporary copy.

Each stage sees the original frozen parents. After comparing its generated
artifacts, restore that stage's frozen archive in the copy before the next
stage. This prevents fresh manifest timestamps from invalidating child hashes.
The original archives are never used as output destinations.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/model_review/execution_stage_reproduction_v1"
STAGES = [
    ("review_pending_execution_v1", "run_review_pending_execution.py"),
    ("fan_rollout_execution_v1", "run_fan_rollout_execution.py"),
    ("pathwise_fan_execution_v1", "run_pathwise_fan_execution.py"),
    ("pathwise_cost_stress_v1", "run_pathwise_cost_stress.py"),
    ("risk_priced_fan_execution_v1", "run_risk_priced_fan_execution.py"),
    ("bounded_target_updates_v1", "run_bounded_target_updates.py"),
    ("selected_controller_cost_stress_v1", "run_selected_controller_cost_stress.py"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identical_artifact(original, generated):
    """Require byte equality, or exact named-array equality for ZIP timestamps."""
    if sha(original) == sha(generated):
        return {"comparison": "byte_exact", "arrays": 0}
    if original.suffix != ".npz":
        raise AssertionError(f"Nonidentical artifact: {original.name}")
    with np.load(original, allow_pickle=False) as left, np.load(generated, allow_pickle=False) as right:
        assert set(left.files) == set(right.files), original.name
        for key in left.files:
            a, b = left[key], right[key]
            assert a.dtype == b.dtype and a.shape == b.shape, (original.name, key)
            if a.dtype.kind in "fc":
                equal = np.array_equal(a, b, equal_nan=True)
            else:
                equal = np.array_equal(a, b)
            assert equal, (original.name, key)
        return {"comparison": "exact_arrays", "arrays": len(left.files)}


def frozen_snapshot(stages):
    paths = {Path(__file__).resolve(), ROOT / "tests/test_execution_stage_reproduction.py"}
    for name, _ in stages:
        folder = ROOT / "results/model_review" / name
        manifest = json.loads((folder / "manifest.json").read_text())
        audit = json.loads((folder / "independent_audit.json").read_text())
        assert audit["status"] == "passed", name
        assert manifest.get("final_scenarios_loaded") is False, name
        ids = manifest.get("scenario_ids_loaded", manifest.get("scenario_ids", []))
        assert ids == list(range(1, 121)), (name, "development scenario identities")
        paths.update(p for p in folder.iterdir() if p.is_file())
        for document in [manifest, audit]:
            for key, values in document.items():
                if not isinstance(values, dict) or key.startswith("output"):
                    continue
                for relative, expected in values.items():
                    candidate = ROOT / relative
                    if isinstance(expected, str) and len(expected) == 64 and candidate.is_file():
                        assert sha(candidate) == expected, (name, key, relative)
                        paths.add(candidate)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def replay_stage(copy_root, name, script):
    copy_root = copy_root.resolve()
    assert copy_root != ROOT.resolve(), "Replay must use a disposable copy"
    original = ROOT / "results/model_review" / name
    target = copy_root / "results/model_review" / name
    assert target.resolve().is_relative_to(copy_root), "Replay output escaped its disposable copy"
    expected = json.loads((original / "manifest.json").read_text())["outputs_sha256"]
    # Removing only this stage in the disposable copy also detects outputs that
    # a runner accidentally depends on instead of recomputing.
    shutil.rmtree(target)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(copy_root / "src"), str(copy_root / "research")])
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(copy_root / "scripts" / script)], cwd=copy_root,
                            env=environment, capture_output=True, text=True)
    clean_log = (result.stdout + result.stderr).replace(str(copy_root), "<isolated-copy>").replace(str(ROOT), "<original-workspace>")
    if result.returncode:
        raise RuntimeError(f"Replay failed for {name}:\n{clean_log[-8000:]}")
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["final_scenarios_loaded"] is False, name
    assert set(manifest["outputs_sha256"]) == set(expected), name
    comparisons = {}
    for relative, digest in expected.items():
        assert sha(original / relative) == digest, (name, relative, "original hash")
        assert sha(target / relative) == manifest["outputs_sha256"][relative], (name, relative, "generated hash")
        comparisons[relative] = identical_artifact(original / relative, target / relative)
    # Child protocols intentionally refer to the frozen parent manifest/audit,
    # rather than to a new timestamped copy of an otherwise identical replay.
    shutil.rmtree(target)
    shutil.copytree(original, target)
    return {"stage": name, "seconds": time.perf_counter() - started,
            "artifact_comparisons": comparisons, "status": "passed"}, clean_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.parent == ROOT / "results/model_review" or not output.is_relative_to(ROOT), "Use a fresh top-level research receipt or an external directory"
    assert not output.exists(), "Choose a new reproduction output directory; existing receipts are preserved"
    started = time.perf_counter()
    before = frozen_snapshot(STAGES)
    receipts, logs = [], {}
    with tempfile.TemporaryDirectory(prefix="safeinvest-development-replay-") as temporary:
        copy_root = Path(temporary)
        for name in ["src", "research", "scripts", "tests", "docs", "data"]:
            shutil.copytree(ROOT / name, copy_root / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "results/model_review", copy_root / "results/model_review",
                        ignore=shutil.ignore_patterns("research_checkpoint_v1", "execution_stage_reproduction_v1"))
        for name, script in STAGES:
            receipt, log = replay_stage(copy_root, name, script)
            receipts.append(receipt); logs[name + ".log"] = log
            print(json.dumps({"stage": name, "status": "passed", "seconds": receipt["seconds"]}), flush=True)
    after = frozen_snapshot(STAGES)
    assert before == after, "Original frozen study files changed during isolated replay"
    output.mkdir(parents=True)
    for name, log in logs.items(): (output / name).write_text(log)
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "elapsed_seconds": time.perf_counter() - started, "stages": receipts,
              "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
              "outputs_sha256": {name: sha(output / name) for name in logs},
              "final_scenarios_loaded": False,
              "scope": "Same-source deterministic replay of seven development execution studies; exact outputs, original archives unchanged",
              "limitations": ["This is reproducibility evidence, not an independent execution algorithm or independent market data",
                              "Complete original scenario files are copied as bytes; guarded runners parse only scenarios1-120 with allow_final=False",
                              "Parent archives are restored between stages to preserve frozen manifest/audit hash contracts",
                              "Replay manifests have new time metadata and are deliberately excluded from artifact equality; no financial numerical tolerance is allowed"]}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    label = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(json.dumps({"status": "passed", "stages": len(receipts), "output": str(label)}), flush=True)


if __name__ == "__main__":
    main()
