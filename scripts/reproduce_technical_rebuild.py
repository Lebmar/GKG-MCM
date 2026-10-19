"""Frozen development-only producers replayed in separate fresh source roots.

This checks implementation reproducibility with the same data/runtime/code.
It is not an independent market test, new policy selection or risk guarantee.
No original producer source or output is modified. A failed attempt and its
workspace are retained; a passed workspace may be removed after comparison.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("docs/technical_rebuild_reproduction_plan.json")
FREEZE = Path("docs/technical_rebuild_reproduction_source_freeze.json")
OUT = Path("results/technical_rebuild_reproduction_v1")
OWN_SOURCES = {"scripts/reproduce_technical_rebuild.py", "tests/test_reproduce_technical_rebuild.py"}
FIRST_PROFILES = ["v2_canonical", "v2b_canonical", "v2_stability", "v2b_stability"]
TIMESTAMPS = ("started_utc", "completed_utc")
PACKAGES = ("safeinvest", "experimental_execution", "safeinvest_research")


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


def relative(name):
    if not isinstance(name, str):
        raise ValueError("Canonical repository-relative path required")
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
        raise ValueError("Canonical repository-relative path required: " + repr(name))
    return path


def merge(target, additions):
    if not isinstance(additions, dict):
        raise ValueError("Hash mapping required")
    for name, digest in additions.items():
        relative(name)
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("Lowercase SHA256 required")
        if name in target and target[name] != digest:
            raise ValueError("Conflicting frozen dependency: " + name)
        target[name] = digest


def current(root, mapping):
    for name, digest in mapping.items():
        if sha(Path(root) / relative(name)) != digest:
            raise RuntimeError("Immutable dependency changed: " + name)


def validate_plan(plan):
    profiles = plan.get("profiles", [])
    if (plan.get("version") != "technical-rebuild-isolated-reproduction-plan-v1"
            or plan.get("execution_ready") is not True or plan.get("allow_final") is not False
            or plan.get("development_ids") != list(range(1, 121))
            or plan.get("python_executable") != ".venv/bin/python"
            or plan.get("normalized_root_manifest_fields") != list(TIMESTAMPS)
            or not 4 <= len(profiles) <= 6 or [p.get("profile_id") for p in profiles[:4]] != FIRST_PROFILES):
        raise RuntimeError("Separately frozen complete development-only replay plan required")
    names, outputs = set(), set()
    for profile in profiles:
        name = profile["profile_id"]
        if not isinstance(name, str) or re.fullmatch("[a-z][a-z0-9_]{0,63}", name) is None or name in names:
            raise ValueError("Unique simple profile IDs required")
        names.add(name)
        for field in ("script", "protocol", "source_freeze", "original_output"):
            relative(profile[field])
        if profile["original_output"] in outputs:
            raise ValueError("Each profile requires a distinct original output")
        outputs.add(profile["original_output"])
    return profiles


def loader_source_gate(path):
    tree = ast.parse(Path(path).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "load_paths"]
    if len(calls) != 1 or len(calls[0].args) < 2 or ast.unparse(calls[0].args[1]) != "range(1, 121)":
        raise RuntimeError("Only the fixed development-ID loader may be replayed")
    keywords = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
    if keywords.get("allow_final") is not False:
        raise RuntimeError("Final loading must remain disabled")


def dependency_closure(root, profile):
    """Expand source-freeze dependencies, including canonical parents of stability.

    Only dependencies of the unchanged producer are copied. Its own original
    manifest/outputs are compared outside the fresh root, never seeded there.
    Necessary predecessor identity receipts are allowed by the frozen maps.
    """
    root = Path(root); result = {}; seen = set()
    original = relative(profile["original_output"])
    top = readj(root / relative(profile["source_freeze"]))
    if top.get("execution_ready") is not True or top.get("protocol_sha256") != sha(root / relative(profile["protocol"])):
        raise RuntimeError("Original producer must already have a frozen ready protocol")
    merge(result, {profile["source_freeze"]: sha(root / profile["source_freeze"]), profile["protocol"]: sha(root / profile["protocol"])})
    queue = [profile["source_freeze"]]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name); frozen = readj(root / relative(name))
        for field in ("source_sha256", "input_sha256"):
            mapping = frozen.get(field)
            if not isinstance(mapping, dict) or not mapping:
                raise RuntimeError("Complete nested source/input freeze required: " + name)
            merge(result, mapping)
        for dependency in sorted(result):
            if dependency.endswith("_source_freeze.json") and dependency not in seen and dependency not in queue:
                if sha(root / dependency) != result[dependency]:
                    raise RuntimeError("Changed nested source-freeze metadata")
                queue.append(dependency)
    for name in result:
        path = relative(name)
        if path == original or original in path.parents:
            raise RuntimeError("Never seed the producer's own target outputs: " + name)
        if path == OUT or OUT in path.parents:
            raise RuntimeError("Replay artifacts may not enter dependency closure")
    current(root, result)
    if profile["script"] not in result:
        raise RuntimeError("The exact producer script must be frozen")
    loader_source_gate(root / profile["script"])
    protocol = readj(root / profile["protocol"])
    if protocol.get("allow_final") is not False or protocol.get("development_ids") != list(range(1, 121)):
        raise RuntimeError("All copied producer protocols remain development only")
    return dict(sorted(result.items()))


def declared_outputs(root, profile, dependencies):
    folder = Path(root) / relative(profile["original_output"]); manifest = readj(folder / "manifest.json")
    top = readj(Path(root) / profile["source_freeze"])
    direct = {}; merge(direct, top["source_sha256"]); merge(direct, top["input_sha256"])
    if (manifest.get("status") != "complete" or manifest.get("final_scenarios_loaded") is not False
            or manifest.get("new_final_reads") != 0 or manifest.get("development_ids_loaded") != list(range(1, 121))
            or manifest.get("protocol_sha256") != sha(Path(root) / profile["protocol"])
            or manifest.get("source_freeze_sha256") != sha(Path(root) / profile["source_freeze"])
            or manifest.get("sources_and_inputs_at_start") != direct
            or manifest.get("sources_and_inputs_at_end") != direct):
        raise RuntimeError("Complete unchanged original development producer required")
    mapping = {}; merge(mapping, manifest.get("outputs_sha256"))
    if not mapping or "manifest.json" in mapping:
        raise RuntimeError("Nonempty declared non-root output inventory required")
    current(folder, mapping)
    return manifest, dict(sorted(mapping.items()))


def required_inputs(root, profiles):
    """Metadata/hash-only construction, usable by root when making the freeze."""
    result = {}; prepared = []
    for profile in profiles:
        dependencies = dependency_closure(root, profile)
        manifest, outputs = declared_outputs(root, profile, dependencies)
        merge(result, dependencies)
        merge(result, {profile["original_output"] + "/manifest.json": sha(Path(root) / profile["original_output"] / "manifest.json")})
        merge(result, {profile["original_output"] + "/" + name: digest for name, digest in outputs.items()})
        prepared.append(dict(profile=profile, dependencies=dependencies, original_manifest=manifest, outputs=outputs))
    return dict(sorted(result.items())), prepared


def ready(root=ROOT):
    root = Path(root); plan, frozen = readj(root / PLAN), readj(root / FREEZE)
    if (frozen.get("version") != "technical-rebuild-isolated-reproduction-source-freeze-v1"
            or frozen.get("execution_ready") is not True or frozen.get("plan_sha256") != sha(root / PLAN)
            or set(frozen.get("source_sha256", {})) != OWN_SOURCES):
        raise RuntimeError("Separately frozen replay source/test/plan required")
    current(root, frozen["source_sha256"])
    current(root, frozen["input_sha256"])
    profiles = validate_plan(plan); inputs, prepared = required_inputs(root, profiles)
    if frozen["input_sha256"] != inputs:
        raise RuntimeError("Replay freeze must bind every complete original/dependency/output hash")
    return plan, frozen, prepared


def copy_dependencies(original_root, workspace, mapping, original_output):
    original_root, workspace = Path(original_root), Path(workspace)
    forbidden = relative(original_output)
    for name in mapping:
        path = relative(name)
        if path == forbidden or forbidden in path.parents:
            raise RuntimeError("Producer target output cannot be a copied dependency")
    current(original_root, mapping)
    for name, digest in mapping.items():
        path = relative(name)
        if path == forbidden or forbidden in path.parents:
            raise RuntimeError("Producer target output cannot be a copied dependency")
        destination = workspace / path
        if destination.exists():
            raise RuntimeError("Never overwrite a fresh-root dependency")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / path, destination)
        if sha(destination) != digest:
            raise RuntimeError("Copied dependency differs")
    if (workspace / forbidden).exists():
        raise RuntimeError("A profile's target directory must not exist before replay")


def normalize_manifest(value):
    result = copy.deepcopy(value)
    for key in TIMESTAMPS:
        stamp = result.get(key)
        if not isinstance(stamp, str):
            raise ValueError("Both declared UTC runtime timestamps must be strings")
        parsed = datetime.fromisoformat(stamp)
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Declared timestamps must be UTC")
        result.pop(key)
    return result


def compare_npz(left_path, right_path):
    arrays = []; differences = []
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        ln, rn = set(left.files), set(right.files)
        if len(left.files) != len(ln) or len(right.files) != len(rn):
            differences.append(dict(kind="duplicate_field_names"))
        if ln != rn:
            differences.append(dict(kind="field_inventory", original_only=sorted(ln-rn), replay_only=sorted(rn-ln)))
        for name in sorted(ln & rn):
            a, b = left[name], right[name]; shape = a.shape == b.shape; dtype = a.dtype == b.dtype
            equal = bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)) if shape and dtype else False
            row = dict(field=name, original_shape=list(a.shape), replay_shape=list(b.shape), original_dtype=str(a.dtype), replay_dtype=str(b.dtype),
                       scalar_values=int(a.size), shape_equal=shape, dtype_equal=dtype, values_and_paired_NaN_equal=equal)
            arrays.append(row)
            if not (shape and dtype and equal):
                differences.append(dict(kind="array", **row))
    return dict(exact=not differences, fields_checked=len(arrays), scalar_values_checked=sum(x["scalar_values"] for x in arrays),
                container_bytes_equal=sha(left_path) == sha(right_path), arrays=arrays, differences=differences)


def compare_outputs(original, replay, declared):
    original, replay = Path(original), Path(replay); files = {}; differences = []
    expected = set(declared) | {"manifest.json"}
    actual = {str(p.relative_to(replay)) for p in replay.rglob("*") if p.is_file()} if replay.is_dir() else set()
    if actual != expected:
        differences.append(dict(kind="replay_inventory", missing=sorted(expected-actual), unexpected=sorted(actual-expected)))
    for name, digest in sorted(declared.items()):
        relative(name); a, b = original / name, replay / name
        if not a.is_file() or not b.is_file():
            differences.append(dict(kind="missing_output", output=name)); continue
        ah, bh = sha(a), sha(b); row = dict(original_sha256=ah, replay_sha256=bh, bytes_equal=ah == bh, original_manifest_hash_equal=ah == digest)
        if ah != digest:
            differences.append(dict(kind="original_artifact_changed", output=name))
        if name.endswith(".npz"):
            row["npz"] = compare_npz(a, b)
            if not row["npz"]["exact"]:
                differences.append(dict(kind="npz_values", output=name))
        elif ah != bh:
            differences.append(dict(kind="artifact_bytes", output=name))
        files[name] = row
    manifest_equal = False
    if (original / "manifest.json").is_file() and (replay / "manifest.json").is_file():
        left, right = readj(original / "manifest.json"), readj(replay / "manifest.json")
        if left.get("outputs_sha256") != declared or right.get("outputs_sha256") != declared:
            differences.append(dict(kind="root_manifest_output_hashes"))
        a, b = normalize_manifest(left), normalize_manifest(right); manifest_equal = a == b
        if not manifest_equal:
            differences.append(dict(kind="root_manifest", differing_fields=sorted(k for k in set(a)|set(b) if (k in a)!=(k in b) or a.get(k)!=b.get(k))))
    else:
        differences.append(dict(kind="missing_root_manifest"))
    return dict(status="passed" if not differences else "differences", declared_files=len(declared), files=files,
                root_manifest_equal_after_two_timestamps=manifest_equal, root_manifest_normalized_fields=list(TIMESTAMPS), differences=differences)


BOOTSTRAP = r'''
import importlib, importlib.metadata, json, os, pathlib, runpy, sys, traceback
workspace, script, target, receipt, expected_venv = map(pathlib.Path, sys.argv[1:])
workspace = workspace.resolve()
sys.dont_write_bytecode = True
sys.path[:0] = [str(workspace / p) for p in ("src", "research", "scripts")]
packages = ("safeinvest", "experimental_execution", "safeinvest_research")
def origins():
    result = {}
    for package in packages:
        importlib.import_module(package)
    for name, module in list(sys.modules.items()):
        if any(name == p or name.startswith(p + ".") for p in packages):
            paths = ([str(module.__file__)] if getattr(module, "__file__", None) else []) + list(map(str, getattr(module, "__path__", [])))
            if not paths or any(not pathlib.Path(p).resolve().is_relative_to(workspace) for p in paths):
                raise RuntimeError("A domain package escaped the isolated source root: " + name + repr(paths))
            result[name] = paths
    return result
meta = dict(python_executable=sys.executable, python_version=sys.version, prefix=sys.prefix, base_prefix=sys.base_prefix,
    isolated=sys.flags.isolated, no_user_site=sys.flags.no_user_site, dont_write_bytecode=sys.dont_write_bytecode,
    runtime_versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")}, cwd=os.getcwd(), workspace=str(workspace))
try:
    if not sys.flags.isolated or not sys.flags.no_user_site or pathlib.Path(sys.prefix) != expected_venv:
        raise RuntimeError("Original virtualenv and isolated Python are required")
    meta["package_origins_before"] = origins()
    sys.argv = [str(script), "--execute", "--output", str(target)]
    runpy.run_path(str(script), run_name="__main__")
    meta["package_origins_after"] = origins()
    meta["status"] = "passed"
except BaseException as exc:
    meta.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    raise
finally:
    receipt.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
'''


def isolated_command(python, workspace, profile, runtime_receipt, venv):
    python = Path(python)
    if not python.is_absolute():
        raise ValueError("Keep the original virtualenv's absolute interpreter path")
    # Intentionally do not resolve the venv executable symlink.
    workspace = Path(workspace)
    return [str(python), "-I", "-c", BOOTSTRAP, str(workspace), str(workspace/relative(profile["script"])),
            str(workspace/relative(profile["original_output"])), str(runtime_receipt), str(venv)]


def clean_environment(environment):
    result = dict(environment)
    result.pop("PYTHONPATH", None); result.pop("PYTHONHOME", None)
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return result


def run(output=None, root=ROOT):
    root = Path(root).absolute(); output = Path(output).absolute() if output is not None else root/OUT
    if output.exists():
        raise RuntimeError("Never overwrite or silently retry a replay attempt")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="incomplete", started_utc=datetime.now(timezone.utc).isoformat(), phase="before_replay_gates",
                    original_market_scenarios=120, independent_additional_scenarios=0, final_scenarios_loaded=False,
                    profiles=[], workspaces=[], automatic_retry=False)
    writej(output/"manifest.json", manifest); workspace = None
    try:
        plan, frozen, prepared = ready(root)
        snapshot = {}; merge(snapshot, frozen["source_sha256"]); merge(snapshot, frozen["input_sha256"])
        merge(snapshot, {str(PLAN):sha(root/PLAN),str(FREEZE):sha(root/FREEZE)})
        manifest.update(phase="guarded_source_replay", plan_sha256=sha(root/PLAN), source_freeze_sha256=sha(root/FREEZE), sources_and_original_artifacts_at_start=snapshot)
        writej(output/"manifest.json", manifest)
        python = root/plan["python_executable"]
        for item in prepared:
            profile = item["profile"]; name = profile["profile_id"]; folder = output/name; folder.mkdir()
            workspace = Path(tempfile.mkdtemp(prefix="gkg-technical-"+name+"-"))
            manifest["workspaces"].append(dict(profile_id=name,path=str(workspace),status="retained_incomplete")); writej(output/"manifest.json", manifest)
            copy_dependencies(root, workspace, item["dependencies"], profile["original_output"])
            writej(folder/"copied_dependencies.json",item["dependencies"])
            command = isolated_command(python, workspace, profile, folder/"runtime.json", root/".venv")
            writej(folder/"invocation.json",dict(argv=command,cwd=str(workspace),removed_environment_keys=["PYTHONHOME","PYTHONPATH"],
                forced_environment=dict(PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")))
            with (folder/"producer.log").open("wb") as log:
                result = subprocess.run(command,cwd=workspace,env=clean_environment(os.environ),stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode != 0:
                raise RuntimeError("Isolated producer failed: "+name+" exit="+str(result.returncode))
            runtime = readj(folder/"runtime.json")
            if runtime.get("status") != "passed":
                raise RuntimeError("Isolated package-origin checks failed")
            current(root,item["dependencies"]); current(workspace,item["dependencies"])
            comparison = compare_outputs(root/profile["original_output"],workspace/profile["original_output"],item["outputs"])
            writej(folder/"comparison.json",comparison)
            if comparison["status"] != "passed":
                raise RuntimeError("Replay differences retained: "+name)
            manifest["profiles"].append(dict(profile_id=name,status="passed",copied_dependencies=len(item["dependencies"]),
                compared_artifacts=len(item["outputs"]),comparison_sha256=sha(folder/"comparison.json"),runtime_sha256=sha(folder/"runtime.json"),producer_log_sha256=sha(folder/"producer.log")))
            shutil.rmtree(workspace); manifest["workspaces"][-1]["status"]="removed_after_exact_comparison"; workspace=None
            writej(output/"manifest.json", manifest)
        current(root,snapshot)
        manifest.update(status="passed",phase="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
            sources_and_original_artifacts_at_end={name:sha(root/name) for name in snapshot},
            profile_count=len(prepared),new_fits_for_selection=0,new_final_reads=0,
            interpretation="Same frozen producer/runtime/data reproduction only; no independent market validation or policy improvement.",
            outputs_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p!=output/"manifest.json"})
        writej(output/"manifest.json",manifest)
    except BaseException as exc:
        (output/"failure.txt").write_text(traceback.format_exc())
        manifest.update(status="failed",phase="failed_evidence_retained",failed_utc=datetime.now(timezone.utc).isoformat(),
                        error_type=type(exc).__name__,error=str(exc),retained_current_workspace=None if workspace is None else str(workspace),
                        failure_sha256=sha(output/"failure.txt"),automatic_retry=False)
        writej(output/"manifest.json",manifest)
        raise
    print(json.dumps(dict(status="passed",profiles=len(manifest["profiles"]),manifest_sha256=sha(output/"manifest.json"))),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.execute:run(args.output)
    else:print(json.dumps(dict(source_only=True,execution_requires_separate_freeze=True,profiles=FIRST_PROFILES,market_values_parsed=False)))
