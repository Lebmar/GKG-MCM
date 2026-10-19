"""All five pre-existing partition repeats of the unchanged six-policy learner.

Source/synthetic preparation precedes any guarded development-only execution.
Each repeat has120 original held-out paths, never600 independent observations.
Own-training Q1 fits are reused job by job; no full120 target is substituted.
The imported producer helpers make this a same-source stability experiment,
not a second independent accounting implementation or new blind validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
from safeinvest.data import load_paths
from experimental_execution.state_budget_execution import simulate_state_budget
from experimental_execution.review_pending import simulate_paired
from experimental_execution.state_budget_selection import CANDIDATES, select_inner, summarize_records
from run_state_budget_rebuild import rows_from_arm, archive_arrays, validate_job, jsonable, null_rows, squeeze

PROTOCOL = Path("docs/state_budget_stability_protocol.json")
FREEZE = Path("docs/state_budget_stability_source_freeze.json")
PRIMARY_PROTOCOL = Path("docs/state_budget_rebuild_protocol.json")
PRIMARY_FREEZE = Path("docs/state_budget_rebuild_source_freeze.json")
JOBS = Path("docs/refit_stability_protocol.json")
PARENT = Path("results/model_review/refit_stability_v1")
FITS = PARENT / "static_refits.json.gz"
DEFAULT_OUT = ROOT / "results/state_budget_stability_v1"
H, L, COST = 30, .4, .0025
REFS = ("same_target_B", "same_target_annual", "exact_thirds_annual")
REQUIRED_SOURCES = {
    "scripts/run_state_budget_stability.py", "tests/test_state_budget_stability.py",
    "scripts/run_state_budget_rebuild.py", "tests/test_state_budget_rebuild_runner.py",
    "research/experimental_execution/state_budget_execution.py", "tests/test_state_budget_execution.py",
    "research/experimental_execution/forecast_ensemble.py", "tests/test_forecast_ensemble.py",
    "research/experimental_execution/state_budget_selection.py", "tests/test_state_budget_selection.py",
    "research/experimental_execution/review_pending.py", "src/safeinvest/core.py", "src/safeinvest/data.py",
    "research/safeinvest_research/metrics.py",
}
REQUIRED_INPUTS = {str(JOBS), str(FITS), str(PARENT / "manifest.json"), str(PARENT / "independent_audit.json"),
                   str(PRIMARY_PROTOCOL), str(PRIMARY_FREEZE), "data/SafeInvest_Synthetic_Data.csv"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def check_hashes(root, mapping):
    for name, digest in mapping.items():
        path = Path(name)
        if (not isinstance(name, str) or path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or not isinstance(digest, str) or len(digest) != 64 or sha(root / path) != digest):
            raise RuntimeError("Frozen stability dependency changed: " + str(name))


def validate_stability_jobs(jobs):
    """Every fixed repeat covers120 IDs once outside,96 once inside each fit."""
    universe = set(range(1, 121))
    if len(jobs) != 125 or len({j["job_id"] for j in jobs}) != 125:
        raise ValueError("Require all125 distinct pre-existing stability jobs")
    for job in jobs:
        if (type(job["repeat_id"]) is not int or job["repeat_id"] not in range(1, 6)
                or type(job["outer_fold"]) is not int or job["outer_fold"] not in range(1, 6)
                or job["parent_context_id"] != 100*job["repeat_id"]+job["outer_fold"]
                or job["parent_context"] != f"stability_r{job['repeat_id']}_outer{job['outer_fold']}"):
            raise ValueError("Repeat/outer/context identity changed")
        for ids in (job["training_ids"], job["evaluation_ids"]):
            if (not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i not in universe for i in ids)):
                raise ValueError("Unique original development integer IDs1-120 required")
    for repeat in range(1, 6):
        outside = []
        for fold in range(1, 6):
            group = [j for j in jobs if j["repeat_id"] == repeat and j["outer_fold"] == fold]
            inner = [j for j in group if j["stage"] == "inner"]
            refits = [j for j in group if j["stage"] == "refit"]
            if len(group) != 5 or len(inner) != 4 or len(refits) != 1 or {j["dynamic_inner_fold"] for j in inner} != {1, 2, 3, 4}:
                raise ValueError("Each of25 contexts needs four inner jobs and one refit")
            refit = refits[0]; train = set(refit["training_ids"]); test = set(refit["evaluation_ids"])
            if len(train) != 96 or len(test) != 24 or train & test or train | test != universe:
                raise ValueError("Every outer refit must partition96/24 whole scenarios")
            outside += refit["evaluation_ids"]; inside = []
            for job in inner:
                tr, ev = set(job["training_ids"]), set(job["evaluation_ids"])
                if len(tr) != 72 or len(ev) != 24 or tr & ev or tr | ev != train:
                    raise ValueError("Inner job must partition the same outer training cohort72/24")
                inside += job["evaluation_ids"]
            if sorted(inside) != sorted(train):
                raise ValueError("Each context's inner OOF must cover96 original IDs once")
        if sorted(outside) != list(range(1, 121)):
            raise ValueError("Each repeat's outer OOF must cover120 original IDs once")


def validate_fits(jobs, fits):
    if set(fits) != {j["job_id"] for j in jobs}:
        raise ValueError("Every job needs its own original static bundle, with no substituted extra fit")
    for job in jobs:
        bundle = fits[job["job_id"]]
        validate_job(job, bundle)
        if bundle["refit"]["outer_test_ids"] != job["evaluation_ids"]:
            raise ValueError("Own static refit evaluation identities differ")


def freeze_gate(root=ROOT):
    """All hashes, identity metadata and readiness checks precede load_paths."""
    root = Path(root); p, frozen = readj(root / PROTOCOL), readj(root / FREEZE)
    if (p.get("version") != "state-budget-five-repeat-stability-v1" or p.get("execution_ready") is not True
            or frozen.get("execution_ready") is not True or frozen.get("protocol_sha256") != sha(root / PROTOCOL)
            or p.get("development_ids") != list(range(1, 121)) or p.get("allow_final") is not False
            or p.get("candidates") != list(CANDIDATES) or p.get("repeat_ids") != list(range(1, 6))
            or p.get("job_protocol") != str(JOBS) or p.get("static_refits") != str(FITS)):
        raise RuntimeError("Freeze the complete development-only five-repeat protocol before market parsing")
    if not REQUIRED_SOURCES <= set(frozen.get("source_sha256", {})) or not REQUIRED_INPUTS <= set(frozen.get("input_sha256", {})):
        raise RuntimeError("Complete runtime/test/parent source and input closure required")
    check_hashes(root, frozen["source_sha256"]); check_hashes(root, frozen["input_sha256"])
    primary, primary_frozen = readj(root / PRIMARY_PROTOCOL), readj(root / PRIMARY_FREEZE)
    if (primary_frozen.get("execution_ready") is not True
            or primary_frozen.get("protocol_sha256") != sha(root / PRIMARY_PROTOCOL)
            or primary["candidates"] != list(CANDIDATES) or primary["service_alpha"] != p["service_alpha"]):
        raise RuntimeError("The exact primary learner contract must already be frozen")
    for key in ("source_sha256", "input_sha256"):
        check_hashes(root, primary_frozen[key])
    parent, audit = readj(root / PARENT / "manifest.json"), readj(root / PARENT / "independent_audit.json")
    bound = audit["sources_and_artifacts_at_start"]
    if (parent["status"] != "complete" or parent["final_scenarios_loaded"] is not False
            or parent["scenario_ids_loaded"] != list(range(1, 121)) or audit["status"] != "passed"
            or audit["final_scenarios_loaded"] is not False or bound != audit["sources_and_artifacts_at_end"]
            or bound.get(str(PARENT / "manifest.json")) != sha(root / PARENT / "manifest.json")
            or bound.get(str(FITS)) != sha(root / FITS) or bound.get(str(JOBS)) != sha(root / JOBS)
            or parent["outputs_sha256"].get(FITS.name) != sha(root / FITS)):
        raise RuntimeError("Immutable audited original stability partitions/static refits are required")
    jobs = readj(root / JOBS)["jobs"]; validate_stability_jobs(jobs)
    with gzip.open(root / FITS, "rt") as stream:
        fits = json.load(stream)
    validate_fits(jobs, fits)
    return p, frozen, jobs, fits, parent["development_arrays_sha256"]


def empty_repeat(repeat):
    return dict(repeat_id=repeat, arrays={}, groups=[], rows=[], selections={}, initial_witnesses=[], completed_jobs=[], summaries=[])


def evaluate_repeat(state, jobs, fits, market, inflation, *, service_alpha, progress=lambda x: None):
    """The same entire inner selector and refit rule on one fixed repeat."""
    repeat = state["repeat_id"]; arrays, groups, rows = state["arrays"], state["groups"], state["rows"]
    def save(group_id, arm, ids, meta):
        retained, omitted = archive_arrays(arm, meta["kind"])
        groups.append(dict(group_id=group_id, scenario_ids=list(map(int, ids)), array_names=list(retained),
                           omitted_diagnostic_arrays=omitted, repeat_id=repeat, **meta))
        arrays.update({group_id+"__"+key: value for key, value in retained.items()})
    def common(job, policy):
        return dict(job_id=job["job_id"], repeat_id=repeat, outer_fold=job["outer_fold"], context_id=job["parent_context_id"], policy=policy)
    def case_reason(bundle):
        return "q1_review" if bundle["q2_case"]["status"] == "risk_review" else "not_balanced_case"
    def references(job, w):
        ev = np.asarray(job["evaluation_ids"])-1; result = {}
        for policy, target, armname in ((REFS[0], w, "pending_hold"), (REFS[1], w, "annual"), (REFS[2], np.ones(3)/3, "annual")):
            arm = None if target is None else squeeze(simulate_paired(market[ev], inflation[ev], target, L, horizon=H, cost_rate=COST)[armname])
            metadata = dict(kind="reference", **common(job, policy), training_ids=job["training_ids"],
                target=None if target is None else np.asarray(target).tolist(), status="unavailable" if arm is None else "funded")
            # repeat_id lives once in save's shared metadata.
            metadata.pop("repeat_id")
            save(job["job_id"]+"__"+policy, arm, job["evaluation_ids"], metadata)
            result[policy] = null_rows(job["evaluation_ids"], common(job, policy), case_reason(fits[job["job_id"]])) if arm is None else rows_from_arm(arm, job["evaluation_ids"], common(job, policy))
        return result
    def candidate(job, spec, w, kind):
        tr, ev = np.asarray(job["training_ids"])-1, np.asarray(job["evaluation_ids"])-1
        policy = "state_budget_learner" if kind == "outer_selected" else spec["candidate_id"]
        result = None if w is None else simulate_state_budget(market[ev], inflation[ev], market[tr], inflation[tr], w,
            L=L, horizon=H, cost_rate=COST, release_fraction=spec["release_fraction"], forecast_rule=spec["forecast_rule"],
            training_ids=job["training_ids"], actual_ids=job["evaluation_ids"], support_mode="calendar")
        status = case_reason(fits[job["job_id"]]) if result is None else result["status"]
        arm = None if result is None else result["arm"]
        if result is not None:
            state["initial_witnesses"].append(dict(job_id=job["job_id"], candidate_id=spec["candidate_id"], scope=kind, status=status, witness=result["initial_witness"]))
        metadata = dict(kind=kind, **common(job, policy), training_ids=job["training_ids"],
            target=None if w is None else w.tolist(), candidate=spec, status=status)
        metadata.pop("repeat_id")
        save(job["job_id"]+"__"+spec["candidate_id"]+"__"+kind, arm, job["evaluation_ids"], metadata)
        return null_rows(job["evaluation_ids"], common(job, policy), status) if arm is None else rows_from_arm(arm, job["evaluation_ids"], common(job, policy))
    for fold in range(1, 6):
        context_jobs = [j for j in jobs if j["repeat_id"] == repeat and j["outer_fold"] == fold]
        inner = {s["candidate_id"]: [] for s in CANDIDATES}; baseline = []
        for job in sorted((j for j in context_jobs if j["stage"] == "inner"), key=lambda j: j["dynamic_inner_fold"]):
            w = validate_job(job, fits[job["job_id"]]); reference = references(job, w)
            baseline += reference[REFS[0]]
            for refrows in reference.values():
                rows.extend(dict(row, scope="inner_reference") for row in refrows)
            for spec in CANDIDATES:
                outcomes = candidate(job, spec, w, "inner_candidate")
                inner[spec["candidate_id"]] += outcomes
                rows.extend(dict(row, scope="inner_candidate") for row in outcomes)
            state["completed_jobs"].append(job["job_id"]); progress(job)
        chosen = select_inner(inner, baseline, alpha=service_alpha)
        job = next(j for j in context_jobs if j["stage"] == "refit")
        w = validate_job(job, fits[job["job_id"]]); reference = references(job, w)
        for refrows in reference.values():
            rows.extend(dict(row, scope="outer_reference") for row in refrows)
        if chosen["selected_candidate_id"] is None:
            outcomes = null_rows(job["evaluation_ids"], common(job, "state_budget_learner"), "no_inner_feasible_candidate")
            metadata = dict(kind="outer_selected", policy="state_budget_learner", job_id=job["job_id"], outer_fold=fold,
                context_id=job["parent_context_id"], training_ids=job["training_ids"], target=None if w is None else w.tolist(),
                candidate=None, status="no_inner_feasible_candidate")
            save(job["job_id"]+"__state_budget_learner__outer_selected", None, job["evaluation_ids"], metadata)
        else:
            spec = next(s for s in CANDIDATES if s["candidate_id"] == chosen["selected_candidate_id"])
            outcomes = candidate(job, spec, w, "outer_selected")
        chosen.update(refit_job_id=job["job_id"], repeat_id=repeat, outer_fold=fold, context_id=job["parent_context_id"],
            inner_job_ids=[j["job_id"] for j in context_jobs if j["stage"] == "inner"],
            final_status="funded" if all(r["assigned"] for r in outcomes) else "execution_review",
            selected_result_summary=summarize_records(outcomes), training_ids=job["training_ids"], evaluation_ids=job["evaluation_ids"])
        state["selections"][str(fold)] = chosen
        rows.extend(dict(row, scope="outer_selected") for row in outcomes)
        state["completed_jobs"].append(job["job_id"]); progress(job)
    if len(state["completed_jobs"]) != 25 or len(groups) != 200 or len(rows) != 4800:
        raise RuntimeError("A complete repeat must retain25jobs,200groups and4800original-ID path rows")
    for policy in ("state_budget_learner", *REFS):
        scope = "outer_selected" if policy == "state_budget_learner" else "outer_reference"
        cohort = [r for r in rows if r["scope"] == scope and r["policy"] == policy]
        if sorted(r["scenario_id"] for r in cohort) != list(range(1, 121)):
            raise RuntimeError("Each repeat/policy must preserve exactly120 unique original OOF IDs")
        state["summaries"].append(dict(repeat_id=repeat, scope=scope, policy=policy, **summarize_records(cohort)))
    return state


def write_repeat(folder, state, status):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state["rows"]).to_csv(folder / "path_metrics.csv.gz", index=False, compression=dict(method="gzip", mtime=0))
    pd.DataFrame(state["summaries"]).to_csv(folder / "summary.csv.gz", index=False, compression=dict(method="gzip", mtime=0))
    writej(folder / "selections.json", state["selections"]); writej(folder / "trace_groups.json", state["groups"])
    writej(folder / "initial_witnesses.json", state["initial_witnesses"])
    np.savez_compressed(folder / "traces.npz", **state["arrays"])
    oversized = [p.name for p in folder.iterdir() if p.is_file() and p.stat().st_size > 50*1024*1024]
    if oversized:
        raise RuntimeError("Predeclared per-repeat archive exceeded50MiB; retain evidence, no silent field deletion: "+str(oversized))
    files = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}
    writej(folder / "manifest.json", dict(status=status, repeat_id=state["repeat_id"], completed_jobs=state["completed_jobs"],
        groups=len(state["groups"]), path_rows=len(state["rows"]), summaries=len(state["summaries"]),
        array_fields=len(state["arrays"]), outputs_sha256=files, final_scenarios_loaded=False,
        original_market_scenarios=120, independent_additional_scenarios=0))


def run(output=DEFAULT_OUT, root=ROOT):
    root, output = Path(root), Path(output).resolve()
    if output.exists():
        raise RuntimeError("Never overwrite an existing complete, incomplete or failed stability attempt")
    p, frozen, jobs, fits, expected_fingerprint = freeze_gate(root)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {**frozen["source_sha256"], **frozen["input_sha256"]}
    manifest = dict(status="incomplete", started_utc=datetime.now(timezone.utc).isoformat(), phase="before_market_parser",
        protocol_sha256=sha(root / PROTOCOL), source_freeze_sha256=sha(root / FREEZE), sources_and_inputs_at_start=snapshot,
        final_scenarios_loaded=False, market_value_access_started=False, completed_repeats=[], completed_jobs=[],
        independence="Five partitions of the same120 paths; never600 independent markets or a best-repeat selection")
    writej(output / "manifest.json", manifest); state = None; completed_summaries = []
    try:
        manifest.update(phase="guarded_development_load", market_value_access_started=True); writej(output / "manifest.json", manifest)
        market, inflation, ids = load_paths(root / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
        digest = hashlib.sha256()
        for array in (ids, market, inflation):
            digest.update(np.ascontiguousarray(array).tobytes())
        if ids.tolist() != list(range(1, 121)) or digest.hexdigest() != expected_fingerprint:
            raise RuntimeError("Guarded original development identities/arrays changed")
        manifest.update(development_ids_loaded=ids.tolist(), development_arrays_sha256=digest.hexdigest())
        def progress(job):
            manifest.update(phase="complete_policy_jobs", current_repeat=job["repeat_id"], current_context=job["parent_context_id"])
            manifest["completed_jobs"].append(job["job_id"]); writej(output / "manifest.json", manifest)
            print(json.dumps(dict(completed_job=job["job_id"], repeat_id=job["repeat_id"])), flush=True)
        for repeat in range(1, 6):
            state = empty_repeat(repeat)
            evaluate_repeat(state, jobs, fits, market, inflation, service_alpha=p["service_alpha"], progress=progress)
            check_hashes(root, snapshot)
            write_repeat(output / f"repeat_{repeat:02d}", state, "complete")
            completed_summaries += state["summaries"]; manifest["completed_repeats"].append(repeat)
            manifest.update(current_repeat=None, phase="repeat_archived"); writej(output / "manifest.json", manifest)
            state = None
        pd.DataFrame(completed_summaries).to_csv(output / "repeat_summary.csv.gz", index=False, compression=dict(method="gzip", mtime=0))
        if len(manifest["completed_jobs"]) != 125 or len(completed_summaries) != 20:
            raise RuntimeError("All125 jobs and20 separate-repeat summaries are required")
        end = {name: sha(root / name) for name in snapshot}
        if end != snapshot or sha(root / PROTOCOL) != manifest["protocol_sha256"] or sha(root / FREEZE) != manifest["source_freeze_sha256"]:
            raise RuntimeError("Frozen stability source/input/protocol changed during execution")
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), phase="complete",
            sources_and_inputs_at_end=end, repeat_count=5, training_jobs=125, outer_fits=25, repeat_summary_rows=len(completed_summaries),
            candidate_count=6, independent_additional_scenarios=0, new_final_reads=0, best_repeat_selected=False,
            outputs_sha256={str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file() and path != output / "manifest.json"})
        writej(output / "manifest.json", manifest)
    except BaseException as exc:
        (output / "failure.txt").write_text(traceback.format_exc())
        if state is not None:
            try:
                write_repeat(output / f"repeat_{state['repeat_id']:02d}_partial", state, "failed_partial")
            except BaseException:
                (output / "failure_archival_error.txt").write_text(traceback.format_exc())
        manifest.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__,
            error=str(exc), automatic_retry=False, source_or_model_changed=False)
        writej(output / "manifest.json", manifest)
        raise
    print(json.dumps(dict(status="complete", repeats=5, jobs=125, output=str(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.execute:
        run(args.output)
    else:
        print(json.dumps(dict(source_only=True, development_only=True, repeats=5, jobs=125, market_values_parsed=False)))
