#!/usr/bin/env python3
"""Bridge existing full-development cost evidence; never fit or replay a policy.

Execution requires a root-frozen protocol.  Base NPZ identity is established
before any nonbase cost table is parsed.  Rejected-candidate research remains
distinct from the current pre-investment Review service decision.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/current_candidate_cost_bridge_protocol.json"
OUT = ROOT / "results/model_review/current_candidate_cost_bridge_v1"
JOINT = ROOT / "results/model_review/joint_liquidity_refit_v1"
OLD = ROOT / "results/model_review/selected_controller_cost_stress_v1"
CURRENT = "c0__joint_tie_set_refit_learner"
HISTORICAL = "base__full_development__C__fixed_target_learner"
SHARED_KERNELS = (
    "research/experimental_execution/risk_priced_fan_rollout.py",
    "research/experimental_execution/fan_rollout.py",
    "research/experimental_execution/review_pending.py",
    "src/safeinvest/core.py",
    "research/safeinvest_research/metrics.py",
)
FINANCIAL = (
    "log_growth", "cagr", "mdd", "volatility", "terminal_real",
    "nominal_fees", "nominal_moved", "trade_fraction", "breach", "excess",
    "first_crossing",
)
POLICY_MAP = {
    "fixed_target_learner": "joint_tie_set_refit_learner",
    "annual_then_pending": "annual_then_pending",
    "equal_thirds_annual": "equal_thirds_annual",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def exact_array(left, right, name: str) -> dict:
    """Strict dtype/shape/value identity, with paired NaNs explicitly allowed."""
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError(f"Array dtype/shape changed: {name}")
    same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
    if not same:
        raise ValueError(f"Array values differ: {name}")
    return {"field": name, "shape": list(a.shape), "dtype": str(a.dtype),
            "values": int(a.size), "exact_values_and_NULL_mask": True}


def array_fields(archive, prefix: str) -> dict:
    tag = prefix + "__"
    return {key[len(tag):]: archive[key] for key in archive.files if key.startswith(tag)}


def pair_arrays(current: dict, old: dict, *, expected_fields: list[str],
                explicit_metadata: dict | None = None) -> list[dict]:
    if sorted(current) != sorted(expected_fields):
        raise ValueError("Current trace field inventory changed")
    alternatives = explicit_metadata or {}
    checked = []
    for field in sorted(current):
        if field in old:
            other = old[field]
            provenance = "old archived same field"
        elif field in alternatives:
            other = alternatives[field]
            provenance = "explicitly declared old protocol metadata; absent old NPZ field"
        else:
            raise ValueError(f"Missing historical bridge field: {field}")
        checked.append({**exact_array(current[field], other, field),
                        "historical_provenance": provenance})
    return checked


def finite_or_none(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    answer = float(value)
    if not np.isfinite(answer):
        raise ValueError("Unexpected nonfinite CSV financial value")
    return answer


def compare_csv_paths(current_rows: list[dict], old_rows: list[dict]) -> dict:
    def indexed(rows):
        result = {}
        for row in rows:
            sid = int(row["scenario_id"])
            if sid in result:
                raise ValueError("Duplicate CSV scenario")
            result[sid] = row
        if sorted(result) != list(range(1, 121)):
            raise ValueError("Full120 scenario inventory changed")
        return result
    a, b = indexed(current_rows), indexed(old_rows)
    maximum = 0.
    for sid in a:
        for field in FINANCIAL:
            left, right = finite_or_none(a[sid][field]), finite_or_none(b[sid][field])
            if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
                if type(left) is not type(right) or left != right:
                    raise ValueError(f"CSV NULL/bool mismatch: {sid}/{field}")
            else:
                maximum = max(maximum, abs(left - right))
                if not np.isclose(left, right, atol=2e-10, rtol=5e-13):
                    raise ValueError(f"CSV serialization mismatch: {sid}/{field}")
    return {"paths": 120, "fields": list(FINANCIAL), "maximum_absolute_error": maximum,
            "numeric_tolerance": {"atol": 2e-10, "rtol": 5e-13},
            "note": "CSV decimal parsing tolerance only; all archived NPZ bridge arrays use exact identity"}


def summarize_paths(rows: list[dict]) -> dict:
    if sorted(int(r["scenario_id"]) for r in rows) != list(range(1, 121)):
        raise ValueError("Every fixed cost/policy needs exactly all original120 IDs")
    def values(field):
        a = np.array([finite_or_none(row[field]) for row in rows], float)
        if not np.isfinite(a).all():
            raise ValueError("Research financial results unexpectedly unavailable")
        return a
    growth, mdd = values("cagr"), values("mdd")
    return {"n": 120, "mean_log_growth": float(values("log_growth").mean()),
            "median_cagr": float(np.median(growth)),
            "p10_cagr": float(np.quantile(growth, .1, method="inverted_cdf")),
            "median_mdd": float(np.median(mdd)),
            "p90_mdd": float(np.quantile(mdd, .9, method="inverted_cdf")),
            "breach_count": int(values("breach").sum()),
            "mean_nominal_fees": float(values("nominal_fees").mean()),
            "mean_trade_fraction": float(values("trade_fraction").mean())}


def main() -> None:
    protocol_digest = sha(PROTOCOL)
    protocol = read_json(PROTOCOL)
    if protocol.get("execution_ready") is not True or not protocol.get("execution_frozen_utc"):
        raise RuntimeError("Bridge protocol must be frozen by root before numerical comparison")
    snapshot = {**protocol["input_sha256"], **protocol["source_sha256"]}
    for name, digest in snapshot.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Frozen dependency changed: {name}")
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite a bridge result directory")
    current_manifest, old_manifest = read_json(JOINT / "manifest.json"), read_json(OLD / "manifest.json")
    for directory in (JOINT, OLD):
        if read_json(directory / "independent_audit.json")["status"] != "passed":
            raise RuntimeError("Both parent numerical audits must have passed")
    for name in SHARED_KERNELS:
        expected = current_manifest["sources"][name]
        if (old_manifest["sources"][name] != expected
                or current_manifest["sources_at_end"][name] != expected
                or old_manifest["sources_at_end"][name] != expected
                or sha(ROOT / name) != expected):
            raise RuntimeError(f"Shared execution kernel mismatch: {name}")
    if current_manifest["development_arrays_sha256"] != old_manifest["development_array_sha256"]:
        raise RuntimeError("Guarded parent development array fingerprints differ")
    groups = read_json(JOINT / "trace_groups.json")["actual"]
    current_context = read_json(JOINT / "context_selections.json")["full_development"]
    plan = read_json(OLD / "frozen_controller_plan.json")["full_development"]
    registry = read_json(ROOT / "docs/main_model_registry.json")
    group = groups[CURRENT]
    ids = list(range(1, 121))
    if (group["scenario_ids"] != ids or current_context["training_ids"] != ids or plan["training_ids"] != ids
            or plan["evaluation_ids"] != ids or group["selected_candidate_order"] != 4
            or group["metadata"]["horizon"] != 30 or group["metadata"]["loss_tolerance"] != .4):
        raise RuntimeError("Full candidate identity or original H/L/fan changed")
    chosen = plan["learners"]["fixed_target_learner"]
    if (chosen["selected_controller_order"] != 4 or chosen["annual_price"] != .03
            or chosen["updates_enabled"] is not False or plan["menu"][1]["weights"] != [.45,.05,.50]
            or registry["current_C_execution"]["final_status"] != "execution_review"
            or registry["current_C_execution"]["original_target"] != [.45,.05,.50]
            or registry["current_C_execution"]["original_selected_eta"] != .03):
        raise RuntimeError("Historical candidate/current rejected candidate metadata differ")
    bridge = {}
    # No raw market data, policy simulator, nonbase traces or cost table used here.
    with np.load(JOINT / "actual_traces.npz", allow_pickle=False) as current, np.load(OLD / "actual_traces.npz", allow_pickle=False) as old:
        for policy, current_policy in POLICY_MAP.items():
            current_prefix = "c0__" + current_policy
            old_prefix = "base__full_development__C__" + policy
            a, b = array_fields(current, current_prefix), array_fields(old, old_prefix)
            exact_array(np.array(groups[current_prefix]["scenario_ids"], dtype=np.int64), b["scenario_ids"], "scenario_ids")
            alternative = None
            if policy != "fixed_target_learner":
                target = plan["menu"][1]["weights"] if policy == "annual_then_pending" else [1/3]*3
                alternative = {"targets": np.asarray(target, dtype=np.float64)}
            bridge[policy] = pair_arrays(a, b, expected_fields=protocol["expected_trace_fields"][policy], explicit_metadata=alternative)
    # Only after exact base-array bridge passes may all historical cost rows be read.
    with (JOINT / "client_paths.csv").open() as handle:
        current_rows = list(csv.DictReader(handle))
    with (OLD / "client_paths.csv").open() as handle:
        old_rows = list(csv.DictReader(handle))
    csv_bridge = {}
    for old_policy, current_policy in POLICY_MAP.items():
        a = [r for r in current_rows if r["context"] == "full_development" and r["policy"] == current_policy]
        b = [r for r in old_rows if r["context"] == "full_development" and r["policy"] == old_policy and r["cost_label"] == "base"]
        csv_bridge[old_policy] = compare_csv_paths(a, b)
    summaries, selected_rows = [], []
    for rate in protocol["rates"]:
        for policy in POLICY_MAP:
            rows = [r for r in old_rows if r["context"] == "full_development" and r["policy"] == policy and r["cost_label"] == rate["label"]]
            if any(float(r["cost_rate"]) != rate["cost_rate"] for r in rows):
                raise ValueError("Fee label/rate mismatch")
            summaries.append({"cost_label": rate["label"], "cost_rate": rate["cost_rate"],
                              "research_policy": policy, **summarize_paths(rows)})
            for r in rows:
                selected_rows.append({"cost_label": rate["label"], "cost_rate": rate["cost_rate"],
                    "research_policy": policy, "scenario_id": int(r["scenario_id"]),
                    **{name: finite_or_none(r[name]) for name in FINANCIAL}})
    if sha(PROTOCOL) != protocol_digest or any(sha(ROOT / name) != digest for name, digest in snapshot.items()):
        raise RuntimeError("Frozen sources or inputs changed during bridge")
    OUT.mkdir()
    write_json(OUT / "bridge.json", {"status": "passed", "protocol_sha256": protocol_digest,
        "exact_NPZ_base_bridge": bridge, "CSV_base_bridge": csv_bridge,
        "shared_kernel_hashes": {name: sha(ROOT / name) for name in SHARED_KERNELS},
        "development_arrays_sha256": current_manifest["development_arrays_sha256"],
        "candidate": {"target": [.45,.05,.50], "eta": .03, "H": 30, "L": .4, "original_ids": ids},
        "current_service": {"status": "execution_review", "financial_outputs": None,
            "stress_admission_recomputed": False,
            "interpretation": "Frozen Review across hypothetical costs is not an empirical robustness result"},
        "limits": protocol["limits"]})
    for name, rows in (("research_paths.csv", selected_rows), ("research_summary.csv", summaries)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "protocol_sha256": protocol_digest, "inputs_and_sources": snapshot,
        "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir())},
        "new_market_data_read": False, "new_policy_replays": 0, "new_fits": 0,
        "final_scenarios_read": False, "current_service_admission_changed": False,
        "research_rows": len(selected_rows), "summary_rows": len(summaries)})
    print(json.dumps({"status": "passed", "research_rows": len(selected_rows), "output": str(OUT)}))


if __name__ == "__main__":
    main()
