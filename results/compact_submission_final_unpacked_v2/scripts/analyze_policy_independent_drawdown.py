"""Policy-independent annual MDD lower bound on guarded development paths.

This is an optimistic clairvoyant, zero-fee relaxation, not an investable
policy.  The theorem uses each interval's product of annual gross envelopes,
not pointwise wealth dominance.  No portfolio is fitted or reselected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from safeinvest.data import load_paths

PROTOCOL = ROOT / "docs/policy_independent_drawdown_bound_protocol.json"
OUT = ROOT / "results/model_review/policy_independent_drawdown_bound_v1"
TOL = 1e-12
ASSETS = ("equity", "government_bond", "bill_proxy")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def drawdown_envelope(returns, inflation, horizon=None):
    """Compute D* and its annual clairvoyant witness, including initial100.

    Inputs are decimal returns(n,T,3), inflation(n,T).  Each year jointly
    select the largest asset real gross factor, with the lowest asset index
    breaking exact ties.  This uses that year's future outcome deliberately
    as a mathematical relaxation.  It is not a causal trading suggestion.
    The annual decision model excludes further trades inside each year.
    """
    market, infl = np.asarray(returns, float), np.asarray(inflation, float)
    if (market.ndim != 3 or market.shape[-1] != 3 or min(market.shape[:2]) < 1
            or infl.shape != market.shape[:2] or not np.all(np.isfinite(market))
            or not np.all(np.isfinite(infl)) or np.any(market <= -1) or np.any(infl <= -1)):
        raise ValueError("Require nonempty finite annual returns/inflation with positive gross factors")
    if horizon is None:
        horizon = market.shape[1]
    if (isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer))
            or not 1 <= horizon <= market.shape[1]):
        raise ValueError("horizon must be an integer between1 and the available years")
    market, infl = market[:, :horizon], infl[:, :horizon]
    asset_log = np.log1p(market)
    best_asset = np.argmax(asset_log, axis=-1)
    log_envelope = np.max(asset_log, axis=-1) - np.log1p(infl)
    cumulative = np.column_stack((np.zeros(len(market)), np.cumsum(log_envelope, axis=1)))
    running_peak = np.maximum.accumulate(cumulative, axis=1)
    drawdowns = np.maximum(-np.expm1(cumulative - running_peak), 0.0)
    end = np.argmax(drawdowns, axis=1)
    peak_index = np.zeros_like(cumulative, dtype=np.int64)
    current_index = np.zeros(len(market), dtype=np.int64)
    for t in range(1, horizon + 1):
        higher = cumulative[:, t] > running_peak[:, t - 1]
        current_index = np.where(higher, t, current_index)
        peak_index[:, t] = current_index
    start = peak_index[np.arange(len(market)), end]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        wealth = 100 * np.exp(cumulative)
        envelope = np.exp(log_envelope)
    if not np.all(np.isfinite(wealth)) or np.any(wealth <= 0) or not np.all(np.isfinite(envelope)) or np.any(envelope <= 0):
        raise ValueError("Clairvoyant evidence path exceeds floating-point numerical range")
    return {"mdd_lower_bound": drawdowns[np.arange(len(market)), end],
            "ideal_real_wealth": wealth, "log_ideal_wealth_factor": cumulative,
            "annual_real_envelope": envelope, "annual_log_envelope": log_envelope,
            "best_asset_index": best_asset, "witness_peak_time": start, "witness_trough_time": end}


def source_snapshot(metadata):
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "src/safeinvest/data.py",
             ROOT / "docs/risk_menu_development_protocol.json", ROOT / "tests/test_policy_independent_drawdown.py"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    for relative, digest in metadata["source_hashes"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError("Cached numerical source changed: " + relative)
        if relative in sources and sources[relative] != digest:
            raise RuntimeError("Source snapshots conflict: " + relative)
        sources[relative] = digest
    return sources


def main():
    protocol_hash = sha(PROTOCOL)  # Before any market-array loading.
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol["allowed_scenario_ids"] != [1, 120] or protocol["forbidden_final_access"] is not True
            or len(protocol["profiles"]) != 5):
        raise ValueError("Only the five declared development1..120 bounds are permitted")
    cache = ROOT / protocol["cached_grid_comparator"]["file"]
    meta_path = ROOT / protocol["cached_grid_comparator"]["metadata"]
    metadata = json.loads(meta_path.read_text())
    input_paths = [cache, meta_path, ROOT / "docs/risk_menu_development_protocol.json"]
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}
    sources = source_snapshot(metadata)
    if sources[str(PROTOCOL.relative_to(ROOT))] != protocol_hash:
        raise RuntimeError("Bound protocol changed while opening inputs")
    if (metadata["scenario_ids"] != list(range(1, 121)) or metadata["candidate_count"] != 232
            or metadata["cost_rate"] != .0025 or metadata["step"] != .05
            or metadata["npz_sha256"] != inputs[str(cache.relative_to(ROOT))]):
        raise RuntimeError("Cached comparator differs from the declared232-target development family")
    prior_profiles = json.loads((ROOT / "docs/risk_menu_development_protocol.json").read_text())["clients"]
    expected = [{key: p[key] for key in ("name", "horizon", "loss_tolerance")} for p in prior_profiles]
    if protocol["profiles"] != expected:
        raise RuntimeError("Bound thresholds differ from the original client profiles")
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for array in (ids, returns, inflation):
        digest.update(np.ascontiguousarray(array).tobytes())
    if (not np.array_equal(ids, np.arange(1, 121)) or returns.shape != (120, 30, 3)
            or digest.hexdigest() != metadata["development_arrays_sha256"]):
        raise RuntimeError("Guarded development arrays differ from cached comparator provenance")
    bounds = {p["horizon"]: drawdown_envelope(returns, inflation, p["horizon"]) for p in protocol["profiles"]}
    clients, path_rows, grid_rows = [], [], []
    max_numerical_violation = 0.0
    with np.load(cache, allow_pickle=False) as archive:
        if not np.array_equal(archive["scenario_ids"], ids):
            raise RuntimeError("Cached scenario identities differ")
        weights = archive["weights"]
        if (weights.shape != (232, 3) or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or not np.allclose(weights.sum(axis=1), 1, atol=TOL, rtol=0)
                or not np.any(np.all(np.abs(weights - 1 / 3) <= TOL, axis=1))):
            raise RuntimeError("Cached comparator is not the fixed232-point simplex catalogue")
        for profile in protocol["profiles"]:
            client, horizon, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
            result = bounds[horizon]
            lower = result["mdd_lower_bound"]
            candidate_mdd = archive[f"H{horizon}__mdd"]
            if (candidate_mdd.shape != (120, 232) or not np.all(np.isfinite(candidate_mdd))
                    or np.any(candidate_mdd < 0) or np.any(candidate_mdd > 1)):
                raise RuntimeError("Malformed cached per-path MDD array")
            violation = float(np.maximum(lower[:, None] - candidate_mdd, 0).max())
            max_numerical_violation = max(max_numerical_violation, violation)
            if violation > TOL:
                raise RuntimeError("Cached annual policy violates the claimed pathwise lower bound")
            forced = lower - loss > TOL
            counts = np.sum(candidate_mdd - loss > TOL, axis=0)
            forced_count, minimum_count = int(forced.sum()), int(counts.min())
            if minimum_count < forced_count:
                raise RuntimeError("A cached policy has fewer breaches than the necessary forced-path count")
            budget_paths = 12
            record = {"client": client, "horizon": horizon, "loss_tolerance": loss, "n_original_paths": 120,
                      "forced_breach_count": forced_count, "forced_breach_fraction": forced_count / 120,
                      "empirical_alpha": .1, "allowed_breach_count": budget_paths,
                      "empirical_budget_impossible_for_entire_declared_class": forced_count > budget_paths,
                      "grid_min_breach_count": minimum_count, "grid_min_breach_fraction": minimum_count / 120,
                      "grid_candidates_attaining_minimum": int(np.count_nonzero(counts == minimum_count)),
                      "finite_grid_empirical_budget_impossible": minimum_count > budget_paths,
                      "necessary_bound_is_not_sufficiency": True}
            clients.append(record)
            for index, scenario in enumerate(ids):
                start, end = int(result["witness_peak_time"][index]), int(result["witness_trough_time"][index])
                path_rows.append({"client": client, "scenario_id": int(scenario), "horizon": horizon,
                                  "loss_tolerance": loss, "mdd_lower_bound": float(lower[index]),
                                  "forced_breach": bool(forced[index]), "witness_peak_time": start, "witness_trough_time": end,
                                  "witness_peak_ideal_real_wealth": float(result["ideal_real_wealth"][index, start]),
                                  "witness_trough_ideal_real_wealth": float(result["ideal_real_wealth"][index, end]),
                                  "ideal_real_terminal": float(result["ideal_real_wealth"][index, -1])})
            for candidate, count in enumerate(counts):
                grid_rows.append({"client": client, "horizon": horizon, "loss_tolerance": loss,
                                  "candidate_index": candidate, "w_equity": float(weights[candidate, 0]),
                                  "w_bond": float(weights[candidate, 1]), "w_bill": float(weights[candidate, 2]),
                                  "breach_count": int(count), "n_original_paths": 120,
                                  "forced_breach_lower_count": forced_count})
    evidence = bounds[30]
    evidence_rows = []
    for row, scenario in enumerate(ids):
        for year in range(30):
            best = int(evidence["best_asset_index"][row, year])
            evidence_rows.append({"scenario_id": int(scenario), "year": year + 1,
                                  "equity_gross": float(1 + returns[row, year, 0]),
                                  "government_bond_gross": float(1 + returns[row, year, 1]),
                                  "bill_proxy_gross": float(1 + returns[row, year, 2]),
                                  "inflation_gross": float(1 + inflation[row, year]),
                                  "clairvoyant_best_asset_index": best, "clairvoyant_best_asset": ASSETS[best],
                                  "zero_fee_real_gross_envelope": float(evidence["annual_real_envelope"][row, year]),
                                  "ideal_real_wealth_before": float(evidence["ideal_real_wealth"][row, year]),
                                  "ideal_real_wealth_after": float(evidence["ideal_real_wealth"][row, year + 1]),
                                  "uses_future_year_return": True, "investable_recommendation": False})
    summary = {"scope": protocol["scope"], "profiles": clients,
               "forced_original_scenario_ids": {profile["name"]: [int(row["scenario_id"]) for row in path_rows
                                                                  if row["client"] == profile["name"] and row["forced_breach"]]
                                                for profile in protocol["profiles"]},
               "theorem": protocol["mathematics"], "admissible_strategy_class": protocol["admissible_strategy_class"],
               "interpretation": protocol["forced_event"],
               "cache_bound_check": {"candidate_path_horizon_comparisons": 120 * 232 * 5,
                                     "max_positive_lower_minus_cached_mdd": max_numerical_violation,
                                     "tolerance": TOL, "passed": True},
               "clairvoyant_evidence": "Zero-fee ideal path uses each realized future annual best asset; this is an optimistic proof witness, not a causal strategy or attainable recommendation.",
               "new_models_fitted": False, "cached_policies_rerun": False, "final_scenarios_loaded": False}
    frames = {"client_counts.csv": pd.DataFrame(clients), "path_bounds.csv": pd.DataFrame(path_rows),
              "grid_client_counts.csv": pd.DataFrame(grid_rows), "clairvoyant_asset_path.csv": pd.DataFrame(evidence_rows)}
    payloads = {name: frame.to_csv(index=False).encode() for name, frame in frames.items()}
    payloads["summary.json"] = json_bytes(summary)
    sources_end = {relative: sha(ROOT / relative) for relative in sources}
    inputs_end = {relative: sha(ROOT / relative) for relative in inputs}
    if sources_end != sources or inputs_end != inputs:
        raise RuntimeError("Source/input drift during diagnostic; no output written")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (OUT / name).write_bytes(payload)
    output_hashes = {name: sha(OUT / name) for name in payloads}
    if output_hashes != {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}:
        raise RuntimeError("Written output bytes differ from declared payloads")
    if ({relative: sha(ROOT / relative) for relative in sources} != sources
            or {relative: sha(ROOT / relative) for relative in inputs} != inputs):
        raise RuntimeError("Source/input drift during output writing; no completion manifest")
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256_before_market_loading": protocol_hash,
                "source_sha256_start": sources, "source_sha256_end": sources_end,
                "input_artifact_sha256_start": inputs, "input_artifact_sha256_end": inputs_end,
                "outputs_sha256": output_hashes, "scenario_ids": ids.tolist(), "development_arrays_sha256": digest.hexdigest(),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "raw_final_returns_loaded": False, "new_policy_fitted_or_selected": False,
                "units": {"mdd_lower_bound": "fraction of prior peak", "forced_breach_fraction": "forced original paths/120",
                          "ideal_real_wealth": "initial100 purchasing-power units under zero-fee clairvoyant relaxation"},
                "scope": protocol["scope"], "admissible_strategy_class": protocol["admissible_strategy_class"],
                "proof": "Per-period factor bound implies every-interval product bound and then MDD bound; pointwise wealth dominance is insufficient.",
                "budget_conclusion": "Only forced_count>12 proves infeasibility of the original120-path10% budget in the declared annual-trading class; lower counts do not prove causal feasibility or a population probability statement."}
    (OUT / "manifest.json").write_bytes(json_bytes(manifest))
    print(json.dumps({"output": str(OUT), "client_counts": clients, "cache_bound_check": summary["cache_bound_check"]}))


if __name__ == "__main__":
    main()
