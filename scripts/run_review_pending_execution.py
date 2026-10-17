"""Replay frozen strict172 effective assignments with a post-crossing hold proxy.

No fitting/calibration routine is imported or called. Financial referrals only
start holding at the next decision; the already observed everbreach is retained.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from safeinvest.data import load_paths
from safeinvest_research.metrics import NUMERIC_TOL, SUMMARY_METADATA, discrete_cvar, path_metrics

PROTOCOL = ROOT / "docs/review_pending_execution_protocol.json"
OUT = ROOT / "results/model_review/review_pending_execution_v1"
FROZEN = ROOT / "results/model_review/positive_asset_specification_v1"
CACHE = ROOT / "results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz"
ARMS = ("annual", "pending_hold")
METRIC_FIELDS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
PAIR_FIELDS = (*METRIC_FIELDS, "nominal_fees", "nominal_moved", "mean_annual_fee_fraction", "mean_annual_moved_fraction", "actual_trade_fraction", "max_end_year_tv_drift", "excess", "added_severity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_manifest():
    files = [Path(__file__).resolve(), ROOT / "tests/test_review_pending_execution.py", PROTOCOL,
             ROOT / "research/experimental_execution/review_pending.py", ROOT / "research/safeinvest_research/metrics.py",
             ROOT / "src/safeinvest/core.py", ROOT / "src/safeinvest/data.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def frozen_dependencies(protocol):
    result = {name: sha(ROOT / name) for name in protocol["frozen_dependencies_sha256"]}
    if result != protocol["frozen_dependencies_sha256"]: raise RuntimeError("A frozen contract/product/cache dependency changed")
    kernel = protocol["execution_kernel"]
    if not kernel["ready"] or kernel["sha256"] != sha(ROOT / kernel["path"]):
        raise RuntimeError("Execution kernel must be source-stable and hash-frozen before market replay")
    frozen_sources = protocol.get("execution_sources_sha256", {})
    if not frozen_sources or any(sha(ROOT / name) != expected for name, expected in frozen_sources.items()):
        raise RuntimeError("Execution sources must be explicitly frozen in the prospective protocol")
    return result


def recovery_from_path(real, loss_tolerance):
    """Independent annual endpoint crossing/recovery with explicit censoring."""
    real = np.asarray(real, float)
    if real.ndim != 1 or len(real) < 2 or np.any(~np.isfinite(real)) or np.any(real <= 0):
        raise ValueError("Require a positive finite annual real-wealth path")
    peak = np.maximum.accumulate(real)
    current = (peak - real) / peak
    running = np.maximum.accumulate(current)
    crossing = np.flatnonzero(running - loss_tolerance > NUMERIC_TOL)
    if not len(crossing):
        return {"triggered": False, "first_crossing": None, "drawdown_at_trigger": None, "peak_at_trigger": None,
                "real_wealth_at_trigger": None, "recovery_endpoint": None, "recovery_time": None,
                "recovered": None, "right_censored": None, "available_followup": None, "observed_followup": None,
                "added_severity": None}
    tau = int(crossing[0]); later = np.flatnonzero(real[tau + 1:] >= peak[tau])
    recovered = bool(len(later)); endpoint = int(tau + 1 + later[0]) if recovered else None
    available = len(real) - 1 - tau
    delay = endpoint - tau if recovered else None
    return {"triggered": True, "first_crossing": tau, "drawdown_at_trigger": float(running[tau]),
            "peak_at_trigger": float(peak[tau]), "real_wealth_at_trigger": float(real[tau]),
            "recovery_endpoint": endpoint, "recovery_time": delay, "recovered": recovered,
            "right_censored": not recovered, "available_followup": available,
            "observed_followup": delay if recovered else available, "added_severity": float(max(running[-1] - running[tau], 0.))}


def records_for_arm(arm, *, context, scope, fold, client, assignment, candidate_local, candidate_global, scenario_ids, horizon, loss_tolerance):
    fields = path_metrics(arm["real_wealth"])
    rows = []
    for index, scenario in enumerate(scenario_ids):
        recovery = recovery_from_path(arm["real_wealth"][index, 0], loss_tolerance)
        for key in ["first_crossing", "recovery_endpoint", "recovery_time", "available_followup", "observed_followup"]:
            expected = recovery[key] if recovery[key] is not None else -1
            if int(arm[key][index, 0]) != expected: raise RuntimeError(f"Core/replay {key} differs")
        for key in ["triggered", "recovered", "right_censored"]:
            if bool(arm[key][index, 0]) != bool(recovery[key]): raise RuntimeError(f"Core/replay {key} differs")
        for key in ["drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger", "added_severity"]:
            actual = arm[key][index, 0]
            if recovery[key] is None:
                if not np.isnan(actual): raise RuntimeError("Untriggered core trigger-value sentinel is not NaN")
            elif not np.isclose(actual, recovery[key], atol=1e-13, rtol=1e-12): raise RuntimeError(f"Core/replay {key} differs")
        row = {"context": context, "scope": scope, "outer_fold": fold, "scenario_id": int(scenario), "client": client,
               "horizon": horizon, "loss_tolerance": loss_tolerance, "assignment": assignment, "assigned": True,
               "candidate_local_index": candidate_local, "candidate_original232_index": candidate_global,
               **{key: float(value[index, 0]) for key, value in fields.items()}, **recovery}
        row.update({"breach": bool(row["mdd"] - loss_tolerance > NUMERIC_TOL), "excess": float(max(row["mdd"] - loss_tolerance, 0.)),
                    "nominal_fees": float(arm["fees"][index, 0].sum()), "nominal_moved": float(arm["moved"][index, 0].sum()),
                    "mean_annual_fee_fraction": float(arm["fee_fraction"][index, 0].mean()),
                    "mean_annual_moved_fraction": float(arm["moved_fraction"][index, 0].mean()),
                    "actual_trade_count": int(arm["trades"][index, 0].sum()), "review_periods": horizon,
                    "actual_trade_fraction": float(arm["trades"][index, 0].mean()),
                    "max_end_year_tv_drift": float(arm["end_year_tv_drift"][index, 0].max()),
                    "terminal_equity": float(arm["end_weights"][index, 0, -1, 0]),
                    "terminal_bond": float(arm["end_weights"][index, 0, -1, 1]),
                    "terminal_bill": float(arm["end_weights"][index, 0, -1, 2]),
                    "financial_referral_reason": "financial_drawdown_crossing" if recovery["triggered"] else None})
        if row["breach"] != row["triggered"]: raise RuntimeError("Absorbing breach flag differs from first crossing")
        rows.append(row)
    return rows


def review_rows(*, context, scope, fold, client, scenario_ids, horizon, loss_tolerance):
    template = {key: None for key in [*PAIR_FIELDS, "breach", "triggered", "first_crossing", "drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger", "recovery_endpoint", "recovery_time", "recovered", "right_censored", "available_followup", "observed_followup", "actual_trade_count", "review_periods", "terminal_equity", "terminal_bond", "terminal_bill"]}
    return [{"context": context, "scope": scope, "outer_fold": fold, "scenario_id": int(sid), "client": client,
             "horizon": horizon, "loss_tolerance": loss_tolerance, "assignment": "Review", "assigned": False,
             "candidate_local_index": None, "candidate_original232_index": None,
             "financial_referral_reason": None, "onboarding_reason": "frozen_effective_review", **template} for sid in scenario_ids]


def verify_baseline(rows, saved, cache):
    """Every annual replay must match frozen effective financial outcomes first."""
    errors = []
    for row in rows:
        found = saved.loc[(saved.context == row["context"]) & (saved.scenario_id == row["scenario_id"]) & (saved.client == row["client"])]
        if len(found) != 1: raise RuntimeError("Frozen annual row is not unique")
        old = found.iloc[0]
        if bool(old.assigned) != row["assigned"] or old.assignment != row["assignment"]:
            raise RuntimeError("Frozen effective assignment changed")
        if not row["assigned"]:
            if not old[list(METRIC_FIELDS)].isna().all(): raise RuntimeError("Original Review is not NULL")
            continue
        if int(old.candidate_index) != row["candidate_local_index"]: raise RuntimeError("Frozen local candidate changed")
        for field in METRIC_FIELDS:
            difference = abs(row[field] - float(old[field])); errors.append(difference)
            if not np.isclose(row[field], old[field], atol=1e-11, rtol=1e-12): raise RuntimeError(f"Annual original replay mismatch {field}")
            expected = cache[f"H{row['horizon']}__{field}"][row["scenario_id"] - 1, row["candidate_original232_index"]]
            if not np.isclose(row[field], expected, atol=1e-11, rtol=1e-12): raise RuntimeError("Annual cache mapping differs")
        for output, key in [("nominal_fees", "fees"), ("nominal_moved", "moved"), ("actual_trade_fraction", "trade_fraction")]:
            expected = cache[f"H{row['horizon']}__{key}"][row["scenario_id"] - 1, row["candidate_original232_index"]]
            if not np.isclose(row[output], expected, atol=1e-10, rtol=1e-12): raise RuntimeError(f"Annual cache accounting differs {output}")
    return max(errors, default=0.)


def verify_pair(annual, pending, returns, inflation, horizon, loss_tolerance):
    """Independent accounting and prefix-coupling checks on actual arrays."""
    cpi = np.concatenate([np.ones((len(returns), 1)), np.cumprod(1 + inflation[:, :horizon], axis=1)], axis=1)
    for arm in [annual, pending]:
        if not np.allclose(arm["cpi"], cpi, atol=1e-13, rtol=1e-12): raise RuntimeError("CPI accumulation differs")
        if not np.allclose(arm["nominal_wealth"], arm["holdings"].sum(axis=-1), atol=1e-11, rtol=1e-12): raise RuntimeError("Wealth differs from actual holdings")
        if not np.allclose(arm["real_wealth"], arm["nominal_wealth"] / cpi[:, None, :], atol=1e-11, rtol=1e-12): raise RuntimeError("Purchasing-power accounting differs")
        if not np.allclose(arm["holdings"][:, :, 1:], arm["posttrade_holdings"] * (1 + returns[:, None, :horizon, :]), atol=1e-11, rtol=1e-12): raise RuntimeError("Actual asset returns differ")
        if not np.allclose(arm["fees"], .0025 * arm["moved"], atol=1e-12, rtol=1e-12): raise RuntimeError("Fee/transfer identity differs")
        if not np.array_equal(arm["trades"], arm["moved"] > NUMERIC_TOL): raise RuntimeError("Actual trade flags differ")
        real_peak = np.maximum.accumulate(arm["real_wealth"], axis=-1)
        dd = (real_peak - arm["real_wealth"]) / real_peak
        running = np.maximum.accumulate(dd, axis=-1)
        if not np.allclose(arm["real_peak"], real_peak, atol=1e-11, rtol=1e-12) or not np.allclose(arm["running_mdd"], running, atol=1e-13, rtol=1e-12): raise RuntimeError("Peak/MDD history differs")
        if not np.array_equal(arm["ever_breached"], running - loss_tolerance > NUMERIC_TOL): raise RuntimeError("Absorbing breach history differs")
    if not np.array_equal(annual["first_crossing"], pending["first_crossing"]) or not np.array_equal(annual["ever_breached"][..., -1], pending["ever_breached"][..., -1]):
        raise RuntimeError("Post-event treatment must not change tau or whole-H everbreach")
    for i in range(len(returns)):
        tau = int(annual["first_crossing"][i, 0]); stop = horizon if tau == -1 else tau
        for key in ["holdings", "nominal_wealth", "real_wealth", "real_peak", "running_mdd", "ever_breached"]:
            if not np.array_equal(annual[key][i, 0, :stop + 1], pending[key][i, 0, :stop + 1]): raise RuntimeError("Crossing prefix states differ")
        for key in ["posttrade_holdings", "fees", "moved", "actions"]:
            if not np.array_equal(annual[key][i, 0, :stop], pending[key][i, 0, :stop]): raise RuntimeError("Crossing prefix actions/fees differ")
        if tau != -1 and (np.any(pending["fees"][i, 0, tau:] != 0) or np.any(pending["moved"][i, 0, tau:] != 0) or np.any(pending["actions"][i, 0, tau:] != "hold")):
            raise RuntimeError("Pending proxy traded after crossing")


def pair_records(annual, pending):
    by_key = {(row["context"], row["scenario_id"], row["client"]): row for row in pending}
    result = []
    for a in annual:
        b = by_key[(a["context"], a["scenario_id"], a["client"])]
        if (a["assigned"], a["triggered"], a["first_crossing"], a["breach"]) != (b["assigned"], b["triggered"], b["first_crossing"], b["breach"]):
            raise RuntimeError("Paired assignment/crossing/breach identity differs")
        row = {key: a[key] for key in ["context", "scope", "outer_fold", "scenario_id", "client", "horizon", "loss_tolerance", "assignment", "assigned", "candidate_local_index", "candidate_original232_index", "triggered", "first_crossing"]}
        for field in PAIR_FIELDS:
            row["annual_" + field], row["pending_" + field] = a[field], b[field]
            row["delta_" + field] = b[field] - a[field] if a[field] is not None and b[field] is not None else None
        row["delta_breach"] = int(b["breach"]) - int(a["breach"]) if a["assigned"] else None
        for name, item in [("annual", a), ("pending", b)]:
            for field in ["recovery_endpoint", "recovery_time", "recovered", "right_censored", "available_followup", "observed_followup"]: row[name + "_" + field] = item[field]
        result.append(row)
    return result


def summaries(records, keys):
    frame = pd.DataFrame(records); result = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple): values = (values,)
        auto = group.loc[group.assigned]; triggered = auto.loc[auto.triggered.astype(bool)]
        for subset, selected in [("all_assigned", auto), ("triggered_only", triggered)]:
            row = {**dict(zip(keys, values)), "subset": subset, "total_profile_paths": len(group), "auto_n": len(auto), "review_n": len(group) - len(auto), "triggered_n": len(triggered), "summary_n": len(selected)}
            names = ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd", "mean_terminal_real", "mean_excess", "mean_added_severity_among_triggered", "mean_nominal_fees", "mean_nominal_moved", "actual_trade_fraction", "breach_count", "breach_probability", "recovery_risk_set_n", "recovered_n", "right_censored_n", "median_recovery_delay_among_recovered_only"]
            row.update({name: None for name in names})
            if len(selected):
                recovery = selected.loc[selected.triggered.astype(bool)]
                recovered = recovery.loc[recovery.recovered.astype(bool)]
                row.update({"mean_log_growth": float(selected.log_growth.mean()), "median_cagr": float(selected.cagr.median()), "p10_cagr": float(np.quantile(selected.cagr, .1, method="inverted_cdf")),
                            "median_mdd": float(selected.mdd.median()), "p90_mdd": float(np.quantile(selected.mdd, .9, method="inverted_cdf")), "cvar90_mdd": float(discrete_cvar(selected.mdd.to_numpy(float))),
                            "mean_terminal_real": float(selected.terminal_real.mean()), "mean_excess": float(selected.excess.mean()),
                            "mean_added_severity_among_triggered": float(recovery.added_severity.mean()) if len(recovery) else None,
                            "mean_nominal_fees": float(selected.nominal_fees.mean()), "mean_nominal_moved": float(selected.nominal_moved.mean()),
                            "actual_trade_fraction": float(selected.actual_trade_count.sum() / selected.review_periods.sum()), "breach_count": int(selected.breach.astype(bool).sum()),
                            "breach_probability": float(selected.breach.astype(bool).mean()), "recovery_risk_set_n": len(recovery), "recovered_n": len(recovered), "right_censored_n": len(recovery) - len(recovered),
                            "median_recovery_delay_among_recovered_only": float(recovered.recovery_time.median()) if len(recovered) else None})
            result.append(row)
    return result


def paired_summaries(rows, keys):
    frame = pd.DataFrame(rows); result = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple): values = (values,)
        auto = group.loc[group.assigned]
        for subset, selected in [("all_assigned", auto), ("triggered_only", auto.loc[auto.triggered.astype(bool)])]:
            result.append({**dict(zip(keys, values)), "subset": subset, "total_profile_paths": len(group), "auto_n": len(auto), "summary_n": len(selected),
                           **{"mean_delta_" + field: float(selected["delta_" + field].mean()) if len(selected) and selected["delta_" + field].notna().any() else None for field in PAIR_FIELDS},
                           **{f"{field}_{direction}_n": int((selected["delta_" + field] > NUMERIC_TOL).sum()) if direction == "increased" else int((selected["delta_" + field] < -NUMERIC_TOL).sum()) if direction == "decreased" else int((selected["delta_" + field].abs() <= NUMERIC_TOL).sum()) for field in ["mdd", "terminal_real"] for direction in ["increased", "decreased", "equal"]},
                           "added_severity_pair_n": int(selected.delta_added_severity.notna().sum()),
                           "sum_delta_breach": int(selected.delta_breach.sum()) if len(selected) else None})
    return result


def main():
    started = time.perf_counter(); protocol = json.loads(PROTOCOL.read_text())
    dependencies, sources = frozen_dependencies(protocol), source_manifest()
    if (protocol["parameters"]["cost_rate"] != .0025 or protocol["parameters"]["numerical_crossing_tolerance"] != NUMERIC_TOL
            or protocol["parameters"]["initial_wealth"] != 100. or protocol["execution_kernel"]["path"] != "research/experimental_execution/review_pending.py"
            or protocol["data"]["allowed_scenario_ids"] != list(range(1, 121)) or protocol["data"]["allow_final"]):
        raise RuntimeError("Frozen cost/data scope changed")
    from experimental_execution.review_pending import simulate_paired
    fits = json.loads((FROZEN / "nested_context_refits.json").read_text()); mapping = json.loads((FROZEN / "candidate_mapping.json").read_text())
    if len(mapping) != 172 or [row["candidate_local_index"] for row in mapping] != list(range(172)):
        raise RuntimeError("Frozen local172 mapping is invalid")
    strict_protocol = json.loads((ROOT / "docs/positive_asset_specification_protocol.json").read_text())
    profiles = strict_protocol["clients"]
    saved = pd.concat([pd.read_csv(FROZEN / "nested_full_paths.csv"), pd.read_csv(FROZEN / "nested_outer_oof_paths.csv")], ignore_index=True)
    saved = saved.loc[saved.view == "effective"]
    returns, inflation, ids = load_paths(ROOT / protocol["data"]["path"], range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in [ids, returns, inflation]: digest.update(np.ascontiguousarray(value).tobytes())
    cache_meta = json.loads(CACHE.with_suffix(".json").read_text())
    if digest.hexdigest() != cache_meta["development_arrays_sha256"]: raise RuntimeError("Guarded input differs from the frozen cache data")
    with np.load(CACHE) as data: cache = {key: data[key] for key in data.files}
    all_rows = {arm: [] for arm in ARMS}; metadata = []; archives = {}; yearly = []; baseline_errors = []
    for context in strict_protocol["contexts"]:
        name = context["name"]; fitted = fits[name]
        if fitted["training_ids"] != context["training_ids"] or fitted["outer_test_ids"] != context["outer_test_ids"]: raise RuntimeError("Frozen context IDs changed")
        scenario_ids = context["training_ids"] if context["context_id"] == 0 else context["outer_test_ids"]
        scope = "full" if context["context_id"] == 0 else "oof"; fold = context["context_id"] or None
        ix = np.asarray(scenario_ids) - 1
        menu = {item["name"]: item for item in fitted["raw_fit"]["menu"]} if fitted["raw_fit"] else {}
        for profile in profiles:
            client, horizon, tolerance = profile["name"], profile["horizon"], profile["loss_tolerance"]
            assignment = fitted["effective_assignments"][client]
            common = {"context": name, "scope": scope, "fold": fold, "client": client, "scenario_ids": scenario_ids, "horizon": horizon, "loss_tolerance": tolerance}
            if assignment == "Review":
                rows = review_rows(**common)
                baseline_errors.append(verify_baseline(rows, saved, cache))
                for arm in ARMS: all_rows[arm].extend([{**row, "arm": arm} for row in rows])
                continue
            product = menu[assignment]; local = product["candidate_index"]; global_index = mapping[local]["candidate_original232_index"]
            target = np.asarray(product["weights"], float)
            if product["candidate_original232_index"] != global_index or not np.allclose(target, cache["weights"][global_index], atol=1e-14, rtol=0) or np.min(target) < .05 - NUMERIC_TOL:
                raise RuntimeError("Frozen positive target/local/global map mismatch")
            pair = simulate_paired(returns[ix], inflation[ix], target, tolerance, horizon=horizon, cost_rate=.0025)
            annual_rows = records_for_arm(pair["annual"], assignment=assignment, candidate_local=local, candidate_global=global_index, **common)
            baseline_errors.append(verify_baseline(annual_rows, saved, cache))
            # Comparative quantities are produced only after the original arm agrees.
            verify_pair(pair["annual"], pair["pending_hold"], returns[ix], inflation[ix], horizon, tolerance)
            pending_rows = records_for_arm(pair["pending_hold"], assignment=assignment, candidate_local=local, candidate_global=global_index, **common)
            prefix = f"{name}__client_{client}"
            metadata.append({"prefix": prefix, "context": name, "scope": scope, "outer_fold": fold, "client": client,
                             "scenario_ids": scenario_ids, "horizon": horizon, "loss_tolerance": tolerance, "assignment": assignment,
                             "candidate_local_index": local, "candidate_original232_index": global_index, "target": target.tolist(),
                             "frozen_training_ids": fitted["training_ids"], "coupling_receipt": pair["coupling"]})
            for arm, rows in [("annual", annual_rows), ("pending_hold", pending_rows)]:
                all_rows[arm].extend([{**row, "arm": arm} for row in rows])
                for key, value in pair[arm].items():
                    if isinstance(value, np.ndarray): archives[f"{prefix}__{arm}__{key}"] = value
                for i, sid in enumerate(scenario_ids):
                    for year in range(horizon + 1):
                        state = pair[arm]
                        record = {"context": name, "scope": scope, "outer_fold": fold, "client": client, "scenario_id": sid, "arm": arm, "year": year,
                                  **{key: float(state[key][i, 0, year]) for key in ["nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd"]},
                                  "cpi": float(state["cpi"][i, year]), "ever_breached": bool(state["ever_breached"][i, 0, year])}
                        for asset, asset_name in enumerate(["equity", "bond", "bill"]):
                            record["holding_" + asset_name] = float(state["holdings"][i, 0, year, asset])
                            record["weight_" + asset_name] = float(state["weights"][i, 0, year, asset])
                        for key in ["fees", "moved", "trades", "actions", "pending_at_decision"]:
                            value = None if year == 0 else state[key][i, 0, year - 1].item()
                            record[key + "_before_return_year"] = value
                        yearly.append(record)
            print(json.dumps({"context": name, "client": client, "paths": len(scenario_ids), "triggered": sum(row["triggered"] for row in annual_rows), "baseline_max_error": baseline_errors[-1]}), flush=True)
    pairs = pair_records(all_rows["annual"], all_rows["pending_hold"])
    records = [*all_rows["annual"], *all_rows["pending_hold"]]
    if len(records) != 2400 or len(pairs) != 1200: raise RuntimeError("Full+OOF denominators changed")
    for scope in ["full", "oof"]:
        for arm in ARMS:
            for profile in profiles:
                selected = [row for row in records if row["scope"] == scope and row["arm"] == arm and row["client"] == profile["name"]]
                if len(selected) != 120 or {row["scenario_id"] for row in selected} != set(range(1, 121)): raise RuntimeError("Missing original profile/scenario denominator")
    if source_manifest() != sources or frozen_dependencies(protocol) != dependencies: raise RuntimeError("Source/frozen dependency changed before writing")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {"client_path_metrics.csv": pd.DataFrame(records), "paired_path_differences.csv": pd.DataFrame(pairs), "yearly_trace.csv": pd.DataFrame(yearly),
              "profile_summary.csv": pd.DataFrame(summaries(records, ["scope", "arm", "client"])),
              "context_profile_summary.csv": pd.DataFrame(summaries(records, ["context", "scope", "arm", "client"])),
              "paired_profile_summary.csv": pd.DataFrame(paired_summaries(pairs, ["scope", "client"])),
              "paired_context_profile_summary.csv": pd.DataFrame(paired_summaries(pairs, ["context", "scope", "client"]))}
    for name, frame in frames.items(): frame.to_csv(OUT / name, index=False)
    np.savez_compressed(OUT / "executed_paths.npz", **archives)
    write_json(OUT / "execution_groups.json", metadata)
    end_sources, end_dependencies = source_manifest(), frozen_dependencies(protocol)
    if end_sources != sources or end_dependencies != dependencies: raise RuntimeError("Source/frozen dependency changed during writing")
    outputs = [*frames, "executed_paths.npz", "execution_groups.json"]
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "sources_at_end": end_sources, "frozen_dependency_sha256": dependencies, "frozen_dependency_sha256_at_end": end_dependencies,
        "outputs_sha256": {name: sha(OUT / name) for name in outputs}, "protocol": protocol, "scenario_ids": ids.tolist(), "final_scenarios_loaded": False,
        "development_arrays_sha256": digest.hexdigest(), "model_refit_or_recalibration": False, "summary_metadata": SUMMARY_METADATA,
        "checks": {"annual_original_path_metrics_max_abs_error": max(baseline_errors), "paired_first_crossing_and_breach_identical": True,
                   "prefix_actions_states_fees_identical_through_crossing": True, "no_pending_trades_after_crossing": True,
                   "recovery_independently_recomputed": True, "review_financial_outcomes_null": True, "client_path_rows": len(records), "paired_rows": len(pairs), "execution_groups": len(metadata)},
        "units": "G is annualized log real wealth growth; fees/moved are cumulative nominal initial-100 wealth units, not return percentage points. Recovery at annual endpoints with fixed trigger peak and explicit right censoring.",
        "scope": "Frozen strict172 products and effective assignments, post-crossing no-human-response hold scenario only. No lower everbreach probability, adviser improvement, optimal hold or final adoption claim. Annual restore is a research comparator and does not resolve Q2 benefit-gating/target updates. Learned OOF policies differ across folds and do not give a pooled fixed-policy certificate."})
    print(f"Wrote review pending execution diagnostic to {OUT}", flush=True)


if __name__ == "__main__": main()
