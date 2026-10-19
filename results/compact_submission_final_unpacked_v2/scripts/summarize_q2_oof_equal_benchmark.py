"""Summarize the saved, matched OOF Balanced-client paths and exact-thirds baseline.

This script imports no market loader, optimizer, or execution simulator.  It pools
the 120 path observations before taking quantiles, rather than averaging fold
quantiles.  It makes no decision about whether to adopt the experimental gates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research")]
from safeinvest_research.metrics import NUMERIC_TOL, SUMMARY_METADATA, summarize_paths

OUT = ROOT / "results/model_review/q2_oof_benchmark_v1"
FIXED = ROOT / "results/model_review/fixed_execution_ablation_v1"
FINITE = ROOT / "results/model_review/finite_horizon_execution_v1"
POLICIES = ["annual_restore", "initial_then_hold", "myopic_gate", "finite_horizon_gate", "equal_weight_annual"]
PATH_FIELDS = ["log_growth", "cagr", "mdd", "volatility", "terminal_real", "nominal_fees", "nominal_moved",
               "mean_annual_fee_fraction", "mean_annual_moved_fraction", "trade_fraction",
               "target_execution_fraction", "max_end_year_tv_drift", "terminal_equity", "terminal_bond", "terminal_bill"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sources():
    return {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__).resolve(), ROOT / "research/safeinvest_research/metrics.py", ROOT / "docs/problem_A.pdf"]}


def inputs():
    return {str(path.relative_to(ROOT)): sha(path) for folder in [FIXED, FINITE]
            for path in [folder / "target_path_metrics.csv", folder / "client_path_metrics.csv"]}


def cohort(frame):
    return frame.loc[(frame.scope == "oof") & (frame.client == "C") & (frame.horizon == 30)].copy()


def check_cohort(frame, original_keys):
    """A single policy must have exactly the original held-out fold per path."""
    if len(frame) != 120 or frame.scenario_id.nunique() != 120 or set(frame.scenario_id) != set(range(1, 121)):
        raise RuntimeError("Require exactly one row for each development scenario 1--120")
    keys = frame[["scenario_id", "fold", "context"]].sort_values("scenario_id").reset_index(drop=True)
    if not keys.equals(original_keys):
        raise RuntimeError("Policy paths do not use the same original OOF fold/context mapping")
    if not frame.groupby("fold").size().eq(24).all():
        raise RuntimeError("Each of the five original OOF folds must contribute 24 paths")
    if not np.isfinite(frame[PATH_FIELDS].to_numpy()).all():
        raise RuntimeError("An assigned or reference path has missing/nonfinite metrics")
    if not np.allclose(np.expm1(frame.log_growth), frame.cagr, atol=1e-14, rtol=1e-12):
        raise RuntimeError("Saved CAGR and annual log growth disagree")
    trade_counts = frame.trade_fraction.to_numpy() * 30
    if not np.allclose(trade_counts, np.rint(trade_counts), atol=1e-11, rtol=0):
        raise RuntimeError("Saved trade fractions do not correspond to actual integer trade counts")


def summarize(frame):
    stats = summarize_paths({field: frame[field].to_numpy() for field in ["log_growth", "cagr", "mdd", "volatility", "terminal_real"]}, .4)
    result = {name: float(value) for name, value in stats.items()}
    result.update({"n_paths": len(frame), "horizon": 30, "loss_tolerance": .4,
                   "breach_count": int((frame.mdd - .4 > NUMERIC_TOL).sum()),
                   "actual_trade_count": int(np.rint((frame.trade_fraction * 30).sum())),
                   "review_periods": 30 * len(frame), "actual_trade_fraction": float(frame.trade_fraction.mean()),
                   "mean_nominal_fees": float(frame.nominal_fees.mean()), "mean_nominal_moved": float(frame.nominal_moved.mean()),
                   "mean_annual_fee_fraction": float(frame.mean_annual_fee_fraction.mean()),
                   "mean_annual_moved_fraction": float(frame.mean_annual_moved_fraction.mean())})
    if not np.isclose(result["cvar90_mdd"], np.sort(frame.mdd.to_numpy())[-12:].mean(), atol=1e-14, rtol=0):
        raise RuntimeError("CVaR90 differs from the worst 12 of 120 equally weighted paths")
    if not np.isclose(result["breach_probability"], result["breach_count"] / 120, atol=1e-14, rtol=0):
        raise RuntimeError("Breach count and frequency disagree")
    return result


def requirement_checklist():
    return {
        "purpose": "Working-research completeness checklist against Problem A pp. 1--3; not a final claim of competition completion",
        "scope": "Current development research only. Earlier final-set inspection prevents any new claim that the same 121--180 data are unseen.",
        "Q1": {
            "development_evidence_present": ["Purchasing-power net wealth, annualized log growth and real CAGR", "Annual-endpoint MDD, tail exceedance/excess and CVaR diagnostics separate large loss from ordinary volatility", "Three static products, transparent profile-horizon-specific assignment or Review", "Explicit policy uncertainty and alternative inner-screen calibration"],
            "pending_for_final_answer": ["Choose and freeze one coherent three-product/onboarding policy without treating exploratory alternatives as simultaneous final recommendations", "State what liquidity labels can and cannot identify, and give brief A--E explanations", "Clarify whether each portfolio uses all three assets, given zero-weight possibilities in explored long-only menus"]
        },
        "Q2": {
            "development_evidence_present": ["Frozen OOF Balanced client C uses its stated 30-year horizon, with annual/hold/myopic/finite execution compared on identical paths and targets", "This artifact explicitly restores exact one-third equity/bond/bill annually with the stated fee as the required baseline", "Median and P10 real CAGR, median and P90 MDD, actual trade fraction, mean annual log growth, fees, breach counts and CVaR90 pooled from all 120 path metrics", "Prespecified annual-policy worst OOF C severe-loss episode, with peak/trough/recovery and actual trades/costs", "No-decision-from-evaluation causal execution and finite-horizon approximation checks"],
            "pending_for_final_answer": ["Decide and state whether to adopt automated rebalancing or target-weight updates; a hold-versus-restore gate keeps its target fixed and is not a target-change rule", "Provide one customer-facing explanation of at most 50 English words for a trade, target change or human review", "Select the coherent final policy and define its authorized final evaluation; this development table does not supply the required final 121--180 assessment or decide a gate winner"]
        },
        "Q3": {
            "development_evidence_present": ["Modeling-assumption diagnostics include synthetic-market dependence/annual stationarity concerns, fee-observation MDD sensitivity and finite-mesh execution approximation", "Known final-set exposure, small effective evidence samples, liquidity nonidentification and risk transfer from static to dynamic policies have been recorded"],
            "pending_for_final_answer": ["Tie one chosen assumption/cost diagnostic to the actual frozen main recommendation and explicitly say whether that recommendation changes", "Freeze the important limitation, precise human-review/suspension trigger, and pending-review action; do not promise a fresh annual reset of the whole-horizon risk budget"]
        },
        "submission": ["Concise recommendation must cover all three portfolios, A--E, Balanced updates and main safeguard", "English report at most 10 A4 pages, font at least 11pt, members named", "AI use disclosure and independently checked calculations/code/references", "Source/supplement ZIP at most 50 MB; preserve dataset/problem CC BY-NC-SA attribution and license requirements"],
        "interpretation": "A development experiment can address an item without resolving the final modeling decision. No safeguard or gate is adopted by this checklist."
    }


def main():
    source_start, input_start = sources(), inputs()
    fixed_clients = cohort(pd.read_csv(FIXED / "client_path_metrics.csv"))
    finite_clients = cohort(pd.read_csv(FINITE / "client_path_metrics.csv"))
    fixed_targets = pd.read_csv(FIXED / "target_path_metrics.csv")
    finite_targets = pd.read_csv(FINITE / "target_path_metrics.csv")
    original = fixed_clients.loc[fixed_clients.policy == "annual_restore"].copy().sort_values("scenario_id")
    if not original.assigned.all() or set(original.assignment) != {"Balanced"}:
        raise RuntimeError("The required original C cohort must retain its recorded Balanced assignment")
    keys = original[["scenario_id", "fold", "context"]].reset_index(drop=True)
    check_cohort(original, keys)
    records, crosscheck_errors = [], {}
    for policy in POLICIES:
        if policy == "equal_weight_annual":
            frame = finite_targets.loc[(finite_targets.scope == "oof") & (finite_targets.horizon == 30) & (finite_targets.policy == policy)].copy()
            if not np.allclose(frame[["w_equity", "w_bond", "w_bill"]], 1 / 3, atol=1e-16, rtol=0):
                raise RuntimeError("Baseline is not the exact-thirds target")
            frame["assignment"] = "EqualWeightReference"
            frame["client"] = "C_horizon_reference"
        else:
            frame = finite_clients.loc[finite_clients.policy == policy].copy()
            if policy in ["myopic_gate", "finite_horizon_gate"]:
                frame = frame.loc[frame.mesh_step == .05].copy()
            if not frame.assigned.all() or set(frame.assignment) != {"Balanced"}:
                raise RuntimeError("Execution policy changed the frozen original assignment")
            joined = frame.merge(original[["scenario_id", "candidate_index"]], on="scenario_id", suffixes=("", "_original"), validate="one_to_one")
            if not np.array_equal(joined.candidate_index, joined.candidate_index_original):
                raise RuntimeError("Execution policy changed the original fold target")
            target_frame = finite_targets.loc[(finite_targets.scope == "oof") & (finite_targets.horizon == 30) & (finite_targets.policy == policy)].copy()
            if policy in ["myopic_gate", "finite_horizon_gate"]:
                target_frame = target_frame.loc[target_frame.mesh_step == .05]
            joined = frame.merge(target_frame, on=["context", "fold", "scenario_id", "policy", "candidate_index"], suffixes=("", "_target"), validate="one_to_one")
            for field in PATH_FIELDS:
                if not np.allclose(joined[field], joined[field + "_target"], atol=1e-12, rtol=1e-12):
                    raise RuntimeError(f"Client/target metric mismatch: {policy} {field}")
            for field in ["w_equity", "w_bond", "w_bill"]:
                frame[field] = joined.set_index("scenario_id").loc[frame.scenario_id, field].to_numpy()
        frame = frame.sort_values("scenario_id").reset_index(drop=True)
        check_cohort(frame, keys)
        if policy in ["annual_restore", "initial_then_hold"]:
            previous = fixed_clients.loc[fixed_clients.policy == policy].sort_values("scenario_id")
            previous_target = fixed_targets.loc[(fixed_targets.scope == "oof") & (fixed_targets.horizon == 30) & (fixed_targets.policy == policy)]
            previous_target = frame[["context", "fold", "scenario_id", "candidate_index"]].merge(previous_target, on=["context", "fold", "scenario_id", "candidate_index"], validate="one_to_one")
            difference = max(float(np.max(np.abs(frame[field].to_numpy() - previous[field].to_numpy()))) for field in PATH_FIELDS)
            difference = max(difference, max(float(np.max(np.abs(frame[field].to_numpy() - previous_target[field].to_numpy()))) for field in PATH_FIELDS))
            if difference > 1e-11: raise RuntimeError("Original execution ablation changed in the prototype outputs")
            crosscheck_errors[policy] = difference
        frame["reference_only"] = policy == "equal_weight_annual"
        frame["loss_tolerance"] = .4
        frame["breach"] = frame.mdd - .4 > NUMERIC_TOL
        keep = ["scenario_id", "fold", "context", "policy", "client", "assignment", "candidate_index", "w_equity", "w_bond", "w_bill", "reference_only", "horizon", "loss_tolerance", "breach"] + PATH_FIELDS
        records.append(frame[keep].copy())
    paths = pd.concat(records, ignore_index=True)
    summaries = [{"policy": policy, "reference_only": policy == "equal_weight_annual", **summarize(paths.loc[paths.policy == policy])} for policy in POLICIES]
    paired = []
    base = paths.loc[paths.policy == "annual_restore"].set_index("scenario_id")
    for policy in POLICIES[1:]:
        for row in paths.loc[paths.policy == policy].itertuples(index=False):
            origin = base.loc[row.scenario_id]
            paired.append({"scenario_id": int(row.scenario_id), "fold": int(row.fold), "policy": policy,
                           "reference_only": policy == "equal_weight_annual", "same_target_as_annual": policy != "equal_weight_annual",
                           **{f"delta_{field}": float(getattr(row, field) - origin[field]) for field in PATH_FIELDS},
                           "delta_breach": int(row.breach) - int(origin.breach)})
    if inputs() != input_start or sources() != source_start: raise RuntimeError("Source or input changed before output")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    paths.to_csv(OUT / "oof_paths.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_differences_vs_annual.csv", index=False)
    write_json(OUT / "summary.json", {"cohort": "Original OOF client C, assigned Balanced in all five folds, H=30, one observation for every scenario 1--120", "loss_tolerance": .4, "quantiles": SUMMARY_METADATA,
                                       "baseline": "Equal-weight annual reference restores the exact target (1/3,1/3,1/3), including charged initial allocation; it uses a different target from the fold-specific Balanced products.",
                                       "fees": "Mean nominal amount paid over 30 years; it is not a return percentage point. mean_annual_fee_fraction separately averages each actually paid fee divided by contemporaneous pre-trade nominal wealth.",
                                       "scope": "Descriptive OOF development summary. Fold training sets overlap and learned policies differ across folds; no pooled fixed-rule confidence certificate or unseen-final claim. No gate adoption or target-update choice is made here.",
                                       "summaries": summaries})
    write_json(OUT / "requirements_checklist.json", requirement_checklist())
    source_end, input_end = sources(), inputs()
    if source_end != source_start or input_end != input_start: raise RuntimeError("Source or input changed during output writing")
    output_names = ["summary.csv", "oof_paths.csv", "paired_differences_vs_annual.csv", "summary.json", "requirements_checklist.json"]
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "sources": source_start, "sources_at_end": source_end,
                                       "input_sha256": input_start, "input_sha256_at_end": input_end, "outputs_sha256": {name: sha(OUT / name) for name in output_names},
                                       "market_or_final_data_loaded": False, "model_refit": False, "model_choice_made": False,
                                       "checks": {"paths_per_policy": 120, "unique_ids_per_policy": list(range(1, 121)), "fold_size": 24,
                                                  "all_policies_original_fold_mapping": True, "client_and_target_metrics_match": True,
                                                  "original_fixed_ablation_max_abs_error": crosscheck_errors,
                                                  "quantiles_taken_after_pooling_paths": True, "cvar90_equals_worst_12_mean": True,
                                                  "paired_difference_rows": len(paired), "long_path_rows": len(paths)}})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
