"""Predeclared original-path diagnostic of an unconditioned calendar fan.

Fixed lambda=1 state comparator; no policy fitting, parameter selection,
trading replay, final-data access, Q calibration or risk certification.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
from safeinvest.data import load_paths
from experimental_execution.calendar_state_diagnostic import (
    CHANNELS, TARGETS, market_log_variables, diagnostic_arrays,
    fit_calendar_state, predict_calendar_state, per_path_losses, validate_folds)

PROTOCOL = ROOT / "docs/forecast_assumption_diagnostic_protocol.json"
OUT = ROOT / "results/model_review/forecast_assumptions_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    if not records:
        raise ValueError("No empty diagnostic table is expected")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def compute_diagnostic(returns, inflation, ids, folds):
    ids = list(map(int, ids)); validate_folds(folds, ids)
    z = market_log_variables(returns, inflation)
    if len(z) != len(ids):
        raise ValueError("Market rows must match original IDs")
    x, targets = diagnostic_arrays(z)
    position = {sid: i for i, sid in enumerate(ids)}
    annual, paths, coefs, scales, calendar = [], [], [], [], []
    for fold in folds:
        fold_id = fold["fold"]
        ti = [position[s] for s in fold["training_ids"]]
        ei = [position[s] for s in fold["evaluation_ids"]]
        for target in TARGETS:
            y = targets[target]
            model = fit_calendar_state(x[ti], y[ti], penalty=1.)
            baseline, state = predict_calendar_state(model, x[ei])
            loss = per_path_losses(y[ei], baseline, state, model["y_scale"])
            lo, hi = np.quantile(y[ti], [.1, .9], axis=0, method="linear")
            minimum, maximum = y[ti].min(axis=0), y[ti].max(axis=0)
            covered = (y[ei] >= lo) & (y[ei] <= hi)
            out_range = (y[ei] < minimum) | (y[ei] > maximum)
            lag_out_range = (x[ei] < x[ti].min(axis=0)) | (x[ei] > x[ti].max(axis=0))
            for j, channel in enumerate(CHANNELS):
                scales.append({"fold": fold_id, "target": target, "channel": channel,
                    "training_paths": len(ti), "training_year_rows": len(ti)*29,
                    "feature_residual_rms": float(model["x_scale"][j]),
                    "outcome_residual_rms": float(model["y_scale"][j]), "fixed_lambda": 1.})
                for k, predictor in enumerate(CHANNELS):
                    coefs.append({"fold": fold_id, "target": target, "outcome": channel,
                        "lag_predictor": predictor,
                        "standardized_coefficient": float(model["standardized_coefficients"][k,j])})
                for t in range(29):
                    calendar.append({"fold": fold_id, "target": target, "channel": channel,
                        "decision_original_year": t+2,
                        "training_lag_mean": float(model["x_calendar_mean"][t,j]),
                        "training_outcome_mean": float(model["y_calendar_mean"][t,j]),
                        "training_outcome_q10": float(lo[t,j]), "training_outcome_q90": float(hi[t,j]),
                        "training_outcome_min": float(minimum[t,j]), "training_outcome_max": float(maximum[t,j])})
                for a, sid in enumerate(fold["evaluation_ids"]):
                    paths.append({"fold": fold_id, "scenario_id": sid, "target": target,
                        "channel": channel, "annual_scored_rows": 29,
                        **{key: float(value[a,j]) for key,value in loss.items()},
                        "interval_10_90_coverage": float(covered[a,:,j].mean()),
                        "outside_training_target_range_fraction": float(out_range[a,:,j].mean()),
                        "lag_outside_training_channel_range_fraction": float(lag_out_range[a,:,j].mean())})
                    for t in range(29):
                        annual.append({"fold": fold_id, "scenario_id": sid, "target": target,
                            "channel": channel, "decision_original_year": t+2,
                            "last_observed_original_year": t+1,
                            "remaining_original_years": 29-t,
                            "actual_log_quantity": float(y[ei[a],t,j]),
                            "calendar_mean_prediction": float(baseline[a,t,j]),
                            "fixed_state_prediction": float(state[a,t,j]),
                            "empirical_q10": float(lo[t,j]), "empirical_q90": float(hi[t,j]),
                            "inside_empirical_interval": bool(covered[a,t,j]),
                            "outside_training_range": bool(out_range[a,t,j])})
    return {"annual_predictions": annual, "path_losses": paths,
            "coefficients": coefs, "training_scales": scales, "calendar_training_summary": calendar}


def aggregate(records, *, by_fold):
    keys = sorted({((r["fold"],r["target"],r["channel"]) if by_fold else
                    (r["target"],r["channel"])) for r in records})
    result=[]
    for key in keys:
        group=[r for r in records if (((r["fold"],r["target"],r["channel"]) if by_fold else
                                      (r["target"],r["channel"])) == key)]
        target, channel = key[-2:]
        row={"target": target, "channel": channel, "original_path_n":len(group),
             "annual_rows_per_original_path":29}
        if by_fold: row={"fold":key[0],**row}
        for name in ("baseline_mse","state_mse","baseline_standardized_mse","state_standardized_mse",
                     "baseline_bias","state_bias","interval_10_90_coverage",
                     "outside_training_target_range_fraction","lag_outside_training_channel_range_fraction"):
            row[name]=float(np.mean([r[name] for r in group]))
        row["state_to_calendar_mse_ratio"] = row["state_standardized_mse"] / row["baseline_standardized_mse"] if row["baseline_standardized_mse"]>0 else None
        diff=np.array([r["state_standardized_mse"]-r["baseline_standardized_mse"] for r in group])
        row["paired_standardized_loss_difference"] = float(diff.mean())
        row["path_loss_difference_q10"],row["path_loss_difference_q50"],row["path_loss_difference_q90"]=map(float,np.quantile(diff,[.1,.5,.9],method="linear"))
        row["path_fraction_state_lower_loss"] = float(np.mean(diff<0))
        result.append(row)
    return result


def vector_summary(path_records, fold_summaries):
    rows=[]
    for target in TARGETS:
        selected=[r for r in path_records if r["target"]==target and r["channel"] in CHANNELS[:3]]
        ids=sorted({r["scenario_id"] for r in selected})
        vectors=[]
        for sid in ids:
            group=[r for r in selected if r["scenario_id"]==sid]
            if len(group)!=3:raise ValueError("Three assets required for every vector loss")
            vectors.append({"scenario_id":sid,"fold":group[0]["fold"],
                "baseline":float(np.mean([r["baseline_standardized_mse"] for r in group])),
                "state":float(np.mean([r["state_standardized_mse"] for r in group]))})
        b=np.array([r['baseline'] for r in vectors]);s=np.array([r['state'] for r in vectors]);diff=s-b
        folds=[]
        for f in sorted({r['fold'] for r in vectors}):
            rr=[r for r in vectors if r['fold']==f]
            folds.append({"fold":f,"original_path_n":len(rr),
                "state_to_calendar_standardized_mse_ratio":float(np.mean([r['state'] for r in rr])/np.mean([r['baseline'] for r in rr]))})
        rows.append({"target":target,"primary_metric":"path-averaged three-real-asset standardized MSE; inflation excluded",
            "original_path_n":len(vectors),"baseline_standardized_vector_mse":float(b.mean()),
            "state_standardized_vector_mse":float(s.mean()),"state_to_calendar_mse_ratio":float(s.mean()/b.mean()),
            "paired_mean_loss_difference":float(diff.mean()),"path_loss_difference_q10_q50_q90":list(map(float,np.quantile(diff,[.1,.5,.9],method='linear'))),
            "path_fraction_state_lower_loss":float(np.mean(diff<0)),"fold_results":folds,
            "interpretation":"Fixed-comparator descriptive development-path prediction diagnostic. No confidence interval, IID annual-row inference, learned trading-Q claim or future risk guarantee."})
    return rows


def ensure_fresh_output(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Diagnostic archive is not empty; preserve existing results")


def main():
    ensure_fresh_output(OUT)
    p=json.loads(PROTOCOL.read_text())
    if not p["execution_ready"] or p["fixed_comparator"]["ridge_lambda"] != 1.:
        raise ValueError("Only the frozen lambda=1 diagnostic may run")
    frozen=p["execution_sources_sha256"]
    source_before={path:sha(ROOT/path) for path in frozen}
    if source_before!=frozen:raise ValueError("Frozen diagnostic source changed")
    inputs=p["frozen_input_sha256"]
    input_before={path:sha(ROOT/path) for path in inputs}
    if input_before!=inputs:raise ValueError("Frozen diagnostic input changed")
    if p["data"]["allow_final"] is not False or p["data"]["scenario_ids"] != list(range(1,121)):
        raise ValueError("Only original development paths 1..120 are allowed")
    validate_folds(p["folds"],p["data"]["scenario_ids"])
    returns,inflation,ids=load_paths(ROOT/p["data"]["path"],p["data"]["scenario_ids"],allow_final=False)
    tables=compute_diagnostic(returns,inflation,ids,p["folds"])
    tables["fold_summary"]=aggregate(tables["path_losses"],by_fold=True)
    tables["summary"]=aggregate(tables["path_losses"],by_fold=False)
    findings={"primary_vector_results":vector_summary(tables["path_losses"],tables["fold_summary"]),
        "limitations":p["interpretation_boundaries"],"no_policy_or_parameter_changes":True,
        "all_training_centering_scaling":True,"annual_rows_treated_independent":False,
        "final_scenarios_loaded":False,"nominal_interval_label":.8,
        "interval_definition":"Training same-calendar 10th to 90th quantiles; numpy linear method; inclusive endpoints. Descriptive marginal interval only."}
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():write_csv(OUT/f"{name}.csv",rows)
    write_json(OUT/'findings.json',findings)
    source_after={path:sha(ROOT/path) for path in frozen}
    input_after={path:sha(ROOT/path) for path in inputs}
    if source_before!=source_after or input_before!=input_after:raise ValueError("Inputs/sources changed during diagnostic")
    outputs={str(path.relative_to(ROOT)):sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name!='manifest.json'}
    manifest={"completed_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":sha(PROTOCOL),"sources":source_before,"sources_at_end":source_after,
        "input_sha256":input_before,"input_sha256_at_end":input_after,"outputs_sha256":outputs,
        "scenario_ids_loaded":ids.tolist(),"final_scenarios_loaded":False,
        "row_counts":{name:len(rows) for name,rows in tables.items()},
        "complete_original_paths":120,"out_of_fold_splits":5,"no_trading_policy_evaluated":True,
        "results_select_parameters":False}
    write_json(OUT/'manifest.json',manifest)
    print(json.dumps({"out":str(OUT.relative_to(ROOT)),"rows":manifest['row_counts'],"primary":findings['primary_vector_results']},indent=2))


if __name__=='__main__':main()
