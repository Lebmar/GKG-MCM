"""Execute the fixed, previously declared state inventory without policy fitting."""
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
from experimental_execution.fixed_state_counterfactual import continue_from_state, compare_actions

PARENT = ROOT / "results/model_review/nested_dynamic_calibration_v1"
PROTOCOL = ROOT / "docs/fixed_state_forecast_diagnostic_protocol.json"
OUT = ROOT / "results/model_review/fixed_state_forecast_diagnostic_v1"
STATE_FIELDS = ("holdings", "cpi", "real_wealth", "real_peak", "running_mdd", "ever_breached",
                "actions", "fees", "moved", "posttrade_holdings")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sources():
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "src/safeinvest/core.py",
             ROOT / "src/safeinvest/data.py", ROOT / "src/safeinvest/__init__.py",
             ROOT / "research/experimental_execution/fixed_state_counterfactual.py",
             ROOT / "tests/test_fixed_state_counterfactual.py",
             ROOT / "tests/test_fixed_state_diagnostic_runner.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def inputs(p):
    result = {f: sha(ROOT / f) for f in p["frozen_input_sha256"]}
    if result != p["frozen_input_sha256"]: raise RuntimeError("A frozen input changed")
    return result


def analysis_scope(scope, context):
    if scope == "validation": return "inner_validation"
    return "full_development_descriptive" if context == "full_development" else "outer_oof"


def summarize(frame):
    result = []
    specs = [("analysis_scope", ["analysis_scope"]),
             ("context", ["analysis_scope", "context"]), ("year", ["analysis_scope", "year"])]
    for level, keys in specs:
        for values, g in frame.groupby(keys, sort=False):
            values = values if isinstance(values, tuple) else (values,)
            identity = dict(zip(keys, values))
            result.append({"level": level, "analysis_scope": identity["analysis_scope"],
                "context": identity.get("context"), "year": int(identity["year"]) if "year" in identity else None,
                "active_state_n": len(g), "unique_scenario_n": g.scenario_id.nunique(),
                "unique_job_n": g.job_id.nunique(), "unique_group_n": g.group.nunique(),
                "in_training_fan_state_n": int(g.in_training_fan.sum()),
                "chosen_restore_n": int(g.chosen_action.eq("restore").sum()),
                "predicted_growth_positive_n": int(g.predicted_growth_sign.eq(1).sum()),
                "realized_growth_positive_n": int(g.realized_suffix_growth_sign.eq(1).sum()),
                "growth_sign_agree_n": int(g.growth_sign_agreement.sum()),
                "score_action_agree_n": int(g.score_action_agreement.sum()),
                "realized_hold_event_n": int(g.realized_hold_event.sum()),
                "realized_restore_event_n": int(g.realized_restore_event.sum()),
                **{"mean_" + col: float(g[col].mean()) for col in ["delta_log_residual", "score_residual",
                    "two_branch_realized_score_shortfall", "predicted_hold_bad_fraction", "predicted_restore_bad_fraction"]},
                "mean_abs_delta_log_residual": float(g.delta_log_residual.abs().mean()),
                "mean_abs_score_residual": float(g.score_residual.abs().mean())})
    return result


def main():
    started = time.perf_counter(); p = json.loads(PROTOCOL.read_text())
    if p.get("execution_ready") is not True: raise RuntimeError("Protocol/source review is incomplete")
    start_source, start_input = sources(), inputs(p)
    if p["execution_sources_sha256"] != {k: v for k, v in start_source.items() if k != str(PROTOCOL.relative_to(ROOT))}:
        raise RuntimeError("Execution source fingerprints are not frozen")
    if p["data"] != {"scenario_ids": list(range(1, 121)), "allow_final": False}:
        raise ValueError("Only the guarded development inventory is authorized")
    if OUT.exists(): raise RuntimeError("Use a new protocol/archive; preserve existing evidence")
    q_audit = json.loads((PARENT / "independent_audit.json").read_text())
    identity_fields = ("scope", "group", "scenario_id", "year", "training_n", "status")
    original_identities = [{k: x[k] for k in identity_fields} for x in q_audit["q_subset_results"]]
    if p["identities"] != original_identities or p["nonautomated_jobs"] != q_audit["q_nonautomated_job_skips"]:
        raise ValueError("Previously declared Q identities or no-replacement skips changed")
    if len(original_identities) != 81 or sum(x["status"] == "audited" for x in original_identities) != 79:
        raise ValueError("Require all 79 active and two pending identities")
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    parent_protocol = json.loads((ROOT / "docs/nested_dynamic_calibration_protocol.json").read_text())
    jobs = {x["job_id"]: x for x in parent_protocol["jobs"]}
    old_groups = json.loads((PARENT / "trace_groups.json").read_text())
    archives = {}
    for scope, name in (("actual", "actual_traces.npz"), ("validation", "validation_traces.npz")):
        with np.load(PARENT / name, allow_pickle=False) as f: archives[scope] = {k: f[k] for k in f.files}
    inventory, branches, comparisons, trace_groups, states = [], [], [], {}, {}
    for slot, item in enumerate(original_identities, 1):
        scope, key, sid, year = item["scope"], item["group"], item["scenario_id"], item["year"]
        group = old_groups[scope][key]; meta = group["metadata"]; job = jobs[meta["job_id"]]
        arm = {name[len(key) + 2:]: value for name, value in archives[scope].items() if name.startswith(key + "__")}
        i, t, state_id = group["scenario_ids"].index(sid), year - 1, f"q{slot:03d}"
        if sid != group["q_subset_scenario_id"] or year not in group["q_subset_years"]:
            raise ValueError("Saved subset metadata and fixed inventory disagree")
        common = {"state_id": state_id, "archive_scope": scope, "analysis_scope": analysis_scope(scope, meta["context"]),
            "group": key, "job_id": meta["job_id"], "context": meta["context"], "scenario_id": sid, "year": year,
            "training_n": item["training_n"], "in_training_fan": sid in job["training_ids"],
            "chosen_action": str(arm["actions"][i, t]), "selected_candidate_order": group["selected_candidate_order"]}
        if bool(arm["pending_at_decision"][i, t]):
            if item["status"] != "pending_skip_no_replacement": raise ValueError("Pending skip mismatch")
            inventory.append({**common, "status": "pending_skip_no_replacement"}); continue
        if item["status"] != "audited": raise ValueError("Active state not in the frozen audited set")
        x, cp, pk, dd = arm["holdings"][i, t], float(arm["cpi"][i, t]), float(arm["real_peak"][i, t]), float(arm["running_mdd"][i, t])
        target = np.asarray(arm["targets"], float); annual = np.asarray(arm["annual_prices"], float).reshape(-1)
        if annual.shape != (1,) or not np.isfinite(annual[0]): raise ValueError("Require the single saved finite selected price")
        price = float(annual[0]); n = int(arm["forecast_n"][i, t])
        if n != item["training_n"] or n != len(job["training_ids"]): raise ValueError("Forecast denominator changed")
        common.update({"annual_price": price, "horizon": 30, "loss_tolerance": .4,
            "initial_equity": float(x[0]), "initial_bond": float(x[1]), "initial_bill": float(x[2]),
            "initial_cpi": cp, "initial_real_peak": pk, "initial_running_mdd": dd,
            "target_equity": float(target[0]), "target_bond": float(target[1]), "target_bill": float(target[2])})
        predicted, realized = {}, {}
        for action in ("hold", "restore"):
            bad_count = int(arm["qbad_" + action + "_count"][i, t])
            if not 0 <= bad_count <= n: raise ValueError("Saved fan counts are invalid")
            predicted[action] = {"log": float(arm["qg_" + action][i, t]), "bad_fraction": bad_count / n}
            value = continue_from_state(x, cp, pk, dd, target, .4, returns[sid - 1], inflation[sid - 1],
                year_index=t, original_horizon=30, current_action=action, ever_breached=bool(arm["ever_breached"][i, t]), cost_rate=.0025)
            realized[action] = value; prefix = state_id + "__" + action
            for field in STATE_FIELDS: states[prefix + "__" + field] = value[field]
            branches.append({**common, "current_action": action, "predicted_total_log_growth": predicted[action]["log"],
                "predicted_bad_count": bad_count, "fan_n": n, "predicted_bad_fraction": bad_count / n,
                "realized_suffix_total_log_growth": value["total_log_growth"], "realized_suffix_breach": value["breach"],
                "history_mdd": value["history_mdd"], "terminal_real": value["terminal_real"],
                "nominal_fees": value["nominal_fees"], "nominal_moved": value["nominal_moved"],
                "first_crossing_original_year": value["first_crossing_original_year"],
                "log_residual": value["total_log_growth"] - predicted[action]["log"],
                "event_residual": int(value["breach"]) - bad_count / n})
        contrast = compare_actions(predicted, realized, annual_price=price, original_horizon=30, chosen_action=common["chosen_action"])
        if contrast["predicted_action"] != common["chosen_action"] or abs(contrast["predicted_score"] - float(arm["predicted_score"][i, t])) > 3e-12:
            raise ValueError("Saved action or technical score does not match the frozen state")
        comparisons.append({**common, **contrast, "realized_hold_event": realized["hold"]["breach"],
            "realized_restore_event": realized["restore"]["breach"],
            "predicted_hold_bad_fraction": predicted["hold"]["bad_fraction"],
            "predicted_restore_bad_fraction": predicted["restore"]["bad_fraction"]})
        inventory.append({**common, "status": "active_evaluated"})
        trace_groups[state_id] = {**common, "npz_prefixes": {a: state_id + "__" + a for a in ("hold", "restore")},
                                  "fields": list(STATE_FIELDS)}
        if time.perf_counter() - started > 60 * p["budget_minutes"]: raise RuntimeError("Diagnostic time budget reached")
    for item in p["nonautomated_jobs"]:
        job = jobs[item["job_id"]]
        for year in item["years"]:
            inventory.append({"state_id": f"q{len(inventory) + 1:03d}", "archive_scope": "actual", "analysis_scope": "outer_oof",
                "job_id": item["job_id"], "context": job["parent_context"], "scenario_id": item["scenario_id"],
                "year": year, "training_n": len(job["training_ids"]), "status": "nonautomated_skip_no_replacement"})
    if (len(inventory), len(branches), len(comparisons)) != (90, 158, 79): raise RuntimeError("Diagnostic inventory is incomplete")
    OUT.mkdir(parents=True)
    tables = {"state_inventory.csv": inventory, "branch_paths.csv": branches,
              "state_comparisons.csv": comparisons, "summary.csv": summarize(pd.DataFrame(comparisons))}
    for name, rows in tables.items(): pd.DataFrame(rows).to_csv(OUT / name, index=False)
    writej(OUT / "trace_groups.json", trace_groups); np.savez_compressed(OUT / "branch_states.npz", **states)
    if sources() != start_source or inputs(p) != start_input: raise RuntimeError("Sources or inputs changed during the diagnostic")
    writej(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "elapsed_seconds": time.perf_counter() - started, "sources": start_source, "sources_at_end": sources(),
        "input_sha256": start_input, "input_sha256_at_end": inputs(p),
        "outputs_sha256": {f.name: sha(f) for f in OUT.iterdir() if f.is_file()},
        "row_counts": {name: len(rows) for name, rows in tables.items()}, "scenario_ids_loaded": ids.tolist(),
        "final_scenarios_loaded": False, "new_policy_fit_or_selection": False,
        "analysis_scope_counts": pd.DataFrame(comparisons).analysis_scope.value_counts().to_dict(),
        "limitations": ["Finite previously fixed states; not a conditional probability calibration test or independent samples",
            "Current action then B is distinct from the learned adaptive policy's realized continuation",
            "Full120 states include the actual scenario in the training fan; explicitly descriptive",
            "Outer OOF has only six active fixed states, two fits; repeated development and overlapping folds",
            "Two-branch realized score shortfall is not an implementable oracle strategy or policy regret",
            "No rejected outer path receives imputed investment outcomes; no parameter or service decision is changed"]})
    print(json.dumps({"status": "complete", "active_states": 79, "branches": 158, "output": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__": main()
