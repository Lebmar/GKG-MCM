"""Fixed-state contrast of every metadata-matched pair in the frozen11 cases.

No policy is fitted, selected or replayed. The sole state is each original
pair's first saved action disagreement; all other years remain unavailable
to this diagnostic's action estimator. A separately frozen protocol is
required before loading market arrays or account values.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
PROTOCOL = ROOT / "docs/matched_parameter_fan_contrast_protocol.json"
OUT = ROOT / "results/model_review/matched_parameter_fan_contrast_v1"
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
KERNELS = ("research/experimental_execution/risk_priced_fan_rollout.py",
           "research/experimental_execution/fan_rollout.py",
           "research/experimental_execution/review_pending.py", "src/safeinvest/core.py")
ENDPOINT_FIELDS = ("holdings", "weights", "nominal_wealth", "real_wealth", "cpi",
                   "real_peak", "current_drawdown", "running_mdd", "ever_breached")
PREVIOUS_PERIOD_FIELDS = ("posttrade_holdings", "fees", "moved", "target_execution",
                          "pending_at_decision", "trades")
TOLERANCE = {"atol": 2e-10, "rtol": 5e-13}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def readj(path):
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as stream:
            return json.load(stream)
    return json.loads(p.read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def enrich_inventory(parent, manifests):
    """Metadata only: no trajectory, first-disagreement year or outcome inputs."""
    out = []
    for original in parent["inventory"]:
        item = dict(original)
        ids = item["training_ids"]
        order = item["selected_candidate_order"]
        if (isinstance(order, bool) or not isinstance(order, int) or not 0 <= order < len(PRICES)
                or not ids or ids != sorted(set(ids))
                or any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 120 for x in ids)
                or item["training_fan_n"] != len(ids)):
            raise ValueError("Invalid frozen fan/controller metadata")
        manifest = manifests[item["folder"]]
        item.update({"original_H": parent["original_H"], "loss_tolerance": parent["loss_tolerance"],
                     "cost_rate": parent["cost_rate"], "annual_eta": PRICES[order],
                     "kernel_sha256": {key: manifest["sources"][key] for key in KERNELS},
                     "development_arrays_sha256": manifest["development_arrays_sha256"]})
        out.append(item)
    if len({item["pair_id"] for item in out}) != len(out):
        raise ValueError("Duplicated original case identity")
    return out


def eligible_pairs(inventory):
    """Retain ALL same-parameter/same-market/different-fan case pairs."""
    result = []
    common = ("scenario_id", "target", "annual_eta", "original_H", "loss_tolerance",
              "cost_rate", "kernel_sha256", "development_arrays_sha256")
    for left, right in combinations(sorted(inventory, key=lambda a: a["pair_id"]), 2):
        if (all(left[field] == right[field] for field in common)
                and left["training_ids"] != right["training_ids"]):
            result.append({"contrast_id": left["pair_id"] + "__vs__" + right["pair_id"],
                           "left_case_id": left["pair_id"], "right_case_id": right["pair_id"]})
    return result


def first_action_disagreement(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Two complete equally long saved action paths required")
    if not np.isin(a, ("hold", "restore")).all() or not np.isin(b, ("hold", "restore")).all():
        raise ValueError("Unknown original action")
    indexes = np.flatnonzero(a != b)
    return None if not len(indexes) else int(indexes[0])


def close_array(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("State dtype/shape mismatch: " + label)
    if a.dtype.kind in "biuUS":
        if not np.array_equal(a, b):
            raise ValueError("Discrete state mismatch: " + label)
        return {"field": label, "exact": True, "maximum_absolute_error": 0.}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Unexpected nonfinite physical state: " + label)
    np.testing.assert_allclose(a, b, **TOLERANCE, err_msg=label)
    return {"field": label, "exact": bool(np.array_equal(a, b)),
            "maximum_absolute_error": float(np.max(np.abs(a - b), initial=0.))}


def verify_prefix(left, right, tau):
    """tau is zero-based next return column; include its predecision endpoint."""
    if tau is None:
        return {"status": "no_action_disagreement", "tau_year": None, "checks": []}
    if isinstance(tau, bool) or not isinstance(tau, int) or tau < 1:
        raise ValueError("A post-inception active disagreement is required")
    checks = []
    for field in ENDPOINT_FIELDS:
        checks.append(close_array(left[field][:tau+1], right[field][:tau+1], field))
    for field in PREVIOUS_PERIOD_FIELDS:
        checks.append(close_array(left[field][:tau], right[field][:tau], field))
    for side in (left, right):
        if bool(side["ever_breached"][tau]) or bool(side["pending_at_decision"][tau]):
            raise ValueError("No active automatic gate at first disagreement")
    return {"status": "passed", "tau_year": tau + 1, "checks": checks,
            "atol": TOLERANCE["atol"], "rtol": TOLERANCE["rtol"],
            "common_state_source": "left original saved predecision state; right state matches above"}


def extract_account(archive, group, ids, scenario):
    if ids.count(scenario) != 1:
        raise ValueError("Original actual scenario identity unavailable/duplicated")
    index, prefix = ids.index(scenario), group + "__"
    common_metadata = {"targets", "loss_tolerances", "annual_prices"}
    result = {}
    for key in archive.files:
        if key.startswith(prefix):
            field = key[len(prefix):]
            value = archive[key]
            result[field] = value.copy() if field in common_metadata else value[index].copy()
    if not result:
        raise ValueError("Original saved model account unavailable")
    return result


def terminal_outcome(account, horizon):
    real = float(account["real_wealth"][-1])
    total = math.log(real) - math.log(float(account["real_wealth"][0]))
    crossing = int(account["first_crossing"])
    return {"terminal_real_wealth": real, "terminal_nominal_wealth": float(account["nominal_wealth"][-1]),
            "total_net_log_real_growth": total, "annual_log_real_growth": total/horizon,
            "cagr": float(np.expm1(total/horizon)), "original_H_MDD": float(account["running_mdd"][-1]),
            "ever_breached": bool(account["ever_breached"][-1]),
            "first_crossing_year": None if crossing < 0 else crossing,
            "nominal_fees": float(np.sum(account["fees"])),
            "trade_fraction_including_initial_and_pending": float(np.mean(account["trades"]))}


def fan_specifications(left, right):
    if (len(left["training_ids"]) != 120 or len(right["training_ids"]) != 96
            or left["scenario_id"] not in left["training_ids"]
            or left["scenario_id"] in right["training_ids"]
            or not set(right["training_ids"]) < set(left["training_ids"])):
        raise ValueError("The prespecified120/self-delete119/outer96 bridge does not apply")
    sid = left["scenario_id"]
    return [("full120", left["training_ids"], "left"),
            ("full_without_actual119", [x for x in left["training_ids"] if x != sid], None),
            ("outer96", right["training_ids"], "right")]


def validate_saved_Q(row, account, tau):
    if row["fan_n"] != int(account["forecast_n"][tau]):
        raise ValueError("Original saved forecast denominator differs")
    errors = {}
    for key in ("qg_hold", "qg_restore", "predicted_score", "predicted_growth_advantage_restore"):
        a, b = float(row[key]), float(account[key][tau])
        if not np.isfinite(b) or not np.isclose(a, b, **TOLERANCE):
            raise ValueError("Original saved Q/score mismatch: " + key)
        errors[key] = abs(a - b)
    for key in ("qbad_hold_count", "qbad_restore_count", "predicted_breach_count_delta_restore"):
        if row[key] != int(account[key][tau]):
            raise ValueError("Original saved bad count mismatch: " + key)
    if row["chosen_action"] != str(account["actions"][tau]) or row["gate_reason"] != str(account["gate_reason"][tau]):
        raise ValueError("Original saved action/reason mismatch")
    return {"status": "passed", "maximum_absolute_Q_score_error": max(errors.values()),
            "per_field_errors": errors, "counts_action_reason_denominator_exact": True}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run():
    if OUT.exists():
        raise RuntimeError("Refuse to overwrite a prior contrast archive")
    protocol_digest = sha(PROTOCOL)
    p = readj(PROTOCOL)
    if p.get("execution_ready") is not True or not p.get("execution_frozen_utc"):
        raise RuntimeError("Protocol/source freeze required before any numerical execution")
    if (p.get("data",{}).get("scenario_ids") != list(range(1,121))
            or p["data"].get("allow_final") is not False
            or p.get("original_H") != 30 or p.get("loss_tolerance") != .4
            or p.get("cost_rate") != .0025 or p.get("output_directory") != str(OUT.relative_to(ROOT))):
        raise ValueError("Original development/accounting contract changed")
    snapshot = {**p["frozen_input_sha256"], **p["execution_sources_sha256"]}
    if any(sha(ROOT / rel) != value for rel, value in snapshot.items()):
        raise RuntimeError("Frozen source/input changed")
    parent = readj(ROOT / p["parent_protocol"])
    manifests = {folder: readj(ROOT/"results/model_review"/folder/"manifest.json") for folder in p["parent_folders"]}
    inventory = enrich_inventory(parent, manifests)
    selected = eligible_pairs(inventory)
    if selected != p["eligible_pairs"] or canonical_sha(parent["inventory"]) != p["parent_inventory_canonical_sha256"]:
        raise ValueError("Complete metadata-derived eligible pair set changed")
    cases = {item["pair_id"]: item for item in inventory}
    groups = {folder: readj(ROOT/"results/model_review"/folder/"trace_groups.json")["actual"] for folder in p["parent_folders"]}
    static = {folder: readj(ROOT/"results/model_review"/folder/filename) for folder,filename in p["static_files"].items()}
    for folder, manifest in manifests.items():
        if readj(ROOT/"results/model_review"/folder/"independent_audit.json")["status"] != "passed":
            raise RuntimeError("Unpassed original account audit")
        if manifest["development_arrays_sha256"] != next(iter(manifests.values()))["development_arrays_sha256"]:
            raise ValueError("Original market array fingerprint differs")
        for key in KERNELS:
            if manifest["sources"][key] != manifest["sources_at_end"][key] or sha(ROOT/key) != manifest["sources"][key]:
                raise ValueError("Original frozen execution kernel differs")
    from safeinvest.data import load_paths
    from experimental_execution.fan_rollout import evaluate_actions
    from experimental_execution.risk_priced_fan_rollout import priced_choices
    returns, inflation, ids = load_paths(ROOT/p["data"]["path"], range(1,121), allow_final=False)
    if p["data"]["scenario_ids"] != list(range(1,121)) or p["data"]["allow_final"] is not False or not np.array_equal(ids,np.arange(1,121)):
        raise ValueError("Guarded original development IDs changed")
    digest = hashlib.sha256()
    for a in (ids,returns,inflation): digest.update(np.ascontiguousarray(a).tobytes())
    if any(manifest["development_arrays_sha256"] != digest.hexdigest() for manifest in manifests.values()):
        raise ValueError("Loaded market differs from saved models")
    archives = {folder: np.load(ROOT/"results/model_review"/folder/"actual_traces.npz",allow_pickle=False) for folder in p["parent_folders"]}
    q_rows, member_rows, pair_results, branch_arrays = [], [], [], {}
    try:
        for comparison in selected:
            left,right = (cases[comparison[key]] for key in ("left_case_id","right_case_id"))
            accounts={}
            for label,item in (("left",left),("right",right)):
                group=groups[item["folder"]][item["model_group"]]; md=group["metadata"]
                bundle=static[item["folder"]][item["job_id"]]
                if (md["job_id"] != item["job_id"] or group["selected_candidate_order"] != item["selected_candidate_order"]
                        or md["horizon"] != item["original_H"] or md["loss_tolerance"] != item["loss_tolerance"]
                        or bundle["refit"]["training_ids"] != item["training_ids"]
                        or bundle["q2_case"]["product"]["weights"] != item["target"]
                        or bundle["training_price_selections"][str(item["selected_beta"])]["selected_candidate_order"] != item["selected_candidate_order"]):
                    raise ValueError("Saved original fit/target/price/fan identity changed")
                account=extract_account(archives[item["folder"]],item["model_group"],group["scenario_ids"],item["scenario_id"])
                if (not np.array_equal(account["targets"],np.array(item["target"]))
                        or not np.array_equal(account["annual_prices"],np.array([item["annual_eta"]]))
                        or not np.array_equal(account["loss_tolerances"],np.array([item["loss_tolerance"]]))):
                    raise ValueError("Archived target/eta/L metadata changed")
                accounts[label]=account
            A,B=accounts["left"],accounts["right"]
            tau=first_action_disagreement(A["actions"],B["actions"])
            prefix=verify_prefix(A,B,tau)
            common_state=None if tau is None else {field: np.asarray(A[field][tau]).tolist() for field in ENDPOINT_FIELDS}
            actual_digest=hashlib.sha256()
            for value in (np.array([left["scenario_id"]],dtype=np.int64),returns[left["scenario_id"]-1],inflation[left["scenario_id"]-1]):
                actual_digest.update(np.ascontiguousarray(value).tobytes())
            pair={**comparison,"original_scenario_id":left["scenario_id"],"tau_year":None if tau is None else tau+1,
                  "left_case":left,"right_case":right,"pre_tau_state_identity":prefix,"common_state":common_state,
                  "same_actual_original_market_path_sha256":actual_digest.hexdigest(),
                  "original_full_horizon_outcomes":{label:terminal_outcome(account,left["original_H"]) for label,account in accounts.items()},
                  "saved_reproduction":{},"state_policy_changes":False}
            outcomes=pair["original_full_horizon_outcomes"]
            pair["original_outcome_right_minus_left"]={key:outcomes["right"][key]-outcomes["left"][key] for key in ("terminal_real_wealth","total_net_log_real_growth","original_H_MDD","nominal_fees")}
            for fan_label,fan_ids,original_side in fan_specifications(left,right):
                row={"contrast_id":comparison["contrast_id"],"original_scenario_id":left["scenario_id"],
                     "year":None if tau is None else tau+1,"fan_label":fan_label,"fan_n":len(fan_ids),
                     "fan_ids_sha256":canonical_sha(fan_ids),"actual_ID_in_fan":left["scenario_id"] in fan_ids,
                     "original_H":left["original_H"],"annual_eta":left["annual_eta"],"cost_rate":left["cost_rate"],
                     **dict.fromkeys(("qg_hold","qg_restore","qbad_hold_count","qbad_restore_count","predicted_growth_advantage_restore",
                                      "predicted_breach_count_delta_restore","predicted_score","chosen_action","gate_reason")),
                     "availability":"no_action_disagreement" if tau is None else "active_fixed_state"}
                if tau is not None:
                    indexes=np.asarray(fan_ids)-1
                    values=evaluate_actions(A["holdings"][tau],A["cpi"][tau],A["real_peak"][tau],A["running_mdd"][tau],
                        A["ever_breached"][tau],returns[indexes],inflation[indexes],year_index=tau,target=left["target"],
                        L=left["loss_tolerance"],cost_rate=left["cost_rate"],horizon=left["original_H"])
                    h,r=values["hold"],values["restore"]
                    choice,score,reason=priced_choices(h["mean_log_growth"],r["mean_log_growth"],h["breach_count"],r["breach_count"],len(fan_ids),left["annual_eta"],left["original_H"])
                    row.update({"qg_hold":float(h["mean_log_growth"][0]),"qg_restore":float(r["mean_log_growth"][0]),
                        "qbad_hold_count":int(h["breach_count"][0]),"qbad_restore_count":int(r["breach_count"][0]),
                        "predicted_growth_advantage_restore":float(r["mean_log_growth"][0]-h["mean_log_growth"][0]),
                        "predicted_breach_count_delta_restore":int(r["breach_count"][0]-h["breach_count"][0]),
                        "predicted_score":float(score[0]),"chosen_action":"restore" if choice[0] else "hold","gate_reason":str(reason[0])})
                    for action,arr in (("hold",h),("restore",r)):
                        prefix_key=comparison["contrast_id"]+"__"+fan_label+"__"+action
                        branch_arrays[prefix_key+"__total_log_growth"]=arr["total_log_growth"].copy()
                        branch_arrays[prefix_key+"__ever_breached"]=arr["ever_breached"].copy()
                    for j,sid in enumerate(fan_ids):
                        member_rows.append({"contrast_id":comparison["contrast_id"],"fan_label":fan_label,"year":tau+1,
                            "fan_scenario_id":sid,"hold_total_log_growth":float(h["total_log_growth"][0,j]),
                            "restore_total_log_growth":float(r["total_log_growth"][0,j]),
                            "hold_ever_breached":bool(h["ever_breached"][0,j]),"restore_ever_breached":bool(r["ever_breached"][0,j])})
                    if original_side is not None:
                        pair["saved_reproduction"][original_side]=validate_saved_Q(row,accounts[original_side],tau)
                q_rows.append(row)
            pair_results.append(pair)
    finally:
        for archive in archives.values(): archive.close()
    if sha(PROTOCOL)!=protocol_digest or any(sha(ROOT/rel)!=value for rel,value in snapshot.items()):
        raise RuntimeError("Protocol/source/input changed during contrast")
    OUT.mkdir()
    writej(OUT/"pairs.json",pair_results)
    write_csv(OUT/"same_state_Q.csv",q_rows)
    if member_rows: write_csv(OUT/"fan_members.csv",member_rows)
    np.savez_compressed(OUT/"fan_branch_arrays.npz",**branch_arrays)
    writej(OUT/"manifest.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"status":"passed",
        "protocol_sha256":protocol_digest,"sources_and_inputs_sha256":snapshot,
        "eligible_pairs":selected,"metadata_selection_uses_outcomes":False,"new_actual_policy_replays":0,"new_fits":0,
        "original_scenario_ids_loaded":ids.tolist(),"development_arrays_sha256":digest.hexdigest(),"final_scenarios_read":False,
        "fixed_state_Q_rows":len(q_rows),"fan_member_records":len(member_rows),"output_sha256":{x.name:sha(x) for x in sorted(OUT.iterdir())},
        "limitations":p["limitations"]})
    print(json.dumps({"status":"passed","pairs":len(pair_results),"Q_rows":len(q_rows),"fan_members":len(member_rows)}))


if __name__ == "__main__":
    run()
