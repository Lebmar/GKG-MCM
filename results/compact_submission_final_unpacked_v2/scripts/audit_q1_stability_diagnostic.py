"""Independent scalar saved-cache audit; imports no production diagnostic.

This validates postprocessing of the existing static cache, not the upstream
market simulator, conditional predictions, population risk or a new policy.
Source is frozen before this auditor loads any cache numerical arrays.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"docs/q1_stability_diagnostic_audit_plan.json"
FREEZE=ROOT/"docs/q1_stability_diagnostic_audit_source_freeze.json"
PROFILES=(("A",5,.1,"high"),("B",15,.25,"medium"),("C",30,.4,"low"),("D",25,.15,"medium"),("E",6,.35,"high"))
FLOORS={"high":.3,"medium":.1,"low":0.}
TOL=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):self.n=0;self.maximum_error=0.
    def equal(self,expected,actual,label=""):
        self.n+=1
        if expected is None or isinstance(expected,(bool,str)):
            if type(actual) is not type(expected) or actual!=expected:raise AssertionError(label+": identity mismatch")
        elif isinstance(expected,int):
            if isinstance(actual,bool) or not isinstance(actual,(int,float)) or actual!=expected:raise AssertionError(label+": integer mismatch")
        elif isinstance(expected,float):
            if not isinstance(actual,(int,float)) or isinstance(actual,bool) or not math.isfinite(actual):raise AssertionError(label+": finite numeric mismatch")
            error=abs(expected-actual);self.maximum_error=max(self.maximum_error,error)
            if error>5e-13:raise AssertionError(f"{label}: scalar difference {error}")
        elif isinstance(expected,list):
            if not isinstance(actual,list) or len(expected)!=len(actual):raise AssertionError(label+": list shape")
            for i,(a,b) in enumerate(zip(expected,actual)):self.equal(a,b,label+f"[{i}]")
        elif isinstance(expected,dict):
            if not isinstance(actual,dict) or set(expected)!=set(actual):raise AssertionError(label+": dictionary keys")
            for k,v in expected.items():self.equal(v,actual[k],label+"."+k)
        else:raise TypeError(type(expected))


def scalar_individual(growth,loss,weights,training_ids,*,L,floor,gamma,menu_index):
    """Separate Python sums/counts and complete set enumeration."""
    n=len(training_ids)
    result=dict(training_n=n,training_ids=training_ids,L=float(L),liquidity_floor=float(floor),
        gamma=None if gamma is None else float(gamma),epsilon_annual_log_growth=.001,
        status="no_selected_gamma" if gamma is None else "pending",event_count_cap=None,
        feasible_n=None,feasible_indices=None,near_optimal_n=None,near_optimal_indices=None,
        best_mean_log_growth=None,weight_min=None,weight_max=None,total_variation_diameter=None,
        raw_menu_index=menu_index,raw_menu_feasible=None,raw_menu_mean_log_growth=None,
        raw_menu_regret_if_feasible=None,raw_menu_in_near_optimal_set=None)
    if gamma is None:return result,[]
    cap=math.floor(gamma*n+TOL)
    counts=[sum(float(loss[sid-1,k])-L>TOL for sid in training_ids) for k in range(len(weights))]
    means=[math.fsum(float(growth[sid-1,k]) for sid in training_ids)/n for k in range(len(weights))]
    feasible=[k for k,w in enumerate(weights) if counts[k]<=cap and float(w[2])>=floor-TOL]
    result.update(event_count_cap=cap,feasible_n=len(feasible),feasible_indices=feasible,
        status="available" if feasible else "empty_feasible_set")
    if menu_index is not None:
        result.update(raw_menu_feasible=menu_index in feasible,raw_menu_mean_log_growth=means[menu_index])
    if not feasible:
        result.update(near_optimal_n=0,near_optimal_indices=[])
        return result,[]
    best=max(means[k] for k in feasible)
    near=[k for k in feasible if means[k]>=best-.001-TOL]
    diameter=max(math.fsum(abs(float(weights[a][c])-float(weights[b][c])) for c in range(3))/2 for a in near for b in near)
    result.update(best_mean_log_growth=best,near_optimal_n=len(near),near_optimal_indices=near,
        weight_min=[min(float(weights[k][c]) for k in near) for c in range(3)],
        weight_max=[max(float(weights[k][c]) for k in near) for c in range(3)],total_variation_diameter=diameter)
    if menu_index is not None:
        result["raw_menu_in_near_optimal_set"]=menu_index in near
        if menu_index in feasible:result["raw_menu_regret_if_feasible"]=best-means[menu_index]
    members=[dict(candidate_local_index=k,weights=[float(v) for v in weights[k]],training_event_count=counts[k],
        mean_training_log_growth=means[k],gap_to_individual_best=best-means[k]) for k in near]
    return result,members


def scalar_summary(rows):
    funded=[r for r in rows if r["assigned"]];n=len(funded)
    result=dict(original_n=len(rows),financial_n=n,review_n=len(rows)-n,coverage=n/len(rows) if rows else None)
    result.update(dict.fromkeys(("breach_count","breach_fraction","mean_log_growth","median_cagr","p10_cagr",
        "median_mdd","p90_mdd","cvar90_mdd","mean_volatility","mean_nominal_fees","mean_trade_fraction")))
    if not n:return result
    def avg(key):return math.fsum(r[key] for r in funded)/n
    def order(key):return sorted(r[key] for r in funded)
    def median(a):return (a[(n-1)//2]+a[n//2])/2
    dd=order("mdd");growth=order("cagr");remaining=n/10;total=0.
    for value in reversed(dd):
        take=min(remaining,1.);total+=take*value;remaining-=take
        if remaining<=0:break
    events=sum(bool(r["breach"]) for r in funded)
    result.update(breach_count=events,breach_fraction=events/n,mean_log_growth=avg("log_growth"),
        median_cagr=median(growth),p10_cagr=growth[math.ceil(n/10)-1],median_mdd=median(dd),
        p90_mdd=dd[math.ceil(9*n/10)-1],cvar90_mdd=total/(n/10),
        mean_volatility=avg("volatility"),mean_nominal_fees=avg("nominal_fees"),mean_trade_fraction=avg("trade_fraction"))
    return result


def compare_csv(path,expected,checks):
    with gzip.open(path,"rt",newline="") as handle:actual=list(csv.DictReader(handle))
    checks.equal(len(expected),len(actual),path.name+".rows")
    for i,(e,a) in enumerate(zip(expected,actual)):
        if set(e)!=set(a):raise AssertionError(path.name+": CSV column schema")
        for k,v in e.items():
            cell=a[k]
            if v is None:parsed=None if cell=="" else cell
            elif isinstance(v,(list,dict)):parsed=json.loads(cell)
            elif isinstance(v,bool):parsed=True if cell=="True" else False if cell=="False" else cell
            elif isinstance(v,int):parsed=int(cell)
            elif isinstance(v,float):parsed=float(cell)
            else:parsed=cell
            checks.equal(v,parsed,f"{path.name}[{i}].{k}")


def validate_ids(values):
    if not values or any(type(v) is not int or not 1<=v<=120 for v in values) or len(set(values))!=len(values):
        raise ValueError("Unique original development IDs only")


def authorize():
    p,f=json.loads(PLAN.read_text()),json.loads(FREEZE.read_text())
    if not f["execution_ready"] or f["plan_sha256"]!=sha(PLAN):raise RuntimeError("Freeze independent source before cache numeric access")
    if p["allow_final"] is not False or p["development_ids"]!=list(range(1,121)):raise RuntimeError("Development-only audit")
    if "scripts/audit_q1_stability_diagnostic.py" not in f["source_sha256"]:raise RuntimeError("Bind audit executable")
    for name,digest in f["source_sha256"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError("Audit source changed")
    source=ROOT/p["producer_directory"];m=json.loads((source/"manifest.json").read_text())
    if (m["status"]!="complete" or sha(source/"manifest.json")!=p["producer_manifest_sha256"]
            or m["final_scenarios_loaded"] is not False or m["new_menu_fits"]!=0):raise RuntimeError("Producer identity/scope")
    for name,digest in m["outputs_sha256"].items():
        if sha(source/name)!=digest:raise RuntimeError("Producer output changed")
    for name,digest in m["sources_and_inputs_at_start"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError("Producer source/input changed")
    if m["sources_and_inputs_at_start"]!=m["sources_and_inputs_at_end"]:raise RuntimeError("Producer execution changed inputs")
    return p,f,source,m


def run():
    p,f,source,m=authorize();receipt=source/"independent_cache_audit.json"
    if receipt.exists():raise RuntimeError("Refuse overwriting any audit result")
    checks=Checks()
    snapshot=dict(plan_sha256=sha(PLAN),audit_source_freeze_sha256=sha(FREEZE),
        producer_manifest_sha256=sha(source/"manifest.json"),
        audit_source_at_start={name:sha(ROOT/name) for name in f["source_sha256"]},
        producer_outputs_at_start={name:sha(source/name) for name in m["outputs_sha256"]})
    try:
        counts=_run_numeric(p,source,checks)
        for name,digest in m["sources_and_inputs_at_start"].items():
            if sha(ROOT/name)!=digest:raise RuntimeError("Input/source changed during audit")
        if snapshot["producer_outputs_at_start"]!={name:sha(source/name) for name in m["outputs_sha256"]}:raise RuntimeError("Producer output changed during audit")
        if snapshot["audit_source_at_start"]!={name:sha(ROOT/name) for name in f["source_sha256"]}:raise RuntimeError("Audit source changed during run")
        if sha(PLAN)!=snapshot["plan_sha256"] or sha(FREEZE)!=snapshot["audit_source_freeze_sha256"]:raise RuntimeError("Audit plan/freeze changed")
        result=dict(status="passed",completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks.n,
            maximum_absolute_numeric_error=checks.maximum_error,counts=counts,**snapshot,
            independent_math="Python scalar counts/fsum/sets/ordinary median/explicit quantile ranks/fractional tail; no production diagnostic import.",
            scope="Saved-cache postprocessing correctness, not upstream market resimulation, joint-menu optimality, predictive calibration or future risk.")
    except BaseException as error:
        result=dict(status="failed",error_type=type(error).__name__,error=str(error),checks=checks.n,**snapshot)
        (source/"independent_cache_audit_failure.txt").write_text(traceback.format_exc())
        receipt.write_text(json.dumps(result,indent=2)+"\n");raise
    receipt.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],checks=checks.n,maximum_error=checks.maximum_error,receipt=str(receipt))))


def _run_numeric(p,source,checks):
    protocol=json.loads((ROOT/p["producer_protocol"]).read_text());inputs=protocol["inputs"]
    jobs=json.loads((ROOT/inputs["job_protocol"]).read_text())["jobs"]
    bundles=json.loads((ROOT/inputs["static_refits"]).read_text())
    for job in jobs:
        validate_ids(job["training_ids"]);validate_ids(job["evaluation_ids"])
        if job["stage"]=="inner" or job["parent_context_id"]!=0:
            if set(job["training_ids"])&set(job["evaluation_ids"]):raise ValueError("Heldout/training overlap")
    with np.load(ROOT/inputs["cache_npz"],allow_pickle=False) as cache:
        checks.equal(list(range(1,121)),[int(v) for v in cache["scenario_ids"]],"cache IDs")
        rational=sorted([(i/20,j/20,(20-i-j)/20) for i in range(21) for j in range(21-i)]+[(1/3,1/3,1/3)])
        original=cache["weights"]
        checks.equal([list(w) for w in rational],original.tolist(),"independent rational grid")
        mapping=[i for i,w in enumerate(rational) if min(w)>=.05-TOL]
        weights=[rational[k] for k in mapping];checks.equal(172,len(weights),"strict universe")
        third=next(k for k,w in enumerate(weights) if all(abs(v-1/3)<1e-15 for v in w))
        fields={h:{key:cache[f"H{h}__{key}"][:,mapping] for key in
            ("log_growth","cagr","mdd","volatility","terminal_real","fees","trade_fraction")} for _,h,_,_ in PROFILES}
    expected_sets,expected_members,expected_metadata,expected_bands,expected_oof,expected_refs=[],[],[],[],[],[]
    for job in jobs:
        bundle=bundles[job["job_id"]];refit=bundle["refit"];raw=refit["raw_fit"]
        checks.equal(job["training_ids"],refit["training_ids"],"own fit IDs")
        gamma=refit["selection"]["selected_gamma"]
        summaries=[r for r in bundle["inner_summaries"] if r["eligible"]]
        if summaries:
            coverage=max(r["raw_auto_path_profile_units"] for r in summaries)
            same=[r for r in summaries if r["raw_auto_path_profile_units"]==coverage]
            maximum=max(r["raw_auto_unit_mean_log_growth"] for r in same)
            band=sorted(float(r["training_screen_gamma"]) for r in same if r["raw_auto_unit_mean_log_growth"]>=maximum-TOL)
        else:band=[]
        expected_bands.append(dict(job_id=job["job_id"],training_ids=job["training_ids"],selected_gamma=gamma,
            objective_band_gammas=band,complete_saved_menu=None if raw is None else raw["menu"],
            interpretation="Saved metadata/objective band, not a unique preference estimate or behavior-equivalence proof."))
        for name,h,L,liquidity in PROFILES:
            assignment=refit["raw_assignments"][name];effective=refit["effective_assignments"][name]
            product=next((r for r in raw["menu"] if r["name"]==assignment),None) if raw else None
            index=None if product is None else product["candidate_index"]
            if product:
                checks.equal(mapping[index],product["candidate_original232_index"],"saved original target index")
                checks.equal(list(weights[index]),product["weights"],"saved target weights")
            admission=next(r for r in refit["admission_details"] if r["client"]==name)
            detail=refit.get("q1_review_details",{}).get(name,{})
            expected_metadata.append(dict(job_id=job["job_id"],context_id=job["parent_context_id"],stage=job["stage"],
                profile=name,horizon=h,L=L,liquidity=liquidity,training_n=len(job["training_ids"]),selected_gamma=gamma,
                raw_assignment=assignment,effective_assignment=effective,admission_reason=admission["reason"],
                review_reasons=detail.get("q1_review_reasons",[]),raw_menu_index=index,
                raw_original232_index=None if product is None else product["candidate_original232_index"],
                raw_weights=None if product is None else product["weights"],
                raw_identified_by_profiles=None if product is None else product["identified_by_profiles"],
                raw_reference15_CVaR=None if product is None else product["reference_cvar90_mdd"]))
            for screen,cap in (("selected_gamma",gamma),("service_alpha_reference",.1)):
                result,members=scalar_individual(fields[h]["log_growth"],fields[h]["mdd"],weights,job["training_ids"],
                    L=L,floor=FLOORS[liquidity],gamma=cap,menu_index=index)
                common=dict(job_id=job["job_id"],profile=name,horizon=h,screen=screen)
                expected_sets.append(dict(common,**result))
                for member in members:
                    expected_members.append(dict(common,**member,candidate_original232_index=mapping[member["candidate_local_index"]]))
            if job["stage"]=="refit" and job["parent_context_id"]!=0:
                selected=next((r["candidate_index"] for r in raw["menu"] if r["name"]==effective),None) if raw else None
                for sid in job["evaluation_ids"]:
                    common=dict(job_id=job["job_id"],scenario_id=sid,profile=name,horizon=h,L=L,
                        raw_assignment=assignment,effective_assignment=effective,admission_reason=admission["reason"])
                    def values(k):
                        if k is None:return dict.fromkeys(("log_growth","cagr","mdd","volatility","terminal_real","nominal_fees","trade_fraction","breach"))
                        result={key:float(fields[h][key][sid-1,k]) for key in ("log_growth","cagr","mdd","volatility","terminal_real","trade_fraction")}
                        result.update(nominal_fees=float(fields[h]["fees"][sid-1,k]),breach=result["mdd"]-L>TOL)
                        return result
                    expected_oof.append(dict(common,assigned=selected is not None,candidate_local_index=selected,**values(selected)))
                    expected_refs.append(dict(common,assigned=True,candidate_local_index=third,**values(third)))
    checks.equal(expected_sets,json.loads((source/"individual_feasible_sets.json").read_text()),"all300 scalar sets")
    checks.equal(expected_bands,json.loads((source/"fit_gamma_bands.json").read_text()),"gamma metadata bands")
    compare_csv(source/"near_optimal_members.csv.gz",expected_members,checks)
    compare_csv(source/"fit_profile_metadata.csv.gz",expected_metadata,checks)
    compare_csv(source/"own_horizon_oof_paths.csv.gz",expected_oof,checks)
    compare_csv(source/"exact_thirds_references.csv.gz",expected_refs,checks)
    expected_summary=[]
    for name,h,L,_ in PROFILES:
        rows=[r for r in expected_oof if r["profile"]==name];refs=[r for r in expected_refs if r["profile"]==name]
        checks.equal(list(range(1,121)),sorted(r["scenario_id"] for r in rows),name+" original120")
        by_ref={r["scenario_id"]:r for r in refs};funded={r["scenario_id"]:r for r in rows if r["assigned"]}
        ids=sorted(funded);parts=dict(neither=[],model_only=[],reference_only=[],both=[])
        for sid in ids:
            a,b=funded[sid]["breach"],by_ref[sid]["breach"]
            parts["both" if a and b else "model_only" if a else "reference_only" if b else "neither"].append(sid)
        expected_summary.append(dict(profile=name,horizon=h,L=L,model=scalar_summary(rows),reference_all_original=scalar_summary(refs),
            same_assigned_reference=scalar_summary([by_ref[s] for s in ids]),assigned_ids=ids,event_id_partition=parts,
            paired_mean_log_growth=math.fsum(funded[s]["log_growth"]-by_ref[s]["log_growth"] for s in ids)/len(ids) if ids else None,
            caution="Same-own-H static product references, not Q2execution approval; profiles share market scenarios and cannot be pooled as IID."))
    checks.equal(expected_summary,json.loads((source/"own_horizon_summary.json").read_text()),"independent ownH summary")
    return dict(individual_sets=len(expected_sets),candidate_statistics=sum(x["gamma"] is not None for x in expected_sets)*172,
        near_members=len(expected_members),metadata_rows=len(expected_metadata),model_OOF_rows=len(expected_oof),
        financial_model_rows=sum(r["assigned"] for r in expected_oof),null_model_rows=sum(not r["assigned"] for r in expected_oof),
        exact_thirds_reference_rows=len(expected_refs),summary_groups=5,raw_market_reads=0,new_policy_runs=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(development_only=True,cache_numeric_loaded=False,production_diagnostic_imported=False),indent=2))
