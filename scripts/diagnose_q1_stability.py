"""Frozen-cache Q1 diagnostics; no raw market parser, new menu or policy fit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"research"),str(ROOT/"src")]
from diagnostics.q1_stability import (
    EPSILON_ANNUAL_LOG_GROWTH, FLOORS, PROFILES, TOL, development_ids,
    individual_set, gamma_objective_band, profile_metadata, own_horizon_rows,
)

PROTOCOL = ROOT/"docs/q1_stability_diagnostic_protocol.json"
FREEZE = ROOT/"docs/q1_stability_diagnostic_source_freeze.json"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):raise ValueError("Nonfinite output is not an implicitNULL")
    return x


def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def write_csv(path,rows,fields=None):
    fields = list(fields) if fields else sorted({key for row in rows for key in row})
    with gzip.GzipFile(filename=str(path),mode="wb",mtime=0) as zipped:
        with io.TextIOWrapper(zipped,newline="") as text:
            writer=csv.DictWriter(text,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def validate_jobs(jobs,bundles):
    if len(jobs)!=30 or len({j["job_id"] for j in jobs})!=30 or set(bundles)!={j["job_id"] for j in jobs}:
        raise ValueError("Require exact30 saved static fit jobs")
    if sorted({j["parent_context_id"] for j in jobs})!=list(range(6)):
        raise ValueError("Require full120 and five outer contexts")
    outer=[]
    for c in range(6):
        group=[j for j in jobs if j["parent_context_id"]==c]
        refits=[j for j in group if j["stage"]=="refit"]
        inner=[j for j in group if j["stage"]=="inner"]
        if len(refits)!=1 or len(inner)!=4:raise ValueError("Each context has4 inner and1 refit")
        refit=refits[0];train=development_ids(refit["training_ids"]);test=development_ids(refit["evaluation_ids"])
        if c==0:
            if sorted(train)!=list(range(1,121)) or sorted(test)!=list(range(1,121)):raise ValueError("Full context identity")
        elif len(train)!=96 or len(test)!=24 or set(train)&set(test) or set(train)|set(test)!=set(range(1,121)):
            raise ValueError("Outer96/24 partition required")
        else:outer+=test
        validation=[]
        for job in inner:
            a=development_ids(job["training_ids"]);b=development_ids(job["evaluation_ids"])
            if set(a)&set(b) or set(a)|set(b)!=set(train):raise ValueError("Inner training/validation separation")
            validation+=b
        if sorted(validation)!=sorted(train):raise ValueError("Inner validation covers training IDs once")
    if sorted(outer)!=list(range(1,121)):raise ValueError("Primary outer IDs must occur once")
    for job in jobs:
        if bundles[job["job_id"]]["refit"]["training_ids"]!=job["training_ids"]:
            raise ValueError("Saved static refit trained on wrong IDs")


def summary(rows):
    rows=list(rows);funded=[r for r in rows if r["assigned"]];n=len(funded)
    result=dict(original_n=len(rows),financial_n=n,review_n=len(rows)-n,
        coverage=n/len(rows) if rows else None)
    keys=("breach_count","breach_fraction","mean_log_growth","median_cagr","p10_cagr",
          "median_mdd","p90_mdd","cvar90_mdd","mean_volatility","mean_nominal_fees","mean_trade_fraction")
    result.update(dict.fromkeys(keys))
    if not n:return result
    def values(key):
        x=[r[key] for r in funded]
        if not all(math.isfinite(v) for v in x):raise ValueError("Funded summaries require finite metrics")
        return x
    def avg(x):return math.fsum(x)/len(x)
    def q(x,p):return sorted(x)[max(0,math.ceil(p*len(x))-1)]
    def med(x):
        x=sorted(x);return (x[(len(x)-1)//2]+x[len(x)//2])/2
    d=values("mdd");g=values("cagr");tail=.1*n
    result.update(breach_count=sum(r["breach"] for r in funded),breach_fraction=sum(r["breach"] for r in funded)/n,
        mean_log_growth=avg(values("log_growth")),median_cagr=med(g),p10_cagr=q(g,.1),
        median_mdd=med(d),p90_mdd=q(d,.9),
        cvar90_mdd=math.fsum(min(max(tail-i,0.),1.)*v for i,v in enumerate(sorted(d,reverse=True)))/tail,
        mean_volatility=avg(values("volatility")),mean_nominal_fees=avg(values("nominal_fees")),
        mean_trade_fraction=avg(values("trade_fraction")))
    return result


def matched_profile_summary(rows,references):
    left={r["scenario_id"]:r for r in rows};right={r["scenario_id"]:r for r in references}
    if len(left)!=len(rows) or len(right)!=len(references) or set(left)!=set(right):raise ValueError("Comparison identity")
    assigned=sorted(s for s,r in left.items() if r["assigned"])
    partition=dict(neither=[],model_only=[],reference_only=[],both=[])
    for sid in assigned:
        a,b=left[sid]["breach"],right[sid]["breach"]
        partition["both" if a and b else "model_only" if a else "reference_only" if b else "neither"].append(sid)
    return dict(model=summary(rows),reference_all_original=summary(references),
        same_assigned_reference=summary([right[s] for s in assigned]),assigned_ids=assigned,
        event_id_partition=partition,
        paired_mean_log_growth=None if not assigned else math.fsum(left[s]["log_growth"]-right[s]["log_growth"] for s in assigned)/len(assigned),
        caution="Same-own-H static product references, not Q2execution approval; profiles share market scenarios and cannot be pooled as IID.")


def authorize():
    p,f=json.loads(PROTOCOL.read_text()),json.loads(FREEZE.read_text())
    if not f["execution_ready"] or f["protocol_sha256"]!=sha(PROTOCOL):raise RuntimeError("Source/protocol freeze required before cache numeric access")
    if (p["development_ids"]!=list(range(1,121)) or p["allow_final"] is not False
            or p["epsilon_annual_log_growth"]!=EPSILON_ANNUAL_LOG_GROWTH
            or p["service_alpha"]!=.1 or p["profiles"]!=list(PROFILES) or p["liquidity_floors"]!=FLOORS):
        raise RuntimeError("Frozen scientific scope changed")
    if "scripts/diagnose_q1_stability.py" not in f["source_sha256"]:raise RuntimeError("Bind executable source")
    for group in ("source_sha256","input_sha256"):
        for name,digest in f[group].items():
            if sha(ROOT/name)!=digest:raise RuntimeError("Frozen file changed: "+name)
    for name in p["inputs"].values():
        if name not in f["input_sha256"]:raise RuntimeError("Every input must be frozen")
    return p,f


def run(output):
    p,f=authorize();output=Path(output).resolve()
    if output.exists():raise RuntimeError("Refuse overwrite")
    output.mkdir(parents=True)
    snapshot=dict(protocol_sha256=sha(PROTOCOL),source_freeze_sha256=sha(FREEZE),
        sources_and_inputs_at_start={**f["source_sha256"],**f["input_sha256"]})
    writej(output/"manifest.json",dict(status="running",**snapshot))
    try:_run_numeric(p,output,snapshot)
    except BaseException as error:
        (output/"failure.txt").write_text(traceback.format_exc())
        writej(output/"manifest.json",dict(status="failed",error_type=type(error).__name__,error=str(error),**snapshot))
        raise


def _run_numeric(p,output,snapshot):
    inputs=p["inputs"]
    metadata=json.loads((ROOT/inputs["cache_metadata"]).read_text())
    if (metadata["scenario_ids"]!=list(range(1,121)) or metadata["candidate_count"]!=232
            or metadata["step"]!=.05 or metadata["cost_rate"]!=.0025
            or metadata["npz_sha256"]!=sha(ROOT/inputs["cache_npz"])):
        raise ValueError("Require the exact development-only static annual cache")
    bundles=json.loads((ROOT/inputs["static_refits"]).read_text())
    jobs=json.loads((ROOT/inputs["job_protocol"]).read_text())["jobs"]
    validate_jobs(jobs,bundles)
    # This import is pure candidate-grid construction; no cache generator or
    # raw-data loader is imported or called.
    from safeinvest_research.static import simplex_grid
    fields={}
    with np.load(ROOT/inputs["cache_npz"],allow_pickle=False) as cache:
        if not np.array_equal(cache["scenario_ids"],np.arange(1,121)):raise ValueError("Final IDs forbidden")
        original=cache["weights"]
        if not np.array_equal(original,simplex_grid(.05)) or original.shape!=(232,3):raise ValueError("Original232 mapping changed")
        mapping=np.flatnonzero(np.all(original>=.05-TOL,axis=1));weights=original[mapping]
        if len(mapping)!=172:raise ValueError("Strict172 mapping required")
        thirds=np.flatnonzero(np.all(np.isclose(weights,1/3,atol=1e-15,rtol=0),axis=1))
        if len(thirds)!=1:raise ValueError("Exact-thirds target must remain unique")
        thirds_index=int(thirds[0])
        for horizon in sorted({q["horizon"] for q in PROFILES}):
            fields[horizon]={}
            for key in ("log_growth","cagr","mdd","volatility","terminal_real","fees","trade_fraction"):
                values=cache[f"H{horizon}__{key}"]
                if values.shape!=(120,232) or not np.isfinite(values).all():raise ValueError("Cache field identity/finite failure")
                fields[horizon][key]=values[:,mapping]
    meta_rows,sets,members,bands,outer_rows,references=[],[],[],[],[],[]
    for job in jobs:
        bundle=bundles[job["job_id"]];raw=bundle["refit"]["raw_fit"]
        if raw:
            for item in raw["menu"]:
                k=item["candidate_index"]
                if (item["candidate_original232_index"]!=int(mapping[k])
                        or not np.array_equal(item["weights"],weights[k])):
                    raise ValueError("Saved fitted menu local/original index mismatch")
        bands.append(dict(job_id=job["job_id"],training_ids=job["training_ids"],
            selected_gamma=bundle["refit"]["selection"]["selected_gamma"],
            objective_band_gammas=gamma_objective_band(bundle),
            complete_saved_menu=None if raw is None else raw["menu"],
            interpretation="Saved metadata/objective band, not a unique preference estimate or behavior-equivalence proof."))
        for profile in PROFILES:
            meta=profile_metadata(job,bundle,profile);meta_rows.append(meta)
            own=fields[profile["horizon"]]
            for screen,gamma in (("selected_gamma",meta["selected_gamma"]),("service_alpha_reference",.1)):
                value,near=individual_set(own["log_growth"],own["mdd"],weights,job["training_ids"],
                    L=profile["loss_tolerance"],liquidity_floor=FLOORS[profile["liquidity"]],gamma=gamma,
                    raw_menu_index=meta["raw_menu_index"])
                common=dict(job_id=job["job_id"],profile=profile["name"],horizon=profile["horizon"],screen=screen)
                sets.append(dict(common,**value))
                for row in near:
                    row=dict(row);row["candidate_original232_index"]=int(mapping[row["candidate_local_index"]])
                    members.append(dict(common,**row))
            if job["stage"]=="refit" and job["parent_context_id"]!=0:
                a,b=own_horizon_rows(job,bundle,profile,own,thirds_index=thirds_index)
                outer_rows+=a;references+=b
    expected={(sid,q["name"]) for sid in range(1,121) for q in PROFILES}
    if len(outer_rows)!=600 or {(r["scenario_id"],r["profile"]) for r in outer_rows}!=expected:
        raise ValueError("Require exactly600 own-horizon OOF profile-paths")
    summaries=[dict(profile=q["name"],horizon=q["horizon"],L=q["loss_tolerance"],
        **matched_profile_summary([r for r in outer_rows if r["profile"]==q["name"]],
                                  [r for r in references if r["profile"]==q["name"]])) for q in PROFILES]
    write_csv(output/"fit_profile_metadata.csv.gz",meta_rows)
    writej(output/"fit_gamma_bands.json",bands)
    writej(output/"individual_feasible_sets.json",sets)
    write_csv(output/"near_optimal_members.csv.gz",members,
        ("job_id","profile","horizon","screen","candidate_local_index","candidate_original232_index",
         "weights","training_event_count","mean_training_log_growth","gap_to_individual_best"))
    write_csv(output/"own_horizon_oof_paths.csv.gz",outer_rows)
    write_csv(output/"exact_thirds_references.csv.gz",references)
    writej(output/"own_horizon_summary.json",summaries)
    end={name:sha(ROOT/name) for name in snapshot["sources_and_inputs_at_start"]}
    if end!=snapshot["sources_and_inputs_at_start"] or sha(PROTOCOL)!=snapshot["protocol_sha256"] or sha(FREEZE)!=snapshot["source_freeze_sha256"]:
        raise RuntimeError("Frozen source/input/protocol changed during diagnostic")
    writej(output/"manifest.json",dict(status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),**snapshot,
        sources_and_inputs_at_end=end,epsilon_annual_log_growth=EPSILON_ANNUAL_LOG_GROWTH,
        metadata_rows=len(meta_rows),individual_sets=len(sets),near_member_rows=len(members),
        own_horizon_oof_rows=len(outer_rows),reference_rows=len(references),new_menu_fits=0,new_policy_runs=0,
        raw_market_reads=0,final_scenarios_loaded=False,old_model_changed=False,
        outputs_sha256={v.name:sha(v) for v in output.iterdir() if v.name!="manifest.json"}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"results/q1_stability_diagnostic_v1")
    args=parser.parse_args()
    if args.execute:run(args.output)
    else:print(json.dumps(dict(development_only=True,cache_values_loaded=False,new_menu_fits=0,
        epsilon_annual_log_growth=EPSILON_ANNUAL_LOG_GROWTH,profiles=PROFILES),indent=2))
