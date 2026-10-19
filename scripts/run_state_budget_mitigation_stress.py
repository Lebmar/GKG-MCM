"""Fixed-selected V2b development stresses, with a complete exact base gate.

No parameter/target is refitted or reselected.  Baseline reproduction for all
six contexts and three references completes before any nonbase setting runs.
Frozen V2b helpers define rows/NULLs/summaries; their sources stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src"),str(ROOT/"research")]
import run_state_budget_mitigation as canonical

PROTOCOL=ROOT/"docs/state_budget_mitigation_stress_protocol.json"
FREEZE=ROOT/"docs/state_budget_mitigation_stress_source_freeze.json"
DEFAULT_OUT=ROOT/"results/state_budget_mitigation_stress_v1"
SETTINGS=(
    dict(setting_id="base",cost_rate=.0025,support_mode="calendar",actual_inflation_shift=0.),
    dict(setting_id="fee_0000",cost_rate=0.,support_mode="calendar",actual_inflation_shift=0.),
    dict(setting_id="fee_00125",cost_rate=.00125,support_mode="calendar",actual_inflation_shift=0.),
    dict(setting_id="fee_005",cost_rate=.005,support_mode="calendar",actual_inflation_shift=0.),
    dict(setting_id="fee_01",cost_rate=.01,support_mode="calendar",actual_inflation_shift=0.),
    dict(setting_id="pooled_support",cost_rate=.0025,support_mode="pooled",actual_inflation_shift=0.),
    dict(setting_id="actual_inflation_plus_02",cost_rate=.0025,support_mode="calendar",actual_inflation_shift=.02),
)
OBJECTS=("state_budget_learner",*canonical.REFS)
TOL=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_array_identity(actual,expected):
    """Exact field/dtype/shape/value comparison, including pairedNaNs."""
    if set(actual)!=set(expected):raise AssertionError("Baseline array field names differ")
    values=0
    for name in sorted(expected):
        a,b=np.asarray(actual[name]),np.asarray(expected[name])
        if a.dtype!=b.dtype or a.shape!=b.shape:raise AssertionError("Baseline dtype/shape differs: "+name)
        equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a,b)
        if not equal:raise AssertionError("Baseline values differ: "+name)
        values+=a.size
    return dict(fields=len(expected),scalar_values=values,all_exact=True)


def thin_arm(arm):
    if arm is None:return {},[]
    retained,omitted={},[]
    for key,value in arm.items():
        if not isinstance(value,np.ndarray):continue
        if key.startswith(("candidate_","mitigation_candidate_")) or key in ("training_previous_state","training_real_gross"):
            omitted.append(key)
        else:retained[key]=value
    return retained,omitted


def actual_inflation_for_setting(inflation,setting):
    pi=np.asarray(inflation,float)
    changed=pi+setting["actual_inflation_shift"]
    if not np.isfinite(changed).all() or np.any(changed<=-1):raise ValueError("Actual stressed inflation must have positive gross")
    return changed


def evaluate_two_phase(contexts,evaluate,check_base,consume,base_passed):
    """The control flow is testable without loading data or running a model."""
    for context in contexts:
        result=evaluate(context,SETTINGS[0])
        check_base(context,result)
        consume(context,SETTINGS[0],result)
    base_passed()
    for setting in SETTINGS[1:]:
        for context in contexts:
            consume(context,setting,evaluate(context,setting))


def context_definitions(jobs,fits,selections,groups):
    canonical.validate_jobs(jobs)
    if set(selections)!={str(i) for i in range(6)}:raise ValueError("Require six canonical context selections")
    result=[]
    for c in range(6):
        job=next(j for j in jobs if j["parent_context_id"]==c and j["stage"]=="refit")
        job={**job,"q1_case_status":fits[job["job_id"]]["q2_case"]["status"]}
        target=canonical.validate_job(job,fits[job["job_id"]])
        chosen=selections[str(c)]
        if chosen["training_ids"]!=job["training_ids"] or chosen["evaluation_ids"]!=job["evaluation_ids"]:
            raise ValueError("Canonical selected context identities differ")
        candidate_id=chosen["selected_candidate_id"]
        spec=next((s for s in canonical.CANDIDATES if s["candidate_id"]==candidate_id),None)
        if candidate_id is not None and spec is None:raise ValueError("Canonical tuple outside frozen six families")
        scope="full_selected" if c==0 else "outer_selected"
        selected=[g for g in groups if g["job_id"]==job["job_id"] and g["kind"]==scope]
        refs={name:[g for g in groups if g["job_id"]==job["job_id"] and g["kind"]=="reference" and g["policy"]==name]
              for name in canonical.REFS}
        if len(selected)!=1 or any(len(x)!=1 for x in refs.values()):raise ValueError("Canonical selected/reference group identity")
        selected=selected[0];reference_groups={name:items[0] for name,items in refs.items()}
        for group in [selected,*reference_groups.values()]:
            if group["scenario_ids"]!=job["evaluation_ids"] or group["training_ids"]!=job["training_ids"]:
                raise ValueError("Canonical arm training/evaluation identities differ")
        if selected.get("target")!=(None if target is None else target.tolist()):raise ValueError("Canonical own-fit target differs")
        if selected.get("candidate")!=spec:raise ValueError("Canonical selected tuple metadata differs")
        for name,group in reference_groups.items():
            expected_target=(np.ones(3)/3).tolist() if name=="exact_thirds_annual" else None if target is None else target.tolist()
            if group.get("target")!=expected_target:raise ValueError("Canonical reference target differs")
        null_reason=("q1_review" if job["q1_case_status"]=="risk_review" else "not_balanced_case") if target is None else "no_inner_feasible_candidate"
        result.append(dict(context_id=c,job=job,target=target,candidate=spec,
            selected_group=selected,reference_groups=reference_groups,null_reason=null_reason,
            cohort="full_resubstitution" if c==0 else "outer_OOF"))
    return result


def evaluate_context(context,setting,returns,inflation,*,simulate_model=None,paired_simulator=None):
    """Fixed tuple/target, changed known fee/support or actual inflation only."""
    if simulate_model is None:
        from experimental_execution.state_budget_mitigation import simulate_state_budget
        simulate_model=simulate_state_budget
    paired_simulator=canonical.simulate_paired if paired_simulator is None else paired_simulator
    job=context["job"];train=np.array(job["training_ids"])-1;ev=np.array(job["evaluation_ids"])-1
    r=np.asarray(returns)[ev];actual_pi=actual_inflation_for_setting(np.asarray(inflation)[ev],setting)
    w=context["target"];spec=context["candidate"]
    if w is None or spec is None:
        model=dict(arm=None,status=context["null_reason"],financial_available=False,initial_witness=None)
    else:
        model=simulate_model(r,actual_pi,np.asarray(returns)[train],np.asarray(inflation)[train],w,
            L=canonical.L,horizon=canonical.H,cost_rate=setting["cost_rate"],release_fraction=spec["release_fraction"],
            forecast_rule=spec["forecast_rule"],training_ids=job["training_ids"],actual_ids=job["evaluation_ids"],support_mode=setting["support_mode"])
    objects={"state_budget_learner":model}
    for name,target,armname in (("same_target_B",w,"pending_hold"),("same_target_annual",w,"annual"),
                                 ("exact_thirds_annual",np.ones(3)/3,"annual")):
        if target is None:
            objects[name]=dict(arm=None,status=context["null_reason"],financial_available=False)
        else:
            arm=canonical.squeeze(paired_simulator(r,actual_pi,np.asarray(target),canonical.L,
                horizon=canonical.H,cost_rate=setting["cost_rate"])[armname])
            objects[name]=dict(arm=arm,status="hypothetical_reference",financial_available=True)
    return objects


def same_cohort_comparisons(rows):
    comparisons=[]
    for setting in SETTINGS:
        for cohort in ("outer_OOF","full_resubstitution"):
            group=[r for r in rows if r["setting_id"]==setting["setting_id"] and r["cohort"]==cohort]
            model=[r for r in group if r["policy"]=="state_budget_learner"]
            if len({r["scenario_id"] for r in model})!=len(model):raise ValueError("Repeated original model ID")
            funded={r["scenario_id"]:r for r in model if r["assigned"]}
            for reference in canonical.REFS:
                all_ref=[r for r in group if r["policy"]==reference];by_ref={r["scenario_id"]:r for r in all_ref}
                if set(by_ref)!={r["scenario_id"] for r in model} or len(by_ref)!=len(all_ref):raise ValueError("Stress comparison identity")
                if any(not by_ref[s]["assigned"] for s in funded):raise ValueError("Same-ID reference unavailable")
                ids=sorted(funded);parts=dict(neither=[],model_only=[],reference_only=[],both=[])
                for sid in ids:
                    a,b=funded[sid]["breach"],by_ref[sid]["breach"]
                    parts["both" if a and b else "model_only" if a else "reference_only" if b else "neither"].append(sid)
                differences={}
                for key in ("log_growth","cagr","mdd","nominal_fees","trade_fraction","automatic_fraction"):
                    values=[funded[s][key]-by_ref[s][key] for s in ids]
                    differences[key]=dict(mean=math.fsum(values)/len(values) if values else None,
                        median=float(np.median(values)) if values else None,
                        positive_n=sum(v>TOL for v in values),negative_n=sum(v < -TOL for v in values),numerical_tie_n=sum(abs(v)<=TOL for v in values))
                comparisons.append(dict(setting_id=setting["setting_id"],cohort=cohort,reference=reference,
                    model_all_original=canonical.summarize_records(model),reference_all_original=canonical.summarize_records(all_ref),
                    same_funded_model=canonical.summarize_records([funded[s] for s in ids]),
                    same_funded_reference=canonical.summarize_records([by_ref[s] for s in ids]),same_funded_ids=ids,
                    initial_noninvested_ids=sorted(set(by_ref)-set(funded)),event_id_partition=parts,paired_differences=differences))
    return comparisons


def freeze_gate():
    p,f=json.loads(PROTOCOL.read_text()),json.loads(FREEZE.read_text())
    if not f["execution_ready"] or f["protocol_sha256"]!=sha(PROTOCOL):raise RuntimeError("Freeze source/protocol before data access")
    if p["development_ids"]!=list(range(1,121)) or p["allow_final"] is not False or p["settings"]!=list(SETTINGS):raise RuntimeError("Fixed development stresses changed")
    if "scripts/run_state_budget_mitigation_stress.py" not in f["source_sha256"]:raise RuntimeError("Bind executable")
    for group in ("source_sha256","input_sha256"):
        for name,digest in f[group].items():
            if sha(ROOT/name)!=digest:raise RuntimeError("Frozen stress dependency changed: "+name)
    for path in p["inputs"].values():
        if path not in f["input_sha256"]:raise RuntimeError("Bind every canonical stress input")
    source=ROOT/p["canonical_directory"];manifest=json.loads((source/"manifest.json").read_text())
    if (manifest["status"]!="complete" or manifest["final_scenarios_loaded"] is not False or manifest["new_final_reads"]!=0
            or sha(source/"manifest.json")!=p["canonical_manifest_sha256"]
            or manifest["sources_and_inputs_at_start"]!=manifest["sources_and_inputs_at_end"]):raise RuntimeError("Canonical producer identity/scope")
    for name,digest in manifest["outputs_sha256"].items():
        if sha(source/name)!=digest:raise RuntimeError("Canonical baseline output changed")
    for name,digest in manifest["sources_and_inputs_at_start"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError("Canonical source/input changed")
    jobs=json.loads((ROOT/p["inputs"]["job_protocol"]).read_text())["jobs"]
    fits=json.loads((ROOT/p["inputs"]["static_refits"]).read_text())
    selections=json.loads((source/"selections.json").read_text());groups=json.loads((source/"trace_groups.json").read_text())
    return p,f,source,manifest,context_definitions(jobs,fits,selections,groups)


def run(output=DEFAULT_OUT):
    p,f,source,parent,contexts=freeze_gate();output=Path(output).resolve()
    if output.exists():raise RuntimeError("Refuse overwriting any stress attempt")
    output.mkdir(parents=True)
    m=dict(status="running",phase="before_development_parser",started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(PROTOCOL),source_freeze_sha256=sha(FREEZE),
        canonical_manifest_sha256=sha(source/"manifest.json"),
        sources_and_inputs_at_start={**f["source_sha256"],**f["input_sha256"]},
        final_scenarios_loaded=False,new_final_reads=0,new_fits=0,new_inner_selections=0,base_reproduction_passed=False)
    canonical.writej(output/"manifest.json",m)
    arrays,groups,rows,witnesses,basechecks={},[],[],[],[]
    try:
        r,pi,ids=canonical.load_paths(ROOT/"data/SafeInvest_Synthetic_Data.csv",range(1,121),allow_final=False)
        if not np.array_equal(ids,np.arange(1,121)):raise RuntimeError("Only development120 may enter")
        m.update(phase="exact_base_reproduction",development_ids_loaded=ids.tolist());canonical.writej(output/"manifest.json",m)
        with np.load(source/"traces.npz",allow_pickle=False) as baseline:
            def evaluate(context,setting):return evaluate_context(context,setting,r,pi)
            def check_base(context,result):
                for name in OBJECTS:
                    expected_group=context["selected_group"] if name=="state_budget_learner" else context["reference_groups"][name]
                    expected={key:baseline[expected_group["group_id"]+"__"+key] for key in expected_group["array_names"]}
                    actual,_=canonical.archive_arrays(result[name]["arm"],expected_group["kind"])
                    check=exact_array_identity(actual,expected)
                    if bool(expected)!=bool(result[name]["financial_available"]):raise AssertionError("Baseline NULL/funded identity")
                    basechecks.append(dict(context_id=context["context_id"],policy=name,canonical_group_id=expected_group["group_id"],**check))
            def consume(context,setting,result):
                job=context["job"]
                for name,item in result.items():
                    common=dict(setting_id=setting["setting_id"],cohort=context["cohort"],context_id=context["context_id"],job_id=job["job_id"],policy=name)
                    current=(canonical.rows_from_arm(item["arm"],job["evaluation_ids"],common) if item["financial_available"]
                             else canonical.null_rows(job["evaluation_ids"],common,item["status"]))
                    rows.extend(current)
                    gid=setting["setting_id"]+"__"+job["job_id"]+"__"+name
                    retained,omitted=thin_arm(item["arm"])
                    for key,value in retained.items():arrays[gid+"__"+key]=value
                    groups.append(dict(group_id=gid,**common,training_ids=job["training_ids"],scenario_ids=job["evaluation_ids"],
                        selected_candidate=context["candidate"] if name=="state_budget_learner" else None,
                        own_fit_target=None if context["target"] is None else context["target"].tolist(),
                        status=item["status"],financial_available=item["financial_available"],setting=setting,
                        array_names=list(retained),omitted_diagnostic_arrays=omitted))
                witnesses.append(dict(context_id=context["context_id"],setting_id=setting["setting_id"],
                    selected_candidate=context["candidate"],status=result["state_budget_learner"]["status"],
                    witness=result["state_budget_learner"].get("initial_witness")))
                print(json.dumps(dict(context=context["context_id"],setting=setting["setting_id"],status=result["state_budget_learner"]["status"])),flush=True)
            def base_passed():
                receipt=dict(status="passed",canonical_manifest_sha256=m["canonical_manifest_sha256"],
                    canonical_traces_sha256=parent["outputs_sha256"]["traces.npz"],contexts=6,objects=24,
                    fields=sum(x["fields"] for x in basechecks),scalar_values=sum(x["scalar_values"] for x in basechecks),checks=basechecks)
                canonical.writej(output/"base_reproduction_receipt.json",receipt)
                m.update(base_reproduction_passed=True,base_reproduction_receipt_sha256=sha(output/"base_reproduction_receipt.json"),phase="fixed_selected_stresses")
                canonical.writej(output/"manifest.json",m)
            evaluate_two_phase(contexts,evaluate,check_base,consume,base_passed)
        summaries=[]
        for setting in SETTINGS:
            for cohort in ("outer_OOF","full_resubstitution"):
                for name in OBJECTS:
                    current=[row for row in rows if row["setting_id"]==setting["setting_id"] and row["cohort"]==cohort and row["policy"]==name]
                    if sorted(row["scenario_id"] for row in current)!=list(range(1,121)):raise RuntimeError("Every originalID must remain once per setting/cohort/object")
                    summaries.append(dict(setting_id=setting["setting_id"],cohort=cohort,policy=name,**canonical.summarize_records(current)))
        comparisons=same_cohort_comparisons(rows)
        if len(summaries)!=56 or len(comparisons)!=42:raise RuntimeError("Declared stress summary schema")
        pd.DataFrame(rows).to_csv(output/"path_metrics.csv.gz",index=False,compression=dict(method="gzip",mtime=0))
        pd.DataFrame(summaries).to_csv(output/"summary.csv.gz",index=False,compression=dict(method="gzip",mtime=0))
        canonical.writej(output/"cohort_comparisons.json",comparisons);canonical.writej(output/"trace_groups.json",groups)
        canonical.writej(output/"initial_witnesses.json",witnesses);np.savez_compressed(output/"traces.npz",**arrays)
        end={name:sha(ROOT/name) for name in m["sources_and_inputs_at_start"]}
        if end!=m["sources_and_inputs_at_start"] or sha(PROTOCOL)!=m["protocol_sha256"] or sha(FREEZE)!=m["source_freeze_sha256"]:raise RuntimeError("Frozen stress lineage changed")
        if sha(source/"manifest.json")!=m["canonical_manifest_sha256"] or any(sha(source/name)!=digest for name,digest in parent["outputs_sha256"].items()):raise RuntimeError("Canonical producer changed during stress")
        m.update(status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),phase="complete",sources_and_inputs_at_end=end,
            summary_rows=len(summaries),cohort_comparisons=len(comparisons),path_rows=len(rows),trace_groups=len(groups),
            output_archive_policy="Uniformly thin only alternative menus and shared training copies after exact full base comparison.",
            outputs_sha256={v.name:sha(v) for v in output.iterdir() if v.name!="manifest.json"})
        canonical.writej(output/"manifest.json",m)
    except BaseException as error:
        (output/"failure.txt").write_text(traceback.format_exc())
        m.update(status="failed",error_type=type(error).__name__,error=str(error),failed_utc=datetime.now(timezone.utc).isoformat())
        canonical.writej(output/"manifest.json",m);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true");parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.execute:run(args.output)
    else:print(json.dumps(dict(settings=SETTINGS,development_only=True,new_selection=False,market_values_parsed=False),indent=2))
