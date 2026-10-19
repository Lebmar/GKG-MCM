"""Read-only extraction and figures for the audited joint-liquidity experiment.

--self-test uses synthetic rows in memory and opens no result files. Actual
extraction is opt-in and first requires the new manifest and independent audit.
No market loader, fitter, forecast, simulator or selection function is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"research"), str(ROOT/"scripts")]
from safeinvest_research.metrics import discrete_cvar
from plot_development_structure import configure, check_text_bounds

NEW = ROOT/"results/model_review/joint_liquidity_refit_v1"
OLD = ROOT/"results/model_review/nested_dynamic_calibration_v1"
OUT = ROOT/"results/model_review/joint_liquidity_summary_v1"
FIG = ROOT/"research/figures/joint_liquidity_refit"
NEW_MAP = {"joint_canonical_learner":"new_canonical", "joint_tie_set_refit_learner":"new_tie_set_refit",
           "annual_then_pending":"new_pending_reference", "equal_thirds_annual":"new_equal_reference"}
OLD_MAP = {"nested_dynamic_learner":"old_floor0_learner", "annual_then_pending":"old_pending_reference",
           "equal_thirds_annual":"old_equal_reference"}
LEARNERS = ["old_floor0_learner", "new_canonical", "new_tie_set_refit"]
ORDER = LEARNERS + ["new_pending_reference", "new_equal_reference", "old_pending_reference", "old_equal_reference"]
PAIRS = [("new_tie_set_refit","new_canonical"), ("new_tie_set_refit","old_floor0_learner"),
         ("new_canonical","old_floor0_learner"),
         *[(p,r) for p in ("new_tie_set_refit","new_canonical") for r in ("new_pending_reference","new_equal_reference")]]
TOL = 1e-12
METRICS = ["log_growth","cagr","mdd","volatility","terminal_real","nominal_fees","nominal_moved","trade_fraction","excess","breach"]
BLUE, GOLD, GRAY, LIGHT, TEXT = "#1B5E8C", "#AD7A19", "#57616B", "#D6DBDF", "#242B32"
PRICE = {0:"0",1:"0.001",2:"0.003",3:"0.01",4:"0.03",5:"0.1",6:"0.3",7:"1",8:"risk-first"}
FIGURE_SOURCE = "Source: HKUST(GZ) 2026 MCM SafeInvest synthetic scenarios 1-120 (CC BY-NC-SA 4.0); team87 saved development analyses."


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def scalar(x): return None if x is None or pd.isna(x) else x.item() if isinstance(x,np.generic) else x


def require_bool(frame, field):
    if field not in frame or not frame[field].map(lambda x:isinstance(x,(bool,np.bool_))).all():
        raise ValueError(f"Observed boolean field required: {field}")


def case_of(label):
    if label=="Balanced": return "balanced_case"
    if label=="Review": return "q1_review"
    if label in ("Conservative","Growth"): return "not_balanced_case"
    raise ValueError("Unknown Q1 effective assignment")


def tail_cvar(values):
    a=np.sort(np.asarray(values,float))[::-1]; mass=.1*len(a)
    if not len(a): return None
    whole=int(math.floor(mass)); fraction=mass-whole
    total=float(a[:whole].sum())+(float(a[whole])*fraction if fraction>TOL else 0)
    independent=total/mass; shared=float(discrete_cvar(a))
    if not np.isclose(independent,shared,atol=1e-13,rtol=0): raise ValueError("CVaR fractional-tail convention differs")
    return independent


def check_financial(frame, *, horizon=30, loss=.4, fees=True):
    require_bool(frame,"assigned"); obs=frame.loc[frame.assigned]; absent=frame.loc[~frame.assigned]
    fields=METRICS if fees else ["log_growth","cagr","mdd","volatility","terminal_real","excess","breach"]
    if not absent[fields].isna().all().all(): raise ValueError("Review or unavailable financial outcomes were imputed")
    if not len(obs): return
    if not np.isfinite(obs[[x for x in fields if x!="breach"]].to_numpy(float)).all(): raise ValueError("Nonfinite observed outcome")
    require_bool(obs,"breach")
    if not np.allclose(obs.log_growth,np.log(obs.terminal_real/100)/horizon,atol=1e-13,rtol=1e-12): raise ValueError("Own-H growth/real wealth identity failed")
    if not np.allclose(obs.cagr,np.expm1(obs.log_growth),atol=1e-13,rtol=1e-12): raise ValueError("CAGR/G identity failed")
    if not np.array_equal(obs.breach.to_numpy(bool),(obs.mdd-loss>TOL).to_numpy()): raise ValueError("Breach definition changed")
    if not np.allclose(obs.excess,np.maximum(obs.mdd-loss,0),atol=1e-13,rtol=0): raise ValueError("Severity excess differs")
    if fees:
        if not np.allclose(obs.nominal_fees,.0025*obs.nominal_moved,atol=1e-11,rtol=1e-12): raise ValueError("Transfer cost identity differs")
        counts=obs.trade_fraction*30
        if not np.allclose(counts,np.rint(counts),atol=1e-10,rtol=0): raise ValueError("Trade periods not count/30 including initial allocation")


def denominators(frame, reference):
    require_bool(frame,"assigned"); cases=frame.q1_effective_assignment.map(case_of)
    if not cases.eq(frame.case_status).all(): raise ValueError("Q1 case label mismatch")
    out={"original_n":len(frame),"balanced_case_n":int(cases.eq("balanced_case").sum()),
         "q1_review_n":int(cases.eq("q1_review").sum()),"not_balanced_case_n":int(cases.eq("not_balanced_case").sum()),
         "reference_only":reference,"assigned_paths":int(frame.assigned.sum()),"financial_null_paths":int((~frame.assigned).sum())}
    if reference:
        if not frame.reference_only.eq(True).all() or not frame.hypothetical_reference.eq(True).all(): raise ValueError("Reference is not hypothetical")
        if not np.array_equal(frame.execution_status,np.where(frame.assigned,"hypothetical_reference","reference_unavailable")): raise ValueError("Reference availability became service admission")
        out.update(case_auto_n=None,case_execReview_n=None,reference_available_n=int(frame.assigned.sum()),
                   case_reference_available_n=int((frame.assigned & cases.eq("balanced_case")).sum()))
    else:
        if not frame.assigned.eq(frame.execution_status.eq("automated")).all(): raise ValueError("Automatic service status mismatch")
        for status in ["automated","execution_review"]:
            if not cases.loc[frame.execution_status.eq(status)].eq("balanced_case").all(): raise ValueError("Outside-case Q2 service")
        for status in ["q1_review","not_balanced_case"]:
            if not cases.loc[frame.execution_status.eq(status)].eq(status).all(): raise ValueError("Q1 Review/outside-case conflated")
        if not set(frame.execution_status).issubset({"automated","execution_review","q1_review","not_balanced_case"}): raise ValueError("Unknown service status")
        out.update(case_auto_n=int(frame.execution_status.eq("automated").sum()),case_execReview_n=int(frame.execution_status.eq("execution_review").sum()),
                   reference_available_n=None,case_reference_available_n=None)
        if out["balanced_case_n"]!=out["case_auto_n"]+out["case_execReview_n"]: raise ValueError("Balanced service partition failed")
    if out["original_n"]!=out["balanced_case_n"]+out["q1_review_n"]+out["not_balanced_case_n"]: raise ValueError("Original path partition failed")
    return out


def financial_summary(frame):
    reference=bool(frame.reference_only.iloc[0]); out=denominators(frame,reference); check_financial(frame)
    obs=frame.loc[frame.assigned]; n=len(obs)
    out.update(scope=str(frame.scope.iloc[0]),comparison_id=str(frame.comparison_id.iloc[0]),policy=str(frame.policy.iloc[0]),
               conditional_financial_n=n,service_auto_fraction=None if reference else out["case_auto_n"]/len(frame),
               financial_population="hypothetical_reference_available_paths" if reference else "automatically_executed_paths")
    names=["breach_count","conditional_breach_probability","breaches_per_original_n","mean_log_growth","median_cagr","p10_cagr","median_volatility","median_mdd","p90_mdd","cvar90_mdd","mean_nominal_fees","mean_trade_fraction","mean_excess"]
    if not n: return {**out,**dict.fromkeys(names)}
    k=int(obs.breach.sum())
    out.update(breach_count=k,conditional_breach_probability=k/n,breaches_per_original_n=k/len(frame),
        mean_log_growth=float(obs.log_growth.mean()),median_cagr=float(obs.cagr.median()),
        p10_cagr=float(np.quantile(obs.cagr,.1,method="inverted_cdf")),median_volatility=float(obs.volatility.median()),median_mdd=float(obs.mdd.median()),
        p90_mdd=float(np.quantile(obs.mdd,.9,method="inverted_cdf")),cvar90_mdd=tail_cvar(obs.mdd),
        mean_nominal_fees=float(obs.nominal_fees.mean()),mean_trade_fraction=float(obs.trade_fraction.mean()),mean_excess=float(obs.excess.mean()))
    return out


def compare_saved(rows,saved, *, old=False):
    maximum=0.
    for row in rows:
        match=saved.loc[(saved.scope==row["scope"]) & (saved.policy==row["policy"])]
        if len(match)!=1: raise ValueError("Saved summary identity missing/duplicated")
        source=match.iloc[0]
        for key,value in row.items():
            source_key="risk_review_n" if old and key=="q1_review_n" else key
            if source_key not in source: continue
            other=scalar(source[source_key]); value=scalar(value)
            if value is None or other is None:
                if value is not None or other is not None: raise ValueError(f"Saved NULL mismatch: {key}")
            elif isinstance(value,(int,float,bool,np.number)):
                error=abs(float(value)-float(other)); maximum=max(maximum,error)
                if not np.isclose(float(value),float(other),atol=1e-12,rtol=1e-12): raise ValueError(f"Saved metric differs: {key}")
    return maximum


def check_cohorts(frame, reference_keys=None):
    for (scope,comp),part in frame.groupby(["scope","comparison_id"],sort=False):
        if len(part)!=120 or sorted(part.scenario_id)!=list(range(1,121)): raise ValueError("Original 120 unique scenario identities required")
        if not part.client.eq("C").all() or not part.horizon.eq(30).all() or not part.loss_tolerance.eq(.4).all(): raise ValueError("Only original C H30/L40 enters Q2")
        if scope=="outer_oof":
            if set(part.context_id)!=set(range(1,6)) or not part.groupby("context_id").size().eq(24).all(): raise ValueError("Original five outerfolds24 required")
            keys=part[["scenario_id","context_id"]].astype(int).sort_values("scenario_id").reset_index(drop=True)
            if reference_keys is not None and not keys.equals(reference_keys): raise ValueError("Old/new original OOF fold identity changed")


def pair_summaries(frame):
    pairs=[]; summaries=[]
    for left,right in PAIRS:
        a=frame.loc[(frame.scope=="outer_oof") & (frame.comparison_id==left)].set_index("scenario_id")
        b=frame.loc[(frame.scope=="outer_oof") & (frame.comparison_id==right)].set_index("scenario_id")
        if set(a.index)!=set(range(1,121)) or set(a.index)!=set(b.index): raise ValueError("Paired comparison dropped original identities")
        rows=[]
        for sid in range(1,121):
            x,y=a.loc[sid],b.loc[sid]
            if int(x.context_id)!=int(y.context_id): raise ValueError("Paired paths use different original folds")
            common=bool(x.assigned and y.assigned)
            row={"left":left,"right":right,"scenario_id":sid,"outer_fold":int(x.context_id),
                 "left_auto":bool(x.assigned),"right_financial_available":bool(y.assigned),
                 "right_reference_only":bool(y.reference_only),"common_evaluable":common}
            for field in ["log_growth","cagr","mdd","excess","nominal_fees","trade_fraction","breach"]:
                row["left_"+field]=float(x[field]) if x.assigned else None
                row["right_"+field]=float(y[field]) if y.assigned else None
                row["delta_"+field]=float(x[field])-float(y[field]) if common else None
            rows.append(row)
        paired=pd.DataFrame(rows);both=paired.loc[paired.common_evaluable]
        summary={"left":left,"right":right,"original_pairs":120,"left_auto_n":int(paired.left_auto.sum()),
                 "right_financial_available_n":int(paired.right_financial_available.sum()),"common_evaluable_n":len(both),
                 "right_reference_only":bool(b.reference_only.iloc[0]),
                 "left_common_breach_count":int(both.left_breach.sum()) if len(both) else None,
                 "right_common_breach_count":int(both.right_breach.sum()) if len(both) else None,
                 **{"mean_delta_"+f:float(both["delta_"+f].mean()) if len(both) else None for f in ["log_growth","cagr","mdd","excess","nominal_fees","trade_fraction","breach"]}}
        summaries.append(summary);pairs.extend(rows)
    return pd.DataFrame(pairs),pd.DataFrame(summaries)


def compare_saved_pairs(summary,saved):
    reverse={v:k for k,v in NEW_MAP.items()};maximum=0.;checked=0
    fields={"original_pairs":"original_pairs","left_auto_n":"learner_auto_n",
            "right_financial_available_n":"reference_available_n","common_evaluable_n":"common_auto_n",
            **{"mean_delta_"+x:"mean_delta_"+x for x in ["log_growth","cagr","mdd","excess","nominal_fees","trade_fraction","breach"]}}
    for row in summary.to_dict("records"):
        if row["left"] not in reverse or row["right"] not in reverse:continue
        match=saved.loc[(saved.scope=="outer_oof")&(saved.learner_policy==reverse[row["left"]])&(saved.reference_policy==reverse[row["right"]])]
        if len(match)!=1:raise ValueError("Saved within-new-study pair summary missing")
        for key,source_key in fields.items():
            a,b=scalar(row[key]),scalar(match.iloc[0][source_key])
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError("Saved paired NULL mask differs")
            else:
                maximum=max(maximum,abs(float(a)-float(b)))
                if not np.isclose(a,b,atol=1e-12,rtol=1e-12):raise ValueError("Saved paired statistic differs")
        checked+=1
    if checked!=5:raise ValueError("All five prespecified within-study pair summaries required")
    return maximum


def context_rows(new,old):
    rows=[]
    for context,value in new.items():
        cid=0 if context=="full_development" else int(context.rsplit("_",1)[1])
        for policy,choice in value["methods"].items():
            evidence=choice["selection_evidence"]; inner=evidence.get("inner_sets") or {}
            selected=evidence.get("selected_price_evidence") or {}
            rows.append({"stage":"new_target_floor_30_10_0","context":context,"context_id":cid,"comparison_id":NEW_MAP[policy],
                "training_n":len(value["training_ids"]),"evaluation_n":len(value["evaluation_ids"]),"static_gamma":value["static_gamma"],
                "q1_effective_assignment":value["case"]["effective_assignment"],
                "target_weights":json.dumps(value["case"]["product"]["weights"]) if value["case"]["product"] else None,
                "selected_beta":choice["selected_beta"],"selected_candidate_order":choice["selected_candidate_order"],
                "execution_status":choice["execution_status"],"execution_reason":choice["execution_reason"],
                "top_inner_beta_set_E":json.dumps(inner.get("top_E",[])),
                "top_inner_auto_path_n":inner.get("maximum_auto_coverage"),
                "refit_attempted_betas":json.dumps([r["training_beta"] for r in evidence.get("refit_attempts",[])]),
                "final_training_breach_cap":selected.get("training_breach_cap"),
                "unit":"one_completed_training_refit_not_a_customer"})
    for context,value in old.items():
        cid=0 if context=="full_development" else int(context.rsplit("_",1)[1])
        rows.append({"stage":"old_floor0_complete_process","context":context,"context_id":cid,"comparison_id":"old_floor0_learner",
            "training_n":len(value["training_ids"]),"evaluation_n":len(value["evaluation_ids"]),"static_gamma":value["static_gamma"],
            "q1_effective_assignment":value["case"]["effective_assignment"],
            "target_weights":json.dumps(value["case"]["product"]["weights"]) if value["case"]["product"] else None,
            "selected_beta":value["beta_selection"]["selected_beta"],"selected_candidate_order":value["selected_candidate_order"],
            "execution_status":value["execution_status"],"execution_reason":value["execution_reason"],
            "top_inner_beta_set_E":None,"top_inner_auto_path_n":value["beta_selection"].get("maximum_auto_coverage"),
            "refit_attempted_betas":None,"final_training_breach_cap":(value.get("selected_price_training_evidence") or {}).get("training_breach_cap"),
            "unit":"one_completed_training_refit_not_a_customer"})
    return pd.DataFrame(rows)


def admission_overlap(frame):
    a=frame.loc[(frame.scope=="outer_oof")&(frame.comparison_id=="new_tie_set_refit")].set_index("scenario_id")
    b=frame.loc[(frame.scope=="outer_oof")&(frame.comparison_id=="new_canonical")].set_index("scenario_id")
    rows=[]
    for sid in range(1,121):
        x,y=a.loc[sid],b.loc[sid]
        group=("both_auto" if x.assigned and y.assigned else "primary_only_auto" if x.assigned else
               "canonical_only_auto" if y.assigned else "neither_auto")
        row={"scenario_id":sid,"outer_fold":int(x.context_id),"admission_group":group,
             "primary_auto":bool(x.assigned),"canonical_auto":bool(y.assigned)}
        for prefix,z in [("primary",x),("canonical",y)]:
            for field in ["breach","log_growth","cagr","mdd","nominal_fees","trade_fraction"]:
                row[prefix+"_"+field]=float(z[field]) if z.assigned else None
        rows.append(row)
    records=pd.DataFrame(rows);summary=[]
    for group in ["both_auto","primary_only_auto","canonical_only_auto","neither_auto"]:
        part=records.loc[records.admission_group==group];r={"admission_group":group,"original_market_path_n":len(part),"unit":"market_scenario_paths_for_given_C"}
        for prefix in ["primary","canonical"]:
            obs=part.loc[part[prefix+"_auto"]];n=len(obs)
            r.update({prefix+"_auto_n":n,prefix+"_breach_count":int(obs[prefix+"_breach"].sum()) if n else None,
                      prefix+"_conditional_breach_probability":float(obs[prefix+"_breach"].mean()) if n else None,
                      prefix+"_mean_log_growth":float(obs[prefix+"_log_growth"].mean()) if n else None})
        for prefix,comp in [("pending_reference","new_pending_reference"),("equal_reference","new_equal_reference")]:
            available=frame.loc[(frame.scope=="outer_oof")&(frame.comparison_id==comp)&frame.scenario_id.isin(part.scenario_id)&frame.assigned]
            r[prefix+"_available_n"]=len(available)
            r[prefix+"_breach_count"]=int(available.breach.sum()) if len(available) else None
        summary.append(r)
    return records,pd.DataFrame(summary)


def compare_common_saved_actions(frame,new_directory,old_directory):
    """Compare every stored named array (including Q); never rerun a controller."""
    group_sets={"new":read_json(new_directory/"trace_groups.json")["actual"],
                "old":read_json(old_directory/"trace_groups.json")["actual"]}
    policy_map={**{v:k for k,v in NEW_MAP.items()},**{v:k for k,v in OLD_MAP.items()}}
    plans=[("new_methods_common_auto","new_tie_set_refit","new_canonical","new","new"),
           ("new_canonical_old_floor0_common_auto","new_canonical","old_floor0_learner","new","old")]
    comparisons=[];all_groups=[];global_error=0.;different_cells=0;unique_ids=set()
    with np.load(new_directory/"actual_traces.npz",allow_pickle=False) as new_archive, np.load(old_directory/"actual_traces.npz",allow_pickle=False) as old_archive:
        archives={"new":new_archive,"old":old_archive}
        def locate(stage,cid,policy):
            candidates=[key for key,value in group_sets[stage].items() if
                value["metadata"].get("context_id")==int(cid) and value["metadata"].get("scope")=="outer_oof" and value["metadata"].get("policy")==policy]
            if len(candidates)!=1:raise ValueError("Saved trace group missing or duplicated")
            return candidates[0]
        for comparison,left_id,right_id,left_stage,right_stage in plans:
            left_rows=frame.loc[(frame.scope=="outer_oof")&(frame.comparison_id==left_id)]
            right_rows=frame.loc[(frame.scope=="outer_oof")&(frame.comparison_id==right_id)]
            common=left_rows.loc[left_rows.assigned & left_rows.scenario_id.isin(right_rows.loc[right_rows.assigned,"scenario_id"])]
            groups=[]
            for cid,part in common.groupby("context_id",sort=True):
                left=locate(left_stage,cid,policy_map[left_id]);right=locate(right_stage,cid,policy_map[right_id])
                la,ra=archives[left_stage],archives[right_stage]
                a,b=group_sets[left_stage][left]["scenario_ids"],group_sets[right_stage][right]["scenario_ids"]
                ids=sorted(part.scenario_id.astype(int));unique_ids.update(ids)
                ia=[a.index(sid) for sid in ids];ib=[b.index(sid) for sid in ids]
                lf={key[len(left)+2:] for key in la.files if key.startswith(left+"__")}
                rf={key[len(right)+2:] for key in ra.files if key.startswith(right+"__")}
                details=[]
                for name in sorted(lf & rf):
                    x,y=la[left+"__"+name],ra[right+"__"+name]
                    if x.ndim and x.shape[0]==len(a):x=x[ia]
                    if y.ndim and y.shape[0]==len(b):y=y[ib]
                    same_shape=x.shape==y.shape;same_dtype=x.dtype==y.dtype
                    floating=x.dtype.kind in "fc" and y.dtype.kind in "fc"
                    equal=bool(np.array_equal(x,y,equal_nan=True)) if same_shape and floating else bool(np.array_equal(x,y)) if same_shape else False
                    error=None;different=None
                    if same_shape:
                        mask=(x==y)
                        if floating:mask=mask | (np.isnan(x)&np.isnan(y))
                        different=int(np.count_nonzero(~mask));different_cells+=different
                        if x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
                            finite=np.isfinite(x)&np.isfinite(y)
                            error=float(np.max(np.abs(x[finite].astype(float)-y[finite].astype(float)),initial=0));global_error=max(global_error,error)
                    details.append({"name":name,"left_shape":list(x.shape),"right_shape":list(y.shape),
                        "left_dtype":str(x.dtype),"right_dtype":str(y.dtype),"same_shape":same_shape,"same_dtype":same_dtype,
                        "equal_nan_applied":floating,"exact_values_equal":equal,"different_cells":different,"max_abs_finite_error":error})
                group={"comparison":comparison,"outer_fold":int(cid),"scenario_ids":ids,"paths":len(ids),
                    "left_group":left,"right_group":right,"named_array_sets_equal":lf==rf,
                    "left_only_arrays":sorted(lf-rf),"right_only_arrays":sorted(rf-lf),"arrays_compared":len(details),
                    "all_named_arrays_exact":lf==rf and all(d["same_shape"] and d["same_dtype"] and d["exact_values_equal"] for d in details),
                    "arrays":details}
                groups.append(group);all_groups.append(group)
            comparisons.append({"comparison":comparison,"left":left_id,"right":right_id,"common_paths":len(common),
                "all_named_arrays_exact":all(g["all_named_arrays_exact"] for g in groups) if len(common) else None,"groups":groups})
    total=sum(c["common_paths"] for c in comparisons)
    return {"scope":"Same-ID/same-fold common automatic OOF paths; read saved arrays, no simulator or forecast called.",
            "comparisons":comparisons,"common_paths":comparisons[0]["common_paths"],
            "comparison_path_rows":total,"unique_scenario_ids_across_comparisons":len(unique_ids),
            "numeric_max_abs_error":global_error if total else None,"different_cells":different_cells if total else None,
            "all_named_arrays_exactly_equal":all(g["all_named_arrays_exact"] for g in all_groups) if total else None,
            "states_and_actions_exactly_equal":all(g["all_named_arrays_exact"] for g in all_groups) if total else None,
            "Q_arrays_included":True,"comparison_rows_not_independent_samples":True,
            "not_a_claim_about_unobserved_actions_on_Review_paths":True}


def q1_extract(assignments,paths,saved,old_static):
    if len(assignments)!=30 or set(assignments.client)!=set("ABCDE"): raise ValueError("Six contexts times five questionnaire examples required")
    own={"A":(5,.1,"high"),"B":(15,.25,"medium"),"C":(30,.4,"low"),"D":(25,.15,"medium"),"E":(6,.35,"high")}
    floors={"high":.3,"medium":.1,"low":0}; rows=[]; comparison=assignments.copy()
    for record in assignments.to_dict("records"):
        h,l,label=own[record["client"]]
        if (record["horizon"],record["loss_tolerance"],record["liquidity"])!=(h,l,label): raise ValueError("Q1 own H/L/liquidity changed")
        if record["effective_assignment"]!="Review":
            weights=json.loads(record["target_weights"])
            if min(weights)<=0 or abs(sum(weights)-1)>TOL or weights[2]<floors[label]-TOL: raise ValueError("New Q1 target fails positive-assets or declared ordinal floor")
    for (scope,client),g in paths.groupby(["scope","client"],sort=False):
        h,l,_=own[client]
        if len(g)!=120 or sorted(g.scenario_id)!=list(range(1,121)): raise ValueError("Q1 annual path cohort incomplete")
        check_financial(g,horizon=h,loss=l,fees=False);auto=g.loc[g.assigned]
        k=int(auto.breach.sum()) if len(auto) else None
        row={"scope":scope,"client":client,"horizon":h,"loss_tolerance":l,"original_n":120,
             "annual_assigned_n":len(auto),"q1_review_n":120-len(auto),"breach_count":k,
             "conditional_breach_probability":k/len(auto) if len(auto) else None,
             "mean_log_growth":float(auto.log_growth.mean()) if len(auto) else None,
             "median_cagr":float(auto.cagr.median()) if len(auto) else None,
             "p10_cagr":float(np.quantile(auto.cagr,.1,method="inverted_cdf")) if len(auto) else None,
             "median_volatility":float(auto.volatility.median()) if len(auto) else None,
             "median_mdd":float(auto.mdd.median()) if len(auto) else None,
             "p90_mdd":float(np.quantile(auto.mdd,.9,method="inverted_cdf")) if len(auto) else None,
             "cvar90_mdd":tail_cvar(auto.mdd) if len(auto) else None,
             "mean_excess":float(auto.excess.mean()) if len(auto) else None,
             "unit":"market_paths_for_this_given_profile_not_population_customers"}
        source=saved.loc[(saved.scope==scope)&(saved.client==client)]
        if len(source)!=1: raise ValueError("Saved Q1 summary missing")
        for key in ["original_n","annual_assigned_n","q1_review_n","breach_count","conditional_breach_probability","mean_log_growth","median_cagr","median_mdd"]:
            a,b=scalar(row[key]),scalar(source.iloc[0][key])
            if a is None or b is None:
                if a is not None or b is not None: raise ValueError("Q1 summary NULL differs")
            elif not np.isclose(a,b,atol=1e-12,rtol=1e-12): raise ValueError("Q1 summary metric differs")
        rows.append(row)
    old_rows=[]
    for context in ["full_development",*[f"outer_fold_{i}" for i in range(1,6)]]:
        cid=0 if context=="full_development" else int(context[-1]);fit=old_static[f"c{cid}_refit"]["refit"]
        menu={p["name"]:p for p in fit["raw_fit"]["menu"]} if fit["raw_fit"] else {}
        for client,label in fit["effective_assignments"].items():
            p=menu.get(label)
            old_rows.append({"context":context,"client":client,"old_floor0_effective_assignment":label,
                             "old_floor0_target_weights":json.dumps(p["weights"]) if p else None})
    comparison=comparison.merge(pd.DataFrame(old_rows),on=["context","client"],validate="one_to_one")
    comparison["assignment_changed_from_old_floor0"]=comparison.effective_assignment.ne(comparison.old_floor0_effective_assignment)
    def targets_differ(row):
        a,b=scalar(row.target_weights),scalar(row.old_floor0_target_weights)
        if a is None or b is None:return a!=b
        return not np.allclose(json.loads(a),json.loads(b),atol=TOL,rtol=0)
    comparison["target_changed_from_old_floor0"]=comparison.apply(targets_differ,axis=1)
    return comparison,pd.DataFrame(rows)


def audited_inputs(directory,names):
    manifest=read_json(directory/"manifest.json");audit=read_json(directory/"independent_audit.json")
    if audit.get("status")!="passed" or manifest.get("final_scenarios_loaded") is not False or audit.get("final_scenarios_loaded") is not False:
        raise RuntimeError("Stable no-final archive and passed independent audit required before reading result rows")
    if manifest.get("scenario_ids_loaded")!=list(range(1,121)) or manifest.get("training_tasks")!=30 or manifest.get("candidate_count")!=9:
        raise RuntimeError("Unexpected source study scope")
    inputs={str((directory/x).relative_to(ROOT)):sha(directory/x) for x in ["manifest.json","independent_audit.json",*names]}
    for name in names:
        if sha(directory/name)!=manifest["outputs_sha256"][name]: raise RuntimeError("Frozen source output changed")
    return manifest,audit,inputs


def finish(fig,name,output):
    check_text_bounds(fig)
    if output is not None:
        fig.savefig(output/(name+".png"),dpi=240,metadata={"Software":"summarize_joint_liquidity_refit.py"})
        fig.savefig(output/(name+".svg"),metadata={"Date":None,"Creator":"summarize_joint_liquidity_refit.py"})
    plt.close(fig)


def service_figure(summary,contexts,output=None,overlap=None):
    oof=summary.loc[summary.scope=="outer_oof"].set_index("comparison_id");learners=oof.loc[LEARNERS]
    fig,axes=plt.subplots(2,2,figsize=(14.2,10.9));fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.84,wspace=.35,hspace=.58)
    labels=["Old floor0\ncanonical","New reserve\ncanonical","New reserve\ntie-set refit"]
    bottom=np.zeros(3);statuses=[("case_auto_n","Automatic execution",BLUE),("case_execReview_n","Execution review",GOLD),
        ("q1_review_n","Q1 review",GRAY),("not_balanced_case_n","Outside Balanced",LIGHT)]
    for field,label,color in statuses:
        values=learners[field].to_numpy(float)
        axes[0,0].bar(range(3),values,bottom=bottom,width=.6,color=color,edgecolor="white",linewidth=.8)
        for x,y,b in zip(range(3),values,bottom):
            if y>=8: axes[0,0].text(x,b+y/2,str(int(y)),ha="center",va="center",fontsize=12,color="white" if color in [BLUE,GRAY] else TEXT)
        bottom+=values
    if not np.array_equal(bottom,[120,120,120]): raise ValueError("Service chart does not partition original120")
    ax=axes[0,0];ax.set_ylim(0,126);ax.set_yticks([0,24,48,72,96,120]);ax.set_xticks(range(3),labels);ax.set_ylabel("Original market paths (N = 120)");ax.set_title("(a) C service decisions",loc="left")
    ax=axes[0,1];probs=learners.conditional_breach_probability.dropna();top=max(.20,float(probs.max())*1.5 if len(probs) else .20)
    for x,row in enumerate(learners.itertuples()):
        n=int(row.case_auto_n)
        if n:
            ax.bar(x,row.conditional_breach_probability,width=.6,color=GRAY if x==0 else BLUE,edgecolor=TEXT,linewidth=.5)
            ax.text(x,row.conditional_breach_probability+.04*top,f"{int(row.breach_count)}/{n}",ha="center",va="bottom",fontsize=12)
        else: ax.text(x,.10*top,"No auto\noutcomes",ha="center",va="bottom",fontsize=12)
    ax.axhline(.1,color=TEXT,linestyle="--",linewidth=1);ax.set_ylim(0,top);ax.set_xlim(-.6,2.6);ax.set_xticks(range(3),labels);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel("Exceedances / automatic paths");ax.set_title("(b) Conditional risk; path subsets can differ",loc="left")
    ax=axes[1,0];refs=oof.loc[["new_pending_reference","new_equal_reference"]]
    if not refs.case_auto_n.isna().all() or not refs.reference_only.all(): raise ValueError("References are not automatic service")
    ax.barh([0,1],refs.reference_available_n,color=LIGHT,height=.55,edgecolor=GRAY)
    for y,row in enumerate(refs.itertuples()):ax.text(.97,y,f"{int(row.reference_available_n)}/120 available\n{int(row.case_reference_available_n)} within Balanced",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=12)
    ax.set_xlim(0,125);ax.set_xticks([0,40,80,120]);ax.set_yticks([0,1],["New pending\nreference","Equal thirds\nreference"]);ax.set_ylim(1.6,-.6);ax.set_xlabel("Hypothetical financial availability");ax.set_title("(c) References are never service fallback",loc="left")
    ax=axes[1,1];ax.axis("off");ax.set_title("(d) Final training refits: beta / price",loc="left")
    cols=["Context","Canonical","Tie-set refit"]
    table_rows=[]
    for context in ["full_development",*[f"outer_fold_{i}" for i in range(1,6)]]:
        row=["Full120" if context=="full_development" else "Outer "+context[-1]]
        for comp in ["new_canonical","new_tie_set_refit"]:
            d=contexts.loc[(contexts.context==context)&(contexts.comparison_id==comp)].iloc[0]
            if d.execution_status=="automated":text=f"{d.selected_beta:g} / {PRICE[int(d.selected_candidate_order)]}"
            elif d.execution_status=="not_balanced_case":text="Not Balanced"
            elif d.execution_status=="q1_review":text="Q1 Review"
            else:text="Exec Review"
            row.append(text)
        table_rows.append(row)
    tab=ax.table(cellText=table_rows,colLabels=cols,loc="center",cellLoc="center",bbox=[0,.01,1,.91],colWidths=[.23,.38,.39]);tab.auto_set_font_size(False);tab.set_fontsize(12)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor(LIGHT);cell.set_linewidth(.6)
        if r==0:cell.set_facecolor(LIGHT)
    for ax in [axes[0,0],axes[0,1]]:ax.grid(axis="y");ax.set_axisbelow(True)
    axes[1,0].grid(axis="x");axes[1,0].set_axisbelow(True)
    fig.text(.075,.965,"Joint target-liquidity fitting: service coverage and conditional risk",fontsize=17,weight="bold",ha="left")
    fig.text(.075,.925,"Original client example C; H = 30, L = 40%; new reserve convention: High/Medium/Low target bills = 30%/10%/no extra floor.",fontsize=12,ha="left")
    fig.legend(handles=[Patch(facecolor=c,label=l) for _,l,c in statuses],loc="upper left",bbox_to_anchor=(.07,.90),ncol=4,frameon=False)
    lead="Full120 is a training refit; all five outer fits complete computation. Different products or reviews are not numerical fit failures."
    if overlap is not None:
        extra=overlap.loc[overlap.admission_group=="primary_only_auto"].iloc[0];n=int(extra.primary_auto_n)
        lead=(f"Primary-only automatic paths: {int(extra.primary_breach_count)}/{n} exceedances. Pooled risk can hide a higher rate within one fit." if n else
              "No primary-only automatic paths; subgroup risk is unestimated. Pooled risk is not a per-fit bound.")
    fig.text(.075,.13,lead+"\nFull120 is a training refit; all five outer fits completed computation. A review response is not a numerical failure.\n120 market scenarios and 5 training fits are different units, not a sample of 120 customers. References are hypothetical.\nRead risk with coverage and matched subsets. Dashed 10% is a service reference, not a confidence bound.",fontsize=12,ha="left",va="top",linespacing=1.45)
    fig.text(.075,.025,FIGURE_SOURCE,fontsize=12,ha="left",va="top")
    finish(fig,"01_service_coverage_and_refits",output)


def paired_figure(summary,output=None):
    labels={"old_floor0_learner":"old floor0","new_canonical":"new canonical","new_tie_set_refit":"new tie-set",
            "new_pending_reference":"new pending ref","new_equal_reference":"equal thirds ref"}
    rows=[]
    for left,right in PAIRS: rows.append(summary.loc[(summary.left==left)&(summary.right==right)].iloc[0])
    names=[f"{labels[r.left]} - {labels[r.right]}\n(n={int(r.common_evaluable_n)})" for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(14.5,12.0),sharey=True);fig.subplots_adjust(left=.28,right=.97,bottom=.18,top=.83,wspace=.28,hspace=.37)
    settings=[("mean_delta_log_growth",100,"(a) Annual log growth","Mean difference (pp per year)"),
        ("mean_delta_mdd",100,"(b) Whole-H maximum drawdown","Mean difference in MDD (pp)"),
        ("mean_delta_nominal_fees",1,"(c) Actual transaction fees","Nominal difference (per initial 100)"),
        ("mean_delta_trade_fraction",100,"(d) Actual trade frequency","Mean difference in trade share (pp)")]
    for ax,(field,scale,title,xlabel) in zip(axes.flat,settings):
        values=np.array([r[field] for r in rows],float)*scale;finite=values[np.isfinite(values)];m=max(float(np.max(np.abs(finite),initial=0)),.01)
        ax.set_xlim(-1.35*m,1.35*m);ax.axvline(0,color=TEXT,linewidth=.8)
        for y,(v,row) in enumerate(zip(values,rows)):
            if row.common_evaluable_n and np.isfinite(v):
                ax.barh(y,v,height=.60,color=BLUE if v>=0 else GOLD,edgecolor=TEXT,linewidth=.4)
                ax.annotate(f"{v:+.3f}",(v,y),xytext=(4 if v>=0 else -4,0),textcoords="offset points",ha="left" if v>=0 else "right",va="center",fontsize=12)
            else: ax.text(0,y,"No common automatic outcomes",ha="center",va="center",fontsize=12)
        ax.set_yticks(range(len(rows)),names);ax.set_ylim(len(rows)-.4,-.6);ax.set_title(title,loc="left");ax.set_xlabel(xlabel);ax.grid(axis="x");ax.set_axisbelow(True)
    fig.text(.075,.962,"Matched C financial outcomes: same original scenario IDs and outer folds",fontsize=17,weight="bold",ha="left")
    fig.text(.075,.919,"Difference = left minus right. Only the automatically served / available intersection is scored; every pair retains all original 120 identities.",fontsize=12,ha="left")
    fig.text(.075,.875,"References are hypothetical financial comparators. Trade share counts actual trading periods / 30, including the initial decision.",fontsize=12,ha="left")
    fig.text(.075,.12,"New floors can change the shared menu. C's Low=0 does not imply an unchanged target; this is a complete-procedure comparison.\nTop-inner E is a beta tie set, not behavioral equivalence or Profile E. H6 Profile E has its own Q1 annual assessment only.\nReused development data and overlapping training folds give neither a fresh blind test nor a fixed-policy risk certificate.\nNo policy winner is selected. References are hypothetical; different common subsets contain different market paths and fit contexts.",fontsize=12,ha="left",va="top",linespacing=1.45)
    fig.text(.075,.025,FIGURE_SOURCE,fontsize=12,ha="left",va="top")
    finish(fig,"02_common_subset_financial_differences",output)


def synthetic_frame(auto=48):
    rows=[]
    for comp in ORDER:
        ref="reference" in comp
        for sid in range(1,121):
            available=(sid<=auto) if comp!="new_equal_reference" and comp!="old_equal_reference" else True
            status=("hypothetical_reference" if available else "reference_unavailable") if ref else ("automated" if available else "execution_review")
            growth=.03; mdd=.45 if sid%7==0 else .2
            metrics=dict(zip(METRICS,[growth,np.expm1(growth),mdd,.1,100*np.exp(30*growth),.1,40.,1/30,max(mdd-.4,0),mdd-.4>TOL])) if available else dict.fromkeys(METRICS,np.nan)
            rows.append({"scope":"outer_oof","comparison_id":comp,"policy":comp,"client":"C","horizon":30,"loss_tolerance":.4,
                "scenario_id":sid,"context_id":(sid-1)//24+1,"q1_effective_assignment":"Balanced","case_status":"balanced_case",
                "assigned":available,"reference_only":ref,"hypothetical_reference":ref,"execution_status":status,**metrics})
    return pd.DataFrame(rows)


def self_test():
    configure()
    for n in [0,24,48,120]:
        frame=synthetic_frame(n);rows=[financial_summary(g) for _,g in frame.groupby("comparison_id",sort=False)]
        refs=[r for r in rows if r["reference_only"]]
        assert all(r["case_auto_n"] is None for r in refs)
        learner=next(r for r in rows if r["comparison_id"]=="new_tie_set_refit")
        assert learner["case_auto_n"]==n
        if not n: assert learner["breach_count"] is None and learner["mean_log_growth"] is None
        pairs,summary=pair_summaries(frame)
        assert len(pairs)==120*len(PAIRS) and summary.original_pairs.eq(120).all()
        overlap_paths,partition=admission_overlap(frame)
        assert len(overlap_paths)==120 and partition.original_market_path_n.sum()==120
        for prefix in ["primary","canonical"]:
            empty=partition.loc[partition[prefix+"_auto_n"].eq(0)]
            assert empty[[prefix+"_breach_count",prefix+"_mean_log_growth"]].isna().all().all()
        if not n: assert summary.mean_delta_mdd.isna().all()
        if n in [0,48]:
            contexts=pd.DataFrame([{"context":ctx,"comparison_id":p,"execution_status":"execution_review" if not n else "automated", "selected_beta":.075,"selected_candidate_order":4}
                for ctx in ["full_development",*[f"outer_fold_{i}" for i in range(1,6)]] for p in ["new_canonical","new_tie_set_refit"]])
            service_figure(pd.DataFrame(rows),contexts);paired_figure(summary)
    frame=synthetic_frame(48);bad=frame.copy();bad.loc[(bad.comparison_id=="old_floor0_learner")&(bad.scenario_id==1),"context_id"]=2
    try: pair_summaries(bad)
    except ValueError: pass
    else: raise AssertionError("Mismatched folds were accepted")
    h6=frame.loc[(frame.comparison_id=="new_canonical")&frame.assigned].copy()
    try: check_financial(h6,horizon=6)
    except ValueError: pass
    else: raise AssertionError("H30 outcomes became H6 evidence")
    extra=frame.comparison_id.eq("new_tie_set_refit")&frame.scenario_id.between(49,72)
    reference=frame.loc[frame.comparison_id.eq("new_equal_reference")&frame.scenario_id.between(49,72)]
    for field in METRICS:frame.loc[extra,field]=reference[field].to_numpy()
    frame.loc[extra,"assigned"]=True;frame.loc[extra,"execution_status"]="automated"
    _,partition=admission_overlap(frame)
    assert partition.set_index("admission_group").original_market_path_n.to_dict()=={"both_auto":48,"primary_only_auto":24,"canonical_only_auto":0,"neither_auto":48}
    for n in [1,24,48,96,120]: tail_cvar(np.linspace(0,.8,n))
    print("Synthetic checks passed: NULL/service/reference units, original120 pairs, zero common subset, own-H, fractional CVaR and text bounds; no result files read.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--include-stable-new",action="store_true")
    parser.add_argument("--confirm-visual-review",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test();return
    if not args.include_stable_new: raise SystemExit("Use --self-test while preparing; audited actual extraction requires --include-stable-new")
    new_names=["client_paths.csv","summary.csv","fit_admission.csv","context_selections.json","beta_summaries.csv",
               "q1_context_assignments.csv","q1_effective_paths.csv","q1_summary.csv","paired_summary.csv","actual_traces.npz","trace_groups.json"]
    old_names=["client_paths.csv","summary.csv","context_selections.json","static_refits.json","actual_traces.npz","trace_groups.json"]
    new_manifest,new_audit,new_inputs=audited_inputs(NEW,new_names)
    old_manifest,old_audit,old_inputs=audited_inputs(OLD,old_names)
    if new_manifest.get("liquidity_floors")!={"high":.3,"medium":.1,"low":0.}: raise RuntimeError("Declared target floor changed")
    if new_manifest.get("primary_method")!="joint_tie_set_refit_learner":raise RuntimeError("Predeclared primary method changed")
    inputs={**new_inputs,**old_inputs}
    protocol_path=ROOT/"docs/joint_liquidity_refit_protocol.json"
    protocol_relative=str(protocol_path.relative_to(ROOT))
    if sha(protocol_path)!=new_manifest["sources"][protocol_relative]:raise RuntimeError("Frozen source protocol changed")
    protocol=read_json(protocol_path);inputs[protocol_relative]=sha(protocol_path)
    attempts=protocol.get("implementation_attempts",[])
    for attempt in attempts:
        receipt=ROOT/attempt["receipt"]
        if sha(receipt)!=attempt["receipt_sha256"]:raise RuntimeError("Implementation failure receipt changed")
        inputs[attempt["receipt"]]=sha(receipt)
    source_files=[Path(__file__).resolve(),ROOT/"scripts/plot_development_structure.py",ROOT/"research/safeinvest_research/metrics.py"]
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    new=pd.read_csv(NEW/"client_paths.csv");old=pd.read_csv(OLD/"client_paths.csv")
    new["comparison_id"]=new.policy.map(NEW_MAP);old["comparison_id"]=old.policy.map(OLD_MAP)
    if new.comparison_id.isna().any() or old.comparison_id.isna().any(): raise RuntimeError("Unplanned policy in source")
    old["case_status"]=old.case_status.replace({"risk_review":"q1_review"});old["execution_status"]=old.execution_status.replace({"risk_review":"q1_review"})
    new["stage"]="new_joint_floor_30_10_0";old["stage"]="old_floor0_complete_process"
    keys=old.loc[(old.scope=="outer_oof")&(old.comparison_id=="old_floor0_learner"),["scenario_id","context_id"]].astype(int).sort_values("scenario_id").reset_index(drop=True)
    frame=pd.concat([old,new],ignore_index=True,sort=False);check_cohorts(frame,keys)
    summaries=pd.DataFrame([financial_summary(g) for _,g in frame.groupby(["scope","comparison_id"],sort=False)])
    new_error=compare_saved(summaries.loc[summaries.comparison_id.isin(NEW_MAP.values())].to_dict("records"),pd.read_csv(NEW/"summary.csv"))
    old_error=compare_saved(summaries.loc[summaries.comparison_id.isin(OLD_MAP.values())].to_dict("records"),pd.read_csv(OLD/"summary.csv"),old=True)
    paired_paths,paired=pair_summaries(frame)
    paired_error=compare_saved_pairs(paired,pd.read_csv(NEW/"paired_summary.csv"))
    overlap_paths,overlap=admission_overlap(frame)
    behavior=compare_common_saved_actions(frame,NEW,OLD)
    contexts=context_rows(read_json(NEW/"context_selections.json"),read_json(OLD/"context_selections.json"))
    fit_admission=pd.read_csv(NEW/"fit_admission.csv")
    if len(fit_admission)!=12: raise ValueError("Six contexts times two final methods required")
    require_bool(fit_admission,"initial_admitted")
    for row in fit_admission.to_dict("records"):
        match=contexts.loc[(contexts.context==row["context"])&(contexts.comparison_id==NEW_MAP[row["policy"]])].iloc[0]
        if row["execution_status"]!=match.execution_status or bool(row["initial_admitted"])!=(match.execution_status=="automated"): raise ValueError("Fit admission scope disagrees")
        for key in ["selected_beta","selected_candidate_order"]:
            a,b=scalar(row[key]),scalar(match[key])
            if a!=b:raise ValueError("Fit selected beta/price differs from context evidence")
    q1,q1_metrics=q1_extract(pd.read_csv(NEW/"q1_context_assignments.csv"),pd.read_csv(NEW/"q1_effective_paths.csv"),pd.read_csv(NEW/"q1_summary.csv"),read_json(OLD/"static_refits.json"))
    beta=pd.read_csv(NEW/"beta_summaries.csv")
    if len(beta)!=30: raise ValueError("Six contexts times five inner beta summaries required")
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    outputs={"c_scope_summary.csv":summaries,"c_oof_comparison_paths.csv":frame.loc[frame.scope=="outer_oof"],
        "common_subset_pairs.csv":paired_paths,"common_subset_summary.csv":paired,"context_selection_summary.csv":contexts,
        "q1_context_comparison.csv":q1,"q1_own_horizon_annual_summary.csv":q1_metrics,"new_inner_beta_evidence.csv":beta}
    outputs["new_admission_overlap_paths.csv"]=overlap_paths
    outputs["new_admission_overlap_summary.csv"]=overlap
    outputs["admission_partition.csv"]=overlap.rename(columns={"original_market_path_n":"original_n",
        "primary_auto_n":"primary_observed_n","canonical_auto_n":"canonical_observed_n"})
    for name,data in outputs.items():data.to_csv(OUT/name,index=False)
    outer=contexts.loc[contexts.context_id.ne(0)]
    fit_counts=[]
    for comp,g in outer.groupby("comparison_id",sort=False):
        fit_counts.append({"comparison_id":comp,"completed_outer_fit_n":len(g),"auto_admission_fit_n":int(g.execution_status.eq("automated").sum()),
            "review_or_outside_case_fit_n":int(g.execution_status.ne("automated").sum()),"unit":"completed_training_refit_runs_not_customer_population"})
    interpretation={"purpose":"Describe the prespecified joint target-liquidity prototype, without selecting a method from OOF outcomes.",
        "primary_method":"new_tie_set_refit","target_reserve":"30/10/0 is normative target composition, not identified cash needs, a live balance floor or withdrawal guarantee.",
        "scope":"All30 tasks refit constrained Q1 menus/gamma/admission in every static subfold; both final methods share training and inner evidence.",
        "E_not_profile_E":"Top inner beta set E retains risk-eligible maximum-auto-count/global-growth-tied parameters; it is not a behavioral-equivalence class. Profile E is a separate H6 questionnaire case.",
        "C_indirect_menu_effect":"C Low0 adds no personal reserve floor, yet the shared product menu can change when other profiles' constraints change.",
        "Q1_vs_Q2":"A-E Q1 annual summaries use their own H/L. Only original C H30 enters this Q2 study; E-Balanced H6 cannot inherit C-H30 execution evidence.",
        "coverage_units":{"five_example_profiles":"Representative fictional inputs, not population frequencies.","outer_fit_counts":fit_counts,
            "market_paths":"Original120 market paths for a given profile. 24 equal paths per outer fit can make fit and path ratios numerically equal without making them customer-population coverage."},
        "pairing":"Only same-ID/same-outer-fold common automatically served or hypothetical-available outcomes are scored. Original120 pairs and allNULLs remain.",
        "new_two_method_common_actual_behavior":behavior,
        "incremental_admission":"Primary-only auto paths are a separately reported market-path subgroup, not matched counterfactual outcomes for the canonical method's NULLs. Pooled frequency does not establish a per-fit risk bound.",
        "full_new_Q1_assignments_and_targets":[{k:scalar(v) for k,v in row.items()} for row in q1.loc[q1.context.eq("full_development"),["client","horizon","loss_tolerance","liquidity","floor","effective_assignment","target_weights","q1_review_kind"]].to_dict("records")],
        "no_causal_floor_claim":"Oldfloor0 vs newcanonical changes the entire constrained fitting procedure. Newtie-set vs oldfloor0 also changes final selection; neither is a randomized causal cash-reserve effect.",
        "references":"Hypothetical financial availability, never customer service, fallback or an imputation of Review returns.",
        "risk_limits":"Conditional frequencies and CVaR are descriptive; overlapping training folds/reused development history are not a blind test or fixed-policy certificate. No CI is manufactured.",
        "implementation_attempt_history":"The first attempt had processed development data before duplicate floor export arguments stopped the first final Q1-record export. Its source/protocol/log are archived separately as attempt1_incomplete. The second attempt repaired dictionary export merging and diagnostic tests under the same model/selection rules; this is not first exposure or a fresh blind evaluation.",
        "implementation_attempt_provenance":attempts,
        "fees_and_trades":"Fees are cumulative nominal amounts per initial100, not return pp. Actual trading fraction uses30 decisions including initial; no target updates in this nine-price family.",
        "new_financial_results_from_this_script":False,"final_scenarios_loaded":False,"policy_winner_selected":False}
    write_json(OUT/"interpretation.json",interpretation)
    write_json(OUT/"common_saved_action_receipt.json",behavior)
    configure();service_figure(summaries,contexts,FIG,overlap);paired_figure(paired,FIG)
    for name in ["c_scope_summary.csv","common_subset_summary.csv","context_selection_summary.csv","new_admission_overlap_summary.csv"]:
        (FIG/name).write_bytes((OUT/name).read_bytes())
    output_files=[*outputs,"interpretation.json","common_saved_action_receipt.json"]
    source_end={str(p.relative_to(ROOT)):sha(p) for p in source_files};input_end={p:sha(ROOT/p) for p in inputs}
    if source_end!=sources or input_end!=inputs: raise RuntimeError("Source/input changed during read-only extraction")
    write_json(OUT/"manifest.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"status":"complete_audited_read_only_extraction",
        "sources":sources,"sources_at_end":source_end,"input_sha256":inputs,"input_sha256_at_end":input_end,
        "outputs_sha256":{name:sha(OUT/name) for name in output_files},
        "checks":{"recomputed_new_summary_max_abs_error":new_error,"recomputed_old_summary_max_abs_error":old_error,
            "recomputed_five_within_new_study_paired_summary_max_abs_error":paired_error,
            "all_original120_path_and_fold_identities_preserved":True,"only_common_subsets_scored":True,"Q1_own_H_verified":True,
            "common_saved_actual_behavior":behavior,
            "reference_availability_not_auto_service":True,"new_and_old_upstream_independent_audits_passed":True},
        "new_model_run_or_policy_selection":False,"raw_market_loaded":False,"final_scenarios_loaded":False})
    figure_inputs={str((OUT/name).relative_to(ROOT)):sha(OUT/name) for name in ["manifest.json","c_scope_summary.csv","common_subset_summary.csv","context_selection_summary.csv","new_admission_overlap_summary.csv","interpretation.json"]}
    figure_outputs=["01_service_coverage_and_refits.png","01_service_coverage_and_refits.svg","02_common_subset_financial_differences.png","02_common_subset_financial_differences.svg",
                    "c_scope_summary.csv","common_subset_summary.csv","context_selection_summary.csv","new_admission_overlap_summary.csv"]
    write_json(FIG/"manifest.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),
        "status":"complete_audited_and_visually_reviewed" if args.confirm_visual_review else "rendered_pending_visual_review",
        "visual_review_confirmed":args.confirm_visual_review,"sources":sources,"sources_at_end":source_end,
        "input_sha256":figure_inputs,"input_sha256_at_end":{p:sha(ROOT/p) for p in figure_inputs},
        "outputs_sha256":{name:sha(FIG/name) for name in figure_outputs},
        "checks":{"text_bounds_passed":True,"min_font_points":12,"PNG_dpi":240,"SVG_text_preserved":True,
                  "visible_source_and_license_attribution":FIGURE_SOURCE},
        "raw_market_loaded":False,"final_scenarios_loaded":False,"model_run_or_policy_winner_selection":False})
    print(f"Wrote {OUT} and {FIG}; visual review confirmed:{args.confirm_visual_review}")
    print("Summary manifest SHA",sha(OUT/"manifest.json"));print("Figure manifest SHA",sha(FIG/"manifest.json"))


if __name__=="__main__": main()
