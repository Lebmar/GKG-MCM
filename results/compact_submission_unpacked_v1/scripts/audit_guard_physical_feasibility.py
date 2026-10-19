"""Independent saved-witness audit; no LP solver or producer import.

Reconstruct original selected first-guard states, then recompute eight-region
primal/dual evidence. Floating numerical evidence remains an approximate
diagnostic, never an exact-arithmetic impossibility or policy guarantee.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PLAN=Path('docs/guard_physical_feasibility_witness_audit_plan.json')
FREEZE=Path('docs/guard_physical_feasibility_witness_audit_source_freeze.json')
LP_PLAN=Path('docs/guard_physical_feasibility_protocol.json')
LP_FREEZE=Path('docs/guard_physical_feasibility_source_freeze.json')
OUT=Path('results/guard_physical_feasibility_v1')
ATTEMPT=Path('results/guard_physical_feasibility_witness_audit_attempt_v1')
RECEIPT=OUT/'independent_witness_audit.json'
SOURCES={'scripts/audit_guard_physical_feasibility.py','tests/test_audit_guard_physical_feasibility.py'}
OUTPUTS={'guard_cases.csv.gz','case_inputs_and_lp_witnesses.json.gz','selected_group_inventory.json','summary.json'}
PARENTS=(('V2b',Path('results/state_budget_mitigation_v1')),('V3',Path('results/state_budget_cushion_v1')))
CERT_TOL=1e-8
COMPARE_TOL=1e-8
ATOL=1e-11
RTOL=1e-11
LABELS=('numerical_physical_feasible_witness_above_budget','all_eight_numerical_optima_below_budget',
    'numerical_boundary_unresolved','incomplete_solver_or_certificate_evidence','solver_or_certificate_unresolved')
CSV_FIELDS=('version','scope','job_id','group_id','scenario_id','account_row','period','current_real_wealth',
    'real_peak','running_mdd','budget_floor','release_fraction','cushion_scale','classification',
    'all_eight_numerically_resolved','passed_certificate_count','best_pattern_index','hold_worst_next_real',
    'best_worst_next_real','best_budget_margin_real','maximum_dual_objective_real','best_nominal_fee','comparison_band_real')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def readj(path):return json.loads(Path(path).read_text())


def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def verify_map(root,mapping):
    for name,digest in mapping.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name or sha(root/p)!=digest:
            raise RuntimeError('Frozen independent witness evidence changed: '+name)


class Checks:
    def __init__(self):self.n=0;self.maximum_numeric_difference=0.;self.reported_rawfun_checks=0
    def check(self,condition,label):
        self.n+=1
        if not condition:raise AssertionError(label)
    def value(self,actual,expected,label,*,exact=False):
        if isinstance(expected,dict):
            self.check(isinstance(actual,dict) and set(actual)==set(expected),label+'/keys')
            for key,value in expected.items():self.value(actual[key],value,label+'/'+key,exact=exact)
        elif expected is None or isinstance(expected,(str,bool,int,np.integer)):
            if isinstance(expected,bool):self.check(isinstance(actual,(bool,np.bool_)),label+'/bool')
            elif isinstance(expected,(int,np.integer)):self.check(isinstance(actual,(int,np.integer)) and not isinstance(actual,(bool,np.bool_)),label+'/integer')
            self.check(actual==expected,label)
        elif isinstance(expected,(list,tuple,np.ndarray)):
            if isinstance(expected,(list,tuple)) and any(isinstance(v,dict) or v is None for v in expected):
                self.check(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+'/length')
                for i,(a,e) in enumerate(zip(actual,expected)):self.value(a,e,label+'/'+str(i),exact=exact)
                return
            a=np.asarray(actual);e=np.asarray(expected)
            self.check(a.shape==e.shape,label+'/shape')
            if e.dtype.kind in 'biuUS':
                if e.dtype.kind=='b':self.check(a.dtype.kind=='b',label+'/typedbool')
                elif e.dtype.kind in 'iu':self.check(a.dtype.kind in 'iu',label+'/typedintegers')
                self.check(np.array_equal(a,e),label+'/values')
            else:
                a=np.asarray(actual,float);e=np.asarray(expected,float)
                self.check(np.isfinite(a).all() and np.isfinite(e).all(),label+'/finite')
                delta=float(np.max(np.abs(a-e),initial=0.));self.maximum_numeric_difference=max(self.maximum_numeric_difference,delta)
                self.check(np.array_equal(a,e) if exact else np.all(np.abs(a-e)<=ATOL+RTOL*np.abs(e)),label+'/values')
        else:
            a=float(actual);e=float(expected);self.check(math.isfinite(a) and math.isfinite(e),label+'/finite')
            delta=abs(a-e);self.maximum_numeric_difference=max(self.maximum_numeric_difference,delta)
            self.check(a==e if exact else delta<=ATOL+RTOL*abs(e),label)


def _vector(value,length,label):
    a=np.asarray(value,float)
    if a.shape!=(length,) or not np.isfinite(a).all():raise AssertionError('Finite vector required: '+label)
    return a.tolist()


def verify_pattern(case,record,index,c):
    """Scalar-loop equations, independent of producer matrix construction."""
    label=f"pattern/{index}";mask=[bool(index&bit) for bit in (4,2,1)]
    x=_vector(case['holdings'],3,'holdings');V=math.fsum(x);cp=float(case['cpi']);R=V/cp
    a=np.asarray(case['support_real_gross'],float);rate=.0025
    c.check(V>0 and cp>0 and np.all(np.asarray(x)>=0) and a.ndim==2 and a.shape[1]==3
        and len(a)>0 and np.isfinite(a).all() and np.all(a>0),'positive physical inputs')
    p=[v/V for v in x];b=1+rate*math.fsum(p[j] for j in range(3) if mask[j])
    c.value(record['pattern_index'],index,label+'/index');c.value(record['buy_mask'],mask,label+'/mask')
    c.value(record['normalized_equality_rhs'],b,label+'/rhs')
    optimal=record.get('solver_status')==0 and record.get('solver_success') is True
    if not optimal:
        c.check(record['certificate_passed'] is False,label+'/nonoptimal_not_certificate')
        c.check(bool(record.get('solver_message') or record.get('error')),label+'/failure_evidence')
        return None
    q=_vector(record['normalized_solution'],4,'solution')
    mi=_vector(record['inequality_marginals'],len(a),'inequalitydual')
    me=_vector(record['equality_marginals'],1,'equalitydual')[0]
    ml=_vector(record['lower_marginals'],4,'lowerdual');mu=_vector(record['upper_marginals'],4,'upperdual')
    growth=[math.fsum(float(row[j])*q[j] for j in range(3)) for row in a]
    slack=[v-q[3] for v in growth];eq=math.fsum((1+rate*mask[j])*q[j] for j in range(3))-b
    lower=[p[j] if mask[j] else 0. for j in range(3)]
    upper=[None if mask[j] else p[j] for j in range(3)]
    violation=[0.,abs(eq)]+[-v for v in slack]+[lower[j]-q[j] for j in range(3)]
    violation += [q[j]-upper[j] for j in range(3) if upper[j] is not None]
    station=[]
    for j in range(4):
        objective=-1. if j==3 else 0.
        column=[1. if j==3 else -float(row[j]) for row in a]
        equality_coefficient=0. if j==3 else 1+rate*mask[j]
        station.append(objective-math.fsum(column[s]*mi[s] for s in range(len(a)))-equality_coefficient*me-ml[j]-mu[j])
    signs=[0.]+mi+[-v for v in ml]+mu+[abs(ml[3]),abs(mu[3])]
    signs += [abs(mu[j]) for j in range(3) if upper[j] is None]
    products=[abs(mi[s]*slack[s]) for s in range(len(a))]
    products += [abs(ml[j]*(q[j]-lower[j])) for j in range(3)]
    products += [abs(mu[j]*(upper[j]-q[j])) for j in range(3) if upper[j] is not None]
    dual=math.fsum([b*me]+[lower[j]*ml[j] for j in range(3)]+[upper[j]*mu[j] for j in range(3) if upper[j] is not None])
    fee_ratio=rate*math.fsum(max(q[j]-p[j],0.) for j in range(3))
    pattern_fee=rate*math.fsum(q[j]-p[j] for j in range(3) if mask[j])
    funding=math.fsum(q[:3])+fee_ratio-1;worst=min(growth)
    residuals=dict(primal_max_violation=max(violation),dual_stationarity_max_abs=max(map(abs,station)),
        dual_sign_max_violation=max(signs),complementary_max_abs=max(products),
        primal_dual_objective_gap_abs=abs(-q[3]-dual),true_self_finance_residual_abs=abs(funding),
        true_vs_pattern_fee_gap_abs=abs(fee_ratio-pattern_fee),worst_support_vs_z_gap_abs=abs(worst-q[3]))
    c.check(set(record['residuals'])==set(residuals)|{'solver_objective_gap_abs'},label+'/residual_schema')
    for name,value in residuals.items():c.value(record['residuals'][name],value,label+'/'+name)
    # raw scipy res.fun was not saved. This reported field can only be
    # checked for range, not independently reconstructed from the archive.
    rawgap=record['residuals']['solver_objective_gap_abs']
    c.check(isinstance(rawgap,(int,float)) and math.isfinite(rawgap) and rawgap>=0,label+'/reported_rawfun_gap_finite_nonnegative')
    c.reported_rawfun_checks+=1
    passed=all(v<=CERT_TOL for v in residuals.values()) and rawgap<=CERT_TOL
    c.value(record['certificate_passed'],passed,label+'/certificate')
    expected=dict(normalized_dual_objective=dual,normalized_primal_objective=-q[3],normalized_worst_support=worst,
        support_slacks=slack,equality_residual=[eq],dual_stationarity=station,posttrade_holdings=[q[j]*V for j in range(3)],
        nominal_fee=fee_ratio*V,worst_next_real=worst*R,dual_objective_maxmin_real=-dual*R,true_self_finance_residual=funding*V)
    for name,value in expected.items():c.value(record[name],value,label+'/'+name)
    return dict(record=record,recomputed=expected) if passed else None


def verify_case(case,c):
    result=case['result'];records=result['patterns']
    c.check(len(records)==8,'eight complete ordered regions');valid=[]
    for index,record in enumerate(records):
        checked=verify_pattern(case,record,index,c)
        if checked is not None:valid.append(checked)
    V=math.fsum(case['holdings']);R=V/case['cpi'];support=case['support_real_gross'];floor=case['budget_floor']
    # All pattern monetary values have been independently verified above.
    # Preserve the producer's exact floating tie ordering on those witnesses.
    best=max(valid,key=lambda z:(z['record']['worst_next_real'],-z['record']['pattern_index'])) if valid else None
    largest=max(z['record']['dual_objective_maxmin_real'] for z in valid) if valid else None
    all8=len(valid)==8;band=COMPARE_TOL*R
    if best is None:classification=LABELS[4]
    elif best['record']['worst_next_real']>floor+band:classification=LABELS[0]
    elif all8 and largest<floor-band:classification=LABELS[1]
    elif all8:classification=LABELS[2]
    else:classification=LABELS[3]
    independent_best=max((z['recomputed']['worst_next_real'] for z in valid),default=None)
    independent_dual=max((z['recomputed']['dual_objective_maxmin_real'] for z in valid),default=None)
    if independent_best is None:independent_classification=LABELS[4]
    elif independent_best>floor+band:independent_classification=LABELS[0]
    elif all8 and independent_dual<floor-band:independent_classification=LABELS[1]
    elif all8:independent_classification=LABELS[2]
    else:independent_classification=LABELS[3]
    c.check(independent_classification==classification,'classification also stable under independently recomputed values')
    hold=min(math.fsum(float(row[j])*case['holdings'][j] for j in range(3))/case['cpi'] for row in support)
    expected=dict(classification=classification,all_eight_numerically_resolved=all8,pattern_count=8,
        passed_certificate_count=len(valid),budget_floor=floor,current_real_wealth=R,comparison_band_real=band,
        hold_worst_next_real=hold,best_pattern_index=None if best is None else best['record']['pattern_index'],
        best_worst_next_real=None if best is None else best['record']['worst_next_real'],
        best_posttrade_holdings=None if best is None else best['record']['posttrade_holdings'],
        best_nominal_fee=None if best is None else best['record']['nominal_fee'],
        best_budget_margin_real=None if best is None else best['record']['worst_next_real']-floor,
        maximum_dual_objective_real=largest,exact_arithmetic_impossibility_proof=False,account_action_changed=False,
        future_or_recursive_feasibility_guarantee=False)
    c.check(set(result)==set(expected)|{'patterns'},'case result schema')
    for name,value in expected.items():c.value(result[name],value,'result/'+name)
    return expected


def original_states(root,c):
    inventory=[];states=[]
    for version,folder in PARENTS:
        groups=[g for g in readj(root/folder/'trace_groups.json') if g['kind'] in ('outer_selected','full_selected')]
        c.check(len(groups)==6 and len({g['group_id'] for g in groups})==6 and sum(g['kind']=='full_selected' for g in groups)==1,'all six selected groups/'+version)
        for scope in ('outer_selected','full_selected'):
            ids=[sid for g in groups if g['kind']==scope for sid in g['scenario_ids']]
            c.check(all(type(sid) is int for sid in ids) and sorted(ids)==list(range(1,121)),'original scope120IDs/'+version+'/'+scope)
        with np.load(root/folder/'traces.npz',allow_pickle=False) as archive:
            for g in groups:
                funded=bool(g['array_names']);count=0;prefix=g['group_id']+'__';n=len(g['scenario_ids'])
                if funded:
                    cache={}
                    def get(name,shape):
                        if name not in cache:cache[name]=archive[prefix+name]
                        v=cache[name];c.check(name in g['array_names'] and v.shape==shape,'source state/'+name);return v
                    first=get('first_guard_trigger',(n,));trigger=get('guard_triggered',(n,30))
                    c.check(first.dtype.kind in 'iu' and trigger.dtype.kind=='b','typed original guard')
                    positions=np.flatnonzero(first>=0);count=len(positions)
                    c.check(np.array_equal(trigger.sum(axis=1),first>=0),'one guard or none/source')
                    for i in positions:
                        t=int(first[i])-1;c.check(1<=t<30 and trigger[i,t],'original firstguard endpoint')
                        x=get('holdings',(n,31,3))[i,t];cp=float(get('cpi',(n,31))[i,t])
                        P=float(get('real_peak',(n,31))[i,t]);M=float(get('running_mdd',(n,31))[i,t])
                        floor=float(get('budget_floor',(n,30))[i,t]);rho=float(g['candidate']['release_fraction'])
                        c.check(not get('was_pending_input',(n,30))[i,t] and get('selected_index',(n,30))[i,t]==-1
                            and not get('support_feasible',(n,30))[i,t] and not get('candidate_feasible',(n,30,13))[i,t].any(),'original active failed13point gate')
                        tr=g['training_ids'];c.check(len(set(tr))==len(tr) and all(type(v) is int and 1<=v<=120 for v in tr),'own training IDs')
                        c.check(g['kind']=='full_selected' or not set(tr)&set(g['scenario_ids']),'outer own support')
                        support=get('training_real_gross',(30,len(tr),3))[t]
                        R=math.fsum(x.tolist())/cp;c.check(abs(floor-max(.6*P,R-rho*(R-.6*P)))<=1e-10*R and M-.4<=1e-12,'unchanged floor and history')
                        states.append(dict(version=version,scope=g['kind'],job_id=g['job_id'],group_id=g['group_id'],
                            scenario_id=int(g['scenario_ids'][i]),account_row=int(i),period=t+1,holdings=x.tolist(),cpi=cp,
                            real_peak=P,running_mdd=M,current_real_wealth=R,budget_floor=floor,release_fraction=rho,
                            cushion_scale=float(get('cushion_scale',(n,30))[i,t]) if version=='V3' else None,
                            training_ids=tr,support_real_gross=support.tolist(),primary_feasible_point_count=0,
                            primary_feasible_interval_count=int(get('interval_feasible',(n,30,4))[i,t].sum()),actual_future_used_in_LP=False))
                inventory.append(dict(version=version,scope=g['kind'],group_id=g['group_id'],job_id=g['job_id'],
                    scenario_ids=g['scenario_ids'],status=g.get('status'),physical_arm_present=funded,first_guard_count=count))
    return inventory,states


def evidence_closure(root=ROOT):
    """Metadata/hash-only assembly; never opens numeric JSON/CSV/NPZ bodies."""
    root=Path(root);m=readj(root/OUT/'manifest.json');f=readj(root/LP_FREEZE)
    snapshot={**f['source_sha256'],**f['input_sha256']}
    if (m['status']!='complete' or m['sources_and_inputs_at_start']!=snapshot or m['sources_and_inputs_at_end']!=snapshot
            or m['protocol_sha256']!=sha(root/LP_PLAN) or m['source_freeze_sha256']!=sha(root/LP_FREEZE)
            or f.get('execution_ready') is not True or f['protocol_sha256']!=sha(root/LP_PLAN)
            or m.get('final_scenarios_loaded') is not False or m.get('raw_market_values_parsed') is not False
            or m.get('account_action_changed') is not False or m.get('new_final_reads')!=0
            or set(m['outputs_sha256'])!=OUTPUTS):raise RuntimeError('Complete unchanged development-only LP producer required')
    outputs={str(OUT/name):digest for name,digest in m['outputs_sha256'].items()}
    verify_map(root,snapshot);verify_map(root,outputs)
    return {**snapshot,**outputs,str(OUT/'manifest.json'):sha(root/OUT/'manifest.json'),str(LP_PLAN):sha(root/LP_PLAN),str(LP_FREEZE):sha(root/LP_FREEZE)}


def preflight(root=ROOT):
    root=Path(root)
    if (root/ATTEMPT).exists() or (root/RECEIPT).exists():raise RuntimeError('Never overwrite or silently retry independent witness audit')
    p,f=readj(root/PLAN),readj(root/FREEZE)
    if (p.get('version')!='guard-physical-feasibility-independent-witness-audit-v1' or p.get('execution_ready') is not True
            or f.get('execution_ready') is not True or f.get('plan_sha256')!=sha(root/PLAN)
            or p.get('parent_manifest_sha256')!=sha(root/OUT/'manifest.json') or p.get('certificate_tolerance')!=CERT_TOL
            or p.get('comparison_tolerance')!=COMPARE_TOL or p.get('audit_atol')!=ATOL or p.get('audit_rtol')!=RTOL
            or p.get('allow_final') is not False or not SOURCES<=set(f.get('source_sha256',{}))):
        raise RuntimeError('Separately freeze exact independent saved-witness audit before numeric parse')
    verify_map(root,f['source_sha256']);verify_map(root,f['input_sha256'])
    closure=evidence_closure(root);frozen={**f['source_sha256'],**f['input_sha256']}
    if any(frozen.get(name)!=digest for name,digest in closure.items()):raise RuntimeError('Complete producer/parent transitive audit closure required')
    return p,f,frozen


def audit_values(root,c):
    with gzip.open(root/OUT/'case_inputs_and_lp_witnesses.json.gz','rt') as stream:cases=json.load(stream)
    inventory,states=original_states(root,c)
    c.value(readj(root/OUT/'selected_group_inventory.json'),inventory,'twelve inventory')
    c.check(len(cases)==len(states),'all original firstguard states retained')
    derived=[]
    for index,(case,state) in enumerate(zip(cases,states)):
        c.check(set(case)==set(state)|{'result'},'state schema')
        for key,value in state.items():c.value(case[key],value,f'state/{index}/{key}',exact=key in ('holdings','support_real_gross','cpi','real_peak','running_mdd','budget_floor'))
        derived.append(verify_case(case,c))
    rows=pd.read_csv(root/OUT/'guard_cases.csv.gz',float_precision='round_trip')
    c.check(list(rows.columns)==list(CSV_FIELDS) and len(rows)==len(cases),'all CSV rows/header')
    for i,(case,result) in enumerate(zip(cases,derived)):
        for key in CSV_FIELDS:
            expected=case.get(key,result.get(key));actual=rows.iloc[i][key]
            if expected is None:c.check(pd.isna(actual),'CSVNULL/'+key)
            else:c.value(actual,expected,'CSV/'+key)
    summaries=[]
    for version,_ in PARENTS:
        for scope in ('outer_selected','full_selected'):
            which=[i for i,case in enumerate(cases) if case['version']==version and case['scope']==scope]
            groups=[g for g in inventory if g['version']==version and g['scope']==scope]
            summaries.append(dict(version=version,scope=scope,original_n=120,group_count=len(groups),
                physical_path_n=sum(len(g['scenario_ids']) for g in groups if g['physical_arm_present']),
                first_guard_n=len(which),LP_count=8*len(which),
                classification_counts={name:sum(derived[i]['classification']==name for i in which) for name in LABELS},
                all_eight_resolved_n=sum(derived[i]['all_eight_numerically_resolved'] for i in which)))
    c.value(readj(root/OUT/'summary.json'),summaries,'four scope summaries')
    manifest=readj(root/OUT/'manifest.json')
    for key,value in dict(selected_group_count=12,first_guard_count=len(cases),LP_count=8*len(cases),
        exact_arithmetic_impossibility_proof=False,all_first_guards_preserved=True,new_policy_runs=0).items():c.value(manifest[key],value,'producer/'+key)
    return dict(case_count=len(cases),pattern_count=8*len(cases),selected_group_count=len(inventory),summary_rows=4,
        summaries=summaries,maximum_numeric_difference=c.maximum_numeric_difference,reported_rawfun_gap_checks=c.reported_rawfun_checks)


def run(root=ROOT):
    root=Path(root);p,f,before=preflight(root);(root/ATTEMPT).mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_saved_witness_parse',parent_manifest_sha256=sha(root/OUT/'manifest.json'),
        plan_sha256=sha(root/PLAN),source_freeze_sha256=sha(root/FREEZE),new_LP_solves=0,final_scenarios_loaded=False)
    writej(root/ATTEMPT/'attempt.json',state)
    try:
        c=Checks();counts=audit_values(root,c)
        after={name:sha(root/name) for name in before};c.check(after==before,'all original/frozen evidence unchanged')
        c.check(sha(root/PLAN)==state['plan_sha256'] and sha(root/FREEZE)==state['source_freeze_sha256'],'audit plan/sourcefreeze unchanged')
        receipt=dict(status='passed',version='guard-physical-feasibility-independent-witness-audit-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
            parent_manifest_sha256=state['parent_manifest_sha256'],plan_sha256=state['plan_sha256'],audit_source_freeze_sha256=state['source_freeze_sha256'],
            checks=c.n,**counts,sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,
            new_LP_solves=0,new_policy_runs=0,producer_modified=False,final_scenarios_loaded=False,exact_arithmetic_impossibility_proof=False,
            raw_solver_objective_reconstructed=False,
            raw_solver_objective_boundary='Original res.fun was not saved. solver_objective_gap_abs checked only finite/nonnegative/threshold; c@q and saved normalized_primal_objective, primal-dual gap and all physical/KKT quantities are independently recomputed.',
            boundary='Scalar saved-witness verification, not a repeated optimization, independent market validation, exact arithmetic proof or rule/parameter reselection. Whole saved state arrays are parsed for index extraction; future endpoints do not enter LP equations.')
        writej(root/RECEIPT,receipt);state.update(status='passed',phase='complete',receipt_sha256=sha(root/RECEIPT));writej(root/ATTEMPT/'attempt.json',state)
    except BaseException as exc:
        (root/ATTEMPT/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(root/ATTEMPT/'attempt.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,new_LP_solves=0,final_scenarios_loaded=False)))
