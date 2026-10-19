"""Prospectively frozen small-family execution diagnostic on reused paths.

Q1 job/context/refit components are immutable inputs. No Q1 optimization,
forecast, beta/eta selection, or old nine-price financial evidence is reused.
Architecture gates are separate and evaluated before all24 outer paths.
"""
from __future__ import annotations
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
import run_joint_liquidity_refit as joint
from safeinvest.data import load_paths
from safeinvest_research.metrics import discrete_cvar
from experimental_execution.simple_drift_execution import simulate_drift,DEFAULT_THRESHOLDS
from experimental_execution.review_pending import simulate_paired

PROTOCOL=ROOT/'docs/simple_execution_comparison_protocol.json'
OUT=ROOT/'results/model_review/simple_execution_comparison_v1'
LEARNERS=('annual_gated_diagnostic','drift_band_gated_diagnostic')
REFERENCES=('annual_then_pending','equal_thirds_annual')
FAMILIES={LEARNERS[0]:(0,),LEARNERS[1]:(0,1,2)}
THRESHOLDS=(0.,.05,.1)
ALPHA=.1;H=30;LOSS=.4;COST=.0025;TOL=1e-12
FINANCIAL=(*joint.FINANCIAL,'mean_fee_fraction')
PAIRINGS=tuple((a,b) for a in LEARNERS for b in REFERENCES)+((LEARNERS[1],LEARNERS[0]),)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_json_gz(path,value):Path(path).write_bytes(gzip.compress(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
def write_csv_gz(path,rows):Path(path).write_bytes(gzip.compress(pd.DataFrame(rows).to_csv(index=False,lineterminator='\n').encode(),mtime=0))
def read_json(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text())


def source_manifest():
    sources=joint.source_manifest()
    for rel in ('scripts/run_simple_execution_comparison.py','research/experimental_execution/simple_drift_execution.py',
                'tests/test_simple_execution_pipeline.py','tests/test_simple_drift_execution.py','requirements-research-pinned.txt'):
        sources[rel]=sha(ROOT/rel)
    sources.pop(str(PROTOCOL.relative_to(ROOT)),None)
    return dict(sorted(sources.items()))


def frozen_inputs(protocol):
    values={rel:sha(ROOT/rel) for rel in protocol['frozen_input_sha256']}
    if values!=protocol['frozen_input_sha256']:raise RuntimeError('A frozen Q1/source input changed')
    return values


def validate_protocol(p,parent=None):
    if (p['data']['scenario_ids']!=list(range(1,121)) or p['data']['allow_final'] is not False
        or p['architectures']!={LEARNERS[0]:[0.],LEARNERS[1]:[0.,.05,.1]}
        or p['candidate_orders']!={'0':0.,'1':.05,'2':.1}
        or p['service_alpha']!=ALPHA or p['original_H']!=H or p['loss_tolerance']!=LOSS or p['cost_rate']!=COST
        or p['liquidity_floors']!=joint.LIQUIDITY_FLOORS
        or p['client']!='C' or p['repeat_count']!=5 or p['outer_fits']!=25 or p['q2_training_tasks']!=125
        or p['initial_nominal_wealth']!=100 or p['initial_weights']!=[0,0,1]
        or p['references']!=list(REFERENCES) or tuple(DEFAULT_THRESHOLDS)!=THRESHOLDS):
        raise ValueError('Predeclared simple-family definition changed')
    if parent is not None:
        for key in ('contexts','jobs','static_contexts'):
            if p[key]!=parent[key]:raise ValueError('Immutable parent whole-path identities changed')
    if len(p['contexts'])!=25 or len(p['jobs'])!=125:raise ValueError('Require25 complete contexts/125 jobs')
    expected=[]
    for c in p['contexts']:
        cid=c['context_id'];group=[j for j in p['jobs'] if j['parent_context_id']==cid]
        if [j['job_id'] for j in group]!=[*[f'c{cid}_inner{i}' for i in range(1,5)],f'c{cid}_refit']:
            raise ValueError('Four inner jobs must precede each final refit')
        if len(c['training_ids'])!=96 or len(c['outer_test_ids'])!=24 or set(c['training_ids'])&set(c['outer_test_ids']):raise ValueError('Original96/24 parent split')
        inner=[]
        for job in group:
            if job['stage']=='inner':
                if len(job['training_ids'])!=72 or len(job['evaluation_ids'])!=24:raise ValueError('Original72/24 inner split')
                if set(job['training_ids'])&set(job['evaluation_ids']) or set(job['training_ids'])|set(job['evaluation_ids'])!=set(c['training_ids']):raise ValueError('Inner complementary original IDs')
                inner.extend(job['evaluation_ids'])
            elif job['training_ids']!=c['training_ids'] or job['evaluation_ids']!=c['outer_test_ids']:raise ValueError('Final identity differs')
        if sorted(inner)!=sorted(c['training_ids']):raise ValueError('Each parent training path must enter inner validation once')
        expected.extend(j['job_id'] for j in group)
    if expected!=[j['job_id'] for j in p['jobs']]:raise ValueError('Complete job order changed')
    for repeat in range(1,6):
        selected=[c for c in p['contexts'] if c['repeat_id']==repeat]
        if len(selected)!=5 or sorted(s for c in selected for s in c['outer_test_ids'])!=list(range(1,121)):raise ValueError('Original120 heldout once per repeat')


def restrict_q1_bundles(p,bundles):
    result={}
    if set(bundles)!=set(j['job_id'] for j in p['jobs']):raise ValueError('Need exactly125 matching Q1 component jobs')
    for job in p['jobs']:
        jid=job['job_id'];b=bundles[jid]
        # Intentionally do not inspect or reuse old training_price_selections,
        # reference_training_breach_count or any nine-price financial metrics.
        q={key:b[key] for key in ('q2_case','context','refit')}
        if q['context']!=p['static_contexts'][jid]:raise ValueError('Q1 job static context identity differs')
        if q['refit']['training_ids']!=job['training_ids'] or q['refit']['outer_test_ids']!=job['evaluation_ids']:raise ValueError('Q1 refit training/evaluation identity differs')
        if q['q2_case']['effective_assignment']!=q['refit']['effective_assignments']['C']:raise ValueError('Q1 effective C label differs')
        if q['q2_case']['public_status']!=joint.case_of(q['q2_case']['effective_assignment']):raise ValueError('Q1 public case differs')
        label=q['q2_case']['effective_assignment'];raw=q['refit'].get('raw_fit')
        menu=raw['menu'] if raw else []
        matches=[product for product in menu if product['name']==label]
        if label!='Review' and len(matches)!=1:raise ValueError('Effective Q1 label needs one recorded menu product')
        expected=matches[0] if label!='Review' else None
        def same_product(actual,wanted):
            if wanted is None:return actual is None
            if not isinstance(actual,dict):return False
            keys=('name','weights','candidate_index','candidate_original232_index')
            if any(key not in actual or key not in wanted or actual[key]!=wanted[key] for key in keys):return False
            if 'candidate_local_index' in actual or 'candidate_local_index' in wanted:
                if actual.get('candidate_local_index')!=wanted.get('candidate_local_index'):return False
            return True
        if not same_product(q['q2_case'].get('q1_product'),expected):raise ValueError('Q1 cached product weights/local/global IDs differ from effective menu product')
        if not same_product(q['q2_case'].get('product'),expected if label=='Balanced' else None):raise ValueError('Q2 cached target weights/local/global IDs differ from effective Balanced menu product')
        result[jid]=q
    return result


def choose_family(candidates,kB,n,orders):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:raise ValueError('Positive training denominator required')
    if isinstance(kB,bool) or not isinstance(kB,(int,np.integer)) or not 0<=kB<=n:raise ValueError('Reference count must be an integer')
    if tuple(orders) not in tuple(FAMILIES.values()):raise ValueError('One predeclared architecture family required')
    if [r['execution_candidate_order'] for r in candidates]!=[0,1,2]:raise ValueError('All3 training summaries required in canonical order')
    cap=min(int(kB),math.floor(ALPHA*n+TOL));classified=[]
    for row in candidates:
        k=row['breach_count']
        if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<=n or row['n_training']!=n:raise ValueError('Complete-path integer training evidence required')
        for key in ('mean_log_growth','mean_excess','mean_nominal_fees'):
            if isinstance(row[key],bool) or not np.isfinite(row[key]):raise ValueError('Finite candidate metric required')
        if row['mean_excess']<0 or row['mean_nominal_fees']<0:raise ValueError('Severity and fees cannot be negative')
        if row['execution_candidate_order'] in orders:classified.append({**row,'training_eligible':k<=cap})
    if candidates[0]['breach_count']!=kB:raise ValueError('kB must come from the newly simulated threshold0 training path')
    eligible=[r for r in classified if r['training_eligible']]
    maximum=max((r['mean_log_growth'] for r in eligible),default=None)
    tied=[r for r in eligible if r['mean_log_growth']>=maximum-TOL] if eligible else []
    chosen=min(tied,key=lambda r:(r['mean_excess'],r['mean_nominal_fees'],r['execution_candidate_order'])) if tied else None
    return dict(selection_performed=True,n_training=int(n),reference_training_breach_count=int(kB),training_breach_cap=cap,
        family_candidate_orders=list(orders),candidate_summaries=classified,maximum_eligible_mean_log_growth=maximum,
        global_growth_tied_candidate_orders=[r['execution_candidate_order'] for r in tied],
        selected_candidate_order=chosen['execution_candidate_order'] if chosen else None,
        selected_threshold=chosen['tv_threshold'] if chosen else None,
        execution_status='automated' if chosen else 'execution_review',
        execution_reason=None if chosen else 'no_admissible_training_candidate')


def common_fields(job,bundle):
    return {**joint.common_fields(job,bundle),'repeat_id':job['repeat_id'],'outer_fold':job['outer_fold']}

def financial_rows(arm,ids,metadata):
    rows=joint.arm_records(arm,ids,metadata)
    for i,row in enumerate(rows):row['mean_fee_fraction']=float(np.mean(arm['fee_fraction'][i]))
    return rows

def null_rows(ids,metadata,status,reason):
    rows=joint.null_records(ids,metadata,status,reason)
    for row in rows:row['mean_fee_fraction']=None
    return rows


def training_summary(rows,order,n):
    frame=pd.DataFrame(rows);observed=frame.loc[frame.assigned]
    result=dict(execution_candidate_order=order,candidate_order=order,tv_threshold=THRESHOLDS[order],delta=THRESHOLDS[order],n_training=n,
                case_status=str(frame.case_status.iloc[0]),simulation_status='evaluated' if len(observed) else str(frame.execution_status.iloc[0]))
    if not len(observed):return {**result,**dict.fromkeys(('breach_count','mean_log_growth','mean_excess','mean_nominal_fees','mean_nominal_moved','mean_trade_fraction','mean_fee_fraction'))}
    if len(observed)!=n:raise ValueError('Training evidence cannot drop paths')
    return {**result,'breach_count':int(observed.breach.sum()),
        **{out:float(observed[col].mean()) for out,col in (('mean_log_growth','log_growth'),('mean_excess','excess'),
            ('mean_nominal_fees','nominal_fees'),('mean_nominal_moved','nominal_moved'),('mean_trade_fraction','trade_fraction'),('mean_fee_fraction','mean_fee_fraction'))}}


def gate_inner(rows,baseline_rows,expected_ids):
    if sorted(r['scenario_id'] for r in rows)!=sorted(expected_ids) or sorted(r['scenario_id'] for r in baseline_rows)!=sorted(expected_ids):raise ValueError('Inner gate needs every original parent training identity once')
    if len(set(expected_ids))!=len(expected_ids):raise ValueError('Original inner identities cannot repeat')
    for row in (*rows,*baseline_rows):
        if not isinstance(row['assigned'],(bool,np.bool_)):raise ValueError('Boolean admission/availability required')
    bmap={r['scenario_id']:r for r in baseline_rows};auto=[r for r in rows if r['assigned']]
    for row in auto:
        ref=bmap[row['scenario_id']]
        if not ref['assigned'] or ref['job_id']!=row['job_id']:raise ValueError('B must be available on the same inner-fit heldout path')
        if not isinstance(row['breach'],(bool,np.bool_)) or not isinstance(ref['breach'],(bool,np.bool_)):raise ValueError('Observed event booleans required')
    n=len(auto);k=sum(int(r['breach']) for r in auto) if n else None
    kb=sum(int(bmap[r['scenario_id']]['breach']) for r in auto) if n else None
    cap=math.floor(ALPHA*n+TOL) if n else None
    reasons=[]
    if not n:reasons.append('no_inner_auto_evidence')
    else:
        if k>cap:reasons.append('inner_absolute_risk_exceeds_reference')
        if k>kb:reasons.append('inner_risk_exceeds_same_U_B')
    case_counts={key:sum(r['case_status']==key for r in rows) for key in ('balanced_case','q1_review','not_balanced_case')}
    return dict(original_n=len(rows),auto_set_scenario_ids=sorted(r['scenario_id'] for r in auto),
        auto_inner_n=n,balanced_case_n=case_counts['balanced_case'],q1_review_n=case_counts['q1_review'],not_balanced_case_n=case_counts['not_balanced_case'],
        inner_execution_review_n=sum(r['execution_status']=='execution_review' for r in rows),
        selected_breach_count=k,same_U_B_breach_count=kb,absolute_alpha_count_cap=cap,
        conditional_breach_probability=k/n if n else None,conditional_B_breach_probability=kb/n if n else None,
        conditional_mean_log_growth=float(np.mean([r['log_growth'] for r in auto])) if n else None,
        eligible=not reasons,reason='passed_inner_gate' if not reasons else reasons[0],reasons=reasons,
        gate_scope='Own architecture inner selected paths; B on identical U and each own inner target; no architecture search')


def performance_summary(rows):
    frame=pd.DataFrame(rows);out=[]
    for policy,g in frame.groupby('policy',sort=False):
        reference=policy in REFERENCES;auto=g.loc[g.assigned];n=len(auto)
        if g.scenario_id.duplicated().any():raise ValueError('One original scenario per summary')
        if not g.loc[~g.assigned,list(FINANCIAL)].isna().all().all():raise ValueError('Review or unavailable outcomes must be NULL')
        if not reference and not g.assigned.eq(g.execution_status.eq('automated')).all():raise ValueError('Automatic execution/availability mismatch')
        cases={k:int(g.case_status.eq(k).sum()) for k in ('balanced_case','q1_review','not_balanced_case')}
        row=dict(policy=policy,scope='outer_oof',original_n=len(g),balanced_case_n=cases['balanced_case'],q1_review_n=cases['q1_review'],not_balanced_case_n=cases['not_balanced_case'],
            case_auto_n=None if reference else int(g.execution_status.eq('automated').sum()),
            case_execReview_n=None if reference else int(g.execution_status.eq('execution_review').sum()),
            assigned_paths=n,financial_null_paths=len(g)-n,reference_only=reference,
            reference_available_n=n if reference else None,
            case_reference_available_n=int((g.assigned & g.case_status.eq('balanced_case')).sum()) if reference else None)
        if not reference and row['balanced_case_n']!=row['case_auto_n']+row['case_execReview_n']:raise ValueError('Balanced service denominator identity')
        k=int(auto.breach.sum()) if n else None
        row.update(breach_count=k,conditional_breach_probability=k/n if n else None,breaches_per_original_n=k/len(g) if n else None,
            mean_log_growth=float(auto.log_growth.mean()) if n else None,median_cagr=float(auto.cagr.median()) if n else None,
            p10_cagr=float(np.quantile(auto.cagr,.1,method='inverted_cdf')) if n else None,
            mean_mdd=float(auto.mdd.mean()) if n else None,median_mdd=float(auto.mdd.median()) if n else None,
            p90_mdd=float(np.quantile(auto.mdd,.9,method='inverted_cdf')) if n else None,
            cvar90_mdd=float(discrete_cvar(auto.mdd.to_numpy())) if n else None,
            **{name:float(auto[field].mean()) if n else None for name,field in (('mean_nominal_fees','nominal_fees'),('mean_trade_fraction','trade_fraction'),('mean_excess','excess'),('mean_fee_fraction','mean_fee_fraction'))})
        out.append(row)
    return out


def paired_outputs(rows):
    frame=pd.DataFrame(rows);pairs=[];summary=[]
    for left,right in PAIRINGS:
        a=frame.loc[frame.policy.eq(left)];b=frame.loc[frame.policy.eq(right)]
        merged=a.merge(b,on=['repeat_id','outer_fold','context','scenario_id'],suffixes=('_left','_right'),validate='one_to_one')
        if len(merged)!=120:raise ValueError('Keep120 original paired identities per repeat')
        current=[]
        for row in merged.to_dict('records'):
            common=bool(row['assigned_left'] and row['assigned_right'])
            current.append(dict(repeat_id=int(row['repeat_id']),outer_fold=int(row['outer_fold']),context=row['context'],scenario_id=int(row['scenario_id']),
                learner_policy=left,reference_policy=right,reference_is_hypothetical=right in REFERENCES,
                learner_auto=bool(row['assigned_left']),reference_available=bool(row['assigned_right']),common_auto=common,
                **{'delta_'+f:float(row[f+'_left'])-float(row[f+'_right']) if common else None for f in ('log_growth','cagr','mdd','excess','nominal_fees','trade_fraction','breach','mean_fee_fraction')}))
        pairs.extend(current);g=pd.DataFrame(current);both=g.loc[g.common_auto]
        summary.append(dict(learner_policy=left,reference_policy=right,reference_is_hypothetical=right in REFERENCES,original_pairs=120,
            learner_auto_n=int(g.learner_auto.sum()),reference_available_n=int(g.reference_available.sum()),common_auto_n=len(both),
            **{'mean_delta_'+f:float(both['delta_'+f].mean()) if len(both) else None for f in ('log_growth','cagr','mdd','excess','nominal_fees','trade_fraction','breach','mean_fee_fraction')}))
    return pairs,summary


def save_trace(archive,groups,key,arm,ids,metadata,*,kind,order=None):
    for name,value in arm.items():
        if isinstance(value,np.ndarray):archive[key+'__'+name]=value
    groups[key]=dict(scenario_ids=list(ids),metadata=metadata,kind=kind,
        selected_candidate_order=order,tv_threshold=THRESHOLDS[order] if order is not None else None,
        diagnostic_scope='Physical simple-execution path; no forecast/Q/price metadata')


def run_pipeline(p,bundles,returns,inflation,ids,*,simulator=simulate_drift,annual_simulator=simulate_paired,guard=None,on_job=None):
    validate_protocol(p);q1=restrict_q1_bundles(p,bundles)
    if list(map(int,ids))!=list(range(1,121)) or np.shape(returns)!=(120,30,3) or np.shape(inflation)!=(120,30):raise ValueError('Original120 complete paths required')
    training=[];summaries=[];inner=[];inner_B=[];gates=[];clients=[];admission=[];fit_summ=[]
    train_choices={};context_choices={};actual={};validation={};groups={'actual':{},'validation':{}};completed=[]
    for c in p['contexts']:
        name=c['name'];context_clients=[]
        for job in [j for j in p['jobs'] if j['parent_context_id']==c['context_id']]:
            if guard:guard()
            tick=time.perf_counter();jid=job['job_id'];b=q1[jid];common=common_fields(job,b)
            case=common['case_status'];tr=np.asarray(job['training_ids'])-1;ev=np.asarray(job['evaluation_ids'])-1
            target=np.asarray(b['q2_case']['product']['weights']) if case=='balanced_case' else None
            stats=[];kB=None
            # All3 complete training metrics are retained, but never full
            # training state trees and never any old dynamic financial rows.
            for order,threshold in enumerate(THRESHOLDS):
                meta={**common,'execution_candidate_order':order,'candidate_order':order,'tv_threshold':threshold,'delta':threshold,'scope':'training_support'}
                if case=='balanced_case':
                    arm=simulator(returns[tr],inflation[tr],target,threshold,L=LOSS,horizon=H,cost_rate=COST)
                    rows=financial_rows(arm,job['training_ids'],meta)
                else:rows=null_rows(job['training_ids'],meta,case,case)
                training.extend(rows);s={**common,**training_summary(rows,order,len(tr))};summaries.append(s);stats.append(s)
            if case=='balanced_case':kB=stats[0]['breach_count']
            parent_gates={}
            if job['stage']=='refit':
                for policy in LEARNERS:
                    gate=gate_inner([r for r in inner if r['context']==name and r['policy']==policy],[r for r in inner_B if r['context']==name],c['training_ids'])
                    parent_gates[policy]=gate;gates.append(dict(context=name,context_id=c['context_id'],repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],policy=policy,**gate))
            choices={}
            for policy,orders in FAMILIES.items():
                if case!='balanced_case':choice=dict(selection_performed=False,family_candidate_orders=list(orders),selected_candidate_order=None,selected_threshold=None,execution_status=case,execution_reason=case)
                elif job['stage']=='refit' and not parent_gates[policy]['eligible']:
                    choice=dict(selection_performed=False,family_candidate_orders=list(orders),selected_candidate_order=None,selected_threshold=None,execution_status='execution_review',execution_reason=parent_gates[policy]['reason'])
                else:choice=choose_family(stats,kB,len(tr),orders)
                choices[policy]=choice
            train_choices[jid]=dict(job_id=jid,training_ids=job['training_ids'],evaluation_ids=job['evaluation_ids'],
                q1_effective_assignment=common['q1_effective_assignment'],case_status=case,reference_training_breach_count=kB,methods=choices)
            if job['stage']=='inner':
                chosen={v['selected_candidate_order'] for v in choices.values() if v['execution_status']=='automated'}
                orders=sorted(chosen|({0} if case=='balanced_case' else set()))
                arms={order:simulator(returns[ev],inflation[ev],target,THRESHOLDS[order],L=LOSS,horizon=H,cost_rate=COST) for order in orders}
                for policy,choice in choices.items():
                    order=choice['selected_candidate_order'];meta={**common,'policy':policy,'scope':'inner_validation','selected_candidate_order':order,'candidate_order':order,'tv_threshold':THRESHOLDS[order] if order is not None else None,'delta':THRESHOLDS[order] if order is not None else None,'reference_only':False}
                    rows=financial_rows(arms[order],job['evaluation_ids'],meta) if choice['execution_status']=='automated' else null_rows(job['evaluation_ids'],meta,choice['execution_status'],choice['execution_reason'])
                    inner.extend(rows)
                    if choice['execution_status']=='automated':save_trace(validation,groups['validation'],jid+'__'+policy,arms[order],job['evaluation_ids'],meta,kind='selected_inner',order=order)
                bm={**common,'policy':REFERENCES[0],'scope':'inner_validation_reference','reference_only':True,'selected_candidate_order':None,'candidate_order':0,'tv_threshold':0.,'delta':0.}
                if case=='balanced_case':
                    inner_B.extend(financial_rows(arms[0],job['evaluation_ids'],bm));save_trace(validation,groups['validation'],jid+'__inner_B',arms[0],job['evaluation_ids'],bm,kind='inner_B',order=0)
                else:inner_B.extend(null_rows(job['evaluation_ids'],bm,'reference_unavailable',case))
            else:
                chosen={v['selected_candidate_order'] for v in choices.values() if v['execution_status']=='automated'}
                orders=sorted(chosen|({0} if case=='balanced_case' else set()))
                arms={order:simulator(returns[ev],inflation[ev],target,THRESHOLDS[order],L=LOSS,horizon=H,cost_rate=COST) for order in orders}
                for policy,choice in choices.items():
                    order=choice['selected_candidate_order'];meta={**common,'policy':policy,'scope':'outer_oof','selected_candidate_order':order,'candidate_order':order,'tv_threshold':THRESHOLDS[order] if order is not None else None,'delta':THRESHOLDS[order] if order is not None else None,'reference_only':False}
                    rows=financial_rows(arms[order],job['evaluation_ids'],meta) if choice['execution_status']=='automated' else null_rows(job['evaluation_ids'],meta,choice['execution_status'],choice['execution_reason'])
                    context_clients.extend(rows)
                    if choice['execution_status']=='automated':save_trace(actual,groups['actual'],jid+'__'+policy,arms[order],job['evaluation_ids'],meta,kind='selected_outer',order=order)
                    admission.append(dict(context=name,context_id=c['context_id'],repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],policy=policy,training_n=len(tr),evaluation_n=len(ev),
                        q1_effective_assignment=common['q1_effective_assignment'],case_status=case,initial_admitted=choice['execution_status']=='automated',inner_gate_passed=parent_gates[policy]['eligible'],
                        selected_candidate_order=order,selected_threshold=choice['selected_threshold'],execution_status=choice['execution_status'],execution_reason=choice['execution_reason']))
                context_choices[name]=dict(training_ids=job['training_ids'],evaluation_ids=job['evaluation_ids'],repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],case=b['q2_case'],
                    static_gamma=common['static_gamma'],inner_gates=parent_gates,methods=choices)
                bm={**common,'policy':REFERENCES[0],'scope':'outer_oof','reference_only':True,'selected_candidate_order':None,'candidate_order':0,'tv_threshold':0.,'delta':0.}
                if case=='balanced_case':
                    context_clients.extend(financial_rows(arms[0],job['evaluation_ids'],bm));save_trace(actual,groups['actual'],jid+'__B',arms[0],job['evaluation_ids'],bm,kind='B_reference',order=0)
                else:context_clients.extend(null_rows(job['evaluation_ids'],bm,'reference_unavailable',case))
                eq=joint.reference.squeeze_pending(annual_simulator(returns[ev],inflation[ev],np.ones(3)/3,LOSS,horizon=H,cost_rate=COST)['annual'])
                eq['targets']=np.ones(3)/3
                em={**common,'policy':REFERENCES[1],'scope':'outer_oof','reference_only':True,'selected_candidate_order':None,'candidate_order':None,'tv_threshold':None,'delta':None,
                    'candidate_local_index':None,'candidate_original232_index':None,'execution_target_type':'exact_thirds_always_annual_reference'}
                context_clients.extend(financial_rows(eq,job['evaluation_ids'],em));save_trace(actual,groups['actual'],jid+'__equal',eq,job['evaluation_ids'],em,kind='equal_thirds_annual')
            if guard:guard()
            entry=dict(job_id=jid,parent_context=name,repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],stage=job['stage'],training_n=len(tr),evaluation_n=len(ev),case_status=case,seconds=time.perf_counter()-tick)
            completed.append(entry)
            if on_job:on_job(entry)
        clients.extend(context_clients);fit_summ.extend(dict(repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],context=name,**r) for r in performance_summary(context_clients))
    repeat_summ=[];pairs=[];paired=[];scenario=[]
    for repeat in range(1,6):
        rows=[r for r in clients if r['repeat_id']==repeat]
        repeat_summ.extend(dict(repeat_id=repeat,**r) for r in performance_summary(rows))
        pr,ps=paired_outputs(rows);pairs.extend(pr);paired.extend(dict(repeat_id=repeat,**r) for r in ps)
    frame=pd.DataFrame(clients)
    for (policy,sid),g in frame.loc[frame.policy.isin(LEARNERS)].groupby(['policy','scenario_id'],sort=True):
        if len(g)!=5 or set(g.repeat_id)!=set(range(1,6)):raise ValueError('Original path must appear once in each repeat')
        obs=g.loc[g.assigned]
        scenario.append(dict(policy=policy,scenario_id=int(sid),repeat_records_n=5,admitted_refits_n=int(g.assigned.sum()),breached_among_admitted_refits_n=int(obs.breach.sum()) if len(obs) else None,
            min_log_growth=float(obs.log_growth.min()) if len(obs) else None,max_log_growth=float(obs.log_growth.max()) if len(obs) else None,
            interpretation='Five heldout fits of same original path; admission perfit before its24 returns, not path screening'))
    return dict(tables=dict(training_paths=training,training_summary=summaries,inner_validation_paths=inner,inner_reference_paths=inner_B,
        inner_gate_summaries=gates,client_paths=clients,fit_admission=admission,fit_summary=fit_summ,summary=repeat_summ,paired_paths=pairs,paired_summary=paired,per_scenario_stability=scenario),
        training_choices=train_choices,context_selections=context_choices,trace_groups=groups,actual_traces=actual,validation_traces=validation,completed_jobs=completed)


def main():
    p=read_json(PROTOCOL)
    if p.get('execution_ready') is not True:raise RuntimeError('Draft study cannot execute before source/protocol review and freeze')
    if OUT.exists():raise RuntimeError('Never overwrite a complete or partial archive')
    parent=read_json(ROOT/'docs/refit_stability_protocol.json');validate_protocol(p,parent)
    source=source_manifest();inputs=frozen_inputs(p);protocol_sha=sha(PROTOCOL)
    if source!=p['execution_sources_sha256']:raise RuntimeError('Execution source snapshot differs')
    parent_manifest=read_json(ROOT/'results/model_review/refit_stability_v1/manifest.json')
    if parent_manifest['status']!='complete' or read_json(ROOT/'results/model_review/refit_stability_v1/independent_audit.json')['status']!='passed':raise RuntimeError('Complete audited Q1 component cache required')
    bundles=read_json(ROOT/p['q1_component_cache']['path'])
    restrict_q1_bundles(p,bundles)
    returns,inflation,ids=load_paths(ROOT/p['data']['path'],p['data']['scenario_ids'],allow_final=False)
    digest=hashlib.sha256()
    for a in (ids,returns,inflation):digest.update(np.ascontiguousarray(a).tobytes())
    if digest.hexdigest()!=parent_manifest['development_arrays_sha256']:raise RuntimeError('Original development array digest differs')
    def guard():
        if source_manifest()!=source or frozen_inputs(p)!=inputs or sha(PROTOCOL)!=protocol_sha:raise RuntimeError('Frozen source/input/protocol changed during job')
    guard();OUT.mkdir(parents=True);started=time.perf_counter();completed=[]
    def on_job(row):
        completed.append(row);write_json(OUT/'progress.json',dict(completed_jobs=completed,final_scenarios_loaded=False))
        print(json.dumps(row),flush=True)
        if time.perf_counter()-started>60*p['budget_numeric_minutes']:raise RuntimeError('Numeric budget exhausted: preserve incomplete archive')
    result=run_pipeline(p,bundles,returns,inflation,ids,simulator=simulate_drift,annual_simulator=simulate_paired,guard=guard,on_job=on_job)
    if len(result['completed_jobs'])!=125:raise RuntimeError('All125 jobs required')
    for name,n in p['expected_rows'].items():
        if len(result['tables'][name])!=n:raise RuntimeError('Required complete row inventory differs: '+name)
    for name,rows in result['tables'].items():write_csv_gz(OUT/(name+'.csv.gz'),rows)
    write_json_gz(OUT/'training_choices.json.gz',result['training_choices']);write_json(OUT/'context_selections.json',result['context_selections']);write_json(OUT/'trace_groups.json',result['trace_groups'])
    np.savez_compressed(OUT/'actual_traces.npz',**result['actual_traces']);np.savez_compressed(OUT/'validation_traces.npz',**result['validation_traces'])
    guard()
    write_json(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        protocol_sha256=protocol_sha,sources=source,sources_at_end=source_manifest(),input_sha256=inputs,input_sha256_at_end=frozen_inputs(p),
        outputs_sha256={f.name:sha(f) for f in OUT.iterdir() if f.is_file()},scenario_ids_loaded=ids.tolist(),final_scenarios_loaded=False,
        development_arrays_sha256=digest.hexdigest(),training_tasks=125,outer_fits=25,repeat_count=5,candidate_count=3,
        architectures=p['architectures'],row_counts={k:len(v) for k,v in result['tables'].items()},
        limitations=['Simplified diagnostic, not final Q2 benefit-cost strategy or causal forecast ablation','B inclusion creates imposed training-feasibility/growth properties, not future safety evidence',
            'Reused original120 paths under5 repeats; not600 independent observations or25 independent deployment trials','Every architecture uses own inner gate; Review and outside-Balanced financial outcomes stayNULL; references hypothetical',
            'Only selected actual/inner and B physical states saved; all3 training metrics independently auditable; no Q/forecast values exist']))
    print(json.dumps(dict(status='complete',output=str(OUT.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
