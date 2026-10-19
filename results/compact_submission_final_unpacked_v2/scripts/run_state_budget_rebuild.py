"""Development-only complete-policy evaluation of a frozen small rebuild.

No final values are loaded. This new exploratory research preserves the old
model and final archive. Reused whole-path folds give mechanical separation,
not renewed independence from the long public development history.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research')]
from safeinvest.data import load_paths
from safeinvest_research.metrics import path_metrics
from experimental_execution.state_budget_selection import CANDIDATES,select_inner,summarize_records
from experimental_execution.review_pending import simulate_paired

PROTOCOL=ROOT/'docs/state_budget_rebuild_protocol.json'
FREEZE=ROOT/'docs/state_budget_rebuild_source_freeze.json'
DEFAULT_OUT=ROOT/'results/state_budget_rebuild_v1'
H,L,COST=30,.4,.0025
REFS=('same_target_B','same_target_annual','exact_thirds_annual')
FINANCIAL=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees',
           'trade_fraction','breach','first_crossing','automatic_fraction',
           'protection_trade_fraction','support_review')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return jsonable(x.tolist())
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def writej(path,x):Path(path).write_text(json.dumps(jsonable(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def validate_job(job,bundle):
    train,test=job['training_ids'],job['evaluation_ids']
    for ids in (train,test):
        if not ids or len(set(ids))!=len(ids) or any(type(i) is not int or not 1<=i<=120 for i in ids):
            raise ValueError('Only complete development IDs1-120 may enter jobs')
    if job['stage']=='inner' and set(train)&set(test):raise ValueError('Inner actual/training IDs overlap')
    if bundle['refit']['training_ids']!=train:raise ValueError('Q1 target fitted on different training identities')
    case=bundle['q2_case']
    if not case['in_balanced_case']:return None
    w=np.array(case['product']['weights'],float)
    if w.shape!=(3,) or np.any(w<=0) or not np.isclose(w.sum(),1,atol=1e-12,rtol=0):
        raise ValueError('A Balanced target needs all three positive assets')
    return w


def rows_from_arm(arm,ids,common):
    metric=path_metrics(arm['real_wealth'])
    if len(ids)!=len(arm['real_wealth']):raise ValueError('Arm/identity mismatch')
    out=[]
    for i,sid in enumerate(ids):
        pending=arm['pending_at_decision'][i]
        reasons=np.asarray(arm.get('gate_reason',np.full((len(ids),H),'')))[i]
        support_review=bool(np.any(np.asarray(arm.get('guard_pending',np.zeros((len(ids),H+1),bool)))[i]))
        guard_triggered=np.asarray(arm.get('guard_triggered',np.zeros((len(ids),H),bool)))[i]
        first=int(arm['first_crossing'][i])
        out.append(dict(**common,scenario_id=int(sid),assigned=True,
            **{k:float(v[i]) for k,v in metric.items()},
            nominal_fees=float(arm['fees'][i].sum()),trade_fraction=float(arm['trades'][i].mean()),
            breach=bool(arm['ever_breached'][i,-1]),first_crossing=first if first>=0 else None,
            automatic_fraction=float(1-np.mean(pending|guard_triggered)),
            protection_trade_fraction=float(np.mean(reasons=='risk_budget_protection')),
            support_review=support_review))
    return out


def null_rows(ids,common,reason):
    return [dict(**common,scenario_id=int(s),assigned=False,review_reason=reason,
                 **dict.fromkeys(FINANCIAL)) for s in ids]


def squeeze(arm,order=0):
    result={}
    for key,value in arm.items():
        if not isinstance(value,np.ndarray):continue
        if key=='cpi':result[key]=value
        elif key in ('targets','loss_tolerances'):continue
        elif value.ndim>=2:result[key]=value[:,order]
    return result


def archive_arrays(arm,kind):
    """Keep every physical tree; bound duplicate inner-menu diagnostic size.

    Full selected-arm candidate menus remain available for the fixed decision
    audit. Inner policies retain all physical states and selected gate fields;
    their per-alternative menus and shared training copies are not archived.
    """
    arrays={};omitted=[]
    if arm is None:return arrays,omitted
    for key,value in arm.items():
        if not isinstance(value,np.ndarray):continue
        if kind=='inner_candidate' and (key.startswith('candidate_') or key in ('training_previous_state','training_real_gross')):
            omitted.append(key)
        else:arrays[key]=value
    return arrays,omitted


def validate_jobs(jobs):
    if len(jobs)!=30 or len({j['job_id'] for j in jobs})!=30:raise RuntimeError('Require all30 fixed complete jobs')
    for j in jobs:
        for ids in (j['training_ids'],j['evaluation_ids']):
            if not ids or len(ids)!=len(set(ids)) or any(type(i) is not int or not 1<=i<=120 for i in ids):
                raise RuntimeError('Only unique integer development IDs1-120 may enter any job')
    contexts=sorted({j['parent_context_id'] for j in jobs})
    if contexts!=list(range(6)):raise RuntimeError('Require full120 plus five24 outer contexts')
    outer_ids=[]
    for c in contexts:
        js=[j for j in jobs if j['parent_context_id']==c]
        inner=[j for j in js if j['stage']=='inner']; refit=[j for j in js if j['stage']=='refit']
        if len(inner)!=4 or len(refit)!=1:raise RuntimeError('Each context has four inner folds and one refit')
        if c==0:
            if refit[0]['training_ids']!=list(range(1,121)) or refit[0]['evaluation_ids']!=list(range(1,121)):
                raise RuntimeError('Full120 is a resubstitution diagnostic, not held-out')
        else:
            if len(refit[0]['training_ids'])!=96 or len(refit[0]['evaluation_ids'])!=24:
                raise RuntimeError('Each outer fold must contain96/24 scenarios')
            if set(refit[0]['training_ids'])|set(refit[0]['evaluation_ids'])!=set(range(1,121)):
                raise RuntimeError('An outer fold must partition all120 scenarios')
            outer_ids+=refit[0]['evaluation_ids']
        train=set(refit[0]['training_ids']);union=[]
        for j in inner:
            if set(j['training_ids'])|set(j['evaluation_ids'])!=train or set(j['training_ids'])&set(j['evaluation_ids']):
                raise RuntimeError('Inner job is not a whole-scenario partition of outer training')
            union+=j['evaluation_ids']
        if len(union)!=len(set(union)) or set(union)!=train:raise RuntimeError('Inner OOF must cover outer training once')
        if c and set(refit[0]['training_ids'])&set(refit[0]['evaluation_ids']):raise RuntimeError('Outer leakage')
    if sorted(outer_ids)!=list(range(1,121)):
        raise RuntimeError('Five outer held-out cohorts must cover each original ID once')


def freeze_gate():
    p=json.loads(PROTOCOL.read_text());f=json.loads(FREEZE.read_text())
    if not f['execution_ready'] or f['protocol_sha256']!=sha(PROTOCOL):raise RuntimeError('Freeze protocol before market parsing')
    if p['development_ids']!=list(range(1,121)) or p['allow_final'] is not False:raise RuntimeError('Development-only study')
    if p['candidates']!=list(CANDIDATES):raise RuntimeError('Candidate family changed')
    for key in ('source_sha256','input_sha256'):
        for name,digest in f[key].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Frozen dependency changed: '+name)
    jobs=json.loads((ROOT/p['job_protocol']).read_text())['jobs']
    validate_jobs(jobs)
    fits=json.loads((ROOT/p['static_refits']).read_text())
    for j in jobs:validate_job(j,fits[j['job_id']])
    return p,f,jobs


def run(output=DEFAULT_OUT):
    from experimental_execution.state_budget_execution import simulate_state_budget
    p,f,jobs=freeze_gate()
    output=Path(output).resolve()
    if output.exists():raise RuntimeError('Refuse overwriting any existing attempt')
    output.mkdir(parents=True)
    m=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),phase='before_market_parser',
           protocol_sha256=sha(PROTOCOL),source_freeze_sha256=sha(FREEZE),
           sources_and_inputs_at_start={**f['source_sha256'],**f['input_sha256']},
           final_scenarios_loaded=False,market_value_access_started=False,
           historical_exposure=p['historical_exposure'])
    writej(output/'manifest.json',m)
    arrays={};groups=[];allrows=[];selections={};initial_witnesses=[]
    def save_group(group_id,arm,ids,meta):
        entry=dict(group_id=group_id,scenario_ids=list(map(int,ids)),**meta)
        entry['array_names']=[]
        retained,omitted=archive_arrays(arm,meta['kind'])
        entry['omitted_diagnostic_arrays']=omitted
        for key,value in retained.items():
            arrays[group_id+'__'+key]=value
            entry['array_names'].append(key)
        groups.append(entry)
    def references(job,r,pi,w):
        out={}
        for name,target,armname in [('same_target_B',w,'pending_hold'),('same_target_annual',w,'annual'),
                                    ('exact_thirds_annual',np.ones(3)/3,'annual')]:
            if target is None:
                reason='q1_review' if job.get('q1_case_status')=='risk_review' else 'not_balanced_case'
                out[name]=null_rows(job['evaluation_ids'],dict(job_id=job['job_id'],policy=name),reason)
                save_group(job['job_id']+'__'+name,None,job['evaluation_ids'],dict(kind='reference',policy=name,
                    job_id=job['job_id'],training_ids=job['training_ids'],target=None,status=reason))
                continue
            arm=squeeze(simulate_paired(r,pi,np.asarray(target),L,horizon=H,cost_rate=COST)[armname])
            common=dict(job_id=job['job_id'],policy=name)
            out[name]=rows_from_arm(arm,job['evaluation_ids'],common)
            save_group(job['job_id']+'__'+name,arm,job['evaluation_ids'],dict(kind='reference',policy=name,
                job_id=job['job_id'],training_ids=job['training_ids'],target=np.asarray(target).tolist()))
        return out
    def candidate(job,spec,r,pi,train,trainpi,w,scope):
        common=dict(job_id=job['job_id'],policy=spec['candidate_id'])
        if w is None:
            reason='q1_review' if job.get('q1_case_status')=='risk_review' else 'not_balanced_case'
            save_group(job['job_id']+'__'+spec['candidate_id']+'__'+scope,None,job['evaluation_ids'],
                dict(kind=scope,policy=spec['candidate_id'],job_id=job['job_id'],training_ids=job['training_ids'],
                     target=None,candidate=spec,status=reason))
            return null_rows(job['evaluation_ids'],common,reason)
        z=simulate_state_budget(r,pi,train,trainpi,w,L=L,horizon=H,cost_rate=COST,
            release_fraction=spec['release_fraction'],forecast_rule=spec['forecast_rule'],
            training_ids=job['training_ids'],actual_ids=job['evaluation_ids'],support_mode='calendar')
        initial_witnesses.append(dict(job_id=job['job_id'],candidate_id=spec['candidate_id'],scope=scope,
            status=z['status'],witness=z['initial_witness']))
        save_group(job['job_id']+'__'+spec['candidate_id']+'__'+scope,z['arm'],job['evaluation_ids'],
            dict(kind=scope,policy=spec['candidate_id'],job_id=job['job_id'],training_ids=job['training_ids'],
                 target=w.tolist(),candidate=spec,status=z['status']))
        if not z['financial_available']:
            return null_rows(job['evaluation_ids'],common,z['status'])
        return rows_from_arm(z['arm'],job['evaluation_ids'],common)
    try:
        m.update(phase='load_development',market_value_access_started=True);writej(output/'manifest.json',m)
        r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
        m.update(development_ids_loaded=ids.tolist(),phase='complete_inner_policies');writej(output/'manifest.json',m)
        fits=json.loads((ROOT/p['static_refits']).read_text())
        for context in range(6):
            js=[j for j in jobs if j['parent_context_id']==context]
            inner={s['candidate_id']:[] for s in CANDIDATES};inner_B=[]
            for job in [j for j in js if j['stage']=='inner']:
                job={**job,'q1_case_status':fits[job['job_id']]['q2_case']['status']}
                w=validate_job(job,fits[job['job_id']]);tr=np.array(job['training_ids'])-1;ev=np.array(job['evaluation_ids'])-1
                refs=references(job,r[ev],pi[ev],w);inner_B+=refs['same_target_B']
                for name,rows in refs.items():allrows.extend([dict(**row,scope='inner_reference',context_id=context) for row in rows])
                for spec in CANDIDATES:
                    rows=candidate(job,spec,r[ev],pi[ev],r[tr],pi[tr],w,'inner_candidate')
                    inner[spec['candidate_id']]+=rows
                    allrows.extend([dict(**row,scope='inner_candidate',context_id=context) for row in rows])
                print(json.dumps(dict(completed_job=job['job_id'],candidate_count=6)),flush=True)
            chosen=select_inner(inner,inner_B,alpha=p['service_alpha'])
            job=next(j for j in js if j['stage']=='refit')
            job={**job,'q1_case_status':fits[job['job_id']]['q2_case']['status']}
            w=validate_job(job,fits[job['job_id']])
            tr=np.array(job['training_ids'])-1;ev=np.array(job['evaluation_ids'])-1
            refs=references(job,r[ev],pi[ev],w)
            for name,rows in refs.items():allrows.extend([dict(**row,scope='outer_reference' if context else 'full_training_reference',context_id=context) for row in rows])
            if chosen['selected_candidate_id'] is None:
                rows=null_rows(job['evaluation_ids'],dict(job_id=job['job_id'],policy='state_budget_learner'),'no_inner_feasible_candidate')
                chosen['final_status']='execution_review'
                save_group(job['job_id']+'__state_budget_learner__selected',None,job['evaluation_ids'],
                    dict(kind='outer_selected' if context else 'full_selected',policy='state_budget_learner',
                         job_id=job['job_id'],training_ids=job['training_ids'],target=w.tolist() if w is not None else None,
                         candidate=None,status='no_inner_feasible_candidate'))
            else:
                spec=next(s for s in CANDIDATES if s['candidate_id']==chosen['selected_candidate_id'])
                rows=candidate(job,spec,r[ev],pi[ev],r[tr],pi[tr],w,'outer_selected' if context else 'full_selected')
                chosen['final_status']='funded' if all(row['assigned'] for row in rows) else 'execution_review'
                for row in rows:row['policy']='state_budget_learner'
            chosen['refit_job_id']=job['job_id'];chosen['selected_result_summary']=summarize_records(rows)
            chosen['training_ids']=job['training_ids'];chosen['evaluation_ids']=job['evaluation_ids']
            selections[str(context)]=chosen
            allrows.extend([dict(**row,scope='outer_selected' if context else 'full_selected',context_id=context) for row in rows])
            print(json.dumps(dict(context_complete=context,selected=chosen['selected_candidate_id'],status=chosen['final_status'])),flush=True)
        # Primary OOF: every original ID once; full120 remains resubstitution.
        primary=[row for row in allrows if row['scope']=='outer_selected']
        if sorted(row['scenario_id'] for row in primary)!=list(range(1,121)):
            raise RuntimeError('Primary outer paths must retain all120 original IDs once')
        summaries=[]
        for scope in ('outer_selected','full_selected'):
            rows=[x for x in allrows if x['scope']==scope]
            summaries.append(dict(scope=scope,policy='state_budget_learner',**summarize_records(rows)))
        for scope in ('outer_reference','full_training_reference'):
            for name in REFS:
                rows=[x for x in allrows if x['scope']==scope and x['policy']==name]
                summaries.append(dict(scope=scope,policy=name,**summarize_records(rows)))
        pd.DataFrame(allrows).to_csv(output/'path_metrics.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        pd.DataFrame(summaries).to_csv(output/'summary.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        writej(output/'selections.json',selections);writej(output/'trace_groups.json',groups)
        writej(output/'initial_witnesses.json',initial_witnesses)
        np.savez_compressed(output/'traces.npz',**arrays)
        m['sources_and_inputs_at_end']={name:sha(ROOT/name) for name in m['sources_and_inputs_at_start']}
        if m['sources_and_inputs_at_end']!=m['sources_and_inputs_at_start']:raise RuntimeError('Frozen source/input changed during run')
        m.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),phase='complete',
            candidate_policy_runs=len(initial_witnesses),trace_groups=len(groups),array_fields=len(arrays),
            path_rows=len(allrows),summary_rows=len(summaries),new_final_reads=0,
            model_history='New exploratory candidate; old main/final artifacts unchanged.',
            outputs_sha256={x.name:sha(x) for x in output.iterdir() if x.name!='manifest.json'})
        writej(output/'manifest.json',m)
    except BaseException as exc:
        (output/'failure.txt').write_text(traceback.format_exc())
        m.update(status='failed',error_type=type(exc).__name__,error=str(exc),failed_utc=datetime.now(timezone.utc).isoformat())
        writej(output/'manifest.json',m)
        raise
    print(json.dumps(dict(status='complete',output=str(output),primary=summaries[0])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.execute:run(args.output)
    else:print(json.dumps(dict(candidates=CANDIDATES,development_only=True,market_values_parsed=False),indent=2))
