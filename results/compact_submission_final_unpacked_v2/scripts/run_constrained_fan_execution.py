"""One chance-constrained net-growth rollout; no price or architecture search.

Reuses audited static components, never old dynamic choices. All admission
decisions precede evaluation returns. Full120 evaluation is in-sample only.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
import run_simple_execution_comparison as helpers
import run_joint_liquidity_refit as joint
from safeinvest.data import load_paths
from experimental_execution.constrained_fan_rollout import simulate_constrained
import run_benefit_cost_execution as identity_helpers
from experimental_execution.simple_drift_execution import simulate_drift
from experimental_execution.review_pending import simulate_paired

PROTOCOL=ROOT/'docs/constrained_fan_execution_protocol.json'
OUT=ROOT/'results/model_review/constrained_fan_execution_v1'
METHODS={'constrained_net_growth_rollout':'fixed_alpha_B_continuation'}
PRIMARY='constrained_net_growth_rollout'
REFERENCES=helpers.REFERENCES
H=30; LOSS=.4; COST=.0025; ALPHA=.1; TOL=1e-12
sha=helpers.sha; read_json=helpers.read_json; write_json=helpers.write_json
write_csv_gz=helpers.write_csv_gz; write_json_gz=helpers.write_json_gz


def source_manifest():
    d=helpers.source_manifest()
    for rel in ('scripts/run_constrained_fan_execution.py','research/experimental_execution/constrained_fan_rollout.py',
                'tests/test_constrained_fan_rollout.py','tests/test_constrained_fan_pipeline.py',
                'scripts/run_benefit_cost_execution.py','research/experimental_execution/benefit_cost_execution.py',
                'tests/test_benefit_cost_execution.py','tests/test_benefit_cost_pipeline.py'):
        d[rel]=sha(ROOT/rel)
    return dict(sorted(d.items()))


def validate_protocol(p):
    if p['methods']!=METHODS or p['primary_method']!=PRIMARY or p['local_alpha']!=ALPHA:
        raise ValueError('One fixed constrained controller required')
    # Reuse only the already-audited context/data/initial-state identity checks.
    identity_helpers.validate_protocol({**p,'methods':identity_helpers.METHODS,'primary_method':identity_helpers.PRIMARY})


def training_decision(summary,kB,n):
    """A fixed controller either qualifies or goes to Review; no candidate search."""
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:raise ValueError('Positive training denominator required')
    if isinstance(kB,bool) or not isinstance(kB,(int,np.integer)) or not 0<=kB<=n:raise ValueError('Valid reference event count required')
    cap=min(int(kB),math.floor(ALPHA*n+TOL))
    k=summary['breach_count']
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<=n:raise ValueError('Complete integer event count required')
    reason='complete_training_risk_exceeds_cap' if k>cap else None
    return dict(training_eligible=reason is None,execution_status='automated' if reason is None else 'execution_review',
                execution_reason=reason,reference_training_breach_count=int(kB),training_breach_cap=cap,n_training=int(n))


def restrict_q1_bundles(p,bundles):
    # The original full120 static refit correctly has no held-out IDs.
    # Its separate dynamic in-sample evaluation uses all120 training IDs.
    # Validate that distinction rather than rewriting the immutable bundle.
    matching={**p,'jobs':[{**j,'evaluation_ids':[]} if j['job_id']=='c0_refit' else j for j in p['jobs']]}
    return helpers.restrict_q1_bundles(matching,bundles)


def summary_rows(arm,n,policy,common):
    base={**common,'policy':policy,'n_training':n}
    if arm is None:
        return {**base,**dict.fromkeys(('breach_count','mean_log_growth','mean_excess','mean_nominal_fees','optional_restores','economic_improvement_restores','risk_repair_restores','growth_sacrificing_restores','locally_infeasible_decisions'))}
    optional=arm['target_execution'].copy();optional[:,0]=False
    delta=arm['qg_restore']-arm['qg_hold']
    return {**base,'breach_count':int(arm['breach'].sum()),'mean_log_growth':float(arm['log_growth'].mean()),
        'mean_excess':float(arm['excess'].mean()),'mean_nominal_fees':float(arm['fees'].sum(axis=1).mean()),
        'optional_restores':int(optional.sum()),'economic_improvement_restores':int((optional & arm['economic_improvement']).sum()),
        'risk_repair_restores':int((optional & arm['risk_cost_justification']).sum()),
        'growth_sacrificing_restores':int((optional & (delta < -TOL)).sum()),
        'locally_infeasible_decisions':int((arm['local_model_infeasible'] & arm['local_constraint_applied']).sum())}


def simulate(r,pi,tr,target,policy):
    return simulate_constrained(r,pi,tr[0],tr[1],target,L=LOSS,horizon=H,cost_rate=COST)


def save_trace(arrays,groups,key,arm,ids,trids,meta,kind):
    for k,v in arm.items():
        if isinstance(v,np.ndarray):arrays[key+'__'+k]=v
    groups[key]=dict(scenario_ids=list(ids),training_ids=list(trids),metadata=meta,kind=kind)


def run_pipeline(p,bundles,r,pi,ids,*,guard=None,on_job=None):
    validate_protocol(p);q1=restrict_q1_bundles(p,bundles)
    if list(map(int,ids))!=list(range(1,121)) or r.shape!=(120,H,3) or pi.shape!=(120,H):raise ValueError('Complete original120 paths required')
    tables={k:[] for k in ('training_paths','training_summary','inner_validation_paths','inner_reference_paths','inner_gate_summaries','client_paths','fit_admission')}
    arrays={'actual':{},'validation':{}};groups={'actual':{},'validation':{}};choices={};completed=[]
    for c in p['contexts']:
        for job in [j for j in p['jobs'] if j['parent_context_id']==c['context_id']]:
            if guard:guard()
            tick=time.perf_counter();jid=job['job_id'];b=q1[jid];common=helpers.common_fields(job,b)
            case=common['case_status'];tr=np.asarray(job['training_ids'])-1;ev=np.asarray(job['evaluation_ids'])-1
            target=np.asarray(b['q2_case']['product']['weights']) if case=='balanced_case' else None
            reference_train=simulate_drift(r[tr],pi[tr],target,0.,L=LOSS,horizon=H,cost_rate=COST) if target is not None else None
            kB=int(reference_train['breach'].sum()) if reference_train is not None else None
            job_choices={}
            for policy in METHODS:
                meta={**common,'policy':policy,'execution_family':METHODS[policy],'reference_only':False}
                arm=None
                if case=='balanced_case':
                    arm=simulate(r[tr],pi[tr],(r[tr],pi[tr]),target,policy)
                summ=summary_rows(arm,len(tr),policy,common);tables['training_summary'].append(summ)
                status=case if case!='balanced_case' else 'automated'
                reason=case if case!='balanced_case' else None
                rows=helpers.financial_rows(arm,job['training_ids'],{**meta,'scope':'training_support'}) if arm is not None else helpers.null_rows(job['training_ids'],{**meta,'scope':'training_support'},status,reason)
                tables['training_paths'].extend(rows)
                decision=training_decision(summ,kB,len(tr)) if case=='balanced_case' else dict(training_eligible=False,execution_status=case,execution_reason=case)
                gate=None
                if job['stage']=='refit':
                    gate=helpers.gate_inner([v for v in tables['inner_validation_paths'] if v['context']==c['name'] and v['policy']==policy],
                        [v for v in tables['inner_reference_paths'] if v['context']==c['name']],c['training_ids'])
                    tables['inner_gate_summaries'].append(dict(context=c['name'],context_id=c['context_id'],repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],policy=policy,**gate))
                    if case=='balanced_case' and not gate['eligible']:decision={**decision,'execution_status':'execution_review','execution_reason':gate['reason']}
                job_choices[policy]={**decision,'inner_gate':gate}
                scope='inner_validation' if job['stage']=='inner' else ('in_sample_full120' if c['context_id']==0 else 'outer_oof')
                em={**meta,'scope':scope}
                evalarm=simulate(r[ev],pi[ev],(r[tr],pi[tr]),target,policy) if decision['execution_status']=='automated' else None
                rows=helpers.financial_rows(evalarm,job['evaluation_ids'],em) if evalarm is not None else helpers.null_rows(job['evaluation_ids'],em,decision['execution_status'],decision['execution_reason'])
                table='inner_validation_paths' if job['stage']=='inner' else 'client_paths';tables[table].extend(rows)
                if evalarm is not None:
                    category='validation' if job['stage']=='inner' else 'actual'
                    save_trace(arrays[category],groups[category],jid+'__'+policy,evalarm,job['evaluation_ids'],job['training_ids'],em,'selected_inner' if job['stage']=='inner' else scope)
                if job['stage']=='refit':tables['fit_admission'].append({**em,**decision,'initial_admitted':decision['execution_status']=='automated','inner_gate_passed':gate['eligible'],
                    'training_n':len(tr),'evaluation_n':len(ev)})
            # B is evaluated separately and never supplies unavailable model financials.
            scope='inner_validation_reference' if job['stage']=='inner' else ('in_sample_full120' if c['context_id']==0 else 'outer_oof')
            bm={**common,'policy':REFERENCES[0],'scope':scope,'reference_only':True}
            B=simulate_drift(r[ev],pi[ev],target,0.,L=LOSS,horizon=H,cost_rate=COST) if target is not None else None
            rows=helpers.financial_rows(B,job['evaluation_ids'],bm) if B is not None else helpers.null_rows(job['evaluation_ids'],bm,'reference_unavailable',case)
            tables['inner_reference_paths' if job['stage']=='inner' else 'client_paths'].extend(rows)
            category='validation' if job['stage']=='inner' else 'actual'
            if B is not None:save_trace(arrays[category],groups[category],jid+'__B',B,job['evaluation_ids'],job['training_ids'],bm,'B_reference')
            if job['stage']=='refit':
                eq=joint.reference.squeeze_pending(simulate_paired(r[ev],pi[ev],np.ones(3)/3,LOSS,horizon=H,cost_rate=COST)['annual']);eq['targets']=np.ones(3)/3
                em={**common,'policy':REFERENCES[1],'scope':scope,'reference_only':True,'candidate_local_index':None,'candidate_original232_index':None}
                tables['client_paths'].extend(helpers.financial_rows(eq,job['evaluation_ids'],em))
                save_trace(arrays['actual'],groups['actual'],jid+'__equal',eq,job['evaluation_ids'],job['training_ids'],em,'equal_thirds_annual')
            choices[jid]=dict(job_id=jid,training_ids=job['training_ids'],evaluation_ids=job['evaluation_ids'],case_status=case,target=target.tolist() if target is not None else None,reference_training_breach_count=kB,methods=job_choices)
            if guard:guard()
            row=dict(job_id=jid,stage=job['stage'],repeat_id=c['repeat_id'],training_n=len(tr),evaluation_n=len(ev),seconds=time.perf_counter()-tick)
            completed.append(row)
            if on_job:on_job(row)
    return dict(tables=tables,choices=choices,arrays=arrays,groups=groups,completed_jobs=completed)


def main():
    p=read_json(PROTOCOL);validate_protocol(p)
    if p.get('execution_ready') is not True:raise RuntimeError('Design/source review must precede execution')
    if OUT.exists():raise RuntimeError('Never overwrite any complete or partial archive')
    sources=source_manifest();inputs=helpers.frozen_inputs(p);protocol_sha=sha(PROTOCOL)
    if sources!=p['execution_sources_sha256']:raise RuntimeError('Frozen execution sources differ')
    bundles=read_json(ROOT/'results/model_review/refit_stability_v1/static_refits.json.gz')
    old=read_json(ROOT/'results/model_review/joint_liquidity_refit_v1/static_refits.json')
    bundles.update({k:v for k,v in old.items() if k.startswith('c0_')});restrict_q1_bundles(p,bundles)
    r,pi,ids=load_paths(ROOT/p['data']['path'],p['data']['scenario_ids'],allow_final=False)
    digest=hashlib.sha256()
    for a in (ids,r,pi):digest.update(np.ascontiguousarray(a).tobytes())
    if digest.hexdigest()!=read_json(ROOT/'results/model_review/refit_stability_v1/manifest.json')['development_arrays_sha256']:raise RuntimeError('Development arrays changed')
    def guard():
        if source_manifest()!=sources or helpers.frozen_inputs(p)!=inputs or sha(PROTOCOL)!=protocol_sha:raise RuntimeError('Execution source/input/protocol changed')
    guard();OUT.mkdir(parents=True);start=time.perf_counter();completed=[]
    def on_job(row):
        completed.append(row);write_json(OUT/'progress.json',dict(completed_jobs=completed,final_scenarios_loaded=False));print(row,flush=True)
        if time.perf_counter()-start>60*p['budget_numeric_minutes']:raise RuntimeError('Preserve incomplete archive on time budget')
    result=run_pipeline(p,bundles,r,pi,ids,guard=guard,on_job=on_job)
    if len(completed)!=130:raise RuntimeError('All130 tasks required')
    for name,n in p['expected_rows'].items():
        if len(result['tables'][name])!=n:raise RuntimeError('Row inventory differs: '+name)
    for name,rows in result['tables'].items():write_csv_gz(OUT/(name+'.csv.gz'),rows)
    write_json_gz(OUT/'job_choices.json.gz',result['choices']);write_json(OUT/'trace_groups.json',result['groups'])
    for category in result['arrays']:np.savez_compressed(OUT/(category+'_traces.npz'),**result['arrays'][category])
    guard();write_json(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
        protocol_sha256=protocol_sha,sources=sources,sources_at_end=source_manifest(),input_sha256=inputs,input_sha256_at_end=helpers.frozen_inputs(p),
        outputs_sha256={f.name:sha(f) for f in OUT.iterdir() if f.is_file()},development_arrays_sha256=digest.hexdigest(),
        scenario_ids_loaded=ids.tolist(),final_scenarios_loaded=False,training_tasks=130,primary_method=PRIMARY,
        row_counts={k:len(v) for k,v in result['tables'].items()},limitations=[
            'Calendar empirical forecasts are model assumptions, not calibrated conditional probabilities',
            'Local alpha feasibility and least violation are model heuristics, not a complete-policy safety certificate',
            'Repeated fits reuse original120 paths; full120 results are in-sample and do not add validation',
            'One constrained repair controller prospectively specified after growth-only training rejection; no more family expansion',
            'Review financial outcomes are NULL; no annual or sensitivity fallback']))
    print('complete',flush=True)


if __name__=='__main__':main()
