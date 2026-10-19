"""Independent financial-account and selector audit of saved stability studies.

Shares the frozen independent V2/V2b physical ledgers and scalar metric/selector
oracles. This is not another independent implementation of those helpers and
does not recompute each decision's forecast or optimize an unarchived menu.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip,hashlib,json,math,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_rebuild as original
import audit_state_budget_mitigation as mitigation
PLAN=ROOT/'docs/cushion_stability_audit_plan.json'
FREEZE=ROOT/'docs/cushion_stability_audit_source_freeze.json'
OUT=ROOT/'results/cushion_stability_audit_v1'
sha=original.sha
writej=original.writej
readj=original.readj


def unique_development(values):
    if not values or any(type(s) is not int or not 1<=s<=120 for s in values) or len(set(values))!=len(values):
        raise ValueError('Require unique original development IDs1-120')


def validate_jobs(jobs,fits):
    if len(jobs)!=125 or len({j['job_id'] for j in jobs})!=125 or set(fits)!={j['job_id'] for j in jobs}:
        raise ValueError('Exactly125 own-fit jobs')
    for repeat in range(1,6):
        outer=[]
        for fold in range(1,6):
            js=[j for j in jobs if j['repeat_id']==repeat and j['outer_fold']==fold]
            inner=[j for j in js if j['stage']=='inner'];refit=[j for j in js if j['stage']=='refit']
            if len(inner)!=4 or len(refit)!=1:raise ValueError('Each repeat/fold has4inner+1refit')
            refit=refit[0];training=set(refit['training_ids']);actual=set(refit['evaluation_ids'])
            if len(training)!=96 or len(actual)!=24 or training&actual or training|actual!=set(range(1,121)):
                raise ValueError('Each outer fit partitions120 as96/24')
            outer+=refit['evaluation_ids'];validation=[]
            for j in js:
                unique_development(j['training_ids']);unique_development(j['evaluation_ids'])
                if fits[j['job_id']]['refit']['training_ids']!=j['training_ids']:raise ValueError('Wrong own-training Q1 fit')
            for j in inner:
                if len(j['training_ids'])!=72 or len(j['evaluation_ids'])!=24 or set(j['training_ids'])&set(j['evaluation_ids']) or set(j['training_ids'])|set(j['evaluation_ids'])!=training:
                    raise ValueError('Inner72/24 whole-path split')
                validation+=j['evaluation_ids']
            if sorted(validation)!=sorted(training):raise ValueError('Inner OOF IDs occur once')
        if sorted(outer)!=list(range(1,121)):raise ValueError('Every repeat preserves original120 once')


def check_frozen_study(study):
    folder=ROOT/study['directory'];m=readj(folder/'manifest.json')
    if (m['status']!='complete' or m['final_scenarios_loaded'] is not False or m['new_final_reads']!=0
        or m['development_ids_loaded']!=list(range(1,121)) or m['repeat_count']!=5 or m['training_jobs']!=125
        or m['independent_additional_scenarios']!=0 or m['best_repeat_selected'] is not False
        or m['sources_and_inputs_at_start']!=m['sources_and_inputs_at_end']
        or sha(folder/'manifest.json')!=study['manifest_sha256']):raise RuntimeError('Complete fixed original study required')
    for name,digest in m['sources_and_inputs_at_start'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen producer source/input changed: '+name)
    for name,digest in m['outputs_sha256'].items():
        if sha(folder/name)!=digest:raise RuntimeError('Producer output changed: '+name)
    return folder,m


def authorize():
    p,f=readj(PLAN),readj(FREEZE)
    studies=p.get('studies',[])
    if (p.get('version')!='cushion-stability-independent-financial-selection-audit-v1'
        or len(studies)!=1 or studies[0].get('name')!='V3' or studies[0].get('variant')!='mitigation'
        or studies[0].get('directory')!='results/state_budget_cushion_stability_v1'):
        raise RuntimeError('Only the complete fixed V3 five-repeat study is in scope')
    if not f['execution_ready'] or f['plan_sha256']!=sha(PLAN) or p['allow_final'] is not False or p['development_ids']!=list(range(1,121)):
        raise RuntimeError('Freeze complete development-only audit before numerical access')
    for group in ('source_sha256','input_sha256'):
        for name,digest in f[group].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Audit dependency changed: '+name)
    if 'scripts/audit_cushion_stability.py' not in f['source_sha256']:raise RuntimeError('Bind executing source')
    studies=[(s,*check_frozen_study(s)) for s in p['studies']]
    return p,f,studies


def null_reason(group,case):
    if group['kind']=='reference' and group['target'] is None:
        return 'q1_review' if case['status']=='risk_review' else 'not_balanced_case'
    return group['status']


def scalar_cushion(real,peak,loss):
    """Independent scalar algebra; no policy/forecast engine is imported."""
    if (not all(math.isfinite(x) for x in (real,peak,loss)) or real<=0 or peak<=0
        or not 0<=loss<=1 or real>peak+1e-12*max(real,peak)):
        raise ValueError('Consistent positive real wealth/peak and original tolerance required')
    if real>=peak:return 1.
    if loss==0:return 0.
    return max(0.,min(1.,(real-(1-loss)*peak)/(loss*real)))


def cushion_witnesses(arm,target,group,c,loss=.4):
    """Only algebra/state witness identity and pending NULL, not a forecast audit."""
    if group['kind']=='reference':
        c.check('cushion_scale' not in arm and 'proposal_target_weights' not in arm,group['group_id']+'/reference has no cushion')
        return dict(active=0,pending=0)
    label=group['group_id'];active=np.asarray(arm['gate_evaluated'])
    n,h=active.shape;scale=np.asarray(arm['cushion_scale']);proposals=np.asarray(arm['proposal_target_weights'])
    c.check(scale.shape==(n,h) and scale.dtype.kind=='f',label+'/m shape/dtype')
    c.check(proposals.shape==(n,h,4,3) and proposals.dtype.kind=='f',label+'/proposal shape/dtype')
    c.check(np.isnan(scale[~active]).all() and np.isnan(proposals[~active]).all(),label+'/pending m/proposal NULL')
    c.check(np.isfinite(scale[active]).all() and np.isfinite(proposals[active]).all(),label+'/active finite witnesses')
    c.check(np.all((scale[active]>=0)&(scale[active]<=1)),label+'/m range')
    for i,t in np.argwhere(active):
        real=math.fsum(map(float,arm['holdings'][i,t]))/float(arm['cpi'][i,t])
        peak=max(float(arm['real_peak'][i,t]),real)
        m=1. if t==0 else scalar_cushion(real,peak,loss)
        c.arr(scale[i,t],m,label+f'/{i}/{t}/scalar m identity',atol=3e-12,rtol=5e-13)
        expected=[[a*m*float(target[j])+((1-a*m) if j==2 else 0.) for j in range(3)] for a in (1.,.75,.5,.25)]
        c.arr(proposals[i,t],expected,label+f'/{i}/{t}/temporary destination identity',atol=3e-12,rtol=5e-13)
    return dict(active=int(active.sum()),pending=int((~active).sum()))


def audit_repeat(folder,repeat,variant,jobs,fits,r,pi,c):
    if variant!='mitigation':raise ValueError('V3 uses the explicit shared independent mitigation ledger')
    oracle=mitigation;financial=mitigation.FIN
    directory=folder/f'repeat_{repeat:02d}'
    manifest=readj(directory/'manifest.json');groups=readj(directory/'trace_groups.json')
    selections=readj(directory/'selections.json')
    frame=pd.read_csv(directory/'path_metrics.csv.gz',float_precision='round_trip')
    saved_summary=pd.read_csv(directory/'summary.csv.gz',float_precision='round_trip')
    own=[j for j in jobs if j['repeat_id']==repeat];byjob={j['job_id']:j for j in own}
    c.check(manifest['status']=='complete' and manifest['completed_jobs']==[j['job_id'] for j in own],f'{repeat}/completed own jobs')
    c.check(len(groups)==200 and len({g['group_id'] for g in groups})==200 and len(frame)==4800 and len(saved_summary)==4,f'{repeat}/full fixed dimensions')
    c.check(not frame.duplicated(['job_id','policy','scope','scenario_id']).any(),f'{repeat}/unique path records')
    derived=[];accounts=0;periods=0;nulls=0;array_fields=0;active_cushion=0;pending_cushion=0
    with np.load(directory/'traces.npz',allow_pickle=False) as archive:
        expected_fields=set()
        for group in groups:
            gid=group['group_id'];job=byjob[group['job_id']];tr,ev=job['training_ids'],job['evaluation_ids']
            c.check(group['scenario_ids']==ev and group['training_ids']==tr and group['repeat_id']==repeat,gid+'/original IDs')
            case=fits[job['job_id']]['q2_case']
            target=np.array(case['product']['weights'],float) if case['in_balanced_case'] else None
            if group['policy']=='exact_thirds_annual':target=np.ones(3)/3
            c.value(group['target'],None,gid+'/no target') if target is None else c.arr(group['target'],target,gid+'/own target')
            scope=('inner_reference' if job['stage']=='inner' else 'outer_reference') if group['kind']=='reference' else group['kind']
            policy=group['policy'];rows=frame[frame.job_id.eq(job['job_id'])&frame.scope.eq(scope)&frame.policy.eq(policy)]
            c.check(len(rows)==len(ev) and set(rows.scenario_id)==set(ev),gid+'/all original paths')
            names=group['array_names'];c.check(len(names)==len(set(names)),gid+'/no duplicate array names')
            arm={name:archive[gid+'__'+name] for name in names};expected_fields|={gid+'__'+name for name in names}
            if arm:
                c.check(rows.assigned.all(),gid+'/uniform initial funding')
                c.check(target is not None,gid+'/funded target exists')
                # The shared independent ledgers check exact fees once, initial
                # allocations, all returns/CPI/peaks/MDD, absorbing pending and
                # the version-specific trigger action. Forecasts not recomputed.
                oracle.physical(arm,r[np.array(ev)-1],pi[np.array(ev)-1],target,group,c)
                witness=cushion_witnesses(arm,target,group,c)
                active_cushion+=witness['active'];pending_cushion+=witness['pending']
                if group['kind']=='outer_selected' and variant!='original':
                    c.arr(arm['mitigation_evaluated'],arm['gate_evaluated']&(arm['selected_index']<0),gid+'/first active failure')
                    c.check(np.all(arm['mitigation_evaluated'].sum(axis=1)<=1),gid+'/one mitigation at most')
                accounts+=len(ev);periods+=30*len(ev)
            else:
                c.check(not rows.assigned.any() and rows[list(financial)].isna().all().all(),gid+'/NULL finance preserved')
                c.check(not names and not group['omitted_diagnostic_arrays'],gid+'/no imaginary Review account')
                c.check(set(rows.review_reason)=={null_reason(group,case)},gid+'/Review reason')
                nulls+=len(ev)
            for i,sid in enumerate(ev):
                actual=rows[rows.scenario_id.eq(sid)].iloc[0]
                row=dict(job_id=job['job_id'],repeat_id=repeat,outer_fold=job['outer_fold'],context_id=job['parent_context_id'],policy=policy,scope=scope,scenario_id=sid,assigned=bool(arm))
                if arm:row.update(oracle.path_metrics(arm,i))
                else:row.update(review_reason=null_reason(group,case),**dict.fromkeys(financial))
                for key,value in row.items():c.value(actual[key],value,gid+f'/{sid}/'+key)
                derived.append(row)
        c.check(set(archive.files)==expected_fields,f'{repeat}/complete NPZ inventory')
        array_fields=len(expected_fields)
    for fold in range(1,6):
        rows=[r for r in derived if r['outer_fold']==fold]
        candidates={s['candidate_id']:[r for r in rows if r['scope']=='inner_candidate' and r['policy']==s['candidate_id']] for s in original.SPECS}
        B=[r for r in rows if r['scope']=='inner_reference' and r['policy']=='same_target_B']
        chosen=original.select(candidates,B,.1);observed=selections[str(fold)]
        for key,value in chosen.items():original.record_check(observed[key],value,c,f'{repeat}/{fold}/selection/'+key)
        refit=next(j for j in own if j['outer_fold']==fold and j['stage']=='refit')
        c.check(observed['refit_job_id']==refit['job_id'] and observed['training_ids']==refit['training_ids'] and observed['evaluation_ids']==refit['evaluation_ids'],f'{repeat}/{fold}/refit identity')
        selected=[g for g in groups if g['job_id']==refit['job_id'] and g['kind']=='outer_selected']
        c.check(len(selected)==1 and (selected[0].get('candidate') or {}).get('candidate_id')==chosen['selected_candidate_id'],f'{repeat}/{fold}/no alternative refit rescue')
        final=[r for r in rows if r['scope']=='outer_selected']
        c.check(observed['final_status']==('funded' if all(r['assigned'] for r in final) else 'execution_review'),f'{repeat}/{fold}/final status')
        original.record_check(observed['selected_result_summary'],oracle.summary(final),c,f'{repeat}/{fold}/selected summary')
    for _,saved in saved_summary.iterrows():
        cohort=[r for r in derived if r['scope']==saved.scope and r['policy']==saved.policy]
        c.check(sorted(r['scenario_id'] for r in cohort)==list(range(1,121)),f'{repeat}/{saved.policy}/original OOF120')
        for key,value in oracle.summary(cohort).items():c.value(saved[key],value,f'{repeat}/{saved.policy}/summary/'+key)
    return dict(repeat_id=repeat,groups=len(groups),path_rows=len(derived),financial_accounts=accounts,periods=periods,null_path_rows=nulls,array_fields=array_fields,active_cushion_witnesses=active_cushion,pending_cushion_witnesses=pending_cushion),saved_summary


def run():
    p,f,studies=authorize()
    if OUT.exists():raise RuntimeError('Refuse overwrite of completed or failed audit')
    OUT.mkdir(parents=True)
    snapshot=dict(plan_sha256=sha(PLAN),source_freeze_sha256=sha(FREEZE),audit_sources_and_inputs={**f['source_sha256'],**f['input_sha256']})
    writej(OUT/'manifest.json',dict(status='running',**snapshot,phase='before_market_parser'))
    c=original.Checks();counts=[]
    try:
        sys.path.insert(0,str(ROOT/'src'))
        from safeinvest.data import load_paths
        r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
        c.check(ids.tolist()==list(range(1,121)),'development IDs only')
        jobs=readj(ROOT/p['job_protocol'])['jobs']
        with gzip.open(ROOT/p['static_refits'],'rt') as handle:fits=json.load(handle)
        validate_jobs(jobs,fits)
        for study,folder,manifest in studies:
            frames=[]
            for repeat in range(1,6):
                result,summary=audit_repeat(folder,repeat,study['variant'],jobs,fits,r,pi,c)
                counts.append(dict(study=study['name'],**result));frames.append(summary)
            merged=pd.concat(frames,ignore_index=True);root_summary=pd.read_csv(folder/'repeat_summary.csv.gz',float_precision='round_trip')
            c.check(list(merged.columns)==list(root_summary.columns) and merged.shape==root_summary.shape,study['name']+'/20 root rows schema')
            for key in merged:
                if merged[key].dtype.kind in 'iufb':c.arr(merged[key],root_summary[key],study['name']+'/root summary/'+key)
                else:c.arr(merged[key].to_numpy(dtype='U'),root_summary[key].to_numpy(dtype='U'),study['name']+'/root summary/'+key)
            check_frozen_study(study)
        for name,digest in snapshot['audit_sources_and_inputs'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Audit evidence changed: '+name)
        writej(OUT/'manifest.json',dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),**snapshot,checks=c.n,maximum_absolute_error=c.max_error,counts=counts,final_scenarios_loaded=False,new_model_runs=0,
            scope='All financial accounts, exact fees/CPI/originalpeak/MDD, metric rows/NULL, inner selection and refit status. Shares frozen independent mitigation ledgers. Adds all active m/temporary destination algebra and pending NULL only; does not recompute forecasts or action menus. Canonical all-active audit is separate; dependent repeats are not new independent scenarios.'))
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        writej(OUT/'manifest.json',dict(status='failed',error_type=type(error).__name__,error=str(error),checks=c.n,**snapshot));raise
    print(json.dumps(dict(status='passed',checks=c.n,repeats=len(counts),output=str(OUT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(development_only=True,market_values_loaded=False,new_model_runs=0)))
