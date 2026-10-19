"""Independent fixed-selected V2b stress accounting, admission and summary audit.

No production helper is imported. The copied frozen independent physical ledger
changes only its known fee parameter; independent scalar metrics are reused.
No forecast or full action-menu optimality is recomputed by this audit.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,math,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_rebuild as base
import audit_state_budget_mitigation as mitigation
PLAN=ROOT/'docs/state_budget_mitigation_stress_audit_plan.json'
FREEZE=ROOT/'docs/state_budget_mitigation_stress_audit_source_freeze.json'
PROTOCOL=ROOT/'docs/state_budget_mitigation_stress_protocol.json'
PRODUCER_FREEZE=ROOT/'docs/state_budget_mitigation_stress_source_freeze.json'
SOURCE=ROOT/'results/state_budget_mitigation_stress_v1'
OUT=ROOT/'results/state_budget_mitigation_stress_audit_v1'
TOL=1e-12
SETTINGS=(
 dict(setting_id='base',cost_rate=.0025,support_mode='calendar',actual_inflation_shift=0.),
 dict(setting_id='fee_0000',cost_rate=0.,support_mode='calendar',actual_inflation_shift=0.),
 dict(setting_id='fee_00125',cost_rate=.00125,support_mode='calendar',actual_inflation_shift=0.),
 dict(setting_id='fee_005',cost_rate=.005,support_mode='calendar',actual_inflation_shift=0.),
 dict(setting_id='fee_01',cost_rate=.01,support_mode='calendar',actual_inflation_shift=0.),
 dict(setting_id='pooled_support',cost_rate=.0025,support_mode='pooled',actual_inflation_shift=0.),
 dict(setting_id='actual_inflation_plus_02',cost_rate=.0025,support_mode='calendar',actual_inflation_shift=.02))
REFS=('same_target_B','same_target_annual','exact_thirds_annual')
OBJECTS=('state_budget_learner',*REFS)
PAIR=('log_growth','cagr','mdd','nominal_fees','trade_fraction','automatic_fraction')
sha,readj,writej=base.sha,base.readj,base.writej


def physical(arm,r,pi,target,group,c,cost):
    label=group['group_id'];x=arm['holdings'];post=arm['posttrade_holdings'];n,h=len(r),r.shape[1]
    c.check(x.shape==(n,h+1,3) and post.shape==(n,h,3),label+'/complete original-H account')
    c.arr(x[:,0],np.tile([0.,0.,100.],(n,1)),label+'/initial100bills')
    c.arr(x[:,1:],post*(1+r),label+'/after-action actual joint nominal returns')
    cp=np.column_stack((np.ones(n),np.cumprod(1+pi,axis=1)))
    c.arr(arm['cpi'],cp,label+'/original CPI')
    wealth=x.sum(axis=2);real=wealth/cp;peak=np.maximum.accumulate(real,axis=1);dd=np.maximum(1-real/peak,0);mdd=np.maximum.accumulate(dd,axis=1)
    for key,value in [('nominal_wealth',wealth),('real_wealth',real),('real_peak',peak),('current_drawdown',dd),('running_mdd',mdd),('pretrade_wealth',wealth[:,:-1])]:c.arr(arm[key],value,label+'/'+key)
    c.arr(arm['weights'],x/wealth[:,:,None],label+'/weights');c.arr(arm['end_weights'],arm['weights'][:,1:],label+'/endweights')
    moved=np.maximum(post-x[:,:-1],0).sum(axis=2)
    c.arr(arm['moved'],moved,label+'/destination amount once');c.arr(arm['fees'],cost*moved,label+'/implicit fee equation once')
    c.arr(post.sum(axis=2)+arm['fees'],wealth[:,:-1],label+'/self finance')
    c.arr(arm['fee_fraction'],arm['fees']/wealth[:,:-1],label+'/fee fraction')
    c.arr(arm['moved_fraction'],moved/wealth[:,:-1],label+'/moved fraction')
    c.arr(arm['trades'],moved>TOL,label+'/physical trade not target label')
    c.arr(arm['pretrade_tv_drift'],.5*np.abs(arm['weights'][:,:-1]-target).sum(axis=2),label+'/pretrade drift')
    c.arr(arm['end_year_tv_drift'],.5*np.abs(arm['weights'][:,1:]-target).sum(axis=2),label+'/end drift')
    bad=np.maximum.accumulate(mdd-.4>TOL,axis=1);c.arr(arm['ever_breached'],bad,label+'/absorbing original breach')
    first=np.where(bad.any(axis=1),bad.argmax(axis=1),-1);c.arr(arm['first_crossing'],first,label+'/first crossing')
    c.arr(arm['triggered'],first>=0,label+'/triggered')
    recovered=np.zeros(n,bool);recovery=np.full(n,-1,int)
    for i,tau in enumerate(first):
        if tau>=0:
            for key,value in [('drawdown_at_trigger',mdd[i,tau]),('peak_at_trigger',peak[i,tau]),('real_wealth_at_trigger',real[i,tau])]:c.value(arm[key][i],value,label+f'/{i}/'+key)
            end=next((e for e in range(tau+1,h+1) if real[i,e]>=peak[i,tau]),-1)
            recovery[i]=end;recovered[i]=end>=0
        else:
            for key in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):c.check(np.isnan(arm[key][i]),label+f'/{i}/'+key+' NULL')
    for key,value in [('recovery_endpoint',recovery),('recovered',recovered),('recovery_time',np.where(recovered,recovery-first,-1)),
            ('right_censored',(first>=0)&~recovered),('available_followup',np.where(first>=0,h-first,-1)),
            ('observed_followup',np.where(recovered,recovery-first,np.where(first>=0,h-first,-1)))]:c.arr(arm[key],value,label+'/'+key)
    severity=np.array([max(mdd[i,-1]-mdd[i,tau],0) if tau>=0 else np.nan for i,tau in enumerate(first)])
    c.arr(arm['added_severity'],severity,label+'/added severity')
    terminal=np.where(bad[:,-1],mdd[:,-1]-.4,0.) if group['kind']!='reference' else np.maximum(mdd[:,-1]-.4,0)
    c.arr(arm['terminal_excess'],terminal,label+'/terminal excess convention')
    if group['kind']=='reference':
        pending=bad[:,:-1] if group['policy']=='same_target_B' else np.zeros((n,h),bool)
        c.arr(arm['pending_at_decision'],pending,label+'/reference original rule')
        c.arr(arm['target_execution'],~pending,label+'/reference annual/pending executions')
        c.arr(arm['actions'],np.where(pending,'hold','restore'),label+'/reference actions')
        c.arr(post[pending],x[:,:-1][pending],label+'/reference pending hold')
        restore=~pending;c.arr(post[restore]/post[restore].sum(axis=1)[:,None],np.broadcast_to(target,post[restore].shape),label+'/reference target',atol=3e-12)
        return
    guard=arm['guard_pending'];new=arm['guard_triggered'];before=guard[:,:-1]|bad[:,:-1]
    c.arr(guard[:,1:],np.maximum.accumulate(new,axis=1),label+'/absorbing guard pending')
    c.arr(arm['was_pending_input'],before,label+'/pre-action pending')
    c.arr(arm['was_guard_pending_input'],guard[:,:-1],label+'/pre-action guard')
    c.arr(arm['pending_at_decision'],before|new,label+'/current guard included pending')
    c.arr(arm['guard_pending_at_decision'],guard[:,:-1]|new,label+'/current guard flag')
    c.arr(arm['automatic_service'],~(before|new),label+'/automatic service')
    c.arr(arm['gate_evaluated'],~before,label+'/only active evaluated')
    c.arr(new,(arm['selected_index']<0)&~before,label+'/new guard only evaluated no feasible')
    c.arr(arm['first_guard_trigger'],np.where(new.any(axis=1),new.argmax(axis=1)+1,-1),label+'/first guard')
    c.arr(post[before],x[:,:-1][before],label+'/pending exact physical hold')
    c.arr(arm['actions'][before],np.full(np.count_nonzero(before),'hold'),label+'/pending action')
    c.check(not arm['target_execution'][before].any(),label+'/pending no target execution')
    c.arr(arm['targets'],target,label+'/unchanged strategic target');c.arr(arm['loss_tolerances'],[.4],label+'/original loss tolerance')
    gross=(1+r)/(1+pi[:,:,None]);c.arr(arm['realized_real_gross'],gross,label+'/after-action realized real gross')
    previous=np.full((n,h,4),np.nan);previous[:,1:,:3]=gross[:,:-1]-1;previous[:,1:,3]=pi[:,:-1]
    c.arr(arm['actual_previous_state'],previous,label+'/actual prefix t-1 only')
    inactive=before
    inactive_float='floor budget_floor selected_score pred_log_growth_hold pred_log_growth_selected uniform_predicted_advantage conditional_predicted_advantage min_predicted_advantage current_log_fee gross_log_advantage_uniform gross_log_advantage_min worst_real_shortfall_hold worst_real_shortfall_selected worst_real_shortfall_improvement selected_lambda'.split()
    for key in inactive_float:c.check(np.isnan(arm[key][inactive]).all(),label+'/'+key+'/pending NULL')
    for key in ('selected_index','selected_proposal'):c.check(np.all(arm[key][inactive]==-1),label+'/'+key+'/pending sentinel')
    for key in ('forecast_n','forecast_member_count'):c.check(np.all(arm[key][inactive]==0),label+'/'+key+'/pending zero')
    for key in arm:
        if key.startswith('candidate_') or key.startswith('interval_'):
            values=arm[key][inactive]
            if arm[key].dtype.kind=='f':c.check(np.isnan(values).all(),label+'/'+key+'/pending NULL')
            elif arm[key].dtype.kind=='i':c.check(np.all(values==-1),label+'/'+key+'/pending sentinel')
            elif arm[key].dtype.kind=='b':c.check(not values.any(),label+'/'+key+'/pending false')
    for key in ('support_feasible','hold_feasible','growth_sacrifice'):c.check(not arm[key][inactive].any(),label+'/'+key+'/pending false')
    expectedreason=np.where(guard[:,:-1],'guard_pending_hold','already_breached_pending_hold')
    c.arr(arm['gate_reason'][inactive],expectedreason[inactive],label+'/pending reason')


def merge(target,addition):
    for name,digest in addition.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix()!=name:raise ValueError('Relative immutable path required')
        if name in target and target[name]!=digest:raise ValueError('Conflicting frozen source/input hash')
        target[name]=digest


def snapshot_complete(folder,manifest,snapshot):
    if manifest['status']!='complete' or manifest['final_scenarios_loaded'] is not False or manifest['new_final_reads']!=0:
        raise RuntimeError('Complete unmodified development producer required')
    if manifest['sources_and_inputs_at_start']!=manifest['sources_and_inputs_at_end']:
        raise RuntimeError('Producer changed during execution')
    merge(snapshot,manifest['sources_and_inputs_at_start'])
    merge(snapshot,{str((folder/'manifest.json').relative_to(ROOT)):sha(folder/'manifest.json')})
    merge(snapshot,{str((folder/name).relative_to(ROOT)):digest for name,digest in manifest['outputs_sha256'].items()})
    base.verify_map(snapshot)


def authorize():
    plan,freeze,protocol,producer_freeze=map(readj,(PLAN,FREEZE,PROTOCOL,PRODUCER_FREEZE))
    if (plan.get('version')!='state-budget-mitigation-stress-independent-audit-v1' or not freeze['execution_ready']
        or freeze['plan_sha256']!=sha(PLAN) or plan['development_ids']!=list(range(1,121)) or plan['allow_final'] is not False
        or plan['settings']!=list(SETTINGS) or protocol['settings']!=list(SETTINGS)
        or protocol['development_ids']!=list(range(1,121)) or protocol['allow_final'] is not False
        or not producer_freeze['execution_ready'] or producer_freeze['protocol_sha256']!=sha(PROTOCOL)):
        raise RuntimeError('Frozen fixed7-setting development audit before any numeric access')
    required={'scripts/audit_state_budget_mitigation_stress.py','tests/test_state_budget_mitigation_stress_audit.py',
        'scripts/audit_state_budget_rebuild.py','scripts/audit_state_budget_mitigation.py','src/safeinvest/data.py'}
    if not required<=set(freeze['source_sha256']):raise RuntimeError('Complete independent audit source required')
    snapshot={};merge(snapshot,freeze['source_sha256']);merge(snapshot,freeze['input_sha256'])
    merge(snapshot,{str(x.relative_to(ROOT)):sha(x) for x in (PLAN,FREEZE,PROTOCOL,PRODUCER_FREEZE)})
    producer=readj(SOURCE/'manifest.json')
    closure={**producer_freeze['source_sha256'],**producer_freeze['input_sha256']}
    if (sha(SOURCE/'manifest.json')!=plan['producer_manifest_sha256'] or producer['protocol_sha256']!=sha(PROTOCOL)
        or producer['source_freeze_sha256']!=sha(PRODUCER_FREEZE) or producer['sources_and_inputs_at_start']!=closure
        or producer['development_ids_loaded']!=list(range(1,121))
        or (producer['trace_groups'],producer['path_rows'],producer['summary_rows'],producer['cohort_comparisons'])!=(168,6720,56,42)
        or producer['new_fits']!=0 or producer['new_inner_selections']!=0 or not producer['base_reproduction_passed']):
        raise RuntimeError('Complete fixedselected producer identity required')
    expected={'path_metrics.csv.gz','summary.csv.gz','cohort_comparisons.json','trace_groups.json','initial_witnesses.json','traces.npz','base_reproduction_receipt.json'}
    if set(producer['outputs_sha256'])!=expected:raise RuntimeError('Bind every declared stress artifact')
    snapshot_complete(SOURCE,producer,snapshot)
    parent=ROOT/protocol['canonical_directory'];canonical=readj(parent/'manifest.json')
    if sha(parent/'manifest.json')!=protocol['canonical_manifest_sha256'] or producer['canonical_manifest_sha256']!=sha(parent/'manifest.json'):
        raise RuntimeError('Fixed canonical parent identity required')
    snapshot_complete(parent,canonical,snapshot)
    canonical_freeze=readj(ROOT/protocol['inputs']['canonical_source_freeze'])
    if (canonical['source_freeze_sha256']!=sha(ROOT/protocol['inputs']['canonical_source_freeze'])
        or canonical['sources_and_inputs_at_start']!={**canonical_freeze['source_sha256'],**canonical_freeze['input_sha256']}):
        raise RuntimeError('Canonical complete source/input closure mismatch')
    return plan,protocol,producer,parent,snapshot


def independent_contexts(jobs,fits,selections,groups,c):
    c.check(len(jobs)==30 and len({j['job_id'] for j in jobs})==30,'complete30 original ownfit jobs')
    c.check(set(selections)=={str(i) for i in range(6)},'all6 canonical selections')
    contexts={};outer=[]
    for context in range(6):
        candidates=[j for j in jobs if j['stage']=='refit' and j['parent_context_id']==context]
        c.check(len(candidates)==1,f'{context}/one own refit');job=candidates[0]
        tr,ev=job['training_ids'],job['evaluation_ids']
        for ids in (tr,ev):c.check(bool(ids) and len(set(ids))==len(ids) and all(type(i) is int and 1<=i<=120 for i in ids),f'{context}/development IDs')
        c.check((tr==ev==list(range(1,121))) if context==0 else (len(tr)==96 and len(ev)==24 and not set(tr)&set(ev) and set(tr)|set(ev)==set(range(1,121))),f'{context}/full or OOF partition')
        if context:outer+=ev
        own=fits[job['job_id']];c.check(own['refit']['training_ids']==tr,f'{context}/own fit training')
        case=own['q2_case'];w=case['product']['weights'] if case['in_balanced_case'] else None
        if w is not None:c.check(len(w)==3 and min(w)>0 and abs(math.fsum(w)-1)<=TOL,f'{context}/three positive target')
        chosen=selections[str(context)]
        c.check(chosen['training_ids']==tr and chosen['evaluation_ids']==ev and chosen['refit_job_id']==job['job_id'],f'{context}/selected original IDs')
        cid=chosen['selected_candidate_id'];spec=next((s for s in base.SPECS if s['candidate_id']==cid),None)
        c.check(cid is None or spec is not None,f'{context}/frozen selected tuple')
        scope='full_selected' if context==0 else 'outer_selected'
        selected=[g for g in groups if g['job_id']==job['job_id'] and g['kind']==scope]
        refs={ref:[g for g in groups if g['job_id']==job['job_id'] and g['kind']=='reference' and g['policy']==ref] for ref in REFS}
        c.check(len(selected)==1 and all(len(v)==1 for v in refs.values()),f'{context}/canonical4 group inventory')
        selected=selected[0];c.check(selected.get('candidate')==spec and selected.get('target')==w,f'{context}/exact canonical tuple target')
        refs={k:v[0] for k,v in refs.items()}
        for ref,g in refs.items():c.check(g.get('target')==([1/3]*3 if ref=='exact_thirds_annual' else w),f'{context}/{ref}/canonical target')
        for g in [selected,*refs.values()]:c.check(g['scenario_ids']==ev and g['training_ids']==tr,f'{context}/canonical fan actual IDs')
        reason=('q1_review' if case['status']=='risk_review' else 'not_balanced_case') if w is None else 'no_inner_feasible_candidate'
        contexts[context]=dict(job=job,target=w,spec=spec,null_reason=reason,selected=selected,references=refs)
    c.check(sorted(outer)==list(range(1,121)),'outer evaluation IDs exactly once')
    return contexts


def initial_admission(target,spec,training_returns,training_inflation,setting):
    if target is None or spec is None:return None
    post,fee,moved=base.fee_endpoint([0.,0.,100.],target,setting['cost_rate'])
    r,pi=np.asarray(training_returns),np.asarray(training_inflation)
    support=[[(1+float(r[i,t,j]))/(1+float(pi[i,t])) for j in range(3)] for i in range(len(r)) for t in (range(30) if setting['support_mode']=='pooled' else (0,))]
    worst=min(math.fsum(a*b for a,b in zip(row,post)) for row in support)
    budget=100-40*spec['release_fraction'];allowed=worst-budget>=-100*TOL
    return dict(allowed=allowed,posttrade=post,fee=fee,moved=moved,budget_floor=budget,worst_next_real=worst,support_members=len(support))


def check_initial(group,arm,witness,context,setting,r,pi,c):
    tr=np.array(context['job']['training_ids'])-1;target=context['target'];spec=context['spec'];label=group['group_id']
    result=initial_admission(target,spec,r[tr],pi[tr],setting)
    c.check(witness['context_id']==group['context_id'] and witness['setting_id']==setting['setting_id'] and witness['selected_candidate']==spec and witness['status']==group['status'],label+'/initial witness identity')
    if result is None:
        c.check(not arm and witness['witness'] is None,label+'/no selected candidate or own target means NULL')
        c.check(group['status']==context['null_reason'] and not group['financial_available'],label+'/fixed unavailable selection')
        return None
    initial=witness['witness'];c.check(initial is not None,label+'/initial fulltarget witness required')
    c.check(bool(arm)==result['allowed']==group['financial_available'],label+'/independent initial support admission')
    c.check(group['status']==('funded' if result['allowed'] else 'preinvestment_review'),label+'/initial status')
    c.value(initial['budget_floor'],result['budget_floor'],label+'/initial original budget')
    c.arr(initial['candidate_posttrade'][1],result['posttrade'],label+'/fulltarget initial endpoint')
    c.value(initial['candidate_fees'][1],result['fee'],label+'/initial fee once')
    c.check(initial['candidate_lambda'][1]==1 and not initial['candidate_feasible'][0],label+'/no hold or partial initial substitute')
    c.check(initial['selected_index']==(1 if result['allowed'] else -1),label+'/only fulltarget or Review')
    c.check(initial['support_feasible']==result['allowed'] and not initial['mitigation_evaluated'],label+'/no initial mitigation rescue')
    if arm:
        c.arr(arm['posttrade_holdings'][:,0],np.tile(result['posttrade'],(len(arm['fees']),1)),label+'/funded initial posttrade')
        c.arr(arm['fees'][:,0],np.full(len(arm['fees']),result['fee']),label+'/funded first fee')
    return dict(context_id=group['context_id'],setting_id=setting['setting_id'],funded=result['allowed'],budget_floor=result['budget_floor'],supported_worst_next_real=result['worst_next_real'],support_members=result['support_members'])


def independent_comparisons(rows):
    output=[]
    for setting in SETTINGS:
        for cohort in ('outer_OOF','full_resubstitution'):
            same=[r for r in rows if r['setting_id']==setting['setting_id'] and r['cohort']==cohort]
            model=[r for r in same if r['policy']=='state_budget_learner'];funded={r['scenario_id']:r for r in model if r['assigned']}
            for ref in REFS:
                reference=[r for r in same if r['policy']==ref];byid={r['scenario_id']:r for r in reference};ids=sorted(funded)
                if set(byid)!={r['scenario_id'] for r in model} or any(not byid[s]['assigned'] for s in ids):raise ValueError('Complete matching reference IDs')
                differences={}
                for metric in PAIR:
                    v=[funded[s][metric]-byid[s][metric] for s in ids]
                    differences[metric]=dict(mean=math.fsum(v)/len(v) if v else None,median=base.median(v) if v else None,
                        positive_n=sum(x>TOL for x in v),negative_n=sum(x < -TOL for x in v),numerical_tie_n=sum(abs(x)<=TOL for x in v))
                partition={k:[] for k in ('neither','model_only','reference_only','both')}
                for sid in ids:
                    a,b=funded[sid]['breach'],byid[sid]['breach'];partition['both' if a and b else 'model_only' if a else 'reference_only' if b else 'neither'].append(sid)
                output.append(dict(setting_id=setting['setting_id'],cohort=cohort,reference=ref,model_all_original=mitigation.summary(model),reference_all_original=mitigation.summary(reference),
                    same_funded_model=mitigation.summary([funded[s] for s in ids]),same_funded_reference=mitigation.summary([byid[s] for s in ids]),same_funded_ids=ids,initial_noninvested_ids=sorted(set(byid)-set(ids)),event_id_partition=partition,paired_differences=differences))
    return output


def baseline_exact(arm,canonical_arm,c,label):
    kept={k:v for k,v in canonical_arm.items() if not k.startswith(('candidate_','mitigation_candidate_')) and k not in ('training_previous_state','training_real_gross')}
    c.check(set(arm)==set(kept),label+'/all thin retained field names')
    values=0
    for name,a in arm.items():
        b=kept[name];c.check(a.dtype==b.dtype and a.shape==b.shape,label+'/'+name+'/exact dtype shape')
        c.check(np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b),label+'/'+name+'/exact values pairedNaN')
        values+=a.size
    return dict(fields=len(kept),scalar_values=values)


def audit_numeric(protocol,parent,r,pi,c):
    jobs=readj(ROOT/protocol['inputs']['job_protocol'])['jobs'];fits=readj(ROOT/protocol['inputs']['static_refits'])
    contexts=independent_contexts(jobs,fits,readj(parent/'selections.json'),readj(parent/'trace_groups.json'),c)
    groups=readj(SOURCE/'trace_groups.json');witnesses=readj(SOURCE/'initial_witnesses.json')
    frame=pd.read_csv(SOURCE/'path_metrics.csv.gz',float_precision='round_trip');summary=pd.read_csv(SOURCE/'summary.csv.gz',float_precision='round_trip')
    c.check(len(groups)==168 and len({g['group_id'] for g in groups})==168 and len(frame)==6720 and len(summary)==56,'complete stress dimensions')
    c.check(not frame.duplicated(['setting_id','cohort','policy','scenario_id']).any(),'all path record identities unique')
    c.check(not summary.duplicated(['setting_id','cohort','policy']).any(),'all summary identities unique')
    witness_by={(x['context_id'],x['setting_id']):x for x in witnesses};c.check(len(witnesses)==len(witness_by)==42,'all42 initial witnesses')
    expected_keys={(s['setting_id'],context,obj) for s in SETTINGS for context in range(6) for obj in OBJECTS}
    c.check({(g['setting_id'],g['context_id'],g['policy']) for g in groups}==expected_keys,'all168 groups no favorable omissions')
    derived=[];initial=[];basefields=basevalues=accounts=nulls=0
    with np.load(SOURCE/'traces.npz',allow_pickle=False) as archive,np.load(parent/'traces.npz',allow_pickle=False) as canonical:
        inventory=set()
        for group in groups:
            gid=group['group_id'];context=contexts[group['context_id']];job=context['job'];policy=group['policy']
            setting=next(s for s in SETTINGS if s['setting_id']==group['setting_id']);cohort='full_resubstitution' if group['context_id']==0 else 'outer_OOF'
            c.check(group['setting']==setting and group['cohort']==cohort and group['job_id']==job['job_id'] and group['training_ids']==job['training_ids'] and group['scenario_ids']==job['evaluation_ids'],gid+'/exact setting fan evaluation metadata')
            c.check(group['own_fit_target']==context['target'] and group['selected_candidate']==(context['spec'] if policy=='state_budget_learner' else None),gid+'/fixed ownfit tuple and target')
            names=group['array_names'];c.check(len(set(names))==len(names),gid+'/unique archive names')
            arm={name:archive[gid+'__'+name] for name in names};inventory|={gid+'__'+name for name in names}
            c.check(bool(arm)==group['financial_available'],gid+'/initial funding metadata')
            target=[1/3]*3 if policy=='exact_thirds_annual' else context['target']
            meta=dict(group_id=gid,kind='outer_selected' if policy=='state_budget_learner' else 'reference',policy=policy,target=target)
            if policy=='state_budget_learner':
                result=check_initial(group,arm,witness_by[(group['context_id'],setting['setting_id'])],context,setting,r,pi,c)
                if result is not None:initial.append(result)
            else:c.check(bool(arm)==(target is not None),gid+'/hypothetical reference availability')
            selected=frame[frame.setting_id.eq(setting['setting_id'])&frame.cohort.eq(cohort)&frame.policy.eq(policy)&frame.context_id.eq(group['context_id'])]
            c.check(len(selected)==len(job['evaluation_ids']) and set(selected.scenario_id)==set(job['evaluation_ids']),gid+'/all original rows')
            if arm:
                full=(mitigation.PRIMARY_FIELDS|mitigation.MIT_FIELDS) if policy=='state_budget_learner' else base.PHYSICAL
                omitted={k for k in full if k.startswith(('candidate_','mitigation_candidate_')) or k in ('training_previous_state','training_real_gross')}
                c.check(set(arm)==full-omitted and set(group['omitted_diagnostic_arrays'])==omitted,gid+'/complete thin inventory')
                ev=np.array(job['evaluation_ids'])-1
                physical(arm,r[ev],pi[ev]+setting['actual_inflation_shift'],target,meta,c,setting['cost_rate'])
                if policy=='state_budget_learner':
                    c.arr(arm['mitigation_evaluated'],arm['gate_evaluated']&(arm['selected_index']<0),gid+'/first primary failure mitigation')
                    c.check(np.all(arm['mitigation_evaluated'].sum(axis=1)<=1),gid+'/one mitigation at most')
                accounts+=len(ev)
            else:
                c.check(not group['omitted_diagnostic_arrays'] and selected[list(mitigation.FIN)].isna().all().all(),gid+'/Review NULL no imaginary account');nulls+=len(selected)
            for i,sid in enumerate(job['evaluation_ids']):
                observed=selected[selected.scenario_id.eq(sid)].iloc[0]
                row=dict(setting_id=setting['setting_id'],cohort=cohort,context_id=group['context_id'],job_id=job['job_id'],policy=policy,scenario_id=sid,assigned=bool(arm))
                if arm:row.update(mitigation.path_metrics(arm,i))
                else:row.update(review_reason=group['status'],**dict.fromkeys(mitigation.FIN))
                for key,value in row.items():c.value(observed[key],value,gid+f'/{sid}/'+key)
                derived.append(row)
            if setting['setting_id']=='base':
                cg=context['selected'] if policy=='state_budget_learner' else context['references'][policy]
                old={name:canonical[cg['group_id']+'__'+name] for name in cg['array_names']}
                count=baseline_exact(arm,old,c,gid);basefields+=count['fields'];basevalues+=count['scalar_values']
        c.check(set(archive.files)==inventory,'complete stress NPZ inventory')
    for _,observed in summary.iterrows():
        cohort=[row for row in derived if row['setting_id']==observed.setting_id and row['cohort']==observed.cohort and row['policy']==observed.policy]
        c.check(sorted(row['scenario_id'] for row in cohort)==list(range(1,121)),'alloriginal120 summary once')
        for key,value in mitigation.summary(cohort).items():c.value(observed[key],value,'summary/'+str(observed.setting_id)+'/'+str(observed.cohort)+'/'+str(observed.policy)+'/'+key)
    comparisons=independent_comparisons(derived);base.record_check(readj(SOURCE/'cohort_comparisons.json'),comparisons,c,'all42 samefunded comparisons')
    return dict(groups=len(groups),path_rows=len(derived),financial_path_rows=accounts,null_path_rows=nulls,summary_rows=len(summary),cohort_comparisons=len(comparisons),initial_witnesses=42,base_thin_exact_fields=basefields,base_thin_exact_scalar_values=basevalues,initial_support_checks=initial)


def run():
    if OUT.exists():raise RuntimeError('Refuse overwrite of any audit attempt')
    OUT.mkdir(parents=True);state=dict(status='running',phase='before_frozen_metadata',new_model_runs=0,final_scenarios_loaded=False)
    writej(OUT/'manifest.json',state);c=base.Checks()
    try:
        plan,protocol,producer,parent,snapshot=authorize();state.update(phase='before_development_parser',plan_sha256=sha(PLAN),source_freeze_sha256=sha(FREEZE),sources_and_inputs_at_start=snapshot);writej(OUT/'manifest.json',state)
        sys.path.insert(0,str(ROOT/'src'))
        from safeinvest.data import load_paths
        r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
        c.check(ids.tolist()==list(range(1,121)),'guarded original120 only')
        counts=audit_numeric(protocol,parent,r,pi,c)
        base.verify_map(snapshot)
        state.update(status='passed',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),checks=c.n,maximum_absolute_error=c.max_error,counts=counts,
            sources_and_inputs_at_end={name:sha(ROOT/name) for name in snapshot},scope='Independent fee-parameterized V2b physical ledger, initial fulltarget support admission, all scalar metrics/NULL/summaries/comparisons, exact thin retained base identity. No independent forecast/action-menu ranking or full13/9-menu base comparison; known modeled and actual fees, not fee misestimation.')
        writej(OUT/'manifest.json',state)
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(error).__name__,error=str(error),checks=c.n,automatic_retry=False);writej(OUT/'manifest.json',state);raise
    print(json.dumps(dict(status='passed',checks=c.n,output=str(OUT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,actual_market_or_saved_values_read=False,settings=list(SETTINGS))))
