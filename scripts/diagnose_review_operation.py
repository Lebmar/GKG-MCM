"""Frozen saved-only development Review diagnostic; no final/raw market loader."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research')]
from diagnostics import review_operation as op

PLAN=Path('docs/review_operation_diagnostic_protocol_v1.json')
FREEZE=Path('docs/review_operation_diagnostic_source_freeze_v1.json')
RELEASE=Path('docs/review_operation_diagnostic_release_v1.json')
SOURCE=Path('results/state_budget_mitigation_v1')
OUT=Path('results/review_operation_diagnostic_v1')
SOURCES={'research/diagnostics/review_operation.py','scripts/diagnose_review_operation.py',
    'tests/test_review_operation.py','src/safeinvest/core.py','src/safeinvest/__init__.py'}
MAIN_FIELDS={'fees','moved','trades','holdings','posttrade_holdings','cpi','nominal_wealth','real_wealth','real_peak',
    'current_drawdown','running_mdd','first_guard_trigger','first_crossing','guard_triggered','guard_pending',
    'was_pending_input','was_guard_pending_input','automatic_service','gate_reason','realized_real_gross'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite result must be explicit NULL')
    return value
def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(root,mapping):
    for name,digest in mapping.items():
        path=Path(name)
        if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix()!=name or sha(root/path)!=digest:
            raise RuntimeError('Frozen Review diagnostic dependency changed: '+name)


def input_closure(root=ROOT):
    """Producer/audit metadata and byte hashing, not numerical NPZ parsing."""
    root=Path(root);m=readj(root/SOURCE/'manifest.json');a=readj(root/SOURCE/'independent_audit.json')
    before=m['sources_and_inputs_at_start'];audit=a['sources_and_artifacts_at_start']
    if (m['status']!='complete' or m['final_scenarios_loaded'] is not False or m['new_final_reads']!=0
            or before!=m['sources_and_inputs_at_end'] or a['status']!='passed'
            or audit!=a['sources_and_artifacts_at_end'] or a['parent_manifest_sha256']!=sha(root/SOURCE/'manifest.json')
            or a['final_scenarios_loaded'] is not False):raise RuntimeError('Current audited immutable development producer required')
    outputs={str(SOURCE/name):digest for name,digest in m['outputs_sha256'].items()}
    originals={**outputs,str(SOURCE/'manifest.json'):sha(root/SOURCE/'manifest.json')}
    if any(audit.get(k)!=v for k,v in originals.items()):raise RuntimeError('Independent audit must bind all original outputs')
    mapping={}
    for group in (before,audit,originals,{str(SOURCE/'independent_audit.json'):sha(root/SOURCE/'independent_audit.json')}):
        for name,digest in group.items():
            if name in mapping and mapping[name]!=digest:raise RuntimeError('Conflicting frozen input')
            mapping[name]=digest
    verify(root,mapping);return mapping


def preflight(root=ROOT):
    root=Path(root);p=readj(root/PLAN);f=readj(root/FREEZE);release=readj(root/RELEASE)
    if (p.get('version')!='review-operation-development-diagnostic-v1' or p.get('execution_ready') is not False
            or p.get('development_ids')!=list(range(1,121)) or p.get('allow_final') is not False
            or p.get('fee_fractions')!=list(op.FEE_FRACTIONS) or p.get('exposure_fractions')!=list(op.EXPOSURE_FRACTIONS)
            or f.get('protocol_sha256')!=sha(root/PLAN) or not SOURCES<=set(f.get('source_sha256',{}))
            or release.get('version')!='review-operation-development-diagnostic-release-v1' or release.get('execution_ready') is not True
            or release.get('protocol_sha256')!=sha(root/PLAN) or release.get('source_freeze_sha256')!=sha(root/FREEZE)):
        raise RuntimeError('Separately recorded exact diagnostic activation required before numerical parsing')
    snapshot={**f['source_sha256'],**f['input_sha256']};verify(root,snapshot)
    if any(snapshot.get(k)!=v for k,v in input_closure(root).items()):raise RuntimeError('Complete frozen development audit closure required')
    return snapshot


def arm_from(archive,group,fields=None):
    names=group['array_names'] if fields is None else sorted(fields)
    if not set(names)<=set(group['array_names']):raise RuntimeError('Missing previously saved state fields')
    return {name:archive[group['group_id']+'__'+name] for name in names}


def write_csv(path,rows):
    keys=sorted({k for row in rows for k in row})
    with gzip.open(path,'wt',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(clean(rows))


def summary(rows):
    if not rows:return dict(original_n=0)
    values=lambda key:[float(r[key]) for r in rows]
    mean=lambda key:math.fsum(values(key))/len(rows)
    quant=lambda key,p:sorted(values(key))[max(0,math.ceil(p*len(rows))-1)]
    return dict(original_n=len(rows),breach_count=sum(r['breach'] for r in rows),mean_log_growth=mean('log_growth'),
        median_cagr=float(np.median(values('cagr'))),p10_cagr=quant('cagr',.1),mean_mdd=mean('mdd'),p90_mdd=quant('mdd',.9),
        mean_nominal_fees=mean('nominal_fees'),mean_trade_fraction=mean('trade_fraction'))


def run(root=ROOT):
    root=Path(root)
    if (root/OUT).exists():raise RuntimeError('Never overwrite any Review diagnostic attempt')
    snapshot=preflight(root);(root/OUT).mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='before_saved_development_arrays',started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(root/PLAN),source_freeze_sha256=sha(root/FREEZE),release_sha256=sha(root/RELEASE),
        sources_and_inputs_at_start=snapshot,raw_market_parser_called=False,final_scenarios_loaded=False,new_fits=0,new_selections=0,
        main_policy_changed=False,new_admission=False)
    writej(root/OUT/'manifest.json',state)
    try:
        groups=readj(root/SOURCE/'trace_groups.json');selected=[g for g in groups if g['kind']=='outer_selected']
        if len(selected)!=5 or sorted(s for g in selected for s in g['scenario_ids'])!=list(range(1,121)):raise RuntimeError('Every original development ID exactly once in five OOF groups')
        annual=[];phases=[];accounts=[];cost_rows=[];break_evens=[];static_rows=[];baselines=[];traces={};trace_groups=[];checks=[]
        with np.load(root/SOURCE/'traces.npz',allow_pickle=False) as archive:
            for g in selected:
                ids=g['scenario_ids'];job=g['job_id'];train=g['training_ids'];n=len(ids)
                if set(ids)&set(train) or len(train)!=96 or any(not 1<=sid<=120 for sid in [*ids,*train]):raise RuntimeError('Own96 training/24 disjoint evaluation development-only')
                a=arm_from(archive,g,MAIN_FIELDS)
                if a['fees'].shape!=(24,30):raise RuntimeError('Original24-path outer fold/full30-year horizon required')
                refs={}
                for policy in op.REFERENCES:
                    candidates=[x for x in groups if x['kind']=='reference' and x['job_id']==job and x['policy']==policy]
                    if len(candidates)!=1 or candidates[0]['scenario_ids']!=ids:raise RuntimeError('Complete same-ID own-job reference required')
                    refs[policy]=arm_from(archive,candidates[0])
                gross,real_gross=op.reconstruct_market(refs['exact_thirds_annual'])
                op.close(real_gross,a['realized_real_gross'],'reference market versus after-action saved real gross')
                y,z,original=op.phase_rows(a,ids,job_id=job);annual+=y;phases+=z;accounts+=original
                for key,value in a.items():traces[job+'__source__'+key]=value
                trace_groups.append(dict(group_id=job+'__source',kind='saved_outer_selected',scenario_ids=ids,training_ids=train,target=g['target'],array_names=sorted(a)))
                for i,sid in enumerate(ids):
                    baselines.append(dict(policy='main',scenario_id=sid,job_id=job,**op.financial(a['real_wealth'][i],a['fees'][i],a['trades'][i])))
                    tau=int(a['first_guard_trigger'][i]);applicable=0<tau<30
                    for policy,r in refs.items():
                        baselines.append(dict(policy=policy,scenario_id=sid,job_id=job,**op.financial(r['real_wealth'][i],r['fees'][i],r['trades'][i])))
                        break_evens.append(dict(job_id=job,scenario_id=sid,reference=policy,guard_year=None if tau<0 else tau,applicable=applicable,
                            **op.break_even(a['real_wealth'][i,-1],r['real_wealth'][i,-1],a['holdings'][i,tau].sum() if applicable else 0,applicable=applicable)))
                for q in op.FEE_FRACTIONS:
                    policy='review_fee_'+str(q);packed={k:[] for k in ('real_wealth','real_peak','running_mdd')}
                    for i,sid in enumerate(ids):
                        tau=int(a['first_guard_trigger'][i]);cost=op.debit_review_cost(a['real_wealth'][i],a['real_peak'][i],a['running_mdd'][i],a['holdings'][i],a['cpi'][i],tau,q)
                        if q==0:
                            for field in packed:
                                if not np.array_equal(cost[field],a[field][i]):raise RuntimeError('Zero service fee must reproduce exact original path')
                        fee=cost['review_fee_nominal'];metrics=op.financial(cost['real_wealth'],a['fees'][i],a['trades'][i])
                        metrics.update(mdd=float(cost['running_mdd'][-1]),breach=cost['breach'],nominal_fees=metrics['nominal_fees']+fee)
                        cost_rows.append(dict(policy=policy,job_id=job,scenario_id=sid,fee_fraction=q,applicable=cost['applicable'],
                            guard_year=None if tau<0 else tau,debit_endpoint=cost['debit_endpoint'],review_fee_nominal=fee,
                            review_fee_real=cost['review_fee_real'],first_crossing=cost['first_crossing'],**metrics))
                        for field in packed:packed[field].append(cost[field])
                    prefix=job+'__'+policy
                    for field,values in packed.items():traces[prefix+'__'+field]=np.asarray(values)
                    trace_groups.append(dict(group_id=prefix,kind='review_cost_sensitivity',scenario_ids=ids,fee_fraction=q,array_names=list(packed)))
                for fraction in op.EXPOSURE_FRACTIONS:
                    target=fraction*np.asarray(g['target'])+(1-fraction)*np.array([0.,0.,1.])
                    for rule in ('hold','annual'):
                        policy='a'+str(fraction)+'_'+rule;r=op.static_reference(gross,a['cpi'],target,rule=rule)
                        if fraction==1 and rule=='annual':checks.append(dict(job_id=job,**op.compare_all_reference_fields(r,refs['same_target_annual'])))
                        prefix=job+'__'+policy
                        for field,value in r.items():traces[prefix+'__'+field]=value
                        trace_groups.append(dict(group_id=prefix,kind='hypothetical_static_reference',scenario_ids=ids,training_ids=train,
                            target=target.tolist(),fraction=fraction,rule=rule,array_names=sorted(r),service_admitted=False))
                        for i,sid in enumerate(ids):static_rows.append(dict(policy=policy,job_id=job,scenario_id=sid,fraction=fraction,rule=rule,
                            service_admitted=False,**op.financial(r['real_wealth'][i],r['fees'][i],r['trades'][i])))
        summaries=[dict(kind=kind,policy=policy,**summary([r for r in rows if r['policy']==policy]))
            for kind,rows in [('original',baselines),('review_cost',cost_rows),('low_exposure',static_rows)] for policy in sorted({r['policy'] for r in rows})]
        pair_rows=[];pair_summaries=[]
        for policy in sorted({r['policy'] for r in static_rows}):
            static={r['scenario_id']:r for r in static_rows if r['policy']==policy}
            for reference in ('main',*op.REFERENCES):
                old={r['scenario_id']:r for r in baselines if r['policy']==reference};parts=dict(neither=0,static_only=0,reference_only=0,both=0)
                part=[]
                for sid in range(1,121):
                    x,y=static[sid],old[sid];parts['both' if x['breach'] and y['breach'] else 'static_only' if x['breach'] else 'reference_only' if y['breach'] else 'neither']+=1
                    row=dict(policy=policy,reference=reference,scenario_id=sid,**{'delta_'+k:x[k]-y[k] for k in ('log_growth','cagr','mdd','nominal_fees','trade_fraction')})
                    part.append(row);pair_rows.append(row)
                pair_summaries.append(dict(policy=policy,reference=reference,paired_n=120,event_partition=parts,
                    mean_differences={k:math.fsum(r['delta_'+k] for r in part)/120 for k in ('log_growth','cagr','mdd','nominal_fees','trade_fraction')}))
        for name,rows in [('annual_phases.csv.gz',annual),('account_phases.csv.gz',phases),('original_accounts.csv.gz',accounts),
                ('review_cost_paths.csv.gz',cost_rows),('review_cost_break_even.csv.gz',break_evens),('low_exposure_paths.csv.gz',static_rows),
                ('baseline_paths.csv.gz',baselines),('financial_summary.csv.gz',summaries),('low_exposure_paired_paths.csv.gz',pair_rows)]:write_csv(root/OUT/name,rows)
        writej(root/OUT/'low_exposure_paired_summary.json',pair_summaries);writej(root/OUT/'annual_reproduction_checks.json',checks)
        writej(root/OUT/'trace_groups.json',trace_groups);np.savez_compressed(root/OUT/'traces.npz',**traces)
        phase_summary={phase:dict(periods=sum(r['phase']==phase for r in annual),
            mean_equity_weight=float(np.mean([r['equity_weight'] for r in annual if r['phase']==phase])) if any(r['phase']==phase for r in annual) else None,
            nominal_fees=math.fsum(r['nominal_fee'] for r in annual if r['phase']==phase),
            total_log_growth=math.fsum(r['net_log_growth'] for r in annual if r['phase']==phase)) for phase in op.PHASES}
        writej(root/OUT/'operation_summary.json',dict(original_n=120,guard_n=sum(r['first_guard'] is not None for r in accounts),
            no_guard_n=sum(r['first_guard'] is None for r in accounts),within_H_review_debits=sum(r['review_debit_within_H'] for r in accounts),phase_summary=phase_summary,
            cost_assumption='Unknown one-time fee scenarios only. Proportional asset debit after trigger-year market at endpoint tau, before decision tau+1, only tau<H. No service benefit assumed; original peak/CPI retained.',
            mdd_caution='Full-H risk is recalculated with original history; segment maxima are never added or treated as causal risk contributions.',
            reference_caution='Eight predeclared simple low-exposure references, not risk-matched/certified/selected products. All120 OOF IDs retained. Adaptive development diagnostic, no independent new sample.'))
        verify(root,snapshot)
        if sha(root/PLAN)!=state['protocol_sha256'] or sha(root/FREEZE)!=state['source_freeze_sha256'] or sha(root/RELEASE)!=state['release_sha256']:raise RuntimeError('Diagnostic activation changed during execution')
        state.update(status='complete',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            sources_and_inputs_at_end={k:sha(root/k) for k in snapshot},original_accounts=120,annual_rows=len(annual),
            cost_rows=len(cost_rows),low_exposure_rows=len(static_rows),paired_rows=len(pair_rows),
            outputs_sha256={p.name:sha(p) for p in (root/OUT).iterdir() if p.name!='manifest.json'})
        writej(root/OUT/'manifest.json',state)
    except BaseException as error:
        (root/OUT/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(error).__name__,error=str(error),automatic_retry=False)
        writej(root/OUT/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:print(json.dumps(dict(source_only=True,raw_market_access=False,actual_diagnostic_executed=False)))
