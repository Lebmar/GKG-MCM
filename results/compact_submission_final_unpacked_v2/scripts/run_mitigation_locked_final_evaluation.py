"""One separately released, nonblind final evaluation of the locked V2b model.

No refit, candidate selection or threshold update occurs. Source preparation
never enables final access: immutable release and release-hash freeze are both
required before importing any market loader or execution kernel.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,math,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=Path('docs/mitigation_locked_final_evaluation_protocol.json')
FREEZE=Path('docs/mitigation_locked_final_evaluation_source_freeze.json')
RELEASE=Path('docs/mitigation_locked_final_evaluation_release.json')
RELEASE_FREEZE=Path('docs/mitigation_locked_final_evaluation_release_freeze.json')
REGISTRY=Path('docs/mitigation_main_model_registry.json')
OUT=Path('results/mitigation_locked_final_evaluation_v1')
TRAIN=list(range(1,121));ACTUAL=list(range(121,181));TOL=1e-12
FIN=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','trade_fraction','breach','first_crossing','automatic_fraction','protection_trade_fraction','support_review','mitigation_trade_fraction','mitigation_nominal_fees')
PAIR=('log_growth','cagr','mdd','nominal_fees','trade_fraction','automatic_fraction')
PROFILES=(('A',5,.1,'high'),('B',15,.25,'medium'),('C',30,.4,'low'),('D',25,.15,'medium'),('E',6,.35,'high'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def merge(mapping,extra):
    for name,digest in extra.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name or not isinstance(digest,str) or len(digest)!=64:raise ValueError('Canonical relative SHA256 mapping required')
        if name in mapping and mapping[name]!=digest:raise ValueError('Conflicting immutable dependency')
        mapping[name]=digest

def verify(root,mapping):
    for name,digest in mapping.items():
        if sha(Path(root)/name)!=digest:raise RuntimeError('Frozen dependency changed: '+name)


def object_specification(registry):
    q1,q2,contract=registry['Q1'],registry['Q2'],registry['final_evaluation_contract']
    required=dict(profile='C',strategic_product='Balanced',target=[.45,.05,.5],horizon=30,L=.4,cost_rate=.0025,selected_candidate_id='release_0.5_uniform',release_fraction=.5,forecast_rule='uniform',support_mode='calendar',service_alpha=.1,persistent_target_relearning=False)
    if any(q2.get(k)!=v for k,v in required.items()):raise ValueError('Locked V2b parameters may not change')
    if (registry['development_ids']!=TRAIN or contract['training_ids']!=TRAIN or contract['actual_ids']!=ACTUAL or contract['adjust_after_final'] is not False or contract['old_final_overwrite'] is not False or contract['new_final_label']!='previously_exposed_nonblind'):
        raise ValueError('Fixed IDs/nonblind/no-postfinal-adjustment contract required')
    assignments=dict(A='Review',B='Conservative',C='Balanced',D='Review',E='Growth')
    portfolios=dict(Conservative=[.25,.3,.45],Balanced=[.45,.05,.5],Growth=[.65,.05,.3])
    if q1['profile_effective_assignments']!=assignments or q1['portfolios']!=portfolios:raise ValueError('Full own120 Q1 menu changed')
    objects=[]
    for name,rule,target in [('mitigation_main','mitigation',q2['target']),('same_target_B','pending_hold',q2['target']),('same_target_annual','annual',q2['target']),('exact_thirds_annual','annual',[1/3]*3)]:
        objects.append(dict(object_id=name,question='Q2',profile='C',horizon=30,L=.4,liquidity='low',assignment='Balanced',target=target,rule=rule,role='locked_development_candidate' if name=='mitigation_main' else 'hypothetical_reference',q1_suitable=True))
    for name,h,L,liquidity in PROFILES:
        if q1['horizons'][name]!=h or q1['loss_tolerances'][name]!=L:raise ValueError('Original profile H/L required')
        assigned=assignments[name];available=assigned!='Review'
        objects.append(dict(object_id='Q1_'+name,question='Q1',profile=name,horizon=h,L=L,liquidity=liquidity,assignment=assigned,target=portfolios[assigned] if available else None,rule='annual' if available else 'Review',role='static_product_reference' if available else 'Q1_Review',q1_suitable=available))
        objects.append(dict(object_id='Q1_'+name+'_thirds',question='Q1_reference',profile=name,horizon=h,L=L,liquidity=liquidity,assignment='exact_thirds_reference',target=[1/3]*3,rule='annual',role='hypothetical_reference',q1_suitable=None))
    return dict(version='locked-V2b-final-objects-v1',cost_rate=.0025,initial_nominal_wealth=100.,asset_order=['equity','bond','bill'],training_ids=TRAIN,evaluation_ids=ACTUAL,objects=objects,main_candidate=dict(candidate_id='release_0.5_uniform',release_fraction=.5,forecast_rule='uniform',support_mode='calendar'),severe_example_rule='Maximum main full-H MDD, exact tie smallest originalID',scope='previously_exposed_nonblind',post_final_adjustment=False)


def authorize_release(plan,freeze,release,release_freeze,*,protocol_sha,freeze_sha,release_sha):
    if (plan.get('version')!='mitigation-locked-final-evaluation-v1' or plan.get('execution_ready') is not True or plan.get('final_execution_ready') is not False or plan.get('training_ids')!=TRAIN or plan.get('evaluation_ids')!=ACTUAL or plan.get('output_directory')!=str(OUT)):
        raise RuntimeError('Frozen preparation keeps final disabled')
    if not freeze.get('execution_ready') or freeze.get('protocol_sha256')!=protocol_sha:raise RuntimeError('Frozen preparation source required')
    if (release.get('version')!='mitigation-locked-final-evaluation-release-v1' or release.get('final_execution_ready') is not True or release.get('protocol_sha256')!=protocol_sha or release.get('source_freeze_sha256')!=freeze_sha or release.get('object_specification_sha256')!=plan['object_specification_sha256'] or release.get('registry_sha256')!=plan['registry_sha256'] or release.get('training_ids')!=TRAIN or release.get('evaluation_ids')!=ACTUAL or release.get('output_directory')!=str(OUT) or release.get('technical_evidence_sha256')!=plan['technical_evidence_sha256'] or release.get('previously_exposed_nonblind') is not True or release.get('adjust_after_final') is not False):
        raise RuntimeError('Separate exact source/spec/evidence/nonblind final release required')
    if (release_freeze.get('version')!='mitigation-locked-final-release-hash-freeze-v1' or release_freeze.get('execution_ready') is not True or release_freeze.get('release_sha256')!=release_sha or release_freeze.get('protocol_sha256')!=protocol_sha or release_freeze.get('source_freeze_sha256')!=freeze_sha):
        raise RuntimeError('Independent release SHA activation gate required')


def preflight(root=ROOT):
    root=Path(root)
    if not (root/RELEASE).is_file() or not (root/RELEASE_FREEZE).is_file():raise RuntimeError('Final release and release-hash activation do not exist')
    p,f,release,activation=map(readj,(root/PROTOCOL,root/FREEZE,root/RELEASE,root/RELEASE_FREEZE))
    authorize_release(p,f,release,activation,protocol_sha=sha(root/PROTOCOL),freeze_sha=sha(root/FREEZE),release_sha=sha(root/RELEASE))
    snapshot={};merge(snapshot,f['source_sha256']);merge(snapshot,f['input_sha256']);merge(snapshot,{str(x):sha(root/x) for x in (PROTOCOL,FREEZE,RELEASE,RELEASE_FREEZE)})
    must={'scripts/run_mitigation_locked_final_evaluation.py','tests/test_mitigation_locked_final_evaluation.py','src/safeinvest/data.py','src/safeinvest/core.py','src/safeinvest/__init__.py','research/experimental_execution/state_budget_mitigation.py','research/experimental_execution/state_budget_execution.py','research/experimental_execution/forecast_ensemble.py','research/experimental_execution/review_pending.py','research/safeinvest_research/metrics.py'}
    if not must<=set(f['source_sha256']) or str(REGISTRY) not in f['input_sha256']:raise RuntimeError('Complete explicit kernel/source/registry closure required')
    verify(root,snapshot)
    if sha(root/REGISTRY)!=p['registry_sha256']:raise RuntimeError('Locked registry changed')
    registry=readj(root/REGISTRY);spec=object_specification(registry)
    if spec!=p['object_specification'] or canonical_sha(spec)!=p['object_specification_sha256']:raise RuntimeError('Locked final objects changed')
    for group in ('sources_sha256','development_evidence_sha256'):
        for name,digest in registry[group].items():
            if snapshot.get(name)!=digest:raise RuntimeError('Frozen source/input must contain locked registry closure')
    fits=readj(root/p['own_static_fit_path']);selection=readj(root/p['canonical_selections_path'])['0']
    own=fits['c0_refit'];case=own['q2_case']
    if own['refit']['training_ids']!=TRAIN or not case['in_balanced_case'] or case['product']['weights']!=[.45,.05,.5]:
        raise RuntimeError('Main target must be the frozen full120 own-fit Q1 C target')
    if selection['training_ids']!=TRAIN or selection['evaluation_ids']!=TRAIN or selection['refit_job_id']!='c0_refit' or selection['selected_candidate_id']!='release_0.5_uniform' or selection['final_status']!='funded':
        raise RuntimeError('Frozen full120 candidate identity mismatch; no new selection')
    if not p['technical_evidence_sha256'] or not set(p['required_receipt_paths'])<=set(p['technical_evidence_sha256']):
        raise RuntimeError('Every required technical receipt must precede release')
    for name,digest in p['technical_evidence_sha256'].items():
        if snapshot.get(name)!=digest:raise RuntimeError('Frozen input must bind every technical receipt')
        receipt=readj(root/name)
        if receipt.get('status') not in ('passed','complete'):raise RuntimeError('All technical receipts must be complete/passed')
        outputs=receipt.get('outputs_sha256',{})
        merge(snapshot,{str(Path(name).parent/relative):value for relative,value in outputs.items()})
    verify(root,snapshot)
    return p,spec,snapshot


def squeeze(arm):
    return {key:value if key=='cpi' else value[:,0] for key,value in arm.items() if isinstance(value,np.ndarray) and key not in ('targets','loss_tolerances') and (key=='cpi' or value.ndim>=2)}


def evaluate_objects(spec,actual_r,actual_pi,train_r,train_pi,*,engine,paired):
    if len(actual_r)!=60 or len(train_r)!=120:raise ValueError('Exact60 actual/120 training arrays required')
    output={};witness=None
    for obj in spec['objects']:
        if obj['rule']=='Review':output[obj['object_id']]=dict(arm=None,status='q1_review',financial_available=False);continue
        if obj['rule']=='mitigation':
            result=engine(actual_r,actual_pi,train_r,train_pi,obj['target'],L=obj['L'],horizon=obj['horizon'],cost_rate=spec['cost_rate'],release_fraction=.5,forecast_rule='uniform',training_ids=TRAIN,actual_ids=ACTUAL,support_mode='calendar')
            output[obj['object_id']]=result;witness=result['initial_witness']
        else:
            arm=paired(actual_r,actual_pi,np.asarray(obj['target']),obj['L'],horizon=obj['horizon'],cost_rate=spec['cost_rate'])[obj['rule']]
            output[obj['object_id']]=dict(arm=squeeze(arm),status=obj['role'],financial_available=True)
    return output,witness


def path_rows(obj,result,ids,*,metrics):
    if len(ids)!=len(set(ids)) or any(type(s) is not int or s not in ACTUAL for s in ids):raise ValueError('Unique original final IDs required')
    common={**obj,'scope':'previously_exposed_nonblind','execution_status':result['status'],'assigned':bool(result['financial_available'])}
    if not common['assigned']:
        if result['arm'] is not None:raise ValueError('Review cannot invent a physical arm')
        return [dict(**common,scenario_id=sid,**dict.fromkeys(FIN)) for sid in ids]
    arm=result['arm'];h=obj['horizon'];n=len(ids)
    if arm['real_wealth'].shape!=(n,h+1) or arm['fees'].shape!=(n,h):raise ValueError('Original ownH and actual IDs required')
    values=metrics(arm['real_wealth']);rows=[]
    for i,sid in enumerate(ids):
        pending=arm['pending_at_decision'][i];guard=arm.get('guard_triggered',np.zeros((n,h),bool))[i]
        reason=arm.get('gate_reason',np.full((n,h),'',dtype='U64'))[i];mit=reason=='risk_mitigation_then_review';trade=arm['trades'][i]
        first=int(arm['first_crossing'][i]);breach=bool(arm['ever_breached'][i,-1])
        if breach!=(float(values['mdd'][i])-obj['L']>TOL):raise ValueError('Original L-event convention differs')
        rows.append(dict(**common,scenario_id=sid,**{k:float(v[i]) for k,v in values.items()},nominal_fees=float(arm['fees'][i].sum()),trade_fraction=float(trade.mean()),breach=breach,first_crossing=None if first<0 else first,automatic_fraction=float(1-np.mean(pending|guard)),protection_trade_fraction=float(np.mean((reason=='risk_budget_protection')|(mit&trade))),support_review=bool(np.any(arm.get('guard_pending',np.zeros((n,h+1),bool))[i])),mitigation_trade_fraction=float(np.mean(mit&trade)),mitigation_nominal_fees=float(arm['fees'][i][mit].sum())))
    return rows


def summary(rows):
    if not rows or len({r['scenario_id'] for r in rows})!=len(rows):raise ValueError('Preserve original unique population')
    if len({r['object_id'] for r in rows})!=1:raise ValueError('Summaries cannot pool objects/horizons')
    funded=[r for r in rows if r['assigned']];n=len(funded)
    result={k:rows[0][k] for k in ('object_id','question','profile','horizon','L','liquidity','assignment','role','q1_suitable','scope')}
    result.update(original_n=len(rows),financial_n=n,initial_review_n=len(rows)-n,initial_coverage=n/len(rows))
    fields=('breach_count','breach_fraction','mean_log_growth','median_cagr','p10_cagr','mean_mdd','median_mdd','p90_mdd','cvar90_mdd','mean_volatility','mean_nominal_fees','mean_trade_fraction','mean_automatic_fraction','funded_support_review_n','mean_protection_trade_fraction','mean_mitigation_trade_fraction','mean_mitigation_nominal_fees')
    result.update(dict.fromkeys(fields))
    for r in rows:
        if not r['assigned'] and any(r[k] is not None for k in FIN):raise ValueError('Noninvestment finance must be NULL')
    if not n:return result
    avg=lambda k:math.fsum(float(r[k]) for r in funded)/n
    quant=lambda v,p:sorted(v)[max(0,math.ceil(p*len(v))-1)]
    median=lambda v:float(np.median(v))
    losses=[r['mdd'] for r in funded];cagrs=[r['cagr'] for r in funded];mass=.1*n
    tail=math.fsum(x*min(max(mass-i,0.),1.) for i,x in enumerate(sorted(losses,reverse=True)))/mass
    result.update(breach_count=sum(r['breach'] for r in funded),breach_fraction=sum(r['breach'] for r in funded)/n,mean_log_growth=avg('log_growth'),median_cagr=median(cagrs),p10_cagr=quant(cagrs,.1),mean_mdd=avg('mdd'),median_mdd=median(losses),p90_mdd=quant(losses,.9),cvar90_mdd=tail,mean_volatility=avg('volatility'),mean_nominal_fees=avg('nominal_fees'),mean_trade_fraction=avg('trade_fraction'),mean_automatic_fraction=avg('automatic_fraction'),funded_support_review_n=sum(r['support_review'] for r in funded),mean_protection_trade_fraction=avg('protection_trade_fraction'),mean_mitigation_trade_fraction=avg('mitigation_trade_fraction'),mean_mitigation_nominal_fees=avg('mitigation_nominal_fees'))
    return result


def compare(model,reference):
    a,b={r['scenario_id']:r for r in model},{r['scenario_id']:r for r in reference}
    if set(a)!=set(b) or len(a)!=len(model) or len(b)!=len(reference):raise ValueError('Complete same original-ID comparison')
    ids=sorted(s for s in a if a[s]['assigned'])
    if any(not b[s]['assigned'] for s in ids):raise ValueError('Same-ID reference unavailable')
    paired=[dict(scenario_id=s,**{k:a[s][k]-b[s][k] for k in PAIR}) for s in ids]
    differences={k:dict(mean=math.fsum(r[k] for r in paired)/len(ids) if ids else None,median=float(np.median([r[k] for r in paired])) if ids else None,positive_n=sum(r[k]>TOL for r in paired),negative_n=sum(r[k]<-TOL for r in paired),numerical_tie_n=sum(abs(r[k])<=TOL for r in paired)) for k in PAIR}
    parts={k:[] for k in ('neither','model_only','reference_only','both')}
    for s in ids:
        x,y=a[s]['breach'],b[s]['breach'];parts['both' if x and y else 'model_only' if x else 'reference_only' if y else 'neither'].append(s)
    def matched_summary(rows,template):
        if rows:return summary(rows)
        value=summary(template)
        for key in list(value):
            if key.startswith(('mean_','median_','p10_','p90_','cvar','breach_')) or key=='funded_support_review_n':value[key]=None
        value.update(original_n=0,financial_n=0,initial_review_n=0,initial_coverage=None)
        return value
    return dict(model=summary(model),reference_all_original=summary(reference),same_funded_model=matched_summary([a[s] for s in ids],model),same_funded_reference=matched_summary([b[s] for s in ids],reference),same_funded_ids=ids,initial_review_ids=sorted(set(a)-set(ids)),paired_differences=differences,event_id_partition=parts),paired


def severe_example(model_rows,results):
    funded=[r for r in model_rows if r['assigned']]
    if not funded:return dict(scenario_id=None,reason='No main financial account',traces=None)
    worst=max(r['mdd'] for r in funded);sid=min(r['scenario_id'] for r in funded if r['mdd']==worst);index=ACTUAL.index(sid)
    keys=('real_wealth','running_mdd','cpi','holdings','posttrade_holdings','fees','trades','guard_triggered','gate_reason','actual_decision_source')
    traces={name:{k:arm[index] for k,arm in results[name]['arm'].items() if k in keys} for name in ('mitigation_main','same_target_B','same_target_annual','exact_thirds_annual')}
    return dict(scenario_id=sid,rule='max fullH main MDD then min exact-tied originalID',main_mdd=worst,traces=traces,scope='Worst-case illustration, not a population estimate')


def write_csv(path,rows):
    keys=sorted({k for r in rows for k in r if k not in ('target',)})
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as stream:
        with io.TextIOWrapper(stream,newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def run(root=ROOT):
    root=Path(root);p,spec,snapshot=preflight(root);output=root/OUT
    if output.exists():raise RuntimeError('Never overwrite completed or failed final evaluation')
    output.mkdir(parents=True);state=dict(status='running',phase='before_market_parser',started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/PROTOCOL),source_freeze_sha256=sha(root/FREEZE),release_sha256=sha(root/RELEASE),release_freeze_sha256=sha(root/RELEASE_FREEZE),registry_sha256=sha(root/REGISTRY),object_specification_sha256=p['object_specification_sha256'],sources_and_inputs_at_start=snapshot,scope='previously_exposed_nonblind',post_final_parameter_changes=False,new_fits=0,new_selections=0)
    writej(output/'manifest.json',state)
    try:
        sys.path[:0]=[str(root/'src'),str(root/'research')]
        from safeinvest.data import load_paths
        from experimental_execution.review_pending import simulate_paired
        from safeinvest_research.metrics import path_metrics
        from experimental_execution.state_budget_mitigation import simulate_state_budget
        train,train_pi,train_ids=load_paths(root/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
        actual,actual_pi,actual_ids=load_paths(root/'data/SafeInvest_Synthetic_Data.csv',range(121,181),allow_final=True)
        if train_ids.tolist()!=TRAIN or actual_ids.tolist()!=ACTUAL:raise RuntimeError('Fixed training/final identity mismatch')
        results,witness=evaluate_objects(spec,actual,actual_pi,train,train_pi,engine=simulate_state_budget,paired=simulate_paired)
        rows=[];groups=[];arrays={}
        for obj in spec['objects']:
            result=results[obj['object_id']];rows+=path_rows(obj,result,ACTUAL,metrics=path_metrics)
            arm=result['arm'];names=[]
            if arm is not None:
                for key,value in arm.items():
                    if isinstance(value,np.ndarray):arrays[obj['object_id']+'__'+key]=value;names.append(key)
            groups.append(dict(**obj,status=result['status'],financial_available=result['financial_available'],scenario_ids=ACTUAL,training_ids=TRAIN if obj['rule']=='mitigation' else None,array_names=names,omitted_diagnostic_arrays=[]))
        q2=[r for r in rows if r['question']=='Q2'];q1=[r for r in rows if r['question']=='Q1'];q1ref=[r for r in rows if r['question']=='Q1_reference']
        summaries=[summary([r for r in rows if r['object_id']==obj['object_id']]) for obj in spec['objects']]
        comparisons=[];pairs=[]
        for model,reference in [('mitigation_main',x) for x in ('same_target_B','same_target_annual','exact_thirds_annual')]+[('Q1_'+name,'Q1_'+name+'_thirds') for name,_,_,_ in PROFILES]:
            value,paired=compare([r for r in rows if r['object_id']==model],[r for r in rows if r['object_id']==reference]);comparisons.append(dict(model_object=model,reference_object=reference,**value));pairs.extend(dict(model_object=model,reference_object=reference,**r) for r in paired)
        if (len(q2),len(q1),len(q1ref),len(summaries),len(comparisons))!=(240,300,300,14,8):raise RuntimeError('Complete frozen object inventory required')
        for filename,data in [('Q2_paths.csv.gz',q2),('Q1_paths.csv.gz',q1),('Q1_thirds_references.csv.gz',q1ref),('summary.csv.gz',summaries),('paired_paths.csv.gz',pairs)]:write_csv(output/filename,data)
        writej(output/'object_specification.json',spec);writej(output/'trace_groups.json',groups);writej(output/'initial_witness.json',witness);writej(output/'comparisons.json',comparisons)
        writej(output/'severe_example.json',severe_example([r for r in q2 if r['object_id']=='mitigation_main'],results));np.savez_compressed(output/'traces.npz',**arrays)
        verify(root,snapshot)
        state.update(status='complete',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),sources_and_inputs_at_end={name:sha(root/name) for name in snapshot},training_ids_loaded=TRAIN,final_ids_loaded=ACTUAL,final_scenarios_loaded=True,Q2_rows=240,Q1_rows=300,Q1_reference_rows=300,summary_rows=14,comparisons=8,trace_groups=len(groups),outputs_sha256={str(x.relative_to(output)):sha(x) for x in output.iterdir() if x.name!='manifest.json'})
        writej(output/'manifest.json',state)
    except BaseException as error:
        (output/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(error).__name__,error=str(error),automatic_retry=False);writej(output/'manifest.json',state);raise
    print(json.dumps(dict(status='complete',output=str(output),manifest_sha256=sha(output/'manifest.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,final_access_enabled=False,requires_separate_release_and_hash_activation=True)))
