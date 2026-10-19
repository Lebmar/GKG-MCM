"""Fixed-learner sensitivity to five predeclared whole-path partitions.

No policy, threshold, floor, family or method is chosen from these outcomes.
Every outer fit repeats the complete frozen joint Q1/Q2 learner. Compressed
archives retain all evidence; repeated path identities are never IID samples.
"""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
import run_joint_liquidity_refit as joint
from safeinvest.data import load_paths
from experimental_execution.nested_static_adapter import make_context,prepare_strict_family
from experimental_execution.joint_liquidity_static import calibrate_liquidity_static,PROFILES
from experimental_execution.nested_dynamic_selection import BETAS,price_selection
PROTOCOL=ROOT/'docs/refit_stability_protocol.json'
OUT=ROOT/'results/model_review/refit_stability_v1'
CACHE=joint.CACHE
SEED_BASE=87020261018000
REPEATS=5
LEARNERS=joint.LEARNERS
FINANCIAL=joint.FINANCIAL
LIQUIDITY_FLOORS=joint.LIQUIDITY_FLOORS
reference=joint.reference
case_of=joint.case_of


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_json_gz(path,value):Path(path).write_bytes(gzip.compress(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
def write_csv_gz(path,rows):Path(path).write_bytes(gzip.compress(pd.DataFrame(rows).to_csv(index=False,lineterminator='\n').encode(),mtime=0))


def build_design():
    contexts=[];jobs=[];static={}
    for repeat in range(1,REPEATS+1):
        chunks=np.array_split(np.random.default_rng(SEED_BASE+repeat).permutation(np.arange(1,121)),5)
        for fold,chunk in enumerate(chunks,1):
            cid=100*repeat+fold;name=f'stability_r{repeat}_outer{fold}'
            evaluation=sorted(map(int,chunk));training=sorted(set(range(1,121))-set(evaluation))
            c=make_context(training,seed=SEED_BASE+1000+cid,name=name,context_id=cid,outer_test_ids=evaluation)
            c.update(repeat_id=repeat,outer_fold=fold)
            contexts.append(c)
            # The final static calibration shares the parent's four folds,
            # exactly as in the frozen original complete-process learner.
            for f in c['inner_folds']:
                number=f['inner_fold'];jid=f'c{cid}_inner{number}';seed=SEED_BASE+100000+10*cid+number
                job=dict(job_id=jid,parent_context=name,parent_context_id=cid,repeat_id=repeat,outer_fold=fold,
                    dynamic_inner_fold=number,stage='inner',training_ids=f['train_ids'],evaluation_ids=f['test_ids'],static_inner_seed=seed)
                jobs.append(job)
                static[jid]=make_context(f['train_ids'],seed=seed,name=jid,context_id=10000+10*cid+number,outer_test_ids=f['test_ids'])
            job=dict(job_id=f'c{cid}_refit',parent_context=name,parent_context_id=cid,repeat_id=repeat,outer_fold=fold,
                dynamic_inner_fold=None,stage='refit',training_ids=training,evaluation_ids=evaluation,static_inner_seed=c['inner_seed'])
            jobs.append(job)
            static[job['job_id']]={k:v for k,v in c.items() if k not in ('repeat_id','outer_fold')}
    return contexts,jobs,static


def source_manifest():
    paths=joint.source_manifest()
    for rel in ['scripts/run_refit_stability.py','docs/refit_stability_protocol.json','tests/test_refit_stability.py','tests/test_refit_stability_pipeline.py']:
        paths[rel]=sha(ROOT/rel)
    return dict(sorted(paths.items()))


def frozen_inputs(p):
    actual={rel:sha(ROOT/rel) for rel in p['frozen_input_sha256']}
    if actual!=p['frozen_input_sha256']:raise RuntimeError('A declared input changed')
    return actual


def validate_protocol(p):
    contexts,jobs,static=build_design()
    if p['contexts']!=contexts or p['jobs']!=jobs or p['static_contexts']!=static:raise ValueError('Predeclared seed or whole-path identities changed')
    if (p['data']['scenario_ids']!=list(range(1,121)) or p['data']['allow_final'] is not False
        or p['clients']!=PROFILES or p['liquidity_floors']!=joint.LIQUIDITY_FLOORS
        or p['learner_policies']!=list(joint.LEARNERS) or p['betas']!=list(BETAS)
        or p['finite_prices']!=list(joint.priced.FINITE_PRICES)
        or p['original_H']!=30 or p['loss_tolerance']!=.4 or p['cost_rate']!=.0025
        or p['target_updates_enabled'] is not False):raise ValueError('Frozen learner definition changed')
    return contexts,jobs


def aggregate_repeats(client_paths,q1_paths):
    summary=[];q1=[];pairs=[];paired=[]
    for repeat in range(1,REPEATS+1):
        rows=[r for r in client_paths if r['repeat_id']==repeat]
        if len(rows)!=480:raise ValueError('Need four policies on all original120 identities in every repeat')
        summary.extend({'repeat_id':repeat,**r} for r in joint.performance_summary(rows))
        q1.extend({'repeat_id':repeat,**r} for r in joint.q1_summary([r for r in q1_paths if r['repeat_id']==repeat]))
        pr,ps=joint.paired_outputs(rows)
        pairs.extend({'repeat_id':repeat,**r} for r in pr);paired.extend({'repeat_id':repeat,**r} for r in ps)
    return summary,q1,pairs,paired


def scenario_stability(rows):
    frame=pd.DataFrame(rows);result=[]
    for (policy,sid),g in frame.loc[frame.policy.isin(joint.LEARNERS)].groupby(['policy','scenario_id'],sort=False):
        if len(g)!=REPEATS or set(g.repeat_id)!=set(range(1,REPEATS+1)):raise ValueError('Each original scenario needs one record perrepeat')
        auto=g.loc[g.assigned]
        result.append(dict(policy=policy,scenario_id=int(sid),repeat_records_n=len(g),admitted_refits_n=len(auto),
            breached_among_admitted_refits_n=int(auto.breach.astype(bool).sum()) if len(auto) else None,
            min_log_growth=float(auto.log_growth.min()) if len(auto) else None,max_log_growth=float(auto.log_growth.max()) if len(auto) else None,
            min_mdd=float(auto.mdd.min()) if len(auto) else None,max_mdd=float(auto.mdd.max()) if len(auto) else None,
            interpretation='Repeated fits on this same market path, not independent market-risk trials'))
    return result


def main():
    from experimental_execution.risk_priced_fan_rollout import simulate_prices
    from experimental_execution.review_pending import simulate_paired
    started=time.perf_counter();p=json.loads(PROTOCOL.read_text())
    if p.get('execution_ready') is not True:raise RuntimeError('Source/protocol not frozen')
    contexts,jobs=validate_protocol(p);source=source_manifest();inputs=frozen_inputs(p)
    if p['execution_sources_sha256']!={k:v for k,v in source.items() if k!=str(PROTOCOL.relative_to(ROOT))}:raise RuntimeError('Execution sources changed')
    if OUT.exists():raise RuntimeError('Never overwrite a saved study')
    returns,inflation,ids=load_paths(ROOT/p['data']['path'],range(1,121),allow_final=False)
    digest=hashlib.sha256()
    for a in (ids,returns,inflation):digest.update(np.ascontiguousarray(a).tobytes())
    meta=json.loads(CACHE.with_suffix('.json').read_text())
    if digest.hexdigest()!=meta['development_arrays_sha256'] or sha(CACHE)!=meta['npz_sha256']:raise RuntimeError('Guarded development cache identity changed')
    with np.load(CACHE,allow_pickle=False) as f:cache={k:f[k] for k in f.files}
    if not np.array_equal(cache['scenario_ids'],ids):raise RuntimeError('Scenario axis changed')
    metrics={h:{k:cache[f'H{h}__{k}'] for k in joint.reference.METRICS} for h in (5,6,15,25,30)}
    allowed,weights,metrics=prepare_strict_family(cache['weights'],metrics)
    thirds=np.flatnonzero(np.all(cache['weights']==np.ones(3)/3,axis=1))
    if len(thirds)!=1:raise RuntimeError('Unique thirds required')
    eqglobal=int(thirds[0]);eqlocal=int(np.flatnonzero(allowed==eqglobal)[0])
    OUT.mkdir(parents=True)
    bundles={};static_paths=[];training_paths=[];training_summaries=[];validation=[];beta_summaries=[]
    selections_by_context={};client_paths=[];q1_assignments=[];q1_paths=[];fit_admission=[];fit_summaries=[]
    actual_archive={};validation_archive={};groups={'actual':{},'validation':{}};completed=[]
    for c in contexts:
        name=c['name'];cid=c['context_id'];repeat=c['repeat_id'];fold=c['outer_fold']
        context_rows=[]
        for job in [j for j in jobs if j['parent_context_id']==cid]:
            tick=time.perf_counter();jid=job['job_id'];train=job['training_ids'];evaluation=job['evaluation_ids']
            b=calibrate_liquidity_static(metrics,weights,p['clients'],p['static_contexts'][jid],candidate_original_indices=allowed)
            static_paths.extend({**r,'job_id':jid,'repeat_id':repeat,'outer_fold':fold} for r in b['inner_oof_paths'])
            bundles[jid]={k:v for k,v in b.items() if k!='inner_oof_paths'}
            common={**joint.common_fields(job,b),'repeat_id':repeat,'outer_fold':fold};case=common['case_status']
            tr=np.asarray(train)-1;ev=np.asarray(evaluation)-1
            target=None;kB=None;family=None;summaries=[]
            if case=='balanced_case':
                target=np.asarray(b['q2_case']['product']['weights']);col=b['q2_case']['product']['candidate_original232_index']
                kB=int((cache['H30__mdd'][tr,col]-.4>1e-12).sum())
                family=simulate_prices(returns[tr],inflation[tr],returns[tr],inflation[tr],target,joint.priced.PRICES,L=.4,cost_rate=.0025,horizon=30)
                for order in range(9):
                    rows=joint.arm_records(joint.truthful_extract(family,order),train,{**common,**joint.priced.candidate_meta(order,30),'scope':'training_support'})
                    training_paths.extend(rows)
                    s=joint.priced.training_summary(rows,order,30)
                    s.update(job_id=jid,context=name,repeat_id=repeat,outer_fold=fold,static_gamma=common['static_gamma'],q1_effective_assignment=common['q1_effective_assignment'])
                    summaries.append(s);training_summaries.append(s)
                price_choices={str(beta):price_selection(summaries,kB,len(train),beta) for beta in BETAS}
            else:
                for order in range(9):
                    training_paths.extend(joint.null_records(train,{**common,**joint.priced.candidate_meta(order,30),'scope':'training_support'},case,case))
                    training_summaries.append(dict(job_id=jid,context=name,repeat_id=repeat,outer_fold=fold,**joint.priced.candidate_meta(order,30),
                        q1_effective_assignment=common['q1_effective_assignment'],n_training=len(train),breach_count=None,mean_log_growth=None,simulation_status=case))
                price_choices={str(beta):dict(training_beta=beta,selected_candidate_order=None,execution_status=case,execution_review_reason=case) for beta in BETAS}
            bundles[jid]['training_price_selections']=price_choices;bundles[jid]['reference_training_breach_count']=kB
            if job['stage']=='inner':
                orders=[s['selected_candidate_order'] for s in price_choices.values()]
                arms=joint.evaluate_unique_orders(simulate_prices,returns[ev],inflation[ev],returns[tr],inflation[tr],target,orders) if case=='balanced_case' else {}
                for order,arm in arms.items():joint.save_trace(validation_archive,groups['validation'],jid+f'__p{order}',arm,evaluation,common,order)
                for beta in BETAS:
                    order=price_choices[str(beta)]['selected_candidate_order']
                    md={**common,'training_beta':beta,'selected_candidate_order':order,'scope':'inner_validation'}
                    if case!='balanced_case':rows=joint.null_records(evaluation,md,case,case)
                    elif order is None:rows=joint.null_records(evaluation,md,'execution_review','no_admissible_training_price')
                    else:rows=joint.arm_records(arms[order],evaluation,md)
                    validation.extend(rows)
            else:
                pool=[r for r in validation if r['context']==name]
                bs=[dict(context=name,repeat_id=repeat,outer_fold=fold,**joint.joint_beta_summary([r for r in pool if r['training_beta']==beta],beta,c['training_ids'])) for beta in BETAS]
                beta_summaries.extend(bs)
                methods=joint.final_choices(common['q1_effective_assignment'],bs,summaries,price_choices,kB,len(train))
                orders=[x['selected_candidate_order'] for x in methods.values() if x['execution_status']=='automated']
                arms=joint.evaluate_unique_orders(simulate_prices,returns[ev],inflation[ev],returns[tr],inflation[tr],target,orders) if orders else {}
                for policy,choice in methods.items():
                    order=choice['selected_candidate_order'];md={**common,'policy':policy,'training_beta':choice['selected_beta'],'selected_candidate_order':order,'scope':'outer_oof','reference_only':False}
                    if choice['execution_status']=='automated':
                        rows=joint.arm_records(arms[order],evaluation,md)
                        joint.save_trace(actual_archive,groups['actual'],f'c{cid}__{policy}',arms[order],evaluation,md,order)
                    else:rows=joint.null_records(evaluation,md,choice['execution_status'],choice['execution_reason'])
                    context_rows.extend(rows)
                    fit_admission.append(dict(context=name,context_id=cid,repeat_id=repeat,outer_fold=fold,scope='outer_oof',policy=policy,training_n=len(train),evaluation_n=len(evaluation),
                        q1_effective_assignment=common['q1_effective_assignment'],case_status=case,initial_admitted=choice['execution_status']=='automated',
                        **{k:choice[k] for k in ('selected_beta','selected_candidate_order','execution_status','execution_reason')}))
                selections_by_context[name]=dict(training_ids=train,evaluation_ids=evaluation,repeat_id=repeat,outer_fold=fold,case=b['q2_case'],static_gamma=common['static_gamma'],methods=methods)
                assignments,paths=joint.q1_records(b,p['clients'],metrics,evaluation,cid,name)
                q1_assignments.extend({**r,'repeat_id':repeat,'outer_fold':fold} for r in assignments);q1_paths.extend({**r,'repeat_id':repeat,'outer_fold':fold} for r in paths)
                md={**common,'scope':'outer_oof','reference_only':True,'training_beta':None,'selected_candidate_order':None}
                bm={**md,'policy':'annual_then_pending'}
                if case=='balanced_case':
                    arm=joint.reference.squeeze_pending(simulate_paired(returns[ev],inflation[ev],target,.4,horizon=30,cost_rate=.0025)['pending_hold']);arm['targets']=target.copy()
                    context_rows.extend(joint.arm_records(arm,evaluation,bm));joint.save_trace(actual_archive,groups['actual'],f'c{cid}__annual_then_pending',arm,evaluation,bm,kind='annual_then_pending')
                else:context_rows.extend(joint.null_records(evaluation,bm,'reference_unavailable',case))
                arm=joint.reference.squeeze_pending(simulate_paired(returns[ev],inflation[ev],np.ones(3)/3,.4,horizon=30,cost_rate=.0025)['annual']);arm['targets']=np.ones(3)/3
                em={**md,'policy':'equal_thirds_annual','candidate_local_index':eqlocal,'candidate_original232_index':eqglobal,'execution_target_type':'exact_thirds_reference'}
                context_rows.extend(joint.arm_records(arm,evaluation,em));joint.save_trace(actual_archive,groups['actual'],f'c{cid}__equal_thirds_annual',arm,evaluation,em,kind='equal_thirds_annual')
            if source_manifest()!=source or frozen_inputs(p)!=inputs:raise RuntimeError('Source/input changed during task')
            completed.append(dict(job_id=jid,repeat_id=repeat,outer_fold=fold,seconds=time.perf_counter()-tick,case_status=case,gamma=common['static_gamma'],training_n=len(train),evaluation_n=len(evaluation)))
            write_json(OUT/'progress.json',dict(completed_jobs=completed,execution_sources_sha256=source,final_scenarios_loaded=False))
            print(json.dumps(completed[-1]),flush=True)
            if time.perf_counter()-started>60*p['budget_numeric_minutes']:raise RuntimeError('Predeclared numeric budget exhausted: incomplete archive')
        client_paths.extend(context_rows)
        fit_summaries.extend(dict(repeat_id=repeat,outer_fold=fold,context=name,context_id=cid,**r) for r in joint.performance_summary(context_rows))
    summary,q1summ,pairs,paired=aggregate_repeats(client_paths,q1_paths)
    frames={'static_inner_paths':static_paths,'training_candidate_paths':training_paths,'training_candidate_summary':training_summaries,
        'inner_validation_paths':validation,'beta_summaries':beta_summaries,'client_paths':client_paths,'summary':summary,'paired_paths':pairs,'paired_summary':paired,
        'q1_context_assignments':q1_assignments,'q1_effective_paths':q1_paths,'q1_summary':q1summ,'fit_admission':fit_admission,'fit_summary':fit_summaries,'per_scenario_stability':scenario_stability(client_paths)}
    for name,rows in frames.items():write_csv_gz(OUT/(name+'.csv.gz'),rows)
    write_json_gz(OUT/'static_refits.json.gz',bundles);write_json(OUT/'context_selections.json',selections_by_context);write_json(OUT/'trace_groups.json',groups)
    np.savez_compressed(OUT/'actual_traces.npz',**actual_archive);np.savez_compressed(OUT/'validation_traces.npz',**validation_archive)
    if source_manifest()!=source or frozen_inputs(p)!=inputs:raise RuntimeError('Source/input changed during export')
    write_json(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        sources=source,sources_at_end=source_manifest(),input_sha256=inputs,input_sha256_at_end=frozen_inputs(p),
        outputs_sha256={x.name:sha(x) for x in OUT.iterdir() if x.is_file()},scenario_ids_loaded=ids.tolist(),development_arrays_sha256=digest.hexdigest(),
        final_scenarios_loaded=False,training_tasks=len(completed),candidate_count=9,repeat_count=REPEATS,outer_fits=len(contexts),
        primary_method=joint.LEARNERS[1],liquidity_floors=joint.LIQUIDITY_FLOORS,row_counts={name:len(rows) for name,rows in frames.items()},
        limitations=['Repeated partition sensitivity on reused original120 paths, not600 independent market observations or25 independent deployment trials',
            'All125 complete jobs or incomplete; no favorable repeat, price, threshold, floor or method selection',
            'Q1 Review and outsideBalanced and execution Review keep distinct financialNULL; references hypothetical only',
            'Every summary identifies repeat/fit/original scenario; conditional results can involve different admitted cohorts',
            'Fixed unconditional calendar fan remains unvalidated as a true conditional law; no risk guarantee or new blind final evaluation',
            'All selected actual/inner state trees saved; unselected nine-price training trees not retained; all their complete-path metrics retained']))
    print(json.dumps(dict(status='complete',seconds=time.perf_counter()-started,output=str(OUT.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
