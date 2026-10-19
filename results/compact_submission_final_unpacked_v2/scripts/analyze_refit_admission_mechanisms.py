"""Describe frozen refit/admission mechanisms without fitting or simulation.

Run only after diagnostic_plan.json exists. All dynamic outcomes are read from
saved training and inner-validation CSVs. The optional numerical cross-target
calculation uses the frozen annual-policy H30 MDD cache only; it does not score
any rejected outer account, generate new dynamic paths or choose a new rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/model_review/refit_admission_diagnostics_v1'
STAGE='results/model_review/nested_dynamic_calibration_v1/'
CACHE='results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz'
BETAS=(0.,.025,.05,.075,.10)
TOL=1e-12
OUTCOME_FIELDS=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees',
 'nominal_moved','trade_fraction','target_execution_fraction','terminal_equity',
 'terminal_bond','terminal_bill','breach','excess','first_crossing')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def bool_value(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if value in ('True','true'):return True
    if value in ('False','false'):return False
    raise ValueError('Require an observed boolean, not NULL or a numeric substitute')


def original_ids(values):
    raw=list(values)
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 1<=x<=120 for x in raw):
        raise ValueError('Require original integer IDs1..120')
    if len(set(raw))!=len(raw):raise ValueError('Duplicate original IDs')
    return sorted(map(int,raw))


def saved_count(group,expected_ids):
    ids=original_ids(group.scenario_id.tolist())
    if ids!=original_ids(expected_ids):raise ValueError('Saved candidate paths differ from its nominated training IDs')
    assigned=group.assigned.map(bool_value)
    if not assigned.all():
        if assigned.any():raise ValueError('Candidate training group mixes assigned and NULL outcomes')
        if group[['log_growth','mdd','breach']].notna().any().any():raise ValueError('Unassigned training outcomes must be NULL')
        return None
    observed=group.breach.map(bool_value).to_numpy()
    event=group.mdd.to_numpy(float)-.4>TOL
    if not np.array_equal(observed,event):raise ValueError('Saved whole-H event differs from MDD')
    return int(observed.sum())


def cap_ledger(n,beta,kB,counts,growth):
    """Exact integer eligibility and one-band objective ties; no new choice."""
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<=0 or isinstance(beta,(bool,np.bool_)) or beta not in BETAS:
        raise ValueError('Invalid predeclared n or beta')
    if isinstance(kB,bool) or not isinstance(kB,(int,np.integer)) or not 0<=kB<=n:
        raise ValueError('Invalid reference event count')
    if set(counts)!=set(range(9)) or set(growth)!=set(range(9)):
        raise ValueError('Require all nine saved nominal price candidates')
    if any(isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<=n for k in counts.values()):
        raise ValueError('Candidate counts must be integer full-training counts')
    if any(not np.isfinite(g) for g in growth.values()):raise ValueError('Growth must be observed')
    raw=math.floor(beta*n+TOL);cap=min(kB,raw);minimum=min(counts.values())
    feasible=[i for i in range(9) if counts[i]<=cap]
    maximum=max((growth[i] for i in feasible),default=None)
    ties=[i for i in feasible if growth[i]>=maximum-TOL]
    return dict(beta=float(beta),n=int(n),raw_beta_floor=raw,reference_count=int(kB),effective_cap=int(cap),
        cap_binding='equal' if kB==raw else 'reference' if kB<raw else 'beta_floor',
        minimum_candidate_count=int(minimum),minimum_count_orders=[i for i in range(9) if counts[i]==minimum],
        minK_minus_cap=int(minimum-cap),feasible_orders=feasible,growth_tied_feasible_orders=ties,
        canonical_selected_order=ties[0] if ties else None,maximum_feasible_growth=maximum,
        minimum_count_fraction=minimum/n,nominal_beta_needed_for_minK=minimum/n,
        minimum_count_attainable_under_reference=minimum<=kB)


def _token(value):
    if value is None or (isinstance(value,(float,np.floating)) and np.isnan(value)):return None
    if isinstance(value,(bool,np.bool_)):return ('bool',bool(value))
    if isinstance(value,(int,np.integer)):return ('number',float(value).hex())
    if isinstance(value,(float,np.floating)):return ('number',float(value).hex())
    return ('str',str(value))


def signature(records,fields):
    ordered=sorted(records,key=lambda r:(str(r.get('job_id','')),int(r['scenario_id'])))
    rows=[[_token(r.get(k)) for k in fields] for r in ordered]
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def equivalence_classes(named_records,fields):
    groups={}
    for name,rows in named_records.items():
        sig=signature(rows,fields);groups.setdefault(sig,[]).append(name)
    return [dict(class_id=i+1,members=members,saved_vector_sha256=sig)
            for i,(sig,members) in enumerate(groups.items())]


def _event_map(group,expected_ids):
    saved_count(group,expected_ids)
    return {int(r.scenario_id):bool_value(r.breach) for r in group.itertuples(index=False)}


def count_decomposition(inner_events,refit_events):
    """Algebra on saved ID-indexed events; not a causal policy decomposition."""
    old=original_ids(inner_events);new=original_ids(refit_events)
    if not set(old)<=set(new):raise ValueError('Inner training must be a subset of its refit training set')
    if any(not isinstance(v,(bool,np.bool_)) for v in [*inner_events.values(),*refit_events.values()]):
        raise ValueError('Events must be booleans')
    added=sorted(set(new)-set(old))
    upward=[i for i in old if not inner_events[i] and refit_events[i]]
    downward=[i for i in old if inner_events[i] and not refit_events[i]]
    added_events=[i for i in added if refit_events[i]]
    k_old=sum(inner_events.values());retained=sum(refit_events[i] for i in old);k_new=sum(refit_events.values())
    if k_new-k_old!=len(upward)-len(downward)+len(added_events):raise AssertionError('Count ledger failed')
    return dict(inner_n=len(old),refit_n=len(new),retained_n=len(old),added_n=len(added),
        inner_count=int(k_old),refit_retained_count=int(retained),refit_added_count=len(added_events),
        refit_count=int(k_new),count_delta=int(k_new-k_old),retained_net_flip_count=len(upward)-len(downward),
        upward_flip_ids=upward,downward_flip_ids=downward,added_event_ids=added_events,
        retained_ids=old,added_ids=added)


def annual_target_cross(mdd,inner_ids,refit_ids,inner_target,refit_target,L=.4):
    """Four saved annual-cache cells, plus both exact decomposition orders."""
    inner=original_ids(inner_ids);refit=original_ids(refit_ids)
    if not set(inner)<=set(refit):raise ValueError('Inner IDs must be retained in refit')
    mdd=np.asarray(mdd)
    if mdd.ndim!=2 or mdd.shape[0]!=120 or not np.isfinite(mdd).all():raise ValueError('Require finite annual-cache original120 rows')
    for index in (inner_target,refit_target):
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<mdd.shape[1]:raise ValueError('Invalid target cache index')
    added=sorted(set(refit)-set(inner));a=mdd[:,inner_target]-L>TOL;b=mdd[:,refit_target]-L>TOL
    def k(v,ids):return int(v[np.asarray(ids,dtype=int)-1].sum()) if ids else 0
    ta,tb,sa,sb=k(a,inner),k(b,inner),k(a,refit),k(b,refit)
    na,nb=k(a,added),k(b,added)
    result=dict(k_T_inner_target=ta,k_T_refit_target=tb,k_S_inner_target=sa,k_S_refit_target=sb,
        added_k_inner_target=na,added_k_refit_target=nb,total_count_delta=sb-ta,
        target_first_on_retained=tb-ta,addition_after_target=nb,
        addition_first_under_inner_target=na,target_after_addition=sb-sa,
        target_by_added_paths_interaction=nb-na,
        T_inner_event_ids=[i for i in inner if a[i-1]],T_refit_event_ids=[i for i in inner if b[i-1]],
        added_inner_event_ids=[i for i in added if a[i-1]],added_refit_event_ids=[i for i in added if b[i-1]])
    if result['total_count_delta']!=result['target_first_on_retained']+result['addition_after_target'] or result['total_count_delta']!=result['addition_first_under_inner_target']+result['target_after_addition']:
        raise AssertionError('Annual cross-target accounting failed')
    return result


def recorded_pair_difference(left,right):
    """Full recorded path vectors can differ despite equal aggregate means."""
    left=sorted(left,key=lambda r:int(r['scenario_id']));right=sorted(right,key=lambda r:int(r['scenario_id']))
    if original_ids(r['scenario_id'] for r in left)!=original_ids(r['scenario_id'] for r in right):
        raise ValueError('Pairwise recorded outcomes require the same original IDs')
    changed={f:[] for f in OUTCOME_FIELDS};any_ids=[]
    for a,b in zip(left,right,strict=True):
        differences=[f for f in OUTCOME_FIELDS if _token(a.get(f))!=_token(b.get(f))]
        if differences:any_ids.append(int(a['scenario_id']))
        for f in differences:changed[f].append(int(a['scenario_id']))
    return dict(recorded_outcome_equal=not any_ids,paths_with_any_recorded_difference=len(any_ids),
        differing_path_ids=any_ids,field_difference_counts={f:len(v) for f,v in changed.items()})


def _group_rows(rows,fields):
    groups={}
    for row in rows:
        key=json.dumps(clean([row.get(f) for f in fields]),sort_keys=True)
        groups.setdefault(key,[]).append(row['beta'])
    return [dict(values=json.loads(key),betas=members) for key,members in groups.items()]


def analyze(protocol,static,context_selections,training,summary,validation,beta_summaries,mdd,weights,scenario_ids):
    if not np.array_equal(scenario_ids,np.arange(1,121)) or np.shape(mdd)!=(120,232) or np.shape(weights)!=(232,3):
        raise ValueError('Require the frozen original120 x232 annual family')
    jobs=protocol['jobs'];task_rows=[];task_info={};platforms={};price_equivalence={};task_counts={}
    for job in jobs:
        jid=job['job_id'];bundle=static[jid];case=bundle['q2_case'];product=case.get('q1_product')
        T=original_ids(job['training_ids']);n=len(T)
        info=dict(job_id=jid,context=job['parent_context'],context_id=job['parent_context_id'],stage=job['stage'],
            dynamic_inner_fold=job['dynamic_inner_fold'],n=n,training_ids=T,case_status=case['status'],
            assignment=case['effective_assignment'],static_gamma=bundle['refit']['selection']['selected_gamma'],
            target_original232_index=product['candidate_original232_index'] if product else None,
            target_weights=product['weights'] if product else None)
        if product and not np.array_equal(np.asarray(product['weights']),weights[product['candidate_original232_index']]):
            raise ValueError('Saved target does not match frozen cache column')
        task_info[jid]=info
        group=summary.loc[summary.job_id==jid].sort_values('execution_candidate_order')
        if len(group)!=9:raise ValueError('Missing candidate summaries: '+jid)
        counts={};growth={};outcomes={}
        for order in range(9):
            paths=training.loc[(training.job_id==jid)&(training.execution_candidate_order==order)].sort_values('scenario_id')
            count=saved_count(paths,T)
            if case['status']=='balanced_case':
                if count is None:raise ValueError('Balanced training outcome is NULL')
                s=group.loc[group.execution_candidate_order==order].iloc[0]
                if int(s.breach_count)!=count or int(s.n_training)!=n:raise ValueError('Saved training summary count mismatch')
                if not np.isclose(paths.log_growth.mean(),s.mean_log_growth,atol=5e-15,rtol=1e-13):raise ValueError('Saved growth summary mismatch')
                counts[order]=count;growth[order]=float(s.mean_log_growth)
            elif count is not None:raise ValueError('Non-Balanced case acquired training financial outcomes')
            outcomes[order]=paths.to_dict('records')
        if counts:
            index=info['target_original232_index'];kB=int((mdd[np.asarray(T)-1,index]-.4>TOL).sum())
            if kB!=bundle['reference_training_breach_count']:raise ValueError('Same-target annual reference differs from saved count')
            task_counts[jid]=(counts,growth,kB)
        for beta in BETAS:
            saved=bundle['training_price_selections'][str(beta)]
            if counts:
                ledger=cap_ledger(n,beta,kB,counts,growth)
                if ledger['effective_cap']!=saved['training_breach_cap'] or ledger['canonical_selected_order']!=saved['selected_candidate_order']:
                    raise ValueError('Read-only eligibility/canonical reconstruction differs from frozen selection')
                saved_feasible=[r['execution_candidate_order'] for r in saved['candidate_summaries'] if r['training_eligible']]
                if ledger['feasible_orders']!=saved_feasible:raise ValueError('Feasible price set differs')
            else:
                ledger=dict(beta=beta,n=n,raw_beta_floor=math.floor(beta*n+TOL),reference_count=None,effective_cap=None,
                    cap_binding='not_balanced_case_or_risk_review',minimum_candidate_count=None,minimum_count_orders=[],
                    minK_minus_cap=None,feasible_orders=[],growth_tied_feasible_orders=[],canonical_selected_order=None,
                    maximum_feasible_growth=None,minimum_count_fraction=None,nominal_beta_needed_for_minK=None,
                    minimum_count_attainable_under_reference=None)
            task_rows.append({**{k:v for k,v in info.items() if k!='training_ids'},**ledger,
                'saved_execution_status':saved['execution_status'],'saved_selected_order':saved['selected_candidate_order']})
        these=task_rows[-5:]
        platforms[jid]=dict(raw_floor_classes=_group_rows(these,['raw_beta_floor']),
            effective_cap_classes=_group_rows(these,['effective_cap']),feasible_set_classes=_group_rows(these,['feasible_orders']),
            selected_order_classes=_group_rows(these,['saved_selected_order','saved_execution_status']))
        price_equivalence[jid]=dict(complete_recorded_outcome_classes=equivalence_classes(outcomes,['scenario_id',*OUTCOME_FIELDS]),
            candidate_catalog=group[['execution_candidate_order','execution_candidate_id','price_kind','annual_price']].to_dict('records'),
            pairwise_recorded_differences=[dict(left_order=i,right_order=j,
                mean_growth_tied_within_existing_tolerance=(abs(growth[i]-growth[j])<=TOL) if counts else None,
                **recorded_pair_difference(outcomes[i],outcomes[j])) for i in range(9) for j in range(i+1,9)],
            interpretation='Identical saved complete-path metrics do not prove identical actions/Q or identify continuous eta; unselected training action traces were not retained.',
            per_beta_growth_ties={str(r['beta']):r['growth_tied_feasible_orders'] for r in these})
    beta_equiv={}
    for context,selection in context_selections.items():
        context_rows=validation.loc[validation.context==context]
        records={str(beta):context_rows.loc[context_rows.training_beta==beta].to_dict('records') for beta in BETAS}
        beta_equiv[context]=dict(selected_beta=selection['beta_selection']['selected_beta'],
            selected_controller_vector_classes=equivalence_classes(records,['job_id','scenario_id','case_status','execution_status',
                'assigned','selected_candidate_order','q1_effective_assignment','candidate_original232_index']),
            recorded_financial_vector_classes=equivalence_classes(records,['job_id','scenario_id','assigned',*OUTCOME_FIELDS]),
            saved_beta_summaries=beta_summaries.loc[beta_summaries.context==context].to_dict('records'),
            interpretation='Same selected-order vector within fixed jobs implies the same frozen deterministic selected controllers. Financial-vector equality alone is weaker. Classes are repeated-development observations, not uniquely identified service parameters.')
    changes=[];beta_changes=[];dynamic=[];annual=[];witnesses={}
    lookup={(r['job_id'],r['beta']):r for r in task_rows}
    for job in jobs:
        if job['stage']!='inner':continue
        jid=job['job_id'];rid=f"c{job['parent_context_id']}_refit";a=task_info[jid];b=task_info[rid]
        base=dict(context=job['parent_context'],inner_job=jid,refit_job=rid,inner_n=a['n'],refit_n=b['n'],
            inner_gamma=a['static_gamma'],refit_gamma=b['static_gamma'],inner_assignment=a['assignment'],refit_assignment=b['assignment'],
            inner_case=a['case_status'],refit_case=b['case_status'],inner_target=a['target_original232_index'],refit_target=b['target_original232_index'],
            inner_weights=a['target_weights'],refit_weights=b['target_weights'],
            same_target=(a['target_original232_index']==b['target_original232_index']) if a['target_weights'] and b['target_weights'] else None,
            target_L1=float(np.abs(np.asarray(a['target_weights'])-np.asarray(b['target_weights'])).sum()) if a['target_weights'] and b['target_weights'] else None,
            retained_n=len(set(a['training_ids'])&set(b['training_ids'])),added_n=len(set(b['training_ids'])-set(a['training_ids'])),
            removed_n=len(set(a['training_ids'])-set(b['training_ids'])),
            same_training_fan=False,controller_identity_preserved=False)
        if base['removed_n']:raise ValueError('Refit lost an inner training path')
        changes.append(base)
        for beta in BETAS:
            x,y=lookup[jid,beta],lookup[rid,beta]
            numeric=x['minimum_candidate_count'] is not None and y['minimum_candidate_count'] is not None
            beta_changes.append({**base,'beta':beta,'inner_minK':x['minimum_candidate_count'],'refit_minK':y['minimum_candidate_count'],
                'inner_cap':x['effective_cap'],'refit_cap':y['effective_cap'],'raw_floor_change':y['raw_beta_floor']-x['raw_beta_floor'],
                'minK_change':y['minimum_candidate_count']-x['minimum_candidate_count'] if numeric else None,
                'effective_cap_change':y['effective_cap']-x['effective_cap'] if numeric else None,
                'gap_change':y['minK_minus_cap']-x['minK_minus_cap'] if numeric else None,
                'inner_minimum_orders':x['minimum_count_orders'],'refit_minimum_orders':y['minimum_count_orders'],
                'inner_feasible':x['feasible_orders'],'refit_feasible':y['feasible_orders'],
                'lost_feasible':sorted(set(x['feasible_orders'])-set(y['feasible_orders'])),
                'gained_feasible':sorted(set(y['feasible_orders'])-set(x['feasible_orders'])),
                'inner_selected_order':x['saved_selected_order'],'refit_selected_order':y['saved_selected_order'],
                'interpretation':'Envelope-count arithmetic; minimizing price, target, fan and endogenous states can all change. Not a causal mechanism estimate.'})
        if a['case_status']=='balanced_case' and b['case_status']=='balanced_case':
            for order in range(9):
                old=training.loc[(training.job_id==jid)&(training.execution_candidate_order==order)]
                new=training.loc[(training.job_id==rid)&(training.execution_candidate_order==order)]
                decomposition=count_decomposition(_event_map(old,a['training_ids']),_event_map(new,b['training_ids']))
                key=f'{jid}__to__{rid}__price{order}'
                witnesses[key]=decomposition
                dynamic.append({**base,'nominal_candidate_order':order,
                    **{k:v for k,v in decomposition.items() if not k.endswith('_ids')},
                    'upward_flip_count':len(decomposition['upward_flip_ids']),'downward_flip_count':len(decomposition['downward_flip_ids']),
                    'witness_key':key,'interpretation':'Same nominal eta/order, different fitted controller. Common market IDs do not hold state/target/forecast fixed.'})
        if a['target_original232_index'] is not None and b['target_original232_index'] is not None:
            cross=annual_target_cross(mdd,a['training_ids'],b['training_ids'],a['target_original232_index'],b['target_original232_index'])
            key=f'{jid}__to__{rid}__annual_cross';witnesses[key]=cross
            annual.append({**base,**{k:v for k,v in cross.items() if not k.endswith('_ids')},'witness_key':key,
                'interpretation':'Four frozen fixed-annual-policy cache cells; target-first/addition-first ledgers are order-dependent, not a decomposition of dynamic policy risk.'})
    focus={}
    for context in ('outer_fold_2','outer_fold_4'):
        choice=context_selections[context];beta=choice['beta_selection']['selected_beta'];cid=next(j['parent_context_id'] for j in jobs if j['parent_context']==context)
        rid=f'c{cid}_refit'
        focus[context]=dict(saved_execution_status=choice['execution_status'],saved_execution_reason=choice['execution_reason'],
            selected_beta=beta,refit_task=task_info[rid],
            selected_beta_refit_ledger=lookup[rid,beta] if beta is not None else None,
            four_inner_ledgers_at_selected_beta=[lookup[f'c{cid}_inner{i}',beta] for i in range(1,5)] if beta is not None else None,
            all_beta_refit_ledgers=[lookup[rid,x] for x in BETAS],
            inner_refit_selected_beta_comparisons=[r for r in beta_changes if r['context']==context and r['beta']==beta],
            fixed_annual_target_cross=[r for r in annual if r['context']==context],
            identification_boundary='The saved ledger identifies why the final rule declares no eligible saved price. It cannot uniquely attribute the changed dynamic event counts to target, fan, added paths or feedback states.')
    findings=dict(task_case_counts=dict(Counter(x['case_status'] for x in task_info.values())),
        contexts={name:dict(selected_beta=v['beta_selection']['selected_beta'],execution_status=v['execution_status'],execution_reason=v['execution_reason']) for name,v in context_selections.items()},
        behavior_equivalent_beta_groups={k:[g['members'] for g in v['selected_controller_vector_classes'] if len(g['members'])>1] for k,v in beta_equiv.items()},
        task_betas_with_multiple_growth_tied_prices=sum(len(x['growth_tied_feasible_orders'])>1 for x in task_rows),
        fixed_annual_cross_pairs_with_order_interaction=sum(r['target_by_added_paths_interaction']!=0 for r in annual),
        interpretation=[
          'Beta maps to integer count platforms and then to finite feasible/selected-controller classes. A canonical smallest tied beta does not identify a continuous risk budget or client preference.',
          'Eta is a finite technical candidate label. Training objective ties and recorded outcome equivalence need not identify a unique eta; even a unique finite training winner is not a true KKT/customer parameter.',
          'At a fixed state, observed action inequalities generally constrain eta to an interval; this script does not reconstruct such intervals because unselected training states/Q are absent and state paths depend on eta.',
          'The selected beta is not guaranteed to admit a price after static target and training fan are refitted. That is a declared Review outcome, not a code fallback failure.',
          'Common-ID dynamic event flips combine changed target, fan and endogenous states; the exact arithmetic ledger is descriptive, not separately identified causal effects.',
          'No new dynamic outcome or rejected outer financial result is computed. No beta, eta, tie, target, coverage or liquidity rule is changed.'])
    return dict(task_rows=task_rows,platforms=platforms,beta_equiv=beta_equiv,price_equivalence=price_equivalence,
        changes=changes,beta_changes=beta_changes,dynamic=dynamic,annual=annual,witnesses=witnesses,focus=focus,findings=findings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();out=args.output.resolve()
    plan_path=out/'diagnostic_plan.json'
    if (out/'manifest.json').exists():raise RuntimeError('Preserve completed diagnostics; do not overwrite')
    plan=json.loads(plan_path.read_text())
    if plan['scope']['new_dynamic_simulation'] or plan['scope']['new_fitting'] or plan['scope']['final_scenarios_loaded']:
        raise ValueError('Diagnostic plan exceeds read-only development scope')
    sources=[Path(__file__).resolve(),ROOT/'tests/test_refit_admission_mechanisms.py',plan_path]
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for rel,digest in plan['frozen_input_sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen evidence changed: '+rel)
        before[rel]=digest
    parent=json.loads((ROOT/(STAGE+'manifest.json')).read_text())
    if parent['scenario_ids_loaded']!=list(range(1,121)) or parent['final_scenarios_loaded'] is not False:
        raise ValueError('Wrong parent data scope')
    if parent['sources']!=parent['sources_at_end'] or parent['input_sha256']!=parent['input_sha256_at_end']:
        raise ValueError('Unstable parent fingerprints')
    for rel,digest in parent['sources'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen model source changed: '+rel)
        before[rel]=digest
    for name,digest in parent['outputs_sha256'].items():
        rel=STAGE+name
        if rel in plan['frozen_input_sha256'] and plan['frozen_input_sha256'][rel]!=digest:
            raise RuntimeError('Saved evidence differs from its parent manifest')
    audit=json.loads((ROOT/(STAGE+'independent_audit.json')).read_text())
    if audit['status']!='passed':raise ValueError('Parent audit did not pass')
    protocol=json.loads((ROOT/'docs/nested_dynamic_calibration_protocol.json').read_text())
    static=json.loads((ROOT/(STAGE+'static_refits.json')).read_text())
    selection=json.loads((ROOT/(STAGE+'context_selections.json')).read_text())
    read=lambda name:pd.read_csv(ROOT/(STAGE+name),float_precision='round_trip')
    training=read('training_candidate_paths.csv');summary=read('training_candidate_summary.csv')
    validation=read('inner_validation_paths.csv');beta_summaries=read('beta_summaries.csv')
    if CACHE not in parent['input_sha256'] or before[CACHE]!=parent['input_sha256'][CACHE]:raise ValueError('Annual cache not the frozen parent input')
    with np.load(ROOT/CACHE,allow_pickle=False) as cache:
        mdd=cache['H30__mdd'];weights=cache['weights'];ids=cache['scenario_ids']
    result=analyze(protocol,static,selection,training,summary,validation,beta_summaries,mdd,weights,ids)
    tables={'task_beta_mechanisms.csv':result['task_rows'],'inner_refit_changes.csv':result['changes'],
      'inner_refit_beta_comparison.csv':result['beta_changes'],'dynamic_training_count_decomposition.csv':result['dynamic'],
      'fixed_annual_target_cross.csv':result['annual']}
    documents={'task_beta_platforms.json':result['platforms'],'inner_beta_equivalence.json':result['beta_equiv'],
      'price_outcome_equivalence.json':result['price_equivalence'],'decomposition_witnesses.json':result['witnesses'],
      'focus_contexts.json':result['focus'],'findings.json':result['findings']}
    for name,rows in tables.items():
        encoded=[{k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,tuple,dict)) else v for k,v in r.items()} for r in rows]
        pd.DataFrame(encoded).to_csv(out/name,index=False)
    for name,value in documents.items():write_json(out/name,value)
    after={rel:sha(ROOT/rel) for rel in before}
    if before!=after:raise RuntimeError('Read-only sources or evidence changed during diagnostics')
    cachemeta=json.loads((ROOT/Path(CACHE).with_suffix('.json')).read_text())
    write_json(out/'manifest.json',dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(plan_path),source_evidence_sha256_start=before,source_evidence_sha256_end=after,
        outputs_sha256={n:sha(out/n) for n in [*tables,*documents]},
        row_counts={n:len(v) for n,v in tables.items()},parent_manifest_sha256=sha(ROOT/(STAGE+'manifest.json')),
        provenance=dict(frozen_annual_cache=CACHE,cache_sha256=before[CACHE],
            development_arrays_sha256=cachemeta['development_arrays_sha256'],
            parent_raw_data_sha256=parent['input_sha256'].get('data/SafeInvest_Synthetic_Data.csv'),
            raw_market_csv_loaded=False,annual_cache_arrays_loaded=['scenario_ids','weights','H30__mdd'],
            dynamic_outcome_sources='saved training_candidate_paths and inner_validation_paths only'),
        final_scenarios_loaded=False,new_dynamic_simulation=False,new_fitting=False,
        rejected_outer_financial_outcomes_computed=False,policy_or_tie_changed=False,
        scope='Read-only descriptive diagnosis on reused development evidence; no new policy selection or causal attribution'))
    print(json.dumps(dict(status='passed',row_counts={n:len(v) for n,v in tables.items()},output=str(out.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
