"""Mechanical six-fit preview using frozen training/inner evidence only.

Never reads client_paths, actual/outer finance, raw markets or market arrays.
No policy is replayed or fitted. The preview does not estimate OOF coverage,
validated risk, benefit or a new learner's deployment performance.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from experimental_execution.refit_aware_selection import (BETAS,top_inner_set,select_refit_aware,inner_behavior_classes)
from experimental_execution.nested_dynamic_selection import beta_summary,choose_beta

CONTRACT=ROOT/'docs/refit_aware_selection_contract.json'
OUT=ROOT/'results/model_review/refit_aware_selection_preview_v1'
PARENT=ROOT/'results/model_review/nested_dynamic_calibration_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if math.isfinite(value) else None
    return value


def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def same_summary(saved,recomputed):
    for key,value in recomputed.items():
        actual=saved[key]
        if value is None:
            if not pd.isna(actual):raise ValueError('NULL inner evidence was imputed: '+key)
        elif isinstance(value,str):
            if actual!=value:raise ValueError('Inner summary mismatch: '+key)
        elif isinstance(value,(bool,np.bool_)):
            if not isinstance(actual,(bool,np.bool_)) or bool(actual)!=bool(value):raise ValueError('Inner eligibility mismatch')
        elif not np.isclose(actual,value,atol=5e-15,rtol=1e-13):raise ValueError('Inner summary mismatch: '+key)


def behavior_records(frame):
    result=[]
    for row in frame.to_dict('records'):
        order=row['selected_candidate_order']
        if pd.isna(order):order=None
        elif not float(order).is_integer():raise ValueError('Noninteger saved candidate order')
        else:order=int(order)
        result.append({k:row[k] for k in ('training_beta','scenario_id','job_id','case_status','execution_status','assigned')}|
                      {'selected_candidate_order':order})
    return result


def training_summary_for_job(summary,paths,job):
    group=summary.loc[summary.job_id==job['job_id']].sort_values('execution_candidate_order')
    if len(group)!=9:raise ValueError('Missing saved refit candidate evidence')
    result=[];ids=sorted(job['training_ids']);n=len(ids)
    for order,row in enumerate(group.to_dict('records')):
        candidate=paths.loc[(paths.job_id==job['job_id'])&(paths.execution_candidate_order==order)]
        if sorted(candidate.scenario_id.tolist())!=ids or candidate.scenario_id.nunique()!=n or not candidate.assigned.all():
            raise ValueError('Refit candidate must contain every original training path once')
        count=int(candidate.breach.astype(bool).sum())
        if not np.array_equal(candidate.breach.astype(bool).to_numpy(),candidate.mdd.to_numpy()-.4>1e-12):
            raise ValueError('Saved training event/MDD mismatch')
        if row['execution_candidate_order']!=order or row['n_training']!=n or row['breach_count']!=count:
            raise ValueError('Refit candidate summary count mismatch')
        if not np.isclose(candidate.log_growth.mean(),row['mean_log_growth'],atol=5e-15,rtol=1e-13):
            raise ValueError('Refit saved growth mismatch')
        result.append({**row,'execution_candidate_order':order,'n_training':n,'breach_count':count})
    return result


def main():
    if OUT.exists():raise RuntimeError('Preserve preview archive; use a separate version for any new study')
    contract=json.loads(CONTRACT.read_text())
    if contract['scope']['outer_financial_data_read'] or contract['scope']['new_simulation'] or contract['scope']['new_fitting']:
        raise ValueError('Preview contract exceeds training-evidence scope')
    sources=[Path(__file__).resolve(),ROOT/'research/experimental_execution/refit_aware_selection.py',
             ROOT/'research/experimental_execution/nested_dynamic_selection.py',ROOT/'tests/test_refit_aware_selection.py',CONTRACT]
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for rel,digest in contract['frozen_input_sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen preview evidence changed: '+rel)
        before[rel]=digest
    parent_manifest=json.loads((PARENT/'manifest.json').read_text())
    old_helper='research/experimental_execution/nested_dynamic_selection.py'
    if before[old_helper]!=parent_manifest['sources'][old_helper]:raise ValueError('Frozen selection helper changed')
    protocol=json.loads((ROOT/'docs/nested_dynamic_calibration_protocol.json').read_text())
    bundles=json.loads((PARENT/'static_refits.json').read_text())
    old=json.loads((PARENT/'context_selections.json').read_text())
    read=lambda name:pd.read_csv(PARENT/name,float_precision='round_trip')
    saved_betas=read('beta_summaries.csv');validation=read('inner_validation_paths.csv')
    training=read('training_candidate_paths.csv');training_summary=read('training_candidate_summary.csv')
    selected={};sets={};attempts=[];comparison=[]
    refits=[j for j in protocol['jobs'] if j['stage']=='refit']
    if len(refits)!=6:raise ValueError('Require the six saved refit training contexts')
    for job in refits:
        context=job['parent_context'];jid=job['job_id'];bundle=bundles[jid]
        values=validation.loc[validation.context==context];summaries=[]
        for beta in BETAS:
            records=values.loc[values.training_beta==beta].to_dict('records')
            summary=beta_summary(records,beta,job['training_ids'])
            saved=saved_betas.loc[(saved_betas.context==context)&(saved_betas.training_beta==beta)]
            if len(saved)!=1:raise ValueError('Missing or duplicate original beta summary')
            same_summary(saved.iloc[0].to_dict(),summary);summaries.append(summary)
        old_recomputed=choose_beta(summaries)
        if old_recomputed['selected_beta']!=old[context]['beta_selection']['selected_beta']:
            raise ValueError('Original smallest-beta choice does not reproduce')
        inner=top_inner_set(summaries)
        classes=inner_behavior_classes(behavior_records(values))
        assignment=bundle['refit']['effective_assignments']['C']
        # Full Q1 assignment remains intact; no outside-case product is forced.
        if assignment=='Balanced':
            price_summaries=training_summary_for_job(training_summary,training,job)
            kB=bundle['reference_training_breach_count']
        else:price_summaries=None;kB=None
        result=select_refit_aware(assignment,summaries,price_summaries,kB,len(job['training_ids']))
        selected[context]={'job_id':jid,'n_training':len(job['training_ids']),
            'static_gamma':bundle['refit']['selection']['selected_gamma'],'q1_effective_assignments':bundle['refit']['effective_assignments'],
            'q1_product':bundle['q2_case']['q1_product'],'old_selected_beta':old_recomputed['selected_beta'],
            'old_execution_status':old[context]['execution_status'],'new_training_only_selection':result,
            'evidence_scope':'Mechanical saved-training fit admission only; no new outer or validation market execution.'}
        sets[context]={'all_original_inner_summaries':summaries,'risk_eligible_and_top_E':inner,
            'separate_selected_controller_behavior_classes':classes,
            'not_interchangeable':'E is not every eligible beta and is not a behavior-equivalence class.'}
        for attempt in result['refit_attempts']:
            feasible=[x['execution_candidate_order'] for x in attempt['candidate_summaries'] if x['training_eligible']]
            attempts.append(dict(context=context,job_id=jid,beta=attempt['training_beta'],n_training=attempt['n_training'],
                reference_kB=attempt['reference_training_breach_count'],cap=attempt['training_breach_cap'],
                feasible_orders=json.dumps(feasible),selected_order=attempt['selected_candidate_order'],
                maximum_eligible_mean_log_growth=attempt['maximum_eligible_mean_log_growth'],
                execution_status=attempt['execution_status']))
        product=bundle['q2_case']['q1_product']
        comparison.append(dict(context=context,job_id=jid,n_training=len(job['training_ids']),assignment=assignment,
            static_gamma=bundle['refit']['selection']['selected_gamma'],target_weights=json.dumps(product['weights']) if product else None,
            eligible_betas=json.dumps(inner['eligible_betas']),top_E=json.dumps(inner['top_E']),
            old_selected_beta=old_recomputed['selected_beta'],old_selected_order=old[context]['selected_candidate_order'],
            old_training_status=old[context]['execution_status'],preview_selected_beta=result['selected_beta'],
            preview_selected_order=result['selected_candidate_order'],preview_training_status=result['execution_status'],
            preview_reason=result['execution_review_reason'],
            mechanical_training_admission_changed=(old[context]['execution_status']=='automated')!=(result['execution_status']=='automated'),
            outer_financial_outcomes_evaluated=False,OOF_coverage_improvement_estimated=False))
    after={rel:sha(ROOT/rel) for rel in before}
    if before!=after:raise RuntimeError('Read-only preview sources/evidence changed')
    OUT.mkdir(parents=True)
    writej(OUT/'per_context_selection.json',selected);writej(OUT/'inner_sets_and_behavior_classes.json',sets)
    pd.DataFrame(attempts).to_csv(OUT/'refit_E_attempts.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'mechanical_training_comparison.csv',index=False)
    writej(OUT/'manifest.json',dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        contract_sha256=sha(CONTRACT),source_evidence_sha256_start=before,source_evidence_sha256_end=after,
        outputs_sha256={name:sha(OUT/name) for name in ('per_context_selection.json','inner_sets_and_behavior_classes.json',
                'refit_E_attempts.csv','mechanical_training_comparison.csv')},
        contexts=6,refit_E_attempts=len(attempts),new_simulation=False,new_fitting=False,raw_market_data_loaded=False,
        market_arrays_loaded=False,outer_financial_data_read=False,final_scenarios_loaded=False,
        previous_stage_or_tie_modified=False,new_learner_definition=True,
        limitations=['Six-fit mechanical training preview only; not new learner OOF coverage/risk or market performance.',
          'Original eligible set, top inner-evidence set E and per-path selected-controller behavior classes remain distinct.',
          'Every attempted beta belongs to E and <=alpha; original price risk cap, fee, target and training evidence unchanged.',
          'No rejected path investment outcomes are generated, and no annual reference becomes a service fallback.',
          'A future full learner evaluation would require a separately frozen protocol; this preview never relabels old OOF.']))
    print(json.dumps({'status':'passed','contexts':6,'refit_E_attempts':len(attempts),'output':str(OUT.relative_to(ROOT))}),flush=True)


if __name__=='__main__':main()
