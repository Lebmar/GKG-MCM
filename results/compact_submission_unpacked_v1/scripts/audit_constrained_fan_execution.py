"""Independent fixed-scope constrained fan audit; no reviewed kernel imports.

Uses previously frozen independent100-step fee accounting and scalar fan only.
All model decisions are re-derived from saved net Q and bounded integer counts;
Q levels are independently reconstructed only at the prospective fixed slots.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_simple_execution_comparison as accounting
import audit_benefit_cost_execution as inherited
from audit_fan_rollout_execution import scalar_fan, reference_actual

PROTOCOL=ROOT/'docs/constrained_fan_execution_protocol.json'
AUDIT_PLAN=ROOT/'docs/constrained_fan_execution_audit_plan.json'
OUT=ROOT/'results/model_review/constrained_fan_execution_v1'
CACHE=inherited.CACHE
METHODS={'constrained_net_growth_rollout':'fixed_alpha_B_continuation'}
PRIMARY=next(iter(METHODS));REFERENCES=inherited.REFERENCES;TOL=1e-12
sha=inherited.sha;readj=inherited.readj;writej=inherited.writej
QFLOATS=('qg_hold','qg_restore','predicted_growth_advantage_restore','predicted_breach_count_delta_restore',
         'quoted_nominal_fee','quoted_nominal_moved','quoted_fee_fraction','quoted_log_fee_penalty')
QCOUNTS=('qbad_hold_count','qbad_restore_count','forecast_n','local_breach_cap')
QBOOLS=('hold_model_feasible','restore_model_feasible','local_model_infeasible','estimate_available',
        'restoration_predicted_growth_sacrifice','economic_improvement','risk_cost_justification','repair_meets_cap','local_constraint_applied')
QSTRINGS=('gate_reason','decision_reason','decision_detail')
# The immutable production array is U40 while its intended inception detail
# has41 characters. Audit its exact serialized prefix and disclose this.
INITIAL_DETAIL_SERIALIZED='mandatory_initial_restore_diagnostic_only'[:40]
EXTRA_REFITS=('c101_refit','c202_refit','c303_refit','c404_refit','c505_refit','c0_refit')
BREAKDOWN=('optional_restores','economic_improvement_restores','risk_repair_restores','growth_sacrificing_restores','locally_infeasible_decisions')


def independent_choice(qh,qr,kh,kr,n):
    """Construct the feasible set, then min violation, then net-Q/tie hold."""
    n=inherited._integer(n,'Positive common original training count',1)
    kh=inherited._integer(kh,'Integer hold breach count');kr=inherited._integer(kr,'Integer restore breach count')
    if max(kh,kr)>n or isinstance(qh,(bool,np.bool_)) or isinstance(qr,(bool,np.bool_)) or not np.isfinite([qh,qr]).all():
        raise ValueError('Finite net Q and bounded integer counts required')
    cap=math.floor(.1*n+TOL);actions=[('hold',float(qh),kh),('restore',float(qr),kr)]
    feasible=[a for a in actions if a[2]<=cap];infeasible=not feasible
    pool=feasible if feasible else [a for a in actions if a[2]==min(kh,kr)]
    best=max(a[1] for a in pool)
    hold=next((a for a in pool if a[0]=='hold'),None)
    selected=hold if hold is not None and best-hold[1]<=TOL else max(pool,key=lambda a:a[1])
    restore=selected[0]=='restore'
    fh,fr=kh<=cap,kr<=cap;delta=float(qr)-float(qh)
    if not np.isfinite(delta):raise ValueError('Finite net Q advantage required')
    reason='hold'
    if restore:
        reason='least_violation_restore' if infeasible else 'growth_improvement' if fh else 'risk_constraint_repair'
    detail=reason if restore else 'least_violation_hold' if infeasible else 'only_hold_model_feasible' if not fr else 'both_feasible_growth_or_tie_hold'
    economic=restore and delta>TOL;risk_cost=restore and not economic
    if risk_cost and not kr<kh:raise AssertionError('Nonpositive/tie net restore must strictly reduce integer risk count')
    return dict(action=selected[0],gate_reason=reason,decision_detail=detail,local_breach_cap=cap,
        hold_model_feasible=fh,restore_model_feasible=fr,local_model_infeasible=infeasible,
        predicted_growth_advantage_restore=delta,predicted_breach_count_delta_restore=float(kr-kh),
        restoration_predicted_growth_sacrifice=restore and delta < -TOL,economic_improvement=economic,
        risk_cost_justification=risk_cost,repair_meets_cap=restore and fr)


def replay_model(r,pi,target,training_n,e,ledger=None):
    """Derive every action from Q/counts, then independently execute accounting."""
    ledger=ledger or accounting.AuditLedger();reasons=[];classes=[]
    def action(t,x,cp,peak,worst,bad):
        active=not bad
        accounting.maybe(e['estimate_available'][t],active,ledger,'decision/estimate_available')
        accounting.maybe(e['local_constraint_applied'][t],active and t>0,ledger,'decision/constraint_applied')
        if bad:
            ledger.require(all(np.isnan(e[k][t]) for k in QFLOATS),'decision/pending_float_NULL')
            ledger.require(all(e[k][t]==-1 for k in QCOUNTS),'decision/pending_integer_sentinel')
            ledger.require(all(not e[k][t] for k in QBOOLS),'decision/pending_bool_sentinel')
            chosen='hold';reason='already_breached_pending_hold';detail=reason;classification=None
        else:
            kh,kr=(e[k][t].item() for k in ('qbad_hold_count','qbad_restore_count'))
            qh,qr=float(e['qg_hold'][t]),float(e['qg_restore'][t])
            classification=independent_choice(qh,qr,kh,kr,training_n)
            accounting.maybe(e['forecast_n'][t],training_n,ledger,'decision/common_training_N')
            for k in ('local_breach_cap','hold_model_feasible','restore_model_feasible','local_model_infeasible',
                      'predicted_growth_advantage_restore','predicted_breach_count_delta_restore'):
                accounting.maybe(e[k][t],classification[k],ledger,'decision/'+k)
            _,fee,moved=accounting.independent_fee_trade(x,target);wealth=float(sum(x))
            for k,v in dict(quoted_nominal_fee=fee,quoted_nominal_moved=moved,quoted_fee_fraction=fee/wealth,
                            quoted_log_fee_penalty=-math.log1p(-fee/wealth)).items():
                ledger.close(e[k][t],v,'decision/'+k)
            optional=('restoration_predicted_growth_sacrifice','economic_improvement','risk_cost_justification','repair_meets_cap')
            for k in optional:accounting.maybe(e[k][t],classification[k] if t>0 else False,ledger,'decision/'+k)
            if t==0:
                chosen='restore';reason='mandatory_initial_restore';detail=INITIAL_DETAIL_SERIALIZED
            else:chosen=classification['action'];reason=classification['gate_reason'];detail=classification['decision_detail']
        for k,v in dict(gate_reason=reason,decision_reason=reason,decision_detail=detail).items():
            accounting.maybe(e[k][t],v,ledger,'decision/'+k)
        reasons.append(reason);classes.append(classification)
        return chosen
    expected=reference_actual(np.asarray(r),np.asarray(pi),np.asarray(target),.4,action)
    expected['decision_reason']=np.asarray(reasons,dtype='U40')
    expected,metrics=inherited._finish_account(expected,target)
    extra=dict(local_model_infeasible_decision_count=sum(c is not None and t>0 and c['local_model_infeasible'] for t,c in enumerate(classes)),
               growth_improvement_count=sum(s=='growth_improvement' for s in reasons),
               risk_repair_count=sum(s=='risk_constraint_repair' for s in reasons),
               least_violation_restore_count=sum(s=='least_violation_restore' for s in reasons),
               growth_sacrifice_count=sum(c is not None and t>0 and c['restoration_predicted_growth_sacrifice'] for t,c in enumerate(classes)))
    return expected,metrics,extra


def validate_membership(p,parent,full):
    if p['methods']!=METHODS or p['primary_method']!=PRIMARY or p['local_alpha']!=.1:
        raise ValueError('Only one frozen constrained model and alpha permitted')
    return inherited.validate_membership({**p,'methods':inherited.METHODS,'primary_method':'benefit_cost_one_year'},parent,full)


def audit_artifacts(p,bundles,r,pi,ids,tables,choices,groups,archives,cache,*,parent,full):
    ledger=accounting.AuditLedger();membership=validate_membership(p,parent,full);start=time.monotonic()
    plan=readj(AUDIT_PLAN)
    def budget():
        if time.monotonic()-start>60*plan['budget_minutes']:raise TimeoutError('Fixed audit scope exceeds declared budget; never shrink or replace slots')
    if list(map(int,ids))!=list(range(1,121)) or r.shape!=(120,30,3) or pi.shape!=(120,30):raise ValueError('Original120 complete paths required')
    for k,n in p['expected_rows'].items():ledger.require(len(tables[k])==n,'inventory/'+k)
    if set(bundles)!=set(j['job_id'] for j in p['jobs']) or set(choices)!=set(bundles):raise ValueError('Complete130-job identity required')
    weights=cache['weights'];allowed=np.flatnonzero(np.all(weights>=.05-TOL,axis=1))
    ledger.require(len(weights)==232 and len(allowed)==172,'Q1/strict172_mapping')
    train=accounting.index_rows(tables['training_paths'],('job_id','policy','scenario_id'))
    summ=accounting.index_rows(tables['training_summary'],('job_id','policy'))
    inner=accounting.index_rows(tables['inner_validation_paths'],('job_id','policy','scenario_id'))
    inner_B=accounting.index_rows(tables['inner_reference_paths'],('job_id','scenario_id'))
    actual=accounting.index_rows(tables['client_paths'],('job_id','policy','scenario_id'))
    gates=accounting.index_rows(tables['inner_gate_summaries'],('context','policy'))
    admissions=accounting.index_rows(tables['fit_admission'],('context','policy'))
    contexts={c['name']:c for c in p['contexts']};expected_groups={'actual':{},'validation':{}};slots=[];initial_cache={}
    counts=dict(training_rows=0,training_summaries=0,admission_gates=0,selected_trace_accounts=0,selected_trace_decisions=0,
                model_trace_accounts=0,independent_forecast_states=0,unique_initial_forecasts=0,unavailable_forecast_slots=0,pending_forecast_slots=0,full120_paired_paths=0)
    def identity(row,job,b,policy,equal=False):
        expected=dict(job_id=job['job_id'],context=job['parent_context'],context_id=job['parent_context_id'],repeat_id=job['repeat_id'],
          outer_fold=job['outer_fold'],client='C',horizon=30,loss_tolerance=.4,case_status=b['q2_case']['public_status'],
          q1_effective_assignment=b['q2_case']['effective_assignment'],policy=policy,selection_training_n=len(job['training_ids']),
          static_gamma=b['refit']['selection']['selected_gamma'],reference_only=policy in REFERENCES,hypothetical_reference=policy in REFERENCES)
        for k,v in expected.items():accounting.maybe(row[k],v,ledger,'identity/'+k)
        product=b['q2_case']['product']
        for k,source in (('candidate_local_index','candidate_index'),('candidate_original232_index','candidate_original232_index')):
            accounting.maybe(row[k],None if equal or product is None else product[source],ledger,'identity/'+k)
        if policy==PRIMARY:accounting.maybe(row['execution_family'],METHODS[PRIMARY],ledger,'identity/execution_family')
    for job in p['jobs']:
        budget();jid=job['job_id'];b=bundles[jid];case=b['q2_case'];refit=b['refit'];choice=choices[jid]
        ledger.require(b['context']==p['static_contexts'][jid] and refit['training_ids']==job['training_ids'],'Q1/context_training')
        ledger.require(refit['outer_test_ids']==([] if jid=='c0_refit' else job['evaluation_ids']),'Q1/static_evaluation_vs_dynamic_in_sample')
        label=refit['effective_assignments']['C'];expected_case={'Balanced':'balanced_case','Review':'q1_review','Conservative':'not_balanced_case','Growth':'not_balanced_case'}[label]
        ledger.require(case['effective_assignment']==label and case['public_status']==expected_case,'Q1/effective_case')
        raw_menu=refit['raw_fit']['menu'] if refit.get('raw_fit') else []
        item=next((x for x in raw_menu if x['name']==label),None)
        if label=='Review':ledger.require(case['product'] is None and case['q1_product'] is None,'Q1/Review_no_product')
        else:
            ledger.require(item is not None and case['q1_product'] is not None,'Q1/effective_product')
            for k in ('name','candidate_index','candidate_original232_index','weights'):
                ledger.require(case['q1_product'][k]==item[k],'Q1/raw_menu/'+k)
                if label=='Balanced':ledger.require(case['product'][k]==item[k],'Q1/Balanced_product/'+k)
        if 'q1_review_details' in refit:
            detail=refit['q1_review_details']['C']
            ledger.require(case.get('q1_review_kind')==detail.get('q1_review_kind') and case.get('q1_review_reasons',[])==detail.get('q1_review_reasons',[]),'Q1/original_review_reasons')
        tx=np.asarray(job['training_ids'])-1;target=None;kb=None
        if expected_case=='balanced_case':
            local,original=item['candidate_index'],item['candidate_original232_index']
            ledger.require(original==int(allowed[local]),'Q1/local_original_index');ledger.close(item['weights'],weights[original],'Q1/exact_target',atol=2e-15,rtol=0)
            target=item['weights'];kb=int(np.sum(cache['H30__mdd'][tx,original]-.4>TOL))
        ledger.require(choice['training_ids']==job['training_ids'] and choice['evaluation_ids']==job['evaluation_ids'] and choice['case_status']==expected_case,'choices/identities')
        ledger.require(choice['target']==target and set(choice['methods'])=={PRIMARY},'choices/exact_target_and_single_model')
        accounting.maybe(choice['reference_training_breach_count'],kb,ledger,'choices/cache_B_count')
        s=summ[(jid,PRIMARY)];rows=[]
        for sid in job['training_ids']:
            row=train[(jid,PRIMARY,sid)];identity(row,job,b,PRIMARY)
            accounting.maybe(row['scope'],'training_support',ledger,'training/scope');accounting.maybe(row['assigned'],target is not None,ledger,'training/availability')
            accounting.maybe(row['execution_status'],'automated' if target is not None else expected_case,ledger,'training/status')
            accounting.maybe(row['execution_reason'],None if target is not None else expected_case,ledger,'training/reason')
            inherited.check_row_arithmetic(row,ledger,available=target is not None);rows.append(row);counts['training_rows']+=1
        accounting.maybe(s['n_training'],len(tx),ledger,'summary/n_training')
        kval=sum(bool(row['breach']) for row in rows) if target is not None else None
        accounting.maybe(s['breach_count'],kval,ledger,'summary/breach_count')
        for k,source in (('mean_log_growth','log_growth'),('mean_excess','excess'),('mean_nominal_fees','nominal_fees')):
            accounting.maybe(s[k],float(np.mean([row[source] for row in rows])) if target is not None else None,ledger,'summary/'+k)
        if target is not None:
            optional=int(round(sum(row['target_execution_fraction']*30 for row in rows)-len(rows)))
            accounting.maybe(s['optional_restores'],optional,ledger,'summary/optional_count')
            for k in BREAKDOWN:
                ledger.require(np.isfinite(s[k]) and s[k]==int(s[k]) and 0<=s[k]<=len(tx)*29,'summary/bounded_integer/'+k)
            ledger.require(s['economic_improvement_restores']+s['risk_repair_restores']==optional,'summary/restore_partition')
            ledger.require(s['growth_sacrificing_restores']<=s['risk_repair_restores'],'summary/sacrifice_subset')
        else:
            for k in BREAKDOWN:accounting.maybe(s[k],None,ledger,'summary/Review_NULL/'+k)
        counts['training_summaries']+=1
        training_eligible=target is not None and kval<=min(kb,math.floor(.1*len(tx)+TOL));saved_choice=choice['methods'][PRIMARY]
        accounting.maybe(saved_choice['training_eligible'],bool(training_eligible),ledger,'admission/training_eligible')
        if target is not None:
            for k,v in dict(reference_training_breach_count=kb,training_breach_cap=min(kb,math.floor(.1*len(tx)+TOL)),n_training=len(tx)).items():
                accounting.maybe(saved_choice[k],v,ledger,'admission/'+k)
        if target is None:status,reason=expected_case,expected_case
        elif not training_eligible:status,reason='execution_review','complete_training_risk_exceeds_cap'
        else:status,reason='automated',None
        if job['stage']=='refit':
            entries=[]
            for ij in (j for j in p['jobs'] if j['parent_context_id']==job['parent_context_id'] and j['stage']=='inner'):
                sr=[inner[(ij['job_id'],PRIMARY,sid)] for sid in ij['evaluation_ids']];autos=[row for row in sr if row['assigned']]
                ledger.require(len(autos) in (0,len(sr)),'gate/uniform_inner_block')
                entries.append(dict(fold=ij['dynamic_inner_fold'],evaluation_ids=ij['evaluation_ids'],automated=bool(autos),
                    breach_count=sum(bool(row['breach']) for row in autos) if autos else None,
                    reference_breach_count=sum(bool(inner_B[(ij['job_id'],row['scenario_id'])]['breach']) for row in autos) if autos else None,
                    mean_log_growth=float(np.mean([row['log_growth'] for row in autos])) if autos else None))
            g=accounting.independent_four_inner_receipt(entries,contexts[job['parent_context']]['training_ids'])
            sg=gates[(job['parent_context'],PRIMARY)];nested=saved_choice['inner_gate']
            values=dict(original_n=g['original_n'],auto_inner_n=g['automatic_n'],selected_breach_count=g['breach_count'],same_U_B_breach_count=g['reference_breach_count_same_automatic_ids'],
                absolute_alpha_count_cap=g['effective_gate_cap'],conditional_mean_log_growth=g['conditional_mean_log_growth'],eligible=g['inner_gate_eligible'],
                conditional_breach_probability=g['breach_count']/g['automatic_n'] if g['automatic_n'] else None,
                conditional_B_breach_probability=g['reference_breach_count_same_automatic_ids']/g['automatic_n'] if g['automatic_n'] else None)
            for k,v in values.items():accounting.maybe(sg[k],v,ledger,'gate/'+k);accounting.maybe(nested[k],v,ledger,'nested_gate/'+k)
            U=sorted(sid for entry in entries if entry['automated'] for sid in entry['evaluation_ids'])
            encoded=sg['auto_set_scenario_ids'];decoded=json.loads(encoded) if isinstance(encoded,str) else encoded
            ledger.require(decoded==U and nested['auto_set_scenario_ids']==U,'gate/exact_same_U')
            reasons=[]
            if not g['automatic_n']:reasons.append('no_inner_auto_evidence')
            else:
                if not g['empirical_risk_eligible']:reasons.append('inner_absolute_risk_exceeds_reference')
                if not g['paired_B_dominance_eligible']:reasons.append('inner_risk_exceeds_same_U_B')
            gr=reasons[0] if reasons else 'passed_inner_gate'
            ledger.require(nested['reasons']==reasons and nested['reason']==gr,'gate/reasons')
            if target is not None and not g['inner_gate_eligible']:status,reason='execution_review',gr
            ar=admissions[(job['parent_context'],PRIMARY)]
            for k,v in dict(training_n=len(tx),evaluation_n=len(job['evaluation_ids']),initial_admitted=status=='automated',inner_gate_passed=g['inner_gate_eligible'],execution_status=status,execution_reason=reason).items():
                accounting.maybe(ar[k],v,ledger,'fit_admission/'+k)
            counts['admission_gates']+=1
        accounting.maybe(saved_choice['execution_status'],status,ledger,'choice/status');accounting.maybe(saved_choice['execution_reason'],reason,ledger,'choice/reason')
        category='validation' if job['stage']=='inner' else 'actual';source=inner if category=='validation' else actual
        scope='inner_validation' if category=='validation' else 'in_sample_full120' if job['parent_context_id']==0 else 'outer_oof'
        for sid in job['evaluation_ids']:
            row=source[(jid,PRIMARY,sid)];identity(row,job,b,PRIMARY)
            for k,v in dict(scope=scope,assigned=status=='automated',execution_status=status,execution_reason=reason).items():accounting.maybe(row[k],v,ledger,'evaluated/'+k)
            inherited.check_row_arithmetic(row,ledger,available=status=='automated')
            if jid=='c0_refit' and status=='automated':
                for field in accounting.METRICS:accounting.maybe(row[field],train[(jid,PRIMARY,sid)][field],ledger,'in_sample_same_path/'+field)
                counts['full120_paired_paths']+=1
        if status=='automated':expected_groups[category][jid+'__'+PRIMARY]=(job,target,PRIMARY)
        else:
            for year in (1,2,*((16,30) if jid in EXTRA_REFITS else ())):
                slots.append(dict(job_id=jid,policy=PRIMARY,scenario_id=min(job['evaluation_ids']),original_year=year,status='unavailable_no_substitution',execution_status=status))
                counts['unavailable_forecast_slots']+=1
        for sid in job['evaluation_ids']:
            br=inner_B[(jid,sid)] if category=='validation' else actual[(jid,REFERENCES[0],sid)];identity(br,job,b,REFERENCES[0])
            accounting.maybe(br['assigned'],target is not None,ledger,'B/availability');inherited.check_row_arithmetic(br,ledger,available=target is not None)
            for field,value in dict(scope='inner_validation_reference' if category=='validation' else scope,
                 execution_status='hypothetical_reference' if target is not None else 'reference_unavailable',execution_reason=None if target is not None else expected_case).items():
                accounting.maybe(br[field],value,ledger,'B/'+field)
            if job['stage']=='refit':
                eq=actual[(jid,REFERENCES[1],sid)];identity(eq,job,b,REFERENCES[1],equal=True)
                accounting.maybe(eq['assigned'],True,ledger,'EQ/availability');inherited.check_row_arithmetic(eq,ledger,available=True)
                accounting.maybe(eq['scope'],scope,ledger,'EQ/scope');accounting.maybe(eq['execution_status'],'hypothetical_reference',ledger,'EQ/status')
        if target is not None:expected_groups[category][jid+'__B']=(job,target,REFERENCES[0])
        if job['stage']=='refit':expected_groups['actual'][jid+'__equal']=(job,np.ones(3)/3,REFERENCES[1])
    for category in ('actual','validation'):
        archive=archives[category];ledger.require(set(groups[category])==set(expected_groups[category]),'trace/complete_inventory')
        ledger.require(all(any(k.startswith(g+'__') for g in expected_groups[category]) for k in archive),'trace/no_orphans')
        for key,(job,target,policy) in expected_groups[category].items():
            budget();meta=groups[category][key]
            ledger.require(meta['scenario_ids']==job['evaluation_ids'] and meta['training_ids']==job['training_ids'],'trace/original_train_eval_identity')
            ledger.require(meta['metadata']['job_id']==job['job_id'] and meta['metadata']['policy']==policy,'trace/job_policy')
            arm={k[len(key)+2:]:archive[k] for k in archive if k.startswith(key+'__')};tx=np.asarray(job['training_ids'])-1;smallest=min(job['evaluation_ids'])
            for i,sid in enumerate(job['evaluation_ids']):
                if policy==PRIMARY:
                    e={k:arm[k][i] for k in (*QFLOATS,*QCOUNTS,*QBOOLS,*QSTRINGS)}
                    expected,metric,extra=replay_model(r[sid-1],pi[sid-1],target,len(tx),e,ledger)
                    for k,v in extra.items():accounting.maybe(arm[k][i],v,ledger,'trace_extra/'+k)
                    counts['model_trace_accounts']+=1
                else:
                    expected=accounting.independent_account(r[sid-1],pi[sid-1],target,0.,hold_after_breach=policy!=REFERENCES[1]);metric=accounting.independent_metrics(expected)
                    metric['mean_fee_fraction']=float(np.mean(expected['fee_fraction']))
                saved={k:arm[k] if k=='targets' else arm[k][i] for k in expected if k in arm and k not in ('delta','loss_tolerance')}
                accounting.compare_saved_account(saved,expected,ledger)
                table=inner_B if category=='validation' and policy==REFERENCES[0] else inner if category=='validation' else actual
                row=table[(job['job_id'],sid)] if table is inner_B else table[(job['job_id'],policy,sid)]
                for k in accounting.METRICS:accounting.maybe(row[k],metric[k],ledger,'replayed_metric/'+k)
                counts['selected_trace_accounts']+=1;counts['selected_trace_decisions']+=30
                if policy==PRIMARY and sid==smallest:
                    for year in (1,2,*((16,30) if job['job_id'] in EXTRA_REFITS else ())):
                        slot=dict(category=category,group=key,job_id=job['job_id'],policy=policy,scenario_id=sid,original_year=year)
                        t=year-1
                        if expected['pending_at_decision'][t]:slot['status']='pending_skip_no_substitution';counts['pending_forecast_slots']+=1
                        else:
                            budget();cachekey=(tuple(map(float,expected['holdings'][t])),float(expected['cpi'][t]),float(expected['real_peak'][t]),
                                float(expected['running_mdd'][t]),tuple(map(float,target)),tuple(job['training_ids']),30,.4,.0025)
                            if t==0 and cachekey in initial_cache:q=initial_cache[cachekey];slot['initial_cache_reused']=True
                            else:
                                q=scalar_fan(expected['holdings'][t],expected['cpi'][t],expected['real_peak'][t],expected['running_mdd'][t],np.asarray(target),.4,r[tx],pi[tx],t)
                                if t==0:initial_cache[cachekey]=q;counts['unique_initial_forecasts']+=1
                            for branch,field in (('hold','hold'),('restore','restore')):
                                ledger.close(arm['qg_'+field][i,t],q[branch]['mean'],'independent_Q/net_'+field)
                                accounting.maybe(arm['qbad_'+field+'_count'][i,t],q[branch]['count'],ledger,'independent_Q/count_'+field)
                            counts['independent_forecast_states']+=1;slot['status']='independently_recomputed'
                        slots.append(slot)
    ledger.require(counts['training_rows']==10080 and counts['training_summaries']==130 and counts['admission_gates']==26,'inventory/complete_training_and_gates')
    budget()
    return dict(status='passed',checks_passed=ledger.checks_passed,maximum_absolute_errors=ledger.maximum_absolute_errors,counts=counts,membership=membership,
                fixed_forecast_state_inventory=slots,elapsed_seconds=time.monotonic()-start,limitations=plan['limitations'],
                training_economic_breakdown_scope='All financial means/event/optional counts aggregated; unrecorded training economic/risk/sacrifice/infeasibility classifications checked for partition and count bounds only, not fully independently regenerated',
                every_saved_Q_level_recomputed=False,unrecorded_full_training_policy_trees_replayed=False,
                frozen_schema_note='Inception decision_detail literal is41 characters but frozen U40 field stores mandatory_initial_restore_diagnostic_onl; exact serialized prefix audited, no action or numeric effect')


def main():
    if (OUT/'independent_audit.json').exists():raise RuntimeError('Never overwrite successful frozen audit')
    if (OUT/'audit_failure_receipt.json').exists():raise RuntimeError('Preserved failed audit requires explicit plan addendum')
    p=readj(PROTOCOL);manifest=readj(OUT/'manifest.json');plan=readj(AUDIT_PLAN)
    if p.get('execution_ready') is not True or manifest.get('status')!='complete' or manifest['training_tasks']!=130:raise RuntimeError('Complete frozen130-job archive required before parsing market values')
    if sha(PROTOCOL)!=manifest['protocol_sha256']:raise RuntimeError('Executed protocol changed')
    for mapping in (p['execution_sources_sha256'],p['frozen_input_sha256']):
        for name,value in mapping.items():
            if sha(ROOT/name)!=value:raise RuntimeError('Frozen source/input changed: '+name)
    for name,value in manifest['outputs_sha256'].items():
        if sha(OUT/name)!=value:raise RuntimeError('Executed numeric artifact changed: '+name)
    paths=[PROTOCOL,AUDIT_PLAN,OUT/'manifest.json',Path(__file__),ROOT/'tests/test_audit_constrained_fan_execution.py',CACHE]
    paths += [ROOT/name for name in ('scripts/audit_simple_execution_comparison.py','scripts/audit_benefit_cost_execution.py','scripts/audit_fan_rollout_execution.py','scripts/audit_review_pending_execution.py')]
    paths += [ROOT/name for name in p['execution_sources_sha256']]+[ROOT/name for name in p['frozen_input_sha256']]+[OUT/name for name in manifest['outputs_sha256']]
    before={str(path.relative_to(ROOT)):sha(path) for path in paths}
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    r,pi,ids=load_paths(ROOT/p['data']['path'],p['data']['scenario_ids'],allow_final=False)
    digest=hashlib.sha256()
    for value in (ids,r,pi):digest.update(np.ascontiguousarray(value).tobytes())
    if digest.hexdigest()!=manifest['development_arrays_sha256'] or ids.tolist()!=list(range(1,121)) or manifest['final_scenarios_loaded']:raise RuntimeError('Guarded development arrays changed')
    parent=readj(ROOT/'docs/refit_stability_protocol.json');full=readj(ROOT/'docs/joint_liquidity_refit_protocol.json')
    bundles=readj(ROOT/'results/model_review/refit_stability_v1/static_refits.json.gz');bundles.update({k:v for k,v in readj(ROOT/'results/model_review/joint_liquidity_refit_v1/static_refits.json').items() if k.startswith('c0_')})
    for name in ('refit_stability_v1','joint_liquidity_refit_v1'):
        if readj(ROOT/f'results/model_review/{name}/independent_audit.json')['status']!='passed':raise RuntimeError('Completed parent Q1 component audit required')
    tables={k:accounting.observed_bools(pd.read_csv(OUT/(k+'.csv.gz'),low_memory=False)) for k in p['expected_rows']}
    with np.load(CACHE,allow_pickle=False) as f:cache={k:f[k] for k in ('weights','H30__mdd')}
    with np.load(OUT/'actual_traces.npz',allow_pickle=False) as a,np.load(OUT/'validation_traces.npz',allow_pickle=False) as v:
        result=audit_artifacts(p,bundles,r,pi,ids,tables,readj(OUT/'job_choices.json.gz'),readj(OUT/'trace_groups.json'),{'actual':a,'validation':v},cache,parent=parent,full=full)
    after={str(path.relative_to(ROOT)):sha(path) for path in paths}
    if before!=after:raise RuntimeError('Frozen source/input/artifact changed during audit')
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(AUDIT_PLAN),
        auditor_sha256=sha(Path(__file__)),sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,scenario_ids_loaded=ids.tolist(),final_scenarios_loaded=False,
        q1_component_audit_scope='Completed parent Q1 components referenced; own130 identities, menu/target/candidate indices and Review reasons checked; no repeated all172 optimizer audit',
        interpretation='Automated arithmetic/action/fixed-subset audit; not human review, conditional calibration, fresh holdout evidence or a risk guarantee')
    writej(OUT/'independent_audit.json',result)
    print(json.dumps(dict(status='passed',checks_passed=result['checks_passed'],counts=result['counts'],audit_sha256=sha(OUT/'independent_audit.json'))))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists() and not (OUT/'independent_audit.json').exists() and not (OUT/'audit_failure_receipt.json').exists():
            writej(OUT/'audit_failure_receipt.json',dict(status='failed_audit_attempt_preserved',completed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(Path(__file__)),audit_plan_sha256=sha(AUDIT_PLAN),
                error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),numeric_archive_modified_by_auditor=False,successful_audit_written=False))
        raise
