"""Independent fixed-state/single-realized-suffix counterfactual audit.

The new counterfactual module, its runner and policy kernels are never imported.
A previous independent100-step fee bisection is the sole reused finance helper.
Forecasts are copied from the already audited parent Q arrays; only the two
realized current-action-then-B branches are independently rebuilt here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_review_pending_execution import independent_trade

PROTOCOL=ROOT/'docs/fixed_state_forecast_diagnostic_protocol.json'
OUT=ROOT/'results/model_review/fixed_state_forecast_diagnostic_v1'
PARENT=ROOT/'results/model_review/nested_dynamic_calibration_v1'
H,L,COST,TOL=30,.4,.0025,1e-12
ARRAY_FIELDS=('holdings','cpi','real_wealth','real_peak','running_mdd','ever_breached',
              'actions','fees','moved','posttrade_holdings')
SCALAR_FIELDS=('total_log_growth','terminal_real','history_mdd','breach',
               'first_crossing_original_year','year_index','original_horizon','nominal_fees','nominal_moved')


def readj(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_branch(holdings,cpi,peak,worst,target,loss,returns,inflation,*,
                       year_index,original_horizon,current_action,ever_breached=False,cost_rate=.0025):
    """Own scalar state recursion using the original calendar and history."""
    if current_action not in ('hold','restore') or ever_breached:
        raise ValueError('Only the declared two active-state branches are audited')
    x=np.array(holdings,dtype=float,copy=True);w=np.asarray(target,dtype=float)
    market=np.asarray(returns,dtype=float);infl=np.asarray(inflation,dtype=float)
    if (x.shape!=(3,) or w.shape!=(3,) or market.shape!=(original_horizon,3)
        or infl.shape!=(original_horizon,) or not 0<=year_index<original_horizon):raise ValueError('Inconsistent original state/clock')
    cp=float(cpi);pk=float(peak);dd=float(worst);bad=False
    current=float(sum(x))/cp
    if cp<=0 or pk<=0 or current<=0 or not np.isfinite([*x,cp,pk,dd,*w]).all():raise ValueError('Invalid saved state')
    if current-pk>TOL*max(current,pk) or max(1-current/pk,0)-dd>TOL or dd-loss>TOL:raise ValueError('Inconsistent active history')
    states={k:[] for k in ('holdings','cpi','real_wealth','real_peak','running_mdd','ever_breached')}
    decisions={k:[] for k in ('actions','fees','moved','posttrade_holdings')}
    first=None
    def append():
        for key,value in (('holdings',x.copy()),('cpi',cp),('real_wealth',float(sum(x))/cp),
                          ('real_peak',pk),('running_mdd',dd),('ever_breached',bad)):
            states[key].append(value)
    append()
    for year in range(year_index,original_horizon):
        # The first action is fixed; all later choices use this branch's own b.
        action=current_action if year==year_index else ('hold' if bad else 'restore')
        if action=='restore':post,fee,moved=independent_trade(x,w,cost_rate)
        else:post,fee,moved=x.copy(),0.,0.
        for key,value in (('actions',action),('fees',float(fee)),('moved',float(moved)),('posttrade_holdings',post.copy())):
            decisions[key].append(value)
        x=np.array([post[j]*(1+market[year,j]) for j in range(3)])
        cp=cp*(1+float(infl[year]));real=float(sum(x))/cp
        if not np.isfinite(real) or real<=0 or cp<=0:raise ValueError('Nonfinite suffix state')
        pk=max(pk,real);dd=max(dd,1-real/pk,0.)
        if dd-loss>TOL:bad=True
        if bad and first is None:first=year+1
        append()
    result={key:np.asarray(values) for key,values in {**states,**decisions}.items()}
    result.update(total_log_growth=math.log(result['real_wealth'][-1]/current),
        terminal_real=float(result['real_wealth'][-1]),history_mdd=float(dd),breach=bool(bad),
        first_crossing_original_year=first,year_index=int(year_index),original_horizon=int(original_horizon),
        nominal_fees=float(sum(result['fees'])),nominal_moved=float(sum(result['moved'])))
    # History peak and worst are retained, rather than re-estimating MDD from
    # the suffix alone. The old worst may exceed any newly realized suffix DD.
    real=result['real_wealth'];interval=max([0.]+[1-real[t]/real[s] for s in range(len(real)) for t in range(s+1,len(real))])
    against_old_peak=max(1-float(v)/peak for v in real)
    if not math.isclose(dd,max(float(worst),interval,against_old_peak,0.),abs_tol=2e-14,rel_tol=2e-14):
        raise AssertionError('Independent all-interval/history MDD differs')
    return result


def independent_comparison(predicted,realized,annual_price,original_horizon,chosen_action):
    qh,qr=predicted['hold']['log'],predicted['restore']['log']
    ph,pr=predicted['hold']['bad_fraction'],predicted['restore']['bad_fraction']
    delta_q=qr-qh;delta_p=pr-ph
    delta_g=realized['restore']['total_log_growth']-realized['hold']['total_log_growth']
    delta_b=int(realized['restore']['breach'])-int(realized['hold']['breach'])
    score_q=delta_q-original_horizon*annual_price*delta_p
    score_r=delta_g-original_horizon*annual_price*delta_b
    sign=lambda v:1 if v>TOL else -1 if v<-TOL else 0
    action_q='restore' if score_q>TOL else 'hold'
    action_r='restore' if score_r>TOL else 'hold'
    branch_scores={a:realized[a]['total_log_growth']-original_horizon*annual_price*int(realized[a]['breach']) for a in ('hold','restore')}
    return dict(predicted_delta_log=float(delta_q),predicted_delta_bad_fraction=float(delta_p),
        realized_suffix_delta_log=float(delta_g),realized_suffix_delta_event=delta_b,
        delta_log_residual=float(delta_g-delta_q),delta_event_residual=float(delta_b-delta_p),
        predicted_score=float(score_q),realized_suffix_score=float(score_r),score_residual=float(score_r-score_q),
        predicted_growth_sign=sign(delta_q),realized_suffix_growth_sign=sign(delta_g),
        growth_sign_agreement=sign(delta_q)==sign(delta_g),predicted_action=action_q,
        realized_suffix_better_current_action=action_r,score_action_agreement=action_q==action_r,
        two_branch_realized_score_shortfall=max(branch_scores.values())-branch_scores[chosen_action])


def self_checks():
    checks={}
    x=np.array([20.,0.,80.]);w=np.array([40/99.95,0.,59.95/99.95])
    post,fee,moved=independent_trade(x,w,.0025)
    assert np.allclose(post,[40.,0.,59.95],atol=2e-14,rtol=0) and abs(fee-.05)<2e-15 and abs(moved-20)<2e-14
    checks['twenty_units_moved_fee_point05']=True
    r=np.array([[99.,0.,0.],[-.5,0.,0.],[1.,0.,0.]])
    b=independent_branch([30.,0.,70.],2.,60.,1/6,[.5,0,.5],.2,r,np.zeros(3),
                         year_index=1,original_horizon=3,current_action='hold',cost_rate=0.)
    assert np.allclose(b['real_wealth'],[50.,42.5,50.]) and b['first_crossing_original_year']==2
    assert b['actions'].tolist()==['hold','hold'] and abs(b['history_mdd']-(1-42.5/60))<1e-15
    checks['original_calendar_CPI_peak_MDD_and_next_decision_pending']=True
    r=np.array([[-.5,0.,0.],[0.,0.,0.]])
    branches={a:independent_branch([0.,0.,100.],1.,100.,0.,[1.,0.,0.],.2,r,np.zeros(2),
          year_index=0,original_horizon=2,current_action=a,cost_rate=0.) for a in ('hold','restore')}
    assert branches['hold']['actions'].tolist()==['hold','restore'] and not branches['hold']['breach']
    assert branches['restore']['actions'].tolist()==['restore','hold'] and branches['restore']['breach']
    checks['each_branch_uses_own_b_not_other_branch']=True
    predicted={a:dict(log=g,bad_fraction=p) for a,g,p in [('hold',.2,.2),('restore',.21,.1)]}
    realized={a:dict(total_log_growth=g,breach=b) for a,g,b in [('hold',.1,False),('restore',.2,True)]}
    compare=independent_comparison(predicted,realized,.03,30,'restore')
    assert math.isclose(compare['predicted_score'],.1,abs_tol=1e-14)
    assert math.isclose(compare['realized_suffix_score'],-.8,abs_tol=1e-14)
    assert math.isclose(compare['two_branch_realized_score_shortfall'],.8,abs_tol=1e-14)
    checks['original_H_score_probability_fraction_and_hindsight_not_policy_value']=True
    return checks


def independent_summary(records):
    """Rebuild descriptive groups; a state count is not an independent n."""
    result=[]
    for level,keys in [('analysis_scope',('analysis_scope',)),
                       ('context',('analysis_scope','context')),('year',('analysis_scope','year'))]:
        groups={}
        for row in records:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
        for values,rows in groups.items():
            identity=dict(zip(keys,values));n=len(rows)
            out=dict(level=level,analysis_scope=identity['analysis_scope'],context=identity.get('context'),
                year=identity.get('year'),active_state_n=n,
                unique_scenario_n=len({r['scenario_id'] for r in rows}),unique_job_n=len({r['job_id'] for r in rows}),
                unique_group_n=len({r['group'] for r in rows}),in_training_fan_state_n=sum(r['in_training_fan'] for r in rows),
                chosen_restore_n=sum(r['chosen_action']=='restore' for r in rows),
                predicted_growth_positive_n=sum(r['predicted_growth_sign']==1 for r in rows),
                realized_growth_positive_n=sum(r['realized_suffix_growth_sign']==1 for r in rows),
                growth_sign_agree_n=sum(r['growth_sign_agreement'] for r in rows),
                score_action_agree_n=sum(r['score_action_agreement'] for r in rows),
                realized_hold_event_n=sum(r['realized_hold_event'] for r in rows),
                realized_restore_event_n=sum(r['realized_restore_event'] for r in rows))
            for field in ('delta_log_residual','score_residual','two_branch_realized_score_shortfall',
                          'predicted_hold_bad_fraction','predicted_restore_bad_fraction'):
                out['mean_'+field]=sum(r[field] for r in rows)/n
            for field in ('delta_log_residual','score_residual'):
                out['mean_abs_'+field]=sum(abs(r[field]) for r in rows)/n
            result.append(out)
    return result


def evidence_snapshot(directory):
    manifest=readj(directory/'manifest.json')
    paths={str((directory/name).relative_to(ROOT)) for name in manifest['outputs_sha256']}
    paths.add(str((directory/'manifest.json').relative_to(ROOT)))
    paths.update(manifest['sources']);paths.update(manifest['input_sha256'])
    paths.update((str(Path(__file__).resolve().relative_to(ROOT)),
                  'scripts/audit_review_pending_execution.py','tests/test_audit_fixed_state_forecast_diagnostic.py'))
    return {rel:sha(ROOT/rel) for rel in sorted(paths)}


def run_audit(directory=OUT):
    directory=Path(directory);synthetic=self_checks();before=evidence_snapshot(directory)
    manifest=readj(directory/'manifest.json');protocol=readj(PROTOCOL)
    if manifest['sources']!=manifest['sources_at_end'] or manifest['input_sha256']!=manifest['input_sha256_at_end']:
        raise AssertionError('Original diagnostic source/input instability')
    for mapping in (manifest['sources'],manifest['input_sha256']):
        for rel,digest in mapping.items():
            if sha(ROOT/rel)!=digest:raise AssertionError('Frozen diagnostic dependency changed: '+rel)
    for name,digest in manifest['outputs_sha256'].items():
        if sha(directory/name)!=digest:raise AssertionError('Original diagnostic artifact SHA mismatch: '+name)
    if manifest['final_scenarios_loaded'] is not False or manifest['scenario_ids_loaded']!=list(range(1,121)):
        raise AssertionError('Unexpected diagnostic market scope')
    old_audit=readj(PARENT/'independent_audit.json')
    fields=('scope','group','scenario_id','year','training_n','status')
    expected=[{k:x[k] for k in fields} for x in old_audit['q_subset_results']]
    if protocol['identities']!=expected or protocol['nonautomated_jobs']!=old_audit['q_nonautomated_job_skips']:
        raise AssertionError('Predeclared fixed-state identities were replaced')
    parent_protocol=readj(ROOT/'docs/nested_dynamic_calibration_protocol.json')
    jobs={x['job_id']:x for x in parent_protocol['jobs']}
    parent_groups=readj(PARENT/'trace_groups.json')
    parent_arrays={}
    for scope,name in [('actual','actual_traces.npz'),('validation','validation_traces.npz')]:
        with np.load(PARENT/name,allow_pickle=False) as archive:parent_arrays[scope]={k:archive[k] for k in archive.files}
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    market,inflation,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
    if not np.array_equal(ids,np.arange(1,121)):raise AssertionError('Original development path IDs changed')
    inventory=pd.read_csv(directory/'state_inventory.csv',float_precision='round_trip')
    branches=pd.read_csv(directory/'branch_paths.csv',float_precision='round_trip')
    contrasts=pd.read_csv(directory/'state_comparisons.csv',float_precision='round_trip')
    summaries=pd.read_csv(directory/'summary.csv',float_precision='round_trip')
    saved_groups=readj(directory/'trace_groups.json')
    with np.load(directory/'branch_states.npz',allow_pickle=False) as archive:saved={k:archive[k] for k in archive.files}
    check_n=0;errors={};independent_states={};results=[];branch_scalars={};branch_periods=0
    def check(condition,label):
        nonlocal check_n
        check_n+=1
        if not condition:raise AssertionError(label)
    def close(actual,expected,label,*,atol=3e-11,rtol=3e-13):
        a,b=np.asarray(actual),np.asarray(expected);check(a.shape==b.shape,label+'/shape')
        if a.dtype.kind in 'bUiS' or b.dtype.kind in 'bUiS':check(np.array_equal(a,b),label+'/exact')
        else:
            delta=np.abs(a.astype(float)-b.astype(float));valid=np.isfinite(delta)
            errors[label.rsplit('/',1)[-1]]=max(errors.get(label.rsplit('/',1)[-1],0.),float(delta[valid].max(initial=0.)))
            check(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),label+'/numeric')
    def maybe(actual,expected,label):
        if expected is None:check(pd.isna(actual),label+'/NULL')
        elif isinstance(expected,str):check(actual==expected,label+'/string')
        elif isinstance(expected,(bool,np.bool_)):
            observed=(actual.lower()=='true') if isinstance(actual,str) and actual.lower() in ('true','false') else actual
            check(isinstance(observed,(bool,np.bool_)) and bool(observed)==bool(expected),label+'/bool')
        else:close(actual,expected,label)
    check(len(expected)==81 and sum(x['status']=='audited' for x in expected)==79,'identity/81_with79active')
    check((len(inventory),len(branches),len(contrasts))==(90,158,79),'inventory/90_158_79')
    check(inventory.state_id.is_unique,'inventory/unique_slots')
    expected_state_ids=[];expected_prefixes=set();pending_n=0
    for number,item in enumerate(expected,1):
        scope,key,sid,year=(item[k] for k in ('scope','group','scenario_id','year'))
        state_id=f'q{number:03d}';row=inventory.loc[inventory.state_id==state_id]
        check(len(row)==1,'inventory/original_slot');row=row.iloc[0]
        group=parent_groups[scope][key];meta=group['metadata'];job=jobs[meta['job_id']]
        arm={name[len(key)+2:]:value for name,value in parent_arrays[scope].items() if name.startswith(key+'__')}
        i=group['scenario_ids'].index(sid);t=year-1
        analysis_scope='inner_validation' if scope=='validation' else 'full_development_descriptive' if meta['context']=='full_development' else 'outer_oof'
        common=dict(state_id=state_id,archive_scope=scope,analysis_scope=analysis_scope,group=key,
            job_id=meta['job_id'],context=meta['context'],scenario_id=sid,year=year,training_n=item['training_n'],
            in_training_fan=sid in job['training_ids'],chosen_action=str(arm['actions'][i,t]),
            selected_candidate_order=group['selected_candidate_order'])
        for field,value in common.items():maybe(row[field],value,'identity/'+field)
        check(sid==group['q_subset_scenario_id'] and year in group['q_subset_years'],'identity/frozen_parent_subset')
        pending=bool(arm['pending_at_decision'][i,t]);check(pending==(item['status']=='pending_skip_no_replacement'),'identity/own_pending_status')
        if pending:
            pending_n+=1;check(row['status']=='pending_skip_no_replacement','pending/no_replacement')
            check(not branches.state_id.eq(state_id).any() and not contrasts.state_id.eq(state_id).any() and state_id not in saved_groups,'pending/no_counterfactual')
            continue
        check(row['status']=='active_evaluated','inventory/active_status');expected_state_ids.append(state_id)
        x=arm['holdings'][i,t].copy();cp=float(arm['cpi'][i,t]);pk=float(arm['real_peak'][i,t]);dd=float(arm['running_mdd'][i,t]);target=np.asarray(arm['targets'])
        price=np.asarray(arm['annual_prices']).reshape(-1);check(price.shape==(1,) and np.isfinite(price[0]),'price/single_original_finite')
        eta=float(price[0]);n=int(arm['forecast_n'][i,t]);check(n==item['training_n']==len(job['training_ids']),'forecast/original_denominator')
        check(not bool(arm['ever_breached'][i,t]),'active/no_history_reset')
        common.update(annual_price=eta,horizon=H,loss_tolerance=L,initial_equity=float(x[0]),initial_bond=float(x[1]),initial_bill=float(x[2]),
            initial_cpi=cp,initial_real_peak=pk,initial_running_mdd=dd,target_equity=float(target[0]),target_bond=float(target[1]),target_bill=float(target[2]))
        for field,value in common.items():maybe(row[field],value,'state/'+field)
        expected_group=saved_groups[state_id]
        for field,value in common.items():maybe(expected_group[field],value,'trace_group/'+field)
        check(expected_group['fields']==list(ARRAY_FIELDS),'trace_group/complete_state_fields')
        predicted={};realized={}
        for action in ('hold','restore'):
            prefix=state_id+'__'+action;expected_prefixes.add(prefix)
            check(expected_group['npz_prefixes'][action]==prefix,'trace_group/prefix')
            result=independent_branch(x,cp,pk,dd,target,L,market[sid-1],inflation[sid-1],year_index=t,original_horizon=H,
                current_action=action,ever_breached=bool(arm['ever_breached'][i,t]),cost_rate=COST)
            realized[action]=result;branch_periods+=H-t
            count=int(arm['qbad_'+action+'_count'][i,t]);check(0<=count<=n,'forecast/saved_count')
            predicted[action]=dict(log=float(arm['qg_'+action][i,t]),bad_fraction=count/n)
            for field in ARRAY_FIELDS:
                close(saved[prefix+'__'+field],result[field],'branch/'+field)
                independent_states[prefix+'__'+field]=result[field]
            independent_states[prefix+'__nominal_wealth']=result['holdings'].sum(axis=1)
            independent_states[prefix+'__original_endpoint_years']=np.arange(t,H+1)
            independent_states[prefix+'__original_decision_years']=np.arange(t+1,H+1)
            for field in SCALAR_FIELDS:
                independent_states[prefix+'__'+field]=np.asarray(-1 if result[field] is None else result[field])
            branch_scalars[prefix]={field:result[field] for field in SCALAR_FIELDS}
            br=branches.loc[(branches.state_id==state_id)&(branches.current_action==action)]
            check(len(br)==1,'branch/exact_two_actions');br=br.iloc[0]
            for field,value in common.items():maybe(br[field],value,'branch_common/'+field)
            expected_values=dict(predicted_total_log_growth=predicted[action]['log'],predicted_bad_count=count,fan_n=n,predicted_bad_fraction=count/n,
                realized_suffix_total_log_growth=result['total_log_growth'],realized_suffix_breach=result['breach'],history_mdd=result['history_mdd'],
                terminal_real=result['terminal_real'],nominal_fees=result['nominal_fees'],nominal_moved=result['nominal_moved'],
                first_crossing_original_year=result['first_crossing_original_year'],
                log_residual=result['total_log_growth']-predicted[action]['log'],event_residual=int(result['breach'])-count/n)
            for field,value in expected_values.items():maybe(br[field],value,'branch_derived/'+field)
            check(np.all(np.abs(result['fees']-COST*result['moved'])<=3e-13),'fee/one_transfer_basis')
        comparison=independent_comparison(predicted,realized,eta,H,common['chosen_action'])
        check(comparison['predicted_action']==common['chosen_action'],'Q/old_action_reproduced')
        close(comparison['predicted_score'],arm['predicted_score'][i,t],'Q/old_technical_score',atol=3e-12,rtol=0)
        derived={**common,**comparison,'realized_hold_event':realized['hold']['breach'],'realized_restore_event':realized['restore']['breach'],
                 'predicted_hold_bad_fraction':predicted['hold']['bad_fraction'],'predicted_restore_bad_fraction':predicted['restore']['bad_fraction']}
        cr=contrasts.loc[contrasts.state_id==state_id];check(len(cr)==1,'comparison/one_per_active');cr=cr.iloc[0]
        for field,value in derived.items():maybe(cr[field],value,'comparison/'+field)
        results.append(derived)
    check(pending_n==2,'pending/exact_two')
    check(set(saved_groups)==set(expected_state_ids),'trace_groups/only79_active')
    check(set(saved)=={prefix+'__'+field for prefix in expected_prefixes for field in ARRAY_FIELDS},'branch_npz/no_missing_or_unrecorded_branches')
    # Three final jobs remain unserved; nine slots are an inventory, never
    # fabricated current states or realized counterfactual investments.
    slot=81
    for item in protocol['nonautomated_jobs']:
        job=jobs[item['job_id']];check(item['scenario_id']==min(job['evaluation_ids']),'noauto/frozen_identity')
        missing_group=f"c{job['parent_context_id']}__nested_dynamic_learner"
        check(missing_group not in parent_groups['actual'],'noauto/no_saved_account_exists')
        for year in item['years']:
            slot+=1;state_id=f'q{slot:03d}';r=inventory.loc[inventory.state_id==state_id];check(len(r)==1,'noauto/inventory_slot');r=r.iloc[0]
            expected_row=dict(archive_scope='actual',analysis_scope='outer_oof',job_id=item['job_id'],context=job['parent_context'],
                scenario_id=item['scenario_id'],year=year,training_n=len(job['training_ids']),status='nonautomated_skip_no_replacement')
            for field,value in expected_row.items():maybe(r[field],value,'noauto/'+field)
            check(not branches.state_id.eq(state_id).any() and not contrasts.state_id.eq(state_id).any() and state_id not in saved_groups,'noauto/no_fallback_finance')
    expected_summaries=independent_summary(results)
    check(len(summaries)==len(expected_summaries),'summary/group_count')
    for expected_row in expected_summaries:
        g=summaries.loc[summaries.level.eq(expected_row['level'])&summaries.analysis_scope.eq(expected_row['analysis_scope'])]
        if expected_row['context'] is not None:g=g.loc[g.context==expected_row['context']]
        else:g=g.loc[g.context.isna()]
        if expected_row['year'] is not None:g=g.loc[g.year==expected_row['year']]
        else:g=g.loc[g.year.isna()]
        check(len(g)==1,'summary/unique_identity');r=g.iloc[0]
        for field,value in expected_row.items():maybe(r[field],value,'summary/'+field)
    expected_scopes={'outer_oof':6,'full_development_descriptive':3,'inner_validation':70}
    check({k:sum(r['analysis_scope']==k for r in results) for k in expected_scopes}==expected_scopes,'scope/6outer3full70inner')
    check(manifest['analysis_scope_counts']==expected_scopes,'scope/manifest_active_counts')
    check(len(branch_scalars)==158,'independent/158branches')
    after=evidence_snapshot(directory);check(before==after,'readonly/source_input_outputs_unchanged')
    independent_npz=directory/'independent_branch_states.npz'
    np.savez_compressed(independent_npz,**independent_states)
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks=check_n,maximum_errors=errors,
        inventory_slots=90,old_fixed_Q_identities=81,active_fixed_states=79,pending_skips=2,nonautomated_slots=9,
        independent_branches=158,independent_branch_periods=branch_periods,analysis_scope_counts=expected_scopes,
        synthetic_checks=synthetic,branch_scalars=branch_scalars,
        source_input_output_sha256_start=before,source_input_output_sha256_end=after,
        independent_branch_states_sha256=sha(independent_npz),independent_branch_state_array_count=len(independent_states),
        parent_manifest_sha256=sha(PARENT/'manifest.json'),parent_Q_audit_sha256=sha(PARENT/'independent_audit.json'),
        original_diagnostic_manifest_sha256=sha(directory/'manifest.json'),final_scenarios_loaded=False,
        numerical_guard=dict(state_atol=3e-11,state_rtol=3e-13,old_score_atol=3e-12,branch_event_and_action_exact=True),
        independence='No new counterfactual module, runner, simulator or metric helper imported. Scalar fee uses the previous independent100-bisection helper; own calendar/state/summary recursion.',
        limits=['Only the fixed79 states and158 current-action-then-B branches; no new dynamic policy evaluated.',
          'Predicted Q values/counts copied from previously independently audited parent arrays, not newly calibrated probabilities.',
          'Six actual outer states, three full-development states and70inner states are kept separate; they are correlated/repeated decision states.',
          'Nine nonautomated slots and two pending states never receive replacement financial accounts.',
          'Saved independent branch traces support arithmetic reproducibility, not oracle policies, causal policy effects or risk certification.'])
    (directory/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    result=run_audit()
    print(json.dumps(dict(status=result['status'],checks=result['checks'],branches=result['independent_branches'],
                         output=str(OUT/'independent_audit.json'))),flush=True)
