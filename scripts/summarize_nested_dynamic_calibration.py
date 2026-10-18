"""Summarize a stable, independently audited complete-process dynamic study.

Preparation: ``--self-test`` reads no research results. Actual extraction is
explicitly opt-in with ``--include-stable-new`` after the root's stable notice.
No market loader, fitter, forecast, execution policy, or winner selector is
imported. The root-owned result directory and all legacy archives stay read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research'))
from safeinvest_research.metrics import discrete_cvar, SUMMARY_METADATA

SOURCE = ROOT/'results/model_review/nested_dynamic_calibration_v1'
LEGACY = ROOT/'results/model_review/q2_execution_development_v2'
LEGACY_PRICE = ROOT/'results/model_review/risk_priced_fan_execution_v1'
OUT = ROOT/'results/model_review/nested_dynamic_calibration_summary_v1'
NEW_POLICIES = ('nested_dynamic_learner','annual_then_pending','equal_thirds_annual')
IDS = {'nested_dynamic_learner':'new_nested_dynamic','annual_then_pending':'new_pending_reference','equal_thirds_annual':'new_equal_reference'}
LEGACY_IDS = {'risk_priced_learner':'legacy_risk_priced','strict_annual':'legacy_strict_annual','annual_then_pending':'legacy_pending_reference','equal_weight_annual':'legacy_equal_reference'}
ORDER = ['legacy_risk_priced','new_nested_dynamic','legacy_strict_annual','legacy_pending_reference','new_pending_reference','legacy_equal_reference','new_equal_reference']
CORE = ['log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','nominal_moved','trade_fraction']
FINANCIAL = CORE + ['breach','excess']
SERVICE_STATUSES = {'automated','execution_review','risk_review','not_balanced_case'}
TOL = 1e-12


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def scalar(value):
    if value is None or pd.isna(value): return None
    return value.item() if isinstance(value,np.generic) else value


def bool_column(frame,name):
    if name not in frame or not frame[name].map(lambda x:isinstance(x,(bool,np.bool_))).all():
        raise RuntimeError(f'{name} must be an observed boolean, not inferred from a string/NULL')


def case_of(label):
    if label=='Balanced': return 'balanced_case'
    if label=='Review': return 'risk_review'
    if label in ('Conservative','Growth'): return 'not_balanced_case'
    raise RuntimeError(f'Unknown Q1 effective assignment: {label}')


def frame_denominators(frame,reference):
    bool_column(frame,'assigned')
    case=frame.q1_effective_assignment.map(case_of)
    if 'case_status' in frame and not case.eq(frame.case_status).all(): raise RuntimeError('Q1 assignment/case classification mismatch')
    result={'original_n':len(frame),'balanced_case_n':int(case.eq('balanced_case').sum()),
            'risk_review_n':int(case.eq('risk_review').sum()),'not_balanced_case_n':int(case.eq('not_balanced_case').sum())}
    if result['original_n']!=sum(result[k] for k in ['balanced_case_n','risk_review_n','not_balanced_case_n']): raise RuntimeError('Original cohort identity failed')
    if reference:
        if not frame.hypothetical_reference.eq(True).all() or not frame.reference_only.eq(True).all(): raise RuntimeError('Reference must stay explicitly hypothetical')
        expected=np.where(frame.assigned,'hypothetical_reference','reference_unavailable')
        if not np.array_equal(frame.execution_status.to_numpy(),expected): raise RuntimeError('Reference financial availability was mistaken for customer service')
        result.update({'case_auto_n':None,'case_execReview_n':None,'service_auto_n':None,
                       'reference_available_n':int(frame.assigned.sum()),'case_reference_available_n':int((frame.assigned & case.eq('balanced_case')).sum())})
    else:
        if not set(frame.execution_status).issubset(SERVICE_STATUSES): raise RuntimeError('Unknown learner service status')
        if not frame.assigned.eq(frame.execution_status.eq('automated')).all(): raise RuntimeError('Learner automatic service/financial availability disagree')
        if not case.loc[frame.execution_status.isin(['automated','execution_review'])].eq('balanced_case').all(): raise RuntimeError('Non-Balanced Q1 response entered Q2 service')
        for status in ['risk_review','not_balanced_case']:
            if not case.loc[frame.execution_status.eq(status)].eq(status).all(): raise RuntimeError('Risk referral and outside-case status were conflated')
        result.update({'case_auto_n':int(frame.execution_status.eq('automated').sum()),'case_execReview_n':int(frame.execution_status.eq('execution_review').sum()),'reference_available_n':None,'case_reference_available_n':None})
        result['service_auto_n']=result['case_auto_n']
        if result['balanced_case_n']!=result['case_auto_n']+result['case_execReview_n']: raise RuntimeError('Balanced-case identity failed')
    result.update({'assigned_paths':int(frame.assigned.sum()),'financial_null_paths':int((~frame.assigned).sum()),'reference_only':bool(reference)})
    return result


def check_financial(frame):
    bool_column(frame,'assigned');observed=frame.loc[frame.assigned];null=frame.loc[~frame.assigned]
    if not np.isfinite(observed[CORE].to_numpy(float)).all(): raise RuntimeError('Available financial outcome has a nonfinite metric')
    if not null[FINANCIAL].isna().all().all(): raise RuntimeError('Nonautomated/unavailable outcome was imputed')
    if observed.empty:return
    if not np.allclose(np.expm1(observed.log_growth),observed.cagr,atol=1e-13,rtol=1e-12): raise RuntimeError('G/CAGR identity failed')
    if not np.allclose(observed.log_growth,np.log(observed.terminal_real/100)/30,atol=1e-13,rtol=1e-12): raise RuntimeError('G/actual terminal wealth identity failed')
    bool_column(observed,'breach')
    if not np.array_equal(observed.breach.to_numpy(bool),(observed.mdd-.4>TOL).to_numpy()): raise RuntimeError('Whole-H breach definition changed')
    if not np.allclose(observed.trade_fraction*30,np.rint(observed.trade_fraction*30),atol=1e-11,rtol=0): raise RuntimeError('Actual trade fraction is not an integer count / H30')
    if not np.allclose(observed.nominal_fees,.0025*observed.nominal_moved,atol=1e-11,rtol=1e-12): raise RuntimeError('Fee counted incorrectly relative to actual transfer')


def independent_cvar(values):
    a=np.sort(np.asarray(values,float))[::-1];mass=.1*len(a);whole=int(math.floor(mass));fraction=mass-whole
    total=float(a[:whole].sum())
    if fraction>TOL: total+=float(a[whole])*fraction
    return total/mass


def summarize_frame(frame):
    reference=bool(frame.reference_only.iloc[0]);denom=frame_denominators(frame,reference);check_financial(frame)
    observed=frame.loc[frame.assigned];n=len(observed)
    out={'scope':str(frame.scope.iloc[0]),'policy':str(frame.policy.iloc[0]),**denom,
         'financial_population':'hypothetical_available_reference_paths' if reference else 'automatic_service_paths',
         'conditional_risk_denominator':n,'service_auto_fraction':None if reference else denom['case_auto_n']/len(frame),
         'reference_financial_availability_fraction':n/len(frame) if reference else None}
    fields=['breach_count','conditional_breach_probability','breaches_per_original_n','mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd','mean_nominal_fees','mean_trade_fraction','mean_excess','actual_trade_count']
    if not n: return {**out,**dict.fromkeys(fields)}
    c=np.sort(observed.cagr.to_numpy());d=np.sort(observed.mdd.to_numpy());k=int((d-.4>TOL).sum())
    tail=float(discrete_cvar(d))
    if not np.isclose(tail,independent_cvar(d),atol=1e-13,rtol=0): raise RuntimeError('Fractional-tail CVaR differs from independent weighted calculation')
    out.update({'breach_count':k,'conditional_breach_probability':k/n,'breaches_per_original_n':k/len(frame),
                'mean_log_growth':float(observed.log_growth.mean()),'median_cagr':float(np.median(c)),'p10_cagr':float(c[math.ceil(.1*n)-1]),
                'median_mdd':float(np.median(d)),'p90_mdd':float(d[math.ceil(.9*n)-1]),'cvar90_mdd':tail,
                'mean_nominal_fees':float(observed.nominal_fees.mean()),'mean_trade_fraction':float(observed.trade_fraction.mean()),
                'mean_excess':float(np.maximum(d-.4,0).mean()),'actual_trade_count':int(np.rint((observed.trade_fraction*30).sum()))})
    return out


def compare_saved(summary,saved):
    maximum=0.
    for row in summary:
        old=saved.loc[(saved.scope==row['scope'])&(saved.policy==row['policy'])]
        if len(old)!=1: raise RuntimeError('Missing/duplicate saved summary identity')
        old=old.iloc[0]
        fields=['original_n','balanced_case_n','risk_review_n','not_balanced_case_n','case_auto_n','case_execReview_n','assigned_paths','financial_null_paths','reference_available_n','case_reference_available_n','breach_count','conditional_breach_probability','breaches_per_original_n','mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd','mean_nominal_fees','mean_trade_fraction','mean_excess']
        for key in fields:
            a,b=scalar(row[key]),scalar(old[key])
            if a is None or b is None:
                if a is not None or b is not None: raise RuntimeError(f'Source summary NULL mask differs: {row["policy"]}/{key}')
            else:
                maximum=max(maximum,abs(float(a)-float(b)))
                if not np.isclose(float(a),float(b),atol=1e-12,rtol=1e-12): raise RuntimeError(f'Recomputed/source summary differs: {row["policy"]}/{key}')
    return maximum


def cohort_check(frame,legacy_keys=None):
    for (scope,policy),part in frame.groupby(['scope','policy'],sort=False):
        if len(part)!=120 or part.scenario_id.nunique()!=120 or set(part.scenario_id)!=set(range(1,121)): raise RuntimeError('Every original policy cohort must contain IDs1--120 exactly once')
        if scope=='outer_oof':
            if not part.groupby('outer_fold').size().eq(24).all(): raise RuntimeError('Five original outer folds of24 are required')
            keys=part[['scenario_id','outer_fold']].astype(int).sort_values('scenario_id').reset_index(drop=True)
            if legacy_keys is not None and not keys.equals(legacy_keys): raise RuntimeError('New/old OOF identity mapping differs')
        if not part.horizon.eq(30).all() or not part.loss_tolerance.eq(.4).all(): raise RuntimeError('Original C H30/L40 changed')


def load_legacy():
    manifest=read_json(LEGACY/'manifest.json')
    for name in ['oof_paths.csv','summary.csv']:
        if sha(LEGACY/name)!=manifest['outputs_sha256'][name]: raise RuntimeError('Frozen legacy Q2 input changed')
    frame=pd.read_csv(LEGACY/'oof_paths.csv');frame=frame.loc[frame.policy.isin(LEGACY_IDS)].copy()
    frame['source_policy']=frame.policy;frame['comparison_id']=frame.policy.map(LEGACY_IDS);frame['stage']='legacy_frozen_targets_and_prices'
    frame['q1_effective_assignment']='Balanced';frame['case_status']='balanced_case';frame['is_service_learner']=frame.policy.eq('risk_priced_learner')
    frame['reference_only']=~frame.is_service_learner;frame['hypothetical_reference']=frame.reference_only
    frame['execution_status']=np.where(frame.is_service_learner,'automated','hypothetical_reference')
    frame['static_gamma']=.075;frame['training_beta']=np.nan
    if not frame.assigned.all(): raise RuntimeError('Legacy comparison expected the original all120 financial cohorts')
    return frame


def matched_comparisons(frame):
    new=frame.loc[frame.comparison_id=='new_nested_dynamic'].set_index('scenario_id')
    rows=[];summaries=[]
    comparators=[p for p in ORDER if p!='new_nested_dynamic']
    for comp_id in comparators:
        comp=frame.loc[frame.comparison_id==comp_id].set_index('scenario_id')
        if set(comp.index)!=set(new.index): raise RuntimeError('Matched comparison dropped an original identity')
        service=bool(comp.is_service_learner.iloc[0])
        for sid in sorted(new.index):
            a,b=new.loc[sid],comp.loc[sid];common=bool(a.assigned and b.assigned)
            if int(a.outer_fold)!=int(b.outer_fold): raise RuntimeError('Matched paths used different outer folds')
            rows.append({'comparator_id':comp_id,'scenario_id':int(sid),'outer_fold':int(a.outer_fold),
                         'new_service_auto':bool(a.assigned),'comparator_financial_available':bool(b.assigned),
                         'comparator_is_service_learner':service,'common_evaluable':common,
                         'common_service_auto':common if service else None,
                         **{'new_'+field:float(a[field]) if a.assigned else None for field in ['log_growth','cagr','mdd','nominal_fees','trade_fraction','breach']},
                         **{'comparator_'+field:float(b[field]) if b.assigned else None for field in ['log_growth','cagr','mdd','nominal_fees','trade_fraction','breach']},
                         **{'delta_'+field:float(a[field])-float(b[field]) if common else None for field in ['log_growth','cagr','mdd','nominal_fees','trade_fraction']},
                         'delta_breach':int(bool(a.breach))-int(bool(b.breach)) if common else None})
        part=pd.DataFrame([r for r in rows if r['comparator_id']==comp_id]);both=part.loc[part.common_evaluable]
        summaries.append({'comparator_id':comp_id,'original_pairs':120,'new_service_auto_n':int(part.new_service_auto.sum()),
                          'comparator_financial_available_n':int(part.comparator_financial_available.sum()),
                          'comparator_is_service_learner':service,'common_evaluable_n':len(both),
                          'common_service_auto_n':len(both) if service else None,
                          'new_common_breach_count':int(both.new_breach.sum()) if len(both) else None,
                          'comparator_common_breach_count':int(both.comparator_breach.sum()) if len(both) else None,
                          'new_common_breach_probability':float(both.new_breach.mean()) if len(both) else None,
                          'comparator_common_breach_probability':float(both.comparator_breach.mean()) if len(both) else None,
                          'new_common_mean_log_growth':float(both.new_log_growth.mean()) if len(both) else None,
                          'comparator_common_mean_log_growth':float(both.comparator_log_growth.mean()) if len(both) else None,
                          'new_common_p90_mdd':float(np.quantile(both.new_mdd,.9,method='inverted_cdf')) if len(both) else None,
                          'comparator_common_p90_mdd':float(np.quantile(both.comparator_mdd,.9,method='inverted_cdf')) if len(both) else None,
                          'new_common_cvar90_mdd':float(discrete_cvar(both.new_mdd.to_numpy())) if len(both) else None,
                          'comparator_common_cvar90_mdd':float(discrete_cvar(both.comparator_mdd.to_numpy())) if len(both) else None,
                          'common_saved_outcomes_exactly_equal':bool(np.array_equal(both[['new_'+f for f in ['log_growth','cagr','mdd','nominal_fees','trade_fraction','breach']]].to_numpy(),both[['comparator_'+f for f in ['log_growth','cagr','mdd','nominal_fees','trade_fraction','breach']]].to_numpy())) if len(both) else None,
                          **{'mean_delta_'+field:float(both['delta_'+field].mean()) if len(both) else None for field in ['log_growth','cagr','mdd','nominal_fees','trade_fraction','breach']}})
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def trace_checks(new_paths,groups,archive):
    checked=0;max_error=0.
    for key,group in groups['actual'].items():
        meta=group['metadata']
        if meta.get('scope')!='outer_oof': continue
        policy=meta['policy'];ids=group['scenario_ids'];rows=new_paths.loc[(new_paths.context==meta['context'])&(new_paths.policy==policy)].set_index('scenario_id')
        if set(rows.index)!=set(ids) or not rows.assigned.all(): raise RuntimeError('Trace group does not represent the saved available paths')
        wealth=archive[key+'__real_wealth'];fees=archive[key+'__fees'];moved=archive[key+'__moved'];trades=archive[key+'__trades']
        if wealth.shape!=(len(ids),31): raise RuntimeError('Unexpected selected actual trace shape')
        mdd=np.max(1-wealth/np.maximum.accumulate(wealth,axis=-1),axis=-1)
        calc={'log_growth':np.log(wealth[:,-1]/100)/30,'cagr':np.expm1(np.log(wealth[:,-1]/100)/30),'mdd':mdd,'terminal_real':wealth[:,-1],
              'nominal_fees':fees.sum(axis=-1),'nominal_moved':moved.sum(axis=-1),'trade_fraction':trades.mean(axis=-1)}
        for field,values in calc.items():
            expected=rows.loc[ids,field].to_numpy(float);err=float(np.max(np.abs(values-expected),initial=0));max_error=max(max_error,err)
            if not np.allclose(values,expected,atol=1e-11,rtol=1e-12): raise RuntimeError(f'Actual saved trace/path metric differs: {key}/{field}')
        checked+=len(ids)
    if checked!=int(new_paths.loc[new_paths.scope=='outer_oof','assigned'].sum()): raise RuntimeError('An available OOF path lacks a corresponding saved actual trace')
    return {'available_oof_paths_checked':checked,'max_abs_error':max_error}



def legacy_behavior_checks(groups,new_archive,old_archive):
    """Describe saved behavior on the new-served subgroup, with no replay."""
    numeric=['nominal_wealth','real_wealth','holdings','weights','posttrade_holdings','fees','moved','real_peak','current_drawdown','running_mdd']
    flags=['actions','trades','pending_at_decision','target_execution','ever_breached','first_crossing']
    results=[];maximum=0.;all_equal=True;flag_differences=0;total=0
    for key,g in groups['actual'].items():
        meta=g['metadata']
        if meta.get('scope')!='outer_oof' or meta['policy']!='nested_dynamic_learner':continue
        old_prefix=meta['context']+'__C__risk_priced_learner__'
        old_ids=old_archive[old_prefix+'scenario_ids'].tolist();ids=g['scenario_ids'];ix=[old_ids.index(sid) for sid in ids]
        differences={};local_max=0.;local_equal=True
        for field in numeric+flags:
            a=new_archive[key+'__'+field];b=old_archive[old_prefix+field][ix]
            if a.shape!=b.shape:raise RuntimeError('Old/new saved behavior comparison shape differs')
            if field in numeric:
                error=float(np.max(np.abs(a-b),initial=0));maximum=max(maximum,error);local_max=max(local_max,error)
                local_equal=local_equal and bool(np.array_equal(a,b,equal_nan=True))
            else:
                differences[field]=int(np.count_nonzero(a!=b));flag_differences+=differences[field]
        all_equal=all_equal and local_equal and not any(differences.values());total+=len(ids)
        results.append({'context':meta['context'],'scenario_ids':ids,'paths':len(ids),'numeric_max_abs_error':local_max,'numeric_arrays_exactly_equal':local_equal,'different_flag_cells':differences})
    return {'scope':'Only the new automatically served OOF subgroup; both inputs are saved actual traces, no simulator called.',
            'common_served_paths':total,'numeric_max_abs_error':maximum,'flag_cell_differences':flag_differences,
            'all_compared_saved_numeric_states_and_actions_exactly_equal':all_equal if total else None,'groups':results}

def context_rows(selections,beta_frame):
    result=[]
    for context,c in selections.items():
        selected=c['beta_selection']['selected_beta'];case=c['case'];product=case.get('product');evidence=None
        if selected is not None:
            e=beta_frame.loc[(beta_frame.context==context)&np.isclose(beta_frame.training_beta,selected,atol=0,rtol=0)]
            if len(e)!=1: raise RuntimeError('Selected beta has no unique saved inner evidence')
            evidence=e.iloc[0]
        price_evidence=c.get('selected_price_training_evidence')
        candidates=price_evidence.get('candidate_summaries',[]) if price_evidence else []
        order=c['selected_candidate_order']
        eta_values=[0.,.001,.003,.01,.03,.1,.3,1.]
        row={'context':context,'q1_effective_assignment':case['effective_assignment'],'case_status':case['status'],
             'service_alpha':.10,'static_training_gamma':c['static_gamma'],'selected_dynamic_training_beta':selected,'selected_price_order':order,'selected_price_kind':None if order is None else 'risk_first' if order==8 else 'finite','selected_annual_eta':eta_values[order] if order is not None and order<8 else None,
             'execution_status':c['execution_status'],'execution_reason':c['execution_reason'],
             'original_training_n':len(c['training_ids']),'evaluation_n':len(c['evaluation_ids']),
             'selected_inner_auto_n':scalar(evidence.case_auto_n) if evidence is not None else None,
             'selected_inner_auto_folds':scalar(evidence.auto_inner_folds) if evidence is not None else None,
             'selected_inner_conditional_breach_probability':scalar(evidence.conditional_breach_probability) if evidence is not None else None,
             'refit_training_breach_cap':price_evidence.get('training_breach_cap') if price_evidence else None,
             'refit_training_reference_breach_count':price_evidence.get('reference_training_breach_count') if price_evidence else None,
             'refit_admissible_price_count':sum(bool(p['training_eligible']) for p in candidates) if candidates else None,
             'refit_min_candidate_breach_count':min(p['breach_count'] for p in candidates) if candidates else None}
        for j,asset in enumerate(['equity','bond','bill']):row['balanced_target_'+asset]=float(product['weights'][j]) if product else None
        result.append(row)
    return pd.DataFrame(result)


def self_test():
    # Case/NULL/reference checks are independent of all market output.
    def rec(sid,label,status,assigned,reference=False):
        return {'scenario_id':sid,'q1_effective_assignment':label,'case_status':case_of(label),'execution_status':status,'assigned':assigned,'reference_only':reference,'hypothetical_reference':reference}
    f=pd.DataFrame([rec(1,'Balanced','automated',True),rec(2,'Balanced','execution_review',False),rec(3,'Review','risk_review',False),rec(4,'Growth','not_balanced_case',False)])
    d=frame_denominators(f,False);assert d['original_n']==4 and d['balanced_case_n']==2 and d['case_auto_n']==1 and d['case_execReview_n']==1 and d['risk_review_n']==d['not_balanced_case_n']==1
    ref=pd.DataFrame([rec(1,'Balanced','hypothetical_reference',True,True),rec(2,'Review','hypothetical_reference',True,True)])
    d=frame_denominators(ref,True);assert d['reference_available_n']==2 and d['case_reference_available_n']==1 and d['case_auto_n'] is None and d['service_auto_n'] is None
    bad=f.copy();bad.loc[3,'execution_status']='automated';bad.loc[3,'assigned']=True
    try:frame_denominators(bad,False)
    except RuntimeError:pass
    else:raise AssertionError('Non-Balanced auto case was admitted')
    for n in [1,24,48,96,120]:
        a=np.linspace(0,.6,n);assert np.isclose(discrete_cvar(a),independent_cvar(a),atol=1e-13,rtol=0)
    # No automatic outcome is unavailable, rather than a zero-risk outcome.
    zero=f.iloc[1:].copy();zero['scope']='outer_oof';zero['policy']='nested_dynamic_learner'
    for field in FINANCIAL:zero[field]=None
    z=summarize_frame(zero)
    assert z['case_auto_n']==0 and z['conditional_risk_denominator']==0 and z['breach_count'] is None and z['conditional_breach_probability'] is None and z['mean_log_growth'] is None
    # References can be available outside Balanced while remaining non-service.
    available=ref.copy();available['scope']='outer_oof';available['policy']='equal_thirds_annual'
    vals={'log_growth':.01,'cagr':float(np.expm1(.01)),'mdd':.2,'volatility':.1,'terminal_real':float(100*np.exp(.3)),'nominal_fees':.1,'nominal_moved':40.,'trade_fraction':1/30,'breach':False,'excess':0.}
    for field,value in vals.items():available[field]=value
    z=summarize_frame(available);assert z['reference_available_n']==2 and z['case_reference_available_n']==1 and z['case_auto_n'] is None and z['conditional_risk_denominator']==2
    # Preserve all original paired identities while scoring only the common
    # automatic/evaluable subgroup; references never become service auto.
    synthetic=[]
    for cid in ORDER:
        for sid in range(1,121):
            serving=(sid<=24) if cid=='new_nested_dynamic' else True
            synthetic.append({'comparison_id':cid,'scenario_id':sid,'outer_fold':1+(sid-1)//24,'assigned':serving,
                              'is_service_learner':cid in ['legacy_risk_priced','new_nested_dynamic'],
                              'log_growth':.02 if serving else np.nan,'cagr':float(np.expm1(.02)) if serving else np.nan,
                              'mdd':.2 if serving else np.nan,'nominal_fees':.1 if serving else np.nan,
                              'trade_fraction':.5 if serving else np.nan,'breach':False if serving else None})
    paired,ps=matched_comparisons(pd.DataFrame(synthetic));assert len(paired)==720 and ps.original_pairs.eq(120).all() and ps.common_evaluable_n.eq(24).all()
    assert ps.loc[ps.comparator_id=='legacy_risk_priced','common_service_auto_n'].iloc[0]==24
    assert ps.loc[ps.comparator_id!='legacy_risk_priced','common_service_auto_n'].isna().all()
    print('Synthetic preparation tests passed: service identities, reference availability, non-Balanced exclusion, zero-auto NULL, matched24/original120 pairs and fractional CVaR; no research result read')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--include-stable-new',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    if not args.include_stable_new:parser.error('Use --self-test during preparation; --include-stable-new requires root stable notice and passed independent audit')
    audit=read_json(SOURCE/'independent_audit.json')
    if audit['status']!='passed':raise RuntimeError('Independent new-study audit must pass before extracting or charting results')
    manifest=read_json(SOURCE/'manifest.json')
    if manifest['final_scenarios_loaded'] or manifest['candidate_count']!=9 or manifest['training_tasks']!=30 or manifest['scenario_ids_loaded']!=list(range(1,121)):raise RuntimeError('New archive is incomplete or outside the declared9-price development scope')
    source_files=[Path(__file__).resolve(),ROOT/'research/safeinvest_research/metrics.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    new_names=['manifest.json','independent_audit.json','client_paths.csv','summary.csv','context_selections.json','beta_summaries.csv','trace_groups.json','actual_traces.npz']
    input_files=[SOURCE/n for n in new_names]+[LEGACY/'manifest.json',LEGACY/'oof_paths.csv',LEGACY/'summary.csv',LEGACY_PRICE/'manifest.json',LEGACY_PRICE/'actual_traces.npz',ROOT/'docs/nested_dynamic_calibration_protocol.json']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    for name in new_names:
        if name not in ['manifest.json','independent_audit.json'] and sha(SOURCE/name)!=manifest['outputs_sha256'][name]:raise RuntimeError(f'Frozen new output changed: {name}')
    new=pd.read_csv(SOURCE/'client_paths.csv');saved=pd.read_csv(SOURCE/'summary.csv')
    if set(new.policy)!=set(NEW_POLICIES) or set(new.scope)!={'full_development','outer_oof'} or not new.client.eq('C').all():raise RuntimeError('Unexpected new scope/policies')
    new['outer_fold']=np.where(new.scope.eq('outer_oof'),new.context_id,np.nan)
    legacy=load_legacy();keys=legacy.loc[legacy.policy=='risk_priced_learner',['scenario_id','outer_fold']].astype(int).sort_values('scenario_id').reset_index(drop=True)
    cohort_check(new,keys);cohort_check(legacy,keys)
    new_summaries=[summarize_frame(g) for _,g in new.groupby(['scope','policy'],sort=False)]
    summary_error=compare_saved(new_summaries,saved)
    groups=read_json(SOURCE/'trace_groups.json')
    old_price_manifest=read_json(LEGACY_PRICE/'manifest.json')
    if sha(LEGACY_PRICE/'actual_traces.npz')!=old_price_manifest['outputs_sha256']['actual_traces.npz']:raise RuntimeError('Frozen old price actual trace changed')
    with np.load(SOURCE/'actual_traces.npz',allow_pickle=False) as archive:
        trace_receipt=trace_checks(new,groups,archive)
        with np.load(LEGACY_PRICE/'actual_traces.npz',allow_pickle=False) as old_archive:behavior_receipt=legacy_behavior_checks(groups,archive,old_archive)
    new['source_policy']=new.policy;new['comparison_id']=new.policy.map(IDS);new['stage']='new_complete_nested_learning';new['is_service_learner']=new.policy.eq('nested_dynamic_learner')
    comparison=pd.concat([legacy,new.loc[new.scope=='outer_oof']],ignore_index=True)
    comparison_summaries=[]
    for cid,g in comparison.groupby('comparison_id',sort=False):
        row=summarize_frame(g);row.update({'comparison_id':cid,'stage':g.stage.iloc[0],'source_policy':g.source_policy.iloc[0]});comparison_summaries.append(row)
    comparison_summary=pd.DataFrame(comparison_summaries).set_index('comparison_id').loc[ORDER].reset_index()
    pairs,pair_summary=matched_comparisons(comparison)
    selections=read_json(SOURCE/'context_selections.json');beta=pd.read_csv(SOURCE/'beta_summaries.csv');contexts=context_rows(selections,beta)
    if len(contexts)!=6 or len(beta)!=30:raise RuntimeError('Need all six contexts and all five predeclared beta summaries')
    expected={'legacy_strict_annual':9,'legacy_pending_reference':9,'legacy_risk_priced':14,'legacy_equal_reference':6}
    for cid,k in expected.items():
        if int(comparison_summary.set_index('comparison_id').loc[cid,'breach_count'])!=k: raise RuntimeError('Declared legacy C cohort changed')
    # The hypothetical exactthird target/cost/H are unchanged even if Q1 service
    # is selective; compare the same120 physical reference outcomes directly.
    old_eq=comparison.loc[comparison.comparison_id=='legacy_equal_reference'].sort_values('scenario_id')
    new_eq=comparison.loc[comparison.comparison_id=='new_equal_reference'].sort_values('scenario_id')
    if not np.allclose(old_eq[CORE],new_eq[CORE],atol=1e-11,rtol=1e-12,equal_nan=True): raise RuntimeError('New exactthird financial reference differs from its unchanged legacy specification')
    outer_contexts=contexts.loc[contexts.context!='full_development']
    def count_labels(values):
        result={}
        for value in values:
            key='NULL' if pd.isna(value) else str(value)
            result[key]=result.get(key,0)+1
        return result
    stability={'scope':'One full120 descriptive fit and five overlapping96-train outer fits; not six independent replications.',
               'selected_beta_counts_all_contexts':count_labels(contexts.selected_dynamic_training_beta),
               'selected_beta_counts_outer_contexts':count_labels(outer_contexts.selected_dynamic_training_beta),
               'outer_service_status_counts':count_labels(outer_contexts.execution_status),
               'outer_Q1_effective_assignment_counts':count_labels(outer_contexts.q1_effective_assignment),
               'outer_review_reason_counts':count_labels(outer_contexts.execution_reason),
               'outer_selected_beta_distinct_including_NULL':len(count_labels(outer_contexts.selected_dynamic_training_beta)),
               'outer_service_status_distinct':len(count_labels(outer_contexts.execution_status)),
               'selected_inner_auto_n_by_context':{r.context:scalar(r.selected_inner_auto_n) for r in contexts.itertuples()},
               'selected_inner_auto_folds_by_context':{r.context:scalar(r.selected_inner_auto_folds) for r in contexts.itertuples()},
               'interpretation':'Report beta, Q1 case and Review/coverage instability jointly. A smaller observed conditional loss rate on fewer served paths is not by itself an improved service. NULL beta or zero auto evidence is not beta0 or zero risk.',
               'parameters':'service_alpha=.10 is the declared service reference; static gamma and dynamic beta are distinct technical training screens, not elicited customer risk probabilities.'}
    full_summary=next(r for r in new_summaries if r['scope']=='full_development' and r['policy']=='nested_dynamic_learner')
    outer_summary=next(r for r in new_summaries if r['scope']=='outer_oof' and r['policy']=='nested_dynamic_learner')
    keys_stability=['original_n','balanced_case_n','case_auto_n','case_execReview_n','risk_review_n','not_balanced_case_n','breach_count','conditional_breach_probability']
    stability['full120_training_refit_in_sample']={k:full_summary[k] for k in keys_stability}
    stability['outer_oof_complete_procedure']={k:outer_summary[k] for k in keys_stability}
    stability['legacy_price_common_served_comparison']=pair_summary.loc[pair_summary.comparator_id=='legacy_risk_priced'].to_dict('records')[0]
    stability['saved_actual_behavior_on_common_served_subgroup']=behavior_receipt
    common_legacy=stability['legacy_price_common_served_comparison']
    same_common=common_legacy['common_saved_outcomes_exactly_equal']
    subgroup_finding=('On the common '+str(common_legacy['common_evaluable_n'])+' automatically served paths, saved new and legacy priced financial outcomes are exactly equal. The smaller new total observed breach count is accompanied by withheld automatic service; it is not evidence of improved trading behavior for that same subgroup.' if same_common else 'Use the same automatically served subgroup for any financial contrast; the overall conditional rates describe different service coverage.')
    stability['served_subgroup_interpretation']=subgroup_finding
    limits=['Reused development; not a fresh blind evaluation or fixed-policy risk certificate.',
            'OOF folds learn different menus/gammas/betas/prices; overlapping training folds do not justify naive pooled binomial certification.',
            'New and legacy raw conditional metrics may describe different automatically served subsets. Coverage is shown alongside risk; only common-evaluable pairs share the same IDs.',
            'Reference assigned_paths means hypothetical financial availability, never automatic customer service; case_auto_n/case_execReview_n stay NULL.',
            'New complete procedure reselects Q1 menu/assignment as well as execution calibration. Differences from legacy are not a pure one-parameter causal effect.',
            'A zero automatic-service denominator gives NULL risk/growth, not zero risk. Observed breaches per original120 do not impute outcomes for Review or outside-Balanced cases.',
            'Risk-only questionnaire process still has an unresolved liquidity service rule; no final service adoption or final policy selection is made.',
            'Nine-price fixed-target family only; no18-controller extension or target update is evaluated here.',
            'Service alpha=.10 is a public policy reference; static gamma and dynamic beta are technical screens, not identified questionnaire preferences.']
    OUT.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(OUT/'oof_comparison_paths.csv',index=False);comparison_summary.to_csv(OUT/'oof_comparison_summary.csv',index=False)
    pd.DataFrame(new_summaries).to_csv(OUT/'new_scope_summary.csv',index=False);pairs.to_csv(OUT/'common_evaluable_pairs.csv',index=False);pair_summary.to_csv(OUT/'common_evaluable_summary.csv',index=False)
    contexts.to_csv(OUT/'context_selections.csv',index=False);beta.to_csv(OUT/'inner_beta_evidence.csv',index=False)
    write_json(OUT/'stability_diagnostics.json',stability)
    write_json(OUT/'interpretation.json',{'purpose':'Descriptive extraction, not a new policy selector.','service_vs_reference':'Only learner automated rows count as customer automatic service. All reference outcomes remain hypothetical financial results.','original_denominator':120,'conditional_denominator':'Observed learner auto_n or reference_available_n as labelled; NULL if zero.','served_subgroup_finding':subgroup_finding,'saved_actual_behavior_on_common_served_subgroup':behavior_receipt,'paired_denominator':'New learner automatic service intersected with comparator financial availability on the same originalscenario and outer fold; common_service_auto applies only to both learners.','summary_metadata':SUMMARY_METADATA,'limits':limits,'final_scenarios_loaded':False,'policy_selected_from_this_comparison':False})
    sources_end={p:sha(ROOT/p) for p in sources};inputs_end={p:sha(ROOT/p) for p in inputs}
    if sources!=sources_end or inputs!=inputs_end:raise RuntimeError('Source or saved input changed during extraction')
    names=['oof_comparison_paths.csv','oof_comparison_summary.csv','new_scope_summary.csv','common_evaluable_pairs.csv','common_evaluable_summary.csv','context_selections.csv','inner_beta_evidence.csv','stability_diagnostics.json','interpretation.json']
    write_json(OUT/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_audited_saved_artifact_extraction','sources':sources,'sources_at_end':sources_end,'input_sha256':inputs,'input_sha256_at_end':inputs_end,'outputs_sha256':{n:sha(OUT/n) for n in names},'checks':{'recomputed_saved_summary_max_abs_error':summary_error,'actual_trace_receipt':trace_receipt,'legacy_common_served_behavior_receipt':behavior_receipt,'all_policy_groups_original120_and_same_oof_fold_mapping':True,'service_and_reference_denominators_separated':True,'Review_and_outside_case_financial_NULL_preserved':True,'common_evaluable_original_pair_count_per_comparator':120},'final_scenarios_loaded':False,'raw_market_loaded':False,'new_fit_forecast_execution_or_winner_selection':False,'limitations':limits})
    print(comparison_summary[['comparison_id','original_n','case_auto_n','reference_available_n','breach_count','conditional_breach_probability']].to_string(index=False));print('Wrote',OUT)


if __name__=='__main__':main()
