"""Post-hoc sufficient scale-equivalence witnesses for all11fixed interventions.

Only saved independently audited arrays are diagnosed.  This does not rerun
markets, policies, forecasts or fitting.  The exact mathematical lemma is
supported separately by synthetic old-reference tests.  All thresholds below
are fixed before the new equivalence computations, with known outcome
motivation explicitly declared in the frozen plan.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/intervention_scale_equivalence_plan.json'
CF=ROOT/'results/model_review/failure_action_intervention_v1'
OUT=ROOT/'results/model_review/intervention_scale_equivalence_v1'
TOL=1e-12
H=30
STATE=('holdings','weights','nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd','ever_breached','cpi')
FLOAT_Q=('qg_hold','qg_restore','predicted_score')
COUNT_Q=('qbad_hold_count','qbad_restore_count','forecast_n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def validate_account(a):
    for name in STATE:
        shape=(H+1,3) if name in ('holdings','weights') else (H+1,)
        value=np.asarray(a[name])
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Complete finite original endpoint states required: '+name)
    if (np.any(a['nominal_wealth']<=0) or np.any(a['real_wealth']<=0) or np.any(a['real_peak']<=0)
        or np.any(a['cpi']<=0) or np.any(a['holdings']<0) or np.any(np.diff(a['running_mdd']) < -TOL)):
        raise ValueError('Positive account/CPI/peak and inherited nonreset MDD required')
    for name in ('actions','policy_actions','forced','pending_at_decision','fees','moved',*FLOAT_Q,*COUNT_Q):
        if np.asarray(a[name]).shape!=(H,):raise ValueError('Complete original period arrays required: '+name)
    if np.asarray(a['posttrade_holdings']).shape!=(H,3):raise ValueError('Complete posttrade holdings required')


def state_errors(c,t,e):
    if not 0<=e<=H:raise ValueError('Original endpoint0..H required')
    scale=float(t['nominal_wealth'][e]/c['nominal_wealth'][e])
    return dict(scale_k=scale,
        max_weight_absolute_gap=float(np.max(np.abs(t['weights'][e]-c['weights'][e]))),
        real_over_peak_absolute_gap=abs(float(t['real_wealth'][e]/t['real_peak'][e]-c['real_wealth'][e]/c['real_peak'][e])),
        running_mdd_absolute_gap=abs(float(t['running_mdd'][e]-c['running_mdd'][e])),
        flags_equal=bool(t['ever_breached'][e]==c['ever_breached'][e]),
        CPI_equal=bool(t['cpi'][e]==c['cpi'][e]))


def equivalent(errors):
    return bool(errors['max_weight_absolute_gap']<=TOL and errors['real_over_peak_absolute_gap']<=TOL
        and errors['running_mdd_absolute_gap']<=TOL and errors['flags_equal'] and errors['CPI_equal'])


def future_checks(c,t,e,k):
    """No rerouting after failure: test every future state and action once."""
    endpoints=slice(e,H+1);periods=slice(e,H)
    errors={}
    def maximum(name,value):errors[name]=float(np.max(np.abs(value),initial=0.))
    for field in ('nominal_wealth','real_wealth','real_peak'):
        maximum(field+'_scale_relative_gap',t[field][endpoints]/c[field][endpoints]/k-1)
    for field in ('weights','current_drawdown','running_mdd'):
        maximum(field+'_absolute_gap',t[field][endpoints]-c[field][endpoints])
    maximum('holdings_normalized_scale_gap',t['holdings'][endpoints]/(k*c['nominal_wealth'][endpoints,None])-c['holdings'][endpoints]/c['nominal_wealth'][endpoints,None])
    for field in ('fees','moved'):
        maximum(field+'_normalized_scale_gap',t[field][periods]/(k*c['nominal_wealth'][e:H])-c[field][periods]/c['nominal_wealth'][e:H])
    maximum('posttrade_holdings_normalized_scale_gap',t['posttrade_holdings'][periods]/(k*c['nominal_wealth'][e:H,None])-c['posttrade_holdings'][periods]/c['nominal_wealth'][e:H,None])
    exact={}
    for field in ('actions','policy_actions','forced','pending_at_decision',*COUNT_Q):exact[field]=bool(np.array_equal(c[field][periods],t[field][periods]))
    exact['ever_breached']=bool(np.array_equal(c['ever_breached'][endpoints],t['ever_breached'][endpoints]))
    exact['cpi']=bool(np.array_equal(c['cpi'][endpoints],t['cpi'][endpoints]))
    exact['no_future_forcing']=bool(not np.any(c['forced'][periods]) and not np.any(t['forced'][periods]))
    for field in FLOAT_Q:
        left,right=c[field][periods],t[field][periods]
        exact[field+'_NULL_mask']=bool(np.array_equal(np.isnan(left),np.isnan(right)))
        valid=np.isfinite(left)&np.isfinite(right)
        maximum(field+'_absolute_gap',left[valid]-right[valid])
    return dict(all_conditions_passed=all(exact.values()) and all(value<=TOL for value in errors.values()),
        maximum_dimensionless_or_Q_error=max(errors.values(),default=0.),maximum_errors=errors,exact_checks=exact,
        future_decision_periods=H-e,future_endpoints_including_witness=H+1-e)


def earliest_attainment(a):
    worst=np.asarray(a['running_mdd']);hits=np.flatnonzero(np.abs(worst-worst[-1])<=TOL)
    return int(hits[0])


def diagnose(c,t,tau):
    validate_account(c);validate_account(t)
    if isinstance(tau,bool) or not isinstance(tau,int) or not 2<=tau<=H:raise ValueError('Original fixed intervention tau required')
    candidate=None;scan=[]
    for e in range(tau,H):
        errors=state_errors(c,t,e);match=equivalent(errors)
        scan.append(dict(endpoint=e,is_candidate=match,**errors))
        if match and candidate is None:candidate=e
    terminal=state_errors(c,t,H)
    attainment_c,attainment_t=earliest_attainment(c),earliest_attainment(t)
    future=None if candidate is None else future_checks(c,t,candidate,state_errors(c,t,candidate)['scale_k'])
    result=dict(earliest_interior_equivalence_endpoint=candidate,
        scale_k=None if candidate is None else state_errors(c,t,candidate)['scale_k'],
        terminal_state_equivalent=equivalent(terminal),terminal_state_errors=terminal,
        control_final_mdd=float(c['running_mdd'][-1]),treatment_final_mdd=float(t['running_mdd'][-1]),
        delta_final_mdd=float(t['running_mdd'][-1]-c['running_mdd'][-1]),
        terminal_mdd_equal_within_declared_tolerance=bool(abs(t['running_mdd'][-1]-c['running_mdd'][-1])<=TOL),
        control_final_mdd_first_attained_endpoint=attainment_c,treatment_final_mdd_first_attained_endpoint=attainment_t,
        equivalence_strictly_before_both_final_mdd_attainments=None if candidate is None else candidate<min(attainment_c,attainment_t),
        future_checks=future,
        status='no_interior_scale_witness' if candidate is None else 'scale_equivalence_verified' if future['all_conditions_passed'] else 'earliest_candidate_failed_future_verification')
    return result,scan


def snapshot(plan):
    result={}
    for key in ('sources_sha256','frozen_input_sha256'):
        if not plan[key]:raise ValueError('Nonempty frozen source/input closure required')
        for rel,digest in plan[key].items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts or len(digest)!=64:raise ValueError('Repository-relative SHA256 closure required')
            actual=sha(ROOT/rel)
            if actual!=digest:raise RuntimeError('Frozen scale diagnostic dependency changed: '+rel)
            result[rel]=actual
    result[str(PLAN.relative_to(ROOT))]=sha(PLAN)
    return result


def main():
    p=readj(PLAN)
    if p.get('execution_ready') is not True:raise RuntimeError('Freeze plan/source/synthetic tests before new equivalence computations')
    if OUT.exists():raise RuntimeError('Refuse to overwrite scale-equivalence archive')
    before=snapshot(p)
    audit=readj(CF/'independent_audit.json');manifest=readj(CF/'manifest.json')
    if audit['status']!='passed' or manifest['status']!='complete' or audit['parent_manifest_sha256']!=sha(CF/'manifest.json'):
        raise RuntimeError('Complete independently audited CF archive required')
    inventory=readj(ROOT/'docs/failure_action_intervention_protocol.json')['inventory']
    records=readj(CF/'intervention_summary.json');by_pair={x['pair_id']:x for x in records}
    if len(records)!=11 or len(by_pair)!=11 or set(by_pair)!={x['pair_id'] for x in inventory}:raise ValueError('All11unchanged original intervention identities required')
    results=[];scans=[]
    with np.load(CF/'full_branches.npz',allow_pickle=False) as archive:
        for item in inventory:
            pid=item['pair_id'];tau=by_pair[pid]['tau']
            if tau is None:
                results.append(dict(pair_id=pid,scenario_id=item['scenario_id'],tau=None,status='no_intervention_preserved',earliest_interior_equivalence_endpoint=None));continue
            arms=[]
            for branch in ('control','treatment'):
                prefix=pid+'__'+branch+'__'
                arms.append({key[len(prefix):]:archive[key] for key in archive.files if key.startswith(prefix)})
            result,scan=diagnose(*arms,tau)
            results.append(dict(pair_id=pid,scenario_id=item['scenario_id'],tau=tau,**result))
            scans.extend(dict(pair_id=pid,scenario_id=item['scenario_id'],tau=tau,**row) for row in scan)
    after=snapshot(p)
    if before!=after:raise RuntimeError('Scale diagnostic source/input/protocol changed')
    OUT.mkdir(parents=True,exist_ok=False)
    writej(OUT/'pair_results.json',results)
    flat=[]
    for row in results:
        flat.append({key:value for key,value in row.items() if not isinstance(value,dict)})
    pd.DataFrame(flat).to_csv(OUT/'pair_summary.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    pd.DataFrame(scans).to_csv(OUT/'endpoint_scan.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    outputs={f.name:sha(f) for f in OUT.iterdir()}
    summary=dict(pairs=11,pairs_with_interior_witness=sum(x.get('earliest_interior_equivalence_endpoint') is not None for x in results),
        pairs_with_verified_scaled_future=sum(x['status']=='scale_equivalence_verified' for x in results),
        failed_earliest_candidates=sum(x['status']=='earliest_candidate_failed_future_verification' for x in results),
        equal_terminal_mdd_pairs=sum(x.get('terminal_mdd_equal_within_declared_tolerance',False) for x in results),
        witnesses_strictly_before_both_mdd_attainments=sum(x.get('equivalence_strictly_before_both_final_mdd_attainments') is True for x in results),
        no_new_market_load=True,new_fits=0,new_policy_simulations=0,admission_changed=False)
    writej(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),
        sources_and_inputs_at_start=before,sources_and_inputs_at_end=after,outputs_sha256=outputs,summary=summary,
        inference='Post-hoc sufficient scaling identity check on all fixed CF paths; no conditional forecast, market or population causal validation'))
    print(json.dumps(summary))

if __name__=='__main__':main()
