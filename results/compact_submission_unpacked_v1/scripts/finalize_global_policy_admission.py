"""No fitting: retain the existing global learner, add one fixed inner guard.

No old parameter/target or rejected fit is changed. All financial outcomes
remain hypothetical development evidence, with new Review outcomes NULL.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/global_policy_admission_protocol.json'
OUT=ROOT/'results/model_review/global_policy_admission_v1'
OLD='joint_tie_set_refit_learner';POLICY='global_policy_with_same_U_guard'
REFS=('annual_then_pending','equal_thirds_annual')
FIN=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','nominal_moved','trade_fraction','target_execution_fraction','terminal_equity','terminal_bond','terminal_bill','breach','excess','first_crossing')
TOL=1e-12

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):
    p=Path(p)
    if p.suffix=='.gz':
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(p.read_text())
def writej(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csv(p,rows):Path(p).write_bytes(gzip.compress(pd.DataFrame(rows).to_csv(index=False,lineterminator='\n').encode(),mtime=0))


def inner_guard(rows,B,expected_ids):
    if len(rows)!=len(expected_ids) or sorted(rows.scenario_id)!=sorted(expected_ids) or rows.scenario_id.duplicated().any():raise ValueError('Complete original inner denominator required')
    if len(B)!=len(expected_ids) or sorted(B.scenario_id)!=sorted(expected_ids) or B.scenario_id.duplicated().any():raise ValueError('Complete B original denominator required')
    u=rows.loc[rows.assigned];kb=None;k=None;n=len(u)
    if n:
        paired=u.merge(B,on=['job_id','scenario_id'],suffixes=('_model','_B'),validate='one_to_one')
        if len(paired)!=n or not paired.assigned_B.all():raise ValueError('B must be available on exactly own U')
        for col in ('candidate_local_index','candidate_original232_index','static_gamma','q1_effective_assignment'):
            if not np.array_equal(paired[col+'_model'].to_numpy(),paired[col+'_B'].to_numpy()):raise ValueError('B own-fit target/label/gamma mismatch')
        k=int(paired.breach_model.sum());kb=int(paired.breach_B.sum())
    cap=math.floor(.1*n+TOL) if n else None
    reasons=[] if n else ['no_inner_auto_evidence']
    if n and k>cap:reasons.append('inner_absolute_risk_exceeds_alpha')
    if n and k>kb:reasons.append('inner_risk_exceeds_same_U_B')
    return dict(original_inner_n=len(rows),auto_inner_n=n,k_inner=k,k_B_same_U=kb,absolute_cap=cap,
        auto_inner_ids=sorted(map(int,u.scenario_id)),eligible=not reasons,reasons=reasons)


def justification(qh,qr,kh,kr,n,eta,H=30):
    if not all(np.isfinite(x) for x in (qh,qr)) or n<1 or not (0<=kh<=n and 0<=kr<=n):raise ValueError('Valid paired netQ/count evidence required')
    dg=qr-qh;dp=(kr-kh)/n
    score=dg-H*eta*dp if eta is not None else None
    restore=(score>TOL) if eta is not None else (kr<kh or kr==kh and dg>TOL)
    kind=('net_growth_improvement' if dg>TOL else 'risk_reduction_with_growth_cost') if restore else 'hold'
    if restore and dg<=TOL and not kr<kh:raise ValueError('Nonpositive net growth restore must reduce model event count')
    return dict(recommend_restore=bool(restore),delta_net_total_log_growth=dg,delta_model_breach_probability=dp,
        risk_adjusted_net_advantage=score,justification=kind,strict_growth_sacrifice=bool(restore and dg < -TOL))


def main():
    p=readj(PROTOCOL)
    if p['execution_ready'] is not True or p['retained_learner']!=OLD or p['final_policy']!=POLICY or p['service_alpha']!=.1:raise RuntimeError('Fixed retention protocol required')
    if OUT.exists():raise RuntimeError('Never overwrite a result or failed attempt')
    source={'scripts/finalize_global_policy_admission.py':sha(__file__),'tests/test_global_policy_admission.py':sha(ROOT/'tests/test_global_policy_admission.py')}
    inputs={rel:sha(ROOT/rel) for rel in p['frozen_input_sha256']}
    if source!=p['execution_sources_sha256'] or inputs!=p['frozen_input_sha256']:raise RuntimeError('Frozen sources/inputs differ')
    protocol_sha=sha(PROTOCOL)
    ref=pd.read_csv(ROOT/'results/model_review/benefit_cost_execution_v1/inner_reference_paths.csv.gz')
    contexts=readj(ROOT/'docs/refit_stability_protocol.json')['contexts']
    full=readj(ROOT/'results/model_review/joint_liquidity_refit_v1/static_refits.json')['c0_refit']['context']
    contexts=contexts+[{**full,'repeat_id':0,'outer_fold':0}]
    archives={};admissions=[];clients=[];costs=[]
    for group,base in [('repeat','refit_stability_v1'),('full','joint_liquidity_refit_v1')]:
        folder=ROOT/'results/model_review'/base;suffix='.csv.gz' if group=='repeat' else '.csv'
        archives[group]=dict(choices=readj(folder/'context_selections.json'),inner=pd.read_csv(folder/('inner_validation_paths'+suffix)),
            clients=pd.read_csv(folder/('client_paths'+suffix)),groups=readj(folder/'trace_groups.json')['actual'],folder=folder)
    for c in contexts:
        a=archives['full' if c['context_id']==0 else 'repeat'];name=c['name'];choice=a['choices'][name]['methods'][OLD]
        beta=choice['selected_beta'];oldauto=choice['execution_status']=='automated'
        B=ref.loc[ref.context.eq(name)].copy()
        if oldauto:
            selected=a['inner'].loc[a['inner'].context.eq(name)&a['inner'].training_beta.eq(beta)].copy()
            gate=inner_guard(selected,B,c['training_ids'])
        else:gate=dict(original_inner_n=len(c['training_ids']),auto_inner_n=None,k_inner=None,k_B_same_U=None,absolute_cap=None,auto_inner_ids=[],eligible=False,reasons=['original_policy_not_automated'])
        approved=oldauto and gate['eligible'];reason=None if approved else (gate['reasons'][0] if oldauto else choice['execution_reason'])
        case=a['choices'][name]['case']['public_status']
        final_status='automated' if approved else (case if case!='balanced_case' else 'execution_review')
        admission=dict(context=name,context_id=c['context_id'],repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],
            original_execution_status=choice['execution_status'],selected_beta=beta,selected_candidate_order=choice['selected_candidate_order'],
            final_automated=bool(approved),execution_status=final_status,execution_reason=reason,**gate)
        admissions.append(admission)
        oldrows=a['clients'].loc[a['clients'].context.eq(name)&a['clients'].policy.isin((OLD,*REFS))]
        if len(oldrows)!=3*len(a['choices'][name]['evaluation_ids']):raise RuntimeError('Complete three-policy evaluated denominator required')
        for row in oldrows.to_dict('records'):
            row.update(repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],scope='in_sample_full120' if c['context_id']==0 else 'outer_oof')
            if row['policy']==OLD:
                row['policy']=POLICY
                if not approved:
                    row.update(assigned=False,execution_status=final_status,execution_reason=reason)
                    row.update(dict.fromkeys(FIN))
            clients.append(row)
        if oldauto:
            candidates=[(key,g) for key,g in a['groups'].items() if g['metadata'].get('context')==name and g['metadata'].get('policy')==OLD]
            if len(candidates)!=1:raise RuntimeError('One unchanged selected old actual trace group required')
            key,g=candidates[0]
            with np.load(a['folder']/'actual_traces.npz',allow_pickle=False) as z:
                arm={k[len(key)+2:]:z[k] for k in z.files if k.startswith(key+'__')}
            raweta=float(arm['annual_prices'].reshape(-1)[0]);eta=raweta if np.isfinite(raweta) else None
            for i,sid in enumerate(g['scenario_ids']):
                for t in range(30):
                    active=t>0 and not bool(arm['pending_at_decision'][i,t]);restored=bool(arm['target_execution'][i,t]);estimate={}
                    if active:
                        estimate=justification(float(arm['qg_hold'][i,t]),float(arm['qg_restore'][i,t]),int(arm['qbad_hold_count'][i,t]),int(arm['qbad_restore_count'][i,t]),int(arm['forecast_n'][i,t]),eta)
                        if estimate['recommend_restore']!=restored:raise RuntimeError('Original net-cost utility decision differs')
                        if eta is not None and abs(estimate['risk_adjusted_net_advantage']-float(arm['predicted_score'][i,t]))>1e-13:raise RuntimeError('Stored netQ score differs; never charge cost twice')
                    costs.append(dict(context=name,repeat_id=c['repeat_id'],scenario_id=int(sid),year=t+1,old_selected_policy=OLD,
                        final_guard_admitted=bool(approved),audit_scope='original_selected_controller_development',initial=t==0,pending=bool(arm['pending_at_decision'][i,t]),
                        active_optional=bool(active),restore=restored,actual_trade=bool(arm['trades'][i,t]),actual_fee=float(arm['fees'][i,t]),
                        actual_fee_fraction=float(arm['fee_fraction'][i,t]),eta_technical=eta,eta_kind='finite' if eta is not None else 'risk_first',**estimate))
    if len(admissions)!=26 or len(clients)!=2160:raise RuntimeError('All26fits/2160model-reference rows required')
    frame=pd.DataFrame(clients);summary=[]
    for (repeat,policy),g in frame.groupby(['repeat_id','policy'],sort=True):
        auto=g.loc[g.assigned]
        if not g.loc[~g.assigned,list(FIN)].isna().all().all():raise RuntimeError('Review financials must remain NULL')
        n=len(auto);summary.append(dict(repeat_id=int(repeat),policy=policy,scope='in_sample_full120' if repeat==0 else 'outer_oof',original_n=len(g),assigned_n=n,
            breach_count=int(auto.breach.sum()) if n else None,conditional_breach_probability=float(auto.breach.mean()) if n else None,
            mean_log_growth=float(auto.log_growth.mean()) if n else None,median_cagr=float(auto.cagr.median()) if n else None,
            p10_cagr=float(np.quantile(auto.cagr,.1,method='inverted_cdf')) if n else None,median_mdd=float(auto.mdd.median()) if n else None,
            p90_mdd=float(np.quantile(auto.mdd,.9,method='inverted_cdf')) if n else None,mean_fees=float(auto.nominal_fees.mean()) if n else None,
            mean_trade_fraction=float(auto.trade_fraction.mean()) if n else None))
    if source!={k:sha(ROOT/k) for k in source} or inputs!={k:sha(ROOT/k) for k in inputs} or protocol_sha!=sha(PROTOCOL):raise RuntimeError('Read-only source/input changed')
    OUT.mkdir(parents=True);csv(OUT/'admission.csv.gz',admissions);csv(OUT/'client_paths.csv.gz',clients);csv(OUT/'summary.csv.gz',summary);csv(OUT/'trade_estimates.csv.gz',costs)
    writej(OUT/'admission.json',admissions)
    writej(OUT/'manifest.json',dict(status='complete',protocol_sha256=protocol_sha,sources=source,sources_at_end=source,input_sha256=inputs,input_sha256_at_end=inputs,
        outputs_sha256={f.name:sha(f) for f in OUT.iterdir() if f.is_file()},new_fits=0,market_data_loaded=False,final_scenarios_loaded=False,
        row_counts=dict(admission=26,client_paths=len(clients),summary=len(summary),trade_estimates=len(costs)),
        limitations=['Original parameter/target choices immutable; this is an additional fit-level guard, no fallback or reselection',
            'Old forecasts already include current and future continuation costs; positive utility need not mean positive financial growth',
            'Eta is a finite-controller search parameter, not questionnaire-identified preference or KKT multiplier',
            'Further development on reused original120; five repeats are dependent; full120in_sample is not validation',
            'Parent independent physical audits reused; this postprocess does not rerun prices or forecast fans']))
    print(pd.DataFrame(admissions).drop(columns='auto_inner_ids').to_string(index=False));print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
