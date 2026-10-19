"""Independent admission/score audit; no producer, simulator or market import.

Pure helpers are synthetic-testable before the overlay protocol is frozen.
Artifact reading requires the frozen protocol and complete producer manifest.
The existing parent physical/Q audits are reused; this does not re-estimate
conditional forecasts or replay any market/account ledger.
"""
from __future__ import annotations
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/global_policy_admission_protocol.json'
OUT=ROOT/'results/model_review/global_policy_admission_v1'
OLD='joint_tie_set_refit_learner'
POLICY='global_policy_with_same_U_guard'
REFS=('annual_then_pending','equal_thirds_annual')
PRICES=(0.,.001,.003,.01,.03,.1,.3,1.,None)
FIN=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','nominal_moved',
     'trade_fraction','target_execution_fraction','terminal_equity','terminal_bond','terminal_bill',
     'breach','excess','first_crossing')
TARGET_FIELDS=('candidate_local_index','candidate_original232_index','static_gamma',
    'q1_effective_assignment','case_status','horizon','loss_tolerance',
    'q1_candidate_local_index','q1_candidate_original232_index','selection_training_n',
    'liquidity_policy_fingerprint')
TOL=1e-12


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def readj(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return json.load(stream)
    return json.loads(path.read_text())


def null(value):
    return value is None or isinstance(value,(float,np.floating)) and np.isnan(value)


def integer(value,name,minimum=0):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(name+' must be an integer')
    return int(value)


def observed_bool(value,name):
    if not isinstance(value,(bool,np.bool_)):raise ValueError(name+' must be an observed boolean')
    return bool(value)


def same(a,b):
    if null(a) and null(b):return True
    if null(a) or null(b):return False
    return a==b


def independent_guard(rows,B,expected_ids):
    """Original complete IDs; only assigned U is paired to genuine own-job B."""
    expected=[integer(x,'original ID',1) for x in expected_ids]
    if len(set(expected))!=len(expected) or any(x>120 for x in expected):raise ValueError('Unique original1–120 identities required')
    model=list(rows);baseline=list(B)
    for label,values in [('model',model),('B',baseline)]:
        ids=[integer(r['scenario_id'],label+' ID',1) for r in values]
        if len(values)!=len(expected) or sorted(ids)!=sorted(expected):raise ValueError('Missing/duplicate/foreign '+label+' original IDs')
    bmap={int(r['scenario_id']):r for r in baseline};auto=[];k=kb=0
    for row in model:
        admitted=observed_bool(row['assigned'],'model assigned')
        if not admitted:
            for field in FIN:
                if field in row and not null(row[field]):raise ValueError('Nonautomatic inner financial result must be NULL')
            continue
        sid=int(row['scenario_id']);ref=bmap[sid]
        if row['job_id']!=ref['job_id'] or not observed_bool(ref['assigned'],'B assigned'):
            raise ValueError('B must exist on the identical own-inner job and ID')
        if ref.get('policy','annual_then_pending')!='annual_then_pending':raise ValueError('Risk-price0 is not annual B')
        for field in TARGET_FIELDS:
            if field in row or field in ref:
                if field not in row or field not in ref or not same(row[field],ref[field]):raise ValueError('Own-target/case identity mismatch: '+field)
        auto.append(sid);k+=observed_bool(row['breach'],'model breach');kb+=observed_bool(ref['breach'],'B breach')
    n=len(auto);cap=math.floor(.10*n+TOL) if n else None
    reasons=[]
    if not n:reasons=['no_inner_auto_evidence']
    else:
        if k>cap:reasons.append('inner_absolute_risk_exceeds_alpha')
        if k>kb:reasons.append('inner_risk_exceeds_same_U_B')
    return dict(original_inner_n=len(expected),auto_inner_n=n,k_inner=k if n else None,
        k_B_same_U=kb if n else None,absolute_cap=cap,auto_inner_ids=sorted(auto),eligible=not reasons,reasons=reasons)


def independent_score(qh,qr,kh,kr,n,eta,horizon=30):
    """Fixed net-Q score or separate lexicographic risk-first; no fee offset."""
    n=integer(n,'fan N',1);h=integer(horizon,'original H',1)
    kh=integer(kh,'hold bad count');kr=integer(kr,'restore bad count')
    if max(kh,kr)>n or not all(np.isfinite(v) for v in (qh,qr)):raise ValueError('Finite paired net logs and bounded counts required')
    if eta is not None and (isinstance(eta,(bool,np.bool_)) or not np.isfinite(eta) or eta<0):raise ValueError('Nonnegative finite technical eta or risk-first label required')
    growth=float(qr)-float(qh);probability=(kr-kh)/n
    if eta is None:
        score=None;restore=kr<kh or kr==kh and growth>TOL
    else:
        score=growth-h*float(eta)*probability
        if not np.isfinite(score):raise ValueError('Score exceeds numeric range')
        restore=score>TOL
    if restore and growth<=TOL and kr>=kh:raise ValueError('Nonpositive-growth restore must strictly reduce model events')
    return dict(recommend_restore=bool(restore),delta_net_total_log_growth=growth,
        delta_model_breach_probability=probability,risk_adjusted_net_advantage=score,
        justification=('net_growth_improvement' if growth>TOL else 'risk_reduction_with_growth_cost') if restore else 'hold',
        strict_growth_sacrifice=bool(restore and growth < -TOL))


class Checks:
    def __init__(self):self.n=0;self.maximum_error=0.
    def check(self,condition,label):
        self.n+=1
        if not condition:raise AssertionError(label)
    def value(self,actual,expected,label):
        if null(expected):self.check(null(actual),label+' NULL');return
        if isinstance(expected,(bool,np.bool_)):self.check(isinstance(actual,(bool,np.bool_)) and bool(actual)==bool(expected),label);return
        if isinstance(expected,(list,dict)):
            self.check(actual==expected,label);return
        if isinstance(expected,(int,np.integer)):
            self.check(not isinstance(actual,(bool,np.bool_)) and not null(actual) and actual==expected,label);return
        if isinstance(expected,(float,np.floating)):
            self.check(not null(actual) and np.isfinite(actual),label+' finite')
            error=abs(float(actual)-float(expected));self.maximum_error=max(self.maximum_error,error)
            self.check(error<=2e-12,label+' numeric');return
        self.check(actual==expected,label)
    def record(self,actual,expected,label):
        for field,value in expected.items():self.value(actual.get(field),value,label+'/'+field)


def current_snapshot(p,manifest):
    paths={PROTOCOL,OUT/'manifest.json',Path(__file__).resolve(),ROOT/'tests/test_global_policy_retention_audit.py'}
    paths.update(ROOT/x for x in p['frozen_input_sha256'])
    paths.update(ROOT/x for x in p['execution_sources_sha256'])
    paths.update(OUT/x for x in manifest['outputs_sha256'])
    return {str(path.relative_to(ROOT)):sha(path) for path in sorted(paths)}


def parent_integrity(p,checks):
    """Parent audits bind immutable accounting evidence; do not replay ledgers."""
    for name in ('joint_liquidity_refit_v1','refit_stability_v1','benefit_cost_execution_v1','constrained_fan_execution_v1'):
        folder=ROOT/'results/model_review'/name;m=readj(folder/'manifest.json');a=readj(folder/'independent_audit.json')
        checks.check(a['status']=='passed','parent/'+name+'/audit passed')
        if 'parent_manifest_sha256' in a:checks.value(a['parent_manifest_sha256'],sha(folder/'manifest.json'),'parent binding/'+name)
        else:
            snapshot=a['sources_and_artifacts_at_start']
            checks.value(snapshot[str((folder/'manifest.json').relative_to(ROOT))],sha(folder/'manifest.json'),'legacy audit binding/'+name)
        checks.check(m['sources']==m['sources_at_end'] and m['input_sha256']==m['input_sha256_at_end'],'parent immutable/'+name)
        for rel,digest in m['outputs_sha256'].items():checks.value(sha(folder/rel),digest,'parent output/'+name+'/'+rel)
        # The B cache exact Q1 bundle inputs must be the same parent output
        # files already checked above, not another target with a shared label.
        if name=='benefit_cost_execution_v1':
            for rel,digest in m['input_sha256'].items():checks.value(sha(ROOT/rel),digest,'B cache input/'+rel)


def independent_summary(rows):
    frame=pd.DataFrame(rows);out=[]
    for (repeat,policy),group in frame.groupby(['repeat_id','policy'],sort=True):
        a=group.loc[group.assigned];n=len(a)
        out.append(dict(repeat_id=int(repeat),policy=policy,scope='in_sample_full120' if repeat==0 else 'outer_oof',
            original_n=len(group),assigned_n=n,breach_count=int(a.breach.sum()) if n else None,
            conditional_breach_probability=float(a.breach.mean()) if n else None,
            mean_log_growth=float(a.log_growth.mean()) if n else None,median_cagr=float(a.cagr.median()) if n else None,
            p10_cagr=float(np.quantile(a.cagr,.1,method='inverted_cdf')) if n else None,
            median_mdd=float(a.mdd.median()) if n else None,p90_mdd=float(np.quantile(a.mdd,.9,method='inverted_cdf')) if n else None,
            mean_fees=float(a.nominal_fees.mean()) if n else None,mean_trade_fraction=float(a.trade_fraction.mean()) if n else None))
    return out


def common_B_pairs(rows):
    frame=pd.DataFrame(rows);pairs=[]
    for repeat in range(6):
        g=frame.loc[frame.repeat_id.eq(repeat)];model=g.loc[g.policy.eq(POLICY)];B=g.loc[g.policy.eq('annual_then_pending')]
        joined=model.merge(B,on=['context','scenario_id'],suffixes=('_model','_B'),validate='one_to_one')
        if len(joined)!=120:raise ValueError('Each repeat retains original120 paired identities')
        u=joined.loc[joined.assigned_model & joined.assigned_B];n=len(u)
        pairs.append(dict(repeat_id=repeat,scope='in_sample_full120' if repeat==0 else 'outer_oof',original_n=120,common_auto_n=n,
            model_breaches=int(u.breach_model.sum()) if n else None,B_breaches=int(u.breach_B.sum()) if n else None,
            **{'mean_delta_'+field:float((u[field+'_model']-u[field+'_B']).mean()) if n else None for field in ('log_growth','mdd','nominal_fees','trade_fraction')}))
    return pairs


def audit_artifacts(p,manifest):
    checks=Checks();parent_integrity(p,checks)
    checks.check(p['retained_learner']==OLD and p['final_policy']==POLICY and p['service_alpha']==.1,'fixed overlay contract')
    checks.check(manifest['new_fits']==0 and manifest['market_data_loaded'] is False and manifest['final_scenarios_loaded'] is False,'no new fitting/market/final')
    for field in ('sources','input_sha256'):checks.check(manifest[field]==manifest[field+'_at_end'],'producer start/end '+field)
    checks.value(manifest['protocol_sha256'],sha(PROTOCOL),'frozen protocol SHA')
    checks.check(manifest['sources']==p['execution_sources_sha256'] and manifest['input_sha256']==p['frozen_input_sha256'],'frozen closure exact')
    for mapping,base in ((p['frozen_input_sha256'],ROOT),(p['execution_sources_sha256'],ROOT),(manifest['outputs_sha256'],OUT)):
        for rel,digest in mapping.items():checks.value(sha(base/rel),digest,'current artifact/'+rel)
    contexts=readj(ROOT/'docs/refit_stability_protocol.json')['contexts']
    c0=readj(ROOT/'results/model_review/joint_liquidity_refit_v1/static_refits.json')['c0_refit']['context']
    contexts=contexts+[{**c0,'repeat_id':0,'outer_fold':0}]
    checks.check(len(contexts)==26 and len({c['name'] for c in contexts})==26,'all26 unique contexts')
    B_all=pd.read_csv(ROOT/'results/model_review/benefit_cost_execution_v1/inner_reference_paths.csv.gz')
    archives={}
    for kind,name in [('repeat','refit_stability_v1'),('full','joint_liquidity_refit_v1')]:
        folder=ROOT/'results/model_review'/name;ext='.csv.gz' if kind=='repeat' else '.csv'
        archives[kind]=dict(folder=folder,choices=readj(folder/'context_selections.json'),
            inner=pd.read_csv(folder/('inner_validation_paths'+ext)),clients=pd.read_csv(folder/('client_paths'+ext)),
            groups=readj(folder/'trace_groups.json')['actual'])
    actual_admissions=readj(OUT/'admission.json');admap={r['context']:r for r in actual_admissions}
    checks.check(len(actual_admissions)==26 and len(admap)==26,'complete JSON admission inventory')
    actual_csv=pd.read_csv(OUT/'admission.csv.gz');checks.check(len(actual_csv)==26,'complete CSV admission inventory')
    client=pd.read_csv(OUT/'client_paths.csv.gz');checks.check(len(client)==2160,'complete client inventory')
    checks.check(not client[['context','policy','scenario_id']].duplicated().any(),'unique original client identity')
    output_client={(r['context'],r['policy'],int(r['scenario_id'])):r for r in client.to_dict('records')}
    trade=pd.read_csv(OUT/'trade_estimates.csv.gz');checks.check(not trade[['context','scenario_id','year']].duplicated().any(),'unique original decision identity')
    output_trade={(r['context'],int(r['scenario_id']),int(r['year'])):r for r in trade.to_dict('records')}
    expected_clients=[];expected_costs=[];account_count=0;active_count=0;auto_fit=0
    for c in contexts:
        name=c['name'];archive=archives['full' if c['context_id']==0 else 'repeat'];old=archive['choices'][name]
        choice=old['methods'][OLD];oldauto=choice['execution_status']=='automated';beta=choice['selected_beta']
        checks.check(sorted(old['training_ids'])==sorted(c['training_ids']),'old training identity/'+name)
        evaluation=old['evaluation_ids'];checks.check(len(evaluation)==(120 if c['context_id']==0 else 24),'original evaluation n/'+name)
        if oldauto:
            rows=archive['inner'].loc[archive['inner'].context.eq(name)&archive['inner'].training_beta.eq(beta)]
            B=B_all.loc[B_all.context.eq(name)]
            gate=independent_guard(rows.to_dict('records'),B.to_dict('records'),c['training_ids'])
        else:gate=dict(original_inner_n=len(c['training_ids']),auto_inner_n=None,k_inner=None,k_B_same_U=None,absolute_cap=None,
                       auto_inner_ids=[],eligible=False,reasons=['original_policy_not_automated'])
        approved=bool(oldauto and gate['eligible']);auto_fit+=approved
        case=old['case']['public_status']
        status='automated' if approved else ('execution_review' if case=='balanced_case' else case)
        reason=None if approved else (gate['reasons'][0] if oldauto else choice['execution_reason'])
        expected=dict(context=name,context_id=c['context_id'],repeat_id=c['repeat_id'],outer_fold=c['outer_fold'],
            original_execution_status=choice['execution_status'],selected_beta=beta,selected_candidate_order=choice['selected_candidate_order'],
            final_automated=approved,execution_status=status,execution_reason=reason,**gate)
        checks.check(name in admap,'admission present/'+name);checks.record(admap[name],expected,'admission/'+name)
        csvrow=actual_csv.loc[actual_csv.context.eq(name)];checks.check(len(csvrow)==1,'one CSV fit/'+name)
        parsed=csvrow.iloc[0].to_dict();parsed['auto_inner_ids']=ast.literal_eval(parsed['auto_inner_ids']);parsed['reasons']=ast.literal_eval(parsed['reasons'])
        checks.record(parsed,expected,'admission CSV/'+name)
        source=archive['clients'].loc[archive['clients'].context.eq(name)&archive['clients'].policy.isin((OLD,*REFS))]
        checks.check(len(source)==3*len(evaluation),'complete original three-policy rows/'+name)
        for row in source.to_dict('records'):
            out={**row,'repeat_id':c['repeat_id'],'outer_fold':c['outer_fold'],'scope':'in_sample_full120' if c['context_id']==0 else 'outer_oof'}
            if row['policy']==OLD:
                out['policy']=POLICY
                if not approved:out.update(assigned=False,execution_status=status,execution_reason=reason,**dict.fromkeys(FIN))
            key=(name,out['policy'],int(row['scenario_id']));checks.check(key in output_client,'client identity/'+str(key))
            checks.record(output_client[key],out,'client/'+str(key));expected_clients.append(out)
        if not oldauto:continue
        found=[(key,g) for key,g in archive['groups'].items() if g['metadata'].get('context')==name and g['metadata'].get('policy')==OLD]
        checks.check(len(found)==1,'exact unchanged original selected actual group/'+name);key,group=found[0]
        checks.check(group['scenario_ids']==evaluation,'trace original evaluation identity/'+name)
        checks.value(group['metadata']['training_beta'],beta,'trace unchanged selected beta/'+name)
        order=integer(choice['selected_candidate_order'],'original selected order');checks.check(order<len(PRICES),'price order range')
        eta=PRICES[order];checks.value(group['selected_candidate_order'],order,'trace selected order/'+name)
        with np.load(archive['folder']/'actual_traces.npz',allow_pickle=False) as arrays:
            arm={field[len(key)+2:]:arrays[field] for field in arrays.files if field.startswith(key+'__')}
        checks.check(arm['real_wealth'].shape==(len(evaluation),31),'original H30 shape/'+name)
        checks.check(np.asarray(arm['annual_prices']).shape==(1,),'single original price/'+name)
        saved_eta=float(arm['annual_prices'][0]);checks.check((np.isposinf(saved_eta) if eta is None else saved_eta==eta),'price metadata/'+name)
        checks.check(np.allclose(arm['targets'],old['case']['product']['weights'],rtol=0,atol=2e-15),'original exact target/'+name)
        for i,sid in enumerate(group['scenario_ids']):
            account_count+=1
            for t in range(30):
                pending=observed_bool(arm['pending_at_decision'][i,t],'saved pending');active=t>0 and not pending
                restored=observed_bool(arm['target_execution'][i,t],'saved action')
                checks.check(str(arm['actions'][i,t])==('restore' if restored else 'hold'),'action/target consistency')
                estimate={}
                if active:
                    active_count+=1
                    n=integer(arm['forecast_n'][i,t],'original fan N',1);checks.check(n==len(c['training_ids']),'own full fan N/'+name)
                    kh=integer(arm['qbad_hold_count'][i,t],'saved hold bad');kr=integer(arm['qbad_restore_count'][i,t],'saved restore bad')
                    estimate=independent_score(float(arm['qg_hold'][i,t]),float(arm['qg_restore'][i,t]),kh,kr,n,eta,30)
                    checks.check(estimate['recommend_restore']==restored,'exact original threshold action')
                    checks.value(arm['predicted_growth_advantage_restore'][i,t],estimate['delta_net_total_log_growth'],'saved delta netQ')
                    checks.value(arm['predicted_breach_count_delta_restore'][i,t],kr-kh,'saved delta count')
                    checks.value(arm['restoration_predicted_growth_sacrifice'][i,t],estimate['strict_growth_sacrifice'],'saved sacrifice')
                    if eta is not None:
                        checks.value(arm['predicted_score'][i,t],estimate['risk_adjusted_net_advantage'],'saved finite net-Q score')
                        expected_reason='finite_score_positive' if restored else 'finite_score_nonpositive'
                    else:
                        checks.check(np.isnan(arm['predicted_score'][i,t]),'risk-first score absent')
                        expected_reason=('risk_first_lower_predicted_risk' if kr<kh else 'risk_first_higher_predicted_risk' if kr>kh
                            else 'risk_first_equal_risk_growth_positive' if restored else 'risk_first_equal_risk_hold_or_tie')
                    checks.value(arm['gate_reason'][i,t],expected_reason,'original gate reason')
                else:
                    checks.check(restored==(t==0),'mandatory inception / absorbing pending action')
                    for field in ('qg_hold','qg_restore','predicted_score','predicted_growth_advantage_restore'):
                        checks.check(np.isnan(arm[field][i,t]),'inactive Q sentinel/'+field)
                    for field in ('qbad_hold_count','qbad_restore_count','forecast_n','predicted_breach_count_delta_restore'):
                        checks.check(arm[field][i,t]==-1,'inactive count sentinel/'+field)
                    checks.check(not arm['restoration_predicted_growth_sacrifice'][i,t],'inactive sacrifice false')
                    checks.value(arm['gate_reason'][i,t],'mandatory_initial_restore' if t==0 else 'already_breached_pending_hold','inactive reason')
                row=dict(context=name,repeat_id=c['repeat_id'],scenario_id=int(sid),year=t+1,old_selected_policy=OLD,
                    final_guard_admitted=approved,audit_scope='original_selected_controller_development',initial=t==0,pending=pending,
                    active_optional=active,restore=restored,actual_trade=bool(arm['trades'][i,t]),actual_fee=float(arm['fees'][i,t]),
                    actual_fee_fraction=float(arm['fee_fraction'][i,t]),eta_technical=eta,eta_kind='finite' if eta is not None else 'risk_first',**estimate)
                identity=(name,int(sid),t+1);checks.check(identity in output_trade,'complete original trade identity')
                target=output_trade[identity];checks.record(target,row,'trade/'+str(identity))
                if not active:
                    for field in ('recommend_restore','delta_net_total_log_growth','delta_model_breach_probability',
                        'risk_adjusted_net_advantage','justification','strict_growth_sacrifice'):checks.check(null(target.get(field)),'absent initial/pending estimate')
                expected_costs.append(row)
    checks.check(account_count==408,'all408 original selected actual accounts')
    checks.check(len(expected_costs)==12240 and len(trade)==12240,'all original408x30 decision rows')
    summary=pd.read_csv(OUT/'summary.csv.gz');expected_summary=independent_summary(expected_clients)
    checks.check(len(summary)==18 and len(expected_summary)==18,'all6scopes x3policies summary')
    for row in expected_summary:
        actual=summary.loc[summary.repeat_id.eq(row['repeat_id'])&summary.policy.eq(row['policy'])]
        checks.check(len(actual)==1,'unique summary');checks.record(actual.iloc[0].to_dict(),row,'summary')
    pairs=common_B_pairs(expected_clients)
    costs=pd.DataFrame(expected_costs);optional=costs.loc[costs.active_optional & costs.restore]
    strict=optional.loc[optional.strict_growth_sacrifice.eq(True)]
    statistics=dict(original_selected_accounts=account_count,original_year_records=len(expected_costs),
        active_optional_decisions=active_count,optional_restores=len(optional),strict_growth_sacrifice_restores=len(strict),
        risk_reduction_cost_restores=int(optional.justification.eq('risk_reduction_with_growth_cost').sum()),
        strict_sacrifice_total_log_q10_q50_q90=np.quantile(strict.delta_net_total_log_growth,[.1,.5,.9]).tolist() if len(strict) else None,
        strict_sacrifice_mean_model_probability_delta=float(strict.delta_model_breach_probability.mean()) if len(strict) else None,
        strict_sacrifice_mean_actual_current_fee=float(strict.actual_fee.mean()) if len(strict) else None,
        note='Original selected development decisions, including newly rejected fits; not new admitted service returns, IID annual trials, or cumulative financial loss')
    return dict(status='passed',checks_passed=checks.n,maximum_numeric_absolute_error=checks.maximum_error,
        fits_checked=26,guard_admitted_fits=auto_fit,client_records_checked=len(client),trade_records_checked=len(trade),
        score_classification=statistics,common_same_U_B_pairs=pairs,
        physical_audit_scope='Unchanged parent physical and Q audits reused; complete admission/identity/NULL/score/action checks only',
        market_data_loaded=False,final_scenarios_loaded=False)


def main():
    p=readj(PROTOCOL);manifest=readj(OUT/'manifest.json')
    if p.get('execution_ready') is not True or manifest.get('status')!='complete':raise RuntimeError('Frozen protocol and complete producer required')
    before=current_snapshot(p,manifest)
    result=audit_artifacts(p,manifest)
    after=current_snapshot(p,manifest)
    if before!=after:raise RuntimeError('Read-only source/input/artifact changed during audit')
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),parent_manifest_sha256=sha(OUT/'manifest.json'),
        protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after)
    path=OUT/'independent_audit.json'
    if path.exists():raise RuntimeError('Never replace an existing independent audit receipt')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','checks':result['checks_passed'],'fits':26,'trade_records':result['trade_records_checked'],
        'audit_sha256':sha(path)},ensure_ascii=False))


if __name__=='__main__':main()
