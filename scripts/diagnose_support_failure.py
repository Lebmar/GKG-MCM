"""Saved-only support and purchasing-power diagnostic. No policy or raw loader."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from diagnostics.support_failure import (original_metrics,four_corners,initial_full_target,
    budget,directional_minimum,one_step_accounting)
PLAN=Path('docs/support_failure_diagnostic_protocol_v1.json')
FREEZE=Path('docs/support_failure_diagnostic_source_freeze_v1.json')
ACTIVATION=Path('docs/support_failure_diagnostic_activation_v1.json')
OUT=Path('results/support_failure_diagnostic_v1')
CANON=Path('results/state_budget_mitigation_v1')
STRESS=Path('results/state_budget_mitigation_stress_v1')
SOURCES=('research/diagnostics/support_failure.py','scripts/diagnose_support_failure.py','tests/test_support_failure.py')
SETTINGS=('base','fee_0000','fee_00125','fee_005','fee_01','pooled_support','actual_inflation_plus_02')
POLICIES=('state_budget_learner','same_target_B','same_target_annual','exact_thirds_annual')
COHORTS=('outer_OOF','full_resubstitution')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):
        if np.isnan(v):return None
        raise ValueError('Infinite diagnostic output')
    return v
def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows):
    if not rows:raise ValueError('Declared diagnostic table unexpectedly empty')
    fields=sorted({k for r in rows for k in r})
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:
        with io.TextIOWrapper(f,newline='') as out:
            w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
            for row in rows:w.writerow({k:json.dumps(clean(v),separators=(',',':')) if isinstance(v,(dict,list,np.ndarray)) else clean(v) for k,v in row.items()})


class Checks:
    def __init__(self):self.count=0;self.max_abs=0.
    def close(self,a,b,atol=3e-10,rtol=5e-13):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError('Saved diagnostic identity mismatch')
        self.count+=a.size;self.max_abs=max(self.max_abs,float(np.max(np.abs(a-b))) if a.size else 0.)


def saved_inputs(root=ROOT):
    """Hash saved artifacts only. Historical raw-file hashes stay opaque."""
    root=Path(root);mapping={}
    for folder,audit_name in [(CANON,CANON/'independent_audit.json'),(STRESS,Path('results/state_budget_mitigation_stress_audit_v1/manifest.json'))]:
        m=readj(root/folder/'manifest.json');audit=readj(root/audit_name)
        if m.get('status')!='complete' or m.get('final_scenarios_loaded') is not False or audit.get('status')!='passed' or audit.get('final_scenarios_loaded') is not False:
            raise RuntimeError('Passed unchanged development-only saved parents required')
        if m['sources_and_inputs_at_start']!=m['sources_and_inputs_at_end']:raise RuntimeError('Parent start/end mismatch')
        parent=str(folder/'manifest.json');mh=sha(root/parent)
        links=[audit.get('parent_manifest_sha256'),audit.get('producer_manifest_sha256')]
        links += [audit.get(k,{}).get(parent) for k in ('sources_and_inputs_at_start','sources_and_artifacts_at_start')]
        if mh not in links:raise RuntimeError('Independent audit must bind this exact parent')
        for key in ('sources_and_inputs','sources_and_artifacts'):
            if key+'_at_start' in audit and audit[key+'_at_start']!=audit[key+'_at_end']:raise RuntimeError('Audit start/end mismatch')
        mapping[parent]=mh;mapping[str(audit_name)]=sha(root/audit_name)
        for name,digest in m['outputs_sha256'].items():
            part=Path(name)
            if part.is_absolute() or '..' in part.parts or part.as_posix()!=name:raise RuntimeError('Canonical saved output path required')
            path=folder/part
            if sha(root/path)!=digest:raise RuntimeError('Saved parent artifact changed')
            mapping[str(path)]=digest
    for name in ['docs/state_budget_mitigation_stress_protocol.json','docs/state_budget_mitigation_analysis_protocol.json']:
        mapping[name]=sha(root/name)
    return mapping


def preflight(root=ROOT):
    root=Path(root);p,f,a=readj(root/PLAN),readj(root/FREEZE),readj(root/ACTIVATION)
    if (p.get('version')!='saved-support-and-inflation-failure-diagnostic-v1' or p.get('execution_ready') is not False
            or p.get('allow_final') is not False or p.get('development_ids')!=list(range(1,121))
            or p.get('settings')!=list(SETTINGS) or p.get('cohorts')!=list(COHORTS) or p.get('policies')!=list(POLICIES)
            or f.get('execution_ready') is not True or f.get('protocol_sha256')!=sha(root/PLAN)
            or a.get('execution_ready') is not True or a.get('protocol_sha256')!=sha(root/PLAN)
            or a.get('source_freeze_sha256')!=sha(root/FREEZE) or not set(SOURCES)<=set(f.get('source_sha256',{}))):
        raise RuntimeError('Prospective source freeze and separate activation required before saved numerical parse')
    snapshot={**f['source_sha256'],**f['input_sha256']}
    if any('data/'==str(k)[:5] or 'final' in str(k) for k in snapshot):raise RuntimeError('No raw/final input in this diagnostic')
    for name,digest in snapshot.items():
        q=Path(name)
        if q.is_absolute() or '..' in q.parts or not (root/q).resolve().is_relative_to(root.resolve()) or sha(root/q)!=digest:
            raise RuntimeError('New diagnostic source/input changed')
    required=saved_inputs(root)
    if any(snapshot.get(k)!=v for k,v in required.items()):raise RuntimeError('Freeze complete consumed saved-artifact closure')
    return p,snapshot


def arm(z,g,keys):
    if not g['financial_available']:return None
    if not set(keys)<=set(g['array_names']):raise ValueError('Required saved physical fields absent')
    return {k:z[g['group_id']+'__'+k] for k in keys}


def reconstruct(cz,cgroups,check):
    gross=np.full((120,30,3),np.nan);cpig=np.full((120,30),np.nan);seen=[]
    refs=[g for g in cgroups if g['kind']=='reference' and g['policy']=='exact_thirds_annual' and g['job_id']!='c0_refit']
    if len(refs)!=5:raise ValueError('All five original OOF thirds groups required')
    for g in refs:
        ids=np.asarray(g['scenario_ids'],int);seen.extend(ids.tolist())
        post=cz[g['group_id']+'__posttrade_holdings'];nextx=cz[g['group_id']+'__holdings'][:,1:];cpi=cz[g['group_id']+'__cpi']
        if post.shape!=(24,30,3) or np.any(post<=0) or np.any(cpi<=0):raise ValueError('Strictly positive saved reference inversion required')
        gross[ids-1]=nextx/post;cpig[ids-1]=cpi[:,1:]/cpi[:,:-1]
    if sorted(seen)!=list(range(1,121)) or not np.isfinite(gross).all() or not np.isfinite(cpig).all() or np.any(gross<=0) or np.any(cpig<=0):
        raise ValueError('All original120 positive saved markets required')
    selected={g['job_id']:g for g in cgroups if g['kind'] in ('outer_selected','full_selected')}
    if set(selected)!={'c0_refit','c1_refit','c2_refit','c3_refit','c4_refit','c5_refit'}:raise ValueError('All six contexts required')
    training={}
    for job,g in selected.items():
        ids=np.asarray(g['scenario_ids']);tr=np.asarray(g['training_ids']);prefix=g['group_id']+'__'
        if job!='c0_refit' and (len(tr)!=96 or len(ids)!=24 or set(tr)&set(ids)):raise ValueError('Own outer96/24 split required')
        if job=='c0_refit' and (tr.tolist()!=list(range(1,121)) or ids.tolist()!=list(range(1,121))):raise ValueError('Full resubstitution identity required')
        train=np.transpose(cz[prefix+'training_real_gross'],(1,0,2));training[job]=train
        check.close(train,gross[tr-1]/cpig[tr-1,:,None],atol=3e-11)
        check.close(cz[prefix+'realized_real_gross'],gross[ids-1]/cpig[ids-1,:,None],atol=3e-11)
    return gross,cpig,selected,training


def initial_tables(groups,witnesses,training,check):
    summaries=[];years=[];members=[]
    for g in groups:
        if g['policy']!='state_budget_learner':continue
        s=g['setting'];train=training[g['job_id']];n,h,_=train.shape
        w=np.asarray(g['own_fit_target']);rho=g['selected_candidate']['release_fraction'];y,fee=initial_full_target(w,s['cost_rate'])
        floor,b=budget(100.,100.,rho);values=np.einsum('nha,a->nh',train,y);pool=float(values.min());cal=float(values[:,0].min())
        if pool>cal+1e-10:raise AssertionError('Pooled inclusion must tighten same-target support')
        declared=pool if s['support_mode']=='pooled' else cal;passed=declared-b>=-1e-10
        witness=next(x['witness'] for x in witnesses if x['setting_id']==g['setting_id'] and x['context_id']==g['context_id'])
        check.close(witness['budget_floor'],b);check.close(witness['candidate_posttrade'][1],y)
        check.close(witness['candidate_min_support_slack'][1],declared-b)
        if passed!=g['financial_available']:raise AssertionError('Fixed initial support admission mismatch')
        summary=dict(setting_id=g['setting_id'],cohort=g['cohort'],context_id=g['context_id'],job_id=g['job_id'],training_n=n,
            evaluated_original_n=len(g['scenario_ids']),target=w.tolist(),rho=rho,cost_rate=s['cost_rate'],support_mode=s['support_mode'],
            status=g['status'],financial_available=g['financial_available'],full_target_fee=fee,original_floor=float(floor),budget_floor=float(b),
            calendar_year1_min=cal,pooled_min=pool,calendar_year1_slack=cal-float(b),pooled_slack=pool-float(b),
            pooled_violating_training_rows=int((values<float(b)-1e-10).sum()),calendar_violating_training_rows=int((values[:,0]<float(b)-1e-10).sum()))
        summaries.append(summary)
        if g['setting_id']!='base':continue
        for t in range(h):
            minimum=float(values[:,t].min());binding=np.flatnonzero(np.abs(values[:,t]-minimum)<=1e-10)
            years.append(dict(job_id=g['job_id'],cohort=g['cohort'],training_year=t+1,training_n=n,min_full_target_next_real=minimum,
                budget_floor=float(b),min_slack=minimum-float(b),violating_rows=int((values[:,t]<float(b)-1e-10).sum()),
                minimum_member_original_ids=[g['training_ids'][i] for i in binding],negative_real_bill_rows=int((train[:,t,2]<1).sum()),min_real_bill_gross=float(train[:,t,2].min())))
            for i,sid in enumerate(g['training_ids']):members.append(dict(job_id=g['job_id'],cohort=g['cohort'],training_id=sid,training_year=t+1,
                equity_real_gross=train[i,t,0],bond_real_gross=train[i,t,1],bill_real_gross=train[i,t,2],
                full_target_next_real=values[i,t],slack=values[i,t]-float(b),budget_floor=float(b),negative_real_bill=bool(train[i,t,2]<1)))
    return summaries,years,members


def support_periods(z,groups,training,gross,cpig,check):
    records=[];summaries=[]
    keys=['holdings','posttrade_holdings','fees','cpi','real_wealth','real_peak','running_mdd','realized_real_gross','gate_evaluated',
        'automatic_service','guard_triggered','was_pending_input','support_feasible','budget_floor','worst_real_shortfall_selected',
        'first_guard_trigger','first_crossing','actual_decision_source']
    for g in groups:
        if g['policy']!='state_budget_learner' or not g['financial_available']:continue
        a=arm(z,g,keys);train=training[g['job_id']];pool=train.reshape(-1,3);ids=np.asarray(g['scenario_ids']);n,h=a['fees'].shape
        shift=g['setting']['actual_inflation_shift'];pi_g=cpig[ids-1]+shift
        check.close(a['cpi'][:,1:]/a['cpi'][:,:-1],pi_g,atol=3e-11)
        check.close(a['realized_real_gross'],gross[ids-1]/pi_g[:,:,None],atol=3e-11)
        check.close(a['holdings'][:,1:],a['posttrade_holdings']*gross[ids-1])
        for t in range(h):
            post=a['posttrade_holdings'][:,t];c=a['cpi'][:,t];real=a['real_wealth'][:,t]
            mc,ic=directional_minimum(train[:,t],post,c);mp,ip=directional_minimum(pool,post,c)
            declared=mp if g['setting']['support_mode']=='pooled' else mc
            floor,b=budget(real,a['real_peak'][:,t],g['selected_candidate']['release_fraction'])
            acc=one_step_accounting(a['holdings'][:,t],post,a['fees'][:,t],c,a['cpi'][:,t+1],gross[ids-1,t]);check.close(acc['next_real'],a['real_wealth'][:,t+1])
            active=a['gate_evaluated'][:,t];check.close(a['budget_floor'][active,t],b[active])
            check.close(a['worst_real_shortfall_selected'][active,t],np.maximum(b-declared,0)[active])
            for i,sid in enumerate(ids):
                row=dict(setting_id=g['setting_id'],cohort=g['cohort'],job_id=g['job_id'],scenario_id=int(sid),year=t+1,
                    gate_evaluated=bool(active[i]),ordinary_service=bool(a['automatic_service'][i,t]),current_guard=bool(a['guard_triggered'][i,t]),
                    prior_pending=bool(a['was_pending_input'][i,t]),support_feasible=bool(a['support_feasible'][i,t]),decision_source=str(a['actual_decision_source'][i,t]),
                    current_real=real[i],original_peak=a['real_peak'][i,t],offline_original_floor=floor[i],offline_budget_floor=b[i],
                    calendar_directional_min=mc[i],pooled_directional_min=mp[i],declared_directional_min=declared[i],actual_next_real=a['real_wealth'][i,t+1],
                    calendar_worst_training_id=g['training_ids'][ic[i]],pooled_worst_training_id=g['training_ids'][ip[i]//h],pooled_worst_training_year=int(ip[i]%h+1),
                    directional_miss=bool(a['real_wealth'][i,t+1]<declared[i]-1e-12*real[i]),
                    budget_failure=bool(a['real_wealth'][i,t+1]<b[i]-1e-12*real[i]),
                    actual_bill_real_gross=acc['real_gross'][i,2],actual_bill_negative_real=bool(acc['real_gross'][i,2]<1),
                    fee_nominal=a['fees'][i,t],fee_current_real=acc['fee_current_real'][i],equity_real_PnL=acc['asset_real_contribution'][i,0],
                    bond_real_PnL=acc['asset_real_contribution'][i,1],bill_real_PnL=acc['asset_real_contribution'][i,2],
                    nominal_asset_gain_at_next_CPI=acc['nominal_asset_gain'][i],nominal_fee_at_next_CPI=acc['nominal_fee_at_next_CPI'][i],
                    inflation_erosion=acc['inflation_erosion'][i],real_wealth_change=acc['change'][i],first_guard_year=int(a['first_guard_trigger'][i]),first_crossing=int(a['first_crossing'][i]))
                records.append(row)
    for setting in SETTINGS:
        for cohort in COHORTS:
            gs=[g for g in groups if g['setting_id']==setting and g['cohort']==cohort and g['policy']=='state_budget_learner']
            rr=[r for r in records if r['setting_id']==setting and r['cohort']==cohort];financial=sum(len(g['scenario_ids']) for g in gs if g['financial_available'])
            original=sum(len(g['scenario_ids']) for g in gs)
            summary=dict(setting_id=setting,cohort=cohort,original_n=original,financial_n=financial,account_periods=len(rr),initial_review_n=original-financial,phases=[])
            for phase,predicate in [('ordinary',lambda r:r['ordinary_service']),('guard_mitigation',lambda r:r['current_guard']),('prior_pending',lambda r:r['prior_pending'])]:
                part=[r for r in rr if predicate(r)];summary['phases'].append(dict(phase=phase,periods=len(part),distinct_original_ids=len({r['scenario_id'] for r in part}),
                    directional_miss_periods=sum(r['directional_miss'] for r in part),budget_failure_periods=sum(r['budget_failure'] for r in part),
                    negative_real_bill_periods=sum(r['actual_bill_negative_real'] for r in part),
                    supported_safe_actual_budget_failures=sum(r['gate_evaluated'] and r['support_feasible'] and r['budget_failure'] for r in part)))
            if sum(p['periods'] for p in summary['phases'])!=len(rr):raise AssertionError('Complete disjoint service phases')
            summaries.append(summary)
    return records,summaries


def inflation_tables(z,groups,check):
    paths=[];endpoints=[];summary=[]
    for cohort in COHORTS:
        for policy in POLICIES:
            current=[]
            bg=[g for g in groups if g['setting_id']=='base' and g['cohort']==cohort and g['policy']==policy]
            for base in bg:
                shift=next(g for g in groups if g['setting_id']=='actual_inflation_plus_02' and g['job_id']==base['job_id'] and g['policy']==policy)
                if base['scenario_ids']!=shift['scenario_ids']:raise ValueError('Complete originalID inflation pairing')
                ids=base['scenario_ids'];meta=dict(cohort=cohort,policy=policy,job_id=base['job_id'])
                if not base['financial_available'] or not shift['financial_available']:
                    for sid in ids:paths.append(dict(**meta,scenario_id=sid,base_financial=base['financial_available'],shift_financial=shift['financial_available'],BB_mdd=None,BS_mdd=None,SB_mdd=None,SS_mdd=None))
                    continue
                keys=['nominal_wealth','cpi','real_wealth','holdings','posttrade_holdings','fees','running_mdd','first_crossing']
                b,s=arm(z,base,keys),arm(z,shift,keys);f=four_corners(b['nominal_wealth'],b['cpi'],s['nominal_wealth'],s['cpi'])
                check.close(f['corners']['BB']['real'],b['real_wealth']);check.close(f['corners']['SS']['real'],s['real_wealth'])
                check.close(f['corners']['BB']['running_mdd'],b['running_mdd']);check.close(f['corners']['SS']['running_mdd'],s['running_mdd'])
                check.close(f['corners']['BB']['first_crossing'],b['first_crossing'],atol=0,rtol=0);check.close(f['corners']['SS']['first_crossing'],s['first_crossing'],atol=0,rtol=0)
                if policy in ('same_target_annual','exact_thirds_annual'):
                    for key in ('nominal_wealth','holdings','posttrade_holdings','fees'):
                        if not np.array_equal(b[key],s[key]):raise AssertionError('Inflation-only annual nominal negative control failed')
                for i,sid in enumerate(ids):
                    row=dict(**meta,scenario_id=sid,base_financial=True,shift_financial=True,terminal_log_nominal_component=f['log_nominal_path_component'][i,-1],terminal_log_CPI_component=f['log_CPI_component'][i,-1])
                    for corner,data in f['corners'].items():
                        for key in ('mdd','first_crossing','breach','log_growth','cagr'):row[corner+'_'+key]=data[key][i]
                    for key in ('MDD_CPI_first','MDD_nominal_after_CPI','MDD_nominal_first','MDD_CPI_after_nominal','MDD_interaction','MDD_total'):row[key]=f[key][i]
                    paths.append(row);current.append(row)
                    for t in range(31):
                        endpoint=dict(**meta,scenario_id=sid,endpoint=t,log_nominal_path_component=f['log_nominal_path_component'][i,t],log_CPI_component=f['log_CPI_component'][i,t])
                        for corner,data in f['corners'].items():
                            for key in ('real','peak','running_mdd'):endpoint[corner+'_'+key]=data[key][i,t]
                        endpoints.append(endpoint)
            if len({g['job_id'] for g in bg})!=(5 if cohort=='outer_OOF' else 1) or sum(len(g['scenario_ids']) for g in bg)!=120:raise ValueError('Separate complete original120 cohort')
            stats=dict(cohort=cohort,policy=policy,original_n=120,paired_financial_n=len(current),corners={},ordered_MDD_means={})
            for corner in ('BB','BS','SB','SS'):
                stats['corners'][corner]=dict(breach_count=sum(r[corner+'_breach'] for r in current) if current else None,mean_mdd=float(np.mean([r[corner+'_mdd'] for r in current])) if current else None,median_cagr=float(np.median([r[corner+'_cagr'] for r in current])) if current else None)
            for key in ('MDD_CPI_first','MDD_nominal_after_CPI','MDD_nominal_first','MDD_CPI_after_nominal','MDD_interaction','MDD_total'):
                stats['ordered_MDD_means'][key]=float(np.mean([r[key] for r in current])) if current else None
            stats['event_quadrants']=[dict(**{k+'_breach':bool(v) for k,v in zip(('BB','BS','SB','SS'),bits)},original_ids=[r['scenario_id'] for r in current if tuple(bool(r[k+'_breach']) for k in ('BB','BS','SB','SS'))==bits]) for bits in __import__('itertools').product((False,True),repeat=4)]
            summary.append(stats)
    return paths,endpoints,summary


def run(root=ROOT):
    root=Path(root)
    if (root/OUT).exists():raise RuntimeError('Never overwrite a diagnostic attempt')
    plan,snapshot=preflight(root);output=root/OUT;output.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_saved_numerical_parse',started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/PLAN),
        source_freeze_sha256=sha(root/FREEZE),activation_sha256=sha(root/ACTIVATION),sources_and_inputs_at_start=snapshot,
        raw_csv_opened=False,raw_market_values_parsed=False,final_scenarios_loaded=False,new_policy_runs=0,new_fits=0,new_selections=0)
    writej(output/'manifest.json',state)
    try:
        cg=readj(root/CANON/'trace_groups.json');sg=readj(root/STRESS/'trace_groups.json');witness=readj(root/STRESS/'initial_witnesses.json')
        stressplan=readj(root/'docs/state_budget_mitigation_stress_protocol.json')
        if [s['setting_id'] for s in stressplan['settings']]!=list(SETTINGS) or len(sg)!=168 or len(witness)!=42:raise ValueError('All7 original stress settings/contexts required')
        inventory=pd.read_csv(root/STRESS/'path_metrics.csv.gz',float_precision='round_trip')
        if len(inventory)!=6720 or set(inventory.scenario_id)!=set(range(1,121)):raise ValueError('Complete120 IDs/settings/policies/cohorts required')
        for setting in SETTINGS:
            for cohort in COHORTS:
                for policy in POLICIES:
                    subset=inventory[(inventory.setting_id==setting)&(inventory.cohort==cohort)&(inventory.policy==policy)]
                    if sorted(subset.scenario_id)!=list(range(1,121)):raise ValueError('All original IDs retained once per group')
        check=Checks()
        with np.load(root/CANON/'traces.npz',allow_pickle=False) as cz,np.load(root/STRESS/'traces.npz',allow_pickle=False) as sz:
            gross,cpig,selected,training=reconstruct(cz,cg,check)
            for g in sg:
                if g['policy']=='state_budget_learner':
                    parent=selected[g['job_id']]
                    if g['training_ids']!=parent['training_ids'] or g['scenario_ids']!=parent['scenario_ids'] or g['own_fit_target']!=parent['target'] or g['selected_candidate']!=parent['candidate']:
                        raise ValueError('No new tuple/target/ID selection')
            init,years,members=initial_tables(sg,witness,training,check)
            periods,support_summary=support_periods(sz,sg,training,gross,cpig,check)
            corners,endpoints,corner_summary=inflation_tables(sz,sg,check)
        inventory.to_csv(output/'path_inventory.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        for name,rows in [('initial_support_contexts',init),('initial_support_by_year',years),('initial_support_members',members),('directional_support_periods',periods),('inflation_four_corner_paths',corners),('inflation_four_corner_endpoints',endpoints)]:write_csv(output/(name+'.csv.gz'),rows)
        writej(output/'directional_support_summary.json',support_summary);writej(output/'inflation_four_corner_summary.json',corner_summary)
        writej(output/'checks.json',dict(status='passed',scalar_checks=check.count,maximum_absolute_difference=check.max_abs,
            context_setting_rows=len(init),initial_year_rows=len(years),initial_training_member_rows=len(members),support_period_rows=len(periods),
            paired_path_rows=len(corners),paired_endpoint_rows=len(endpoints),source_used='Saved arrays only; no raw CSV or policy calls'))
        after={name:sha(root/name) for name in snapshot}
        if after!=snapshot or sha(root/PLAN)!=state['protocol_sha256'] or sha(root/FREEZE)!=state['source_freeze_sha256'] or sha(root/ACTIVATION)!=state['activation_sha256']:
            raise RuntimeError('Diagnostic dependencies changed during analysis')
        state.update(status='complete',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),sources_and_inputs_at_end=after,
            outputs_sha256={p.name:sha(p) for p in output.iterdir() if p.name!='manifest.json'},new_physical_policy_runs=0)
        writej(output/'manifest.json',state)
    except BaseException as exc:
        (output/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(output/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print('Source-only. Separate frozen activation required; no numeric parse performed.')
