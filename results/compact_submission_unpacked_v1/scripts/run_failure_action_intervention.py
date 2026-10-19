"""Prespecified one-action interventions on eleven archived mechanism cases."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research')]
from experimental_execution.failure_action_intervention import replay_intervention

PROTOCOL=ROOT/'docs/failure_action_intervention_protocol.json'
OUT=ROOT/'results/model_review/failure_action_intervention_v1'
PARENT=ROOT/'results/model_review/failure_path_mechanisms_v1'
STATE_FIELDS=('holdings','weights','nominal_wealth','real_wealth','cpi','real_peak',
              'current_drawdown','running_mdd','ever_breached')
PERIOD_FIELDS=('original_year','actions','policy_actions','forced','pending_at_decision',
               'posttrade_holdings','fees','moved','qg_hold','qg_restore','qbad_hold_count',
               'qbad_restore_count','forecast_n','predicted_score','gate_reason')
PRICES=(0.,.001,.003,.01,.03,.1,.3,1.,None)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def snapshots(p,digest):
    if sha(PROTOCOL)!=digest:raise RuntimeError('Frozen intervention protocol changed')
    if not p.get('execution_sources_sha256') or not p.get('frozen_input_sha256'):
        raise RuntimeError('Nonempty frozen source/input closure is required')
    sources={f:sha(ROOT/f) for f in p['execution_sources_sha256']}
    inputs={f:sha(ROOT/f) for f in p['frozen_input_sha256']}
    if sources!=p['execution_sources_sha256'] or inputs!=p['frozen_input_sha256']:
        raise RuntimeError('Frozen intervention source/input changed')
    inputs[str(PROTOCOL.relative_to(ROOT))]=digest
    return sources,inputs


def validate(p):
    if p.get('execution_ready') is not True:raise RuntimeError('Wait for intervention source freeze')
    if p['data']!={'path':'data/SafeInvest_Synthetic_Data.csv','scenario_ids':list(range(1,121)),'allow_final':False}:
        raise ValueError('Only original development market paths authorized')
    if (p['original_H'],p['loss_tolerance'],p['cost_rate'])!=(30,.4,.0025):raise ValueError('Original account contract differs')
    if p['output_directory']!='results/model_review/failure_action_intervention_v1':raise ValueError('Fixed output directory required')
    design=readj(ROOT/'docs/failure_action_intervention_design.json')
    inventory=p['inventory']
    canonical=json.dumps(inventory,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    if (len(inventory)!=11 or [r['pair_id'] for r in inventory]!=design['pair_ids']
        or hashlib.sha256(canonical).hexdigest()!=design['parent_inventory_canonical_sha256']):
        raise ValueError('Prespecified eleven-pair intervention inventory differs')
    manifest,audit=readj(PARENT/'manifest.json'),readj(PARENT/'independent_audit.json')
    if manifest['status']!='complete' or audit['status']!='passed':raise RuntimeError('Audited first-stage paths required')
    binding=audit.get('parent_manifest_sha256',audit.get('producer_manifest_sha256'))
    if binding!=sha(PARENT/'manifest.json'):raise RuntimeError('Parent independent audit binding differs')
    return inventory


def extract(z,group,ids,sid):
    i=ids.index(sid);prefix=group+'__';metadata={'targets','loss_tolerances','annual_prices'}
    return {key[len(prefix):]:(z[key].copy() if key[len(prefix):] in metadata else z[key][i].copy())
            for key in z.files if key.startswith(prefix)}


def close(a,b,name,errors,atol=2e-10,rtol=3e-13):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:raise AssertionError('Shape differs: '+name)
    if a.dtype.kind in 'bUSO' or b.dtype.kind in 'bUSO':
        if not np.array_equal(a,b):raise AssertionError('Exact label/flag differs: '+name)
        errors[name]=max(errors.get(name,0.),0.);return
    if not np.array_equal(np.isnan(a),np.isnan(b)):raise AssertionError('NULL mask differs: '+name)
    finite=np.isfinite(a)&np.isfinite(b)
    error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    errors[name]=max(errors.get(name,0.),error)
    if not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True):raise AssertionError('Numeric identity differs: '+name)


def full_branch(original,suffix,t):
    result={}
    for key in STATE_FIELDS:
        result[key]=np.concatenate([original[key][:t],suffix[key]],axis=0)
    for key in PERIOD_FIELDS:
        prefix=(np.arange(1,31) if key=='original_year' else original['actions'] if key=='policy_actions'
                else np.zeros(30,bool) if key=='forced' else original[key])
        result[key]=np.concatenate([prefix[:t],suffix[key]],axis=0)
    result['fee_fraction']=result['fees']/result['nominal_wealth'][:-1]
    result['trades']=result['moved']>1e-12
    return result


def branch_metrics(arm):
    bad=arm['ever_breached'];crossing=np.flatnonzero(bad)
    return dict(terminal_real=float(arm['real_wealth'][-1]),mdd=float(arm['running_mdd'][-1]),
        breach=bool(bad[-1]),first_crossing=int(crossing[0]) if len(crossing) else None,
        total_log_growth=float(np.log(arm['real_wealth'][-1]/100)),
        nominal_fees=float(arm['fees'].sum()),nominal_moved=float(arm['moved'].sum()),
        trade_fraction=float(np.mean(arm['trades'])),restore_fraction=float(np.mean(arm['actions']=='restore')))


def witness_rows(item,branch,arm,tau):
    rows=[];seen={}
    for role,year in [('tau',tau),('tau_plus_one',None if tau is None else tau+1),('original_H',30)]:
        slot=f"{item['pair_id']}:{branch}:{role}";alias=None
        if tau is None:status='no_action_difference_skip'
        elif year>30:status='outside_original_horizon_skip'
        elif year in seen:status='duplicate_year_alias';alias=seen[year]
        elif bool(arm['pending_at_decision'][year-1]):status='already_breached_pending_skip';seen[year]=slot
        else:status='active_Q';seen[year]=slot
        row=dict(slot_id=slot,pair_id=item['pair_id'],branch=branch,role=role,year=year,status=status,alias_of=alias,
            scenario_id=item['scenario_id'],training_fan_n=item['training_fan_n'],
            forced_at_decision=None,qg_hold=None,qg_restore=None,qbad_hold_count=None,qbad_restore_count=None,
            forecast_n=None,predicted_score=None,policy_action=None,executed_action=None)
        if status=='active_Q':
            t=year-1
            row.update(forced_at_decision=bool(arm['forced'][t]),qg_hold=float(arm['qg_hold'][t]),
                qg_restore=float(arm['qg_restore'][t]),qbad_hold_count=int(arm['qbad_hold_count'][t]),
                qbad_restore_count=int(arm['qbad_restore_count'][t]),forecast_n=int(arm['forecast_n'][t]),
                predicted_score=float(arm['predicted_score'][t]),policy_action=str(arm['policy_actions'][t]),
                executed_action=str(arm['actions'][t]))
        rows.append(row)
    return rows


def main():
    p=readj(PROTOCOL);items=validate(p)
    if OUT.exists():raise RuntimeError('Never overwrite an intervention archive')
    digest=sha(PROTOCOL);sources,inputs=snapshots(p,digest);started=time.monotonic()
    from safeinvest.data import load_paths
    r,pi,ids=load_paths(ROOT/p['data']['path'],p['data']['scenario_ids'],allow_final=False)
    market_digest=hashlib.sha256()
    for array in (ids,r,pi):market_digest.update(np.ascontiguousarray(array).tobytes())
    parent_manifest=readj(PARENT/'manifest.json')
    if market_digest.hexdigest()!=parent_manifest['development_array_sha256']:raise RuntimeError('Original guarded market arrays differ')
    annual=pd.read_csv(PARENT/'annual_states.csv.gz',float_precision='round_trip')
    endpoints=pd.read_csv(PARENT/'account_endpoints.csv.gz',float_precision='round_trip')
    groups={folder:readj(ROOT/'results/model_review'/folder/'trace_groups.json') for folder in {x['folder'] for x in items}}
    archives={(f,s):np.load(ROOT/'results/model_review'/f/(s+'_traces.npz'),allow_pickle=False)
              for f,s in {(x['folder'],x['trace_scope']) for x in items}}
    summaries=[];branches=[];qrows=[];trees={};trace_groups={};checks=[]
    OUT.mkdir();writej(OUT/'manifest.json',dict(status='incomplete',protocol_sha256=digest))
    try:
        for item in items:
            if time.monotonic()-started>p['budget_minutes']*60:raise TimeoutError('Frozen intervention budget exceeded')
            pid=item['pair_id'];errors={}
            ma=annual.loc[annual.pair_id.eq(pid)&annual.arm.eq('model')].sort_values('year')
            ba=annual.loc[annual.pair_id.eq(pid)&annual.arm.eq('B')].sort_values('year')
            if ma.year.tolist()!=list(range(1,31)) or ba.year.tolist()!=list(range(1,31)):raise ValueError('All original paired action years required')
            different=np.flatnonzero(ma.action.to_numpy()!=ba.action.to_numpy());tau=int(different[0]+1) if len(different) else None
            summary=dict(pair_id=pid,stage=item['stage'],scenario_id=item['scenario_id'],tau=tau,
                status='no_action_difference' if tau is None else 'completed_intervention',original_action=None,forced_action=None,
                control_terminal_real=None,treatment_terminal_real=None,control_mdd=None,treatment_mdd=None,
                delta_mdd=None,delta_log_growth=None,control_breach=None,treatment_breach=None,delta_breach=None,
                control_first_crossing=None,treatment_first_crossing=None,delta_nominal_fees=None,
                zero_transfer_intervention=None,hypothetical_diagnostic=True)
            if tau is None:
                for branch in ('control','treatment'):
                    qrows.extend(witness_rows(item,branch,None,None))
                    branches.append(dict(pair_id=pid,branch=branch,status='no_action_difference',tau=None))
                summaries.append(summary);checks.append(dict(pair_id=pid,status='no_action_difference',maximum_errors={}));continue
            t=tau-1
            if not 2<=tau<=30:raise ValueError('First differing action must follow mandatory inception')
            me=endpoints.loc[endpoints.pair_id.eq(pid)&endpoints.arm.eq('model')].sort_values('endpoint')
            be=endpoints.loc[endpoints.pair_id.eq(pid)&endpoints.arm.eq('B')].sort_values('endpoint')
            if me.endpoint.tolist()!=list(range(31)) or be.endpoint.tolist()!=list(range(31)):raise ValueError('Complete source history required')
            for column in ('nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd','cpi','ever_breached',
                           'holding_equity','holding_bond','holding_bill','weight_equity','weight_bond','weight_bill'):
                close(me[column].to_numpy()[:t+1],be[column].to_numpy()[:t+1],'pre_tau/'+column,errors)
            if bool(me.ever_breached.iloc[t]) or bool(be.ever_breached.iloc[t]):raise ValueError('Both arms must be active before first disagreement')
            group=groups[item['folder']][item['trace_scope']][item['model_group']]
            old=extract(archives[item['folder'],item['trace_scope']],item['model_group'],group['scenario_ids'],item['scenario_id'])
            close(old['targets'],item['target'],'original_target',errors,atol=0,rtol=0)
            close(old['actions'],ma.action.to_numpy(),'original_actions',errors)
            old_action=str(ma.action.iloc[t]);forced_action=str(ba.action.iloc[t]);summary.update(original_action=old_action,forced_action=forced_action)
            if old_action==forced_action:raise AssertionError('First disagreement metadata lost')
            fan=np.asarray(item['training_ids'])-1;arms={};metrics={}
            for branch,first in [('control',old_action),('treatment',forced_action)]:
                suffix=replay_intervention(r[item['scenario_id']-1],pi[item['scenario_id']-1],r[fan],pi[fan],
                    item['target'],PRICES[item['selected_candidate_order']],holdings=old['holdings'][t],cpi=old['cpi'][t],
                    real_peak=old['real_peak'][t],running_mdd=old['running_mdd'][t],year_index=t,
                    forced_first_action=first,horizon=30,loss=.4,cost_rate=.0025)
                if str(suffix['policy_actions'][0])!=old_action:raise AssertionError('Original policy at forced state does not match source action')
                arm=full_branch(old,suffix,t);arms[branch]=arm;metrics[branch]=branch_metrics(arm)
                close(arm['original_year'],np.arange(1,31),'original_clock',errors,atol=0,rtol=0)
                if branch=='control':
                    for key in STATE_FIELDS+('actions','posttrade_holdings','fees','moved','qg_hold','qg_restore','qbad_hold_count','qbad_restore_count','forecast_n','predicted_score'):
                        close(arm[key],old[key],'identity_control/'+key,errors)
                    close(arm['policy_actions'],old['actions'],'identity_control/policy_actions',errors)
                else:
                    close(arm['holdings'][:t+1],old['holdings'][:t+1],'treatment/shared_prefix_holdings',errors)
                prefix=pid+'__'+branch
                for key,value in arm.items():trees[prefix+'__'+key]=value
                trace_groups[prefix]=dict(pair_id=pid,branch=branch,scenario_id=item['scenario_id'],tau=tau,
                    original_job=item['job_id'],selected_candidate_order=item['selected_candidate_order'],
                    original_action=old_action,forced_first_action=first)
                branches.append(dict(pair_id=pid,branch=branch,status='completed_intervention',tau=tau,
                    scenario_id=item['scenario_id'],**metrics[branch],
                    suffix_total_log_growth=suffix['total_remaining_log_growth'],suffix_nominal_fees=suffix['nominal_fees']))
                qrows.extend(witness_rows(item,branch,arm,tau))
            c,cf=metrics['control'],metrics['treatment']
            summary.update(control_terminal_real=c['terminal_real'],treatment_terminal_real=cf['terminal_real'],
                control_mdd=c['mdd'],treatment_mdd=cf['mdd'],delta_mdd=cf['mdd']-c['mdd'],
                delta_log_growth=float(np.log(cf['terminal_real']/c['terminal_real'])),
                control_breach=c['breach'],treatment_breach=cf['breach'],delta_breach=int(cf['breach'])-int(c['breach']),
                control_first_crossing=c['first_crossing'],treatment_first_crossing=cf['first_crossing'],
                delta_nominal_fees=cf['nominal_fees']-c['nominal_fees'],
                zero_transfer_intervention=bool(np.allclose(arms['control']['posttrade_holdings'][t],arms['treatment']['posttrade_holdings'][t],atol=1e-12,rtol=0)))
            summaries.append(summary);checks.append(dict(pair_id=pid,status='identity_control_passed',maximum_errors=errors))
        if len(summaries)!=11 or len(branches)!=22 or len(qrows)!=66:raise AssertionError('All prespecified pairs/branches/Q roles must remain')
        for name,rows in [('intervention_summary',summaries),('branch_summary',branches),('q_witness_inventory',qrows)]:
            pd.DataFrame(rows).to_csv(OUT/(name+'.csv.gz'),index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
        writej(OUT/'intervention_summary.json',summaries);writej(OUT/'trace_groups.json',trace_groups)
        writej(OUT/'identity_and_precondition_checks.json',dict(status='passed',pairs=checks))
        np.savez_compressed(OUT/'full_branches.npz',**trees)
        sources_end,inputs_end=snapshots(p,digest)
        outputs={f.name:sha(f) for f in OUT.iterdir() if f.name!='manifest.json'}
        writej(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started,protocol_sha256=digest,sources=sources,sources_at_end=sources_end,
            input_sha256=inputs,input_sha256_at_end=inputs_end,outputs_sha256=outputs,
            development_arrays_sha256=market_digest.hexdigest(),scenario_ids_loaded=ids.tolist(),final_scenarios_loaded=False,
            pairs=11,branches=22,Q_witness_role_slots=66,tree_arrays=len(trees),new_fits=0,
            original_policy_changed=False,admission_changed=False,current_C_financials_added=False,
            estimand='One prespecified action intervention plus all subsequent original-policy feedback on a fixed observed market path'))
    except Exception as error:
        writej(OUT/'failure_receipt.json',dict(status='failed_attempt_preserved',error=str(error),traceback=traceback.format_exc(),
            protocol_sha256=digest,source_sha256=sha(Path(__file__))))
        raise
    finally:
        for archive in archives.values():archive.close()
    print(json.dumps(dict(status='complete',pairs=11,branches=22,witness_roles=66,manifest_sha256=sha(OUT/'manifest.json'))))


if __name__=='__main__':main()
