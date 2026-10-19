"""Independent fixed first-disagreement intervention audit.

New diagnostic producer/runner and all project execution/policy/metric kernels
are never imported.  Original full-H physical accounts use the old independent
100-step-fee scalar reference.  Fixed predeclared Q slots use old scalar_fan;
all other saved active Qs receive score algebra/availability checks only.
"""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_fan_rollout_execution import reference_actual, scalar_fan

PROTOCOL=ROOT/'docs/failure_action_intervention_protocol.json'
PLAN=ROOT/'docs/failure_action_intervention_audit_plan.json'
OUT=ROOT/'results/model_review/failure_action_intervention_v1'
H,LOSS,COST,TOL=30,.4,.0025,1e-12
PRICES=(0.,.001,.003,.01,.03,.1,.3,1.,None)
QFLOAT=('qg_hold','qg_restore','predicted_score','predicted_growth_advantage_restore')
QCOUNT=('qbad_hold_count','qbad_restore_count','forecast_n','predicted_breach_count_delta_restore')


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):d.update(block)
    return d.hexdigest()


def readj(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return json.load(stream)
    return json.loads(path.read_text())


def null(v):return v is None or isinstance(v,(float,np.floating)) and np.isnan(v)


def original_tau(model_actions,B_actions):
    m,b=np.asarray(model_actions),np.asarray(B_actions)
    if m.shape!=b.shape or m.ndim!=1 or len(m)!=H:raise ValueError('Complete same originalH actions required')
    if not np.isin(m,('restore','hold')).all() or not np.isin(b,('restore','hold')).all():raise ValueError('Unknown original action')
    changed=np.flatnonzero(m!=b)
    tau=int(changed[0]+1) if len(changed) else None
    if tau is not None and tau<2:raise ValueError('Mandatory same initial restore cannot be an intervention')
    return tau


def gate(qh,qr,kh,kr,n,eta):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:raise ValueError('Positive integer own fan size required')
    if any(isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<=n for k in (kh,kr)):
        raise ValueError('Integer bounded model branch bad counts required')
    if not np.isfinite([qh,qr]).all():raise ValueError('Finite net total logs required')
    if eta is None:return dict(score=None,action='restore' if kr<kh or kr==kh and qr-qh>TOL else 'hold')
    if isinstance(eta,bool) or not np.isfinite(eta) or eta<0:raise ValueError('Finite technical eta required')
    score=float(qr-qh-H*eta*(kr-kh)/n)
    return dict(score=score,action='restore' if score>TOL else 'hold')


def nominal_slots(tau):
    """Keep six branch/role records per pair, including aliases and skips."""
    if tau is not None and (isinstance(tau,bool) or not isinstance(tau,int) or not 2<=tau<=H):
        raise ValueError('Original first disagreement tau2..H orNULL required')
    rows=[]
    for branch in ('control','treatment'):
        first={}
        for role,year in (('tau',tau),('tau_plus_one',None if tau is None else tau+1),('original_H',H)):
            if tau is None:status='no_intervention';alias=None
            elif year>H:status='beyond_original_H';alias=None
            elif year in first:status='alias';alias=first[year]
            else:status='candidate';alias=None;first[year]=role
            rows.append(dict(branch=branch,role=role,year=year,status=status,alias_of=alias))
    return rows


def replay(returns,inflation,target,actions):
    actions=list(actions)
    if len(actions)!=H or actions[0]!='restore':raise ValueError('Original full history with initial fee required')
    def action(t,x,cpi,peak,mdd,bad):
        if actions[t] not in ('hold','restore'):raise ValueError('Unknown saved action')
        if bad and actions[t]!='hold':raise AssertionError('Absorbing pending cannot resume')
        return actions[t]
    return reference_actual(returns,inflation,target,LOSS,action,cost_rate=COST)


def outcomes(ref):
    tau=int(ref['first_crossing'])
    return dict(terminal_real=float(ref['real_wealth'][-1]),total_log_growth=math.log(float(ref['real_wealth'][-1]/ref['real_wealth'][0])),
        mean_annual_log_growth=math.log(float(ref['real_wealth'][-1]/ref['real_wealth'][0]))/H,
        mdd=float(ref['running_mdd'][-1]),breach=bool(ref['ever_breached'][-1]),first_crossing=None if tau<0 else tau,
        nominal_fees=float(sum(ref['fees'])),nominal_moved=float(sum(ref['moved'])))


class Checks:
    def __init__(self):self.n=0;self.errors={}
    def check(self,condition,label):
        self.n+=1
        if not condition:raise AssertionError(label)
    def close(self,actual,expected,label,*,atol=3e-10,rtol=3e-13):
        a,b=np.asarray(actual),np.asarray(expected);self.check(a.shape==b.shape,label+'/shape')
        if a.dtype.kind in 'bUS' or b.dtype.kind in 'bUS':self.check(np.array_equal(a,b),label+'/exact');return
        err=np.abs(a.astype(float)-b.astype(float));valid=np.isfinite(err)
        field=label.split('/')[-1];self.errors[field]=max(self.errors.get(field,0.),float(err[valid].max(initial=0)))
        self.check(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),label+'/numeric')
    def scalar(self,actual,expected,label):
        if null(expected):self.check(null(actual),label+'/NULL')
        elif isinstance(expected,str):self.check(actual==expected,label+'/text')
        elif isinstance(expected,(bool,np.bool_)):self.check(isinstance(actual,(bool,np.bool_)) and actual==expected,label+'/bool')
        else:self.close(actual,expected,label)


def check_original_prefix(m,b,control,treatment,tau,checks,label):
    checks.scalar(tau,original_tau(m['actions'],b['actions']),label+'/original first disagreement')
    if tau is None:return
    t=tau-1
    checks.check(not bool(m['pending_at_decision'][t]) and not bool(b['pending_at_decision'][t]),label+'/both own original states active')
    for field in ('holdings','weights','nominal_wealth','real_wealth','cpi','real_peak','current_drawdown','running_mdd','ever_breached'):
        checks.close(m[field][:tau],b[field][:tau],label+'/same original M/B prestate '+field)
        checks.close(control[field][:tau],m[field][:tau],label+'/control whole prefix '+field)
        checks.close(treatment[field][:tau],m[field][:tau],label+'/treatment whole prefix '+field)
    checks.close(control['actions'],m['actions'],label+'/control all original actions')
    checks.close(treatment['actions'][:t],m['actions'][:t],label+'/treatment original action prefix')
    checks.scalar(control['actions'][t],m['actions'][t],label+'/control tau action')
    checks.scalar(treatment['actions'][t],b['actions'][t],label+'/treatment forced Btau action')


def check_saved_Q(source,eta,n,*,tau,branch,checks,label):
    active=~np.asarray(source['pending_at_decision'],bool);active[0]=False
    for field in QFLOAT:
        value=np.asarray(source[field],float)
        checks.check(value.shape==(H,) and np.isfinite(value[active]).all() and np.isnan(value[~active]).all(),label+'/Qfloat availability '+field)
    for field in QCOUNT:
        value=np.asarray(source[field])
        checks.check(value.shape==(H,) and value.dtype.kind in 'iu' and (value[~active]==-1).all(),label+'/Qinteger availability '+field)
    for t in np.flatnonzero(active):
        checks.scalar(int(source['forecast_n'][t]),n,label+'/own fan size')
        kh,kr=int(source['qbad_hold_count'][t]),int(source['qbad_restore_count'][t])
        result=gate(source['qg_hold'][t],source['qg_restore'][t],kh,kr,n,eta)
        checks.close(source['predicted_score'][t],result['score'],label+'/saved score',atol=2e-12,rtol=0)
        checks.scalar(source['predicted_growth_advantage_restore'][t],source['qg_restore'][t]-source['qg_hold'][t],label+'/saved delta net growth')
        checks.scalar(int(source['predicted_breach_count_delta_restore'][t]),kr-kh,label+'/saved delta event count')
        if branch=='treatment' and tau==t+1:continue
        checks.scalar(str(source['actions'][t]),result['action'],label+'/unforced own-state score action')
    return int(sum(active))



STATE_FIELDS=('holdings','weights','nominal_wealth','real_wealth','cpi','real_peak','current_drawdown','running_mdd','ever_breached')
PERIOD_FIELDS=('actions','pending_at_decision','posttrade_holdings','fees','moved')
BASE_Q_FIELDS=('qg_hold','qg_restore','qbad_hold_count','qbad_restore_count','forecast_n','predicted_score')


def independent_Q_slot(ref,full,year,training_returns,training_inflation,target,eta,checks,label,cache):
    if not 1<=year<=H:raise ValueError('Original in-horizon slot required')
    t=year-1
    if year==1:return dict(status='initial_not_optional',year=year)
    if bool(ref['pending_at_decision'][t]):
        for field in QFLOAT:checks.check(np.isnan(full[field][t]),label+'/pending Q float unavailable')
        for field in QCOUNT:checks.check(full[field][t]==-1,label+'/pending Q count unavailable')
        return dict(status='pending_unavailable',year=year)
    digest=hashlib.sha256()
    for value in (np.asarray([year,H,LOSS,COST]),ref['holdings'][t],np.asarray([ref['cpi'][t],ref['real_peak'][t],ref['running_mdd'][t]]),
        np.asarray(target),training_returns,training_inflation):digest.update(np.ascontiguousarray(value).tobytes())
    key=digest.hexdigest();reused=key in cache
    if not reused:
        cache[key]=scalar_fan(ref['holdings'][t],float(ref['cpi'][t]),float(ref['real_peak'][t]),float(ref['running_mdd'][t]),
            target,LOSS,training_returns,training_inflation,t,cost_rate=COST)
    q=cache[key];n=len(training_returns)
    qh,qr=q['hold']['mean'],q['restore']['mean'];kh,kr=q['hold']['count'],q['restore']['count']
    checks.close(full['qg_hold'][t],qh,label+'/independent net total-log Qhold',atol=2e-12,rtol=0)
    checks.close(full['qg_restore'][t],qr,label+'/independent net total-log Qrestore',atol=2e-12,rtol=0)
    checks.scalar(int(full['qbad_hold_count'][t]),kh,label+'/independent fullfan kh')
    checks.scalar(int(full['qbad_restore_count'][t]),kr,label+'/independent fullfan kr')
    checks.scalar(int(full['forecast_n'][t]),n,label+'/independent ownfan N')
    chosen=gate(qh,qr,kh,kr,n,eta)
    checks.close(full['predicted_score'][t],chosen['score'],label+'/independent Q score',atol=2e-12,rtol=0)
    checks.scalar(str(full['policy_actions'][t]),chosen['action'],label+'/independent unfixed policy action')
    return dict(status='recomputed',year=year,unique_state_key=key,reused_identical_Q_state=reused,training_n=n,
        independent_qg_hold=qh,independent_qg_restore=qr,independent_qbad_hold_count=kh,independent_qbad_restore_count=kr,
        independent_score=chosen['score'],independent_policy_action=chosen['action'])

PARENT=ROOT/'results/model_review/failure_path_mechanisms_v1'
EXPECTED_PAIRS={'full_35','inner_35','full_112','inner_112','full_52','inner_52','full_98','inner_98','outer_r2_52','outer_r4_112','outer_r5_112'}


def snapshot(p,m):
    paths={PROTOCOL,PLAN,OUT/'manifest.json',Path(__file__).resolve(),ROOT/'tests/test_audit_failure_action_intervention.py',
        ROOT/'scripts/audit_fan_rollout_execution.py',ROOT/'scripts/audit_review_pending_execution.py'}
    for key in ('execution_sources_sha256','frozen_input_sha256'):
        for rel,digest in p[key].items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts or len(digest)!=64:raise ValueError('Repository-relative frozen SHA256 closure required')
            if sha(ROOT/rel)!=digest:raise AssertionError('Frozen CF source/input changed: '+rel)
            paths.add(ROOT/rel)
    for name,digest in m['outputs_sha256'].items():
        if Path(name).name!=name or sha(OUT/name)!=digest:raise AssertionError('CF output hash changed: '+name)
        paths.add(OUT/name)
    return {str(path.relative_to(ROOT)):sha(path) for path in sorted(paths)}


def extract_parent(archive,group,meta,sid):
    ids=meta['scenario_ids']
    if len(set(ids))!=len(ids) or sid not in ids:raise ValueError('Unique original parent scenario required')
    prefix=group+'__';index=ids.index(sid);out={}
    for key in archive.files:
        if key.startswith(prefix):
            field=key[len(prefix):];values=archive[key]
            out[field]=values.copy() if field in ('targets','annual_prices','loss_tolerances') else values[index].copy()
    return out


def extract_branch(archive,prefix):
    head=prefix+'__';raw={k[len(head):]:archive[k].copy() for k in archive.files if k.startswith(head)}
    if not raw:raise ValueError('Intervention full-H branch arrays missing')
    raw['target_execution']=raw['actions']=='restore'
    raw['predicted_growth_advantage_restore']=raw['qg_restore']-raw['qg_hold']
    active=~raw['pending_at_decision'];active[0]=False
    raw['predicted_breach_count_delta_restore']=np.where(active,raw['qbad_restore_count']-raw['qbad_hold_count'],-1)
    return raw


def branch_metrics(ref):
    metric=outcomes(ref);metric.pop('mean_annual_log_growth')
    metric.update(trade_fraction=float(np.mean(ref['trades'])),restore_fraction=float(np.mean(ref['actions']=='restore')))
    return metric


def run():
    receipt=OUT/'independent_audit.json'
    if receipt.exists():raise RuntimeError('Never overwrite independent CF audit receipt')
    p=readj(PROTOCOL)
    if p.get('execution_ready') is not True:raise RuntimeError('Wait for frozen execution-ready CF protocol')
    if (p['data']!={'path':'data/SafeInvest_Synthetic_Data.csv','scenario_ids':list(range(1,121)),'allow_final':False}
        or (p['original_H'],p['loss_tolerance'],p['cost_rate'])!=(H,LOSS,COST)
        or p['output_directory']!=str(OUT.relative_to(ROOT))):raise ValueError('Original guarded CF accounting contract differs')
    items=p['inventory'];design=readj(ROOT/'docs/failure_action_intervention_design.json')
    canonical=json.dumps(items,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    if len(items)!=11 or {x['pair_id'] for x in items}!=EXPECTED_PAIRS or [x['pair_id'] for x in items]!=design['pair_ids'] or hashlib.sha256(canonical).hexdigest()!=design['parent_inventory_canonical_sha256']:
        raise ValueError('Frozen original11pair inventory differs')
    m=readj(OUT/'manifest.json')
    expected_inputs={**p['frozen_input_sha256'],str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    if (m.get('status')!='complete' or m['protocol_sha256']!=sha(PROTOCOL)
        or m['sources']!=m['sources_at_end'] or m['sources']!=p['execution_sources_sha256']
        or m['input_sha256']!=m['input_sha256_at_end'] or m['input_sha256']!=expected_inputs):
        raise RuntimeError('Complete CF producer with frozen matching source/input/protocol required before values')
    before=snapshot(p,m);checks=Checks()
    for key,value in [('pairs',11),('branches',22),('Q_witness_role_slots',66),('new_fits',0),('original_policy_changed',False),('admission_changed',False),('current_C_financials_added',False),('final_scenarios_loaded',False)]:checks.scalar(m[key],value,'manifest/'+key)
    checks.check(m['scenario_ids_loaded']==list(range(1,121)),'development original scope')
    pa=readj(PARENT/'independent_audit.json')
    checks.check(pa['status']=='passed' and pa['parent_manifest_sha256']==sha(PARENT/'manifest.json'),'audited first-stage source binding')
    checks.check(readj(PARENT/'manifest.json')['status']=='complete','first-stage source complete')
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    market,inflation,ids=load_paths(ROOT/p['data']['path'],range(1,121),allow_final=False)
    digest=hashlib.sha256()
    for x in (ids,market,inflation):digest.update(np.ascontiguousarray(x).tobytes())
    checks.check(np.array_equal(ids,np.arange(1,121)) and digest.hexdigest()==m['development_arrays_sha256'],'independent guarded original development arrays')
    summaries=readj(OUT/'intervention_summary.json');summarycsv=pd.read_csv(OUT/'intervention_summary.csv.gz',float_precision='round_trip')
    branches=pd.read_csv(OUT/'branch_summary.csv.gz',float_precision='round_trip');qrows=pd.read_csv(OUT/'q_witness_inventory.csv.gz',float_precision='round_trip')
    trace_groups=readj(OUT/'trace_groups.json');producerchecks=readj(OUT/'identity_and_precondition_checks.json')
    checks.check(len(summaries)==11 and {x['pair_id'] for x in summaries}==EXPECTED_PAIRS,'11complete JSON interventions')
    checks.check(len(summarycsv)==11 and set(summarycsv.pair_id)==EXPECTED_PAIRS and not summarycsv.pair_id.duplicated().any(),'11complete CSV interventions')
    checks.check(len(branches)==22 and not branches.duplicated(['pair_id','branch']).any() and set(branches.pair_id)==EXPECTED_PAIRS,'22complete unique branches')
    checks.check(len(qrows)==66 and not qrows.duplicated(['pair_id','branch','role']).any() and set(qrows.pair_id)==EXPECTED_PAIRS,'66fixed nominal Qroles')
    checks.check(not qrows.slot_id.duplicated().any(),'unique explicit Q slotIDs')
    checks.check(producerchecks['status']=='passed' and len(producerchecks['pairs'])==11,'all identity/precondition producer records retained')
    summarymap={x['pair_id']:x for x in summaries};groups={};parents={};Qcache={};slotresults=[];pairresults=[];physical_periods=0;active_algebra=0
    first_protocol=readj(ROOT/'docs/failure_path_mechanism_protocol.json')
    checks.check(items==first_protocol['inventory'],'exact unchanged first-stage pair metadata')
    try:
        with np.load(OUT/'full_branches.npz',allow_pickle=False) as new:
            for item in items:
                pid=item['pair_id'];folder=item['folder'];scope=item['trace_scope'];sid=item['scenario_id'];eta=PRICES[item['selected_candidate_order']];n=item['training_fan_n']
                if folder not in groups:groups[folder]=readj(ROOT/'results/model_review'/folder/'trace_groups.json')
                key=(folder,scope)
                if key not in parents:parents[key]=np.load(ROOT/'results/model_review'/folder/(scope+'_traces.npz'),allow_pickle=False)
                meta=groups[folder][scope][item['model_group']];old=extract_parent(parents[key],item['model_group'],meta,sid)
                checks.close(old['targets'],item['target'],pid+'/original own target',atol=0,rtol=0)
                checks.check(meta['metadata']['job_id']==item['job_id'] and meta['metadata']['selection_training_n']==n,pid+'/own original job/fan size')
                r,z=market[sid-1],inflation[sid-1]
                B=reference_actual(r,z,item['target'],LOSS,lambda t,x,cp,pk,dd,bad:'hold' if bad else 'restore',cost_rate=COST)
                tau=original_tau(old['actions'],B['actions']);summary=summarymap[pid];csv=summarycsv.loc[summarycsv.pair_id.eq(pid)].iloc[0].to_dict()
                for kind,row in [('JSON',summary),('CSV',csv)]:
                    for field,value in [('pair_id',pid),('stage',item['stage']),('scenario_id',sid),('tau',tau),('hypothetical_diagnostic',True)]:checks.scalar(row[field],value,pid+'/'+kind+' '+field)
                if tau is None:
                    for record in (summary,csv):
                        checks.scalar(record['status'],'no_action_difference',pid+'/no disagreement status')
                        for field in ('original_action','forced_action','control_terminal_real','treatment_terminal_real','control_mdd','treatment_mdd','delta_mdd','delta_log_growth','control_breach','treatment_breach','delta_breach','control_first_crossing','treatment_first_crossing','delta_nominal_fees','zero_transfer_intervention'):checks.scalar(record[field],None,pid+'/no disagreement '+field)
                    for branch in ('control','treatment'):
                        records=branches.loc[branches.pair_id.eq(pid)&branches.branch.eq(branch)]
                        checks.check(len(records)==1,pid+'/preserve no-op branch')
                        checks.scalar(records.iloc[0]['status'],'no_action_difference',pid+'/no-op branch status')
                    branch_full={};branch_refs={}
                else:
                    branch_full={};branch_refs={};metrics={};t=tau-1
                    for branch in ('control','treatment'):
                        prefix=pid+'__'+branch;full=extract_branch(new,prefix);branch_full[branch]=full
                        ref=replay(r,z,item['target'],full['actions']);branch_refs[branch]=ref;physical_periods+=H
                        for field in (*STATE_FIELDS,*PERIOD_FIELDS,'target_execution','trades'):
                            checks.close(full[field],ref[field],prefix+'/independent whole-H '+field)
                        checks.close(full['fee_fraction'],ref['fees']/ref['nominal_wealth'][:-1],prefix+'/fee fraction')
                        checks.close(full['original_year'],np.arange(1,H+1),prefix+'/original full clock',atol=0,rtol=0)
                        checks.close(full['forced'],np.arange(H)==t,prefix+'/one forced decision only')
                        active_algebra+=check_saved_Q(full,eta,n,tau=tau,branch=branch,checks=checks,label=prefix)
                        for year in range(H):
                            if full['pending_at_decision'][year]:expected='hold'
                            elif year==0:expected='restore'
                            else:expected=gate(full['qg_hold'][year],full['qg_restore'][year],int(full['qbad_hold_count'][year]),int(full['qbad_restore_count'][year]),n,eta)['action']
                            checks.scalar(str(full['policy_actions'][year]),expected,prefix+'/policy recommendation all years')
                        if branch=='control':
                            for field in (*STATE_FIELDS,*PERIOD_FIELDS,*BASE_Q_FIELDS):
                                checks.close(full[field],old[field],prefix+'/complete control identity '+field,atol=2e-12 if field in BASE_Q_FIELDS else 3e-10,rtol=0 if field in BASE_Q_FIELDS else 3e-13)
                        tg=trace_groups[prefix]
                        for field,value in dict(pair_id=pid,branch=branch,scenario_id=sid,tau=tau,original_job=item['job_id'],selected_candidate_order=item['selected_candidate_order'],
                            original_action=str(old['actions'][t]),forced_first_action=str(old['actions'][t]) if branch=='control' else str(B['actions'][t])).items():checks.scalar(tg[field],value,prefix+'/trace metadata '+field)
                        rows=branches.loc[branches.pair_id.eq(pid)&branches.branch.eq(branch)];checks.check(len(rows)==1,prefix+'/one branch summary')
                        record=rows.iloc[0].to_dict();metric=branch_metrics(ref);metrics[branch]=metric
                        for field,value in dict(pair_id=pid,branch=branch,status='completed_intervention',tau=tau,scenario_id=sid,**metric,
                            suffix_total_log_growth=math.log(ref['real_wealth'][-1]/ref['real_wealth'][t]),suffix_nominal_fees=float(sum(ref['fees'][t:]))).items():checks.scalar(record[field],value,prefix+'/independent branch metric '+field)
                    c,cf=metrics['control'],metrics['treatment'];cm,tm=branch_full['control'],branch_full['treatment']
                    check_original_prefix(old,B,cm,tm,tau,checks,pid)
                    expected=dict(status='completed_intervention',original_action=str(old['actions'][t]),forced_action=str(B['actions'][t]),
                        control_terminal_real=c['terminal_real'],treatment_terminal_real=cf['terminal_real'],control_mdd=c['mdd'],treatment_mdd=cf['mdd'],
                        delta_mdd=cf['mdd']-c['mdd'],delta_log_growth=math.log(cf['terminal_real']/c['terminal_real']),
                        control_breach=c['breach'],treatment_breach=cf['breach'],delta_breach=int(cf['breach'])-int(c['breach']),
                        control_first_crossing=c['first_crossing'],treatment_first_crossing=cf['first_crossing'],
                        delta_nominal_fees=cf['nominal_fees']-c['nominal_fees'],
                        zero_transfer_intervention=bool(np.allclose(cm['posttrade_holdings'][t],tm['posttrade_holdings'][t],atol=1e-12,rtol=0)))
                    for kind,row in [('JSON',summary),('CSV',csv)]:
                        for field,value in expected.items():checks.scalar(row[field],value,pid+'/'+kind+' independent contrast '+field)
                    pairresults.append(dict(pair_id=pid,tau=tau,control_breach=c['breach'],treatment_breach=cf['breach'],delta_mdd=expected['delta_mdd'],delta_log_growth=expected['delta_log_growth']))
                for slot in nominal_slots(tau):
                    branch,role,year=slot['branch'],slot['role'],slot['year'];slotid=pid+':'+branch+':'+role
                    records=qrows.loc[qrows.slot_id.eq(slotid)];checks.check(len(records)==1,slotid+'/one nominal role')
                    row=records.iloc[0].to_dict()
                    status={'no_intervention':'no_action_difference_skip','beyond_original_H':'outside_original_horizon_skip','alias':'duplicate_year_alias'}.get(slot['status'])
                    if status is None:status='already_breached_pending_skip' if branch_refs[branch]['pending_at_decision'][year-1] else 'active_Q'
                    alias=None if slot['alias_of'] is None else pid+':'+branch+':'+slot['alias_of']
                    for field,value in dict(slot_id=slotid,pair_id=pid,branch=branch,role=role,year=year,status=status,alias_of=alias,scenario_id=sid,training_fan_n=n).items():checks.scalar(row[field],value,slotid+'/identity '+field)
                    financial=('forced_at_decision','qg_hold','qg_restore','qbad_hold_count','qbad_restore_count','forecast_n','predicted_score','policy_action','executed_action')
                    if status=='active_Q':
                        full=branch_full[branch];t=year-1
                        values=dict(forced_at_decision=bool(full['forced'][t]),qg_hold=float(full['qg_hold'][t]),qg_restore=float(full['qg_restore'][t]),
                            qbad_hold_count=int(full['qbad_hold_count'][t]),qbad_restore_count=int(full['qbad_restore_count'][t]),forecast_n=int(full['forecast_n'][t]),
                            predicted_score=float(full['predicted_score'][t]),policy_action=str(full['policy_actions'][t]),executed_action=str(full['actions'][t]))
                        for field,value in values.items():checks.scalar(row[field],value,slotid+'/copied slot '+field)
                        tr=np.asarray(item['training_ids'])-1
                        independent=independent_Q_slot(branch_refs[branch],full,year,market[tr],inflation[tr],item['target'],eta,checks,slotid,Qcache)
                    else:
                        for field in financial:checks.scalar(row[field],None,slotid+'/unavailable slot financial '+field)
                        independent=dict(status=status,year=year)
                    slotresults.append(dict(slot_id=slotid,pair_id=pid,branch=branch,role=role,**independent))
    finally:
        for archive in parents.values():archive.close()
    checks.check(len(slotresults)==66,'all fixed nominal roles retained including aliases/skips')
    after=snapshot(p,m);checks.check(before==after,'CF source/input/protocol/output immutable throughout audit')
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks_passed=checks.n,
        maximum_absolute_errors=checks.errors,maximum_numeric_absolute_error=max(checks.errors.values(),default=0.),
        parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(PLAN),auditor_sha256=sha(Path(__file__)),
        test_sha256=sha(ROOT/'tests/test_audit_failure_action_intervention.py'),pairs_checked=11,complete_full_history_branches_independently_replayed=physical_periods//H,
        full_history_annual_account_periods_checked=physical_periods,active_saved_Q_score_algebra_checked=active_algebra,
        nominal_Q_role_slots=66,independently_recomputed_Q_role_slots=sum(x['status']=='recomputed' for x in slotresults),unique_Q_states_recomputed=len(Qcache),
        Q_skip_and_alias_records=sum(x['status']!='recomputed' for x in slotresults),all_adaptively_visited_Q_states_independently_recomputed=False,
        final_scenarios_loaded=False,new_fits=0,admission_changed=False,
        independence='Previous separate100-step fee/state accounting and scalar_fan only; new intervention producer/runner not imported. Control compared with original savedM, treatment accounting replayed from recorded actions; fullfan Q independently rebuilt only at predeclared fixed roles.',
        inference_limits=['One action intervention plus original-policy feedback on these previously selected fixed market paths; no population treatment effect or conditional forecast calibration.',
            'Same original market IDs and pairs overlap across fits; branch counts are not independent trials.',
            'No controller, target, fan, alpha, eta, scenario exclusion or admission search.'],
        sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,pair_results=pairresults,Q_slot_results=slotresults)
    receipt.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','complete_full_history_branches_independently_replayed','independently_recomputed_Q_role_slots','unique_Q_states_recomputed','maximum_numeric_absolute_error')}))
    return result

if __name__=='__main__':run()
