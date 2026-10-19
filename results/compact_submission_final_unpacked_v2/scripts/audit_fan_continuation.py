"""Independent fixed-state fan/deletion/continuation diagnostic audit.

No new diagnostic producer or execution kernel is imported. Numerical entry
point will only read a complete frozen archive after its protocol is ready.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_fan_rollout_execution import scalar_fan
import audit_simple_execution_comparison as accounting

PROTOCOL=ROOT/'docs/fan_continuation_diagnostic_protocol.json'
AUDIT_PLAN=ROOT/'docs/fan_continuation_audit_plan.json'
OUT=ROOT/'results/model_review/fan_continuation_diagnostic_v1'
TOL=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return accounting.read_json(path)
def writej(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def recommendation(qh,qr,kh,kr,n,eta,original_horizon):
    """Original total-log score at fixed eta; no fee subtraction or refit."""
    for value,label in ((n,'Positive original fan N'),(original_horizon,'Original horizon')):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError(label)
    for value in (kh,kr):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 0<=value<=n:raise ValueError('Integer original-H branch event count required')
    if isinstance(qh,(bool,np.bool_)) or isinstance(qr,(bool,np.bool_)) or not np.isfinite([qh,qr]).all():raise ValueError('Finite net total-log Q required')
    if eta is not None and (isinstance(eta,(bool,np.bool_)) or not np.isfinite(eta) or eta<0):raise ValueError('Fixed nonnegative finite eta or separate risk_first required')
    delta=float(qr)-float(qh);dp=(int(kr)-int(kh))/int(n)
    score=None if eta is None else delta-int(original_horizon)*float(eta)*dp
    restore=(kr<kh or (kr==kh and delta>TOL)) if eta is None else score>TOL
    if restore and delta<=TOL and not kr<kh:raise AssertionError('Nonpositive-growth restore must strictly reduce predicted event count')
    return dict(delta_net_log_growth=delta,delta_breach_probability=dp,score=score,restore=bool(restore),action='restore' if restore else 'hold')


def explicit_delete_one(hold_logs,restore_logs,hold_events,restore_events,training_ids,eta,original_horizon=30):
    """Delete whole members by explicit slices, not sum-minus saved outputs."""
    h,r=np.asarray(hold_logs,float),np.asarray(restore_logs,float)
    hb,rb=np.asarray(hold_events),np.asarray(restore_events);ids=list(training_ids);n=len(ids)
    if (n<2 or h.shape!=(n,) or r.shape!=(n,) or hb.shape!=(n,) or rb.shape!=(n,)
        or hb.dtype.kind!='b' or rb.dtype.kind!='b' or not np.isfinite(h).all() or not np.isfinite(r).all()
        or len(set(ids))!=n or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 1<=i<=120 for i in ids)):
        raise ValueError('Complete finite original-ID paired branch members required')
    original=recommendation(float(np.mean(h)),float(np.mean(r)),int(hb.sum()),int(rb.sum()),n,eta,original_horizon)
    rows=[]
    for j,sid in enumerate(ids):
        keep=[i for i in range(n) if i!=j]
        qh,qr=float(np.mean(h[keep])),float(np.mean(r[keep]));kh,kr=int(hb[keep].sum()),int(rb[keep].sum())
        choice=recommendation(qh,qr,kh,kr,n-1,eta,original_horizon)
        # Single-branch event probability is in[0,1]; paired event difference
        # is in[-1,1], hence the latter has the distinct2/N deletion bound.
        if abs(kh/(n-1)-float(hb.mean()))>1/n+5e-15 or abs(kr/(n-1)-float(rb.mean()))>1/n+5e-15:raise AssertionError('Single-branch deletion probability bound')
        if abs(choice['delta_breach_probability']-original['delta_breach_probability'])>2/n+5e-15:raise AssertionError('Paired deletion probability bound')
        rows.append(dict(deleted_training_id=int(sid),n_remaining=n-1,hold_Q=qh,restore_Q=qr,hold_k=kh,restore_k=kr,
                         **choice,action_flip=choice['restore']!=original['restore']))
    return original,rows


def replay_suffix(holdings,cpi,peak,worst,target,returns,inflation,year_index,*,first_action=None,fixed_actions=None,loss=.4,cost_rate=.0025):
    """Same archived state, original suffix, current action then branch-own B.

    fixed_actions instead replays the saved adaptive continuation from the
    identical state. Actual future is used only for this post-hoc diagnostic.
    """
    r,pi=np.asarray(returns,float),np.asarray(inflation,float);x=np.asarray(holdings,float).copy();w=np.asarray(target,float)
    if (r.ndim!=2 or r.shape[1]!=3 or pi.shape!=(len(r),) or isinstance(year_index,(bool,np.bool_))
        or not isinstance(year_index,(int,np.integer)) or not 0<=year_index<len(r)
        or not np.isfinite(r).all() or not np.isfinite(pi).all() or np.any(r<=-1) or np.any(pi<=-1)
        or x.shape!=(3,) or not np.isfinite(x).all() or np.any(x<0) or x.sum()<=0 or w.shape!=(3,)
        or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1,atol=1e-13,rtol=0)):
        raise ValueError('Original chronological suffix and positive same state required')
    if (first_action is None)==(fixed_actions is None):raise ValueError('Specify exactly one current branch or fixed saved action sequence')
    if first_action is not None and first_action not in ('hold','restore'):raise ValueError('Unknown current action')
    if fixed_actions is not None and len(fixed_actions)!=len(r)-year_index:raise ValueError('Complete saved suffix actions required')
    cp=float(cpi);p=float(peak);d=float(worst)
    if not np.isfinite([cp,p,d,loss,cost_rate]).all() or cp<=0 or p<=0 or not 0<=d<=1 or not 0<loss<1 or not 0<=cost_rate<1:raise ValueError('Finite valid original state/cost/loss required')
    initial_real=float(x.sum())/cp
    if p<initial_real-1e-10 or d<max(1-initial_real/p,0)-TOL:raise ValueError('Historical CPI/peak/MDD cannot be reset')
    bad=d-loss>TOL;first=None;rows=[]
    states=[dict(original_endpoint=year_index,holdings=x.copy(),cpi=cp,real_wealth=initial_real,real_peak=p,running_mdd=d,ever_breached=bad)]
    for j,t in enumerate(range(year_index,len(r))):
        action=str(fixed_actions[j]) if fixed_actions is not None else str(first_action) if j==0 else ('hold' if bad else 'restore')
        if action not in ('hold','restore') or bad and action!='hold':raise ValueError('Unknown action or violation of absorbing pending hold')
        pre=float(x.sum());post,fee,moved=accounting.independent_fee_trade(x,w,cost_rate=cost_rate) if action=='restore' else (x.copy(),0.,0.)
        rows.append(dict(original_year=t+1,action=action,pending_at_decision=bool(bad),pretrade_wealth=pre,posttrade_holdings=post.copy(),fee=fee,moved=moved))
        x=np.asarray([float(post[a])*(1+float(r[t,a])) for a in range(3)])
        cp*=1+float(pi[t]);real=float(x.sum())/cp;p=max(p,real);d=max(d,max(1-real/p,0.))
        if not bad and d-loss>TOL:first=t+1
        bad=bool(bad or d-loss>TOL)
        states.append(dict(original_endpoint=t+1,holdings=x.copy(),cpi=cp,real_wealth=real,real_peak=p,running_mdd=d,ever_breached=bad))
    return dict(total_log_growth=math.log(states[-1]['real_wealth']/initial_real),terminal_real=states[-1]['real_wealth'],
        terminal_mdd=d,ever_breached=bad,nominal_fees=float(sum(row['fee'] for row in rows)),current_fee=rows[0]['fee'],
        first_crossing_original_year=first,states=states,periods=rows)

BOOL_COLUMNS=('final_guard_admitted','actual_path_in_fan','recomputed_restore','self_deleted_action_flip',
              'is_actual_source_path','hold_ever_breached','restore_ever_breached','self_deleted','drop_restore','action_flip',
              'B_hold_ever_breached','B_restore_ever_breached','adaptive_ever_breached')
PRICES=(0.,.001,.003,.01,.03,.1,.3,1.,None)
PARENT_NAMES=('joint_liquidity_refit_v1','refit_stability_v1')


def check_mapping(root,mapping):
    for name,expected in mapping.items():
        if sha(Path(root)/name)!=expected:raise RuntimeError('Frozen file changed: '+str(name))


def read_table(path):
    import pandas as pd
    try:frame=pd.read_csv(path,low_memory=False)
    except pd.errors.EmptyDataError:return pd.DataFrame()
    for col in BOOL_COLUMNS:
        if col not in frame:continue
        def value(v):
            if pd.isna(v):return np.nan
            if isinstance(v,(bool,np.bool_)):return bool(v)
            if str(v) not in ('True','False','true','false'):raise ValueError('Literal boolean required: '+col)
            return str(v) in ('True','true')
        frame[col]=frame[col].map(value)
    return frame


def validate_inventory(p,plan,admissions,parents):
    if (p['data']['scenario_ids']!=list(range(1,121)) or p['data']['allow_final'] is not False
        or p['original_H']!=30 or p['loss_tolerance']!=.4 or p['cost_rate']!=.0025
        or p['expected_slots']!=39 or len(p['inventory'])!=39):raise ValueError('Frozen original39-slot diagnostic required')
    old={r['context']:r for r in admissions if r['original_execution_status']=='automated'}
    if len(old)!=13:raise ValueError('Exactly13 original old-auto fits required')
    expected={(r['context'],r['scenario_id'],r['original_year']):r for r in plan['inventory']['slots']}
    if set(expected)!={(s['context'],s['scenario_id'],s['year']) for s in p['inventory']}:raise ValueError('Prospective39 witness identities changed')
    if len({s['slot_id'] for s in p['inventory']})!=39 or {s['context'] for s in p['inventory']}!=set(old):raise ValueError('No absent, extra or repeated fit/slot allowed')
    for s in p['inventory']:
        a=old[s['context']];parent=parents[s['folder']];g=parent['groups'][s['group']];choice=parent['choices'][s['context']]
        retained=choice['methods']['joint_tie_set_refit_learner'];meta=g['metadata'];e=expected[(s['context'],s['scenario_id'],s['year'])]
        if (s['folder'] not in PARENT_NAMES or s['group']!=e['group'] or s['context_id']!=e['context_id']
            or s['repeat_id']!=e['repeat_id'] or s['final_guard_admitted']!=e['final_guard_admitted']
            or s['selected_beta']!=a['selected_beta'] or s['selected_candidate_order']!=a['selected_candidate_order']
            or s['final_guard_admitted']!=a['final_automated'] or retained['execution_status']!='automated'
            or retained['selected_beta']!=s['selected_beta'] or retained['selected_candidate_order']!=s['selected_candidate_order']
            or s['training_ids']!=choice['training_ids'] or g['scenario_ids']!=choice['evaluation_ids']
            or s['scenario_id']!=g['q_subset_scenario_id'] or s['scenario_id']!=min(g['scenario_ids']) or g['q_subset_years']!=[2,16,30]
            or meta['context']!=s['context'] or meta['context_id']!=s['context_id'] or meta['policy']!='joint_tie_set_refit_learner'
            or meta['training_beta']!=s['selected_beta'] or meta['horizon']!=30 or meta['loss_tolerance']!=.4
            or g['selected_candidate_order']!=s['selected_candidate_order']):raise ValueError('Archived policy/state/target lineage changed')
        train=s['training_ids'];evaluation=choice['evaluation_ids']
        if len(train)!=(120 if s['context_id']==0 else 96) or len(set(train))!=len(train) or any(i not in range(1,121) for i in train):raise ValueError('Own whole original training membership changed')
        if s['context_id']==0:
            if train!=list(range(1,121)) or evaluation!=train or s['folder']!=PARENT_NAMES[0] or s['repeat_id']!=0:raise ValueError('Full training resubstitution identity changed')
        elif set(train)&set(evaluation) or set(train)|set(evaluation)!=set(range(1,121)) or len(evaluation)!=24 or s['folder']!=PARENT_NAMES[1]:raise ValueError('OOF96/24 whole-path partition changed')
    return True


def independent_summary(states,members,deletions):
    """Pandas group/arithmetic reconciliation after all state values are audited."""
    import pandas as pd
    sf=pd.DataFrame(states);df=pd.DataFrame(deletions)
    def describe(series):
        x=series.dropna().to_numpy(float)
        return dict(n=len(x),minimum=float(x.min()) if len(x) else None,median=float(np.quantile(x,.5)) if len(x) else None,
                    maximum=float(x.max()) if len(x) else None,mean=float(sum(x)/len(x)) if len(x) else None)
    active=sf.loc[sf.state_status.eq('active')];groups=[]
    for scope in ('full_training_in_sample','outer_oof'):
        for guard in (None,False,True):
            selected=sf.loc[sf.scope.eq(scope)]
            if guard is not None:selected=selected.loc[selected.final_guard_admitted.eq(guard)]
            a=selected.loc[selected.state_status.eq('active')];d=df.loc[df.slot_id.isin(selected.slot_id)] if len(df) else pd.DataFrame()
            sd=d.loc[d.self_deleted] if len(d) else pd.DataFrame()
            groups.append(dict(scope=scope,final_guard_admitted=guard,slots=len(selected),active_slots=len(a),pending_slots=len(selected)-len(a),
                fits=selected.group.nunique(),distinct_original_market_ids=sorted(map(int,selected.scenario_id.unique())),
                active_distinct_original_market_ids=sorted(map(int,a.scenario_id.unique())),active_distinct_original_market_n=a.scenario_id.nunique(),
                year30_active_control_count=int(a.year.eq(30).sum()),deletion_records=len(d),deletion_action_flips=int(d.action_flip.sum()) if len(d) else 0,
                slots_with_any_deletion_flip=int(a.deletion_action_flips.gt(0).sum()),self_deletion_records=len(sd),self_deletion_action_flips=int(sd.action_flip.sum()) if len(sd) else 0,
                continuation_log_gap=describe(a.continuation_gap_total_log_growth),continuation_mdd_gap=describe(a.continuation_gap_history_mdd),
                continuation_fee_gap=describe(a.continuation_gap_nominal_fees),continuation_event_gap=describe(a.continuation_gap_event),
                future_action_mismatches=int(a.future_action_mismatches.sum())))
    return dict(slots=len(sf),fits=sf.group.nunique(),distinct_original_market_ids=sorted(map(int,sf.scenario_id.unique())),
        distinct_original_market_n=sf.scenario_id.nunique(),guard_admitted_slots=int(sf.final_guard_admitted.sum()),active_slots=len(active),
        active_distinct_original_market_ids=sorted(map(int,active.scenario_id.unique())),active_distinct_original_market_n=active.scenario_id.nunique(),
        year30_active_control_count=int(active.year.eq(30).sum()),pending_slots=len(sf)-len(active),branch_member_records=len(members),deletion_records=len(deletions),groups=groups)


def compare_tree(actual,expected,ledger,name):
    if isinstance(expected,dict):
        ledger.require(isinstance(actual,dict) and set(actual)==set(expected),name+'/keys')
        for key,value in expected.items():compare_tree(actual[key],value,ledger,name+'/'+key)
    elif isinstance(expected,list):
        ledger.require(isinstance(actual,list) and len(actual)==len(expected),name+'/length')
        for i,value in enumerate(expected):compare_tree(actual[i],value,ledger,name+'/'+str(i))
    else:accounting.maybe(actual,expected,ledger,name)


def audit_tables(p,plan,parents,returns,inflation,tables,summary):
    import time
    ledger=accounting.AuditLedger();started=time.monotonic()
    inventory=accounting.index_rows(tables['state_inventory'],('slot_id',));states=accounting.index_rows(tables['state_diagnostics'],('slot_id',))
    members=accounting.index_rows(tables['branch_members'],('slot_id','training_scenario_id')) if len(tables['branch_members']) else {}
    deletions=accounting.index_rows(tables['deletions'],('slot_id','training_scenario_id')) if len(tables['deletions']) else {}
    ledger.require(len(inventory)==39 and set(inventory)==set(states)=={(s['slot_id'],) for s in p['inventory']},'inventory/all39states')
    checked_members=set();checked_deletions=set();witnesses=[];active_count=0;adaptive_periods=0;branch_periods=0
    def check(row,expected,prefix):
        for key,value in expected.items():accounting.maybe(row[key],value,ledger,prefix+'/'+key)
    def budget():
        if time.monotonic()-started>60*plan['budget_minutes']:raise TimeoutError('Fixed audit budget exceeded; preserve scope and failures')
    for s in p['inventory']:
        budget();key=(s['slot_id'],);sr=states[key];iv=inventory[key]
        for field,value in s.items():
            if field=='training_ids':ledger.require(json.loads(iv[field])==value,'inventory/exact_trainingIDs')
            else:accounting.maybe(iv[field],value,ledger,'inventory/'+field)
        check(sr,{k:v for k,v in s.items() if k!='training_ids'},'state/metadata')
        parent=parents[s['folder']];group=parent['groups'][s['group']];archive=parent['traces'];i=group['scenario_ids'].index(s['scenario_id']);t=s['year']-1
        arm={name[len(s['group'])+2:]:archive[name] for name in archive if name.startswith(s['group']+'__')}
        x=arm['holdings'][i,t];cp=float(arm['cpi'][i,t]);peak=float(arm['real_peak'][i,t]);worst=float(arm['running_mdd'][i,t]);target=arm['targets']
        ledger.close(target,parent['choices'][s['context']]['case']['product']['weights'],'state/Q1_target',atol=2e-15,rtol=0)
        eta=PRICES[s['selected_candidate_order']];ledger.close(arm['annual_prices'],[np.inf if eta is None else eta],'state/original_eta')
        ledger.close(arm['loss_tolerances'],[.4],'state/original_loss')
        pending=bool(arm['pending_at_decision'][i,t]);action=str(arm['actions'][i,t]);n=len(s['training_ids'])
        ledger.require(pending==bool(arm['ever_breached'][i,t])==(worst-.4>TOL),'state/absorbing_status')
        fixed=dict(scope='full_training_in_sample' if s['context_id']==0 else 'outer_oof',state_status='pending_skip' if pending else 'active',actual_action=action,
            actual_path_in_fan=s['scenario_id'] in s['training_ids'],fan_n=n,eta_technical=eta,eta_kind='risk_first' if eta is None else 'finite_price',
            holdings_equity=float(x[0]),holdings_bond=float(x[1]),holdings_bill=float(x[2]),cpi=cp,real_peak=peak,running_mdd=worst,current_real=float(x.sum())/cp,
            target_equity=float(target[0]),target_bond=float(target[1]),target_bill=float(target[2]))
        check(sr,fixed,'state/saved_state')
        witness=dict(slot_id=s['slot_id'],context=s['context'],scenario_id=s['scenario_id'],original_year=s['year'],final_guard_admitted=s['final_guard_admitted'],status=fixed['state_status'])
        if pending:
            ledger.require(action=='hold' and np.isnan(arm['qg_hold'][i,t]) and np.isnan(arm['qg_restore'][i,t]) and arm['forecast_n'][i,t]==-1,'pending/archived_sentinels')
            metadata=set(s)-{'training_ids'}|set(fixed)
            for field in sr:
                if field not in metadata:accounting.maybe(sr[field],None,ledger,'pending/new_diagnostics_NULL/'+field)
            ledger.require(not any(k[0]==s['slot_id'] for k in members) and not any(k[0]==s['slot_id'] for k in deletions),'pending/no_replacement_records')
            witnesses.append(witness);continue
        active_count+=1;tr=np.asarray(s['training_ids'])-1
        fan=scalar_fan(x,cp,peak,worst,target,.4,returns[tr],inflation[tr],t)
        full,drops=explicit_delete_one(fan['hold']['logs'],fan['restore']['logs'],fan['hold']['bad'],fan['restore']['bad'],s['training_ids'],eta,30)
        ledger.require(full['action']==action,'fullfan/saved_action')
        for branch in ('hold','restore'):
            ledger.close(arm['qg_'+branch][i,t],fan[branch]['mean'],'fullfan/saved_Q_'+branch)
            accounting.maybe(arm['qbad_'+branch+'_count'][i,t],fan[branch]['count'],ledger,'fullfan/saved_count_'+branch)
        accounting.maybe(arm['forecast_n'][i,t],n,ledger,'fullfan/saved_N')
        if eta is not None:ledger.close(arm['predicted_score'][i,t],full['score'],'fullfan/saved_score')
        else:ledger.require(np.isnan(arm['predicted_score'][i,t]),'risk_first/score_sentinel')
        expected=dict(saved_qg_hold=float(arm['qg_hold'][i,t]),saved_qg_restore=float(arm['qg_restore'][i,t]),
            saved_bad_hold=fan['hold']['count'],saved_bad_restore=fan['restore']['count'],saved_score=None if eta is None else float(arm['predicted_score'][i,t]),
            recomputed_qg_hold=fan['hold']['mean'],recomputed_qg_restore=fan['restore']['mean'],recomputed_bad_hold=fan['hold']['count'],
            recomputed_bad_restore=fan['restore']['count'],recomputed_score=full['score'],recomputed_restore=full['restore'],
            delta_net_log_growth=full['delta_net_log_growth'],delta_breach_probability=full['delta_breach_probability'],deletion_n=n,
            deletion_action_flips=sum(d['action_flip'] for d in drops),self_deleted_action_flip=None)
        for j,sid in enumerate(s['training_ids']):
            mk=(s['slot_id'],sid);mr=members[mk];dr=deletions[mk];drop=drops[j];self_member=sid==s['scenario_id']
            check(mr,dict(slot_id=s['slot_id'],training_scenario_id=sid,member_index=j,is_actual_source_path=self_member,
                hold_total_log_growth=float(fan['hold']['logs'][j]),restore_total_log_growth=float(fan['restore']['logs'][j]),
                hold_ever_breached=bool(fan['hold']['bad'][j]),restore_ever_breached=bool(fan['restore']['bad'][j]),
                delta_net_log_growth=float(fan['restore']['logs'][j]-fan['hold']['logs'][j]),delta_breach_event=int(fan['restore']['bad'][j])-int(fan['hold']['bad'][j])),'member')
            drop_expect=dict(slot_id=s['slot_id'],training_scenario_id=sid,self_deleted=self_member,member_index=j,remaining_fan_n=n-1,
                drop_qg_hold=drop['hold_Q'],drop_qg_restore=drop['restore_Q'],drop_bad_hold=drop['hold_k'],drop_bad_restore=drop['restore_k'],
                drop_score=drop['score'],drop_restore=drop['restore'],drop_delta_net_log_growth=drop['delta_net_log_growth'],drop_delta_breach_probability=drop['delta_breach_probability'],
                hold_probability_change=drop['hold_k']/(n-1)-fan['hold']['count']/n,restore_probability_change=drop['restore_k']/(n-1)-fan['restore']['count']/n,
                paired_probability_change=drop['delta_breach_probability']-full['delta_breach_probability'],action_flip=drop['action_flip'])
            check(dr,drop_expect,'deletion');checked_members.add(mk);checked_deletions.add(mk)
            if self_member:expected['self_deleted_action_flip']=drop['action_flip']
        branches={}
        for branch in ('hold','restore'):
            b=replay_suffix(x,cp,peak,worst,target,returns[s['scenario_id']-1],inflation[s['scenario_id']-1],t,first_action=branch);branches[branch]=b;branch_periods+=len(b['periods'])
            for field,value in dict(total_log_growth=b['total_log_growth'],ever_breached=b['ever_breached'],history_mdd=b['terminal_mdd'],nominal_fees=b['nominal_fees'],terminal_real=b['terminal_real']).items():expected['B_'+branch+'_'+field]=value
            expected['forecast_realized_'+branch+'_log_residual']=b['total_log_growth']-fan[branch]['mean']
            expected['forecast_realized_'+branch+'_event_residual']=int(b['ever_breached'])-fan[branch]['count']/n
        chosen=branches[action]
        ledger.close(chosen['periods'][0]['posttrade_holdings'],arm['posttrade_holdings'][i,t],'chosen/current_posttrade')
        ledger.close(chosen['current_fee'],arm['fees'][i,t],'chosen/current_fee')
        adaptive=replay_suffix(x,cp,peak,worst,target,returns[s['scenario_id']-1],inflation[s['scenario_id']-1],t,fixed_actions=arm['actions'][i,t:])
        adaptive_periods+=len(adaptive['periods'])
        for j,state in enumerate(adaptive['states']):
            endpoint=t+j
            for field in ('holdings','cpi','real_wealth','real_peak','running_mdd','ever_breached'):
                if field=='ever_breached':accounting.maybe(arm[field][i,endpoint],state[field],ledger,'adaptive/state/'+field)
                else:ledger.close(arm[field][i,endpoint],state[field],'adaptive/state/'+field)
        for j,period in enumerate(adaptive['periods']):
            ledger.close(arm['fees'][i,t+j],period['fee'],'adaptive/fee');ledger.close(arm['posttrade_holdings'][i,t+j],period['posttrade_holdings'],'adaptive/posttrade')
        for field,key in (('total_log_growth','total_log_growth'),('ever_breached','ever_breached'),('history_mdd','terminal_mdd'),('nominal_fees','nominal_fees'),('terminal_real','terminal_real')):
            expected['adaptive_'+field]=adaptive[key]
            if field=='ever_breached':expected['continuation_gap_event']=int(adaptive[key])-int(chosen[key])
            else:expected['continuation_gap_'+field]=adaptive[key]-chosen[key]
        future=list(map(str,arm['actions'][i,t+1:]));b_future=[row['action'] for row in chosen['periods'][1:]]
        expected.update(future_decisions=29-t,future_action_mismatches=sum(a!=b for a,b in zip(future,b_future)),
                        adaptive_future_restores=future.count('restore'),B_future_restores=b_future.count('restore'))
        check(sr,expected,'state/independent_values')
        if s['year']==30:
            for field in ('total_log_growth','history_mdd','nominal_fees','terminal_real','event'):ledger.close(expected['continuation_gap_'+field],0.,'year30/zero_control/'+field,atol=2e-12,rtol=0)
        witnesses.append(witness)
    ledger.require(checked_members==set(members) and checked_deletions==set(deletions),'inventory/all_members_and_deletions_exact')
    expected_summary=independent_summary(tables['state_diagnostics'].to_dict('records'),tables['branch_members'].to_dict('records'),tables['deletions'].to_dict('records'))
    actual_summary={k:v for k,v in summary.items() if k!='interpretation'};compare_tree(actual_summary,expected_summary,ledger,'summary')
    ledger.require(isinstance(summary.get('interpretation'),list) and len(summary['interpretation'])==6,'summary/interpretation_scope')
    budget()
    return dict(status='passed',checks_passed=ledger.checks_passed,maximum_numeric_absolute_error=max(ledger.maximum_absolute_errors.values(),default=0.),
        maximum_absolute_errors=ledger.maximum_absolute_errors,state_inventory=witnesses,slots_checked=39,active_slots_checked=active_count,pending_slots_checked=39-active_count,
        branch_members_checked=len(checked_members),deletions_checked=len(checked_deletions),actual_B_periods_checked=branch_periods,adaptive_periods_checked=adaptive_periods,
        elapsed_seconds=time.monotonic()-started,scope_limits=plan['limitations']+['Fixed witness inventory excludes previously discussed failure IDs35/52/98/112; these diagnostics do not explain their root causes'])


def main():
    import pandas as pd
    if (OUT/'independent_audit.json').exists():raise RuntimeError('Never overwrite completed audit')
    if (OUT/'audit_failure_receipt.json').exists():raise RuntimeError('Preserved failed audit requires an explicit plan addendum')
    p=readj(PROTOCOL);plan=readj(AUDIT_PLAN);manifest=readj(OUT/'manifest.json')
    if p.get('execution_ready') is not True or manifest.get('status')!='complete':raise RuntimeError('Complete frozen diagnostic required before parsing market values')
    if sha(PROTOCOL)!=manifest['protocol_sha256'] or manifest['new_fits']!=0 or manifest['policy_actions_changed']!=0 or manifest['final_scenarios_loaded'] is not False:raise RuntimeError('Frozen diagnostic contract changed')
    for mapping in (p['execution_sources_sha256'],p['frozen_input_sha256'],plan['metadata_and_independent_helper_sha256']):check_mapping(ROOT,mapping)
    expected_inputs={**p['frozen_input_sha256'],str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    if manifest['sources']!=p['execution_sources_sha256'] or manifest['sources']!=manifest['sources_at_end'] or manifest['input_sha256']!=expected_inputs or manifest['input_sha256']!=manifest['input_sha256_at_end']:raise RuntimeError('Producer snapshots differ from frozen inputs/source')
    check_mapping(OUT,manifest['outputs_sha256'])
    admissions=readj(ROOT/'results/model_review/global_policy_admission_v1/admission.json');parents={};paths=[PROTOCOL,AUDIT_PLAN,Path(__file__),ROOT/'tests/test_audit_fan_continuation.py',OUT/'manifest.json']
    paths += [ROOT/name for name in p['execution_sources_sha256']]+[ROOT/name for name in p['frozen_input_sha256']]+[OUT/name for name in manifest['outputs_sha256']]
    paths += [ROOT/name for name in plan['metadata_and_independent_helper_sha256']]
    for folder in PARENT_NAMES:
        base=ROOT/'results/model_review'/folder;m=readj(base/'manifest.json');a=readj(base/'independent_audit.json')
        binding=a.get('parent_manifest_sha256',a.get('sources_and_artifacts_at_start',{}).get(str((base/'manifest.json').relative_to(ROOT))))
        if a.get('status')!='passed' or binding!=sha(base/'manifest.json') or a['sources_and_artifacts_at_start']!=a['sources_and_artifacts_at_end']:raise RuntimeError('Frozen parent independent audit binding changed')
        check_mapping(ROOT,a['sources_and_artifacts_at_start'])
        if m['sources']!=m['sources_at_end'] or m['input_sha256']!=m['input_sha256_at_end']:raise RuntimeError('Parent source/input snapshot differs')
        for values in (m['sources'],m['input_sha256']):check_mapping(ROOT,values)
        check_mapping(base,m['outputs_sha256'])
        parents[folder]=dict(groups=readj(base/'trace_groups.json')['actual'],choices=readj(base/'context_selections.json'),manifest=m)
        paths += [base/'manifest.json',base/'independent_audit.json',base/'actual_traces.npz']+[ROOT/name for name in a['sources_and_artifacts_at_start']]
    validate_inventory(p,plan,admissions,parents)
    before={str(path.relative_to(ROOT)):sha(path) for path in paths}
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    returns,inflation,ids=load_paths(ROOT/p['data']['path'],p['data']['scenario_ids'],allow_final=False)
    digest=hashlib.sha256()
    for array in (ids,returns,inflation):digest.update(np.ascontiguousarray(array).tobytes())
    if ids.tolist()!=list(range(1,121)) or manifest['scenario_ids_loaded']!=ids.tolist() or digest.hexdigest()!=manifest['development_arrays_sha256'] or any(parent['manifest']['development_arrays_sha256']!=digest.hexdigest() for parent in parents.values()):raise RuntimeError('Guarded original120 arrays changed')
    tables={name:read_table(OUT/(name+'.csv.gz')) for name in ('state_inventory','state_diagnostics','branch_members','deletions')}
    for name,count in manifest['row_counts'].items():
        if len(tables[name])!=count:raise RuntimeError('Manifest row count mismatch: '+name)
    try:
        for folder,parent in parents.items():parent['traces']=np.load(ROOT/'results/model_review'/folder/'actual_traces.npz',allow_pickle=False)
        result=audit_tables(p,plan,parents,returns,inflation,tables,readj(OUT/'summary.json'))
    finally:
        for parent in parents.values():
            if 'traces' in parent:parent['traces'].close()
    after={str(path.relative_to(ROOT)):sha(path) for path in paths}
    if before!=after:raise RuntimeError('Source/input/output changed during independent audit')
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(AUDIT_PLAN),auditor_sha256=sha(Path(__file__)),
        sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,scenario_ids_loaded=ids.tolist(),final_scenarios_loaded=False,
        interpretation='Independent fixed-state arithmetic/member/deletion/continuation audit, not calibration, refit stability, causal effect, new admission or policy improvement')
    writej(OUT/'independent_audit.json',result)
    print(json.dumps(dict(status='passed',checks=result['checks_passed'],active_slots=result['active_slots_checked'],pending_slots=result['pending_slots_checked'],members=result['branch_members_checked'],deletions=result['deletions_checked'],audit_sha256=sha(OUT/'independent_audit.json'))))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists() and not (OUT/'independent_audit.json').exists() and not (OUT/'audit_failure_receipt.json').exists():
            writej(OUT/'audit_failure_receipt.json',dict(status='failed_audit_attempt_preserved',completed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(Path(__file__)),audit_plan_sha256=sha(AUDIT_PLAN),error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),numeric_archive_modified=False))
        raise
