"""Independent V2b audit; shares the frozen independent V2 numerical oracle.

No production engine/forecast/selector/executor/metric is imported. Original
scalar math and adapted ledger/integration code come from the earlier
independent auditor; the9point fallback and actual-branch checks are new.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,math,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_rebuild as base
PROTOCOL=ROOT/'docs/state_budget_mitigation_protocol.json'
FREEZE=ROOT/'docs/state_budget_mitigation_source_freeze.json'
AUDIT_FREEZE=ROOT/'docs/state_budget_mitigation_audit_source_freeze.json'
PLAN=ROOT/'docs/state_budget_mitigation_audit_plan.json'
OUT=ROOT/'results/state_budget_mitigation_v1'
ATTEMPT=ROOT/'results/state_budget_mitigation_audit_attempt_v1'
TOL=base.TOL
Checks=base.Checks
SPECS=base.SPECS
FIN=(*base.FIN,'mitigation_trade_fraction','mitigation_nominal_fees')
sha,readj,writej=base.sha,base.readj,base.writej
verify_map,record_check=base.verify_map,base.record_check
select=base.select
scalar_forecasts=base.scalar_forecasts
PRIMARY_FIELDS=base.PHYSICAL|base.EXTRAS
MIT_FIELDS=set('mitigation_evaluated mitigation_trade mitigation_selected_index mitigation_selected_proposal mitigation_selected_q mitigation_worst_next_real_hold mitigation_worst_next_real_selected mitigation_worst_next_real_improvement actual_decision_source primary_failed_hold_score primary_failed_hold_pred_log_growth primary_failed_hold_fee primary_failed_hold_shortfall primary_failed_hold_posttrade primary_gate_reason primary_selected_score primary_pred_log_growth_selected'.split())
MIT_FIELDS|={'mitigation_candidate_'+k for k in ('q','fees','moved','worst_next_real','shortfall','proposal','worst_support_member','posttrade')}
MIT_FIELDS|={'mitigation_candidate_member_'+k for k in ('advantage','log_growth','gross_advantage','predicted_next_breach')}


def choose_worst(worst,fees,proposal,real):
    maximum=max(worst);tolerance=TOL*real
    if maximum<=worst[0]+tolerance:return 0
    band=[i for i in range(9) if worst[i]>=maximum-tolerance and worst[i]>worst[0]+tolerance]
    return min(band,key=lambda i:(i!=0,float(fees[i]),int(proposal[i]),i))


def scalar_mitigation(x,cpi,support,*,target,budget,growth,weights,peak,worst,L=.4,cost=.0025):
    x=np.asarray(x,float);target=np.asarray(target,float);support=np.asarray(support,float);growth=np.asarray(growth,float);weights=np.asarray(weights,float)
    wealth=math.fsum(x);real=wealth/cpi
    post=np.empty((9,3));post[0]=x;fee=np.zeros(9);moved=np.zeros(9);q=np.zeros(9);proposal=np.full(9,-1,int)
    for j,b in enumerate((1.,.75,.5,.25)):
        end,f,m=base.fee_endpoint(x,b*target+(1-b)*np.array([0.,0.,1.]),cost)
        for p,fraction in enumerate((.5,1.)):
            k=1+2*j+p;post[k]=x+fraction*(end-x);fee[k]=fraction*f;moved[k]=fraction*m;q[k]=fraction;proposal[k]=j
    values=np.array([[math.fsum(row[a]*post[k,a] for a in range(3))/cpi for row in support] for k in range(9)])
    minimum=values.min(axis=1);binding=values.argmin(axis=1);selected=choose_worst(minimum,fee,proposal,real)
    shortfall=np.maximum(budget-minimum,0.)
    hold_next=[math.fsum(x[a]*row[a] for a in range(3))/cpi for row in growth]
    member=np.empty((9,len(weights)));full=member.copy();gross=member.copy();bad=member.copy()
    for k in range(9):
        nextreal=[math.fsum(post[k,a]*row[a] for a in range(3))/cpi for row in growth]
        w=post[k]/math.fsum(post[k]);logratio=[math.log(v/h) for v,h in zip(nextreal,hold_next)]
        logfull=[math.log(v/real) for v in nextreal]
        grossratio=[math.log(math.fsum(w[a]*row[a] for a in range(3))/math.fsum((x[a]/wealth)*row[a] for a in range(3))) for row in growth]
        events=[max(worst,1-v/max(peak,v))-L>TOL for v in nextreal]
        for m,prob in enumerate(weights):
            member[k,m]=math.fsum(prob[j]*logratio[j] for j in range(len(growth)))
            full[k,m]=math.fsum(prob[j]*logfull[j] for j in range(len(growth)))
            gross[k,m]=math.fsum(prob[j]*grossratio[j] for j in range(len(growth)))
            bad[k,m]=math.fsum(prob[j]*events[j] for j in range(len(growth)))
    return dict(selected_index=selected,selected_proposal=int(proposal[selected]),selected_q=q[selected],posttrade_holdings=post[selected],
        fee=fee[selected],moved=moved[selected],trade=selected>0 and moved[selected]>TOL,worst_next_real_hold=minimum[0],
        worst_next_real_selected=minimum[selected],worst_next_real_improvement=minimum[selected]-minimum[0],
        worst_real_shortfall_hold=shortfall[0],worst_real_shortfall_selected=shortfall[selected],worst_real_shortfall_improvement=shortfall[0]-shortfall[selected],
        candidate_posttrade=post,candidate_q=q,candidate_fees=fee,candidate_moved=moved,candidate_proposal=proposal,candidate_worst_next_real=minimum,
        candidate_worst_support_member=binding,candidate_shortfall=shortfall,candidate_member_advantage=member,candidate_member_log_growth=full,
        candidate_member_gross_advantage=gross,candidate_member_predicted_next_breach=bad)


def blank(members):
    result=dict(mitigation_evaluated=False,mitigation_trade=False,mitigation_selected_index=-1,mitigation_selected_proposal=-1,
        mitigation_selected_q=np.nan,mitigation_worst_next_real_hold=np.nan,mitigation_worst_next_real_selected=np.nan,
        mitigation_worst_next_real_improvement=np.nan,actual_decision_source='primary',primary_failed_hold_score=np.nan,
        primary_failed_hold_pred_log_growth=np.nan,primary_failed_hold_fee=np.nan,primary_failed_hold_shortfall=np.nan,
        primary_failed_hold_posttrade=np.full(3,np.nan))
    for key in ('q','fees','moved','worst_next_real','shortfall'):result['mitigation_candidate_'+key]=np.full(9,np.nan)
    for key in ('proposal','worst_support_member'):result['mitigation_candidate_'+key]=np.full(9,-1,int)
    result['mitigation_candidate_posttrade']=np.full((9,3),np.nan)
    for key in ('advantage','log_growth','gross_advantage','predicted_next_breach'):result['mitigation_candidate_member_'+key]=np.full((9,members),np.nan)
    return result


def scalar_decision(x,cpi,peak,worst,support,growth,weights,*,target,release,rule,initial=False,L=.4,cost=.0025):
    result=base.scalar_decision(x,cpi,peak,worst,support,growth,weights,target=target,release=release,rule=rule,initial=initial,L=L,cost=cost)
    result.update(blank(len(weights)));result.update(primary_gate_reason=result['gate_reason'],primary_selected_score=result['selected_score'],primary_pred_log_growth_selected=result['pred_log_growth_selected'])
    if initial or result['status']!='no_feasible_action':return result
    result.update(primary_failed_hold_score=result['selected_score'],primary_failed_hold_pred_log_growth=result['pred_log_growth_selected'],
        primary_failed_hold_fee=result['fee'],primary_failed_hold_shortfall=result['worst_real_shortfall_selected'],
        primary_failed_hold_posttrade=result['posttrade_holdings'].copy(),actual_decision_source='mitigation')
    m=scalar_mitigation(x,cpi,support,target=target,budget=result['budget_floor'],growth=growth,weights=weights,peak=peak,worst=worst,L=L,cost=cost)
    s=m['selected_index'];adv=m['candidate_member_advantage'];full=m['candidate_member_log_growth'];gross=m['candidate_member_gross_advantage']
    result.update(mitigation_evaluated=True,mitigation_trade=m['trade'],mitigation_selected_index=s,mitigation_selected_proposal=m['selected_proposal'],
        mitigation_selected_q=m['selected_q'],mitigation_worst_next_real_hold=m['worst_next_real_hold'],mitigation_worst_next_real_selected=m['worst_next_real_selected'],
        mitigation_worst_next_real_improvement=m['worst_next_real_improvement'])
    for key in ('posttrade','q','fees','moved','proposal','worst_next_real','worst_support_member','shortfall'):result['mitigation_candidate_'+key]=m['candidate_'+key]
    for key in ('advantage','log_growth','gross_advantage','predicted_next_breach'):result['mitigation_candidate_member_'+key]=m['candidate_member_'+key]
    score=adv[s,0] if rule=='uniform' else min(adv[s])
    result.update(posttrade_holdings=m['posttrade_holdings'],fee=m['fee'],moved=m['moved'],action='mitigate' if m['trade'] else 'hold',
        gate_reason='risk_mitigation_then_review',selected_score=score,pred_log_growth_selected=full[s,0],uniform_predicted_advantage=adv[s,0],
        conditional_predicted_advantage=adv[s,1] if len(weights)>1 else np.nan,min_predicted_advantage=min(adv[s]),
        current_log_fee=math.log1p(-m['fee']/math.fsum(x)),gross_log_advantage_uniform=gross[s,0],gross_log_advantage_min=min(gross[s]),
        worst_real_shortfall_selected=m['worst_real_shortfall_selected'],worst_real_shortfall_improvement=m['worst_real_shortfall_improvement'],growth_sacrifice=score<-TOL)
    return result


def decision_check(observed,expected,c,label,*,x,cp,support,target):
    key='mitigation_candidate_worst_support_member';a,b=np.asarray(observed[key],int),np.asarray(expected[key],int)
    c.check(a.shape==b.shape,label+'/'+key+'/shape')
    for k in range(9):
        if a[k]==b[k]:continue
        post=expected['mitigation_candidate_posttrade'][k]
        values=np.asarray(support)@post/cp;valid=0<=a[k]<len(values) and abs(values[a[k]]-min(values))<=3e-10+5e-13*math.fsum(x)/cp
        c.check(valid,label+'/'+key+f'/{k}/equivalent binding')
        c.numeric_binding_ties.append(dict(field=label+'/'+key,index=k,observed=int(a[k]),scalar=int(b[k])))
    base.decision_check({k:v for k,v in observed.items() if k!=key},{k:v for k,v in expected.items() if k!=key},c,label,x=x,cp=cp,support=support,target=target)


def path_metrics(arm,i):
    result=base.path_metrics(arm,i);h=len(arm['fees'][i]);reason=arm.get('gate_reason',np.full_like(arm['actions'],'',dtype='U64'))[i]
    trade=arm['trades'][i];mit=(reason=='risk_mitigation_then_review')
    result.update(protection_trade_fraction=np.count_nonzero((reason=='risk_budget_protection')|(mit&trade))/h,
        mitigation_trade_fraction=np.count_nonzero(mit&trade)/h,mitigation_nominal_fees=math.fsum(arm['fees'][i][mit]))
    return result


def summary(rows):
    rows=list(rows);result=base.summary(rows);fin=[r for r in rows if r['assigned']]
    result.update(mean_mitigation_trade_fraction=math.fsum(r.get('mitigation_trade_fraction',0.) for r in fin)/len(fin) if fin else None,
        mean_mitigation_fees=math.fsum(r.get('mitigation_nominal_fees',0.) for r in fin)/len(fin) if fin else None)
    return result


def primary_view(arm):
    """Diagnostic primary decision at the actual pre-state; never a physical replay."""
    view={k:v for k,v in arm.items() if k in PRIMARY_FIELDS};mit=arm['mitigation_evaluated']
    replacements={'posttrade_holdings':arm['primary_failed_hold_posttrade'],'fees':arm['primary_failed_hold_fee'],
        'moved':np.zeros_like(arm['moved']),'actions':np.full_like(arm['actions'],'hold'),'gate_reason':arm['primary_gate_reason'],
        'selected_score':arm['primary_selected_score'],'pred_log_growth_selected':arm['primary_pred_log_growth_selected'],
        'worst_real_shortfall_selected':arm['primary_failed_hold_shortfall'],'worst_real_shortfall_improvement':np.zeros_like(arm['fees']),
        'target_execution':np.zeros_like(arm['target_execution']),'growth_sacrifice':np.zeros_like(arm['growth_sacrifice'])}
    for key in ('uniform_predicted_advantage','conditional_predicted_advantage','min_predicted_advantage','current_log_fee','gross_log_advantage_uniform','gross_log_advantage_min'):replacements[key]=np.zeros_like(arm[key])
    for key,values in replacements.items():
        view[key]=arm[key].copy();view[key][mit]=values[mit]
    return view


def schema(arm,group,n,train_n,c):
    if group['kind']=='reference':base.schema(arm,group,n,train_n,c);return
    omitted={k for k in PRIMARY_FIELDS|MIT_FIELDS if k.startswith(('candidate_','mitigation_candidate_')) or k in ('training_previous_state','training_real_gross')}
    expected=(PRIMARY_FIELDS|MIT_FIELDS)-omitted if group['kind']=='inner_candidate' else PRIMARY_FIELDS|MIT_FIELDS
    c.check(set(arm)==expected,group['group_id']+'/complete V2b inventory')
    c.check(set(group['array_names'])==set(arm) and len(group['array_names'])==len(arm),group['group_id']+'/full metadata inventory')
    c.check(set(group['omitted_diagnostic_arrays'])==(omitted if group['kind']=='inner_candidate' else set()),group['group_id']+'/deliberately omitted inventory')
    original={k:v for k,v in arm.items() if k in PRIMARY_FIELDS};meta=dict(group,array_names=list(original),omitted_diagnostic_arrays=[k for k in group['omitted_diagnostic_arrays'] if k in PRIMARY_FIELDS])
    base.schema(original,meta,n,train_n,c)
    for key in MIT_FIELDS&set(arm):
        shape=(n,30)
        if key=='primary_failed_hold_posttrade':shape=(n,30,3)
        elif key=='mitigation_candidate_posttrade':shape=(n,30,9,3)
        elif key.startswith('mitigation_candidate_member_'):shape=(n,30,9,5)
        elif key.startswith('mitigation_candidate_'):shape=(n,30,9)
        c.check(arm[key].shape==shape,group['group_id']+'/'+key+'/shape')
        kind='b' if key in ('mitigation_evaluated','mitigation_trade') else 'i' if key in ('mitigation_selected_index','mitigation_selected_proposal','mitigation_candidate_proposal','mitigation_candidate_worst_support_member') else 'U' if key in ('actual_decision_source','primary_gate_reason') else 'f'
        c.check(arm[key].dtype.kind==kind,group['group_id']+'/'+key+'/dtype')


def model_gates(arm,train,trainpi,ids,group,c):
    view=primary_view(arm);base.model_gates(view,train,trainpi,ids,group,c)
    label=group['group_id'];n,h=arm['fees'].shape;active=arm['gate_evaluated'];mit=arm['mitigation_evaluated'];initial=np.zeros((n,h),bool);initial[:,0]=True
    expected=active&~initial&(arm['selected_index']<0)
    c.arr(mit,expected,label+'/only first funded primary failure')
    c.check(np.all(mit.sum(axis=1)<=1),label+'/at most one mitigation per started account')
    c.arr(arm['actual_decision_source'],np.where(~active,'pending',np.where(mit,'mitigation','primary')),label+'/actual branch identity')
    c.arr(arm['primary_gate_reason'][active],view['gate_reason'][active],label+'/original primary reason')
    c.arr(arm['primary_selected_score'][active],view['selected_score'][active],label+'/original primary score')
    c.arr(arm['primary_pred_log_growth_selected'][active],view['pred_log_growth_selected'][active],label+'/original primary growth')
    c.arr(arm['primary_gate_reason'][~active],np.full(np.count_nonzero(~active),''),label+'/pending primary reason empty')
    for key in ('primary_selected_score','primary_pred_log_growth_selected'):
        c.check(np.isnan(arm[key][~active]).all(),label+'/'+key+'/pending NULL')
    c.check(not arm['support_feasible'][mit].any() and arm['guard_triggered'][mit].all(),label+'/no new floor certificate')
    c.check(not arm['automatic_service'][mit].any() and arm['pending_at_decision'][mit].all() and not arm['was_pending_input'][mit].any(),label+'/mitigate before ordinary-service suspension')
    c.arr(arm['mitigation_trade'],mit&(arm['mitigation_selected_index']>0)&(arm['moved']>TOL),label+'/actual mitigation trade')
    for key in MIT_FIELDS:
        if key not in arm or key in ('actual_decision_source','primary_gate_reason','primary_selected_score','primary_pred_log_growth_selected','mitigation_evaluated','mitigation_trade'):continue
        values=arm[key][~mit]
        if arm[key].dtype.kind=='f':c.check(np.isnan(values).all(),label+'/'+key+'/nonmitigation NULL')
        elif arm[key].dtype.kind=='i':c.check(np.all(values==-1),label+'/'+key+'/nonmitigation sentinel')
    for i,t in np.argwhere(mit):
        cp=float(arm['cpi'][i,t]);x=arm['holdings'][i,t];target=group['target'];real=float(arm['real_wealth'][i,t]);spec=group['candidate']
        support=(1+train[:,t])/(1+trainpi[:,t,None]);trainprev=np.column_stack(((1+train[:,t-1])/(1+trainpi[:,t-1,None])-1,trainpi[:,t-1]))
        actualprev=arm['actual_previous_state'][i,t];weights=base.scalar_forecasts(ids,trainprev,actualprev)
        result=scalar_decision(x,cp,arm['real_peak'][i,t],arm['running_mdd'][i,t],support,support,weights,target=target,release=spec['release_fraction'],rule=spec['forecast_rule'])
        c.check(result['selected_index']==-1 and result['mitigation_evaluated'],label+f'/{i}/{t}/independent primary failure')
        # Thin inner groups retain the executed/scalar fields, not all menus.
        observed={};expected_result={}
        for key,value in result.items():
            source={'fee':'fees','action':'actions'}.get(key,key)
            if key=='status':actual='no_feasible_action'
            elif key=='real_wealth':actual=arm['real_wealth'][i,t]
            elif source in arm:actual=arm[source][i,t]
            else:continue
            observed[key]=actual;expected_result[key]=value
        if 'mitigation_candidate_worst_support_member' in expected_result:
            decision_check(observed,expected_result,c,label+f'/{i}/{t}/all_mitigation_scalar',x=x,cp=cp,support=support,target=target)
        else:base.decision_check(observed,expected_result,c,label+f'/{i}/{t}/thin_mitigation_scalar',x=x,cp=cp,support=support,target=target)
        chosen=int(result['mitigation_selected_index']);improvement=result['mitigation_worst_next_real_improvement']
        if chosen>0:c.check(improvement>TOL*real,label+f'/{i}/{t}/strict afterfee worstwealth improvement')
        else:c.check(result['fee']==0 and result['moved']==0,label+f'/{i}/{t}/no-improvement hold no fee')
        c.mitigation_checks.append(dict(group_id=label,scenario_id=group['scenario_ids'][i],original_year=int(t+1),selected_index=chosen,
            trade=bool(result['mitigation_trade']),nominal_fee=float(result['fee']),worst_real_improvement=float(improvement),
            residual_real_shortfall=float(result['worst_real_shortfall_selected']),predicted_paired_score=float(result['selected_score']),
            support_certificate=False))


def source_review():
    import ast
    text=(ROOT/'scripts/run_state_budget_mitigation.py').read_text();tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_paths']
    if len(calls)!=1 or ast.unparse(calls[0].args[1])!='range(1, 121)' or {k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}.get('allow_final') is not False:raise AssertionError('Only fixed development loader allowed')
    return dict(status='passed_static_source_review',original_primary_source_sha256=sha(ROOT/'research/experimental_execution/state_budget_execution.py'),
        information='Current actual returns advance after choice; forecast/support use own-training arrays; first active funded failure only.',
        limitation='Frozen-source/API review and saved prefix/branch checks, not a formal proof against arbitrary future edits')


# Shared independent vector ledger, adapted so only prior pending forbids a trade.
def physical(arm,r,pi,target,group,c):
    label=group['group_id'];x=arm['holdings'];post=arm['posttrade_holdings'];n,h=len(r),r.shape[1]
    c.check(x.shape==(n,h+1,3) and post.shape==(n,h,3),label+'/complete original-H account')
    c.arr(x[:,0],np.tile([0.,0.,100.],(n,1)),label+'/initial100bills')
    c.arr(x[:,1:],post*(1+r),label+'/after-action actual joint nominal returns')
    cp=np.column_stack((np.ones(n),np.cumprod(1+pi,axis=1)))
    c.arr(arm['cpi'],cp,label+'/original CPI')
    wealth=x.sum(axis=2);real=wealth/cp;peak=np.maximum.accumulate(real,axis=1);dd=np.maximum(1-real/peak,0);mdd=np.maximum.accumulate(dd,axis=1)
    for key,value in [('nominal_wealth',wealth),('real_wealth',real),('real_peak',peak),('current_drawdown',dd),('running_mdd',mdd),('pretrade_wealth',wealth[:,:-1])]:c.arr(arm[key],value,label+'/'+key)
    c.arr(arm['weights'],x/wealth[:,:,None],label+'/weights');c.arr(arm['end_weights'],arm['weights'][:,1:],label+'/endweights')
    moved=np.maximum(post-x[:,:-1],0).sum(axis=2)
    c.arr(arm['moved'],moved,label+'/destination amount once');c.arr(arm['fees'],.0025*moved,label+'/implicit fee equation once')
    c.arr(post.sum(axis=2)+arm['fees'],wealth[:,:-1],label+'/self finance')
    c.arr(arm['fee_fraction'],arm['fees']/wealth[:,:-1],label+'/fee fraction')
    c.arr(arm['moved_fraction'],moved/wealth[:,:-1],label+'/moved fraction')
    c.arr(arm['trades'],moved>TOL,label+'/physical trade not target label')
    c.arr(arm['pretrade_tv_drift'],.5*np.abs(arm['weights'][:,:-1]-target).sum(axis=2),label+'/pretrade drift')
    c.arr(arm['end_year_tv_drift'],.5*np.abs(arm['weights'][:,1:]-target).sum(axis=2),label+'/end drift')
    bad=np.maximum.accumulate(mdd-.4>TOL,axis=1);c.arr(arm['ever_breached'],bad,label+'/absorbing original breach')
    first=np.where(bad.any(axis=1),bad.argmax(axis=1),-1);c.arr(arm['first_crossing'],first,label+'/first crossing')
    c.arr(arm['triggered'],first>=0,label+'/triggered')
    recovered=np.zeros(n,bool);recovery=np.full(n,-1,int)
    for i,tau in enumerate(first):
        if tau>=0:
            for key,value in [('drawdown_at_trigger',mdd[i,tau]),('peak_at_trigger',peak[i,tau]),('real_wealth_at_trigger',real[i,tau])]:c.value(arm[key][i],value,label+f'/{i}/'+key)
            end=next((e for e in range(tau+1,h+1) if real[i,e]>=peak[i,tau]),-1)
            recovery[i]=end;recovered[i]=end>=0
        else:
            for key in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):c.check(np.isnan(arm[key][i]),label+f'/{i}/'+key+' NULL')
    for key,value in [('recovery_endpoint',recovery),('recovered',recovered),('recovery_time',np.where(recovered,recovery-first,-1)),
            ('right_censored',(first>=0)&~recovered),('available_followup',np.where(first>=0,h-first,-1)),
            ('observed_followup',np.where(recovered,recovery-first,np.where(first>=0,h-first,-1)))]:c.arr(arm[key],value,label+'/'+key)
    severity=np.array([max(mdd[i,-1]-mdd[i,tau],0) if tau>=0 else np.nan for i,tau in enumerate(first)])
    c.arr(arm['added_severity'],severity,label+'/added severity')
    terminal=np.where(bad[:,-1],mdd[:,-1]-.4,0.) if group['kind']!='reference' else np.maximum(mdd[:,-1]-.4,0)
    c.arr(arm['terminal_excess'],terminal,label+'/terminal excess convention')
    if group['kind']=='reference':
        pending=bad[:,:-1] if group['policy']=='same_target_B' else np.zeros((n,h),bool)
        c.arr(arm['pending_at_decision'],pending,label+'/reference original rule')
        c.arr(arm['target_execution'],~pending,label+'/reference annual/pending executions')
        c.arr(arm['actions'],np.where(pending,'hold','restore'),label+'/reference actions')
        c.arr(post[pending],x[:,:-1][pending],label+'/reference pending hold')
        restore=~pending;c.arr(post[restore]/post[restore].sum(axis=1)[:,None],np.broadcast_to(target,post[restore].shape),label+'/reference target',atol=3e-12)
        return
    guard=arm['guard_pending'];new=arm['guard_triggered'];before=guard[:,:-1]|bad[:,:-1]
    c.arr(guard[:,1:],np.maximum.accumulate(new,axis=1),label+'/absorbing guard pending')
    c.arr(arm['was_pending_input'],before,label+'/pre-action pending')
    c.arr(arm['was_guard_pending_input'],guard[:,:-1],label+'/pre-action guard')
    c.arr(arm['pending_at_decision'],before|new,label+'/current guard included pending')
    c.arr(arm['guard_pending_at_decision'],guard[:,:-1]|new,label+'/current guard flag')
    c.arr(arm['automatic_service'],~(before|new),label+'/automatic service')
    c.arr(arm['gate_evaluated'],~before,label+'/only active evaluated')
    c.arr(new,(arm['selected_index']<0)&~before,label+'/new guard only evaluated no feasible')
    c.arr(arm['first_guard_trigger'],np.where(new.any(axis=1),new.argmax(axis=1)+1,-1),label+'/first guard')
    c.arr(post[before],x[:,:-1][before],label+'/pending exact physical hold')
    c.arr(arm['actions'][before],np.full(np.count_nonzero(before),'hold'),label+'/pending action')
    c.check(not arm['target_execution'][before].any(),label+'/pending no target execution')
    c.arr(arm['targets'],target,label+'/unchanged strategic target');c.arr(arm['loss_tolerances'],[.4],label+'/original loss tolerance')
    gross=(1+r)/(1+pi[:,:,None]);c.arr(arm['realized_real_gross'],gross,label+'/after-action realized real gross')
    previous=np.full((n,h,4),np.nan);previous[:,1:,:3]=gross[:,:-1]-1;previous[:,1:,3]=pi[:,:-1]
    c.arr(arm['actual_previous_state'],previous,label+'/actual prefix t-1 only')
    inactive=before
    inactive_float='floor budget_floor selected_score pred_log_growth_hold pred_log_growth_selected uniform_predicted_advantage conditional_predicted_advantage min_predicted_advantage current_log_fee gross_log_advantage_uniform gross_log_advantage_min worst_real_shortfall_hold worst_real_shortfall_selected worst_real_shortfall_improvement selected_lambda'.split()
    for key in inactive_float:c.check(np.isnan(arm[key][inactive]).all(),label+'/'+key+'/pending NULL')
    for key in ('selected_index','selected_proposal'):c.check(np.all(arm[key][inactive]==-1),label+'/'+key+'/pending sentinel')
    for key in ('forecast_n','forecast_member_count'):c.check(np.all(arm[key][inactive]==0),label+'/'+key+'/pending zero')
    for key in arm:
        if key.startswith('candidate_') or key.startswith('interval_'):
            values=arm[key][inactive]
            if arm[key].dtype.kind=='f':c.check(np.isnan(values).all(),label+'/'+key+'/pending NULL')
            elif arm[key].dtype.kind=='i':c.check(np.all(values==-1),label+'/'+key+'/pending sentinel')
            elif arm[key].dtype.kind=='b':c.check(not values.any(),label+'/'+key+'/pending false')
    for key in ('support_feasible','hold_feasible','growth_sacrifice'):c.check(not arm[key][inactive].any(),label+'/'+key+'/pending false')
    expectedreason=np.where(guard[:,:-1],'guard_pending_hold','already_breached_pending_hold')
    c.arr(arm['gate_reason'][inactive],expectedreason[inactive],label+'/pending reason')

# Adapted independent V2 artifact integration, unchanged folds/selector.
def preflight():
    if ATTEMPT.exists() or (OUT/'independent_audit.json').exists():raise RuntimeError('Never overwrite or silently retry an independent audit')
    p,f,m,a=readj(PROTOCOL),readj(FREEZE),readj(OUT/'manifest.json'),readj(AUDIT_FREEZE)
    if (p['development_ids']!=list(range(1,121)) or p['allow_final'] is not False or not f['execution_ready']
        or f['protocol_sha256']!=sha(PROTOCOL)):raise RuntimeError('Frozen development-only protocol required before values')
    closure={**f['source_sha256'],**f['input_sha256']}
    if (m['status']!='complete' or m['protocol_sha256']!=sha(PROTOCOL) or m['source_freeze_sha256']!=sha(FREEZE)
        or m['sources_and_inputs_at_start']!=m['sources_and_inputs_at_end'] or m['sources_and_inputs_at_start']!=closure
        or m['final_scenarios_loaded'] is not False or m['new_final_reads']!=0 or m['development_ids_loaded']!=list(range(1,121))
        or (m['path_rows'],m['summary_rows'],m['trace_groups'])!=(6360,8,240)):
        raise RuntimeError('Complete producer with unchanged development-only identities required')
    if not a.get('execution_ready') or a.get('plan_sha256')!=sha(PLAN):raise RuntimeError('Separately frozen independent auditor required')
    expected_outputs={'path_metrics.csv.gz','summary.csv.gz','selections.json','trace_groups.json','initial_witnesses.json','traces.npz'}
    if set(m['outputs_sha256'])!=expected_outputs:raise RuntimeError('All six declared producer outputs must be hash bound')
    must={'scripts/audit_state_budget_mitigation.py','tests/test_audit_state_budget_mitigation.py','docs/state_budget_mitigation_audit_plan.json','src/safeinvest/data.py','scripts/audit_state_budget_rebuild.py'}
    if not must<=set(a['source_sha256']):raise RuntimeError('Complete independent source/test/plan/loader freeze required')
    verify_map(a['source_sha256']);verify_map(closure);verify_map(m['outputs_sha256'],OUT)
    if str(PLAN.relative_to(ROOT)) not in f['input_sha256'] or f['input_sha256'][str(PLAN.relative_to(ROOT))]!=sha(PLAN):raise RuntimeError('Producer must bind the predeclared independent audit plan')
    names=set(closure)|set(a['source_sha256'])|{str(x.relative_to(ROOT)) for x in (PLAN,PROTOCOL,FREEZE,AUDIT_FREEZE,OUT/'manifest.json')}
    names|={str((OUT/k).relative_to(ROOT)) for k in m['outputs_sha256']}
    return p,m,{name:sha(ROOT/name) for name in sorted(names)}

def audit_values(p,m,before,state):
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    state.update(phase='loading_development_only',market_value_access_started=True);writej(ATTEMPT/'attempt.json',state)
    r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
    c=Checks();c.mitigation_checks=[];c.check(ids.tolist()==list(range(1,121)),'exact development identities')
    jobs=readj(ROOT/p['job_protocol'])['jobs'];jobmap={j['job_id']:j for j in jobs};fits=readj(ROOT/p['static_refits'])
    c.check(len(jobmap)==30 and set(j['parent_context_id'] for j in jobs)==set(range(6)),'30jobs6contexts')
    for job in jobs:
        tr,ev=job['training_ids'],job['evaluation_ids'];c.check(len(tr)==len(set(tr)) and len(ev)==len(set(ev)),'unique complete job IDs')
        c.check(set(tr)|set(ev)<=set(range(1,121)),'all job IDs development only')
        c.check(fits[job['job_id']]['refit']['training_ids']==tr,'own fitted Q1 training IDs')
        if job['stage']=='inner' or job['parent_context_id']:
            c.check(not set(tr)&set(ev),'held-out actual excluded from support and fit')
    for context in range(6):
        js=[j for j in jobs if j['parent_context_id']==context];inner=[j for j in js if j['stage']=='inner'];refit=[j for j in js if j['stage']=='refit']
        c.check(len(inner)==4 and len(refit)==1,'four inner folds and own refit')
        validation=[sid for j in inner for sid in j['evaluation_ids']]
        c.check(sorted(validation)==sorted(refit[0]['training_ids']),'inner OOF complete once on own training')
        for job in inner:c.check(set(job['training_ids'])|set(job['evaluation_ids'])==set(refit[0]['training_ids']),'inner exact partition of outer training')
    groups=readj(OUT/'trace_groups.json');selections=readj(OUT/'selections.json');initial=readj(OUT/'initial_witnesses.json')
    frame=pd.read_csv(OUT/'path_metrics.csv.gz',float_precision='round_trip');saved_summary=pd.read_csv(OUT/'summary.csv.gz',float_precision='round_trip')
    c.check(len(groups)==240 and len({g['group_id'] for g in groups})==240 and len(frame)==6360 and len(saved_summary)==8,'complete fixed artifact denominators')
    for job in jobs:
        own=[g for g in groups if g['job_id']==job['job_id']];references=[g for g in own if g['kind']=='reference']
        c.check(len(references)==3 and {g['policy'] for g in references}=={'same_target_B','same_target_annual','exact_thirds_annual'},job['job_id']+'/all3references')
        if job['stage']=='inner':c.check(len(own)==9 and {g['policy'] for g in own if g['kind']=='inner_candidate'}=={s['candidate_id'] for s in SPECS},job['job_id']+'/all6inner candidates')
        else:c.check(len(own)==4 and len([g for g in own if g['kind']==('outer_selected' if job['parent_context_id'] else 'full_selected')])==1,job['job_id']+'/selected group retained even Review')
    c.check(not frame.duplicated(['job_id','policy','scope','scenario_id']).any(),'unique full path records')
    counts=dict(groups=240,physical_accounts=0,periods=0,endpoints=0,initial_review_rows=0,initial_scalar_decisions=0,nominal_decision_slots=54,active_scalar_decisions=0,pending_slots=0,unfunded_slots=0)
    derived=[];witnesses=[];initial_map={(z['job_id'],z['candidate_id'],z['scope']):z for z in initial};seen_initial=set();inventories={}
    with np.load(OUT/'traces.npz',allow_pickle=False) as archive:
      expected_archive=set()
      for group in groups:
        gid=group['group_id'];job=jobmap[group['job_id']];context=job['parent_context_id'];kind=group['kind'];eval_ids=job['evaluation_ids'];train_ids=job['training_ids']
        c.check(group['scenario_ids']==eval_ids and group['training_ids']==train_ids,gid+'/original train/eval order')
        fit=fits[job['job_id']];case=fit['q2_case'];base=np.array(case['product']['weights'],float) if case['in_balanced_case'] else None
        target=np.ones(3)/3 if group['policy']=='exact_thirds_annual' else base
        c.value(group['target'],target.tolist() if target is not None else None,gid+'/own fit target') if target is None else c.arr(group['target'],target,gid+'/own fit target')
        scope=('inner_reference' if job['stage']=='inner' else 'outer_reference' if context else 'full_training_reference') if kind=='reference' else kind
        policy=group['policy'] if kind not in ('outer_selected','full_selected') else 'state_budget_learner'
        rows=frame.loc[frame.job_id.eq(job['job_id'])&frame.scope.eq(scope)&frame.policy.eq(policy)]
        c.check(len(rows)==len(eval_ids) and set(rows.scenario_id)==set(eval_ids),gid+'/complete cohort paths')
        arm={name:archive[gid+'__'+name] for name in group['array_names']};expected_archive|={gid+'__'+name for name in group['array_names']}
        tr=np.array(train_ids)-1;ev=np.array(eval_ids)-1
        if arm:
            c.check(rows.assigned.all(),gid+'/all-funded same initial state');schema(arm,group,len(ev),len(tr),c)
            physical(arm,r[ev],pi[ev],target,group,c)
            if kind!='reference':model_gates(arm,r[tr],pi[tr],train_ids,group,c)
            inventories[gid]={k:dict(dtype=str(v.dtype),shape=list(v.shape)) for k,v in arm.items()}
            counts['physical_accounts']+=len(ev);counts['periods']+=30*len(ev);counts['endpoints']+=31*len(ev)
        else:
            c.check(not rows.assigned.any() and rows[list(FIN)].isna().all().all(),gid+'/preinvestment Review complete NULL')
            c.check(group['array_names']==[] and group['omitted_diagnostic_arrays']==[],gid+'/no physical review tree')
            c.check(set(rows.review_reason)=={group['status']},gid+'/Review reason identity');counts['initial_review_rows']+=len(ev)
        for i,sid in enumerate(eval_ids):
            actual=rows.loc[rows.scenario_id.eq(sid)].iloc[0]
            row=dict(job_id=job['job_id'],policy=policy,scope=scope,context_id=context,scenario_id=sid,assigned=bool(arm))
            if not arm:row['review_reason']=group['status']
            row.update(path_metrics(arm,i) if arm else dict.fromkeys(FIN));derived.append(row)
            for key,value in row.items():c.value(actual[key],value,gid+f'/{sid}/path/'+key)
        if kind!='reference' and group.get('candidate') is not None and target is not None:
            spec=group['candidate'];c.check(spec in SPECS,gid+'/fixed candidate tuple')
            key=(job['job_id'],spec['candidate_id'],kind);c.check(key in initial_map,gid+'/initial witness exists');seen_initial.add(key)
            gross=(1+r[tr])/(1+pi[tr,:,None]);initial_expected=scalar_decision([0.,0.,100.],1.,100.,0.,gross[:,0],gross[:,0],np.full((1,len(tr)),1/len(tr)),
                target=target,release=spec['release_fraction'],rule=spec['forecast_rule'],initial=True)
            decision_check(initial_map[key]['witness'],initial_expected,c,gid+'/initial_scalar',x=np.array([0.,0.,100.]),cp=1.,support=gross[:,0],target=target)
            c.check(initial_map[key]['status']==('funded' if initial_expected['selected_index']>=0 else 'preinvestment_review'),gid+'/uniform initial admission')
            c.check(bool(arm)==(initial_expected['selected_index']>=0),gid+'/initial exact admission vs physical availability');counts['initial_scalar_decisions']+=1
        if kind in ('outer_selected','full_selected'):
            ordered=sorted(eval_ids);selected_ids=(ordered[0],ordered[len(ordered)//2],ordered[-1])
            for sid in selected_ids:
              i=eval_ids.index(sid)
              for year in (2,16,30):
                t=year-1;w=dict(group_id=gid,context_id=context,scenario_id=sid,original_year=year,replacement=None)
                if not arm:w['status']='skipped_preinvestment_Review';counts['unfunded_slots']+=1
                elif arm['was_pending_input'][i,t]:w['status']='skipped_prior_pending';counts['pending_slots']+=1
                else:
                    trainprev=np.column_stack(((1+r[tr,t-1])/(1+pi[tr,t-1,None])-1,pi[tr,t-1]));actualprev=np.r_[(1+r[sid-1,t-1])/(1+pi[sid-1,t-1])-1,pi[sid-1,t-1]]
                    weights=scalar_forecasts(train_ids,trainprev,actualprev);gross=(1+r[tr,t])/(1+pi[tr,t,None]);spec=group['candidate'];x=arm['holdings'][i,t];cp=arm['cpi'][i,t]
                    expected=scalar_decision(x,cp,arm['real_peak'][i,t],arm['running_mdd'][i,t],gross,gross,weights,target=target,release=spec['release_fraction'],rule=spec['forecast_rule'])
                    observed={}
                    for key in expected:
                        if key=='status':value='selected' if arm['selected_index'][i,t]>=0 else 'no_feasible_action'
                        elif key=='fee':value=arm['fees'][i,t]
                        elif key=='action':value=arm['actions'][i,t]
                        elif key=='real_wealth':value=arm['real_wealth'][i,t]
                        else:value=arm[key][i,t]
                        observed[key]=value
                    decision_check(observed,expected,c,gid+f'/{sid}/{year}/scalar',x=x,cp=cp,support=gross,target=target)
                    w.update(status='passed_independent13point_decision',selected_index=int(expected['selected_index']),action=expected['action'],
                        score=float(expected['selected_score']),forecast_members=int(len(weights)),training_n=len(tr),
                        forecast_weights_sha256=hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest())
                    counts['active_scalar_decisions']+=1
                witnesses.append(w)
      c.check(set(archive.files)==expected_archive,'complete NPZ inventory no missing or hidden arrays')
    c.check(seen_initial==set(initial_map) and len(initial_map)==len(initial)==m['candidate_policy_runs'],'all initial witnesses unique and independently audited')
    c.check(len(witnesses)==54 and counts['active_scalar_decisions']+counts['pending_slots']+counts['unfunded_slots']==54,'all54nominal slots no replacements')
    c.check(len(derived)==6360 and sorted(r['scenario_id'] for r in derived if r['scope']=='outer_selected')==list(range(1,121)),'complete all-original once OOF')
    for context in range(6):
        rows=[r for r in derived if r['context_id']==context]
        inner={s['candidate_id']:[r for r in rows if r['scope']=='inner_candidate' and r['policy']==s['candidate_id']] for s in SPECS}
        B=[r for r in rows if r['scope']=='inner_reference' and r['policy']=='same_target_B']
        result=select(inner,B,p['service_alpha']);observed=selections[str(context)]
        for key,value in result.items():record_check(observed[key],value,c,f'context{context}/selection/'+key)
        chosenrows=[r for r in rows if r['scope'] in ('outer_selected','full_selected')]
        c.check(observed['final_status']==('funded' if all(r['assigned'] for r in chosenrows) else 'execution_review'),f'context{context}/no hidden refit rescue')
        record_check(observed['selected_result_summary'],summary(chosenrows),c,f'context{context}/own refit summary')
        job=next(j for j in jobs if j['stage']=='refit' and j['parent_context_id']==context)
        c.check(observed['refit_job_id']==job['job_id'] and observed['training_ids']==job['training_ids'] and observed['evaluation_ids']==job['evaluation_ids'],f'context{context}/refit identity')
        group=next(g for g in groups if g['job_id']==job['job_id'] and g['kind'] in ('outer_selected','full_selected'))
        c.check((group.get('candidate') or {}).get('candidate_id')==observed['selected_candidate_id'],f'context{context}/selected tuple applied unchanged')
    for _,row in saved_summary.iterrows():
        expected=summary([r for r in derived if r['scope']==row.scope and r['policy']==row.policy])
        for key,value in expected.items():c.value(row[key],value,'summary/'+row.scope+'/'+row.policy+'/'+key)
    after={name:sha(ROOT/name) for name in before};c.check(after==before,'all frozen evidence unchanged during audit')
    return dict(status='passed',version='state-budget-mitigation-independent-audit-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(PLAN),audit_source_freeze_sha256=sha(AUDIT_FREEZE),
        checks=c.n,maximum_absolute_numeric_error=c.max_error,counts=dict(counts,mitigation_evaluations=len(c.mitigation_checks),mitigation_trades=sum(z['trade'] for z in c.mitigation_checks)),fixed_decision_witnesses=witnesses,all_mitigation_events=c.mitigation_checks,numeric_binding_ties=c.numeric_binding_ties,
        field_inventory=inventories,sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,source_review=source_review(),
        development_ids_loaded=ids.tolist(),final_scenarios_loaded=False,new_fits=0,producer_modified=False,
        shared_independent_engine='Frozen audit_state_budget_rebuild fee/forecast/primary/selection math; adapted vector ledger and integration',boundary='All-account actual physics, complete selection, all initial decisions,54fixed primary slots and every first mitigation event independently recomputed. Other primary forecasts are saved algebra. Prior development exposure, finite support and lack of recursive feasibility remain.')

def run():
    p,m,before=preflight();ATTEMPT.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='guards_passed_no_market_values_read',started_utc=datetime.now(timezone.utc).isoformat(),
        market_value_access_started=False,final_scenarios_loaded=False,auditor_source_sha256=sha(__file__),parent_manifest_sha256=sha(OUT/'manifest.json'))
    writej(ATTEMPT/'attempt.json',state)
    try:
        result=audit_values(p,m,before,state);writej(OUT/'independent_audit.json',result)
        state.update(status='passed',phase='complete',receipt_sha256=sha(OUT/'independent_audit.json'));writej(ATTEMPT/'attempt.json',state)
        print(json.dumps(dict(status='passed',checks=result['checks'],counts=result['counts'],receipt_sha256=state['receipt_sha256'])))
    except BaseException as exc:
        (ATTEMPT/'error.log').write_text(traceback.format_exc());state.update(status='failed',error=str(exc),exception_type=type(exc).__name__,rerun_allowed=False,error_log_sha256=sha(ATTEMPT/'error.log'))
        writej(ATTEMPT/'attempt.json',state);raise

if __name__=='__main__':run()
