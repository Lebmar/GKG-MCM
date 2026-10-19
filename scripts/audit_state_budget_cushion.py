"""V3 independent all-active-decision audit, reusing frozen prior audit math.

No production helper is imported. The prior independent scalar primary is
copied with only its four destinations changed using drawdown algebra;
the prior independent unscaled nine-point fallback and ledger are reused.
Every funded active decision is recomputed, not just selected witness slots.
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
import audit_state_budget_mitigation as previous
PROTOCOL=ROOT/'docs/state_budget_cushion_protocol.json'
FREEZE=ROOT/'docs/state_budget_cushion_source_freeze.json'
AUDIT_FREEZE=ROOT/'docs/state_budget_cushion_audit_source_freeze.json'
PLAN=ROOT/'docs/state_budget_cushion_audit_plan.json'
OUT=ROOT/'results/state_budget_cushion_v1'
ATTEMPT=ROOT/'results/state_budget_cushion_audit_attempt_v1'
TOL=base.TOL
Checks,SPECS,FIN=base.Checks,base.SPECS,previous.FIN
sha,readj,writej=base.sha,base.readj,base.writej
verify_map,record_check=base.verify_map,base.record_check
select,scalar_forecasts=base.select,base.scalar_forecasts
fee_endpoint,scalar_interval,choose_band=base.fee_endpoint,base.scalar_interval,base.choose_band
blank,scalar_mitigation=previous.blank,previous.scalar_mitigation
path_metrics,summary,physical=previous.path_metrics,previous.summary,previous.physical
CUSHION_FIELDS={'cushion_scale','proposal_target_weights'}


def cushion_scale(real,peak,L):
    """Independent dimensionless drawdown form, not producer wealth algebra."""
    if not 0<real<math.inf or not 0<peak<math.inf or not 0<=L<=1:
        raise ValueError('Positive finite wealth/peak and L in[0,1] required')
    if real>peak+TOL*max(real,peak):raise ValueError('Historical peak inconsistent')
    if real>=peak or L==1:return 1.
    if L==0:return 0.
    d=1-real/peak
    return min(1.,max(0.,(L-d)/(L*(1-d))))


def scalar_primary(x,cpi,peak,worst,support,growth,weights,*,target,release,rule,initial=False,L=.4,cost=.0025):
    x=np.asarray(x,float);target=np.asarray(target,float);support=np.asarray(support,float);growth=np.asarray(growth,float)
    weights=np.asarray(weights,float);v=math.fsum(x);real=v/cpi;peak=max(peak,real)
    floor=(1-L)*peak;budget=max(floor,real-release*(real-floor))
    hs=np.array([math.fsum(x[k]*row[k] for k in range(3))/cpi for row in support])
    hg=np.array([math.fsum(x[k]*row[k] for k in range(3))/cpi for row in growth])
    scale=1. if initial else cushion_scale(real,peak,L)
    proposals=[b*scale*target+(1-b*scale)*np.array([0.,0.,1.]) for b in (1.,.75,.5,.25)]
    endpoints=[fee_endpoint(x,w,cost) for w in proposals]
    intervals=[]
    for endpoint,_,_ in endpoints:
        slope=[math.fsum((endpoint[k]-x[k])*row[k] for k in range(3))/(cpi*real) for row in support]
        intervals.append(scalar_interval((hs-budget)/real,slope))
    post=np.full((13,3),np.nan);lam=np.full(13,np.nan);fees=np.full(13,np.nan);moved=np.full(13,np.nan)
    proposal=np.full(13,-1,np.int64);eligible=np.zeros(13,bool)
    post[0]=x;lam[0]=fees[0]=moved[0]=0.;hold_safe=bool(min((hs-budget)/real)>=-TOL)
    eligible[0]=hold_safe and not initial
    for j,(endpoint,fee,transfer) in enumerate(endpoints):
        interval=intervals[j]
        values=[1.] if initial and j==0 else [] if initial or not interval['feasible'] else [interval['lo'],(interval['lo']+interval['hi'])/2,interval['hi']]
        for k,q in enumerate(values):
            index=1+3*j+k;lam[index]=q;fees[index]=q*fee;moved[index]=q*transfer;post[index]=x+q*(endpoint-x);proposal[index]=j
            slacks=[math.fsum(post[index,a]*row[a] for a in range(3))/cpi-budget for row in support]
            eligible[index]=min(slacks)/real>=-TOL
    member=np.full((13,len(weights)),np.nan);loggrowth=member.copy();gross=member.copy();breach=member.copy()
    logfee=np.full(13,np.nan);score=np.full(13,np.nan);slack=np.full(13,np.nan);worstmember=np.full(13,-1,np.int64)
    for index in range(13):
        if not math.isfinite(lam[index]):continue
        next_real=[math.fsum(post[index,k]*row[k] for k in range(3))/cpi for row in growth]
        next_weights=post[index]/math.fsum(post[index])
        ratio=[math.log(a/b) for a,b in zip(next_real,hg)]
        full=[math.log(a/real) for a in next_real]
        grosslogs=[math.log(math.fsum(next_weights[k]*row[k] for k in range(3))/math.fsum((x[k]/v)*row[k] for k in range(3))) for row in growth]
        bad=[max(worst,1-a/max(peak,a))-L>TOL for a in next_real]
        for m,prob in enumerate(weights):
            member[index,m]=math.fsum(prob[j]*ratio[j] for j in range(len(growth)))
            loggrowth[index,m]=math.fsum(prob[j]*full[j] for j in range(len(growth)))
            gross[index,m]=math.fsum(prob[j]*grosslogs[j] for j in range(len(growth)))
            breach[index,m]=math.fsum(prob[j]*bad[j] for j in range(len(growth)))
        logfee[index]=math.log1p(-fees[index]/v)
        score[index]=member[index,0] if rule=='uniform' else min(member[index])
        slacks=[math.fsum(post[index,k]*row[k] for k in range(3))/cpi-budget for row in support]
        worstmember[index]=min(range(len(slacks)),key=slacks.__getitem__);slack[index]=slacks[worstmember[index]]
    choice=choose_band(score,eligible,fees,proposal,initial=initial,hold_feasible=hold_safe);s=max(choice,0)
    reason=('initial_target_support_infeasible' if initial else 'no_feasible_declared_proposal_pending') if choice<0 else ('mandatory_initial_full_target_support_pass' if initial else 'risk_budget_protection' if not hold_safe else 'support_safe_hold' if s==0 else 'support_safe_growth')
    action='hold' if s==0 or moved[s]<=TOL else 'restore' if lam[s]==1 else 'partial'
    sf=np.maximum(-slack,0.)
    result=dict(status='selected' if choice>=0 else 'initial_review' if initial else 'no_feasible_action',
        selected_index=choice,posttrade_holdings=post[s],fee=fees[s],moved=moved[s],action=action,gate_reason=reason,
        hold_feasible=hold_safe,support_feasible=choice>=0,guard_triggered=choice<0,floor=floor,budget_floor=budget,real_wealth=real,
        selected_score=score[s],pred_log_growth_hold=loggrowth[0,0],pred_log_growth_selected=loggrowth[s,0],
        uniform_predicted_advantage=member[s,0],conditional_predicted_advantage=member[s,1] if len(weights)>1 else np.nan,
        min_predicted_advantage=min(member[s]),current_log_fee=logfee[s],gross_log_advantage_uniform=gross[s,0],
        gross_log_advantage_min=min(gross[s]),worst_real_shortfall_hold=sf[0],worst_real_shortfall_selected=sf[s],
        worst_real_shortfall_improvement=sf[0]-sf[s],growth_sacrifice=choice>=0 and score[s]<-TOL,
        interval_lo=np.array([q['lo'] for q in intervals]),interval_hi=np.array([q['hi'] for q in intervals]),
        interval_feasible=np.array([q['feasible'] for q in intervals]),interval_lower_member=np.array([q['lower_member'] for q in intervals]),
        interval_upper_member=np.array([q['upper_member'] for q in intervals]),candidate_posttrade=post,candidate_lambda=lam,
        candidate_fees=fees,candidate_moved=moved,candidate_proposal=proposal,candidate_feasible=eligible,candidate_score=score,
        candidate_min_support_slack=slack,candidate_worst_support_member=worstmember,candidate_member_advantage=member,
        candidate_member_log_growth=loggrowth,candidate_member_gross_advantage=gross,candidate_log_fee=logfee,candidate_predicted_next_breach=breach)
    result.update(cushion_scale=scale,proposal_target_weights=np.array(proposals))
    return result

def scalar_decision(x,cpi,peak,worst,support,growth,weights,*,target,release,rule,initial=False,L=.4,cost=.0025):
    result=scalar_primary(x,cpi,peak,worst,support,growth,weights,target=target,release=release,rule=rule,initial=initial,L=L,cost=cost)
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

def source_review():
    import ast
    text=(ROOT/'scripts/run_state_budget_cushion.py').read_text();tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_paths']
    if len(calls)!=1 or ast.unparse(calls[0].args[1])!='range(1, 121)' or {k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}.get('allow_final') is not False:raise AssertionError('Only fixed development loader allowed')
    return dict(status='passed_static_source_review',prior_independent_primary_sha256=sha(ROOT/'scripts/audit_state_budget_rebuild.py'),
        information='Current actual returns advance after choice; forecast/support use own-training arrays; first active funded failure only.',
        limitation='Frozen-source/API review and saved prefix/branch checks, not a formal proof against arbitrary future edits')

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
    must={'scripts/audit_state_budget_cushion.py','tests/test_audit_state_budget_cushion.py','docs/state_budget_cushion_audit_plan.json','scripts/audit_state_budget_mitigation.py','src/safeinvest/data.py','scripts/audit_state_budget_rebuild.py'}
    if not must<=set(a['source_sha256']):raise RuntimeError('Complete independent source/test/plan/loader freeze required')
    verify_map(a['source_sha256']);verify_map(closure);verify_map(m['outputs_sha256'],OUT)
    if str(PLAN.relative_to(ROOT)) not in f['input_sha256'] or f['input_sha256'][str(PLAN.relative_to(ROOT))]!=sha(PLAN):raise RuntimeError('Producer must bind the predeclared independent audit plan')
    names=set(closure)|set(a['source_sha256'])|{str(x.relative_to(ROOT)) for x in (PLAN,PROTOCOL,FREEZE,AUDIT_FREEZE,OUT/'manifest.json')}
    names|={str((OUT/k).relative_to(ROOT)) for k in m['outputs_sha256']}
    return p,m,{name:sha(ROOT/name) for name in sorted(names)}

def decision_check(observed,expected,c,label,*,x,cp,support,target,oracle=None):
    """Binding certificates use scaled primary endpoints; no old tie checker."""
    full=expected if oracle is None else oracle
    forecast_scalars=set('selected_score pred_log_growth_hold pred_log_growth_selected uniform_predicted_advantage conditional_predicted_advantage min_predicted_advantage current_log_fee gross_log_advantage_uniform gross_log_advantage_min primary_failed_hold_score primary_failed_hold_pred_log_growth primary_selected_score primary_pred_log_growth_selected'.split())
    c.check(set(observed)==set(expected),label+'/complete retained helper fields')
    for key,value in expected.items():
        if key not in ('candidate_worst_support_member','mitigation_candidate_worst_support_member','interval_lower_member','interval_upper_member'):
            if key in forecast_scalars:c.arr(np.asarray(observed[key],float),np.asarray(value,float),label+'/'+key,atol=3e-11)
            else:record_check(observed[key],value,c,label+'/'+key)
            continue
        a,b=np.asarray(observed[key],int),np.asarray(value,int)
        c.check(a.shape==b.shape,label+'/'+key+'/shape')
        for k in range(len(b)):
            if a[k]==b[k]:continue
            valid=False
            if key.endswith('worst_support_member'):
                points=full['mitigation_candidate_posttrade' if key.startswith('mitigation_') else 'candidate_posttrade']
                post=points[k]
                if np.isfinite(post).all() and 0<=a[k]<len(support):
                    values=np.asarray(support)@post/cp
                    valid=abs(values[a[k]]-min(values))<=3e-10+5e-13*math.fsum(x)/cp
            elif full['interval_feasible'][k]:
                lower=key=='interval_lower_member';bound=full['interval_lo' if lower else 'interval_hi'][k]
                if a[k]==-1:valid=abs(bound-(0 if lower else 1))<=3e-11
                elif 0<=a[k]<len(support):
                    endpoint=fee_endpoint(x,full['proposal_target_weights'][k])[0]
                    row=np.asarray(support)[a[k]];real=full['real_wealth']
                    margin=(math.fsum(row[j]*x[j] for j in range(3))/cp-full['budget_floor'])/real
                    slope=math.fsum(row[j]*(endpoint[j]-x[j]) for j in range(3))/(cp*real)
                    valid=abs(margin+bound*slope)<=3e-11 and (slope>0 if lower else slope<0)
            c.check(valid,label+'/'+key+f'/{k}/equivalent scaled binding row')
            c.numeric_binding_ties.append(dict(field=label+'/'+key,index=k,observed=int(a[k]),scalar=int(b[k])))


def schema(arm,group,n,train_n,c):
    if group['kind']=='reference':previous.schema(arm,group,n,train_n,c);return
    old={k:v for k,v in arm.items() if k not in CUSHION_FIELDS}
    c.check(set(arm)==set(old)|CUSHION_FIELDS,group['group_id']+'/complete V3 two-field extension')
    meta=dict(group,array_names=list(old));previous.schema(old,meta,n,train_n,c)
    c.check(set(group['array_names'])==set(arm) and len(group['array_names'])==len(arm),group['group_id']+'/full V3 metadata inventory')
    for name,shape in (('cushion_scale',(n,30)),('proposal_target_weights',(n,30,4,3))):
        c.check(arm[name].shape==shape and arm[name].dtype.kind=='f',group['group_id']+'/'+name+'/shape and dtype')


def model_gates(arm,train,trainpi,ids,group,c):
    label=group['group_id'];n,h=arm['fees'].shape;active=arm['gate_evaluated'];mit=arm['mitigation_evaluated'];spec=group['candidate']
    gross=(1+train[:,:h])/(1+trainpi[:,:h,None])
    if 'training_real_gross' in arm:c.arr(arm['training_real_gross'],np.transpose(gross,(1,0,2)),label+'/own training gross')
    lag=np.full((h,len(ids),4),np.nan);lag[1:,:,:3]=np.transpose(gross[:,:-1]-1,(1,0,2));lag[1:,:,3]=trainpi[:,:-1].T
    if 'training_previous_state' in arm:c.arr(arm['training_previous_state'],lag,label+'/own previous-only training state')
    for key in CUSHION_FIELDS:c.check(np.isnan(arm[key][~active]).all(),label+'/'+key+'/pending NULL')
    c.arr(arm['actual_decision_source'],np.where(~active,'pending',np.where(mit,'mitigation','primary')),label+'/actual branch source')
    c.arr(arm['primary_gate_reason'][~active],np.full(np.count_nonzero(~active),''),label+'/pending primary reason')
    for key in ('primary_selected_score','primary_pred_log_growth_selected'):c.check(np.isnan(arm[key][~active]).all(),label+'/'+key+'/pending NULL')
    c.check(np.all(mit.sum(axis=1)<=1) and not mit[:,0].any() and not mit[~active].any(),label+'/one funded active noninitial mitigation only')
    for key in previous.MIT_FIELDS:
        if key not in arm or key in ('actual_decision_source','primary_gate_reason','primary_selected_score','primary_pred_log_growth_selected','mitigation_evaluated','mitigation_trade'):continue
        values=arm[key][~mit]
        if arm[key].dtype.kind=='f':c.check(np.isnan(values).all(),label+'/'+key+'/nonmitigation NULL')
        elif arm[key].dtype.kind=='i':c.check(np.all(values==-1),label+'/'+key+'/nonmitigation sentinel')
    for key in ('mitigation_evaluated','mitigation_trade'):c.check(not arm[key][~active].any(),label+'/'+key+'/pending false')
    slots=np.argwhere(active);mit_n=0
    for i,t in slots:
        cp=float(arm['cpi'][i,t]);x=arm['holdings'][i,t];target=group['target']
        weights=scalar_forecasts(ids) if t==0 else scalar_forecasts(ids,lag[t],arm['actual_previous_state'][i,t])
        expected=scalar_decision(x,cp,arm['real_peak'][i,t],arm['running_mdd'][i,t],gross[:,t],gross[:,t],weights,
            target=target,release=spec['release_fraction'],rule=spec['forecast_rule'],initial=t==0)
        observed={};retained={};K=len(weights)
        for key,value in expected.items():
            source={'fee':'fees','action':'actions'}.get(key,key)
            if key=='status':continue
            if key=='real_wealth':actual=arm['real_wealth'][i,t]
            elif source in arm:
                actual=arm[source][i,t]
                if key.startswith(('candidate_member_','mitigation_candidate_member_')) or key=='candidate_predicted_next_breach':
                    c.check(np.isnan(actual[...,K:]).all(),label+f'/{i}/{t}/'+key+'/unused member padding NULL')
                    actual=actual[...,:K]
            else:continue
            observed[key]=actual;retained[key]=value
        decision_check(observed,retained,c,label+f'/{i}/{t}/all_active_scalar',x=x,cp=cp,support=gross[:,t],target=target,oracle=expected)
        c.value(arm['forecast_n'][i,t],len(ids),label+f'/{i}/{t}/training count')
        c.value(arm['forecast_member_count'][i,t],K,label+f'/{i}/{t}/member count')
        idx=expected['selected_index'];s=max(idx,0)
        c.value(arm['selected_proposal'][i,t],int(expected['candidate_proposal'][s]) if idx>=0 else -1,label+f'/{i}/{t}/primary proposal')
        c.value(arm['selected_lambda'][i,t],expected['candidate_lambda'][s] if idx>=0 else np.nan,label+f'/{i}/{t}/primary lambda')
        execution=expected['mitigation_selected_index']>0 if expected['mitigation_evaluated'] else idx>0
        c.value(arm['target_execution'][i,t],execution,label+f'/{i}/{t}/executed branch target label')
        if expected['mitigation_evaluated']:
            mit_n+=1;real=expected['real_wealth'];chosen=expected['mitigation_selected_index'];improvement=expected['mitigation_worst_next_real_improvement']
            c.check(not arm['support_feasible'][i,t] and arm['guard_triggered'][i,t] and not arm['automatic_service'][i,t],label+f'/{i}/{t}/mitigation not recertified')
            c.check(improvement>TOL*real if chosen>0 else expected['fee']==0 and expected['moved']==0,label+f'/{i}/{t}/strict afterfee improvement or free hold')
            c.mitigation_checks.append(dict(group_id=label,scenario_id=group['scenario_ids'][i],original_year=int(t+1),selected_index=int(chosen),
                trade=bool(expected['mitigation_trade']),nominal_fee=float(expected['fee']),worst_real_improvement=float(improvement),
                residual_real_shortfall=float(expected['worst_real_shortfall_selected']),predicted_paired_score=float(expected['selected_score']),support_certificate=False))
    c.check(mit_n==np.count_nonzero(mit),label+'/all fallback events independently enumerated')
    c.active_decision_groups.append(dict(group_id=label,active_decisions=len(slots),mitigation_decisions=mit_n,
        original_year_and_account_slots_sha256=hashlib.sha256(np.ascontiguousarray(slots).tobytes()).hexdigest()))

def audit_values(p,m,before,state):
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    state.update(phase='loading_development_only',market_value_access_started=True);writej(ATTEMPT/'attempt.json',state)
    r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
    c=Checks();c.mitigation_checks=[];c.active_decision_groups=[];c.check(ids.tolist()==list(range(1,121)),'exact development identities')
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
    expected_summary={('outer_selected','state_budget_learner'),('full_selected','state_budget_learner')}
    expected_summary|={(scope,policy) for scope in ('outer_reference','full_training_reference') for policy in ('same_target_B','same_target_annual','exact_thirds_annual')}
    c.check(set(zip(saved_summary.scope,saved_summary.policy))==expected_summary,'all eight distinct original summary cohorts')
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
    return dict(status='passed',version='state-budget-cushion-independent-audit-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(PLAN),audit_source_freeze_sha256=sha(AUDIT_FREEZE),
        checks=c.n,maximum_absolute_numeric_error=c.max_error,counts=dict(counts,all_active_independent_decisions=sum(z['active_decisions'] for z in c.active_decision_groups),mitigation_evaluations=len(c.mitigation_checks),mitigation_trades=sum(z['trade'] for z in c.mitigation_checks)),fixed_decision_witnesses=witnesses,all_mitigation_events=c.mitigation_checks,numeric_binding_ties=c.numeric_binding_ties,
        all_active_decision_groups=c.active_decision_groups,field_inventory=inventories,sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,source_review=source_review(),
        development_ids_loaded=ids.tolist(),final_scenarios_loaded=False,new_fits=0,producer_modified=False,
        shared_independent_engine='Frozen V2 audit fee/forecast/selection, copied independent primary with drawdown-scaled endpoints, frozen V2b independent fallback/ledger/metrics',boundary='Every active decision in every funded group independently recomputed, including thin inner retained fields; full selected groups compare all13+9 candidate arrays. All original accounting/NULL/selection and initial witnesses checked.54 slots remain a redundant fixed witness inventory. Reused development, finite support and lack of recursive feasibility remain.')

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
