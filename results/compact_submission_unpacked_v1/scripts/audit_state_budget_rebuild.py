"""Independent scalar math and complete saved-account audit of the rebuild.

No producer engine/forecast/selection/trade/metrics is imported. The sole raw
loader is guarded development-only and is invoked after frozen complete gates.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, math, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/state_budget_rebuild_protocol.json'
FREEZE=ROOT/'docs/state_budget_rebuild_source_freeze.json'
AUDIT_FREEZE=ROOT/'docs/state_budget_rebuild_audit_source_freeze.json'
PLAN=ROOT/'docs/state_budget_rebuild_audit_plan.json'
OUT=ROOT/'results/state_budget_rebuild_v1'
ATTEMPT=ROOT/'results/state_budget_rebuild_audit_attempt_v1'
TOL=1e-12
FIN=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','trade_fraction','breach',
     'first_crossing','automatic_fraction','protection_trade_fraction','support_review')
SPECS=tuple(dict(candidate_id=f'release_{r:g}_{f}',release_fraction=r,forecast_rule=f,order=i)
            for i,(r,f) in enumerate((r,f) for r in (.5,.75,1.) for f in ('uniform','ensemble_min')))
PHYSICAL=set('holdings weights nominal_wealth real_wealth real_peak current_drawdown running_mdd ever_breached cpi posttrade_holdings fees moved pretrade_wealth fee_fraction moved_fraction pretrade_tv_drift end_year_tv_drift target_execution pending_at_decision actions trades end_weights first_crossing triggered drawdown_at_trigger peak_at_trigger real_wealth_at_trigger recovery_endpoint recovery_time recovered right_censored available_followup observed_followup added_severity terminal_excess'.split())
EXTRAS=set('floor budget_floor selected_score pred_log_growth_hold pred_log_growth_selected uniform_predicted_advantage conditional_predicted_advantage min_predicted_advantage current_log_fee gross_log_advantage_uniform gross_log_advantage_min worst_real_shortfall_hold worst_real_shortfall_selected worst_real_shortfall_improvement guard_pending_at_decision was_pending_input was_guard_pending_input automatic_service support_feasible hold_feasible growth_sacrifice guard_triggered gate_evaluated guard_pending first_guard_trigger selected_index selected_proposal selected_lambda forecast_n forecast_member_count actual_previous_state training_previous_state interval_lo interval_hi interval_feasible interval_lower_member interval_upper_member candidate_lambda candidate_fees candidate_moved candidate_score candidate_min_support_slack candidate_log_fee candidate_feasible candidate_proposal candidate_worst_support_member candidate_posttrade candidate_member_advantage candidate_member_log_growth candidate_member_gross_advantage candidate_predicted_next_breach targets loss_tolerances training_real_gross realized_real_gross'.split())
EXTRAS.add('gate_reason')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def isnull(x):return x is None or isinstance(x,(float,np.floating)) and np.isnan(x)


class Checks:
    def __init__(self):self.n=0;self.max_error=0.;self.numeric_binding_ties=[]
    def check(self,condition,label):
        self.n+=1
        if not condition:raise AssertionError(label)
    def arr(self,a,b,label,*,atol=3e-10,rtol=5e-13):
        a,b=np.asarray(a),np.asarray(b);self.check(a.shape==b.shape,label+'/shape')
        if a.dtype.kind in 'bUiuS' or b.dtype.kind in 'bUiuS':self.check(np.array_equal(a,b),label+'/exact');return
        self.check(np.array_equal(np.isnan(a),np.isnan(b)),label+'/NULLmask')
        self.check(np.array_equal(np.isfinite(a),np.isfinite(b)),label+'/finite mask')
        ok=np.isfinite(a)&np.isfinite(b)
        e=float(np.max(np.abs(a[ok]-b[ok]),initial=0.));self.max_error=max(self.max_error,e)
        self.check(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),label+'/value')
    def value(self,a,b,label):
        if isnull(b):self.check(isnull(a),label+'/NULL')
        elif isinstance(b,(str,bool,np.bool_)):self.check(a==b,label+'/exact')
        else:self.arr(a,b,label)


def fee_endpoint(x,w,cost=.0025):
    """Own scalar monotone bisection, not the piecewise producer executor."""
    x=[float(v) for v in x];w=[float(v) for v in w];total=math.fsum(x);sw=math.fsum(w)
    if total<=0 or sw<=0 or any(v<0 for v in x+w) or not 0<=cost<1:raise ValueError('Physical positive account/target/rate required')
    w=[v/sw for v in w];lo,hi=0.,cost*total
    for _ in range(100):
        f=(lo+hi)/2;buy=math.fsum(max((total-f)*w[j]-x[j],0.) for j in range(3))
        if f>cost*buy:hi=f
        else:lo=f
    f=(lo+hi)/2 if cost else 0.;post=np.array([(total-f)*v for v in w])
    moved=math.fsum(max(post[j]-x[j],0.) for j in range(3))
    return post,f,moved


def scalar_forecasts(ids,train_previous=None,actual_previous=None):
    """Own four-channel population scaling, kernel, shrink and sorted-ID blocks."""
    ids=list(map(int,ids));n=len(ids)
    if not ids or len(set(ids))!=n or min(ids)<1 or max(ids)>120:raise ValueError('Development training IDs required')
    uniform=[1/n]*n
    if train_previous is None and actual_previous is None:return np.array([uniform])
    if n<3 or train_previous is None or actual_previous is None:raise ValueError('Both lagged states and3training rows required')
    z=[[float(v) for v in row] for row in train_previous];a=list(map(float,actual_previous))
    if len(z)!=n or len(a)!=4 or any(len(row)!=4 for row in z):raise ValueError('Four-channel lagged inputs required')
    means=[math.fsum(row[k] for row in z)/n for k in range(4)]
    scales=[max(math.sqrt(math.fsum((row[k]-means[k])**2 for row in z)/n),1e-8) for k in range(4)]
    kernels=[]
    for row in z:
        differences=[(a[k]-row[k])/scales[k] for k in range(4)]
        distance=math.fsum(d*d for d in differences)
        kernels.append(math.exp(-distance/8))
    total=math.fsum(kernels)
    conditional=[.5/n+.5*v/total for v in kernels] if math.isfinite(total) and total>0 else uniform.copy()
    order=sorted(range(n),key=lambda i:ids[i]);base,extra=divmod(n,3);cursor=0;blocks=[]
    for k in range(3):
        size=base+(k<extra);chosen=order[cursor:cursor+size];cursor+=size
        blocks.append([1/size if j in chosen else 0. for j in range(n)])
    return np.asarray([uniform,conditional,*blocks])


def scalar_interval(margins,slopes):
    lower,upper=0.,1.;lower_id=upper_id=-1
    for j,(a,b) in enumerate(zip(margins,slopes)):
        if b==0 and a<0:return dict(feasible=False,lo=np.nan,hi=np.nan,lower_member=j,upper_member=j)
        if b>0 and -a/b>lower:lower,lower_id=-a/b,j
        if b<0 and -a/b<upper:upper,upper_id=-a/b,j
    feasible=lower<=upper and lower<=1 and upper>=0
    return dict(feasible=feasible,lo=lower if feasible else np.nan,hi=upper if feasible else np.nan,
                lower_member=lower_id,upper_member=upper_id)


def choose_band(scores,eligible,fees,proposals,*,initial,hold_feasible):
    available=[i for i,v in enumerate(eligible) if v]
    if not available:return -1
    best=max(float(scores[i]) for i in available)
    if not initial and hold_feasible and best<=TOL:return 0
    band=[i for i in available if scores[i]>=best-TOL and (initial or not hold_feasible or i==0 or scores[i]>TOL)]
    return min(band,key=lambda i:(i!=0,float(fees[i]),int(proposals[i]),i))


def scalar_decision(x,cpi,peak,worst,support,growth,weights,*,target,release,rule,initial=False,L=.4,cost=.0025):
    x=np.asarray(x,float);target=np.asarray(target,float);support=np.asarray(support,float);growth=np.asarray(growth,float)
    weights=np.asarray(weights,float);v=math.fsum(x);real=v/cpi;peak=max(peak,real)
    floor=(1-L)*peak;budget=max(floor,real-release*(real-floor))
    hs=np.array([math.fsum(x[k]*row[k] for k in range(3))/cpi for row in support])
    hg=np.array([math.fsum(x[k]*row[k] for k in range(3))/cpi for row in growth])
    proposals=[b*target+(1-b)*np.array([0.,0.,1.]) for b in (1.,.75,.5,.25)]
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
    return result


def quantile(values,p):
    v=sorted(map(float,values));return v[max(0,min(len(v)-1,math.ceil(len(v)*p)-1))]
def median(values):
    v=sorted(map(float,values));n=len(v);return v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2
def cvar(values):
    v=sorted(map(float,values),reverse=True);mass=len(v)/10
    return math.fsum(x*max(0.,min(1.,mass-i)) for i,x in enumerate(v))/mass


def summary(rows):
    rows=list(rows);ids=[int(r['scenario_id']) for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Unique original IDs required')
    fin=[r for r in rows if r['assigned']];n=len(rows);k=len(fin)
    outside=sum(r.get('review_reason')=='not_balanced_case' for r in rows)
    q1_review=sum(r.get('review_reason')=='q1_review' for r in rows)
    result=dict(original_n=n,financial_n=k,initial_noninvested_n=n-k,not_balanced_n=outside,q1_review_n=q1_review,
        initial_execution_review_n=n-k-outside-q1_review,initial_coverage=k/n if n else None)
    result.update(dict.fromkeys('breach_count breach_fraction mean_log_growth median_cagr p10_cagr mean_mdd median_mdd p90_mdd cvar90_mdd mean_volatility mean_nominal_fees mean_trade_fraction support_review_n mean_automatic_fraction mean_protection_trade_fraction'.split()))
    if not fin:return result
    def avg(key):return math.fsum(float(r[key]) for r in fin)/k
    losses=[r['mdd'] for r in fin];growth=[r['cagr'] for r in fin];bad=sum(bool(r['breach']) for r in fin)
    result.update(breach_count=bad,breach_fraction=bad/k,mean_log_growth=avg('log_growth'),median_cagr=median(growth),p10_cagr=quantile(growth,.1),
        mean_mdd=avg('mdd'),median_mdd=median(losses),p90_mdd=quantile(losses,.9),cvar90_mdd=cvar(losses),mean_volatility=avg('volatility'),
        mean_nominal_fees=avg('nominal_fees'),mean_trade_fraction=avg('trade_fraction'),support_review_n=sum(bool(r['support_review']) for r in fin),
        mean_automatic_fraction=avg('automatic_fraction'),mean_protection_trade_fraction=avg('protection_trade_fraction'))
    return result


def select(candidate_rows,B,alpha=.1):
    b={int(r['scenario_id']):r for r in B};evidence=[]
    for spec in SPECS:
        rows=candidate_rows[spec['candidate_id']]
        if len(rows)!=len(b) or {int(r['scenario_id']) for r in rows}!=set(b):raise ValueError('Identical complete original population required')
        s=summary(rows);ids=[int(r['scenario_id']) for r in rows if r['assigned']]
        if any(not b[i]['assigned'] for i in ids):raise ValueError('Funded same-U B required')
        same=summary([b[i] for i in ids]);reasons=[]
        if not ids:reasons.append('no_invested_inner_paths')
        else:
            if s['breach_count']>math.floor(alpha*len(ids)+TOL):reasons.append('inner_event_frequency_above_reference_alpha')
            if s['mean_mdd']>same['mean_mdd']+TOL:reasons.append('inner_mean_mdd_worse_than_same_cohort_B')
            if s['cvar90_mdd']>same['cvar90_mdd']+TOL:reasons.append('inner_cvar90_mdd_worse_than_same_cohort_B')
        evidence.append(dict(**spec,summary=s,same_cohort_B=same,eligible=not reasons,reasons=reasons))
    eligible=[r for r in evidence if r['eligible']]
    if not eligible:return dict(status='execution_review',selected_candidate_id=None,evidence=evidence,reason='no_inner_candidate_meets_declared_risk_requirements')
    n=max(r['summary']['financial_n'] for r in eligible);eligible=[r for r in eligible if r['summary']['financial_n']==n]
    q=min(r['summary']['support_review_n'] for r in eligible);eligible=[r for r in eligible if r['summary']['support_review_n']==q]
    g=max(r['summary']['mean_log_growth'] for r in eligible);eligible=[r for r in eligible if r['summary']['mean_log_growth']>=g-TOL]
    f=min(r['summary']['mean_nominal_fees'] for r in eligible);eligible=[r for r in eligible if r['summary']['mean_nominal_fees']<=f+TOL]
    chosen=min(eligible,key=lambda r:r['order'])
    return dict(status='inner_selected_pending_refit',selected_candidate_id=chosen['candidate_id'],evidence=evidence,
        tied_candidate_ids=[r['candidate_id'] for r in eligible],reason='coverage_then_fewer_support_reviews_then_net_growth')


def record_check(actual,expected,c,label):
    if isinstance(expected,dict):
        c.check(set(actual)==set(expected),label+'/complete keys')
        for key,value in expected.items():record_check(actual[key],value,c,label+'/'+key)
    elif isinstance(expected,(list,tuple)):
        c.check(len(actual)==len(expected),label+'/length')
        for i,value in enumerate(expected):record_check(actual[i],value,c,label+f'/{i}')
    elif isinstance(expected,np.ndarray):
        c.arr(np.asarray(actual,dtype=expected.dtype),expected,label,atol=3e-11)
    else:c.value(actual,expected,label)


def binding_check(name,observed,expected,result,x,cp,support,target,c,label):
    """Certify equivalent scalar summation ties, never change a decision tie."""
    a,b=np.asarray(observed,int),np.asarray(expected,int);c.check(a.shape==b.shape,label+'/shape')
    for k in range(len(a)):
        if a[k]==b[k]:continue
        valid=False
        if name=='candidate_worst_support_member' and np.isfinite(result['candidate_lambda'][k]):
            slacks=np.asarray(support)@result['candidate_posttrade'][k]/cp-result['budget_floor']
            valid=0<=a[k]<len(slacks) and abs(slacks[a[k]]-slacks.min())<=3e-10+5e-13*result['real_wealth']
        elif name in ('interval_lower_member','interval_upper_member') and result['interval_feasible'][k]:
            bound=result['interval_lo' if name=='interval_lower_member' else 'interval_hi'][k]
            if a[k]==-1:valid=abs(bound-(0 if name=='interval_lower_member' else 1))<=3e-11
            elif 0<=a[k]<len(support):
                endpoint=fee_endpoint(x,(1.,.75,.5,.25)[k]*target+(1-(1.,.75,.5,.25)[k])*np.array([0.,0.,1.]))[0]
                row=support[a[k]];margin=(float(row@x)/cp-result['budget_floor'])/result['real_wealth']
                slope=float(row@(endpoint-x))/(cp*result['real_wealth'])
                valid=abs(margin+bound*slope)<=3e-11 and (slope>0 if name=='interval_lower_member' else slope<0)
        c.check(valid,label+f'/{k}/equivalent binding row')
        c.numeric_binding_ties.append(dict(field=label,index=k,observed=int(a[k]),scalar=int(b[k])))


def decision_check(observed,expected,c,label,*,x,cp,support,target):
    c.check(set(observed)==set(expected),label+'/complete helper fields')
    for key,value in expected.items():
        if key in ('candidate_worst_support_member','interval_lower_member','interval_upper_member'):
            binding_check(key,observed[key],value,expected,np.asarray(x),cp,np.asarray(support),np.asarray(target),c,label+'/'+key)
        else:record_check(observed[key],value,c,label+'/'+key)


def path_metrics(arm,i):
    real=list(map(float,arm['real_wealth'][i]));h=len(real)-1;annual=[real[t+1]/real[t]-1 for t in range(h)]
    mu=math.fsum(annual)/h;vol=math.sqrt(math.fsum((x-mu)**2 for x in annual)/(h-1)) if h>1 else 0.
    growth=math.log(real[-1]/100)/h;mdd=max([0.]+[1-real[t]/real[s] for s in range(h) for t in range(s+1,h+1)])
    tau=next((t for t in range(1,h+1) if float(arm['running_mdd'][i,t])-.4>TOL),None)
    pending=arm['pending_at_decision'][i];guard=arm.get('guard_triggered',np.zeros_like(arm['pending_at_decision']))[i]
    reason=arm.get('gate_reason',np.full_like(arm['actions'],'',dtype='U64'))[i]
    return dict(log_growth=growth,cagr=math.expm1(growth),mdd=mdd,volatility=vol,terminal_real=real[-1],
        nominal_fees=math.fsum(arm['fees'][i]),trade_fraction=float(np.count_nonzero(arm['moved'][i]>TOL))/h,
        breach=mdd-.4>TOL,first_crossing=tau,automatic_fraction=float(np.count_nonzero(~(pending|guard)))/h,
        protection_trade_fraction=float(np.count_nonzero(reason=='risk_budget_protection'))/h,
        support_review=bool(np.any(arm.get('guard_pending',np.zeros((len(arm['real_wealth']),h+1),bool))[i])))


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
    c.arr(post[before|new],x[:,:-1][before|new],label+'/pending exact physical hold')
    c.arr(arm['actions'][before|new],np.full(np.count_nonzero(before|new),'hold'),label+'/pending action')
    c.check(not arm['target_execution'][before|new].any(),label+'/pending no target execution')
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


def model_gates(arm,train,trainpi,ids,group,c):
    label=group['group_id'];n,h=arm['fees'].shape;realgross=(1+train[:,:h])/(1+trainpi[:,:h,None]);spec=group['candidate']
    if 'training_real_gross' in arm:c.arr(arm['training_real_gross'],np.transpose(realgross,(1,0,2)),label+'/training gross identity')
    if 'training_previous_state' in arm:
        previous=np.full((h,len(train),4),np.nan);previous[1:,:,:3]=np.transpose(realgross[:,:-1]-1,(1,0,2));previous[1:,:,3]=trainpi[:,:-1].T
        c.arr(arm['training_previous_state'],previous,label+'/train prefix t-1 only')
    active=arm['gate_evaluated'];initial=np.zeros((n,h),bool);initial[:,0]=True
    r=arm['real_wealth'][:,:-1];floor=(1-.4)*arm['real_peak'][:,:-1];budget=np.maximum(floor,r-spec['release_fraction']*(r-floor))
    c.arr(arm['floor'][active],floor[active],label+'/original floor')
    c.arr(arm['budget_floor'][active],budget[active],label+'/clamped released floor')
    c.arr(arm['forecast_n'][active],np.full(np.count_nonzero(active),len(ids),int),label+'/training count')
    count=np.tile(np.r_[1,np.full(h-1,5)],(n,1));c.arr(arm['forecast_member_count'][active],count[active],label+'/member count')
    for t in range(h):
        rows=np.flatnonzero(active[:,t])
        if not len(rows):continue
        x=arm['holdings'][rows,t];post=arm['posttrade_holdings'][rows,t];cp=arm['cpi'][rows,t]
        support=realgross[:,t];holdmargin=(x@support.T/cp[:,None]-budget[rows,t,None]).min(axis=1)
        safe=holdmargin/r[rows,t]>=-TOL;c.arr(arm['hold_feasible'][rows,t],safe,label+f'/{t}/hold floor')
        selectedmargin=(post@support.T/cp[:,None]-budget[rows,t,None]).min(axis=1)
        supportok=arm['selected_index'][rows,t]>=0
        c.arr(arm['support_feasible'][rows,t],supportok,label+f'/{t}/selected feasible flag')
        c.check(np.all(selectedmargin[supportok]/r[rows[supportok],t]>=-TOL),label+f'/{t}/selected support roundoff convention')
        c.arr(arm['worst_real_shortfall_hold'][rows,t],np.maximum(-holdmargin,0),label+f'/{t}/hold shortfall')
        c.arr(arm['worst_real_shortfall_selected'][rows,t],np.maximum(-selectedmargin,0),label+f'/{t}/selected shortfall')
        c.arr(arm['worst_real_shortfall_improvement'][rows,t],np.maximum(-holdmargin,0)-np.maximum(-selectedmargin,0),label+f'/{t}/shortfall improvement')
        chosen=arm['selected_index'][rows,t];score=arm['selected_score'][rows,t]
        expected=arm['uniform_predicted_advantage'][rows,t] if spec['forecast_rule']=='uniform' else arm['min_predicted_advantage'][rows,t]
        c.arr(score,expected,label+f'/{t}/paired selected score')
        c.arr(arm['current_log_fee'][rows,t],np.log1p(-arm['fees'][rows,t]/arm['pretrade_wealth'][rows,t]),label+f'/{t}/fee log once')
        c.arr(arm['uniform_predicted_advantage'][rows,t],arm['current_log_fee'][rows,t]+arm['gross_log_advantage_uniform'][rows,t],label+f'/{t}/uniform net-gross identity')
        c.arr(arm['min_predicted_advantage'][rows,t],arm['current_log_fee'][rows,t]+arm['gross_log_advantage_min'][rows,t],label+f'/{t}/min net-gross identity')
        c.arr(arm['growth_sacrifice'][rows,t],(chosen>=0)&(score<-TOL),label+f'/{t}/sacrifice flag')
        reasons=np.where(chosen<0,'no_feasible_declared_proposal_pending',np.where(~safe,'risk_budget_protection',np.where(chosen==0,'support_safe_hold','support_safe_growth')))
        if t==0:reasons=np.full(len(rows),'mandatory_initial_full_target_support_pass')
        c.arr(arm['gate_reason'][rows,t],reasons,label+f'/{t}/reason')
        c.check(not np.any((t>0)&safe&(chosen>0)&(score<=TOL)),label+f'/{t}/safe trade strictly net beneficial')
        c.arr(arm['target_execution'][rows,t],chosen>0,label+f'/{t}/proposal execution label')
        p=arm['selected_proposal'][rows,t];lam=arm['selected_lambda'][rows,t]
        for z,i in enumerate(rows):
            if chosen[z]<0:continue
            if chosen[z]==0:
                c.arr(post[z],x[z],label+f'/{t}/{i}/selected hold');continue
            c.check(0<=p[z]<4 and 0<=lam[z]<=1,label+f'/{t}/{i}/declared proposal/lambda')
            if lam[z]>0:
                end=x[z]+(post[z]-x[z])/lam[z];v=end.sum();weight=(1.,.75,.5,.25)[p[z]]*np.asarray(group['target'])+(1-(1.,.75,.5,.25)[p[z]])*np.array([0.,0.,1.])
                c.arr(end/v,weight,label+f'/{t}/{i}/exact endpoint destination',atol=3e-11)
            action='hold' if arm['moved'][i,t]<=TOL else 'restore' if lam[z]==1 else 'partial'
            c.check(arm['actions'][i,t]==action,label+f'/{t}/{i}/action class')
        if 'candidate_score' not in arm:continue
        for z,i in enumerate(rows):
            q=arm['candidate_lambda'][i,t];points=arm['candidate_posttrade'][i,t];fee=arm['candidate_fees'][i,t]
            finite=np.isfinite(q);c.check(q[0]==0 and finite[0],label+f'/{t}/{i}/hold slot always present')
            expectedq=np.full(13,np.nan);expectedq[0]=0.;expectedproposal=np.full(13,-1,int)
            if t==0:expectedq[1]=1.;expectedproposal[1]=0
            else:
                for j in range(4):
                    if arm['interval_feasible'][i,t,j]:
                        lo,hi=arm['interval_lo'][i,t,j],arm['interval_hi'][i,t,j]
                        c.check(0<=lo<=hi<=1,label+f'/{t}/{i}/{j}/feasible interval')
                        expectedq[1+3*j:4+3*j]=[lo,(lo+hi)/2,hi];expectedproposal[1+3*j:4+3*j]=j
                    else:c.check(np.isnan(arm['interval_lo'][i,t,j]) and np.isnan(arm['interval_hi'][i,t,j]),label+f'/{t}/{i}/{j}/empty interval NULL')
            c.arr(q,expectedq,label+f'/{t}/{i}/finite13point menu')
            c.arr(arm['candidate_proposal'][i,t],expectedproposal,label+f'/{t}/{i}/proposal menu order')
            buy=np.maximum(points[finite]-x[z],0).sum(axis=1)
            c.arr(arm['candidate_moved'][i,t,finite],buy,label+f'/{t}/{i}/all candidate physical moved')
            c.arr(fee[finite],.0025*buy,label+f'/{t}/{i}/all candidate fee once')
            c.arr(points[finite].sum(axis=1)+fee[finite],np.full(sum(finite),sum(x[z])),label+f'/{t}/{i}/all candidate funding')
            minimum=(points[finite]@support.T/cp[z]-budget[i,t]).min(axis=1)
            expectedslack=np.full(13,np.nan);expectedslack[finite]=minimum
            c.arr(arm['candidate_min_support_slack'][i,t],expectedslack,label+f'/{t}/{i}/all supported minima')
            eligible=np.zeros(13,bool);eligible[finite]=minimum/r[i,t]>=-TOL
            if t==0:eligible[0]=False
            c.arr(arm['candidate_feasible'][i,t],eligible,label+f'/{t}/{i}/all support eligibility')
            c.check(np.isnan(points[~finite]).all() and np.isnan(fee[~finite]).all() and np.isnan(arm['candidate_moved'][i,t,~finite]).all(),label+f'/{t}/{i}/unavailable candidate NULL')
            K=int(arm['forecast_member_count'][i,t]);candidate=arm['candidate_member_advantage'][i,t,:,:K]
            score0=candidate[:,0] if spec['forecast_rule']=='uniform' else candidate.min(axis=1)
            c.arr(arm['candidate_score'][i,t],score0,label+f'/{t}/{i}/all candidate score algebra')
            c.arr(candidate,arm['candidate_member_gross_advantage'][i,t,:,:K]+arm['candidate_log_fee'][i,t,:,None],label+f'/{t}/{i}/all fee decomposition')
            if K<5:
                for key in ('candidate_member_advantage','candidate_member_log_growth','candidate_member_gross_advantage','candidate_predicted_next_breach'):c.check(np.isnan(arm[key][i,t,:,K:]).all(),label+f'/{t}/{i}/{key}/padding NULL')
            idx=choose_band(arm['candidate_score'][i,t],arm['candidate_feasible'][i,t],arm['candidate_fees'][i,t],arm['candidate_proposal'][i,t],initial=t==0,hold_feasible=bool(safe[z]))
            c.check(idx==int(chosen[z]),label+f'/{t}/{i}/all13global tie selection')
            s=max(idx,0)
            for saved,key in [('posttrade_holdings','candidate_posttrade'),('fees','candidate_fees'),('moved','candidate_moved')]:c.arr(arm[saved][i,t],arm[key][i,t,s],label+f'/{t}/{i}/selected from/'+key)


def schema(arm,group,n,train_n,c):
    full=PHYSICAL|EXTRAS
    omitted={k for k in full if k.startswith('candidate_') or k in ('training_previous_state','training_real_gross')}
    expected=PHYSICAL if group['kind']=='reference' else full-omitted if group['kind']=='inner_candidate' else full
    c.check(set(arm)==expected,group['group_id']+'/complete declared array inventory')
    c.check(set(group['array_names'])==set(arm) and len(group['array_names'])==len(arm),group['group_id']+'/array metadata inventory')
    c.check(set(group['omitted_diagnostic_arrays'])==(omitted if group['kind']=='inner_candidate' else set()),group['group_id']+'/exact deliberate omitted diagnostics')
    scalar=set('first_crossing triggered drawdown_at_trigger peak_at_trigger real_wealth_at_trigger recovery_endpoint recovery_time recovered right_censored available_followup observed_followup added_severity terminal_excess first_guard_trigger'.split())
    end=set('nominal_wealth real_wealth real_peak current_drawdown running_mdd ever_breached cpi guard_pending'.split())
    booleans=set('ever_breached target_execution pending_at_decision trades triggered recovered right_censored guard_pending guard_pending_at_decision was_pending_input was_guard_pending_input automatic_service support_feasible hold_feasible growth_sacrifice guard_triggered gate_evaluated interval_feasible candidate_feasible'.split())
    ints=set('first_crossing recovery_endpoint recovery_time available_followup observed_followup first_guard_trigger selected_index selected_proposal forecast_n forecast_member_count interval_lower_member interval_upper_member candidate_proposal candidate_worst_support_member'.split())
    for key,value in arm.items():
        shape=(n,30)
        if key in scalar:shape=(n,)
        elif key in end:shape=(n,31)
        elif key in ('holdings','weights'):shape=(n,31,3)
        elif key in ('posttrade_holdings','end_weights','realized_real_gross'):shape=(n,30,3)
        elif key=='targets':shape=(3,)
        elif key=='loss_tolerances':shape=(1,)
        elif key=='training_previous_state':shape=(30,train_n,4)
        elif key=='training_real_gross':shape=(30,train_n,3)
        elif key=='actual_previous_state' or key.startswith('interval_'):shape=(n,30,4)
        elif key=='candidate_posttrade':shape=(n,30,13,3)
        elif key.startswith('candidate_member_') or key=='candidate_predicted_next_breach':shape=(n,30,13,5)
        elif key.startswith('candidate_'):shape=(n,30,13)
        c.check(value.shape==shape,group['group_id']+'/'+key+'/shape')
        kind='b' if key in booleans else 'i' if key in ints else 'U' if key in ('actions','gate_reason') else 'f'
        c.check(value.dtype.kind==kind,group['group_id']+'/'+key+'/dtype')


def verify_map(mapping,prefix=ROOT):
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty immutable hash closure required')
    for name,digest in mapping.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not isinstance(digest,str) or len(digest)!=64:raise ValueError('Relative SHA256 map required')
        if sha(prefix/path)!=digest:raise ValueError('Frozen evidence changed: '+name)


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
    must={'scripts/audit_state_budget_rebuild.py','tests/test_audit_state_budget_rebuild.py','docs/state_budget_rebuild_audit_plan.json','src/safeinvest/data.py'}
    if not must<=set(a['source_sha256']):raise RuntimeError('Complete independent source/test/plan/loader freeze required')
    verify_map(a['source_sha256']);verify_map(closure);verify_map(m['outputs_sha256'],OUT)
    if str(PLAN.relative_to(ROOT)) not in f['input_sha256'] or f['input_sha256'][str(PLAN.relative_to(ROOT))]!=sha(PLAN):raise RuntimeError('Producer must bind the predeclared independent audit plan')
    names=set(closure)|set(a['source_sha256'])|{str(x.relative_to(ROOT)) for x in (PLAN,PROTOCOL,FREEZE,AUDIT_FREEZE,OUT/'manifest.json')}
    names|={str((OUT/k).relative_to(ROOT)) for k in m['outputs_sha256']}
    return p,m,{name:sha(ROOT/name) for name in sorted(names)}


def source_review():
    import ast
    runner=(ROOT/'scripts/run_state_budget_rebuild.py').read_text();tree=ast.parse(runner)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_paths']
    if len(calls)!=1:raise AssertionError('One fixed guarded development parser required')
    call=calls[0]
    if ast.unparse(call.args[1])!='range(1, 121)' or {k.arg:ast.literal_eval(k.value) for k in call.keywords}.get('allow_final') is not False:raise AssertionError('No final/parser ID override')
    engine=(ROOT/'research/experimental_execution/state_budget_execution.py').read_text();et=ast.parse(engine)
    helper=next(n for n in et.body if isinstance(n,ast.FunctionDef) and n.name=='decide_state_action')
    args={a.arg for a in helper.args.args+helper.args.kwonlyargs}
    if any(name in args for name in ('actual_returns','actual_inflation','actual_ids','future')):raise AssertionError('Actual future/ID cannot enter active helper')
    return dict(status='passed_static_source_review',parser='range(1,121),allow_final=False',
        active_helper_has_no_actual_future_or_scenario_ID_argument=True,
        limitation='Static frozen-source API/timing review plus exact saved previous-state identities; not a formal security proof')


def audit_values(p,m,before,state):
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    state.update(phase='loading_development_only',market_value_access_started=True);writej(ATTEMPT/'attempt.json',state)
    r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
    c=Checks();c.check(ids.tolist()==list(range(1,121)),'exact development identities')
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
    return dict(status='passed',version='state-budget-rebuild-independent-audit-v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(PLAN),audit_source_freeze_sha256=sha(AUDIT_FREEZE),
        checks=c.n,maximum_absolute_numeric_error=c.max_error,counts=counts,fixed_decision_witnesses=witnesses,numeric_binding_ties=c.numeric_binding_ties,
        field_inventory=inventories,sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,source_review=source_review(),
        development_ids_loaded=ids.tolist(),final_scenarios_loaded=False,new_fits=0,producer_modified=False,
        boundary='Independent all-account physics/selection/metrics plus initial and54fixed decision slots. Nonwitness forecast means are saved-gate algebra, not full independent forecast reestimation. Historical adaptive exposure and support/recursive-feasibility limitations remain.')


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
