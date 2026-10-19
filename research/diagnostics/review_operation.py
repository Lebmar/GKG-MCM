"""Saved-path Review accounting and predeclared development references.

No raw data loader, learner, admission rule or final-sample access lives here.
The fixed static references are hypothetical exposures, not new products.
"""
from __future__ import annotations
import math
import numpy as np
from safeinvest.core import execute_target

TOL=1e-12
ABS_TOL=1e-9
REL_TOL=5e-12
ASSETS=('equity','bond','bill')
PHASES=('before_guard','guard_year','after_guard','no_guard')
FEE_FRACTIONS=(0.,.001,.0025,.005,.01)
EXPOSURE_FRACTIONS=(.25,.5,.75,1.)
REFERENCES=('same_target_B','same_target_annual','exact_thirds_annual')


def close(a,b,name):
    if not np.allclose(a,b,atol=ABS_TOL,rtol=REL_TOL,equal_nan=True):raise ValueError('Saved accounting identity failed: '+name)


def financial(real,fees,trades,L=.4):
    real=np.asarray(real,float);h=len(real)-1
    peak=np.maximum.accumulate(real);mdd=float(np.max(1-real/peak))
    return dict(terminal_real=float(real[-1]),log_growth=math.log(real[-1]/real[0])/h,
        cagr=float((real[-1]/real[0])**(1/h)-1),mdd=mdd,breach=mdd-L>TOL,
        nominal_fees=float(np.sum(fees)),trade_fraction=float(np.mean(trades)))


def reconstruct_market(reference):
    post=np.asarray(reference['posttrade_holdings']);after=np.asarray(reference['holdings'])[:,1:]
    cpi=np.asarray(reference['cpi'])
    if post.shape!=after.shape or np.any(post<=0) or np.any(cpi<=0):raise ValueError('Positive three-asset annual reference required')
    gross=after/post;real=gross/(cpi[:,1:]/cpi[:,:-1])[:,:,None]
    if not np.isfinite(gross).all() or np.any(gross<=0):raise ValueError('Finite positive joint gross required')
    return gross,real


def audit_saved_arm(arm):
    """Physical identities and absorbing pending continuation, without a policy call."""
    x=np.asarray(arm['holdings']);p=np.asarray(arm['posttrade_holdings']);cp=np.asarray(arm['cpi']);fees=np.asarray(arm['fees'])
    n,h=fees.shape;g=np.asarray(arm['realized_real_gross'])
    if x.shape!=(n,h+1,3) or p.shape!=(n,h,3) or cp.shape!=(n,h+1) or g.shape!=(n,h,3):raise ValueError('Original fullH shapes required')
    if np.any(x<0) or np.any(p<0) or np.any(cp<=0) or np.any(g<=0):raise ValueError('Positive finite physical inputs required')
    for value in (x,p,cp,g,fees):
        if not np.isfinite(value).all():raise ValueError('Finite physical state required')
    close(x.sum(-1),arm['nominal_wealth'],'nominal wealth');close(x.sum(-1)/cp,arm['real_wealth'],'real wealth')
    close(p.sum(-1)+fees,x[:,:-1].sum(-1),'self financing')
    close(p*g/cp[:,:-1,None],x[:,1:]/cp[:,1:,None],'per-asset realized gross')
    close(np.maximum.accumulate(arm['real_wealth'],axis=1),arm['real_peak'],'original real peak')
    close(np.maximum.accumulate(1-arm['real_wealth']/arm['real_peak'],axis=1),arm['running_mdd'],'full running MDD')
    mask=np.asarray(arm['was_pending_input'],bool)
    if np.any(arm['fees'][mask]!=0) or np.any(arm['trades'][mask]):raise ValueError('Existing pending must have no trading fee or action')
    close(p[mask],x[:,:-1][mask],'existing pending is physical hold')
    for i in range(n):
        guards=np.flatnonzero(arm['guard_triggered'][i]);tau=int(arm['first_guard_trigger'][i])
        if len(guards)>1 or tau!=(int(guards[0])+1 if len(guards) else -1):raise ValueError('First guard convention mismatch')
        if tau>=0 and (not np.all(arm['guard_pending'][i,tau:]) or not np.all(mask[i,tau:])):raise ValueError('Guard must be absorbing from next decision')
    return dict(accounts=n,periods=n*h,existing_pending_periods=int(mask.sum()),guards=int(np.sum(arm['guard_triggered'])))


def phase_rows(arm,ids,*,job_id):
    audit_saved_arm(arm);n,h=arm['fees'].shape
    if len(ids)!=n:raise ValueError('IDs and state rows differ')
    annual=[];segments=[];accounts=[]
    for i,sid in enumerate(ids):
        if not isinstance(sid,(int,np.integer)) or not 1<=sid<=120:raise ValueError('Development IDs only')
        tau=int(arm['first_guard_trigger'][i]);r=arm['real_wealth'][i];cp=arm['cpi'][i]
        rows=[]
        for t in range(h):
            phase='no_guard' if tau<0 else 'before_guard' if t+1<tau else 'guard_year' if t+1==tau else 'after_guard'
            x=arm['holdings'][i,t];p=arm['posttrade_holdings'][i,t];fee=float(arm['fees'][i,t]);v=float(x.sum());weights=p/p.sum()
            fee_log=math.log1p(-fee/v);market_log=math.log(float(weights@arm['realized_real_gross'][i,t]))
            net_log=math.log(float(r[t+1]/r[t]));gains=p*(arm['realized_real_gross'][i,t]-1)/cp[t]
            close(fee_log+market_log,net_log,'annual log fee plus real portfolio growth')
            close(-fee/cp[t]+gains.sum(),r[t+1]-r[t],'annual real-dollar asset gains minus fee')
            row=dict(job_id=job_id,scenario_id=int(sid),period=t+1,phase=phase,guard_year=None if tau<0 else tau,
                existing_pending=bool(arm['was_pending_input'][i,t]),guard_triggered=bool(arm['guard_triggered'][i,t]),
                guard_pending_before=bool(arm['was_guard_pending_input'][i,t]),ordinary_service=bool(arm['automatic_service'][i,t]),
                trade=bool(arm['trades'][i,t]),nominal_fee=fee,real_fee_at_charge=fee/float(cp[t]),
                net_log_growth=net_log,log_fee=fee_log,real_portfolio_log_growth=market_log,
                real_wealth_start=float(r[t]),real_wealth_end=float(r[t+1]),original_peak_end=float(arm['real_peak'][i,t+1]),
                original_current_dd_end=float(arm['current_drawdown'][i,t+1]),running_mdd_start=float(arm['running_mdd'][i,t]),
                running_mdd_end=float(arm['running_mdd'][i,t+1]),gate_reason=str(arm['gate_reason'][i,t]))
            for k,asset in enumerate(ASSETS):row[asset+'_weight']=float(weights[k]);row[asset+'_real_gain']=float(gains[k])
            rows.append(row);annual.append(row)
        close(sum(row['net_log_growth'] for row in rows),math.log(r[-1]/r[0]),'full log growth equals phase sums')
        for phase in PHASES:
            part=[row for row in rows if row['phase']==phase]
            segments.append(dict(job_id=job_id,scenario_id=int(sid),phase=phase,periods=len(part),
                net_log_growth=math.fsum(x['net_log_growth'] for x in part),log_fee=math.fsum(x['log_fee'] for x in part),
                nominal_fees=math.fsum(x['nominal_fee'] for x in part),trades=sum(x['trade'] for x in part),
                mean_equity_weight=float(np.mean([x['equity_weight'] for x in part])) if part else None,
                mean_bond_weight=float(np.mean([x['bond_weight'] for x in part])) if part else None,
                mean_bill_weight=float(np.mean([x['bill_weight'] for x in part])) if part else None,
                running_mdd_start=part[0]['running_mdd_start'] if part else None,running_mdd_end=part[-1]['running_mdd_end'] if part else None,
                max_original_peak_drawdown=max(x['original_current_dd_end'] for x in part) if part else None))
        accounts.append(dict(job_id=job_id,scenario_id=int(sid),first_guard=None if tau<0 else tau,
            first_crossing=None if arm['first_crossing'][i]<0 else int(arm['first_crossing'][i]),
            post_guard_periods=h-tau if tau>=0 else None,
            review_debit_within_H=bool(0<tau<h),**financial(r,arm['fees'][i],arm['trades'][i])))
    return annual,segments,accounts


def debit_review_cost(real,peak,running_mdd,holdings,cpi,tau,q,*,L=.4):
    """One proportional service debit at endpoint tau, before decision tau+1.

    The pre-debit trigger-year peak and MDD remain in history. No human benefit,
    liquidation, new forecast, new market or artificial waiting duration is added.
    """
    if not 0<=q<1:raise ValueError('Fee scenario must be a fraction in[0,1)')
    r=np.asarray(real,float);p=np.asarray(peak,float);m=np.asarray(running_mdd,float);h=len(r)-1
    applicable=bool(0<tau<h);out=r.copy();peaks=p.copy();mdds=m.copy();fee=0.
    if applicable and q:
        fee=float(q*np.asarray(holdings)[tau].sum());out[tau:]*=1-q
        prior_peak=float(p[tau]);prior_mdd=float(m[tau])
        for u in range(tau,h+1):
            prior_peak=max(prior_peak,float(out[u]));prior_mdd=max(prior_mdd,1-float(out[u])/prior_peak)
            peaks[u]=prior_peak;mdds[u]=prior_mdd
    crossing=np.flatnonzero(mdds-L>TOL)
    return dict(real_wealth=out,real_peak=peaks,running_mdd=mdds,applicable=applicable,
        review_fee_nominal=fee,review_fee_real=fee/float(cpi[tau]) if applicable else 0.,
        debit_endpoint=tau if applicable else None,first_crossing=int(crossing[0]) if len(crossing) else None,
        breach=bool(mdds[-1]-L>TOL))


def break_even(main_terminal,reference_terminal,nominal_at_review,*,applicable):
    if not applicable:return dict(signed_fraction_threshold=None,max_nonnegative_fee_fraction=None,max_nominal_fee=None,zero_fee_already_worse=None)
    threshold=1-float(reference_terminal)/float(main_terminal)
    available=threshold>=0
    return dict(signed_fraction_threshold=threshold,max_nonnegative_fee_fraction=threshold if available else None,
        max_nominal_fee=threshold*float(nominal_at_review) if available else None,zero_fee_already_worse=not available)


def static_reference(nominal_gross,cpi,target,*,rule,cost_rate=.0025,L=.4):
    """Independent scalar-time accounting for fixed initial hold or annual restore."""
    g=np.asarray(nominal_gross,float);cp=np.asarray(cpi,float);w=np.asarray(target,float);w=w/w.sum()
    n,h,_=g.shape
    if rule not in ('hold','annual') or cp.shape!=(n,h+1) or np.any(g<=0) or np.any(cp<=0):raise ValueError('Fixed positive joint paths and rule required')
    a={k:np.zeros((n,h)) for k in ('fees','moved','pretrade_wealth','fee_fraction','moved_fraction','pretrade_tv_drift','end_year_tv_drift')}
    a.update({k:np.zeros((n,h+1)) for k in ('nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd')})
    a.update(holdings=np.zeros((n,h+1,3)),weights=np.zeros((n,h+1,3)),posttrade_holdings=np.zeros((n,h,3)),
        ever_breached=np.zeros((n,h+1),bool),target_execution=np.zeros((n,h),bool),pending_at_decision=np.zeros((n,h),bool),
        actions=np.full((n,h),'hold',dtype='U7'),first_crossing=np.full(n,-1,dtype=np.int64),cpi=cp.copy())
    for name in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):a[name]=np.full(n,np.nan)
    a['holdings'][:,0,2]=100.;a['weights'][:,0,2]=1.
    for name in ('nominal_wealth','real_wealth','real_peak'):a[name][:,0]=100.
    for t in range(h):
        x=a['holdings'][:,t];v=x.sum(-1);post=x.copy()
        if t==0 or rule=='annual':
            post,fee,moved=execute_target(x,np.broadcast_to(w,x.shape),cost_rate)
            a['fees'][:,t]=fee;a['moved'][:,t]=moved;a['target_execution'][:,t]=True;a['actions'][:,t]='restore'
        a['posttrade_holdings'][:,t]=post;a['pretrade_wealth'][:,t]=v
        a['fee_fraction'][:,t]=a['fees'][:,t]/v;a['moved_fraction'][:,t]=a['moved'][:,t]/v
        a['pretrade_tv_drift'][:,t]=.5*np.abs(a['weights'][:,t]-w).sum(-1)
        after=post*g[:,t];nominal=after.sum(-1);real=nominal/cp[:,t+1];peak=np.maximum(a['real_peak'][:,t],real)
        dd=np.maximum(1-real/peak,0);mdd=np.maximum(a['running_mdd'][:,t],dd);ever=a['ever_breached'][:,t]|(mdd-L>TOL)
        first=(a['first_crossing']<0)&ever;a['first_crossing'][first]=t+1
        for key,values in (('drawdown_at_trigger',mdd),('peak_at_trigger',peak),('real_wealth_at_trigger',real)):a[key][first]=values[first]
        for key,value in (('holdings',after),('weights',after/nominal[:,None]),('nominal_wealth',nominal),('real_wealth',real),
                ('real_peak',peak),('current_drawdown',dd),('running_mdd',mdd),('ever_breached',ever)):a[key][:,t+1]=value
        a['end_year_tv_drift'][:,t]=.5*np.abs(a['weights'][:,t+1]-w).sum(-1)
    triggered=a['first_crossing']>=0;endpoint=np.arange(h+1)[None,:]
    recovery=triggered[:,None]&(endpoint>a['first_crossing'][:,None])&(a['real_wealth']>=a['peak_at_trigger'][:,None])
    recovered=recovery.any(1);where=recovery.argmax(1)
    a.update(triggered=triggered,recovered=recovered,recovery_endpoint=np.where(recovered,where,-1),
        recovery_time=np.where(recovered,where-a['first_crossing'],-1),right_censored=triggered&~recovered,
        available_followup=np.where(triggered,h-a['first_crossing'],-1),
        observed_followup=np.where(recovered,where-a['first_crossing'],np.where(triggered,h-a['first_crossing'],-1)),
        added_severity=np.where(triggered,np.maximum(a['running_mdd'][:,-1]-a['drawdown_at_trigger'],0),np.nan),
        terminal_excess=np.maximum(a['running_mdd'][:,-1]-L,0),trades=a['moved']>TOL,end_weights=a['weights'][:,1:].copy())
    return a


def compare_all_reference_fields(actual,expected):
    if set(actual)!=set(expected):raise ValueError('All old annual-reference fields must be reconstructed')
    maximum=0.
    for name in expected:
        x,y=np.asarray(actual[name]),np.asarray(expected[name])
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Original shape/dtype mismatch: '+name)
        if x.dtype.kind in 'bUSiu':
            if not np.array_equal(x,y):raise ValueError('Original discrete field mismatch: '+name)
        else:
            close(x,y,'all old reference fields '+name);finite=np.isfinite(x)&np.isfinite(y)
            if finite.any():maximum=max(maximum,float(np.max(np.abs(x[finite]-y[finite]))))
    return dict(fields=len(expected),maximum_absolute_difference=maximum,absolute_tolerance=ABS_TOL,relative_tolerance=REL_TOL)
