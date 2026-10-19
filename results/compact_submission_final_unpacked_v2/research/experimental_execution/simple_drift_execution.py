"""Pure fixed-target TV-band execution with absorbing pending hold.

This is a simplified execution comparator, not a benefit-cost trading model
or a final Q2 strategy. It fits no threshold, target, forecast or risk price
and performs no file or market-data I/O. Initial admission is external.

The original pre-fee account is100/all bills. At the start of year1 it
restores to the caller's frozen target. Thereafter, before observing that
year's return, an active account restores iff pre-return half-L1 weight
drift is strictly greater than its declared band. Threshold0 deliberately
restores every active year, including exact zero drift, to reproduce the
annual-until-breach positive control's action/execution flags exactly.

At the first annual endpoint with running real MDD-L>1e-12, the account
becomes permanently pending. Only future decisions hold; no fee or action
from the crossing year is reversed. CPI, original horizon, peak, running
MDD and absorbing event history never reset. There is no terminal trade.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target

NUMERIC_TOL=1e-12
DEFAULT_THRESHOLDS=(0.,.05,.10)


def _threshold(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
        raise ValueError('Threshold must be one declared real TV band, not a boolean')
    value=float(value)
    if not np.isfinite(value) or value not in DEFAULT_THRESHOLDS:
        raise ValueError('Only the predeclared TV thresholds0,.05,.10 are supported')
    return value


def restoration_mask(pretrade_tv: ArrayLike,pending: ArrayLike,threshold: float,*,initial: bool=False):
    """Pre-return action rule; threshold equality holds, without extra tolerance.

    Threshold0 is the explicit annual positive control. It counts a target
    execution even when the exact self-financing executor finds no transfer;
    actual trades and fees still require actual movement.
    """
    delta=_threshold(threshold)
    drift=np.asarray(pretrade_tv,float);flag=np.asarray(pending)
    if (not isinstance(initial,(bool,np.bool_)) or flag.dtype.kind!='b'
            or drift.shape!=flag.shape or not np.isfinite(drift).all()
            or np.any(drift<0) or np.any(drift>1)):
        raise ValueError('Require matching finite TV[0,1]/boolean pending arrays and a boolean initial flag')
    return ~flag & (bool(initial) or delta==0. or (drift>delta))


def _inputs(returns,inflation,target,threshold,L,horizon,cost_rate):
    market,pi=np.asarray(returns,float),np.asarray(inflation,float)
    w=np.asarray(target,float)
    if (market.ndim!=3 or market.shape[-1]!=3 or min(market.shape[:2])<1
            or pi.shape!=market.shape[:2] or not np.isfinite(market).all()
            or not np.isfinite(pi).all() or np.any(market<=-1) or np.any(pi<=-1)):
        raise ValueError('Require complete finite nonempty(n,T,3)/(n,T) positive-gross market arrays')
    if (w.shape!=(3,) or not np.isfinite(w).all() or np.any(w<0)
            or not np.isclose(w.sum(),1.,atol=NUMERIC_TOL,rtol=0)):
        raise ValueError('Target must be one finite long-only three-asset simplex point')
    w=w/w.sum();delta=_threshold(threshold)
    if (isinstance(L,(bool,np.bool_)) or not isinstance(L,Real)
            or not np.isfinite(L) or not 0<=L<=1):
        raise ValueError('L must be a finite scalar loss fraction in[0,1]')
    if (isinstance(cost_rate,(bool,np.bool_)) or not isinstance(cost_rate,Real)
            or not np.isfinite(cost_rate) or not 0<=cost_rate<1):
        raise ValueError('Cost rate must be a finite scalar fraction in[0,1)')
    h=market.shape[1] if horizon is None else horizon
    if (isinstance(h,(bool,np.bool_)) or not isinstance(h,(int,np.integer))
            or not 1<=h<=market.shape[1]):
        raise ValueError('Original horizon must be an integer in[1,T]')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        cpi=np.column_stack((np.ones(len(market)),np.cumprod(1+pi[:,:h],axis=1)))
    if not np.isfinite(cpi).all() or np.any(cpi<=0):
        raise ValueError('Cumulative CPI exceeds the finite positive numeric range')
    return market[:,:h],w,delta,float(L),int(h),float(cost_rate),cpi


def simulate_drift(returns: ArrayLike,inflation: ArrayLike,target: ArrayLike,
                   threshold: float,*,L: float=.4,horizon: int|None=None,
                   cost_rate: float=.0025) -> dict:
    """Return one squeezed actual arm, compatible with joint arm/trace helpers.

    Input returns/inflation(n,T,3)/(n,T); target(3,); threshold0,.05,.10.
    This always creates a new original-H account; it cannot reset a saved
    client. Targets, admission and threshold ordering are the caller's fixed
    protocol, never inferred here. H defaults to the supplied original T;
    one-year synthetic paths are supported without a sample-volatility claim.

    Endpoints(n,H+1): nominal_wealth,real_wealth,real_peak,current_drawdown,
    running_mdd,ever_breached,cpi. Holdings/weights add the asset axis3.
    Periods(n,H): fees,moved,pretrade_wealth,fee_fraction,moved_fraction,
    pretrade_tv_drift,end_year_tv_drift,actions,target_execution,trades,
    pending_at_decision,decision_reason. Posttrade holdings(n,H,3).
    'trades' means actual moved>1e-12; 'target_execution' is the restore
    decision, which can execute no transfer when already at the target.

    Per-path first_crossing=1..H or-1, all trigger/recovery/censor fields
    follow the old pending schema: absent trigger values areNaN and missing
    times are-1. Serialize inapplicable fields asNULL, never zero risk/time.
    Reaching the saved trigger peak records recovery but cannot end pending.
    Extra per-path metrics/counts summarize these saved physical paths.
    All returned ndarray values are read-only; no forecast/Q arrays exist.
    """
    market,w,delta,loss,h,rate,cpi=_inputs(returns,inflation,target,threshold,L,horizon,cost_rate)
    n=len(market);periods=(n,h);endpoints=(n,h+1)
    arm={key:np.zeros(periods) for key in ('fees','moved','pretrade_wealth','fee_fraction','moved_fraction','pretrade_tv_drift','end_year_tv_drift')}
    arm.update({key:np.zeros(endpoints) for key in ('nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd')})
    arm['holdings']=np.zeros(endpoints+(3,));arm['weights']=np.zeros(endpoints+(3,))
    arm['posttrade_holdings']=np.zeros(periods+(3,))
    arm['ever_breached']=np.zeros(endpoints,dtype=bool)
    arm['target_execution']=np.zeros(periods,dtype=bool)
    arm['pending_at_decision']=np.zeros(periods,dtype=bool)
    arm['actions']=np.full(periods,'hold',dtype='U7')
    arm['decision_reason']=np.full(periods,'',dtype='U40')
    arm['first_crossing']=np.full(n,-1,dtype=np.int64)
    for key in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):arm[key]=np.full(n,np.nan)
    arm['holdings'][:,0,2]=100.;arm['weights'][:,0,2]=1.
    for key in ('nominal_wealth','real_wealth','real_peak'):arm[key][:,0]=100.
    targets=np.broadcast_to(w,(n,3))
    for t in range(h):
        current=arm['holdings'][:,t,:];wealth=arm['nominal_wealth'][:,t]
        drift=.5*np.abs(arm['weights'][:,t,:]-w).sum(axis=-1)
        pending=arm['first_crossing']>=0
        restore=restoration_mask(drift,pending,delta,initial=t==0)
        post=current.copy();fee=np.zeros(n);moved=np.zeros(n)
        if np.any(restore):
            post[restore],fee[restore],moved[restore]=execute_target(current[restore],targets[restore],rate)
        arm['posttrade_holdings'][:,t,:]=post
        arm['fees'][:,t],arm['moved'][:,t]=fee,moved
        arm['pretrade_wealth'][:,t]=wealth
        arm['fee_fraction'][:,t],arm['moved_fraction'][:,t]=fee/wealth,moved/wealth
        arm['pretrade_tv_drift'][:,t]=drift
        arm['target_execution'][:,t],arm['pending_at_decision'][:,t]=restore,pending
        arm['actions'][:,t]=np.where(restore,'restore','hold')
        reason=np.where(restore,'tv_exceeds_threshold_restore','tv_within_threshold_hold').astype('U40')
        if delta==0.:reason[:]='annual_restore_threshold_zero'
        if t==0:reason[:]='mandatory_initial_restore'
        reason[pending]='already_breached_pending_hold'
        arm['decision_reason'][:,t]=reason
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            after=post*(1+market[:,t,:]);nominal=after.sum(axis=-1)
            real=nominal/cpi[:,t+1];end_weights=after/nominal[:,None]
        if (not np.isfinite(after).all() or not np.isfinite(real).all()
                or np.any(nominal<=0) or np.any(real<=0) or not np.isfinite(end_weights).all()):
            raise ValueError('Executed holdings or wealth exceeds the finite positive numeric range')
        peak=np.maximum(arm['real_peak'][:,t],real)
        dd=np.maximum(1-real/peak,0.)
        worst=np.maximum(arm['running_mdd'][:,t],dd)
        ever=arm['ever_breached'][:,t] | (worst-loss>NUMERIC_TOL)
        first=(arm['first_crossing']<0)&ever
        arm['first_crossing'][first]=t+1
        for key,value in (('drawdown_at_trigger',worst),('peak_at_trigger',peak),('real_wealth_at_trigger',real)):arm[key][first]=value[first]
        for key,value in (('holdings',after),('weights',end_weights),('nominal_wealth',nominal),('real_wealth',real),
                          ('real_peak',peak),('current_drawdown',dd),('running_mdd',worst),('ever_breached',ever)):
            if key in ('holdings','weights'):arm[key][:,t+1,:]=value
            else:arm[key][:,t+1]=value
        arm['end_year_tv_drift'][:,t]=.5*np.abs(end_weights-w).sum(axis=-1)
    triggered=arm['first_crossing']>=0;times=np.arange(h+1)[None,:]
    possible=triggered[:,None]&(times>arm['first_crossing'][:,None])&(arm['real_wealth']>=arm['peak_at_trigger'][:,None])
    recovered=possible.any(axis=-1);endpoint=possible.argmax(axis=-1)
    arm['triggered'],arm['recovered']=triggered,recovered
    arm['recovery_endpoint']=np.where(recovered,endpoint,-1).astype(np.int64)
    arm['recovery_time']=np.where(recovered,endpoint-arm['first_crossing'],-1).astype(np.int64)
    arm['right_censored']=triggered&~recovered
    arm['available_followup']=np.where(triggered,h-arm['first_crossing'],-1).astype(np.int64)
    arm['observed_followup']=np.where(recovered,arm['recovery_time'],arm['available_followup']).astype(np.int64)
    arm['added_severity']=np.where(triggered,np.maximum(arm['running_mdd'][:,-1]-arm['drawdown_at_trigger'],0),np.nan)
    arm['terminal_excess']=np.maximum(arm['running_mdd'][:,-1]-loss,0)
    arm['trades']=arm['moved']>NUMERIC_TOL
    arm['end_weights']=arm['weights'][:,1:,:]
    arm['cpi']=cpi.copy();arm['targets']=w.copy();arm['loss_tolerances']=np.asarray([loss])
    arm['threshold']=delta;arm['original_horizon']=h;arm['cost_rate']=rate
    arm['trade_count']=arm['trades'].sum(axis=1).astype(np.int64)
    arm['target_execution_count']=arm['target_execution'].sum(axis=1).astype(np.int64)
    arm['nominal_fees']=arm['fees'].sum(axis=1);arm['nominal_moved']=arm['moved'].sum(axis=1)
    real=arm['real_wealth'];peaks=np.maximum.accumulate(real,axis=1)
    logg=(np.log(real[:,-1])-np.log(100.))/h
    annual=real[:,1:]/real[:,:-1]-1
    arm.update(log_growth=logg,cagr=np.expm1(logg),mdd=np.max((peaks-real)/peaks,axis=1),
        volatility=np.std(annual,axis=1,ddof=1) if h>1 else np.zeros(n),terminal_real=real[:,-1].copy(),
        breach=arm['ever_breached'][:,-1].copy())
    arm['excess']=np.maximum(arm['mdd']-loss,0)
    for value in arm.values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    return arm
