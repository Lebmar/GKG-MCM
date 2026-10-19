"""Pure saved-path algebra; no loaders, execution policies or forecast fitting."""
from __future__ import annotations
import numpy as np


def original_metrics(real, L=.4):
    a=np.asarray(real,float)
    if a.ndim!=2 or a.shape[1]<2 or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('Positive complete original real paths required')
    if not np.allclose(a[:,0],100.,atol=3e-10,rtol=0):raise ValueError('Original time0 wealth100 required')
    peak=np.maximum.accumulate(a,axis=1);dd=1-a/peak;mdd=np.maximum.accumulate(dd,axis=1)
    events=mdd-L>1e-12
    first=np.where(events.any(axis=1),events.argmax(axis=1),-1)
    log=np.log(a[:,-1]/100.)/(a.shape[1]-1)
    return dict(real=a,peak=peak,drawdown=dd,running_mdd=mdd,mdd=mdd[:,-1],
                first_crossing=first,breach=events[:,-1],log_growth=log,cagr=np.expm1(log))


def four_corners(nominal_base,cpi_base,nominal_shift,cpi_shift,L=.4):
    vb,cb,vs,cs=map(lambda x:np.asarray(x,float),(nominal_base,cpi_base,nominal_shift,cpi_shift))
    if len({x.shape for x in (vb,cb,vs,cs)})!=1 or any(np.any(x<=0) or not np.isfinite(x).all() for x in (vb,cb,vs,cs)):
        raise ValueError('Aligned positive complete nominal/CPI paths required')
    if not np.allclose(cb[:,0],1.,atol=1e-14,rtol=0) or not np.allclose(cs[:,0],1.,atol=1e-14,rtol=0):
        raise ValueError('Original CPI1 required')
    values={k:original_metrics(v/c,L) for k,v,c in [('BB',vb,cb),('BS',vb,cs),('SB',vs,cb),('SS',vs,cs)]}
    nominal=np.log(vs/vb);inflation=-np.log(cs/cb)
    if not np.allclose(np.log(values['SS']['real']/values['BB']['real']),nominal+inflation,atol=3e-12,rtol=5e-13):
        raise AssertionError('Log nominal/CPI identity')
    bb,bs,sb,ss=[values[k]['mdd'] for k in ('BB','BS','SB','SS')]
    return dict(corners=values,log_nominal_path_component=nominal,log_CPI_component=inflation,
                MDD_CPI_first=bs-bb,MDD_nominal_after_CPI=ss-bs,
                MDD_nominal_first=sb-bb,MDD_CPI_after_nominal=ss-sb,
                MDD_interaction=ss-sb-bs+bb,MDD_total=ss-bb)


def initial_full_target(target,cost_rate):
    w=np.asarray(target,float);c=float(cost_rate)
    if w.shape!=(3,) or np.any(w<=0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1.,atol=1e-12,rtol=0) or not np.isfinite(c) or c<0:
        raise ValueError('Positive original three-asset target and nonnegative fee required')
    y=100*w/(1+c*(1-w[2]));fee=c*(y[0]+y[1])
    if not np.isclose(y.sum()+fee,100.,atol=3e-12,rtol=0):raise AssertionError('Initial self-finance')
    return y,fee


def budget(real,peak,rho,L=.4):
    real,peak=np.asarray(real,float),np.asarray(peak,float)
    floor=(1-float(L))*peak
    return floor,np.maximum(floor,real-float(rho)*(real-floor))


def directional_minimum(support,post,cpi):
    s,y,c=np.asarray(support,float),np.asarray(post,float),np.asarray(cpi,float)
    if s.ndim!=2 or s.shape[1]!=3 or y.ndim!=2 or y.shape[1]!=3 or c.shape!=(len(y),) or len(s)==0:
        raise ValueError('Joint support(N,3), post(n,3), CPI(n) required')
    if not np.isfinite(s).all() or not np.isfinite(y).all() or not np.isfinite(c).all() or np.any(s<=0) or np.any(y<0) or np.any(c<=0):
        raise ValueError('Positive gross/CPI, finite nonnegative holdings required')
    wealth=(s@y.T)/c[None,:];which=wealth.argmin(axis=0)
    return wealth[which,np.arange(len(y))],which


def one_step_accounting(pre,post,fee,cpi_pre,cpi_next,nominal_gross):
    x,y,f,cp,cn,g=map(lambda v:np.asarray(v,float),(pre,post,fee,cpi_pre,cpi_next,nominal_gross))
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or g.shape!=x.shape or any(v.shape!=(len(x),) for v in (f,cp,cn)):
        raise ValueError('Aligned complete one-step physical quantities required')
    if np.any(cp<=0) or np.any(cn<=0) or np.any(g<=0) or not all(np.isfinite(v).all() for v in (x,y,f,cp,cn,g)):
        raise ValueError('Finite positive gross/CPI required')
    v=x.sum(axis=1)
    if not np.allclose(y.sum(axis=1)+f,v,atol=3e-10,rtol=5e-13):raise AssertionError('Fee must be paid once')
    pi=cn/cp-1;realgross=g/(cn/cp)[:,None]
    next_real=(y*g).sum(axis=1)/cn;change=next_real-v/cp
    asset_real=y/cp[:,None]*(realgross-1)
    fee_current=-f/cp
    nominal_asset_gain=(y*(g-1)).sum(axis=1)/cn
    nominal_fee=-f/cn;inflation_erosion=-v*pi/cn
    if not np.allclose(change,asset_real.sum(axis=1)+fee_current,atol=3e-10,rtol=5e-13):raise AssertionError('Real asset P&L identity')
    if not np.allclose(change,nominal_asset_gain+nominal_fee+inflation_erosion,atol=3e-10,rtol=5e-13):raise AssertionError('Nominal/inflation P&L identity')
    return dict(next_real=next_real,change=change,asset_real_contribution=asset_real,fee_current_real=fee_current,
                nominal_asset_gain=nominal_asset_gain,nominal_fee_at_next_CPI=nominal_fee,
                inflation_erosion=inflation_erosion,real_gross=realgross)
