"""Causal, cost-aware decisions using a frozen joint block scenario model."""
from __future__ import annotations
import numpy as np
from .core import execute_target


def forecast_bank(returns, inflation, size=256, block_length=5, seed=1729):
    """Joint real-return blocks from fitting paths only; never separate assets.

    Six independently sampled five-year blocks form each 30-year projection.
    This is an empirical stationarity assumption, not a market forecast claim.
    """
    real_gross = (1 + np.asarray(returns)) / (1 + np.asarray(inflation))[..., None]
    n, years, _ = real_gross.shape
    rng = np.random.default_rng(seed)
    pieces = []
    for _ in range(int(np.ceil(30 / block_length))):
        scene = rng.integers(n, size=size)
        start = rng.integers(years - block_length + 1, size=size)
        pieces.append(real_gross[scene[:, None], start[:, None] + np.arange(block_length), :])
    return np.cumprod(np.concatenate(pieces, axis=1)[:, :30], axis=1)


def target_family(base, dynamic=False):
    base = np.asarray(base, float)
    if not dynamic:
        return base[None, :]
    targets = []
    for delta in [-.10, -.05, 0., .05, .10]:
        equity = base[0] + delta
        if not 0 <= equity < 1 or 1-base[0] <= 1e-12:
            continue
        other = base[1:] * ((1-equity)/(1-base[0]))
        w = np.r_[equity, other]
        if min(w) >= -1e-12 and w[2] >= .10-1e-12:
            targets.append(w)
    return np.asarray(targets)


def projected_score(holdings, post, bank, remaining, peak_ratio, past_mdd, risk_aversion=4.):
    """Expected log terminal growth minus full-path drawdown penalty.

    Future holdings drift without additional forecast trades. Realized strategy
    decisions are recomputed each year. Fees are already in post; no second fee.
    """
    wealth = holdings.sum(axis=-1)
    allocations = post / wealth[:, None]
    relative = np.einsum('na,kta->nkt', allocations, bank[:, :remaining, :], optimize=True)
    # Annual observations match metric_arrays; costs reduce each future value,
    # but an intra-year fee dip later recovered by returns is not a new point.
    peaks = np.maximum(np.maximum.accumulate(relative,axis=2), peak_ratio[:,None,None])
    drawdown = np.maximum(np.max(1-relative/peaks,axis=2), past_mdd[:,None])
    return np.mean(np.log(relative[:,:,-1])-risk_aversion*drawdown,axis=1)


def simulate(returns, inflation, base_target, mode='annual', bank=None,
             cost_rate=.0025, epsilon=.002, review_threshold=None, risk_aversion=4.):
    """Simulate decisions before each year's returns; 100 initially in bills.

    Modes: annual (unconditional reference), gate, dynamic. Review is permanent
    buy-and-hold pending human feedback; pending years remain in denominators.
    """
    returns = np.asarray(returns,float)
    inflation = np.asarray(inflation,float)
    n,H,_ = returns.shape
    if inflation.shape != (n,H) or mode not in ('annual','gate','dynamic'):
        raise ValueError('invalid path shapes or policy mode')
    if mode != 'annual' and bank is None:
        raise ValueError('cost-aware policies require a fitting-only forecast bank')
    x = np.zeros((n,3)); x[:,2] = 100.
    wealth = np.full((n,H+1),100.)
    weights = np.zeros((n,H,3)); targets = np.zeros_like(weights)
    costs=np.zeros((n,H)); moved=np.zeros((n,H)); trades=np.zeros((n,H),bool)
    pending=np.zeros((n,H),bool); benefit=np.zeros((n,H))
    price=np.ones(n); peak=np.full(n,100.); past_mdd=np.zeros(n)
    active_target=np.broadcast_to(np.asarray(base_target,float),(n,3)).copy()
    family=target_family(base_target,mode=='dynamic')
    locked=np.zeros(n,bool)
    for t in range(H):
        real=x.sum(axis=1)/price
        current_dd=1-real/peak
        if review_threshold is not None and t>0:
            locked |= current_dd >= review_threshold
        pending[:,t]=locked
        if t==0 or mode=='annual':
            proposed,c,m=execute_target(x,active_target,cost_rate)
            accept=~locked
            if t==0 and mode!='annual':
                hold_score=projected_score(x,x,bank,H,peak/real,past_mdd,risk_aversion)
                trade_score=projected_score(x,proposed,bank,H,peak/real,past_mdd,risk_aversion)
                benefit[:,t]=trade_score-hold_score
                accept &= trade_score>hold_score+epsilon
        else:
            ratio=peak/real
            hold_score=projected_score(x,x,bank,H-t,ratio,past_mdd,risk_aversion)
            best_score=hold_score.copy(); proposed=x.copy(); c=np.zeros(n);m=np.zeros(n)
            chosen=active_target.copy()
            for target in family:
                post,fee,amount=execute_target(x,target,cost_rate)
                score=projected_score(x,post,bank,H-t,ratio,past_mdd,risk_aversion)
                eligible=np.abs(active_target[:,0]-target[0])<=.05000000001
                improve=(score>best_score)&eligible
                best_score=np.where(improve,score,best_score)
                proposed=np.where(improve[:,None],post,proposed)
                c=np.where(improve,fee,c); m=np.where(improve,amount,m)
                chosen=np.where(improve[:,None],target,chosen)
            benefit[:,t]=best_score-hold_score
            accept=(best_score>hold_score+epsilon)&(~locked)
            active_target=np.where(accept[:,None],chosen,active_target)
        x=np.where(accept[:,None],proposed,x)
        costs[:,t]=np.where(accept,c,0.);moved[:,t]=np.where(accept,m,0.)
        trades[:,t]=moved[:,t]>1e-9
        weights[:,t]=x/x.sum(axis=1)[:,None]
        targets[:,t]=active_target
        x=x*(1+returns[:,t,:]); price*=1+inflation[:,t]
        wealth[:,t+1]=x.sum(axis=1)
        real=wealth[:,t+1]/price;peak=np.maximum(peak,real)
        past_mdd=np.maximum(past_mdd,1-real/peak)
    return dict(wealth=wealth,weights=weights,targets=targets,costs=costs,moved=moved,
                trades=trades,pending=pending,benefit=benefit)
