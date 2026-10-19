"""Annual model liquidity diagnostics without inventing client liabilities."""
from __future__ import annotations
import numpy as np
TOL=1e-12


def bill_liquidation(holdings,cost_rate=.0025):
    """Sell all nonbill assets once, pay destination fee from the transfer.

    Available bill amount = existing bills + sum(nonbills)/(1+cost).
    This uses the contest's annual tradability assumptions; no intrayear
    access, external execution delay, capital protection or withdrawal claim.
    """
    x=np.asarray(holdings,float)
    if x.shape[-1:]!=(3,) or not np.isfinite(x).all() or np.any(x<0) or np.any(x.sum(axis=-1)<=0):
        raise ValueError('Finite positive-total nonnegative holdings in three assets')
    if isinstance(cost_rate,(bool,np.bool_)) or not np.isfinite(cost_rate) or not 0<=cost_rate<1:
        raise ValueError('Finite fee fraction in[0,1)')
    transferred=x[...,:2].sum(axis=-1)/(1+cost_rate)
    return dict(available_bills=x[...,2]+transferred,nominal_fee=cost_rate*transferred,
                transferred=transferred)


def annual_target_liquidity(weights,returns,*,floor,cost_rate=.0025):
    """End-of-year shares after a fixed annual target, one scenario only.

    A current end-of-year realization describes holdings after that year.
    It never selects a target or enters that year's pre-return decision.
    Ratios cancel the common post-fee wealth scale. Hypothetical liquidation
    is evaluated separately and is not charged to the investment path.
    """
    w=np.asarray(weights,float);r=np.asarray(returns,float)
    if w.shape!=(3,) or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1,rtol=0,atol=TOL):
        raise ValueError('Three-asset simplex target')
    if r.ndim!=2 or r.shape[1]!=3 or not len(r) or np.any(r<=-1) or not np.isfinite(r).all():
        raise ValueError('Positive-gross annual three-asset returns')
    if isinstance(floor,(bool,np.bool_)) or not np.isfinite(floor) or not 0<=floor<=1:
        raise ValueError('Normative target bill floor in[0,1]')
    if w[2]<floor-TOL:raise ValueError('Given target violates its declared target floor')
    endpoint=w*(1+r);shares=endpoint/endpoint.sum(axis=1,keepdims=True)
    liquid=bill_liquidation(shares,cost_rate)
    return dict(end_bill_share=shares[:,2],below_target_floor=shares[:,2]<floor-TOL,
        hypothetical_liquidation_drag=liquid['nominal_fee'],hypothetical_available_bill_ratio=liquid['available_bills'])
