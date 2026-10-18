"""One finite-calendar-fan, chance-constrained rollout heuristic.

At each active post-inception state, compare current hold/restore followed
by the SAME frozen annual-until-breach B continuation. Qg is already net of
every executed fee and uses TOTAL remaining real log growth. Feasible
actions have empirical bad count <=floor(.1*N+1e-12); within that set choose
the larger Qg, retaining hold within1e-12. If neither is feasible, explicitly
flag local_model_infeasible and choose fewer bad paths, then higher Qg,
then hold. This least-violation action is not a claim of local feasibility.

There is no price, penalty or positive-growth deployment condition. A
risk-constraint repair may sacrifice predicted growth. The unconditional
calendar fan and repeatedly replanned B continuation are model assumptions;
local feasibility does not establish global-policy or population safety.
External complete-policy training and own-U admission may still Review.
Initial allocation follows Q1 mandatorily and pays its exact fee once.
Actual breach permanently holds; original H, CPI, peak and MDD never reset.
No terminal liquidation, market I/O, fitting or candidate search occurs.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike

from safeinvest.core import execute_target
from .fan_rollout import (NUMERIC_TOL, _advance, _market, _parameters,
                          _recovery, evaluate_actions as _base_evaluate)

LOCAL_ALPHA = .10
POLICY = "constrained_fan_rollout"


def _readonly(value):
    if isinstance(value, dict):
        for entry in value.values():_readonly(entry)
    elif isinstance(value, np.ndarray):value.setflags(write=False)
    return value


def constrained_choices(qg_hold: ArrayLike, qg_restore: ArrayLike,
                        bad_hold: ArrayLike, bad_restore: ArrayLike,
                        fan_n: int) -> dict:
    """Choose one action from paired net Qg and exact bounded integer counts.

    Inputs broadcast over active states; N is a positive common integer.
    Counts are not probabilities. Local alpha=.1 is fixed, never optimized.
    The growth tolerance is in total-log units at every original date. All
    holds have reason 'hold'; local_model_infeasible separately distinguishes
    a least-violation hold from a hold that meets the local empirical cap.
    sacrifice flags executed optional restore with DeltaQg < -1e-12; exact
    unrounded advantages are always returned, including smaller negatives.
    economic_improvement and risk_cost_justification partition recommended
    restores at DeltaQg>1e-12. The latter always strictly reduces bad count;
    it is not a claim of positive unpriced financial gain. repair_meets_cap
    reports whether the chosen restore actually meets the local cap.
    """
    if (isinstance(fan_n, (bool, np.bool_))
            or not isinstance(fan_n, (int, np.integer)) or fan_n < 1):
        raise ValueError("fan_n must be one positive integer")
    qh, qr, kh, kr = np.broadcast_arrays(np.asarray(qg_hold, float),
        np.asarray(qg_restore, float), np.asarray(bad_hold), np.asarray(bad_restore))
    if (not np.isfinite(qh).all() or not np.isfinite(qr).all()
            or kh.dtype.kind not in 'iu' or kr.dtype.kind not in 'iu'
            or np.any(kh < 0) or np.any(kr < 0)
            or np.any(kh > fan_n) or np.any(kr > fan_n)):
        raise ValueError("Require finite net log values and bounded integer bad counts")
    cap = math.floor(LOCAL_ALPHA * int(fan_n) + NUMERIC_TOL)
    fh, fr = kh <= cap, kr <= cap
    infeasible = ~(fh | fr)
    advantage = qr - qh
    if not np.isfinite(advantage).all():raise ValueError("Net log advantage exceeds numerical range")
    growth_better = advantage > NUMERIC_TOL
    # Convert before subtraction: unsigned integer count differences must
    # not wrap. The actual fan has at most120 paths in this research stage.
    count_delta = kr.astype(np.float64) - kh.astype(np.float64)
    restore = ((fr & ~fh) | (fr & fh & growth_better)
               | (infeasible & ((kr < kh) | ((kr == kh) & growth_better))))
    reason = np.full(qh.shape, 'hold', dtype='U32')
    reason = np.where(restore & fh & fr, 'growth_improvement', reason)
    reason = np.where(restore & ~fh & fr, 'risk_constraint_repair', reason)
    reason = np.where(restore & infeasible, 'least_violation_restore', reason)
    detail = np.where(restore, reason, np.where(infeasible,'least_violation_hold',
                      np.where(~fr,'only_hold_model_feasible','both_feasible_growth_or_tie_hold')))
    return _readonly(dict(recommend_restore=restore, gate_reason=reason,
        decision_detail=detail,
        local_breach_cap=cap, hold_model_feasible=fh, restore_model_feasible=fr,
        local_model_infeasible=infeasible,
        predicted_growth_advantage_restore=advantage,
        predicted_breach_count_delta_restore=count_delta,
        restoration_predicted_growth_sacrifice=restore & (advantage < -NUMERIC_TOL),
        economic_improvement=restore & growth_better,
        risk_cost_justification=restore & ~growth_better,
        repair_meets_cap=restore & fr))


def evaluate_actions(holdings: ArrayLike, cpi: ArrayLike, real_peak: ArrayLike,
                     running_mdd: ArrayLike, ever_breached: ArrayLike,
                     training_returns: ArrayLike, training_inflation: ArrayLike, *,
                     year_index: int, target: ArrayLike, L: float,
                     cost_rate: float = .0025, horizon: int | None = None) -> dict:
    """Return frozen B-continuation branch values plus constrained choices.

    Input/state/branch schemas match frozen fan.evaluate_actions. Only the
    new constrained recommendation is used by simulate_constrained; the
    base evaluator's growth/Pareto recommendations remain diagnostic fields.
    No current fee is subtracted from its already-net Qg a second time.
    """
    result = _base_evaluate(holdings, cpi, real_peak, running_mdd, ever_breached,
        training_returns, training_inflation, year_index=year_index, target=target,
        L=L, cost_rate=cost_rate, horizon=horizon)
    result.update(constrained_choices(result['hold']['mean_log_growth'],
        result['restore']['mean_log_growth'], result['hold']['breach_count'],
        result['restore']['breach_count'], result['fan_n']))
    x=np.asarray(holdings,float)
    if x.ndim==1:x=x[None,:]
    post,fee,moved=execute_target(x,target,cost_rate)
    rho=fee/x.sum(axis=-1)
    result.update(current_restore_posttrade_holdings=post,quoted_nominal_fee=fee,
        quoted_nominal_moved=moved,quoted_fee_fraction=rho,
        quoted_log_fee_penalty=-np.log1p(-rho))
    return _readonly(result)


def simulate_constrained(actual_returns: ArrayLike, actual_inflation: ArrayLike,
                         training_returns: ArrayLike, training_inflation: ArrayLike,
                         target: ArrayLike, *, L: float,
                         cost_rate: float = .0025,
                         horizon: int | None = None) -> dict:
    """Execute one declared heuristic and return a squeezed physical arm.

    actual(n,T,3)/(n,T), training(N,Ttrain,3)/(N,Ttrain), target(3,).
    H is original horizon, not restarted remaining years. Initial restore
    is mandatory without a new short-term positive-growth qualification.
    Already-breached accounts hold without evaluating a forecast. The year's
    actual return is read only after the current pre-return action.

    Period diagnostics(n,H): qg_hold/qg_restore, qbad_hold_count/
    qbad_restore_count, forecast_n,local_breach_cap,hold_model_feasible,
    restore_model_feasible,local_model_infeasible,estimate_available,
    predicted_growth_advantage_restore,predicted_breach_count_delta_restore,
    restoration_predicted_growth_sacrifice,economic_improvement,
    risk_cost_justification,repair_meets_cap,gate_reason/decision_reason.
    Inception has diagnostic Q/count/current-fee evidence but its mandatory
    action ignores the local recommendation; the optional economic/repair
    flags and sacrifice areFalse. local_constraint_applied isFalse at
    inception/pending andTrue for optional active decisions. Pending Q/quotes
    are absent: floatsNaN, integers-1, flagsFalse, estimate_availableFalse.
    Inactive sentinels must serialize NULL.
    All ordinary simple/fan physical states, recovery and metrics persist.
    """
    market, pi = _market(actual_returns, actual_inflation, 'actual paths')
    train, train_pi = _market(training_returns, training_inflation, 'training fan')
    requested = market.shape[1] if horizon is None else horizon
    w, loss, rate, h = _parameters(target, L, cost_rate, requested,
                                  min(market.shape[1], train.shape[1]))
    n = len(market); periods, endpoints = (n,h), (n,h+1)
    arm = {key:np.zeros(periods) for key in ('fees','moved','pretrade_wealth',
        'fee_fraction','moved_fraction','pretrade_tv_drift','end_year_tv_drift')}
    arm.update({key:np.zeros(endpoints) for key in ('nominal_wealth','real_wealth',
        'real_peak','current_drawdown','running_mdd')})
    arm.update({key:np.full(periods,np.nan) for key in ('qg_hold','qg_restore',
        'predicted_growth_advantage_restore','predicted_breach_count_delta_restore',
        'quoted_nominal_fee','quoted_nominal_moved','quoted_fee_fraction','quoted_log_fee_penalty')})
    arm.update({key:np.full(periods,-1,dtype=np.int64) for key in ('qbad_hold_count',
        'qbad_restore_count','forecast_n','local_breach_cap')})
    arm.update({key:np.zeros(periods,dtype=bool) for key in ('hold_model_feasible',
        'restore_model_feasible','local_model_infeasible','estimate_available',
        'restoration_predicted_growth_sacrifice','economic_improvement',
        'risk_cost_justification','repair_meets_cap','local_constraint_applied',
        'target_execution','pending_at_decision')})
    arm['holdings'], arm['weights'] = np.zeros(endpoints+(3,)), np.zeros(endpoints+(3,))
    arm['posttrade_holdings'] = np.zeros(periods+(3,))
    arm['ever_breached'] = np.zeros(endpoints,dtype=bool)
    arm['cpi'] = np.ones(endpoints)
    arm['actions'] = np.full(periods,'hold',dtype='U7')
    arm['gate_reason'] = np.full(periods,'',dtype='U40')
    arm['decision_detail'] = np.full(periods,'',dtype='U40')
    arm['first_crossing'] = np.full(n,-1,dtype=np.int64)
    for key in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):
        arm[key] = np.full(n,np.nan)
    arm['holdings'][:,0,2], arm['weights'][:,0,2] = 100.,1.
    for key in ('nominal_wealth','real_wealth','real_peak'):arm[key][:,0] = 100.
    direct = ('hold_model_feasible','restore_model_feasible','local_model_infeasible',
        'predicted_growth_advantage_restore','predicted_breach_count_delta_restore',
        'restoration_predicted_growth_sacrifice','economic_improvement',
        'risk_cost_justification','repair_meets_cap','gate_reason','decision_detail',
        'quoted_nominal_fee','quoted_nominal_moved','quoted_fee_fraction','quoted_log_fee_penalty')
    for year in range(h):
        current, wealth = arm['holdings'][:,year,:], arm['nominal_wealth'][:,year]
        pending = arm['ever_breached'][:,year]
        restore = np.zeros(n,dtype=bool)
        post, fee, moved = current.copy(),np.zeros(n),np.zeros(n)
        arm['gate_reason'][pending,year] = 'already_breached_pending_hold'
        arm['decision_detail'][pending,year] = 'already_breached_pending_hold'
        active = ~pending
        if np.any(active):
            values = evaluate_actions(current[active],arm['cpi'][active,year],
                arm['real_peak'][active,year],arm['running_mdd'][active,year],pending[active],
                train,train_pi,year_index=year,target=w,L=loss,cost_rate=rate,horizon=h)
            chosen=np.ones(np.sum(active),dtype=bool) if year==0 else values['recommend_restore']
            restore[active] = chosen
            arm['qg_hold'][active,year] = values['hold']['mean_log_growth']
            arm['qg_restore'][active,year] = values['restore']['mean_log_growth']
            arm['qbad_hold_count'][active,year] = values['hold']['breach_count']
            arm['qbad_restore_count'][active,year] = values['restore']['breach_count']
            arm['forecast_n'][active,year] = values['fan_n']
            arm['local_breach_cap'][active,year] = values['local_breach_cap']
            arm['estimate_available'][active,year] = True
            arm['local_constraint_applied'][active,year] = year>0
            for key in direct:arm[key][active,year] = values[key]
            if year==0:
                arm['gate_reason'][:,year] = 'mandatory_initial_restore'
                arm['decision_detail'][:,year] = 'mandatory_initial_restore_diagnostic_only'
                for key in ('economic_improvement','risk_cost_justification','repair_meets_cap',
                            'restoration_predicted_growth_sacrifice'):
                    arm[key][:,year]=False
            chosen_ids=np.flatnonzero(active)[chosen]
            post[chosen_ids]=values['current_restore_posttrade_holdings'][chosen]
            fee[chosen_ids]=values['quoted_nominal_fee'][chosen]
            moved[chosen_ids]=values['quoted_nominal_moved'][chosen]
        arm['posttrade_holdings'][:,year] = post
        arm['fees'][:,year],arm['moved'][:,year] = fee,moved
        arm['pretrade_wealth'][:,year] = wealth
        arm['fee_fraction'][:,year],arm['moved_fraction'][:,year] = fee/wealth,moved/wealth
        arm['pretrade_tv_drift'][:,year] = .5*np.abs(arm['weights'][:,year]-w).sum(axis=-1)
        arm['target_execution'][:,year],arm['pending_at_decision'][:,year] = restore,pending
        arm['actions'][:,year] = np.where(restore,'restore','hold')
        after,cp,nominal,real,weights,peak,drawdown,worst,ever = _advance(
            post,1+market[:,year,:],arm['cpi'][:,year],1+pi[:,year],
            arm['real_peak'][:,year],arm['running_mdd'][:,year],pending,loss)
        first = (arm['first_crossing']<0)&ever
        arm['first_crossing'][first] = year+1
        for key,value in (('drawdown_at_trigger',worst),('peak_at_trigger',peak),('real_wealth_at_trigger',real)):
            arm[key][first] = value[first]
        for key,value in (('holdings',after),('cpi',cp),('nominal_wealth',nominal),('real_wealth',real),
            ('weights',weights),('real_peak',peak),('current_drawdown',drawdown),('running_mdd',worst),('ever_breached',ever)):
            arm[key][:,year+1] = value
        arm['end_year_tv_drift'][:,year] = .5*np.abs(weights-w).sum(axis=-1)
    _recovery(arm,loss,h)
    arm['trades'] = arm['moved']>NUMERIC_TOL
    arm['end_weights'] = arm['weights'][:,1:,:]
    arm['decision_reason'] = arm['gate_reason']
    arm['targets'],arm['loss_tolerances'] = w.copy(),np.asarray([loss])
    arm['original_horizon'],arm['cost_rate'],arm['local_alpha'],arm['policy'] = h,rate,LOCAL_ALPHA,POLICY
    arm['trade_count'] = arm['trades'].sum(axis=1).astype(np.int64)
    arm['target_execution_count'] = arm['target_execution'].sum(axis=1).astype(np.int64)
    arm['nominal_fees'],arm['nominal_moved'] = arm['fees'].sum(axis=1),arm['moved'].sum(axis=1)
    arm['local_model_infeasible_decision_count'] = (arm['local_model_infeasible'] & arm['local_constraint_applied']).sum(axis=1).astype(np.int64)
    for reason,key in (('growth_improvement','growth_improvement_count'),('risk_constraint_repair','risk_repair_count'),
                       ('least_violation_restore','least_violation_restore_count')):
        arm[key] = (arm['gate_reason']==reason).sum(axis=1).astype(np.int64)
    arm['growth_sacrifice_count'] = arm['restoration_predicted_growth_sacrifice'].sum(axis=1).astype(np.int64)
    real = arm['real_wealth']; peaks = np.maximum.accumulate(real,axis=1)
    logg = (np.log(real[:,-1])-np.log(100.))/h
    annual = real[:,1:]/real[:,:-1]-1
    arm.update(log_growth=logg,cagr=np.expm1(logg),mdd=np.max((peaks-real)/peaks,axis=1),
        volatility=np.std(annual,axis=1,ddof=1) if h>1 else np.zeros(n),
        terminal_real=real[:,-1].copy(),breach=arm['ever_breached'][:,-1].copy())
    arm['excess'] = np.maximum(arm['mdd']-loss,0)
    return _readonly(arm)
