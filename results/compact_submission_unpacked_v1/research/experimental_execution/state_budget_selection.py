"""Whole-policy development selection for the state-budget rebuild.

This is an empirical service convention, not a confidence bound. Every
candidate is evaluated on whole inner held-out paths, including subsequent
funded review and its remaining market exposure. Initial abstentions are NULL.
"""
from __future__ import annotations

import math
import numpy as np

from safeinvest_research.metrics import discrete_cvar

TOL = 1e-12
CANDIDATES = tuple(
    dict(candidate_id=f"release_{release:g}_{forecast}", release_fraction=release,
         forecast_rule=forecast, order=order)
    for order, (release, forecast) in enumerate(
        (r, f) for r in (.5, .75, 1.) for f in ("uniform", "ensemble_min")))


def summarize_records(rows):
    """All-original population and conditional invested-path outcomes."""
    rows = list(rows)
    ids = [int(x['scenario_id']) for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Each original scenario must occur exactly once in a summary')
    financial = [r for r in rows if r['assigned']]
    outside=sum(r.get('review_reason')=='not_balanced_case' for r in rows)
    q1_review=sum(r.get('review_reason')=='q1_review' for r in rows)
    result = dict(original_n=len(rows), financial_n=len(financial),
                  initial_noninvested_n=len(rows)-len(financial),
                  not_balanced_n=outside,q1_review_n=q1_review,
                  initial_execution_review_n=len(rows)-len(financial)-outside-q1_review,
                  initial_coverage=len(financial)/len(rows) if rows else None)
    keys = ['breach_count','breach_fraction','mean_log_growth','median_cagr','p10_cagr',
            'mean_mdd','median_mdd','p90_mdd','cvar90_mdd','mean_volatility',
            'mean_nominal_fees','mean_trade_fraction','support_review_n',
            'mean_automatic_fraction','mean_protection_trade_fraction']
    result.update(dict.fromkeys(keys))
    if not financial:
        return result
    def a(key):
        values = np.array([r[key] for r in financial], float)
        if not np.all(np.isfinite(values)):
            raise ValueError('Invested paths require finite complete-H metrics')
        return values
    mdd, cagr = a('mdd'), a('cagr')
    result.update(breach_count=sum(bool(r['breach']) for r in financial),
        breach_fraction=sum(bool(r['breach']) for r in financial)/len(financial),
        mean_log_growth=float(a('log_growth').mean()),median_cagr=float(np.median(cagr)),
        p10_cagr=float(np.quantile(cagr,.1,method='inverted_cdf')),
        mean_mdd=float(mdd.mean()),median_mdd=float(np.median(mdd)),
        p90_mdd=float(np.quantile(mdd,.9,method='inverted_cdf')),
        cvar90_mdd=float(discrete_cvar(mdd)),mean_volatility=float(a('volatility').mean()),
        mean_nominal_fees=float(a('nominal_fees').mean()),
        mean_trade_fraction=float(a('trade_fraction').mean()),
        support_review_n=sum(bool(r.get('support_review',False)) for r in financial),
        mean_automatic_fraction=float(np.mean([r.get('automatic_fraction',1.) for r in financial])),
        mean_protection_trade_fraction=float(np.mean([r.get('protection_trade_fraction',0.) for r in financial])))
    return result


def select_inner(candidate_rows, reference_rows, *, alpha=.1, candidate_specs=CANDIDATES):
    """Select using inner held-out outcomes only, with no refit rescue.

    Feasible: nonzero initial coverage, event fraction<=alpha, and mean/CVaR90
    MDD no greater than the same-target annual/pending reference on identical
    funded IDs. These are descriptive requirements, not significance tests.
    Maximize initial coverage, minimize support-infeasibility referrals, then
    maximize full-H net log growth; fee and declared order resolve true ties.
    """
    if isinstance(alpha,bool) or not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('alpha is a declared service fraction')
    reference_rows=list(reference_rows)
    ref={int(r['scenario_id']):r for r in reference_rows}
    if len(ref)!=len(reference_rows): raise ValueError('Duplicate reference original IDs')
    original_set=set(ref)
    evidence=[]
    for spec in candidate_specs:
        rows=list(candidate_rows[spec['candidate_id']])
        if {int(r['scenario_id']) for r in rows}!=original_set:
            raise ValueError('Every candidate must retain identical original IDs')
        summary=summarize_records(rows)
        ids=[int(r['scenario_id']) for r in rows if r['assigned']]
        if any(not ref[sid]['assigned'] for sid in ids):
            raise ValueError('A funded candidate lacks the same-target reference')
        same=summarize_records([ref[sid] for sid in ids])
        reasons=[]
        if not ids: reasons.append('no_invested_inner_paths')
        else:
            if summary['breach_count']>math.floor(alpha*len(ids)+TOL):
                reasons.append('inner_event_frequency_above_reference_alpha')
            if summary['mean_mdd']>same['mean_mdd']+TOL:
                reasons.append('inner_mean_mdd_worse_than_same_cohort_B')
            if summary['cvar90_mdd']>same['cvar90_mdd']+TOL:
                reasons.append('inner_cvar90_mdd_worse_than_same_cohort_B')
        evidence.append(dict(**spec,summary=summary,same_cohort_B=same,
                             eligible=not reasons,reasons=reasons))
    eligible=[x for x in evidence if x['eligible']]
    if not eligible:
        return dict(status='execution_review',selected_candidate_id=None,evidence=evidence,
                    reason='no_inner_candidate_meets_declared_risk_requirements')
    # Successive global tie bands avoid order-dependent pairwise tolerance.
    eligible=[x for x in eligible if x['summary']['financial_n']==max(z['summary']['financial_n'] for z in eligible)]
    eligible=[x for x in eligible if x['summary']['support_review_n']==min(z['summary']['support_review_n'] for z in eligible)]
    best=max(x['summary']['mean_log_growth'] for x in eligible)
    tied=[x for x in eligible if x['summary']['mean_log_growth']>=best-TOL]
    minfee=min(x['summary']['mean_nominal_fees'] for x in tied)
    tied=[x for x in tied if x['summary']['mean_nominal_fees']<=minfee+TOL]
    chosen=min(tied,key=lambda x:x['order'])
    return dict(status='inner_selected_pending_refit',selected_candidate_id=chosen['candidate_id'],
                tied_candidate_ids=[x['candidate_id'] for x in tied],evidence=evidence,
                reason='coverage_then_fewer_support_reviews_then_net_growth')
