"""Pure training-evidence prototype: refit feasibility within top inner set E.

This is a new learner definition, not a revision to any archived selection.
No market loading, fitting, simulation or financial-outcome imputation occurs.
All risk caps and price tie rules are the original frozen ones. E contains all
risk-eligible, maximum-coverage, globally growth-tied inner betas; it is neither
all eligible betas nor a class of per-path behavioral equivalence.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .nested_dynamic_selection import BETAS,ALPHA,GROWTH_TOL,price_selection,classify_assignment


def _int(value,label):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(label+' must be an integer')
    return int(value)


def _beta(value):
    if isinstance(value,(bool,np.bool_)) or value not in BETAS or value>ALPHA:
        raise ValueError('Require an original beta no larger than service alpha')
    return float(value)


def top_inner_set(summaries):
    """Validate original inner risk evidence and retain all top tied betas.

    Risk eligibility is reconstructed from the original auto denominator and
    whole-H count. Then maximize exact automatic validation path count, find
    one global maximum mean annual-log growth, and retain g>=max_g-1e-12.
    No pairwise chaining or minimum-beta collapse is performed here.
    """
    if len(summaries)!=len(BETAS):raise ValueError('Require all original inner beta summaries')
    betas=[_beta(r['training_beta']) for r in summaries]
    if betas!=list(BETAS):raise ValueError('Beta summaries must be unique and in original order')
    normalized=[];original_n=None
    for row,beta in zip(summaries,betas,strict=True):
        n=_int(row['original_n'],'original_n');auto=_int(row['case_auto_n'],'case_auto_n')
        if not 0<=auto<=n or n<1:raise ValueError('Invalid original/automatic validation denominator')
        if original_n is None:original_n=n
        if original_n!=n:raise ValueError('All beta summaries need the same original validation paths')
        if all(k in row for k in ('balanced_case_n','case_execReview_n','risk_review_n','not_balanced_case_n')):
            case,review,risk,outside=[_int(row[k],k) for k in ('balanced_case_n','case_execReview_n','risk_review_n','not_balanced_case_n')]
            if min(case,review,risk,outside)<0 or case!=auto+review or n!=case+risk+outside:
                raise ValueError('Case/service denominators do not partition original paths')
        k=row['breach_count'];g=row['conditional_mean_log_growth']
        if auto:
            k=_int(k,'breach_count')
            if not 0<=k<=auto or isinstance(g,(bool,np.bool_)) or not np.isfinite(g):
                raise ValueError('Automatic paths need observed whole-H events and growth')
            g=float(g)
        elif k is not None or g is not None:
            raise ValueError('Zero automatic evidence must retain NULL count and growth')
        eligible=bool(auto) and k<=math.floor(ALPHA*auto+GROWTH_TOL)
        if not isinstance(row['eligible'],(bool,np.bool_)) or bool(row['eligible'])!=eligible:
            raise ValueError('Saved eligibility differs from the original conditional-auto screen')
        normalized.append(dict(training_beta=beta,case_auto_n=auto,breach_count=k,
            conditional_mean_log_growth=g,eligible=eligible))
    eligible=[r for r in normalized if r['eligible']]
    coverage=max((r['case_auto_n'] for r in eligible),default=None)
    covered=[r for r in eligible if r['case_auto_n']==coverage]
    maximum=max((r['conditional_mean_log_growth'] for r in covered),default=None)
    E=[r['training_beta'] for r in covered if r['conditional_mean_log_growth']>=maximum-GROWTH_TOL]
    return dict(eligible_betas=[r['training_beta'] for r in eligible],top_E=E,
        maximum_auto_coverage=coverage,maximum_mean_log_growth_at_coverage=maximum,
        growth_tolerance=GROWTH_TOL,original_n=original_n,
        status='top_inner_set_available' if E else 'no_top_inner_beta',
        definition='All risk-eligible betas at exact maximum auto path count and within one global growth band; not all eligible and not behavioral classes.')


def _validate_training(summaries,reference_count,n):
    n=_int(n,'n_training');kB=_int(reference_count,'reference_count')
    if n<1 or not 0<=kB<=n:raise ValueError('Invalid complete-training/reference denominator')
    if len(summaries)!=9:raise ValueError('Require every original price candidate')
    if [_int(r['execution_candidate_order'],'candidate order') for r in summaries]!=list(range(9)):
        raise ValueError('Require nine candidates in original price order')
    for row in summaries:
        if _int(row['n_training'],'candidate n_training')!=n:raise ValueError('Candidate training denominators differ')
        k=_int(row['breach_count'],'candidate breach count');g=row['mean_log_growth']
        if not 0<=k<=n or isinstance(g,(bool,np.bool_)) or not np.isfinite(g):raise ValueError('Invalid complete-policy training metric')
    return n,kB


def select_refit_aware(effective_assignment,inner_summaries,training_summaries,reference_count,n):
    """Only ascending E may supply the first refit-feasible original price.

    Q1 effective-case priority is evaluated first. Outside Balanced, no Q2
    selection is attempted. Balanced with empty E or all-E training failure is
    execution Review; no beta outside E or reference fallback is considered.
    Every E attempt uses the unchanged frozen price_selection function.
    """
    case=classify_assignment(effective_assignment)
    base=dict(case_status=case,effective_assignment=effective_assignment,selected_beta=None,
        selected_candidate_order=None,selected_price_evidence=None,refit_attempts=[],inner_sets=None,
        financial_outcomes_evaluated=False,new_learner_definition=True)
    if case!='balanced_case':
        return {**base,'execution_status':case,'execution_review_reason':case}
    n,kB=_validate_training(training_summaries,reference_count,n)
    inner=top_inner_set(inner_summaries);base['inner_sets']=inner
    if not inner['top_E']:
        return {**base,'execution_status':'execution_review','execution_review_reason':'no_top_inner_beta'}
    attempts=[]
    for beta in sorted(inner['top_E']):
        result=price_selection(training_summaries,kB,n,beta)
        attempts.append(deepcopy(result))
        if result['selected_candidate_order'] is not None:
            return {**base,'inner_sets':inner,'refit_attempts':attempts,
                'selected_beta':beta,'selected_candidate_order':result['selected_candidate_order'],
                'selected_price_evidence':deepcopy(result),'execution_status':'automated',
                'execution_review_reason':None}
    return {**base,'inner_sets':inner,'refit_attempts':attempts,
        'execution_status':'execution_review','execution_review_reason':'no_refit_feasible_beta_in_top_E'}


def inner_behavior_classes(records):
    """Separate saved per-path controller vectors; never use to enlarge E.

    Inputs are ID/case/status/admission/selected-price metadata only. Fields
    concerning actual returns or growth are neither required nor inspected.
    """
    grouped={b:[] for b in BETAS}
    for row in records:
        b=_beta(row['training_beta']);sid=_int(row['scenario_id'],'scenario_id')
        if not 1<=sid<=120:raise ValueError('Only original development IDs1..120')
        auto=row['assigned']
        if not isinstance(auto,(bool,np.bool_)):raise ValueError('assigned must be a boolean')
        order=row.get('selected_candidate_order')
        if order is not None:
            order=_int(order,'selected_candidate_order')
            if not 0<=order<9:raise ValueError('Invalid selected price order')
        grouped[b].append((sid,str(row['job_id']),str(row['case_status']),str(row['execution_status']),bool(auto),order))
    original_ids=None;classes={}
    for b,rows in grouped.items():
        rows=sorted(rows)
        ids=[r[0] for r in rows]
        if not rows or len(set(ids))!=len(ids):raise ValueError('Each beta needs every unique original validation path')
        if original_ids is None:original_ids=ids
        if ids!=original_ids:raise ValueError('Behavior vectors must share original path identities')
        classes.setdefault(tuple(rows),[]).append(float(b))
    return [dict(class_id=i+1,betas=members,original_validation_n=len(vector),
        selected_controller_vector=[dict(scenario_id=r[0],job_id=r[1],case_status=r[2],execution_status=r[3],assigned=r[4],selected_candidate_order=r[5]) for r in vector])
        for i,(vector,members) in enumerate(classes.items())]
