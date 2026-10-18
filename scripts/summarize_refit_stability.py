"""Read-only summaries and figures after the complete stability study audit.

--self-test uses only synthetic memory rows. It opens no result archive,
raw market, forecast library or policy-selection module. Repeats retain
original path identities; they are not new market samples or a time series.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from plot_development_structure import configure,check_text_bounds

BASE=ROOT/'results/model_review/refit_stability_v1'
OUT=ROOT/'results/model_review/refit_stability_summary_v1'
FIG=ROOT/'research/figures/refit_stability'
PROTOCOL=ROOT/'docs/refit_stability_protocol.json'
LEARNERS=('joint_canonical_learner','joint_tie_set_refit_learner')
REFERENCES=('annual_then_pending','equal_thirds_annual')
POLICIES=LEARNERS+REFERENCES
PAIRINGS=tuple((a,b) for a in LEARNERS for b in REFERENCES)+((LEARNERS[1],LEARNERS[0]),)
PRICE={0:'0',1:'.001',2:'.003',3:'.01',4:'.03',5:'.1',6:'.3',7:'1',8:'risk-first'}
METRICS=('log_growth','cagr','mdd','excess','nominal_fees','trade_fraction','breach')
FINANCIAL=(*METRICS,'volatility','terminal_real','nominal_moved')
BLUE,GOLD,GRAY,LIGHT,TEXT='#1B5E8C','#AD7A19','#57616B','#D6DBDF','#242B32'
SOURCE='Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120 (CC BY-NC-SA 4.0); team 87 development results.'
TOL=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text())
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if value is None:return None
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def bool_field(frame,field):
    if not frame[field].map(lambda v:isinstance(v,(bool,np.bool_))).all():raise ValueError('Boolean required: '+field)

def case_of(label):
    if label=='Balanced':return 'balanced_case'
    if label=='Review':return 'q1_review'
    if label in ('Conservative','Growth'):return 'not_balanced_case'
    raise ValueError('Unknown effective Q1 label')


def financial_summary(group):
    if group.empty:raise ValueError('Original identities must remain present')
    if group.scenario_id.duplicated().any():raise ValueError('Repeated original identity within one summary')
    if group.policy.nunique()!=1:raise ValueError('One policy per financial summary')
    policy=str(group.policy.iloc[0]);reference=policy in REFERENCES
    if policy not in POLICIES:raise ValueError('Unknown policy')
    bool_field(group,'assigned')
    cases=group.q1_effective_assignment.map(case_of)
    if not cases.eq(group.case_status).all():raise ValueError('Q1 case identity mismatch')
    auto=group.loc[group.assigned];null=group.loc[~group.assigned]
    if not null[list(FINANCIAL)].isna().all().all():raise ValueError('Nonautomatic financial results must stay NULL')
    if reference:
        if not group.reference_only.eq(True).all() or not group.hypothetical_reference.eq(True).all():raise ValueError('Hypothetical reference required')
        if not np.array_equal(group.execution_status,np.where(group.assigned,'hypothetical_reference','reference_unavailable')):raise ValueError('Reference availability is not service admission')
    else:
        if not group.assigned.eq(group.execution_status.eq('automated')).all():raise ValueError('Execution/admission differs')
        for status in ('automated','execution_review'):
            if not cases.loc[group.execution_status.eq(status)].eq('balanced_case').all():raise ValueError('Outside-case Q2 execution')
        for status in ('q1_review','not_balanced_case'):
            if not cases.loc[group.execution_status.eq(status)].eq(status).all():raise ValueError('Q1 Review/outside case mixed')
    row=dict(policy=policy,reference_only=reference,original_n=len(group),
        balanced_case_n=int(cases.eq('balanced_case').sum()),q1_review_n=int(cases.eq('q1_review').sum()),
        not_balanced_case_n=int(cases.eq('not_balanced_case').sum()),
        case_auto_n=None if reference else int(group.execution_status.eq('automated').sum()),
        case_execReview_n=None if reference else int(group.execution_status.eq('execution_review').sum()),
        reference_available_n=len(auto) if reference else None,
        financial_available_n=len(auto),financial_null_n=len(null))
    if row['original_n']!=row['balanced_case_n']+row['q1_review_n']+row['not_balanced_case_n']:raise ValueError('Original denominator partition')
    if not reference and row['balanced_case_n']!=row['case_auto_n']+row['case_execReview_n']:raise ValueError('Balanced denominator partition')
    names=('breach_count','conditional_breach_probability','mean_log_growth','mean_mdd','median_mdd','p90_mdd',
           'mean_nominal_fees','mean_trade_fraction','mean_excess','median_cagr')
    if not len(auto):return {**row,**dict.fromkeys(names)}
    bool_field(auto,'breach')
    if not np.isfinite(auto[[v for v in FINANCIAL if v!='breach']].to_numpy(float)).all():raise ValueError('Nonfinite automatic financial outcome')
    if not np.array_equal(auto.breach.to_numpy(bool),(auto.mdd-.4>TOL).to_numpy()):raise ValueError('Full-H breach definition')
    if not np.allclose(auto.log_growth,np.log(auto.terminal_real/100)/30,atol=1e-12,rtol=0):raise ValueError('Original-H growth identity')
    if not np.allclose(auto.nominal_fees,.0025*auto.nominal_moved,atol=1e-10,rtol=1e-12):raise ValueError('Fee identity')
    k=int(auto.breach.sum())
    row.update(breach_count=k,conditional_breach_probability=k/len(auto),mean_log_growth=float(auto.log_growth.mean()),
        mean_mdd=float(auto.mdd.mean()),median_mdd=float(auto.mdd.median()),p90_mdd=float(np.quantile(auto.mdd,.9,method='inverted_cdf')),
        mean_nominal_fees=float(auto.nominal_fees.mean()),mean_trade_fraction=float(auto.trade_fraction.mean()),
        mean_excess=float(auto.excess.mean()),median_cagr=float(auto.cagr.median()))
    return row


def repeat_and_fit_summaries(frame):
    repeats=[];fits=[]
    for (repeat,policy),g in frame.groupby(['repeat_id','policy'],sort=True):
        if sorted(g.scenario_id)!=list(range(1,121)):raise ValueError('Each repeat/policy needs all original120 paths')
        repeats.append(dict(repeat_id=int(repeat),**financial_summary(g)))
    for (repeat,fold,context,policy),g in frame.groupby(['repeat_id','outer_fold','context','policy'],sort=True):
        if len(g)!=24:raise ValueError('Each fit/policy needs24 held-out paths')
        fits.append(dict(repeat_id=int(repeat),outer_fold=int(fold),context=context,**financial_summary(g)))
    return pd.DataFrame(repeats),pd.DataFrame(fits)


def compare_saved_summary(computed,saved,identity,renames=None):
    renames=renames or {};checks=0;maximum=0.
    for row in computed.to_dict('records'):
        match=saved
        for field in identity:match=match.loc[match[renames.get(field,field)].eq(row[field])]
        if len(match)!=1:raise ValueError('One saved summary row per complete identity required')
        source=match.iloc[0]
        for key,value in row.items():
            source_key=renames.get(key,key)
            if source_key not in source:continue
            a,b=clean(value),clean(source[source_key]);checks+=1
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError('Saved summary NULL differs: '+key)
            elif isinstance(a,(bool,int,float)) and isinstance(b,(bool,int,float)):
                error=abs(float(a)-float(b));maximum=max(maximum,error)
                if not np.isclose(a,b,atol=2e-12,rtol=1e-12):raise ValueError('Saved summary numeric differs: '+key)
            elif a!=b:raise ValueError('Saved summary identity differs: '+key)
    return {'checked_cells':checks,'maximum_absolute_difference':maximum}


def scenario_frequency(frame):
    selected=frame.loc[frame.policy.isin(LEARNERS)]
    rows=[]
    for (policy,sid),g in selected.groupby(['policy','scenario_id'],sort=True):
        if len(g)!=5 or set(g.repeat_id)!=set(range(1,6)):raise ValueError('Each original scenario requires exactly5 repeats')
        auto=g.loc[g.assigned]
        rows.append(dict(policy=policy,scenario_id=int(sid),repeat_records_n=5,admitted_refits_n=int(g.assigned.sum()),
            execution_review_refits_n=int(g.execution_status.eq('execution_review').sum()),
            q1_review_refits_n=int(g.execution_status.eq('q1_review').sum()),
            outside_balanced_refits_n=int(g.execution_status.eq('not_balanced_case').sum()),
            breached_among_admitted_refits_n=int(auto.breach.sum()) if len(auto) else None,
            min_log_growth=float(auto.log_growth.min()) if len(auto) else None,
            max_log_growth=float(auto.log_growth.max()) if len(auto) else None,
            min_mdd=float(auto.mdd.min()) if len(auto) else None,max_mdd=float(auto.mdd.max()) if len(auto) else None,
            interpretation='Five refits on one original market path; not five independent risk trials'))
    freq=pd.DataFrame(rows);distribution=[]
    for policy,g in freq.groupby('policy',sort=True):
        if len(g)!=120:raise ValueError('Original path distribution must have120 rows')
        for k in range(6):distribution.append(dict(policy=policy,admitted_of_5=k,original_scenario_n=int(g.admitted_refits_n.eq(k).sum()),original_scenario_total=120))
    return freq,pd.DataFrame(distribution)


def paired_tables(frame):
    allpairs=[];summaries=[];fitpairs=[]
    for repeat in range(1,6):
        part=frame.loc[frame.repeat_id.eq(repeat)]
        for left,right in PAIRINGS:
            a=part.loc[part.policy.eq(left)].set_index('scenario_id');b=part.loc[part.policy.eq(right)].set_index('scenario_id')
            if set(a.index)!=set(range(1,121)) or set(a.index)!=set(b.index):raise ValueError('Pair retains all original120 identities')
            rows=[]
            for sid in range(1,121):
                x,y=a.loc[sid],b.loc[sid]
                if x.context!=y.context or x.outer_fold!=y.outer_fold:raise ValueError('Pair must share repeat/fold/path')
                common=bool(x.assigned and y.assigned)
                row=dict(repeat_id=repeat,outer_fold=int(x.outer_fold),context=x.context,scenario_id=sid,
                    left_policy=left,right_policy=right,right_reference_only=right in REFERENCES,
                    left_auto=bool(x.assigned),right_financial_available=bool(y.assigned),common_evaluable=common)
                for field in METRICS:
                    row['left_'+field]=float(x[field]) if x.assigned else None
                    row['right_'+field]=float(y[field]) if y.assigned else None
                    row['delta_'+field]=float(x[field])-float(y[field]) if common else None
                rows.append(row)
            pair=pd.DataFrame(rows);allpairs.extend(rows)
            def summarize(g):
                both=g.loc[g.common_evaluable]
                return dict(original_pairs=len(g),left_auto_n=int(g.left_auto.sum()),right_financial_available_n=int(g.right_financial_available.sum()),
                    common_evaluable_n=len(both),left_common_breach_count=int(both.left_breach.sum()) if len(both) else None,
                    right_common_breach_count=int(both.right_breach.sum()) if len(both) else None,
                    **{'mean_delta_'+f:float(both['delta_'+f].mean()) if len(both) else None for f in METRICS})
            summaries.append(dict(repeat_id=repeat,left_policy=left,right_policy=right,right_reference_only=right in REFERENCES,**summarize(pair)))
            for (fold,context),g in pair.groupby(['outer_fold','context'],sort=True):
                fitpairs.append(dict(repeat_id=repeat,outer_fold=int(fold),context=context,left_policy=left,right_policy=right,right_reference_only=right in REFERENCES,**summarize(g)))
    return pd.DataFrame(allpairs),pd.DataFrame(summaries),pd.DataFrame(fitpairs)


def training_margin(candidate_rows,kB,n,beta,selected_order):
    if n!=96:raise ValueError('Every outer refit must use the same96 original training paths')
    if len(candidate_rows)!=9 or set(candidate_rows.execution_candidate_order.astype(int))!=set(range(9)):
        raise ValueError('All9 saved training candidates are required')
    cap=min(int(kB),math.floor(float(beta)*n+TOL));minimum=int(candidate_rows.breach_count.min())
    chosen=None if selected_order is None else candidate_rows.loc[candidate_rows.execution_candidate_order.eq(selected_order)]
    if chosen is not None and len(chosen)!=1:raise ValueError('Selected saved candidate identity')
    selected_k=int(chosen.breach_count.iloc[0]) if chosen is not None else None
    return dict(training_n=n,reference_training_breach_count=int(kB),training_beta=float(beta),training_breach_cap=cap,
        minimum_nine_candidate_breach_count=minimum,min_candidate_slack=cap-minimum,
        feasible_training_candidate_n=int(candidate_rows.breach_count.le(cap).sum()),
        beta_selected_candidate_order=selected_order,beta_selected_training_breach_count=selected_k,
        beta_selected_training_slack=cap-selected_k if selected_k is not None else None)


def context_tables(contexts,bundles,training,fit_summary):
    rows=[];margins=[]
    for context,c in contexts.items():
        repeat,fold=int(c['repeat_id']),int(c['outer_fold'])
        train_n=len(c['training_ids']);eval_n=len(c['evaluation_ids'])
        if (train_n,eval_n)!=(96,24):raise ValueError('All refits must be96/24')
        cid=100*repeat+fold;jid=f'c{cid}_refit';bundle=bundles[jid]
        case=c['case'];product=case.get('q1_product') or case.get('product')
        stats=training.loc[training.job_id.eq(jid)]
        kB=bundle['reference_training_breach_count']
        minK=int(stats.breach_count.min()) if stats.breach_count.notna().all() else None
        for policy in LEARNERS:
            choice=c['methods'][policy];evidence=choice['selection_evidence'];inner=evidence.get('inner_sets') or {}
            E=inner.get('top_E',[])
            executed=choice['execution_status']=='automated'
            order=choice['selected_candidate_order'];beta=choice['selected_beta']
            attempt_betas=([beta] if beta is not None and case['effective_assignment']=='Balanced' else []) if policy==LEARNERS[0] else [a['training_beta'] for a in evidence.get('refit_attempts',[])]
            # Every top-E point gets a saved-training ledger; algorithm_attempted
            # distinguishes actual ascending attempts from descriptive entries
            # after the algorithm stopped. No outcome selects any new point.
            method_margins=[]
            for b in E if case['effective_assignment']=='Balanced' else []:
                evidence_at_beta=bundle['training_price_selections'][str(float(b))]
                m=training_margin(stats,kB,train_n,b,evidence_at_beta['selected_candidate_order'])
                m.update(context=context,repeat_id=repeat,outer_fold=fold,policy=policy,
                    top_E_rank=sorted(E).index(b)+1,algorithm_attempted=b in attempt_betas,
                    executed_beta=executed and beta==b,
                    ledger_scope='Saved final-refit training outcomes only; no additional outer price evaluated')
                margins.append(m);method_margins.append(m)
            selected_m=next((m for m in method_margins if executed and m['training_beta']==beta),None)
            fs=fit_summary.loc[fit_summary.context.eq(context)&fit_summary.policy.eq(policy)]
            if len(fs)!=1:raise ValueError('One fit financial summary per method')
            row=dict(context=context,context_id=cid,repeat_id=repeat,outer_fold=fold,policy=policy,
                training_n=train_n,evaluation_n=eval_n,static_gamma=c['static_gamma'],
                q1_effective_assignment=case['effective_assignment'],case_status=case['public_status'],
                q1_review_kind=case.get('q1_review_kind'),q1_review_reasons=json.dumps(case.get('q1_review_reasons',[])),
                q1_target_weights=json.dumps(product['weights']) if product else None,
                execution_target_weights=json.dumps(case['product']['weights']) if case.get('product') else None,
                selected_beta=beta,selected_candidate_order=order,selected_annual_price_label=PRICE[int(order)] if order is not None else None,
                execution_status=choice['execution_status'],execution_reason=choice['execution_reason'],
                top_inner_beta_set_E=json.dumps(E),top_inner_auto_n=inner.get('maximum_auto_coverage'),
                top_inner_mean_log_growth=inner.get('maximum_mean_log_growth_at_coverage'),
                actual_refit_beta_attempt_n=len(attempt_betas),refit_margin_available=case['effective_assignment']=='Balanced',reference_training_breach_count=kB,
                minimum_nine_training_breach_count=minK,selected_training_breach_cap=selected_m['training_breach_cap'] if selected_m else None,
                selected_training_breach_count=selected_m['beta_selected_training_breach_count'] if selected_m else None,
                selected_training_slack=selected_m['beta_selected_training_slack'] if selected_m else None,
                **{'heldout_'+k:clean(v) for k,v in fs.iloc[0].to_dict().items() if k not in ('repeat_id','outer_fold','context','policy')})
            rows.append(row)
    margin_columns=['training_n','reference_training_breach_count','training_beta','training_breach_cap',
        'minimum_nine_candidate_breach_count','min_candidate_slack','feasible_training_candidate_n',
        'beta_selected_candidate_order','beta_selected_training_breach_count','beta_selected_training_slack',
        'context','repeat_id','outer_fold','policy','top_E_rank','algorithm_attempted','executed_beta','ledger_scope']
    return pd.DataFrame(rows),pd.DataFrame(margins,columns=margin_columns)


def self_test():
    def row(sid,assigned,*,policy=LEARNERS[0],case='balanced_case',status=None):
        status=status or ('automated' if assigned else 'execution_review')
        label={'balanced_case':'Balanced','q1_review':'Review','not_balanced_case':'Growth'}[case]
        ref=policy in REFERENCES
        if ref:status='hypothetical_reference' if assigned else 'reference_unavailable'
        result=dict(scenario_id=sid,policy=policy,assigned=assigned,q1_effective_assignment=label,case_status=case,
            execution_status=status,reference_only=ref,hypothetical_reference=ref,repeat_id=1,outer_fold=(sid-1)//24+1,context=f'c{(sid-1)//24+1}')
        for f in FINANCIAL:result[f]=None
        if assigned:
            result.update(log_growth=.03,cagr=np.expm1(.03),mdd=.2,excess=0.,nominal_fees=.25,nominal_moved=100.,
                trade_fraction=.2,breach=False,volatility=.1,terminal_real=100*np.exp(.9))
        return result
    base=pd.DataFrame([row(s,False) for s in range(1,121)])
    z=financial_summary(base)
    assert z['case_auto_n']==0 and z['case_execReview_n']==120 and z['conditional_breach_probability'] is None
    mixed=pd.DataFrame([row(s,False,case='q1_review',status='q1_review') if s<=24 else row(s,False,case='not_balanced_case',status='not_balanced_case') for s in range(1,121)])
    z=financial_summary(mixed);assert z['q1_review_n']==24 and z['not_balanced_case_n']==96 and z['case_execReview_n']==0
    ref=pd.DataFrame([row(s,True,policy=REFERENCES[1]) for s in range(1,121)])
    assert financial_summary(ref)['case_auto_n'] is None and financial_summary(ref)['reference_available_n']==120
    corrupt=base.copy();corrupt.loc[0,'mdd']=.1
    try:financial_summary(corrupt)
    except ValueError:pass
    else:raise AssertionError('Imputation not detected')
    frames=[]
    for repeat in range(1,6):
        for policy in POLICIES:
            for sid in range(1,121):
                r=row(sid,policy in REFERENCES or (sid<=24 if policy==LEARNERS[0] else sid>96),policy=policy)
                r['repeat_id']=repeat;frames.append(r)
    frame=pd.DataFrame(frames);pairs,summary,_=paired_tables(frame)
    common=summary.loc[summary.left_policy.eq(LEARNERS[1])&summary.right_policy.eq(LEARNERS[0])]
    assert common.common_evaluable_n.eq(0).all() and common.mean_delta_mdd.isna().all()
    freq,dist=scenario_frequency(frame)
    assert len(freq)==240 and dist.groupby('policy').original_scenario_n.sum().eq(120).all()
    assert freq.loc[freq.scenario_id.eq(1)&freq.policy.eq(LEARNERS[0])].admitted_refits_n.iloc[0]==5
    assert freq.loc[freq.scenario_id.eq(30)].admitted_refits_n.eq(0).all()
    stats=pd.DataFrame([dict(execution_candidate_order=i,breach_count=6+i//3) for i in range(9)])
    m=training_margin(stats,9,96,.05,None);assert m['training_breach_cap']==4 and m['min_candidate_slack']==-2
    m=training_margin(stats,9,96,.075,0);assert m['training_breach_cap']==7 and m['beta_selected_training_slack']==1
    return {'status':'passed','synthetic_boundary_cases':8,'market_or_result_files_opened':False}


def save(fig,name):
    check_text_bounds(fig)
    fig.savefig(FIG/(name+'.png'),dpi=240,metadata={'Software':'summarize_refit_stability.py'})
    p=FIG/(name+'.svg');fig.savefig(p,metadata={'Date':None,'Creator':'summarize_refit_stability.py'})
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n');plt.close(fig)


def plot_repeat(summary):
    data=summary.loc[summary.policy.isin(LEARNERS)]
    fig,axes=plt.subplots(1,2,figsize=(14.8,8.4))
    fig.subplots_adjust(left=.08,right=.98,bottom=.29,top=.73,wspace=.35)
    x=np.arange(5);width=.33
    for j,(policy,color,label) in enumerate(zip(LEARNERS,(GOLD,BLUE),('Canonical tie rule','Retained top-set refit'),strict=True)):
        g=data.loc[data.policy.eq(policy)].sort_values('repeat_id');xx=x+(j-.5)*width
        n=g.case_auto_n.to_numpy(int);k=g.breach_count.to_numpy(float);risk=g.conditional_breach_probability.to_numpy(float)
        axes[0].bar(xx,n,width,color=color,label=label)
        for at,nn in zip(xx,n,strict=True):axes[0].text(at,nn+2,str(nn),ha='center',fontsize=12)
        axes[1].plot(xx,risk,'o' if j else 's',linestyle='none',color=color,markersize=8,label=label)
        for at,nn,kk,rr in zip(xx,n,k,risk,strict=True):
            axes[1].text(at,(.012+.022*j) if not nn else rr+.012+.022*j,'NULL' if not nn else f'{int(kk)}/{nn}',ha='center',fontsize=12,color=color,bbox={'facecolor':'white','edgecolor':'none','pad':.3})
    axes[0].set_ylim(0,140);axes[0].set_yticks([0,24,48,72,96,120]);axes[0].set_ylabel('Automatically evaluated paths / original 120')
    axes[0].set_title('(a) Admission within each complete repeat',loc='left',pad=14)
    axes[1].set_ylim(0,max(.20,float(data.conditional_breach_probability.max())+.09))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[1].set_ylabel('Observed full-H breach fraction among auto paths')
    axes[1].axhline(.1,color=GRAY,linestyle='--',linewidth=.9)
    axes[1].set_title('(b) Conditional events; changing admitted cohorts',loc='left',pad=14)
    for ax in axes:ax.set_xticks(x,[f'Repeat {i}' for i in range(1,6)]);ax.set_xlim(-.65,4.65);ax.grid(axis='y');ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.08,.78),ncol=2,frameon=False,fontsize=12)
    fig.text(.06,.95,'Fixed-learner sensitivity to whole-path partitions',fontsize=18,weight='bold')
    fig.text(.06,.90,'Five separately reported repeats, each original 120 scenarios; 25 outer fits, all training n = 96.',fontsize=12)
    fig.text(.06,.855,'Repeat labels identify partitions, not time. The same original market paths recur; no best seed is selected.',fontsize=12)
    fig.text(.06,.20,'Dots use only admitted paths; zero admission has NULL conditional risk. Review and outside-Balanced outcomes stay NULL.\n'
        'The 10% line is a service reference, not a validated guarantee. No confidence intervals or independent-repeat inference.\n'
        'Hypothetical financial references are excluded from automatic-service bars and are never customer fallbacks.\n'+SOURCE,fontsize=12,va='top',linespacing=1.6)
    save(fig,'01_repeat_admission_and_conditional_risk')


def plot_scenario_and_fits(distribution,contexts):
    fig,axes=plt.subplots(1,2,figsize=(15.2,8.8),gridspec_kw={'width_ratios':[.9,1.1]})
    fig.subplots_adjust(left=.07,right=.975,bottom=.30,top=.72,wspace=.38)
    width=.34;x=np.arange(6)
    for j,(policy,color,label) in enumerate(zip(LEARNERS,(GOLD,BLUE),('Canonical','Retained top-set refit'),strict=True)):
        g=distribution.loc[distribution.policy.eq(policy)].sort_values('admitted_of_5');xx=x+(j-.5)*width
        axes[0].bar(xx,g.original_scenario_n,width,color=color,label=label)
        for at,n in zip(xx,g.original_scenario_n,strict=True):axes[0].text(at,n+1.8,str(n),ha='center',fontsize=12)
    axes[0].set_ylim(0,max(20,int(distribution.original_scenario_n.max())+12));axes[0].set_xticks(x)
    axes[0].set_xlabel('Number of admitting refits out of five')
    axes[0].set_ylabel('Count of original market paths (120 per method)')
    axes[0].set_title('(a) Same-path admission sensitivity',loc='left',pad=14)
    axes[0].grid(axis='y');axes[0].set_axisbelow(True)
    primary=contexts.loc[contexts.policy.eq(LEARNERS[1])]
    colors={'automated':BLUE,'execution_review':'#F2E4C8','q1_review':LIGHT,'not_balanced_case':'white'}
    for r in primary.itertuples(index=False):
        xx,yy=int(r.outer_fold)-1,int(r.repeat_id)-1
        axes[1].add_patch(Rectangle((xx-.5,yy-.5),1,1,facecolor=colors[r.execution_status],edgecolor='white',linewidth=2))
        if r.execution_status=='automated':text=f'β {r.selected_beta:g}\nη {r.selected_annual_price_label}';color='white'
        elif r.execution_status=='q1_review':text='Q1\nReview';color=TEXT
        elif r.execution_status=='not_balanced_case':text='Outside\nBalanced';color=TEXT
        else:text='Inner fail' if 'inner' in str(r.execution_reason) or 'top_inner' in str(r.execution_reason) else 'Refit fail';color=TEXT
        axes[1].text(xx,yy,text,ha='center',va='center',fontsize=12,color=color)
    axes[1].set(xlim=(-.5,4.5),ylim=(4.5,-.5),xticks=np.arange(5),xticklabels=[f'Fold {i}' for i in range(1,6)],yticks=np.arange(5),yticklabels=[f'Repeat {i}' for i in range(1,6)])
    axes[1].set_title('(b) Primary retained-set fit status / parameters',loc='left',pad=14)
    axes[1].tick_params(length=0)
    for spine in axes[1].spines.values():spine.set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.06,.765),ncol=2,frameon=False,fontsize=12)
    fig.text(.06,.95,'Admission depends on the fitted training partition',fontsize=18,weight='bold')
    fig.text(.06,.90,'Each original scenario is evaluated once in each repeat; five appearances are repeated fits on one market path.',fontsize=12)
    fig.text(.06,.855,'Matrix: blue = automated, pale gold = execution Review, grey = Q1 Review, white = outside Balanced.',fontsize=12)
    fig.text(.06,.205,'β controls the complete-training breach cap; η is a technical price. Different selected prices need not imply different actions.\n'
        'Every outer fit uses 96 training paths and four inner 72-path fits; cross-fit changes cannot be attributed to changing sample size.\n'
        'Each fit makes one admission decision for all 24 held-out paths before their returns; frequencies track held-out fits.\n'
        'Repeated development paths and overlapping training fits are not 600 independent markets or 25 independent deployment trials.\n'+SOURCE,fontsize=12,va='top',linespacing=1.6)
    save(fig,'02_scenario_admission_and_primary_fit_status')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--confirm-visual-review',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    source_paths=[Path(__file__).resolve(),ROOT/'scripts/plot_development_structure.py',ROOT/'tests/test_refit_stability_summary.py']
    source_before={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    # No study table is opened until both gates pass.
    manifest=readj(BASE/'manifest.json');audit=readj(BASE/'independent_audit.json')
    if manifest.get('status')!='complete' or audit.get('status')!='passed':raise RuntimeError('Complete manifest and passed independent audit required')
    if manifest['scenario_ids_loaded']!=list(range(1,121)) or manifest['training_tasks']!=125 or manifest['outer_fits']!=25:raise ValueError('Complete original120/125job/25fit inventory required')
    files=[BASE/name for name in manifest['outputs_sha256']]+[BASE/'manifest.json',BASE/'independent_audit.json',PROTOCOL,
        ROOT/'docs/refit_stability_audit_plan.json',ROOT/'docs/refit_stability_preexecution_review.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in files}
    for name,value in manifest['outputs_sha256'].items():
        if sha(BASE/name)!=value:raise ValueError('Frozen study output changed')
    synthetic_precheck=self_test()
    frame=pd.read_csv(BASE/'client_paths.csv.gz')
    if not frame.client.eq('C').all() or not frame.horizon.eq(30).all() or not frame.loss_tolerance.eq(.4).all():raise ValueError('Only original C H30/L40 dynamic service')
    training=pd.read_csv(BASE/'training_candidate_summary.csv.gz')
    contexts=readj(BASE/'context_selections.json');bundles=readj(BASE/'static_refits.json.gz')
    for _,g in frame.loc[frame.policy.isin(LEARNERS)].groupby(['context','policy']):
        if len(g)!=24 or g.assigned.nunique()!=1 or g.execution_status.nunique()!=1:raise ValueError('One pre-evaluation admission decision for all24 held-out paths per fit')
    repeat,fit=repeat_and_fit_summaries(frame)
    freq,dist=scenario_frequency(frame)
    pairs,paired,fitpaired=paired_tables(frame)
    context,margins=context_tables(contexts,bundles,training,fit)
    names={'financial_available_n':'assigned_paths','financial_null_n':'financial_null_paths'}
    verification={'repeat_summary':compare_saved_summary(repeat,pd.read_csv(BASE/'summary.csv.gz'),['repeat_id','policy'],names),
        'fit_summary':compare_saved_summary(fit,pd.read_csv(BASE/'fit_summary.csv.gz'),['repeat_id','outer_fold','context','policy'],names),
        'paired_summary':compare_saved_summary(paired,pd.read_csv(BASE/'paired_summary.csv.gz'),['repeat_id','left_policy','right_policy'],
            {'left_policy':'learner_policy','right_policy':'reference_policy','right_reference_only':'reference_is_hypothetical',
             'left_auto_n':'learner_auto_n','right_financial_available_n':'reference_available_n','common_evaluable_n':'common_auto_n'})}
    if (len(frame),len(repeat),len(fit),len(freq),len(dist),len(pairs),len(paired),len(fitpaired),len(context))!=(2400,20,100,240,12,3000,25,125,50):raise ValueError('Summary inventory differs')
    for policy,g in freq.groupby('policy'):
        total=int(g.admitted_refits_n.sum());actual=int(frame.loc[frame.policy.eq(policy),'assigned'].sum())
        if total!=actual:raise ValueError('Repeated admission identity')
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    tables={'repeat_summary':repeat,'fit_financial_summary':fit,'fit_contexts':context,'refit_training_margins':margins,
        'scenario_admission_frequency':freq,'scenario_admission_distribution':dist,'repeat_paired_paths':pairs,
        'repeat_paired_summary':paired,'fit_paired_summary':fitpaired}
    for name,data in tables.items():data.to_csv(OUT/(name+'.csv'),index=False)
    learners=repeat.loc[repeat.policy.isin(LEARNERS)]
    findings={'repeat_method_summaries':repeat.to_dict('records'),
        'admitted_fit_count_by_method':{p:int(context.loc[context.policy.eq(p),'execution_status'].eq('automated').sum()) for p in LEARNERS},
        'repeat_admission_range_by_method':{p:[int(learners.loc[learners.policy.eq(p),'case_auto_n'].min()),int(learners.loc[learners.policy.eq(p),'case_auto_n'].max())] for p in LEARNERS},
        'scenario_admission_distribution':dist.to_dict('records'),
        'units':{'original_market_paths':120,'repeat_count':5,'outer_fit_count':25,'training_n_each_outer_fit':96,'training_n_each_inner_fit':72,'market_samples_not_600':True,'overlapping_fits_not_independent':True},
        'selection':'No favorable repeat, seed, floor, threshold, beta or eta selected from these summaries.',
        'training_margins':'Every top-E point reports saved training cap, minimum9-price count, beta-selected count and same-target annual kB. Slack is cap minus count: an integer in-sample feasibility margin, not an out-of-sample safety guarantee. algorithm_attempted distinguishes actual method attempts from descriptive remaining-E entries; no extra outer price evaluated.',
        'interpretation_boundaries':['Each fit admits or refers all24 held-out paths before observing their returns. Per-scenario frequency tracks which training fit held that ID out across five repeats, not future-path screening or a client-population decision.',
            'Admission frequencies describe repeated fits of one original path, not client-population coverage or independent market risks.',
            'Conditional growth/risk/fees retain Review and case denominators and may use different admitted cohorts.',
            'Financial deltas use only common automated/available paths sharing repeat, fold and original scenario; noncommon values remain NULL.',
            'Different beta/eta or target frequencies do not establish different actions, unique risk preferences or a causal source of performance changes.',
            'Outer training counts96 and inner72 are fixed across these fits. Inner partitions and outer memberships both vary; do not attribute differences solely to outer sample size.',
            'All data are repeatedly used development scenarios1–120. No independent-repeat inference, confidence interval, future10% guarantee, fresh blind test or paper approval.'],
        'upstream_audit_checks':audit['checks_passed'],'recomputed_saved_summary_agreement':verification}
    writej(OUT/'findings.json',findings)
    configure();plt.rcParams['svg.hashsalt']='refit-stability-summary-v1'
    plot_repeat(repeat);plot_scenario_and_fits(dist,context)
    after={str(p.relative_to(ROOT)):sha(p) for p in files}
    if before!=after:raise ValueError('Frozen study source changed during summary')
    source_after={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    if source_before!=source_after:raise ValueError('Summary sources changed during execution')
    receipt={'completed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'complete' if args.confirm_visual_review else 'rendered_pending_visual_review','visual_review_confirmed':args.confirm_visual_review,
        'sources':source_before,'sources_at_end':source_after,'input_sha256':before,'input_sha256_at_end':after,
        'outputs_sha256':{str(p.relative_to(ROOT)):sha(p) for directory in (OUT,FIG) for p in sorted(directory.iterdir()) if p.is_file() and p.name!='manifest.json'},
        'row_counts':{name:len(data) for name,data in tables.items()},'raw_market_loaded':False,'final_scenarios_loaded':False,
        'new_policy_or_fit':False,'original_study_modified':False,'synthetic_boundary_validation':synthetic_precheck,
        'source_input_artifact_sha256':{**source_before,**before},
        'checks':{'font_min_points':12,'png_dpi':240,'svg_text_preserved':True,'text_bounds_passed':True,
            'source_attribution_visible':True,'repeat_path_fit_units_visible':True,'no_ci_or_best_seed_selection':True}}
    writej(OUT/'manifest.json',receipt)
    print(json.dumps({'status':receipt['status'],'output':str(OUT.relative_to(ROOT)),'row_counts':receipt['row_counts']}))


if __name__=='__main__':main()
