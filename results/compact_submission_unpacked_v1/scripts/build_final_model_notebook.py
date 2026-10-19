"""Audited read-only research companion: main framework and admission convergence.

Write this builder before reading final-guard results. Run only after the
parent authorizes complete independently audited results. No market parser,
execution kernel, learner or selector is imported. Old notebook builders and
frozen failed experiments are never edited or executed by this companion.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
STUDY=Path('results/model_review/global_policy_admission_v1')
PROTOCOL=Path('docs/global_policy_admission_protocol.json')
FIGURES=Path('research/figures/final_model_development')
NOTEBOOK=Path('notebooks/final_model_development.ipynb')
HTML=Path('notebooks/final_model_development.html')
POLICY='global_policy_with_same_U_guard'
OLD='joint_tie_set_refit_learner'
B='annual_then_pending';EQUAL='equal_thirds_annual'
PREVIOUS=Path('results/model_review/benefit_cost_execution_v1')
LOCAL=Path('results/model_review/constrained_fan_execution_v1')
JOINT=Path('results/model_review/joint_liquidity_refit_v1')
STABILITY=Path('results/model_review/refit_stability_v1')
SOURCE_NOTE='Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120; CC BY-NC-SA 4.0; Team 87.'
TOL=1e-12


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def readj(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return json.load(stream)
    return json.loads(path.read_text())


def writej(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def check_mapping(root,values):
    for rel,digest in values.items():
        path=Path(root)/rel
        if not path.is_file() or sha(path)!=digest:raise RuntimeError('Current source/input/artifact differs: '+str(path))


def require_complete(root=ROOT):
    """Hard audit/manifest/current-byte gates precede numeric table reads."""
    root=Path(root);manifest_path=root/STUDY/'manifest.json';audit_path=root/STUDY/'independent_audit.json'
    if not manifest_path.is_file() or not audit_path.is_file():raise RuntimeError('Wait for complete final guard and independent audit')
    manifest,audit=readj(manifest_path),readj(audit_path)
    if manifest.get('status')!='complete' or audit.get('status')!='passed':raise RuntimeError('Incomplete or unaudited final guard')
    if audit.get('parent_manifest_sha256')!=sha(manifest_path):raise RuntimeError('Final audit must bind the current parent manifest')
    protocol=readj(root/PROTOCOL)
    if (sha(root/PROTOCOL)!=manifest['protocol_sha256'] or audit.get('protocol_sha256')!=sha(root/PROTOCOL)
        or protocol.get('execution_ready') is not True or protocol['retained_learner']!=OLD
        or protocol['final_policy']!=POLICY or protocol['service_alpha']!=.1
        or manifest['new_fits']!=0 or manifest['market_data_loaded'] is not False
        or manifest['final_scenarios_loaded'] is not False):raise RuntimeError('Frozen postfit-only method/data boundary changed')
    if manifest['sources']!=manifest['sources_at_end'] or manifest['input_sha256']!=manifest['input_sha256_at_end']:
        raise RuntimeError('Producer source/input snapshots differ')
    before,after=audit['sources_and_artifacts_at_start'],audit['sources_and_artifacts_at_end']
    if before!=after:raise RuntimeError('Independent audit source/artifact snapshots differ')
    for mapping in (manifest['sources'],manifest['input_sha256'],protocol['execution_sources_sha256'],protocol['frozen_input_sha256'],before):
        check_mapping(root,mapping)
    check_mapping(root/STUDY,manifest['outputs_sha256'])
    snapshot={str(PROTOCOL):sha(root/PROTOCOL),str(STUDY/'manifest.json'):sha(manifest_path),str(STUDY/'independent_audit.json'):sha(audit_path)}
    snapshot.update(manifest['sources']);snapshot.update(manifest['input_sha256']);snapshot.update(before)
    for name,digest in manifest['outputs_sha256'].items():snapshot[str(STUDY/name)]=digest
    parents={}
    for folder in (PREVIOUS,LOCAL,JOINT,STABILITY):
        m=readj(root/folder/'manifest.json');a=readj(root/folder/'independent_audit.json');m_sha=sha(root/folder/'manifest.json')
        completed=(m.get('status')=='complete' or (folder==JOINT and m.get('status') is None
            and bool(m.get('completed_utc')) and m.get('training_tasks')==30 and m.get('candidate_count')==9))
        if not completed or a.get('status')!='passed':raise RuntimeError('Completed audited parent required: '+str(folder))
        if a.get('parent_manifest_sha256') is not None:
            if a['parent_manifest_sha256']!=m_sha:raise RuntimeError('Parent audit binding differs: '+str(folder))
        else:
            # Older independent receipts bind the manifest in their complete
            # root-relative source/artifact snapshots rather than a top key.
            if a['sources_and_artifacts_at_start'].get(str(folder/'manifest.json'))!=m_sha:raise RuntimeError('Older audit manifest snapshot differs')
        ab,ae=a['sources_and_artifacts_at_start'],a['sources_and_artifacts_at_end']
        if ab!=ae:raise RuntimeError('Parent independent audit snapshots differ')
        for values in (m['sources'],m['input_sha256'],ab):check_mapping(root,values)
        if m['sources']!=m['sources_at_end'] or m['input_sha256']!=m['input_sha256_at_end']:raise RuntimeError('Parent source/input snapshots differ')
        check_mapping(root/folder,m['outputs_sha256'])
        for rel in (folder/'manifest.json',folder/'independent_audit.json'):snapshot[str(rel)]=sha(root/rel)
        snapshot.update(m['sources']);snapshot.update(m['input_sha256']);snapshot.update(ab)
        for name,digest in m['outputs_sha256'].items():snapshot[str(folder/name)]=digest
        parents[str(folder)]=m
    return dict(manifest=manifest,audit=audit,protocol=protocol,parents=parents,input_artifact_sha256=dict(sorted(snapshot.items())))


def _bools(frame,columns):
    import pandas as pd
    for col in columns:
        if col not in frame:continue
        values=frame[col].dropna()
        if values.map(lambda v:isinstance(v,(bool,__import__('numpy').bool_))).all():continue
        mapping={'True':True,'False':False,'true':True,'false':False}
        if not values.map(lambda v:str(v) in mapping).all():raise ValueError('Observed boolean required: '+col)
        frame[col]=frame[col].map(lambda v:None if pd.isna(v) else mapping[str(v)])
    return frame


def load_tables(root=ROOT):
    """Use only audited outputs; no market CSV, NPZ forecast or fitting read."""
    import numpy as np
    import pandas as pd
    root=Path(root)
    admission=_bools(pd.read_csv(root/STUDY/'admission.csv.gz'),['final_automated','eligible'])
    clients=_bools(pd.read_csv(root/STUDY/'client_paths.csv.gz'),['assigned','breach'])
    summary=pd.read_csv(root/STUDY/'summary.csv.gz')
    trades=_bools(pd.read_csv(root/STUDY/'trade_estimates.csv.gz'),['final_guard_admitted','initial','pending','active_optional','restore','actual_trade','recommend_restore','strict_growth_sacrifice'])
    if len(admission)!=26 or admission.context.duplicated().any() or set(admission.repeat_id)!=set(range(6)):raise ValueError('All26fits required')
    if len(clients)!=2160 or clients[['repeat_id','context','scenario_id','policy']].duplicated().any():raise ValueError('All2160 unique policy/path records required')
    if len(summary)!=18 or summary[['repeat_id','policy']].duplicated().any() or set(summary.policy)!={POLICY,B,EQUAL}:raise ValueError('All18 summaries required')
    financial=['breach_count','conditional_breach_probability','mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','mean_fees','mean_trade_fraction']
    if not summary.original_n.eq(120).all() or not summary.loc[summary.assigned_n.eq(0),financial].isna().all().all():raise ValueError('Empty auto population must retain NULL financial results')
    if not summary.loc[summary.repeat_id.eq(0),'scope'].eq('in_sample_full120').all() or not summary.loc[summary.repeat_id.ne(0),'scope'].eq('outer_oof').all():raise ValueError('Never mix full in-sample and repeated OOF')
    if admission.loc[admission.original_execution_status.ne('automated'),'final_automated'].any():raise ValueError('Guard may never rescue an old refusal')
    old_clients=_bools(pd.read_csv(root/STABILITY/'client_paths.csv.gz'),['assigned','breach'])
    old_choices=readj(root/JOINT/'context_selections.json')['full_development']['methods'][OLD]
    # Figure1 counts: same original 120 paths in each repeat. A denotes the
    # outer admitted set, deliberately distinct from the inner guard's U.
    repeat=[]
    for repeat_id in range(1,6):
        old=old_clients.loc[old_clients.repeat_id.eq(repeat_id)&old_clients.policy.eq(OLD)]
        final=clients.loc[clients.repeat_id.eq(repeat_id)&clients.policy.eq(POLICY)]
        base=clients.loc[clients.repeat_id.eq(repeat_id)&clients.policy.eq(B)]
        active=final.loc[final.assigned]
        pair=active.merge(base,on=['repeat_id','context','job_id','scenario_id'],suffixes=('_model','_B'),validate='one_to_one')
        if len(old)!=120 or len(final)!=120 or len(pair)!=len(active) or (len(pair) and not pair.assigned_B.all()):raise ValueError('Exact same-ID outer denominator required')
        row=summary.loc[summary.repeat_id.eq(repeat_id)&summary.policy.eq(POLICY)].iloc[0]
        n=len(active);k=int(active.breach.sum()) if n else None;kb=int(pair.breach_B.sum()) if n else None
        if n!=int(row.assigned_n) or n>int(old.assigned.sum()) or (n and k!=int(row.breach_count)):raise ValueError('Coverage/event summaries disagree')
        repeat.append(dict(repeat_id=repeat_id,original_n=120,original_auto_n=int(old.assigned.sum()),final_auto_n=n,
            model_k=k,B_same_A_k=kb,model_rate=k/n if n else None,B_same_A_rate=kb/n if n else None,
            mean_delta_log_growth=float((pair.log_growth_model-pair.log_growth_B).mean()) if n else None,
            mean_delta_mdd=float((pair.mdd_model-pair.mdd_B).mean()) if n else None,
            mean_delta_nominal_fees=float((pair.nominal_fees_model-pair.nominal_fees_B).mean()) if n else None))
    repeat=pd.DataFrame(repeat)
    # Figure2 financial training quantities stay separate from inner-U counts.
    previous_training=pd.read_csv(root/PREVIOUS/'training_summary.csv.gz');previous_fit=pd.read_csv(root/PREVIOUS/'fit_admission.csv.gz')
    local_training=pd.read_csv(root/LOCAL/'training_summary.csv.gz');local_fit=pd.read_csv(root/LOCAL/'fit_admission.csv.gz')
    full=admission.loc[admission.repeat_id.eq(0)]
    if len(full)!=1:raise ValueError('One final full-development admission required')
    full=full.iloc[0]
    full=full.to_dict()
    full_old_trades=trades.loc[trades.context.eq(full['context'])&trades.active_optional]
    if len(full_old_trades):
        kinds=full_old_trades.eta_kind.unique();etas=full_old_trades.eta_technical.dropna().unique()
        if len(kinds)!=1 or (kinds[0]=='finite' and len(etas)!=1):raise ValueError('Original selected controller eta must stay fixed')
        full['eta_technical_display']='risk_first' if kinds[0]=='risk_first' else str(float(etas[0]))
    else:full['eta_technical_display']='unavailable'
    comparison=[]
    for label,policy,tr,fit in [('One-year positive net gate','benefit_cost_one_year',previous_training,previous_fit),('Local alpha rollout','constrained_net_growth_rollout',local_training,local_fit)]:
        ts=tr.loc[tr.job_id.eq('c0_refit')&tr.policy.eq(policy)];fa=fit.loc[fit.job_id.eq('c0_refit')&fit.policy.eq(policy)]
        if len(ts)!=1 or len(fa)!=1:raise ValueError('Named failed full-fit diagnostic required')
        ts,fa=ts.iloc[0],fa.iloc[0]
        comparison.append(dict(method=label,training_n=int(ts.n_training),training_k=ts.breach_count,
            training_cap=fa.training_breach_cap,B_training_k=fa.reference_training_breach_count,service_status=fa.execution_status,
            detail='Completed failed simplification; unchanged'))
    evidence=old_choices['selection_evidence'].get('selected_price_evidence');order=old_choices['selected_candidate_order']
    if evidence is not None and order is not None:
        candidate=next(x for x in evidence['candidate_summaries'] if x['execution_candidate_order']==order)
        global_row=dict(training_n=int(evidence['n_training']),training_k=candidate['breach_count'],training_cap=evidence['training_breach_cap'],B_training_k=evidence['reference_training_breach_count'])
    else:global_row=dict(training_n=120,training_k=None,training_cap=None,B_training_k=None)
    comparison.append(dict(method='Retained global learner',**global_row,service_status=full['execution_status'],detail='Old chosen controller + additional same-U guard'))
    comparison=pd.DataFrame(comparison)
    if not comparison.training_n.eq(120).all():raise ValueError('Keep complete full 120 training stage only in left panel')
    # Trade attribution is archived evidence first. Only explicit guarded
    # admissions supply the plotted final-procedure OOF decision subset.
    if len(trades):
        if not trades.audit_scope.eq('original_selected_controller_development').all():raise ValueError('Trade archive scope label missing')
        if trades[['context','scenario_id','year']].duplicated().any():raise ValueError('One archived decision per original fit/path/year')
        expected=trades.context.map(admission.set_index('context').final_automated)
        if expected.isna().any() or not trades.final_guard_admitted.eq(expected).all():raise ValueError('Trade admission flag disagrees with fit decision')
        optional=trades.loc[trades.active_optional]
        if not optional.loc[optional.restore,'recommend_restore'].all():raise ValueError('Stored selected action must match audited score')
        nonpositive=optional.loc[optional.restore&optional.delta_net_total_log_growth.le(TOL)]
        if not nonpositive.delta_model_breach_probability.lt(0).all():raise ValueError('Nonpositive-growth restore must strictly reduce predicted risk')
    served=trades.loc[trades.final_guard_admitted&trades.active_optional&trades.restore&trades.repeat_id.between(1,5)].copy()
    served['trade_class']=np.where(served.delta_net_total_log_growth>TOL,'Economic net gain',np.where(served.delta_net_total_log_growth < -TOL,'Risk reduction / strict growth cost','Risk reduction / growth tie'))
    trade_counts=[]
    for repeat_id in range(1,6):
        n=int(repeat.loc[repeat.repeat_id.eq(repeat_id),'final_auto_n'].iloc[0]);group=served.loc[served.repeat_id.eq(repeat_id)]
        trade_counts.append(dict(repeat_id=repeat_id,admitted_path_records=n,optional_restores=len(group) if n else None,
            economic_gain=int(group.trade_class.eq('Economic net gain').sum()) if n else None,
            risk_tie=int(group.trade_class.eq('Risk reduction / growth tie').sum()) if n else None,
            growth_cost=int(group.trade_class.eq('Risk reduction / strict growth cost').sum()) if n else None))
    trade_counts=pd.DataFrame(trade_counts)
    if not (trade_counts.loc[trade_counts.admitted_path_records.gt(0),['economic_gain','risk_tie','growth_cost']].sum(axis=1)==trade_counts.loc[trade_counts.admitted_path_records.gt(0),'optional_restores']).all():raise ValueError('Trade categories must partition optional restores')
    return dict(admission=admission,clients=clients,summary=summary,trades=trades,repeat=repeat,comparison=comparison,
                full=full,old_full_choice=old_choices,served_trades=served,trade_counts=trade_counts)


def plot_style():
    import matplotlib
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'axes.titlesize':15,
        'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'svg.fonttype':'none','svg.hashsalt':'final-model-development-v1',
        'figure.facecolor':'white','axes.facecolor':'white','text.color':'#27313a','axes.labelcolor':'#27313a','axes.edgecolor':'#a6afb5'})


def plot_repeats(repeat):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_style();fig,axes=plt.subplots(1,2,figsize=(17,8));fig.subplots_adjust(left=.065,right=.97,bottom=.23,top=.76,wspace=.3)
    x=np.arange(5);left,right=axes
    left.bar(x-.19,repeat.original_auto_n,.36,color='white',edgecolor='#64717b',linewidth=1.6,label='Original learner')
    left.bar(x+.19,repeat.final_auto_n,.36,color='#1f5b7a',label='After fixed same-U guard')
    for j,row in enumerate(repeat.itertuples()):
        left.text(j-.19,row.original_auto_n+3,str(row.original_auto_n),ha='center',fontsize=12)
        left.text(j+.19,row.final_auto_n+3,str(row.final_auto_n),ha='center',fontsize=12)
    left.set(xticks=x,xticklabels=[f'R{i}' for i in range(1,6)],xlim=(-.6,4.6),ylim=(0,136),ylabel='Automatic path records / original 120',title='Coverage per repeat')
    left.legend(loc='upper left',bbox_to_anchor=(0,1.19),frameon=False)
    valid=repeat.final_auto_n.gt(0)
    for offset,column,count_column,label,color in [(-.19,'model_rate','model_k','Guarded learner','#1f5b7a'),(.19,'B_same_A_rate','B_same_A_k','B on the same IDs A','#b27c2b')]:
        heights=repeat.loc[valid,column].to_numpy(float)*100
        right.bar(x[valid]+offset,heights,.36,color=color,label=label)
        for j,height in zip(x[valid],heights):
            row=repeat.iloc[j];right.text(j+offset,height+1.3,f'{int(row[count_column])}/{int(row.final_auto_n)}',ha='center',fontsize=12)
    upper=max([28.]+[float(v)*100+8 for v in repeat[['model_rate','B_same_A_rate']].to_numpy().ravel() if v==v])
    for j in x[~valid]:right.text(j,2.5,'n=0\nrate=NULL',ha='center',va='bottom',fontsize=12,color='#64717b')
    right.axhline(10,color='#64717b',linestyle='--',linewidth=1.3,label='10% benchmark')
    right.set(xticks=x,xticklabels=[f'R{i}' for i in range(1,6)],xlim=(-.6,4.6),ylim=(0,upper),ylabel='Observed full-H breach share on A (%)',title='Outer events on the same admitted IDs')
    right.legend(loc='upper left',bbox_to_anchor=(0,1.19),frameon=False,ncol=2,columnspacing=1.1)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.065,.94,'Retained global framework: coverage and same-ID outer evidence',fontsize=20,weight='bold')
    fig.text(.065,.875,'Each repeat reuses the same original 120 scenarios; A is its outer admitted set, distinct from the inner guard set U.',fontsize=13)
    fig.text(.065,.15,'Counts describe simulated path records, not new customers. Repeats and overlapping fits are dependent; no pooled risk certificate.',fontsize=12)
    fig.text(.065,.095,'The guard only removes old admissions. NULL risk/returns at zero coverage are preserved; admission does not guarantee outer risk<=10%.',fontsize=12)
    fig.text(.065,.035,SOURCE_NOTE,fontsize=12)
    return fig


def plot_full_stages(comparison,full,old_choice):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_style();fig,axes=plt.subplots(1,2,figsize=(17,8.3),gridspec_kw={'width_ratios':[1.25,1]})
    fig.subplots_adjust(left=.065,right=.965,bottom=.25,top=.76,wspace=.32);left,right=axes;x=np.arange(3)
    colors=['#9b8c7a','#7b526d','#1f5b7a'];labels=['One-year positive\nnet-growth gate','Local alpha\nrollout','Retained global\nlearner']
    for j,row in enumerate(comparison.itertuples()):
        if row.training_k==row.training_k:
            left.bar(j,row.training_k,.6,color=colors[j]);left.text(j,row.training_k+.7,f'k={int(row.training_k)}/120',ha='center',fontsize=12)
        else:left.text(j,1,'not selected\nk=NULL',ha='center',fontsize=12)
        if row.training_cap==row.training_cap:left.scatter(j,row.training_cap,marker='D',s=70,facecolors='white',edgecolors='#27313a',zorder=4)
    B_values=comparison.B_training_k.dropna().unique()
    if len(B_values)==1:left.axhline(B_values[0],color='#b27c2b',linestyle='--',linewidth=1.4,label=f'Same-target B training k={int(B_values[0])}')
    left.axhline(12,color='#64717b',linestyle=':',linewidth=1.5,label='Declared alpha x120=12')
    left.scatter([],[],marker='D',s=70,facecolors='white',edgecolors='#27313a',label='Effective complete-training cap')
    left.set(xticks=x,xticklabels=labels,xlim=(-.6,2.6),ylim=(0,max([36]+list(comparison.training_k.dropna()+5))),ylabel='Original full-H breach count /120',title='Complete full 120 training only')
    left.legend(loc='upper left',bbox_to_anchor=(0,1.19),frameon=False,ncol=2,columnspacing=1.1)
    n=full['auto_inner_n'];k=full['k_inner'];kb=full['k_B_same_U'];cap=full['absolute_cap']
    if n is not None and n==n and int(n)>0:
        vals=[int(k),int(kb)];right.bar([0,1],vals,.55,color=['#1f5b7a','#b27c2b'])
        for j,value in enumerate(vals):right.text(j,value+.4,f'{value}/{int(n)}',ha='center',fontsize=13)
        right.axhline(cap,color='#64717b',linestyle='--',linewidth=1.3,label=f'Absolute inner cap={int(cap)}')
        right.set(xticks=[0,1],xticklabels=['Old selected beta\ninner procedure','B on identical\ninner IDs U'],xlim=(-.6,1.6),ylim=(0,max(vals+[int(cap)])+5),ylabel='Inner validation full-H event count',title='Fixed own-inner guard: separate stage')
        right.legend(loc='upper left',bbox_to_anchor=(0,1.19),frameon=False)
    else:
        right.axis('off');right.text(.5,.5,'Original learner unavailable\nInner guard evidence=NULL',ha='center',va='center',transform=right.transAxes,fontsize=15)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    beta=old_choice['selected_beta'];order=old_choice['selected_candidate_order'];state=full['execution_status']
    fig.text(.065,.95,'Full-development fit: training feasibility and final inner guard',fontsize=20,weight='bold')
    fig.text(.065,.885,'The same 120 original IDs can appear at different stages: left is complete training; right is own-inner predictions, not the same metric.',fontsize=13)
    fig.text(.065,.17,f'Retained choices fixed: beta={beta}; eta={full["eta_technical_display"]} (technical); order={order}. Full-fit: {state}.',fontsize=13,weight='bold')
    fig.text(.065,.115,'Full 120 is in-sample. Passing a complete-training cap does not override the additional same-U B guard or establish deployability.',fontsize=12)
    fig.text(.065,.07,'Two completed simplifications remain failed experiments. Framework retention is a later methodological decision, not retrospective preregistration.',fontsize=12)
    fig.text(.065,.025,SOURCE_NOTE,fontsize=12)
    return fig


def plot_trade_attribution(trade_counts,served_trades,full):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_style();fig,axes=plt.subplots(1,2,figsize=(17,8.3));fig.subplots_adjust(left=.065,right=.965,bottom=.26,top=.76,wspace=.33)
    left,right=axes;x=np.arange(5);colors=['#1f5b7a','#b27c2b','#7b526d'];names=['Economic net gain','Risk reduction / growth tie','Risk reduction / strict growth cost']
    bottom=np.zeros(5);valid=trade_counts.admitted_path_records.gt(0).to_numpy()
    for column,label,color,hatch in [('economic_gain',names[0],colors[0],None),('risk_tie',names[1],colors[1],'//'),('growth_cost',names[2],colors[2],'xx')]:
        values=trade_counts[column].fillna(0).to_numpy(float);left.bar(x[valid],values[valid],.65,bottom=bottom[valid],color=color,label=label,hatch=hatch,edgecolor='white');bottom+=values
    high=max(1,float(bottom.max()))
    for j,row in enumerate(trade_counts.itertuples()):
        text='Review\ncounts=NULL' if row.admitted_path_records==0 else f'{int(row.optional_restores)} restores\n{row.admitted_path_records} path records'
        left.text(j,(high*.025 if row.admitted_path_records==0 else bottom[j]+high*.025),text,ha='center',fontsize=12)
    left.set(xticks=x,xticklabels=[f'R{i}' for i in range(1,6)],xlim=(-.6,4.6),ylim=(0,high*1.3),ylabel='Admitted OOF optional restore decisions',title='Economic explanation of executed restores')
    left.legend(loc='upper left',bbox_to_anchor=(0,1.19),frameon=False,ncol=2,columnspacing=1.)
    if len(served_trades):
        for label,color,marker in zip(names,colors,['o','s','^']):
            selected=served_trades.loc[served_trades.trade_class.eq(label)]
            if len(selected):right.scatter(selected.delta_net_total_log_growth,-100*selected.delta_model_breach_probability,s=25,alpha=.45,color=color,marker=marker,label=label)
        right.axvline(0,color='#64717b',linewidth=1.2);right.axhline(0,color='#64717b',linewidth=1.2)
        right.set(xlabel='Restore minus hold net Q (remaining total-log units)',ylabel='Predicted event-probability reduction (percentage points)',title='Paired forecasts at those same decision states')
    else:
        right.axis('off');right.text(.5,.5,'No final-admitted optional restores\nQ distribution unavailable',ha='center',va='center',transform=right.transAxes,fontsize=15)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.text(.065,.95,'Guard-admitted development decisions: net growth and risk tradeoffs',fontsize=20,weight='bold')
    fig.text(.065,.885,'Only final_guard_admitted AND active_optional AND restore; initial allocation and pending hold are excluded.',fontsize=13)
    fig.text(.065,.175,f'Full 120 service status: {full["execution_status"]}. Plotted repeat decisions remain simulated development records, not live deployment.',fontsize=12)
    fig.text(.065,.12,'Q already contains current and continuation fees. Negative net-Q gaps are a predicted risk tradeoff; overlapping Q gaps are never summed as loss.',fontsize=12)
    fig.text(.065,.07,'Repeated paths, fits and annual decisions are dependent. Rejected old controllers remain archived evidence, not served returns or extra customers.',fontsize=12)
    fig.text(.065,.025,SOURCE_NOTE,fontsize=12)
    return fig


def save_figures(root,tables,ready):
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/FIGURES;folder.mkdir(parents=True,exist_ok=True);builder_rel=str(Path(__file__).resolve().relative_to(ROOT))
    sources={builder_rel:sha(__file__)};outputs={}
    plot_data={'repeat_plot_data':tables['repeat'],'training_stage_plot_data':tables['comparison'],
        'trade_count_plot_data':tables['trade_counts'],'admitted_trade_plot_data':tables['served_trades']}
    for name,frame in plot_data.items():
        path=folder/(name+'.csv');frame.to_csv(path,index=False,lineterminator='\n');outputs[str(path.relative_to(root))]=sha(path)
    writej(folder/'full_guard_plot_data.json',tables['full']);outputs[str(FIGURES/'full_guard_plot_data.json')]=sha(folder/'full_guard_plot_data.json')
    figures=[('01_repeated_admission_and_same_ID_events',plot_repeats(tables['repeat'])),
        ('02_full_training_and_inner_guard',plot_full_stages(tables['comparison'],tables['full'],tables['old_full_choice'])),
        ('03_admitted_trade_cost_attribution',plot_trade_attribution(tables['trade_counts'],tables['served_trades'],tables['full']))]
    for name,fig in figures:
        png,svg=folder/(name+'.png'),folder/(name+'.svg');fig.savefig(png,dpi=240);fig.savefig(svg,metadata={'Date':None});svg.write_text(svg.read_text().rstrip());plt.close(fig)
        for path in (png,svg):outputs[str(path.relative_to(root))]=sha(path)
    after=require_complete(root)
    if ready['input_artifact_sha256']!=after['input_artifact_sha256'] or sources[builder_rel]!=sha(__file__):raise RuntimeError('Read-only source/input changed while drawing')
    writej(folder/'manifest.json',dict(status='complete',source_sha256=sources,sources_at_end=sources,input_artifact_sha256=ready['input_artifact_sha256'],
        input_artifact_sha256_at_end=after['input_artifact_sha256'],outputs_sha256=outputs,visual_review_confirmed=False,
        visual_scope='All three PNGs await actual parent inspection; no whole-HTML browser preview claim',source_note=SOURCE_NOTE,
        format={'png_dpi':240,'minimum_font_points':12,'svg_text':True,'svg_trailing_whitespace_stripped_before_hash':True}))


def confirm_figures(root=ROOT):
    root=Path(root);require_complete(root);path=root/FIGURES/'manifest.json';value=readj(path)
    for key in ('source_sha256','input_artifact_sha256','outputs_sha256'):check_mapping(root,value[key])
    value.update(visual_review_confirmed=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),visual_scope='All three PNGs actually inspected by parent; HTML execution/static checking only, no browser-preview claim')
    writej(path,value)


def observed_summary(tables):
    rows=tables['repeat'];counts='、'.join(str(int(n)) for n in rows.final_auto_n)
    events='；'.join('NULL（0自动）' if int(r.final_auto_n)==0 else f'{int(r.model_k)}/{int(r.final_auto_n)}' for r in rows.itertuples())
    full=tables['full'];reason=full['execution_reason']
    return f'五次重复在额外 guard 后的自动路径记录数为 **{counts} / 每次原120条**，对应完整期限事件为 **{events}**。全120最终拟合为 **`{full["execution_status"]}`**，原因 `{reason}`。主框架已收敛，自动部署成功尚未成立。'


def make_notebook(tables):
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(dedent(text).strip()))
    md('''
    # SafeInvest：主框架与准入收敛

    这是研究 companion，不是比赛论文。当前保留原有全策略有限族受约束学习框架，并对其**已经选定**的策略增加固定 own-inner same-U B 准入门槛。旧β、η／控制器、Q1目标、top-E逻辑和原拒绝均不改变，没有新拟合、重新排名、兜底或救回旧拒绝。

    这项主框架采用和额外guard是**后续公开的开发选择**。两次近期简化实验按各自冻结规则失败，结果继续保留；不能追认某个最终赢家一直预声明。相同原120条市场路径已被反复使用，不产生新的盲测或确认性安全证书。
    ''')
    md(observed_summary(tables))
    md('''
    ## 1. 当前来源和独立审计门槛

    本笔记只读已经完成、独立审计通过的产物，并验证审计绑定当前父manifest、当前源码／输入／输出SHA以及起止快照一致。父实验的完整审计和当前文件同样核验。只对字节流计算SHA，不解析原市场CSV，不载入收益数组、模拟器、拟合器或选择器。原121—180曾在早期V1接触过，本轮不读取，也不把它们称新的盲测。
    ''')
    code('''
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    from IPython.display import display
    ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'docs/global_policy_admission_protocol.json').is_file())
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_final_model_notebook as companion
    ready=companion.require_complete(ROOT)
    tables=companion.load_tables(ROOT)
    figures=companion.readj(ROOT/companion.FIGURES/'manifest.json')
    assert figures['status']=='complete' and figures['visual_review_confirmed'] is True
    for key in ('source_sha256','input_artifact_sha256','outputs_sha256'):companion.check_mapping(ROOT,figures[key])
    SOURCE_NOTE=companion.SOURCE_NOTE;POLICY=companion.POLICY;OLD=companion.OLD;B=companion.B;EQUAL=companion.EQUAL
    plot_style=companion.plot_style
    def show(frame,formats=None):display(frame.style.format(formats or {},na_rep='NULL').hide(axis='index'))
    print('完整产物、独立审计绑定和当前文件SHA检查通过。只读保存结果，没有新拟合或市场解析。')
    ''')
    md(r'''
    ## 2. 先固定数学对象，再解释技术参数

    客户C沿用原$H=30,L=0.40$，初始票据财富100。真实购买力为$R_t=W_t/\prod_{s=1}^{t}(1+\pi_s)$，名义财富已扣实际费用；$D_H=\max_{0\le t\le H}(1-R_t/\max_{s\le t}R_s)$。完整期限事件是$D_H>L$，不是单年概率，也没有重置历史。

    核心增长目标是$G_H(\pi)=E[\log(R_H/100)]/H$；风险服务基准是$P(D_H>L)\le\alpha$，声明$\alpha=0.10$。它不是问卷识别的风险偏好、题目强制概率或统计保证。$L$是客户问卷阈值，也不是财富保证。报告同时保留题目要求的CAGR中位数/P10和MDD中位数/P90；这些分位数不是置信区间。

    实现保留一个有限因果策略族。每个训练β下，在原9个技术控制器中选平均年化log增长最高、且$k_j\le\min(k_B,\lfloor\beta n\rfloor)$的候选。β属于固定网格$\{0,.025,.05,.075,.10\}$，它与η都不是客户问卷识别量。

    内层仅保留自身自动子群有证据、且满足原绝对风险门槛的β程序，先最大化自动路径覆盖，再按同一全局$10^{-12}$增长容差保留top集合$E$。最终只在升序$E$内找原完整训练可行选择。它是带覆盖优先服务逻辑的有限族过程，不是对所有可能策略的纯增长全局最优解。
    ''')
    md(r'''
    ## 3. η只生成控制器，费用只计一次

    决策前只使用实际到$t-1$的余额、CPI、历史峰值、运行MDD和原期限。用自己的训练联合路径calendar后缀，评估当前hold或paid restore，并在其后使用冻结年度B直到该分支首次超限、以后持有。经验fan、B续行近似、实际下一年重规划均为明确假设，不是校准条件概率或精确动态规划。

    $Q_g$是至原$H$的净真实购买力**总log增长**，已经包含当前与续行费用及其对pending路径的影响。有限η使用
    $$\Delta Q_g-H\eta\,\Delta\widehat p>10^{-12},\quad\Delta\widehat p=(k_R-k_H)/N.$$
    η是冻结有限族中生成策略的技术索引，不是问卷风险厌恶、货币交换意愿、KKT乘子或精确对偶价格。第9个`risk_first`为单独的整数量词序规则：先较少事件，再净增长，并非数值无穷大运算。

    若可选恢复的$\Delta Q_g>10^{-12}$，是模型内费后增长改善；若不为正，必须严格降低预测事件计数，由风险项解释其代价。严格$\Delta Q_g<-10^{-12}$另标增长牺牲，不能称正现金利润、必要或唯一可行动作。正局部分数仍不证明完整策略风险合格。

    当前恢复费用由$C=0.0025\sum_a[(W-C)w_a-x_a]_+$自融资求解，交易后总财富$W-C$。报价可单列，但不能再从已净值化$Q_g$扣一次。由于后续分支历史／pending／费用可能不同，一般不能把完整B续行优势简化成无费优势减当前费。初配强制并收费；原模式的初配／pending不伪造可选经济证据，真实首次超限后永久持有，无终端清算费。
    ''')
    md(r'''
    ## 4. 新准入只检查旧已选β的自身U

    只有旧结果已经自动的fit才可能获准。取它**原最终已选β**的四个内层预测块；各块仍用该内层自己的目标和原η选择，不能把最终refit的η倒施给内层。$U$恰是这些记录中的原自动ID，完整原内层分母继续保留。

    新guard固定要求$|U|>0$、$k_\pi(U)\le\lfloor0.10|U|\rfloor$，且$k_\pi(U)\le k_B(U)$。B逐job、原ID、Q1目标、静态γ和原H/L/费用/pending约定匹配。失败后整组24外层路径或全120路径统一Review、金融字段NULL，不尝试另一个β、η、目标、架构或B补值。

    这个准入决定在该fit外层收益前，只用其旧内层证据；但guard和主框架采用的研究开发历史已有适应性。因而不能把新冻结称新市场信息或独立确认试验。
    ''')
    code('''
    admission=tables['admission']
    show(admission[['repeat_id','outer_fold','original_execution_status','selected_beta','selected_candidate_order',
        'auto_inner_n','k_inner','k_B_same_U','absolute_cap','final_automated','execution_status','execution_reason']])
    ''')
    md('''
    ## 5. 五次重复：覆盖收缩与相同外层ID事件

    左图比较原学习程序与额外guard后的自动路径记录数。右图只在每次重复的外层获准集合$A$上逐ID配对模型和B，$A$不同于内层guard集合$U$。零准入的事件率保留NULL。重复共享同一原120条路径、拟合训练集合重叠；不能把5次当600条独立市场或把路径记录称新增客户。内层guard通过也不能保证该外层小子群事件率不超过10%。
    ''')
    code(inspect.getsource(plot_repeats)+"\nfig=plot_repeats(tables['repeat'])\ndisplay(fig)\nimport matplotlib.pyplot as plt\nplt.close(fig)")
    code('''
    show(tables['repeat'],{'model_rate':'{:.2%}','B_same_A_rate':'{:.2%}'})
    ''')
    md('''
    ## 6. 失败的简化规则与全开发拟合分开呈现

    以下左图全部是完整120条训练路径的策略事件计数：一年正净增长gate、固定局部α规则、保留的旧global已选控制器。它们不是外层验证收益排名，也没有从本图另选赢家。两个简化规则失败并不能证明所有安全盈利策略不存在；局部正经济性或每个状态局部10%不是原完整策略机会约束的唯一实现。

    右图是保留框架已选β的自身**内层预测**事件，与逐ID同U的B相比。即使样本数也等于120，它与左图完整训练是不同阶段、不同预测过程。额外same-U比较失败仍应Review。全120最终结果为in-sample，不能当第六次验证。
    ''')
    code(inspect.getsource(plot_full_stages)+"\nfig=plot_full_stages(tables['comparison'],tables['full'],tables['old_full_choice'])\ndisplay(fig)\nplt.close(fig)")
    code('''
    show(tables['comparison'])
    show(pd.DataFrame([{k:tables['full'][k] for k in ['selected_beta','selected_candidate_order','auto_inner_n','k_inner','k_B_same_U','absolute_cap','execution_status','execution_reason']}]))
    ''')
    md('''
    ## 7. 题目要求的金融摘要及其条件分母

    这里只展示最终获准子群的保存财务记录；参考策略为假设性参照，Review不以B补值。中位数和P10/P90均直接在当前条件自动路径上计算，离散分位数使用`inverted_cdf`。自动人群变化后，不同子群均值或分位数相减不能解释成策略因果收益。全开发0号scope与5次重复分开；不合并、平均分位数或生成IID置信证书。
    ''')
    code('''
    summary=tables['summary']
    names={POLICY:'保留程序+固定guard',B:'同目标年度B参照',EQUAL:'等权年度参照'}
    for scope in ['outer_oof','in_sample_full120']:
        frame=summary.loc[summary.scope.eq(scope),['repeat_id','policy','original_n','assigned_n','breach_count','conditional_breach_probability','median_cagr','p10_cagr','median_mdd','p90_mdd','mean_fees','mean_trade_fraction']].copy()
        frame['policy']=frame.policy.map(names)
        print(scope)
        show(frame,{k:'{:.2%}' for k in ['conditional_breach_probability','median_cagr','p10_cagr','median_mdd','p90_mdd','mean_trade_fraction']}|{'mean_fees':'{:.4f}'})
    print('mean_fees 是初始名义财富100下实际逐年费用之和的路径均值；不是费率或终端价值损失。')
    ''')
    code('''
    paired=tables['repeat'][['repeat_id','final_auto_n','model_k','B_same_A_k','mean_delta_log_growth','mean_delta_mdd','mean_delta_nominal_fees']].copy()
    paired['年化log增长差_pp']=100*paired.mean_delta_log_growth
    paired['MDD差_pp']=100*paired.mean_delta_mdd
    paired=paired.rename(columns={'final_auto_n':'同外层A路径记录数','model_k':'模型事件k','B_same_A_k':'同A_B事件k','mean_delta_nominal_fees':'平均名义费用差_初始100'})
    show(paired[['repeat_id','同外层A路径记录数','模型事件k','同A_B事件k','年化log增长差_pp','MDD差_pp','平均名义费用差_初始100']],
        {'年化log增长差_pp':'{:+.4f}','MDD差_pp':'{:+.4f}','平均名义费用差_初始100':'{:+.4f}'})
    measured=tables['repeat'].loc[tables['repeat'].final_auto_n.gt(0)]
    if len(measured) and (measured.mean_delta_log_growth.gt(0)&measured.mean_delta_nominal_fees.lt(0)&measured.mean_delta_mdd.gt(0)&measured.model_k.gt(measured.B_same_A_k)).all():
        print('所有非空重复在同A上：平均log增长较高、费用较低，同时MDD和超限事件更多。收益/费用好处伴随风险代价，不能称风险支配B。')
    print('以上为相同自动子群的描述性配对，不是因果效果或总体部署保证。年化log差用百分点；MDD差也用百分点，二者不是同一种风险收益量。')
    ''')
    md(r'''
    ## 8. 只在guard获准记录中解释可选交易的代价

    `trade_estimates`覆盖所有旧已选控制器，且明确标为`original_selected_controller_development`；后来被guard拒绝的旧轨迹仅是档案证据。下图严格筛`final_guard_admitted & active_optional & restore`，不把拒绝轨迹当新服务收益，也不把强制初配或pending计入可选恢复。

    经济改善、风险降低但增长在$10^{-12}$内并列、严格增长牺牲三类互斥。散点是同一状态下两种动作的预测差，单位是剩余完整期限总log-Q与模型事件概率百分点；不是实现损益。Q覆盖的未来区间互相重叠，**不得跨年累加负Q差当累计损失**，交易／年／重复记录也不独立。全120若为Review，不制造它的获准交易分布。
    ''')
    code(inspect.getsource(plot_trade_attribution)+"\nfig=plot_trade_attribution(tables['trade_counts'],tables['served_trades'],tables['full'])\ndisplay(fig)\nplt.close(fig)")
    code('''
    show(tables['trade_counts'])
    archived=tables['trades']
    archive_scope=archived.groupby(['repeat_id','final_guard_admitted'],dropna=False).agg(
        archived_path_records=('scenario_id','size'),original_contexts=('context','nunique')).reset_index()
    archive_scope=archive_scope.rename(columns={'archived_path_records':'年度决策记录数_非客户数'})
    show(archive_scope)
    admitted=tables['clients'].loc[tables['clients'].policy.eq(POLICY)&tables['clients'].assigned&tables['clients'].repeat_id.between(1,5)]
    print(f'获准OOF记录共{len(admitted)}条，对应{admitted.scenario_id.nunique()}个不同原市场ID；它们不是独立新客户。只按重复保留描述统计，不累加forecast Q缺口。')
    ''')
    md('''
    ## 9. 收敛的是框架与准入判断

    当前保留的是完整策略约束、有限技术控制器、覆盖优先内层top-E与明确拒绝选项的数学程序，额外same-U guard只收紧准入。它不消除参数/目标不唯一、跨拟合不稳定、无条件calendar预测近似、有限尾部样本和反复开发的局限。

    全开发C如果仍为Review，准确结论是当前固定规则与证据**尚不支持自动C部署**。部分外层fit获准、代码审计通过、甚至完整训练可行，都不能替代这一结论，也不能推出一切安全盈利策略不存在。主框架收敛不等于已经获得可部署投资策略。下一步应解释假设、证据和局限，而不是为覆盖率改阈值、换beta/eta或回填参考收益。

    本companion保存三张可追溯科研图与可编辑绘图代码，等待实际PNG检查后执行。HTML仅承诺执行和静态输出检查，不声称整个HTML的浏览器视觉预览通过。论文仍未生成或提交。
    ''')
    code('''
    end=companion.require_complete(ROOT)
    assert ready['input_artifact_sha256']==end['input_artifact_sha256']
    selected=[companion.PROTOCOL,companion.STUDY/'manifest.json',companion.STUDY/'independent_audit.json',
        companion.STUDY/'admission.csv.gz',companion.STUDY/'summary.csv.gz',companion.STUDY/'trade_estimates.csv.gz',
        Path('scripts/build_final_model_notebook.py'),companion.FIGURES/'manifest.json']
    show(pd.DataFrame([{'来源路径':str(path),'SHA256':companion.sha(ROOT/path)} for path in selected]))
    print('完整来源映射存于科研图manifest；起止相同。没有市场解析、新拟合、替换准入或paper写入。')
    ''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (.venv)','language':'python'},
        'language_info':{'name':'python','version':'3.12'},'builder':'scripts/build_final_model_notebook.py',
        'scope':'Read-only audited retained global framework + fixed same-U guard; open subsequent development choice; no deployment success claim'})
    nbf.validate(notebook);return notebook


def build(root=ROOT):
    root=Path(root);source=sha(__file__);ready=require_complete(root);tables=load_tables(root)
    save_figures(root,tables,ready);notebook=make_notebook(tables);(root/NOTEBOOK).parent.mkdir(parents=True,exist_ok=True);nbf.write(notebook,root/NOTEBOOK)
    if source!=sha(__file__):raise RuntimeError('Builder changed during build')
    print(json.dumps(dict(status='built; parent must actually inspect all3 PNGs before --confirm-figures and --execute',
        notebook=str(NOTEBOOK),code_cells=sum(c.cell_type=='code' for c in notebook.cells),builder_sha256=source),ensure_ascii=False))


def execute(root=ROOT):
    import os
    root=Path(root);ready=require_complete(root);figures=readj(root/FIGURES/'manifest.json')
    if figures.get('visual_review_confirmed') is not True:raise RuntimeError('Parent must inspect all3 PNGs before explicit --confirm-figures')
    for key in ('source_sha256','input_artifact_sha256','outputs_sha256'):check_mapping(root,figures[key])
    env=dict(os.environ);env['JUPYTER_PATH']=str(root/'.venv/share/jupyter')
    subprocess.run([sys.executable,'-m','nbconvert','--execute','--to','notebook','--inplace',str(root/NOTEBOOK),'--ExecutePreprocessor.timeout=180'],cwd=root,env=env,check=True)
    subprocess.run([sys.executable,'-m','nbconvert','--to','html','--no-input',str(root/NOTEBOOK),'--output',HTML.name],cwd=root,env=env,check=True)
    notebook=nbf.read(root/NOTEBOOK,as_version=4);nbf.validate(notebook);codes=[c for c in notebook.cells if c.cell_type=='code']
    if [c.execution_count for c in codes]!=list(range(1,len(codes)+1)) or any(o.output_type=='error' for c in codes for o in c.outputs):raise RuntimeError('Notebook execution incomplete or contains errors')
    after=require_complete(root)
    if ready['input_artifact_sha256']!=after['input_artifact_sha256']:raise RuntimeError('Read-only notebook inputs changed')
    writej(root/FIGURES/'notebook_execution.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        builder_sha256=sha(__file__),parent_manifest_sha256=sha(root/STUDY/'manifest.json'),figure_manifest_sha256=sha(root/FIGURES/'manifest.json'),
        notebook_sha256=sha(root/NOTEBOOK),html_sha256=sha(root/HTML),code_cells=len(codes),errors=0,
        html_browser_preview_passed=False,input_artifact_sha256=ready['input_artifact_sha256'],input_artifact_sha256_at_end=after['input_artifact_sha256']))
    print(json.dumps(dict(status='executed/exported',notebook_sha256=sha(root/NOTEBOOK),html_sha256=sha(root/HTML),code_cells=len(codes),errors=0,html_browser_preview_passed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--confirm-figures',action='store_true',help='Only after actual parent inspection of all3 PNGs')
    group.add_argument('--execute',action='store_true',help='Execute/export only after explicit figure review')
    args=parser.parse_args()
    if args.confirm_figures:confirm_figures()
    elif args.execute:execute()
    else:build()
