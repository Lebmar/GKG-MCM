"""Plot only stable, independently audited nested dynamic extraction.

Preparation ``--self-test`` renders synthetic figures in memory for layout and
NULL handling only; it reads no research results and writes no plots. Real
figures require --include-stable-new after root authorization. Use the visual
confirmation flag only after viewing both actual PNGs with view_image.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from plot_development_structure import configure,check_text_bounds

DATA=ROOT/'results/model_review/nested_dynamic_calibration_summary_v1'
UPSTREAM=ROOT/'results/model_review/nested_dynamic_calibration_v1'
OUT=ROOT/'research/figures/nested_dynamic_calibration'
BLUE='#1B5E8C';GOLD='#AD7A19';GRAY='#57616B';LIGHT='#D6DBDF';TEXT='#242B32'
COMPARATORS=['legacy_risk_priced','legacy_strict_annual','legacy_pending_reference','new_pending_reference','legacy_equal_reference','new_equal_reference']
LABELS={'legacy_risk_priced':'Legacy priced learner','legacy_strict_annual':'Legacy strict annual ref',
        'legacy_pending_reference':'Legacy pending ref','new_pending_reference':'New pending ref',
        'legacy_equal_reference':'Legacy equal thirds ref','new_equal_reference':'New equal thirds ref'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def finish(fig,name,output):
    check_text_bounds(fig)
    if output is not None:
        fig.savefig(output/(name+'.png'),dpi=240,metadata={'Software':'plot_nested_dynamic_calibration.py'})
        fig.savefig(output/(name+'.svg'),metadata={'Date':None,'Creator':'plot_nested_dynamic_calibration.py'})
    plt.close(fig)


def service_and_risk(summary,output=None,paired=None):
    indexed=summary.set_index('comparison_id');learners=indexed.loc[['legacy_risk_priced','new_nested_dynamic']]
    if learners.reference_only.any():raise RuntimeError('A hypothetical reference entered the automatic service chart')
    fig,axes=plt.subplots(1,3,figsize=(13.8,6.5));fig.subplots_adjust(left=.07,right=.98,bottom=.30,top=.73,wspace=.48)
    positions=np.arange(2);bottom=np.zeros(2)
    statuses=[('case_auto_n','Automatic service',BLUE,''),('case_execReview_n','Execution review',GOLD,''),('risk_review_n','Q1 risk review',GRAY,''),('not_balanced_case_n','Outside Balanced',LIGHT,'//')]
    for field,label,color,hatch in statuses:
        vals=learners[field].to_numpy(float)
        if not np.isfinite(vals).all():raise RuntimeError('Learner service counts must be observed, not reference NULLs')
        axes[0].bar(positions,vals,bottom=bottom,width=.57,color=color,edgecolor='white',linewidth=.8,hatch=hatch)
        for x,y,b in zip(positions,vals,bottom):
            if y>=8:axes[0].text(x,b+y/2,str(int(y)),ha='center',va='center',fontsize=12,color='white' if color in [BLUE,GRAY] else TEXT)
        bottom+=vals
    if not np.array_equal(bottom,np.array([120.,120.])):raise RuntimeError('Service chart must partition the full original120, not the reference available cohort')
    axes[0].set_ylim(0,126);axes[0].set_yticks([0,24,48,72,96,120]);axes[0].set_xticks(positions,['Legacy\npriced','New\nnested']);axes[0].set_ylabel('Original paths (N = 120)');axes[0].set_title('(a) Service decisions',loc='left')
    probs=[float(x) for x in learners.conditional_breach_probability if pd.notna(x)];top=max(.20,max(probs,default=0)*1.4)
    for x,row in enumerate(learners.itertuples()):
        n=int(row.case_auto_n)
        if n:
            p=float(row.conditional_breach_probability);axes[1].bar(x,p,width=.57,color=BLUE if x else GRAY,edgecolor=TEXT,linewidth=.5)
            axes[1].text(x,p+.035*top,f'{int(row.breach_count)}/{n}',ha='center',va='bottom',fontsize=12)
        else:axes[1].text(x,.12*top,'No automatic\noutcomes; risk NULL',ha='center',va='bottom',fontsize=12)
    axes[1].axhline(.10,color=TEXT,linestyle='--',linewidth=1);axes[1].set_ylim(0,top);axes[1].set_xlim(-.6,1.6);axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_xticks(positions,['Legacy\npriced','New\nnested']);axes[1].set_ylabel('Exceedances / automatic paths');axes[1].set_title('(b) Conditional service risk',loc='left')
    refs=indexed.loc[['new_pending_reference','new_equal_reference']]
    if not refs.reference_only.all() or not refs.case_auto_n.isna().all() or not refs.case_execReview_n.isna().all():raise RuntimeError('Reference availability was relabeled as automatic service')
    available=refs.reference_available_n.to_numpy(float);case_available=refs.case_reference_available_n.to_numpy(float)
    axes[2].barh([0,1],available,height=.57,color=LIGHT,edgecolor=GRAY,linewidth=.8)
    for y,n,c in zip([0,1],available,case_available):axes[2].text(.98,y,f'{int(n)}/120 available\n{int(c)} within Balanced',transform=axes[2].get_yaxis_transform(),ha='right',va='center',fontsize=12)
    axes[2].set_yticks([0,1],['Pending\nreference','Equal thirds\nreference']);axes[2].set_ylim(1.6,-.6);axes[2].set_xlim(0,125);axes[2].set_xticks([0,40,80,120]);axes[2].set_xlabel('Available financial paths');axes[2].set_title('(c) Hypothetical references',loc='left')
    for ax in axes[:2]:ax.grid(axis='y');ax.set_axisbelow(True)
    axes[2].grid(axis='x');axes[2].set_axisbelow(True)
    fig.text(.07,.957,'Nested dynamic calibration: service coverage and conditional risk',fontsize=17,weight='bold',ha='left')
    fig.text(.07,.911,'OOF client C; H = 30, L = 40%; original 120 identities retained; automatic and reference populations differ.',fontsize=12,ha='left')
    fig.legend(handles=[Patch(facecolor=c,edgecolor=GRAY if c==LIGHT else c,hatch=h,label=label) for _,label,c,h in statuses],loc='upper left',bbox_to_anchor=(.065,.865),ncol=4,frameon=False,columnspacing=1.8)
    lead='Dashed line: 10% service reference, not a confidence bound. Zero automatic service leaves risk and growth unestimated.'
    if paired is not None:
        common=paired.loc[paired.comparator_id=='legacy_risk_priced'].iloc[0];n=int(common.common_evaluable_n)
        lead=(f'Matched automatic subset: legacy {int(common.comparator_common_breach_count)}/{n}; new {int(common.new_common_breach_count)}/{n}. Dashed 10% = service reference, not a confidence bound.' if n else 'No common automatic paths: matched risk is unestimated. Dashed 10% is a service reference, not a confidence bound.')
    fig.text(.07,.18,lead+'\nReference availability is hypothetical finance, never automatic service; its case-auto and case-execution-review fields remain NULL.\nReused development data and overlapping training folds; Q1 menus and served subsets can change. Liquidity suitability is unresolved.',fontsize=12,ha='left',va='top',linespacing=1.5)
    finish(fig,'01_service_and_risk',output)


def financial_comparison(paired,output=None):
    paired=paired.set_index('comparator_id').loc[COMPARATORS]
    fig,axes=plt.subplots(2,2,figsize=(13.8,9.6),sharey=True);fig.subplots_adjust(left=.265,right=.98,bottom=.20,top=.81,wspace=.28,hspace=.38)
    y=np.arange(len(COMPARATORS));labels=[f'{LABELS[c]} (n={int(paired.loc[c,"common_evaluable_n"])})' for c in COMPARATORS]
    settings=[('mean_delta_log_growth',100,'(a) Annual log growth','Mean difference (pp per year)'),('mean_delta_mdd',100,'(b) Whole-horizon MDD','Mean difference in MDD (pp)'),('mean_delta_nominal_fees',1,'(c) Actual transaction fees','Mean nominal difference (per initial 100)'),('mean_delta_trade_fraction',100,'(d) Actual trade frequency','Mean difference in trade share (pp)')]
    for ax,(field,scale,title,xlabel) in zip(axes.flat,settings):
        values=pd.to_numeric(paired[field],errors='coerce').to_numpy(float)*scale
        magnitude=max(float(np.max(np.abs(values[np.isfinite(values)]),initial=0)),.01)
        ax.set_xlim(-1.35*magnitude,1.35*magnitude);ax.axvline(0,color=TEXT,linewidth=.8)
        for i,(v,n) in enumerate(zip(values,paired.common_evaluable_n)):
            if n and np.isfinite(v):
                ax.barh(i,v,height=.54,color=BLUE if v>=0 else GOLD,edgecolor=TEXT,linewidth=.4)
                ax.annotate(f'{v:+.3f}',(v,i),xytext=(4 if v>=0 else -4,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=12)
            else:ax.text(.03,i,'No common automatic paths; NULL',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=12)
        ax.set_yticks(y,labels);ax.set_ylim(len(y)-.55,-.55);ax.set_title(title,loc='left',pad=9);ax.set_xlabel(xlabel);ax.grid(axis='x');ax.set_axisbelow(True)
    fig.text(.07,.958,'Financial comparison on the same automatic-service subgroup',fontsize=17,weight='bold',ha='left')
    fig.text(.07,.912,'Difference = new nested learner minus comparator, using matching scenario IDs and original outer folds.',fontsize=12,ha='left')
    fig.text(.07,.863,'Each comparator retains all 120 paired identities. Only the new-auto / comparator-available intersection is scored.',fontsize=12,ha='left')
    fig.text(.07,.115,'For the legacy learner, n is common automatic service. For each reference, n means common financial evaluability; references are hypothetical.\nNew Q1 products, static screens and dynamic calibration may all differ; these are complete-procedure contrasts, not a pure execution ablation.\nRead these conditional outcomes together with coverage, selected-beta stability and Review reasons. No winner is selected.',fontsize=12,ha='left',va='top',linespacing=1.5)
    finish(fig,'02_financial_comparison',output)


def synthetic_summary(auto_n):
    rows=[]
    for cid,n,k in [('legacy_risk_priced',120,14),('new_nested_dynamic',auto_n,0 if auto_n else np.nan)]:
        rows.append({'comparison_id':cid,'reference_only':False,'case_auto_n':n,'case_execReview_n':48-n if n<=48 else 0,'risk_review_n':24 if n<=48 else 0,'not_balanced_case_n':48 if n<=48 else 0,'conditional_breach_probability':k/n if n else np.nan,'breach_count':k,'reference_available_n':np.nan,'case_reference_available_n':np.nan})
    for cid,avail,case in [('new_pending_reference',48,48),('new_equal_reference',120,48)]:rows.append({'comparison_id':cid,'reference_only':True,'case_auto_n':np.nan,'case_execReview_n':np.nan,'risk_review_n':24,'not_balanced_case_n':48,'conditional_breach_probability':.1,'breach_count':0,'reference_available_n':avail,'case_reference_available_n':case})
    return pd.DataFrame(rows)


def self_test():
    configure()
    for n in [0,24,48]:
        service_and_risk(synthetic_summary(n))
        pairs=[]
        for i,cid in enumerate(COMPARATORS):
            vals=[(i-2)*.001,(2-i)*.01,(i-3)*.02,(3-i)*.01]
            pairs.append({'comparator_id':cid,'common_evaluable_n':n,**{name:value if n else np.nan for name,value in zip(['mean_delta_log_growth','mean_delta_mdd','mean_delta_nominal_fees','mean_delta_trade_fraction'],vals)}})
        financial_comparison(pd.DataFrame(pairs))
    print('Synthetic in-memory layout tests passed: zero-auto NULL, selective coverage, reference availability, signed matched metrics and text bounds; no files/results read or plots written')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--include-stable-new',action='store_true');parser.add_argument('--confirm-visual-review',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    if not args.include_stable_new:parser.error('--include-stable-new requires root stable authorization and passed upstream independent audit')
    audit=read_json(UPSTREAM/'independent_audit.json')
    if audit['status']!='passed':raise RuntimeError('Upstream independent audit must pass before plotting')
    manifest=read_json(DATA/'manifest.json')
    if manifest['status']!='complete_audited_saved_artifact_extraction' or manifest['final_scenarios_loaded']:raise RuntimeError('Extraction is incomplete or outside development scope')
    names=['manifest.json','oof_comparison_summary.csv','common_evaluable_summary.csv','context_selections.csv','inner_beta_evidence.csv','stability_diagnostics.json','interpretation.json']
    input_files=[DATA/name for name in names]+[UPSTREAM/'manifest.json',UPSTREAM/'independent_audit.json']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    for name in names[1:]:
        if sha(DATA/name)!=manifest['outputs_sha256'][name]:raise RuntimeError('Frozen extraction output changed')
    source_files=[Path(__file__).resolve(),ROOT/'scripts/plot_development_structure.py'];sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    summary=pd.read_csv(DATA/'oof_comparison_summary.csv');pairs=pd.read_csv(DATA/'common_evaluable_summary.csv');stability=read_json(DATA/'stability_diagnostics.json')
    if not summary.original_n.eq(120).all() or not pairs.original_pairs.eq(120).all():raise RuntimeError('Original cohort dropped or reweighted')
    configure();plt.rcParams['svg.hashsalt']='nested-dynamic-calibration-research-v1';OUT.mkdir(parents=True,exist_ok=True)
    service_and_risk(summary,OUT,paired=pairs);financial_comparison(pairs,OUT)
    summary.to_csv(OUT/'service_and_conditional_metrics.csv',index=False);pairs.to_csv(OUT/'matched_financial_metrics.csv',index=False)
    write_json(OUT/'stability_diagnostics.json',stability)
    end_sources={p:sha(ROOT/p) for p in sources};end_inputs={p:sha(ROOT/p) for p in inputs}
    if sources!=end_sources or inputs!=end_inputs:raise RuntimeError('Plot source or saved input changed')
    outputs=['01_service_and_risk.png','01_service_and_risk.svg','02_financial_comparison.png','02_financial_comparison.svg','service_and_conditional_metrics.csv','matched_financial_metrics.csv','stability_diagnostics.json']
    write_json(OUT/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_audited_and_visually_reviewed' if args.confirm_visual_review else 'rendered_pending_visual_review','visual_review_confirmed':args.confirm_visual_review,'independent_upstream_audit_passed':True,'sources':sources,'sources_at_end':end_sources,'input_sha256':inputs,'input_sha256_at_end':end_inputs,'outputs_sha256':{name:sha(OUT/name) for name in outputs},'checks':{'reference_financial_availability_not_auto_service':True,'original_120_retained':True,'NULL_metrics_at_zero_auto_preserved':True,'matched_subgroup_counts_labelled':True,'text_bounds_passed':True,'font_min_points':12,'PNG_dpi':240,'SVG_text_preserved':True},'final_scenarios_loaded':False,'raw_market_loaded':False,'new_model_run_or_policy_choice':False,'limitations':read_json(DATA/'interpretation.json')['limits']})
    print('Wrote',OUT,'; visual review confirmed:',args.confirm_visual_review)


if __name__=='__main__':main()
