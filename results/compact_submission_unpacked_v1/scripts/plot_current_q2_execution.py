"""Render current Q2 standalone research figures from reviewed extraction only.

Run summarize_current_q2_execution.py first. No market data or model is read.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from plot_development_structure import configure, check_text_bounds

DATA = ROOT/'results/model_review/q2_execution_development_v2'
OUT = ROOT/'research/figures/q2_execution_development_v2'
ORDER = ['strict_annual','annual_then_pending','risk_priced_learner','equal_weight_annual','expanded_target_learner']
LABELS = {'strict_annual':'Annual','annual_then_pending':'Pending','risk_priced_learner':'Priced','equal_weight_annual':'Equal thirds','expanded_target_learner':'Expanded'}
COLORS = {'strict_annual':'#1B5E8C','annual_then_pending':'#AD7A19','risk_priced_learner':'#C45C2D','equal_weight_annual':'#57616B','expanded_target_learner':'#6C773C'}
STYLES = {'strict_annual':'-','annual_then_pending':'--','risk_priced_learner':'-.','equal_weight_annual':':','expanded_target_learner':(0,(5,1,1,1,1,1))}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save(fig,name):
    check_text_bounds(fig)
    fig.savefig(OUT/f'{name}.png',dpi=240,metadata={'Software':'plot_current_q2_execution.py'})
    fig.savefig(OUT/f'{name}.svg',metadata={'Date':None,'Creator':'plot_current_q2_execution.py'})
    plt.close(fig)


def summary_chart(summary, paths, context_choices):
    policies = [p for p in ORDER if p in summary.policy.tolist()]
    d=summary.set_index('policy').loc[policies];x=np.arange(len(policies));colors=[COLORS[p] for p in policies]
    fig,axes=plt.subplots(3,2,figsize=(12,11.3))
    fig.subplots_adjust(left=.085,right=.98,bottom=.13,top=.85,wspace=.25,hspace=.58)
    panels=[('median_cagr',100,'(a) Median real CAGR with P10','Annual rate (%)'),('median_mdd',100,'(b) Median real MDD with tail metrics','Drawdown (%)'),('breach_count',1,'(c) Whole-horizon exceedances','Paths exceeding L = 40%'),('mean_log_growth',100,'(d) Mean annual log growth','Annual log rate (%)'),('mean_nominal_fees',1,'(e) Total transaction fees','Nominal amount per initial 100'),('mean_trade_fraction',100,'(f) Executed actions','Decisions (%)')]
    for ax,(field,scale,title,ylabel) in zip(axes.flat,panels):
        vals=d[field].to_numpy()*scale
        if field=='mean_trade_fraction':
            ax.bar(x-.13,vals,width=.27,color=colors,edgecolor='#242B32',linewidth=.6)
            ax.bar(x+.16,d.mean_target_update_fraction.to_numpy()*100,width=.25,color='white',edgecolor=colors,hatch='///',linewidth=1.2)
            ax.legend(handles=[Patch(facecolor='#A2A7AC',label='Trades / 30'),Patch(facecolor='white',edgecolor='#57616B',hatch='///',label='Target updates / 29')],loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,handlelength=1.2,columnspacing=.8)
            ax.set_ylim(0,133)
            for i,v in enumerate(vals):ax.text(i-.13,v+3,f'{v:.1f}',ha='center',va='bottom',fontsize=12)
        else:
            ax.bar(x,vals,width=.55,color=colors,edgecolor='#242B32',linewidth=.6,zorder=2)
            high=max(vals.max(),0.)
            if field=='median_cagr':
                p10=d.p10_cagr.to_numpy()*100;ax.scatter(x,p10,marker='_',s=180,color='#242B32',linewidths=2,zorder=4)
                ax.legend(handles=[Line2D([],[],color='#242B32',marker='_',linestyle='none',markersize=12,label='P10 (empirical)')],loc='upper left',frameon=False)
                high=max(high,p10.max());ax.set_ylim(min(0,p10.min()*1.2),high*1.42)
                for i,v in enumerate(vals):ax.text(i,v+.1,f'{v:.2f}',ha='center',va='bottom',fontsize=12)
            elif field=='median_mdd':
                p90=d.p90_mdd.to_numpy()*100;tail=d.cvar90_mdd.to_numpy()*100
                ax.scatter(x,p90,marker='o',s=45,facecolor='white',edgecolor='#242B32',zorder=4)
                ax.scatter(x,tail,marker='x',s=50,color='#242B32',linewidths=1.7,zorder=4)
                ax.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#242B32',linestyle='none',label='P90'),Line2D([],[],marker='x',color='#242B32',linestyle='none',label='CVaR90')],loc='upper left',ncol=2,frameon=False,columnspacing=.8)
                ax.set_ylim(0,tail.max()*1.32)
                for i,v in enumerate(vals):ax.text(i,v+.7,f'{v:.1f}',ha='center',va='bottom',fontsize=12)
            elif field=='breach_count':
                ax.axhline(12,color='#242B32',linestyle='--',lw=1,zorder=1)
                ax.text(.01,.94,'10% reference = 12/120',transform=ax.transAxes,va='top',fontsize=12)
                ax.set_ylim(0,max(16,high+4));ax.set_yticks([0,4,8,12,16])
                for i,v in enumerate(vals):ax.text(i,v+.25,f'{int(v)}/120',ha='center',va='bottom',fontsize=12)
            else:
                ax.set_ylim(0,high*1.28)
                for i,v in enumerate(vals):ax.text(i,v+high*.035,f'{v:.2f}' if field=='mean_log_growth' else f'{v:.3f}',ha='center',va='bottom',fontsize=12)
        ax.set_title(title,loc='left',pad=9);ax.set_ylabel(ylabel);ax.set_xticks(x,['Equal\nthirds' if p=='equal_weight_annual' else LABELS[p] for p in policies]);ax.grid(axis='y',zorder=0);ax.set_axisbelow(True)
    fig.text(.085,.962,'Client C: pooled development OOF outcomes',fontsize=17,weight='bold',ha='left')
    fig.text(.085,.928,'H = 30 years; all 120 original scenarios per policy; no policy selected',fontsize=12,ha='left')
    if 'expanded_target_learner' in policies:
        update_modes=sum(bool(c['selected_controller']['updates_enabled']) for c in context_choices if c['selected_controller'] is not None)
        pa=paths.loc[paths.policy=='risk_priced_learner'].sort_values('scenario_id')
        ex=paths.loc[paths.policy=='expanded_target_learner'].sort_values('scenario_id')
        same=np.allclose(pa[['log_growth','mdd','nominal_fees','trade_fraction']],ex[['log_growth','mdd','nominal_fees','trade_fraction']],atol=1e-13,rtol=0)
        note=f'Expanded chose update-enabled mode in {update_modes}/{len(context_choices)} training fits'
        note += '; its OOF outcomes coincide with Priced.' if same else '; selected outer policies are evaluated without reselection.'
        fig.text(.085,.901,note,fontsize=12,ha='left')
    caption = 'Annual uses the strict Balanced target; Pending holds after crossing. Equal thirds uses a different target.\n'
    caption += ('Priced selects a fixed-target controller; Expanded selects fixed/update controllers, using training paths only.\n' if 'expanded_target_learner' in policies else 'Priced selects its execution price using training paths only; current targets remain fixed.\n')
    caption += 'Quantiles describe path dispersion, not confidence intervals. Fees are nominal amounts, not return points.'
    fig.text(.085,.078,caption,fontsize=12,ha='left',va='top',linespacing=1.5)
    save(fig,'01_current_oof_comparison')


def episode_chart(yearly,episode):
    policies=[p for p in ORDER if p in yearly.policy.tolist()]
    fig,axes=plt.subplots(4,1,figsize=(12,12.5),gridspec_kw={'height_ratios':[1.65,1.1,1.1,.6]})
    fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.85,hspace=.55)
    ax=axes[0]
    for p in policies:
        d=yearly.loc[yearly.policy==p].sort_values('year_endpoint')
        ax.plot(d.year_endpoint,d.real_wealth,color=COLORS[p],linestyle=STYLES[p],lw=2.2,label=LABELS[p])
    annual=episode['policies']['strict_annual'];tau=annual['first_crossing']
    ax.axhline(100,color='#A2A7AC',lw=.8)
    ax.axhline(annual['trigger_peak_real_wealth'],color='#1B5E8C',linestyle=(0,(2,4)),alpha=.65,lw=.9)
    ax.axvline(tau,color='#57616B',lw=.9,linestyle='--')
    ax.scatter([annual['mdd_peak_year'],annual['mdd_trough_year']],[annual['mdd_peak_real_wealth'],annual['mdd_trough_real_wealth']],color=COLORS['strict_annual'],s=35,zorder=5)
    ax.annotate(f'Annual peak: year {annual["mdd_peak_year"]}',(annual['mdd_peak_year'],annual['mdd_peak_real_wealth']),xytext=(10,28),textcoords='offset points',fontsize=12,arrowprops={'arrowstyle':'-','color':'#57616B'})
    ax.annotate(f'First crossing / trough: year {tau}',(tau,annual['mdd_trough_real_wealth']),xytext=(15,25),textcoords='offset points',fontsize=12,arrowprops={'arrowstyle':'-','color':'#57616B'})
    ax.set_ylabel('Actual real wealth\n(initial 100)');ax.set_title('(a) Net wealth; original peak and whole-horizon history retained',loc='left')
    for p in policies:
        d=yearly.loc[yearly.policy==p].sort_values('year_endpoint');axes[1].plot(d.year_endpoint,d.current_drawdown*100,color=COLORS[p],linestyle=STYLES[p],lw=2)
    axes[1].axhline(40,color='#242B32',lw=1,linestyle='--');axes[1].text(.99,.9,'L = 40%',transform=axes[1].transAxes,ha='right',fontsize=12)
    axes[1].set_ylabel('Current drawdown (%)');axes[1].set_title('(b) Drawdown from each policy\'s unreset annual real peak',loc='left')
    for p in policies:
        d=yearly.loc[yearly.policy==p].sort_values('year_endpoint');axes[2].plot(d.year_endpoint,d.actual_equity_weight*100,color=COLORS[p],linestyle=STYLES[p],lw=2)
        trade=d.loc[d.actual_trade_in_period.eq(True),'year_endpoint'].to_numpy()-1
        y=policies.index(p)
        axes[3].plot(trade,np.full(len(trade),y),linestyle='none',marker='|',markersize=11,color=COLORS[p])
        updates=d.loc[d.target_updated_in_period.eq(True),'year_endpoint'].to_numpy()-1
        if len(updates):axes[3].plot(updates,np.full(len(updates),y),linestyle='none',marker='D',markersize=5,color=COLORS[p])
    axes[2].set_ylim(0,100);axes[2].set_ylabel('Actual equity weight (%)')
    axes[2].set_title('(c) Actual year-end allocation',loc='left')
    axes[3].set_yticks(range(len(policies)),[LABELS[p] for p in policies]);axes[3].set_ylim(len(policies)-.5,-.5)
    axes[3].set_title('(d) Saved positive-transfer trades; diamonds mark target changes',loc='left');axes[3].set_xlabel('Decision year (0 is inception); financial panels use annual endpoints')
    for ax in axes:
        ax.set_xlim(0,30);ax.set_xticks(np.arange(0,31,5));ax.grid(axis='y');ax.set_axisbelow(True)
    fig.text(.12,.97,f'One prespecified severe episode: scenario {episode["scenario_id"]}, outer fold {episode["outer_fold"]}',ha='left',fontsize=17,weight='bold')
    subtitle = 'Chosen by largest strict-annual OOF MDD; this episode is not a representative-path comparison.'
    if 'expanded_target_learner' in policies:
        pa=yearly.loc[yearly.policy=='risk_priced_learner'].sort_values('year_endpoint')
        ex=yearly.loc[yearly.policy=='expanded_target_learner'].sort_values('year_endpoint')
        if np.allclose(pa[['real_wealth','actual_equity_weight']],ex[['real_wealth','actual_equity_weight']],atol=1e-12,rtol=0):subtitle = 'Chosen by largest strict-annual OOF MDD; Priced and Expanded coincide here. This path is not representative.'
    fig.text(.12,.936,subtitle,fontsize=12,ha='left')
    fig.legend(handles=[Line2D([],[],color=COLORS[p],linestyle=STYLES[p],lw=2.2,label=LABELS[p]) for p in policies],loc='upper left',bbox_to_anchor=(.115,.916),ncol=len(policies),frameon=False,columnspacing=1.5,handlelength=2.5)
    fig.text(.12,.063,'Equal thirds uses a different target. After crossing, pending rules hold while human response is absent.\nRecovery is measured against the fixed pre-crossing real peak; previous exceedance remains recorded.\nUnknown intrayear price movements are not observed. This episode cannot establish overall policy superiority.',ha='left',va='top',fontsize=12,linespacing=1.45)
    save(fig,'02_prespecified_annual_severe_episode')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-visual-review',action='store_true',help='Use only after inspecting the rendered PNGs with view_image')
    args=parser.parse_args()
    source_paths=[Path(__file__).resolve(),ROOT/'scripts/plot_development_structure.py',ROOT/'docs/q2_execution_development_protocol.json']
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    manifest=json.loads((DATA/'manifest.json').read_text()); input_names=['manifest.json','summary.csv','oof_paths.csv','episode.json','episode_yearly.csv','summary.json']
    inputs={str((DATA/n).relative_to(ROOT)):sha(DATA/n) for n in input_names}
    for n,h in manifest['outputs_sha256'].items():
        if sha(DATA/n)!=h:raise RuntimeError('Extraction output hash changed')
    summary=pd.read_csv(DATA/'summary.csv');paths=pd.read_csv(DATA/'oof_paths.csv');yearly=pd.read_csv(DATA/'episode_yearly.csv');episode=json.loads((DATA/'episode.json').read_text())
    if not summary.total_paths.eq(120).all() or not summary.auto_n.eq(120).all():raise RuntimeError('Plot needs explicit adaptation for Review-dependent coverage, never fill null with zero')
    for p,frame in paths.groupby('policy'):
        if len(frame)!=120 or frame.scenario_id.nunique()!=120:raise RuntimeError('Chart cohort is not original120')
    configure();plt.rcParams['svg.hashsalt']='current-q2-development-v2';OUT.mkdir(parents=True,exist_ok=True)
    summary_json=json.loads((DATA/'summary.json').read_text())
    summary_chart(summary,paths,summary_json['expanded_training_choices']);episode_chart(yearly,episode)
    summary.to_csv(OUT/'summary_chart_data.csv',index=False);yearly.to_csv(OUT/'episode_chart_data.csv',index=False)
    sources_end={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    if sources!=sources_end:raise RuntimeError('Chart source changed')
    end={p:sha(ROOT/p) for p in inputs}
    if inputs!=end:raise RuntimeError('Chart input changed')
    names=['01_current_oof_comparison.png','01_current_oof_comparison.svg','02_prespecified_annual_severe_episode.png','02_prespecified_annual_severe_episode.svg','summary_chart_data.csv','episode_chart_data.csv']
    write_json(OUT/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'status':manifest['status'] if args.confirm_visual_review else 'rendered_pending_visual_review','visual_review_confirmed':args.confirm_visual_review,'sources':sources,'sources_at_end':sources_end,'inputs_sha256':inputs,'inputs_sha256_at_end':end,'outputs_sha256':{n:sha(OUT/n) for n in names},'checks':{'text_bounds_passed':True,'cohort_per_policy_original120':True,'quantiles_not_averaged_across_folds':True,'minimum_font_points':12,'PNG_dpi':240,'SVG_text_preserved':True,'policy_chosen':False},'final_scenarios_loaded':False,'limitations':json.loads((DATA/'summary.json').read_text())['limitations']})
    print('Rendered summary and episode PNG/SVG; visual review confirmed' if args.confirm_visual_review else 'Rendered summary and episode PNG/SVG; visual review still required')


if __name__=='__main__':main()
