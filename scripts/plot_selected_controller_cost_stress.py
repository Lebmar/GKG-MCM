"""One known-cost sensitivity figure from frozen saved C OOF paths only.

No raw market data, simulator, fitter, or unselected controller is read.
Run with --confirm-visual-review only after inspecting the PNG and after the
independent selected-cost audit has passed. Prior Q2 figures remain untouched.
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
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research')]
from plot_development_structure import configure,check_text_bounds
from safeinvest_research.metrics import summarize_paths,SUMMARY_METADATA

SOURCE=ROOT/'results/model_review/selected_controller_cost_stress_v1'
OUT=ROOT/'research/figures/selected_controller_cost_stress'
RATE_LABELS=['zero','half','base','double','quadruple']
RATES=[0.,.00125,.0025,.005,.01]
POLICIES=['fixed_target_learner','expanded_target_learner','annual_then_pending','equal_thirds_annual']
COLORS={'fixed_target_learner':'#C45C2D','annual_then_pending':'#AD7A19','equal_thirds_annual':'#57616B'}
MARKERS={'fixed_target_learner':'o','annual_then_pending':'s','equal_thirds_annual':'^'}
STYLES={'fixed_target_learner':'-','annual_then_pending':'--','equal_thirds_annual':':'}
LABELS={'fixed_target_learner':'Fixed + expanded (coincident)','annual_then_pending':'Annual then pending','equal_thirds_annual':'Equal thirds annual'}
TOL=1e-12


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def recompute(paths,saved):
    rows=[];mapping=None;max_error=0.
    for policy in POLICIES:
        for label,rate in zip(RATE_LABELS,RATES):
            part=paths.loc[(paths.policy==policy)&(paths.cost_label==label)].sort_values('scenario_id')
            if len(part)!=120 or part.scenario_id.nunique()!=120 or part.scenario_id.tolist()!=list(range(1,121)):
                raise RuntimeError('Every policy/rate must retain all original120 paths once')
            keys=part[['scenario_id','outer_fold']].astype(int).reset_index(drop=True)
            if mapping is None:mapping=keys
            elif not mapping.equals(keys):raise RuntimeError('Cost/policy changed the original OOF fold mapping')
            if not part.groupby('outer_fold').size().eq(24).all():raise RuntimeError('Require five held-out folds of24')
            if not part.cost_rate.eq(rate).all() or not part.horizon.eq(30).all() or not part.loss_tolerance.eq(.4).all():raise RuntimeError('Wrong cost/H/L')
            auto=part.loc[part.assigned]
            financial=['log_growth','cagr','mdd','terminal_real','nominal_fees','nominal_moved','trade_fraction','target_update_count']
            if not np.isfinite(auto[financial]).all().all() or not part.loc[~part.assigned,financial].isna().all().all():raise RuntimeError('Automatic finite/Review NULL masks failed')
            if len(auto)!=120:raise RuntimeError('Chart needs explicit coverage adaptation; never impute a Review outcome')
            if not np.allclose(np.expm1(auto.log_growth),auto.cagr,atol=1e-13,rtol=1e-12):raise RuntimeError('G/CAGR identity failed')
            if not np.allclose(auto.nominal_fees,rate*auto.nominal_moved,atol=1e-12,rtol=1e-12):raise RuntimeError('Fee/actual-transfer identity failed')
            if not np.allclose(auto.trade_fraction*30,np.rint(auto.trade_fraction*30),atol=1e-11,rtol=0):raise RuntimeError('Actual trade count must be integer over30 decisions')
            if not np.allclose(auto.target_update_count/30,auto.target_update_fraction,atol=1e-13,rtol=0):raise RuntimeError('Parent update count/30 identity failed')
            stats=summarize_paths({k:auto[k].to_numpy() for k in ['log_growth','cagr','mdd','volatility','terminal_real']},.4)
            row={'policy':policy,'cost_label':label,'cost_rate':rate,'cost_percent':rate*100,'total_paths':120,'auto_n':len(auto),'review_n':0,
                 **{k:float(stats[k]) for k in ['mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd']},
                 'breach_count':int((auto.mdd-.4>TOL).sum()),'mean_nominal_fees':float(auto.nominal_fees.mean()),
                 'mean_nominal_moved':float(auto.nominal_moved.mean()),'mean_trade_fraction':float(auto.trade_fraction.mean()),
                 'actual_trade_count':int(np.rint((auto.trade_fraction*30).sum())),'trade_opportunities':3600,
                 'target_update_count':int(auto.target_update_count.sum()),'target_update_opportunities':3480,
                 'mean_target_update_fraction_per29':float((auto.target_update_count/29).mean())}
            original=saved.loc[(saved.scope=='outer_oof')&(saved.client=='C')&(saved.policy==policy)&(saved.cost_label==label)]
            if len(original)!=1:raise RuntimeError('Saved summary identity missing')
            original=original.iloc[0]
            for field in ['mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd','breach_count','mean_nominal_fees','mean_nominal_moved','mean_trade_fraction']:
                err=abs(float(original[field])-row[field]);max_error=max(max_error,err)
                if not np.isclose(original[field],row[field],atol=1e-12,rtol=1e-12):raise RuntimeError(f'Pooled/source mismatch: {policy}/{label}/{field}')
            # Independent order-statistic checks, not means of fold summaries.
            c=np.sort(auto.cagr);d=np.sort(auto.mdd)
            if not np.allclose([row['median_cagr'],row['p10_cagr'],row['median_mdd'],row['p90_mdd'],row['cvar90_mdd']],[(c[59]+c[60])/2,c[11],(d[59]+d[60])/2,d[107],d[-12:].mean()],atol=1e-13,rtol=0):raise RuntimeError('Pooled quantile/CVaR order statistic failed')
            rows.append(row)
    return pd.DataFrame(rows),max_error


def individual_nonmonotonicity(paths):
    rows=[]
    for policy in POLICIES:
        part=paths.loc[paths.policy==policy]
        for field in ['log_growth','nominal_fees','trade_fraction']:
            a=part.pivot(index='scenario_id',columns='cost_label',values=field).loc[:,RATE_LABELS].to_numpy()
            delta=np.diff(a,axis=1);both=(delta>TOL).any(axis=1)&(delta<-TOL).any(axis=1)
            rows.append({'policy':policy,'field':field,'original_paths':120,'cost_settings_per_path':5,'nonmonotonic_path_count':int(both.sum()),'definition':'Both a positive and negative adjacent cost-setting difference, absolute threshold1e-12','interpretation':'Descriptive repeated-cost behavior on the same120 paths; not independent cost samples or an inference test'})
    return pd.DataFrame(rows)


def draw(summary):
    plotted=['fixed_target_learner','annual_then_pending','equal_thirds_annual']
    fig,axes=plt.subplots(2,2,figsize=(12,8.7))
    fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.78,wspace=.26,hspace=.42)
    settings=[('breach_count',1,'(a) Whole-horizon exceedances','Paths exceeding L = 40%'),('mean_log_growth',100,'(b) Mean annual log growth','Annual log rate (%)'),('mean_nominal_fees',1,'(c) Actual transaction fees','Mean total nominal fees\n(per initial 100)'),('mean_trade_fraction',100,'(d) Learner trade frequency (focused)','Positive-transfer trades / 30 (%)')]
    for ax,(field,scale,title,ylabel) in zip(axes.flat,settings):
        for policy in plotted if field!='mean_trade_fraction' else plotted[:1]:
            part=summary.loc[summary.policy==policy].set_index('cost_label').loc[RATE_LABELS]
            ax.plot(part.cost_percent,part[field]*scale,color=COLORS[policy],marker=MARKERS[policy],linestyle=STYLES[policy],lw=2,ms=6)
            if field=='breach_count':
                for x,y in zip(part.cost_percent,part[field]):ax.annotate(str(int(y)),(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=12)
            if field=='mean_trade_fraction':
                for x,y in zip(part.cost_percent,part[field]*100):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=12)
        ax.axvline(.25,color='#A2A7AC',lw=.8,linestyle=(0,(2,4)),zorder=0)
        ax.set_title(title,loc='left',pad=10);ax.set_ylabel(ylabel);ax.set_xlabel('Known fee rate (%)')
        ax.set_xticks(np.array(RATES)*100,['0','0.125','0.25','0.5','1.0']);ax.set_xlim(-.045,1.045);ax.grid(axis='y');ax.set_axisbelow(True)
    axes[0,0].axhline(12,color='#242B32',lw=.9,linestyle='--');axes[0,0].set_ylim(0,17);axes[0,0].set_yticks([0,4,8,12,16]);axes[0,0].text(.99,.17,'10% reference = 12/120',transform=axes[0,0].transAxes,ha='right',va='top',fontsize=12)
    axes[0,1].set_ylim(3.3,4.45);axes[0,1].yaxis.set_major_locator(MultipleLocator(.25))
    axes[1,0].set_ylim(0,4.7)
    axes[1,1].set_ylim(24.1,25.15);axes[1,1].yaxis.set_major_locator(MultipleLocator(.25))
    fig.text(.10,.958,'Client C: known-cost sensitivity of frozen selected controllers',fontsize=17,weight='bold',ha='left')
    fig.text(.10,.917,'OOF: the same 120 scenarios at every rate; H = 30 years; no policy selected from these outcomes.',fontsize=12,ha='left')
    fig.legend(handles=[Line2D([],[],color=COLORS[p],marker=MARKERS[p],linestyle=STYLES[p],lw=2,label=LABELS[p]) for p in plotted],loc='upper left',bbox_to_anchor=(.095,.889),ncol=3,frameon=False,columnspacing=1.5,handlelength=2.7)
    fig.text(.10,.835,'Five fee settings are sensitivity cases, not five independent market samples. Dotted vertical line: default 0.25%.',fontsize=12,ha='left')
    fig.text(.10,.115,'Menus, selected prices/modes and service status remain frozen; the fee is known to both actual execution and forecast fans.\nNo selection or risk recalibration is repeated. Fees are nominal amounts, not return percentage points.\nTrades include mandatory inception (30 opportunities). Target updates exclude inception (29 opportunities): all zero here.\nReference trade shares: Pending 97.31%, Equal thirds 100% at every rate. The learners\' tail-risk issue persists.',fontsize=12,ha='left',va='top',linespacing=1.45)
    check_text_bounds(fig)
    fig.savefig(OUT/'known_cost_oof_sensitivity.png',dpi=240,metadata={'Software':'plot_selected_controller_cost_stress.py'})
    fig.savefig(OUT/'known_cost_oof_sensitivity.svg',metadata={'Date':None,'Creator':'plot_selected_controller_cost_stress.py'})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirm-visual-review',action='store_true');args=parser.parse_args()
    audit_path=SOURCE/'independent_audit.json';audit_passed=audit_path.exists() and json.loads(audit_path.read_text())['status']=='passed'
    if args.confirm_visual_review and not audit_passed:raise RuntimeError('Independent selected-cost audit must pass before final freeze')
    source_files=[Path(__file__).resolve(),ROOT/'scripts/plot_development_structure.py',ROOT/'research/safeinvest_research/metrics.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    input_files=[SOURCE/'manifest.json',SOURCE/'summary.csv',SOURCE/'client_paths.csv',ROOT/'docs/selected_controller_cost_stress_protocol.json']
    if audit_passed:input_files.append(audit_path)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    if manifest['final_scenarios_loaded'] or not manifest['controller_parameters_frozen_at_default_cost'] or not manifest['same_actual_forecast_known_cost'] or manifest['training_risk_recalibrated_at_stress_costs']:raise RuntimeError('Upstream scope differs from the declared known-cost experiment')
    for name in ['summary.csv','client_paths.csv']:
        if sha(SOURCE/name)!=manifest['outputs_sha256'][name]:raise RuntimeError('Frozen saved source changed')
    paths=pd.read_csv(SOURCE/'client_paths.csv');paths=paths.loc[(paths.scope=='outer_oof')&(paths.client=='C')].copy()
    saved=pd.read_csv(SOURCE/'summary.csv');summary,max_error=recompute(paths,saved)
    a=paths.loc[paths.policy=='fixed_target_learner'].set_index(['cost_label','scenario_id']).sort_index()
    b=paths.loc[paths.policy=='expanded_target_learner'].set_index(['cost_label','scenario_id']).sort_index()
    compare=['log_growth','cagr','mdd','terminal_real','nominal_fees','nominal_moved','trade_fraction','target_update_count']
    if not a.index.equals(b.index) or not np.array_equal(a[compare].to_numpy(),b[compare].to_numpy()):raise RuntimeError('Cannot combine noncoincident fixed/expanded series')
    if not summary.target_update_count.eq(0).all():raise RuntimeError('Figure update-frequency annotation no longer matches')
    configure();plt.rcParams['svg.hashsalt']='selected-controller-known-cost-sensitivity-v1';OUT.mkdir(parents=True,exist_ok=True)
    draw(summary)
    summary.to_csv(OUT/'oof_chart_data.csv',index=False);paths.to_csv(OUT/'oof_paths.csv',index=False);individual_nonmonotonicity(paths).to_csv(OUT/'individual_path_nonmonotonicity.csv',index=False)
    outputs=['known_cost_oof_sensitivity.png','known_cost_oof_sensitivity.svg','oof_chart_data.csv','oof_paths.csv','individual_path_nonmonotonicity.csv']
    sources_end={p:sha(ROOT/p) for p in sources};inputs_end={p:sha(ROOT/p) for p in inputs}
    if sources!=sources_end or inputs!=inputs_end:raise RuntimeError('Plot source/input changed')
    write_json(OUT/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_audited_and_visually_reviewed' if args.confirm_visual_review else 'rendered_pending_audit_or_visual_review','independent_audit_passed':audit_passed,'visual_review_confirmed':args.confirm_visual_review,'sources':sources,'sources_at_end':sources_end,'input_sha256':inputs,'input_sha256_at_end':inputs_end,'outputs_sha256':{n:sha(OUT/n) for n in outputs},'scope':{'client':'C','H':30,'L':.4,'same_original_scenarios':120,'outer_folds':5,'paths_per_fold':24,'cost_cases':5,'cost_settings_are_not_market_samples':True,'selection_frozen_at_cost':.0025,'actual_and_forecast_cost_equal':True,'risk_recalibrated':False,'target_update_denominator_excluding_inception':29,'trade_denominator_including_inception':30},'summary_metadata':SUMMARY_METADATA,'checks':{'pooled_path_recomputation_matches_saved_summary':True,'summary_max_abs_error':max_error,'independent_order_statistics_and_worst12_CVaR':True,'all_policy_rate_groups_same_120_fold_mapping':True,'fixed_expanded_all_path_metrics_exactly_equal':True,'fee_equals_cost_times_actual_transfer':True,'trade_and_update_integer_counts_checked':True,'text_bounds_passed':True,'minimum_font_points':12,'PNG_dpi':240,'SVG_text_preserved':True},'final_scenarios_loaded':False,'raw_market_loaded':False,'new_model_fit_or_replay':False,'new_policy_selected':False,'interpretation':'OOF mean annual growth decreases and nominal fee means increase across these five cases; learner mean trade frequency is nonmonotonic. A separate descriptive repeated-path table records individual nonmonotonicity without treating cost cases as independent observations. No risk guarantee, recalibration, winner choice or unchanged-policy-after-refitting claim.'})
    print('Wrote',OUT/'known_cost_oof_sensitivity.png');print('Audit passed:',audit_passed,'visual review confirmed:',args.confirm_visual_review,'pooled-summary max error:',max_error)


if __name__=='__main__':main()
