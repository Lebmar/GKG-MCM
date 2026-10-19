"""Plot locked market-prediction diagnostics only; no raw market or model run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_development_structure import configure, check_text_bounds

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/model_review/forecast_assumptions_v2'
OUT=ROOT/'research/figures/forecast_assumptions_v2'
BLUE,GOLD,TEXT,GRAY='#1B5E8C','#AD7A19','#242B32','#57616B'
TARGETS=('next_year','remaining_to_original30_mean_annual')
CHANNELS=('equity_real_log','bond_real_log','bill_real_log','inflation_log')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduction(ratio):return 100*(1-float(ratio))


def paired_bars(ax,first,second,labels):
    y=np.arange(len(labels));height=.28
    ax.barh(y-height/1.6,first,height,color=BLUE,label='Next-year log quantity')
    ax.barh(y+height/1.6,second,height,facecolor='white',edgecolor=GOLD,hatch='///',linewidth=1.,label='Remaining-to-year-30 mean log')
    for pos,values,color in ((y-height/1.6,first,BLUE),(y+height/1.6,second,GOLD)):
        for at,value in zip(pos,values,strict=True):
            # All signs remain explicit, including near-zero worsening.
            ax.text(value+.025,at,f'{value:+.3f}%',ha='left',va='center',fontsize=12,color=color)
    ax.set_yticks(y,labels)
    ax.set_ylim(len(labels)-.55,-.55)
    ax.set_xlim(-.20,1.90)
    ax.set_xticks([0,.5,1.,1.5],['0%','0.5%','1.0%','1.5%'])
    ax.axvline(0,color=TEXT,linewidth=1.)
    ax.grid(axis='x');ax.set_axisbelow(True)
    ax.set_xlabel('Relative reduction in standardized MSE (%)')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirm-visual-review',action='store_true')
    args=parser.parse_args()
    source_paths=[Path(__file__).resolve(),ROOT/'scripts/plot_development_structure.py']
    source_before={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    paths=[BASE/'manifest.json',BASE/'independent_audit.json',BASE/'findings.json',
        BASE/'summary.csv',BASE/'fold_summary.csv',
        ROOT/'docs/forecast_assumption_diagnostic_protocol.json',
        ROOT/'docs/forecast_assumption_diagnostic_judgment.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    audit=json.loads((BASE/'independent_audit.json').read_text())
    manifest=json.loads((BASE/'manifest.json').read_text())
    if audit['status']!='passed' or manifest['scenario_ids_loaded']!=list(range(1,121)):
        raise AssertionError('Complete original-path diagnostic and passed audit required')
    for rel,value in manifest['outputs_sha256'].items():
        if sha(ROOT/rel)!=value:raise AssertionError('Upstream frozen output changed')
    finding=json.loads((BASE/'findings.json').read_text())
    summary=pd.read_csv(BASE/'summary.csv')
    folds=[]
    for entry in finding['primary_vector_results']:
        if entry['original_path_n']!=120:raise AssertionError('Primary vector original N changed')
        for f in entry['fold_results']:
            if f['original_path_n']!=24:raise AssertionError('Original fold N changed')
            folds.append({'target':entry['target'],'fold':f['fold'],
                'original_path_n':f['original_path_n'],
                'state_to_calendar_standardized_mse_ratio':f['state_to_calendar_standardized_mse_ratio'],
                'relative_mse_reduction_percent':reduction(f['state_to_calendar_standardized_mse_ratio'])})
    fold=pd.DataFrame(folds)
    channel=summary.loc[:,['target','channel','original_path_n','state_to_calendar_mse_ratio']].copy()
    if len(channel)!=8 or not channel.original_path_n.eq(120).all():raise AssertionError('Channel inventory changed')
    channel['relative_mse_reduction_percent']=channel.state_to_calendar_mse_ratio.map(reduction)
    channel['included_in_primary_vector']=channel.channel.ne('inflation_log')
    configure();plt.rcParams['svg.hashsalt']='forecast-assumptions-v2-fixed-comparator'
    fig,axes=plt.subplots(1,2,figsize=(14.5,8.1))
    fig.subplots_adjust(left=.11,right=.975,bottom=.30,top=.73,wspace=.60)
    f=lambda target:fold.loc[fold.target.eq(target)].sort_values('fold').relative_mse_reduction_percent.to_numpy()
    paired_bars(axes[0],f(TARGETS[0]),f(TARGETS[1]),[f'Fold {i} (24 paths)' for i in range(1,6)])
    axes[0].set_title('(a) Primary three-real-asset vector',loc='left',pad=14)
    c=lambda target:channel.loc[channel.target.eq(target)].set_index('channel').loc[list(CHANNELS)].relative_mse_reduction_percent.to_numpy()
    paired_bars(axes[1],c(TARGETS[0]),c(TARGETS[1]),['Equity real log','Bond real log','Bill real log','Inflation log\n(separate target)'])
    axes[1].axhline(2.5,color=GRAY,linestyle=':',linewidth=.9)
    axes[1].set_title('(b) Individual channels; 120 paths each',loc='left',pad=14)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.10,.785),ncol=2,frameon=False,fontsize=12)
    fig.text(.06,.95,'Calendar fan versus a fixed lag-state comparator',fontsize=18,weight='bold')
    fig.text(.06,.90,'120 whole development paths; original five 96/24 folds; fixed ridge lambda = 1.',fontsize=12)
    fig.text(.06,.855,'Previous-year observations only; calendar means and scales use training paths. No policy or parameter update.',fontsize=12)
    fig.text(.06,.215,
        'Change = 100 × (1 − state MSE / calendar MSE); positive values mean lower prediction loss.\n'
        'Each path first averages 29 annual errors; the primary vector averages three real-asset standardized losses.\n'
        'Reused development paths; overlapping training folds. No confidence interval, Q or tail-probability calibration.\n'
        'Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120 (CC BY-NC-SA 4.0); team 87 development diagnostics.',
        fontsize=12,va='top',linespacing=1.6)
    check_text_bounds(fig)
    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/'01_fixed_lag_state_prediction_losses'
    fig.savefig(stem.with_suffix('.png'),dpi=240,metadata={'Software':'plot_forecast_assumptions_v2.py'})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None,'Creator':'plot_forecast_assumptions_v2.py'})
    svg=stem.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    fold.to_csv(OUT/'fold_plot_data.csv',index=False)
    channel.to_csv(OUT/'channel_plot_data.csv',index=False)
    after={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if before!=after:raise AssertionError('Frozen diagnostic changed during plotting')
    source_after={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    if source_before!=source_after:raise AssertionError('Plot sources changed during execution')
    receipt={'completed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'complete' if args.confirm_visual_review else 'rendered_pending_visual_review',
        'visual_review_confirmed':args.confirm_visual_review,'sources':source_before,'sources_at_end':source_after,
        'input_sha256':before,'input_sha256_at_end':after,
        'outputs_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'},
        'raw_market_loaded':False,'final_scenarios_loaded':False,'new_policy_or_fit':False,
        'checks':{'font_min_points':12,'png_dpi':240,'svg_text_preserved':True,
            'text_bounds_passed':True,'zero_reference_visible':True,
            'primary_three_real_assets_and_separate_inflation_visible':True,
            'uncertainty_not_fabricated':True,'source_attribution_visible':True,
            'not_q_or_tail_calibration_visible':True}}
    (OUT/'manifest.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':receipt['status'],'figure':str(stem.with_suffix('.png').relative_to(ROOT))}))


if __name__=='__main__':main()
