"""Read completed diagnostic controls alongside the frozen learned controllers."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plot_development_structure import configure,check_text_bounds

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/model_review'
NEW=BASE/'simple_execution_comparison_v1'
OLD=BASE/'refit_stability_v1'
OUT=BASE/'simple_execution_summary_v1'
FIG=ROOT/'research/figures/simple_execution'
METHODS=('joint_canonical_learner','joint_tie_set_refit_learner','annual_gated_diagnostic','drift_band_gated_diagnostic')
LABELS=('Original tie rule','Retained-set learner','Annual positive control','TV-band diagnostic')
COLORS=('#AD7A19','#1B5E8C','#57616B','#2F6E53')
FIN=('log_growth','mdd','nominal_fees','trade_fraction','breach')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def writej(p,value):Path(p).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def summarize(frame):
    rows=[]
    for (repeat,policy),g in frame.groupby(['repeat_id','policy'],sort=True):
        if sorted(g.scenario_id)!=list(range(1,121)):raise ValueError('Original120 per repeat/method required')
        if not g.loc[~g.assigned,list(FIN)].isna().all().all():raise ValueError('Review financialNULL required')
        if not g.assigned.eq(g.execution_status.eq('automated')).all():raise ValueError('Admission identity')
        auto=g.loc[g.assigned];n=len(auto);k=int(auto.breach.sum()) if n else None
        result=dict(repeat_id=int(repeat),policy=policy,original_n=120,automatic_n=n,
            execution_review_n=int(g.execution_status.eq('execution_review').sum()),
            q1_review_n=int(g.execution_status.eq('q1_review').sum()),outside_balanced_n=int(g.execution_status.eq('not_balanced_case').sum()),
            breach_n=k,conditional_breach_probability=k/n if n else None,
            mean_log_growth=float(auto.log_growth.mean()) if n else None,mean_mdd=float(auto.mdd.mean()) if n else None,
            mean_nominal_fees=float(auto.nominal_fees.mean()) if n else None,mean_trade_fraction=float(auto.trade_fraction.mean()) if n else None)
        if sum(result[k] for k in ('automatic_n','execution_review_n','q1_review_n','outside_balanced_n'))!=120:raise ValueError('Original denominator partition')
        rows.append(result)
    return pd.DataFrame(rows)


def paired(frame,reference):
    pairs=[];summaries=[]
    comparisons=[(method,'annual_then_pending') for method in METHODS]+[(method,METHODS[1]) for method in METHODS[2:]]
    keys=['repeat_id','outer_fold','context','scenario_id']
    for repeat in range(1,6):
        for left,right in comparisons:
            a=frame.loc[frame.repeat_id.eq(repeat)&frame.policy.eq(left)]
            pool=reference if right=='annual_then_pending' else frame
            b=pool.loc[pool.repeat_id.eq(repeat)&pool.policy.eq(right)]
            joined=a.merge(b,on=keys,suffixes=('_left','_right'),validate='one_to_one')
            if len(joined)!=120:raise ValueError('Match repeat/fold/original-ID before comparing')
            these=[]
            for row in joined.to_dict('records'):
                both=bool(row['assigned_left'] and row['assigned_right'])
                these.append(dict(repeat_id=repeat,outer_fold=int(row['outer_fold']),context=row['context'],scenario_id=int(row['scenario_id']),
                    left_policy=left,right_policy=right,common_auto=both,left_auto=bool(row['assigned_left']),right_available=bool(row['assigned_right']),
                    left_breach=int(row['breach_left']) if both else None,right_breach=int(row['breach_right']) if both else None,
                    **{'delta_'+metric:float(row[metric+'_left'])-float(row[metric+'_right']) if both else None for metric in FIN}))
            group=pd.DataFrame(these);both=group.loc[group.common_auto]
            summaries.append(dict(repeat_id=repeat,left_policy=left,right_policy=right,original_pairs=120,
                common_auto_n=len(both),left_breach_n=int(both.left_breach.sum()) if len(both) else None,
                right_breach_n=int(both.right_breach.sum()) if len(both) else None,
                **{'mean_delta_'+metric:float(both['delta_'+metric].mean()) if len(both) else None for metric in FIN}))
            pairs.extend(these)
    return pd.DataFrame(pairs),pd.DataFrame(summaries)


def plot(summary):
    configure();plt.rcParams['svg.hashsalt']='simple-execution-diagnostic-v1'
    fig,(ax,table)=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[1.15,1.]})
    fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.69,wspace=.22)
    fig.suptitle('Smaller execution families: admission and held-out risk',x=.05,y=.97,ha='left',fontsize=20,fontweight='bold')
    fig.text(.05,.91,'Same fixed25 Q1 fits; original120 market paths repeated five times. Each fit decides before its24 evaluation returns.',fontsize=12)
    fig.text(.05,.865,'Annual and TV rules are diagnostic controls; neither estimates per-trade benefit>cost for a final Q2 strategy.',fontsize=12)
    x=np.arange(5);width=.18
    for i,(policy,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
        part=summary.loc[summary.policy.eq(policy)].sort_values('repeat_id')
        bars=ax.bar(x+(i-1.5)*width,part.automatic_n,width=width,color=color,label=label)
        for bar,value in zip(bars,part.automatic_n):ax.text(bar.get_x()+bar.get_width()/2,value+2+8*(i%2),str(value),ha='center',va='bottom',fontsize=12)
    ax.set_xticks(x,[f'Repeat {i}' for i in range(1,6)]);ax.set_ylim(0,140);ax.set_yticks(np.arange(0,121,24))
    ax.set_ylabel('Automatically evaluated paths / original120');ax.set_title('(a) Initial admission per repeat',loc='left',pad=18)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.13),ncol=2,frameon=False,fontsize=12)
    table.set_xlim(0,4);table.set_ylim(0,5);table.invert_yaxis()
    for i,policy in enumerate(METHODS):
        for j in range(5):
            r=summary.loc[summary.policy.eq(policy)&summary.repeat_id.eq(j+1)].iloc[0]
            risk=r.conditional_breach_probability
            fill='#EEF1F2' if pd.isna(risk) else '#F3D6D5' if risk>.1+1e-12 else '#DBE8EF'
            table.add_patch(Rectangle((i,j),1,1,facecolor=fill,edgecolor='white',linewidth=3))
            label='NULL\n0 auto' if pd.isna(risk) else f'{int(r.breach_n)}/{int(r.automatic_n)}\n{risk:.1%}'
            table.text(i+.5,j+.5,label,ha='center',va='center',fontsize=12)
    table.set_xticks(np.arange(4)+.5,['Original\ntie','Retained\nset','Annual\ncontrol','TV-band\ndiagnostic'])
    table.set_yticks(np.arange(5)+.5,[f'Repeat {i}' for i in range(1,6)]);table.tick_params(length=0)
    table.set_title('(b) Conditional full-H events k/n',loc='left',pad=18)
    table.grid(False)
    for spine in table.spines.values():spine.set_visible(False)
    notes=['B inclusion makes supported training feasibility mechanical; only separate inner and held-out observations bear on risk.',
        'Conditional cohorts differ. Zero admission has NULL risk; the 10% reference is not a future guarantee.',
        'Repeat labels are partitions, not time. No best seed/architecture, confidence interval or effective N=600 is inferred.',
        'Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120 (CC BY-NC-SA 4.0); team 87 development results.']
    for i,note in enumerate(notes):fig.text(.05,.17-.035*i,note,fontsize=12)
    check_text_bounds(fig)
    stem=FIG/'01_simple_controls_and_learners'
    fig.savefig(stem.with_suffix('.png'),dpi=240,metadata={'Software':'SafeInvest team87'})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
    path=stem.with_suffix('.svg');path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--confirm-visual-review',action='store_true');args=parser.parse_args()
    for folder in (NEW,OLD):
        if readj(folder/'manifest.json')['status']!='complete' or readj(folder/'independent_audit.json')['status']!='passed':raise RuntimeError('Complete independent audit required')
        manifest=readj(folder/'manifest.json');audit=readj(folder/'independent_audit.json')
        for name,expected in manifest['outputs_sha256'].items():
            if sha(folder/name)!=expected:raise RuntimeError('Audited upstream output changed: '+name)
        snapshots=audit['sources_and_artifacts_at_start'] if 'sources_and_artifacts_at_start' in audit else audit.get('source_and_artifact_sha256_start',{})
        relative=str((folder/'manifest.json').relative_to(ROOT))
        expected=snapshots.get(relative,audit.get('parent_manifest_sha256'))
        if expected!=sha(folder/'manifest.json'):raise RuntimeError('Audit does not bind current upstream manifest')
    sources=[Path(__file__).resolve(),ROOT/'scripts/plot_development_structure.py']
    files=[d/f for d in (NEW,OLD) for f in ('manifest.json','independent_audit.json','client_paths.csv.gz')]
    files+=[NEW/'inner_gate_summaries.csv.gz',NEW/'fit_admission.csv.gz',ROOT/'docs/simple_execution_comparison_protocol.json',ROOT/'docs/simple_drift_cost_interpretation.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in files};start={str(p.relative_to(ROOT)):sha(p) for p in sources}
    old=pd.read_csv(OLD/'client_paths.csv.gz');new=pd.read_csv(NEW/'client_paths.csv.gz')
    frame=pd.concat([old.loc[old.policy.isin(METHODS[:2])],new.loc[new.policy.isin(METHODS[2:])]],ignore_index=True)
    reference=new.loc[new.policy.eq('annual_then_pending')]
    keys=['repeat_id','outer_fold','context','scenario_id','policy']
    refs=old.loc[old.policy.isin(('annual_then_pending','equal_thirds_annual'))].merge(new,on=keys,suffixes=('_old','_new'),validate='one_to_one')
    if len(refs)!=1200:raise ValueError('Both hypothetical references require all original repeat identities')
    for field in FIN:
        if not np.allclose(refs[field+'_old'].astype(float),refs[field+'_new'].astype(float),atol=1e-12,rtol=0,equal_nan=True):raise ValueError('Reference changed: '+field)
    summary=summarize(frame);paths,pairs=paired(frame,reference)
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    for name,data in [('repeat_methods',summary),('common_path_pairs',paths),('paired_comparisons',pairs)]:data.to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
    plot(summary)
    writej(OUT/'findings.json',dict(repeat_methods=summary.to_dict('records'),paired_comparisons=pairs.to_dict('records'),
        interpretation='Diagnostic positive controls; no new policy/threshold or final architecture selection',original_paths=120,repeat_count=5,overlapping_fits=25,
        reference_identity='1200 old/new hypothetical reference records agree; no rejected-path fallback',final_scenarios_loaded=False))
    end={str(p.relative_to(ROOT)):sha(p) for p in files};source_end={str(p.relative_to(ROOT)):sha(p) for p in sources}
    if before!=end or start!=source_end:raise RuntimeError('Source or input changed during saved-result summary')
    outputs={str(p.relative_to(ROOT)):sha(p) for d in (OUT,FIG) for p in d.iterdir() if p.is_file() and p.name!='manifest.json'}
    writej(OUT/'manifest.json',dict(status='complete' if args.confirm_visual_review else 'rendered_pending_visual_review',completed_utc=datetime.now(timezone.utc).isoformat(),
        sources=start,sources_at_end=source_end,input_sha256=before,input_sha256_at_end=end,outputs_sha256=outputs,
        row_counts=dict(repeat_methods=len(summary),common_path_pairs=len(paths),paired_comparisons=len(pairs)),
        visual_review_confirmed=args.confirm_visual_review,raw_market_loaded=False,final_scenarios_loaded=False,new_fit_or_selection=False))
    print('Saved read-only comparison:',OUT)


if __name__=='__main__':main()
