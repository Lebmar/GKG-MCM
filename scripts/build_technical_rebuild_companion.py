"""Build a saved-evidence technical companion; no market parser/model imports."""
from __future__ import annotations
import ast
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/technical_rebuild_2026_10_19.ipynb'
ASSETS=ROOT/'notebooks/technical_rebuild_2026_10_19_assets'
VERSIONS={'V2':'state_budget_rebuild_v1','V2b':'state_budget_mitigation_v1','V3':'state_budget_cushion_v1'}
STABILITY={'V2':'state_budget_stability_v1','V2b':'state_budget_mitigation_stability_v1','V3':'state_budget_cushion_stability_v1'}
ANALYSES={'V2':'state_budget_analysis_v1','V2b':'state_budget_mitigation_analysis_v1','V3':'state_budget_cushion_analysis_v1'}
COLORS={'V2':'#7B8494','V2b':'#236B9B','V3':'#B87A16','同目标B':'#5F6641','等权年调':'#B15C70'}


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return d.hexdigest()


def readj(path):return json.loads(Path(path).read_text())


def prepare_data():
    """Verify producer output hashes before reading numerical saved artifacts."""
    ASSETS.mkdir(parents=True,exist_ok=True);sources={};statuses=[]
    def tracked(path):
        path=Path(path);name=str(path.relative_to(ROOT))
        if 'final' in name or name.startswith('data/') or any(s in name for s in ('121_180','121-180')):
            raise RuntimeError('Companion may read development saved artifacts only')
        sources[name]=sha(path);return path
    def producer(directory,audit=None,binding=None):
        folder=ROOT/'results'/directory;manifest=readj(tracked(folder/'manifest.json'))
        if manifest.get('status') not in ('complete','passed') or manifest.get('final_scenarios_loaded') is not False:
            raise RuntimeError('Complete development artifact required: '+directory)
        for name,digest in manifest['outputs_sha256'].items():
            p=folder/name
            if sha(tracked(p))!=digest:raise RuntimeError('Saved output changed: '+str(p))
        if audit:
            receipt=readj(tracked(ROOT/audit));mh=sha(folder/'manifest.json')
            if receipt.get('status')!='passed':raise RuntimeError('Independent audit not passed')
            manifest_name=str((folder/'manifest.json').relative_to(ROOT))
            links=[receipt.get('parent_manifest_sha256'),receipt.get('producer_manifest_sha256')]
            links += [receipt.get(key,{}).get(manifest_name) for key in ('audit_sources_and_inputs','sources_and_inputs_at_start','sources_and_artifacts_at_start')]
            if mh not in links:raise RuntimeError('Audit does not bind producer manifest: '+directory)
        statuses.append(dict(artifact=directory,status='完成并独立核验' if audit else '完成；保存结果后处理',manifest_sha256=sha(folder/'manifest.json')))
        return folder,manifest
    canonical={};frames={};summaries=[];ordinary=[]
    for version,directory in VERSIONS.items():
        folder,m=producer(directory,f'results/{directory}/independent_audit.json');canonical[version]=folder
        df=pd.read_csv(folder/'path_metrics.csv.gz',float_precision='round_trip');frames[version]=df
        sm=pd.read_csv(folder/'summary.csv.gz',float_precision='round_trip');row=sm.query('scope=="outer_selected"').iloc[0].to_dict();row['version']=version;summaries.append(row)
        groups=readj(folder/'trace_groups.json');n=0;active=np.zeros(30)
        with np.load(folder/'traces.npz',allow_pickle=False) as archive:
            for g in groups:
                if g['kind']!='outer_selected' or not g['array_names']:continue
                flags=archive[g['group_id']+'__automatic_service'];n+=len(flags);active+=flags.sum(axis=0)
        assert n==row['financial_n']
        ordinary.extend(dict(version=version,year=t+1,ordinary_service_fraction=active[t]/n if n else np.nan,financial_n=n) for t in range(30))
    admission=pd.DataFrame(summaries)
    for version in ('V2b','V3'):
        q=frames[version].query('scope=="outer_selected"');assert len(q)==120 and q.assigned.all()
    distribution=[]
    for version in ('V2b','V3'):
        q=frames[version].query('scope=="outer_selected"')
        distribution.extend(dict(series=version,scenario_id=int(r.scenario_id),cagr=r.cagr,mdd=r.mdd) for r in q.itertuples())
    for policy,label in (('same_target_B','同目标B'),('exact_thirds_annual','等权年调')):
        a=frames['V2b'].query('scope=="outer_reference" and policy==@policy').sort_values('scenario_id')
        b=frames['V3'].query('scope=="outer_reference" and policy==@policy').sort_values('scenario_id')
        assert np.array_equal(a[['cagr','mdd']].values,b[['cagr','mdd']].values)
        distribution.extend(dict(series=label,scenario_id=int(r.scenario_id),cagr=r.cagr,mdd=r.mdd) for r in a.itertuples())
    repeat=[]
    for version,directory in STABILITY.items():
        audit='results/cushion_stability_audit_v1/manifest.json' if version=='V3' else 'results/technical_stability_audit_v1/manifest.json'
        folder,m=producer(directory,audit);df=pd.read_csv(folder/'repeat_summary.csv.gz',float_precision='round_trip')
        q=df.query('policy=="state_budget_learner"').copy();assert len(q)==5
        repeat.extend(dict(version=version,repeat_id=int(r.repeat_id),financial_n=int(r.financial_n),initial_coverage=r.initial_coverage,breach_count=r.breach_count,breach_fraction=r.breach_fraction,ordinary_service_fraction=r.mean_automatic_fraction) for r in q.itertuples())
    prediction=[]
    for version,directory in ANALYSES.items():
        folder,m=producer(directory);d=readj(folder/'prediction_summary.json')
        for r in d['market']:
            if r['phase']=='positive_period':prediction.append(dict(version=version,unit='资产均值预测',phase=r['phase'],forecast=r['forecast'],metric='标准化MSE',value=r['scenario_equal_standardized_mse']))
        for r in d['policy']:
            if r['forecast']=='uniform' and r['phase'] in ('growth','protection','mitigation_trade'):
                prediction.append(dict(version=version,unit='已投入策略一步',phase=r['phase'],forecast='uniform',metric='配对收益RMSE',value=r['benefit_rmse']))
    q1,pm=producer('q1_stability_diagnostic_v1','results/q1_stability_diagnostic_v1/independent_cache_audit.json')
    q1summary=readj(q1/'own_horizon_summary.json');q1table=[]
    for q in q1summary:
        m=q['model'];q1table.append(dict(profile=q['profile'],horizon=q['horizon'],L=q['L'],financial_n=m['financial_n'],review_n=m['review_n'],breach_count=m['breach_count'],median_cagr=m['median_cagr'],p10_cagr=m['p10_cagr'],p90_mdd=m['p90_mdd']))
    near=pd.read_csv(q1/'near_optimal_members.csv.gz',float_precision='round_trip').query('job_id=="c0_refit" and profile=="B" and screen=="selected_gamma"').copy()
    sets=readj(q1/'individual_feasible_sets.json');near_meta=next(q for q in sets if q['job_id']=='c0_refit' and q['profile']=='B' and q['screen']=='selected_gamma')
    weights=np.array([ast.literal_eval(v) for v in near.weights]);assert len(near)==5 and near_meta['near_optimal_n']==5
    near[['equity','bond','bill']]=weights
    liquidity,lm=producer('liquidity_capacity_diagnostic_v1','results/liquidity_capacity_diagnostic_v1/independent_audit.json')
    annual_liquidity=pd.read_csv(liquidity/'annual_capacity.csv.gz',float_precision='round_trip')
    liqsummary=pd.DataFrame(readj(liquidity/'summary.json'))
    physical,phm=producer('guard_physical_feasibility_v1','results/guard_physical_feasibility_v1/independent_witness_audit.json')
    physical_summary=[]
    for r in readj(physical/'summary.json'):
        physical_summary.append({**{k:v for k,v in r.items() if k!='classification_counts'},**r['classification_counts']})
    physical_summary=pd.DataFrame(physical_summary)
    classification_keys=['numerical_physical_feasible_witness_above_budget','all_eight_numerical_optima_below_budget','numerical_boundary_unresolved','incomplete_solver_or_certificate_evidence','solver_or_certificate_unresolved']
    assert len(physical_summary)==4
    assert np.array_equal(physical_summary[classification_keys].sum(axis=1),physical_summary.first_guard_n)
    stress=None;stress_status='等待独立审计；不展示未审计数值'
    stress_dir=ROOT/'results/state_budget_mitigation_stress_v1'
    candidates=[stress_dir/'independent_audit.json',stress_dir/'independent_stress_audit.json',ROOT/'results/state_budget_mitigation_stress_audit_v1/manifest.json']
    for candidate in candidates:
        if candidate.exists():
            folder,m=producer('state_budget_mitigation_stress_v1',str(candidate.relative_to(ROOT)))
            stress=pd.read_csv(folder/'summary.csv.gz',float_precision='round_trip');stress_status='完成并独立核验';break
    if stress is None:statuses.append(dict(artifact='state_budget_mitigation_stress_v1',status=stress_status,manifest_sha256=sha(tracked(stress_dir/'manifest.json'))))
    # Prespecified illustrative rule: worst V2b OOF MDD, tie smallest original ID.
    worst=frames['V2b'].query('scope=="outer_selected" and assigned').sort_values(['mdd','scenario_id'],ascending=[False,True]).iloc[0]
    sid=int(worst.scenario_id);jid=worst.job_id;folder=canonical['V2b'];groups=readj(folder/'trace_groups.json');severe=[];guard=None
    with np.load(folder/'traces.npz',allow_pickle=False) as archive:
        for policy,label in (('state_budget_learner','V2b'),('same_target_B','同目标B'),('exact_thirds_annual','等权年调')):
            g=next(g for g in groups if g['job_id']==jid and ((g['kind']=='outer_selected') if policy=='state_budget_learner' else g['kind']=='reference' and g['policy']==policy))
            i=g['scenario_ids'].index(sid);prefix=g['group_id']+'__'
            if label=='V2b':guard=int(archive[prefix+'first_guard_trigger'][i])
            for t in range(31):severe.append(dict(scenario_id=sid,job_id=jid,series=label,endpoint=t,real_wealth=float(archive[prefix+'real_wealth'][i,t]),running_mdd=float(archive[prefix+'running_mdd'][i,t])))
    tables=dict(admission=admission,distribution=pd.DataFrame(distribution),repeats=pd.DataFrame(repeat),ordinary=pd.DataFrame(ordinary),prediction=pd.DataFrame(prediction),q1=pd.DataFrame(q1table),near=near,liquidity=annual_liquidity,liquidity_summary=liqsummary,physical_feasibility=physical_summary,severe=pd.DataFrame(severe))
    if stress is not None:tables['stress']=stress
    for name,table in tables.items():table.to_csv(ASSETS/(name+'_source.csv'),index=False)
    metadata=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),sources=sources,statuses=statuses,stress_status=stress_status,severe_rule='argmax V2b OOF MDD, exact ties minimum scenario ID',severe_id=sid,severe_first_guard=guard,
        B_full_gamma=near_meta['gamma'],B_epsilon=.001,B_near_n=len(near),final_values_read=False,raw_values_read=False,new_policy_runs=0,new_fits=0)
    (ASSETS/'source_manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    return dict(tables=tables,meta=metadata,colors=COLORS,assets=ASSETS)


SETUP='''from pathlib import Path
import sys, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Markdown
ROOT = Path.cwd()
if not (ROOT/'scripts/build_technical_rebuild_companion.py').exists():
    ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT/'scripts'))
from build_technical_rebuild_companion import prepare_data
state = prepare_data(); tables=state['tables']; meta=state['meta']; colors=state['colors']; assets=state['assets']
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':12,'axes.titlesize':15,'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'savefig.facecolor':'white'})
def finish(fig, name, source):
    fig.tight_layout(); fig.savefig(assets/(name+'.png'),dpi=240,bbox_inches='tight'); fig.savefig(assets/(name+'.svg'),bbox_inches='tight')
    source.to_csv(assets/(name+'_data.csv'),index=False)
    display(fig); plt.close(fig)
    display(Markdown('图数据：`'+name+'_data.csv`；输入SHA256见同目录 `source_manifest.json`。'))
pd.set_option('display.max_columns',12)
'''

CHARTS=[]
CHARTS.append(('准入与转介分开看',r'''d=tables['admission'].set_index('version').loc[['V2','V2b','V3']]
fig,ax=plt.subplots(figsize=(12,5.4)); x=np.arange(3)
initial=120-d.financial_n; referral=d.support_review_n; remaining=d.financial_n-referral
ax.bar(x,initial,color='#DCE0E5',edgecolor='#6A707A',label='初始未投入 Review')
ax.bar(x,referral,bottom=initial,color='#E4C48A',edgecolor='#87631F',label='已投入，之后触发支持转介')
ax.bar(x,remaining,bottom=initial+referral,color='#236B9B',label='已投入，未触发支持转介')
for i,v in enumerate(d.index):
    ax.text(i,123,f"投入 {int(d.loc[v,'financial_n'])}/120\n转介 {int(referral.loc[v])}",ha='center',fontsize=12)
ax.set(xticks=x,xticklabels=['V2\n条件投入 N=24','V2b\n投入 N=120','V3\n投入 N=120'],ylim=(0,143),ylabel='原始情景数（每版120）',title='初始覆盖与投入后转介不是同一指标')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3);finish(fig,'01_admission_referral',d.reset_index())
display(Markdown('**负面结果：** V2 的较高持续服务比例只针对投入的24条路径；不能当作全120条路径的表现。V2b/V3虽全数初始投入，绝大多数路径随后进入转介持有。'))
'''))
CHARTS.append(('同一120条路径的收益与回撤分布',r'''d=tables['distribution'];fig,axes=plt.subplots(1,2,figsize=(14,5.4))
styles={'V2b':('-',2.2),'V3':('--',2.2),'同目标B':('-.',1.8),'等权年调':(':',2.4)}
for label in ['V2b','V3','同目标B','等权年调']:
    q=d[d.series==label]
    for ax,key in zip(axes,['cagr','mdd']):
        v=np.sort(q[key]);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=label,color=colors[label],ls=styles[label][0],lw=styles[label][1])
for ax,key,title in zip(axes,['cagr','mdd'],['30年实际年化收益 CAGR','30年原始峰值最大回撤 MDD']):
    ax.set(xlabel=title,ylabel='经验累计比例（N=120）',ylim=(0,1.02));ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
axes[1].axvline(.4,color='#444444',ls='--',lw=1.2,label='L=40%');axes[0].legend();axes[1].legend()
fig.suptitle('同一 OOF 120条合成路径；经验分布不是置信区间',fontsize=16,y=1.04)
finish(fig,'02_same120_ecdf',d)
display(Markdown('V2b/V3压低了此开发样本的回撤，也牺牲了收益。这里不加入只投入24条路径的V2，以免混淆比较总体。'))
'''))
CHARTS.append(('五次划分的稳定性',r'''d=tables['repeats'];fig,axes=plt.subplots(1,3,figsize=(16,5.2))
for version,marker in [('V2','o'),('V2b','s'),('V3','^')]:
    q=d[d.version==version].sort_values('repeat_id')
    for ax,key in zip(axes,['initial_coverage','breach_fraction','ordinary_service_fraction']):
        ax.plot(q.repeat_id,q[key],marker=marker,color=colors[version],label=version,lw=1.8)
for ax,title in zip(axes,['初始投入 / 原始120','超限 / 各自投入路径','普通自动服务期 / 投入路径']):
    ax.set(xticks=range(1,6),xlabel='重复划分编号',ylabel='比例',title=title);ax.yaxis.set_major_formatter(PercentFormatter(1))
axes[0].set_ylim(0,1.08);axes[2].set_ylim(0,1.02);axes[1].set_ylim(bottom=0)
axes[0].legend()
fig.suptitle('五次划分重用同一120条路径：不是 N=600，不平均分位数\nV2重复5全Review；风险与服务统计为NULL',fontsize=14,y=1.08)
finish(fig,'03_partition_stability',d)
display(d[['version','repeat_id','financial_n','breach_count','ordinary_service_fraction']].round(4))
'''))
CHARTS.append(('普通服务在第几年结束',r'''d=tables['ordinary'];fig,axes=plt.subplots(1,2,figsize=(14,5.3),sharey=True)
for version in ['V2','V2b','V3']:
    q=d[d.version==version];ax=axes[0] if version=='V2' else axes[1]
    ax.step(q.year,q.ordinary_service_fraction,where='mid',color=colors[version],label=f"{version}（投入 N={int(q.financial_n.iloc[0])}）",ls='--' if version=='V3' else '-',lw=2)
for ax,title in zip(axes,['V2：仅条件投入24条路径','V2b / V3：同一投入120条路径']):
    ax.set(xlim=(1,30),xticks=[1,5,10,15,20,25,30],ylim=(0,1.03),xlabel='原始年份',title=title);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend()
axes[0].set_ylabel('该年仍提供普通自动服务的比例')
fig.suptitle('触发guard的当年即排除普通服务；当年仍可能执行一次自动缓解',fontsize=15,y=1.04)
finish(fig,'04_ordinary_service',d)
'''))
CHARTS.append(('费用与预测假设的压力检查',r'''fig,axes=plt.subplots(2,2,figsize=(15,10))
if 'stress' not in tables:
    plt.close(fig);fig,ax=plt.subplots(figsize=(12,2.8));ax.axis('off')
    ax.text(.03,.6,'V2b压力结果等待独立审计；本次不展示未审计数值。\n审计完成后重跑本notebook，将补齐费用、通胀与pooled支持三个情景。',transform=ax.transAxes,fontsize=15)
    stress_source=pd.DataFrame([{'status':meta['stress_status']}])
else:
    s=tables['stress'].query('cohort=="outer_OOF"').copy();costs={'fee_0000':0.,'fee_00125':.00125,'base':.0025,'fee_005':.005,'fee_01':.01}
    for policy,label in [('state_budget_learner','V2b'),('same_target_B','同目标B'),('exact_thirds_annual','等权年调')]:
        q=s[(s.policy==policy)&s.setting_id.isin(costs)].copy();q['cost']=q.setting_id.map(costs);q=q.sort_values('cost')
        for ax,key in zip(axes[0],['median_cagr','mean_automatic_fraction']):ax.plot(q.cost,q[key],marker='o',label=label,color=colors[label])
    for ax,title in zip(axes[0],['各费率下的中位CAGR','各费率下的普通服务期比例']):
        ax.set(xlabel='单次买入费用率',title=title);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend()
    q=s.query('policy=="state_budget_learner"').set_index('setting_id').loc[['base','actual_inflation_plus_02','pooled_support']]
    names=['基准','实际通胀+2pp','跨年份 pooled 支持']
    axes[1,0].bar(names,q.initial_coverage,color='#236B9B');axes[1,0].set(title='V2b 初始覆盖（原始120）',ylim=(0,1.08));axes[1,0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1,1].bar(names,q.breach_fraction,color='#B87A16');axes[1,1].set(title='V2b 超限比例（投入分母）',ylim=(0,.25),xlim=(-.6,2.6),xticks=range(3),xticklabels=names);axes[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    for i,(_,row) in enumerate(q.iterrows()):
        if row.financial_n==0:axes[1,1].text(i,.025,'Review\n风险=NULL',ha='center',fontsize=12)
        else:axes[1,1].text(i,row.breach_fraction+.008,f"{int(row.breach_count)}/{int(row.financial_n)}",ha='center',fontsize=12)
    stress_source=s
fig.suptitle('冻结V2b的情景检查；不重新选择参数，也不把Review填成0风险',fontsize=16,y=1.01)
finish(fig,'05_stress',stress_source)
display(Markdown('**状态：** '+meta['stress_status']+'。实际通胀平移属于假设改变；pooled支持属于更保守的独立压力情景，不是事后补救主规则。'))
'''))
CHARTS.append(('Q1 的权重不是三个独立区间',r'''d=tables['near'].sort_values('candidate_local_index');fig,axes=plt.subplots(1,2,figsize=(14,5.6));x=np.arange(len(d));bottom=np.zeros(len(d))
for key,label,color in [('equity','股票','#236B9B'),('bond','债券','#B87A16'),('bill','国库券','#858B62')]:
    axes[0].barh(x,d[key],left=bottom,label=label,color=color);bottom+=d[key].to_numpy()
axes[0].set(yticks=x,yticklabels=[f"候选 {int(v)}" for v in d.candidate_local_index],xlabel='完整组合权重（每行合计100%）',title='B：完整ε近最优集合中的5个耦合组合',xlim=(0,1));axes[0].xaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13))
axes[1].bar(x,d.gap_to_individual_best,color='#236B9B');axes[1].axhline(.001,color='#444444',ls='--',label='ε=0.001 年化log增长');axes[1].set(xticks=x,xticklabels=[str(int(v)) for v in d.candidate_local_index],xlabel='候选编号',ylabel='与训练最优的年化log增长差',title=f"技术筛选 γ={meta['B_full_gamma']:g}；不是心理风险厌恶")
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13));finish(fig,'06_q1_coupled_weights',d)
display(Markdown('这是B原始15年目标的静态个体可行集诊断，省略共同三产品限制；不能用它替换冻结菜单，也不能从每列最小/最大自由拼接新权重。ε是年化log增长差，不是CAGR差0.1个百分点。'))
'''))
CHARTS.append(('目标券份额与年末漂移',r'''d=tables['liquidity'];fig,axes=plt.subplots(1,3,figsize=(16,5.3))
for ax,profile in zip(axes,['B','C','E']):
    q=d[d.profile==profile];summary=tables['liquidity_summary'].query('profile==@profile').iloc[0]
    v=np.sort(q.end_bill_share);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color='#236B9B',lw=2)
    ax.axvline(summary.target_floor,color='#444444',ls='--',label=f"目标下限 {summary.target_floor:.0%}")
    ax.set(title=f"{profile}：原H={int(summary.horizon)}，年度端点 n={len(q)}",xlabel='年度末国库券份额',ylabel='年度端点经验累计比例',ylim=(0,1.02),xlim=(0,.7),xticks=[0,.2,.4,.6]);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend()
fig.suptitle('静态Q1年度参考；年度观测互相依赖，未推断年内提现能力',fontsize=16,y=1.04)
finish(fig,'07_bill_floor_drift',d)
display(tables['liquidity_summary'][['profile','horizon','financial_n','review_n','mean_fraction_end_below_target_floor','maximum_hypothetical_liquidation_drag']].round(5))
display(Markdown('A/D为Review，没有金融数值或年度端点。年末低于目标券下限是漂移，不等于原始目标违反下限。假想全转券费用只计容量比值，未从投资路径再次扣除。'))
'''))
CHARTS.append(('事前规则选出的严重合成案例',r'''d=tables['severe'];fig,axes=plt.subplots(1,2,figsize=(14,5.6))
for label,ls in [('V2b','-'),('同目标B','--'),('等权年调',':')]:
    q=d[d.series==label]
    axes[0].plot(q.endpoint,q.real_wealth,label=label,color=colors[label],ls=ls,lw=2)
    axes[1].plot(q.endpoint,q.running_mdd,label=label,color=colors[label],ls=ls,lw=2)
for ax in axes:
    ax.set(xlim=(0,30),xticks=[0,5,10,15,20,25,30],xlabel='原始年度端点');ax.legend()
    if meta['severe_first_guard']>0:ax.axvline(meta['severe_first_guard']-1,color='#444444',ls='-.',lw=1.2)
axes[0].set(ylabel='实际财富（初始100）',title='原始CPI下的完整财富路径')
axes[1].set(ylabel='自原始历史峰值累计MDD',title='峰值与MDD不在转介后重置');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].axhline(.4,color='#444444',ls='--',lw=1.2)
fig.suptitle(f"合成情景ID {meta['severe_id']}：V2b OOF最大MDD；同值取最小ID",fontsize=16,y=1.04)
finish(fig,'08_severe_synthetic_path',d)
display(Markdown(f"点划线为guard触发的决策时刻（第{meta['severe_first_guard']}年开始前）。这是按固定最大MDD规则选取的合成案例，不是历史危机事件或随机代表情景。"))
'''))
CHARTS.append(('有限提案失败与宽持仓可行性分开',r'''d=tables['physical_feasibility'];fig,axes=plt.subplots(1,2,figsize=(14,5.8))
keys=['numerical_physical_feasible_witness_above_budget','all_eight_numerical_optima_below_budget']
for ax,scope,title in zip(axes,['outer_selected','full_selected'],['OOF：各自训练支持','全开发重拟合：自身纳入训练支持']):
    q=d[d.scope==scope].set_index('version').loc[['V2b','V3']];x=np.arange(2)
    feasible=q[keys[0]];below=q[keys[1]]
    ax.bar(x,feasible,color='#77A8C6',edgecolor='#236B9B',label='宽持仓存在可行数值见证')
    ax.bar(x,below,bottom=feasible,color='#D5B579',edgecolor='#87631F',label='8区域数值最优均低于预算')
    for i,v in enumerate(q.index):
        ax.text(i,feasible.loc[v]/2,str(int(feasible.loc[v])),ha='center',va='center')
        ax.text(i,feasible.loc[v]+below.loc[v]/2,str(int(below.loc[v])),ha='center',va='center')
        ax.text(i,q.loc[v,'first_guard_n']+3,f"首次guard {int(q.loc[v,'first_guard_n'])}",ha='center')
    ax.set(xticks=x,xticklabels=['V2b','V3'],ylim=(0,134),ylabel='首次guard路径数',title=title)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.08),ncol=2)
fig.suptitle('只解释触发当期的经验支持与规范预算；不改原策略行动',fontsize=16,y=1.04)
finish(fig,'09_guard_physical_feasibility',d)
display(Markdown('**关键判断：** OOF中，V2b的109次首次guard里20次存在更宽持仓的可行数值见证，V3为13/108；有限提案失败不能一概归为物理不可行。其余89/95次获得全部8个买卖区域的数值证书，表明该次训练支持下最优值仍低于预算。证书经过独立代数与KKT检查，但不是精确算术不可能性证明，也不是未来市场保证。全开发结果与OOF分开展示，不能合并成新的独立样本。'))
display(d[['version','scope','first_guard_n',*keys,'numerical_boundary_unresolved','incomplete_solver_or_certificate_evidence','solver_or_certificate_unresolved']])
'''))


def build():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbconvert import HTMLExporter
    cells=[]
    def md(x):
        # Plain mathematical notation keeps the exported companion offline.
        formulas={r'$a m w+(1-am)e_{bill}$':'`a·m·w+(1−a·m)·e_bill`',r'$m=\operatorname{clip}[(L-d)/(L(1-d)),0,1]$':'`m=clip((L−d)/(L(1−d)),0,1)`',r'$d=1-R/P$':'`d=1−R/P`',r'$F=(1-L)P$':'`F=(1−L)P`',r'$\max(F,R-\rho(R-F))$':'`max(F,R−ρ(R−F))`',r'$drag=c(1-s_{bill})/(1+c)$':'`drag=c·(1−s_bill)/(1+c)`',r'$capacity=1-drag$':'`capacity=1−drag`'}
        for source,replacement in formulas.items():x=x.replace(source,replacement)
        cells.append(nbformat.v4.new_markdown_cell(x))
    def code(x):cells.append(nbformat.v4.new_code_cell(x))
    md('# 技术研究伴随文档 · 2026-10-19\n\n**Team87 · Chengxuan YUAN · Lebmar**\n\n这是模型开发、审计与负面证据的可复现companion，不是参赛论文。只读取冻结开发结果；不运行新模型、不拟合、不读取原始市场CSV或121–180最终值。')
    md('## tl;dr\n\n本轮比较V2、V2b与V3。初始覆盖、投入后的普通服务持续性、回撤与收益分别衡量；不能把“初始全投入”解释为“持续自动服务”。以下数字由保存结果在执行时读取并核验。')
    code(SETUP)
    code("q=tables['admission'].set_index('version'); display(Markdown(f\"V2初始投入 **{int(q.loc['V2','financial_n'])}/120**；V2b与V3均为 **120/120**。其OOF普通服务期平均比例分别为 **{q.loc['V2b','mean_automatic_fraction']:.1%} / {q.loc['V3','mean_automatic_fraction']:.1%}**；中位CAGR为 **{q.loc['V2b','median_cagr']:.2%} / {q.loc['V3','median_cagr']:.2%}**。低回撤伴随转介与收益代价。\"))")
    md('## Context & Methods\n\n### 关键假设与边界\n\n开发情景是合成年度路径。研究已反复使用开发数据，历史最终数据曝光也保留记录；此次OOF划分和复现不会产生新的盲测。原始H、CPI、历史峰值及MDD不重置；实数手续费按买入金额支付一次。无置信区间，无“获奖”或真实未来保护保证。\n\n- V2：一步费后配对增长目标与经验联合支持过滤。\n- V2b：首次支持族无可行行动时，仅一次最大化扣费后支持最坏下一期财富，随后进入Review持有。\n- V3：只把主要提案的临时目标改为 $a m w+(1-am)e_{bill}$，其中 $m=\\operatorname{clip}[(L-d)/(L(1-d)),0,1]$，$d=1-R/P$；原始9点缓解保持不缩放。\n\n$F=(1-L)P$，预算下限为 $\\max(F,R-\\rho(R-F))$。经验支持上的一步可行，不等于真实未来支持包含、递归可行或全程回撤保证。')
    md('## Data\n\n数据源是下列已保存产物；每个读取文件保留SHA256，图旁另存CSV数据。`source_manifest.json`列出全部产物身份。独立审计与同源复现各自提供不同证据。')
    code("display(pd.DataFrame(meta['statuses'])[['artifact','status']]); display(Markdown(f\"已绑定保存文件 **{len(meta['sources'])}** 个；原始市场读取=0，模型运行=0。压力状态：{meta['stress_status']}。\"))")
    md('## Results')
    for i,(title,body) in enumerate(CHARTS,1):
        md(f'### {i}. {title}');code(body)
        if i==6:
            md('### Q1各自原始期限：Review保留NULL\n\nQ1为静态年度参考；不能把它的有效分配当作Q2自动执行准入。分位数是各自已投入情景的经验分布统计。')
            code("q=tables['q1'].rename(columns={'profile':'客户','horizon':'原H','financial_n':'投入数','review_n':'Review数','breach_count':'超限数','median_cagr':'中位CAGR','p10_cagr':'P10 CAGR','p90_mdd':'P90 MDD'}); display(q.style.format({'L':'{:.0%}','中位CAGR':'{:.2%}','P10 CAGR':'{:.2%}','P90 MDD':'{:.2%}'},na_rep='NULL'))")
    md('### 10. 一步预测诊断与完整策略结果相互区分\n\n资产MSE每个OOF市场/年份只出现一次，不因服务拒绝而删行；策略预测只在有效决策时存在。成员与决策分别聚合，先按情景平均。缓解行动的配对增长分数只是注释，实际排序目标是支持最坏财富。')
    code("display(tables['prediction'].round(6))")
    md('### 11. 流动性容量的适用边界\n\n年度末假想全转券拖累为 $drag=c(1-s_{bill})/(1+c)$，$capacity=1-drag$。这笔假想手续费没有从投资路径再次扣除；它不是客户支出计划、年内流动性承诺，也不是第9节联合支持预算可行性的另一种说法。')
    md('## Takeaways\n\n1. 保留负面结果：覆盖、普通服务持续性、收益和风险同时报告，不用单一漂亮指标替代。\n2. V2与全120投入版本的条件总体不同；同120比较只在明确匹配的V2b/V3及参考中进行。五次划分是同120情景的依赖重复，不是N600。\n3. V3增加状态依赖提案，不保证优于V2b；最终技术候选的锁定须结合完整策略、稳定性、成本和复杂度判断。\n4. 所有图表都是开发证据，不能抹除历史曝光或声称新盲测。当前文档没有读取最终集。')
    code("from build_technical_rebuild_companion import sha\nassert all(sha(ROOT/name)==digest for name,digest in meta['sources'].items())\ndisplay(Markdown('保存产物在notebook执行前后SHA256一致。所有图已输出PNG 240dpi、可编辑文字SVG与对应CSV。'))")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'technical-dev','display_name':'Frozen development companion','language':'python'},'language_info':{'name':'python','version':sys.version.split()[0]}})
    nbformat.validate(nb);ASSETS.mkdir(parents=True,exist_ok=True);NOTEBOOK.parent.mkdir(exist_ok=True)
    kernels=ASSETS/'jupyter/kernels';spec=kernels/'technical-dev';spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(argv=[str(ROOT/'.venv/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Technical development companion',language='python')))
    manager=KernelManager(kernel_name='technical-dev',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(ROOT)}})
    client.execute();nbformat.validate(nb);nbformat.write(nb,NOTEBOOK)
    exporter=HTMLExporter(template_name='lab');exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    # Static text/tables/embedded PNG outputs remain usable without a CDN.
    import re
    html=re.sub(r'<script\b[^>]*\bsrc=["\']https?://[^"\']+["\'][^>]*>\s*</script>','',html,flags=re.I)
    responsive='''<style id="technical-companion-reading-style">
html,body {width:auto!important;min-width:0!important;max-width:100%!important;box-sizing:border-box;margin:0;}
body {padding:18px 14px!important;line-height:1.55;}
.jp-Notebook {width:auto!important;min-width:0!important;max-width:1180px!important;margin:0 auto!important;padding:12px!important;}
.jp-Cell,.jp-Cell-inputWrapper,.jp-OutputArea,.jp-OutputArea-child,.jp-OutputArea-output,.jp-RenderedHTMLCommon {min-width:0!important;max-width:100%!important;box-sizing:border-box;}
.jp-RenderedHTMLCommon {overflow-wrap:anywhere;word-break:normal;}
.jp-RenderedHTMLCommon table {display:block;width:max-content;max-width:100%;overflow-x:auto;}
.jp-OutputArea-output {overflow-x:auto;}
pre {max-width:100%;overflow-x:auto;white-space:pre;}
img,svg {max-width:100%!important;height:auto!important;}
@media(max-width:600px){body{padding:8px!important}.jp-Notebook{padding:4px!important}h1{font-size:26px!important}h2{font-size:22px!important}h3{font-size:19px!important}.jp-Cell{padding-left:0!important;padding-right:0!important}}
</style>'''
    html=html.replace('</head>',responsive+'\n</head>')
    htmlpath=NOTEBOOK.with_suffix('.executed.html');htmlpath.write_text(html)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(ASSETS.glob('*')) if p.is_file() and p.name not in ('build_manifest.json','visual_qa.json')}
    manifest=dict(status='executed',notebook=str(NOTEBOOK.relative_to(ROOT)),notebook_sha256=sha(NOTEBOOK),html=str(htmlpath.relative_to(ROOT)),html_sha256=sha(htmlpath),builder_sha256=sha(__file__),figure_count=len(CHARTS),artifact_sha256=artifacts,visual_QA='pending_actual_image_review',new_policy_runs=0,raw_market_reads=0,final_values_read=False)
    (ASSETS/'build_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='executed',notebook=str(NOTEBOOK),figures=len(CHARTS),html=str(htmlpath)),ensure_ascii=False))


if __name__=='__main__':build()
