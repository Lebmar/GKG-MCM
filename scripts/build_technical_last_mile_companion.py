"""Build a local technical reading companion from audited development outputs.

No market parser, policy execution, parameter choice or final evaluation.
Standalone research figures retain their complete source CSVs.
"""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import html
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OP = ROOT/'results/review_operation_diagnostic_v1'
SP = ROOT/'results/support_failure_diagnostic_v2'
OUT = ROOT/'notebooks/technical_last_mile_2026_10_19_assets'
PAGE = ROOT/'notebooks/technical_last_mile_2026_10_19.html'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def readj(path): return json.loads(Path(path).read_text())
def readcsv(path): return pd.read_csv(path, float_precision='round_trip')
def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def verify_parent(folder, sources):
    m = readj(folder/'manifest.json')
    a = readj(folder/'independent_audit.json')
    assert m['status'] == 'complete' and a['status'] == 'passed'
    assert m['sources_and_inputs_at_start'] == m['sources_and_inputs_at_end']
    assert m['final_scenarios_loaded'] is False
    assert a['producer_manifest_sha256'] == sha(folder/'manifest.json')
    start, end = {
        'review_operation_diagnostic_v1': ('source_and_input_sha256_at_start', 'source_and_input_sha256_at_end'),
        'support_failure_diagnostic_v2': ('sources_and_artifacts_at_start', 'sources_and_artifacts_at_end')
    }[folder.name]
    assert a[start] == a[end]
    for name, digest in m['outputs_sha256'].items():
        p = folder/name
        assert sha(p) == digest, str(p)
        sources[str(p.relative_to(ROOT))] = digest
    for name in ('manifest.json', 'independent_audit.json'):
        p = folder/name; sources[str(p.relative_to(ROOT))] = sha(p)
    return m, a


def cvar90(values):
    a = sorted(map(float, values), reverse=True); mass = .1*len(a)
    return math.fsum(min(1., max(mass-i, 0.))*v for i, v in enumerate(a))/mass


def build():
    sources = {}
    verify_parent(OP, sources); verify_parent(SP, sources)
    crosswalk = ROOT/'docs/technical_evidence_closure_2026_10_19.json'
    sources[str(crosswalk.relative_to(ROOT))] = sha(crosswalk)
    annual = readcsv(OP/'annual_phases.csv.gz')
    original = readcsv(OP/'baseline_paths.csv.gz')
    static = readcsv(OP/'low_exposure_paths.csv.gz')
    costs = readcsv(OP/'review_cost_paths.csv.gz')
    summaries = readcsv(OP/'financial_summary.csv.gz')
    corners = readcsv(SP/'inflation_four_corner_paths.csv.gz')
    direction = readcsv(SP/'directional_support_periods.csv.gz')
    initial = readcsv(SP/'initial_support_contexts.csv.gz')
    assert len(annual) == 3600 and len(static) == 960 and len(costs) == 600
    assert len(original) == 480 and len(corners) == 960
    all_paths = pd.concat([original, static], ignore_index=True)
    metrics = []
    for policy, frame in all_paths.groupby('policy', sort=True):
        assert sorted(frame.scenario_id) == list(range(1, 121))
        metrics.append(dict(policy=policy, n=120, median_cagr=float(frame.cagr.median()),
            mean_log_growth=float(frame.log_growth.mean()), mean_mdd=float(frame.mdd.mean()),
            cvar90_mdd=cvar90(frame.mdd), breach_n=int(frame.breach.sum()),
            mean_nominal_fees=float(frame.nominal_fees.mean())))
    metrics = pd.DataFrame(metrics)
    aliases = {'main':'M', 'same_target_B':'B', 'same_target_annual':'A', 'exact_thirds_annual':'T'}
    for i, key in enumerate(sorted(static.policy.unique()), 1): aliases[key] = 'S'+str(i)
    metrics['code'] = metrics.policy.map(aliases)
    pairs = []
    with np.load(OP/'traces.npz', allow_pickle=False) as archive:
        for group in readj(OP/'trace_groups.json'):
            if group['kind'] != 'saved_outer_selected': continue
            def field(k): return archive[group['group_id']+'__'+k]
            for i, sid in enumerate(group['scenario_ids']):
                tau = int(field('first_guard_trigger')[i])
                x = dict(scenario_id=sid, job_id=group['group_id'], guard_year=None if tau < 0 else tau,
                         guard_equity=None, terminal_equity=None)
                if tau >= 0:
                    p, e = field('posttrade_holdings')[i, tau-1], field('holdings')[i, -1]
                    x.update(guard_equity=float(p[0]/p.sum()), terminal_equity=float(e[0]/e.sum()))
                pairs.append(x)
    exposure = pd.DataFrame(pairs).sort_values('scenario_id')
    assert exposure.scenario_id.tolist() == list(range(1, 121))
    funded = exposure.dropna(subset=['guard_equity'])
    assert len(funded) == 109
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':12,'axes.titlesize':15,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
        'axes.axisbelow':True,'figure.facecolor':'white','svg.fonttype':'none'})
    figures = []
    def save(fig, name, frame):
        fig.tight_layout(); fig.savefig(OUT/(name+'.png'), dpi=220, bbox_inches='tight')
        fig.savefig(OUT/(name+'.svg'), bbox_inches='tight'); plt.close(fig)
        frame.to_csv(OUT/(name+'_data.csv'), index=False); figures.append(name)

    phases = ['before_guard','guard_year','after_guard','no_guard']
    agg = annual.groupby('phase').agg(periods=('period','size'), equity=('equity_weight','mean'),
        bond=('bond_weight','mean'),bill=('bill_weight','mean')).reindex(phases)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    ax[0].scatter(funded.guard_equity, funded.terminal_equity, color='#236b9b', s=28, alpha=.7)
    upper = max(funded.guard_equity.max(), funded.terminal_equity.max())*1.08
    ax[0].plot([0,upper],[0,upper],ls='--',color='#555555',lw=1)
    ax[0].set(xlabel='转介决策后股票占比',ylabel='原30年到期股票占比',xlim=(0,upper),ylim=(0,upper),title='同109个转介账户的持有后权重漂移')
    bottom = np.zeros(4)
    for asset, label, color in [('equity','股票','#236b9b'),('bond','债券','#b87a16'),('bill','国库券','#87916b')]:
        ax[1].bar(range(4), agg[asset], bottom=bottom, color=color,label=label); bottom += agg[asset].to_numpy()
    ax[1].set(xticks=range(4),xticklabels=['转介前\n704期','触发年\n109期','转介后\n2457期','未转介\n330期'],ylim=(0,1.16),title='分阶段、按路径年加权的平均实际暴露')
    ax[1].legend(loc='upper center',ncol=3)
    for a in ax:
        a.yaxis.set_major_formatter(PercentFormatter(1))
    ax[0].xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('开发OOF；转介后继续持有，原始峰值和期限持续计量',y=1.02,fontsize=16)
    save(fig,'01_review_exposure',exposure)
    agg.to_csv(OUT/'phase_exposure_source.csv')

    fig, ax = plt.subplots(figsize=(12,6.5))
    for row in metrics.itertuples():
        if row.policy == 'same_target_annual': continue  # same physical policy as S7
        main = row.policy == 'main'; is_static = row.policy.startswith('a')
        marker = '*' if main else ('s' if row.policy.endswith('annual') else '^') if is_static else 'D'
        color = '#b87a16' if main else '#236b9b' if is_static else '#596246'
        ax.scatter(row.cvar90_mdd,row.median_cagr,s=210 if main else 70,marker=marker,color=color,
            facecolors='none' if is_static and row.policy.endswith('hold') else color)
        text = row.code+'=A' if row.policy == 'a1.0_annual' else row.code
        offset = (8,-18) if row.policy == 'a1.0_annual' else (6,-15) if row.policy == 'a0.25_hold' else (6,6)
        ax.annotate(text,(row.cvar90_mdd,row.median_cagr),xytext=offset,textcoords='offset points')
    ax.set(xlabel='完整30年 MDD 的最差10%均值（CVaR90）',ylabel='实际年化收益 CAGR 中位数',title='全部预先固定参照：风险、增长和费用仍需共同判断')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.5,-.01,'同120个开发OOF ID；S为假想低暴露参照，未做准入、胜者筛选或严格风险匹配。代码见下表。',ha='center',fontsize=12)
    save(fig,'02_fixed_reference_tradeoffs',metrics)

    main = corners[(corners.cohort=='outer_OOF') & (corners.policy=='state_budget_learner')]
    assert sorted(main.scenario_id) == list(range(1,121))
    labels = ['BB\n原名义／原CPI','BS\n原名义／高CPI','SB\n变名义／原CPI','SS\n变名义／高CPI']
    corner_table = pd.DataFrame([dict(corner=k,breach_count=int(main[k+'_breach'].sum()),
        mean_mdd=float(main[k+'_mdd'].mean()),median_cagr=float(main[k+'_cagr'].median()),
        mean_log_growth=float(main[k+'_log_growth'].mean())) for k in ('BB','BS','SB','SS')])
    fig, ax = plt.subplots(1,2,figsize=(14,5.4))
    ax[0].bar(range(4),corner_table.mean_mdd,color=['#236b9b','#b87a16','#87916b','#b15c70'])
    ax[0].set(xticks=range(4),xticklabels=labels,ylabel='原始峰值完整30年平均MDD',title='两套名义路径 × 两套CPI的四角重估',ylim=(0,max(corner_table.mean_mdd)*1.25))
    for i,row in enumerate(corner_table.itertuples()):ax[0].text(i,row.mean_mdd+.007,f'{row.breach_count}/120 超限',ha='center')
    component = [float(main.terminal_log_CPI_component.mean()/30),float(main.terminal_log_nominal_component.mean()/30)]
    total=float((main.SS_log_growth-main.BB_log_growth).mean())
    assert math.isclose(sum(component),total,abs_tol=1e-12)
    ax[1].bar(range(2),component,color=['#b87a16','#236b9b']);ax[1].axhline(0,color='#444444',lw=1)
    for i,value in enumerate(component):
        inside=abs(value)>.003
        ax[1].text(i,value/2 if inside else value-.0005,f'{value*100:+.3f} 个百分点／年',
            ha='center',va='center' if inside else 'top',color='white' if inside else '#222222')
    ax[1].axhline(total,color='#555555',ls='--',label=f'合计 {total*100:+.3f} 个百分点／年')
    ax[1].set(xticks=range(2),xticklabels=['CPI重估项','名义路径反馈项'],ylabel='平均年度对数增长变化',title='终值对数恒等式；含费用和动作反馈')
    ax[1].legend(loc='upper left')
    for a in ax:a.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('实际年通胀率逐年加2个百分点；训练支持不变；交叉重估不是可执行新策略',fontsize=16,y=1.03)
    save(fig,'03_inflation_four_corners',corner_table)
    pd.DataFrame([dict(component=k,mean_annual_log_growth_delta=v) for k,v in
        [('CPI_revaluation',component[0]),('nominal_path_feedback',component[1]),('SS_minus_BB_total',total)]]).to_csv(OUT/'inflation_log_components_source.csv',index=False)
    main.to_csv(OUT/'inflation_all120_path_source.csv',index=False)
    shifted = direction[(direction.cohort=='outer_OOF') & (direction.setting_id=='actual_inflation_plus_02')]
    first_crossings = shifted[shifted.year==shifted.first_crossing]
    assert len(first_crossings)==21 and first_crossings.prior_pending.all()
    assert set(first_crossings.scenario_id)==set(main.loc[main.SS_breach,'scenario_id'])
    assert set(main.loc[main.BS_breach,'scenario_id'])==set(main.loc[main.SS_breach,'scenario_id'])
    first_crossings.to_csv(OUT/'inflation_first_crossings_source.csv',index=False)
    writej(OUT/'inflation_mechanism_interpretation.json',dict(
        original_n=120,BS_SS_same_event_ids=sorted(int(v) for v in main.loc[main.SS_breach,'scenario_id']),
        first_crossing_n=len(first_crossings),prior_pending_n=int(first_crossings.prior_pending.sum()),
        negative_real_bill_n=int(first_crossings.actual_bill_negative_real.sum()),
        directional_miss_n=int(first_crossings.directional_miss.sum()),
        positive_nominal_asset_gain_n=int((first_crossings.nominal_asset_gain_at_next_CPI>0).sum()),
        mean_annual_log_CPI_component=component[0],mean_annual_log_nominal_path_component=component[1],
        mean_thirty_year_CPI_ratio=float(np.exp(-main.terminal_log_CPI_component).mean()),
        interpretation='Saved-path cross valuation and accounting under a constructed inflation stress, not unique real-world causal attribution. First-crossing-only quantities condition on21 events.'))

    initial = initial[(initial.setting_id=='base') & (initial.cohort=='outer_OOF')].sort_values('job_id')
    assert len(initial)==5
    fig, ax=plt.subplots(figsize=(12,5.5));x=np.arange(5)
    ax.bar(x-.17,initial.calendar_year1_slack,width=.34,color='#236b9b',label='训练第1年支持')
    ax.bar(x+.17,initial.pooled_slack,width=.34,color='#b87a16',label='全部30年训练支持')
    ax.axhline(0,color='#444444',lw=1);ax.set(xticks=x,xticklabels=initial.job_id,
        ylabel='最低下一年实际财富 − 初始风险预算（初始100）',title='同一初始目标及费用：更宽联合支持收紧最低财富条件')
    ax.legend();save(fig,'04_support_initial_slack',initial)

    base=summaries[summaries.policy=='main'].iloc[0]
    fee_summary=summaries[summaries.kind=='review_cost'].copy().sort_values('policy')
    positive_cost=costs[costs.fee_fraction==.01]
    capacity=float(positive_cost.review_fee_nominal.mean()/.01)
    fee_thresholds=[]
    for name in ('same_target_B','same_target_annual','exact_thirds_annual'):
        ref=summaries[summaries.policy==name].iloc[0]
        fee_thresholds.append(dict(reference=name,mean_nominal_fee_equalizing_q=float((ref.mean_nominal_fees-base.mean_nominal_fees)/capacity),
            meaning='Unknown one-time proportional service debit; average cumulative nominal fee accounting only, not risk-protection value or service-price estimate.'))
    writej(OUT/'fee_accounting_thresholds.json',fee_thresholds)
    exposure_stats=dict(guard_accounts=109,no_guard_accounts=11,
        higher_terminal_equity_n=int((funded.terminal_equity-funded.guard_equity>1e-12).sum()),
        mean_guard_equity=float(funded.guard_equity.mean()),mean_terminal_equity=float(funded.terminal_equity.mean()),
        post_guard_periods=2457,original_account_periods=3600)
    writej(OUT/'exposure_interpretation.json',exposure_stats)
    metrics.to_csv(OUT/'all12_reference_metrics.csv',index=False)
    fee_summary.to_csv(OUT/'all5_review_cost_summary.csv',index=False)
    def table(frame,percent=()):
        q=frame.copy()
        for key in percent:q[key]=q[key].map(lambda v:f'{v:.2%}')
        return '<div class="table">'+q.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}',na_rep='NULL')+'</div>'
    def img(name):
        data=base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()
        rel=OUT.name+'/'+name
        extra={'01_review_exposure':('phase_exposure_source.csv','阶段暴露CSV'),
               '03_inflation_four_corners':('inflation_log_components_source.csv','对数分解CSV')}.get(name)
        more='' if extra is None else f' · <a href="{OUT.name}/{extra[0]}">{extra[1]}</a>'
        return f'<figure><a href="{rel}.png"><img alt="{html.escape(name)}" src="data:image/png;base64,{data}"></a><figcaption>可点击查看完整图；<a href="{rel}.svg">SVG</a> · <a href="{rel}_data.csv">图源CSV</a>{more}</figcaption></figure>'
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最后一轮技术诊断 · 开发证据</title><style>body{font:17px/1.7 system-ui,sans-serif;color:#18232e;max-width:1160px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:23px;margin-top:36px}p,li{overflow-wrap:anywhere}.note{padding:16px;border-left:4px solid #236b9b;background:#eef4f7}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:14px;color:#485460}.table{max-width:100%;overflow-x:auto}table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dce2e6;white-space:nowrap;text-align:right}a{color:#236b9b}</style><body><h1>最后一轮技术诊断 · 2026-10-19</h1><p>Team87 · Lebmar。用于检查模型机制、费用与假设；参赛论文继续暂缓。</p><p class="note">主模型、参数与已保存最终评估均冻结。本页新增数字来自原1–120开发路径，五折各用自己的目标和内层选择；没有新盲测，没有利用最终60条路径重新选参数。全120重拟合结果在原始诊断中另列，本页图表仅使用外折OOF。</p>'''
    body+='<h2>转介后仍持续持有风险</h2><p>109个账户转介、11个未转介；转介当期108个账户执行保护交易、1个保持持仓。2457／3600个路径年位于转介之后。109个转介账户中，98个到期股票占比高于转介决策之后，配对平均股票占比约10.61%→32.50%。阶段平均权重使用不同路径年分母，不能将两个阶段均值直接解释为逐账户因果变化。未模拟人工接手、重新准入或人工投资收益。</p>'+img(figures[0])
    body+='<h2>简单降低暴露也能减少风险</h2><p>S1–S8分别固定a＝0.25／0.5／0.75／1，与每折原目标组合配比，其余为额外国库券；每个a保留年度再平衡与初始配置后持有两种规则。没有挑选胜者，也没有认证这些参照可以准入。S3的0／120超限、较低回撤和较低增长说明：现有结果不足以把风险改善全部归因于复杂控制。初始配置后持有同样可能产生显著回撤。</p>'+img(figures[1])
    body+=table(metrics[['code','policy','n','breach_n','median_cagr','mean_log_growth','mean_mdd','cvar90_mdd','mean_nominal_fees']],('median_cagr','mean_log_growth','mean_mdd','cvar90_mdd'))
    body+='<p>M为开发主策略，B为同目标年调且超限后持有，A为同目标持续年调，T为等权持续年调；S7与A为同一政策。mean_log_growth是平均年度对数增长，与CAGR中位数口径不同。名义费用按初始名义财富100列示，为各期未折现名义金额之和。</p>'
    body+='<h2>审核服务费用尚无实测依据</h2><p>在触发年末按资产比例扣除一次假设审核费，仅对原期限内还有后续年份的转介账户生效；继续原持有，不假设人工改善收益。保留扣费前历史峰值，并把同一年度边界的即时扣费损失纳入这组费用情景的回撤。因此该情景含额外费用事件，与原模型纯年度端点评分的口径有明确区别。</p>'
    body+=table(fee_summary[['policy','breach_count','median_cagr','mean_mdd','mean_nominal_fees']],('median_cagr','mean_mdd'))
    body+='<p>一次0.25%资产比例的假设审核费，已令平均累计名义总费用接近同目标年度再平衡参照；这只检查原有手续费优势对未计服务费的敏感程度，不能当作真实报价，也没有为风险保护价值定价。终值相对参照的费用余量与上述费用账本是不同概念。</p>'
    body+='<h2>通胀与支持集：结论依赖的条件</h2>'+img(figures[2])+table(corner_table,('mean_mdd','median_cagr','mean_log_growth'))
    body+='<p>BB、SS是已保存的原／压力路径；BS、SB是交叉计价诊断。每条都从原始100重新计算峰值与全期限MDD。终值对数增长可以精确分解，MDD分解随顺序变化，不能归为唯一因果份额。所有配对与不利路径均保留。</p>'
    body+='<p>保持原名义路径而仅换用更高CPI，已产生与完整压力情景相同的21个超限ID。完整压力情景的21次首次超限全部发生于转介后的持有期间，当期国库券实际收益均为负；其中仅6次跌破方向性支持下界，另外15次没有。因此不能把全部损失归于新的尾部逃逸。14次首次超限当年的组合名义资产收益仍为正，购买力损失不可由名义收益正负替代判断。以上是构造压力下的配对核算，不能据此识别真实市场中的唯一因果机制。</p>'
    body+=img(figures[3])
    body+='<p>第1年联合支持是全部30年联合支持的子集，因此同一持仓在更宽支持下的最低财富不会提高。这里说明固定初始目标为何被更严条件拒绝，不能推论全部可行配置均无法服务。国库券名义稳定也不等于购买力不下降；方向性下界失守比例不能当作联合概率覆盖率。</p>'
    body+='<h2>参数与答题口径</h2><p>γ＝0.075是Q1训练经验超限比例筛选门槛；α＝0.10是服务经验频率约定；ρ＝0.5是单步风险空间释放比例。三者均不构成已识别的客户心理参数。目标国库券下限是流动性代理，缺少现金负债和赎回日程时，不能宣称持续现金准备充足。</p><p>原题允许A、D转人工；各项答题对象已有技术证据。长期无人接手与此前最终数据暴露，仍是实质性运行／验证局限。逐条对照见 <a href="../docs/technical_evidence_closure_2026_10_19.json">18项技术证据对照</a>。</p><p>图源均为题给合成情景的分析转换，沿用组织者与Jordà–Schularick–Taylor Macrohistory R6所选美国序列的归属及CC BY-NC-SA4.0使用说明；不是实际历史或未来预测。AI参与代码与核验，数值独立审计不等于外部人工同行评审。</p></body></html>'
    PAGE.write_text(body)
    for name,digest in sources.items(): assert sha(ROOT/name)==digest, name
    writej(OUT/'build_manifest.json',dict(status='built_pending_visual_review',built_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,builder_sha256=sha(Path(__file__)),page_sha256=sha(PAGE),
        figures=figures,final_market_reads=0,new_model_fits=0,new_model_selections=0,
        assets_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name!='build_manifest.json'}))
    print('Built',PAGE.name,'with',len(figures),'standalone figures; visual review pending.')


if __name__=='__main__': build()
