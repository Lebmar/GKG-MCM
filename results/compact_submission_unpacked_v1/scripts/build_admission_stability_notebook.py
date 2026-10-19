"""Build a focused research companion from completed, audited saved artifacts.

No market CSV, policy fit, parameter selection or final evaluation is executed.
"""
from pathlib import Path
from textwrap import dedent
import json
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/model_review'


def build():
    stable=BASE/'refit_stability_v1'
    audit=json.loads((stable/'independent_audit.json').read_text())
    if audit['status']!='passed':raise RuntimeError('Complete independent stability audit first')
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(dedent(s).strip()))
    def code(s):cells.append(nb.v4.new_code_cell(dedent(s).strip()))
    md(r'''
    # SafeInvest：新增准入风险、重拟合稳定性与预测假设

    **研究记录，2026-10-18；原客户C，H=30，L=40%。本轮没有重新确定客户偏好，也没有写提交论文。**

    本笔记分别检查三件事：新准入的24条市场路径付出了什么风险代价；预先固定的五组划分下，完整学习算法的准入与参数是否稳定；加入简单上一年状态，对原日历预测能改善多少。所有模型规则、下限与候选族保持冻结，不按外折结果选一个“最好种子”。

    **数据身份：** 只使用原开发情景1–120，数据已反复参与设计。125个训练任务是25个外层拟合各4次动态内层训练加最终重拟合；每次完整重选Q1菜单/γ及Q2预算/价格。五次重复的600条评估记录仍来自同一120条原情景，不能当600独立观测，也不能估计真实客户人口覆盖。旧V1最终集曾被查看，本轮未重新解析121–180。

    本轮保持30/10/0目标票据规范约定、原172候选、原风险门槛和九个交易控制器。更多自动准入不会被直接称为更安全；Review收益留空，参照不是服务兜底。
    ''')
    code('''
    from pathlib import Path
    import base64, hashlib, html, json
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML
    ROOT=Path.cwd().resolve()
    if not (ROOT/'docs/refit_stability_protocol.json').exists(): ROOT=ROOT.parent
    BASE=ROOT/'results/model_review'
    def readj(p): return json.loads(Path(p).read_text())
    def show(df,formats=None):
        default={c:'{:,.0f}' for c in df.select_dtypes(include='number') if len(df[c].dropna()) and np.all(np.abs(df[c].dropna()-np.rint(df[c].dropna()))<1e-12)}
        default.update(formats or {})
        display(df.style.format(default,na_rep='—').hide(axis='index'))
    def figure(path,alt):
        encoded=base64.b64encode(Path(path).read_bytes()).decode('ascii')
        display(HTML('<img alt="'+html.escape(alt,quote=True)+'" src="data:image/png;base64,'+encoded+'" style="max-width:100%;height:auto">'))
    S=BASE/'refit_stability_v1'; D=BASE/'incremental_admission_risk_v1'; F=BASE/'forecast_assumptions_v2'
    assert readj(S/'manifest.json')['training_tasks']==125
    assert readj(S/'independent_audit.json')['status']=='passed'
    assert readj(F/'independent_audit.json')['status']=='passed'
    assert readj(D/'manifest.json')['sources_and_artifacts_at_start']==readj(D/'manifest.json')['sources_and_artifacts_at_end']
    assert not any(readj(p/'manifest.json')['final_scenarios_loaded'] for p in [S,D,F])
    print('数据范围：原1–120；只读已保存结果，无拟合/策略更新/最终评价。')
    ''')
    md(r'''
    ## 1. 新增24条路径：费用节省与更深回撤同时存在

    本组来自旧第二外折的一次新准入，不是24位不同客户。比较保留全部24个原情景，使用同一训练目标40/25/35（股票/债券/票据）；它与全120条描述拟合的45/5/50目录不同。B为每年恢复该目标、首次超限后持仓等待的假想参照。等权参照另列，不把不同目标的差异全部归为执行方式。

    相对同目标B，主策略在24/24条路径的最大回撤都更大；18/24条年化log增长更高，22/24条费用更低。超限是3对1，尾部严重度也需同时呈现。24条经验CVaR90使用分数尾质量，不能当置信上界或总体尾风险。
    ''')
    code('''
    summary=pd.read_csv(D/'incremental_summary.csv')
    table=summary[['policy','n','breach_n','mean_mdd','p90_mdd','cvar90_mdd','mean_log_growth','mean_nominal_fees','mean_actual_trade_count']].copy()
    show(table, {c:'{:.2%}' for c in ['mean_mdd','p90_mdd','cvar90_mdd','mean_log_growth']})
    figure(ROOT/'research/figures/incremental_admission_risk/01_new_admission_all_24_pairs.png',
           '完整24情景的同ID回撤与增长配对；每条连接同一市场情景。')
    ''')
    md(r'''
    ## 2. 相同当期权重，仍然可能有不同回撤

    所有主策略独有超限事件为ID35、112，分别在第19、26年跨线；共同超限ID38也完整保留在见证库存。35、112跨线当年，两策略均恢复相同目标，年末权重一致，但历史财富和峰值不同。当前实现保留这些历史状态，核查没有发现峰值被遗漏或重置。

    在同一年，当前回撤$d_t=1-R_t/P_t$。下面的恒等式只作状态记账：

    $$\log\frac{1-d_t^{new}}{1-d_t^{B}}=\log\frac{R_t^{new}}{R_t^B}-\log\frac{P_t^{new}}{P_t^B}.$$

    它解释为什么不能只凭跨线时股票比例推断超限原因，也不能把“更多现金”当作已经验证的修复方案。过去交易、暴露与峰值变化的独立因果贡献尚未隔离。已超限后的等待也不撤销事件，且损失可能继续深化；截至第30年尚未恢复触发峰值属于右删失，不能说永不恢复。
    ''')
    code('''
    states=pd.read_csv(D/'incremental_annual_states.csv')
    for key,g in states.groupby(['scenario_id','policy']):
        g=g.sort_values('year'); w=np.r_[100.,g.real_wealth_end]
        peak=np.maximum.accumulate(w)
        assert np.allclose(peak[1:],g.real_peak_end,atol=1e-10)
        assert np.allclose(1-w[1:]/peak[1:],g.current_drawdown_end,atol=1e-12)
    rows=[]
    for sid,year in [(35,19),(112,26)]:
        group=states[(states.scenario_id==sid)&(states.year==year)].set_index('policy')
        a=group.loc['joint_tie_set_refit_learner']; b=group.loc['annual_then_pending']
        lhs=np.log((1-a.current_drawdown_end)/(1-b.current_drawdown_end))
        rhs=np.log(a.real_wealth_end/b.real_wealth_end)-np.log(a.real_peak_end/b.real_peak_end)
        assert abs(lhs-rhs)<1e-12
        rows.append(dict(scenario_id=sid,year=year,primary_wealth=a.real_wealth_end,reference_wealth=b.real_wealth_end,
                         primary_peak=a.real_peak_end,reference_peak=b.real_peak_end,
                         primary_current_DD=a.current_drawdown_end,reference_current_DD=b.current_drawdown_end,
                         same_end_equity_bill=np.allclose([a.end_year_equity,a.end_year_bill],[b.end_year_equity,b.end_year_bill])))
    show(pd.DataFrame(rows),{'primary_current_DD':'{:.2%}','reference_current_DD':'{:.2%}'})
    figure(ROOT/'research/figures/incremental_admission_risk/02_primary_only_full_history.png',
           '预定全部两条主策略独有超限路径完整0–30年财富、峰值、当前回撤与累计MDD。')
    ''')
    md(r'''
    ## 3. 五组完整重拟合：同规则下的划分敏感性

    外层每次96条训练、24条评估；动态内层每次72条。所有拟合样本量相同，因此本批跨拟合变化不能解释为n变化造成的取整。训练路径、目标、预测扇、内层划分及状态都可能改变，不能把变化全归于某一个因素。

    新的五组划分在读结果前已固定。保留所有25个拟合、所有失败原因和原情景分母；不挑一个最好的划分当最终成绩。外层风险频率、训练安全余量与参数频率是不同统计对象。参数不同不必然意味着同一情景动作不同；这里的每情景准入次数也不是客户人口覆盖率。

    准入是训练集S与客户C的函数A(S,C)，在读取评估收益前确定；同一拟合的24条留出路径统一准入或复核。“某原情景五次准入几次”只记录五次留出该ID时相应训练拟合的决定，不能解释为控制器提前识别并挑选了较安全的未来路径。

    **结果：** 主方案的五组自动路径数为0、48、120、72、48，相应事件数为NULL、8、12、7、3。第二组8/48=16.67%，不支持稳定满足10%的结论；第一组全部执行复核，风险不是零。25个最终拟合的C均为Balanced，变化发生在具体配置、γ与Q2准入；不能把失败解释为C被重新判成了别的客户类型。

    主方案12/25个拟合准入，7个没有内层可准入的β集合，6个顶层集合在重拟合后均不可行；旧规则7/25准入。两方法共同自动子群的增长、回撤与费用差异均为零，增加的是准入集合。主方案12个准入拟合中，8个选中控制器的训练事件数正好等于上限，4个只剩1次事件的整数余量。这是样本内可行余量，不是未来风险置信余量。
    ''')
    code('''
    repeat=pd.read_csv(S/'summary.csv.gz')
    show(repeat[repeat.policy.isin(['joint_canonical_learner','joint_tie_set_refit_learner'])][
        ['repeat_id','policy','original_n','balanced_case_n','q1_review_n','not_balanced_case_n','case_auto_n','case_execReview_n','breach_count','conditional_breach_probability','mean_log_growth','p90_mdd']],
         {'conditional_breach_probability':'{:.2%}','mean_log_growth':'{:.2%}','p90_mdd':'{:.2%}'})
    T=BASE/'refit_stability_summary_v1'
    assert readj(T/'manifest.json')['status']=='complete'
    fits=pd.read_csv(T/'fit_contexts.csv').query("policy=='joint_tie_set_refit_learner'")
    show(fits[['repeat_id','outer_fold','static_gamma','execution_status','execution_reason',
               'execution_target_weights','selected_beta','selected_annual_price_label',
               'selected_training_breach_cap','selected_training_breach_count','selected_training_slack']])
    assert fits.q1_effective_assignment.eq('Balanced').all()
    assert fits.loc[fits.execution_status.eq('automated'),'selected_training_slack'].value_counts().to_dict()=={0.:8,1.:4}
    figure(ROOT/'research/figures/refit_stability/01_repeat_admission_and_conditional_risk.png',
           '五组固定重复的原120路径准入及条件k/n，零准入保留NULL；不是时间趋势。')
    figure(ROOT/'research/figures/refit_stability/02_scenario_admission_and_primary_fit_status.png',
           '同一原情景五次准入频次及全部25个拟合状态，重复记录不是新增独立市场样本。')
    ''')
    md(r'''
    ## 4. 预测假设：固定上一年状态比较器，不更新交易策略

    使用原五折，每个日历年的均值仅用96条训练路径估计；中心化后，尺度由训练96×29期残差按通道池化计算。固定λ=1岭回归，只输入t−1年的三个实际资产log收益及log通胀，分别预测下一年和到原第30年剩余期的平均年log量。没有选λ、追加特征或将比较器装入交易控制器。该λ是正则化参数，和旧风险偏好系数不是同一对象。

    每条原路径先平均29期误差，再按路径聚合；三实际资产的标准化平方误差作为主指标，通胀单列。下一年误差约减少0.939%，剩余期只约减少0.109%，说明简单状态有小幅关联。它没有给出条件交易Q、联合尾分布或完整超限概率的替代模型。

    原日历均值在等大五折下，合并OOF平均偏差几乎为零是代数抵消，不能当作预测校准证据。经验10–90%分位区间覆盖约78.1–78.8%，96条训练路径的样本分位区间本来不保证精确80%，不能直接称失校准。范围外率也不是多维条件支持失效的证明。
    ''')
    code('''
    forecast=pd.read_csv(F/'summary.csv')
    show(forecast[['target','channel','original_path_n','state_to_calendar_mse_ratio',
                   'interval_10_90_coverage','outside_training_target_range_fraction','path_fraction_state_lower_loss']],
         {'state_to_calendar_mse_ratio':'{:.6f}','interval_10_90_coverage':'{:.2%}',
          'outside_training_target_range_fraction':'{:.2%}','path_fraction_state_lower_loss':'{:.2%}'})
    figure(ROOT/'research/figures/forecast_assumptions_v2/01_fixed_lag_state_prediction_losses.png',
           '固定λ1上一年状态比较器的五折与各通道误差变化，改善幅度不作显著性或Q/尾概率解释。')
    ''')
    md(r'''
    ## 5. 当前判断与后续边界

    更多准入不能替代增长—损失—费用的联合判断。新增24条路径的回撤代价已经确认；五组完整重拟合又显示准入0–120的大幅波动，且每个成功拟合只有0或1次训练事件的余量。当前规则继续保留为研究比较器，不能因原一组结果或某个最好划分而认定准入已经稳定。日历扇的简单状态遗漏检验只有小幅均值预测改善，未验证条件尾概率；暂不据此替换交易预测模型。

    所有证据来自已重复使用的开发情景。独立算法重算、闭式边界测试、来源哈希及可重复执行证明计算与声明相符，不能增加独立市场样本，也不能代替参赛者人工审查或最终服务政策审定。下一步的模型改变需要先写清要修复的机制及完整检验流程，不能按本轮最差或最好路径临时改门槛。
    ''')
    notebook=nb.v4.new_notebook(cells=cells)
    notebook.metadata.update(kernelspec=dict(display_name='SafeInvest review',language='python',name='safeinvest-review'),
        language_info=dict(name='python',version='3.12'),safeinvest_scope='Saved development1–120 evidence only; no fitting or final evaluation in this companion.')
    out=ROOT/'notebooks/admission_stability_forecasts.ipynb';nb.write(notebook,out);print(out)


if __name__=='__main__':build()
