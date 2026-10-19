"""Read-only companion of the prospectively fixed benefit-cost model study.

Build only after complete study, bound independent audit and complete summary.
No market parser, simulator, fitting or architecture selection is imported.
Figure review is an explicit later step; it is never inferred from execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path('results/model_review/benefit_cost_execution_v1')
SUMMARY = Path('results/model_review/benefit_cost_summary_v1')
FIGURES = Path('research/figures/benefit_cost')
NOTEBOOK = Path('notebooks/benefit_cost_model.ipynb')
HTML = Path('notebooks/benefit_cost_model.html')
PROTOCOL = Path('docs/benefit_cost_execution_protocol.json')
METHODS = ('benefit_cost_one_year', 'benefit_cost_remaining_H')
PRIMARY = METHODS[0]
METHOD_LABELS = {METHODS[0]: 'One year (primary)', METHODS[1]: 'Remaining H (sensitivity)'}
CHINESE_LABELS = {METHODS[0]: '一年净收益主模型', METHODS[1]: '原H剩余期敏感性'}
SOURCE_NOTE = 'Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120; CC BY-NC-SA 4.0; Team 87.'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def check_mapping(root, values):
    for rel, digest in values.items():
        path = Path(root) / rel
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError('Current source/input/artifact differs: ' + str(rel))


def require_complete(root=ROOT):
    """All completion/binding gates precede reading any numeric result table.

    Input hashing streams file bytes (including the original dataset file),
    but does not parse market rows or construct a return/forecast array.
    """
    root = Path(root)
    gates = ((STUDY/'manifest.json', 'complete'),
             (STUDY/'independent_audit.json', 'passed'),
             (SUMMARY/'manifest.json', 'complete'))
    values = []
    for rel, status in gates:
        if not (root/rel).is_file():
            raise RuntimeError('Waiting for complete audited study and summary: ' + str(rel))
        value = readj(root/rel)
        if value.get('status') != status:
            raise RuntimeError('Incomplete or unaudited artifact: ' + str(rel))
        values.append(value)
    study, audit, summary = values
    study_sha = sha(root/STUDY/'manifest.json')
    if audit.get('parent_manifest_sha256') != study_sha:
        raise RuntimeError('Independent audit is not bound to the current study manifest')
    if summary.get('parent_manifest_sha256') != study_sha:
        raise RuntimeError('Summary is not bound to the current study manifest')
    if study['sources'] != study['sources_at_end'] or study['input_sha256'] != study['input_sha256_at_end']:
        raise RuntimeError('Study start/end source or input snapshots differ')
    if summary['artifacts_at_start'] != summary['artifacts_at_end']:
        raise RuntimeError('Summary read-only parent snapshots differ')
    check_mapping(root, study['sources'])
    check_mapping(root, study['input_sha256'])
    check_mapping(root, summary['source_sha256'])
    check_mapping(root, summary['artifacts_at_start'])
    check_mapping(root/STUDY, study['outputs_sha256'])
    check_mapping(root/SUMMARY, summary['outputs_sha256'])
    if study['protocol_sha256'] != sha(root/PROTOCOL):
        raise RuntimeError('Prospective protocol changed')
    protocol = readj(root/PROTOCOL)
    if (protocol['primary_method'] != PRIMARY or study['primary_method'] != PRIMARY
            or protocol['methods'] != {METHODS[0]:'one_year', METHODS[1]:'remaining_to_original_H'}
            or protocol['data']['scenario_ids'] != list(range(1,121))
            or protocol['data']['allow_final'] is not False
            or study['scenario_ids_loaded'] != list(range(1,121))
            or study['final_scenarios_loaded'] is not False
            or protocol['execution_ready'] is not True):
        raise RuntimeError('Prospective primary, original paths or final-data boundary differs')
    snapshot = {str(rel):sha(root/rel) for rel, _ in gates}
    snapshot[str(PROTOCOL)] = sha(root/PROTOCOL)
    for name in summary['outputs_sha256']:
        snapshot[str(SUMMARY/name)] = sha(root/SUMMARY/name)
    for name in ('training_summary.csv.gz', 'inner_gate_summaries.csv.gz'):
        snapshot[str(STUDY/name)] = sha(root/STUDY/name)
    return dict(study=study, audit=audit, summary=summary, protocol=protocol,
                input_artifact_sha256=dict(sorted(snapshot.items())))


def load_tables(root=ROOT):
    import pandas as pd
    root = Path(root)
    repeat = pd.read_csv(root/SUMMARY/'repeat_summary.csv.gz')
    paired = pd.read_csv(root/SUMMARY/'paired_B_summary.csv.gz')
    try:
        decisions = pd.read_csv(root/SUMMARY/'decision_summary.csv.gz')
    except pd.errors.EmptyDataError:
        decisions = pd.DataFrame(columns=['policy','category','scope','optional_decisions',
            'optional_restores','growth_nonpositive_holds','positive_growth_risk_veto_holds',
            'invalid_optional_restores','minimum_estimated_net_restore'])
    full = pd.DataFrame(readj(root/SUMMARY/'full_development_admission.json'))
    training = pd.read_csv(root/STUDY/'training_summary.csv.gz')
    inner = pd.read_csv(root/STUDY/'inner_gate_summaries.csv.gz')
    selected = repeat.loc[repeat.policy.isin(METHODS)]
    if len(selected) != 12 or selected[['repeat_id','policy']].duplicated().any():
        raise RuntimeError('Keep two fixed methods across all five repeats and full-development fit')
    if not selected.groupby('policy').repeat_id.apply(lambda x:set(x)==set(range(6))).all():
        raise RuntimeError('Missing one of the original six reporting scopes')
    if (not repeat.original_n.eq(120).all()
            or not (repeat.assigned_n+repeat.financial_null_n).eq(repeat.original_n).all()
            or not repeat.loc[repeat.repeat_id.eq(0),'scope'].eq('in_sample_full120').all()
            or not repeat.loc[repeat.repeat_id.ne(0),'scope'].eq('outer_oof').all()):
        raise RuntimeError('Population, NULL partition or evaluation scope changed')
    financial = ['breach_count','conditional_breach_probability','mean_log_growth','mean_mdd','mean_fees','mean_trade_fraction','p90_mdd']
    if not repeat.loc[repeat.assigned_n.eq(0),financial].isna().all().all():
        raise RuntimeError('Zero coverage must retain NULL financial/risk outcomes')
    if len(full)!=2 or set(full.policy)!=set(METHODS) or not full.repeat_id.eq(0).all():
        raise RuntimeError('One full-development admission decision per fixed method required')
    if len(paired)!=12 or paired[['repeat_id','policy']].duplicated().any():
        raise RuntimeError('Keep all twelve same-cohort paired rows, including NULLs')
    pair_financial = [c for c in paired if c.startswith('mean_delta_')] + ['model_breaches','B_breaches']
    if not paired.loc[paired.common_auto_n.eq(0),pair_financial].isna().all().all():
        raise RuntimeError('No common automatic paths means NULL paired financial outcomes')
    if len(decisions) and (decisions.invalid_optional_restores.ne(0).any()
            or decisions.optional_restores.gt(decisions.optional_decisions).any()):
        raise RuntimeError('Saved optional trade contract violation')
    return dict(repeat=repeat, paired=paired, decisions=decisions, full=full,
                training=training, inner=inner)


def plot_style():
    import matplotlib
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
        'axes.labelsize':13,'axes.titlesize':15,'xtick.labelsize':12,'ytick.labelsize':12,
        'legend.fontsize':12,'svg.fonttype':'none','svg.hashsalt':'benefit-cost-model-v1',
        'figure.facecolor':'white','axes.facecolor':'white','text.color':'#27313a',
        'axes.labelcolor':'#27313a','axes.edgecolor':'#a6afb5','savefig.facecolor':'white'})


def plot_repeats(repeat):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    plot_style()
    fig, axes = plt.subplots(1,2,figsize=(17,8.7),gridspec_kw={'width_ratios':[1.05,1]})
    fig.subplots_adjust(left=.07,right=.98,top=.75,bottom=.22,wspace=.23)
    fig.text(.07,.95,'Benefit-cost models: repeated admission and held-out events',fontsize=20,weight='bold')
    fig.text(.07,.90,'One year was named primary before execution; remaining H is a fixed sensitivity, never a fallback.',fontsize=13)
    fig.text(.07,.855,'Original 120 development paths reused in five overlapping fits; each fit decides before its held-out returns.',fontsize=13)
    colors = {METHODS[0]:'#205f8a',METHODS[1]:'#a87418'}
    for j,method in enumerate(METHODS):
        rows = repeat.loc[repeat.policy.eq(method)&repeat.repeat_id.between(1,5)].sort_values('repeat_id')
        if len(rows)!=5:raise ValueError('All five repeated evaluations required')
        xx = np.arange(1,6)+(j-.5)*.27
        axes[0].bar(xx,rows.assigned_n,width=.25,color=colors[method],label=METHOD_LABELS[method],
                    hatch=None if j==0 else '//',edgecolor='white')
        for x,row in zip(xx,rows.itertuples()):
            axes[0].text(x,int(row.assigned_n)+3,str(int(row.assigned_n)),ha='center',va='bottom',fontsize=13)
        for i,row in enumerate(rows.itertuples()):
            if int(row.assigned_n)==0:
                text='NULL\n0 auto';fill='#edf0f2'
            else:
                text=f'{int(row.breach_count)}/{int(row.assigned_n)}\n{row.conditional_breach_probability:.1%}'
                fill='#e4eef4' if row.conditional_breach_probability<=.1+1e-12 else '#f4e9d5'
            axes[1].add_patch(Rectangle((j,i),1,1,facecolor=fill,edgecolor='white',linewidth=3))
            axes[1].text(j+.5,i+.5,text,ha='center',va='center',fontsize=14)
    axes[0].set(title='(a) Automatic service / original 120',xlabel='Repeat partition (not time)',
                ylabel='Automatically executed original paths',xticks=range(1,6),ylim=(0,142),yticks=range(0,121,24))
    axes[0].spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.16),frameon=False,ncol=1)
    axes[1].set(title='(b) Conditional full-H events k/n',xlim=(0,2),ylim=(5,0),
        xticks=[.5,1.5],xticklabels=['One year\nprimary','Remaining H\nsensitivity'],
        yticks=np.arange(5)+.5,yticklabels=[f'Repeat {i}' for i in range(1,6)])
    axes[1].tick_params(length=0)
    for spine in axes[1].spines.values():spine.set_visible(False)
    fig.text(.07,.14,'Every optional restore needs positive estimated net benefit and a full-H buy-and-hold risk-direction check.',fontsize=12)
    fig.text(.07,.10,'Observed k/n and the 10% service reference do not certify future risk; zero admission is NULL risk, not zero.',fontsize=12)
    fig.text(.07,.055,SOURCE_NOTE,fontsize=12)
    return fig


def plot_full_admission(full, training, inner):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plot_style()
    fig, ax = plt.subplots(figsize=(17,8.7))
    fig.subplots_adjust(left=.04,right=.98,top=.72,bottom=.23)
    fig.text(.05,.95,'Full-development fit: economic eligibility and service admission',fontsize=20,weight='bold')
    fig.text(.05,.90,'This uses all 120 previously used development paths: in-sample fitting/evaluation, not a sixth validation repeat.',fontsize=13)
    fig.text(.05,.855,'The named one-year primary is retained whether it qualifies or goes to Review. Sensitivity does not replace it.',fontsize=13)
    headers=['Initial net log benefit','Complete training risk','Own inner automatic set U','Final service decision']
    for j,label in enumerate(headers):ax.text(j+.5,-.19,label,ha='center',va='bottom',fontsize=14,weight='bold')
    for i,method in enumerate(METHODS):
        row = full.loc[full.policy.eq(method)].iloc[0]
        ts = training.loc[training.job_id.eq(row.job_id)&training.policy.eq(method)]
        gate = inner.loc[inner.context.eq(row.context)&inner.policy.eq(method)]
        if len(ts)!=1 or len(gate)!=1:raise ValueError('Full fit must have one matching training summary and inner gate')
        ts, gate = ts.iloc[0], gate.iloc[0]
        def count(value):return 'NULL' if value is None or value!=value else str(int(value))
        net = row.initial_mean_net_log_gain
        net_text = 'NULL\nQ1/case unavailable' if net is None or net!=net else f'{net:+.6f} total log\n'+('positive' if net>1e-12 else 'not positive enough')
        cap = row.get('training_breach_cap')
        train_text = f'k={count(ts.breach_count)}; cap={count(cap)}\nn={int(ts.n_training)}\n'+('passed' if bool(row.get('training_eligible',False)) else 'not eligible')
        inner_text = f'k/U={count(gate.selected_breach_count)}/{int(gate.auto_inner_n)}\nB k={count(gate.same_U_B_breach_count)}; cap={count(gate.absolute_alpha_count_cap)}\n'+('passed' if bool(gate.eligible) else 'not eligible')
        status = str(row.execution_status)
        status_text = 'AUTOMATIC' if status=='automated' else 'REVIEW / unavailable'
        texts=[net_text,train_text,inner_text,status_text]
        ax.text(-.08,i+.5,METHOD_LABELS[method].replace(' (','\n('),ha='right',va='center',fontsize=13)
        for j,text in enumerate(texts):
            fill = '#e4eef4' if (j==3 and status=='automated') else '#edf0f2'
            ax.add_patch(Rectangle((j,i),1,1,facecolor=fill,edgecolor='white',linewidth=4))
            ax.text(j+.5,i+.5,text,ha='center',va='center',fontsize=13)
    ax.set(xlim=(-.80,4),ylim=(2,0));ax.axis('off')
    fig.text(.05,.16,'Inception requires its positive economic estimate, then pays its mandatory allocation fee; no initial all-bill risk veto.',fontsize=12)
    fig.text(.05,.115,'Final admission requires the complete-policy training cap AND own-U inner gate. Review financial outcomes stay NULL.',fontsize=12)
    fig.text(.05,.07,'No annual/TV/sensitivity fallback; original peak, CPI, horizon and absorbing pending hold remain intact.',fontsize=12)
    fig.text(.05,.025,SOURCE_NOTE,fontsize=12)
    return fig


def save_figures(root, tables, ready):
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/FIGURES;folder.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve()
    source={str(script.relative_to(ROOT)):sha(script)}
    builders=[('01_repeated_admission_and_events',plot_repeats(tables['repeat'])),
              ('02_full_development_admission',plot_full_admission(tables['full'],tables['training'],tables['inner']))]
    outputs={}
    for name,fig in builders:
        png,svg=folder/(name+'.png'),folder/(name+'.svg')
        fig.savefig(png,dpi=240)
        fig.savefig(svg,metadata={'Date':None})
        svg.write_text(svg.read_text().rstrip())  # finalize bytes BEFORE hashing
        plt.close(fig)
        for path in (png,svg):outputs[str(path.relative_to(root))]=sha(path)
    after=require_complete(root)
    source_end={rel:sha(root/rel) for rel in source}
    if source!=source_end:raise RuntimeError('Plotting source changed while drawing')
    if ready['input_artifact_sha256']!=after['input_artifact_sha256']:
        raise RuntimeError('Read-only numeric inputs changed while drawing')
    manifest=dict(status='complete',source_sha256=source,sources_at_end=source_end,
        input_artifact_sha256=ready['input_artifact_sha256'],
        input_artifact_sha256_at_end=after['input_artifact_sha256'],outputs_sha256=outputs,
        visual_review_confirmed=False,visual_scope='PNG inspection pending; no whole-HTML browser claim',
        source_note=SOURCE_NOTE,format={'png_dpi':240,'minimum_font_points':12,'svg_text':True,'svg_trailing_whitespace_stripped_before_hash':True})
    writej(folder/'manifest.json',manifest)
    return manifest


def confirm_figures(root=ROOT):
    root=Path(root);require_complete(root);path=root/FIGURES/'manifest.json';value=readj(path)
    check_mapping(root,value['source_sha256']);check_mapping(root,value['input_artifact_sha256'])
    check_mapping(root,value['outputs_sha256'])
    value['visual_review_confirmed']=True
    value['visual_scope']='Both saved PNGs actually inspected; HTML only static inspection, no browser preview claim'
    value['reviewed_utc']=datetime.now(timezone.utc).isoformat()
    writej(path,value)


def observed_summary(tables):
    lines=[]
    for method in METHODS:
        rows=tables['repeat'].loc[tables['repeat'].policy.eq(method)&tables['repeat'].repeat_id.between(1,5)].sort_values('repeat_id')
        coverage='、'.join(str(int(x)) for x in rows.assigned_n)
        events='；'.join('NULL（0自动）' if int(r.assigned_n)==0 else f'{int(r.breach_count)}/{int(r.assigned_n)}' for r in rows.itertuples())
        final=tables['full'].loc[tables['full'].policy.eq(method)].iloc[0]
        lines.append(f'- **{CHINESE_LABELS[method]}**：五次重复自动路径数为{coverage}；条件事件为{events}。全开发集拟合的最终状态为`{final.execution_status}`，原因`{final.execution_reason}`。')
    return '\n'.join(lines)


def make_notebook(tables):
    cells=[]
    def md(value):cells.append(nbf.v4.new_markdown_cell(dedent(value).strip()))
    def code(value):cells.append(nbf.v4.new_code_cell(dedent(value).strip()))
    md(r'''
    # SafeInvest：正净收益交易与完整策略准入

    这份模型研究笔记记录事前固定的一年主模型与剩余期限敏感性。没有按结果更换主模型，不写提交论文，也不以年度／TV交易规则兜底。
    ''')
    md(observed_summary(tables))
    md(r'''
    ## 1. 先核验来源，再读取保存结果

    独立审计必须绑定当前研究manifest；摘要必须绑定同一manifest。运行时重新核对当前所有冻结源码、输入及输出哈希。输入哈希流式读取源文件字节，**不解析市场行、不构造收益数组、不模拟、不拟合**。本阶段仍只使用情景1—120；旧V1曾接触121—180，不恢复“全新盲测”声明。
    ''')
    code(r'''
    from pathlib import Path
    import sys
    import json
    import numpy as np
    import pandas as pd
    from IPython.display import display
    ROOT = next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents]
                if (p/'docs/benefit_cost_execution_protocol.json').is_file())
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_benefit_cost_notebook as companion
    METHODS=companion.METHODS
    METHOD_LABELS=companion.METHOD_LABELS
    SOURCE_NOTE=companion.SOURCE_NOTE
    plot_style=companion.plot_style
    ready = companion.require_complete(ROOT)
    tables = companion.load_tables(ROOT)
    figure_manifest = companion.readj(ROOT/companion.FIGURES/'manifest.json')
    assert figure_manifest['status']=='complete' and figure_manifest['visual_review_confirmed'] is True
    companion.check_mapping(ROOT,figure_manifest['source_sha256'])
    companion.check_mapping(ROOT,figure_manifest['input_artifact_sha256'])
    companion.check_mapping(ROOT,figure_manifest['outputs_sha256'])
    def show(frame,formats=None):
        display(frame.style.format(formats or {},na_rep='—').hide(axis='index'))
    print('完整研究、父manifest绑定、独立审计、源码/输入/输出哈希与图像检查记录通过。')
    ''')
    md(r'''
    ## 2. 毛收益、当前费用与净收益严格分开

    在原日历年决策前，实际余额为$x$、财富$V=\sum_i x_i$。精确自融资执行器报价得到$\mathrm{post},C,m$。令$u=x/V$、$v=\mathrm{post}/(V-C)$，当前log费用为
    $$\kappa=-\log(1-C/V).$$
    对训练情景$s$的联合资产总收益因子$Q_s$，
    $$A_s=\log(v^\top Q_s)-\log(u^\top Q_s),\qquad \Delta_s=A_s-\kappa.$$
    一年主模型只取下一年的$Q_s$；敏感性取当前年至**原始H**的逐资产累计总收益因子。两分支均假设当前动作之后持有。它们保留同情景资产联合关系；不独立打乱资产，也不把一年行视为额外独立市场路径。

    每笔可选恢复必须$\widehat\Delta>10^{-12}$。这是固定数值边界，不是另调的经济利润参数；没有$\lambda$、$\eta$或置信系数搜索。实际执行直接使用已报价的post/C，不重复收费。通胀在配对log财富比中抵消，仍完整参与峰值和MDD。

    剩余期限估计中的“此后不再交易”与实际每年重新gate不同，因此它是滚动近似；不能称为真实自适应续行价值。若展示原目标$G_H$的对应单位，使用原H缩放，不能重置期限。正估计净益也不保证实现收益为正或全策略优于B。
    ''')
    code(r'''
    methods = pd.DataFrame([
        {'方法':'一年主模型','收益估计期限':'下一年','当前费用':'精确 -log(1-C/V)','后续假设':'本估计期限内持有','主次':'事前主模型'},
        {'方法':'剩余期敏感性','收益估计期限':'当前年至原H','当前费用':'同一精确当前费用','后续假设':'至原H持有','主次':'不替换主模型'}])
    show(methods)
    ''')
    md(r'''
    ## 3. 初配资格、方向检查与服务准入是三件事

    初配先计算经济估计；若不满足正净估计，该候选明确Review，不产生客户服务金融结果。通过后，原Q1目标的初配是mandatory并实际支付一次费用；初配不受“相对全票据”的风险方向veto约束，静态Q1与完整策略准入承担风险检验。

    后续可选交易在正净收益之外，共同要求：同一训练fan、继承当前CPI／原峰值／历史MDD，在**原H买后持有**分支中$k_R\le k_H$。这是经验模型内风险方向，不是条件尾概率校准或10%保证。预测诊断中市场均值损失的小幅改善不能替它提供这种保证。

    完整训练路径要求$k\le\min(k_B,\lfloor0.1n\rfloor)$；各内折自己训练、自己目标并实际执行规则。父层只在本架构自动集合U上检查：$|U|>0$、$k_U\le\lfloor0.1|U|\rfloor$、$k_U\le k_{B,U}$。最终训练与自身内层gate均通过才准入。零U不是零风险证据；不合格时保留Review和NULL。

    实际首次年度MDD超限后，未来决策永久pending hold。财富继续变动，CPI、峰值、历史事件及原H均不重置；没有终端清算费。
    ''')
    code(r'''
    full = tables['full'].copy()
    full['方法']=full.policy.map(companion.CHINESE_LABELS)
    columns=['方法','initial_mean_net_log_gain','training_eligible','training_breach_cap',
             'inner_gate_passed','execution_status','execution_reason']
    show(full.reindex(columns=columns),{'initial_mean_net_log_gain':'{:+.6f}'})
    print('上表仅为全开发集拟合的服务决定，不是新留出验证；Review不会以B或敏感性结果填充。')
    ''')
    md(r'''
    ## 4. 五次重复分别报告，原120条路径不能变成600条独立样本

    每次重复的原120条情景各评估一次；训练集合重叠。每个拟合在观察其留出收益之前，统一决定该折的24条路径是否执行。不同自动子群的条件金融数值不能直接当相同人群上的策略效应。
    ''')
    code(r'''
    repeat = tables['repeat']
    for method in companion.METHODS:
        print(companion.CHINESE_LABELS[method])
        frame=repeat.loc[repeat.policy.eq(method)&repeat.repeat_id.between(1,5),[
            'repeat_id','original_n','assigned_n','financial_null_n','breach_count',
            'conditional_breach_probability','mean_log_growth','mean_mdd','mean_fees']]
        show(frame.rename(columns={'repeat_id':'重复','original_n':'原情景n','assigned_n':'自动n',
            'financial_null_n':'金融NULL_n','breach_count':'超限k','conditional_breach_probability':'条件k/n',
            'mean_log_growth':'平均年化log增长','mean_mdd':'平均MDD','mean_fees':'平均名义费用_初始100'}),
            {'条件k/n':'{:.2%}','平均年化log增长':'{:.3%}','平均MDD':'{:.2%}','平均名义费用_初始100':'{:.4f}'})
    ''')
    md('### 重复准入与全期限事件图\n\n图中重复是划分编号，不是时间趋势；NULL不会画成0风险。下面保留可编辑绘图函数。')
    code(inspect.getsource(plot_repeats)+"\nfig=plot_repeats(tables['repeat'])\ndisplay(fig)\nimport matplotlib.pyplot as plt\nplt.close(fig)")
    md(r'''
    ## 5. 全开发集拟合只用于最终研究状态说明

    这个拟合使用全部120条既有开发路径；对应表现是in-sample，不能与五次OOF相加，也不能称第六次验证。无论敏感性是否更容易准入，事前的一年主模型都保留其真实结论。
    ''')
    code(inspect.getsource(plot_full_admission)+"\nfig=plot_full_admission(tables['full'],tables['training'],tables['inner'])\ndisplay(fig)\nplt.close(fig)")
    md(r'''
    ## 6. 金融差异只与同路径B配对

    B是同训练目标、相同情景上的假想参照，不是被拒客户的服务结果。下面分开列五次OOF与全开发集in-sample；共同自动集为空时，所有金融差值与事件保留NULL。log增长差与MDD差显示为百分点；费用以初始100名义单位表示。
    ''')
    code(r'''
    paired=tables['paired'].copy()
    paired['方法']=paired.policy.map(companion.CHINESE_LABELS)
    paired['log增长差_pp']=paired.mean_delta_log_growth*100
    paired['MDD差_pp']=paired.mean_delta_mdd*100
    for scope,frame in paired.groupby('scope',sort=False):
        print(scope)
        show(frame[['repeat_id','方法','common_auto_n','model_breaches','B_breaches',
            'log增长差_pp','MDD差_pp','mean_delta_nominal_fees']],
            {'log增长差_pp':'{:+.3f}','MDD差_pp':'{:+.3f}','mean_delta_nominal_fees':'{:+.4f}'})
    ''')
    md(r'''
    ## 7. 可选交易的执行证据

    正净益与风险方向条件都是每个可选恢复的必要条件。下表按保存trace的类别与scope分别汇总；一年内多次决策互相依赖，不能作为独立成功试验。若没有可选交易，不制造散点、均值或“交易成功率”；初配另有部署资格约束。
    ''')
    code(r'''
    decisions=tables['decisions']
    if decisions.empty:
        print('没有已保存的可选决策组；保留缺失，不能制造交易散点或0风险结论。')
    else:
        group=decisions.groupby(['category','scope','policy'],dropna=False).agg(
            已保存路径组=('group','size'),可选决策=('optional_decisions','sum'),可选恢复=('optional_restores','sum'),
            经济不足持有=('growth_nonpositive_holds','sum'),正收益被方向veto=('positive_growth_risk_veto_holds','sum'),
            不合规恢复=('invalid_optional_restores','sum'),最小交易净log估计=('minimum_estimated_net_restore','min')).reset_index()
        group['policy']=group.policy.map(companion.CHINESE_LABELS)
        show(group,{'最小交易净log估计':'{:+.6f}'})
        assert group['不合规恢复'].eq(0).all()
    ''')
    md(r'''
    ## 8. 判断与限制

    本阶段将“当前费用已经计入”变为逐笔可审查的毛收益、精确log费用和净收益记录；只有在明确经验预测与持有续行假设下，才说交易通过正估计门槛。它不能保证真实收益、完整期限安全或条件预测概率校准。

    准入若失败，先保留失败机制及服务覆盖代价，不能按已看到的留出结果放宽gate、改α、重新选期限或转用年度／TV兜底。本笔记没有把敏感性选作新主模型，也没有把全开发集in-sample表现变成独立证据。

    已保存PNG需实际检查后才允许执行导出。整页HTML只做静态结构与执行输出核对；本地文件浏览器预览的已有策略限制仍保留，不绕过，也不声称整页视觉通过。
    ''')
    code(r'''
    end=companion.require_complete(ROOT)
    assert ready['input_artifact_sha256']==end['input_artifact_sha256']
    evidence={**ready['input_artifact_sha256'],
        'scripts/build_benefit_cost_notebook.py':companion.sha(ROOT/'scripts/build_benefit_cost_notebook.py'),
        str(companion.FIGURES/'manifest.json'):companion.sha(ROOT/companion.FIGURES/'manifest.json')}
    show(pd.DataFrame([{'来源路径':path,'SHA256':digest} for path,digest in evidence.items()]))
    print('只读呈现结束；来源起止一致。原120重复开发，全开发集in-sample，零准入风险NULL。')
    ''')
    result=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'name':'python3','display_name':'Python 3 (.venv)','language':'python'},
        'language_info':{'name':'python','version':'3.12'},
        'builder':'scripts/build_benefit_cost_notebook.py',
        'scope':'Read-only audited study companion; no parsed market rows, simulation, fitting or policy switch'})
    nbf.validate(result)
    return result


def build(root=ROOT):
    root=Path(root);start=sha(__file__);ready=require_complete(root);tables=load_tables(root)
    save_figures(root,tables,ready)
    notebook=make_notebook(tables);(root/NOTEBOOK).parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,root/NOTEBOOK)
    if start!=sha(__file__):raise RuntimeError('Builder source changed during build')
    print(json.dumps({'notebook':str(NOTEBOOK),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
        'builder_sha256':start,'status':'built; inspect both PNGs before --confirm-figures and --execute'},ensure_ascii=False))


def execute(root=ROOT):
    root=Path(root);ready=require_complete(root)
    manifest=readj(root/FIGURES/'manifest.json')
    if manifest.get('visual_review_confirmed') is not True:
        raise RuntimeError('Actually inspect both PNGs, then explicitly --confirm-figures')
    check_mapping(root,manifest['source_sha256']);check_mapping(root,manifest['outputs_sha256'])
    import os
    env=dict(os.environ);env['JUPYTER_PATH']=str(root/'.venv/share/jupyter')
    subprocess.run([sys.executable,'-m','nbconvert','--execute','--to','notebook','--inplace',str(root/NOTEBOOK),
                    '--ExecutePreprocessor.timeout=180'],cwd=root,env=env,check=True)
    subprocess.run([sys.executable,'-m','nbconvert','--to','html','--no-input',str(root/NOTEBOOK),
                    '--output',HTML.name],cwd=root,env=env,check=True)
    saved=nbf.read(root/NOTEBOOK,as_version=4);nbf.validate(saved)
    code=[c for c in saved.cells if c.cell_type=='code']
    if [c.execution_count for c in code]!=list(range(1,len(code)+1)):
        raise RuntimeError('Notebook was not executed top to bottom')
    if any(o.output_type=='error' for c in code for o in c.outputs):raise RuntimeError('Saved notebook contains execution error')
    after=require_complete(root)
    if ready['input_artifact_sha256']!=after['input_artifact_sha256']:raise RuntimeError('Read-only inputs changed')
    print(json.dumps({'notebook_sha256':sha(root/NOTEBOOK),'html_sha256':sha(root/HTML),
        'code_cells':len(code),'errors':0,'html_browser_preview_passed':False},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--confirm-figures',action='store_true',help='Only after actual view_image inspection of both PNGs')
    action.add_argument('--execute',action='store_true',help='Execute and export the existing built notebook after figure review')
    args=parser.parse_args()
    if args.confirm_figures:confirm_figures()
    elif args.execute:execute()
    else:build()
