"""Read-only companion for the single fixed-cap constrained rollout repair.

Do not run before the parent reports complete audited numeric results and
summary. No market parser, simulator, policy fitting or selection is imported.
The previous growth-only notebook builder and frozen studies remain intact.
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

ROOT=Path(__file__).resolve().parents[1]
STUDY=Path('results/model_review/constrained_fan_execution_v1')
SUMMARY=Path('results/model_review/constrained_fan_summary_v1')
OLD_STUDY=Path('results/model_review/benefit_cost_execution_v1')
FIGURES=Path('research/figures/constrained_fan')
NOTEBOOK=Path('notebooks/constrained_fan_model.ipynb')
HTML=Path('notebooks/constrained_fan_model.html')
PROTOCOL=Path('docs/constrained_fan_execution_protocol.json')
POLICY='constrained_net_growth_rollout'
SOURCE_NOTE='Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120; CC BY-NC-SA 4.0; Team 87.'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def readj(path):return json.loads(Path(path).read_text())


def writej(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def check_mapping(root,values):
    for rel,digest in values.items():
        path=Path(root)/rel
        if not path.is_file() or sha(path)!=digest:raise RuntimeError('Current source/input/artifact differs: '+str(rel))


def require_complete(root=ROOT):
    """Completion and parent binding precede reading any numeric result table.

    Current input hashes stream file bytes, including the original CSV file;
    no market rows are parsed and no return arrays are constructed.
    """
    root=Path(root);values=[]
    gates=((STUDY/'manifest.json','complete'),(STUDY/'independent_audit.json','passed'),
           (SUMMARY/'manifest.json','complete'))
    for rel,status in gates:
        if not (root/rel).is_file():raise RuntimeError('Wait for complete independently audited study and summary: '+str(rel))
        value=readj(root/rel)
        if value.get('status')!=status:raise RuntimeError('Incomplete or unaudited artifact: '+str(rel))
        values.append(value)
    study,audit,summary=values;parent_sha=sha(root/STUDY/'manifest.json')
    if audit.get('parent_manifest_sha256')!=parent_sha or summary.get('parent_manifest_sha256')!=parent_sha:
        raise RuntimeError('Audit/summary must bind the current study manifest')
    if study['sources']!=study['sources_at_end'] or study['input_sha256']!=study['input_sha256_at_end']:
        raise RuntimeError('Study source/input start and end snapshots differ')
    if summary['artifacts_at_start']!=summary['artifacts_at_end']:raise RuntimeError('Summary parent changed')
    for mapping in (study['sources'],study['input_sha256'],summary['source_sha256'],summary['artifacts_at_start']):
        check_mapping(root,mapping)
    check_mapping(root/STUDY,study['outputs_sha256']);check_mapping(root/SUMMARY,summary['outputs_sha256'])
    if sha(root/PROTOCOL)!=study['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
    protocol=readj(root/PROTOCOL)
    if (protocol['methods']!={POLICY:'fixed_alpha_B_continuation'} or protocol['primary_method']!=POLICY
        or study['primary_method']!=POLICY or protocol['local_alpha']!=.1
        or protocol['data']['scenario_ids']!=list(range(1,121)) or protocol['data']['allow_final'] is not False
        or study['scenario_ids_loaded']!=list(range(1,121)) or study['final_scenarios_loaded'] is not False
        or protocol['execution_ready'] is not True):raise RuntimeError('Single method, alpha or data boundary changed')
    # The rejected positive-growth architecture is retained as an audited
    # development diagnostic, not replaced or treated as a financial control.
    old=readj(root/OLD_STUDY/'manifest.json');old_audit=readj(root/OLD_STUDY/'independent_audit.json')
    if old.get('status')!='complete' or old_audit.get('status')!='passed':raise RuntimeError('Previous development diagnostic needs its completed audit')
    if old_audit.get('parent_manifest_sha256')!=sha(root/OLD_STUDY/'manifest.json'):raise RuntimeError('Previous audit binding differs')
    check_mapping(root/OLD_STUDY,old['outputs_sha256'])
    snapshot={str(rel):sha(root/rel) for rel,_ in gates}
    for rel in (PROTOCOL,OLD_STUDY/'manifest.json',OLD_STUDY/'independent_audit.json',
                OLD_STUDY/'training_summary.csv.gz',OLD_STUDY/'fit_admission.csv.gz',
                STUDY/'training_summary.csv.gz',STUDY/'inner_gate_summaries.csv.gz'):
        snapshot[str(rel)]=sha(root/rel)
    for name in summary['outputs_sha256']:snapshot[str(SUMMARY/name)]=sha(root/SUMMARY/name)
    return dict(study=study,audit=audit,summary=summary,protocol=protocol,
                input_artifact_sha256=dict(sorted(snapshot.items())))


def load_tables(root=ROOT):
    import pandas as pd
    root=Path(root)
    repeat=pd.read_csv(root/SUMMARY/'repeat_summary.csv.gz')
    paired=pd.read_csv(root/SUMMARY/'paired_B_summary.csv.gz')
    try:decisions=pd.read_csv(root/SUMMARY/'decision_summary.csv.gz')
    except pd.errors.EmptyDataError:decisions=pd.DataFrame()
    full=pd.DataFrame(readj(root/SUMMARY/'full_development_admission.json'))
    training=pd.read_csv(root/STUDY/'training_summary.csv.gz');inner=pd.read_csv(root/STUDY/'inner_gate_summaries.csv.gz')
    selected=repeat.loc[repeat.policy.eq(POLICY)]
    if len(selected)!=6 or set(selected.repeat_id)!=set(range(6)):raise RuntimeError('Five repeated evaluations and separate full-development scope required')
    if (repeat[['repeat_id','policy']].duplicated().any() or not repeat.original_n.eq(120).all()
        or not (repeat.assigned_n+repeat.financial_null_n).eq(120).all()
        or not repeat.loc[repeat.repeat_id.eq(0),'scope'].eq('in_sample_full120').all()
        or not repeat.loc[repeat.repeat_id.ne(0),'scope'].eq('outer_oof').all()):raise RuntimeError('Population, NULL partition or scope changed')
    financial=['breach_count','conditional_breach_probability','mean_log_growth','mean_mdd','mean_fees','mean_trade_fraction','p90_mdd']
    if not repeat.loc[repeat.assigned_n.eq(0),financial].isna().all().all():raise RuntimeError('Zero automatic coverage must have NULL financial/risk results')
    if len(full)!=1 or full.policy.iloc[0]!=POLICY or int(full.repeat_id.iloc[0])!=0:raise RuntimeError('Exactly one full-development model decision required')
    if len(paired)!=6 or set(paired.repeat_id)!=set(range(6)) or not paired.policy.eq(POLICY).all():raise RuntimeError('Keep all six same-U B comparisons, including NULLs')
    fields=[x for x in paired if x.startswith('mean_delta_')]+['model_breaches','B_breaches']
    if not paired.loc[paired.common_auto_n.eq(0),fields].isna().all().all():raise RuntimeError('Empty common-auto comparison must be NULL')
    if len(decisions):
        if (decisions.invalid_optional_restores.ne(0).any()
            or not (decisions.economic_improvement_restores+decisions.risk_cost_justified_restores).eq(decisions.optional_restores).all()
            or decisions.strict_growth_sacrifice_restores.gt(decisions.risk_cost_justified_restores).any()
            or decisions.optional_restores.gt(decisions.optional_decisions).any()):raise RuntimeError('Economic/constraint-repair classification inconsistent')
    old_fits=pd.read_csv(root/OLD_STUDY/'fit_admission.csv.gz')
    old_training=pd.read_csv(root/OLD_STUDY/'training_summary.csv.gz')
    old_fit=old_fits.loc[old_fits.repeat_id.eq(0)&old_fits.policy.eq('benefit_cost_one_year')]
    if len(old_fit)!=1:raise RuntimeError('Keep the rejected named one-year primary diagnostic')
    old_train=old_training.loc[old_training.job_id.eq(old_fit.job_id.iloc[0])&old_training.policy.eq('benefit_cost_one_year')]
    if len(old_train)!=1:raise RuntimeError('Previous complete-training diagnostic missing')
    previous=dict(n=int(old_train.n_training.iloc[0]),k=None if pd.isna(old_train.breach_count.iloc[0]) else int(old_train.breach_count.iloc[0]),
        B_k=int(old_fit.reference_training_breach_count.iloc[0]),status=str(old_fit.execution_status.iloc[0]),reason=str(old_fit.execution_reason.iloc[0]))
    return dict(repeat=repeat,paired=paired,decisions=decisions,full=full,training=training,inner=inner,previous=previous)


def plot_style():
    import matplotlib
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,
        'axes.titlesize':15,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
        'svg.fonttype':'none','svg.hashsalt':'constrained-fan-model-v1','figure.facecolor':'white',
        'axes.facecolor':'white','text.color':'#27313a','axes.labelcolor':'#27313a','axes.edgecolor':'#a6afb5'})


def plot_repeats(repeat,paired):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    plot_style();fig,axes=plt.subplots(1,2,figsize=(17,8.7),gridspec_kw={'width_ratios':[1.05,1]})
    fig.subplots_adjust(left=.07,right=.98,top=.75,bottom=.22,wspace=.24)
    fig.text(.07,.95,'Constrained net-growth rollout: repeated admission and events',fontsize=20,weight='bold')
    fig.text(.07,.90,'One repair controller fixed before this run; no price, horizon or architecture search follows its outcomes.',fontsize=13)
    fig.text(.07,.855,'Original 120 reused development paths; each fit decides before its held-out returns. Repeats are partitions, not time.',fontsize=13)
    rows=repeat.loc[repeat.policy.eq(POLICY)&repeat.repeat_id.between(1,5)].sort_values('repeat_id')
    pairs=paired.loc[paired.repeat_id.between(1,5)].set_index('repeat_id')
    axes[0].bar(rows.repeat_id,rows.assigned_n,width=.55,color='#205f8a')
    for row in rows.itertuples():
        axes[0].text(row.repeat_id,int(row.assigned_n)+3,str(int(row.assigned_n)),ha='center',fontsize=13)
        values=[(row.breach_count,row.assigned_n),(pairs.loc[row.repeat_id,'B_breaches'],pairs.loc[row.repeat_id,'common_auto_n'])]
        for j,(k,n) in enumerate(values):
            text='NULL\n0 auto' if int(n)==0 else f'{int(k)}/{int(n)}\n{k/n:.1%}'
            fill='#edf0f2' if int(n)==0 else ('#e4eef4' if k/n<=.1+1e-12 else '#f4e9d5')
            axes[1].add_patch(Rectangle((j,row.repeat_id-1),1,1,facecolor=fill,edgecolor='white',linewidth=3))
            axes[1].text(j+.5,row.repeat_id-.5,text,ha='center',va='center',fontsize=14)
    axes[0].set(title='(a) Automatic service / original 120',xlabel='Repeat partition (not time)',
        ylabel='Automatically executed original paths',xticks=range(1,6),ylim=(0,142),yticks=range(0,121,24))
    axes[0].spines[['top','right']].set_visible(False)
    axes[1].set(title='(b) Conditional full-H events, identical U',xlim=(0,2),ylim=(5,0),
        xticks=[.5,1.5],xticklabels=['Constrained\nmodel','B on the same\nautomatic paths'],
        yticks=np.arange(5)+.5,yticklabels=[f'Repeat {i}' for i in range(1,6)])
    axes[1].tick_params(length=0)
    for spine in axes[1].spines.values():spine.set_visible(False)
    fig.text(.07,.14,'Local cap and least-violation actions do not certify the complete policy or future population risk.',fontsize=12)
    fig.text(.07,.10,'Zero admission means NULL risk. B is a hypothetical same-target reference, never a Review fallback.',fontsize=12)
    fig.text(.07,.055,SOURCE_NOTE,fontsize=12)
    return fig


def plot_full_gate(full,training,inner):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plot_style();fig,ax=plt.subplots(figsize=(17,7.5))
    fig.subplots_adjust(left=.04,right=.98,top=.67,bottom=.27)
    fig.text(.05,.94,'Full-development fit: contract, complete risk and service gate',fontsize=20,weight='bold')
    fig.text(.05,.87,'All 120 paths are used for this fit and its displayed evaluation: in-sample, not a sixth validation repeat.',fontsize=13)
    row=full.iloc[0];ts=training.loc[training.job_id.eq(row.job_id)&training.policy.eq(POLICY)]
    gate=inner.loc[inner.context.eq(row.context)&inner.policy.eq(POLICY)]
    if len(ts)!=1 or len(gate)!=1:raise ValueError('Exactly one matching full-fit training summary and own-U gate required')
    ts,gate=ts.iloc[0],gate.iloc[0]
    def count(x):return 'NULL' if x is None or x!=x else str(int(x))
    label=str(row.get('q1_effective_assignment',row.get('case_status','Q1 unavailable')))
    headers=['Q1 / mandatory initial contract','Complete-policy training risk','Own inner automatic set U','Final service decision']
    texts=[label+'\nExact initial fee if executed\nNo short-growth deployment veto',
        f'k={count(ts.breach_count)}; cap={count(row.get("training_breach_cap"))}\nn={int(ts.n_training)}\n'+('passed' if bool(row.get('training_eligible',False)) else 'not eligible'),
        f'k/U={count(gate.selected_breach_count)}/{int(gate.auto_inner_n)}\nB k={count(gate.same_U_B_breach_count)}; cap={count(gate.absolute_alpha_count_cap)}\n'+('passed' if bool(gate.eligible) else 'not eligible'),
        'AUTOMATIC' if row.execution_status=='automated' else 'REVIEW / unavailable']
    for j,(header,text) in enumerate(zip(headers,texts)):
        ax.text(j+.5,-.16,header,ha='center',va='bottom',fontsize=13,weight='bold')
        ax.add_patch(Rectangle((j,0),1,1,facecolor='#e4eef4' if j==3 and row.execution_status=='automated' else '#edf0f2',edgecolor='white',linewidth=4))
        ax.text(j+.5,.5,text,ha='center',va='center',fontsize=13)
    ax.set(xlim=(0,4),ylim=(1,0));ax.axis('off')
    fig.text(.05,.19,'Choose largest already-net total-log Q within the local feasible actions; unresolved empty sets remain explicitly flagged.',fontsize=12)
    fig.text(.05,.135,'A nonpositive-net optional restore must strictly reduce predicted bad count; that is a risk tradeoff, not positive cash profit.',fontsize=12)
    fig.text(.05,.08,'Final complete training and own-U gates can still require Review. References never replace missing service outcomes.',fontsize=12)
    fig.text(.05,.025,SOURCE_NOTE,fontsize=12)
    return fig


def save_figures(root,tables,ready):
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/FIGURES;folder.mkdir(parents=True,exist_ok=True)
    source={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)}
    figures=[('01_repeated_admission_and_events',plot_repeats(tables['repeat'],tables['paired'])),
             ('02_full_development_gate',plot_full_gate(tables['full'],tables['training'],tables['inner']))]
    outputs={}
    for name,fig in figures:
        png,svg=folder/(name+'.png'),folder/(name+'.svg')
        fig.savefig(png,dpi=240);fig.savefig(svg,metadata={'Date':None})
        svg.write_text(svg.read_text().rstrip())
        plt.close(fig)
        for path in (png,svg):outputs[str(path.relative_to(root))]=sha(path)
    after=require_complete(root);source_end={rel:sha(root/rel) for rel in source}
    if source!=source_end or ready['input_artifact_sha256']!=after['input_artifact_sha256']:raise RuntimeError('Read-only source/inputs changed while drawing')
    writej(folder/'manifest.json',dict(status='complete',source_sha256=source,sources_at_end=source_end,
        input_artifact_sha256=ready['input_artifact_sha256'],input_artifact_sha256_at_end=after['input_artifact_sha256'],
        outputs_sha256=outputs,visual_review_confirmed=False,
        visual_scope='Both PNGs await actual inspection; whole-HTML browser preview not passed',source_note=SOURCE_NOTE,
        format={'png_dpi':240,'minimum_font_points':12,'svg_text':True,'svg_trailing_whitespace_stripped_before_hash':True}))


def confirm_figures(root=ROOT):
    root=Path(root);require_complete(root);path=root/FIGURES/'manifest.json';value=readj(path)
    for key in ('source_sha256','input_artifact_sha256','outputs_sha256'):check_mapping(root,value[key])
    value.update(visual_review_confirmed=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
        visual_scope='Both PNGs actually inspected; HTML only static checking, no browser preview claim')
    writej(path,value)


def observed_summary(tables):
    rows=tables['repeat'].loc[tables['repeat'].policy.eq(POLICY)&tables['repeat'].repeat_id.between(1,5)].sort_values('repeat_id')
    counts='、'.join(str(int(x)) for x in rows.assigned_n)
    events='；'.join('NULL（0自动）' if int(r.assigned_n)==0 else f'{int(r.breach_count)}/{int(r.assigned_n)}' for r in rows.itertuples())
    final=tables['full'].iloc[0]
    return f'五次重复自动路径数为**{counts}**；对应条件全期限事件为**{events}**。全开发集拟合的服务状态为`{final.execution_status}`，原因`{final.execution_reason}`。这组全开发结果是in-sample，不能增加验证样本量。'


def make_notebook(tables):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(dedent(s).strip()))
    md(r'''
    # SafeInvest：风险可行集合内的净增长主模型

    这是一个固定局部cap、B续行的受约束rollout启发式。本轮只保留一个事前确定的原则修正候选，没有风险价格或更多期限／架构搜索。局部空集合、完整策略拒绝与Review都会保留，不为争取覆盖率而补入年度／TV兜底。
    ''')
    md(observed_summary(tables))
    md(r'''
    ## 1. 当前来源与审计绑定

    先验证新研究完整、独立审计通过并绑定当前manifest、摘要绑定相同manifest；再逐项核对当前源码、输入和保存输出哈希。哈希流式读取源字节，不解析市场行，不调用模拟器或拟合器。新研究仍只使用既有情景1—120，不能把它们称新盲测；旧V1曾经接触121—180。
    ''')
    code(r'''
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    from IPython.display import display
    ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents]
              if (p/'docs/constrained_fan_execution_protocol.json').is_file())
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_constrained_fan_notebook as companion
    ready=companion.require_complete(ROOT)
    tables=companion.load_tables(ROOT)
    figures=companion.readj(ROOT/companion.FIGURES/'manifest.json')
    assert figures['status']=='complete' and figures['visual_review_confirmed'] is True
    for key in ('source_sha256','input_artifact_sha256','outputs_sha256'):companion.check_mapping(ROOT,figures[key])
    POLICY=companion.POLICY;SOURCE_NOTE=companion.SOURCE_NOTE;plot_style=companion.plot_style
    def show(frame,formats=None):display(frame.style.format(formats or {},na_rep='—').hide(axis='index'))
    print('新旧研究完整性、审计父manifest绑定及当前来源哈希检查通过。未解析原始市场数据。')
    ''')
    md(r'''
    ## 2. 为什么保留失败的增长门槛实验

    前一架构要求每次可选恢复都有严格正净增长。已完成的开发诊断显示，这可以阻止降低风险所需的再平衡；初配正经济估计和逐笔正预期增长不等于完整策略风险可行。下面读取冻结的原一年主模型全开发训练记录，不修改其结果，也不把其NULL金融结果当投资收益对照。

    修正原则是与原目标一致：先满足独立声明的风险约束，再在允许动作中最大化费后log增长。风险合规作为非补偿性服务收益，不另引入概率与金钱交换系数。该修正是在新一轮运行之前固定的；开发数据已反复使用，冻结并不能创造未见市场。
    ''')
    code(r'''
    previous=tables['previous']
    show(pd.DataFrame([{'旧主模型':'一年正净增长gate','完整训练n':previous['n'],
        '策略超限k':previous['k'],'同目标B超限k':previous['B_k'],
        '状态':previous['status'],'拒绝原因':previous['reason']}]))
    print('这是旧开发训练的反例诊断；不与新方法的OOF结果拼接，不用它重选随机种子或放宽α。')
    ''')
    md(r'''
    ## 3. 局部可行集合内选择已经扣费的净Q

    决策时的实际余额、CPI、历史真实峰值、运行MDD和原H只包含此前实现信息。用该任务自己的完整训练联合路径、相同calendar的$t:H$后缀，分别评估当前hold或paid restore，然后每年恢复原Q1目标，直到各预测分支自己的首次MDD超限，之后永久持有。这是明确的B续行近似，实际下一年会重新求局部决策，两者不能混称。

    $Q_g(a)$是当前至原H的**净实际购买力总log增长**，已包含当前和将来的执行费用；$k(a)$是包含原历史的完整期限ever-breach整数计数。设$K=\lfloor0.1N+10^{-12}\rfloor$，$\mathcal A=\{a:k(a)\le K\}$。

    若集合非空，在其中选最高净$Q_g$，总log差不超过$10^{-12}$时保留hold。若集合为空，明确记录`local_model_infeasible`，先选较少的整数k，再选净Q，再hold；这是定义完整因果路径的尽力动作，**不等于修复成功**。

    每次检查原期限事件并不建立条件风险预算递推；重复检查10%不能证明全局10%。预测仍是无条件calendar经验fan，历史状态中可能包含该fan忽略的信息，没有精确条件Q或全局策略改善定理。
    ''')
    code(r'''
    show(pd.DataFrame([
        {'局部情况':'至少一动作满足cap','首先比较':'限制在可行集合','然后比较':'最大净Qg，总log容差1e-12','最终并列':'hold'},
        {'局部情况':'两个动作都不满足cap','首先比较':'最小整数k，明确不可行','然后比较':'该集合内最大净Qg','最终并列':'hold'}]))
    print('没有η/λ，也不再给已经净值化的Q追加一次费用。当前quoted费用与实际收费分别记录。')
    ''')
    md(r'''
    ## 4. 费用、风险修复与强制初配透明区分

    可选恢复若$\Delta Q_g>10^{-12}$，标为`economic_improvement`；若$\Delta Q_g\le10^{-12}$，必须严格有$k_R<k_H$，标为`risk_cost_justification`。这代表承担已记录的净增长代价换取模型内风险降低，不能称负净增长是正金钱利润。另列严格$\Delta Q_g<-10^{-12}$的增长牺牲、是否仍处空集合，以及恢复后是否达到cap。

    `gate_reason`表示选择机制；它与上述成本解释分类不同。例如只有恢复可行且其净Q也更高，机制仍是约束修复，但不能把它误计为增长牺牲。

    初配依Q1为mandatory、精确收费一次。初配也记录相同B续行的Q／计数／报价，仅作诊断，不以局部推荐、短期正增长或全票据比较覆盖原配置。初配不计入可选经济／风险修复分类。实际年度超限后，未来永久pending hold，所有真实财富／CPI／峰值／原H继续累计，无终端清算费。
    ''')
    md(r'''
    ## 5. 五次重复的覆盖与事件

    同一原120条路径重复五次，不是600条独立市场路径；训练集合重叠。每个fit在观察其24条留出收益之前统一决定准入。零准入时风险与金融结果是NULL，不能记成0。
    ''')
    code(r'''
    repeat=tables['repeat']
    frame=repeat.loc[repeat.policy.eq(POLICY)&repeat.repeat_id.between(1,5),[
        'repeat_id','original_n','assigned_n','financial_null_n','breach_count',
        'conditional_breach_probability','mean_log_growth','mean_mdd','mean_fees']]
    show(frame.rename(columns={'repeat_id':'重复','original_n':'原情景n','assigned_n':'自动n',
        'financial_null_n':'金融NULL_n','breach_count':'超限k','conditional_breach_probability':'条件k/n',
        'mean_log_growth':'平均年化log增长','mean_mdd':'平均MDD','mean_fees':'平均名义费用_初始100'}),
        {'条件k/n':'{:.2%}','平均年化log增长':'{:.3%}','平均MDD':'{:.2%}','平均名义费用_初始100':'{:.4f}'})
    ''')
    code(inspect.getsource(plot_repeats)+"\nfig=plot_repeats(tables['repeat'],tables['paired'])\ndisplay(fig)\nimport matplotlib.pyplot as plt\nplt.close(fig)")
    md(r'''
    ## 6. 完整策略与自身U门槛仍可拒绝

    支持Balanced的完整训练要求$k_\pi\le\min(k_B,\lfloor0.1n\rfloor)$。各内折用自己的目标与市场训练完整规则，只有合格内折才能在其留出路径提供自动证据。父层只在自己的U上要求正样本量、绝对事件cap、且不超过同U、各自内折目标的B事件。Q1分类、最终完整训练与自身内层gate都要通过；参照从不补缺失服务收益。

    全开发集拟合使用全部120条既有路径；表现是in-sample，不是第六次验证，也不能补救不稳定OOF结果。本轮结果后不扩张新交易族、搜索价格或放宽α强求准入。
    ''')
    code(inspect.getsource(plot_full_gate)+"\nfig=plot_full_gate(tables['full'],tables['training'],tables['inner'])\ndisplay(fig)\nplt.close(fig)\nshow(tables['full'][['policy','training_eligible','training_breach_cap','inner_gate_passed','execution_status','execution_reason']])")
    md(r'''
    ## 7. 同路径金融比较与实际可选决策证据

    B比较只在同repeat／fold／context／原ID的共同自动子群上进行。不同准入人群的条件均值不能直接相减当政策收益。原增长门槛若全Review，则其金融效果为NULL；覆盖率变化不是可测的投资回报改善。
    ''')
    code(r'''
    paired=tables['paired'].copy()
    paired['log增长差_pp']=paired.mean_delta_log_growth*100
    paired['MDD差_pp']=paired.mean_delta_mdd*100
    for scope,group in paired.groupby('scope',sort=False):
        print(scope)
        show(group[['repeat_id','common_auto_n','model_breaches','B_breaches','log增长差_pp','MDD差_pp','mean_delta_nominal_fees']],
            {'log增长差_pp':'{:+.3f}','MDD差_pp':'{:+.3f}','mean_delta_nominal_fees':'{:+.4f}'})
    ''')
    code(r'''
    decisions=tables['decisions']
    if decisions.empty:
        print('没有自动服务trace的可选决策组；不制造交易散点、利润或0风险。')
    else:
        fields=['optional_decisions','optional_restores','economic_improvement_restores',
            'risk_cost_justified_restores','strict_growth_sacrifice_restores',
            'locally_infeasible_decisions','locally_infeasible_restores','invalid_optional_restores']
        counts=decisions.groupby(['category','scope','repeat_id'],dropna=False)[fields].sum().reset_index()
        show(counts)
        assert counts.invalid_optional_restores.eq(0).all()
        print('按scope和重复分别保留决策计数；一年内/同一路径的多次决策不独立。负净优势的修复是风险代价，非正现金盈利。')
    ''')
    md(r'''
    ## 8. 可解释的主模型与证据边界

    本轮用独立的风险可行性优先级约束净增长目标，保留了逐笔费用、净优势、预测事件差、风险修复代价及不可行状态。它是两动作、B续行、经验forecast下的rollout模型，不是完整受约束动态规划的精确解；局部门槛不证明真实市场安全，数值独立审计不验证预测概率。

    若完整证据仍不支持自动C服务，应保留Review并说明当前固定目标／两动作／forecast与证据门槛的局限；不能宣称所有可能的安全盈利策略都不存在，也不能把全Review包装成可投资策略。若获准，也须同时解释覆盖、OOF风险与forecast限制。

    两个保存PNG需实际检查后才允许执行本笔记和导出。HTML仅做执行输出与静态结构核对；已有本地文件浏览预览策略限制保留，不绕过，不声称整页浏览器视觉检查通过。
    ''')
    code(r'''
    end=companion.require_complete(ROOT)
    assert ready['input_artifact_sha256']==end['input_artifact_sha256']
    evidence={**ready['input_artifact_sha256'],
        'scripts/build_constrained_fan_notebook.py':companion.sha(ROOT/'scripts/build_constrained_fan_notebook.py'),
        str(companion.FIGURES/'manifest.json'):companion.sha(ROOT/companion.FIGURES/'manifest.json')}
    show(pd.DataFrame([{'来源路径':path,'SHA256':digest} for path,digest in evidence.items()]))
    print('来源起止一致；只读保存结果，没有市场解析、新拟合或策略选择。')
    ''')
    result=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (.venv)','language':'python'},
        'language_info':{'name':'python','version':'3.12'},'builder':'scripts/build_constrained_fan_notebook.py',
        'scope':'Read-only audited fixed-cap constrained rollout; original120 reused, full120 in-sample'})
    nbf.validate(result);return result


def build(root=ROOT):
    root=Path(root);source=sha(__file__);ready=require_complete(root);tables=load_tables(root)
    save_figures(root,tables,ready);notebook=make_notebook(tables)
    (root/NOTEBOOK).parent.mkdir(parents=True,exist_ok=True);nbf.write(notebook,root/NOTEBOOK)
    if source!=sha(__file__):raise RuntimeError('Builder changed during build')
    print(json.dumps({'notebook':str(NOTEBOOK),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
        'builder_sha256':source,'status':'built; actually inspect both PNGs then --confirm-figures and --execute'},ensure_ascii=False))


def execute(root=ROOT):
    import os
    root=Path(root);ready=require_complete(root);manifest=readj(root/FIGURES/'manifest.json')
    if manifest.get('visual_review_confirmed') is not True:raise RuntimeError('Inspect both PNGs then explicitly --confirm-figures')
    for key in ('source_sha256','input_artifact_sha256','outputs_sha256'):check_mapping(root,manifest[key])
    env=dict(os.environ);env['JUPYTER_PATH']=str(root/'.venv/share/jupyter')
    subprocess.run([sys.executable,'-m','nbconvert','--execute','--to','notebook','--inplace',str(root/NOTEBOOK),
        '--ExecutePreprocessor.timeout=180'],cwd=root,env=env,check=True)
    subprocess.run([sys.executable,'-m','nbconvert','--to','html','--no-input',str(root/NOTEBOOK),
        '--output',HTML.name],cwd=root,env=env,check=True)
    saved=nbf.read(root/NOTEBOOK,as_version=4);nbf.validate(saved);code=[c for c in saved.cells if c.cell_type=='code']
    if [c.execution_count for c in code]!=list(range(1,len(code)+1)) or any(o.output_type=='error' for c in code for o in c.outputs):
        raise RuntimeError('Notebook execution incomplete or contains an error')
    after=require_complete(root)
    if ready['input_artifact_sha256']!=after['input_artifact_sha256']:raise RuntimeError('Read-only inputs changed')
    print(json.dumps({'notebook_sha256':sha(root/NOTEBOOK),'html_sha256':sha(root/HTML),'code_cells':len(code),
        'errors':0,'html_browser_preview_passed':False},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--confirm-figures',action='store_true',help='Only after actual inspection of both PNGs')
    group.add_argument('--execute',action='store_true',help='Execute built notebook/export HTML after figure review')
    args=parser.parse_args()
    if args.confirm_figures:confirm_figures()
    elif args.execute:execute()
    else:build()
