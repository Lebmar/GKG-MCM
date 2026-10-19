"""Read audited saved outputs; build a separate nonblind final reading copy.

No market CSV parser, model, fitter or parameter selection is used here.
The development companion remains separate from this final descriptive view.
"""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'results/mitigation_locked_final_evaluation_v1'
REPLAY = ROOT/'results/mitigation_locked_final_reproduction_v1'
OUT = ROOT/'notebooks/mitigation_locked_final_2026_10_19_assets'
PAGE = ROOT/'notebooks/mitigation_locked_final_2026_10_19.html'
LABELS = {'mitigation_main': '主模型 V2b', 'same_target_B': '同目标年调／超限后持有',
          'same_target_annual': '同目标持续年调', 'exact_thirds_annual': '等权持续年调'}
COLORS = ['#236B9B', '#5F6641', '#B87A16', '#B15C70']


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): d.update(block)
    return d.hexdigest()


def readj(path): return json.loads(Path(path).read_text())


def build():
    manifest = readj(PARENT/'manifest.json')
    audit = readj(PARENT/'independent_audit.json')
    replay = readj(REPLAY/'manifest.json')
    assert manifest['status'] == 'complete' and audit['status'] == replay['status'] == 'passed'
    assert audit['producer_manifest_sha256'] == sha(PARENT/'manifest.json')
    assert manifest['scope'] == 'previously_exposed_nonblind'
    assert manifest['new_fits'] == manifest['new_selections'] == 0
    assert manifest['sources_and_inputs_at_start'] == manifest['sources_and_inputs_at_end']
    assert replay['sources_and_original_artifacts_at_start'][str((PARENT/'manifest.json').relative_to(ROOT))] == sha(PARENT/'manifest.json')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               [PARENT/'manifest.json', PARENT/'independent_audit.json', REPLAY/'manifest.json',
                ROOT/'docs/mitigation_final_precommit_incident.json']}
    for name, digest in manifest['outputs_sha256'].items():
        p = PARENT/name
        assert sha(p) == digest, name
        sources[str(p.relative_to(ROOT))] = digest
    summary = pd.read_csv(PARENT/'summary.csv.gz', float_precision='round_trip')
    paths = pd.read_csv(PARENT/'Q2_paths.csv.gz', float_precision='round_trip')
    paired = readj(PARENT/'comparisons.json')[:3]
    severe = readj(PARENT/'severe_example.json')
    with np.load(PARENT/'traces.npz', allow_pickle=False) as archive:
        guard_years = archive['mitigation_main__first_guard_trigger']
        ordinary = archive['mitigation_main__automatic_service']
        crossing = archive['mitigation_main__first_crossing']
        assert guard_years.shape == (60,) and ordinary.shape == (60, 30)
        assert np.array_equal(ordinary, np.arange(30)[None, :] < guard_years[:, None]-1)
        timing = pd.DataFrame(dict(scenario_id=range(121,181), first_guard_decision_year=guard_years,
                                   first_breach_endpoint=crossing, ordinary_years=ordinary.sum(axis=1)))
    assert len(summary) == 14 and len(paths) == 240
    for oid in LABELS:
        assert sorted(paths.loc[paths.object_id == oid, 'scenario_id']) == list(range(121, 181))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'font.size': 12,
                         'axes.titlesize': 15, 'axes.labelsize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 12, 'figure.facecolor': 'white',
                         'axes.facecolor': 'white', 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'axes.axisbelow': True})
    images = []

    def finish(fig, name, frame):
        fig.tight_layout()
        fig.savefig(OUT/(name+'.png'), dpi=240, bbox_inches='tight')
        fig.savefig(OUT/(name+'.svg'), bbox_inches='tight')
        frame.to_csv(OUT/(name+'_data.csv'), index=False)
        plt.close(fig)
        images.append(name)

    main = paths[(paths.object_id == 'mitigation_main') & paths.assigned]
    ids = set(main.scenario_id)
    comparable = paths[paths.scenario_id.isin(ids)].copy()
    assert comparable.assigned.all()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for (oid, label), color, style in zip(LABELS.items(), COLORS, ['-', '-.', '--', ':']):
        q = comparable[comparable.object_id == oid]
        for ax, key in zip(axes, ['cagr', 'mdd']):
            values = np.sort(q[key].to_numpy())
            if len(values): ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', label=label, color=color, ls=style, lw=2)
    for ax, text in zip(axes, ['30年实际年化收益 CAGR', '原始实际峰值最大回撤 MDD']):
        ax.set(xlabel=text, ylabel=f'经验累计比例（同一投入分母 N={len(ids)}）', ylim=(0, 1.02))
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc='lower right')
    axes[1].axvline(.4, color='#444444', ls='--', lw=1)
    fig.suptitle('固定策略最终描述：121–180此前已暴露；经验分布不是新盲测或置信区间', fontsize=16, y=1.04)
    finish(fig, '01_final_same_ids_ecdf', comparable)

    q2 = summary[summary.question == 'Q2'].set_index('object_id').loc[list(LABELS)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(4); width = .25
    for j, (key, label, color) in enumerate([('initial_coverage', '初始投入 / 60', '#95B6C9'), ('mean_automatic_fraction', '普通自动服务期 / 原始30年', '#236B9B'), ('mean_trade_fraction', '实际交易期 / 原始30年', '#B87A16')]):
        axes[0].bar(x+(j-1)*width, q2[key], width=width, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=['主模型', '同目标B', '同目标年调', '等权年调'], ylim=(0, 1.16), title='初始投入、普通服务与交易分别计量')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.13), fontsize=12)
    axes[1].bar(x, q2.mean_nominal_fees, color=COLORS)
    for i, row in enumerate(q2.itertuples()):
        axes[1].text(i, row.mean_nominal_fees+.025, f'{row.mean_nominal_fees:.3f}', ha='center', fontsize=12)
    axes[1].set(xticks=x, xticklabels=['主模型', '同目标B', '同目标年调', '等权年调'], ylabel='平均累计名义手续费（初始名义100）', title='较低费用伴随较少交易与更多转介', ylim=(0, max(q2.mean_nominal_fees)*1.2))
    fig.suptitle('主模型全部60条已投入路径最终转人工；费用未计人工服务成本', fontsize=16, y=1.04)
    finish(fig, '02_service_and_fees', q2.reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5)); severe_rows = []
    if severe['scenario_id'] is not None:
        for (oid, label), color, style in zip(LABELS.items(), COLORS, ['-', '-.', '--', ':']):
            arm = severe['traces'][oid]
            for ax, key in zip(axes, ['real_wealth', 'running_mdd']): ax.plot(range(31), arm[key], label=label, color=color, ls=style, lw=2)
            severe_rows.extend(dict(object_id=oid, scenario_id=severe['scenario_id'], endpoint=t, real_wealth=arm['real_wealth'][t], running_mdd=arm['running_mdd'][t]) for t in range(31))
        for ax in axes: ax.legend(); ax.set_xlabel('原始年度端点')
        axes[0].set(ylabel='实际财富（初始100）', title='转介后继续承受实际持仓收益')
        axes[1].set(ylabel='原始峰值累计MDD', title='转介不重置峰值、期限或超限事件')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].axhline(.4, color='#444444', ls='--', lw=1)
        fig.suptitle(f"合成情景ID {severe['scenario_id']}：按主模型最大MDD选择，同值取最小原始ID", fontsize=16, y=1.04)
    else:
        for ax in axes: ax.text(.5, .5, '无已投入主模型路径；财富与风险为NULL', ha='center', transform=ax.transAxes)
    finish(fig, '03_severe_synthetic_path', pd.DataFrame(severe_rows))
    summary.to_csv(OUT/'all14_summary.csv', index=False)
    timing.to_csv(OUT/'all60_guard_timing.csv', index=False)
    pairtable = pd.DataFrame([dict(reference=r['reference_object'], mean_paired_log_growth=r['paired_differences']['log_growth']['mean'], mean_paired_mdd=r['paired_differences']['mdd']['mean'], growth_higher_n=r['paired_differences']['log_growth']['positive_n'], growth_lower_n=r['paired_differences']['log_growth']['negative_n'], mdd_higher_n=r['paired_differences']['mdd']['positive_n'], mdd_lower_n=r['paired_differences']['mdd']['negative_n']) for r in paired])
    pairtable.to_csv(OUT/'paired_comparison.csv', index=False)
    display = summary[['object_id', 'horizon', 'L', 'financial_n', 'breach_count', 'median_cagr', 'p10_cagr', 'median_mdd', 'p90_mdd', 'mean_nominal_fees', 'mean_automatic_fraction']].copy()
    for key in ['L', 'median_cagr', 'p10_cagr', 'median_mdd', 'p90_mdd', 'mean_automatic_fraction']:
        display[key] = display[key].map(lambda v: 'NULL' if pd.isna(v) else f'{v:.2%}')
    display = display.fillna('NULL')
    def img(name):
        return '<figure><img alt="'+html.escape(name)+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'"></figure>'
    text = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定V2b最终技术核验</title><style>body{font:17px/1.65 system-ui,sans-serif;max-width:1180px;margin:auto;padding:24px;color:#18232e}h1{font-size:28px}h2{font-size:23px}img{max-width:100%;height:auto}.table{max-width:100%;overflow-x:auto}table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dce2e6;text-align:right;white-space:nowrap}p{overflow-wrap:anywhere}figure{margin:24px 0}.note{background:#eef4f7;padding:16px;border-left:4px solid #236b9b}</style><body><h1>固定V2b最终技术核验 · 2026-10-19</h1><p>Team87 · Chengxuan YUAN · Lebmar。这里展示独立审计通过后的保存结果，供技术讨论；不是参赛论文。</p><p class="note">121–180已被早先版本评估过。本次在完整规则冻结后计算，不再拟合或调参，但不能恢复为新的盲测。所有指标仅描述这60条合成路径；回撤按年度实际财富端点计算。</p><h2>结果与代价</h2><p>主模型2/60条超限；同目标B为8/60，等权年调为6/60。主模型中位实际年化收益约2.35%，低于参照的3.56%和3.38%。全部60条主模型路径最终转人工，普通自动服务期约19.4%。较低回撤伴随收益与服务连续性代价，不能概括为全面优于基准。</p>'''
    text += '<p>事后完整时序描述：全部60条路径在第4–7个决策年触发转介，其中54条在第7年；普通服务共350个账户年，平均每账户5.833年。两次实际超限发生于原始端点14、23，均晚于首次转介。这个时序不是新的准入阈值，也不能单凭时间集中认定唯一因果机制。</p>'
    text += img(images[0])+img(images[1])+img(images[2])
    text += '<h2>完整14对象与原始期限</h2><p>Q1_A/D未投入，财务指标保留NULL；其等权结果只是各自期限的假想参照。B/C静态产品的最终超限频率高于10%，E的P10收益为负。它们不构成独立自动服务的批准证明。P10/P90使用经验逆CDF，中位数使用通常中位数；期限与L不同的客户不作同口径风险排名。</p><div class="table">'+display.to_html(index=False, border=0)+'</div>'
    text += '<h2>核验与可复现边界</h2><p>已独立重建720个完整账户、全部840行与每个主模型活跃决策；隔离目录复现11个生产输出，所有NPZ值精确一致。它们核验实现与复现，不证明预测正确。运行前源文件／参数／放行哈希已冻结；因文件名错误，预期Git提交晚于启动，完整工程记录保留在docs/mitigation_final_precommit_incident.json。人工转介后假设暂时持有；未假定人工响应、未加入人工服务费。</p><p>源数据为题给合成情景，源自Jordà–Schularick–Taylor Macrohistory Database R6所选美国序列。保留组织者及原作者归属，适用CC BY-NC-SA4.0；图表为分析转换，不是真实历史路径。AI参与建模、代码和核验；参赛者仍须审核结论。</p></body></html>'
    PAGE.write_text(text)
    assert all(sha(ROOT/name) == digest for name, digest in sources.items())
    record = dict(status='built_pending_visual_QA', built_utc=datetime.now(timezone.utc).isoformat(), sources_sha256=sources, source_sha256=sha(__file__), page=str(PAGE.relative_to(ROOT)), page_sha256=sha(PAGE), figures=images, raw_market_reads=0, new_policy_runs=0, new_fits=0, scope='previously_exposed_nonblind', output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'build_manifest.json'})
    (OUT/'build_manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='built', figures=len(images), page=str(PAGE)), ensure_ascii=False))


if __name__ == '__main__': build()
