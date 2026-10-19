"""Build a compact source panel from completed saved research, without fitting."""
import json
from pathlib import Path

import pandas as pd

from build_execution_sources_receipt import item

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/model_review'
OUT = BASE / 'research_checkpoint_v1/admission_sources_receipt.json'


def main():
    stable = BASE / 'refit_stability_v1'
    forecast = BASE / 'forecast_assumptions_v2'
    incremental = BASE / 'incremental_admission_risk_v1'
    for folder in (stable, forecast):
        assert json.loads((folder / 'independent_audit.json').read_text())['status'] == 'passed'
    assert json.loads((stable / 'manifest.json').read_text())['status'] == 'complete'
    caveats = ['仅原开发情景1–120，已重复使用；没有新盲测。',
               'Review金融结果留空；假想参照不是自动服务兜底。',
               '数值审计与测试不增加市场样本，也不是外部人类同行评审。']
    inc = pd.read_csv(incremental / 'incremental_summary.csv')
    rows = [{'策略': r.policy, '新增原情景数': int(r.n), '超限数': int(r.breach_n),
             '平均MDD_pct': 100 * r.mean_mdd, 'P90_MDD_pct': 100 * r.p90_mdd,
             '经验CVaR90_MDD_pct': 100 * r.cvar90_mdd,
             '平均年化log增长_pct': 100 * r.mean_log_growth,
             '总名义费用_初始100': r.mean_nominal_fees} for r in inc.itertuples()]
    items = [item('incremental-24', '旧第二外折新增24条：增长、费用与回撤代价',
        ['docs/incremental_admission_risk_diagnostic_protocol.json',
         'results/model_review/incremental_admission_risk_v1/incremental_summary.csv',
         'results/model_review/incremental_admission_risk_v1/incremental_paired_summary.csv',
         'results/model_review/incremental_admission_risk_v1/consistency_checks.json'], rows,
        [{'label': '比较对象', 'definition': '同一原情景、同一40/25/35目标；全期限30年实际峰值回撤。等权参照目标不同。'},
         {'label': 'CVaR90', 'definition': '24个观察的经验最差10%分数尾质量均值，不是置信上界。'}],
        "saved = read_csv('incremental_summary.csv')\n# no fitting, path filtering or new policy selection",
        caveats + ['两学习器共同准入为48条；新增24条单列，不把二者混为共同72条。'])]
    repeats = pd.read_csv(BASE / 'refit_stability_summary_v1/repeat_summary.csv')
    rows = [{'重复划分': int(r.repeat_id), '方法': r.policy, '原情景数': int(r.original_n),
             '自动准入数': int(r.case_auto_n), 'Review或范围外数': int(r.financial_null_n),
             '准入后超限数': None if pd.isna(r.breach_count) else int(r.breach_count),
             '条件超限_pct': None if pd.isna(r.conditional_breach_probability) else 100 * r.conditional_breach_probability}
            for r in repeats[repeats.policy.str.endswith('learner')].itertuples()]
    items.append(item('five-refits', '五组完整重拟合：准入与条件风险',
        ['docs/refit_stability_protocol.json', 'results/model_review/refit_stability_v1/manifest.json',
         'results/model_review/refit_stability_v1/independent_audit.json',
         'results/model_review/refit_stability_summary_v1/repeat_summary.csv',
         'results/model_review/refit_stability_summary_v1/fit_contexts.csv'], rows,
        [{'label': '分母', 'definition': '每组仍为原120条；外层训练96、评估24，完整25个外层拟合、125训练任务。每拟合在看评估收益前作同一客户准入决定，统一作用于24条路径。'},
         {'label': '解释', 'definition': '600条重复记录不是600独立市场样本；25个拟合重叠，不能据此建立二项置信区间。'}],
        "saved = read_csv('repeat_summary.csv')\n# all five preregistered repeats retained; no favorable seed selected",
        caveats + ['不同准入集合的条件风险不能直接解释为全人群安全优劣。']))
    judgment = json.loads((ROOT / 'docs/forecast_assumption_diagnostic_judgment.json').read_text())
    rows = [{'预测对象': r['target'], '原路径数': r['original_path_n'],
             '状态模型相对日历模型MSE': r['state_to_calendar_mse_ratio'],
             '相对误差减少_pct': 100 * (1 - r['state_to_calendar_mse_ratio']),
             '状态模型误差更低路径占比_pct': 100 * r['path_fraction_state_lower_loss']}
            for r in judgment['primary_results']]
    items.append(item('lag-state', '固定上一年状态比较器：均值预测假设诊断',
        ['docs/forecast_assumption_diagnostic_protocol.json', 'docs/forecast_assumption_diagnostic_judgment.json',
         'results/model_review/forecast_assumptions_v2/summary.csv',
         'results/model_review/forecast_assumptions_v2/independent_audit.json'], rows,
        [{'label': '主指标', 'definition': '每原路径先平均29期，再平均三实际资产通道的训练尺度标准化平方误差；通胀另列。'},
         {'label': '模型', 'definition': '固定岭系数1；逐日历年训练均值中心化、训练96×29残差通道池化尺度；未装入交易控制器。'}],
        "ratio = saved_state_standardized_MSE / saved_calendar_standardized_MSE\n# no lambda search, new trading forecast or conditional tail probability claim",
        caveats + ['小幅均值预测改善不是收益提高，也不证明Q值或30年超限概率校准。']))
    OUT.write_text(json.dumps({'schemaVersion': 1, 'items': items}, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
