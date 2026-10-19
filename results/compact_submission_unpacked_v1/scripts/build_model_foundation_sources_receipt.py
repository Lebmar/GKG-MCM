"""Extend the saved three-diagnostic Sources panel with audited controls."""
from pathlib import Path
import json
import pandas as pd
from build_execution_sources_receipt import item

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/model_review'


def main():
    for name in ('simple_execution_comparison_v1','refit_stability_v1'):
        assert json.loads((BASE/name/'independent_audit.json').read_text())['status']=='passed'
    assert json.loads((BASE/'simple_execution_summary_v1/manifest.json').read_text())['status']=='complete'
    assert json.loads((BASE/'simple_execution_reproduction_v1/manifest.json').read_text())['status']=='passed'
    payload=json.loads((BASE/'research_checkpoint_v1/admission_sources_receipt.json').read_text())
    frame=pd.read_csv(BASE/'simple_execution_summary_v1/repeat_methods.csv')
    rows=[{'划分':int(r.repeat_id),'方法':r.policy,'原情景数':120,'自动路径数':int(r.automatic_n),
        '事件数':None if pd.isna(r.breach_n) else int(r.breach_n),
        '条件超限_pct':None if pd.isna(r.conditional_breach_probability) else 100*r.conditional_breach_probability}
        for r in frame.itertuples()]
    payload['items'].append(item('simple-controls','简化执行：训练可行性、准入与成本风险取舍',
        ['docs/simple_execution_comparison_protocol.json','docs/simple_drift_cost_interpretation.json',
         'docs/simple_execution_diagnostic_judgment.json','results/model_review/simple_execution_summary_v1/repeat_methods.csv',
         'results/model_review/simple_execution_summary_v1/paired_comparisons.csv',
         'results/model_review/simple_execution_comparison_v1/independent_audit.json',
         'results/model_review/simple_execution_reproduction_v1/manifest.json'],rows,
        [{'label':'组件与样本','definition':'精确复用125个已审计Q1组件；新增125个Q2训练任务，原120情景重复五次。'},
         {'label':'诊断角色','definition':'年度正控制与0/5%/10%TV规则不逐笔估计收益超过费用，不能充当完整最终Q2策略。'},
         {'label':'配对','definition':'增长/损失/费用差仅在同repeat/fold/原ID共同可用路径上计算，Review金融留空。'}],
        "saved=read_csv('simple_execution_summary_v1/repeat_methods.csv')\n# no refit, threshold selection or pooling as600 independent markets",
        ['包含B使训练可行属于构造性质，不能证明未来风险。',
         'TV省费用且均值增长略高，但同群体平均MDD略高，事件并未支配年度参照。',
         '同源闭包复现与独立算法核算不增加市场样本，也不是外部人类同行评议。']))
    out=BASE/'research_checkpoint_v1/model_foundation_sources_receipt.json'
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(out)


if __name__=='__main__':main()
