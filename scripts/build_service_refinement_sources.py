"""Native Sources payload from audited saved service-refinement artifacts only."""
import json
from pathlib import Path
import pandas as pd
from build_execution_sources_receipt import item
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/model_review'
OUT=BASE/'research_checkpoint_v1/service_refinement_sources_receipt.json'


def number(x,integer=False,percent=False):
    if pd.isna(x):return None
    return int(x) if integer else float(x)*(100 if percent else 1)


def main():
    joint=BASE/'joint_liquidity_refit_v1'
    assert json.loads((joint/'independent_audit.json').read_text())['status']=='passed'
    caveats=['所有评估来自已重复使用的开发情景1–120；没有新的盲测。',
             'AI协助的独立算法审计与同源复现不等于外部人类同行审稿或未来风险保证。']
    cards=[]
    rows=[{'旧训练环境':'外折2','内层自动路径':96,'内层超限':6,'并列β':'0.05 / 0.075 / 0.10','旧选β':.05,'重拟合允许超限':4,'九价最小超限':6},
          {'旧训练环境':'外折4','内层自动路径':72,'内层超限':7,'并列β':'仅0.05位于目标最优集合','旧选β':.05,'重拟合允许超限':4,'九价最小超限':7}]
    cards.append(item('mechanism','参数并列与重拟合失效的不同原因',
        ['results/model_review/refit_admission_diagnostics_v1/findings.json','results/model_review/refit_admission_diagnostics_v1/inner_refit_beta_comparison.csv','docs/coverage_admission_policy_review.json'],rows,
        [{'label':'参数身份','definition':'β/η是有限技术筛选或控制器候选；内层目标并列不识别客户风险厌恶。'},
         {'label':'旧外折4','definition':'较大β新增一内折5/24次超限；共同三折仍7/72；不是同一分母上的风险变化。'}],
        '# Saved descriptive training ledger only; no rejected outer financial outcome or refit rescue.',caveats))
    assign=pd.read_csv(joint/'q1_context_assignments.csv').query("scope=='full_development'")
    rows=[{'画像':r.client,'期限_年':int(r.horizon),'回撤限_pct':100*r.loss_tolerance,'流动性':r.liquidity,'目标票据下限_pct':100*r.floor,
           '有效分配':r.effective_assignment,'权重':None if pd.isna(r.target_weights) else r.target_weights,
           '复核原因':None if pd.isna(r.q1_review_kind) else r.q1_review_kind} for r in assign.itertuples()]
    cards.append(item('liquidity','30/10/0规范约定进入每层完整菜单拟合',
        ['docs/liquidity_policy_refinement_review.json','docs/joint_liquidity_refit_protocol.json','results/model_review/joint_liquidity_refit_v1/q1_context_assignments.csv'],rows,
        [{'label':'下限','definition':'High30%、Medium10%、Low不额外加下限；规范选择，非问卷数据唯一识别，非漂移后余额或提款保证。'},
         {'label':'产品身份','definition':'表为一个full120描述性拟合；E的H6-Balanced不继承C的H30动态证据。'}],
        '# All static subfolds jointly refit with reserve constraints; not a postfilter of an old menu.',caveats))
    frame=pd.read_csv(joint/'summary.csv').query("scope=='outer_oof'")
    rows=[{'方法':r.policy,'原始市场路径':int(r.original_n),'自动路径':number(r.case_auto_n,True),'执行复核路径':number(r.case_execReview_n,True),
           'Q1复核路径':int(r.q1_review_n),'非Balanced路径':int(r.not_balanced_case_n),'参照有结果路径':number(r.reference_available_n,True),
           '有结果路径超限':number(r.breach_count,True),'条件超限_pct':number(r.conditional_breach_probability,percent=True),
           '平均年化log增长_pct':number(r.mean_log_growth,percent=True)} for r in frame.itertuples()]
    cards.append(item('joint','固定两种末端规则的完整嵌套评估',
        ['results/model_review/joint_liquidity_refit_v1/summary.csv','results/model_review/joint_liquidity_refit_v1/context_selections.json','results/model_review/joint_liquidity_refit_v1/paired_summary.csv','results/model_review/joint_liquidity_refit_v1/independent_audit.json'],rows,
        [{'label':'自动准入','definition':'同一C画像的训练fit准入指示；路径加权比例不是客户总体覆盖率，五个训练集重叠。'},
         {'label':'保留集合','definition':'仅保留内层风险合格、自动路径最多及全局增长并列集合，再选最小重拟合可行β；不在全部合格β中回退。'},
         {'label':'参照','definition':'年度等待与三等分均为假想金融参照，不计作自动服务，拒绝路径不补收益。'}],
        "rows=read_csv('joint_liquidity_refit_v1/summary.csv').query(\"scope=='outer_oof'\")\n# Both selectors and primary were fixed before this run; no outcome-based winner selection.",caveats))
    fixed=pd.read_csv(BASE/'fixed_state_forecast_diagnostic_v1/summary.csv').query("level=='analysis_scope'")
    rows=[{'范围':r.analysis_scope,'固定有效状态':int(r.active_state_n),'增长方向一致':int(r.growth_sign_agree_n),'风险定价动作一致':int(r.score_action_agree_n)} for r in fixed.itertuples()]
    cards.append(item('fixed-state','固定状态局部预测诊断及其证据边界',
        ['docs/fixed_state_forecast_diagnostic_protocol.json','results/model_review/fixed_state_forecast_diagnostic_v1/state_comparisons.csv','results/model_review/fixed_state_forecast_diagnostic_v1/independent_audit.json'],rows,
        [{'label':'反事实','definition':'同一当前状态、现在hold或restore、两支后续均接原年度等待参照；不是学习器自适应续策。'},
         {'label':'外折六点','definition':'非随机、仅两fit；六点风险事件预测差和实现差均0，η局部消去；方向不一致不能直接证明条件概率失准。'}],
        '# One realized suffix per fixed state; no population forecast-accuracy estimate, confidence interval, or new policy.',caveats))
    checkpoint=json.loads((BASE/'research_checkpoint_v1/verification.json').read_text())
    replay=json.loads((BASE/'joint_liquidity_reproduction_v1/manifest.json').read_text())
    assert checkpoint['status']==replay['status']=='passed'
    audit=json.loads((joint/'independent_audit.json').read_text())
    rows=[{'测试通过数':checkpoint['unit_tests']['passed'],'已核不同文件':checkpoint['distinct_hashed_files'],'已执行笔记代码单元':checkpoint['notebook']['executed_code_cells'],
           '联合阶段任务数':30,'同源隔离复现':replay['status'],'独立审计':audit['status']}]
    cards.append(item('checks','可复查的计算证据',
        ['results/model_review/research_checkpoint_v1/verification.json','results/model_review/joint_liquidity_reproduction_v1/manifest.json','results/model_review/joint_liquidity_refit_v1/independent_audit.json'],rows,
        [{'label':'同源复现','definition':'临时副本只允许解析1–120；确定性产物逐字节或具名数组精确比较，运行时间元数据另列。'}],
        '# No model fit, market CSV read, policy selection, or final evaluation in this Sources builder.',caveats))
    OUT.write_text(json.dumps({'schemaVersion':1,'items':cards},indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(str(OUT))


if __name__=='__main__':main()
