"""Build the compact Sources payload from saved, verified development results."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/model_review"
OUT = BASE / "research_checkpoint_v1/execution_sources_receipt.json"


def item(identity, title, files, rows, definitions, methods, caveats):
    labels = [{"label": path} for path in files]
    return {"id": identity, "title": title,
            "assumptions": ["原始开发情景1–120；整路径外折；已重复使用的数据。"],
            "queries": [{"id": identity + "-saved", "source": {
                "label": title, "files": labels, "metricDefinitions": definitions,
                "filters": ["只读取已保存研究产物，不重新拟合或载入市场收益"],
                "caveats": caveats},
                "sourceRoles": [{"kind": "file", "label": path, "role": "保存的方法、结果或核查证据"} for path in files],
                "reportingPeriod": "2026-10-17执行阶段开发研究",
                "columns": list(rows[0]) if rows else [], "rows": rows,
                "methods": [{"language": "python", "code": methods}]}]}


def main():
    caveats = ["OOF由不同训练折学得，不能作单一固定策略的风险证书。",
               "V1已看过121–180结果；本轮没有新的盲测集。",
               "所有新实验均未成为最终参赛模型；没有上传CMT。"]
    q2 = pd.read_csv(BASE / "q2_execution_development_v2/summary.csv")
    rows = [{"策略": r.policy, "原路径数": int(r.total_paths), "超限数": int(r.breach_count),
             "中位实际CAGR_pct": 100 * r.median_cagr, "P90实际回撤_pct": 100 * r.p90_mdd,
             "平均总名义费用_初始100": r.mean_nominal_fees,
             "实际交易比例_pct": 100 * r.mean_trade_fraction,
             "更新比例_29次机会_pct": 100 * r.mean_target_update_fraction} for r in q2.itertuples()]
    items = [item("current-c", "C客户：同120条完整路径的五策略比较",
                  ["results/model_review/q2_execution_development_v2/summary.csv",
                   "results/model_review/q2_execution_development_v2/manifest.json",
                   "docs/q2_execution_development_protocol.json"], rows,
                  [{"label": "超限", "definition": "30年年度端点最大实际回撤超过40%+1e−12；原120条作分母。"},
                   {"label": "费用与比例", "definition": "总名义费用按初始100计；交易分母30次，更新分母29次。"}],
                  "summary = read_csv('q2_execution_development_v2/summary.csv')\n# no path removal, policy selection or final evaluation; pct =100 *saved_fraction",
                  caveats + ["精确三等分年度对照使用不同目标，不能把差异全归因于执行。", "Expanded与Priced相同；六个训练环境均选择不更新模式。"])]
    cost = pd.read_csv(BASE / "selected_controller_cost_stress_v1/summary.csv").query("scope == 'outer_oof'")
    rows = [{"费率_pct": 100 * r.cost_rate, "策略": r.policy, "原路径数": int(r.total_paths),
             "超限数": int(r.breach_count), "平均年化log增长_pct": 100 * r.mean_log_growth,
             "平均总名义费用_初始100": r.mean_nominal_fees,
             "实际交易比例_pct": 100 * r.mean_trade_fraction} for r in cost.itertuples()]
    items.append(item("known-fees", "冻结控制器的五档已知费用压力",
                      ["results/model_review/selected_controller_cost_stress_v1/summary.csv",
                       "results/model_review/selected_controller_cost_stress_v1/independent_audit.json",
                       "docs/selected_controller_cost_stress_protocol.json"], rows,
                      [{"label": "压力口径", "definition": "真实费用与预测费用同时改变；固定默认成本下选中的目标、价格、模式和准入。"}],
                      "rows = read_csv('selected_controller_cost_stress_v1/summary.csv').query(\"scope =='outer_oof'\")\n# same120 paths reused at each known rate; no new training selection",
                      caveats + ["不是费用估错压力或重新风险校准。", "学习器交易占比的均值非单调；不可据此选择最佳费率。 "]))
    verification = json.loads((BASE / "research_checkpoint_v1/verification.json").read_text())
    reproduction = json.loads((BASE / "execution_stage_reproduction_v1/manifest.json").read_text())
    assert verification["status"] == reproduction["status"] == "passed"
    rows = [{"通过测试数": verification["unit_tests"]["passed"],
             "已核来源快照数": verification["saved_snapshots_checked"],
             "不同文件数": verification["distinct_hashed_files"],
             "隔离复现实验数": len(reproduction["stages"]),
             "已执行代码单元数": verification["notebook"]["executed_code_cells"]}]
    items.append(item("execution-evidence", "计算与来源证据的范围",
                      ["results/model_review/research_checkpoint_v1/verification.json",
                       "results/model_review/execution_stage_reproduction_v1/manifest.json",
                       "scripts/verify_research_checkpoint.py", "scripts/reproduce_execution_stage.py"], rows,
                      [{"label": "同源隔离复现", "definition": "原归档前后不变；七runner在临时副本重跑，输出逐字节或NPZ逐数组精确一致。"}],
                      "assert verification['status'] ==reproduction['status'] =='passed'\n# source-file checks, test cases and numeric replay are not additional market samples",
                      caveats + ["独立数值算法由AI协助另写，不是外部人类同行审稿。", "测试、数百万次机械检查与同源复现均不能保证未来风险。 "]))
    original = json.loads((BASE / "research_checkpoint_v1/sources_receipt.json").read_text())
    OUT.write_text(json.dumps({"schemaVersion": 1, "items": original["items"] + items}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(str(OUT.relative_to(ROOT)))


if __name__ == "__main__": main()
