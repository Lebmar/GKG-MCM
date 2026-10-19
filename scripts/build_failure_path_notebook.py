"""Audited fixed-path mechanisms and interventions research companion.

Only saved results are parsed. No raw market CSV, NPZ execution tree, fit,
forecast evaluator or policy simulator is imported. All four study audits
must bind complete current manifests before result tables are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = Path("results/model_review/failure_path_mechanisms_v1")
INTERVENTIONS = Path("results/model_review/failure_action_intervention_v1")
COSTS = Path("results/model_review/current_candidate_cost_bridge_v1")
FANS = Path("results/model_review/matched_parameter_fan_contrast_v1")
SCALE = Path("results/model_review/intervention_scale_equivalence_v1")
FIGURES = Path("research/figures/failure_path_development")
NOTEBOOK = Path("notebooks/failure_path_development.ipynb")
HTML = Path("notebooks/failure_path_development.html")
SOURCE_NOTE = "Source: HKUST(GZ) 2026 MCM synthetic development scenarios 1–120; CC BY-NC-SA 4.0; Team 87."


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def check_mapping(root, mapping):
    for name, digest in mapping.items():
        if sha(Path(root) / name) != digest:
            raise RuntimeError("Frozen source/input/artifact changed: " + name)


def require_complete(root=ROOT):
    root = Path(root); snapshot = {}; studies = {}
    for folder in (MECHANISMS, INTERVENTIONS, COSTS, FANS):
        mp, ap = root / folder / "manifest.json", root / folder / "independent_audit.json"
        if not mp.is_file() or not ap.is_file():
            raise RuntimeError("Wait for complete study and bound independent audit: " + str(folder))
        m, a = readj(mp), readj(ap)
        expected_status = "passed" if folder in (COSTS, FANS) else "complete"
        audit_binding = a.get("parent_manifest_sha256", a.get("producer_manifest_sha256"))
        if m.get("status") != expected_status or a.get("status") != "passed" or audit_binding != sha(mp):
            raise RuntimeError("Complete study and current-manifest passed audit required: " + str(folder))
        final_key = "final_scenarios_read" if folder in (COSTS, FANS) else "final_scenarios_loaded"
        if m.get(final_key) is not False:
            raise RuntimeError("Study must explicitly exclude final scenario values")
        if folder in (COSTS, FANS):
            # The bridge records one binding map and explicitly checks it
            # again before writing. Do not claim two saved producer maps.
            mapping = m["inputs_and_sources" if folder == COSTS else "sources_and_inputs_sha256"]
            check_mapping(root, mapping); snapshot.update(mapping)
            pp = Path("docs/current_candidate_cost_bridge_protocol.json" if folder == COSTS else "docs/matched_parameter_fan_contrast_protocol.json")
            protocol = readj(root / pp)
            if sha(root / pp) != m["protocol_sha256"] or protocol.get("execution_ready") is not True:
                raise RuntimeError("Frozen complete cost evidence bridge required")
            expected_map = ({**protocol["input_sha256"], **protocol["source_sha256"]} if folder == COSTS else
                            {**protocol["frozen_input_sha256"], **protocol["execution_sources_sha256"]})
            if mapping != expected_map:
                raise RuntimeError("Bridge source/input map differs from frozen protocol")
            snapshot[str(pp)] = sha(root / pp)
        else:
            for start, end in (("sources", "sources_at_end"), ("input_sha256", "input_sha256_at_end")):
                if m.get(start) != m.get(end) or not m.get(start):
                    raise RuntimeError("Source/input start/end identity required")
                check_mapping(root, m[start]); snapshot.update(m[start])
        before, after = a["sources_and_artifacts_at_start"], a["sources_and_artifacts_at_end"]
        if before != after:
            raise RuntimeError("Independent audit snapshots differ")
        check_mapping(root, before); snapshot.update(before)
        outputs = m["output_sha256" if folder == FANS else "outputs_sha256"]
        check_mapping(root / folder, outputs)
        snapshot.update({str(folder / name): digest for name, digest in outputs.items()})
        snapshot[str(folder / "manifest.json")] = sha(mp)
        snapshot[str(folder / "independent_audit.json")] = sha(ap)
        studies[str(folder)] = dict(manifest=m, audit=a)
    # This last post-hoc diagnosis is not a fifth independent market replay.
    # Its already-audited CF input, frozen source/lemma synthetic tests and
    # complete saved-array witness checks are bound separately and labelled.
    mp = root / SCALE / "manifest.json"; m = readj(mp)
    plan_path = Path("docs/intervention_scale_equivalence_plan.json"); plan = readj(root / plan_path)
    if (m.get("status") != "complete" or plan.get("execution_ready") is not True
            or m["plan_sha256"] != sha(root / plan_path)
            or m["sources_and_inputs_at_start"] != m["sources_and_inputs_at_end"]
            or m["summary"]["no_new_market_load"] is not True
            or m["summary"]["new_fits"] != 0 or m["summary"]["new_policy_simulations"] != 0
            or m["summary"]["admission_changed"] is not False):
        raise RuntimeError("Complete frozen saved-array scale diagnosis required")
    expected = {**plan["sources_sha256"], **plan["frozen_input_sha256"], str(plan_path): sha(root / plan_path)}
    if m["sources_and_inputs_at_start"] != expected:
        raise RuntimeError("Scale witness source/input closure differs")
    check_mapping(root, expected); snapshot.update(expected)
    check_mapping(root / SCALE, m["outputs_sha256"])
    snapshot.update({str(SCALE / name): digest for name, digest in m["outputs_sha256"].items()})
    snapshot[str(SCALE / "manifest.json")] = sha(mp)
    return dict(studies=studies, scale_manifest=m, input_artifact_sha256=dict(sorted(snapshot.items())))


def plot_style():
    import matplotlib
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15,
        "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "svg.fonttype": "none", "svg.hashsalt": "failure-path-development-v1",
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#24323c",
        "axes.labelcolor": "#24323c", "axes.edgecolor": "#9ca9b1"})


def plot_mdd_trajectories(endpoints):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plot_style(); fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.97, bottom=.13, top=.83, hspace=.33, wspace=.18)
    specifications = (("full", "model", "Full M; fan 120", "#1f5b7a", "-"),
        ("full", "B", "Full own-target B", "#1f5b7a", "--"),
        ("inner", "model", "Inner M; fan 90", "#b27c2b", "-"),
        ("inner", "B", "Inner own-target B", "#b27c2b", "--"))
    for ax, sid in zip(axes.ravel(), (35, 112, 52, 98), strict=True):
        for stage, arm, label, color, style in specifications:
            rows = endpoints.loc[endpoints.pair_id.eq(f"{stage}_{sid}") & endpoints.arm.eq(arm)].sort_values("endpoint")
            if rows.endpoint.tolist() != list(range(31)):
                raise ValueError("Complete original 31 checkpoints required")
            ax.plot(rows.endpoint, rows.running_mdd, color=color, linestyle=style,
                    linewidth=2.4 if arm == "model" else 1.8, label=label)
        ax.axhline(.4, color="#222222", linestyle=":", linewidth=1.5)
        ax.set(title=f"Original scenario {sid}", xlim=(0, 30), ylim=(0, .5), xticks=[0, 5, 10, 15, 20, 25, 30])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.grid(axis="y", alpha=.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Whole-history running MDD")
    for ax in axes[-1]:
        ax.set_xlabel("Original year endpoint")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .92), ncol=2, frameon=False)
    fig.suptitle("Full versus own-inner trajectories: changed targets and training fans", x=.52, y=.98, fontsize=18)
    fig.text(.075, .045, "Dotted line: original 40% loss threshold. Each M/B pair shares its own target.\n"
             "Full and inner differ in inception targets and training inclusion; these four selected IDs are not an independent risk sample.", fontsize=12)
    fig.text(.075, .012, SOURCE_NOTE, fontsize=12, color="#52616b")
    return fig


def load_tables(root=ROOT):
    """Read only the three audited saved-result table families."""
    import numpy as np
    import pandas as pd
    root = Path(root)
    endpoints = pd.read_csv(root / MECHANISMS / "account_endpoints.csv.gz")
    annual = pd.read_csv(root / MECHANISMS / "annual_states.csv.gz")
    paired = pd.read_csv(root / MECHANISMS / "paired_annual.csv.gz")
    summary = pd.read_csv(root / MECHANISMS / "paired_summary.csv.gz")
    cross = pd.read_csv(root / MECHANISMS / "cross_stage_summary.csv.gz")
    interventions = pd.read_csv(root / INTERVENTIONS / "intervention_summary.csv.gz")
    costs = pd.read_csv(root / COSTS / "research_summary.csv")
    fans = pd.read_csv(root / FANS / "same_state_Q.csv")
    scale = pd.read_csv(root / SCALE / "pair_summary.csv.gz")
    scale_details = readj(root / SCALE / "pair_results.json")
    bridge = readj(root / COSTS / "bridge.json")
    registry = readj(root / "docs/main_model_registry.json")
    if (len(endpoints), len(annual), len(paired), len(summary), len(cross), len(interventions), len(costs)) != (682, 660, 330, 11, 4, 11, 15):
        raise ValueError("Complete fixed inventories required")
    for frame, keys in ((endpoints, ["pair_id", "arm", "endpoint"]), (annual, ["pair_id", "arm", "year"]),
                        (paired, ["pair_id", "year"]), (summary, ["pair_id"]), (cross, ["scenario_id"]),
                        (interventions, ["pair_id"]), (costs, ["cost_label", "research_policy"])):
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate result identity")
    if set(summary.pair_id) != set(interventions.pair_id) or set(cross.scenario_id) != {35, 112, 52, 98}:
        raise ValueError("No replacement or omitted fixed pair permitted")
    if len(fans) != 3 or len(scale) != 11 or set(scale.pair_id) != set(summary.pair_id):
        raise ValueError("Three fixed fans and all eleven scale cases required")
    if sorted(fans.fan_n.tolist()) != [96, 119, 120] or fans.contrast_id.nunique() != 1 or fans.year.nunique() != 1:
        raise ValueError("One same-state120/119/96contrast required")
    detail_index = {row["pair_id"]: row for row in scale_details}
    if len(detail_index) != 11 or set(detail_index) != set(summary.pair_id):
        raise ValueError("Complete scale detail identity required")
    scale["future_normalized_state_actions_and_Q_verified"] = scale.pair_id.map(
        lambda pid: None if detail_index[pid].get("future_checks") is None else detail_index[pid]["future_checks"]["all_conditions_passed"])
    if not costs.n.eq(120).all() or sorted(costs.cost_rate.unique()) != [0., .00125, .0025, .005, .01]:
        raise ValueError("Five original cost rates and complete120paths required")
    if bridge["current_service"]["status"] != "execution_review" or bridge["current_service"]["financial_outputs"] is not None:
        raise ValueError("Research cost stress must not create C service financials")
    if registry["current_C_execution"]["final_status"] != "execution_review":
        raise ValueError("This fixed companion assumes the unchanged rejected full C result")
    active = interventions.tau.notna()
    if not np.allclose(interventions.loc[active, "delta_mdd"], interventions.loc[active, "treatment_mdd"] - interventions.loc[active, "control_mdd"], atol=1e-12, rtol=0):
        raise ValueError("Intervention difference direction must remain treatment minus control")
    if not np.allclose(interventions.loc[active, "delta_log_growth"], np.log(interventions.loc[active, "treatment_terminal_real"] / interventions.loc[active, "control_terminal_real"]), atol=1e-12, rtol=0):
        raise ValueError("Intervention growth difference is total log, not annualized")
    tables = dict(endpoints=endpoints, annual=annual, paired=paired, summary=summary, cross=cross,
                  interventions=interventions, costs=costs, bridge=bridge, fans=fans, scale=scale)
    tables["crossing_components"] = crossing_components(tables)
    return tables


def plot_interventions(interventions):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_style(); frame = interventions.copy(); y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(17, 11), sharey=True)
    fig.subplots_adjust(left=.20, right=.96, bottom=.24, top=.84, wspace=.25)
    fields = (("delta_mdd", 100., "Whole-history MDD difference (pp)", "#1f5b7a"),
              ("delta_log_growth", 1., "Net TOTAL log-growth difference", "#b27c2b"))
    for ax, (field, scale, xlabel, color) in zip(axes, fields, strict=True):
        active = frame.tau.notna(); values = frame.loc[active, field].to_numpy(float) * scale
        ax.barh(y[active], values, height=.62, color=color)
        ax.axvline(0, color="#6f7e87", linewidth=1)
        span = max(float(np.max(np.abs(values))) if len(values) else 0., .001)
        ax.set_xlim(min([0.] + values.tolist()) - span * .25, max([0.] + values.tolist()) + span * .35)
        for pos, value in zip(y[active], values, strict=True):
            label = "≈0" if abs(value) <= (1e-12 * scale) else f"{value:+.4f}"
            ax.annotate(label, (value, pos), xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center", fontsize=12)
        for pos in y[~active]:
            ax.text(0, pos, "No disagreement: NULL", va="center", fontsize=12)
        ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.18)
    labels = [f"{r.pair_id} / τ={int(r.tau)}" if r.tau == r.tau else f"{r.pair_id} / τ=NULL" for r in frame.itertuples()]
    axes[0].set(yticks=y, yticklabels=labels, ylim=(len(frame) - .5, -.5))
    fig.suptitle("One fixed first-action intervention, then the original policy replans", x=.54, y=.96, fontsize=18)
    fig.text(.06, .09, "Every value is treatment minus control. Only the first saved M/B action disagreement is replaced by B's action.\n"
             "Target, training fan, technical price and original history remain fixed; future actions follow each branch's own state.\n"
             "All 11 selected pairs are retained. This is a path intervention, not a new admitted strategy or a population effect.", fontsize=12)
    fig.text(.06, .025, SOURCE_NOTE, fontsize=12, color="#52616b")
    return fig


def plot_cost_bridge(costs):
    import matplotlib.pyplot as plt
    plot_style(); fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.subplots_adjust(left=.08, right=.96, bottom=.31, top=.77, wspace=.25)
    specifications = (("fixed_target_learner", "Rejected fixed M", "#1f5b7a", "-", "o"),
        ("annual_then_pending", "Own-target B", "#b27c2b", "--", "s"),
        ("equal_thirds_annual", "Exact-thirds annual", "#667681", ":", "^"))
    for policy, label, color, style, marker in specifications:
        group = costs.loc[costs.research_policy.eq(policy)].sort_values("cost_rate")
        if len(group) != 5:
            raise ValueError("No omitted fee point allowed")
        x = 100 * group.cost_rate.to_numpy(float)
        for ax, field, scale in ((axes[0], "mean_log_growth", 100.), (axes[1], "breach_count", 1.)):
            ax.plot(x, scale * group[field], label=label, color=color, linestyle=style, marker=marker, linewidth=2, markersize=7)
    for ax in axes:
        ax.set(xlabel="Known transaction fee rate (%)", xticks=[0, .125, .25, .5, 1.], xlim=(-.03, 1.05)); ax.grid(axis="y", alpha=.18)
        ax.axvline(.25, color="#aaaaaa", linestyle=":", linewidth=1)
    axes[0].set(ylabel="Mean annual log growth (%)", title="Same fixed target and technical price")
    axes[1].set(ylabel="Observed MDD > 40% crossings / 120", title="Complete development resubstitution")
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .90), ncol=3, frameon=False)
    fig.suptitle("Reused cost evidence: exact base-array identity before five-rate comparison", x=.52, y=.98, fontsize=18)
    fig.text(.06, .115, "Full-development 120 paths; fixed target 45/5/50, technical η=0.03, original H=30 and L=40%.\n"
             "Known fees enter actual and forecast calculations. No cost misestimation, refit or admission test is performed.\n"
             "Current C remains Review; these are archived research outcomes, not served investment returns.", fontsize=12)
    fig.text(.06, .025, SOURCE_NOTE, fontsize=12, color="#52616b")
    return fig


def save_figures(root, tables, ready, *, refresh_draft=False):
    import matplotlib.pyplot as plt
    root = Path(root); folder = root / FIGURES
    if folder.exists():
        previous = readj(folder / "manifest.json")
        if not refresh_draft or previous.get("visual_review_confirmed") is not False:
            raise RuntimeError("Only an explicitly unconfirmed presentation draft may be refreshed")
    else:
        previous = {}; folder.mkdir(parents=True)
    plots = (("01_full_inner_running_mdd", plot_mdd_trajectories(tables["endpoints"])),
             ("02_crossing_wealth_peak_identity", plot_crossing_components(tables["crossing_components"])),
             ("03_first_action_interventions", plot_interventions(tables["interventions"])),
             ("04_reused_known_cost_evidence", plot_cost_bridge(tables["costs"])))
    outputs = {}
    for name, figure in plots:
        for extension in ("png", "svg"):
            path = folder / (name + "." + extension)
            figure.savefig(path, dpi=240, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
            if extension == "svg":
                # Strip EACH line, not merely the file's final whitespace.
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            outputs[str(path.relative_to(root))] = sha(path)
        plt.close(figure)
    writej(folder / "manifest.json", dict(status="presentation_draft", completed_utc=datetime.now(timezone.utc).isoformat(),
        presentation_draft_revision=previous.get("presentation_draft_revision", 0) + 1,
        figure_count=4, source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
            "tests/test_build_failure_path_notebook.py": sha(ROOT / "tests/test_build_failure_path_notebook.py")},
        input_artifact_sha256=ready["input_artifact_sha256"], outputs_sha256=outputs,
        visual_review_confirmed=False, visual_review_scope="Pending actual inspection of every saved PNG; whole HTML browser preview not claimed"))


def observed_summary(tables):
    frame = tables["interventions"].loc[tables["interventions"].tau.notna()]
    down = int(frame.delta_mdd.lt(-1e-12).sum()); up = int(frame.delta_mdd.gt(1e-12).sum())
    tied = len(frame) - down - up
    removed = int(frame.delta_breach.lt(0).sum()); added = int(frame.delta_breach.gt(0).sum())
    return (f"已核验 **11组配对、22条完整账户、682个期末状态和660个年度决策**。五个模型独有超限发生点中，"
        f"财富与历史峰值影响不同，不能统一归因为当期股票过多或没有再平衡。预先固定的首次动作干预保留全部11组："
        f"在{len(frame)}组有动作分歧的配对中，MDD下降 **{down}** 组、上升 **{up}** 组、仅舍入差 **{tied}** 组；事件消失 **{removed}** 组、新增 **{added}** 组。"
        "这些是已知失效路径上的受控诊断，不构成可部署的新策略。**当前全开发客户C仍为Review，服务金融结果仍为NULL。**")


def make_notebook(tables):
    cells = []
    def md(value):
        cells.append(nbf.v4.new_markdown_cell(dedent(value).strip()))
    def code(value):
        cells.append(nbf.v4.new_code_cell(dedent(value).strip()))
    md("""
    # SafeInvest：失效路径、历史峰值与一次固定动作干预

    本笔记是可复查的模型研究记录，不是比赛论文。它连接三项已经独立核验的工作：逐年路径会计、单次动作干预、历史成本结果的严格身份复用。原模型、参数网格、准入和所有旧研究产物均不改变。

    ## tl;dr
    """)
    md(observed_summary(tables))
    md(r"""
    ## Context & Methods

    当前全120拟合的已选控制器训练超限为9/120；它已选β的自身内层程序是10/120，而同内层job/目标/ID的年度B为8/120。因此当前比较准入失败，不能用训练合格覆盖它。full固定控制器与四个inner学习结果不是同一个固定策略；这次只追踪已知差异ID35、112、52、98，不能把它们当代表性或独立新市场样本。

    ### Key Assumptions

    - 原客户C的$H=30,L=0.40$、初始100票据、真实CPI、原历史峰值和运行MDD始终保留。首次$MDD-L>10^{-12}$后永久pending hold，不能恢复或重置风险历史。
    - 恢复目标只计一次实际转入交易额的0.25%自融资费用，初配收费，无终端清算费。$Q_g$已包含当前及续行费用，不再重复扣一次报价。
    - 预测fan是自己的训练路径无条件calendar联合后缀；当前动作后以各分支年度B续行。实际策略逐年重规划。这些是声明的近似，没有被本次会计检验变成校准条件概率。
    - 风险偏好技术索引η不被重新解释为识别出的客户偏好；10%仍是规范性服务参考，不能当统计保证。
    - 这里只读保存结果，不解析市场CSV或NPZ树。原121—180早期曾接触，不称新盲测，本笔记也不读取其收益值。
    """)
    md("""
    ## Data

    读表前检查路径会计、动作干预、成本复用和固定状态fan比较四项研究的完整manifest、独立审计对当前manifest的绑定及所有当前SHA。前两项producer起止输入/源码快照必须相同；成本复用与fan比较producer保存单一绑定map并在输出前复核，不能称它们保存了两份起止map。四项独立审计各自有起止快照。

    比例等价是另外一项事后数学机制检查：绑定已独立核验的CF输入、冻结的lemma/合成测试源码、保存数组检查及其起止SHA。它没有新独立市场重算或新的预测校准证明，不把它计作第五次独立风险验证。

    科研图使用可编辑Matplotlib代码，PNG为240dpi，SVG保留文字。每张图真实检查后才允许执行最终笔记。HTML只承诺执行与静态内容检查；未声称整页浏览器视觉预览或完全离线数学渲染。
    """)
    code("""
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    from IPython.display import display
    ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'scripts/build_failure_path_notebook.py').is_file())
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_failure_path_notebook as companion
    ready=companion.require_complete(ROOT)
    tables=companion.load_tables(ROOT)
    figures=companion.readj(ROOT/companion.FIGURES/'manifest.json')
    assert figures['visual_review_confirmed'] is True
    for name in ('source_sha256','input_artifact_sha256','outputs_sha256'):
        companion.check_mapping(ROOT,figures[name])
    SOURCE_NOTE=companion.SOURCE_NOTE
    plot_style=companion.plot_style
    def show(frame,formats=None):
        display(frame.style.format(formats or {},na_rep='NULL').hide(axis='index'))
    print('四项研究当前SHA/绑定独立审计和事后比例机制来源检查通过；无新市场解析、拟合或准入。')
    show(pd.DataFrame([{'研究':name,'producer状态':value['manifest']['status'],
        '独立审计':value['audit']['status'],'审计绑定manifest':value['audit'].get('parent_manifest_sha256',value['audit'].get('producer_manifest_sha256'))}
        for name,value in ready['studies'].items()]))
    """)
    md("""
    ## Results

    ### 1. 跨full/inner，第一步目标差异先于后来动作差异

    四个原ID在full与其own-inner下，都从第1年初配后的暴露和财富开始分歧；原因是初始目标已经不同。首次动作分歧可以迟至第15年，不能当第一次状态分歧。full和inner还改变训练fan及样本纳入关系；η均为0.03仍不能分离目标与fan作用。

    下图M是各自冻结模型，B是同job同目标年度至超限后持有的参照。每个M/B配对目标相同，full/inner之间目标不同。纵轴是各账户完整历史运行MDD。
    """)
    code(inspect.getsource(plot_mdd_trajectories) + "\nfig=plot_mdd_trajectories(tables['endpoints'])\ndisplay(fig)\nimport matplotlib.pyplot as plt\nplt.close(fig)")
    code("""
    show(tables['cross'][['scenario_id','full_target','inner_target','full_training_fan_n','inner_training_fan_n',
        'first_action_divergence_year','first_posttrade_weights_divergence_year',
        'full_first_crossing','inner_first_crossing','full_mdd','inner_mdd']],
        {'full_mdd':'{:.2%}','inner_mdd':'{:.2%}'})
    """)
    md(r"""
    ### 2. 财富更高，仍可能回撤更高

    对同市场、同目标的M/B，用各自实际财富$R$及完整历史峰值$P$，当期回撤差严格满足
    $$DD_M-DD_B=(R_B-R_M)/P_B+R_M(1/P_B-1/P_M).$$
    两项可能异号。它们分解的是**超限当年的current DD**；不能把这些项直接加总为终身running MDD的因果贡献。running MDD始终是每条账户自己全部历史current DD的最大值。

    inner112和R5/112在超限年都恢复到与B相同目标，财富也更高；更高的历史峰值仍令回撤比例更大。R4/112超限年反而持有更少股票，当年组合收益还优于B；统一解释为“当期股票太多导致亏得更多”不成立。
    """)
    code(inspect.getsource(plot_crossing_components) + "\nfig=plot_crossing_components(tables['crossing_components'])\ndisplay(fig)\nplt.close(fig)")
    code("""
    show(tables['crossing_components'],{'current_DD_gap_pp':'{:+.4f}','wealth_ratio_pp':'{:+.4f}','peak_ratio_pp':'{:+.4f}'})
    """)
    md(r"""
    ### 3. 实际费差与配置收益差：精确会计，保留正负号

    每年的真实log增长差等于费用占财富比的log差加交易后组合毛收益的log差，共同通胀严格抵消。完整年度区间互不重叠，因此这些**实现会计量**可望远镜求和到终值log财富比。每年预测Q覆盖的未来区间却重叠，不能把负Q差累加成实现损失。

    名义财富也保存两种等价递推恒等式：直接继承财富差−本年fee差＋三资产P&L差；或以B当年毛收益为次序的继承/费用/暴露三项。两种式子是替代表达，不再互相相加，也不是唯一因果归因。
    """)
    code("""
    frame=tables['summary'][['pair_id','first_action_divergence_year','model_first_crossing','B_first_crossing',
        'model_mdd','B_mdd','total_nominal_fee_gap','total_log_fee_drag_gap','total_log_posttrade_gross_gap',
        'terminal_log_real_wealth_ratio']]
    show(frame,{'model_mdd':'{:.2%}','B_mdd':'{:.2%}',
        'total_nominal_fee_gap':'{:+.5f}','total_log_fee_drag_gap':'{:+.6f}',
        'total_log_posttrade_gross_gap':'{:+.6f}','terminal_log_real_wealth_ratio':'{:+.6f}'})
    assert np.allclose(frame.total_log_fee_drag_gap+frame.total_log_posttrade_gross_gap,
        frame.terminal_log_real_wealth_ratio,atol=2e-12,rtol=0)
    print('名义费用差以初始100为单位；log量是完整期限总log，不是货币损失或年化百分比。')
    """)
    md(r"""
    ### 4. 只强制一次预定动作，然后让原策略继续重规划

    每组只取原M/B第一次动作分歧τ，强制把M当年动作替换为B原动作。随后使用同一个原目标、训练fan、η、原H和该分支自己的历史继续运行原π。对照也从同一状态重算，必须复现冻结M持仓、实际动作、Q和终值。超限后的pending对各分支独立生效，绝不重置。

    这与“把之后每年都改成B”不同，也与比较$Q^π$两动作最优值不同。τ不按改善结果搜索；全部11组保留，不推荐强制动作，不把干预当新准入策略。所有差值方向为**干预−对照**；增长差是剩余完整期限总log财富比，未年化。

    图内≈0仅表示$|\Delta MDD|\le10^{-12}$的舍入量级差异，不更改原超限判定。全部11组的完整期限事件和首次超限年均未改变，不能说首次动作干预消除了这些失效。
    """)
    code(inspect.getsource(plot_interventions) + "\nfig=plot_interventions(tables['interventions'])\ndisplay(fig)\nplt.close(fig)")
    code("""
    show(tables['interventions'][['pair_id','tau','original_action','forced_action','control_breach','treatment_breach',
        'control_first_crossing','treatment_first_crossing','delta_mdd','delta_log_growth','delta_nominal_fees']],
        {'delta_mdd':'{:+.6f}','delta_log_growth':'{:+.6f}','delta_nominal_fees':'{:+.6f}'})
    """)
    md(r"""
    ### 4a. 九组MDD近似不变的充分机制：未来比例等价

    这项检查由已知“9/11组MDD只差舍入、终值财富却改变”的结果启发，属于**事后数学机制分析**。全部11组都保留。若某个共同期末满足持仓与历史峰值同时乘同一个$k>0$、CPI相同、运行MDD与pending相同，且目标、fan、η、H/L都相同，则比例费用与交易后持仓同样按$k$缩放。两动作分支的终值/当前财富比例和DD不变，故Q增长、事件计数、分数及推荐动作相同；按同一实际市场归纳可保持未来比例。

    数学引理是精确等价条件；“数值接近”本身不保证临近分数阈值的动作相同。因此保存数组还逐一检查了见证点之后的所有动作、Q、计数、状态、费用与持仓比例。见证点只取强制干预后首次满足条件的内部期末；若后续检查失败，不能换一个更晚见证点。终端才相同比例不算未来机制证据。

    九组均在最终MDD形成之前达到并维持这种充分比例等价；full98和R4/112未找到内部见证点，也确实有实质MDD变化。无见证点不能证明其他机制不可能。先前已形成的不同回撤不能被后来的比例等价追溯抹去；前缀总费用和历史首次超限也不要求按比例相同。该检查基于旧CF独立审计加合成数学测试和保存数组检查，未进行新的独立市场或全策略重算。
    """)
    code("""
    show(tables['scale'][['pair_id','earliest_interior_equivalence_endpoint','scale_k',
        'control_final_mdd_first_attained_endpoint','treatment_final_mdd_first_attained_endpoint',
        'equivalence_strictly_before_both_final_mdd_attainments',
        'future_normalized_state_actions_and_Q_verified','delta_final_mdd','status']],
        {'scale_k':'{:.6f}','delta_final_mdd':'{:+.6f}'})
    """)
    md(r"""
    ### 4b. 同一状态下的特定fan替换改变动作

    对原11组案例先按元数据枚举，唯一目标、η、市场、H/L、费用及源码相同而fan不同的跨fit组合，是full112与R5/112。两者第5年首次动作不同；该年之前的实际持仓、CPI、财富、峰值和MDD一致。把这个共同状态固定，仅替换full120、去掉实际112后的119、原outer96三个fan，重新计算同一冻结的两动作B续行估值。

    120与119都推荐restore，96推荐hold：**删除实际路径自身不足以解释这次翻转**。整体fan成员与分母差异足以改变该固定状态的技术分数与动作，这一局部计算输入机制已被固定状态比较隔离。它不识别完整期限的风险差异有多少由fan造成，不证明哪个fan的概率正确，也不是LOOCV重拟合或新的可部署策略。fan变化与条件预测校准是不同问题。
    """)
    code("""
    show(tables['fans'][['year','fan_label','fan_n','actual_ID_in_fan','qg_hold','qg_restore',
        'qbad_hold_count','qbad_restore_count','predicted_growth_advantage_restore','predicted_score','chosen_action']],
        {k:'{:+.6f}' for k in ['qg_hold','qg_restore','predicted_growth_advantage_restore','predicted_score']})
    print('Q是剩余完整期限总log；事件计数各除以自己的N。该同状态fan比较独立复算335条paired成员/670个动作分支。')
    """)
    md("""
    ### 5. 当前被拒绝固定候选的历史成本证据

    先要求旧成本实验base下固定候选的48个NPZ字段与当前full候选逐dtype/shape/value/NaN mask完全相同，两个参照各36字段也逐一匹配；仅旧参照缺失的target元数据由旧冻结协议明确补充。通过后才复用五个原成本点，不继承旧菜单拟合、外层fit、扩展更新策略或准入结论。

    下图全部为同120条开发路径上的固定45/5/50、η=0.03候选和假设性参照，属于resubstitution研究。成本在实际与预测中都已知；不是费用估计错误压力试验。不同已知费率可能改变动作和历史，因此不应只把费用从同一毛收益扣去。没有在新费率重新拟合或重新做准入；当前C仍Review。
    """)
    code(inspect.getsource(plot_cost_bridge) + "\nfig=plot_cost_bridge(tables['costs'])\ndisplay(fig)\nplt.close(fig)")
    code("""
    show(tables['costs'][['cost_label','cost_rate','research_policy','n','breach_count','mean_log_growth',
        'median_cagr','p10_cagr','median_mdd','p90_mdd','mean_nominal_fees']],
        {k:'{:.2%}' for k in ['cost_rate','mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd']}|
        {'mean_nominal_fees':'{:.5f}'})
    """)
    md("""
    ## Takeaways

    这次把“不稳定”落实到了具体原ID、原年份和账户历史，而不是用单一平均收益解释全部风险。跨拟合初始目标差异、后续动作、财富及峰值路径互相影响；有更高财富仍更高MDD的合法情形。精确会计和独立计算核验排除了本次检查中的峰值重置、费用重复或配对身份错误，并未校准预测概率或证明总体安全。

    单次受控干预回答一个有限的路径问题。它的正负结果全部保留；不能据此挑一条改善交易、删除不利情景、放宽门槛或声称所有安全策略不可能存在。原模型准入判断不变；论文、最终评估和提交材料仍是后续阶段。
    """)
    code("""
    end=companion.require_complete(ROOT)
    assert ready['input_artifact_sha256']==end['input_artifact_sha256']
    refs=[companion.MECHANISMS,companion.INTERVENTIONS,companion.COSTS,companion.FANS]
    show(pd.DataFrame([{'研究':str(folder),'manifest_SHA256':companion.sha(ROOT/folder/'manifest.json'),
        '独立审计_SHA256':companion.sha(ROOT/folder/'independent_audit.json')} for folder in refs]))
    print('比例机制manifest：',companion.sha(ROOT/companion.SCALE/'manifest.json'),
          '；验证类型：已审CF输入＋冻结数学合成测试与保存数组核验；不声称新独立市场重算。')
    print('完整输入/源码/输出绑定起止一致；没有市场解析、新拟合、参数选择或服务金融回填。')
    """)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "builder": "scripts/build_failure_path_notebook.py",
        "scope": "Audited descriptive fixed-path mechanisms, single-action interventions and strict historical cost evidence bridge; current C remains Review"})
    nbf.validate(notebook)
    return notebook


def build(root=ROOT, *, refresh_draft=False):
    root = Path(root)
    if (root / HTML).exists() or ((root / NOTEBOOK).exists() and not refresh_draft):
        raise RuntimeError("Do not replace an existing notebook companion")
    source = sha(__file__); ready = require_complete(root); tables = load_tables(root)
    save_figures(root, tables, ready, refresh_draft=refresh_draft)
    notebook = make_notebook(tables)
    nbf.write(notebook, root / NOTEBOOK)
    if source != sha(__file__):
        raise RuntimeError("Builder changed during authoring")
    print(json.dumps(dict(status="built_pending_actual_PNG_inspection", notebook=str(NOTEBOOK), figure_count=4,
                         code_cells=sum(c.cell_type == "code" for c in notebook.cells)), ensure_ascii=False))


def confirm_figures(root=ROOT):
    root = Path(root); path = root / FIGURES / "manifest.json"; manifest = readj(path)
    if manifest["visual_review_confirmed"] is True:
        raise RuntimeError("Figure review was already confirmed")
    for key in ("source_sha256", "input_artifact_sha256", "outputs_sha256"):
        check_mapping(root, manifest[key])
    manifest.update(status="complete", visual_review_confirmed=True, visual_review_utc=datetime.now(timezone.utc).isoformat(),
        visual_review_scope="Actual four saved PNGs inspected at reading size; values, signs, scales, labels, legends and clipping checked. Whole HTML browser preview not claimed.")
    writej(path, manifest)


def execute(root=ROOT):
    root = Path(root); ready = require_complete(root); fm = readj(root / FIGURES / "manifest.json")
    if fm["visual_review_confirmed"] is not True or fm.get("status") != "complete":
        raise RuntimeError("Actually inspect every PNG before explicit --confirm-figures")
    for key in ("source_sha256", "input_artifact_sha256", "outputs_sha256"):
        check_mapping(root, fm[key])
    env = dict(os.environ); env["JUPYTER_PATH"] = str(root / ".venv/share/jupyter")
    subprocess.run([sys.executable, "-m", "nbconvert", "--execute", "--to", "notebook", "--inplace", str(root / NOTEBOOK), "--ExecutePreprocessor.timeout=240"], cwd=root, env=env, check=True)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--no-input", str(root / NOTEBOOK), "--output", HTML.name], cwd=root, env=env, check=True)
    notebook = nbf.read(root / NOTEBOOK, as_version=4); nbf.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if ([cell.execution_count for cell in code_cells] != list(range(1, len(code_cells) + 1))
            or any(output.output_type == "error" for cell in code_cells for output in cell.outputs)):
        raise RuntimeError("Notebook execution incomplete or contains errors")
    html = (root / HTML).read_text()
    from html.parser import HTMLParser
    class ContentCounter(HTMLParser):
        def __init__(self):
            super().__init__(); self.tables = 0; self.inline_images = 0
        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.tables += 1
            if tag == "img" and dict(attrs).get("src", "").startswith("data:image/png;base64,"):
                self.inline_images += 1
    counter = ContentCounter(); counter.feed(html)
    if counter.tables < 9 or counter.inline_images != 4:
        raise RuntimeError("Expected nine result/provenance tables and four embedded figures")
    end = require_complete(root)
    if ready["input_artifact_sha256"] != end["input_artifact_sha256"]:
        raise RuntimeError("Source evidence changed during notebook execution")
    execution = dict(status="passed", completed_utc=datetime.now(timezone.utc).isoformat(),
        notebook_sha256=sha(root / NOTEBOOK), html_sha256=sha(root / HTML), builder_sha256=sha(__file__),
        figure_manifest_sha256=sha(root / FIGURES / "manifest.json"),
        code_cells=len(code_cells), sequential_execution=True, errors=0,
        figures_png_actually_inspected=4, html_tables=counter.tables, html_inline_pngs=counter.inline_images,
        input_artifact_sha256=ready["input_artifact_sha256"],
        entire_HTML_browser_visual_preview=False, fully_offline_math_rendering_claimed=False,
        scope="Read-only saved results; no raw market parser, execution/forecast kernel, fitter or selector imported")
    writej(root / FIGURES / "notebook_execution.json", execution)
    print(json.dumps({k: execution[k] for k in ("status", "code_cells", "errors", "html_tables", "html_inline_pngs", "notebook_sha256", "html_sha256")}))


def crossing_components(tables):
    import pandas as pd
    rows = []
    names = {"inner_35": "Inner 35 / year 19", "inner_112": "Inner 112 / year 26", "outer_r2_52": "R2 52 / year 28",
             "outer_r4_112": "R4 112 / year 26", "outer_r5_112": "R5 112 / year 26"}
    for pair, name in names.items():
        summary = tables["summary"].loc[tables["summary"].pair_id.eq(pair)].iloc[0]
        year = int(summary.model_first_crossing)
        row = tables["paired"].loc[tables["paired"].pair_id.eq(pair) & tables["paired"].year.eq(year)].iloc[0]
        total, wealth, peak = (100 * float(row[k]) for k in ("current_drawdown_gap_end", "current_DD_wealth_ratio_part", "current_DD_peak_ratio_part"))
        if abs(total - wealth - peak) > 1e-10:
            raise ValueError("Signed current-DD components must add exactly")
        rows.append(dict(pair_id=pair, label=name, crossing_year=year, current_DD_gap_pp=total, wealth_ratio_pp=wealth, peak_ratio_pp=peak))
    return pd.DataFrame(rows)


def plot_crossing_components(components):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_style(); fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.subplots_adjust(left=.23, right=.95, bottom=.30, top=.77)
    y = np.arange(len(components)); height = .28
    ax.barh(y - .18, components.wealth_ratio_pp, height=height, color="#1f5b7a", label="Current wealth ratio part")
    ax.barh(y + .18, components.peak_ratio_pp, height=height, color="#b27c2b", label="Inherited peak ratio part")
    ax.scatter(components.current_DD_gap_pp, y, s=60, marker="D", color="#242a30", zorder=4, label="Sum: current DD gap")
    for j, row in enumerate(components.itertuples()):
        for value, offset in ((row.wealth_ratio_pp, -.18), (row.peak_ratio_pp, .18)):
            ax.text(value + (.12 if value >= 0 else -.12), j + offset, f"{value:+.2f}",
                    ha="left" if value >= 0 else "right", va="center", fontsize=12)
        ax.annotate(f"sum {row.current_DD_gap_pp:+.2f}", (row.current_DD_gap_pp, j), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=12,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=1.5))
    ax.axvline(0, color="#6f7e87", linewidth=1)
    ax.set(yticks=y, yticklabels=components.label, xlim=(-6, 10.3), ylim=(len(y) - .5, -.6),
           xlabel="Current drawdown difference: model minus own B (percentage points)")
    ax.grid(axis="x", alpha=.18); ax.legend(loc="upper left", bbox_to_anchor=(-.27, 1.24), ncol=2, frameon=False)
    fig.suptitle("Five model-only crossings: wealth and historical peak can offset", y=.97, fontsize=18)
    fig.text(.06, .115, "Exact identity: DD(M)-DD(B) = [R(B)-R(M)]/P(B) + R(M) × [1/P(B)-1/P(M)].\n"
             "Signed components describe current DD at the model's crossing. They are not additive lifetime-MDD or causal effects.\n"
             "In Inner 112 and R5 112, higher model wealth coexists with a higher drawdown. Repeated 112 contexts are dependent.", fontsize=12)
    fig.text(.06, .035, SOURCE_NOTE, fontsize=12, color="#52616b")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--refresh-draft", action="store_true")
    group.add_argument("--confirm-figures", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    elif args.refresh_draft:
        build(refresh_draft=True)
    elif args.confirm_figures:
        confirm_figures()
    else:
        execute()
