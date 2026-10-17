"""Plot saved internal OOF execution comparisons; never execute a policy.

Only completed development-result artifacts are read.  Numerical sources,
frozen inputs and result outputs are checked against the execution manifest.
The chart-ready CSVs retain every plotted value and its denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/model_review/finite_horizon_execution_v1"
STYLE_PATH = ROOT / "scripts/plot_development_structure.py"
STYLE_SPEC = importlib.util.spec_from_file_location("execution_figure_style", STYLE_PATH)
STYLE = importlib.util.module_from_spec(STYLE_SPEC)
STYLE_SPEC.loader.exec_module(STYLE)
CLIENTS = ("B", "C", "E")
POLICIES = ("annual_restore", "initial_then_hold", "myopic_gate", "finite_horizon_gate")
LABELS = {"annual_restore": "Annual", "initial_then_hold": "Hold",
          "myopic_gate": "Myopic", "finite_horizon_gate": "Finite-H"}
COLORS = {"annual_restore": STYLE.BLUE, "initial_then_hold": STYLE.GRAY,
          "myopic_gate": STYLE.GOLD, "finite_horizon_gate": "#A65378"}
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_snapshot(manifest):
    if (manifest["scenario_ids"] != list(range(1, 121)) or manifest["final_scenarios_loaded"]
            or manifest["sources"] != manifest["sources_at_end"]
            or manifest["frozen_input_hashes"] != manifest["frozen_input_hashes_at_end"]):
        raise ValueError("A completed, unchanged development-only execution run is required")
    for collection in (manifest["sources"], manifest["frozen_input_hashes"]):
        for relative, digest in collection.items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError(f"Execution provenance changed: {relative}")
    for relative, digest in manifest["outputs_sha256"].items():
        if sha(SOURCE / relative) != digest:
            raise RuntimeError(f"Execution output changed: {relative}")
    audit = manifest["independent_synthetic_tree_audit"]
    if not audit["passed"] or sha(ROOT / audit["path"]) != audit["sha256"]:
        raise RuntimeError("Completed synthetic numerical audit is missing or changed")


def load_artifacts():
    manifest_path = SOURCE / "manifest.json"
    manifest_digest = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    verify_snapshot(manifest)
    paths = {
        "execution_manifest": manifest_path,
        "client_summary": SOURCE / "client_summary.csv",
        "client_paths": SOURCE / "client_path_metrics.csv",
        "resolution_actions": SOURCE / "resolution_action_comparison.csv",
        "model_metadata": SOURCE / "model_metadata.json",
        "profiles": ROOT / "docs/risk_menu_development_protocol.json",
        "plot_style": STYLE_PATH,
        "synthetic_audit": ROOT / manifest["independent_synthetic_tree_audit"]["path"],
    }
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths.values()}
    if hashes[str(manifest_path.relative_to(ROOT))] != manifest_digest:
        raise RuntimeError("Completion manifest changed while loading")
    values = {name: pd.read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text())
              for name, path in paths.items() if name != "plot_style"}
    return values, hashes


def boolean_column(series):
    if series.isna().any() or not series.isin([True, False, "True", "False"]).all():
        raise ValueError("Expected explicit non-null boolean records")
    return series.isin([True, "True"]).to_numpy()


def check_close(actual, expected, field):
    if not np.isclose(actual, expected, atol=2e-12, rtol=2e-12):
        raise ValueError(f"Saved summary disagrees with path records: {field}")


def chart_data(values):
    summary, paths = values["client_summary"], values["client_paths"]
    profiles = {p["name"]: p for p in values["profiles"]["clients"]}
    rows = []
    for client in CLIENTS:
        profile = profiles[client]
        for policy in POLICIES:
            meshes = (.05, .025) if policy in ("myopic_gate", "finite_horizon_gate") else (None,)
            for mesh in meshes:
                matches = (summary.scope == "oof") & (summary.client == client) & (summary.policy == policy)
                matches &= summary.mesh_step.isna() if mesh is None else summary.mesh_step == mesh
                selected_summary = summary.loc[matches]
                if len(selected_summary) != 1:
                    raise ValueError("Every displayed client/policy/mesh needs exactly one OOF summary")
                s = selected_summary.iloc[0]
                matches = (paths.scope == "oof") & (paths.client == client) & (paths.policy == policy)
                matches &= paths.mesh_step.isna() if mesh is None else paths.mesh_step == mesh
                selected = paths.loc[matches]
                if (len(selected) != 120 or set(selected.scenario_id) != set(range(1, 121))
                        or not boolean_column(selected.assigned).all()
                        or not (selected.horizon == profile["horizon"]).all()
                        or not np.allclose(selected.loss_tolerance, profile["loss_tolerance"], atol=0, rtol=0)):
                    raise ValueError("Displayed OOF clients require the complete 120 frozen assigned paths at their own H,L")
                h, loss = profile["horizon"], profile["loss_tolerance"]
                breaches = selected.mdd.to_numpy() - loss > TOL
                if not np.array_equal(breaches, boolean_column(selected.breach)):
                    raise ValueError("Breach flags differ from actual executed MDD")
                g = float(selected.log_growth.mean())
                count = int(breaches.sum())
                trade_fraction = float(selected.trade_fraction.mean())
                count_transfers = float((selected.trade_fraction * h).sum())
                if not 0 <= trade_fraction <= 1 or abs(count_transfers - round(count_transfers)) > 1e-8:
                    raise ValueError("Actual transfer count cannot be recovered from annual fractions")
                initial = int(boolean_column(selected.initial_trade).sum())
                for actual, expected, field in ((s.mean_log_growth, g, "mean_log_growth"),
                                                 (s.breach_count, count, "breach_count"),
                                                 (s.breach_probability, count / 120, "breach_probability"),
                                                 (s.mean_trade_fraction, trade_fraction, "mean_trade_fraction"),
                                                 (s.initial_trade_count, initial, "initial_trade_count")):
                    check_close(actual, expected, field)
                if s.total_paths != 120 or s.assigned_paths != 120 or s.review_paths != 0:
                    raise ValueError("Summary denominators differ from saved paths")
                rows.append({"client": client, "horizon_years": h, "loss_tolerance": loss,
                             "policy": policy, "mesh_step": mesh, "main_figure": mesh is None or mesh == .05,
                             "original_oof_paths": 120, "assigned_paths": 120, "review_paths": 0,
                             "mean_annual_log_growth": g, "annual_geometric_real_growth_percent": 100 * np.expm1(g),
                             "breach_count": count, "breach_denominator_paths": 120,
                             "observed_breach_percent": 100 * count / 120,
                             "actual_transfer_count": int(round(count_transfers)),
                             "annual_decision_denominator": 120 * h,
                             "mean_annual_actual_transfer_percent": 100 * trade_fraction,
                             "initial_trade_count": initial,
                             "initial_allocation_rule": "mandatory" if mesh is None else "optional gate"})
    oof = paths.loc[paths.scope == "oof"]
    reviews = oof.loc[~boolean_column(oof.assigned)]
    financial = ["candidate_index", "log_growth", "cagr", "mdd", "volatility", "terminal_real", "nominal_fees",
                 "nominal_moved", "moved_to_initial_wealth", "mean_annual_fee_fraction", "mean_annual_moved_fraction",
                 "trade_fraction", "target_execution_fraction", "max_end_year_tv_drift", "terminal_equity",
                 "terminal_bond", "terminal_bill", "breach", "excess", "initial_trade"]
    if not reviews[financial].isna().all().all():
        raise ValueError("Frozen Review rows must keep financial outcomes null")
    frame = pd.DataFrame(rows)
    return frame, {"saved_OOF_client_policy_mesh_rows": len(oof), "retained_Review_rows": len(reviews),
                   "all_Review_financial_fields_null": True,
                   "displayed_clients": list(CLIENTS),
                   "omitted_clients": "A and D retain Review; no adviser or fallback returns are plotted"}


def resolution_data(values, data):
    models = [m for m in values["model_metadata"] if m["scope"] == "oof" and m["mesh_step"] == .05]
    if len(models) != 5:
        raise ValueError("Resolution comparison needs all five OOF training contexts")
    resolution = values["resolution_actions"]
    rows = []
    for client in CLIENTS:
        for policy in ("myopic_gate", "finite_horizon_gate"):
            coarse = data.loc[(data.client == client) & (data.policy == policy) & (data.mesh_step == .05)].iloc[0]
            fine = data.loc[(data.client == client) & (data.policy == policy) & (data.mesh_step == .025)].iloc[0]
            selected = []
            for model in models:
                assigned = model["frozen_assignments"][client]
                columns = [i for i, target in enumerate(model["targets"]) if target["name"] == assigned]
                if len(columns) != 1:
                    raise ValueError("Resolution must compare each client's actual frozen assigned target")
                record = resolution.loc[(resolution.context == model["context"]) & (resolution.policy == policy)
                                         & (resolution.horizon == coarse.horizon_years)
                                         & (resolution.target_column == columns[0])]
                if len(record) != 1:
                    raise ValueError("Missing assigned-client resolution record")
                selected.append(record.iloc[0])
            n = sum(int(r.path_year_decisions) for r in selected)
            if n != int(coarse.annual_decision_denominator):
                raise ValueError("Resolution decision denominator differs from assigned OOF client paths")
            rows.append({"client": client, "horizon_years": int(coarse.horizon_years), "policy": policy,
                         "coarse_step": .05, "fine_step": .025,
                         "coarse_breach_count": int(coarse.breach_count), "fine_breach_count": int(fine.breach_count),
                         "breach_denominator_paths": 120,
                         "coarse_geometric_growth_percent": coarse.annual_geometric_real_growth_percent,
                         "fine_geometric_growth_percent": fine.annual_geometric_real_growth_percent,
                         "coarse_actual_transfer_count": int(coarse.actual_transfer_count),
                         "fine_actual_transfer_count": int(fine.actual_transfer_count),
                         "paired_path_year_decisions": n,
                         "exact_common_visited_states": sum(int(r.exact_common_visited_states) for r in selected),
                         "action_disagreements_on_exact_common_states": sum(int(r.action_disagreements_on_exact_common_states) for r in selected),
                         "action_disagreements_all_paired_path_years": sum(int(r.action_disagreements_all_paired_path_years) for r in selected)})
    result = pd.DataFrame(rows)
    if not np.array_equal(result.coarse_breach_count, result.fine_breach_count):
        raise ValueError("Fine-grid breach counts changed; revise the intended figure annotation")
    return result


def plot_comparison(data, resolution, output):
    main = data.loc[data.main_figure]
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.7))
    fig.subplots_adjust(left=.105, right=.985, bottom=.235, top=.792, hspace=.26, wspace=.20)
    x = np.arange(4)
    colors = [COLORS[p] for p in POLICIES]
    growth_max = max(1.0, np.ceil(main.annual_geometric_real_growth_percent.max() * 1.20))
    growth_min = min(0.0, np.floor(main.annual_geometric_real_growth_percent.min() * 1.20))
    breach_max = max(20, 10 * np.ceil(main.observed_breach_percent.max() * 1.20 / 10))
    fields = ("annual_geometric_real_growth_percent", "observed_breach_percent", "mean_annual_actual_transfer_percent")
    limits = ((growth_min, growth_max), (0, breach_max), (0, 114))
    for column, client in enumerate(CLIENTS):
        panel = main.loc[main.client == client].set_index("policy").loc[list(POLICIES)]
        h, loss = int(panel.horizon_years.iloc[0]), panel.loss_tolerance.iloc[0]
        axes[0, column].set_title(f"Client {client}  |  H = {h} years, L = {loss:.0%}", loc="left", pad=15)
        for row, field in enumerate(fields):
            ax = axes[row, column]
            bars = ax.bar(x, panel[field], color=colors, width=.63, zorder=3)
            for bar, (_, record) in zip(bars, panel.iterrows()):
                height = bar.get_height()
                label = f"{height:.2f}" if row == 0 else (f"{int(record.breach_count)}/120\n{height:.1f}%" if row == 1 else f"{height:.1f}%")
                pad = (limits[row][1] - limits[row][0]) * (.025 if row != 1 else .02)
                ax.text(bar.get_x() + bar.get_width() / 2, height + pad, label,
                        ha="center", va="bottom", fontsize=10.8, linespacing=1.1)
            ax.set_ylim(*limits[row])
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS[p] for p in POLICIES] if row == 2 else [""] * 4, fontsize=11)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            if row == 0:
                ax.yaxis.set_major_locator(MultipleLocator(2 if growth_max > 5 else 1))
            if row == 1:
                ax.set_yticks(np.arange(0, breach_max + 1, 10))
                ax.axhline(10, color=STYLE.TEXT, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
            if row == 2:
                ax.set_yticks([0, 25, 50, 75, 100])
    axes[0, 0].set_ylabel("Annual geometric\nreal growth (%)", labelpad=12)
    axes[1, 0].set_ylabel("Observed MDD > L\nfrequency (%)", labelpad=12)
    axes[2, 0].set_ylabel("Annual decisions with\nactual transfers (%)", labelpad=12)
    handles = [Patch(facecolor=COLORS[p], label={"annual_restore": "Annual restoration", "initial_then_hold": "Initial allocation, then hold",
                                               "myopic_gate": "Myopic growth gate", "finite_horizon_gate": "Finite-horizon growth gate"}[p]) for p in POLICIES]
    handles.append(Line2D([], [], color=STYLE.TEXT, linestyle=(0, (4, 3)), linewidth=1,
                          label="10% original static screen (reference)"))
    fig.text(.105, .966, "Frozen targets, different execution policies", fontsize=17)
    fig.text(.105, .922, "Internal OOF outcomes on 120 development paths per client; primary state mesh = 0.05", fontsize=12)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.098, .896), frameon=False,
               ncol=3, columnspacing=1.5, handlelength=1.6, labelspacing=.65, fontsize=11)
    finite = resolution.loc[resolution.policy == "finite_horizon_gate"]
    differences = int(finite.action_disagreements_all_paired_path_years.sum())
    decisions = int(finite.paired_path_year_decisions.sum())
    footers = [
        "Growth = exp(mean annual net real log growth) - 1. Transfers / H decisions, averaged over paths; initial trades included.",
        "Five frozen 96/24 splits; overlapping training sets. Pooled OOF counts are descriptive, not independent safety certificates.",
        "Gate initial allocation is optional; annual/hold initial allocation is mandatory. Dynamic rules inherit no risk approval.",
        f"Mesh 0.025: same B/C/E breach counts; {differences:,}/{decisions:,} paired finite-H actions differ. This is not an error bound.",
    ]
    for y, text in zip((.158, .115, .072, .029), footers):
        fig.text(.04, y, text, fontsize=11.1)
    STYLE.check_text_bounds(fig)
    name = "01_oof_growth_risk_and_trading"
    fig.savefig(output / f"{name}.png", dpi=240, metadata={"Software": "plot_finite_horizon_execution.py"})
    fig.savefig(output / f"{name}.svg", metadata={"Date": None, "Creator": "plot_finite_horizon_execution.py"})
    plt.close(fig)
    return {"figure": name, "layout": "3 metric rows by 3 client columns; no double axes; all bar baselines are zero",
            "displayed_policies": list(POLICIES), "policy_colors": COLORS,
            "primary_mesh_step": .05, "fine_mesh_check": .025,
            "finite_horizon_mesh_action_differences": differences, "paired_client_path_years": decisions,
            "source_fields": {
                "growth": "client_summary.mean_log_growth; checked as mean(client_path_metrics.log_growth)",
                "risk": "client_summary.breach_count/breach_probability/assigned_paths; checked from executed mdd - client L > 1e-12",
                "trading": "client_summary.mean_trade_fraction; checked as mean(client_path_metrics.trade_fraction)",
                "fine_mesh": "resolution_action_comparison indexed by each fold's frozen assigned target from model_metadata"},
            "units_and_denominators": {
                "growth": "100*expm1(mean_paths(log(net real wealth_H/100)/H)), per-year percent; 120 equally weighted paths",
                "risk": "100*count(MDD-L>1e-12)/120, with count/120 labels; each client's own H and L",
                "trading": "100*sum(actual-positive-transfer decisions)/(120*client H); initial decision included; no review-frequency claim"},
            "data_files": ["01_oof_execution_comparison_data.csv", "01_mesh_resolution_check_data.csv"],
            "axis_limits": {"growth": list(limits[0]), "risk": list(limits[1]), "trading": list(limits[2])},
            "figure_size_inches": [13.4, 10.7], "png_dpi": 240,
            "count_labels_show_all_120_paths": True, "population_CI_or_winner_selected": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/figures/execution")
    args = parser.parse_args()
    script_hash = sha(Path(__file__))
    values, hashes = load_artifacts()
    data, review = chart_data(values)
    resolution = resolution_data(values, data)
    STYLE.configure()
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "01_oof_execution_comparison_data.csv", index=False)
    resolution.to_csv(args.output / "01_mesh_resolution_check_data.csv", index=False)
    chart = plot_comparison(data, resolution, args.output)
    verify_snapshot(values["execution_manifest"])
    if ({relative: sha(ROOT / relative) for relative in hashes} != hashes or sha(Path(__file__)) != script_hash):
        raise RuntimeError("Figure source or result artifacts changed; no completion manifest written")
    outputs = ["01_oof_growth_risk_and_trading.png", "01_oof_growth_risk_and_trading.svg",
               "01_oof_execution_comparison_data.csv", "01_mesh_resolution_check_data.csv"]
    json_write(args.output / "manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Research figure of saved development-only internal OOF execution diagnostics; no policy or target selection",
        "source_artifact_sha256_start": hashes, "source_artifact_sha256_end": hashes,
        "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": script_hash,
        "execution_sources_checked_current": True, "execution_inputs_checked_current": True,
        "all_execution_outputs_checked_against_completion_manifest": True,
        "market_returns_or_final_results_read": False, "executor_or_policy_called": False,
        "plotted_values_recomputed_from_saved_path_metrics": True,
        "review_accounting": review, "chart": chart,
        "limitations": [
            "The fivefold training sets overlap; 120 pooled OOF outcomes are not an independent-population binomial certificate.",
            "Growth, risk and trading frequency are different empirical outcomes; the figure does not select a winner.",
            "Original annual-policy targets and assignments are frozen; dynamic execution inherits no suitability or drawdown approval.",
            "Optional initial gate versus mandatory initial annual/hold allocation is part of the declared intervention.",
            "Finite-state Bellman tables use an empirical stationary annual mixture and interpolated future values, with no continuous-state error bound.",
            "Referrals retain null outcomes; no equal-weight customer fallback or adviser outcome is invented.",
            "All wealth is net of exact initial/later execution fees and actual inflation; MDD is observed annually, including initial100.",
        ],
        "verification": {"numerical_reconstruction": "passed", "text_bounds": "passed",
                         "visual_inspection": "Pending view_image review of rendered PNG"},
        "outputs_sha256": {name: sha(args.output / name) for name in outputs},
    })
    print(json.dumps({"output": str(args.output), "figure": chart["figure"], "path_values_checked": len(data),
                      "finite_mesh_action_differences": chart["finite_horizon_mesh_action_differences"]}))


if __name__ == "__main__":
    main()
