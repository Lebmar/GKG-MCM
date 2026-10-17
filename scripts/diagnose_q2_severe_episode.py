"""One prespecified severe OOF episode, read from frozen execution artifacts.

No market loader, solver or fit is imported. The path is selected solely by
the original annual-policy client-C OOF MDD, so it cannot rank overall policy
quality. Equal thirds is a different-target reference, clearly separated.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research")]
from safeinvest_research.metrics import path_metrics

PROTOCOL = ROOT / "docs/q2_severe_episode_protocol.json"
OUT = ROOT / "results/model_review/q2_severe_episode_v1"
PRIMARY = ROOT / "results/model_review/risk_menu_v1"
FIXED = ROOT / "results/model_review/fixed_execution_ablation_v1"
FINITE = ROOT / "results/model_review/finite_horizon_execution_v1"
CASE = "alpha_10__risk_only"
COLORS = {"annual_restore": "#17659c", "initial_then_hold": "#b07c15",
          "finite_horizon_gate": "#18816e", "equal_weight_annual": "#7c838c"}
LABELS = {"annual_restore": "Annual restore", "initial_then_hold": "Initial then hold",
          "finite_horizon_gate": "Finite-horizon gate (.05)", "equal_weight_annual": "Equal thirds (reference)"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def input_manifest():
    paths = [PRIMARY / "validation_oof_paths.csv", PRIMARY / "validation_fold_menus.json", PRIMARY / "validation_manifest.json",
             FIXED / "manifest.json", FIXED / "executed_paths.npz", FIXED / "target_path_metrics.csv",
             FINITE / "manifest.json", FINITE / "executed_wealth.npz", FINITE / "action_traces.npz",
             FINITE / "model_metadata.json", FINITE / "target_path_metrics.csv"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def source_manifest():
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "research/safeinvest_research/metrics.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def drawdown_episode(real):
    wealth = np.asarray(real, float)
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    trough = int(np.argmax(drawdown))
    peak_year = int(np.argmax(wealth[:trough + 1]))
    recovered = np.flatnonzero(wealth[trough + 1:] >= wealth[peak_year] - 1e-10)
    recovery_year = int(trough + 1 + recovered[0]) if len(recovered) else None
    return drawdown, {"peak_year": peak_year, "trough_year": trough, "peak_real_wealth": float(wealth[peak_year]),
                      "trough_real_wealth": float(wealth[trough]), "mdd": float(drawdown[trough]),
                      "recovery_year": recovery_year, "recovered_by_horizon": recovery_year is not None,
                      "horizon_year": len(wealth) - 1}


def reconcile_policy(record, summary_row):
    metrics = path_metrics(record["real_wealth"][None, None, :])
    weights = record["end_weights"]
    drift = .5 * np.abs(weights - record["target"]).sum(axis=-1)
    values = {name: float(value[0, 0]) for name, value in metrics.items()}
    values.update({"nominal_fees": float(record["fees"].sum()), "nominal_moved": float(record["moved"].sum()),
                   "trade_fraction": float(record["trades"].mean()), "max_end_year_tv_drift": float(drift.max()),
                   "terminal_equity": float(weights[-1, 0]), "terminal_bond": float(weights[-1, 1]), "terminal_bill": float(weights[-1, 2])})
    errors = {}
    for name, actual in values.items():
        expected = float(summary_row[name])
        if not np.isclose(actual, expected, atol=1e-11, rtol=1e-12):
            raise RuntimeError(f"Saved summary differs from extracted actual path: {name}")
        errors[name] = abs(actual - expected)
    if not np.allclose(record["fees"], .0025 * record["moved"], atol=1e-12, rtol=1e-12):
        raise RuntimeError("Actual fees differ from .0025 times actual transferred amounts")
    if not np.array_equal(record["trades"], record["moved"] > 1e-12):
        raise RuntimeError("Trade years do not match actual positive transferred amounts")
    if not np.allclose(weights.sum(axis=-1), 1., atol=1e-13, rtol=0):
        raise RuntimeError("End weights do not sum to one")
    return values, drift, errors


def plot_episode(records, annual_episode, selected, target, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "q2-severe-episode-v1"})
    fig, axes = plt.subplots(4, 1, figsize=(11.6, 10.4), sharex=True,
                              gridspec_kw={"height_ratios": [2.8, 1.5, 1.5, 1.5]})
    fig.subplots_adjust(left=.09, right=.985, top=.835, bottom=.125, hspace=.25)
    years = np.arange(31)
    handles = []
    markers = {"annual_restore": "o", "initial_then_hold": "s", "finite_horizon_gate": "^", "equal_weight_annual": "D"}
    for policy, record in records.items():
        color = COLORS[policy]; style = "--" if policy == "equal_weight_annual" else "-"
        line, = axes[0].plot(years, record["real_wealth"], color=color, ls=style, lw=2, label=LABELS[policy])
        handles.append(line)
        axes[1].plot(years, record["drawdown"], color=color, ls=style, lw=1.7)
        axes[2].plot(years[1:], record["drift"], color=color, ls=style, lw=1.7)
        traded = record["trades"]
        axes[3].vlines(years[1:][traded], 0, record["fees"][traded], color=color, alpha=.5, lw=.8)
        axes[3].scatter(years[1:][traded], record["fees"][traded], marker=markers[policy], color=color,
                        facecolors="none" if policy == "initial_then_hold" else color, s=30, linewidths=1.1, zorder=3)
    peak, trough = annual_episode["peak_year"], annual_episode["trough_year"]
    for ax in axes:
        ax.axvspan(peak, trough, color=COLORS["annual_restore"], alpha=.055, lw=0)
        ax.axvline(peak, color=COLORS["annual_restore"], ls=":", lw=.9, alpha=.7)
        ax.axvline(trough, color=COLORS["annual_restore"], ls=":", lw=.9, alpha=.7)
        ax.grid(axis="y", color="#e6e8eb", lw=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 30)
    annual = records["annual_restore"]
    axes[0].scatter([peak, trough], annual["real_wealth"][[peak, trough]], color=COLORS["annual_restore"], s=35, zorder=5)
    axes[1].axhline(.40, color="#333333", ls=":", lw=1.)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Net real wealth\n(initial = 100)")
    axes[1].set_ylabel("Annual-end\ndrawdown")
    axes[2].set_ylabel("Drift from own\ntarget (TV)")
    axes[3].set_ylabel("Nominal\ntransfer fee")
    axes[3].set_xlabel("Return year (fees are paid at its start)")
    axes[3].set_ylim(bottom=0)
    axes[3].set_xticks(range(0, 31, 5))
    fig.suptitle(f"Scenario {selected['scenario_id']}, fold {selected['fold']}: a severe annual-policy episode", fontsize=16, y=.985)
    fig.text(.5, .953, f"OOF client C | H = 30 | fixed target (equity, bond, bill) = ({target[0]:.0%}, {target[1]:.0%}, {target[2]:.0%})",
             ha="center", fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .934), ncol=4, frameon=False, columnspacing=1.2)
    recovery = annual_episode["recovery_year"]
    recovery_text = f"recovered in year {recovery}" if recovery is not None else "not recovered by year 30"
    fig.text(.5, .871, f"Annual policy: {annual_episode['mdd']:.1%} MDD | peak year {peak} to trough year {trough} | {recovery_text}",
             ha="center", fontsize=12)
    fig.text(.09, .064, "Selected as annual restore's worst original OOF client-C path; this example cannot rank overall policy quality.", fontsize=12)
    fig.text(.09, .040, "Shading: annual peak-to-trough episode. Dotted loss line: C's 40% tolerance. Equal thirds uses a different target.", fontsize=12)
    fig.text(.09, .016, "Saved execution artifacts only; annual observations, actual transfers and fees. No intrayear price path is inferred.", fontsize=12)
    fig.savefig(path.with_suffix(".png"), dpi=240, facecolor="white", metadata={"Software": "SafeInvest MCM saved-artifact research"})
    fig.savefig(path.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main():
    sources, inputs = source_manifest(), input_manifest()
    original = pd.read_csv(PRIMARY / "validation_oof_paths.csv")
    candidates = original.loc[(original.case == CASE) & (original.client == "C") & original.assigned]
    if len(candidates) != 120 or not set(candidates.scenario_id).issubset(range(1, 121)):
        raise RuntimeError("Require the complete original development OOF client-C cohort")
    selected = candidates.sort_values(["mdd", "scenario_id"], ascending=[False, True], kind="stable").iloc[0]
    scenario, fold_number, candidate = int(selected.scenario_id), int(selected.fold), int(selected.candidate_index)
    context = f"fold_{fold_number}_test"
    fold = next(item for item in json.loads((PRIMARY / "validation_fold_menus.json").read_text())[CASE] if item["fold"] == fold_number)
    if scenario not in fold["test_ids"] or scenario in fold["train_ids"] or fold["assignments"]["C"] != selected.assignment:
        raise RuntimeError("Selected path is not the recorded frozen OOF assignment")
    column = next(i for i, item in enumerate(fold["menu"]) if item["name"] == selected.assignment and item["candidate_index"] == candidate)
    target = np.asarray(fold["menu"][column]["weights"])
    model = next(item for item in json.loads((FINITE / "model_metadata.json").read_text()) if item["context"] == context and item["mesh_step"] == .05)
    if model["scope"] != "oof" or model["train_ids"] != fold["train_ids"] or model["evaluation_ids"] != fold["test_ids"]:
        raise RuntimeError("Finite-horizon model is not the same saved OOF training/evaluation context")
    for folder, files in [(FIXED, ["executed_paths.npz", "target_path_metrics.csv"]),
                          (FINITE, ["executed_wealth.npz", "action_traces.npz", "model_metadata.json", "target_path_metrics.csv"])]:
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["final_scenarios_loaded"] or manifest["scenario_ids"] != list(range(1, 121)):
            raise RuntimeError("Saved execution manifest is not guarded development-only")
        for name in files:
            if sha(folder / name) != manifest["outputs_sha256"][name]: raise RuntimeError("Saved execution artifact hash mismatch")
    records = {}
    with np.load(FIXED / "executed_paths.npz") as archive:
        for policy in ["annual_restore", "initial_then_hold"]:
            prefix = f"{context}__{policy}"
            ids = archive[f"{prefix}__scenario_ids"]
            row = int(np.flatnonzero(ids == scenario)[0])
            np.testing.assert_allclose(archive[f"{prefix}__targets"][column], target, atol=0, rtol=0)
            records[policy] = {field: archive[f"{prefix}__{field}"][row, column].copy()
                               for field in ["real_wealth", "nominal_wealth", "end_weights", "fees", "moved", "trades"]}
            records[policy]["target"] = target.copy()
    row = fold["test_ids"].index(scenario)
    dynamic = f"{context}__mesh_050__H30__finite_horizon_gate"
    with np.load(FINITE / "executed_wealth.npz") as wealth, np.load(FINITE / "action_traces.npz") as actions:
        if actions[f"{dynamic}__scenario_ids"][row] != scenario:
            raise RuntimeError("Finite scenario ordering differs")
        records["finite_horizon_gate"] = {field: wealth[f"{dynamic}__{field}"][row, column].copy()
                                           for field in ["real_wealth", "nominal_wealth", "end_weights"]}
        records["finite_horizon_gate"].update({field: actions[f"{dynamic}__{field}"][row, column].copy()
                                               for field in ["fees", "moved", "trades"]})
        records["finite_horizon_gate"]["target"] = target.copy()
        equal = f"{context}__equal_weight_annual"
        records["equal_weight_annual"] = {field: wealth[f"{equal}__{field}"][row, 0].copy()
                                          for field in ["real_wealth", "nominal_wealth", "end_weights", "fees", "moved", "trades"]}
        records["equal_weight_annual"]["target"] = np.full(3, 1 / 3)
    fixed_summary = pd.read_csv(FIXED / "target_path_metrics.csv")
    prototype_summary = pd.read_csv(FINITE / "target_path_metrics.csv")
    metrics, rows, checks = {}, [], {}
    for policy, record in records.items():
        summary = fixed_summary if policy in ["annual_restore", "initial_then_hold"] else prototype_summary
        match = summary.loc[(summary.context == context) & (summary.scenario_id == scenario) & (summary.horizon == 30) & (summary.policy == policy)]
        match = match.loc[match.candidate_index == candidate] if policy != "equal_weight_annual" else match
        if policy == "finite_horizon_gate": match = match.loc[match.mesh_step == .05]
        if len(match) != 1: raise RuntimeError("Policy summary does not uniquely identify the saved path")
        values, drift, errors = reconcile_policy(record, match.iloc[0])
        drawdown, episode = drawdown_episode(record["real_wealth"])
        record["drift"], record["drawdown"] = drift, drawdown
        values.update({"target": record["target"].tolist(), "reference_only": policy == "equal_weight_annual",
                       "trade_return_years": (np.flatnonzero(record["trades"]) + 1).tolist(),
                       "own_maximum_drawdown_episode": episode})
        metrics[policy], checks[policy] = values, errors
        for year in range(31):
            weights = np.array([0., 0., 1.]) if year == 0 else record["end_weights"][year - 1]
            rows.append({"scenario_id": scenario, "fold": fold_number, "policy": policy, "year": year,
                         "reference_only": policy == "equal_weight_annual", "real_wealth": float(record["real_wealth"][year]),
                         "nominal_wealth": float(record["nominal_wealth"][year]), "drawdown": float(drawdown[year]),
                         "annual_running_peak": float(np.max(record["real_wealth"][:year + 1])),
                         "w_equity": float(weights[0]), "w_bond": float(weights[1]), "w_bill": float(weights[2]),
                         "weight_observation": "initial_preallocation" if year == 0 else "annual_end",
                         "end_year_tv_drift": None if year == 0 else float(drift[year - 1]),
                         "nominal_fee_before_return_year": None if year == 0 else float(record["fees"][year - 1]),
                         "nominal_moved_before_return_year": None if year == 0 else float(record["moved"][year - 1]),
                         "traded_before_return_year": None if year == 0 else bool(record["trades"][year - 1])})
    if not np.isclose(metrics["annual_restore"]["mdd"], selected.mdd, atol=1e-13, rtol=0):
        raise RuntimeError("Extracted annual MDD differs from the prespecified selecting record")
    annual_episode = metrics["annual_restore"]["own_maximum_drawdown_episode"]
    for policy, record in records.items():
        peak, trough = annual_episode["peak_year"], annual_episode["trough_year"]
        metrics[policy]["at_annual_episode_dates"] = {"peak_year_real_wealth": float(record["real_wealth"][peak]),
                                                     "trough_year_real_wealth": float(record["real_wealth"][trough]),
                                                     "trough_year_own_running_peak_drawdown": float(record["drawdown"][trough])}
    if source_manifest() != sources or input_manifest() != inputs: raise RuntimeError("Sources or inputs changed before output")
    OUT.mkdir(parents=True, exist_ok=True)
    selected_info = {"scenario_id": scenario, "fold": fold_number, "client": "C", "case": CASE,
                     "horizon": 30, "loss_tolerance": .40, "assignment": str(selected.assignment), "candidate_index": candidate,
                     "target": target.tolist(), "original_selecting_mdd": float(selected.mdd)}
    write_json(OUT / "episode.json", {"selected": selected_info, "annual_episode": annual_episode,
                                      "policy_metrics": metrics, "summary_reconciliation_abs_errors": checks,
                                      "selection_bias": "Annual policy's worst original OOF client-C path; no overall policy ranking or typical-path inference", 
                                      "timing": "All MDD observations are original annual endpoints. Actual transfer/fee year t precedes its return; no extra fee observation or intrayear market path is inserted."})
    pd.DataFrame(rows).to_csv(OUT / "yearly_execution.csv", index=False)
    plot_episode(records, annual_episode, selected_info, target, OUT / "severe_episode")
    files = ["episode.json", "yearly_execution.csv", "severe_episode.png", "severe_episode.svg"]
    end_sources, end_inputs = source_manifest(), input_manifest()
    if end_sources != sources or end_inputs != inputs: raise RuntimeError("Sources or inputs changed during output writing")
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                                       "sources_at_end": end_sources, "input_sha256": inputs, "input_sha256_at_end": end_inputs,
                                       "outputs_sha256": {file: sha(OUT / file) for file in files}, "protocol": json.loads(PROTOCOL.read_text()),
                                       "market_or_final_data_loaded": False, "model_refit": False,
                                       "figure": {"size_inches": [11.6, 10.4], "png_dpi": 240, "minimum_font_pt": 12,
                                                  "svg_text": True, "panels": ["Net real wealth", "Annual drawdown", "Own-target TV drift", "Actual nominal fees and trade years"]}})
    print(json.dumps({"selected": selected_info, "annual_episode": annual_episode,
                      "policy_metrics": {policy: {key: value[key] for key in ["mdd", "cagr", "nominal_fees", "nominal_moved", "trade_fraction", "max_end_year_tv_drift", "trade_return_years"]}
                                         for policy, value in metrics.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
