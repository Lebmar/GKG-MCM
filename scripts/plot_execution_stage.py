"""Saved-output execution-stage charts; no market parsing, replay, or model fit.

Pending-only views condition on the original annual rule's first crossing and
cannot rank population policy quality.  Censoring remains explicit, never an
invented finite recovery time.  All summaries start from individual OOF paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research")]
from plot_development_structure import configure, check_text_bounds
from safeinvest_research.metrics import NUMERIC_TOL, SUMMARY_METADATA, discrete_cvar

PENDING = ROOT / "results/model_review/review_pending_execution_v1"
FAN = ROOT / "results/model_review/fan_rollout_execution_v1"
PATHWISE = ROOT / "results/model_review/pathwise_fan_execution_v1"
OUT = ROOT / "research/figures/execution_stage"
CLIENTS = ("B", "C", "E")
BLUE, GOLD, GRAY, LIGHT = "#1B5E8C", "#AD7A19", "#57616B", "#D8DDE1"
CLIENT_STYLE = {"B": (BLUE, "o"), "C": (GOLD, "s"), "E": (GRAY, "^")}
ARMS = ("annual", "pending_hold")
ARM_LABEL = {"annual": "Annual restore", "pending_hold": "Then pending hold"}
POLICY_STYLES = {
    "strict_annual": ("Strict annual", BLUE),
    "annual_then_pending": ("Annual then pending", GOLD),
    "growth_only_rollout": ("Growth-only fan", "#B7623D"),
    "pareto_hold_rollout": ("Pareto-hold fan", GRAY),
}
FIVE_POLICY_STYLES = {**POLICY_STYLES, "pathwise_hold_rollout": ("Pathwise-hold fan", "#657A3D")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_manifest():
    return {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__).resolve(), ROOT / "scripts/plot_development_structure.py", ROOT / "research/safeinvest_research/metrics.py"]}


def save(fig, name, out):
    check_text_bounds(fig)
    fig.savefig(out / (name + ".png"), dpi=240, metadata={"Software": "plot_execution_stage.py"})
    fig.savefig(out / (name + ".svg"), metadata={"Date": None, "Creator": "plot_execution_stage.py"})
    plt.close(fig)


def compare_value(actual, expected, field):
    if pd.isna(actual) and pd.isna(expected): return
    if not np.isclose(actual, expected, atol=1e-12, rtol=1e-11):
        raise RuntimeError(f"Saved summary mismatch: {field}: {actual} vs {expected}")


def path_summary(group):
    auto = group.loc[group.assigned]
    result = {"total_n": len(group), "auto_n": len(auto), "review_n": len(group) - len(auto)}
    if not len(auto):
        return {**result, **{name: None for name in ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd", "breach_count", "breach_probability", "actual_trade_fraction", "mean_nominal_fees"]}}
    return {**result, "mean_log_growth": float(auto.log_growth.mean()), "median_cagr": float(auto.cagr.median()),
            "p10_cagr": float(np.quantile(auto.cagr, .1, method="inverted_cdf")), "median_mdd": float(auto.mdd.median()),
            "p90_mdd": float(np.quantile(auto.mdd, .9, method="inverted_cdf")), "cvar90_mdd": float(discrete_cvar(auto.mdd.to_numpy(float))),
            "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": float(auto.breach.astype(bool).mean()),
            "actual_trade_fraction": float(auto.actual_trade_count.sum() / auto.review_periods.sum()), "mean_nominal_fees": float(auto.nominal_fees.mean())}


def load_pending():
    manifest = json.loads((PENDING / "manifest.json").read_text())
    audit = json.loads((PENDING / "independent_audit.json").read_text())
    if manifest["final_scenarios_loaded"] or manifest["scenario_ids"] != list(range(1, 121)) or audit["status"] != "passed":
        raise RuntimeError("Require completed, independently checked, development-only pending artifacts")
    names = ["client_path_metrics.csv", "profile_summary.csv", "paired_path_differences.csv", "paired_profile_summary.csv"]
    for name in names:
        if sha(PENDING / name) != manifest["outputs_sha256"][name]: raise RuntimeError("Pending output SHA mismatch")
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in [PENDING / name for name in [*names, "manifest.json", "independent_audit.json"]]}
    paths = pd.read_csv(PENDING / "client_path_metrics.csv")
    paths = paths.loc[paths.scope == "oof"].copy()
    if len(paths) != 1200: raise RuntimeError("Require all original600 OOF units in both arms")
    for (arm, client), group in paths.groupby(["arm", "client"]):
        if arm not in ARMS or len(group) != 120 or set(group.scenario_id) != set(range(1, 121)) or not group.groupby("outer_fold").size().eq(24).all():
            raise RuntimeError("Duplicate/missing original OOF path or fold")
        if client in CLIENTS:
            if not group.assigned.all(): raise RuntimeError("A served profile lost an assigned path")
        elif group.assigned.any() or not group[["log_growth", "mdd", "breach", "triggered", "first_crossing", "recovery_time"]].isna().all().all():
            raise RuntimeError("Onboarding Review financial outcomes were filled")
    complete_summary = pd.read_csv(PENDING / "profile_summary.csv")
    summary = complete_summary.loc[(complete_summary.scope == "oof") & (complete_summary.subset == "all_assigned")]
    summaries = []
    for (arm, client), group in paths.groupby(["arm", "client"], sort=False):
        result = path_summary(group); old = summary.loc[(summary.arm == arm) & (summary.client == client)].iloc[0]
        for field in ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd", "breach_count", "breach_probability", "actual_trade_fraction", "mean_nominal_fees"]:
            compare_value(result[field], old[field], field)
        summaries.append({"arm": arm, "client": client, **result})
    key = ["context", "outer_fold", "scenario_id", "client"]
    annual = paths.loc[(paths.arm == "annual") & paths.assigned].copy()
    pending = paths.loc[(paths.arm == "pending_hold") & paths.assigned].copy()
    paired = annual.merge(pending, on=key, suffixes=("_annual", "_pending"), validate="one_to_one")
    if len(paired) != 360: raise RuntimeError("All360 automatically assigned OOF pairs are needed")
    for field in ["candidate_local_index", "candidate_original232_index", "horizon", "loss_tolerance", "first_crossing", "triggered", "breach", "available_followup"]:
        left, right = paired[field + "_annual"], paired[field + "_pending"]
        if not ((left == right) | (left.isna() & right.isna())).all(): raise RuntimeError(f"Pending coupling/pair identity differs: {field}")
    triggered = paired.loc[paired.triggered_annual.astype(bool)].copy()
    if not triggered.groupby("client").size().eq(9).all(): raise RuntimeError("Expected fixed nine triggered OOF pairs per B/C/E")
    rows = []
    for row in triggered.itertuples(index=False):
        recover_a, recover_p = bool(row.recovered_annual), bool(row.recovered_pending)
        status = "both_recovered" if recover_a and recover_p else "annual_only_recovered" if recover_a else "pending_only_recovered" if recover_p else "both_censored"
        rows.append({"context": row.context, "outer_fold": int(row.outer_fold), "scenario_id": int(row.scenario_id), "client": row.client,
                     "horizon": int(row.horizon_annual), "loss_tolerance": row.loss_tolerance_annual,
                     "candidate_local_index": int(row.candidate_local_index_annual), "candidate_original232_index": int(row.candidate_original232_index_annual),
                     "first_crossing": int(row.first_crossing_annual), "available_followup": int(row.available_followup_annual),
                     **{f"{arm}_{field}": getattr(row, f"{field}_{'annual' if arm == 'annual' else 'pending'}") for arm in ARMS for field in ["mdd", "added_severity", "log_growth", "cagr", "terminal_real", "nominal_fees", "actual_trade_fraction", "recovery_endpoint", "recovery_time", "recovered", "right_censored", "observed_followup"]},
                     "delta_mdd": row.mdd_pending - row.mdd_annual, "delta_added_severity": row.added_severity_pending - row.added_severity_annual,
                     "delta_annual_log_growth_pp": 100 * (row.log_growth_pending - row.log_growth_annual),
                     "nominal_fees_saved": row.nominal_fees_annual - row.nominal_fees_pending,
                     "paired_recovery_status": status})
    details = pd.DataFrame(rows).sort_values(["client", "scenario_id"]).reset_index(drop=True)
    if not (details.available_followup == details.horizon - details.first_crossing).all(): raise RuntimeError("Pair-specific follow-up does not preserve original H")
    old_pairs = pd.read_csv(PENDING / "paired_profile_summary.csv")
    old_pairs = old_pairs.loc[(old_pairs.scope == "oof") & (old_pairs.subset == "triggered_only")]
    details.loc[details.nominal_fees_saved.abs() <= NUMERIC_TOL, "nominal_fees_saved"] = 0.
    status_order = ["both_recovered", "annual_only_recovered", "pending_only_recovered", "both_censored"]
    severity, recovery_counts = [], []
    for client in CLIENTS:
        group = details.loc[details.client == client]
        check = paired.loc[paired.triggered_annual.astype(bool) & (paired.client == client)].sort_values("scenario_id")
        if not np.array_equal(group.scenario_id, check.scenario_id): raise RuntimeError("Triggered detail rows are not aligned by original scenario")
        for arm in ARMS:
            suffix = "annual" if arm == "annual" else "pending"
            if not np.allclose(group[arm + "_added_severity"], np.maximum(check["mdd_" + suffix] - check["drawdown_at_trigger_" + suffix], 0), atol=1e-13, rtol=1e-11): raise RuntimeError("Added severity definition differs")
            severity.append({"client": client, "arm": arm, "triggered_n": len(group),
                             "mean_added_severity": float(group[arm + "_added_severity"].mean()),
                             "median_added_severity": float(group[arm + "_added_severity"].median()),
                             "mean_mdd": float(group[arm + "_mdd"].mean()), "median_mdd": float(group[arm + "_mdd"].median()),
                             "pending_lower_mdd_n": int((group.delta_mdd < -NUMERIC_TOL).sum()),
                             "equal_mdd_n": int((group.delta_mdd.abs() <= NUMERIC_TOL).sum()),
                             "pending_higher_mdd_n": int((group.delta_mdd > NUMERIC_TOL).sum()),
                             "mean_nominal_fees_saved": float(group.nominal_fees_saved.mean()),
                             "mean_delta_annual_log_growth_pp": float(group.delta_annual_log_growth_pp.mean())})
            original = complete_summary.loc[(complete_summary.scope == "oof") & (complete_summary.client == client) & (complete_summary.arm == arm) & (complete_summary.subset == "triggered_only")].iloc[0]
            compare_value(severity[-1]["mean_added_severity"], original.mean_added_severity_among_triggered, "mean added severity")
            compare_value(int(group[arm + "_recovered"].astype(bool).sum()), original.recovered_n, "recovered count")
            compare_value(int(group[arm + "_right_censored"].astype(bool).sum()), original.right_censored_n, "censored count")
        previous = old_pairs.loc[old_pairs.client == client].iloc[0]
        for name, actual in [("mdd_decreased_n", (group.delta_mdd < -NUMERIC_TOL).sum()), ("mdd_increased_n", (group.delta_mdd > NUMERIC_TOL).sum()), ("mdd_equal_n", (group.delta_mdd.abs() <= NUMERIC_TOL).sum())]: compare_value(actual, previous[name], name)
        compare_value(group.delta_annual_log_growth_pp.mean(), previous.mean_delta_log_growth * 100, "paired annual log growth pp")
        compare_value(-group.nominal_fees_saved.mean(), previous.mean_delta_nominal_fees, "paired nominal fees")
        for status in status_order:
            recovery_counts.append({"client": client, "paired_recovery_status": status, "count": int((group.paired_recovery_status == status).sum()),
                                    "triggered_pair_n": len(group), "minimum_available_followup": int(group.available_followup.min()), "maximum_available_followup": int(group.available_followup.max())})
    return paths, details, pd.DataFrame(severity), pd.DataFrame(recovery_counts), pd.DataFrame(summaries), inputs


def pending_severity(details, severity, out):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.0))
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.20, wspace=.30, hspace=.63)
    low = np.floor(min(details.annual_mdd.min(), details.pending_hold_mdd.min()) * 10) / 10
    high = np.ceil(max(details.annual_mdd.max(), details.pending_hold_mdd.max()) * 10) / 10
    maximum_added = max(severity.mean_added_severity.max(), severity.median_added_severity.max())
    for column, client in enumerate(CLIENTS):
        group = details.loc[details.client == client]; summary = severity.loc[severity.client == client].iloc[0]
        top, lower = axes[0, column], axes[1, column]
        top.plot([low, high], [low, high], color=GRAY, ls="--", lw=1.)
        top.scatter(group.annual_mdd, group.pending_hold_mdd, color=BLUE, edgecolor="white", s=50, linewidth=.6, zorder=3)
        top.set(xlim=(low - .015, high + .015), ylim=(low - .015, high + .015),
                title=f"{client} | H={int(group.horizon.iloc[0])}, L={group.loss_tolerance.iloc[0]:.0%}", xlabel="Annual MDD", ylabel="Pending MDD")
        top.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); top.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        top.xaxis.set_major_locator(MultipleLocator(.1)); top.yaxis.set_major_locator(MultipleLocator(.1))
        top.set_aspect("equal", adjustable="box")
        x = np.array([0, 1]); width = .28
        values_by_arm = {}
        for offset, arm, color in [(-width / 2, "annual", BLUE), (width / 2, "pending_hold", GOLD)]:
            row = severity.loc[(severity.client == client) & (severity.arm == arm)].iloc[0]
            values = [100 * row.mean_added_severity, 100 * row.median_added_severity]
            values_by_arm[arm] = values
            lower.bar(x + offset, values, width=width, color=color)
        for i, location in enumerate(x):
            value_a, value_p = values_by_arm["annual"][i], values_by_arm["pending_hold"][i]
            label = "0 / 0" if max(value_a, value_p) <= NUMERIC_TOL else f"{value_a:.2f} / {value_p:.2f}"
            lower.text(location, max(value_a, value_p) + .2, label, ha="center", va="bottom", fontsize=12)
        lower.set(xticks=x, xticklabels=["Mean", "Median"], xlim=(-.55, 1.55), ylim=(0, maximum_added * 150 + .5), ylabel="Added MDD after trigger (pp)")
        lower.set_title(f"MDD lower / equal / higher: {int(summary.pending_lower_mdd_n)} / {int(summary.equal_mdd_n)} / {int(summary.pending_higher_mdd_n)}", fontsize=12)
        for ax in [top, lower]: ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.text(.075, .965, "Post-crossing hold: paired severity on triggered OOF paths", fontsize=16)
    fig.text(.075, .920, "Nine matched paths per client; dots can overlap. The diagonal means equal full-horizon MDD.", fontsize=12)
    fig.legend([Line2D([0], [0], color=BLUE, lw=7), Line2D([0], [0], color=GOLD, lw=7)], [ARM_LABEL[a] for a in ARMS],
               ncol=2, frameon=False, bbox_to_anchor=(.5, .89), loc="upper center")
    fig.text(.075, .115, "Added severity = final MDD minus MDD at crossing; bar labels show annual / pending in percentage points.", fontsize=12)
    fig.text(.075, .073, "Lower / equal / higher compares pending with annual MDD. All nine crossings per client persist in both arms.", fontsize=12)
    fig.text(.075, .031, "Triggered subgroup only; annual observations and frozen fold-specific targets. These are descriptive outcomes, not CIs.", fontsize=12)
    save(fig, "01_pending_triggered_severity", out)


def pending_growth_recovery(details, counts, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.09, right=.985, top=.75, bottom=.34, wspace=.33)
    for client, (color, marker) in CLIENT_STYLE.items():
        group = details.loc[details.client == client]
        axes[0].scatter(group.nominal_fees_saved, group.delta_annual_log_growth_pp, color=color, marker=marker, edgecolor="white", linewidth=.5, s=55, label=f"{client} (n=9)")
    axes[0].axhline(0, color=GRAY, lw=1)
    axes[0].set(xlabel="Nominal fees saved\n(initial wealth = 100 units)", ylabel="Pending minus annual\nannual log growth (pp/year)", title="(a) Paired fees and growth")
    axes[0].set_xlim(left=-.02)
    profile_handles, profile_labels = axes[0].get_legend_handles_labels()
    fig.legend(profile_handles, profile_labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.31, .855))
    status_styles = [("both_recovered", "Both recovered", BLUE), ("annual_only_recovered", "Annual only recovered", GRAY),
                     ("pending_only_recovered", "Pending only recovered", GOLD), ("both_censored", "Both right-censored", LIGHT)]
    x = np.arange(3); bottom = np.zeros(3)
    for status, label, color in status_styles:
        values = np.array([counts.loc[(counts.client == client) & (counts.paired_recovery_status == status), "count"].iloc[0] for client in CLIENTS])
        axes[1].bar(x, values, bottom=bottom, width=.55, color=color, edgecolor="white", linewidth=.8, label=label)
        for position, value, start in zip(x, values, bottom):
            if value: axes[1].text(position, start + value / 2, str(value), ha="center", va="center", color="white" if color != LIGHT else GRAY, fontsize=12)
        bottom += values
    axes[1].set(xticks=x, xticklabels=CLIENTS, ylim=(0, 10), yticks=[0, 3, 6, 9], ylabel="Matched triggered paths", title="(b) Recovery status at original H")
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.48, -.16), columnspacing=1., handlelength=1.)
    for ax in axes: ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.text(.09, .955, "Pending hold: cost changes and paired recovery status", fontsize=16)
    fig.text(.09, .895, "Same 27 triggered OOF pairs; each pair has the same first crossing and available follow-up H minus tau.", fontsize=12)
    fig.text(.06, .100, "Recovery uses each pair's fixed peak at the trigger. Unrecovered paths are right-censored, not assigned a recovery time.", fontsize=12)
    fig.text(.06, .045, "Nominal fees are money amounts, not return percentage points. No human response or adviser benefit is simulated.", fontsize=12)
    save(fig, "02_pending_growth_cost_recovery", out)


def comparison_figure(summary, out, *, styles=POLICY_STYLES, filename="03_four_strategy_oof_comparison", five=False):
    """Render only when independently checked saved fan results are supplied."""
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 10.0) if five else (11.5, 9.4))
    fig.subplots_adjust(left=.095, right=.98, top=.78 if five else .81, bottom=.20, wspace=.30, hspace=.48)
    specs = [("median_cagr", "(a) Median real CAGR", "Real CAGR"),
             ("p10_cagr", "(b) P10 real CAGR", "Real CAGR"),
             ("p90_mdd", "(c) P90 real MDD", "Annual-endpoint MDD"),
             ("cvar90_mdd", "(d) CVaR90 of real MDD", "Annual-endpoint MDD"),
             ("breach_count", "(e) Paths exceeding individual L", "Breach count / 120"),
             ("actual_trade_fraction", "(f) Actual trading frequency", "Periods with trade")]
    served = summary.loc[summary.client.isin(CLIENTS)]
    growth_values = served[["median_cagr", "p10_cagr"]].to_numpy(float)
    growth_limits = (min(0., float(growth_values.min()) * 1.15), max(.01, float(growth_values.max()) * 1.15))
    risk_max = max(.1, np.ceil(served[["p90_mdd", "cvar90_mdd"]].to_numpy(float).max() * 10) / 10)
    count_max = max(5, int(np.ceil(served.breach_count.max() / 5) * 5))
    x = np.arange(3); width = .7 / len(styles)
    for ax, (field, title, ylabel) in zip(axes.flat, specs):
        for index, (policy, (label, color)) in enumerate(styles.items()):
            values = [summary.loc[(summary.policy == policy) & (summary.client == client), field].iloc[0] for client in CLIENTS]
            positions = x + (index - (len(styles) - 1) / 2) * width
            ax.bar(positions, values, width=width, color=color, label=label,
                   hatch=["", "//", "..", "xx", "||"][index], edgecolor="#39424A", linewidth=.4)
            if field == "breach_count":
                for location, value in zip(positions, values): ax.text(location, value + .25, str(int(value)), ha="center", va="bottom", fontsize=12)
        ax.set(title=title, ylabel=ylabel, xticks=x, xticklabels=["B (15y)", "C (30y)", "E (6y)"])
        if field in ["median_cagr", "p10_cagr"]: ax.set_ylim(*growth_limits)
        elif field in ["p90_mdd", "cvar90_mdd"]: ax.set_ylim(0, risk_max)
        elif field == "breach_count":
            ax.set_ylim(0, count_max * 1.1); ax.yaxis.set_major_locator(MultipleLocator(max(1, count_max // 5)))
        else: ax.set_ylim(0, 1.05)
        if field != "breach_count": ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y"); ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, bbox_to_anchor=(.55, .92), loc="upper center", columnspacing=2.)
    fig.text(.095, .975, ("Five" if five else "Four") + " execution rules on the same frozen strict-positive products", fontsize=16)
    fig.text(.095, .938, "OOF: 120 paths per served profile and strategy; each profile keeps its original horizon and fold-specific target.", fontsize=12)
    fig.text(.06, .125, "Individual L: B=25%, C=40%, E=35%. A and D retain 120 Review rows each per rule, with financial outcomes NULL.", fontsize=12)
    fig.text(.06, .080, "P10/P90 use the empirical inverse CDF; CVaR90 averages the worst 10% mass. Annual observations only.", fontsize=12)
    fig.text(.06, .035, "Reused development paths and overlapping training folds; fan gates are model-relative heuristics, not risk certificates.", fontsize=12)
    save(fig, filename, out)


def load_fan(pending_paths):
    """Normalize saved fan schema, retaining every original Review path."""
    manifest = json.loads((FAN / "manifest.json").read_text())
    if manifest["final_scenarios_loaded"] or manifest["scenario_ids_loaded"] != list(range(1, 121)):
        raise RuntimeError("Require a completed development-only fan experiment")
    names = ["client_paths.csv", "summary.csv"]
    for name in names:
        if sha(FAN / name) != manifest["outputs_sha256"][name]: raise RuntimeError("Fan result hash mismatch")
    files = [FAN / name for name in [*names, "manifest.json"]] + [ROOT / "docs/fan_rollout_execution_protocol.json"]
    if sha(files[-1]) != manifest["sources"]["docs/fan_rollout_execution_protocol.json"]: raise RuntimeError("Fan design protocol changed")
    audit_path = FAN / "independent_audit.json"
    audit_status = "pending"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text()); audit_status = audit["status"]
        if audit_status != "passed": raise RuntimeError("Fan independent audit did not pass")
        files.append(audit_path)
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in files}
    fan = pd.read_csv(FAN / "client_paths.csv"); fan = fan.loc[fan.scope == "outer_oof"].copy()
    if len(fan) != 1800 or set(fan.policy) != set(list(POLICY_STYLES)[1:]): raise RuntimeError("Require all three declared fan policies with600 original OOF units each")
    fan["scope"] = "oof"
    fan["actual_trade_fraction"] = fan.trade_fraction
    fan["review_periods"] = np.where(fan.assigned, fan.horizon, np.nan)
    counts = fan.horizon * fan.trade_fraction
    if not np.allclose(counts[fan.assigned], np.rint(counts[fan.assigned]), atol=1e-11, rtol=0): raise RuntimeError("Fan trading fractions are not actual integer period counts")
    fan["actual_trade_count"] = np.where(fan.assigned, np.rint(counts), np.nan)
    reference = pending_paths.loc[pending_paths.arm == "pending_hold"]
    keys = ["context", "outer_fold", "scenario_id", "client"]
    for policy, group in fan.groupby("policy", sort=False):
        if policy not in list(POLICY_STYLES)[1:] or len(group) != 600: raise RuntimeError("Unexpected fan policy/population")
        merged = group.merge(reference, on=keys, suffixes=("_fan", "_pending"), validate="one_to_one")
        if len(merged) != 600: raise RuntimeError("Fan policy does not use exactly the original OOF context mapping")
        for field in ["assignment", "assigned", "candidate_local_index", "candidate_original232_index", "horizon", "loss_tolerance"]:
            a, b = merged[field + "_fan"], merged[field + "_pending"]
            if not ((a == b) | (a.isna() & b.isna())).all(): raise RuntimeError("Fan changed frozen target/assignment or horizon")
        for client, rows in group.groupby("client"):
            if len(rows) != 120 or set(rows.scenario_id) != set(range(1, 121)) or not rows.groupby("outer_fold").size().eq(24).all(): raise RuntimeError("Fan OOF identity/fold coverage differs")
            if client in CLIENTS:
                if not rows.assigned.all(): raise RuntimeError("A served fan profile lost coverage")
                if not np.allclose(np.expm1(rows.log_growth), rows.cagr, atol=1e-14, rtol=1e-12): raise RuntimeError("Fan CAGR/log-growth relation differs")
                if not np.array_equal(rows.breach.astype(bool), rows.mdd - rows.loss_tolerance > NUMERIC_TOL): raise RuntimeError("Fan breach definition differs")
            elif rows.assigned.any() or not rows[["log_growth", "mdd", "breach", "nominal_fees", "trade_fraction"]].isna().all().all(): raise RuntimeError("Fan filled onboarding Review finances")
        if policy == "annual_then_pending":
            for field in ["log_growth", "cagr", "mdd", "volatility", "terminal_real", "nominal_fees", "nominal_moved", "actual_trade_fraction", "first_crossing", "breach"]:
                a, b = merged[field + "_fan"].to_numpy(float), merged[field + "_pending"].to_numpy(float)
                if not np.allclose(a, b, atol=1e-11, rtol=1e-12, equal_nan=True): raise RuntimeError(f"Fan pending baseline differs from independent paired artifact: {field}")
    strict = pending_paths.loc[pending_paths.arm == "annual"].copy(); strict["policy"] = "strict_annual"
    all_paths = pd.concat([strict, fan], ignore_index=True)
    original_summary = pd.read_csv(FAN / "summary.csv"); original_summary = original_summary.loc[original_summary.scope == "outer_oof"]
    summaries = []
    for (policy, client), group in all_paths.groupby(["policy", "client"], sort=False):
        result = path_summary(group)
        if policy != "strict_annual":
            old = original_summary.loc[(original_summary.policy == policy) & (original_summary.client == client)].iloc[0]
            for field in ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd", "breach_count", "breach_probability", "mean_nominal_fees"]: compare_value(result[field], old[field], "fan " + field)
            compare_value(result["actual_trade_fraction"], old.mean_trade_fraction, "fan actual trade fraction")
        summaries.append({"policy": policy, "client": client, **result})
    baseline = all_paths.loc[all_paths.policy == "strict_annual"]
    pairs = []
    for policy in list(POLICY_STYLES)[1:]:
        joined = all_paths.loc[all_paths.policy == policy].merge(baseline, on=keys, suffixes=("_policy", "_annual"), validate="one_to_one")
        for row in joined.itertuples(index=False):
            record = {key: getattr(row, key) for key in keys}
            record.update({"policy": policy, "assigned": bool(row.assigned_policy)})
            for field in ["log_growth", "cagr", "mdd", "terminal_real", "nominal_fees", "nominal_moved", "actual_trade_fraction", "breach"]:
                a, b = getattr(row, field + "_policy"), getattr(row, field + "_annual")
                record["delta_" + field] = float(a) - float(b) if row.assigned_policy else None
            pairs.append(record)
    return all_paths, pd.DataFrame(summaries), pd.DataFrame(pairs), inputs, audit_status


def pathwise_training_tables(directory, metadata):
    source = pd.read_csv(directory / "training_support_paths.csv")
    saved = pd.read_csv(directory / "training_support_summary.csv")
    properties, outcomes = [], []
    with np.load(directory / "training_traces.npz") as traces:
        for (context, client), group in source.groupby(["context", "client"], sort=False):
            expected_ids = metadata[context]["training_ids"]
            if len(group) != len(expected_ids) or set(group.scenario_id) != set(expected_ids) or not set(group.scenario_id).issubset(range(1, 121)):
                raise RuntimeError("Training support identity/context differs")
            auto = group.loc[group.assigned]; old = saved.loc[(saved.context == context) & (saved.client == client)].iloc[0]
            row = {"context": context, "client": client, "training_support_paths": len(group), "assigned_paths": len(auto), "review_paths": len(group) - len(auto)}
            names = ["minimum_total_log_growth_delta", "mean_total_log_growth_delta", "strictly_negative_computed_growth_delta_count", "negative_beyond_audit_guard_count", "breach_order_violations", "reference_breach_count", "pathwise_breach_count"]
            row.update({name: None for name in names})
            if len(auto):
                delta = auto.pathwise_total_log_growth - auto.reference_total_log_growth
                if not np.allclose(delta, auto.total_log_growth_delta, atol=1e-13, rtol=1e-11): raise RuntimeError("Training total-log delta differs")
                row.update({"minimum_total_log_growth_delta": float(delta.min()), "mean_total_log_growth_delta": float(delta.mean()),
                            "strictly_negative_computed_growth_delta_count": int((delta < 0).sum()), "negative_beyond_audit_guard_count": int((delta < -1e-10).sum()),
                            "breach_order_violations": int((auto.pathwise_breach.astype(bool) & ~auto.reference_breach.astype(bool)).sum()),
                            "reference_breach_count": int(auto.reference_breach.astype(bool).sum()), "pathwise_breach_count": int(auto.pathwise_breach.astype(bool).sum())})
            elif not group[["reference_total_log_growth", "pathwise_total_log_growth", "total_log_growth_delta", "reference_breach", "pathwise_breach"]].isna().all().all():
                raise RuntimeError("Training Review financial outcomes were filled")
            for name in names: compare_value(row[name], old[name], "training " + name)
            properties.append(row)
            for policy, prefix_field in [("annual_then_pending", "reference"), ("pathwise_hold_rollout", "pathwise")]:
                outcome = {"sample": "training_support", "context": context, "client": client, "policy": policy,
                           "total_n": len(group), "auto_n": len(auto), "review_n": len(group) - len(auto),
                           "mean_log_growth": None, "breach_count": None, "breach_probability": None, "actual_trade_fraction": None,
                           "sampling_note": "Context-specific support replay; five96 training contexts overlap, not independent validation"}
                if len(auto):
                    prefix = f"{context}__{client}__{policy}"
                    ids = traces[prefix + "__scenario_ids"]
                    if set(ids) != set(expected_ids): raise RuntimeError("Training trace IDs differ")
                    rows = auto.set_index("scenario_id").loc[ids]
                    real = traces[prefix + "__real_wealth"]
                    H = int(rows.horizon.iloc[0]); total_log = np.log(real[:, -1] / 100.)
                    if not np.allclose(total_log, rows[prefix_field + "_total_log_growth"], atol=1e-13, rtol=1e-11): raise RuntimeError("Training G differs from saved actual wealth")
                    flags = traces[prefix + "__ever_breached"][:, -1]
                    if not np.array_equal(flags, rows[prefix_field + "_breach"].astype(bool)): raise RuntimeError("Training breach flag differs")
                    trades = traces[prefix + "__trades"]
                    if trades.shape != (len(ids), H): raise RuntimeError("Training trading horizon differs")
                    outcome.update({"mean_log_growth": float((total_log / H).mean()), "breach_count": int(flags.sum()),
                                    "breach_probability": float(flags.mean()), "actual_trade_fraction": float(trades.mean())})
                outcomes.append(outcome)
    if len(source) != 3000 or source.groupby("client").size().to_dict() != {name: 600 for name in ["A", "B", "C", "D", "E"]}:
        raise RuntimeError("Training records must retain120 plus five overlapping96 per original profile")
    return source, pd.DataFrame(properties), pd.DataFrame(outcomes)


def load_pathwise(four_paths):
    directory = PATHWISE
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["final_scenarios_loaded"] or manifest["scenario_ids_loaded"] != list(range(1, 121)): raise RuntimeError("Require development-only pathwise artifacts")
    names = ["client_paths.csv", "summary.csv", "training_support_paths.csv", "training_support_summary.csv", "training_traces.npz", "contexts.json"]
    for name in names:
        if sha(directory / name) != manifest["outputs_sha256"][name]: raise RuntimeError("Pathwise result hash mismatch")
    protocol_path = ROOT / "docs/pathwise_fan_execution_protocol.json"
    if sha(protocol_path) != manifest["sources"][str(protocol_path.relative_to(ROOT))]: raise RuntimeError("Pathwise prospective protocol changed")
    files = [directory / name for name in [*names, "manifest.json"]] + [protocol_path]
    audit_path = directory / "independent_audit.json"; audit_status = "pending"
    if audit_path.exists():
        audit_status = json.loads(audit_path.read_text())["status"]
        if audit_status != "passed": raise RuntimeError("Pathwise independent audit did not pass")
        files.append(audit_path)
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in files}
    all_v2 = pd.read_csv(directory / "client_paths.csv")
    if len(all_v2) != 2400 or set(all_v2.policy) != {"annual_then_pending", "pathwise_hold_rollout"}: raise RuntimeError("Incomplete pathwise population")
    original_v1 = pd.read_csv(FAN / "client_paths.csv")
    reference_v2 = all_v2.loc[all_v2.policy == "annual_then_pending"]
    reference_v1 = original_v1.loc[original_v1.policy == "annual_then_pending"]
    keys = ["scope", "context", "outer_fold", "scenario_id", "client"]
    duplicate = reference_v2.merge(reference_v1, on=keys, suffixes=("_v2", "_v1"), validate="one_to_one")
    if len(duplicate) != 1200: raise RuntimeError("V2 repeated baseline loses full or OOF paths")
    for field in ["assigned", "assignment", "candidate_local_index", "candidate_original232_index", "horizon", "loss_tolerance"]:
        a, b = duplicate[field + "_v2"], duplicate[field + "_v1"]
        if not ((a == b) | (a.isna() & b.isna())).all(): raise RuntimeError("V2 repeated baseline changes its frozen assignment or target")
    for field in ["log_growth", "cagr", "mdd", "volatility", "terminal_real", "nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill", "breach", "excess", "first_crossing"]:
        if not np.allclose(duplicate[field + "_v2"].to_numpy(float), duplicate[field + "_v1"].to_numpy(float), atol=1e-11, rtol=1e-12, equal_nan=True):
            raise RuntimeError("V2 repeated full/OOF baseline differs fromV1: " + field)
    new = all_v2.loc[(all_v2.scope == "outer_oof") & (all_v2.policy == "pathwise_hold_rollout")].copy()
    new["scope"] = "oof"; new["actual_trade_fraction"] = new.trade_fraction
    new["review_periods"] = np.where(new.assigned, new.horizon, np.nan)
    counts = new.horizon * new.trade_fraction
    if not np.allclose(counts[new.assigned], np.rint(counts[new.assigned]), atol=1e-11, rtol=0): raise RuntimeError("Pathwise actual trade counts are noninteger")
    new["actual_trade_count"] = np.where(new.assigned, np.rint(counts), np.nan)
    if len(new) != 600: raise RuntimeError("Need600 original profile/path units in the new OOF candidate")
    base = four_paths.loc[four_paths.policy == "strict_annual"]
    oof_keys = ["context", "outer_fold", "scenario_id", "client"]
    check = new.merge(base, on=oof_keys, suffixes=("_new", "_base"), validate="one_to_one")
    if len(check) != 600: raise RuntimeError("New pathwise candidate changed OOF fold mapping")
    for field in ["assigned", "assignment", "candidate_local_index", "candidate_original232_index", "horizon", "loss_tolerance"]:
        a, b = check[field + "_new"], check[field + "_base"]
        if not ((a == b) | (a.isna() & b.isna())).all(): raise RuntimeError("Pathwise candidate changed target/assignment")
    for client, group in new.groupby("client"):
        if len(group) != 120 or set(group.scenario_id) != set(range(1, 121)) or not group.groupby("outer_fold").size().eq(24).all(): raise RuntimeError("Pathwise candidate misses original120 or fold membership")
        if client in CLIENTS:
            if not group.assigned.all() or not np.allclose(np.expm1(group.log_growth), group.cagr, atol=1e-14, rtol=1e-12): raise RuntimeError("Pathwise financial metric consistency failed")
        elif group.assigned.any() or not group[["log_growth", "mdd", "breach", "nominal_fees", "trade_fraction"]].isna().all().all(): raise RuntimeError("Pathwise Review finances filled")
    five_paths = pd.concat([four_paths, new], ignore_index=True)
    summary = []
    original_summary = pd.read_csv(directory / "summary.csv"); original_summary = original_summary.loc[(original_summary.scope == "outer_oof") & (original_summary.policy == "pathwise_hold_rollout")]
    for (policy, client), group in five_paths.groupby(["policy", "client"], sort=False):
        result = path_summary(group)
        if policy == "pathwise_hold_rollout":
            old = original_summary.loc[original_summary.client == client].iloc[0]
            for field in ["mean_log_growth", "median_cagr", "p10_cagr", "median_mdd", "p90_mdd", "cvar90_mdd", "breach_count", "breach_probability", "mean_nominal_fees"]: compare_value(result[field], old[field], "pathwise " + field)
            compare_value(result["actual_trade_fraction"], old.mean_trade_fraction, "pathwise actual trade fraction")
        summary.append({"policy": policy, "client": client, **result})
    paired = []
    for policy in list(FIVE_POLICY_STYLES)[1:]:
        joined = five_paths.loc[five_paths.policy == policy].merge(base, on=oof_keys, suffixes=("_policy", "_annual"), validate="one_to_one")
        for row in joined.itertuples(index=False):
            paired.append({**{key: getattr(row, key) for key in oof_keys}, "policy": policy, "assigned": bool(row.assigned_policy),
                           **{"delta_" + field: float(getattr(row, field + "_policy")) - float(getattr(row, field + "_annual")) if row.assigned_policy else None for field in ["log_growth", "cagr", "mdd", "terminal_real", "nominal_fees", "nominal_moved", "actual_trade_fraction", "breach"]}})
    metadata = json.loads((directory / "contexts.json").read_text())
    support_paths, properties, training_outcomes = pathwise_training_tables(directory, metadata)
    oof_outcomes = []
    for (policy, client), group in five_paths.loc[five_paths.policy.isin(["annual_then_pending", "pathwise_hold_rollout"])].groupby(["policy", "client"], sort=False):
        result = path_summary(group)
        oof_outcomes.append({"sample": "outer_oof", "context": "pooled_unique120", "client": client, "policy": policy,
                             **{key: result[key] for key in ["total_n", "auto_n", "review_n", "mean_log_growth", "breach_count", "breach_probability", "actual_trade_fraction"]},
                             "sampling_note": "One held-out observation per original scenario; different overlapping-fold learned policies, outside the same-fan property"})
    return five_paths, pd.DataFrame(summary), pd.DataFrame(paired), support_paths, properties, pd.concat([training_outcomes, pd.DataFrame(oof_outcomes)], ignore_index=True), inputs, audit_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--include-fan", action="store_true", help="Read only the completed, frozen fan result artifacts and add the four-rule comparison")
    parser.add_argument("--include-pathwise", action="store_true", help="Also read the completed pathwise candidate; keep four-rule assets and add separate five-rule assets")
    args = parser.parse_args(); source_start = source_manifest(); configure(); plt.rcParams["svg.hashsalt"] = "execution-stage-research-v1"
    if args.include_pathwise: args.include_fan = True
    args.out.mkdir(parents=True, exist_ok=True)
    paths, details, severity, counts, all_summary, inputs = load_pending()
    pending_severity(details, severity, args.out); pending_growth_recovery(details, counts, args.out)
    csvs = {"pending_oof_paths.csv": paths, "pending_triggered_pairs.csv": details, "pending_severity_summary.csv": severity,
            "pending_recovery_status_counts.csv": counts, "pending_all_profile_summary.csv": all_summary}
    fan_audit = None
    pathwise_audit = None
    if args.include_fan:
        fan_paths, fan_summary, fan_pairs, fan_inputs, fan_audit = load_fan(paths)
        comparison_figure(fan_summary, args.out)
        csvs.update({"four_strategy_oof_paths.csv": fan_paths, "four_strategy_summary.csv": fan_summary, "four_strategy_paired_vs_annual.csv": fan_pairs})
        inputs.update(fan_inputs)
    if args.include_pathwise:
        five_paths, five_summary, five_pairs, support_paths, support_properties, support_outcomes, new_inputs, pathwise_audit = load_pathwise(fan_paths)
        comparison_figure(five_summary, args.out, styles=FIVE_POLICY_STYLES, filename="04_five_strategy_oof_comparison", five=True)
        csvs.update({"five_strategy_oof_paths.csv": five_paths, "five_strategy_summary.csv": five_summary, "five_strategy_paired_vs_annual.csv": five_pairs,
                     "pathwise_training_support_paths.csv": support_paths, "pathwise_training_context_properties.csv": support_properties, "pathwise_training_and_oof_outcomes.csv": support_outcomes})
        inputs.update(new_inputs)
    for name, frame in csvs.items(): frame.to_csv(args.out / name, index=False)
    artifact_summary = {"purpose": "Descriptive execution-stage figures and chart-ready path summaries; no policy adoption or final conclusion",
                        "pending_triggered_pairs_per_client": {client: int((details.client == client).sum()) for client in CLIENTS},
                        "pending_severity_rows": json.loads(severity.to_json(orient="records", double_precision=15)),
                        "paired_recovery_status_rows": json.loads(counts.to_json(orient="records", double_precision=15)),
                        "four_strategy_rows": json.loads(fan_summary.to_json(orient="records", double_precision=15)) if args.include_fan else None,
                        "five_strategy_rows": json.loads(five_summary.to_json(orient="records", double_precision=15)) if args.include_pathwise else None,
                        "pathwise_training_context_properties": json.loads(support_properties.to_json(orient="records", double_precision=15)) if args.include_pathwise else None,
                        "pathwise_support_note": "Keep six training contexts separate: full120 plus five overlapping96 count600 support identities per profile, not600 independent validation scenarios. The pathwise statement concerns the same fixed training fan, never arbitrary OOF paths." if args.include_pathwise else None,
                        "units": {"cagr_mdd": "CSV fractions, rendered as percent", "added_severity": "CSV fraction; plotted after multiplication by100 as MDD percentage points",
                                  "growth_delta": "100 times annual log-growth difference, percentage points/year", "fees": "Cumulative nominal money amounts, initial account100; not return percentage points",
                                  "recovery": "Paired observed status at original H with same trigger/follow-up. Right-censored recovery times remain unknown and are never filled."}}
    write_json(args.out / "artifact_summary.json", artifact_summary)
    input_end = {name: sha(ROOT / name) for name in inputs}; source_end = source_manifest()
    if input_end != inputs or source_end != source_start: raise RuntimeError("Saved sources/inputs changed during plotting")
    outputs = [*csvs, "artifact_summary.json", *[name + suffix for name in ["01_pending_triggered_severity", "02_pending_growth_cost_recovery"] for suffix in [".png", ".svg"]]]
    if args.include_fan: outputs.extend(["03_four_strategy_oof_comparison.png", "03_four_strategy_oof_comparison.svg"])
    if args.include_pathwise: outputs.extend(["04_five_strategy_oof_comparison.png", "04_five_strategy_oof_comparison.svg"])
    complete = (not args.include_fan or fan_audit == "passed") and (not args.include_pathwise or pathwise_audit == "passed")
    write_json(args.out / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "stage": "five_strategy" if args.include_pathwise else "four_strategy" if args.include_fan else "pending_only", "fan_independent_audit_status": fan_audit, "pathwise_independent_audit_status": pathwise_audit,
        "status": "complete" if complete else "provisional_pending_independent_audit", "input_sha256": inputs, "input_sha256_at_end": input_end,
        "sources_sha256": source_start, "sources_sha256_at_end": source_end,
        "outputs_sha256": {name: sha(args.out / name) for name in outputs}, "market_or_final_data_loaded": False, "model_executed_or_fitted": False,
        "summary_metadata": SUMMARY_METADATA, "checks": {"all120_paths_each_profile_arm": True, "original_five_fold_mapping": True, "summary_recomputed_from_paths": True, "nine_triggered_pairs_per_BCE": True, "same_tau_and_followup_per_pair": True, "text_bounds_checked": True,
                                                            "fan_baseline_matches_pending_pathwise": True if args.include_fan else None, "all_four_rules_preserve600_original_OOF_rows_including_Review": True if args.include_fan else None,
                                                            "v2_duplicate_baseline_matches_v1_full_and_OOF_all_common_metrics": True if args.include_pathwise else None, "all_five_rules_preserve600_original_OOF_rows_including_Review": True if args.include_pathwise else None,
                                                            "training_support_contexts_kept_separate_and_summary_recomputed": True if args.include_pathwise else None},
        "style": {"font_min_pt": 12, "png_dpi": 240, "svg_text": True, "white_background": True,
                  "redundant_encoding": "Profiles use distinct marker shapes; four rules retain fixed left-to-right order and distinct hatches as well as colors; paired-arm bars have direct annual/pending labels."},
        "limitations": ["Triggered OOF subgroup is selected and small (nine per served profile), not a typical path or population effect estimate.",
                        "Annual endpoints only; no intrayear prices or human responses are inferred. Frozen targets and overlapping-fold learned policies are reused.",
                        "Added severity is an increment of historical MDD, not a new loss relative to trigger wealth. Counts lower/equal/higher compare pending minus annual.",
                        "Recovery is scored against the fixed trigger peak and same pair-specific H-tau window. Censored outcomes keep unknown recovery times; no Kaplan-Meier or censoring-independent inference.",
                        "Outcome quantiles are descriptive, not confidence intervals. Everbreach equality applies only to annual versus its post-crossing pending arm, not the pre-crossing fan gates.",
                        "Fan gates use fixed training-only calendar-suffix reference continuations. Replanning the unconditional fan is not a dynamically consistent conditional forecast or a whole-policy risk certificate; model-relative Pareto decisions need not preserve realized event counts.",
                        "All declared rules remain research comparators. These figures select no winner, elicit no new liquidity/risk preference and do not resolve the final service or untouched-final-evaluation requirement.",
                        "The pathwise candidate's finite-support property is only for the identical fixed training fan; its full120 plus overlapping96 supports must not be counted as600 independent scenarios or used to certify OOF risk."]})
    print(f"Wrote execution-stage research charts to {args.out}", flush=True)


if __name__ == "__main__": main()
