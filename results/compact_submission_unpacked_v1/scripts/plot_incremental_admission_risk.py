"""Notebook figures from the locked incremental-admission diagnostic only.

No raw market files, simulation, fitting, policy selection, or frozen-artifact
edits. All 24 original newly admitted market IDs remain in the first figure;
the second includes every primary-only breach and the complete 30-year paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from plot_development_structure import configure, check_text_bounds

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/model_review/incremental_admission_risk_v1"
OUT = ROOT / "research/figures/incremental_admission_risk"
PRIMARY, REFERENCE = "joint_tie_set_refit_learner", "annual_then_pending"
BLUE, GOLD, GRAY, RED, TEXT = "#1B5E8C", "#AD7A19", "#8A949D", "#A1433F", "#242B32"
SOURCE = "Source: HKUST(GZ) 2026 MCM synthetic scenarios 1–120 (CC BY-NC-SA 4.0); team 87 saved development diagnostics."


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path): return json.loads(Path(path).read_text())


def save(fig, name):
    check_text_bounds(fig)
    fig.savefig(OUT / (name + ".png"), dpi=240, metadata={"Software": "plot_incremental_admission_risk.py"})
    svg = OUT / (name + ".svg")
    fig.savefig(svg, metadata={"Date": None, "Creator": "plot_incremental_admission_risk.py"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def all_path_pairs(paths, findings):
    p = paths.loc[paths.policy.eq(PRIMARY)].set_index("scenario_id").sort_index()
    b = paths.loc[paths.policy.eq(REFERENCE)].set_index("scenario_id").sort_index()
    if len(p) != 24 or p.index.tolist() != b.index.tolist() or p.index.tolist() != findings["incremental_scenario_ids"]:
        raise ValueError("All predeclared 24 original-ID pairs required")
    if not (p.context.eq("outer_fold_2").all() and b.context.eq("outer_fold_2").all()):
        raise ValueError("New admission must retain its original training-fit context")
    mdd_delta, growth_delta = p.mdd - b.mdd, p.log_growth - b.log_growth
    if not ((mdd_delta > 1e-12).sum() == 24 and (growth_delta > 1e-12).sum() == 18):
        raise ValueError("Locked same-ID direction counts changed")
    plot_data = pd.DataFrame({"scenario_id": p.index,
                              "primary_mdd": p.mdd.values, "reference_mdd": b.mdd.values,
                              "primary_log_growth": p.log_growth.values, "reference_log_growth": b.log_growth.values,
                              "mdd_delta": mdd_delta.values, "log_growth_delta": growth_delta.values})
    plot_data.to_csv(OUT / "all_24_pair_plot_data.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(1, 2, figsize=(15., 12.), sharey=True)
    fig.subplots_adjust(left=.095, right=.97, top=.80, bottom=.17, wspace=.20)
    y = np.arange(24)
    for ax, field in zip(axes, ("mdd", "log_growth")):
        for j, (left, right) in enumerate(zip(b[field], p[field])):
            ax.plot([left, right], [j, j], color=GRAY, linewidth=1.25, alpha=.7, zorder=1)
        ax.scatter(b[field], y, marker="s", s=38, color=GOLD, edgecolor="white", linewidth=.5, label="Annual B (hypothetical)", zorder=3)
        ax.scatter(p[field], y, marker="o", s=42, color=BLUE, edgecolor="white", linewidth=.5, label="Primary learner", zorder=4)
        ax.grid(axis="x"); ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(23.7, -.7)
    axes[0].set_yticks(y, [str(sid) for sid in p.index]); axes[0].set_ylabel("Original scenario ID (same client C)")
    axes[0].axvline(.4, color=RED, linestyle="--", linewidth=1.2)
    axes[0].set_xlim(0, .59); axes[0].set_xticks(np.arange(0, .6, .1))
    axes[0].set_xlabel("Maximum drawdown over the original 30 years")
    axes[0].set_title("(a) Higher MDD on all 24 primary paths", loc="left", pad=15)
    axes[0].text(.402, -.95, "L = 40%", color=RED, fontsize=12, ha="left", va="bottom")
    axes[1].axvline(0, color=GRAY, linestyle=":", linewidth=1.)
    axes[1].set_xlabel("Annualized net real log growth")
    axes[1].set_title("(b) Higher log growth on 18 of 24 paths", loc="left", pad=15)
    lo = min(float(p.log_growth.min()), float(b.log_growth.min()))
    hi = max(float(p.log_growth.max()), float(b.log_growth.max()))
    axes[1].set_xlim(lo - .005, hi + .006)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.08, .885), frameon=False, ncol=2)
    fig.text(.055, .955, "Newly admitted paths: growth gains accompany larger drawdowns", fontsize=18, weight="bold")
    fig.text(.055, .91, "All 24 original market IDs from outer fit 2; one client C, H = 30, L = 40%. Lines connect same-ID outcomes.", fontsize=12)
    fig.text(.055, .105,
             f"Primary minus annual B: mean MDD +{100*mdd_delta.mean():.3f} percentage points; mean log growth +{100*growth_delta.mean():.3f} pp/year.\n"
             "B restores the same target each year until its own breach, then holds pending review; it is a hypothetical reference.\n"
             "Reused development scenarios: paired descriptions, not customer coverage, causal attribution, or a risk guarantee.",
             fontsize=12, va="top", linespacing=1.6)
    fig.text(.055, .025, SOURCE, fontsize=12)
    save(fig, "01_new_admission_all_24_pairs")


def crossing_history(annual, pairs):
    selected = pairs.loc[pairs.reference_policy.eq(REFERENCE) & pairs.primary_breach & ~pairs.reference_breach,
                         "scenario_id"].astype(int).sort_values().tolist()
    if selected != [35, 112]:
        raise ValueError("Every locked primary-only breach must appear")
    data = annual.loc[annual.scenario_id.isin(selected) & annual.policy.isin((PRIMARY, REFERENCE))].copy()
    order = {PRIMARY: 0, REFERENCE: 1}
    data["_policy_order"] = data.policy.map(order)
    data = data.sort_values(["scenario_id", "_policy_order", "year"]).drop(columns="_policy_order")
    if len(data) != 120:
        raise ValueError("Full 30 years for both arms of both IDs required")
    data.to_csv(OUT / "primary_only_full_history_plot_data.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.6), sharex=True, sharey="row")
    fig.subplots_adjust(left=.085, right=.975, bottom=.255, top=.755, hspace=.38, wspace=.19)
    max_wealth = 0.; annotations = []
    for j, sid in enumerate(selected):
        p = data.loc[data.scenario_id.eq(sid) & data.policy.eq(PRIMARY)].sort_values("year")
        b = data.loc[data.scenario_id.eq(sid) & data.policy.eq(REFERENCE)].sort_values("year")
        if p.year.tolist() != list(range(1, 31)) or b.year.tolist() != list(range(1, 31)):
            raise ValueError("Original calendar identity changed")
        cross = p.loc[p.first_crossing_this_year, "year"].tolist()
        if len(cross) != 1:
            raise ValueError("Exactly one first crossing per witness")
        tau = int(cross[0]); pr, br = p.loc[p.year.eq(tau)].iloc[0], b.loc[b.year.eq(tau)].iloc[0]
        if pr.action != "restore" or br.action != "restore":
            raise ValueError("Locked crossing-year action correspondence changed")
        if not np.allclose([pr.end_year_equity, pr.end_year_bond, pr.end_year_bill],
                           [br.end_year_equity, br.end_year_bond, br.end_year_bill], atol=2e-13, rtol=0):
            raise ValueError("Crossing-year end weights must match before making the state comparison")
        for frame, color in ((p, BLUE), (b, GOLD)):
            wealth = np.r_[frame.real_wealth_start.iloc[0], frame.real_wealth_end]
            peak = np.r_[frame.real_wealth_start.iloc[0], frame.real_peak_end]
            if wealth[0] != 100 or peak[0] != 100:
                raise ValueError("Original wealth/peak must start at 100; no path reset")
            years = np.arange(31)
            axes[0, j].plot(years, wealth, color=color, linewidth=2.)
            axes[0, j].plot(years, peak, color=color, linestyle="--", linewidth=1.5)
            axes[1, j].plot(years, np.r_[0., frame.current_drawdown_end], color=color, linewidth=2.)
            axes[1, j].plot(years, np.r_[0., frame.running_mdd_end], color=color, linestyle="--", linewidth=1.5)
            max_wealth = max(max_wealth, float(max(peak)))
        for ax in axes[:, j]:
            ax.axvline(tau, color=RED, linestyle=":", linewidth=1.3)
            ax.set_xlim(0, 30); ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.grid(axis="y"); ax.set_axisbelow(True)
        axes[0, j].set_title(f"ID {sid}: primary first crossing at year {tau}", loc="left", pad=14)
        axes[1, j].axhline(.4, color=RED, linestyle="--", linewidth=1.1)
        axes[1, j].set_ylim(0, .5); axes[1, j].set_yticks([0, .1, .2, .3, .4, .5])
        axes[1, j].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, j].set_xlabel("Original calendar year; full 0–30 horizon")
        annotations.append((sid, tau, 100*pr.end_year_equity, 100*pr.end_year_bill,
                            100*pr.current_drawdown_end, 100*br.current_drawdown_end))
    for ax in axes[0]: ax.set_ylim(0, max_wealth * 1.08)
    axes[0, 0].set_ylabel("Real wealth / historical peak\n(initial = 100)")
    axes[1, 0].set_ylabel("Drawdown\n(current / running maximum)")
    axes[1, 1].text(.6, .405, "L = 40%", va="bottom", color=RED, fontsize=12, ha="left")
    legend = [Line2D([0], [0], color=BLUE, lw=2, label="Primary: wealth / current DD"),
              Line2D([0], [0], color=GOLD, lw=2, label="Annual B: wealth / current DD"),
              Line2D([0], [0], color=BLUE, lw=1.5, linestyle="--", label="Primary: historical peak / running MDD"),
              Line2D([0], [0], color=GOLD, lw=1.5, linestyle="--", label="Annual B: historical peak / running MDD")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.07, .88), frameon=False, ncol=2, columnspacing=2.2)
    fig.text(.055, .96, "History matters even when the crossing-year allocation matches", fontsize=18, weight="bold")
    fig.text(.055, .915, "Every primary-only breach versus annual B: IDs 35 and 112. Both use the same target; no historical peak or MDD is reset.", fontsize=12)
    for x, (sid, tau, equity, bill, pdd, bdd) in zip((.085, .565), annotations):
        fig.text(x, .192, f"ID {sid}, year {tau}: both restore; end weights match.\n"
                 f"Equity {equity:.2f}%, bill {bill:.2f}%; drawdown {pdd:.2f}% vs {bdd:.2f}%.",
                 fontsize=12, va="top", linespacing=1.6)
    fig.text(.055, .105, "Vertical dotted lines mark the primary's first crossing; L = 40% is a pathwise loss tolerance, not an event-probability level.\n"
             "Subsequent pending hold preserves history and can deepen losses. Saved comparisons do not isolate the causal effect of earlier trades.",
             fontsize=12, va="top", linespacing=1.6)
    fig.text(.055, .025, SOURCE, fontsize=12)
    save(fig, "02_primary_only_full_history")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--confirm-visual-review", action="store_true"); args = parser.parse_args()
    manifest = readj(BASE / "manifest.json")
    if readj(BASE / "consistency_checks.json")["status"] != "passed":
        raise ValueError("Saved diagnostic consistency must pass")
    for name, expected in manifest["outputs_sha256"].items():
        if sha(BASE / name) != expected: raise ValueError("Locked diagnostic output changed: " + name)
    inputs = [BASE / name for name in ("manifest.json", "findings.json", "consistency_checks.json", "incremental_paths.csv", "incremental_paired_paths.csv", "incremental_annual_states.csv")]
    inputs.append(ROOT / "docs/incremental_admission_risk_diagnostic_protocol.json")
    source_paths = [Path(__file__).resolve(), ROOT / "scripts/plot_development_structure.py"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    OUT.mkdir(parents=True, exist_ok=True); configure()
    all_path_pairs(pd.read_csv(BASE / "incremental_paths.csv"), readj(BASE / "findings.json"))
    crossing_history(pd.read_csv(BASE / "incremental_annual_states.csv"), pd.read_csv(BASE / "incremental_paired_paths.csv"))
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    if before != after: raise ValueError("Frozen input changed while rendering")
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                  status="complete" if args.confirm_visual_review else "rendered_pending_visual_review",
                  visual_review_confirmed=args.confirm_visual_review,
                  sources=sources, sources_at_end={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
                  input_sha256=before, input_sha256_at_end=after,
                  outputs_sha256={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"},
                  fixed_original_client="C", fixed_original_horizon=30, fixed_loss_tolerance=.4,
                  units="Original market paths for one client; no customer coverage claim",
                  checks=dict(all_24_pairs_visible=True, every_primary_only_breach_visible=True, full_30_year_history=True,
                              crossing_weights_equality_checked=True, fonts_min_points=12, text_bounds_passed=True,
                              source_attribution_visible=True, svg_text_preserved=True, svg_trailing_whitespace_normalized_before_hash=True),
                  raw_market_paths_loaded=False, new_policy_or_fit=False, final_scenarios_loaded=False)
    (OUT / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], manifest_sha256=sha(OUT / "manifest.json"), output=str(OUT.relative_to(ROOT)))))


if __name__ == "__main__": main()
