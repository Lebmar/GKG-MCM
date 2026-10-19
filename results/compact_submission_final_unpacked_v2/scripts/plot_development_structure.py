"""Render two research figures from the saved development-structure JSON only.

No CSV, forecast library, policy simulator or final-evaluation output is read.
Run ``.venv/bin/python scripts/plot_development_structure.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("equity", "government_bond", "bill_proxy")
ASSET_LABELS = ("Equities", "Government bonds", "Bill proxy")
BLUE = "#1B5E8C"
GOLD = "#AD7A19"
GRAY = "#57616B"
TEXT = "#242B32"
ASSET_COLORS = (BLUE, GOLD, GRAY)
HORIZONS = (5, 6, 15, 25, 30)


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9AA2AA", "axes.linewidth": .7,
        "grid.color": "#E6E9EC", "grid.linewidth": .7,
        "svg.fonttype": "none", "svg.hashsalt": "development-structure-research-v1",
    })


def check_text_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = list(fig.texts)
    for ax in fig.axes:
        visible += [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        visible += [lab for loc, lab in zip(ax.get_xticks(), ax.get_xticklabels())
                    if min(ax.get_xlim()) <= loc <= max(ax.get_xlim())]
        visible += [lab for loc, lab in zip(ax.get_yticks(), ax.get_yticklabels())
                    if min(ax.get_ylim()) <= loc <= max(ax.get_ylim())]
        if ax.get_legend() is not None:
            visible += list(ax.get_legend().get_texts())
    for legend in fig.legends:
        visible += list(legend.get_texts())
    for item in visible:
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                        box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1):
            raise ValueError(f"clipped text: {item.get_text()!r}")


def save(fig, output, name):
    check_text_bounds(fig)
    fig.savefig(output / f"{name}.png", dpi=240,
                metadata={"Software": "plot_development_structure.py"})
    fig.savefig(output / f"{name}.svg", metadata={"Date": None, "Creator": "plot_development_structure.py"})
    plt.close(fig)


def annual_frequency_and_magnitude(source, output):
    assets = source["market"]["assets"]
    frequency = np.array([assets[a]["real_annual_loss_share_scenario_equal"] for a in ASSETS]) * 100
    annual_count = source["market"]["annual_sample_description"]["annual_rows"]
    counts = np.array([assets[a]["negative_real_annual_count"] for a in ASSETS])
    np.testing.assert_allclose(frequency / 100, counts / annual_count, atol=1e-12, rtol=0)
    quantiles = np.array([[assets[a]["real_return_annual_distribution"][k]
                           for k in ("p05", "median", "p95")] for a in ASSETS]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.subplots_adjust(left=.17, right=.98, bottom=.28, top=.75, wspace=.37)
    positions = np.arange(3)
    axes[0].barh(positions, frequency, height=.43, color=ASSET_COLORS, alpha=.9)
    for y, value, count in zip(positions, frequency, counts):
        axes[0].text(value + 1.1, y, f"{value:.2f}%", va="center", fontsize=12)
    axes[0].set(xlim=(0, 52), xlabel="Observed years with real loss (%)",
                title="(a) Frequency of purchasing-power loss")
    axes[0].xaxis.set_major_locator(MultipleLocator(10))
    for y, (lo, median, hi), color in zip(positions, quantiles, ASSET_COLORS):
        axes[1].errorbar(median, y, xerr=[[median - lo], [hi - median]],
                         color=color, fmt="o", ms=7, elinewidth=2, capsize=6, capthick=1.2)
        axes[1].annotate(f"{lo:.1f}%", (lo, y), xytext=(0, -12),
                         textcoords="offset points", va="top", ha="center", color=color, fontsize=12)
    axes[1].axvline(0, color="#9AA2AA", lw=.9)
    axes[1].set(xlim=(-31, 42), xlabel="Annual real return (%)",
                title="(b) Magnitude: P5-P95 and median")
    axes[1].xaxis.set_major_locator(MultipleLocator(20))
    for ax in axes:
        ax.set_axisbelow(True); ax.grid(axis="x")
        ax.set_yticks(positions, ASSET_LABELS); ax.set_ylim(2.6, -.6)
    fig.text(.17, .94, "Annual loss frequency and return magnitude are different", fontsize=16)
    fig.text(.17, .865, "Development only: 120 synthetic scenarios, 30 annual observations per scenario", fontsize=12)
    fig.text(.04, .145, "Bars describe the 3,600 observed years; years within a scenario are not independent trials.", fontsize=12)
    fig.text(.04, .085, "Intervals are P5-P95 annual outcomes, not confidence bounds; dots mark medians.", fontsize=12)
    fig.text(.04, .025, "A higher loss frequency alone does not establish greater overall investment risk.", fontsize=12)
    save(fig, output, "01_annual_frequency_and_magnitude")
    return {
        "question": "Separate how often real annual returns are negative from the amplitude of their distribution.",
        "source_keys": {
            "frequency": "market.assets.{asset}.real_annual_loss_share_scenario_equal",
            "frequency_counts": "market.assets.{asset}.negative_real_annual_count",
            "magnitude": "market.assets.{asset}.real_return_annual_distribution.{p05,median,p95}",
        },
        "asset_order": list(ASSETS),
        "frequency_percent": frequency.tolist(),
        "negative_year_counts": counts.tolist(),
        "p05_median_p95_percent": quantiles.tolist(),
        "definitions": {
            "real_return": "(1+nominal_asset_return)/(1+inflation)-1, excluding portfolio trading fees",
            "frequency": "average of scenario-specific negative-year fractions; each original scenario has equal weight",
            "magnitude_quantiles": "linear interpolation, matching the saved descriptive annual summaries; median is np.median",
            "sampling": "3600 descriptive annual rows, not3600 independent risk-certification trials",
        },
    }


def horizon_risks(source, output):
    settings = {
        "all_bills": {"label": "All bills", "color": GRAY, "marker": "o", "style": "-"},
        "equal_weight_annual": {"label": "Equal weights, annual rebalance", "color": BLUE, "marker": "s", "style": "--"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=.085, right=.98, bottom=.27, top=.76, wspace=.29)
    values = {}
    handles = []
    for strategy, style in settings.items():
        by_h = source["predeclared_diagnostic_paths"][strategy]["horizons"]
        if any(by_h[str(h)]["n_scenarios"] != 120 for h in HORIZONS):
            raise ValueError("horizon chart requires120 development scenarios at each horizon")
        mdd = np.array([by_h[str(h)]["mdd_p90_inverted_cdf"] for h in HORIZONS]) * 100
        terminal = np.array([by_h[str(h)]["terminal_purchasing_power_loss_share"] for h in HORIZONS]) * 100
        count = [by_h[str(h)]["terminal_purchasing_power_loss_count"] for h in HORIZONS]
        np.testing.assert_allclose(terminal, np.array(count) / 120 * 100, atol=1e-12, rtol=0)
        line, = axes[0].plot(HORIZONS, mdd, color=style["color"], marker=style["marker"],
                             linestyle=style["style"], lw=1.8, ms=6)
        handles.append(line)
        axes[1].plot(HORIZONS, terminal, color=style["color"], marker=style["marker"],
                     linestyle=style["style"], lw=1.8, ms=6)
        values[strategy] = {"mdd_p90_percent": mdd.tolist(), "terminal_loss_percent": terminal.tolist(),
                            "terminal_loss_counts": count}
    for ax in axes:
        ax.set_axisbelow(True); ax.grid(axis="y")
        ax.set_xlim(3.5, 31.5); ax.set_xticks(HORIZONS)
        ax.set_xlabel("Investment horizon H (years)")
        ax.yaxis.set_major_locator(MultipleLocator(10))
    axes[0].set(title="(a) Peak-to-trough path risk", ylabel="Real MDD P90 (%)", ylim=(0, 40))
    axes[1].set(title="(b) Final purchasing-power shortfall", ylabel="Terminal-loss frequency (%)", ylim=(0, 35))
    fig.text(.085, .945, "Path loss and terminal loss answer different questions", fontsize=16)
    fig.legend(handles, [s["label"] for s in settings.values()], ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(.54, .88), columnspacing=2.0, handlelength=2.4)
    fig.text(.04, .145, "Same 120 synthetic scenarios; net real wealth starts at 100. Terminal loss means final wealth <100.", fontsize=12)
    fig.text(.04, .085, "Annual observations; MDD P90 uses the empirical inverse CDF (inverted_cdf).", fontsize=12)
    fig.text(.04, .025, "Longer windows permit deeper prior-peak losses even when final-loss frequency is lower.", fontsize=12)
    save(fig, output, "02_horizon_path_and_terminal_risk")
    return {
        "question": "Compare pathwise maximum drawdown with final purchasing-power shortfall at stated horizons.",
        "source_keys": {
            "mdd": "predeclared_diagnostic_paths.{strategy}.horizons.{H}.mdd_p90_inverted_cdf",
            "terminal_frequency": "predeclared_diagnostic_paths.{strategy}.horizons.{H}.terminal_purchasing_power_loss_share",
            "terminal_counts": "predeclared_diagnostic_paths.{strategy}.horizons.{H}.terminal_purchasing_power_loss_count",
        },
        "horizons": list(HORIZONS), "values": values,
        "definitions": {
            "path_risk": "Across-scenario empirical P90 of each path's maximum relative drawdown in inflation-adjusted net wealth, including initial100",
            "terminal_risk": "Fraction of120 scenarios with real final wealth strictly below initial100",
            "quantile": "inverted_cdf; empirical CDF generalized inverse, not the interpolated P90 also present in the source JSON",
            "portfolio_execution": "all bills or fixed equal weights restored annually; identical contest fee convention; initial allocation charged; no deposits, withdrawals or terminal liquidation",
            "observations": "Annual checks only; no intrayear drawdown observation. Each horizon reuses prefixes of the same120 scenarios, not a new independent sample.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/model_review/development_structure.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/figures/development_structure")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source["scope"]["scenario_ids"] != list(range(1, 121)) or source["scope"]["final_scenarios_loaded"]:
        raise ValueError("only the archived development1--120 diagnostic is permitted")
    configure(); args.output.mkdir(parents=True, exist_ok=True)
    charts = {
        "01_annual_frequency_and_magnitude": annual_frequency_and_magnitude(source, args.output),
        "02_horizon_path_and_terminal_risk": horizon_risks(source, args.output),
    }
    manifest = {
        "source_json": str(args.input.resolve().relative_to(ROOT)),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Development1--120 only; research graphics, not competition-report assets or product selection",
        "no_csv_or_final_results_read": True, "model_reexecuted": False,
        "output_dimensions_inches": [11.5, 4.8], "minimum_font_points": 12, "png_dpi": 240,
        "charts": charts,
        "limitations": [
            "Descriptive outcomes from supplied synthetic development scenarios, not forecasts or real-market guarantees.",
            "Annual rows are dependent within scenarios; no independent-annual binomial certification.",
            "Displayed outcome quantiles are not confidence intervals.",
            "Path MDD uses annual wealth observations; intrayear losses are unavailable.",
            "Horizon prefixes share scenarios; no paired or causal inference is implied by these display lines.",
            "Loss frequency alone does not rank overall investment risk; portfolio suitability and liquidity remain separate modelling questions.",
        ],
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(args.output.iterdir()) if p.suffix in (".png", ".svg")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": list(manifest["outputs"]),
                      "input": manifest["source_json"], "no_model_or_csv_read": True}, indent=2))


if __name__ == "__main__":
    main()
