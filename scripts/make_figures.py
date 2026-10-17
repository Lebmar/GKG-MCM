"""Render paper figures from archived study output, without rerunning policies.

Run from any directory with ``.venv/bin/python scripts/make_figures.py``.
All inputs are read-only. Figure metadata records the input file hashes.
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
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#1B5E8C"
GOLD = "#AD7A19"
GRAY = "#57616B"
LIGHT = "#D9DEE3"
TEXT = "#242B32"
ORDER = ("S0_equal", "S1_annual", "S2_gate", "S3_dynamic", "S2_review", "S3_review")
LABELS = {"S0_equal": "S0 equal", "S1_annual": "S1 annual", "S2_gate": "S2 gate",
          "S3_dynamic": "S3 dynamic", "S2_review": "S2 + review", "S3_review": "S3 + review"}
COLORS = {"S0_equal": GRAY, "S1_annual": "#8D959D", "S2_gate": BLUE,
          "S3_dynamic": GOLD, "S2_review": BLUE, "S3_review": GOLD}


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9AA2AA", "axes.linewidth": .7,
        "grid.color": "#E6E9EC", "grid.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "savefig.transparent": False,
    })


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out, name):
    # A fixed canvas width preserves legible type when inserted at 6.3 inches.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = list(fig.texts)
    for ax in fig.axes:
        visible_text += [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        for loc, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if min(ax.get_xlim()) <= loc <= max(ax.get_xlim()):
                visible_text.append(label)
        for loc, label in zip(ax.get_yticks(), ax.get_yticklabels()):
            if min(ax.get_ylim()) <= loc <= max(ax.get_ylim()):
                visible_text.append(label)
        if ax.get_legend() is not None:
            visible_text += list(ax.get_legend().get_texts())
    for legend in fig.legends:
        visible_text += list(legend.get_texts())
    for item in visible_text:
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                        box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1):
            raise ValueError(f"Figure {name} clips text: {item.get_text()!r}")
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf", metadata={"Title": name, "Creator": "make_figures.py"})
    plt.close(fig)


def _paths_metrics(archive, strategy):
    wealth = archive[f"{strategy}__wealth"]
    inf = archive["inflation"]
    if wealth.shape != (len(inf), inf.shape[1] + 1):
        raise ValueError(f"invalid archived wealth shape for {strategy}")
    price = np.concatenate((np.ones((len(inf), 1)), np.cumprod(1 + inf, axis=1)), axis=1)
    real = wealth / price
    cagr = (real[:, -1] / real[:, 0]) ** (1 / inf.shape[1]) - 1
    mdd = np.max(1 - real / np.maximum.accumulate(real, axis=1), axis=1)
    return real, cagr, mdd


def portfolio_grid(grid, portfolios, out):
    d = grid.loc[grid.horizon == 30].copy()
    if d.empty or d.n_scenarios.nunique() != 1:
        raise ValueError("grid must contain a consistent 30-year development population")
    n = int(d.n_scenarios.iloc[0])
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    fig.subplots_adjust(left=.13, right=.97, bottom=.22, top=.86)
    ax.set_axisbelow(True); ax.grid(True)
    ax.scatter(d.mdd_p90 * 100, d.cagr_median * 100, s=15, color=LIGHT,
               edgecolor="none", zorder=2)
    settings = {
        "Conservative": (GRAY, "s", (-12, 24), "right"),
        "Balanced": (BLUE, "o", (16, -33), "left"),
        "Growth": (GOLD, "^", (-10, -32), "right"),
    }
    for name, weights in portfolios.items():
        match = np.all(np.isclose(d[["w_equity", "w_bond", "w_bill"]], weights), axis=1)
        if match.sum() != 1:
            raise ValueError(f"frozen target {name} is not unique in the saved grid")
        row = d.loc[match].iloc[0]
        color, marker, offset, align = settings[name]
        x, y = row.mdd_p90 * 100, row.cagr_median * 100
        ax.scatter([x], [y], s=74, marker=marker, color=color,
                   edgecolor="white", linewidth=.9, zorder=4)
        ax.annotate(name, xy=(x, y), xytext=offset, textcoords="offset points",
                    ha=align, va="center", color=color, fontsize=12,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .8},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set(xlabel="Real maximum drawdown P90 (%)", ylabel="Median real CAGR (%)",
           xlim=(0, max(70, d.mdd_p90.max() * 105)), ylim=(0, max(7.2, d.cagr_median.max() * 115)))
    ax.xaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_major_locator(MultipleLocator(2))
    fig.text(.13, .94, "Fixed target portfolios before final evaluation", fontsize=13)
    fig.text(.13, .05, f"Development: {n} synthetic scenarios, 30 years; 5-point grid.", fontsize=12)
    _save(fig, out, "01_portfolio_grid")


def final_distributions(archive, summary, selected, out):
    n, years = archive["inflation"].shape
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.1), sharey=True)
    fig.subplots_adjust(left=.29, right=.97, bottom=.26, top=.85, wspace=.25)
    rng = np.random.default_rng(291)
    jitter = rng.uniform(-.11, .11, n)
    metrics = {s: _paths_metrics(archive, s)[1:] for s in ORDER}
    for panel, (ax, metric, title) in enumerate(zip(
            axes, (0, 1), ("(a) Real CAGR (%)", "(b) Real MDD (%)"))):
        ax.set_axisbelow(True); ax.grid(axis="x")
        for position, strategy in enumerate(ORDER):
            values = metrics[strategy][metric] * 100
            q10, q25, q50, q75, q90 = np.quantile(values, [.10, .25, .50, .75, .90])
            stats = {"med": q50, "q1": q25, "q3": q75, "whislo": q10, "whishi": q90}
            color = COLORS[strategy]
            fill = color if strategy in ("S0_equal", "S2_review", "S3_review") else "white"
            ax.scatter(values, np.full(n, position) + jitter, s=7, color=color,
                       alpha=.35, linewidth=0, zorder=2)
            artists = ax.bxp([stats], positions=[position], orientation="horizontal", widths=.48,
                             showfliers=False, patch_artist=True,
                             boxprops={"edgecolor": color, "facecolor": fill, "lw": 1.2},
                             medianprops={"color": TEXT, "lw": 1.5},
                             whiskerprops={"color": color, "lw": 1.1},
                             capprops={"color": color, "lw": 1.1}, zorder=3)
            artists["boxes"][0].set_alpha(.35 if fill != "white" else .88)
            if strategy in ("S3_dynamic", "S3_review"):
                artists["boxes"][0].set_hatch("//")
        ax.set_yticks(range(len(ORDER)), [LABELS[s] for s in ORDER])
        ax.set_ylim(5.65, -.65)
        ax.set_xlabel(title, labelpad=6)
        for tick, strategy in zip(ax.get_yticklabels(), ORDER):
            if strategy == selected:
                tick.set_fontweight("bold"); tick.set_color(BLUE)
        if metric == 0:
            lower = min(0., min(metrics[s][0].min() * 100 for s in ORDER))
            upper = max(metrics[s][0].max() * 100 for s in ORDER)
            ax.set_xlim(np.floor(lower / 2) * 2, np.ceil(upper / 2) * 2)
            ax.xaxis.set_major_locator(MaxNLocator(4))
        else:
            ax.set_xlim(0, min(100, max(70, np.ceil(max(metrics[s][1].max() for s in ORDER) * 10) * 10)))
            ax.xaxis.set_major_locator(MultipleLocator(20))
    fig.text(.13, .935, f"Final outcomes · {n} synthetic scenarios", fontsize=13)
    fig.text(.04, .10, f"{years}-year paths; boxes P25–P75; whiskers P10–P90.", fontsize=12)
    fig.text(.04, .035, "Bold: frozen selected policy. Dots: all scenarios.", fontsize=12)
    _save(fig, out, "02_final_distributions")


def severe_scenario(archive, manifest, selected, out):
    sid = int(manifest["severe_scenario"])
    index = np.flatnonzero(archive["scenario_ids"] == sid)
    if len(index) != 1:
        raise ValueError("severe scenario is absent or duplicated")
    index = int(index[0]); years = archive["inflation"].shape[1]
    strategies = list(dict.fromkeys(("S0_equal", selected, "S3_review")))
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.35), sharex=True,
                            gridspec_kw={"height_ratios": [1.45, 1.0]})
    fig.subplots_adjust(left=.17, right=.97, bottom=.22, top=.79, hspace=.20)
    for ax in axes:
        ax.set_axisbelow(True); ax.grid(axis="y")
        ax.set_xlim(0, years); ax.xaxis.set_major_locator(MultipleLocator(5))
    handles = []
    first_reviews = []
    for strategy in strategies:
        color = COLORS[strategy]; style = "--" if strategy == "S3_review" else "-"
        real, _, _ = _paths_metrics(archive, strategy)
        line, = axes[0].plot(np.arange(years + 1), real[index], color=color,
                             linestyle=style, lw=1.8)
        handles.append(line)
        trades = np.flatnonzero(archive[f"{strategy}__trades"][index])
        axes[0].scatter(trades, real[index, trades], color=color, s=15,
                        marker="o", linewidth=0, zorder=4)
        weights = archive[f"{strategy}__weights"][index, :, 0] * 100
        axes[1].plot(np.arange(years), weights, color=color, linestyle=style, lw=1.6)
        pending = np.flatnonzero(archive[f"{strategy}__pending"][index])
        if len(pending):
            first = int(pending[0]); first_reviews.append(first)
    for first in sorted(set(first_reviews)):
        for ax in axes:
            ax.axvline(first, color="#87919A", linestyle=":", lw=1.0, zorder=1)
    if first_reviews:
        first = min(first_reviews)
        axes[0].annotate(f"Review at t={first}", xy=(first, .97),
                         xycoords=("data", "axes fraction"), xytext=(7, -2),
                         textcoords="offset points", ha="left", va="top", fontsize=12)
    axes[0].axhline(100, color="#ADB4BA", linestyle=":", lw=.8)
    axes[0].set_ylabel("Real wealth\n(initial 100)")
    axes[0].set_ylim(0, None)
    axes[1].set_ylabel("Equity at\ncheck (%)")
    axes[1].set_ylim(0, 60); axes[1].yaxis.set_major_locator(MultipleLocator(20))
    axes[1].set_xlabel("Years elapsed (checks at t=0,…,29)")
    fig.text(.17, .95, f"Severe synthetic scenario {sid}", fontsize=13)
    fig.legend(handles, [LABELS[s] for s in strategies], loc="upper center",
               bbox_to_anchor=(.55, .905), ncol=3, frameon=False,
               handlelength=1.5, columnspacing=.8, handletextpad=.4)
    if first_reviews and len(set(first_reviews)) == 1:
        note = f"Trade circles on wealth; review policies pending from t={first_reviews[0]}."
    else:
        note = "Trade circles on wealth; dotted lines mark review onset."
    fig.text(.04, .035, note, fontsize=12)
    _save(fig, out, "03_severe_scenario")


def cost_sensitivity(data, selected, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3))
    fig.subplots_adjust(left=.20, right=.97, bottom=.26, top=.78, wspace=.65)
    for strategy in ("S0_equal", selected):
        d = data.loc[data.strategy == strategy].sort_values("cost_rate")
        x = d.cost_rate * 100
        axes[0].plot(x, d.cost_mean, color=COLORS[strategy], marker="o", ms=5, lw=1.7)
        axes[1].plot(x, d.trade_fraction * 100, color=COLORS[strategy], marker="o", ms=5, lw=1.7)
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_xlabel("Transfer fee (%)")
        ax.set_xticks([.125, .25, .5], ["0.125", "0.25", "0.50"])
        ax.set_xlim(.10, .525)
    axes[0].set_ylabel("Mean fees\n(wealth units)"); axes[0].set_ylim(0, 2.1)
    axes[1].set_ylabel("Trading checks (%)"); axes[1].set_ylim(0, 105)
    handles = [Line2D([], [], color=COLORS[s], marker="o", lw=1.7) for s in ("S0_equal", selected)]
    fig.legend(handles, [LABELS[s] for s in ("S0_equal", selected)],
               loc="upper center", bbox_to_anchor=(.55, .94), ncol=2, frameon=False)
    fig.text(.06, .035, "Synthetic scenarios (n=60); initial wealth 100; frozen policy.", fontsize=12)
    _save(fig, out, "04_cost_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve(); out = root / "report/figures"; out.mkdir(parents=True, exist_ok=True)
    configure()
    inputs = ["results/frozen/grid.csv", "results/frozen/portfolios.json",
              "results/final/evaluation_manifest.json", "results/final/paths.npz",
              "results/final/summary.csv", "results/final/cost_sensitivity.csv"]
    missing = [f for f in inputs if not (root / f).exists()]
    if missing:
        raise FileNotFoundError("Required archived results are missing: " + ", ".join(missing))
    manifest = _read_json(root / "results/final/evaluation_manifest.json")
    selected = manifest["selected"]
    if selected not in ORDER:
        raise ValueError("unrecognized frozen selected policy")
    with np.load(root / "results/final/paths.npz", allow_pickle=False) as saved:
        archive = {key: saved[key] for key in saved.files}
    if archive["scenario_ids"].tolist() != manifest["scenario_ids"]:
        raise ValueError("scenario IDs do not match final evaluation manifest")
    summary = pd.read_csv(root / "results/final/summary.csv")
    for strategy in ORDER:
        _, cagr, mdd = _paths_metrics(archive, strategy)
        row = summary.loc[summary.strategy == strategy]
        if len(row) != 1:
            raise ValueError(f"summary does not uniquely identify {strategy}")
        np.testing.assert_allclose([np.median(cagr), np.quantile(cagr, .1),
                                    np.median(mdd), np.quantile(mdd, .9)],
                                   row.iloc[0][["cagr_median", "cagr_p10", "mdd_median", "mdd_p90"]].astype(float),
                                   rtol=1e-10, atol=1e-12)
    portfolio_grid(pd.read_csv(root / "results/frozen/grid.csv"),
                   _read_json(root / "results/frozen/portfolios.json"), out)
    final_distributions(archive, summary, selected, out)
    severe_scenario(archive, manifest, selected, out)
    cost_sensitivity(pd.read_csv(root / "results/final/cost_sensitivity.csv"), selected, out)
    receipt = {"input_sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in inputs},
               "selected": selected, "severe_scenario": manifest["severe_scenario"],
               "metric_reconciliation": "all six saved summaries match archived path calculations",
               "output_width_inches": 6.3, "minimum_font_points": 12,
               "png_dpi": 300, "model_reexecuted": False,
               "figures": [p.name for p in sorted(out.glob("*.png"))]}
    (out / "figure_manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
