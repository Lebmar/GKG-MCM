"""Standalone research figures from saved development artifacts only.

No market CSV, final result, executor, simulator, or policy selector is read or
called. Every plotted value, transformation, source hash and output hash is
recorded in research/figures/risk_menu/manifest.json and chart-ready CSVs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import permutations
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
SOURCE = ROOT / "results/model_review/risk_menu_v1"
CASE = "alpha_10__risk_only"
STYLE_PATH = ROOT / "scripts/plot_development_structure.py"
STYLE_SPEC = importlib.util.spec_from_file_location("development_plot_style", STYLE_PATH)
STYLE = importlib.util.module_from_spec(STYLE_SPEC)
STYLE_SPEC.loader.exec_module(STYLE)

BLUE, GOLD, GRAY = STYLE.BLUE, STYLE.GOLD, STYLE.GRAY
PINK = "#A65378"
TEXT = STYLE.TEXT
ROLE_STYLE = {
    "Conservative": {"color": BLUE, "marker": "o"},
    "Balanced": {"color": GOLD, "marker": "s"},
    "Growth": {"color": PINK, "marker": "^"},
}
ASSETS = ("Equities", "Government bonds", "Bill proxy")
ASSET_COLORS = (BLUE, GOLD, GRAY)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(fig, output, name):
    STYLE.check_text_bounds(fig)
    fig.savefig(output / f"{name}.png", dpi=240, metadata={"Software": "plot_risk_menu_research.py"})
    fig.savefig(output / f"{name}.svg", metadata={"Date": None, "Creator": "plot_risk_menu_research.py"})
    plt.close(fig)


def load_artifacts():
    bootstrap = SOURCE / "bootstrap" / CASE
    paths = {
        "protocol": ROOT / "docs/risk_menu_development_protocol.json",
        "grid": SOURCE / "grid_by_client.csv",
        "full_fit": SOURCE / "full_development_menus.json",
        "fit_manifest": SOURCE / "fit_manifest.json",
        "oof_summary": SOURCE / "validation_oof_summary.csv",
        "validation_manifest": SOURCE / "validation_manifest.json",
        "bootstrap_summary": bootstrap / "summary.json",
        "bootstrap_assignments": bootstrap / "assignments.csv",
        "bootstrap_menus": bootstrap / "menus.json",
        "independent_audit": SOURCE / "independent_audit.json",
        "plot_style": STYLE_PATH,
    }
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths.values()}
    values = {name: pd.read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text())
              for name, path in paths.items() if name != "plot_style"}
    protocol = values["protocol"]
    if protocol["allowed_scenarios"] != list(range(1, 121)) or not protocol["forbidden_new_final_access"]:
        raise RuntimeError("Only the declared development-only research may be plotted")
    if not values["independent_audit"]["passed"]:
        raise RuntimeError("The saved independent numerical audit did not pass")
    for manifest_name in ("fit_manifest", "validation_manifest", "bootstrap_summary"):
        for relative, digest in values[manifest_name]["sources"].items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError(f"Saved {manifest_name} is inconsistent with its source snapshot")
    if values["fit_manifest"]["menus_sha256"] != sha(paths["full_fit"]):
        raise RuntimeError("Full menu artifact does not match its fit manifest")
    return values, hashes


def growth_risk_panels(values, output):
    grid = values["grid"].copy()
    full = values["full_fit"][CASE]
    profiles = values["protocol"]["clients"]
    grid["annual_geometric_growth_percent"] = np.expm1(grid.mean_log_growth) * 100
    grid["exceedance_percent"] = grid.breach_probability * 100
    selected = {product["candidate_index"]: product["name"] for product in full["menu"]}
    grid["main_menu_role"] = grid.candidate_index.map(selected).fillna("")
    grid.to_csv(output / "01_client_growth_risk_data.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.21, top=.82, wspace=.17, hspace=.38)
    for ax, profile in zip(axes.flat, profiles):
        client = profile["name"]
        data = grid.loc[grid.client == client]
        if len(data) != 232 or data.candidate_index.nunique() != 232:
            raise ValueError("Each client panel requires all 232 pre-fixed candidates")
        if data.horizon.nunique() != 1 or int(data.horizon.iloc[0]) != profile["horizon"]:
            raise ValueError("Grid rows do not match the questionnaire horizon")
        ax.scatter(data.exceedance_percent, data.annual_geometric_growth_percent,
                   color="#ADB5BC", s=13, alpha=.5, edgecolors="none", zorder=2)
        ax.axvline(10, color=TEXT, linestyle=(0, (4, 3)), lw=1.15, zorder=1)
        for product in full["menu"]:
            point = data.loc[data.candidate_index == product["candidate_index"]].iloc[0]
            style = ROLE_STYLE[product["name"]]
            ax.scatter(point.exceedance_percent, point.annual_geometric_growth_percent,
                       color=style["color"], marker=style["marker"], s=74,
                       edgecolors="white", linewidths=.8, zorder=4)
            if full["assignments"][client] == product["name"]:
                ax.scatter(point.exceedance_percent, point.annual_geometric_growth_percent,
                           facecolors="none", edgecolors=TEXT, marker="o", s=142, linewidths=1.2, zorder=5)
        ax.set_title(f"Client {client}  |  H={profile['horizon']}y, L={profile['loss_tolerance']:.0%}", loc="left")
        ax.text(.025, .965, f"Full-fit assignment: {full['assignments'][client]}",
                transform=ax.transAxes, va="top", fontsize=10.5)
        ax.set(xlim=(-2, 102), ylim=(0, 8))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_axisbelow(True)
        ax.grid(axis="y")
    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    handles = [Line2D([], [], marker="o", color="#ADB5BC", markersize=5, linestyle="none", label="232 fixed targets")]
    handles += [Line2D([], [], marker=style["marker"], color=style["color"], markersize=8,
                       linestyle="none", label=role) for role, style in ROLE_STYLE.items()]
    handles += [Line2D([], [], marker="o", color=TEXT, markerfacecolor="none", markersize=10,
                       linestyle="none", label="Assigned product"),
                Line2D([], [], color=TEXT, linestyle="--", lw=1.15, label="Empirical budget alpha = 10%")]
    legend_ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=.8, labelspacing=1.1)
    fig.text(.09, .955, "Static candidate growth and client-specific loss frequency", fontsize=16)
    fig.text(.09, .90, "Full-development fit: 120 previously used synthetic scenarios; risk-only policy case", fontsize=12)
    fig.text(.49, .145, "Observed full-path exceedances (%)", ha="center", fontsize=12)
    fig.text(.018, .52, "exp(mean annual log growth) - 1 (%)", rotation=90, va="center", fontsize=12)
    fig.text(.09, .075, "Loss event: annual-observation real MDD exceeds the stated L. Net wealth includes initial and later fees.", fontsize=11)
    fig.text(.09, .025, "Training frequencies are not real-world probabilities or safety certificates; no final scenarios are plotted.", fontsize=11)
    _save(fig, output, "01_client_growth_risk")
    return {
        "question": "Show each client's growth-frequency trade-off and the fixed common menu.",
        "fields": ["client", "horizon", "loss_tolerance", "candidate_index", "mean_log_growth", "breach_probability"],
        "source": "grid_by_client.csv plus full_development_menus.json", "rows": len(grid),
        "x": "100*breach_probability, denominator120 original development paths",
        "y": "100*expm1(mean_log_growth); geometric growth, not mean or median CAGR",
        "alpha": .1, "full_fit_assignments": full["assignments"],
        "main_menu": full["menu"],
        "data_file": "01_client_growth_risk_data.csv",
    }


def risk_and_coverage(values, output):
    full = values["full_fit"][CASE]
    oof = values["oof_summary"].loc[values["oof_summary"]["case"] == CASE]
    bootstrap = values["bootstrap_summary"]
    assignments = values["bootstrap_assignments"]
    n_boot = bootstrap["replicates"]
    if len(assignments) != n_boot or assignments.replicate.nunique() != n_boot:
        raise ValueError("Bootstrap assignment denominator is inconsistent")
    rows = []
    for client in ("B", "C", "E"):
        chosen = [item for item in full["client_product_details"] if item["client"] == client and item["chosen"]]
        held = oof.loc[oof.client == client]
        if len(chosen) != 1 or len(held) != 1:
            raise ValueError("Each breach panel client needs a fixed full-fit choice and one OOF summary")
        held = held.iloc[0]
        rows += [{"client": client, "series": "Full fit", "breaches": chosen[0]["breach_count"], "n": chosen[0]["n"]},
                 {"client": client, "series": "Fold-frozen OOF", "breaches": int(held.breach_count), "n": int(held.assigned_test_paths)}]
    if any(row["n"] != 120 for row in rows):
        raise ValueError("Count bars require the same120-outcome denominator; use rates if that changes")
    breach_frame = pd.DataFrame(rows)
    breach_frame.to_csv(output / "02_breach_count_data.csv", index=False)
    coverage = []
    for profile in values["protocol"]["clients"]:
        client = profile["name"]
        automatic = int(np.count_nonzero(assignments[client] != "Review"))
        frequency = automatic / n_boot
        if not np.isclose(frequency, bootstrap["automatic_frequency"][client], rtol=0, atol=1e-12):
            raise ValueError("Bootstrap frequency disagrees with assignment records")
        coverage.append({"client": client, "automatic_replicates": automatic, "n_replicates": n_boot,
                         "automatic_percent": 100 * frequency})
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(output / "02_bootstrap_coverage_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    fig.subplots_adjust(left=.075, right=.975, bottom=.28, top=.72, wspace=.30)
    x = np.arange(3)
    for offset, series, color in ((-.18, "Full fit", BLUE), (.18, "Fold-frozen OOF", GOLD)):
        data = breach_frame.loc[breach_frame.series == series]
        bars = axes[0].bar(x + offset, data.breaches, width=.31, color=color, label=series)
        for bar, (_, row) in zip(bars, data.iterrows()):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .27,
                         f"{int(row.breaches)}/{int(row.n)}", ha="center", va="bottom", fontsize=11)
    axes[0].axhline(12, color=TEXT, lw=1.05, linestyle=(0, (4, 3)), label="10% of120")
    axes[0].set(xticks=x, xticklabels=["B", "C", "E"], ylim=(0, 15), xlabel="Client",
                ylabel="Observed breach count (paths)")
    axes[0].yaxis.set_major_locator(MultipleLocator(3))
    axes[0].set_title("(a) Training and held-out path counts", loc="left", pad=43)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, ncol=3,
                   handlelength=1.4, columnspacing=1.0, fontsize=10.5)
    positions = np.arange(len(coverage))
    automatic = coverage_frame.automatic_percent.to_numpy()
    axes[1].barh(positions, automatic, height=.52, color=BLUE, label="Automatic")
    axes[1].barh(positions, 100 - automatic, left=automatic, height=.52, color="#E0E4E8", label="Review")
    for y, row in coverage_frame.iterrows():
        p = row.automatic_percent
        label = f"{int(row.automatic_replicates)}/{n_boot} ({p:.1f}%)"
        place = p / 2 if p >= 60 else p + (100 - p) / 2
        axes[1].text(place, y, label, va="center", ha="center", color="white" if p >= 60 else TEXT, fontsize=11)
    axes[1].set(yticks=positions, yticklabels=coverage_frame.client, xlim=(0, 100),
                ylim=(len(coverage) - .4, -.6), xlabel="Reselected menus with automatic assignment (%)")
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_title(f"(b) Automatic coverage across {n_boot:,} resamples", loc="left", pad=43)
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, ncol=2, fontsize=10.5)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax is axes[0] else "x")
    fig.text(.075, .955, "Observed risk calibration and assignment stability", fontsize=16)
    fig.text(.075, .895, "Primary alpha10 / risk-only procedure; complete synthetic scenarios are the sampling unit", fontsize=12)
    fig.text(.075, .16, "OOF uses five fold-frozen menus with overlapping training sets; its pooled counts are not an iid risk certificate.", fontsize=11)
    fig.text(.075, .10, "Referrals remain in coverage denominators and have no fabricated investment outcomes.", fontsize=11)
    fig.text(.075, .04, "Bootstrap frequencies describe empirical reselection stability; they are not confidence intervals or future safety probabilities.", fontsize=11)
    _save(fig, output, "02_risk_and_assignment_stability")
    return {"question": "Separate observed breach counts from stability of automatic-versus-review decisions.",
            "case": CASE, "breach_count_values": rows, "bootstrap_coverage_values": coverage,
            "fields": {"training": "client_product_details.{chosen,breach_count,n}",
                       "OOF": "validation_oof_summary.{breach_count,assigned_test_paths}",
                       "bootstrap": "assignments.csv client columns; checked against summary.automatic_frequency"},
            "data_files": ["02_breach_count_data.csv", "02_bootstrap_coverage_data.csv"]}


def aligned_weight_ranges(values, output):
    base = values["full_fit"][CASE]
    summary = values["bootstrap_summary"]
    reference = np.asarray([product["weights"] for product in base["menu"]])
    aligned = []
    for record in values["bootstrap_menus"]:
        weights = np.asarray(record["weights_by_risk_label"], dtype=float)
        if len(weights) != len(reference):
            continue
        order = min(permutations(range(len(weights))), key=lambda order: np.abs(weights[list(order)] - reference).sum())
        aligned.append(weights[list(order)])
    if not aligned:
        raise ValueError("No bootstrap menu has the reference cardinality")
    quantiles = np.quantile(np.asarray(aligned), [.025, .5, .975], axis=0, method="linear")
    np.testing.assert_allclose(quantiles, summary["aligned_weight_quantiles_025_50_975"], rtol=0, atol=1e-12)
    if len(aligned) != summary["aligned_replicates"]:
        raise ValueError("Aligned menu denominator disagrees with bootstrap summary")
    rows = []
    for product, item in enumerate(base["menu"]):
        for asset, label in enumerate(ASSETS):
            rows.append({"reference_role": item["name"], "asset": label,
                         "p025_weight": quantiles[0, product, asset], "median_weight": quantiles[1, product, asset],
                         "p975_weight": quantiles[2, product, asset], "full_fit_weight": reference[product, asset],
                         "aligned_replicates": len(aligned), "total_replicates": summary["replicates"]})
    pd.DataFrame(rows).to_csv(output / "03_marginal_weight_data.csv", index=False)
    fig, axes = plt.subplots(1, len(reference), figsize=(12.8, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.98, bottom=.32, top=.68, wspace=.28)
    for product, ax in enumerate(np.atleast_1d(axes)):
        for asset, color in enumerate(ASSET_COLORS):
            lo, median, hi = quantiles[:, product, asset] * 100
            ax.errorbar(median, asset, xerr=[[median - lo], [hi - median]], color=color,
                        fmt="o", ms=6.5, elinewidth=2.0, capsize=4, capthick=1.2)
            ax.plot(reference[product, asset] * 100, asset - .13, marker="D", markersize=6.5,
                    markerfacecolor="white", markeredgecolor=TEXT, linestyle="none")
        ax.set_title(base["menu"][product]["name"], pad=12)
        ax.set(xlim=(-2, 102), ylim=(2.55, -.55), yticks=np.arange(3), yticklabels=ASSETS,
               xlabel="Asset weight (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_axisbelow(True)
        ax.grid(axis="x")
    handles = [Line2D([], [], color=GRAY, marker="o", lw=2, markersize=6,
                      label="Marginal P2.5-P97.5 and median"),
               Line2D([], [], color=TEXT, marker="D", markerfacecolor="white", linestyle="none",
                      markersize=6.5, label="Full-development target")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .835), ncol=2,
               frameon=False, handlelength=2.3, columnspacing=2.1, fontsize=11.5)
    fig.text(.075, .955, "Marginal target-weight variation after menu alignment", fontsize=16)
    fig.text(.075, .895, f"{len(aligned):,}/{summary['replicates']:,} whole-scenario resamples aligned to the full-fit menu by minimum total L1 distance", fontsize=12)
    fig.text(.075, .205, "Coordinates are summarized separately: quantile weights need not sum to100% or form an observed portfolio.", fontsize=11)
    fig.text(.075, .145, "These are marginal empirical reselection quantiles, not confidence intervals or market outcome ranges.", fontsize=11)
    fig.text(.075, .085, "Alignment conditions on the same menu cardinality; reference risk labels do not impose a universal risk ranking.", fontsize=11)
    _save(fig, output, "03_aligned_weight_variation")
    return {"question": "Describe parameter variability separately from automatic coverage stability.",
            "case": CASE, "alignment": summary["weight_alignment"],
            "aligned_replicates": len(aligned), "total_replicates": summary["replicates"],
            "quantile_method": "linear coordinate quantiles after reperforming minimum-L1 permutation alignment",
            "fields": ["menus.json.weights_by_risk_label", "full_development_menus.json.menu.weights"],
            "values": rows, "data_file": "03_marginal_weight_data.csv",
            "limitation": "Separate coordinate quantiles are not necessarily feasible jointly; no confidence interpretation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/figures/risk_menu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    STYLE.configure()
    values, input_hashes = load_artifacts()
    figures = {
        "01_client_growth_risk": growth_risk_panels(values, args.output),
        "02_risk_and_assignment_stability": risk_and_coverage(values, args.output),
        "03_aligned_weight_variation": aligned_weight_ranges(values, args.output),
    }
    if any(sha(ROOT / relative) != digest for relative, digest in input_hashes.items()):
        raise RuntimeError("A plotted source changed during rendering")
    outputs = {path.name: sha(path) for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "manifest.json"}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha(Path(__file__)), "input_artifact_sha256": input_hashes,
                "output_artifact_sha256": outputs, "case": CASE,
                "scope": "Saved 1--120 synthetic development artifacts only; no raw market or final-result access.",
                "figures": figures,
                "verification": {"all_text_bounds_checked": True, "bootstrap_counts_recomputed": True,
                                 "weight_alignment_and_quantiles_recomputed": True,
                                 "visual_inspection": "Pending visual review of the rendered PNGs."}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"figures": list(figures), "output": str(args.output), "market_data_loaded": False}))


if __name__ == "__main__":
    main()
