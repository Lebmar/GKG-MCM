"""Editable reading figure of audited fixed final outcomes; no model execution."""
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = Path("results/final_locked_analysis_v1")
FINAL = Path("results/final_locked_evaluation_v1")
REPLAY = Path("results/final_locked_reproduction_v1")
PLAN = Path("docs/final_evaluation_analysis_plan.json")
OUT = Path("research/figures/locked_final_evaluation")
STEM = "paired_growth_drawdown_and_service_status"
EVENTS = {
    "neither": ("Neither", "#326C9B", "o"),
    "shadow_only": ("Shadow only", "#AD7000", "^"),
    "reference_only": ("B only", "#A33D73", "s"),
    "both": ("Both", "#363D47", "D"),
}
OBJECTS = ["guarded_C_service", "unapproved_fixed_shadow", "same_target_B", "exact_thirds_annual"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def checked_inputs(root):
    """Bind every plotted number to the audited immutable producer and analysis."""
    analysis, final = readj(root / ANALYSIS / "manifest.json"), readj(root / FINAL / "manifest.json")
    audit, replay = readj(root / FINAL / "independent_audit.json"), readj(root / REPLAY / "manifest.json")
    comparison = readj(root / REPLAY / "comparison.json")
    parent = sha(root / FINAL / "manifest.json")
    if (final["status"] != "complete" or audit["status"] != "passed" or audit["parent_manifest_sha256"] != parent
            or analysis["status"] != "complete" or analysis["parent_manifest_sha256"] != parent
            or analysis["analysis_plan_sha256"] != sha(root / PLAN)
            or analysis["sources_and_artifacts_at_start"] != analysis["sources_and_artifacts_at_end"]
            or analysis["no_new_fit_or_simulation"] is not True or analysis["model_or_admission_changed"] is not False
            or replay["status"] != "passed" or replay["differences"] != 0
            or replay["original_final_producer_sha256_end"]["manifest.json"] != parent
            or comparison["status"] != "passed" or comparison["differences"]
            or comparison["manifest_equal_after_declared_normalization"] is not True):
        raise RuntimeError("Passed, mutually bound fixed final audit, analysis and replay required")
    inputs = dict(analysis["sources_and_artifacts_at_start"])
    inputs.update({str(ANALYSIS / name): digest for name, digest in analysis["outputs_sha256"].items()})
    for path in (ANALYSIS / "manifest.json", REPLAY / "manifest.json", REPLAY / "comparison.json"):
        inputs[str(path)] = sha(root / path)
    for name, digest in inputs.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(root / name) != digest:
            raise RuntimeError("Audited figure input changed: " + name)
    return dict(sorted(inputs.items()))


def draft(root=ROOT):
    root = Path(root); out = root / OUT
    if (out / "manifest.json").exists():
        raise RuntimeError("Confirmed research figure may not be overwritten")
    inputs = checked_inputs(root)
    all_pairs = pd.read_csv(root / ANALYSIS / "paired_differences.csv.gz")
    pairs = all_pairs[all_pairs.reference == "same_target_B"].sort_values("scenario_id")
    summary = pd.read_csv(root / ANALYSIS / "official_summary.csv.gz")
    summary = summary[summary.question == "Q2"].set_index("object_id").loc[OBJECTS]
    if (len(pairs) != 60 or pairs.scenario_id.tolist() != list(range(121, 181))
            or pairs.scenario_id.duplicated().any() or not set(pairs.event_pair) <= set(EVENTS)
            or not np.isfinite(pairs[["delta_log_growth", "delta_mdd"]].to_numpy()).all()
            or not (summary.original_n == 60).all() or summary.loc[OBJECTS[0], "financial_n"] != 0
            or not summary.loc[OBJECTS[0], ["median_cagr", "p10_cagr", "p90_mdd", "breach_count"]].isna().all()):
        raise ValueError("All 60 same-ID pairs and the original four service/reference rows are required")
    counts = {event: int((pairs.event_pair == event).sum()) for event in EVENTS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.labelsize": 14,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "axes.titlesize": 15, "svg.fonttype": "none",
        "text.color": "#252B33", "axes.labelcolor": "#252B33", "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(19.2, 11.2), facecolor="white")
    ax = fig.add_axes((.075, .34, .435, .49)); side = fig.add_axes((.555, .31, .425, .54)); side.axis("off")
    fig.text(.075, .95, "Locked final evaluation: paired outcomes and service status", fontsize=21, weight="bold")
    fig.text(.075, .91, "Fixed unapproved shadow, H = 30 years, L = 40%; forecast fan unchanged (IDs 1–120)", fontsize=14)
    ax.set_title("Unapproved shadow − same-target B", loc="left", pad=14, weight="bold")
    ax.axhline(0, color="#515A65", lw=1.1, zorder=1); ax.axvline(0, color="#515A65", lw=1.1, zorder=1)
    for event, (_, color, marker) in EVENTS.items():
        rows = pairs[pairs.event_pair == event]
        ax.scatter(100 * rows.delta_log_growth, 100 * rows.delta_mdd, s=72, marker=marker,
            facecolor="white" if event == "reference_only" else color, edgecolor=color if event == "reference_only" else "white",
            linewidth=1.0, alpha=.9, zorder=3)
    x, y = 100 * pairs.delta_log_growth.to_numpy(), 100 * pairs.delta_mdd.to_numpy()
    for data, setter in ((x, ax.set_xlim), (y, ax.set_ylim)):
        lo, hi = min(0., float(data.min())), max(0., float(data.max())); pad = .09 * (hi - lo)
        setter(lo - pad, hi + pad)
    ax.set_xlabel("Δ annual real log G (percentage points)", labelpad=12)
    ax.set_ylabel("Δ annual-endpoint real MDD (percentage points)", labelpad=11)
    ax.grid(color="#E7EAED", lw=.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#8B939C")
    handles = [Line2D([], [], marker=marker, linestyle="none", markersize=9,
        markerfacecolor="white" if event == "reference_only" else color, markeredgecolor=color,
        label=f"{label} (n={counts[event]})") for event, (label, color, marker) in EVENTS.items()]
    ax.legend(handles=handles, title="MDD > 40% event pair (shadow / B)", ncol=2, frameon=False,
        loc="upper left", bbox_to_anchor=(-.015, -.15), fontsize=12.5, title_fontsize=12.5, columnspacing=1.6)
    side.text(0., .99, "All four fixed Q2 objects · 60 original IDs each", fontsize=15, weight="bold", va="top")
    labels = ["Main C service\nReview", "Shadow\nnot admitted", "Same-target B\nreference", "Exact thirds\nannual reference"]
    table_rows = []
    for label, obj in zip(labels, OBJECTS):
        row = summary.loc[obj]
        table_rows.append([label] + (["NULL"] * 4 if obj == OBJECTS[0] else
            [f"{100 * row.median_cagr:.2f}%", f"{100 * row.p10_cagr:.2f}%", f"{100 * row.p90_mdd:.2f}%", f"{int(row.breach_count)}/60"]))
    table = side.table(cellText=table_rows, colLabels=["Object", "Median\nCAGR", "P10\nCAGR", "P90\nMDD", "Events"],
        cellLoc="center", colWidths=[.31, .18, .17, .18, .16], bbox=(0., .37, 1., .53))
    table.auto_set_font_size(False); table.set_fontsize(12.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D4D9DE"); cell.set_linewidth(.6)
        cell.set_facecolor("#F2F4F6" if row == 0 else "white")
        if row == 0:
            cell.get_text().set_weight("bold")
        if col == 0:
            cell.get_text().set_horizontalalignment("left"); cell.PAD = .055
    side.text(0., .29, "Main C: auto 0 / 60; Review 60 / 60.\nNo main-service financial points are plotted.", fontsize=12.5, va="top", linespacing=1.5)
    shadow, b, thirds = (summary.loc[obj] for obj in OBJECTS[1:])
    side.text(0., .125, "Mean fees (nominal units; initial wealth = 100):\n"
        f"shadow {shadow.mean_nominal_fees:.3f}; B {b.mean_nominal_fees:.3f}; thirds {thirds.mean_nominal_fees:.3f}.",
        fontsize=12.5, va="top", linespacing=1.5)
    side.text(0., -.015, "Mean trade fraction (initial and pending included):\n"
        f"shadow {100*shadow.mean_trade_fraction:.1f}%; B {100*b.mean_trade_fraction:.1f}%; thirds {100*thirds.mean_trade_fraction:.1f}%.",
        fontsize=12.5, va="top", linespacing=1.5)
    notes = [
        "All 60 paired IDs 121–180 are retained, including adverse outcomes. Δ = unapproved shadow − same-target B; main C finance is NULL.",
        "Annual real log G = ln(real terminal / 100) / 30; CAGR = exp(G) − 1. P10 / P90 use the empirical inverse CDF; medians are ordinary medians.",
        "MDD is measured at annual endpoints, not intrayear peaks. Event rule: MDD − 40% > 10⁻¹². Shadow finance does not grant service admission.",
        "Old V1 exposed these final paths. This fixed descriptive evaluation is not newly blind and does not certify future risk. No model or rule was changed.",
    ]
    for index, note in enumerate(notes):
        fig.text(.075, .165 - .033 * index, note, fontsize=12, va="top")
    fig.text(.075, .025, "Sources: passed final independent audit, fixed analysis plan and paired tables, and exact isolated reproduction; hashes in the figure manifest.", fontsize=12)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (STEM + ".png"), dpi=240)
    fig.savefig(out / (STEM + ".svg"), metadata={"Date": None})
    plt.close(fig)
    svg = out / (STEM + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if checked_inputs(root) != inputs:
        raise RuntimeError("Audited input binding changed during plotting")
    source = Path(__file__).resolve().relative_to(root).as_posix()
    writej(out / "draft_manifest.json", dict(status="draft_pending_root_visual_QA", created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={source: sha(root / source)}, input_sha256=inputs,
        outputs_sha256={STEM + ext: sha(out / (STEM + ext)) for ext in (".png", ".svg")},
        plotted_reference="same_target_B", paired_n=60, original_ids=pairs.scenario_id.tolist(), event_counts=counts,
        complete_Q2_object_order=OBJECTS, axes_units="percentage points", x="100 * delta_log_growth (annual real G)", y="100 * delta_mdd",
        figure_inches=[19.2, 11.2], png_dpi=240, minimum_font_points=12, SVG_text_editable=True,
        all_adverse_outcomes_retained=True, main_C_finance=None, policy_or_admission_changed=False,
        new_model_or_market_execution=False, author_visual_QA=False, root_visual_QA=False))
    print(json.dumps(dict(status="draft", png=str(out / (STEM + ".png")), svg=str(svg), paired_n=60, event_counts=counts)))


def confirm(root=ROOT):
    """Use only after the author and root actually view the exported PNG."""
    root = Path(root); out = root / OUT
    if (out / "manifest.json").exists():
        raise RuntimeError("Confirmed research figure may not be overwritten")
    value = readj(out / "draft_manifest.json")
    if checked_inputs(root) != value["input_sha256"]:
        raise RuntimeError("Figure input binding changed before visual confirmation")
    for name, digest in value["source_sha256"].items():
        if sha(root / name) != digest:
            raise RuntimeError("Editable plotting source changed")
    for name, digest in value["outputs_sha256"].items():
        if sha(out / name) != digest:
            raise RuntimeError("Export changed before visual confirmation")
    value.update(status="complete", author_visual_QA=True, root_visual_QA=True, confirmed_utc=datetime.now(timezone.utc).isoformat())
    value["draft_manifest_sha256"] = sha(out / "draft_manifest.json")
    writej(out / "manifest.json", value)
    print(json.dumps(dict(status="complete", manifest_sha256=sha(out / "manifest.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--confirm-visual-qa", action="store_true")
    args = parser.parse_args()
    draft() if args.draft else confirm()
