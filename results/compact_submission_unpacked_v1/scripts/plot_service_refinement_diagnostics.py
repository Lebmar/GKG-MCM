"""Explain saved refit ledgers and fixed-state checks; no market model is run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_development_structure import configure, check_text_bounds

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/model_review"
OUT = ROOT / "research/figures/service_refinement"
BLUE, GOLD, GRAY, TEXT = "#1B5E8C", "#AD7A19", "#57616B", "#242B32"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    check_text_bounds(fig)
    fig.savefig(OUT / (name + ".png"), dpi=240, metadata={"Software": "plot_service_refinement_diagnostics.py"})
    svg = OUT / (name + ".svg")
    fig.savefig(svg, metadata={"Date": None, "Creator": "plot_service_refinement_diagnostics.py"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def refit_plot(mechanisms, selections, beta_summary, inner):
    rows = []
    for context, item in selections.items():
        beta = item["beta_selection"]["selected_beta"]
        group = mechanisms.loc[mechanisms.context.eq(context) & mechanisms.stage.eq("refit")]
        row = group.iloc[0] if beta is None else group.loc[np.isclose(group.beta, beta, atol=0, rtol=0)].iloc[0]
        rows.append({"context": context, "n": int(row["n"]), "beta": beta,
            "cap": None if beta is None else int(row.effective_cap), "minimum_count": int(row.minimum_candidate_count),
            "status": item["execution_status"]})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.3))
    fig.subplots_adjust(left=.18, right=.98, bottom=.25, top=.73, wspace=.55)
    y = np.arange(len(rows)); labels = []
    for i, row in enumerate(rows):
        label = "Full fit (descriptive)" if i == 0 else f"Outer training fit {i}"
        labels.append(f"{label}\nn = {row['n']}; beta = {row['beta'] if row['beta'] is not None else 'NULL'}")
        if row["cap"] is not None:
            axes[0].barh(i, row["cap"], height=.53, color=GOLD, alpha=.45)
            axes[0].text(row["cap"] + .12, i + .19, f"cap {row['cap']}", va="center", fontsize=12, color=GOLD)
        else:
            axes[0].text(.15, i + .17, "No inner-eligible beta", va="center", fontsize=12, color=GRAY)
        axes[0].plot(row["minimum_count"], i - .1, "o", color=BLUE, markersize=8)
        axes[0].text(row["minimum_count"] + .16, i - .16, f"min {row['minimum_count']}", va="center", fontsize=12, color=BLUE)
    axes[0].set_yticks(y, labels); axes[0].set_ylim(len(rows) - .5, -.6)
    axes[0].set_xlim(0, max(x["minimum_count"] for x in rows) + 2.7)
    axes[0].set_xlabel("Whole-H training breach count (integer)")
    axes[0].set_title("(a) Final refit: cap versus nine-price minimum", loc="left", pad=14)
    axes[0].grid(axis="x"); axes[0].set_axisbelow(True)
    axes[0].axhline(.5, color=GRAY, linestyle=":", linewidth=.8)
    data = inner.loc[inner.context.eq("outer_fold_4")]
    low = data.loc[np.isclose(data.training_beta, .05, atol=0, rtol=0)].set_index("scenario_id")
    high = data.loc[np.isclose(data.training_beta, .075, atol=0, rtol=0)].set_index("scenario_id")
    common = low.index[low.assigned]; added = high.index[high.assigned & ~low.assigned]
    counts = [int(low.loc[common].breach.astype(bool).sum()), int(high.loc[added].breach.astype(bool).sum())]
    ns = [len(common), len(added)]
    if (counts, ns) != ([7, 5], [72, 24]): raise AssertionError("Saved fold4 grouping changed")
    vals = np.array(counts) / ns
    axes[1].bar([0, 1], vals, width=.58, color=[BLUE, GOLD])
    for i, (v, k, n) in enumerate(zip(vals, counts, ns)):
        axes[1].text(i, v + .009, f"{k}/{n}", ha="center", fontsize=12)
    axes[1].set_xticks([0, 1], ["Three admitted\ninner fits", "One additional\ninner fit"])
    axes[1].set_ylabel("Observed events / validation market paths")
    axes[1].set_ylim(0, .29); axes[1].set_yticks([0, .1, .2], ["0%", "10%", "20%"])
    axes[1].axhline(.1, color=GRAY, linestyle="--", linewidth=.8)
    axes[1].set_title("(b) Outer fit 4: changing admission", loc="left", pad=14)
    axes[1].text(.5, .26, "Higher-beta union: 12/96 = 12.5%", ha="center", fontsize=12)
    axes[1].grid(axis="y"); axes[1].set_axisbelow(True)
    fig.text(.06, .95, "Refit admission fails through distinct mechanisms", fontsize=18, weight="bold")
    fig.text(.06, .895, "Client C: H = 30, L = 40%. All five outer fits computed successfully; three abstained from automatic execution.", fontsize=12)
    fig.text(.06, .84, "Fit 2 has inner-equivalent beta values; fit 4 changes which inner fits are admitted. These are not the same failure.", fontsize=12)
    fig.text(.06, .16, "At selected beta = .05, the unrounded refit allowance is 4.8; minima 6 and 7 still exceed it. Rounding alone is insufficient.\n"
        "A same-name price can change target, training fan and endogenous state. Count ledgers do not identify separate causal effects.\n"
        "Source: HKUST(GZ) 2026 MCM SafeInvest synthetic scenarios 1–120 (CC BY-NC-SA 4.0); saved team 87 development results.",
        fontsize=12, va="top", linespacing=1.6)
    save(fig, "01_refit_admission_mechanisms")
    pd.DataFrame(rows).to_csv(OUT / "refit_cap_plot_data.csv", index=False)


def forecast_plot(comparisons, summary):
    outer = comparisons.loc[comparisons.analysis_scope.eq("outer_oof")].copy()
    if len(outer) != 6 or outer.scenario_id.nunique() != 2: raise AssertionError("Fixed outer inventory changed")
    agg = summary.loc[summary.level.eq("analysis_scope")].set_index("analysis_scope")
    scopes = ["full_development_descriptive", "outer_oof", "inner_validation"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.0))
    fig.subplots_adjust(left=.08, right=.98, bottom=.31, top=.73, wspace=.34)
    x = np.arange(6); width = .36
    axes[0].bar(x-width/2, 100*outer.predicted_delta_log, width, label="Training fan prediction", color=BLUE)
    axes[0].bar(x+width/2, 100*outer.realized_suffix_delta_log, width, label="One realized suffix", color=GOLD)
    axes[0].axhline(0, color=TEXT, linewidth=.8)
    labels = [f"Fit {1 if 'c1__' in r.group else 5}\nID {r.scenario_id}\nYear {r.year}" for r in outer.itertuples()]
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("Restore minus hold: total net real log growth (pp)")
    axes[0].set_title("(a) Six fixed outer states; two scenario IDs", loc="left", pad=14)
    axes[0].legend(loc="upper left", frameon=False); axes[0].grid(axis="y"); axes[0].set_axisbelow(True)
    nums = [int(agg.loc[s].active_state_n) for s in scopes]
    for j, (field, label, color) in enumerate([("growth_sign_agree_n", "Growth-direction agreement", BLUE),
                                              ("score_action_agree_n", "Priced-action agreement", GOLD)]):
        counts = [int(agg.loc[s, field]) for s in scopes]
        xx = np.arange(3) + (j-.5)*.36
        axes[1].bar(xx, np.array(counts)/nums, .36, label=label, color=color)
        for pos, k, n in zip(xx, counts, nums): axes[1].text(pos, k/n+.025, f"{k}/{n}", ha="center", fontsize=12)
    axes[1].set_ylim(0, 1.12); axes[1].set_yticks([0, .5, 1], ["0", "0.5", "1"])
    axes[1].set_xticks([0, 1, 2], ["Full fit\n(in training fan)", "Outer\n(two fits)", "Inner\n(24 fitted groups)"])
    axes[1].set_ylabel("Descriptive fraction on fixed states")
    axes[1].set_title("(b) Scope-specific state counts", loc="left", pad=14)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, 1.04), frameon=False)
    axes[1].grid(axis="y"); axes[1].set_axisbelow(True)
    fig.text(.06, .95, "Fixed-state forecast diagnostic: current action, then the same B", fontsize=18, weight="bold")
    fig.text(.06, .895, "79 active, predetermined states: 6 outer + 70 inner + 3 full-fit states. Two pending and nine no-admission slots stay skipped.", fontsize=12)
    fig.text(.06, .84, "At all six outer states, both predicted and realized action differences in the breach event are zero; eta drops out locally.", fontsize=12)
    fig.text(.06, .19, "One realized suffix is not a true conditional probability. Sign disagreements alone do not establish forecast miscalibration.\n"
        "B is annual restoration until that branch's own breach, then pending hold; it is not the learned adaptive-policy continuation.\n"
        "The minimum-ID / fixed-year states are nonrandom and repeated; no population accuracy, confidence interval or new policy is inferred.\n"
        "Source: HKUST(GZ) 2026 MCM SafeInvest synthetic scenarios 1–120 (CC BY-NC-SA 4.0); saved team 87 development diagnostics.",
        fontsize=12, va="top", linespacing=1.6)
    save(fig, "02_fixed_state_forecast_diagnostic")
    outer.to_csv(OUT / "fixed_outer_state_plot_data.csv", index=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--confirm-visual-review", action="store_true"); args = parser.parse_args()
    ref = BASE / "refit_admission_diagnostics_v1"; fixed = BASE / "fixed_state_forecast_diagnostic_v1"
    if json.loads((fixed / "independent_audit.json").read_text())["status"] != "passed": raise AssertionError("Upstream audit required")
    paths = [ref / "manifest.json", ref / "task_beta_mechanisms.csv", ref / "findings.json",
             BASE / "nested_dynamic_calibration_v1/context_selections.json", BASE / "nested_dynamic_calibration_v1/beta_summaries.csv",
             BASE / "nested_dynamic_calibration_v1/inner_validation_paths.csv", fixed / "manifest.json", fixed / "independent_audit.json",
             fixed / "state_comparisons.csv", fixed / "summary.csv"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    source_paths = [Path(__file__).resolve(), ROOT / "scripts/plot_development_structure.py"]
    source = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    configure(); OUT.mkdir(parents=True, exist_ok=True)
    refit_plot(pd.read_csv(ref / "task_beta_mechanisms.csv"), json.loads(paths[3].read_text()), pd.read_csv(paths[4]), pd.read_csv(paths[5]))
    forecast_plot(pd.read_csv(fixed / "state_comparisons.csv"), pd.read_csv(fixed / "summary.csv"))
    after = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if before != after: raise AssertionError("Frozen input changed during plotting")
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "complete" if args.confirm_visual_review else "rendered_pending_visual_review",
        "visual_review_confirmed": args.confirm_visual_review, "sources": source, "sources_at_end": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "input_sha256": before, "input_sha256_at_end": after,
        "outputs_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"},
        "final_scenarios_loaded": False, "raw_market_loaded": False, "new_policy_or_fit": False,
        "checks": {"font_min_points": 12, "text_bounds_passed": True, "source_attribution_visible": True,
                   "fixed_state_scopes_and_denominators_visible": True, "risk_differences_not_probability_tests": True}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "output": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__": main()
