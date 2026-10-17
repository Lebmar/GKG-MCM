"""Plot frozen nested OOF artifacts; no market reader or model selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/model_review/nested_static_calibration_v1"
OUT = ROOT / "research/figures/nested_calibration"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if (manifest["scenario_ids"] != list(range(1, 121)) or manifest["final_scenarios_loaded"]
            or manifest["sources"] != manifest["sources_at_end"]
            or manifest["frozen_input_hashes"] != manifest["frozen_input_hashes_at_end"]):
        raise ValueError("Require an unchanged development-only completed run")
    for group in (manifest["sources"], manifest["frozen_input_hashes"]):
        for relative, digest in group.items():
            if sha(ROOT / relative) != digest:
                raise ValueError(f"Changed source/input: {relative}")
    for relative, digest in manifest["outputs_sha256"].items():
        if sha(SOURCE / relative) != digest:
            raise ValueError(f"Changed result: {relative}")
    summary = pd.read_csv(SOURCE / "outer_vs_primary_summary.csv")
    path_data = pd.read_csv(SOURCE / "outer_oof_paths.csv")
    rows = []
    for client in ("B", "C", "E"):
        for view in ("primary_gamma10", "effective"):
            match = summary.loc[(summary.client == client) & (summary.view == view)]
            if len(match) != 1:
                raise ValueError("Need one OOF summary per client and procedure")
            row = match.iloc[0]
            if (row.total_paths, row.auto_n, row.review_n) != (120, 120, 0):
                raise ValueError("This paired chart requires equal full automatic coverage")
            if view == "effective":
                paths = path_data.loc[(path_data.client == client) & (path_data.view == view)]
                if (len(paths) != 120 or set(paths.scenario_id) != set(range(1, 121))
                        or not paths.assigned.eq(True).all()
                        or int(paths.breach.sum()) != int(row.breach_count)
                        or not np.isclose(paths.log_growth.mean(), row.mean_log_growth, atol=1e-12, rtol=0)):
                    raise ValueError("Effective summary differs from frozen OOF paths")
            rows.append({"client": client, "procedure": view, "assigned_paths": 120,
                         "review_paths": 0, "breach_count": int(row.breach_count),
                         "breach_probability": float(row.breach_probability),
                         "mean_annual_log_growth": float(row.mean_log_growth),
                         "mean_annual_log_growth_percent": 100 * float(row.mean_log_growth),
                         "service_alpha": .10,
                         "solver_gamma": .10 if view == "primary_gamma10" else .075})
    data = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "01_oof_calibration_comparison_data.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    colors, labels = ("#62738A", "#137F83"), ("Fixed training screen 10%", "Inner-selected screen 7.5%")
    x, width = np.arange(3), .32
    for panel, (field, ylabel, limit) in enumerate([
        ("breach_count", "Full-horizon exceedances / 120 paths", (0, 16)),
        ("mean_annual_log_growth_percent", "Mean annual log net real growth (%)", (0, 6)),
    ]):
        ax = axes[panel]
        for j, view in enumerate(("primary_gamma10", "effective")):
            values = data.loc[data.procedure == view].set_index("client").loc[["B", "C", "E"], field].to_numpy()
            bars = ax.bar(x + (j - .5) * width, values, width, color=colors[j])
            ax.bar_label(bars, labels=[f"{v:.0f}" if panel == 0 else f"{v:.2f}" for v in values], padding=4, fontsize=10)
        if panel == 0:
            ax.axhline(12, color="#9C5D1B", linestyle="--", linewidth=1.2)
            ax.text(2.5, 12.4, "10% empirical reference", ha="right", color="#845017", fontsize=9)
            ax.yaxis.set_major_locator(MultipleLocator(4))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set(xticks=x, xticklabels=["B · 15 years", "C · 30 years", "E · 6 years"],
               ylabel=ylabel, ylim=limit)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Learning a training buffer: observed risk–growth trade-off", x=.06, y=.97, ha="left", fontsize=16, weight="bold")
    fig.text(.06, .90, "Client service benchmark stays at 10%; annual restoration is unchanged.", fontsize=11)
    fig.legend([Patch(color=c) for c in colors], labels, loc="lower left", bbox_to_anchor=(.055, .115), ncol=2, frameon=False)
    fig.text(.06, .08, "Five whole-path outer folds; gamma chosen using four inner folds. All displayed cohorts: n=120.", fontsize=9)
    fig.text(.06, .04, "Reused development data; dependent folds. Counts are empirical diagnostics, not probability guarantees.", fontsize=9, color="#495462")
    fig.subplots_adjust(left=.07, right=.98, bottom=.27, top=.81, wspace=.30)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"01_oof_risk_growth_tradeoff.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [
        SOURCE / "manifest.json", SOURCE / "outer_vs_primary_summary.csv", SOURCE / "outer_oof_paths.csv",
        ROOT / "docs/nested_static_calibration_protocol.json", Path(__file__).resolve()]}
    outputs = {p.name: sha(p) for p in OUT.iterdir() if p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": inputs, "outputs_sha256": outputs,
        "scope": "Frozen nested OOF comparison only; no final scenarios, re-selection or probability certificate",
        "validation": "Full equal automatic cohorts and effective OOF path-to-summary arithmetic checked"}, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
