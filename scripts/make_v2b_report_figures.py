"""Generate two English figures for the V2b report from frozen final arrays."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "v2b_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(ROOT / "results" / "mitigation_locked_final_evaluation_v1" / "summary.csv.gz")
q2 = summary[summary.question == "Q2"].set_index("object_id")
traces = np.load(ROOT / "results" / "mitigation_locked_final_evaluation_v1" / "traces.npz", allow_pickle=True)
sid = 161
idx = 161 - 121

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

colors = {"mitigation_main": "#1f4e79", "same_target_B": "#c55a11", "exact_thirds_annual": "#548235"}
labels = {"mitigation_main": "V2b", "same_target_B": "Same-target pending", "exact_thirds_annual": "Equal thirds"}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.15]})

ax = axes[0]
for oid in ("mitigation_main", "same_target_B", "exact_thirds_annual"):
    row = q2.loc[oid]
    x, y = 100 * row.median_cagr, 100 * row.p90_mdd
    ax.scatter(x, y, s=70, color=colors[oid], edgecolor="white", zorder=3, label=labels[oid])
    offset = (7, 7) if oid != "exact_thirds_annual" else (7, -20)
    ax.annotate(f"{labels[oid]}\n{int(row.breach_count)}/60 breaches", (x, y),
                textcoords="offset points", xytext=offset, fontsize=8, color=colors[oid])
ax.set_xlabel("Median real CAGR (%)")
ax.set_ylabel("P90 real MDD (%)")
ax.set_title("Risk-growth trade-off, final 121-180")
ax.grid(alpha=.25, linewidth=.5)
ax.set_xlim(1.9, 4.0)
ax.set_ylim(25, 47)
ax.legend(frameon=False, loc="upper left", fontsize=7.5)

ax = axes[1]
years = np.arange(0, 31)
for oid in ("mitigation_main", "same_target_B", "exact_thirds_annual"):
    dd = traces[f"{oid}__current_drawdown"][idx] * 100
    ax.plot(years, dd, color=colors[oid], linewidth=1.4, label=labels[oid])
ax.axhline(40, color="black", linestyle="--", linewidth=.8)
ax.text(0.3, 41.2, "L = 40%", fontsize=7.5)
ax.set_xlabel("Year")
ax.set_ylabel("Real drawdown from peak (%)")
ax.set_title("Severe case: synthetic ID 161")
ax.set_ylim(0, 60)
ax.set_xlim(0, 30)
ax.grid(alpha=.25, linewidth=.5)
ax.legend(frameon=False, loc="upper left", fontsize=7.5)

fig.tight_layout(pad=.4)
fig.savefig(OUT / "q2_tradeoff_severe.pdf", bbox_inches="tight")
fig.savefig(OUT / "q2_tradeoff_severe.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "q2_tradeoff_severe.pdf")
