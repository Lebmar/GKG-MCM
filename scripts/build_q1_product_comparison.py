"""Build the Q1 5x3 profile-by-product static comparison.

Descriptive only: the assignments remain the frozen V2b registry decisions.
Development 1-120 is the selection screen; final 121-180 is a post-freeze
descriptive check. This script does not fit, select or change a model.
"""
from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from safeinvest.data import load_paths
from experimental_execution.review_pending import simulate_paired
from safeinvest_research.metrics import path_metrics, summarize_paths

OUT = ROOT / "results" / "q1_product_comparison_v1"
PROFILES = (("A", 5, .10, "high"), ("B", 15, .25, "medium"), ("C", 30, .40, "low"), ("D", 25, .15, "medium"), ("E", 6, .35, "high"))
PRODUCTS = (("Conservative", [.25, .30, .45]), ("Balanced", [.45, .05, .50]), ("Growth", [.65, .05, .30]))
ASSIGNMENTS = {"A": "Review", "B": "Conservative", "C": "Balanced", "D": "Review", "E": "Growth"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(split: str, ids: list[int], allow_final: bool):
    returns, inflation, actual_ids = load_paths(ROOT / "data" / "SafeInvest_Synthetic_Data.csv", ids, allow_final=allow_final)
    rows = []
    for profile, horizon, loss, liquidity in PROFILES:
        for product, target in PRODUCTS:
            arm = simulate_paired(returns, inflation, np.asarray(target, dtype=float), loss,
                                  horizon=horizon, cost_rate=.0025)["annual"]
            real = arm["real_wealth"][:, 0, :]
            metrics = path_metrics(real)
            summary = summarize_paths(metrics, loss)
            breach = int(np.sum(metrics["mdd"] - loss > 1e-12))
            row = dict(
                split=split, profile=profile, horizon=horizon, loss_tolerance=loss, liquidity=liquidity,
                product=product, target_equity=target[0], target_bond=target[1], target_bill=target[2],
                median_cagr=float(summary["median_cagr"]), p10_cagr=float(summary["p10_cagr"]),
                median_mdd=float(summary["median_mdd"]), p90_mdd=float(summary["p90_mdd"]),
                cvar90_mdd=float(summary["cvar90_mdd"]), median_volatility=float(summary["median_volatility"]),
                breach_count=breach, breach_fraction=breach / len(actual_ids),
                mean_nominal_fees=float(arm["fees"][:, 0, :].sum(axis=1).mean()),
                mean_trade_fraction=float(arm["trades"][:, 0, :].mean(axis=1).mean()),
                selected_by_frozen_rule=ASSIGNMENTS[profile] == product,
                frozen_assignment=ASSIGNMENTS[profile],
                selected=ASSIGNMENTS[profile] == product,
            )
            rows.append(row)
    return rows


def validate(dev, final):
    """Selected rows must reproduce the published Q1 tables."""
    final_sum = pd.read_csv(ROOT / "results" / "mitigation_locked_final_evaluation_v1" / "summary.csv.gz")
    dev_sum = pd.read_csv(ROOT / "results" / "model_review" / "joint_liquidity_summary_v1" / "q1_own_horizon_annual_summary.csv")
    fields = ("median_cagr", "p10_cagr", "median_mdd", "p90_mdd")
    checks = []
    for rows, table, scope, label in (
        (dev, dev_sum[dev_sum.scope == "full_development"], "full_development", "development"),
        (final, final_sum, "final", "final"),
    ):
        key = "profile" if "profile" in table.columns else "client"
        for profile in ("B", "C", "E"):
            row = next(r for r in rows if r["profile"] == profile and r["selected"])
            if "object_id" in table.columns:
                ref = table[table.object_id == "Q1_" + profile].iloc[0]
            else:
                ref = table[table[key] == profile].iloc[0]
            for field in fields:
                checks.append(dict(split=label, profile=profile, field=field,
                                   computed=row[field], published=float(ref[field]),
                                   abs_error=abs(row[field] - float(ref[field]))))
            checks.append(dict(split=label, profile=profile, field="breach_count",
                               computed=row["breach_count"], published=int(ref["breach_count"]),
                               abs_error=abs(row["breach_count"] - int(ref["breach_count"]))))
    max_err = max(c["abs_error"] for c in checks)
    bad = [c for c in checks if c["abs_error"] > 1e-6]
    if bad:
        raise RuntimeError(f"Published Q1 rows do not reproduce: {bad[:3]}")
    return dict(checks=len(checks), max_abs_error=max_err, all_checked=True)


def pct(x):
    return f"{100*x:.2f}%"


def lpct(x):
    return f"{100*x:g}%"


def markdown(dev, final, validation):
    by = {(r["profile"], r["product"]): r for r in dev}
    lines = []
    lines.append("# Q1 — 5×3 profile × product comparison")
    lines.append("")
    lines.append("Static annual target-restoration products with exact 0.25% fees and each")
    lines.append("profile's own horizon and loss tolerance. **Development 1–120** is the")
    lines.append("selection screen; **final 121–180** is a post-freeze descriptive check. The")
    lines.append("frozen assignments were not changed by this table.")
    lines.append("")
    lines.append("## Development 1–120 (selection screen)")
    lines.append("")
    lines.append("| Profile | Product | Median CAGR | P10 CAGR | Median MDD | P90 MDD | Breaches | Frozen assignment |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for profile, horizon, loss, liquidity in PROFILES:
        for product, _ in PRODUCTS:
            r = by[(profile, product)]
            mark = " **selected**" if r["selected"] else ""
            lines.append(f"| {profile} (H={horizon}, L={lpct(loss)}, {liquidity}) | {product}{mark} | {pct(r['median_cagr'])} | {pct(r['p10_cagr'])} | {pct(r['median_mdd'])} | {pct(r['p90_mdd'])} | {r['breach_count']}/120 | {r['frozen_assignment']} |")
    lines.append("")
    lines.append("All selected B/C/E products pass the declared empirical 10% loss-event screen")
    lines.append("on development data: B 9/120, C 9/120, E 6/120. A and D have no")
    lines.append("risk-feasible product (34–104/120 and 67–104/120 breaches respectively), so")
    lines.append("they remain Review. The final selected-product table is the evaluation table")
    lines.append("in [TASK_ANSWER_SHEET.md](TASK_ANSWER_SHEET.md); it retains the adverse final")
    lines.append("drift (B 11/60, C 8/60, E 5/60, E P10 −1.58%).")
    lines.append("")
    lines.append("## Final 121–180 (descriptive, not used to assign)")
    lines.append("")
    lines.append("| Profile | Product | Median CAGR | P10 CAGR | Median MDD | P90 MDD | Breaches |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    fby = {(r["profile"], r["product"]): r for r in final}
    for profile, horizon, loss, liquidity in PROFILES:
        for product, _ in PRODUCTS:
            r = fby[(profile, product)]
            mark = " **selected**" if r["selected"] else ""
            lines.append(f"| {profile} (H={horizon}, L={lpct(loss)}, {liquidity}) | {product}{mark} | {pct(r['median_cagr'])} | {pct(r['p10_cagr'])} | {pct(r['median_mdd'])} | {pct(r['p90_mdd'])} | {r['breach_count']}/60 |")
    lines.append("")
    lines.append("The final 5×3 table is a descriptive check only. In particular, B/C static")
    lines.append("final frequencies exceed the 10% training reference and E Growth has negative")
    lines.append("P10 growth; these are disclosed, not repaired by changing assignments.")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"The selected B/C/E rows reproduce the published development and final Q1 tables; "
                 f"{validation['checks']} checks, maximum absolute error {validation['max_abs_error']:.3g}.")
    lines.append("")
    lines.append("Source: [development CSV](../results/q1_product_comparison_v1/development_5x3.csv), "
                 "[final CSV](../results/q1_product_comparison_v1/final_5x3.csv), "
                 "[manifest](../results/q1_product_comparison_v1/manifest.json).")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = build("development", list(range(1, 121)), False)
    final = build("final", list(range(121, 181)), True)
    validation = validate(dev, final)
    for name, rows in (("development_5x3.csv", dev), ("final_5x3.csv", final)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    manifest = dict(
        version="q1-product-5x3-comparison-v1",
        status="complete",
        scope="Descriptive static Q1 product comparison; assignments fixed by the frozen V2b registry.",
        development_ids=[1, 120], final_ids=[121, 180],
        profiles={p: dict(horizon=h, loss=L, liquidity=q, assignment=ASSIGNMENTS[p]) for p, h, L, q in PROFILES},
        products={n: t for n, t in PRODUCTS},
        new_fits=0, new_selections=0, assignments_changed=False,
        final_5x3_used_for_selection=False,
        final_5x3_read_scope="Post-freeze descriptive check only; not used to change any assignment.",
        validation=validation,
        source_files={
            "scripts/build_q1_product_comparison.py": sha(ROOT / "scripts/build_q1_product_comparison.py"),
            "data/SafeInvest_Synthetic_Data.csv": sha(ROOT / "data/SafeInvest_Synthetic_Data.csv"),
            "research/experimental_execution/review_pending.py": sha(ROOT / "research/experimental_execution/review_pending.py"),
            "research/safeinvest_research/metrics.py": sha(ROOT / "research/safeinvest_research/metrics.py"),
        },
        outputs={n: sha(OUT / n) for n in ("development_5x3.csv", "final_5x3.csv")},
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "submission" / "Q1_PRODUCT_COMPARISON.md").write_text(markdown(dev, final, validation), encoding="utf-8")
    print("wrote", OUT, "and submission/Q1_PRODUCT_COMPARISON.md")
    print("validation", validation)
    print("development selected B/C/E breaches", [(r["profile"], r["breach_count"]) for r in dev if r["selected"]])
    print("final selected B/C/E breaches", [(r["profile"], r["breach_count"]) for r in final if r["selected"]])


if __name__ == "__main__":
    main()
