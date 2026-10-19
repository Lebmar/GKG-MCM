"""Audit saved results without importing model code or rerunning a policy.

Final market rows are read only to verify the arithmetic of already archived
actions. All decisions remain cached: no policy is called or rerun here.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


STRATEGIES = ("S0_equal", "S1_annual", "S2_gate", "S3_dynamic", "S2_review", "S3_review")
CLIENTS = {
    "A": (5, 0.10, "high"), "B": (15, 0.25, "medium"),
    "C": (30, 0.40, "low"), "D": (25, 0.15, "medium"),
    "E": (6, 0.35, "high"),
}
ATOL = 2e-11
RTOL = 2e-11


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_final_market(path: Path):
    """Read original final rows for authorized replay of frozen actions only."""
    paths = {scenario: {} for scenario in range(121, 181)}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in paths:
                continue
            period = int(row["period"])
            if period in paths[scenario]:
                raise ValueError(f"duplicate original market key: {(scenario, period)}")
            paths[scenario][period] = [float(row[key]) for key in ("equity_return", "bond_return", "bill_proxy_return", "inflation")]
    for scenario, values in paths.items():
        if set(values) != set(range(1, 31)):
            raise ValueError(f"incomplete original final market path: {scenario}")
    values = np.asarray([[paths[scenario][year] for year in range(1, 31)] for scenario in range(121, 181)])
    if not np.all(np.isfinite(values)) or np.any(values <= -1):
        raise ValueError("original final market rows have invalid gross factors")
    return values[:, :, :3], values[:, :, 3]


def quantile(values, probability, axis=-1):
    """Linear empirical quantile from sorted order statistics, independently."""
    ordered = np.sort(np.asarray(values), axis=axis)
    position = (ordered.shape[axis] - 1) * probability
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return np.take(ordered, lower, axis=axis) * (1 - fraction) + np.take(ordered, upper, axis=axis) * fraction


def independent_metrics(wealth, inflation):
    """Use log price accumulation and explicit annual peak recursion."""
    periods = inflation.shape[1]
    log_prices = np.column_stack((np.zeros(len(inflation)), np.cumsum(np.log1p(inflation), axis=1)))
    real = wealth * np.exp(-log_prices)
    peak = real[:, 0].copy()
    mdd = np.zeros(len(real))
    for year in range(1, periods + 1):
        peak = np.maximum(peak, real[:, year])
        mdd = np.maximum(mdd, (peak - real[:, year]) / peak)
    real_returns = real[:, 1:] / real[:, :-1] - 1
    centered = real_returns - real_returns.mean(axis=1, keepdims=True)
    volatility = np.sqrt(np.sum(centered * centered, axis=1) / (periods - 1)) if periods > 1 else np.zeros(len(real))
    cagr = np.expm1((np.log(real[:, -1]) - np.log(real[:, 0])) / periods)
    return {"real": real, "cagr": cagr, "mdd": mdd, "volatility": volatility, "terminal_real": real[:, -1]}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "passed": bool(passed), **details})

    def equal(self, name, actual, expected):
        self.check(name, actual == expected, actual=actual, expected=expected)

    def close(self, name, actual, expected):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        shapes_match = actual.shape == expected.shape
        error = float(np.max(np.abs(actual - expected))) if shapes_match and actual.size else None
        self.check(name, shapes_match and np.allclose(actual, expected, atol=ATOL, rtol=RTOL), max_abs_error=error)


def independent_summary(name, archive, inflation, metrics, cost_rate):
    field = lambda key: archive[f"{name}__{key}"]
    utility = np.log(metrics["terminal_real"] / 100) - 4 * metrics["mdd"]
    return {
        "cost_rate": cost_rate, "n": len(inflation), "horizon": inflation.shape[1],
        "cagr_median": quantile(metrics["cagr"], 0.5), "cagr_p10": quantile(metrics["cagr"], 0.1),
        "mdd_median": quantile(metrics["mdd"], 0.5), "mdd_p90": quantile(metrics["mdd"], 0.9),
        "volatility_median": quantile(metrics["volatility"], 0.5),
        "terminal_real_median": quantile(metrics["terminal_real"], 0.5),
        "mean_utility": utility.sum() / len(utility),
        "trade_fraction": np.count_nonzero(field("trades")) / field("trades").size,
        "subsequent_trade_fraction": np.count_nonzero(field("trades")[:, 1:]) / field("trades")[:, 1:].size,
        "cost_mean": np.sum(field("costs")) / len(inflation),
        "review_fraction": np.count_nonzero(np.any(field("pending"), axis=1)) / len(inflation),
    }


def audit_archive(audit, folder, expected_ids, cost_rate, original_market=None):
    label = folder.name
    metrics = {}
    with np.load(folder / "paths.npz", allow_pickle=False) as archive:
        inflation = archive["inflation"]
        n, horizon = inflation.shape
        audit.close(f"{label}.scenario_ids", archive["scenario_ids"], expected_ids)
        audit.check(f"{label}.inflation_finite_positive_gross", np.all(np.isfinite(inflation)) and np.all(inflation > -1))
        if original_market is not None:
            original_returns, original_inflation = original_market
            audit.equal(f"{label}.original_market_shape", list(original_returns.shape), [n, horizon, 3])
            audit.close(f"{label}.cached_inflation_matches_original", inflation, original_inflation)
        summaries = {row["strategy"]: row for row in read_csv(folder / "summary.csv")}
        audit.equal(f"{label}.summary_strategy_set", sorted(summaries), sorted(STRATEGIES))
        for name in STRATEGIES:
            field = lambda key: archive[f"{name}__{key}"]
            wealth, costs, moved = field("wealth"), field("costs"), field("moved")
            weights, targets = field("weights"), field("targets")
            trades, pending = field("trades"), field("pending")
            prefix = f"{label}.{name}"
            audit.equal(f"{prefix}.wealth_shape", list(wealth.shape), [n, horizon + 1])
            audit.equal(f"{prefix}.weights_shape", list(weights.shape), [n, horizon, 3])
            audit.equal(f"{prefix}.targets_shape", list(targets.shape), [n, horizon, 3])
            for key in ("costs", "moved", "trades", "pending", "benefit"):
                audit.equal(f"{prefix}.{key}_shape", list(field(key).shape), [n, horizon])
                audit.check(f"{prefix}.{key}_finite", np.all(np.isfinite(field(key))))
            audit.close(f"{prefix}.initial_wealth", wealth[:, 0], np.full(n, 100.0))
            audit.check(f"{prefix}.wealth_positive_finite", np.all(np.isfinite(wealth)) and np.all(wealth > 0))
            audit.check(f"{prefix}.amounts_nonnegative", np.all(costs >= 0) and np.all(moved >= 0))
            audit.check(f"{prefix}.fees_funded", np.all(costs < wealth[:, :-1]))
            audit.check(f"{prefix}.moved_within_investable_wealth", np.all(moved <= wealth[:, :-1] - costs + ATOL))
            audit.close(f"{prefix}.fee_equation", costs, cost_rate * moved)
            for key, values in (("weights", weights), ("targets", targets)):
                audit.check(f"{prefix}.{key}_nonnegative_finite", np.all(np.isfinite(values)) and np.all(values >= -ATOL))
                audit.close(f"{prefix}.{key}_sum_one", values.sum(axis=2), np.ones((n, horizon)))
            audit.close(f"{prefix}.trade_flags", trades, moved > 1e-9)
            audit.check(f"{prefix}.no_fees_without_trade", np.all(costs[~trades] <= cost_rate * 1e-9 + ATOL))
            audit.check(f"{prefix}.pending_persistent", np.all(np.diff(pending.astype(int), axis=1) >= 0))
            audit.check(f"{prefix}.no_trades_while_pending", not np.any(trades & pending))
            audit.check(f"{prefix}.no_fees_while_pending", np.all(costs[pending] == 0))
            if original_market is not None:
                # Replay archived allocations; never generate or choose actions.
                pre = np.zeros((n, 3))
                pre[:, 2] = 100.0
                pretrade, posttrade, endholdings = [], [], []
                for year in range(horizon):
                    post = (wealth[:, year] - costs[:, year])[:, None] * weights[:, year]
                    end = post * (1.0 + original_returns[:, year])
                    pretrade.append(pre.copy())
                    posttrade.append(post)
                    endholdings.append(end)
                    pre = end
                pretrade, posttrade, endholdings = map(np.stack, (pretrade, posttrade, endholdings))
                pretrade, posttrade, endholdings = (np.swapaxes(v, 0, 1) for v in (pretrade, posttrade, endholdings))
                actual_transfers = np.maximum(posttrade - pretrade, 0).sum(axis=2)
                sale_proceeds = np.maximum(pretrade - posttrade, 0).sum(axis=2)
                audit.check(f"{prefix}.reconstructed_holdings_nonnegative", np.all(pretrade >= 0) and np.all(posttrade >= 0) and np.all(endholdings >= 0))
                audit.close(f"{prefix}.original_market_wealth_recursion", endholdings.sum(axis=2), wealth[:, 1:])
                audit.close(f"{prefix}.pretrade_wealth", pretrade.sum(axis=2), wealth[:, :-1])
                audit.close(f"{prefix}.self_financing_posttrade_wealth", posttrade.sum(axis=2), wealth[:, :-1] - costs)
                audit.close(f"{prefix}.actual_transfer_amounts", actual_transfers, moved)
                audit.close(f"{prefix}.fees_on_reconstructed_transfers", cost_rate * actual_transfers, costs)
                audit.close(f"{prefix}.sale_funds_transfers_and_fees", sale_proceeds, actual_transfers + costs)
                audit.close(f"{prefix}.no_trade_preserves_asset_amounts", posttrade[~trades], pretrade[~trades])
                audit.close(f"{prefix}.pending_preserves_asset_amounts", posttrade[pending], pretrade[pending])
                audit.close(f"{prefix}.trades_achieve_cached_target", weights[trades], targets[trades])
            m = independent_metrics(wealth, inflation)
            metrics[name] = m
            # A second, direct deflator cross-checks the log-based route.
            price = np.column_stack((np.ones(n), np.multiply.accumulate(1 + inflation, axis=1)))
            audit.close(f"{prefix}.real_wealth_deflation", m["real"] * price, wealth)
            audit.check(f"{prefix}.real_metrics_finite", all(np.all(np.isfinite(v)) for v in m.values()))
            audit.check(f"{prefix}.drawdown_bounds", np.all(m["mdd"] >= 0) and np.all(m["mdd"] < 1))
            recomputed = independent_summary(name, archive, inflation, m, cost_rate)
            for key, value in recomputed.items():
                audit.close(f"{prefix}.summary.{key}", value, float(summaries[name][key]))
        expected_fields = {"inflation", "scenario_ids"} | {f"{name}__{key}" for name in STRATEGIES for key in ("wealth", "weights", "targets", "costs", "moved", "trades", "pending", "benefit")}
        audit.equal(f"{label}.archive_field_set", sorted(archive.files), sorted(expected_fields))
        severe = int(archive["scenario_ids"][np.argmax(metrics["S0_equal"]["mdd"])])
    return metrics, severe


def independent_paired(a, b, protocol):
    count = len(a["cagr"])
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(count, size=(protocol["bootstrap_replicates"], count))
    ua = np.log(a["terminal_real"] / 100) - 4 * a["mdd"]
    ub = np.log(b["terminal_real"] / 100) - 4 * b["mdd"]
    estimators = {
        "mean_utility": ((ua - ub).mean(), (ua[draws] - ub[draws]).mean(axis=1)),
        "median_cagr": (quantile(a["cagr"], .5) - quantile(b["cagr"], .5), quantile(a["cagr"][draws], .5) - quantile(b["cagr"][draws], .5)),
        "p10_cagr": (quantile(a["cagr"], .1) - quantile(b["cagr"], .1), quantile(a["cagr"][draws], .1) - quantile(b["cagr"][draws], .1)),
        "p90_mdd": (quantile(a["mdd"], .9) - quantile(b["mdd"], .9), quantile(a["mdd"][draws], .9) - quantile(b["mdd"][draws], .9)),
    }
    return {name: {"estimate": estimate, "low95": quantile(values, .025), "high95": quantile(values, .975)} for name, (estimate, values) in estimators.items()}


def compare_paired(audit, label, actual, expected):
    audit.equal(f"{label}.metric_set", sorted(actual), sorted(expected))
    for metric, values in actual.items():
        for statistic, value in values.items():
            audit.close(f"{label}.{metric}.{statistic}", value, expected[metric][statistic])


def audit_onboarding(audit, root, frozen, protocol):
    rows = read_csv(root / "results/frozen/onboarding.csv")
    audit.equal("onboarding.frozen_row_count", len(rows), len(CLIENTS) * len(frozen["portfolios"]))
    feasible = {client: [] for client in CLIENTS}
    alpha = protocol["onboarding_breach_probability"]
    for row in rows:
        client, portfolio = row["client"], row["portfolio"]
        horizon, tolerance, liquidity = CLIENTS[client]
        prefix = f"onboarding.frozen.{client}.{portfolio}"
        audit.equal(f"{prefix}.horizon", int(row["horizon"]), horizon)
        audit.close(f"{prefix}.tolerance", float(row["tolerance"]), tolerance)
        audit.equal(f"{prefix}.development_n", int(row["n"]), 120)
        target = [float(row[key]) for key in ("w_equity", "w_bond", "w_bill")]
        audit.close(f"{prefix}.weights", target, frozen["portfolios"][portfolio])
        probability = int(row["breach_count"]) / int(row["n"])
        liquid = target[2] >= protocol["liquidity_bill_floors"][liquidity] - 1e-12
        eligible = probability <= alpha + 1e-12 and liquid
        audit.close(f"{prefix}.breach_probability", probability, float(row["breach_probability"]))
        audit.equal(f"{prefix}.liquidity", liquid, row["liquidity_ok"] == "True")
        audit.equal(f"{prefix}.eligibility", eligible, row["eligible"] == "True")
        if eligible:
            feasible[client].append((float(row["cagr_median"]), portfolio))
    reconstructed = {client: max(options)[1] if options else "Human review" for client, options in feasible.items()}
    audit.equal("onboarding.assignments_from_frozen_evidence", reconstructed, frozen["assignments"])
    diagnostic_rows = read_csv(root / "results/final/onboarding_diagnostics.csv")
    audit.equal("onboarding.final_diagnostic_row_count", len(diagnostic_rows), len(rows))
    adverse = []
    for row in diagnostic_rows:
        client = row["client"]
        audit.equal(f"onboarding.final.{client}.{row['portfolio']}.assignment_unchanged", row["frozen_assignment"], frozen["assignments"][client])
        if row["portfolio"] == frozen["assignments"][client] and row["eligible"] != "True":
            adverse.append(client)
    return sorted(set(adverse))


def verify(root: Path):
    audit = Audit()
    frozen_path = root / "results/frozen/manifest.json"
    frozen = read_json(frozen_path)
    final = read_json(root / "results/final/evaluation_manifest.json")
    protocol = read_json(root / "docs/protocol.json")
    audit.equal("provenance.frozen_manifest_sha256", sha256(frozen_path), final["frozen_manifest_sha256"])
    audit.equal("provenance.final_and_frozen_fingerprints", final["fingerprints"], frozen["fingerprints"])
    for relative, expected in frozen["fingerprints"].items():
        audit.equal(f"provenance.sha256.{relative}", sha256(root / relative), expected)
    frozen_sources = sorted(name for name in frozen["fingerprints"] if name.startswith("src/"))
    actual_sources = sorted(str(path.relative_to(root)) for path in (root / "src").rglob("*.py"))
    audit.equal("provenance.source_file_set", actual_sources, frozen_sources)
    audit.equal("provenance.forecast_sha256", sha256(root / "results/frozen/forecast_bank.npy"), frozen["forecast_sha256"])
    audit.equal("provenance.portfolio_artifact", read_json(root / "results/frozen/portfolios.json"), frozen["portfolios"])
    audit.equal("provenance.architecture_decision", read_json(root / "results/development/architecture_decision.json"), frozen["validation_decision"])
    audit.close("protocol.utility_risk_aversion", protocol["risk_aversion"]["Balanced"], 4.0)
    audit.equal("selection.final_matches_frozen", final["selected"], frozen["selected"])
    audit.equal("selection.frozen_matches_validation", frozen["selected"], frozen["validation_decision"]["selected"])
    audit.equal("selection.change_flag", final["selection_changed_after_evaluation"], False)
    audit.check("provenance.freeze_precedes_final", datetime.fromisoformat(frozen["created_utc"]) < datetime.fromisoformat(final["completed_utc"]))
    original_market = read_final_market(root / "data/SafeInvest_Synthetic_Data.csv")
    metrics, severe = audit_archive(audit, root / "results/final", np.arange(121, 181), protocol["cost_rate"], original_market)
    audit.close("final.manifest_scenario_ids", final["scenario_ids"], np.arange(121, 181))
    audit.equal("final.severe_scenario_rule", severe, final["severe_scenario"])
    comparisons = read_json(root / "results/final/paired_comparisons.json")
    pair_map = {name: (name, "S0_equal") for name in STRATEGIES if name != "S0_equal"}
    pair_map.update({"S3_review_minus_S2_review": ("S3_review", "S2_review"), "S2_gate_minus_S1_annual": ("S2_gate", "S1_annual"), "S2_review_minus_S2_gate": ("S2_review", "S2_gate")})
    audit.equal("final.paired_comparison_set", sorted(comparisons), sorted(pair_map))
    for label, (a, b) in pair_map.items():
        compare_paired(audit, f"final.paired.{label}", independent_paired(metrics[a], metrics[b], protocol), comparisons[label])
    development, _ = audit_archive(audit, root / "results/development", np.arange(81, 121), protocol["cost_rate"])
    expected_validation = frozen["validation_decision"]["paired_dynamic_minus_gate"]
    compare_paired(audit, "development.frozen_validation_comparison", independent_paired(development["S3_review"], development["S2_review"], protocol), expected_validation)
    adverse = audit_onboarding(audit, root, frozen, protocol)
    # Only the base-cost sensitivity rows have archived policy paths.
    main_rows = {row["strategy"]: row for row in read_csv(root / "results/final/summary.csv")}
    base_rows = [row for row in read_csv(root / "results/final/cost_sensitivity.csv") if float(row["cost_rate"]) == protocol["cost_rate"]]
    audit.equal("sensitivity.base_row_count", len(base_rows), 2)
    for row in base_rows:
        for field, value in row.items():
            if field != "strategy":
                audit.close(f"sensitivity.base.{row['strategy']}.{field}", float(value), float(main_rows[row["strategy"]][field]))
    failed = [check for check in audit.checks if not check["passed"]]
    artifacts = (
        "results/frozen/manifest.json", "results/frozen/portfolios.json", "results/frozen/forecast_bank.npy",
        "results/frozen/onboarding.csv", "results/development/architecture_decision.json",
        "results/development/paths.npz", "results/development/summary.csv",
        "results/final/evaluation_manifest.json", "results/final/paths.npz", "results/final/summary.csv",
        "results/final/paired_comparisons.json", "results/final/onboarding_diagnostics.csv",
        "results/final/cost_sensitivity.csv", "results/final/horizon15.csv",
    )
    return {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_script_sha256": sha256(Path(__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "audited_artifact_sha256": {relative: sha256(root / relative) for relative in artifacts},
        "status": "PASS" if not failed else "FAIL",
        "checks_total": len(audit.checks), "checks_passed": len(audit.checks) - len(failed),
        "frozen_policy": frozen["selected"], "frozen_assignments": frozen["assignments"],
        "final_diagnostic_assigned_portfolio_screen_failures": adverse,
        "scope": "Independent cached-output arithmetic, archived-action replay against original final market rows, and provenance audit; no policy decisions, fitting, or parameter selection are rerun.",
        "limitations": [
            "Final wealth and transfer arithmetic are checked against original market returns, but the forecast scores, admissibility rules, and action-selection logic are not independently re-executed.",
            "Non-base-cost sensitivity runs and separately recomputed 15-year policy paths are not cached; their aggregate calculations cannot be independently reconstructed here.",
            "Onboarding assignments are reconstructed from frozen diagnostic rows; client-specific wealth paths are not archived, so their risk metrics are not independently recomputed.",
            "Correct arithmetic and frozen hashes do not establish market stationarity, future performance, tail-risk coverage, or suitability guarantees.",
            "The frozen prior-exposure disclosure remains applicable: pooled descriptive statistics over all scenarios were seen before formal model development.",
        ],
        "checks": audit.checks, "failures": failed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    result = verify(root)
    output = arguments.output or root / "results/verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"],
                      "checks_total": result["checks_total"], "audit_file": str(output)}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
