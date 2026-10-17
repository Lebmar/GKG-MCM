"""Independent development-only accounting and frozen-artifact audit.

No production execution, simulation, metric or selection functions are used.
The sole imported project helper is the guarded development-path CSV loader.
This verifies numerical consistency, not investment suitability or model truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from safeinvest.data import load_paths

OUT = ROOT / "results/model_review/risk_menu_v1"
PROTOCOL = ROOT / "docs/risk_menu_development_protocol.json"
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = Counter()
        self.failures = []
        self.errors = {}

    def check(self, condition, identity, context=None):
        self.checks[identity] += 1
        if not bool(condition):
            self.failures.append({"check": identity, "context": context})

    def close(self, actual, expected, identity, context=None, atol=1e-12, rtol=1e-12):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        if actual.shape != expected.shape:
            self.check(False, identity, {"context": context, "actual_shape": list(actual.shape),
                                         "expected_shape": list(expected.shape)})
            return
        finite = np.isfinite(actual) & np.isfinite(expected)
        difference = np.abs(actual - expected)
        maximum = float(np.max(difference, initial=0)) if np.all(finite) else None
        scaled = float(np.max(difference / (1 + np.abs(expected)), initial=0)) if np.all(finite) else None
        record = self.errors.setdefault(identity, {"maximum_absolute_error": 0.,
                                                   "maximum_error_divided_by_one_plus_abs_reference": 0.,
                                                   "compared_values": 0, "atol": atol, "rtol": rtol})
        record["compared_values"] += int(actual.size)
        if maximum is not None:
            record["maximum_absolute_error"] = max(record["maximum_absolute_error"], maximum)
            record["maximum_error_divided_by_one_plus_abs_reference"] = max(
                record["maximum_error_divided_by_one_plus_abs_reference"], scaled)
        self.check(np.all(finite & (difference <= atol + rtol * np.abs(expected))), identity,
                   {"identity": context, "maximum_absolute_error": maximum})


def bisection_trade(holdings, weights, rate):
    """Solve C - c*sum((w*(V-C)-x)+)=0 independently by 80 bisections."""
    total = holdings.sum(axis=-1)
    lower = np.zeros_like(total)
    upper = rate * total
    for _ in range(80):
        middle = lower + (upper - lower) / 2
        purchases = np.maximum((total - middle)[..., None] * weights - holdings, 0).sum(axis=-1)
        positive = middle > rate * purchases
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)
    fee = lower + (upper - lower) / 2
    post = (total - fee)[..., None] * weights
    purchases = np.maximum(post - holdings, 0).sum(axis=-1)
    return post, fee, purchases


def synthetic_checks(audit):
    rate = .0025
    x = np.asarray([[[0., 0., 100.], [20., 30., 50.], [0., 0., 100.]]])
    w = np.asarray([[1., 0., 0.], [.5, .4, .1], [0., 0., 1.]])
    post, fees, moved = bisection_trade(x, w, rate)
    expected = np.asarray([[100 * rate / (1 + rate), .1 / 1.00225, 0.]])
    audit.close(fees, expected, "synthetic_analytic_fee", atol=1e-14, rtol=1e-14)
    audit.close(post.sum(axis=-1) + fees, x.sum(axis=-1), "synthetic_funding", atol=1e-13)
    audit.close(fees, rate * moved, "synthetic_transfer_fee", atol=1e-14)
    _, free, _ = bisection_trade(x, w, 0.)
    audit.close(free, np.zeros_like(free), "synthetic_zero_cost", atol=0, rtol=0)


def independent_replay(returns, inflation, weights, rate, horizons, audit):
    n, years, _ = returns.shape
    normalized = weights / weights.sum(axis=1, keepdims=True)
    holdings = np.zeros((n, len(weights), 3))
    holdings[..., 2] = 100.
    nominal = np.full((n, len(weights), years + 1), 100.)
    fee_series = np.zeros((n, len(weights), years))
    moved_series = np.zeros_like(fee_series)
    for year in range(years):
        before = holdings.sum(axis=-1)
        post, fee, moved = bisection_trade(holdings, normalized[None, ...], rate)
        sales = np.maximum(holdings - post, 0).sum(axis=-1)
        audit.close(post.sum(axis=-1) + fee, before, "self_financing_wealth", year + 1,
                    atol=1e-10, rtol=1e-13)
        audit.close(fee, rate * moved, "fee_equals_rate_times_incoming_transfer", year + 1,
                    atol=1e-11, rtol=1e-13)
        audit.close(sales - moved, fee, "sales_fund_purchases_and_fee", year + 1,
                    atol=1e-10, rtol=1e-13)
        audit.check(np.all(post >= 0) and np.all(fee >= 0), "nonnegative_holdings_and_fee", year + 1)
        fee_series[..., year], moved_series[..., year] = fee, moved
        holdings = post * (1 + returns[:, None, year, :])
        nominal[..., year + 1] = holdings.sum(axis=-1)
    price = np.ones((n, years + 1))
    for year in range(years):
        price[:, year + 1] = price[:, year] * (1 + inflation[:, year])
    real = nominal / price[:, None, :]
    metrics = {}
    for horizon in horizons:
        path = real[..., :horizon + 1]
        annual = path[..., 1:] / path[..., :-1] - 1
        center = annual.mean(axis=-1, keepdims=True)
        growth = np.log(path[..., -1] / 100.) / horizon
        metrics[horizon] = {
            "log_growth": growth,
            "cagr": (path[..., -1] / 100.) ** (1 / horizon) - 1,
            "mdd": np.max(1 - path / np.maximum.accumulate(path, axis=-1), axis=-1),
            "volatility": np.sqrt(np.sum((annual - center) ** 2, axis=-1) / (horizon - 1)),
            "terminal_real": path[..., -1],
            "fees": fee_series[..., :horizon].sum(axis=-1),
            "moved": moved_series[..., :horizon].sum(axis=-1),
            "trade_fraction": np.count_nonzero(moved_series[..., :horizon] > TOL, axis=-1) / horizon,
        }
    return metrics


def fractional_cvar(values):
    ordered = np.sort(values, axis=0)[::-1]
    tail_n = .1 * len(ordered)
    complete = int(np.floor(tail_n))
    remainder = tail_n - complete
    total = ordered[:complete].sum(axis=0)
    if remainder:
        total = total + remainder * ordered[complete]
    return total / tail_n


def frontier_ids(growth, risk):
    kept = []
    for i in range(len(growth)):
        dominators = [j for j in range(len(growth))
                      if growth[j] >= growth[i] - TOL and risk[j] <= risk[i] + TOL
                      and (growth[j] > growth[i] + TOL or risk[j] < risk[i] - TOL)]
        if not dominators:
            kept.append(i)
    return kept


def expected_completion(identified, growth, risk, frontier):
    selected, extension = list(identified), []
    while len(selected) < 3:
        remaining = [i for i in frontier if i not in selected]
        if not remaining:
            break
        if not selected:
            chosen = min(remaining, key=lambda i: (risk[i], -growth[i], i))
        else:
            distances = {i: min(abs(risk[i] - risk[j]) for j in selected) for i in remaining}
            largest = max(distances.values())
            if largest <= TOL:
                break
            tied = [i for i in remaining if distances[i] >= largest - TOL]
            chosen = min(tied, key=lambda i: (-growth[i], i))
        selected.append(chosen)
        extension.append(chosen)
    return sorted(selected, key=lambda i: (risk[i], growth[i], i)), extension


def objective(ids, growth, feasible, oracle):
    if not ids:
        return np.asarray([0., 0., 0.])
    best = np.max(np.where(feasible[:, ids], growth[:, ids], -np.inf), axis=1)
    covered = np.isfinite(best)
    regrets = oracle[covered] - best[covered]
    return np.asarray([covered.sum(), max(regrets, default=0.), regrets.mean() if len(regrets) else 0.])


def audit_fit(fitted, case, train_ids, protocol, weights, metrics, audit):
    train = np.asarray(train_ids) - 1
    profiles = protocol["clients"]
    alpha, floors = fitted["alpha"], fitted["liquidity_floors"]
    menu = fitted["menu"]
    ids = [item["candidate_index"] for item in menu]
    names = [item["name"] for item in menu]
    audit.check(len(ids) <= 3 and len(set(ids)) == len(ids), "shared_menu_unique_capacity", case)
    audit.check(len(set(names)) == len(names), "shared_menu_unique_names", case)
    audit.check(names == ["Conservative", "Balanced", "Growth"][:len(menu)], "shared_menu_risk_role_names", case)
    audit.check(set(fitted["assignments"]) == {p["name"] for p in profiles}, "all_profiles_have_assignment", case)
    audit.check(fitted["n_training"] == len(train), "training_denominator", case)
    for item in menu:
        audit.close(item["weights"], weights[item["candidate_index"]], "menu_target_matches_cache_id", case)
    growth, feasible, excesses, cvars = [], [], [], []
    choice_by_client = {}
    details = fitted["client_product_details"]
    audit.check(len(details) == len(profiles) * len(menu), "complete_client_product_detail_table", case)
    for profile in profiles:
        horizon, loss = profile["horizon"], profile["loss_tolerance"]
        mdd = metrics[horizon]["mdd"][train]
        breaches = mdd - loss > TOL
        counts = breaches.sum(axis=0)
        probabilities = counts / len(train)
        g = metrics[horizon]["log_growth"][train].mean(axis=0)
        excess = np.where(breaches, mdd - loss, 0).mean(axis=0)
        cvar = fractional_cvar(mdd)
        liquid = weights[:, 2] >= floors[profile["liquidity"]] - TOL
        permitted = (probabilities <= alpha + TOL) & liquid
        growth.append(g)
        feasible.append(permitted)
        excesses.append(excess)
        cvars.append(cvar)
        audit.check(fitted["oracle_candidate_count"][profile["name"]] == int(permitted.sum()),
                    "oracle_feasible_candidate_count", (case, profile["name"]))
        allowed = [i for i in ids if permitted[i]]
        chosen = None
        if allowed:
            highest = max(g[i] for i in allowed)
            tied = [i for i in allowed if g[i] >= highest - TOL]
            chosen = min(tied, key=lambda i: (excess[i], cvar[i], i))
        expected_assignment = "Review" if chosen is None else names[ids.index(chosen)]
        audit.check(fitted["assignments"][profile["name"]] == expected_assignment,
                    "client_choice_from_shared_feasible_menu", (case, profile["name"]))
        choice_by_client[profile["name"]] = chosen
        for name, i in zip(names, ids):
            matches = [row for row in details if row["client"] == profile["name"] and row["portfolio"] == name]
            audit.check(len(matches) == 1, "unique_client_product_detail", (case, profile["name"], name))
            if len(matches) != 1:
                continue
            row = matches[0]
            context = (case, profile["name"], name)
            expected_flags = {
                "candidate_index": i, "horizon": horizon, "loss_tolerance": loss,
                "n": len(train), "breach_count": int(counts[i]), "liquidity_pass": bool(liquid[i]),
                "sample_feasible": bool(permitted[i]), "chosen": chosen == i,
                "one_path_boundary": counts[i] == int(np.floor(len(train) * alpha + TOL))}
            for key, expected in expected_flags.items():
                audit.check(row[key] == expected, "training_detail_" + key, context)
            for key, expected in {"breach_probability": probabilities[i], "expected_excess": excess[i],
                                  "cvar90_mdd": cvar[i], "mean_log_growth": g[i]}.items():
                audit.close(row[key], expected, "training_detail_" + key, context)
    growth, feasible = np.asarray(growth), np.asarray(feasible)
    oracle_feasible = feasible.any(axis=1)
    oracle = np.asarray([max(g[ok], default=0.) for g, ok in zip(growth, feasible)])
    selection = fitted["selection"]
    audit.close(selection["oracle_growth"], oracle, "oracle_growth_train_only", case)
    audit.check(selection["oracle_feasible"] == oracle_feasible.tolist(), "oracle_has_feasible_target", case)
    identified = selection["identified_indices"]
    audit.close(selection["objective"], objective(identified, growth, feasible, oracle),
                "identified_menu_objective", case)
    completed = objective(ids, growth, feasible, oracle)
    audit.close(fitted["post_completion_objective"], completed, "completed_menu_objective", case)
    limits = selection["global_thresholds"]
    audit.check(completed[0] == limits["coverage"] and completed[1] <= limits["maximum_regret_upper"]
                and completed[2] <= limits["mean_regret_upper"], "completed_objective_within_frozen_global_bands", case)
    audit.check(fitted["post_completion_objective_verified"] is True, "recorded_completion_verified", case)
    ref_growth = metrics[15]["log_growth"][train].mean(axis=0)
    ref_risk = fractional_cvar(metrics[15]["mdd"][train])
    frontier = frontier_ids(ref_growth, ref_risk)
    expected_ids, extension = expected_completion(identified, ref_growth, ref_risk, frontier)
    completion = fitted["completion"]
    audit.check(completion["reference_frontier"] == frontier, "reference_frontier_train_only", case)
    audit.check(ids == expected_ids == completion["indices"], "reference_completion_and_label_order", case)
    audit.check(completion["extension_indices"] == extension, "completion_extension_ids", case)
    audit.check(completion["complete"] == (len(ids) == 3), "completion_capacity_flag", case)
    distinct = []
    for i in ids:
        if not distinct or ref_risk[i] - distinct[-1] > TOL:
            distinct.append(ref_risk[i])
    audit.check(completion["distinct_reference_risk_count"] == len(distinct)
                and completion["three_distinct_reference_risks"] == (len(distinct) >= 3),
                "completion_distinct_risk_flags", case)
    for item in menu:
        i = item["candidate_index"]
        audit.close(item["reference_mean_log_growth"], ref_growth[i], "menu_reference_growth_train_only", case)
        audit.close(item["reference_cvar90_mdd"], ref_risk[i], "menu_reference_cvar_train_only", case)
        audit.check(item["identified_by_profiles"] == (i in identified), "menu_role_origin", case)
    return choice_by_client


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Unexpected CSV boolean: {value!r}")
    return value == "True"


def inverse_cdf(values, probability):
    values = sorted(values)
    return values[max(0, int(np.ceil(probability * len(values))) - 1)]


def median(values):
    values = sorted(values)
    n = len(values)
    return (values[(n - 1) // 2] + values[n // 2]) / 2


def audit_oof(paths, summaries, fold_menus, protocol, metrics, choices, audit):
    fold_specs = {f["fold"]: f for f in protocol["validation"]["folds"]}
    profiles = {p["name"]: p for p in protocol["clients"]}
    fitted = {(case, f["fold"]): f for case, folds in fold_menus.items() for f in folds}
    group_rows, identities, auto_count, review_count = defaultdict(list), set(), 0, 0
    outcomes = ("candidate_index", "cagr", "log_growth", "mdd", "breach", "excess")
    for row in paths:
        case, client = row["case"], row["client"]
        fold, sid = int(row["fold"]), int(row["scenario_id"])
        identity = (case, client, sid)
        audit.check(identity not in identities, "unique_oof_case_client_scenario", identity)
        identities.add(identity)
        group_rows[case, client].append(row)
        spec, fit, profile = fold_specs[fold], fitted[case, fold], profiles[client]
        audit.check(sid in spec["test_ids"] and sid not in spec["train_ids"], "oof_test_only_membership", identity)
        audit.check(int(row["horizon"]) == profile["horizon"] and float(row["loss_tolerance"]) == profile["loss_tolerance"],
                    "oof_profile_horizon_and_loss", identity)
        audit.close(float(row["alpha"]), fit["alpha"], "oof_alpha", identity)
        audit.check(row["liquidity_case"] == case.split("__", 1)[1], "oof_liquidity_case", identity)
        audit.check(row["assignment"] == fit["assignments"][client], "oof_frozen_fold_assignment", identity)
        candidate = choices[case, fold][client]
        assigned = candidate is not None
        audit.check(boolean(row["assigned"]) == assigned, "oof_assigned_flag", identity)
        if not assigned:
            review_count += 1
            audit.check(row["assignment"] == "Review" and all(row[field] == "" for field in outcomes),
                        "review_outcomes_all_null", identity)
            continue
        auto_count += 1
        audit.check(all(row[field] != "" for field in outcomes), "automatic_outcomes_not_null", identity)
        # Pandas writes this nullable integer column as values such as "83.0".
        # Direct float equality also rejects a nonintegral candidate identity.
        audit.check(float(row["candidate_index"]) == candidate, "oof_frozen_fold_candidate", identity)
        horizon, loss = profile["horizon"], profile["loss_tolerance"]
        for field in ("cagr", "log_growth", "mdd"):
            audit.close(float(row[field]), metrics[horizon][field][sid - 1, candidate], "oof_replay_" + field, identity)
        mdd = metrics[horizon]["mdd"][sid - 1, candidate]
        breach = mdd - loss > TOL
        audit.check(boolean(row["breach"]) == breach, "oof_replay_breach", identity)
        audit.close(float(row["excess"]), mdd - loss if breach else 0., "oof_replay_excess", identity)
    expected_groups = {(case, client) for case in fold_menus for client in profiles}
    audit.check(set(group_rows) == expected_groups, "complete_oof_case_client_groups")
    summary_lookup = {(row["case"], row["client"]): row for row in summaries}
    audit.check(len(summary_lookup) == len(summaries) == len(expected_groups), "unique_complete_oof_summaries")
    for key, group in group_rows.items():
        audit.check(len(group) == 120 and {int(r["scenario_id"]) for r in group} == set(range(1, 121)),
                    "coverage_denominator_all_120", key)
        by_fold = Counter(int(r["fold"]) for r in group)
        audit.check(by_fold == Counter({i: 24 for i in fold_specs}), "each_fold_contributes_24_oof_paths", key)
        summary = summary_lookup[key]
        audit.close(float(summary["alpha"]), float(group[0]["alpha"]), "oof_summary_alpha", key)
        auto = [row for row in group if boolean(row["assigned"])]
        auto_folds = len({int(row["fold"]) for row in auto})
        expected_counts = {"total_test_paths": 120, "assigned_test_paths": len(auto),
                           "review_test_paths": 120 - len(auto), "auto_folds": auto_folds,
                           "review_folds": 5 - auto_folds}
        for field, value in expected_counts.items():
            audit.check(int(summary[field]) == value, "oof_summary_" + field, key)
        statistic_fields = ("breach_count", "breach_probability", "mean_log_growth", "median_cagr", "p10_cagr",
                            "median_mdd", "p90_mdd", "expected_excess", "internal_calibration_warning")
        if not auto:
            audit.check(all(summary[field] == "" for field in statistic_fields), "review_summary_outcomes_all_null", key)
            continue
        count = sum(boolean(row["breach"]) for row in auto)
        cagr, mdd = [float(r["cagr"]) for r in auto], [float(r["mdd"]) for r in auto]
        expected = {"breach_count": count, "breach_probability": count / len(auto),
                    "mean_log_growth": sum(float(r["log_growth"]) for r in auto) / len(auto),
                    "median_cagr": median(cagr), "p10_cagr": inverse_cdf(cagr, .1),
                    "median_mdd": median(mdd), "p90_mdd": inverse_cdf(mdd, .9),
                    "expected_excess": sum(float(r["excess"]) for r in auto) / len(auto)}
        for field, value in expected.items():
            audit.close(float(summary[field]), value, "oof_summary_" + field, key)
        audit.check(boolean(summary["internal_calibration_warning"]) == (count / len(auto) > float(summary["alpha"]) + TOL),
                    "oof_summary_calibration_warning", key)
    return {"oof_rows": len(paths), "distinct_case_client_scenario_identities": len(identities),
            "automatic_path_outcomes": auto_count, "review_path_outcomes_all_null": review_count,
            "case_client_denominators": len(group_rows)}


def run(audit):
    synthetic_checks(audit)
    artifacts = [PROTOCOL, OUT / "full_development_menus.json", OUT / "validation_fold_menus.json",
                 OUT / "validation_oof_paths.csv", OUT / "validation_oof_summary.csv", OUT / "fit_manifest.json",
                 OUT / "validation_manifest.json", OUT / "cache/step_0500__cost_002500.npz",
                 OUT / "cache/step_0500__cost_002500.json"]
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in artifacts}
    protocol = json.loads(PROTOCOL.read_text())
    audit.check(protocol["allowed_scenarios"] == list(range(1, 121)) and protocol["forbidden_new_final_access"] is True,
                "protocol_development_only_scope")
    cache_manifest = json.loads(artifacts[-1].read_text())
    family = protocol["candidate_family"]
    audit.check(cache_manifest["cost_rate"] == family["cost_rate"] == .0025
                and cache_manifest["step"] == family["weight_grid_step"] == .05
                and family["initial_wealth"] == 100. and family["initial_asset"] == "bills",
                "cache_execution_parameters_match_protocol")
    audit.check(cache_manifest["scenario_ids"] == list(range(1, 121)), "cache_development_only_scope")
    audit.check(cache_manifest["npz_sha256"] == sha(artifacts[-2]), "cache_npz_manifest_hash")
    with np.load(artifacts[-2], allow_pickle=False) as archive:
        cache = {k: archive[k] for k in archive.files}
    ids = cache["scenario_ids"]
    audit.check(np.array_equal(ids, np.arange(1, 121)), "cache_scenario_axis_ids")
    weights = cache["weights"]
    independent_grid = [[i / 20, j / 20, (20 - i - j) / 20]
                        for i in range(21) for j in range(21 - i)] + [[1 / 3] * 3]
    independent_grid = np.asarray(sorted(independent_grid))
    audit.close(weights, independent_grid, "fixed_232_simplex_grid", atol=1e-15, rtol=0)
    audit.check(cache_manifest["candidate_count"] == len(weights) == 232, "fixed_candidate_family_count")
    returns, inflation, loaded_ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv",
                                               range(1, 121), allow_final=False)
    audit.check(np.array_equal(loaded_ids, ids), "market_cache_scenario_axis_alignment")
    development_hash = hashlib.sha256()
    for value in (loaded_ids, returns, inflation):
        development_hash.update(np.ascontiguousarray(value).tobytes())
    audit.check(development_hash.hexdigest() == cache_manifest["development_arrays_sha256"], "development_input_array_hash")
    for relative, digest in cache_manifest["source_hashes"].items():
        audit.check(sha(ROOT / relative) == digest, "cache_source_hash", relative)
    full = json.loads((OUT / "full_development_menus.json").read_text())
    folds = json.loads((OUT / "validation_fold_menus.json").read_text())
    source_provenance = {}
    for stage in ("fit", "validation"):
        manifest = json.loads((OUT / f"{stage}_manifest.json").read_text())
        for relative, digest in manifest["sources"].items():
            actual = sha(ROOT / relative)
            source_provenance[relative] = actual
            audit.check(actual == digest, "frozen_" + stage + "_source_hash", relative)
        if stage == "fit":
            audit.check(manifest["menus_sha256"] == sha(OUT / "full_development_menus.json"), "fit_menu_manifest_hash")
            audit.check(manifest["scenario_ids"] == list(range(1, 121)), "fit_manifest_development_ids")
    metrics = independent_replay(returns, inflation, weights, cache_manifest["cost_rate"], protocol["horizons"], audit)
    for horizon, fields in metrics.items():
        for field, values in fields.items():
            units = field in ("terminal_real", "fees", "moved")
            audit.close(cache[f"H{horizon}__{field}"], values, "cache_replay_" + field, horizon,
                        atol=2e-9 if units else 1e-12, rtol=2e-13 if units else 1e-12)
    expected_cases = {f"alpha_{round(a * 100):02d}__{name}" for a in protocol["alpha_cases"]
                      for name in protocol["liquidity_cases"]}
    audit.check(set(full) == set(folds) == expected_cases, "all_nine_predeclared_cases")
    fold_specs = {f["fold"]: f for f in protocol["validation"]["folds"]}
    all_tests = []
    for spec in fold_specs.values():
        train, test = spec["train_ids"], spec["test_ids"]
        audit.check(len(train) == len(set(train)) == 96 and len(test) == len(set(test)) == 24
                    and not set(train) & set(test) and set(train) | set(test) == set(range(1, 121)),
                    "protocol_fold_96_24_disjoint_partition", spec["fold"])
        all_tests.extend(test)
    audit.check(Counter(all_tests) == Counter(range(1, 121)), "each_development_path_tested_exactly_once")
    all_menu_targets = set()
    choices = {}
    for case, fitted in full.items():
        liquidity = fitted["liquidity_case"]
        audit.check(fitted["liquidity_floors"] == protocol["liquidity_cases"][liquidity]
                    and case == f"alpha_{round(fitted['alpha'] * 100):02d}__{liquidity}",
                    "full_case_policy_alignment", case)
        audit_fit(fitted, case + "/full", list(range(1, 121)), protocol, weights, metrics, audit)
        all_menu_targets.update(item["candidate_index"] for item in fitted["menu"])
        audit.check(len(folds[case]) == 5 and {f["fold"] for f in folds[case]} == set(fold_specs),
                    "complete_five_fold_menu_fits", case)
        for fitted_fold in folds[case]:
            number = fitted_fold["fold"]
            spec = fold_specs[number]
            audit.check(fitted_fold["train_ids"] == spec["train_ids"] and fitted_fold["test_ids"] == spec["test_ids"],
                        "stored_fold_matches_predeclared_split", (case, number))
            audit.check(fitted_fold["alpha"] == fitted["alpha"]
                        and fitted_fold["liquidity_floors"] == fitted["liquidity_floors"], "fold_case_policy_alignment", (case, number))
            choices[case, number] = audit_fit(fitted_fold, case + f"/fold{number}", spec["train_ids"],
                                             protocol, weights, metrics, audit)
            all_menu_targets.update(item["candidate_index"] for item in fitted_fold["menu"])
    counts = audit_oof(read_rows(OUT / "validation_oof_paths.csv"), read_rows(OUT / "validation_oof_summary.csv"),
                       folds, protocol, metrics, choices, audit)
    counts.update({"scenarios": len(ids), "fixed_targets": len(weights), "horizons": len(protocol["horizons"]),
                   "independent_scenario_target_paths": len(ids) * len(weights),
                   "annual_execution_identities": len(ids) * len(weights) * returns.shape[1],
                   "cache_metric_values": sum(v.size for fields in metrics.values() for v in fields.values()),
                   "full_case_fits": len(full), "fold_case_fits": sum(map(len, folds.values())),
                   "client_training_fit_identities": (len(full) + sum(map(len, folds.values()))) * len(protocol["clients"]),
                   "distinct_targets_appearing_in_full_or_fold_menus": len(all_menu_targets),
                   "full_and_fold_menu_target_ids": sorted(all_menu_targets)})
    audit.check({str(p.relative_to(ROOT)): sha(p) for p in artifacts} == provenance, "input_artifacts_unchanged_during_audit")
    audit.check({relative: sha(ROOT / relative) for relative in source_provenance} == source_provenance,
                "frozen_sources_unchanged_during_audit")
    audit.check({relative: sha(ROOT / relative) for relative in cache_manifest["source_hashes"]}
                == cache_manifest["source_hashes"], "cache_sources_unchanged_during_audit")
    return {"identities": counts, "input_artifact_sha256": provenance,
            "source_sha256": source_provenance, "development_input_arrays_sha256": development_hash.hexdigest(),
            "cache_source_sha256": cache_manifest["source_hashes"]}


def main():
    audit = Audit()
    report = {"audit_version": "independent-development-numerical-audit-v1",
              "completed_utc": None, "audit_script_sha256": sha(Path(__file__)),
              "scope": "Only scenarios 1–120; independent bisection accounting and training/OOF artifact reconstruction.",
              "method": "80-step vectorized fee bisection; independent G/CAGR/MDD/volatility and fractional-tail arithmetic; no production executor/simulator/metrics/selector imports.",
              "not_established": ["Population or real-market safety", "Validity of normative alpha/liquidity/menu preferences",
                                  "A fresh unseen holdout", "Global menu optimality on actual data (not re-enumerated here)",
                                  "Historical runtime provenance beyond hashes, stored splits and reconstructed train-only statistics"],
              "final_scenarios_loaded": False}
    try:
        report.update(run(audit))
    except Exception as exc:
        audit.failures.append({"check": "audit_execution_exception", "type": type(exc).__name__,
                               "message": str(exc), "traceback": traceback.format_exc()})
    report.update({"passed": not audit.failures, "check_invocations": dict(audit.checks),
                   "total_check_invocations": sum(audit.checks.values()),
                   "numerical_errors": audit.errors, "failures": audit.failures,
                   "completed_utc": datetime.now(timezone.utc).isoformat()})
    destination = OUT / "independent_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": report["total_check_invocations"],
                      "failures": report["failures"], "identities": report.get("identities"),
                      "audit": str(destination)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
