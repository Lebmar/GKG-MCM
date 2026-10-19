"""Independent selection/accounting/Q-subset audit of the price learner.

Training candidate metrics are independently aggregated, not represented as
an independent replay of their unrecorded complete training decision trees.
All saved selected actual transitions use our own100-step fee bisection.
No reviewed price/fan/runner/selector/metric implementation is imported.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_fan_rollout_execution import reference_actual, scalar_fan
from audit_review_pending_execution import independent_metrics, independent_cvar

OUT = ROOT / "results/model_review/risk_priced_fan_execution_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PENDING = ROOT / "results/model_review/review_pending_execution_v1"
FAN = ROOT / "results/model_review/fan_rollout_execution_v1"
PROTOCOL = ROOT / "docs/risk_priced_fan_execution_protocol.json"
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill")
FINANCIAL = (*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing")
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
POLICIES = ("annual_then_pending", "risk_priced_learner")
TOL = 1e-12


def readj(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    manifest = readj(OUT / "manifest.json")
    paths = [OUT / name for name in manifest["outputs_sha256"]]
    paths += [OUT / "manifest.json", Path(__file__).resolve(), ROOT / "scripts/audit_fan_rollout_execution.py", ROOT / "scripts/audit_review_pending_execution.py"]
    paths += [ROOT / name for name in manifest["sources"]]
    paths += [ROOT / name for name in manifest["input_hashes"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def choice(gh, gr, kh, kr, n, price, horizon):
    growth = gr - gh
    if price is None:
        return ("restore" if kr < kh or (kr == kh and growth > TOL) else "hold"), None
    score = growth - horizon * price * ((kr - kh) / n)
    return ("restore" if score > TOL else "hold"), score


def main():
    before = snapshot()
    manifest, protocol = readj(OUT / "manifest.json"), readj(PROTOCOL)
    selections, fits = readj(OUT / "context_selections.json"), readj(STRICT / "nested_context_refits.json")
    profiles = readj(ROOT / "docs/positive_asset_specification_protocol.json")["clients"]
    rows = pd.read_csv(OUT / "client_paths.csv")
    candidates = pd.read_csv(OUT / "training_candidate_paths.csv")
    summaries = pd.read_csv(OUT / "training_candidate_summary.csv")
    try:
        gates = pd.read_csv(OUT / "decision_gates.csv")
    except pd.errors.EmptyDataError:
        gates = pd.DataFrame(columns=["context", "client", "scenario_id", "year"])
    pending = pd.read_csv(PENDING / "client_path_metrics.csv")
    old_fan = pd.read_csv(FAN / "client_paths.csv")
    checks, errors, q_results, q_differences = 0, {}, [], []
    replay = {"actual_arm_paths": 0, "actual_decisions": 0}
    decisions = {"finite_active": 0, "finite_restore": 0, "lex_active": 0, "lex_restore": 0}
    selected_records = []

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)

    def close(actual, expected, label, *, atol=3e-11, rtol=3e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape, label + "/shape")
        if a.dtype.kind in "bUiS" or b.dtype.kind in "bUiS":
            check(np.array_equal(a, b), label + "/exact")
        else:
            difference = np.abs(a - b); finite = np.isfinite(difference)
            field = label.split("/")[-1]
            errors[field] = max(errors.get(field, 0.), float(difference[finite].max(initial=0)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + "/numeric")

    def metadata(row, order, horizon):
        check(row.execution_candidate_order == order and row.execution_candidate_id == (f"eta_{PRICES[order]:g}" if order < 8 else "risk_first"), "candidate_identity")
        check(row.price_kind == ("finite" if order < 8 else "risk_first"), "candidate_price_kind")
        if order < 8:
            close(row.annual_price, PRICES[order], "candidate/annual_price")
            close(row.original_H_total_price, horizon * PRICES[order], "candidate/original_H_total_price")
        else:
            check(pd.isna(row.annual_price) and pd.isna(row.original_H_total_price), "risk_first_no_infinite_JSON_price")

    check(tuple(protocol["candidate_family"]["annual_prices"]) == PRICES[:8], "frozen_price_grid")
    check(manifest["unselected_outer_prices_evaluated"] is False and manifest["product_or_gamma_refit"] is False, "stated_execution_scope")
    for name, expected in protocol["frozen_input_sha256"].items():
        check(sha(ROOT / name) == expected, "frozen_input/" + name)
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation):
        digest.update(np.ascontiguousarray(value).tobytes())
    check(ids.tolist() == list(range(1, 121)) and digest.hexdigest() == manifest["development_array_sha256"], "guarded_development_arrays")
    cache_path = ROOT / "results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        annual_mdd = {profile["horizon"]: archive[f'H{profile["horizon"]}__mdd'].copy() for profile in profiles}
    rkeys, ckeys = ["context", "client", "scenario_id", "policy"], ["context", "client", "execution_candidate_order", "scenario_id"]
    check(len(rows) == 2400 and not rows.duplicated(rkeys).any(), "all_original_main_rows")
    check(len(candidates) == 27000 and not candidates.duplicated(ckeys).any(), "all_nine_training_candidate_rows")
    check(len(summaries) == 270 and not summaries.duplicated(ckeys[:-1]).any(), "all_training_candidate_summaries")
    check(candidates.scope.eq("training_support").all(), "training_rows_never_mislabeled_OOF")
    check(rows.loc[~rows.assigned, list(FINANCIAL)].isna().all().all(), "main_Review_financial_null")
    check(candidates.loc[~candidates.assigned, list(FINANCIAL)].isna().all().all(), "training_frozen_Review_financial_null")
    check(not gates.duplicated(["context", "client", "scenario_id", "year"]).any(), "unique_selected_decisions")
    rl, gl = rows.set_index(rkeys).sort_index(), gates.set_index(["context", "client", "scenario_id", "year"]).sort_index()
    with np.load(OUT / "actual_traces.npz", allow_pickle=False) as archive:
        for context, fit in fits.items():
            ctx = selections[context]
            training_ids = fit["training_ids"]
            evaluation_ids = training_ids if context == "full_development" else fit["outer_test_ids"]
            scope = "full_development" if context == "full_development" else "outer_oof"
            check(ctx["training_ids"] == training_ids and ctx["evaluation_ids"] == evaluation_ids, "exact_train_eval_context")
            check(context == "full_development" or not set(training_ids) & set(evaluation_ids), "outer_train_test_disjoint")
            tx = np.asarray(training_ids) - 1
            for profile in profiles:
                client, h, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
                assignment = fit["effective_assignments"][client]
                selected = ctx["profiles"][client]
                family = candidates.loc[candidates.context.eq(context) & candidates.client.eq(client)]
                family_summary = summaries.loc[summaries.context.eq(context) & summaries.client.eq(client)].set_index("execution_candidate_order", drop=False)
                check(len(family) == 9 * len(training_ids) and set(family.execution_candidate_order) == set(range(9)), "training_family_complete")
                check(selected["risk_assignment"] == assignment and selected["n_training"] == len(training_ids), "frozen_risk_assignment_and_denominator")
                if assignment == "Review":
                    check(selected["risk_assigned"] is False and selected["selected_candidate_order"] is None
                          and selected["execution_status"] == "risk_review" and selected["candidate_summaries"] == [], "frozen_risk_Review_selection")
                    check(not family.assigned.any() and family.risk_assignment.eq("Review").all(), "frozen_training_Review_not_readmitted")
                    order, target, item = None, None, None
                else:
                    item = next(product for product in fit["raw_fit"]["menu"] if product["name"] == assignment)
                    target = np.asarray(item["weights"])
                    check(np.min(target) >= .05 - TOL, "strict_three_asset_target")
                    reference_count = int((annual_mdd[h][tx, item["candidate_original232_index"]] - loss > TOL).sum())
                    cap = min(reference_count, int(math.floor(.10 * len(training_ids) + TOL)))
                    check(selected["reference_training_breach_count"] == reference_count and selected["training_breach_cap"] == cap
                          and selected["absolute_alpha_count_cap"] == int(math.floor(.10 * len(training_ids) + TOL)), "same_context_training_risk_cap")
                    performance = []
                for candidate_order in range(9):
                    group = family.loc[family.execution_candidate_order.eq(candidate_order)]
                    summary = family_summary.loc[candidate_order]
                    check(len(group) == len(training_ids) and set(group.scenario_id) == set(training_ids), "each_candidate_exact_training_ids")
                    check(group.context_evaluation_scope.eq(scope).all(), "training_evaluation_scope_metadata")
                    metadata(summary, candidate_order, h)
                    for record in group.itertuples(index=False):
                        metadata(record, candidate_order, h)
                    if assignment == "Review":
                        check(pd.isna(summary.breach_count) and pd.isna(summary.mean_log_growth) and pd.isna(summary.training_eligible) and not summary.selected,
                              "frozen_Review_candidate_summary_null")
                        continue
                    check(group.assigned.all() and group.risk_assignment.eq(assignment).all()
                          and group.candidate_local_index.eq(item["candidate_local_index"]).all()
                          and group.candidate_original232_index.eq(item["candidate_original232_index"]).all(), "training_candidate_product_identity")
                    check((group.terminal_real > 0).all() and group.mdd.between(0, 1).all(), "training_financial_ranges")
                    close(group.cagr.to_numpy(), np.expm1(group.log_growth.to_numpy()), "training_algebra/cagr", atol=3e-12)
                    close(group.terminal_real.to_numpy(), 100 * np.exp(h * group.log_growth.to_numpy()), "training_algebra/terminal_real")
                    close(group.nominal_fees.to_numpy(), .0025 * group.nominal_moved.to_numpy(), "training_algebra/fees")
                    close(group.excess.to_numpy(), np.maximum(group.mdd.to_numpy() - loss, 0), "training_algebra/excess")
                    check(np.array_equal(group.breach.to_numpy(bool), group.mdd.to_numpy() - loss > TOL), "training_MDD_everbreach")
                    check(group.loc[~group.breach.astype(bool), "first_crossing"].isna().all()
                          and group.loc[group.breach.astype(bool), "first_crossing"].between(1, h).all(), "training_crossing_or_null")
                    stats = {"mean_log_growth": group.log_growth.mean(), "breach_count": int(group.breach.astype(bool).sum()),
                             "mean_nominal_fees": group.nominal_fees.mean(), "mean_nominal_moved": group.nominal_moved.mean(), "mean_trade_fraction": group.trade_fraction.mean()}
                    evidence = selected["candidate_summaries"][candidate_order]
                    for field, value in stats.items():
                        close(summary[field], value, "training_summary/" + field)
                        close(evidence[field], value, "selection_evidence/" + field)
                    eligible = stats["breach_count"] <= cap
                    check(summary.n_training == len(training_ids) and summary.training_breach_cap == cap
                          and summary.training_eligible == eligible and evidence["training_eligible"] == eligible, "complete_policy_training_eligibility")
                    performance.append((candidate_order, stats["mean_log_growth"], eligible))
                if assignment != "Review":
                    eligible = [(candidate_order, growth) for candidate_order, growth, admit in performance if admit]
                    maximum = max((growth for _, growth in eligible), default=None)
                    order = next((candidate_order for candidate_order, growth in eligible if growth >= maximum - TOL), None)
                    check(selected["selected_candidate_order"] == order, "global_max_band_price_selection")
                    check(selected["execution_status"] == ("automated" if order is not None else "execution_review"), "execution_admission_status")
                    check(selected["execution_review_reason"] == (None if order is not None else "execution_family_no_admissible_policy"), "execution_admission_reason")
                    check(np.array_equal(family_summary.selected.to_numpy(bool), np.asarray([i == order for i in range(9)])), "selected_candidate_flags")
                    if order is None:
                        check(selected["selected_candidate"] is None, "execution_Review_selected_price_null")
                    else:
                        metadata(pd.Series(selected["selected_candidate"]), order, h)
                    selected_records.append({"context": context, "client": client, "reference_training_breach_count": reference_count,
                        "training_cap": cap, "eligible_candidates": [i for i, _, eligible in performance if eligible], "selected_order": order,
                        "selected_annual_price": PRICES[order] if order is not None else None})
                actual_paths = {}
                for policy in POLICIES:
                    group = rows.loc[rows.context.eq(context) & rows.client.eq(client) & rows.policy.eq(policy)]
                    served = assignment != "Review" and (policy == POLICIES[0] or order is not None)
                    check(len(group) == len(evaluation_ids) and set(group.scenario_id) == set(evaluation_ids)
                          and group.assigned.eq(served).all() and group.risk_assigned.eq(assignment != "Review").all(), "main_coverage_and_original_risk_assignment")
                    check(group.risk_assignment.eq(assignment).all(), "main_original_risk_assignment_retained")
                    prefix = f"{context}__{client}__{policy}__"
                    if not served:
                        check(not any(key.startswith(prefix) for key in archive.files), "no_fabricated_Review_account")
                        if policy == POLICIES[1]:
                            check(group[["execution_candidate_order", "execution_candidate_id", "annual_price", "original_H_total_price"]].isna().all().all(), "Review_price_null")
                        continue
                    saved = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                    check(np.array_equal(saved["scenario_ids"], evaluation_ids), "actual_scenario_order")
                    if policy == POLICIES[1]:
                        for record in group.itertuples(index=False):
                            metadata(record, order, h)
                    for i, sid in enumerate(evaluation_ids):
                        def action(t, x, cpi, peak, worst, bad):
                            active = t > 0 and not bad
                            if t == 0:
                                expected = "restore"
                            elif bad:
                                expected = "hold"
                            elif policy == POLICIES[0]:
                                expected = "restore"
                            else:
                                kh, kr, n = int(saved["qbad_hold_count"][i, t]), int(saved["qbad_restore_count"][i, t]), int(saved["forecast_n"][i, t])
                                check(n == len(training_ids) and 0 <= kh <= n and 0 <= kr <= n, "actual_gate_integer_risk_denominator")
                                expected, score = choice(saved["qg_hold"][i, t], saved["qg_restore"][i, t], kh, kr, n, PRICES[order], h)
                                kind = "finite" if order < 8 else "lex"
                                decisions[kind + "_active"] += 1
                                decisions[kind + "_restore"] += expected == "restore"
                                if score is None:
                                    check(np.isnan(saved["predicted_score"][i, t]), "risk_first_score_is_null_not_infinity")
                                else:
                                    close(saved["predicted_score"][i, t], score, "actual_score/original_H_score", atol=3e-12, rtol=0)
                                    check(expected != "restore" or score > TOL, "restoration_positive_risk_adjusted_net_benefit")
                                check(saved["predicted_breach_count_delta_restore"][i, t] == kr - kh, "actual_signed_risk_count_change")
                            check(saved["actions"][i, t] == expected, "actual_selected_rule_and_timing")
                            if policy == POLICIES[1]:
                                for field in ("qg_hold", "qg_restore"):
                                    check(np.isfinite(saved[field][i, t]) if active else np.isnan(saved[field][i, t]), "active_Q_mask")
                                for field in ("qbad_hold_count", "qbad_restore_count", "forecast_n"):
                                    check(saved[field][i, t] >= 0 if active else saved[field][i, t] == -1, "active_count_mask")
                            return expected
                        path = reference_actual(returns[sid - 1, :h], inflation[sid - 1, :h], target, loss, action)
                        replay["actual_arm_paths"] += 1; replay["actual_decisions"] += h
                        actual_paths[(sid, policy)] = path
                        for field in path:
                            if field in saved:
                                close(saved[field][i], path[field], "actual_trace/" + field)
                        record = rl.loc[(context, client, sid, policy)]
                        own = independent_metrics(path, loss)
                        own["trade_fraction"] = own.pop("actual_trade_fraction")
                        own["target_execution_fraction"] = float(np.mean(path["target_execution"]))
                        for field in (*METRICS, *ACCOUNTS, "breach", "excess"):
                            close(record[field], own[field], "actual_metrics/" + field)
                        check(pd.isna(record.first_crossing) if path["first_crossing"] < 0 else record.first_crossing == path["first_crossing"], "actual_first_crossing_or_null")
                        if policy == POLICIES[0]:
                            old = pending.loc[pending.context.eq(context) & pending.client.eq(client) & pending.scenario_id.eq(sid) & pending.arm.eq("pending_hold")]
                            check(len(old) == 1, "frozen_pending_reference_identity")
                            for field in (*METRICS, "nominal_fees", "nominal_moved"):
                                close(record[field], old.iloc[0][field], "frozen_pending/" + field)
                        else:
                            gs = gl.loc[(context, client, sid)]
                            check(list(gs.index) == list(range(1, h + 1)), "selected_CSV_decisions_complete_H")
                            for t in range(h):
                                gate = gs.loc[t + 1]
                                metadata(gate, order, h)
                                check(gate.action == path["actions"][t] and gate.actual_trade == path["trades"][t]
                                      and gate.pending_at_decision == path["pending_at_decision"][t], "CSV_gate_action_flags")
                                close(gate.nominal_fee, path["fees"][t], "CSV_gate/fee")
                                close(gate.nominal_moved, path["moved"][t], "CSV_gate/moved")
                                active = t > 0 and not path["ever_breached"][t]
                                for field in ("qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "forecast_n"):
                                    if active:
                                        close(gate[field], saved[field][i, t], "CSV_gate/" + field)
                                    else:
                                        check(pd.isna(gate[field]), "CSV_inactive_Q_null")
                                if active:
                                    growth = saved["qg_restore"][i, t] - saved["qg_hold"][i, t]
                                    probability = (int(saved["qbad_restore_count"][i, t]) - int(saved["qbad_hold_count"][i, t])) / len(training_ids)
                                    close(gate.unpriced_growth_advantage, growth, "CSV_gate/unpriced_growth")
                                    close(gate.predicted_breach_probability_change, probability, "CSV_gate/probability_change")
                                    if order < 8:
                                        close(gate.computed_score_advantage, growth - h * PRICES[order] * probability, "CSV_gate/score")
                                    else:
                                        check(pd.isna(gate.computed_score_advantage), "CSV_risk_first_score_null")
                                    check(gate.restore_sacrifices_unpriced_growth == (gate.action == "restore" and growth < -TOL), "CSV_growth_sacrifice_label")
                sid = min(evaluation_ids)
                for year in sorted({2, h // 2 + 1, h}):
                    identity = {"context": context, "client": client, "scenario_id": sid, "year": year}
                    if assignment == "Review" or order is None:
                        q_results.append({**identity, "status": "auxiliary_frozen_risk_Review_skip" if assignment == "Review" else "auxiliary_execution_Review_skip"})
                        continue
                    path = actual_paths[(sid, POLICIES[1])]
                    gate = gl.loc[(context, client, sid, year)]
                    if pd.isna(gate.qg_hold):
                        check(path["ever_breached"][year - 1], "prospective_Q_skip_pending")
                        q_results.append({**identity, "status": "skipped_pending_no_replacement"})
                        continue
                    fan = scalar_fan(path["holdings"][year - 1], path["cpi"][year - 1], path["real_peak"][year - 1], path["running_mdd"][year - 1],
                        target, loss, returns[tx, :h], inflation[tx, :h], year - 1)
                    hold, restore = fan["hold"], fan["restore"]
                    close(gate.qg_hold, hold["mean"], "Q_subset/qg_hold", atol=3e-12, rtol=0)
                    close(gate.qg_restore, restore["mean"], "Q_subset/qg_restore", atol=3e-12, rtol=0)
                    check(gate.qbad_hold_count == hold["count"] and gate.qbad_restore_count == restore["count"], "Q_subset_integer_terminal_risk")
                    own_action, own_score = choice(hold["mean"], restore["mean"], hold["count"], restore["count"], len(training_ids), PRICES[order], h)
                    if own_score is not None:
                        close(gate.computed_score_advantage, own_score, "Q_subset/score", atol=3e-12, rtol=0)
                    if gate.action != own_action:
                        q_differences.append({**identity, "saved": gate.action, "independent": own_action, "independent_score": own_score})
                    q_results.append({**identity, "status": "audited" if gate.action == own_action else "numeric_action_review_needed", "selected_order": order,
                        "qg_hold": hold["mean"], "qg_restore": restore["mean"], "bad_hold": hold["count"], "bad_restore": restore["count"],
                        "training_paths": len(training_ids), "independent_score": own_score, "independent_action": own_action})

    # Eta0 full120 is also available as a genuinely frozen earlier artifact.
    zero = candidates.loc[candidates.context.eq("full_development") & candidates.execution_candidate_order.eq(0) & candidates.assigned]
    old = old_fan.loc[old_fan.context.eq("full_development") & old_fan.policy.eq("growth_only_rollout") & old_fan.assigned].set_index(["client", "scenario_id"])
    for record in zero.itertuples(index=False):
        prior = old.loc[(record.client, record.scenario_id)]
        for field in (*METRICS, *ACCOUNTS, "breach", "excess"):
            close(getattr(record, field), prior[field], "frozen_full_eta0/" + field)
    check(len(zero) == 360, "all_full_eta0_served_rows")
    check(len(q_results) == 90, "all_prospective_Q_and_auxiliary_Review_identities")
    for (scope, client, policy), group in rows.groupby(["scope", "client", "policy"]):
        check(len(group) == 120 and set(group.scenario_id) == set(range(1, 121)), "full_OOF_original120_denominators")
    expected_decisions = int(rows.loc[rows.assigned & rows.policy.eq(POLICIES[1]), "horizon"].sum())
    check(len(gates) == expected_decisions, "no_missing_or_extra_selected_decision")
    check(replay["actual_arm_paths"] == int(rows.assigned.sum()) and replay["actual_decisions"] == int(rows.loc[rows.assigned, "horizon"].sum()), "complete_selected_and_reference_replay")
    all_summary = pd.read_csv(OUT / "summary.csv")
    for record in all_summary.itertuples(index=False):
        group = rows.loc[rows.scope.eq(record.scope) & rows.client.eq(record.client) & rows.policy.eq(record.policy)]
        auto = group.loc[group.assigned]
        check(record.total_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "summary_coverage")
        if not len(auto):
            check(pd.isna(record.mean_log_growth) and pd.isna(record.breach_probability), "summary_Review_null")
            continue
        stats = {"mean_log_growth": auto.log_growth.mean(), "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
                 "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(auto.mdd),
                 "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": auto.breach.astype(bool).mean(), "mean_excess": auto.excess.mean(),
                 **{"mean_" + field: auto[field].mean() for field in ACCOUNTS}}
        for field, value in stats.items():
            close(getattr(record, field), value, "summary/" + field)
    coverage = pd.read_csv(OUT / "coverage.csv").set_index(["scope", "client", "policy"]).sort_index()
    expected_coverage = all_summary.set_index(["scope", "client", "policy"]).sort_index()[coverage.columns]
    check(coverage.equals(expected_coverage), "coverage_matches_full_summary")
    pair = pd.read_csv(OUT / "paired_paths.csv")
    pl = pair.set_index(["context", "client", "scenario_id"]).sort_index()
    check(len(pair) == 1200 and not pair.duplicated(["context", "client", "scenario_id"]).any(), "all_original_pair_identities")
    for key, group in rows.groupby(["context", "client", "scenario_id"]):
        a, b = [group.loc[group.policy.eq(policy)].iloc[0] for policy in POLICIES]
        observed = pl.loc[key]
        common = bool(a.assigned and b.assigned)
        check(observed.common_auto == common and observed.learner_assigned == b.assigned and observed.reference_assigned == a.assigned, "paired_common_auto_admission")
        for field in (*METRICS, *ACCOUNTS, "excess", "breach"):
            if common:
                close(observed["delta_" + field], float(b[field]) - float(a[field]), "pair_delta/" + field)
            else:
                check(pd.isna(observed["delta_" + field]), "unserved_pair_no_imputed_gain")
    for record in pd.read_csv(OUT / "paired_summary.csv").itertuples(index=False):
        group = pair.loc[pair.scope.eq(record.scope) & pair.client.eq(record.client)]
        common = group.loc[group.common_auto]
        check(record.total_original_pairs == 120 and record.learner_auto_n == int(group.learner_assigned.sum())
              and record.reference_auto_n == int(group.reference_assigned.sum()) and record.common_auto_n == len(common), "paired_summary_original_and_common_denominators")
        for field in pair.columns:
            if field.startswith("delta_"):
                if len(common):
                    close(getattr(record, "mean_" + field), common[field].mean(), "paired_summary/" + field)
                else:
                    check(pd.isna(getattr(record, "mean_" + field)), "paired_summary_unserved_null")
    for kind in ("sources", "input_hashes"):
        for name, expected in manifest[kind].items():
            check(sha(ROOT / name) == expected, kind + "/" + name)
    for name, expected in manifest["outputs_sha256"].items():
        check(sha(OUT / name) == expected, "original_output/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_hashes"] == manifest["input_hashes_at_end"], "original_start_end_hashes")
    after = snapshot(); check(before == after, "independent_source_artifact_stability")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not q_differences else "numerical_action_review_needed",
        "checks_passed": checks, "maximum_errors": errors, "replay_counts": replay, "decision_counts": decisions,
        "training_candidate_rows": len(candidates), "training_candidate_summaries": len(summaries), "served_context_price_selections": selected_records,
        "client_rows": len(rows), "assigned_reference_rows": int(rows.loc[rows.policy.eq(POLICIES[0]), "assigned"].sum()),
        "assigned_learner_rows": int(rows.loc[rows.policy.eq(POLICIES[1]), "assigned"].sum()),
        "Q_subset_results": q_results, "Q_subset_audited": sum(item["status"] == "audited" for item in q_results),
        "Q_subset_pending_skipped": sum(item["status"] == "skipped_pending_no_replacement" for item in q_results), "Q_subset_action_differences": q_differences,
        "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
        "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(), "final_scenarios_read": False,
        "independence": "Own100-step scalar fee bisection/actual state replay, scalar fan current action then B, path metrics/tails and independent CSV selection aggregation; no reviewed implementation imported",
        "limitations": ["Training candidate metrics are independently aggregated/algebra-checked; their unrecorded complete training decision trees were not independently replayed",
            "All saved selected actual/reference transitions are replayed; independent future fan Q recomputation covers only the prospective fixed subset",
            "Static annual cache is used only to reconstruct each original training reference breach count, justified by the independently audited first-breach coupling",
            "Prior full120 eta0 metrics are compared to the old frozen growth-only artifact; outer training eta0 trace equivalence is a runner assertion, not a separately stored immutable trace",
            "Risk cap/price selection is an empirical development screen, not a probability or dual certificate; no unselected outer candidate or final returns are evaluated",
            "Financial outcomes condition on automatic admission; original120 coverage and common-auto pairs are retained and must be considered together",
            "No price/menu/gamma retuning or final adoption recommendation"], "original_manifests_modified": False}
    destination = OUT / "independent_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_passed", "replay_counts", "decision_counts", "Q_subset_audited", "Q_subset_pending_skipped", "Q_subset_action_differences", "assigned_reference_rows", "assigned_learner_rows", "maximum_errors")}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
