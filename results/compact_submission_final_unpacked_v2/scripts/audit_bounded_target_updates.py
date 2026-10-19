"""Independent accounting, target state, selection and four-action Q audit.

The reference reuses only our own scalar100-step fee/state implementation.
Training summaries are independently reconstructed from saved path metrics;
unrecorded eighteen-family training decision trees are not claimed replayed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_fan_rollout_execution import reference_actual, scalar_fan
from audit_review_pending_execution import independent_metrics, independent_cvar

OUT = ROOT / "results/model_review/bounded_target_updates_v1"
PARENT = ROOT / "results/model_review/risk_priced_fan_execution_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PROTOCOL = ROOT / "docs/bounded_target_update_protocol.json"
POLICIES = ("fixed_target_learner", "expanded_target_learner")
ROLES = ("Conservative", "Balanced", "Growth")
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity", "terminal_bond", "terminal_bill")
H, LOSS, TOL = 30, .40, 1e-12


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


def independent_choice(qg, counts, available, n, price, current):
    """One global comparison band after strict admissible-benefit filtering."""
    qg, counts, available = np.asarray(qg), np.asarray(counts), np.asarray(available, bool)
    priority = [0, current + 1, *[a for a in range(1, 4) if a != current + 1]]
    scores = np.full(4, np.nan)
    if price is not None:
        scores[available] = qg[available] - qg[0] - H * price * ((counts[available].astype(float) - float(counts[0])) / n)
        eligible = available & (scores > TOL)
        maximum = max(scores[eligible], default=None)
        action = next((a for a in priority if eligible[a] and scores[a] >= maximum - TOL), 0)
    else:
        minimum = min(counts[available])
        eligible = available & (counts == minimum)
        if counts[0] == minimum:
            eligible &= qg - qg[0] > TOL
        maximum = max(qg[eligible], default=None)
        action = next((a for a in priority if eligible[a] and qg[a] >= maximum - TOL), 0)
    return action, scores


def reference_target_path(returns, inflation, matrix, decide, *, cost_rate=.0025):
    """Adapt our fixed fee/state reference with a locally owned target buffer.

    The decision updates the buffer BEFORE the independent fee equation.
    Pre-trade drift is then recomputed against the old identity, while end
    drift remains against the new identity. No reviewed executor is called.
    """
    target = np.asarray(matrix[1], float).copy()
    current, before, after, actions = 1, [], [], []

    def callback(t, x, cpi, peak, worst, bad):
        nonlocal current
        before.append(current)
        action = decide(t, x, cpi, peak, worst, bad, current)
        if not isinstance(action, (int, np.integer)) or action not in range(4):
            raise AssertionError("Invalid independent target action")
        if action:
            current = int(action) - 1
        target[:] = matrix[current]
        after.append(current); actions.append(int(action))
        return "restore" if action else "hold"

    result = reference_actual(returns, inflation, target, LOSS, callback, cost_rate=cost_rate)
    result["pretrade_tv_drift"] = .5 * np.abs(result["weights"][:-1] - matrix[np.asarray(before)]).sum(axis=-1)
    result["current_target_before"] = np.asarray(before)
    result["current_target_after"] = np.asarray(after)
    result["current_target_index"] = np.asarray([1, *after])
    result["action_index"] = np.asarray(actions)
    result["target_updated"] = (np.asarray(before) != after) & (np.arange(len(after)) > 0)
    return result


def self_check():
    matrix = np.array([[.25, .25, .50], [.50, .40, .10], [.80, .10, .10]])
    path = reference_target_path(np.array([[1., 0., 0.], [-.2, 0., 0.]]), np.zeros(2), matrix,
                                 lambda t, *args: 2 if t == 0 else 1)
    # Fee is nonzero in the production reference, so independently verify
    # each actual post-fee target and wealth conservation instead of using
    # the zero-cost142.5 closed form as its expected value.
    for t, index in enumerate((1, 0)):
        assert np.allclose(path["posttrade_holdings"][t], (path["nominal_wealth"][t] - path["fees"][t]) * matrix[index], atol=1e-14)
        assert abs(path["fees"][t] - .0025 * path["moved"][t]) < 1e-14
    assert path["current_target_index"].tolist() == [1, 1, 0]
    assert path["target_updated"].tolist() == [False, True]
    qg = np.array([0., .9e-12, 1.1e-12, -1.])
    action, _ = independent_choice(qg, np.zeros(4, int), np.ones(4, bool), 2, 0., 0)
    assert action == 2  # below-threshold priority action is excluded.


def main():
    self_check()
    before = snapshot()
    manifest, protocol = readj(OUT / "manifest.json"), readj(PROTOCOL)
    fits, selections = readj(STRICT / "nested_context_refits.json"), readj(OUT / "context_selections.json")
    parent_selections = readj(PARENT / "context_selections.json")
    parent_rows, parent_train = pd.read_csv(PARENT / "client_paths.csv"), pd.read_csv(PARENT / "training_candidate_paths.csv")
    rows, train = pd.read_csv(OUT / "client_paths.csv"), pd.read_csv(OUT / "training_candidate_paths.csv")
    summaries, gates = pd.read_csv(OUT / "training_candidate_summary.csv"), pd.read_csv(OUT / "decision_gates.csv")
    checks, errors, q_results, q_disagreements, selection_results = 0, {}, [], [], []
    replay_paths = replay_decisions = target_updates = changed_trades = 0

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

    for name, expected in protocol["frozen_input_sha256"].items():
        check(sha(ROOT / name) == expected, "frozen_input/" + name)
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation):
        digest.update(np.ascontiguousarray(value).tobytes())
    check(ids.tolist() == list(range(1, 121)) and digest.hexdigest() == manifest["development_arrays_sha256"]
          and digest.hexdigest() == readj(PARENT / "manifest.json")["development_array_sha256"], "same_guarded_parent_data")
    check(len(rows) == 480 and not rows.duplicated(["context", "scenario_id", "policy"]).any(), "all_original_C_rows")
    check(len(train) == 10800 and not train.duplicated(["context", "controller_order", "scenario_id"]).any(), "all18_training_candidate_rows")
    check(len(summaries) == 108 and not summaries.duplicated(["context", "controller_order"]).any(), "all18_training_summaries")
    check(train.scope.eq("training_support").all() and rows.client.eq("C").all(), "original_profile_and_training_scope")
    check(rows.loc[~rows.assigned, [*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing", "target_update_fraction", "target_update_count", "terminal_target_index"]].isna().all().all(), "execution_Review_null")
    rl = rows.set_index(["context", "scenario_id", "policy"]).sort_index()
    gl = gates.set_index(["context", "scenario_id", "year"]).sort_index()
    check(not gates.duplicated(["context", "scenario_id", "year"]).any(), "unique_expanded_decisions")
    with np.load(OUT / "actual_traces.npz", allow_pickle=False) as archive, np.load(PARENT / "actual_traces.npz", allow_pickle=False) as parent_archive:
        for context, fit in fits.items():
            chosen = selections[context]
            training_ids = fit["training_ids"]
            evaluation_ids = training_ids if context == "full_development" else fit["outer_test_ids"]
            check(chosen["training_ids"] == training_ids and chosen["evaluation_ids"] == evaluation_ids, "frozen_context_ids")
            check(context == "full_development" or not set(training_ids) & set(evaluation_ids), "outer_disjoint")
            menu = {item["name"]: item for item in fit["raw_fit"]["menu"]}
            check(chosen["menu"] == [menu[role] for role in ROLES] and fit["effective_assignments"]["C"] == "Balanced", "frozen_three_products")
            matrix = np.asarray([menu[role]["weights"] for role in ROLES])
            check(np.min(matrix) >= .05 - TOL, "strict_positive_product_weights")
            parent_choice = parent_selections[context]["profiles"]["C"]
            cap = min(parent_choice["reference_training_breach_count"], int(np.floor(.1 * len(training_ids) + TOL)))
            check(chosen["training_breach_cap"] == cap, "same_parent_training_cap")
            family = train.loc[train.context.eq(context)]
            table = summaries.loc[summaries.context.eq(context)].set_index("controller_order")
            values = []
            for order in range(18):
                group = family.loc[family.controller_order.eq(order)].set_index("scenario_id").sort_index()
                record = table.loc[order]
                check(len(group) == len(training_ids) and set(group.index) == set(training_ids) and group.assigned.all(), "each_controller_original_train_ids")
                check(group.updates_enabled.eq(order >= 9).all() and group.price_order.eq(order % 9).all(), "controller_price_mode_mapping")
                check(group.horizon.eq(H).all() and group.loss_tolerance.eq(LOSS).all()
                      and group.candidate_local_index.eq(menu["Balanced"]["candidate_local_index"]).all()
                      and group.candidate_original232_index.eq(menu["Balanced"]["candidate_original232_index"]).all(), "training_original_profile_and_product_identity")
                close(group.cagr.to_numpy(), np.expm1(group.log_growth.to_numpy()), "training_algebra/cagr", atol=3e-12)
                close(group.terminal_real.to_numpy(), 100 * np.exp(H * group.log_growth.to_numpy()), "training_algebra/terminal_real")
                close(group.nominal_fees.to_numpy(), .0025 * group.nominal_moved.to_numpy(), "training_algebra/fees")
                check(np.array_equal(group.breach.to_numpy(bool), group.mdd.to_numpy() - LOSS > TOL), "training_everbreach_event")
                close(group.target_update_fraction.to_numpy(), group.target_update_count.to_numpy() / H, "training_target_update_fraction")
                if order < 9:
                    old = parent_train.loc[parent_train.context.eq(context) & parent_train.client.eq("C") & parent_train.execution_candidate_order.eq(order)].set_index("scenario_id").loc[group.index]
                    for field in (*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing"):
                        close(group[field].to_numpy(), old[field].to_numpy(), "parent_training_fixed/" + field)
                    check(group.target_update_count.eq(0).all() and group.terminal_target_index.eq(1).all(), "fixed_training_target_identity")
                stats = {"mean_log_growth": group.log_growth.mean(), "breach_count": int(group.breach.astype(bool).sum()),
                    "mean_nominal_fees": group.nominal_fees.mean(), "mean_nominal_moved": group.nominal_moved.mean(),
                    "mean_trade_fraction": group.trade_fraction.mean(), "mean_target_update_fraction": group.target_update_fraction.mean()}
                for field, value in stats.items():
                    close(record[field], value, "training_summary/" + field)
                    close(chosen["candidate_summaries"][order][field], value, "selection_evidence/" + field)
                eligible = stats["breach_count"] <= cap
                check(record.training_eligible == eligible and record.training_breach_cap == cap and record.n_training == len(training_ids)
                      and chosen["candidate_summaries"][order]["training_eligible"] == eligible
                      and chosen["candidate_summaries"][order]["training_breach_cap"] == cap, "complete_controller_training_eligibility")
                values.append((order, stats["mean_log_growth"], eligible))
            maximum = max((growth for _, growth, eligible in values if eligible), default=None)
            order = next((i for i, growth, eligible in values if eligible and growth >= maximum - TOL), None)
            check(chosen["selected_controller_order"] == order, "global_max_growth_band_and_simplicity_tie")
            oldmax = max((growth for i, growth, eligible in values if i < 9 and eligible), default=None)
            check(oldmax is None or maximum >= oldmax, "expanded_family_contains_training_fixed_optimum")
            if maximum is None:
                check(chosen["maximum_eligible_mean_log_growth"] is None, "no_feasible_maximum_null")
            else:
                close(chosen["maximum_eligible_mean_log_growth"], maximum, "selection/maximum_G")
            if oldmax is None:
                check(chosen["fixed_family_maximum_eligible_mean_log_growth"] is None, "no_feasible_fixed_maximum_null")
            else:
                close(chosen["fixed_family_maximum_eligible_mean_log_growth"], oldmax, "selection/fixed_maximum_G")
            check(chosen["parent_fixed_selected_candidate_order"] == parent_choice["selected_candidate_order"], "frozen_parent_choice")
            selection_results.append({"context": context, "training_cap": cap, "selected_controller_order": order,
                "parent_fixed_price_order": parent_choice["selected_candidate_order"], "fixed_max_training_G": oldmax, "expanded_max_training_G": maximum})
            paths = {}
            for policy in POLICIES:
                served = parent_choice["selected_candidate_order"] is not None if policy == POLICIES[0] else order is not None
                controller = parent_choice["selected_candidate_order"] if policy == POLICIES[0] else order
                prefix = f"{context}__C__{policy}__"
                group = rows.loc[rows.context.eq(context) & rows.policy.eq(policy)]
                check(len(group) == len(evaluation_ids) and set(group.scenario_id) == set(evaluation_ids) and group.assigned.eq(served).all(), "main_original_C_coverage")
                if not served:
                    check(not any(key.startswith(prefix) for key in archive.files), "no_Review_fabricated_account")
                    continue
                updates, price_order = controller >= 9, controller % 9
                check(group.controller_order.eq(controller).all() and group.price_order.eq(price_order).all()
                      and group.updates_enabled.eq(updates).all(), "actual_selected_controller_metadata")
                if price_order < 8:
                    close(group.annual_price.to_numpy(), np.full(len(group), PRICES[price_order]), "actual_metadata/annual_price")
                    close(group.original_H_total_price.to_numpy(), np.full(len(group), H * PRICES[price_order]), "actual_metadata/original_H_total_price")
                else:
                    check(group.annual_price.isna().all() and group.original_H_total_price.isna().all(), "actual_risk_first_price_metadata_null")
                saved = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                check(np.array_equal(saved["scenario_ids"], evaluation_ids), "actual_scenario_order")
                if policy == POLICIES[0]:
                    oldprefix = f"{context}__C__risk_priced_learner__"
                    shared = [key for key in parent_archive.files if key.startswith(oldprefix)]
                    for key in shared:
                        suffix = key[len(oldprefix):]
                        if suffix in saved:
                            close(saved[suffix], parent_archive[key], "parent_selected_trace/" + suffix)
                for i, sid in enumerate(evaluation_ids):
                    def decision(t, x, cpi, peak, worst, bad, current):
                        active = t > 0 and not bad
                        check(saved["current_target_before"][i, t] == current and saved["current_target_index"][i, t] == current, "current_target_before_identity")
                        if t == 0:
                            action = 2
                        elif bad:
                            action = 0
                        else:
                            qg, qb, available = saved["qg_actions"][i, t], saved["qbad_actions"][i, t], saved["available_actions"][i, t]
                            expected_available = np.ones(4, bool) if updates else np.arange(4) == 0
                            if not updates:
                                expected_available[current + 1] = True
                            check(np.array_equal(available, expected_available), "current_action_availability")
                            check(saved["forecast_n"][i, t] == len(training_ids) and np.all(qb[available] >= 0) and np.all(qb[available] <= len(training_ids)), "fan_integer_risk_denominator")
                            action, scores = independent_choice(qg, qb, available, len(training_ids), PRICES[price_order], current)
                            close(saved["action_scores"][i, t], scores, "actual_gate/action_scores", atol=3e-12, rtol=0)
                            if PRICES[price_order] is not None:
                                close(saved["selected_score"][i, t], scores[action], "actual_gate/selected_score", atol=3e-12, rtol=0)
                                check(action == 0 or scores[action] > TOL, "chosen_restore_or_retarget_positive_net_benefit")
                            else:
                                check(np.isnan(saved["selected_score"][i, t]), "risk_first_selected_score_null")
                        check(saved["action_index"][i, t] == action, "actual_action_rule")
                        after = current if action == 0 else action - 1
                        check(saved["current_target_after"][i, t] == after and saved["current_target_index"][i, t + 1] == after
                              and saved["target_updated"][i, t] == (t > 0 and after != current), "target_state_transition_and_update_flag")
                        if not active:
                            check(not saved["available_actions"][i, t].any() and np.isnan(saved["qg_actions"][i, t]).all()
                                  and np.all(saved["qbad_actions"][i, t] == -1), "initial_pending_no_Q_no_reentry")
                        return int(action)
                    path = reference_target_path(returns[sid - 1, :H], inflation[sid - 1, :H], matrix, decision)
                    paths[(sid, policy)] = path
                    replay_paths += 1; replay_decisions += H
                    target_updates += int(path["target_updated"].sum())
                    changed_trades += int((path["target_updated"] & path["trades"]).sum())
                    for field in path:
                        if field in saved:
                            close(saved[field][i], path[field], "actual_trace/" + field)
                    record = rl.loc[(context, sid, policy)]
                    own = independent_metrics(path, LOSS)
                    own["trade_fraction"] = own.pop("actual_trade_fraction")
                    own["target_execution_fraction"] = float(path["target_execution"].mean())
                    for field in (*METRICS, *ACCOUNTS, "breach", "excess"):
                        close(record[field], own[field], "actual_metrics/" + field)
                    check(pd.isna(record.first_crossing) if path["first_crossing"] < 0 else record.first_crossing == path["first_crossing"], "actual_crossing_or_null")
                    check(record.target_update_count == int(path["target_updated"].sum()) and record.terminal_target_index == path["current_target_index"][-1], "actual_update_count_terminal_identity")
                    close(record.target_update_fraction, path["target_updated"].mean(), "actual_metrics/target_update_fraction")
                    if policy == POLICIES[1]:
                        gg = gl.loc[(context, sid)]
                        check(list(gg.index) == list(range(1, H + 1)), "expanded_gate_all_original_H")
                        for t in range(H):
                            gate = gg.loc[t + 1]
                            check(gate.controller_order == controller and gate.price_order == price_order and gate.updates_enabled == updates,
                                  "CSV_selected_controller_metadata")
                            check(gate.action_index == path["action_index"][t] and gate.target_before == path["current_target_before"][t]
                                  and gate.target_after == path["current_target_after"][t] and gate.target_updated == path["target_updated"][t]
                                  and gate.actual_trade == path["trades"][t], "CSV_target_action_state")
                            close(gate.nominal_fee, path["fees"][t], "CSV_gate/fee")
                            close(gate.nominal_moved, path["moved"][t], "CSV_gate/moved")
                            active = t > 0 and not path["ever_breached"][t]
                            check(gate.active_gate == active and gate.pending_at_decision == path["ever_breached"][t], "CSV_gate_active_pending")
                            for a in range(4):
                                available = active and bool(saved["available_actions"][i, t, a])
                                check(gate[f"action_{a}_available"] == available, "CSV_action_availability")
                                for stem, key in (("qg", "qg_actions"), ("bad_count", "qbad_actions"), ("score", "action_scores")):
                                    meaningful = available and (stem != "score" or price_order < 8)
                                    if meaningful:
                                        close(gate[f"action_{a}_{stem}"], saved[key][i, t, a], "CSV_action/" + stem)
                                    else:
                                        check(pd.isna(gate[f"action_{a}_{stem}"]), "CSV_inactive_action_null")
            sid = min(evaluation_ids)
            for year in (2, 16, 30):
                identity = {"context": context, "scenario_id": sid, "year": year}
                if order is None:
                    q_results.append({**identity, "status": "execution_Review_skip"}); continue
                path = paths[(sid, POLICIES[1])]
                gate = gl.loc[(context, sid, year)]
                if not gate.active_gate:
                    check(path["ever_breached"][year - 1], "Q_subset_pending_skip")
                    q_results.append({**identity, "status": "pending_skip_no_replacement"}); continue
                current = int(path["current_target_index"][year - 1])
                fan = [scalar_fan(path["holdings"][year - 1], path["cpi"][year - 1], path["real_peak"][year - 1], path["running_mdd"][year - 1],
                       matrix[m], LOSS, returns[np.asarray(training_ids) - 1, :H], inflation[np.asarray(training_ids) - 1, :H], year - 1) for m in range(3)]
                qg = np.asarray([fan[current]["hold"]["mean"], *[value["restore"]["mean"] for value in fan]])
                qb = np.asarray([fan[current]["hold"]["count"], *[value["restore"]["count"] for value in fan]])
                available = np.ones(4, bool) if order >= 9 else (np.arange(4) == 0) | (np.arange(4) == current + 1)
                for a in np.flatnonzero(available):
                    close(gate[f"action_{a}_qg"], qg[a], "Q_subset/qg", atol=3e-12, rtol=0)
                    check(gate[f"action_{a}_bad_count"] == qb[a], "Q_subset_integer_terminal_risk")
                action, score = independent_choice(qg, qb, available, len(training_ids), PRICES[order % 9], current)
                if gate.action_index != action:
                    q_disagreements.append({**identity, "saved": int(gate.action_index), "independent": int(action), "independent_scores": [float(value) if np.isfinite(value) else None for value in score]})
                q_results.append({**identity, "status": "audited" if gate.action_index == action else "numeric_action_review_needed", "current_target_index": current,
                    "controller_order": order, "training_paths": len(training_ids), "available_actions": available.tolist(),
                    "independent_qg_actions": qg.tolist(), "independent_bad_actions": qb.tolist(), "independent_action": int(action)})
    check(replay_paths == int(rows.assigned.sum()) and replay_decisions == int(rows.loc[rows.assigned, "horizon"].sum()), "all_saved_real_transitions_replayed")
    check(len(gates) == int(rows.loc[rows.assigned & rows.policy.eq(POLICIES[1]), "horizon"].sum()), "all_expanded_decisions_only")
    check(len(q_results) == 18, "prospective_all_action_Q_identities")
    for (scope, policy), group in rows.groupby(["scope", "policy"]):
        check(len(group) == 120 and set(group.scenario_id) == set(range(1, 121)), "original_C120_denominators")
    for record in pd.read_csv(OUT / "summary.csv").itertuples(index=False):
        group = rows.loc[rows.scope.eq(record.scope) & rows.policy.eq(record.policy)]
        auto = group.loc[group.assigned]
        check(record.total_paths == 120 and record.assigned_paths == len(auto) and record.review_paths == 120 - len(auto), "summary_coverage")
        if not len(auto):
            check(pd.isna(record.mean_log_growth) and pd.isna(record.breach_probability), "summary_Review_null"); continue
        stats = {"mean_log_growth": auto.log_growth.mean(), "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
            "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"), "cvar90_mdd": independent_cvar(auto.mdd),
            "breach_count": int(auto.breach.astype(bool).sum()), "breach_probability": auto.breach.astype(bool).mean(), "mean_excess": auto.excess.mean(),
            "mean_target_update_fraction": auto.target_update_fraction.mean(), **{"mean_" + field: auto[field].mean() for field in ACCOUNTS}}
        for field, value in stats.items():
            close(getattr(record, field), value, "summary/" + field)
    pairs = pd.read_csv(OUT / "paired_paths.csv")
    check(len(pairs) == 240 and not pairs.duplicated(["context", "scenario_id"]).any(), "all_C_original_pairs")
    fields = (*METRICS, *ACCOUNTS, "excess", "breach", "target_update_fraction")
    for record in pairs.itertuples(index=False):
        fixed, expanded = [rl.loc[(record.context, record.scenario_id, policy)] for policy in POLICIES]
        common = bool(fixed.assigned and expanded.assigned)
        check(record.common_auto == common and record.fixed_assigned == fixed.assigned and record.expanded_assigned == expanded.assigned, "pair_admission")
        for field in fields:
            if common:
                close(getattr(record, "delta_" + field), float(expanded[field]) - float(fixed[field]), "paired_path/" + field)
            else:
                check(pd.isna(getattr(record, "delta_" + field)), "unserved_pair_null")
    for record in pd.read_csv(OUT / "paired_summary.csv").itertuples(index=False):
        group = pairs.loc[pairs.scope.eq(record.scope)]; auto = group.loc[group.common_auto]
        check(record.total_original_pairs == 120 and record.common_auto_n == len(auto)
              and record.fixed_auto_n == int(group.fixed_assigned.sum()) and record.expanded_auto_n == int(group.expanded_assigned.sum()), "paired_summary_denominators")
        for field in fields:
            if len(auto):
                close(getattr(record, "mean_delta_" + field), auto["delta_" + field].mean(), "paired_summary/" + field)
            else:
                check(pd.isna(getattr(record, "mean_delta_" + field)), "paired_summary_Review_null")
    for kind in ("sources", "input_hashes"):
        for name, expected in manifest[kind].items():
            check(sha(ROOT / name) == expected, kind + "/" + name)
    for name, expected in manifest["outputs_sha256"].items():
        check(sha(OUT / name) == expected, "output_hash/" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_hashes"] == manifest["input_hashes_at_end"], "original_start_end_hashes")
    after = snapshot(); check(before == after, "audit_source_artifact_stability")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not q_disagreements else "numerical_action_review_needed",
        "checks_passed": checks, "maximum_errors": errors, "actual_arm_paths_replayed": replay_paths, "actual_decisions_replayed": replay_decisions,
        "target_updates_replayed": target_updates, "target_updates_with_actual_trade": changed_trades,
        "training_candidate_rows_aggregated": len(train), "training_summaries_reconstructed": len(summaries), "selections": selection_results,
        "Q_subset_results": q_results, "Q_subset_audited": sum(item["status"] == "audited" for item in q_results),
        "Q_subset_pending_skipped": sum(item["status"] == "pending_skip_no_replacement" for item in q_results), "Q_subset_action_disagreements": q_disagreements,
        "source_and_artifact_sha256_start": before, "source_and_artifact_sha256_end": after,
        "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(), "final_scenarios_read": False,
        "independence": "Own100-step bisection/state replay with independently carried target identity; own scalar fan per target and independent global action/selection comparisons; no reviewed core/runner/selector imported",
        "limitations": ["Saved training path metrics, all18 summaries, eligibility and selection are independently reaggregated; unrecorded full training decision trees are not independently replayed",
            "All selected actual and fixed-reference financial/target transitions are replayed; independent four-action future Q computation covers only the fixed18-identity subset",
            "Target labels do not imply H30 risk ordering or inherited static suitability for Profile C; full-policy empirical screens are not probability certificates",
            "The expanded family contains fixed candidates only as a training-family property; different prices/coverage and repeated development use prevent a causal or fresh-test improvement claim",
            "No price/target/grid/threshold retuning, winner adoption or final-set access"], "original_manifests_modified": False}
    destination = OUT / "independent_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_passed", "actual_arm_paths_replayed", "actual_decisions_replayed", "target_updates_replayed", "Q_subset_audited", "Q_subset_pending_skipped", "Q_subset_action_disagreements", "maximum_errors")}))
    print(json.dumps({"output": str(destination), "output_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
