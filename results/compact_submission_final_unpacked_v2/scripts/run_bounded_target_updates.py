"""Incremental18-controller study for the original Balanced Profile C.

Nine fixed-target candidates are retained alongside nine update-enabled
candidates.  Full-policy training selects once; only that candidate is used
on outer tests.  No target or price grid is expanded from test outcomes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
import run_fan_rollout_execution as reference
import run_risk_priced_fan_execution as priced
from safeinvest.data import load_paths

PROTOCOL = ROOT / "docs/bounded_target_update_protocol.json"
OUT = ROOT / "results/model_review/bounded_target_updates_v1"
PARENT = priced.OUT
POLICIES = ("fixed_target_learner", "expanded_target_learner")
ROLES = ("Conservative", "Balanced", "Growth")
H, L, TOL = 30, .40, 1e-12
sha, write_json = priced.sha, priced.write_json


def sources():
    names = ["scripts/run_bounded_target_updates.py", "tests/test_bounded_target_updates.py",
        "research/experimental_execution/bounded_target_update.py", "tests/test_bounded_target_update.py",
        "research/experimental_execution/risk_priced_fan_rollout.py", "research/experimental_execution/fan_rollout.py",
        "research/experimental_execution/review_pending.py", "scripts/run_risk_priced_fan_execution.py",
        "scripts/run_fan_rollout_execution.py", "research/safeinvest_research/metrics.py",
        "src/safeinvest/core.py", "src/safeinvest/data.py", "docs/bounded_target_update_protocol.json"]
    return {name: sha(ROOT / name) for name in names}


def inputs(protocol):
    result = {name: sha(ROOT / name) for name in protocol["frozen_input_sha256"]}
    if result != protocol["frozen_input_sha256"]: raise RuntimeError("Frozen incremental-study input changed")
    return result


def controller_meta(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order < 18:
        raise ValueError("Controller order must be0..17")
    price_order, updates = int(order % 9), order >= 9
    return {"controller_order": int(order), "controller_id": ("update__" if updates else "fixed__") + priced.CANDIDATE_IDS[price_order],
            "updates_enabled": updates, "price_order": price_order, **priced.candidate_meta(price_order, H)}


def select_expanded(summaries, reference_count, n_training):
    """One global growth band, with the simpler mode first in declared order."""
    if (len(summaries) != 18 or [r["controller_order"] for r in summaries] != list(range(18))
            or isinstance(n_training, (bool, np.bool_)) or not isinstance(n_training, (int, np.integer)) or n_training < 1
            or isinstance(reference_count, (bool, np.bool_)) or int(reference_count) != reference_count or not 0 <= reference_count <= n_training):
        raise ValueError("Need all18 ordered full-training summaries and valid reference count")
    cap = min(int(reference_count), int(np.floor(.10 * n_training + TOL)))
    classified = []
    for row in summaries:
        count = row["breach_count"]
        if (isinstance(count, (bool, np.bool_)) or int(count) != count or not 0 <= count <= n_training
                or row["n_training"] != n_training or not np.isfinite(row["mean_log_growth"])):
            raise ValueError("Controller summary must cover all original training paths")
        classified.append({**row, "training_eligible": int(count) <= cap, "training_breach_cap": cap})
    eligible = [row for row in classified if row["training_eligible"]]
    maximum = max((row["mean_log_growth"] for row in eligible), default=None)
    selected = next((row for row in eligible if row["mean_log_growth"] >= maximum - TOL), None)
    old = [row for row in classified[:9] if row["training_eligible"]]
    fixed_max = max((row["mean_log_growth"] for row in old), default=None)
    if fixed_max is not None and (maximum is None or maximum < fixed_max): raise RuntimeError("Expanded family lost an old feasible candidate")
    return {"n_training": int(n_training), "reference_training_breach_count": int(reference_count), "training_breach_cap": cap,
            "maximum_eligible_mean_log_growth": maximum, "fixed_family_maximum_eligible_mean_log_growth": fixed_max,
            "selected_controller_order": selected["controller_order"] if selected else None,
            "selected_controller": controller_meta(selected["controller_order"]) if selected else None,
            "execution_status": "automated" if selected else "execution_review",
            "execution_review_reason": None if selected else "expanded_family_no_admissible_policy",
            "candidate_summaries": classified, "growth_tie_tolerance": TOL,
            "selected_from_training_only": True}


def extract_controller(family, order):
    result = {}; n, p = family["real_wealth"].shape[:2]
    metadata = {"cpi", "targets", "target_matrix", "annual_prices", "loss_tolerances", "initial_target_index"}
    for key, value in family.items():
        if not isinstance(value, np.ndarray): continue
        if key in metadata: result[key] = value.copy()
        elif value.ndim >= 2 and value.shape[:2] == (n, p): result[key] = value[:, order].copy()
        else: raise ValueError(f"Unexpected controller axis for{key}")
    return result


def rows_for(arm, ids, common):
    rows = priced.financial_rows(arm, ids, common)
    for i, row in enumerate(rows):
        row["target_update_fraction"] = float(arm["target_updated"][i].mean())
        row["target_update_count"] = int(arm["target_updated"][i].sum())
        row["terminal_target_index"] = int(arm["current_target_index"][i, -1])
    return rows


def candidate_summary(rows, order):
    frame = pd.DataFrame(rows)
    if not frame.assigned.all() or frame.scenario_id.nunique() != len(frame): raise RuntimeError("Training controller path missing")
    return {**controller_meta(order), "n_training": len(frame), "breach_count": int(frame.breach.astype(bool).sum()),
        "mean_log_growth": float(frame.log_growth.mean()), "mean_nominal_fees": float(frame.nominal_fees.mean()),
        "mean_nominal_moved": float(frame.nominal_moved.mean()), "mean_trade_fraction": float(frame.trade_fraction.mean()),
        "mean_target_update_fraction": float(frame.target_update_fraction.mean())}


def verify_parent_candidate(rows, parent, context, price_order):
    expected = parent.loc[(parent.context == context) & (parent.client == "C") & (parent.execution_candidate_order == price_order)].set_index("scenario_id")
    actual = pd.DataFrame(rows).set_index("scenario_id")
    if len(actual) != len(expected) or actual.index.has_duplicates or set(actual.index) != set(expected.index):
        raise RuntimeError("Parent training identity changed")
    actual = actual.loc[expected.index]
    for field in (*reference.METRICS, *reference.ACCOUNTS, "breach", "excess"):
        np.testing.assert_allclose(actual[field].to_numpy(float), expected[field].to_numpy(float), atol=2e-10, rtol=5e-13)
    verify_crossings(actual.first_crossing, expected.first_crossing)


def verify_crossings(actual, expected):
    a, b = pd.Series(actual).reset_index(drop=True), pd.Series(expected).reset_index(drop=True)
    np.testing.assert_array_equal(a.isna().to_numpy(), b.isna().to_numpy())
    np.testing.assert_array_equal(a[a.notna()].to_numpy(), b[b.notna()].to_numpy())


def actual_gate_rows(arm, ids, common, selected):
    rows = []
    for i, sid in enumerate(ids):
        for t in range(H):
            active = t > 0 and not arm["pending_at_decision"][i, t]
            row = {**common, "scenario_id": sid, "year": t + 1, **selected,
                "action": str(arm["actions"][i, t]), "action_index": int(arm["action_index"][i, t]),
                "target_before": int(arm["current_target_before"][i, t]), "target_after": int(arm["current_target_after"][i, t]),
                "target_updated": bool(arm["target_updated"][i, t]), "actual_trade": bool(arm["trades"][i, t]),
                "nominal_fee": float(arm["fees"][i, t]), "nominal_moved": float(arm["moved"][i, t]),
                "pending_at_decision": bool(arm["pending_at_decision"][i, t]), "active_gate": active,
                "forecast_n": int(arm["forecast_n"][i, t]) if active else None,
                "selected_score": float(arm["selected_score"][i, t]) if active and selected["price_kind"] == "finite" else None,
                "gate_reason": str(arm["gate_reason"][i, t])}
            for a in range(4):
                available = active and bool(arm["available_actions"][i, t, a])
                row[f"action_{a}_available"] = available
                row[f"action_{a}_qg"] = float(arm["qg_actions"][i, t, a]) if available else None
                row[f"action_{a}_bad_count"] = int(arm["qbad_actions"][i, t, a]) if available else None
                row[f"action_{a}_score"] = float(arm["action_scores"][i, t, a]) if available and selected["price_kind"] == "finite" else None
            if active and row["action_index"] > 0 and selected["price_kind"] == "finite" and not row["selected_score"] > TOL:
                raise RuntimeError("Trade or target update lacks strictly positive estimated net benefit")
            rows.append(row)
    return rows


def paired_outputs(frame):
    keys = ["scope", "context", "outer_fold", "client", "scenario_id"]
    joined = frame.loc[frame.policy == POLICIES[1]].merge(frame.loc[frame.policy == POLICIES[0]], on=keys,
        suffixes=("_expanded", "_fixed"), validate="one_to_one")
    rows = []
    fields = (*reference.METRICS, *reference.ACCOUNTS, "excess", "breach", "target_update_fraction")
    for item in joined.to_dict("records"):
        common = bool(item["assigned_expanded"] and item["assigned_fixed"])
        rows.append({**{key: item[key] for key in keys}, "common_auto": common,
            "expanded_assigned": bool(item["assigned_expanded"]), "fixed_assigned": bool(item["assigned_fixed"]),
            **{"delta_" + field: float(item[field + "_expanded"]) - float(item[field + "_fixed"]) if common else None for field in fields}})
    pairs = pd.DataFrame(rows); summaries = []
    for scope, group in pairs.groupby("scope", sort=False):
        auto = group.loc[group.common_auto]
        summaries.append({"scope": scope, "total_original_pairs": len(group), "common_auto_n": len(auto),
            "expanded_auto_n": int(group.expanded_assigned.sum()), "fixed_auto_n": int(group.fixed_assigned.sum()),
            **{"mean_" + field: float(auto[field].mean()) if len(auto) else None for field in pairs if field.startswith("delta_")}})
    return pairs, pd.DataFrame(summaries)


def main():
    from experimental_execution.bounded_target_update import simulate_controllers
    started = time.perf_counter(); protocol = json.loads(PROTOCOL.read_text())
    if not protocol["execution_ready"]: raise RuntimeError("Complete synthetic checks and freeze protocol before market execution")
    source_start, input_start = sources(), inputs(protocol)
    protocol_key = str(PROTOCOL.relative_to(ROOT))
    if protocol["execution_sources_sha256"] != {k: v for k, v in source_start.items() if k != protocol_key}:
        raise RuntimeError("Execution source/test snapshot differs from protocol")
    if json.loads((PARENT / "independent_audit.json").read_text())["status"] != "passed": raise RuntimeError("Parent independent audit must pass first")
    fits = json.loads((reference.STRICT / "nested_context_refits.json").read_text()); reference.validate_contexts(fits)
    mapping = json.loads((reference.STRICT / "candidate_mapping.json").read_text())
    parent_candidates = pd.read_csv(PARENT / "training_candidate_paths.csv")
    parent_main = pd.read_csv(PARENT / "client_paths.csv"); parent_main = parent_main.loc[(parent_main.client == "C") & (parent_main.policy == priced.LEARNER)].copy()
    parent_selections = json.loads((PARENT / "context_selections.json").read_text())
    returns, inflation, ids = load_paths(reference.DATA, range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    if (not np.array_equal(ids, np.arange(1, 121))
            or digest.hexdigest() != json.loads((PARENT / "manifest.json").read_text())["development_array_sha256"]):
        raise RuntimeError("Guarded development arrays differ from the frozen parent")
    main_rows, training_rows, summaries, gates, archive, selections, checks = [], [], [], [], {}, {}, {}
    for context, fit in fits.items():
        train_ids = fit["training_ids"]; full = context == "full_development"
        eval_ids = train_ids if full else fit["outer_test_ids"]
        train_ix, eval_ix = np.asarray(train_ids) - 1, np.asarray(eval_ids) - 1
        scope = "full_development" if full else "outer_oof"
        menu = {item["name"]: item for item in fit["raw_fit"]["menu"]}
        if fit["effective_assignments"]["C"] != "Balanced" or set(menu) != set(ROLES): raise RuntimeError("Original C/menu context changed")
        matrix = np.array([reference.validate_product(menu[role], mapping) for role in ROLES])
        base = parent_selections[context]["profiles"]["C"]
        common = {"context": context, "scope": scope, "outer_fold": fit["context_id"] or None, "client": "C",
            "horizon": H, "loss_tolerance": L, "risk_assignment": "Balanced", "risk_assigned": True,
            "candidate_local_index": menu["Balanced"]["candidate_local_index"],
            "candidate_original232_index": menu["Balanced"]["candidate_original232_index"]}
        train_r, train_i = returns[train_ix, :H], inflation[train_ix, :H]
        family_rows, family_summaries, families = [], [], {}
        for updates in (False, True):
            family = simulate_controllers(train_r, train_i, train_r, train_i, matrix, priced.PRICES,
                initial_target_index=1, updates_enabled=updates, L=L, cost_rate=.0025, horizon=H)
            families[updates] = family
            for price_order in range(9):
                order = price_order + (9 if updates else 0)
                arm = extract_controller(family, price_order)
                candidate = rows_for(arm, train_ids, {**common, "scope": "training_support", "policy": "training_candidate",
                    "assignment": "Balanced", **controller_meta(order)})
                if not updates: verify_parent_candidate(candidate, parent_candidates, context, price_order)
                family_rows.extend(candidate); family_summaries.append(candidate_summary(candidate, order))
        selection = select_expanded(family_summaries, base["reference_training_breach_count"], len(train_ids))
        selection.update({"training_ids": train_ids, "evaluation_ids": eval_ids, "menu": [menu[role] for role in ROLES],
            "parent_fixed_selected_candidate_order": base["selected_candidate_order"], "profile": {"name": "C", "horizon": H, "loss_tolerance": L}})
        selections[context] = selection; training_rows.extend(family_rows)
        summaries.extend({"context": context, "client": "C", **row} for row in selection["candidate_summaries"])
        old = parent_main.loc[parent_main.context == context].copy()
        if len(old) != len(eval_ids) or set(old.scenario_id) != set(eval_ids): raise RuntimeError("Parent selected C denominator changed")
        old["policy"] = POLICIES[0]; old["target_update_fraction"] = np.where(old.assigned, 0., np.nan)
        old["target_update_count"] = np.where(old.assigned, 0, np.nan); old["terminal_target_index"] = np.where(old.assigned, 1, np.nan)
        old["price_order"] = old.execution_candidate_order
        old["controller_order"] = old.execution_candidate_order
        old["controller_id"] = ["fixed__" + str(value) if pd.notna(value) else None for value in old.execution_candidate_id]
        old["updates_enabled"] = [False if served else None for served in old.assigned]
        main_rows.extend(old.to_dict("records"))
        if base["selected_candidate_order"] is not None:
            price_order = base["selected_candidate_order"]
            fixed_eval_family = families[False] if full else simulate_controllers(returns[eval_ix, :H], inflation[eval_ix, :H],
                train_r, train_i, matrix, np.array([priced.PRICES[price_order]]), initial_target_index=1, updates_enabled=False, L=L, horizon=H)
            fixed_arm = extract_controller(fixed_eval_family, price_order if full else 0)
            actual_fixed = pd.DataFrame(rows_for(fixed_arm, eval_ids, {**common, "assignment": "Balanced"})).set_index("scenario_id").loc[old.scenario_id]
            for field in (*reference.METRICS, *reference.ACCOUNTS, "breach", "excess"):
                np.testing.assert_allclose(actual_fixed[field].to_numpy(float), old[field].to_numpy(float), atol=2e-10, rtol=5e-13)
            verify_crossings(actual_fixed.first_crossing, old.first_crossing)
            prefix = f"{context}__C__{POLICIES[0]}"
            archive[prefix + "__scenario_ids"] = np.asarray(eval_ids)
            for key, value in fixed_arm.items():
                if isinstance(value, np.ndarray): archive[prefix + "__" + key] = value
        order = selection["selected_controller_order"]
        new_common = {**common, "policy": POLICIES[1], "assignment": "Balanced" if order is not None else "Review",
            "selected_execution_status": selection["execution_status"], "execution_review_reason": selection["execution_review_reason"],
            "training_breach_cap": selection["training_breach_cap"],
            **(controller_meta(order) if order is not None else dict.fromkeys(controller_meta(0)))}
        if order is None:
            nulls = priced.null_rows(eval_ids, new_common)
            for row in nulls: row.update(dict.fromkeys(["target_update_fraction", "target_update_count", "terminal_target_index"]))
            main_rows.extend(nulls)
        else:
            updates, price_order = order >= 9, order % 9
            if full: arm = extract_controller(families[updates], price_order)
            else:
                evaluated = simulate_controllers(returns[eval_ix, :H], inflation[eval_ix, :H], train_r, train_i, matrix,
                    np.array([priced.PRICES[price_order]]), initial_target_index=1, updates_enabled=updates, L=L, horizon=H)
                if evaluated["real_wealth"].shape[1] != 1: raise RuntimeError("Unselected OOF controller evaluated")
                arm = extract_controller(evaluated, 0)
            main_rows.extend(rows_for(arm, eval_ids, new_common)); gates.extend(actual_gate_rows(arm, eval_ids, new_common, controller_meta(order)))
            prefix = f"{context}__C__{POLICIES[1]}"; archive[prefix + "__scenario_ids"] = np.asarray(eval_ids)
            for key, value in arm.items():
                if isinstance(value, np.ndarray): archive[prefix + "__" + key] = value
        checks[context] = {"original_nine_training_candidates_reproduced": True, "selected_parent_financials_reproduced": True,
            "expanded_retains_training_optimum": True, "no_unselected_outer_controller_evaluated": True}
        print(json.dumps({"context": context, "selected": selection["selected_controller"]["controller_id"] if order is not None else None,
            "elapsed_seconds": time.perf_counter() - started}), flush=True)
    frame = pd.DataFrame(main_rows)
    if len(frame) != 480 or frame.duplicated(["scope", "context", "scenario_id", "policy"]).any(): raise RuntimeError("Original C population changed")
    for (_, _), group in frame.groupby(["scope", "policy"]):
        if len(group) != 120 or set(group.scenario_id) != set(range(1, 121)): raise RuntimeError("Original120 C denominator changed")
    if not frame.loc[~frame.assigned, [*priced.FINANCIAL, "target_update_fraction"]].isna().all().all(): raise RuntimeError("ExecutionReview has imputed financial outcome")
    pairs, pair_summary = paired_outputs(frame); summary = pd.DataFrame(reference.summarize(frame))
    updates = frame.groupby(["scope", "client", "policy"], sort=False).target_update_fraction.mean().rename("mean_target_update_fraction").reset_index()
    summary = summary.merge(updates, on=["scope", "client", "policy"], validate="one_to_one")
    end_sources, end_inputs = sources(), inputs(protocol)
    if source_start != end_sources or input_start != end_inputs: raise RuntimeError("Frozen source/input changed during target-update study")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {"client_paths.csv": frame, "summary.csv": summary, "paired_paths.csv": pairs, "paired_summary.csv": pair_summary,
        "training_candidate_paths.csv": pd.DataFrame(training_rows), "training_candidate_summary.csv": pd.DataFrame(summaries),
        "decision_gates.csv": pd.DataFrame(gates)}
    for name, value in frames.items(): value.to_csv(OUT / name, index=False)
    np.savez_compressed(OUT / "actual_traces.npz", **archive)
    write_json(OUT / "context_selections.json", selections); write_json(OUT / "accounting_checks.json", checks)
    names = [*frames, "actual_traces.npz", "context_selections.json", "accounting_checks.json"]
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": source_start, "sources_at_end": end_sources, "input_hashes": input_start, "input_hashes_at_end": end_inputs,
        "outputs_sha256": {name: sha(OUT / name) for name in names}, "scenario_ids_loaded": ids.tolist(), "final_scenarios_loaded": False,
        "development_arrays_sha256": digest.hexdigest(), "client_path_rows": len(frame), "training_candidate_path_rows": len(training_rows),
        "decision_rows": len(gates), "scope": "Original Balanced Profile C H30,18-controller training-selected incremental target-update procedure. No OOF winner choice, new targets, fresh holdout or inherited dynamic risk certificate."})
    print(f"Wrote bounded target-update learner to{OUT}", flush=True)


if __name__ == "__main__": main()
