"""Known-cost inputs for frozen, previously selected Profile-C controllers.

No training candidate grid, cap/admission test or parameter selection is
repeated.  Source/audit/selection snapshots must be frozen before execution.
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
import run_bounded_target_updates as bounded
import run_risk_priced_fan_execution as priced
from experimental_execution.bounded_target_update import simulate_controllers
from experimental_execution.review_pending import simulate_paired
from safeinvest.data import load_paths

PROTOCOL = ROOT / "docs/selected_controller_cost_stress_protocol.json"
OUT = ROOT / "results/model_review/selected_controller_cost_stress_v1"
PARENTS = {"fixed": priced.OUT, "expanded": bounded.OUT}
RATES = (("zero", 0.), ("half", .00125), ("base", .0025), ("double", .005), ("quadruple", .01))
POLICIES = ("fixed_target_learner", "expanded_target_learner", "annual_then_pending", "equal_thirds_annual")
H, L = 30, .4
reference = bounded.reference
sha, write_json = priced.sha, priced.write_json
EXTRA_FINANCIAL = ("target_update_fraction", "target_update_count", "terminal_target_index")
FINANCIAL = (*priced.FINANCIAL, *EXTRA_FINANCIAL)


def plain(value):
    if isinstance(value, np.generic): value = value.item()
    return None if pd.isna(value) else value


def prototype(frame, context, policy, ids):
    part = frame.loc[(frame.context == context) & (frame.client == "C") & (frame.policy == policy)]
    if len(part) != len(ids) or set(part.scenario_id) != set(ids) or part.scenario_id.duplicated().any():
        raise RuntimeError("Parent selected original C identities changed")
    metadata = {}
    for key in part:
        if key in (*FINANCIAL, "scenario_id"): continue
        if part[key].nunique(dropna=False) != 1: raise RuntimeError(f"Parent context metadata unexpectedly varies:{key}")
        metadata[key] = plain(part[key].iloc[0])
    return metadata


def build_controller_plan(fits, price_choices, update_choices, price_rows, update_rows):
    """Read immutable parent choices; do not compute or select a new learner."""
    result = {}
    for context, fit in fits.items():
        full = context == "full_development"
        ids = fit["training_ids"] if full else fit["outer_test_ids"]
        menu = {item["name"]: item for item in fit["raw_fit"]["menu"]}
        if fit["effective_assignments"]["C"] != "Balanced" or set(menu) != set(bounded.ROLES):
            raise RuntimeError("Original Balanced C risk/menu assignment changed")
        parent = price_choices[context]["profiles"]["C"]
        expanded = update_choices[context]
        fixed_order, new_order = parent["selected_candidate_order"], expanded["selected_controller_order"]
        if fixed_order != expanded["parent_fixed_selected_candidate_order"]:
            raise RuntimeError("Expanded parent does not retain the same fixed parent choice")
        learners = {}
        for policy, order, updates in ((POLICIES[0], fixed_order, False), (POLICIES[1], new_order, new_order is not None and new_order >= 9)):
            metadata = prototype(update_rows, context, policy, ids)
            if bool(metadata["assigned"]) != (order is not None): raise RuntimeError("Parent executionReview/choice mismatch")
            if order is not None and (isinstance(order, bool) or not isinstance(order, int) or not 0 <= order < (9 if policy == POLICIES[0] else 18)):
                raise RuntimeError("Invalid frozen parent controller identity")
            price_order = order % 9 if order is not None else None
            if order is not None and (int(metadata["execution_candidate_order"]) != price_order
                or int(metadata["controller_order"]) != order or bool(metadata["updates_enabled"]) != updates):
                raise RuntimeError("Parent selected price/update metadata changed")
            learners[policy] = {"selected_controller_order": order, "price_order": price_order,
                "price_kind": "risk_first" if price_order == 8 else "finite" if price_order is not None else None,
                "annual_price": float(priced.PRICES[price_order]) if price_order is not None and price_order < 8 else None,
                "updates_enabled": bool(updates) if order is not None else None,
                "parent_row_metadata": metadata}
        # The fixed parent's own selected identities must also match its copy
        # in the incremental study; financial reproduction happens after replay.
        old_fixed = prototype(price_rows, context, priced.LEARNER, ids)
        if bool(old_fixed["assigned"]) != bool(learners[POLICIES[0]]["parent_row_metadata"]["assigned"]):
            raise RuntimeError("Fixed learner service status differs across parents")
        result[context] = {"training_ids": fit["training_ids"], "evaluation_ids": ids,
            "scope": "full_development" if full else "outer_oof", "outer_fold": fit["context_id"] or None,
            "menu": [menu[role] for role in bounded.ROLES], "learners": learners,
            "reference_row_metadata": prototype(price_rows, context, POLICIES[2], ids)}
    return result


def validate_protocol(protocol, *, ready=True):
    if ([(r["label"], r["cost_rate"]) for r in protocol["rates"]] != list(RATES)
        or any(isinstance(r["cost_rate"], bool) for r in protocol["rates"])
        or protocol["policies"] != list(POLICIES) or protocol["client"]["horizon"] != H
        or protocol["client"]["loss_tolerance"] != L or protocol["data"]["allowed_scenario_ids"] != [1, 120]
        or protocol["data"]["forbidden_scenario_ids"] != [121, 180]):
        raise ValueError("Prospective rates/client/policies/development scope changed")
    if ready and not (protocol.get("execution_ready") is True and protocol.get("parents_ready") is True):
        raise RuntimeError("Completed audited parents and exact source freeze are required")


def checked_parent(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    audit = json.loads((directory / "independent_audit.json").read_text())
    if (audit.get("status") != "passed" or manifest["sources"] != manifest["sources_at_end"]
        or manifest["input_hashes"] != manifest["input_hashes_at_end"] or manifest["final_scenarios_loaded"] is not False
        or manifest["scenario_ids_loaded"] != list(range(1, 121))):
        raise RuntimeError("Parent completion/audit/scope snapshot failed")
    inputs = {str((directory / name).relative_to(ROOT)): digest for name, digest in manifest["outputs_sha256"].items()}
    for mapping in (manifest["sources"], manifest["input_hashes"]):
        for key, digest in mapping.items():
            if key in inputs and inputs[key] != digest: raise RuntimeError("Parent source/input/output hash claims conflict")
            inputs[key] = digest
    inputs[str((directory / "manifest.json").relative_to(ROOT))] = sha(directory / "manifest.json")
    inputs[str((directory / "independent_audit.json").relative_to(ROOT))] = sha(directory / "independent_audit.json")
    start, end = audit.get("source_and_artifact_sha256_start"), audit.get("source_and_artifact_sha256_end")
    if start is not None:
        if start != end: raise RuntimeError("Parent independent-audit snapshot changed")
        for key, digest in start.items():
            if key in inputs and inputs[key] != digest: raise RuntimeError("Parent audit/manifest hash claims conflict")
            inputs[key] = digest
    if any(sha(ROOT / key) != value for key, value in inputs.items()):
        raise RuntimeError("Parent source/input/output/audit is stale")
    return manifest, inputs


def sources():
    names = [*bounded.sources(), *priced.source_manifest(), "scripts/run_selected_controller_cost_stress.py",
        "tests/test_selected_controller_cost_stress.py", "docs/selected_controller_cost_stress_protocol.json"]
    return {name: sha(ROOT / name) for name in dict.fromkeys(names)}


def freeze_protocol():
    """Provenance-only finalization, after explicit parent-ready notification."""
    protocol = json.loads(PROTOCOL.read_text()); validate_protocol(protocol, ready=False)
    if protocol["execution_ready"]: raise RuntimeError("Protocol is already frozen")
    inputs = {}
    for directory in PARENTS.values():
        _, dependency = checked_parent(directory)
        for key, digest in dependency.items():
            if key in inputs and inputs[key] != digest: raise RuntimeError("Parent shared dependency hashes disagree")
            inputs[key] = digest
    fits = json.loads((reference.STRICT / "nested_context_refits.json").read_text()); reference.validate_contexts(fits)
    plan = build_controller_plan(fits, json.loads((PARENTS["fixed"] / "context_selections.json").read_text()),
        json.loads((PARENTS["expanded"] / "context_selections.json").read_text()),
        pd.read_csv(PARENTS["fixed"] / "client_paths.csv"), pd.read_csv(PARENTS["expanded"] / "client_paths.csv"))
    protocol["prospective_draft_sha256"] = sha(PROTOCOL)
    protocol["frozen_controller_plan"] = plan
    protocol["frozen_input_sha256"] = inputs
    protocol["implementation_freeze_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["execution_sources_sha256"] = {k: v for k, v in sources().items() if k != str(PROTOCOL.relative_to(ROOT))}
    protocol["parents_ready"] = protocol["execution_ready"] = True
    write_json(PROTOCOL, protocol)
    print(json.dumps({"protocol_sha256": sha(PROTOCOL), "parents_audit_passed": True}))


def frozen_inputs(protocol):
    expected = protocol["frozen_input_sha256"]
    if {key: sha(ROOT / key) for key in expected} != expected: raise RuntimeError("Frozen parent/input changed")
    key = str(PROTOCOL.relative_to(ROOT))
    if {k: v for k, v in sources().items() if k != key} != protocol["execution_sources_sha256"]:
        raise RuntimeError("Implementation/test/dependency snapshot changed")
    return expected.copy()


def compare_frame(actual, expected, fields, keys):
    if len(actual) != len(expected) or actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise RuntimeError("Base parent comparison identity/denominator changed")
    a, e = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(e.index): raise RuntimeError("Parent scenario identities changed")
    maximum = 0.
    for key in fields:
        if key in keys: continue
        if key not in a or key not in e: raise RuntimeError(f"Missing reproduced parent field:{key}")
        keep = e[key].notna()
        if not np.array_equal(a[key].isna(), e[key].isna()): raise RuntimeError(f"Parent NULL mask changed:{key}")
        if not keep.any(): continue
        if pd.api.types.is_numeric_dtype(e[key]) and not pd.api.types.is_bool_dtype(e[key]):
            av, ev = a.loc[keep, key].to_numpy(float), e.loc[keep, key].to_numpy(float)
            np.testing.assert_allclose(av, ev, atol=2e-10, rtol=5e-13)
            maximum = max(maximum, float(np.max(np.abs(av - ev), initial=0)))
        else: np.testing.assert_array_equal(a.loc[keep, key], e.loc[keep, key])
    return {"rows": len(actual), "fields": len(fields), "max_abs_error": maximum, "NULL_masks_and_identities_match": True}


def compare_parent_trace(arm, ids, archive, prefix, selected_price):
    """Reproduce behavior/Q/state, honestly excluding legacy price catalog."""
    keys = [key for key in archive.files if key.startswith(prefix + "__")]
    if not keys: raise RuntimeError("Missing selected parent trace")
    maximum, copied_catalog, checked = 0., None, 0
    for key in keys:
        field = key[len(prefix) + 2:]; expected = archive[key]
        if field == "annual_prices":
            if not np.any(expected == selected_price): raise RuntimeError("Selected price absent from parent legacy catalog")
            copied_catalog = expected.tolist()
            continue
        actual = np.asarray(ids) if field == "scenario_ids" else arm.get(field)
        if actual is None or actual.shape != expected.shape: raise RuntimeError(f"Parent field/shape missing:{field}")
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=5e-13, equal_nan=True)
            good = np.isfinite(expected)
            maximum = max(maximum, float(np.max(np.abs(actual[good] - expected[good]), initial=0)))
        else: np.testing.assert_array_equal(actual, expected)
        checked += 1
    return {"behavior_state_Q_fields": checked, "max_abs_error": maximum,
            "excluded_metadata_fields": ["annual_prices"] if copied_catalog is not None else [],
            "parent_catalog_metadata": {"finite_values_or_null": [None if not np.isfinite(x) else float(x) for x in copied_catalog],
                "risk_first_indices": np.flatnonzero(np.isposinf(copied_catalog)).tolist()} if copied_catalog is not None else None,
            "parent_catalog_risk_first_label_present": bool(copied_catalog is not None and np.isposinf(copied_catalog).any()),
            "new_actual_price_catalog_size": len(arm["annual_prices"]) if "annual_prices" in arm else None}


def augment_reference(arm, target_index):
    result = dict(arm); n, points = arm["real_wealth"].shape; h = points - 1
    result["current_target_index"] = np.full((n, points), target_index, dtype=np.int64)
    for key in ("current_target_before", "current_target_after"): result[key] = np.full((n, h), target_index, dtype=np.int64)
    result["target_updated"] = np.zeros((n, h), bool)
    result["action_index"] = np.where(arm["target_execution"], target_index + 1 if target_index >= 0 else -1, 0)
    return result


def evaluate_frozen_controller(actual, inflation, train, train_inflation, matrix, chosen, cost_rate):
    """Replay exactly one frozen controller, or keep inherited Review null."""
    order = chosen["selected_controller_order"]
    if order is None: return None
    p = chosen["price_order"]; updates = chosen["updates_enabled"]
    if (isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < 9
        or not isinstance(updates, bool) or order != p + (9 if updates else 0)):
        raise RuntimeError("Frozen mode/controller/price identity changed")
    expected_kind = "risk_first" if p == 8 else "finite"
    if chosen["price_kind"] != expected_kind or (p < 8 and chosen["annual_price"] != float(priced.PRICES[p])):
        raise RuntimeError("Frozen annual price changed")
    if p == 8 and chosen["annual_price"] is not None: raise RuntimeError("Risk-first is a label, not a numeric infinity utility")
    price = float(priced.PRICES[p])
    family = simulate_controllers(actual, inflation, train, train_inflation, matrix, np.array([price]),
        initial_target_index=1, updates_enabled=updates, L=L, horizon=H, cost_rate=cost_rate)
    if family["real_wealth"].shape[1] != 1: raise RuntimeError("An unselected controller was evaluated")
    return bounded.extract_controller(family, 0)


def financial_rows(arm, ids, common):
    rows = priced.financial_rows(arm, ids, common)
    for i, row in enumerate(rows):
        row.update({"target_update_fraction": float(arm["target_updated"][i].mean()),
            "target_update_count": int(arm["target_updated"][i].sum()), "terminal_target_index": int(arm["current_target_index"][i, -1])})
    return rows


def gates_for(arm, ids, common, learner):
    rows = []
    for i, sid in enumerate(ids):
        for t in range(H):
            gate = learner and t > 0 and not arm["pending_at_decision"][i, t]
            row = {**common, "scenario_id": sid, "year": t + 1, "action": str(arm["actions"][i, t]),
                "action_index": int(arm["action_index"][i, t]), "actual_trade": bool(arm["trades"][i, t]),
                "target_before": int(arm["current_target_before"][i, t]), "target_after": int(arm["current_target_after"][i, t]),
                "target_updated": bool(arm["target_updated"][i, t]), "pending_at_decision": bool(arm["pending_at_decision"][i, t]),
                "ever_breached": bool(arm["ever_breached"][i, t]), "automatic_gate_evaluated": bool(gate),
                "nominal_fee": float(arm["fees"][i, t]), "nominal_moved": float(arm["moved"][i, t]),
                "forecast_n": int(arm["forecast_n"][i, t]) if gate else None,
                "selected_score": float(arm["selected_score"][i, t]) if gate and common["price_kind"] == "finite" else None}
            row["gate_reason"] = str(arm["gate_reason"][i, t]) if learner else "specified_reference"
            for a in range(4):
                available = gate and bool(arm["available_actions"][i, t, a])
                row[f"action_{a}_available"] = bool(available)
                row[f"action_{a}_qg"] = float(arm["qg_actions"][i, t, a]) if available else None
                row[f"action_{a}_bad_count"] = int(arm["qbad_actions"][i, t, a]) if available else None
                row[f"action_{a}_score"] = float(arm["action_scores"][i, t, a]) if available and common["price_kind"] == "finite" else None
            rows.append(row)
    return rows


def paired_table(frame, *, contrasts=False):
    comparisons = [(p, p, label, "base") for label, _ in RATES for p in POLICIES] if not contrasts else [
        (a, b, label, label) for label, _ in RATES for a, b in ((POLICIES[1], POLICIES[0]), (POLICIES[0], POLICIES[2]), (POLICIES[1], POLICIES[2]), (POLICIES[3], POLICIES[2]))]
    keys = ["scope", "context", "client", "scenario_id"]
    rows = []
    for policy, comparison, label, compare_label in comparisons:
        a = frame.loc[(frame.policy == policy) & (frame.cost_label == label)]
        b = frame.loc[(frame.policy == comparison) & (frame.cost_label == compare_label)]
        joined = a.merge(b, on=keys, suffixes=("", "_comparison"), validate="one_to_one")
        if len(joined) != len(a) or len(joined) != len(b): raise RuntimeError("Missing original paired cost/policy identities")
        for item in joined.to_dict("records"):
            common = bool(item["assigned"] and item["assigned_comparison"])
            rows.append({**{k: item[k] for k in keys}, "cost_label": label, "policy": policy,
                "comparison_cost_label": compare_label, "comparison_policy": comparison,
                "assigned": bool(item["assigned"]), "comparison_assigned": bool(item["assigned_comparison"]), "common_auto": common,
                **{"delta_" + f: float(item[f]) - float(item[f + "_comparison"]) if common else None for f in (*reference.METRICS, *reference.ACCOUNTS, "breach", "excess", "target_update_fraction", "target_update_count")}})
    pairs = pd.DataFrame(rows); summaries = []
    for labels, group in pairs.groupby(["scope", "cost_label", "policy", "comparison_cost_label", "comparison_policy"], sort=False):
        good = group.loc[group.common_auto]
        summaries.append({**dict(zip(("scope", "cost_label", "policy", "comparison_cost_label", "comparison_policy"), labels)),
            "original_pairs": len(group), "common_auto_pairs": len(good), "assigned_paths": int(group.assigned.sum()),
            "comparison_assigned_paths": int(group.comparison_assigned.sum()),
            **{"mean_" + col: float(good[col].mean()) if len(good) else None for col in pairs if col.startswith("delta_")}})
    return pairs, pd.DataFrame(summaries)


def decision_changes(gates, paths):
    keys = ["context", "client", "scenario_id", "policy", "year"]
    compare = gates.loc[gates.cost_label == "base", keys + ["action", "actual_trade", "pending_at_decision", "ever_breached", "target_after", "target_updated"]]
    rows = []
    for label, rate in RATES:
        group = gates.loc[gates.cost_label == label]
        if len(group) != len(compare): raise RuntimeError("Changed served decision denominator across frozen cost settings")
        joined = group.merge(compare, on=keys, suffixes=("", "_base"), validate="one_to_one")
        if len(joined) != len(group): raise RuntimeError("Unpaired cost/base decision identities")
        for labels, part in joined.groupby(keys[:-1], sort=False):
            both = (part.year > 1) & ~part.ever_breached & ~part.ever_breached_base
            row = {**dict(zip(keys[:-1], labels)), "cost_label": label, "cost_rate": rate, "all_decisions": len(part),
                "both_prebreach_postinitial_decisions": int(both.sum())}
            for field in ("action", "actual_trade", "pending_at_decision", "target_after", "target_updated"):
                changed = part[field] != part[field + "_base"]
                row[field + "_changed_count"] = int(changed.sum()); row[field + "_changed_fraction"] = float(changed.mean())
            row["both_prebreach_action_changed_count"] = int(((part.action != part.action_base) & both).sum())
            rows.append(row)
    counts = pd.DataFrame(rows)
    fields = ["cost_label", "cost_rate", "context", "client", "scenario_id", "policy"]
    merged = paths[[*fields, "scope", "assigned", "horizon"]].merge(counts, on=fields, how="left", validate="one_to_one")
    if (merged.loc[merged.assigned, "all_decisions"].isna().any() or not merged.loc[merged.assigned, "all_decisions"].eq(H).all()
        or merged.loc[~merged.assigned, "all_decisions"].notna().any()): raise RuntimeError("Missing served H30 trace or imputed Review decisions")
    summary = []
    for labels, part in merged.groupby(["scope", "cost_label", "policy"], sort=False):
        auto = part.loc[part.assigned]; denom = int(auto.all_decisions.sum()) if len(auto) else None
        both = int(auto.both_prebreach_postinitial_decisions.sum()) if len(auto) else None
        row = {**dict(zip(("scope", "cost_label", "policy"), labels)), "original_paths": len(part), "assigned_paths": len(auto),
            "review_paths": len(part) - len(auto), "all_decisions": denom, "both_prebreach_postinitial_decisions": both}
        for f in ("action", "actual_trade", "pending_at_decision", "target_after", "target_updated"):
            row[f + "_changed_count"] = int(auto[f + "_changed_count"].sum()) if len(auto) else None
            row[f + "_changed_fraction"] = row[f + "_changed_count"] / denom if denom else None
        row["both_prebreach_action_changed_fraction"] = float(auto.both_prebreach_action_changed_count.sum()) / both if both else None
        summary.append(row)
    return merged, pd.DataFrame(summary)


def check_coverage(frame):
    if len(frame) != 4800 or frame.duplicated(["context", "scenario_id", "policy", "cost_label"]).any():
        raise RuntimeError("Expected4800 original C full/OOF policy/cost rows")
    if (set(frame.policy) != set(POLICIES) or set(frame.cost_label) != {x[0] for x in RATES}
        or set(frame.scope) != {"full_development", "outer_oof"} or set(frame.client) != {"C"}
        or not frame.horizon.eq(H).all() or not frame.loss_tolerance.eq(L).all()
        or not frame.cost_rate.eq(frame.cost_label.map(dict(RATES))).all()):
        raise RuntimeError("Unexpected policy/rate/client/H/L scope")
    for (_, _, _), part in frame.groupby(["scope", "policy", "cost_label"]):
        if len(part) != 120 or set(part.scenario_id) != set(range(1, 121)): raise RuntimeError("Original120 population changed")
    if not frame.loc[~frame.assigned, list(FINANCIAL)].isna().all().all(): raise RuntimeError("Review financial outcome imputed")
    for (_, _), part in frame.groupby(["context", "policy"]):
        if (part.groupby("cost_label").assigned.sum().nunique() != 1
            or part.groupby("scenario_id").assigned.nunique().gt(1).any()): raise RuntimeError("Cost changed frozen admission")


def main():
    started = time.perf_counter(); protocol = json.loads(PROTOCOL.read_text()); validate_protocol(protocol)
    input_start, source_start = frozen_inputs(protocol), sources()
    parent_manifests = {name: checked_parent(path)[0] for name, path in PARENTS.items()}
    fits = json.loads((reference.STRICT / "nested_context_refits.json").read_text()); reference.validate_contexts(fits)
    mapping = json.loads((reference.STRICT / "candidate_mapping.json").read_text())
    plan = protocol["frozen_controller_plan"]
    fresh = build_controller_plan(fits, json.loads((PARENTS["fixed"] / "context_selections.json").read_text()),
        json.loads((PARENTS["expanded"] / "context_selections.json").read_text()),
        pd.read_csv(PARENTS["fixed"] / "client_paths.csv"), pd.read_csv(PARENTS["expanded"] / "client_paths.csv"))
    if fresh != plan: raise RuntimeError("Frozen controller choices/targets/Review status changed")
    returns, inflation, ids = load_paths(reference.DATA, range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)): raise RuntimeError("Development IDs changed")
    digest = hashlib.sha256()
    for a in (ids, returns, inflation): digest.update(np.ascontiguousarray(a).tobytes())
    if (digest.hexdigest() != parent_manifests["fixed"]["development_array_sha256"]
        or digest.hexdigest() != parent_manifests["expanded"]["development_arrays_sha256"]):
        raise RuntimeError("Guarded development arrays differ from parents")
    rows, gates, archive, checks, catalog = [], [], {}, {}, {}
    parent_archives = {name: np.load(path / "actual_traces.npz", allow_pickle=False) for name, path in PARENTS.items()}
    for label, rate in RATES:
        for context, entry in plan.items():
            train_ids, eval_ids = entry["training_ids"], entry["evaluation_ids"]
            tx, ex = np.asarray(train_ids) - 1, np.asarray(eval_ids) - 1
            matrix = np.array([reference.validate_product(item, mapping) for item in entry["menu"]])
            commons = {p: {**item["parent_row_metadata"], "cost_label": label, "cost_rate": rate} for p, item in entry["learners"].items()}
            commons[POLICIES[2]] = {**entry["reference_row_metadata"], "cost_label": label, "cost_rate": rate}
            commons[POLICIES[3]] = {"context": context, "scope": entry["scope"], "outer_fold": entry["outer_fold"],
                "client": "C", "horizon": H, "loss_tolerance": L, "risk_assignment": "Balanced", "risk_assigned": True,
                "assignment": "EqualThirdsReference", "assigned": True, "policy": POLICIES[3], "price_kind": None,
                "annual_price": None, "updates_enabled": False, "selected_execution_status": "hypothetical_research_reference",
                "cost_label": label, "cost_rate": rate}
            arms = {}
            for policy in POLICIES[:2]:
                chosen = entry["learners"][policy]; common = commons[policy]
                if chosen["selected_controller_order"] is None:
                    rows.extend({**common, "scenario_id": sid, **dict.fromkeys(FINANCIAL)} for sid in eval_ids)
                    continue
                arms[policy] = evaluate_frozen_controller(returns[ex, :H], inflation[ex, :H], returns[tx, :H], inflation[tx, :H], matrix, chosen, rate)
            balanced = simulate_paired(returns[ex, :H], inflation[ex, :H], matrix[1], L, horizon=H, cost_rate=rate)
            arms[POLICIES[2]] = augment_reference(reference.squeeze_pending(balanced["pending_hold"]), 1)
            equal = simulate_paired(returns[ex, :H], inflation[ex, :H], np.array([1 / 3] * 3), L, horizon=H, cost_rate=rate)
            arms[POLICIES[3]] = augment_reference(reference.squeeze_pending(equal["annual"]), -1)
            for policy, arm in arms.items():
                common = commons[policy]
                rows.extend(financial_rows(arm, eval_ids, common)); gates.extend(gates_for(arm, eval_ids, common, policy in POLICIES[:2]))
                prefix = f"{label}__{context}__C__{policy}"
                archive[prefix + "__scenario_ids"] = np.asarray(eval_ids)
                for field, value in arm.items():
                    if isinstance(value, np.ndarray): archive[prefix + "__" + field] = value
                if label == "base":
                    if policy in POLICIES[:2]:
                        price = np.inf if entry["learners"][policy]["price_kind"] == "risk_first" else entry["learners"][policy]["annual_price"]
                        key = f"{context}__C__{policy}"
                        checks["bounded__" + key] = compare_parent_trace(arm, eval_ids, parent_archives["expanded"], key, price)
                        catalog["bounded__" + key] = checks["bounded__" + key]["parent_catalog_metadata"]
                        if policy == POLICIES[0]:
                            key = f"{context}__C__{priced.LEARNER}"
                            checks["price__" + key] = compare_parent_trace(arm, eval_ids, parent_archives["fixed"], key, price)
                    elif policy == POLICIES[2]:
                        key = f"{context}__C__{policy}"
                        checks["price__" + key] = compare_parent_trace(arm, eval_ids, parent_archives["fixed"], key, np.nan)
            print(json.dumps({"cost_label": label, "context": context, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    for a in parent_archives.values(): a.close()
    frame = pd.DataFrame(rows); check_coverage(frame)
    for parent, filename, source_policy, policy in (("expanded", "client_paths.csv", POLICIES[0], POLICIES[0]),
        ("expanded", "client_paths.csv", POLICIES[1], POLICIES[1]), ("fixed", "client_paths.csv", priced.LEARNER, POLICIES[0]),
        ("fixed", "client_paths.csv", POLICIES[2], POLICIES[2])):
        expected = pd.read_csv(PARENTS[parent] / filename); expected = expected.loc[(expected.client == "C") & (expected.policy == source_policy)].copy()
        expected["policy"] = policy
        actual = frame.loc[(frame.cost_label == "base") & (frame.policy == policy)]
        checks[f"{parent}__{policy}__CSV"] = compare_frame(actual, expected, list(expected), ["context", "client", "scenario_id", "policy"])
    summary = []
    for label, rate in RATES:
        part = frame.loc[frame.cost_label == label]
        for row in reference.summarize(part):
            selected = part.loc[(part.scope == row["scope"]) & (part.policy == row["policy"]) & part.assigned]
            row.update({"cost_label": label, "cost_rate": rate,
                "mean_target_update_fraction": float(selected.target_update_fraction.mean()) if len(selected) else None,
                "mean_target_update_count": float(selected.target_update_count.mean()) if len(selected) else None})
            summary.append(row)
    default_pairs, default_summary = paired_table(frame)
    contrasts, contrast_summary = paired_table(frame, contrasts=True)
    gate_frame = pd.DataFrame(gates); changes, change_summary = decision_changes(gate_frame, frame)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in {"client_paths.csv": frame, "summary.csv": pd.DataFrame(summary), "paired_vs_default_paths.csv": default_pairs,
        "paired_vs_default_summary.csv": default_summary, "policy_contrasts_paths.csv": contrasts, "policy_contrasts_summary.csv": contrast_summary,
        "decision_gates.csv": gate_frame, "decision_changes_by_path.csv": changes, "decision_changes_summary.csv": change_summary}.items():
        table.to_csv(OUT / name, index=False)
    np.savez_compressed(OUT / "actual_traces.npz", **archive)
    write_json(OUT / "frozen_controller_plan.json", plan); write_json(OUT / "parent_catalog_metadata.json", catalog)
    write_json(OUT / "base_reproduction.json", checks)
    input_end, source_end = frozen_inputs(protocol), sources()
    if input_start != input_end or source_start != source_end: raise RuntimeError("Source/input changed during selected-controller cost stress")
    names = [p.name for p in OUT.iterdir() if p.name not in ("manifest.json", "independent_audit.json")]
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": source_start, "sources_at_end": source_end, "input_hashes": input_start, "input_hashes_at_end": input_end,
        "outputs_sha256": {name: sha(OUT / name) for name in names}, "scenario_ids_loaded": ids.tolist(), "final_scenarios_loaded": False,
        "development_array_sha256": digest.hexdigest(), "client_path_rows": len(frame), "decision_rows": len(gate_frame),
        "cost_rates": [{"label": label, "cost_rate": rate} for label, rate in RATES], "controller_parameters_frozen_at_default_cost": True,
        "same_actual_forecast_known_cost": True, "training_risk_recalibrated_at_stress_costs": False,
        "scope": "Original C H30 fixed selected controllers, known-cost sensitivity; no candidate re-selection, risk certification or best-cost winner"})
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "client_path_rows": len(frame), "decision_rows": len(gate_frame)}))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "freeze": freeze_protocol()
    elif len(sys.argv) == 1: main()
    else: raise SystemExit("Use no arguments to execute the frozen study, or 'freeze' only after audited parents are ready")
