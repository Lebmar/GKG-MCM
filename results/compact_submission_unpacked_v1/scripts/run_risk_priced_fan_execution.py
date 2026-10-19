"""Frozen-product, training-only finite-family risk-priced execution learner.

All nine candidates are replayed on the context's training scenarios.  A single
training-selected candidate is applied to its outer tests, or execution Review
is preserved as null.  No unselected OOF candidate is simulated or ranked.
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
from safeinvest.data import load_paths
from safeinvest_research.metrics import NUMERIC_TOL

PROTOCOL = ROOT / "docs/risk_priced_fan_execution_protocol.json"
OUT = ROOT / "results/model_review/risk_priced_fan_execution_v1"
STRICT = ROOT / "results/model_review/positive_asset_specification_v1"
PENDING = ROOT / "results/model_review/review_pending_execution_v1"
CACHE = ROOT / "results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz"
FINITE_PRICES = (0., .001, .003, .01, .03, .1, .3, 1.)
PRICES = np.array([*FINITE_PRICES, np.inf])
CANDIDATE_IDS = tuple([f"eta_{value:g}" for value in FINITE_PRICES] + ["risk_first"])
LEARNER = "risk_priced_learner"
POLICIES = ("annual_then_pending", LEARNER)
FINANCIAL = (*reference.METRICS, *reference.ACCOUNTS, "breach", "excess", "first_crossing")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_manifest():
    paths = [Path(__file__).resolve(), ROOT / "tests/test_risk_priced_fan_execution.py", PROTOCOL,
             ROOT / "research/experimental_execution/risk_priced_fan_rollout.py", ROOT / "tests/test_risk_priced_fan_rollout.py", ROOT / "research/experimental_execution/fan_rollout.py",
             ROOT / "research/experimental_execution/review_pending.py", ROOT / "scripts/run_fan_rollout_execution.py",
             ROOT / "research/safeinvest_research/metrics.py", ROOT / "src/safeinvest/core.py", ROOT / "src/safeinvest/data.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def frozen_inputs(protocol):
    result = {name: sha(ROOT / name) for name in protocol["frozen_input_sha256"]}
    if result != protocol["frozen_input_sha256"]: raise RuntimeError("A prospectively frozen dependency changed")
    for path in [CACHE, CACHE.with_suffix(".json"), PENDING / "manifest.json", PENDING / "client_path_metrics.csv", PENDING / "independent_audit.json",
                 reference.OUT / "manifest.json"]:
        result[str(path.relative_to(ROOT))] = sha(path)
    previous = json.loads((reference.OUT / "manifest.json").read_text())
    for key in ["scripts/run_fan_rollout_execution.py", "src/safeinvest/core.py", "src/safeinvest/data.py", "research/safeinvest_research/metrics.py"]:
        if sha(ROOT / key) != previous["sources"][key]: raise RuntimeError("Read-only reference implementation changed")
    return result


def candidate_meta(order, horizon):
    if not 0 <= order < 9: raise ValueError("Candidate order must be0..8")
    finite = order < 8
    return {"execution_candidate_order": int(order), "execution_candidate_id": CANDIDATE_IDS[order],
            "price_kind": "finite" if finite else "risk_first", "annual_price": float(PRICES[order]) if finite else None,
            "original_H_total_price": float(horizon * PRICES[order]) if finite else None}


def select_training_candidate(summaries, reference_breach_count, n_training, *, alpha=.10, tolerance=1e-12):
    """A single global growth band; cap is an integer empirical path count."""
    if isinstance(n_training, bool) or not isinstance(n_training, (int, np.integer)) or n_training <= 0:
        raise ValueError("n_training must be a positive integer")
    if (isinstance(reference_breach_count, bool) or int(reference_breach_count) != reference_breach_count
            or not 0 <= reference_breach_count <= n_training): raise ValueError("Invalid reference whole-policy breach count")
    if len(summaries) != 9 or [row["execution_candidate_order"] for row in summaries] != list(range(9)):
        raise ValueError("All nine candidates must be supplied in predeclared order")
    cap = min(int(reference_breach_count), int(np.floor(alpha * n_training + NUMERIC_TOL)))
    classified = []
    for row in summaries:
        count = row["breach_count"]
        if (isinstance(count, bool) or int(count) != count or not 0 <= count <= n_training
                or row["n_training"] != n_training or not np.isfinite(row["mean_log_growth"])):
            raise ValueError("Candidate growth/risk summaries must describe all training paths")
        classified.append({**row, "training_breach_cap": cap, "training_eligible": int(count) <= cap})
    eligible = [row for row in classified if row["training_eligible"]]
    maximum = max(row["mean_log_growth"] for row in eligible) if eligible else None
    selected = next((row for row in eligible if row["mean_log_growth"] >= maximum - tolerance), None)
    return {"n_training": int(n_training), "service_alpha_reference": float(alpha), "reference_training_breach_count": int(reference_breach_count),
            "absolute_alpha_count_cap": int(np.floor(alpha * n_training + NUMERIC_TOL)), "training_breach_cap": cap,
            "candidate_summaries": classified, "maximum_eligible_mean_log_growth": maximum,
            "selected_candidate_order": selected["execution_candidate_order"] if selected else None,
            "execution_status": "automated" if selected else "execution_review",
            "execution_review_reason": None if selected else "execution_family_no_admissible_policy",
            "global_growth_tie_tolerance": tolerance, "selected_from_training_only": True}


def extract_price(simulation, order):
    """Remove only price axis1; CPI never has a price axis, including H+1=p."""
    real = np.asarray(simulation["real_wealth"])
    if real.ndim != 3 or not 0 <= order < real.shape[1]: raise ValueError("Require(n,p,H+1) wealth and a valid price order")
    n, prices = real.shape[:2]; result = {}
    excluded = {"cpi", "targets", "target", "loss_tolerances", "annual_prices", "prices"}
    for key, value in simulation.items():
        if not isinstance(value, np.ndarray): continue
        if key in excluded: result[key] = value.copy()
        elif value.ndim >= 2 and value.shape[:2] == (n, prices): result[key] = value[:, order].copy()
        else: raise ValueError(f"Unexpected price-axis shape for{key}")
    return result


def financial_rows(arm, ids, common):
    values = reference.path_values(arm, common["horizon"])
    result = []
    for i, scenario in enumerate(ids):
        first = int(arm["first_crossing"][i])
        row = {**common, "scenario_id": int(scenario), "assigned": True,
               **{name: float(values[name][i]) for name in (*reference.METRICS, *reference.ACCOUNTS)},
               "breach": bool(arm["ever_breached"][i, -1]), "excess": float(max(values["mdd"][i] - common["loss_tolerance"], 0.)),
               "first_crossing": first if first >= 0 else None}
        if row["breach"] != (row["mdd"] - common["loss_tolerance"] > NUMERIC_TOL): raise RuntimeError("Whole-policy breach history/metric differ")
        result.append(row)
    return result


def null_rows(ids, common):
    return [{**common, "scenario_id": int(scenario), "assigned": False, **dict.fromkeys(FINANCIAL)} for scenario in ids]


def training_summary(rows, order, horizon):
    frame = pd.DataFrame(rows)
    if not frame.assigned.all() or frame.scenario_id.nunique() != len(frame): raise ValueError("Each candidate needs every unique assigned training path")
    return {**candidate_meta(order, horizon), "n_training": len(frame), "breach_count": int(frame.breach.astype(bool).sum()),
            "mean_log_growth": float(frame.log_growth.mean()), "mean_nominal_fees": float(frame.nominal_fees.mean()),
            "mean_nominal_moved": float(frame.nominal_moved.mean()), "mean_trade_fraction": float(frame.trade_fraction.mean())}


def evaluate_selected(simulator, actual_returns, actual_inflation, training_returns, training_inflation, target, order, *, L, horizon):
    """Evaluate zero or one candidate, never search the outer family."""
    if order is None: return None
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order < 9: raise ValueError("Invalid selected training candidate")
    family = simulator(actual_returns, actual_inflation, training_returns, training_inflation, target,
                       np.array([PRICES[order]]), L=L, cost_rate=.0025, horizon=horizon)
    if family["real_wealth"].shape[1] != 1: raise RuntimeError("An unselected OOF price was evaluated")
    return extract_price(family, 0)


def verify_training_annual(arm, ids, horizon, global_index, cache):
    values = reference.path_values(arm, horizon)
    for field in reference.METRICS:
        np.testing.assert_allclose(values[field], cache[f"H{horizon}__{field}"][np.asarray(ids) - 1, global_index], atol=2e-10, rtol=5e-13)
    for field, cached in [("nominal_fees", "fees"), ("nominal_moved", "moved"), ("trade_fraction", "trade_fraction")]:
        np.testing.assert_allclose(values[field], cache[f"H{horizon}__{cached}"][np.asarray(ids) - 1, global_index], atol=2e-10, rtol=5e-13)


def verify_price_zero(price_zero, old):
    errors = {}
    for key in ["holdings", "posttrade_holdings", "real_wealth", "nominal_wealth", "fees", "moved", "real_peak", "running_mdd", "first_crossing", "ever_breached", "trades", "actions", "target_execution", "pending_at_decision"]:
        a, b = price_zero[key], old[key]
        if np.issubdtype(np.asarray(a).dtype, np.number) and np.asarray(a).dtype != bool:
            np.testing.assert_allclose(a, b, atol=2e-10, rtol=5e-13); errors[key] = float(np.max(np.abs(a - b), initial=0.))
        elif not np.array_equal(a, b): raise RuntimeError("Eta0 does not reproduce the old training growth-only policy: " + key)
    return errors


def gate_records(arm, ids, common, order):
    records = []; H = common["horizon"]; finite = order < 8
    for i, sid in enumerate(ids):
        for t in range(H):
            active = t > 0 and not arm["pending_at_decision"][i, t]
            row = {**common, "scenario_id": int(sid), "year": t + 1, **candidate_meta(order, H),
                   "action": str(arm["actions"][i, t]), "actual_trade": bool(arm["trades"][i, t]),
                   "nominal_fee": float(arm["fees"][i, t]), "nominal_moved": float(arm["moved"][i, t]),
                   "pending_at_decision": bool(arm["pending_at_decision"][i, t]),
                   "qg_hold": None, "qg_restore": None, "qbad_hold_count": None, "qbad_restore_count": None,
                   "forecast_n": None, "computed_score_advantage": None, "unpriced_growth_advantage": None,
                   "predicted_breach_probability_change": None, "restore_sacrifices_unpriced_growth": None,
                   "core_gate_reason": str(arm["gate_reason"][i, t])}
            if active:
                gh, gr = float(arm["qg_hold"][i, t]), float(arm["qg_restore"][i, t])
                kh, kr, n = int(arm["qbad_hold_count"][i, t]), int(arm["qbad_restore_count"][i, t]), int(arm["forecast_n"][i, t])
                if not np.isfinite(gh + gr) or not 0 <= kh <= n or not 0 <= kr <= n or n <= 0: raise RuntimeError("Invalid selected gate forecasts")
                growth, probability = gr - gh, (kr - kh) / n
                score = growth - H * PRICES[order] * probability if finite else None
                expected_restore = score > NUMERIC_TOL if finite else kr < kh or (kr == kh and growth > NUMERIC_TOL)
                if (row["action"] == "restore") != expected_restore: raise RuntimeError("Selected action differs from the declared score/risk-first rule")
                if finite and not np.isclose(score, arm["predicted_score"][i, t], atol=1e-12, rtol=1e-11): raise RuntimeError("Original-H total score differs")
                row.update({"qg_hold": gh, "qg_restore": gr, "qbad_hold_count": kh, "qbad_restore_count": kr, "forecast_n": n,
                            "computed_score_advantage": score, "unpriced_growth_advantage": growth,
                            "predicted_breach_probability_change": probability,
                            "restore_sacrifices_unpriced_growth": bool(expected_restore and growth < -NUMERIC_TOL),
                            "runner_gate_reason": "finite_positive_score" if finite and expected_restore else "finite_nonpositive_score" if finite else "risk_first_lower_bad" if kr < kh else "risk_first_higher_bad" if kr > kh else "risk_first_growth_tie"})
            else:
                row["runner_gate_reason"] = "mandatory_inception" if t == 0 else "pending_hold"
            records.append(row)
    return records


def paired_outputs(frame):
    keys = ["scope", "context", "outer_fold", "client", "scenario_id"]
    joined = frame.loc[frame.policy == LEARNER].merge(frame.loc[frame.policy == "annual_then_pending"], on=keys, suffixes=("_learner", "_reference"), validate="one_to_one")
    rows = []
    for item in joined.itertuples(index=False):
        common = bool(item.assigned_learner and item.assigned_reference)
        rows.append({**{key: getattr(item, key) for key in keys}, "learner_assigned": bool(item.assigned_learner), "reference_assigned": bool(item.assigned_reference), "common_auto": common,
                     **{"delta_" + field: float(getattr(item, field + "_learner")) - float(getattr(item, field + "_reference")) if common else None for field in (*reference.METRICS, *reference.ACCOUNTS, "excess", "breach")}})
    paired = pd.DataFrame(rows); summaries = []
    for (scope, client), group in paired.groupby(["scope", "client"], sort=False):
        selected = group.loc[group.common_auto]
        summaries.append({"scope": scope, "client": client, "total_original_pairs": len(group), "learner_auto_n": int(group.learner_assigned.sum()),
                          "reference_auto_n": int(group.reference_assigned.sum()), "common_auto_n": len(selected),
                          **{"mean_" + field: float(selected[field].mean()) if len(selected) else None for field in paired if field.startswith("delta_")}})
    return paired, pd.DataFrame(summaries)


def main():
    # Import only after root has stabilized/tested/reviewed the new kernel and
    # explicitly authorized the market run. This runner is not a scheduler.
    from experimental_execution.risk_priced_fan_rollout import simulate_prices
    from experimental_execution.fan_rollout import simulate_policy
    started = time.perf_counter(); protocol = json.loads(PROTOCOL.read_text())
    sources, inputs = source_manifest(), frozen_inputs(protocol)
    protocol_key = str(PROTOCOL.relative_to(ROOT))
    if protocol.get("execution_sources_sha256") != {name: value for name, value in sources.items() if name != protocol_key}:
        raise RuntimeError("Execution source/test SHA must be frozen in the protocol before market replay")
    if (tuple(protocol["candidate_family"]["annual_prices"]) != FINITE_PRICES
            or protocol["candidate_family"]["additional_candidate"] != "risk_first"):
        raise RuntimeError("Prospectively declared nine-price family changed")
    fits = json.loads((STRICT / "nested_context_refits.json").read_text())
    mapping = json.loads((STRICT / "candidate_mapping.json").read_text())
    strict_protocol = json.loads((ROOT / "docs/positive_asset_specification_protocol.json").read_text())
    if len(mapping) != 172: raise RuntimeError("Require frozen strict172 candidate map")
    original = pd.concat([pd.read_csv(STRICT / "nested_full_paths.csv"), pd.read_csv(STRICT / "nested_outer_oof_paths.csv")], ignore_index=True)
    pending_saved = pd.read_csv(PENDING / "client_path_metrics.csv")
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    cache_meta = json.loads(CACHE.with_suffix(".json").read_text())
    if (digest.hexdigest() != cache_meta["development_arrays_sha256"] or cache_meta["npz_sha256"] != sha(CACHE)
            or cache_meta["scenario_ids"] != list(range(1, 121))): raise RuntimeError("Guarded data/cache identity differs from frozen development cache")
    if json.loads((PENDING / "independent_audit.json").read_text())["status"] != "passed": raise RuntimeError("Pending reference must have completed its independent audit")
    with np.load(CACHE) as file: cache = {key: file[key] for key in file.files}
    candidate_paths, candidate_summaries, main_rows, gates = [], [], [], []
    selections, checks, archives = {}, {}, {}
    for context_info in strict_protocol["contexts"]:
        context = context_info["name"]; fitted = fits[context]
        if fitted["training_ids"] != context_info["training_ids"] or fitted["outer_test_ids"] != context_info["outer_test_ids"]:
            raise RuntimeError("Frozen original training/outer partition changed")
        training_ids = fitted["training_ids"]; full = context_info["context_id"] == 0
        evaluation_ids = training_ids if full else fitted["outer_test_ids"]
        scope = "full_development" if full else "outer_oof"
        train_ix, eval_ix = np.asarray(training_ids) - 1, np.asarray(evaluation_ids) - 1
        selections[context] = {"training_ids": training_ids, "evaluation_ids": evaluation_ids, "outer_fold": context_info["context_id"] or None, "profiles": {}}
        for profile in strict_protocol["clients"]:
            client, H, L = profile["name"], profile["horizon"], profile["loss_tolerance"]
            risk_assignment = fitted["effective_assignments"][client]
            product = next((item for item in fitted["raw_fit"]["menu"] if item["name"] == risk_assignment), None) if fitted["raw_fit"] else None
            common = {"scope": scope, "context": context, "outer_fold": context_info["context_id"] or None, "client": client,
                      "horizon": H, "loss_tolerance": L, "risk_assignment": risk_assignment, "risk_assigned": product is not None,
                      "candidate_local_index": product["candidate_local_index"] if product else None,
                      "candidate_original232_index": product["candidate_original232_index"] if product else None,
                      "selection_training_n": len(training_ids)}
            training_common = {**common, "scope": "training_support", "context_evaluation_scope": scope}
            if product is None:
                reference.verify_review(original, context, client, evaluation_ids)
                selection = {"risk_assignment": risk_assignment, "risk_assigned": False, "n_training": len(training_ids),
                             "reference_training_breach_count": None, "training_breach_cap": None, "selected_candidate_order": None,
                             "execution_status": "risk_review", "execution_review_reason": "frozen_risk_review", "candidate_summaries": []}
                selections[context]["profiles"][client] = selection
                for order in range(9):
                    metadata = candidate_meta(order, H)
                    null_common = {**training_common, **metadata, "assignment": "Review", "training_breach_cap": None,
                                   "selected_execution_status": "risk_review", "execution_review_reason": "frozen_risk_review"}
                    candidate_paths.extend(null_rows(training_ids, null_common))
                    candidate_summaries.append({"context": context, "client": client, **metadata, "risk_assignment": risk_assignment,
                                               "n_training": len(training_ids), "breach_count": None, "mean_log_growth": None,
                                               "mean_nominal_fees": None, "mean_nominal_moved": None, "mean_trade_fraction": None,
                                               "training_breach_cap": None, "training_eligible": None, "selected": False,
                                               "simulation_status": "not_simulated_frozen_risk_review"})
                for policy in POLICIES:
                    main_rows.extend(null_rows(evaluation_ids, {**common, "policy": policy, "assignment": "Review",
                        "selected_execution_status": "risk_review", "execution_review_reason": "frozen_risk_review",
                        **dict.fromkeys(candidate_meta(0, H)), "reference_training_breach_count": None, "training_breach_cap": None}))
                continue
            target = reference.validate_product(product, mapping)
            train_r, train_i = returns[train_ix, :H], inflation[train_ix, :H]
            pair_train = reference.simulate_paired(train_r, train_i, target, L, horizon=H, cost_rate=.0025)
            annual_train = reference.squeeze_pending(pair_train["annual"]); baseline_train = reference.squeeze_pending(pair_train["pending_hold"])
            verify_training_annual(annual_train, training_ids, H, product["candidate_original232_index"], cache)
            reference_count = int(baseline_train["ever_breached"][:, -1].sum())
            family = simulate_prices(train_r, train_i, train_r, train_i, target, PRICES, L=L, cost_rate=.0025, horizon=H)
            if family["real_wealth"].shape != (len(training_ids), 9, H + 1): raise RuntimeError("Training simulator omitted candidate price or original horizon")
            price_arms, summaries = [], []
            for order in range(9):
                arm = extract_price(family, order); price_arms.append(arm)
                rows = financial_rows(arm, training_ids, {**training_common, **candidate_meta(order, H), "assignment": risk_assignment,
                                      "selected_execution_status": "training_candidate", "execution_review_reason": None})
                candidate_paths.extend(rows); summaries.append(training_summary(rows, order, H))
            old_growth = simulate_policy(train_r, train_i, train_r, train_i, target, policy="growth_only_rollout", L=L, cost_rate=.0025)
            zero_errors = verify_price_zero(price_arms[0], old_growth)
            selection = select_training_candidate(summaries, reference_count, len(training_ids))
            selection.update({"risk_assignment": risk_assignment, "risk_assigned": True, "target": target.tolist(),
                              "candidate_local_index": product["candidate_local_index"], "candidate_original232_index": product["candidate_original232_index"]})
            order = selection["selected_candidate_order"]
            selection["selected_candidate"] = candidate_meta(order, H) if order is not None else None
            selections[context]["profiles"][client] = selection
            for summary in selection["candidate_summaries"]:
                candidate_summaries.append({"context": context, "client": client, "risk_assignment": risk_assignment,
                                           **summary, "selected": summary["execution_candidate_order"] == order, "simulation_status": "complete_training_replay"})
            checks[f"{context}__{client}"] = {"training_reference_coupling": pair_train["coupling"], "training_eta0_old_growth_only_max_errors": zero_errors}
            if full:
                annual_eval, baseline_eval = annual_train, baseline_train
            else:
                pair_eval = reference.simulate_paired(returns[eval_ix, :H], inflation[eval_ix, :H], target, L, horizon=H, cost_rate=.0025)
                annual_eval, baseline_eval = reference.squeeze_pending(pair_eval["annual"]), reference.squeeze_pending(pair_eval["pending_hold"])
                checks[f"{context}__{client}"]["evaluation_reference_coupling"] = pair_eval["coupling"]
            checks[f"{context}__{client}"]["evaluation_annual_saved_errors"] = reference.verify_original(annual_eval, original, context, client, evaluation_ids, product, H)
            reference.verify_pending_reference(baseline_eval, pending_saved, context, client, evaluation_ids, H)
            baseline_common = {**common, "policy": "annual_then_pending", "assignment": risk_assignment,
                               "selected_execution_status": "reference", "execution_review_reason": None,
                               **dict.fromkeys(candidate_meta(0, H)), "reference_training_breach_count": reference_count, "training_breach_cap": selection["training_breach_cap"]}
            main_rows.extend(financial_rows(baseline_eval, evaluation_ids, baseline_common))
            prefix = f"{context}__{client}__annual_then_pending"
            archives[prefix + "__scenario_ids"] = np.asarray(evaluation_ids)
            for key, value in baseline_eval.items():
                if isinstance(value, np.ndarray): archives[prefix + "__" + key] = value
            learner_common = {**common, "policy": LEARNER, "assignment": risk_assignment if order is not None else "Review",
                              "selected_execution_status": selection["execution_status"], "execution_review_reason": selection["execution_review_reason"],
                              **(candidate_meta(order, H) if order is not None else dict.fromkeys(candidate_meta(0, H))),
                              "reference_training_breach_count": reference_count, "training_breach_cap": selection["training_breach_cap"]}
            if order is None:
                main_rows.extend(null_rows(evaluation_ids, learner_common))
            else:
                if full:
                    selected_eval = price_arms[order]
                else:
                    # The outer simulator receives exactly ONE training-selected
                    # price; all other held-out candidate returns remain unknown.
                    selected_eval = evaluate_selected(simulate_prices, returns[eval_ix, :H], inflation[eval_ix, :H], train_r, train_i,
                                                      target, order, L=L, horizon=H)
                main_rows.extend(financial_rows(selected_eval, evaluation_ids, learner_common))
                gates.extend(gate_records(selected_eval, evaluation_ids, learner_common, order))
                prefix = f"{context}__{client}__{LEARNER}"
                archives[prefix + "__scenario_ids"] = np.asarray(evaluation_ids)
                for key, value in selected_eval.items():
                    if isinstance(value, np.ndarray): archives[prefix + "__" + key] = value
            print(json.dumps({"context": context, "client": client, "training_n": len(training_ids), "risk_cap": selection["training_breach_cap"],
                              "selected": CANDIDATE_IDS[order] if order is not None else None, "status": selection["execution_status"],
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    frame = pd.DataFrame(main_rows)
    if len(frame) != 2400 or frame.duplicated(["scope", "context", "client", "scenario_id", "policy"]).any(): raise RuntimeError("Main original population changed")
    for (_, _, _), group in frame.groupby(["scope", "client", "policy"]):
        if len(group) != 120 or set(group.scenario_id) != set(range(1, 121)): raise RuntimeError("Missing original120 service denominator")
    if not frame.loc[~frame.assigned, list(FINANCIAL)].isna().all().all(): raise RuntimeError("Review has imputed financial outcomes")
    paired, paired_summary = paired_outputs(frame)
    summaries = pd.DataFrame(reference.summarize(frame))
    if source_manifest() != sources or frozen_inputs(protocol) != inputs: raise RuntimeError("Frozen source/input changed before output")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {"training_candidate_paths.csv": pd.DataFrame(candidate_paths), "training_candidate_summary.csv": pd.DataFrame(candidate_summaries),
              "client_paths.csv": frame, "summary.csv": summaries, "paired_paths.csv": paired, "paired_summary.csv": paired_summary,
              "decision_gates.csv": pd.DataFrame(gates), "coverage.csv": summaries[["scope", "client", "policy", "total_paths", "assigned_paths", "review_paths"]]}
    for name, value in frames.items(): value.to_csv(OUT / name, index=False)
    np.savez_compressed(OUT / "actual_traces.npz", **archives)
    write_json(OUT / "context_selections.json", selections); write_json(OUT / "accounting_checks.json", checks)
    source_end, input_end = source_manifest(), frozen_inputs(protocol)
    if source_end != sources or input_end != inputs: raise RuntimeError("Frozen source/input changed while writing output")
    names = [*frames, "actual_traces.npz", "context_selections.json", "accounting_checks.json"]
    write_json(OUT / "manifest.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "sources_at_end": source_end, "input_hashes": inputs, "input_hashes_at_end": input_end,
        "outputs_sha256": {name: sha(OUT / name) for name in names}, "scenario_ids_loaded": ids.tolist(), "final_scenarios_loaded": False,
        "development_array_sha256": digest.hexdigest(), "policies": list(POLICIES), "price_order": [candidate_meta(i, 1) for i in range(9)],
        "client_path_rows": len(frame), "training_candidate_path_rows": len(candidate_paths), "decision_rows": len(gates),
        "unselected_outer_prices_evaluated": False, "product_or_gamma_refit": False,
        "scope": "Training-selected finite-family execution learner on frozen strict172 products. All original rows and execution referrals remain; no held-out candidate search or final winner selection.",
        "limitations": ["Eta is a technical execution search parameter, not identified customer risk aversion or a dual/risk-budget estimate.",
                        "The integer empirical cap is a conservative development rule, not probability certification; the same-fan local score does not guarantee the repeatedly replanned policy.",
                        "Training contexts overlap. Conditional automatic metrics can change their served subgroup after execution referrals; common-auto paired differences and original coverage are reported separately.",
                        "Previously exposed final data remain excluded; reused development internal validation is not a fresh blind final evaluation. Liquidity and target updating remain unresolved."]})
    print(f"Wrote risk-priced execution learner to {OUT}", flush=True)


if __name__ == "__main__": main()
