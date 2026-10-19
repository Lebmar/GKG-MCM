"""Independent arithmetic, identity, accounting and scalar-fan audit.

Only the guarded development loader is imported from the project. Reviewed
adapter/selector/runner/simulator/metric modules are never imported. Previous
independent audit references provide a separate 100-step fee bisection and
pathwise accounting/fan implementation. This does not rerun menu optimization
or invent a replay of unrecorded nine-price training state trees.
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

OUT = ROOT / "results/model_review/nested_dynamic_calibration_v1"
PROTOCOL = ROOT / "docs/nested_dynamic_calibration_protocol.json"
CACHE = ROOT / "results/model_review/risk_menu_v1/cache/step_0500__cost_002500.npz"
BETAS = (0., .025, .05, .075, .1)
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction",
            "terminal_equity", "terminal_bond", "terminal_bill")
FINANCIAL = (*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing")
LEARNER = "nested_dynamic_learner"
H, LOSS, COST, TOL = 30, .4, .0025, 1e-12


def readj(path): return json.loads(Path(path).read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    m = readj(OUT / "manifest.json")
    paths = [OUT / name for name in m["outputs_sha256"]]
    paths += [OUT / "manifest.json", Path(__file__).resolve(),
              ROOT / "scripts/audit_fan_rollout_execution.py",
              ROOT / "scripts/audit_review_pending_execution.py"]
    paths += [ROOT / name for name in m["sources"]]
    paths += [ROOT / name for name in m["input_sha256"]]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def case_of(assignment):
    return {"Balanced": "balanced_case", "Review": "risk_review",
            "Conservative": "not_balanced_case", "Growth": "not_balanced_case"}[assignment]


def choose_price(stats, k_b, n, beta):
    cap = min(k_b, math.floor(beta * n + TOL))
    eligible = [(i, g) for i, (k, g) in enumerate(stats) if k <= cap]
    maximum = max((g for _, g in eligible), default=None)
    order = next((i for i, g in eligible if g >= maximum - TOL), None)
    return order, cap, maximum


def choose_budget(stats):
    eligible = [x for x in stats if x["eligible"]]
    if not eligible: return None, None, None
    coverage = max(x["case_auto_n"] for x in eligible)
    covered = [x for x in eligible if x["case_auto_n"] == coverage]
    growth = max(x["conditional_mean_log_growth"] for x in covered)
    return min(x["training_beta"] for x in covered
               if x["conditional_mean_log_growth"] >= growth - TOL), coverage, growth


def gate(qh, qr, kh, kr, n, order):
    if order == 8:
        return ("restore" if kr < kh or kr == kh and qr - qh > TOL else "hold"), None
    score = qr - qh - H * PRICES[order] * ((kr - kh) / n)
    return ("restore" if score > TOL else "hold"), score


def run_audit():
    before = snapshot()
    manifest, protocol = readj(OUT / "manifest.json"), readj(PROTOCOL)
    legacy = readj(ROOT / "docs/positive_asset_specification_protocol.json")
    old_refits = readj(ROOT / "results/model_review/positive_asset_specification_v1/nested_context_refits.json")
    bundles = readj(OUT / "static_refits.json")
    choices = readj(OUT / "context_selections.json")
    groups = readj(OUT / "trace_groups.json")
    tables = {name: pd.read_csv(OUT / (name + ".csv")) for name in
              ("static_inner_paths", "training_candidate_paths", "training_candidate_summary",
               "inner_validation_paths", "beta_summaries", "client_paths", "summary", "paired_paths", "paired_summary")}
    jobs = protocol["jobs"]
    checks, errors, q_results, selections = 0, {}, [], []
    replay = {"actual_arm_paths": 0, "validation_arm_paths": 0, "actual_decisions": 0,
              "validation_decisions": 0, "active_finite_gates": 0, "active_lex_gates": 0}

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition: raise AssertionError(label)

    def close(a, b, label, *, atol=3e-11, rtol=3e-13):
        a, b = np.asarray(a), np.asarray(b)
        check(a.shape == b.shape, label + "/shape")
        if a.dtype.kind in "bUiS" or b.dtype.kind in "bUiS":
            check(np.array_equal(a, b), label + "/exact")
        else:
            with np.errstate(invalid="ignore"):
                difference = np.abs(a - b)
            finite = np.isfinite(difference)
            key = label.split("/")[-1]
            errors[key] = max(errors.get(key, 0.), float(difference[finite].max(initial=0.)))
            check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + "/numeric")

    def maybe(a, b, label):
        if b is None: check(a is None or pd.isna(a), label + "/NULL")
        else: close(a, b, label)

    def ids_of(frame, expected, label):
        check(len(frame) == len(expected) and not frame.scenario_id.duplicated().any()
              and set(frame.scenario_id) == set(expected), label + "/original_IDs")

    def price_metadata(frame, order):
        check(frame.execution_candidate_order.eq(order).all(), "price/candidate_order")
        check(frame.execution_candidate_id.eq(f"eta_{PRICES[order]:g}" if order < 8 else "risk_first").all(), "price/candidate_ID")
        check(frame.price_kind.eq("finite" if order < 8 else "risk_first").all(), "price/kind")
        if order < 8:
            close(frame.annual_price, np.full(len(frame), PRICES[order]), "price/annual_eta")
            close(frame.original_H_total_price, np.full(len(frame), H * PRICES[order]), "price/original_H_scale")
        else:
            check(frame.annual_price.isna().all() and frame.original_H_total_price.isna().all(), "price/lex_not_infinite_numeric_score")

    def mapping(value):
        if isinstance(value, list):
            for item in value: mapping(item)
        elif isinstance(value, dict):
            if "candidate_index" in value:
                i = value["candidate_index"]
                check(i is None or isinstance(i, int) and 0 <= i < 172, "static/local172_index")
                check(value["candidate_local_index"] == i and value["candidate_original232_index"] ==
                      (None if i is None else int(allowed[i])), "static/local_global_mapping")
            for key in ("identified_indices", "indices", "search_pool_indices", "extension_indices", "reference_frontier"):
                if key in value:
                    local = value[key]
                    check(all(isinstance(i, int) and 0 <= i < 172 for i in local)
                          and value[key + "_original232"] == [int(allowed[i]) for i in local],
                          "static/local_global_index_list")
            for item in value.values(): mapping(item)

    def static_fit(fit, train_ids, gamma):
        """Audit recorded feasible-product/assignment arithmetic, not menu search."""
        mapping(fit)
        check(fit["n_training"] == len(train_ids) and fit["training_screen_gamma"] == gamma, "static/fit_scope")
        tx = np.asarray(train_ids) - 1
        for item in fit["menu"]:
            close(item["weights"], weights[item["candidate_original232_index"]], "static/weights", atol=0, rtol=0)
            check(min(item["weights"]) >= .05 - TOL, "static/positive_three_assets")
        for profile in legacy["clients"]:
            name, h, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
            candidates = []
            for detail in (x for x in fit["client_product_details"] if x["client"] == name):
                col = detail["candidate_original232_index"]
                dd = cache[f"H{h}__mdd"][tx, col]
                g = float(np.mean(cache[f"H{h}__log_growth"][tx, col]))
                breached = dd - loss > TOL
                k = int(sum(breached)); p = k / len(tx)
                excess = float(np.mean(np.where(breached, dd - loss, 0)))
                cvar = independent_cvar(dd)
                check(detail["breach_count"] == k and detail["n"] == len(tx), "static/whole_H_count")
                for field, v in (("mean_log_growth", g), ("breach_probability", p),
                                 ("expected_excess", excess), ("cvar90_mdd", cvar)):
                    close(detail[field], v, "static/detail_" + field)
                feasible = p <= gamma + TOL
                check(detail["training_screen_feasible"] == feasible, "static/empirical_feasibility")
                if feasible: candidates.append((detail, g, excess, cvar))
            if candidates:
                high = max(x[1] for x in candidates)
                selected = min((x for x in candidates if x[1] >= high - TOL),
                               key=lambda x: (x[2], x[3], x[0]["candidate_index"]))[0]
                expected = selected["portfolio"]
            else: expected = "Review"
            check(fit["assignments"][name] == expected, "static/independent_raw_assignment")

    def static_bundle(job):
        jid, train_ids = job["job_id"], job["training_ids"]
        b = bundles[jid]; sc = protocol["static_contexts"][jid]
        check(b["context"] == sc and sc["training_ids"] == train_ids, "static/saved_context")
        train = set(train_ids); seen = []
        chunks = np.array_split(np.random.default_rng(sc["inner_seed"]).permutation(sorted(train)), 4)
        for f, chunk in zip(sc["inner_folds"], chunks, strict=True):
            check(f["test_ids"] == sorted(map(int, chunk)) and set(f["train_ids"]) == train - set(chunk), "static/seeded_subfold")
            check(not set(f["train_ids"]) & set(f["test_ids"]), "static/subfold_disjoint")
            seen.extend(f["test_ids"])
        check(sorted(seen) == sorted(train_ids), "static/validation_covers_T_once")
        data = tables["static_inner_paths"].loc[lambda x: x.job_id.eq(jid)]
        check(len(data) == 25 * len(train_ids), "static/all_gamma_profile_rows")
        score = []
        for gamma in BETAS:
            all_auto, all_g, any_failure, profiles = 0, [], False, {}
            saved_summary = next(x for x in b["inner_summaries"] if x["training_screen_gamma"] == gamma)
            gamma_rows = data.loc[data.training_screen_gamma.eq(gamma)]
            for fold in sc["inner_folds"]:
                fit = b["inner_fold_fits"][str(gamma)][fold["inner_fold"] - 1]
                check(fit["train_ids"] == fold["train_ids"] and fit["test_ids"] == fold["test_ids"], "static/fit_IDs")
                static_fit(fit, fold["train_ids"], gamma)
                for profile in legacy["clients"]:
                    rows = gamma_rows.loc[gamma_rows.inner_fold.eq(fold["inner_fold"]) & gamma_rows.client.eq(profile["name"])]
                    ids_of(rows, fold["test_ids"], "static/validation")
                    assignment = fit["assignments"][profile["name"]]
                    check(rows.assignment.eq(assignment).all() and rows.assigned.eq(assignment != "Review").all(), "static/validation_assignment")
                    if assignment == "Review":
                        check(rows[[*METRICS, "breach", "excess", "candidate_index"]].isna().all().all(), "static/Review_NULL")
                    else:
                        item = next(x for x in fit["menu"] if x["name"] == assignment)
                        check(rows.candidate_index.eq(item["candidate_index"]).all(), "static/validation_target")
                        for field in METRICS:
                            close(rows[field], cache[f'H{profile["horizon"]}__{field}'][rows.scenario_id.to_numpy(int) - 1, item["candidate_original232_index"]], "static/validation_" + field)
                        dd = rows.mdd.to_numpy() - profile["loss_tolerance"]
                        close(rows.breach.to_numpy(bool), dd > TOL, "static/validation_event")
                        close(rows.excess, np.where(dd > TOL, dd, 0), "static/validation_excess")
            for profile in legacy["clients"]:
                rows = gamma_rows.loc[gamma_rows.client.eq(profile["name"])]
                ids_of(rows, train_ids, "static/pooled_validation")
                auto = rows.loc[rows.assigned]
                n, k = len(auto), int(auto.breach.astype(bool).sum()) if len(auto) else None
                p = k / n if n else None
                passed = p <= .1 + TOL if n else None
                profiles[profile["name"]] = n
                saved = saved_summary["profiles"][profile["name"]]
                check(saved["total_paths"] == len(train_ids) and saved["auto_n"] == n and saved["review_n"] == len(train_ids) - n, "static/pooled_profile_coverage")
                check(saved["breach_count"] == k and saved["empirical_service_risk_pass"] == passed, "static/pooled_profile_risk")
                maybe(saved["conditional_breach_probability"], p, "static/profile_probability")
                maybe(saved["conditional_mean_log_growth"], float(auto.log_growth.mean()) if n else None, "static/profile_growth")
                any_failure |= passed is False
                all_auto += n; all_g.extend(auto.log_growth.tolist())
            eligible = all_auto > 0 and not any_failure
            check(saved_summary["eligible"] == eligible and saved_summary["raw_auto_path_profile_units"] == all_auto, "static/gamma_eligibility_coverage")
            g = float(np.mean(all_g)) if all_g else None
            maybe(saved_summary["raw_auto_unit_mean_log_growth"], g, "static/gamma_growth")
            score.append((eligible, all_auto, g, gamma, profiles))
        good = [x for x in score if x[0]]
        if good:
            coverage = max(x[1] for x in good); maximum = max(x[2] for x in good if x[1] == coverage)
            gamma = min(x[3] for x in good if x[1] == coverage and x[2] >= maximum - TOL)
        else: gamma = None
        refit = b["refit"]
        check(refit["selection"]["selected_gamma"] == gamma, "static/independent_gamma_selection")
        if gamma is None:
            check(refit["raw_fit"] is None, "static/unsupported_no_fit")
        else: static_fit(refit["raw_fit"], train_ids, gamma)
        evidence = next((x[4] for x in score if x[3] == gamma), {})
        for profile in legacy["clients"]:
            name = profile["name"]
            raw = refit["raw_fit"]["assignments"][name] if gamma is not None else "Review"
            effective = raw if raw == "Review" or evidence.get(name, 0) else "Review"
            check(refit["raw_assignments"][name] == raw and refit["effective_assignments"][name] == effective, "static/admission_guard")
        assignment = refit["effective_assignments"]["C"]
        check(b["q2_case"]["status"] == case_of(assignment) and b["q2_case"]["effective_assignment"] == assignment, "case/after_effective_Q1")
        if job["stage"] == "refit": check(refit == old_refits[job["parent_context"]], "static/legacy_literal_reproduction")
        return b

    check(protocol["execution_ready"] is True and tuple(protocol["calibration"]["dynamic_betas"]) == BETAS, "protocol/frozen_ready")
    check(len(jobs) == 30 and set(bundles) == {x["job_id"] for x in jobs}, "jobs/all30")
    for parent in legacy["contexts"]:
        cid = parent["context_id"]
        grouped = [j for j in jobs if j["parent_context_id"] == cid]
        check([j["job_id"] for j in grouped] == [*[f"c{cid}_inner{i}" for i in range(1, 5)], f"c{cid}_refit"], "jobs/four_inner_before_refit")
        for job, fold in zip(grouped[:4], parent["inner_folds"], strict=True):
            check(job["stage"] == "inner" and job["training_ids"] == fold["train_ids"]
                  and job["evaluation_ids"] == fold["test_ids"]
                  and job["parent_context"] == parent["name"]
                  and job["static_inner_seed"] == 8702026101800 + 100 * cid + fold["inner_fold"], "jobs/exact_original_dynamic_inner_partition")
            check(set(job["training_ids"]) | set(job["evaluation_ids"]) == set(parent["training_ids"])
                  and not set(job["training_ids"]) & set(parent["outer_test_ids"]), "jobs/no_outer_path_in_inner_pipeline")
        final = grouped[-1]
        check(final["stage"] == "refit" and final["training_ids"] == parent["training_ids"]
              and final["evaluation_ids"] == (parent["training_ids"] if cid == 0 else parent["outer_test_ids"])
              and final["static_inner_seed"] == parent["inner_seed"], "jobs/exact_original_final_partition")
    for name, expected in protocol["frozen_input_sha256"].items(): check(sha(ROOT / name) == expected, "input/" + name)
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / protocol["data"]["path"], range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for a in (ids, returns, inflation): digest.update(np.ascontiguousarray(a).tobytes())
    check(ids.tolist() == list(range(1, 121)) and digest.hexdigest() == manifest["development_arrays_sha256"], "data/guarded120")
    with np.load(CACHE, allow_pickle=False) as f: cache = {k: f[k] for k in f.files}
    weights = cache["weights"]; allowed = np.flatnonzero(np.all(weights >= .05 - TOL, axis=1))
    check(len(allowed) == 172 and len(weights) == 232, "data/strict172_mapping")
    expected_price = {}; expected_orders = {"actual": {}, "validation": {}}
    for job in jobs:
        jid, train_ids, eval_ids = job["job_id"], job["training_ids"], job["evaluation_ids"]
        check(set(train_ids) <= set(range(1, 121)) and set(eval_ids) <= set(range(1, 121)), "IDs/no_final")
        check(job["stage"] == "refit" and job["parent_context_id"] == 0 or not set(train_ids) & set(eval_ids), "IDs/fit_eval_disjoint")
        b = static_bundle(job); case = b["q2_case"]["status"]
        family = tables["training_candidate_paths"].loc[lambda x: x.job_id.eq(jid)]
        saved_stats = tables["training_candidate_summary"].loc[lambda x: x.job_id.eq(jid)].set_index("execution_candidate_order")
        check(len(family) == 9 * len(train_ids) and set(saved_stats.index) == set(range(9)), "training/all9")
        check(family.scope.eq("training_support").all(), "training/not_OOF")
        stats = []
        if case == "balanced_case":
            item = b["q2_case"]["product"]
            k_b = int((cache["H30__mdd"][np.asarray(train_ids) - 1, item["candidate_original232_index"]] - LOSS > TOL).sum())
            check(b["reference_training_breach_count"] == k_b, "training/same_target_B_count")
        else: k_b = None
        for order in range(9):
            data = family.loc[family.execution_candidate_order.eq(order)]
            ids_of(data, train_ids, "training/candidate")
            price_metadata(data, order)
            summary = saved_stats.loc[order]
            check(data.q1_effective_assignment.eq(b["q2_case"]["effective_assignment"]).all(), "training/Q1_assignment")
            if case != "balanced_case":
                check(not data.assigned.any() and data[list(FINANCIAL)].isna().all().all(), "training/noncase_NULL")
                check(pd.isna(summary.breach_count) and pd.isna(summary.mean_log_growth), "training/noncase_summary_NULL")
                continue
            check(data.assigned.all() and data.candidate_original232_index.eq(item["candidate_original232_index"]).all(), "training/frozen_target")
            close(data.cagr, np.expm1(data.log_growth), "training/cagr_algebra", atol=3e-12)
            close(data.terminal_real, 100 * np.exp(H * data.log_growth), "training/terminal_algebra")
            close(data.nominal_fees, COST * data.nominal_moved, "training/fee_algebra")
            close(data.breach.to_numpy(bool), data.mdd.to_numpy() - LOSS > TOL, "training/everbreach")
            close(data.excess, np.maximum(data.mdd - LOSS, 0), "training/excess")
            check(data.loc[~data.breach.astype(bool), "first_crossing"].isna().all()
                  and data.loc[data.breach.astype(bool), "first_crossing"].between(1, H).all(), "training/crossing_or_NULL")
            k, g = int(data.breach.astype(bool).sum()), float(data.log_growth.mean())
            for field, val in (("breach_count", k), ("mean_log_growth", g), ("mean_nominal_fees", data.nominal_fees.mean()),
                               ("mean_nominal_moved", data.nominal_moved.mean()), ("mean_trade_fraction", data.trade_fraction.mean())):
                close(summary[field], val, "training/summary_" + field)
            stats.append((k, g))
        for beta in BETAS:
            saved = b["training_price_selections"][str(beta)]
            if case == "balanced_case":
                order, cap, maximum = choose_price(stats, k_b, len(train_ids), beta)
                check(saved["training_breach_cap"] == cap and saved["selected_candidate_order"] == order, "training/independent_price_selection")
                maybe(saved["maximum_eligible_mean_log_growth"], maximum, "training/max_growth")
                for i, detail in enumerate(saved["candidate_summaries"]):
                    check(detail["training_eligible"] == (stats[i][0] <= cap), "training/all9_eligibility")
            else: order = None
            expected_price[jid, beta] = order
            if job["stage"] == "inner":
                rows = tables["inner_validation_paths"].loc[lambda x: x.job_id.eq(jid) & x.training_beta.eq(beta)]
                ids_of(rows, eval_ids, "dynamic/inner_validation")
                status = case if case != "balanced_case" else "automated" if order is not None else "execution_review"
                check(rows.execution_status.eq(status).all() and rows.assigned.eq(status == "automated").all(), "dynamic/inner_admission")
                check(rows.q1_effective_assignment.eq(b["q2_case"]["effective_assignment"]).all()
                      and rows.case_status.eq(case).all(), "dynamic/inner_Q1_case")
                if order is None:
                    check(rows[list(FINANCIAL)].isna().all().all(), "dynamic/inner_NULL")
                else:
                    check(rows.selected_candidate_order.eq(order).all(), "dynamic/no_validation_price_search")
                    expected_orders["validation"][jid + f"__p{order}"] = (job, order)
        selections.append({"job_id": jid, "static_gamma": b["refit"]["selection"]["selected_gamma"],
                           "case_status": case, "reference_training_count": k_b,
                           "prices_by_beta": {str(beta): expected_price[jid, beta] for beta in BETAS}})

    def counts(rows):
        result = {"original_n": len(rows), "balanced_case_n": int(rows.case_status.eq("balanced_case").sum()),
                  "risk_review_n": int(rows.case_status.eq("risk_review").sum()),
                  "not_balanced_case_n": int(rows.case_status.eq("not_balanced_case").sum()),
                  "case_auto_n": int(rows.assigned.sum()),
                  "case_execReview_n": int(rows.execution_status.eq("execution_review").sum())}
        check(result["original_n"] == result["balanced_case_n"] + result["risk_review_n"] + result["not_balanced_case_n"], "counts/original_case_identity")
        check(result["balanced_case_n"] == result["case_auto_n"] + result["case_execReview_n"], "counts/case_admission_identity")
        return result

    main_rows = tables["client_paths"]
    check(len(main_rows) == 720 and not main_rows.duplicated(["context", "policy", "scenario_id"]).any(), "main/720_original_rows")
    check(main_rows.loc[~main_rows.assigned, list(FINANCIAL)].isna().all().all(), "main/nonauto_NULL")
    for context in legacy["contexts"]:
        name, cid = context["name"], context["context_id"]
        stats = []
        for beta in BETAS:
            data = tables["inner_validation_paths"].loc[lambda x: x.context.eq(name) & x.training_beta.eq(beta)]
            ids_of(data, context["training_ids"], "dynamic/pooled_inner_once")
            c = counts(data); auto = data.loc[data.assigned]
            k = int(auto.breach.astype(bool).sum()) if len(auto) else None
            g = float(auto.log_growth.mean()) if len(auto) else None
            eligible = len(auto) > 0 and k <= math.floor(.1 * len(auto) + TOL)
            row = tables["beta_summaries"].loc[lambda x: x.context.eq(name) & x.training_beta.eq(beta)].iloc[0]
            for field, value in c.items(): check(row[field] == value, "dynamic/beta_" + field)
            check(row.eligible == eligible and row.auto_inner_folds == auto.dynamic_inner_fold.nunique(), "dynamic/beta_eligibility_folds")
            maybe(row.breach_count, k, "dynamic/beta_count")
            maybe(row.conditional_mean_log_growth, g, "dynamic/beta_growth")
            maybe(row.conditional_breach_probability, k / len(auto) if len(auto) else None, "dynamic/beta_probability")
            stats.append({"training_beta": beta, **c, "breach_count": k, "conditional_mean_log_growth": g, "eligible": eligible})
        beta, coverage, growth = choose_budget(stats)
        selected = choices[name]; job = next(x for x in jobs if x["parent_context_id"] == cid and x["stage"] == "refit")
        jid = job["job_id"]; case = bundles[jid]["q2_case"]["status"]
        order = expected_price[jid, beta] if beta is not None else None
        status = case if case != "balanced_case" else "execution_review" if beta is None or order is None else "automated"
        reason = case if case != "balanced_case" else "no_eligible_inner_beta" if beta is None else "no_admissible_training_price" if order is None else None
        check(selected["beta_selection"]["selected_beta"] == beta and selected["selected_candidate_order"] == order, "dynamic/independent_beta_refit_choice")
        maybe(selected["beta_selection"]["maximum_auto_coverage"], coverage, "dynamic/max_coverage")
        maybe(selected["beta_selection"]["maximum_mean_log_growth_at_coverage"], growth, "dynamic/max_growth_at_coverage")
        check(selected["execution_status"] == status and selected["execution_reason"] == reason, "dynamic/refit_reason_priority")
        eval_ids = job["evaluation_ids"]
        for policy in (LEARNER, "annual_then_pending", "equal_thirds_annual"):
            data = main_rows.loc[main_rows.context.eq(name) & main_rows.policy.eq(policy)]
            ids_of(data, eval_ids, "main/context_policy")
            check(data.q1_effective_assignment.eq(bundles[jid]["q2_case"]["effective_assignment"]).all()
                  and data.case_status.eq(case).all(), "main/Q1_case_retained")
            if policy == LEARNER:
                check(data.execution_status.eq(status).all() and data.assigned.eq(status == "automated").all(), "main/service_status")
                if status == "automated": expected_orders["actual"][f"c{cid}__{LEARNER}"] = (job, order)
            elif policy == "annual_then_pending":
                check(data.reference_only.all() and data.hypothetical_reference.all() and data.assigned.eq(case == "balanced_case").all(), "main/B_hypothetical_availability")
                if case == "balanced_case": expected_orders["actual"][f"c{cid}__annual_then_pending"] = (job, None)
            else:
                check(data.reference_only.all() and data.hypothetical_reference.all() and data.assigned.all(), "main/EQ_hypothetical_availability")
                eq = int(np.flatnonzero(np.all(weights == np.ones(3) / 3, axis=1))[0])
                check(data.candidate_original232_index.eq(eq).all(), "main/EQ_actual_target_index")
                expected_orders["actual"][f"c{cid}__equal_thirds_annual"] = (job, None)

    unavailable_q_jobs = []
    for job in jobs:
        scope = "validation" if job["stage"] == "inner" else "actual"
        available = any(value[0]["job_id"] == job["job_id"] and value[1] is not None
                        for value in expected_orders[scope].values())
        if not available:
            unavailable_q_jobs.append({"job_id": job["job_id"], "scope": scope,
                                       "scenario_id": min(job["evaluation_ids"]), "years": [2, 16, 30],
                                       "reason": "no_automated_selected_price_no_replacement"})

    # No selected trajectory may be absent or invented. All branch fans remain T.
    for scope in ("actual", "validation"):
        check(set(groups[scope]) == set(expected_orders[scope]), scope + "/exact_selected_groups")
        with np.load(OUT / ("actual_traces.npz" if scope == "actual" else "validation_traces.npz"), allow_pickle=False) as archive:
            check(all(any(k.startswith(prefix + "__") for prefix in groups[scope]) for k in archive.files), scope + "/no_orphan_trace_arrays")
            for key, (job, order) in expected_orders[scope].items():
                group = groups[scope][key]; ids_ = job["evaluation_ids"]
                check(group["scenario_ids"] == ids_ and group["q_subset_scenario_id"] == min(ids_) and group["q_subset_years"] == [2, 16, 30], scope + "/predeclared_Q_identity")
                arm = {k[len(key) + 2:]: archive[k] for k in archive.files if k.startswith(key + "__")}
                jid = job["job_id"]; b = bundles[jid]
                policy = group["metadata"].get("policy", LEARNER)
                target = np.ones(3) / 3 if policy == "equal_thirds_annual" else np.asarray(b["q2_case"]["product"]["weights"])
                # The frozen kernel normalizes input weights by their sum.
                # A grid tuple summing to 0.9999999999999999 can move one ULP;
                # this is independently reconstructed, not a new weight rule.
                normalized_target = target / float(sum(target))
                close(arm["targets"], normalized_target, scope + "/saved_target", atol=0, rtol=0)
                if order is not None: close(arm["annual_prices"], [PRICES[order] if order < 8 else np.inf], scope + "/single_executed_price", atol=0, rtol=0)
                for i, sid in enumerate(ids_):
                    def action(t, x, cp, peak, worst, bad):
                        if order is not None and (t == 0 or bad):
                            check(np.isnan(arm["qg_hold"][i, t]) and np.isnan(arm["qg_restore"][i, t])
                                  and np.isnan(arm["predicted_score"][i, t])
                                  and arm["qbad_hold_count"][i, t] == -1
                                  and arm["qbad_restore_count"][i, t] == -1
                                  and arm["forecast_n"][i, t] == -1, "actual/inactive_Q_NULL_sentinels")
                        if t == 0: expected = "restore"
                        elif policy == "equal_thirds_annual": expected = "restore"
                        elif bad: expected = "hold"
                        elif order is None: expected = "restore"
                        else:
                            kh, kr, n = int(arm["qbad_hold_count"][i, t]), int(arm["qbad_restore_count"][i, t]), int(arm["forecast_n"][i, t])
                            check(n == len(job["training_ids"]) and 0 <= kh <= n and 0 <= kr <= n, "actual/gate_integer_T_denominator")
                            expected, score = gate(float(arm["qg_hold"][i, t]), float(arm["qg_restore"][i, t]), kh, kr, n, order)
                            replay["active_lex_gates" if order == 8 else "active_finite_gates"] += 1
                            if score is None: check(np.isnan(arm["predicted_score"][i, t]), "actual/lex_NULL_score")
                            else: close(arm["predicted_score"][i, t], score, "actual/original_H_score", atol=3e-12, rtol=0)
                        check(arm["actions"][i, t] == expected, "actual/action_rule")
                        return expected
                    ref = reference_actual(returns[sid - 1], inflation[sid - 1], target, LOSS, action, cost_rate=COST)
                    if policy == "equal_thirds_annual": ref["pending_at_decision"] = np.zeros(H, bool)
                    for field, value in ref.items():
                        if field in arm: close(arm[field][i], value, scope + "/state_" + field)
                    metrics = independent_metrics(ref, LOSS)
                    metrics.update({"trade_fraction": float(np.mean(ref["trades"])),
                                    "target_execution_fraction": float(np.mean(ref["target_execution"]))})
                    if scope == "actual":
                        row_groups = [main_rows.loc[main_rows.context.eq(job["parent_context"]) & main_rows.policy.eq(policy) & main_rows.scenario_id.eq(sid)]]
                    else:
                        row_groups = [tables["inner_validation_paths"].loc[lambda x: x.job_id.eq(jid) & x.selected_candidate_order.eq(order) & x.scenario_id.eq(sid)]]
                    for data in row_groups:
                        check(len(data) > 0, "actual/saved_metric_rows")
                        for field in (*METRICS, *ACCOUNTS, "breach", "excess"):
                            for value in data[field]: close(value, metrics[field], scope + "/metric_" + field)
                        expected_tau = ref["first_crossing"] if ref["first_crossing"] >= 0 else None
                        for value in data.first_crossing: maybe(value, expected_tau, scope + "/first_crossing")
                    replay[scope + "_arm_paths"] += 1; replay[scope + "_decisions"] += H
                if order is not None:
                    i = ids_.index(min(ids_)); sid = ids_[i]
                    for year in (2, 16, 30):
                        t = year - 1
                        entry = {"scope": scope, "group": key, "scenario_id": sid, "year": year, "training_n": len(job["training_ids"])}
                        if arm["pending_at_decision"][i, t]:
                            q_results.append({**entry, "status": "pending_skip_no_replacement"}); continue
                        tx = np.asarray(job["training_ids"]) - 1
                        q = scalar_fan(arm["holdings"][i, t], arm["cpi"][i, t], arm["real_peak"][i, t], arm["running_mdd"][i, t],
                                       target, LOSS, returns[tx], inflation[tx], t, cost_rate=COST)
                        for a in ("hold", "restore"):
                            close(arm["qg_" + a][i, t], q[a]["mean"], "Q/" + a + "_log", atol=3e-12, rtol=0)
                            check(arm["qbad_" + a + "_count"][i, t] == q[a]["count"], "Q/" + a + "_count")
                        expected, score = gate(q["hold"]["mean"], q["restore"]["mean"], q["hold"]["count"], q["restore"]["count"], len(tx), order)
                        check(expected == arm["actions"][i, t], "Q/independent_action")
                        q_results.append({**entry, "status": "audited", "hold_bad_count": q["hold"]["count"],
                                          "restore_bad_count": q["restore"]["count"], "independent_action": expected})

    # Summaries and pairwise outcomes preserve original denominators and NULLs.
    for (scope, policy), data in main_rows.groupby(["scope", "policy"]):
        ids_of(data, range(1, 121), "summary/complete_original120")
        row = tables["summary"].loc[lambda x: x.scope.eq(scope) & x.policy.eq(policy)].iloc[0]
        auto = data.loc[data.assigned]
        if policy == LEARNER:
            for field, value in counts(data).items(): check(row[field] == value, "summary/" + field)
            check(pd.isna(row.reference_available_n) and pd.isna(row.case_reference_available_n), "summary/learner_no_reference_admission")
        else:
            check(pd.isna(row.case_auto_n) and pd.isna(row.case_execReview_n), "summary/reference_not_service")
            check(row.reference_available_n == len(auto) and row.case_reference_available_n == int((data.assigned & data.case_status.eq("balanced_case")).sum()), "summary/reference_availability")
        check(row.original_n == 120 and row.assigned_paths == len(auto) and row.financial_null_paths == 120 - len(auto), "summary/financial_denominators")
        if not len(auto):
            for field in ("breach_count", "conditional_breach_probability", "mean_log_growth", "median_cagr", "p90_mdd"):
                check(pd.isna(row[field]), "summary/zero_auto_NULL")
            continue
        values = {"breach_count": int(auto.breach.astype(bool).sum()), "conditional_breach_probability": float(auto.breach.astype(bool).mean()),
                  "breaches_per_original_n": float(auto.breach.astype(bool).sum() / 120), "mean_log_growth": auto.log_growth.mean(),
                  "median_cagr": auto.cagr.median(), "p10_cagr": np.quantile(auto.cagr, .1, method="inverted_cdf"),
                  "median_mdd": auto.mdd.median(), "p90_mdd": np.quantile(auto.mdd, .9, method="inverted_cdf"),
                  "cvar90_mdd": independent_cvar(auto.mdd), "mean_nominal_fees": auto.nominal_fees.mean(),
                  "mean_trade_fraction": auto.trade_fraction.mean(), "mean_excess": auto.excess.mean()}
        for field, value in values.items(): close(row[field], value, "summary/" + field)
    check(len(tables["paired_paths"]) == 480, "paired/all_original_units")
    for row in tables["paired_paths"].itertuples(index=False):
        learner = main_rows.loc[main_rows.context.eq(row.context) & main_rows.policy.eq(LEARNER) & main_rows.scenario_id.eq(row.scenario_id)].iloc[0]
        reference = main_rows.loc[main_rows.context.eq(row.context) & main_rows.policy.eq(row.reference_policy) & main_rows.scenario_id.eq(row.scenario_id)].iloc[0]
        common = bool(learner.assigned and reference.assigned)
        check(row.learner_auto == learner.assigned and row.reference_available == reference.assigned and row.common_auto == common, "paired/coverage")
        for field in ("log_growth", "cagr", "mdd", "excess", "nominal_fees", "trade_fraction", "breach"):
            maybe(getattr(row, "delta_" + field), float(learner[field]) - float(reference[field]) if common else None, "paired/delta_" + field)
    for row in tables["paired_summary"].itertuples(index=False):
        data = tables["paired_paths"].loc[lambda x: x.scope.eq(row.scope) & x.reference_policy.eq(row.reference_policy)]
        common = data.loc[data.common_auto]
        check(row.original_pairs == 120 and row.learner_auto_n == data.learner_auto.sum()
              and row.reference_available_n == data.reference_available.sum() and row.common_auto_n == len(common), "paired_summary/denominators")
        for field in data.columns:
            if field.startswith("delta_"): maybe(getattr(row, "mean_" + field), float(common[field].mean()) if len(common) else None, "paired_summary/" + field)
    for name, value in manifest["outputs_sha256"].items(): check(sha(OUT / name) == value, "manifest/output_" + name)
    check(manifest["sources"] == manifest["sources_at_end"] and manifest["input_sha256"] == manifest["input_sha256_at_end"], "manifest/source_input_start_end")
    after = snapshot(); check(before == after, "audit/all_sources_artifacts_start_end")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "checks_passed": checks,
            "row_counts": {k: len(v) for k, v in tables.items()}, "replay": replay, "maximum_absolute_errors": errors,
            "training_selections": selections, "q_subset_results": q_results,
            "q_subset_audited": sum(x["status"] == "audited" for x in q_results),
            "q_subset_pending_skips": sum(x["status"] != "audited" for x in q_results),
            "q_nonautomated_job_skips": unavailable_q_jobs,
            "summary": json.loads(tables["summary"].to_json(orient="records")),
            "sources_and_artifacts_at_start": before, "sources_and_artifacts_at_end": after,
            "scenario_ids_loaded": ids.tolist(), "final_scenarios_loaded": False,
            "reviewed_implementation_imported": False, "original_manifest_modified": False,
            "scope": "Independent saved-evidence selection arithmetic; all selected actual/inner financial replay; prespecified scalar-fan Q subset",
            "limitations": ["No independent rerun of static global menu search or unrecorded nine-price training state trees",
                            "Saved memberships/hash consistency is not a full historical execution-provenance proof",
                            "Repeated-development nested selection and overlapping folds are not fresh external evaluation or a probability certificate",
                            "Conditional automatic performance does not erase Review or outside-Balanced denominators",
                            "Calibration does not repair the unconditional fan's conditional-law misspecification",
                            "AI-assisted independent arithmetic audit does not replace the entrant's review"]}


def main():
    result = run_audit()
    path = OUT / "independent_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks_passed"], "replay": result["replay"],
                      "q_audited": result["q_subset_audited"], "q_pending_skips": result["q_subset_pending_skips"],
                      "output": str(path.relative_to(ROOT)), "sha256": sha(path)}, ensure_ascii=False))


if __name__ == "__main__": main()
