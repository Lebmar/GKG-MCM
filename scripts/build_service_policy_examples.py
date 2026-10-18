"""Build conditional A-E governance examples from frozen training details only.

No raw market, simulator, optimizer or final-scenario data is imported. These
examples deliberately fail the new whole-policy admission gate until matching
joint validation evidence actually exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from experimental_execution import service_governance as governance

CONTRACT = ROOT / "docs/service_policy_refinement_contract.json"
REVIEW = ROOT / "docs/liquidity_policy_refinement_review.json"
FIT = ROOT / "results/model_review/positive_asset_specification_v1/nested_context_refits.json"
OUT = ROOT / "results/model_review/service_policy_design_v1"
EXPECTED_EXECUTION = "new_ordinal_liquidity_whole_service_pipeline_v1_not_yet_validated"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_training_inputs(full_context, profiles, *, source_artifact_sha256):
    """Translate all 15 saved pairs, without carrying old assignments as decisions."""
    fit = full_context["raw_fit"]
    if full_context["training_ids"] != list(range(1, 121)):
        raise ValueError("Use the frozen full120 training catalogue only")
    products = [{"name": p["name"], "weights": p["weights"]} for p in fit["menu"]]
    weights = {p["name"]: p["weights"] for p in products}
    rows = fit["client_product_details"]
    expected = {(p["id"], product) for p in profiles for product in governance.PRODUCTS}
    if len(rows) != 15 or {(r["client"], r["portfolio"]) for r in rows} != expected:
        raise ValueError("Require the complete five-by-three saved training detail set")
    candidates = {p["id"]: [] for p in profiles}
    for d in rows:
        if d["n"] != len(full_context["training_ids"]):
            raise ValueError("Training denominator mismatch")
        c = {"portfolio": d["portfolio"], "weights": weights[d["portfolio"]],
             "horizon": d["horizon"], "loss_tolerance": d["loss_tolerance"],
             "n": d["n"], "breach_count": d["breach_count"],
             "training_gamma": fit["training_screen_gamma"],
             "cost_rate": governance.COST_RATE,
             "mean_annual_log_growth": d["mean_log_growth"], "cvar90_mdd": d["cvar90_mdd"],
             "expected_excess": d["expected_excess"],
             "risk_metric_id": governance.RISK_METRIC, "growth_metric_id": governance.GROWTH_METRIC,
             "severity_metric_id": governance.SEVERITY_METRIC, "training_rule_id": governance.TRAINING_RULE,
             "excess_metric_id": governance.EXCESS_METRIC,
             "source_policy_id": "archived_strict172_risk_only_annual_restore",
             "source_context": "full_development_training_120", "complete": True,
             "fit_lineage": {"source_artifact_sha256": source_artifact_sha256,
                             "source_record": "full_development/raw_fit",
                             "training_ids": list(full_context["training_ids"])}}
        feasible = d["breach_count"] <= int(fit["training_screen_gamma"]*d["n"] + 1e-12)
        if feasible != d["training_screen_feasible"]:
            raise ValueError("Saved training feasibility differs from its integer count screen")
        candidates[d["client"]].append(c)
    return products, candidates


def build_examples(profiles, products, candidates, cases):
    full, flat, pair_rows, coverage = [], [], [], []
    for case in cases:
        results = []
        for profile in profiles:
            proposal = governance.propose_assignment(profile, products, candidates[profile["id"]], floors=case["floors"])
            admission = governance.assess_admission(proposal, evidence=None,
                expected_execution_policy_id=EXPECTED_EXECUTION)
            case_status = governance.classify_balanced_case(proposal, admission)
            result = {"policy_case": case["id"], "profile": profile,
                      "proposal": proposal, "admission": admission, "Balanced_case": case_status}
            results.append(result)
            scores = proposal.get("selected_training_scores", {})
            flat.append({"policy_case": case["id"], "profile": profile["id"],
                "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                "liquidity": profile["liquidity"], "target_bill_floor": proposal["target_bill_floor"],
                "conditional_assignment": proposal["assignment"], "proposal_review_type": proposal["review_type"],
                "proposal_reasons": "|".join(proposal["reason_codes"]),
                "new_service_admitted": admission["service_admitted"],
                "new_effective_assignment": admission["effective_assignment"],
                "new_service_status": admission["service_status"], "new_service_review_type": admission["review_type"],
                "conditional_Q2_case": case_status["conditional_case"],
                "effective_Q2_case": case_status["effective_case"],
                "training_evidence_fingerprint": proposal["training_evidence_fingerprint"],
                "selected_training_mean_annual_log_growth": scores.get("mean_annual_log_growth"),
                "selected_training_cvar90_mdd": scores.get("cvar90_mdd"),
                "selected_training_expected_excess": scores.get("expected_excess"),
                "selected_training_breach_count": scores.get("training_breach_count"),
                "selected_training_n": scores.get("training_n"),
                "financial_outcome": None})
            for pair in proposal["pair_assessments"]:
                pair_rows.append({"policy_case": case["id"], "profile": profile["id"],
                    "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                    "liquidity": profile["liquidity"], "target_bill_floor": proposal["target_bill_floor"],
                    **pair, "selected_conditional_proposal": pair["portfolio"] == proposal["assignment"]})
        full.extend(results)
        conditional_n = sum(r["proposal"]["proposal_status"] == "conditional_proposal" for r in results)
        admitted_n = sum(r["admission"]["service_admitted"] for r in results)
        coverage.append({"policy_case": case["id"], "example_profile_n": len(results),
            "conditional_proposal_n": conditional_n,
            "conditional_proposal_share_of_five_examples": conditional_n/len(results) if results else None,
            "proposal_review_reasons": dict(Counter(r["proposal"]["review_type"] for r in results if r["proposal"]["review_type"])),
            "formal_service_coverage": governance.coverage_summary(original_n=len(results),
                case_n=len(results), automated_n=admitted_n, unit="illustrative_profile_cases"),
            "formal_service_review_reasons": dict(Counter(r["admission"]["review_type"] for r in results if r["admission"]["review_type"])),
            "customer_population_coverage_estimated": False})
    return full, flat, pair_rows, coverage


def runtime_examples():
    """Invented current-state examples only, never market paths or customer facts."""
    profile = {"id": "E", "horizon": 6, "loss_tolerance": .35, "liquidity": "High"}
    target, actual = [.45, .05, .5], [.65, .1, .25]
    history = {"original_horizon": 6, "elapsed": 2, "holdings": [65, 10, 25],
               "cpi": 1, "real_peak": 110, "running_mdd": .2, "ever_breached": False}
    pending_history = {**history, "running_mdd": .42, "ever_breached": True}
    cases = {
        "actual_drift_only_notice": governance.runtime_notices(profile, target, actual, history=history),
        "known_need_change_no_invented_investment_action": governance.runtime_notices(profile, target, actual,
            known_need_changed=True, history=history),
        "existing_pending_no_risk_reset_or_automatic_reentry": governance.runtime_notices(profile, target,
            [.2, .1, .7], review_pending=True, history=pending_history),
        "reject_incompatible_new_target_without_modifying_weights": governance.target_compatibility(profile, [.7, .1, .2])}
    if cases["actual_drift_only_notice"]["request_review"]:
        raise AssertionError("Drift silently added a live-floor review gate")
    if any(c["decision_action"] is not None for c in cases.values()):
        raise AssertionError("Governance examples invented an investment action")
    if cases["existing_pending_no_risk_reset_or_automatic_reentry"]["history"] != pending_history:
        raise AssertionError("History was reset")
    return {"scope": "synthetic governance states, not observed returns, population facts or financial results", "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    inputs = contract["frozen_input_sha256"]
    actual_inputs = {p: sha(ROOT/p) for p in inputs}
    if actual_inputs != inputs:
        raise RuntimeError("A contract-frozen input changed")
    strict_protocol = json.loads((ROOT/"docs/positive_asset_specification_protocol.json").read_text())
    if (strict_protocol["candidate_family"]["cost_rate"] != governance.COST_RATE
            or contract["policy"]["cost_rate"] != governance.COST_RATE):
        raise ValueError("Training artifact/common service cost rate mismatch")
    source_files = [CONTRACT, REVIEW, Path(__file__).resolve(),
        ROOT/"research/experimental_execution/service_governance.py", ROOT/"tests/test_service_governance.py"]
    source_start = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    review = json.loads(REVIEW.read_text())
    profiles = [{"id": p["id"], "horizon": p["H"], "loss_tolerance": p["L"], "liquidity": p["liquidity"]}
                for p in review["source_reading"]["profiles"]]
    full_context = json.loads(FIT.read_text())["full_development"]
    products, candidates = normalized_training_inputs(full_context, profiles, source_artifact_sha256=sha(FIT))
    cases = [{"id": "reference_reserve_primary_prototype", "floors": contract["policy"]["target_bill_floors"]},
             {"id": "larger_reserve_norm_sensitivity", "floors": contract["policy"]["floor_sensitivity"]["target_bill_floors"]}]
    examples, rows, pairs, coverage = build_examples(profiles, products, candidates, cases)
    primary = {r["profile"]["id"]: r["proposal"]["assignment"] for r in examples if r["policy_case"] == cases[0]["id"]}
    if primary != contract["examples"]["expected_conditional_assignments"]:
        raise AssertionError("Conditional illustration differs from its declared expected mapping")
    if any(r["admission"]["service_admitted"] or r["admission"]["financial_outcome"] is not None for r in examples):
        raise AssertionError("Legacy training evidence became unearned formal admission/results")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    outputs = []
    def save_json(name, value):
        write_json(out/name, value); outputs.append(name)
    def save_csv(name, value):
        write_csv(out/name, value); outputs.append(name)
    save_json("source_catalogue_and_training_candidates.json", {
        "scope": "old full120 risk-only annual-restoration training details only",
        "catalogue": products, "profiles": profiles, "candidates_by_profile": candidates,
        "new_fit_or_validation": False})
    save_json("profile_proposals.json", {"status": "conditional_catalogue_examples_not_new_joint_validation", "examples": examples})
    save_csv("profile_proposals.csv", rows)
    save_csv("training_candidate_screen.csv", pairs)
    save_json("coverage_and_units.json", {"profile_examples": coverage,
        "unit_arithmetic_only_not_new_experiment": {
            "two_of_five_training_refits": governance.coverage_summary(original_n=5, case_n=5, automated_n=2, unit="training_refit_runs"),
            "forty_eight_of_120_market_paths": governance.coverage_summary(original_n=120, case_n=120, automated_n=48, unit="market_scenario_paths")},
        "interpretation": "The two displayed .4 ratios have distinct units. Neither estimates customer-population coverage; actual profile examples are five fictional questionnaire cases, not 120 customers."})
    save_json("runtime_examples.json", runtime_examples())
    save_json("interpretation.json", {
        "main_rule": "High/Medium/Low target-bill floors30/10/0 are a disclosed contest convention, not uniquely identified preferences or withdrawal guarantees.",
        "conditional_mapping": primary,
        "A_D": "Risk-training candidates absent in this fixed catalogue; not referrals merely for missing extra client facts.",
        "B_C_E": "Conditional Conservative/Balanced/Balanced from code. Old training details do not jointly validate the new liquidity/service process.",
        "E": "H6 Balanced candidate uses H6 metrics. E-Growth inner evidence and C-H30 execution evidence cannot transfer.",
        "runtime": "Actual bill drift is notice only. New needs/questionnaire/model-scope changes request review; no investment action or emergency response is fabricated.",
        "coverage": "Five illustrative profiles, training refit runs and market scenario paths are different denominators. No population estimate is made.",
        "no_winner": "Larger reserve case is descriptive normative sensitivity, not selected using growth, coverage or risk.",
        "next_step": "Freeze one complete joint menu/onboarding/execution procedure and validate within its own training/inner/outer identities before admitting any new service.",
        "financial_results": None, "final_scenarios_loaded": False})
    source_end = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    input_end = {p: sha(ROOT/p) for p in inputs}
    if source_start != source_end or actual_inputs != input_end:
        raise RuntimeError("Source/input changed during example extraction")
    write_json(out/"manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_pure_governance_prototype_not_jointly_validated",
        "sources": source_start, "sources_at_end": source_end,
        "input_sha256": actual_inputs, "input_sha256_at_end": input_end,
        "outputs_sha256": {name: sha(out/name) for name in outputs},
        "checks": {"saved_training_pairs_read": 15, "policy_cases": 2, "example_profile_rows": len(rows),
            "conditional_primary_assignments_recomputed": primary, "new_formally_admitted_examples": 0,
            "financial_results_generated": False, "drift_only_not_a_review_gate": True,
            "history_preserved": True, "coverage_units_separated": True},
        "new_market_data_loaded": False, "final_scenarios_loaded": False,
        "new_fit_forecast_or_simulation": False, "old_archives_modified": False,
        "limitations": ["Conditional training-catalogue examples only; not new joint validation or a final service.",
            "Pure receipt matching cannot authenticate evidence; caller must verify hashes/audit and substantive scope.",
            "Fit-lineage matching prevents cross-fit receipt reuse; it is not a demand for independent new-market evaluation of each final refit or a risk certificate.",
            "Normative reserve convention is not a live cash floor, liability model or uniquely identified preference.",
            "No market path, return forecast, financial fallback, adviser response or emergency liquidation is produced."]})
    print(f"Wrote {out}; conditional primary: {primary}; formal service admissions:0")
    print(f"Manifest SHA256 {sha(out/'manifest.json')}")


if __name__ == "__main__":
    main()
