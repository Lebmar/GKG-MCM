"""Independent decision-boundary examples; no market data or return simulation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from experimental_execution import service_governance as g


def fixture(liquidity="High", horizon=6, loss=.35):
    profile = {"id": "E", "horizon": horizon, "loss_tolerance": loss, "liquidity": liquidity}
    products = [{"name": n, "weights": w} for n, w in zip(g.PRODUCTS,
        [[.25, .30, .45], [.45, .05, .50], [.70, .10, .20]])]
    candidates = []
    for p, k, growth, severity in zip(products, [0, 2, 9], [.02, .03, .04], [.2, .3, .5]):
        candidates.append({"portfolio": p["name"], "weights": p["weights"],
            "horizon": horizon, "loss_tolerance": loss, "n": 120, "breach_count": k,
            "training_gamma": .075, "mean_annual_log_growth": growth, "cvar90_mdd": severity,
            "expected_excess": .01,
            "cost_rate": .0025,
            "risk_metric_id": g.RISK_METRIC, "growth_metric_id": g.GROWTH_METRIC,
            "severity_metric_id": g.SEVERITY_METRIC, "training_rule_id": g.TRAINING_RULE,
            "excess_metric_id": g.EXCESS_METRIC,
            "source_policy_id": "legacy_risk_only", "source_context": "synthetic_context",
            "fit_lineage": {"source_artifact_sha256": "c"*64, "source_record": "synthetic_context/raw_fit",
                            "training_ids": list(range(1, 121))},
            "complete": True})
    return profile, products, candidates


def matching_toy_receipt(proposal):
    """Synthetic schema receipt, not a claim that any real pipeline was validated."""
    p, s = proposal["profile"], proposal["service_policy"]
    return {"service_policy_id": s["policy_id"], "service_policy_fingerprint": s["fingerprint"],
        "receipt_schema": g.JOINT_RECEIPT_SCHEMA,
        "catalogue_fingerprint": proposal["catalogue_fingerprint"], "profile_id": p["id"],
        "training_evidence_fingerprint": proposal["training_evidence_fingerprint"],
        "horizon": p["horizon"], "loss_tolerance": p["loss_tolerance"], "liquidity": p["liquidity"],
        "product": proposal["assignment"], "target_weights": proposal["target_weights"],
        "risk_metric_id": g.RISK_METRIC, "growth_metric_id": g.GROWTH_METRIC,
        "severity_metric_id": g.SEVERITY_METRIC, "execution_policy_id": "toy_complete_pipeline_v1",
        "excess_metric_id": g.EXCESS_METRIC, "proposal_sort_rule_id": g.PROPOSAL_SORT_RULE,
        "cost_rate": .0025, "service_alpha": .10,
        "validation_scope": "joint_service_and_execution_pipeline", "validation_unit": "whole_scenario_pipeline",
        "status": "passed", "training_candidate_only": False, "receipts_verified_by_caller": True,
        "source_manifest_sha256": "a"*64, "independent_audit_sha256": "b"*64}


class ConditionalProposalTests(unittest.TestCase):
    def test_high_cash_need_can_exclude_highest_growth_without_relabeling_risk(self):
        p, products, candidates = fixture()
        result = g.propose_assignment(p, products, candidates)
        self.assertEqual(result["assignment"], "Balanced")
        growth = result["pair_assessments"][2]
        self.assertTrue(growth["risk_training_candidate"])
        self.assertFalse(growth["liquidity_target_compatible"])
        self.assertIsNone(result["financial_outcome"])
        self.assertFalse(result["training_evidence_is_joint_service_validation"])

    def test_low_adds_no_floor_but_does_not_create_zero_asset_rule(self):
        p, products, candidates = fixture("Low")
        result = g.propose_assignment(p, products, candidates)
        self.assertEqual(result["assignment"], "Growth")
        self.assertEqual(result["target_bill_floor"], 0)
        self.assertGreater(min(result["target_weights"]), 0)

    def test_count_screen_is_integer_and_not_replaced_by_service_alpha(self):
        p, products, candidates = fixture("Low")
        candidates[2]["breach_count"] = 10  # 10/120 < .10 but exceeds gamma .075 (9/120).
        result = g.propose_assignment(p, products, candidates)
        self.assertEqual(result["assignment"], "Balanced")
        self.assertEqual(result["pair_assessments"][2]["training_count_limit"], 9)
        self.assertEqual(result["service_policy"]["service_alpha"], .10)

    def test_screen_above_the_declared_service_alpha_is_not_a_candidate(self):
        p, products, candidates = fixture("Low")
        for c in candidates: c["training_gamma"] = .20
        result = g.propose_assignment(p, products, candidates)
        self.assertEqual(result["review_type"], "model_review")
        self.assertIn("Growth:training_gamma_above_service_alpha", result["reason_codes"])

    def test_risk_liquidity_and_model_referrals_remain_distinct(self):
        p, products, candidates = fixture()
        for c in candidates: c["breach_count"] = 10
        risk = g.propose_assignment(p, products, candidates)
        self.assertEqual(risk["review_type"], "risk_review")
        for product, c in zip(products, candidates):
            product["weights"] = [.5, .3, .2]
            c["weights"] = [.5, .3, .2]
            c["breach_count"] = 0
        liquidity = g.propose_assignment(p, products, candidates)
        self.assertEqual(liquidity["review_type"], "liquidity_review")
        model = g.propose_assignment(p, products, candidates[:2])
        self.assertEqual(model["review_type"], "model_review")
        self.assertTrue(all(x["financial_outcome"] is None for x in [risk, liquidity, model]))

    def test_complete_training_set_does_not_cherry_pick_usable_pairs(self):
        p, products, candidates = fixture()
        candidates[2]["complete"] = False
        result = g.propose_assignment(p, products, candidates)
        self.assertEqual(result["assignment"], "Review")
        self.assertEqual(result["review_type"], "model_review")

    def test_H_L_target_and_metric_mismatches_are_model_issues(self):
        p, products, candidates = fixture()
        for key, value in [("horizon", 30), ("loss_tolerance", .40),
                           ("weights", [.6, .1, .3]), ("risk_metric_id", "nominal_or_old_metric"),
                           ("cost_rate", 0.0)]:
            with self.subTest(key=key):
                altered = deepcopy(candidates); altered[1][key] = value
                result = g.propose_assignment(p, products, altered)
                self.assertEqual(result["review_type"], "model_review")
        altered = deepcopy(candidates); altered[0]["source_context"] = "another_train_sample"
        self.assertEqual(g.propose_assignment(p, products, altered)["review_type"], "model_review")

    def test_global_growth_tolerance_cannot_accumulate_pairwise(self):
        p, products, candidates = fixture("Low")
        for c, growth, severity in zip(candidates,
                [1, 1-.75*g.TOL, 1-1.5*g.TOL], [.5, .3, .01]):
            c["mean_annual_log_growth"], c["cvar90_mdd"] = growth, severity
        result = g.propose_assignment(p, products, candidates)
        self.assertEqual(result["assignment"], "Balanced")

    def test_global_severity_tie_uses_canonical_order_only_inside_band(self):
        p, products, candidates = fixture("Low")
        for c, severity in zip(candidates, [.2+1.5*g.TOL, .2+.75*g.TOL, .2]):
            c["mean_annual_log_growth"], c["cvar90_mdd"] = .03, severity
        self.assertEqual(g.propose_assignment(p, products, candidates)["assignment"], "Balanced")

    def test_expected_excess_precedes_CVaR_with_global_tolerance(self):
        p, products, candidates = fixture("Low")
        for c, excess, cvar in zip(candidates,
                [.02+1.5*g.TOL, .02+.75*g.TOL, .02], [.001, .002, .3]):
            c["mean_annual_log_growth"] = .03
            c["expected_excess"], c["cvar90_mdd"] = excess, cvar
        self.assertEqual(g.propose_assignment(p, products, candidates)["assignment"], "Balanced")

    def test_catalogue_and_policy_fingerprints_ignore_input_order_not_policy_values(self):
        p, products, candidates = fixture()
        a = g.propose_assignment(p, products, candidates)
        b = g.propose_assignment(p, products[::-1], candidates[::-1])
        self.assertEqual(a, b)
        self.assertNotEqual(g.make_policy()["fingerprint"], g.make_policy(g.LARGER_FLOORS)["fingerprint"])

    def test_invalid_customer_product_or_nonmonotone_policy_is_rejected(self):
        p, products, candidates = fixture()
        for key, value in [("horizon", 0), ("horizon", 31), ("horizon", True),
                           ("loss_tolerance", 0), ("loss_tolerance", float("nan")), ("liquidity", "Unknown")]:
            with self.subTest(key=key, value=value):
                altered = {**p, key: value}
                with self.assertRaises(ValueError): g.propose_assignment(altered, products, candidates)
        for weights in [[.8, .2, 0], [.8, .2, .1], [.8, -.1, .3], [.5, float("inf"), .5]]:
            altered = deepcopy(products); altered[0]["weights"] = weights
            with self.assertRaises(ValueError): g.propose_assignment(p, altered, candidates)
        with self.assertRaises(ValueError): g.make_policy({"High": .1, "Medium": .3, "Low": 0})


class JointEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.proposal = g.propose_assignment(*fixture())

    def assess(self, evidence=None):
        return g.assess_admission(self.proposal, evidence, expected_execution_policy_id="toy_complete_pipeline_v1")

    def test_old_training_results_never_automatically_admit_new_service(self):
        result = self.assess()
        self.assertEqual(result["conditional_assignment"], "Balanced")
        self.assertEqual(result["effective_assignment"], "Review")
        self.assertEqual(result["review_type"], "model_review")
        self.assertIsNone(result["financial_outcome"])

    def test_exact_whole_pipeline_receipt_is_only_a_version_match_not_guarantee(self):
        result = self.assess(matching_toy_receipt(self.proposal))
        self.assertTrue(result["service_admitted"])
        self.assertTrue(result["not_a_risk_guarantee"])
        self.assertIsNone(result["financial_outcome"])

    def test_old_risk_only_product_H_and_execution_versions_cannot_transfer(self):
        for key, value in [("service_policy_id", "old_risk_only"),
                           ("validation_scope", "static_training_only"),
                           ("product", "Growth"), ("horizon", 30), ("loss_tolerance", .40),
                           ("liquidity", "Low"), ("target_weights", [.7, .1, .2]),
                           ("execution_policy_id", "old_v0"), ("receipts_verified_by_caller", False),
                           ("training_candidate_only", True), ("independent_audit_sha256", ""),
                           ("risk_metric_id", "old_nominal_metric"), ("cost_rate", 0),
                           ("service_alpha", .20), ("receipts_verified_by_caller", 1)]:
            with self.subTest(key=key):
                evidence = matching_toy_receipt(self.proposal); evidence[key] = value
                self.assertFalse(self.assess(evidence)["service_admitted"])

    def test_malformed_evidence_is_pending_model_review_not_an_automatic_exception(self):
        result = self.assess(["passed"])
        self.assertFalse(result["service_admitted"])
        self.assertEqual(result["review_type"], "model_review")

    def test_numeric_boolean_is_not_a_one_year_horizon_receipt(self):
        proposal = g.propose_assignment(*fixture(horizon=1))
        evidence = matching_toy_receipt(proposal)
        evidence["horizon"] = True
        result = g.assess_admission(proposal, evidence, expected_execution_policy_id="toy_complete_pipeline_v1")
        self.assertFalse(result["service_admitted"])

    def test_same_catalogue_metrics_and_n_do_not_allow_different_training_ids(self):
        p, products, candidates = fixture()
        for c in candidates:
            c["n"] = 96
            c["breach_count"] = min(c["breach_count"], 7)
            c["fit_lineage"]["training_ids"] = list(range(1, 97))
        first = g.propose_assignment(p, products, candidates)
        evidence = matching_toy_receipt(first)
        changed = deepcopy(candidates)
        for c in changed: c["fit_lineage"]["training_ids"] = list(range(25, 121))
        second = g.propose_assignment(p, products, changed)
        self.assertEqual(first["catalogue_fingerprint"], second["catalogue_fingerprint"])
        self.assertEqual(first["assignment"], second["assignment"])
        self.assertNotEqual(first["training_evidence_fingerprint"], second["training_evidence_fingerprint"])
        self.assertFalse(g.assess_admission(second, evidence,
            expected_execution_policy_id="toy_complete_pipeline_v1")["service_admitted"])

    def test_source_fit_artifact_hash_is_not_replaceable_by_same_aggregate_scores(self):
        p, products, candidates = fixture()
        first = g.propose_assignment(p, products, candidates)
        evidence = matching_toy_receipt(first)
        for c in candidates: c["fit_lineage"]["source_artifact_sha256"] = "d"*64
        second = g.propose_assignment(p, products, candidates)
        self.assertNotEqual(first["training_evidence_fingerprint"], second["training_evidence_fingerprint"])
        self.assertFalse(g.assess_admission(second, evidence,
            expected_execution_policy_id="toy_complete_pipeline_v1")["service_admitted"])

    def test_stale_training_payload_hash_and_invalid_lineage_cannot_admit(self):
        proposal = deepcopy(self.proposal)
        evidence = matching_toy_receipt(proposal)
        proposal["training_evidence_payload"]["training_candidates"][0]["mean_annual_log_growth"] += .001
        self.assertFalse(g.assess_admission(proposal, evidence,
            expected_execution_policy_id="toy_complete_pipeline_v1")["service_admitted"])
        p, products, candidates = fixture()
        candidates[0]["fit_lineage"]["training_ids"][0] = 121
        self.assertEqual(g.propose_assignment(p, products, candidates)["review_type"], "model_review")

    def test_mutable_proposal_hash_labels_do_not_authenticate_changed_contents(self):
        evidence = matching_toy_receipt(self.proposal)
        altered = deepcopy(self.proposal)
        altered["service_policy"]["target_bill_floors"]["High"] = .20
        result = g.assess_admission(altered, evidence, expected_execution_policy_id="toy_complete_pipeline_v1")
        self.assertFalse(result["service_admitted"])
        self.assertIn("proposal_policy_fingerprint_or_content_inconsistent", result["reason_codes"])
        altered = deepcopy(self.proposal)
        altered["catalogue"][1]["weights"] = [.5, .1, .4]
        result = g.assess_admission(altered, evidence, expected_execution_policy_id="toy_complete_pipeline_v1")
        self.assertFalse(result["service_admitted"])
        self.assertIn("proposal_catalogue_fingerprint_inconsistent", result["reason_codes"])
        altered = deepcopy(self.proposal)
        altered["target_bill_floor"] = 0
        self.assertFalse(g.assess_admission(altered, evidence,
            expected_execution_policy_id="toy_complete_pipeline_v1")["service_admitted"])

    def test_floor_change_requires_a_new_matching_receipt(self):
        evidence = matching_toy_receipt(self.proposal)
        alternative = g.propose_assignment(*fixture(), floors=g.LARGER_FLOORS)
        self.assertFalse(g.assess_admission(alternative, evidence,
            expected_execution_policy_id="toy_complete_pipeline_v1")["service_admitted"])

    def test_existing_risk_review_is_not_overwritten_as_model_failure(self):
        p, products, candidates = fixture()
        for c in candidates: c["breach_count"] = 30
        proposal = g.propose_assignment(p, products, candidates)
        result = g.assess_admission(proposal, expected_execution_policy_id="toy_complete_pipeline_v1")
        self.assertEqual(result["review_type"], "risk_review")

    def test_nonBalanced_and_pending_Balanced_cases_are_not_forced_auto(self):
        proposal = g.propose_assignment(*fixture("Low"))
        admission = g.assess_admission(proposal, expected_execution_policy_id="toy_complete_pipeline_v1")
        case = g.classify_balanced_case(proposal, admission)
        self.assertEqual(case["conditional_case"], "not_balanced_case")
        self.assertEqual(case["effective_case"], "model_review")
        pending = self.assess()
        case = g.classify_balanced_case(self.proposal, pending)
        self.assertEqual(case["conditional_case"], "conditional_balanced_case")
        self.assertEqual(case["effective_case"], "model_review")
        self.assertEqual(case["original_horizon"], 6)


class RuntimeAndCoverageTests(unittest.TestCase):
    def test_actual_drift_is_a_notice_and_issues_no_review_trade_or_hold(self):
        p, _, _ = fixture()
        r = g.runtime_notices(p, [.45, .05, .5], [.65, .1, .25])
        self.assertIn("bill_share_below_target", r["notices"])
        self.assertIn("bill_share_below_policy_target_floor", r["notices"])
        self.assertFalse(r["request_review"])
        self.assertFalse(r["freeze_unapproved_new_targets"])
        self.assertIsNone(r["decision_action"])
        self.assertFalse(r["live_cash_floor_guaranteed"])

    def test_new_facts_request_review_without_inventing_liquidation_or_emergency_action(self):
        p, _, _ = fixture()
        for flags in [{"known_need_changed": True}, {"questionnaire_changed": True}, {"model_applicable": False}]:
            r = g.runtime_notices(p, [.45, .05, .5], [.45, .05, .5], **flags)
            self.assertTrue(r["request_review"])
            self.assertTrue(r["freeze_unapproved_new_targets"])
            self.assertIsNone(r["decision_action"])
            self.assertIsNone(r["emergency_liability_action"])

    def test_pending_does_not_clear_when_cash_recovers_or_reset_history(self):
        p, _, _ = fixture()
        history = {"original_horizon": 6, "elapsed": 4, "holdings": [30, 15, 45],
                   "cpi": 1.2, "real_peak": 110, "running_mdd": .42, "ever_breached": True,
                   "referral_reasons": ["observed_financial_crossing"]}
        before = deepcopy(history)
        r = g.runtime_notices(p, [.45, .05, .5], [.2, .1, .7], review_pending=True, history=history)
        self.assertTrue(r["request_review"])
        self.assertEqual(r["history"], before)
        self.assertEqual(history, before)
        self.assertFalse(r["history_reset"])
        r["history"]["referral_reasons"].append("changed_copy")
        self.assertEqual(history, before)

    def test_incompatible_target_is_distinct_from_actual_drift(self):
        p, _, _ = fixture()
        r = g.runtime_notices(p, [.7, .1, .2], [.1, .1, .8])
        self.assertTrue(r["request_review"])
        self.assertIn("current_target_incompatible_with_declared_policy", r["review_reason_codes"])
        self.assertEqual(r["notices"], [])

    def test_new_target_is_blocked_without_silently_adding_bills(self):
        p, _, _ = fixture()
        bad = [.7, .1, .2]
        result = g.target_compatibility(p, bad)
        self.assertFalse(result["target_reserve_compatible"])
        self.assertEqual(result["target_weights"], bad)
        self.assertIsNone(result["decision_action"])
        self.assertFalse(result["risk_or_execution_approved"])

    def test_zero_denominators_remain_null_not_zero_risk(self):
        r = g.coverage_summary(original_n=0, case_n=0, automated_n=0, unit="illustrative_profile_cases")
        self.assertIsNone(r["auto_over_original"])
        self.assertIsNone(r["auto_given_case"])
        r = g.coverage_summary(original_n=120, case_n=0, automated_n=0, unit="market_scenario_paths")
        self.assertEqual(r["auto_over_original"], 0)
        self.assertIsNone(r["auto_given_case"])
        self.assertIsNone(r["financial_outcome"])

    def test_equal_numerical_refit_and_scenario_ratios_do_not_imply_population_coverage(self):
        refits = g.coverage_summary(original_n=5, case_n=5, automated_n=2, unit="training_refit_runs")
        paths = g.coverage_summary(original_n=120, case_n=120, automated_n=48, unit="market_scenario_paths")
        self.assertEqual(refits["auto_over_original"], paths["auto_over_original"])
        self.assertNotEqual(refits["unit"], paths["unit"])
        self.assertFalse(paths["customer_population_coverage_estimated"])
        with self.assertRaises(ValueError):
            g.coverage_summary(original_n=5, case_n=6, automated_n=2, unit="illustrative_profile_cases")
        with self.assertRaises(ValueError):
            g.coverage_summary(original_n=120, case_n=120, automated_n=48, unit="customer_population")


if __name__ == "__main__":
    unittest.main()
