"""Independent closed-form tests; no raw or prospective numeric-study inputs."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_simple_execution_comparison import (
    AuditLedger, compare_saved_account, independent_account,
    independent_drift_action, independent_fee_trade, independent_four_inner_receipt,
    independent_metrics, independent_training_choice, validate_original_membership,
    validate_protocol_membership,
)
import audit_simple_execution_comparison as audit_module


def training_records():
    return [dict(candidate_order=i, delta=d, n=96, breach_count=2,
                 mean_log_growth=.05, mean_excess=.01, mean_nominal_fees=.5)
            for i, d in enumerate((0., .05, .1))]


def inner_records():
    return [dict(fold=i + 1, evaluation_ids=list(range(24 * i + 1, 24 * (i + 1) + 1)),
                 automated=True, breach_count=2, reference_breach_count=2, mean_log_growth=.03)
            for i in range(4)]


def synthetic_protocol_membership():
    """Only ID sets and lineage; contains no market observations or fit values."""
    contexts, jobs, static = [], [], {}
    all_ids = set(range(1, 121))
    for repeat in range(1, 6):
        for fold in range(1, 6):
            cid = 100 * repeat + fold; name = f"synthetic_r{repeat}_outer{fold}"
            evaluation = list(range(24 * (fold - 1) + 1, 24 * fold + 1))
            training = sorted(all_ids - set(evaluation))
            inner = [dict(inner_fold=i + 1, test_ids=training[24 * i:24 * (i + 1)],
                          train_ids=sorted(set(training) - set(training[24 * i:24 * (i + 1)]))) for i in range(4)]
            context = dict(context_id=cid, name=name, repeat_id=repeat, outer_fold=fold,
                           training_ids=training, outer_test_ids=evaluation, inner_seed=cid, inner_folds=inner)
            contexts.append(context)
            for number in range(1, 6):
                is_refit = number == 5
                train = training if is_refit else inner[number - 1]["train_ids"]
                test = evaluation if is_refit else inner[number - 1]["test_ids"]
                jid = f"c{cid}_refit" if is_refit else f"c{cid}_inner{number}"
                job = dict(job_id=jid, parent_context=name, parent_context_id=cid, repeat_id=repeat, outer_fold=fold,
                           dynamic_inner_fold=None if is_refit else number, stage="refit" if is_refit else "inner",
                           training_ids=train, evaluation_ids=test, static_inner_seed=cid * 10 + number)
                jobs.append(job); part = len(train) // 4
                static[jid] = dict(name=name if is_refit else jid, training_ids=train, outer_test_ids=test,
                                   inner_seed=job["static_inner_seed"],
                                   inner_folds=[dict(inner_fold=i + 1, test_ids=train[part * i:part * (i + 1)],
                                                     train_ids=sorted(set(train) - set(train[part * i:part * (i + 1)]))) for i in range(4)])
    return dict(data=dict(scenario_ids=list(range(1, 121)), allow_final=False), original_H=30, loss_tolerance=.4,
                service_alpha=.1, cost_rate=.0025, liquidity_floors=dict(high=.3, medium=.1, low=0.),
                architectures={"annual_gated_diagnostic": [0.], "drift_band_gated_diagnostic": [0., .05, .1]},
                contexts=contexts, jobs=jobs, static_contexts=static)


class SimpleExecutionIndependentAuditTests(unittest.TestCase):
    def test_full_initial_switch_closed_form_fee_once(self):
        post, fee, moved = independent_fee_trade([0., 0., 100.], [1., 0., 0.])
        expected_fee = .0025 * 100 / 1.0025
        self.assertAlmostEqual(fee, expected_fee, places=13)
        np.testing.assert_allclose(post, [100 - expected_fee, 0., 0.], atol=2e-13, rtol=0)
        self.assertAlmostEqual(moved, 100 - expected_fee, places=13)
        self.assertAlmostEqual(fee, .0025 * moved, places=14)

    def test_three_assets_charge_total_buys_not_buys_plus_sells(self):
        target = np.array([.4, .25, .35])
        post, fee, moved = independent_fee_trade([0., 0., 100.], target)
        expected_fee = 100 * .0025 * .65 / (1 + .0025 * .65)
        self.assertAlmostEqual(fee, expected_fee, places=13)
        self.assertAlmostEqual(moved, .65 * (100 - fee), places=13)
        self.assertAlmostEqual(sum(post) + fee, 100., places=13)

    def test_no_move_no_fee_and_target_restore_is_not_necessarily_trade(self):
        account = independent_account(np.zeros((3, 3)), np.zeros(3), [0., 0., 1.], 0.)
        self.assertEqual(account["actions"].tolist(), ["restore"] * 3)
        self.assertEqual(account["target_execution"].tolist(), [True] * 3)
        self.assertFalse(account["trades"].any())
        np.testing.assert_allclose(account["fees"], 0., atol=1e-28, rtol=0)

    def test_strict_drift_gate_has_no_extra_default_tolerance(self):
        target = [.25, .25, .5]
        self.assertEqual(independent_drift_action([30., 20., 50.], target, .05, 1, False)[0], "hold")
        above = [30.00000000001, 19.99999999999, 50.]
        self.assertEqual(independent_drift_action(above, target, .05, 1, False)[0], "restore")
        self.assertEqual(independent_drift_action(above, target, .05, 1, False, decision_tolerance=1e-12)[0], "hold")

    def test_initial_restore_and_pending_precedes_annual_or_large_drift(self):
        self.assertEqual(independent_drift_action([0., 0., 100.], [.4, .25, .35], .1, 0, False)[0], "restore")
        for delta in (0., .05, .1):
            self.assertEqual(independent_drift_action([100., 0., 0.], [.4, .25, .35], delta, 2, True)[0], "hold")
        with self.assertRaises(ValueError):
            independent_drift_action([0., 0., 100.], [.4, .25, .35], .1, 0, True)

    def test_known_three_year_account_breach_recovery_and_absorbing_pending(self):
        market = np.array([[-.6, 0., 0.], [-.6, 0., 0.], [4., 0., 0.]])
        account = independent_account(market, np.zeros(3), [.5, 0., .5], 0., cost_rate=0.)
        np.testing.assert_allclose(account["real_wealth"], [100., 70., 49., 105.], atol=1e-13, rtol=0)
        self.assertEqual(account["actions"].tolist(), ["restore", "restore", "hold"])
        self.assertEqual(account["ever_breached"].tolist(), [False, False, True, True])
        self.assertEqual(account["pending_at_decision"].tolist(), [False, False, True])
        self.assertEqual(account["first_crossing"], 2)
        self.assertAlmostEqual(account["running_mdd"][-1], .51)
        self.assertAlmostEqual(account["real_peak"][-1], 105.)
        metrics = independent_metrics(account)
        self.assertAlmostEqual(metrics["log_growth"], math.log(1.05) / 3)
        self.assertAlmostEqual(metrics["excess"], .11)
        self.assertAlmostEqual(metrics["trade_fraction"], 2 / 3)

    def test_inflation_preserves_peak_and_breach_for_nominally_flat_bills(self):
        account = independent_account(np.zeros((2, 3)), np.array([1., 0.]), [0., 0., 1.], 0.)
        np.testing.assert_array_equal(account["nominal_wealth"], [100., 100., 100.])
        np.testing.assert_array_equal(account["cpi"], [1., 2., 2.])
        np.testing.assert_array_equal(account["real_wealth"], [100., 50., 50.])
        self.assertEqual(account["first_crossing"], 1)
        self.assertEqual(account["actions"].tolist(), ["restore", "hold"])
        self.assertTrue(independent_metrics(account)["breach"])

    def test_terminal_crossing_has_no_extra_decision_or_liquidation_fee(self):
        account = independent_account(np.array([[-.9, 0., 0.]]), np.zeros(1), [.5, 0., .5], .1)
        self.assertEqual(account["first_crossing"], 1)
        self.assertEqual(len(account["fees"]), 1)
        self.assertEqual(account["pending_at_decision"].tolist(), [False])
        fee = 100 * .0025 * .5 / (1 + .0025 * .5)
        self.assertAlmostEqual(account["fees"].sum(), fee, places=13)

    def test_always_annual_reference_keeps_breach_history_without_pending_hold(self):
        market = np.array([[-.6, 0., 0.], [-.6, 0., 0.], [4., 0., 0.]])
        account = independent_account(market, np.zeros(3), [.5, 0., .5], 0., cost_rate=0., hold_after_breach=False)
        self.assertEqual(account["actions"].tolist(), ["restore"] * 3)
        self.assertEqual(account["first_crossing"], 2)
        self.assertEqual(account["ever_breached"].tolist(), [False, False, True, True])
        self.assertFalse(account["pending_at_decision"].any())
        self.assertAlmostEqual(account["real_wealth"][-1], 147.)
        with self.assertRaises(ValueError):
            independent_account(market, np.zeros(3), [.5, 0., .5], .05, hold_after_breach=False)

    def test_invalid_shapes_gross_returns_and_out_of_family_threshold_fail(self):
        for r, pi in [(np.zeros((2, 2)), np.zeros(2)), (np.array([[-1., 0., 0.]]), np.zeros(1)),
                      (np.zeros((1, 3)), np.array([-1.])), (np.zeros((1, 3)), np.zeros(2))]:
            with self.subTest(shape=r.shape), self.assertRaises(ValueError):
                independent_account(r, pi, [.4, .25, .35], .05)
        with self.assertRaises(ValueError):
            independent_account(np.zeros((1, 3)), np.zeros(1), [.4, .25, .35], .2)

    def test_saved_account_corruption_does_not_supply_its_own_actions(self):
        expected = independent_account(np.array([[.2, 0., 0.], [0., 0., 0.]]), np.zeros(2), [.5, 0., .5], .1)
        compare_saved_account(expected, expected)
        for field, index, value in [("fees", 0, .5), ("actions", 1, "restore"), ("ever_breached", -1, True)]:
            corrupt = copy.deepcopy(expected); corrupt[field][index] = value
            with self.subTest(field=field), self.assertRaises(ValueError): compare_saved_account(corrupt, expected)
        corrupt = copy.deepcopy(expected); corrupt["first_crossing"] = -.5
        with self.assertRaises(ValueError): compare_saved_account(corrupt, expected)
        corrupt = copy.deepcopy(expected); corrupt["trades"] = corrupt["trades"].astype(int)
        with self.assertRaises(ValueError): compare_saved_account(corrupt, expected)

    def test_training_cap_uses_B_and_whole_original_n(self):
        rows = training_records(); rows[0]["breach_count"] = 7; rows[1]["breach_count"] = 8; rows[2]["breach_count"] = 9
        receipt = independent_training_choice(rows, 96, 7)
        self.assertEqual(receipt["effective_cap"], 7)
        self.assertEqual(receipt["eligible_candidate_orders"], [0])
        receipt = independent_training_choice(rows, 96, 12)
        self.assertEqual(receipt["effective_cap"], 9)
        self.assertEqual(receipt["eligible_candidate_orders"], [0, 1, 2])

    def test_annual_architecture_cannot_borrow_a_higher_growth_band_control(self):
        rows = training_records(); rows[2]["mean_log_growth"] = .1
        self.assertEqual(independent_training_choice(rows, 96, 7, candidate_orders=(0,))["selected_candidate_order"], 0)
        self.assertEqual(independent_training_choice(rows, 96, 7)["selected_candidate_order"], 2)
        with self.assertRaises(ValueError): independent_training_choice(rows, 96, 7, candidate_orders=(1, 2))

    def test_growth_ties_use_one_global_band_not_pairwise_tolerance_chaining(self):
        rows = training_records()
        for row, growth, excess in zip(rows, [.05, .05 + .75e-12, .05 + 1.5e-12], [0., .001, .002]):
            row.update(mean_log_growth=growth, mean_excess=excess)
        receipt = independent_training_choice(rows, 96, 7)
        self.assertEqual(receipt["selected_candidate_order"], 1)

    def test_secondary_growth_band_keys_are_exact_excess_then_fee_then_order(self):
        rows = training_records()
        rows[0].update(mean_excess=0., mean_nominal_fees=.8)
        rows[1].update(mean_excess=1e-14, mean_nominal_fees=0.)
        self.assertEqual(independent_training_choice(rows, 96, 7)["selected_candidate_order"], 0)
        rows[1]["mean_excess"] = 0.
        self.assertEqual(independent_training_choice(rows, 96, 7)["selected_candidate_order"], 1)
        rows[2].update(mean_excess=0., mean_nominal_fees=0.)
        self.assertEqual(independent_training_choice(rows, 96, 7)["selected_candidate_order"], 1)

    def test_no_feasible_candidate_and_malformed_family_keep_unavailable_selection(self):
        rows = training_records()
        receipt = independent_training_choice(rows, 96, 0)
        self.assertIsNone(receipt["selected_candidate_order"])
        self.assertIsNone(receipt["selected_delta"])
        self.assertIsNone(receipt["maximum_eligible_mean_log_growth"])
        bad = copy.deepcopy(rows); bad[1]["breach_count"] = 1.5
        with self.assertRaises(ValueError): independent_training_choice(bad, 96, 7)
        with self.assertRaises(ValueError): independent_training_choice(rows[::-1], 96, 7)

    def test_four_inner_gate_uses_B_on_the_identical_automatic_ID_subset(self):
        rows = inner_records()
        receipt = independent_four_inner_receipt(rows, list(range(1, 97)))
        self.assertTrue(receipt["inner_gate_eligible"])
        self.assertEqual((receipt["automatic_n"], receipt["breach_count"], receipt["reference_breach_count_same_automatic_ids"]), (96, 8, 8))
        rows[3].update(automated=False, breach_count=None, reference_breach_count=None, mean_log_growth=None)
        receipt = independent_four_inner_receipt(rows, list(range(1, 97)))
        self.assertEqual((receipt["automatic_n"], receipt["review_n"]), (72, 24))
        self.assertFalse(receipt["all_four_automated"])
        self.assertTrue(receipt["inner_gate_eligible"])
        rows[0]["reference_breach_count"] = 1
        receipt = independent_four_inner_receipt(rows, list(range(1, 97)))
        self.assertTrue(receipt["empirical_risk_eligible"])
        self.assertFalse(receipt["paired_B_dominance_eligible"])
        self.assertFalse(receipt["inner_gate_eligible"])

    def test_inner_zero_auto_NULLs_and_partition_overlap_are_not_filled(self):
        rows = inner_records()
        for row in rows: row.update(automated=False, breach_count=None, reference_breach_count=None, mean_log_growth=None)
        receipt = independent_four_inner_receipt(rows, list(range(1, 97)))
        self.assertFalse(receipt["inner_gate_eligible"])
        self.assertIsNone(receipt["breach_count"])
        self.assertIsNone(receipt["conditional_mean_log_growth"])
        rows[0]["breach_count"] = 0
        with self.assertRaises(ValueError): independent_four_inner_receipt(rows, list(range(1, 97)))
        rows = inner_records(); rows[1]["evaluation_ids"][0] = 1
        with self.assertRaises(ValueError): independent_four_inner_receipt(rows, list(range(1, 97)))

    def test_original_ID_membership_rejects_overlap_and_final_ID(self):
        self.assertEqual(validate_original_membership(dict(training_ids=[1, 2], evaluation_ids=[3])), dict(training_n=2, evaluation_n=1))
        for job in [dict(training_ids=[1, 2], evaluation_ids=[2]), dict(training_ids=[1, 2], evaluation_ids=[121]),
                    dict(training_ids=[1, True], evaluation_ids=[3])]:
            with self.subTest(job=job), self.assertRaises(ValueError): validate_original_membership(job)

    def test_complete_nested_membership_retains_125_jobs_and_500_static_folds(self):
        result = validate_protocol_membership(synthetic_protocol_membership())
        self.assertEqual(result["dynamic_training_jobs"], 125)
        self.assertEqual(result["static_inner_folds"], 500)
        self.assertEqual(result["repeated_outer_original_ID_records"], 600)

    def test_nested_membership_rejects_outer_leakage_and_wrong_Q1_cache_job(self):
        p = synthetic_protocol_membership(); jid = p["jobs"][0]["job_id"]
        p["static_contexts"][jid]["inner_folds"][0]["train_ids"][0] = p["jobs"][0]["evaluation_ids"][0]
        with self.assertRaises(ValueError): validate_protocol_membership(p)
        p = synthetic_protocol_membership(); p["static_contexts"][jid]["name"] = "another_job"
        with self.assertRaises(ValueError): validate_protocol_membership(p)


class SimpleExecutionFullAuditSyntheticTests(unittest.TestCase):
    """The tested simulator supplies fixtures only, never audit implementation."""
    @classmethod
    def setUpClass(cls):
        from test_simple_execution_pipeline import SimpleExecutionPipelineTests as Fixture
        import run_simple_execution_comparison as runner
        cls.protocol, cls.bundles, cls.returns, cls.inflation, cls.ids = Fixture.fixture()
        simulator, _ = Fixture.engine()
        result = runner.run_pipeline(cls.protocol, cls.bundles, cls.returns, cls.inflation, cls.ids, simulator=simulator)
        cls.result = result
        cls.tables = {name: audit_module.observed_bools(audit_module.pd.DataFrame(rows)) for name, rows in result["tables"].items()}
        # Deliberately fake grid/cache for this closed synthetic account; only
        # index/target/first-event identities are needed by the component audit.
        allowed = list(range(5, 22)) + [23] + list(range(24, 37)) + [57] + list(range(58, 198))
        assert len(allowed) == 172 and allowed[8] == 13 and allowed[17] == 23 and allowed[31] == 57
        weights = np.tile([1., 0., 0.], (232, 1)); weights[allowed] = [.3, .3, .4]
        weights[13], weights[23], weights[57] = [.2, .3, .5], [.4, .3, .3], [.7, .2, .1]
        cls.cache = {"weights": weights, "H30__mdd": np.zeros((120, 232))}
        digest = hashlib.sha256()
        for value in (cls.ids, cls.returns, cls.inflation): digest.update(np.ascontiguousarray(value).tobytes())
        cls.digest = digest.hexdigest()

    def audit(self, *, tables=None, choices=None, bundles=None):
        return audit_module.audit_pipeline(self.protocol, self.bundles if bundles is None else bundles,
            self.returns, self.inflation, self.ids, self.tables if tables is None else tables,
            self.result["training_choices"] if choices is None else choices, self.result["context_selections"],
            self.result["trace_groups"], {"actual": self.result["actual_traces"], "validation": self.result["validation_traces"]},
            self.cache, data_digest=self.digest)

    def test_full125_account_selection_gate_and_denominator_audit(self):
        result = self.audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["checked_training_metric_rows"], 28800)
        self.assertEqual(result["checked_inner_method_rows"], 4800)
        self.assertEqual(result["checked_outer_all_policy_rows"], 2400)
        self.assertEqual(result["unique_scalar_physical_accounts_replayed"], 480)

    def test_training_financial_and_Q1_semantic_tampering_fail_independently(self):
        tables = dict(self.tables); tables["training_paths"] = tables["training_paths"].copy()
        tables["training_paths"].loc[0, "log_growth"] += .01
        with self.assertRaises(ValueError): self.audit(tables=tables)
        bundles = copy.deepcopy(self.bundles); jid = self.protocol["jobs"][0]["job_id"]
        bundles[jid]["q2_case"]["product"]["weights"] = [.45, .25, .3]
        with self.assertRaises(ValueError): self.audit(bundles=bundles)

    def test_final_gate_choice_tampering_cannot_relabel_outer_service(self):
        choices = copy.deepcopy(self.result["training_choices"])
        jid = next(j["job_id"] for j in self.protocol["jobs"] if j["stage"] == "refit")
        choices[jid]["methods"]["annual_gated_diagnostic"]["execution_status"] = "execution_review"
        with self.assertRaises(ValueError): self.audit(choices=choices)


class SimpleExecutionAuditIOGuardTests(unittest.TestCase):
    def test_completed_audit_guard_precedes_any_source_or_market_read(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); (out / "independent_audit.json").write_text("{}")
            with patch.object(audit_module, "OUT", out), patch.object(audit_module, "read_json", side_effect=AssertionError("No input read allowed")):
                with self.assertRaises(RuntimeError): audit_module.main()
                with self.assertRaises(RuntimeError): audit_module.prepare_audit_plan()

    def test_plan_binds_source_and_protocol_before_numerical_audit_without_market(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / "results"; protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"execution_ready": True}))
            with patch.object(audit_module, "OUT", out), patch.object(audit_module, "PROTOCOL", protocol), patch.object(audit_module, "AUDIT_PLAN", root / "audit_plan.json"):
                path = audit_module.prepare_audit_plan(); before = path.read_bytes()
                plan = json.loads(before)
                self.assertFalse(plan["allow_final"])
                self.assertEqual(plan["intended_scope"]["training_metric_rows"], 28800)
                self.assertEqual(audit_module.prepare_audit_plan().read_bytes(), before)
                out.mkdir(); (out / "audit_failure_receipt.json").write_text("{}")
                with self.assertRaises(RuntimeError): audit_module.main()
                with self.assertRaises(RuntimeError): audit_module.prepare_audit_plan()


if __name__ == "__main__": unittest.main()
