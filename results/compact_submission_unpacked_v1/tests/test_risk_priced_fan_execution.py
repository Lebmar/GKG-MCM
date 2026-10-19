"""Training selection and evaluation boundaries, using synthetic arrays only."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research"), str(ROOT / "src")]
import run_risk_priced_fan_execution as runner


def candidates(n=96, counts=None, growth=None):
    counts = counts if counts is not None else [7] * 9
    growth = growth if growth is not None else [.03] * 9
    return [{**runner.candidate_meta(i, 30), "n_training": n, "breach_count": counts[i], "mean_log_growth": growth[i]} for i in range(9)]


class TrainingPriceSelectionTests(unittest.TestCase):
    def test_integer_cap_preserves_reference_and_alpha_caps(self):
        value = runner.select_training_candidate(candidates(counts=[7, 8, 9, 0, 6, 7, 8, 9, 10]), 7, 96)
        self.assertEqual(value["absolute_alpha_count_cap"], 9)
        self.assertEqual(value["training_breach_cap"], 7)
        self.assertEqual([x["training_eligible"] for x in value["candidate_summaries"]], [True, False, False, True, True, True, False, False, False])
        self.assertEqual(runner.select_training_candidate(candidates(n=120, counts=[12] * 9), 15, 120)["training_breach_cap"], 12)

    def test_global_growth_band_does_not_accumulate_pairwise_tolerance(self):
        t = 1e-12
        growth = [1 - 1.5 * t, 1 - .75 * t, 1.] + [0.] * 6
        value = runner.select_training_candidate(candidates(growth=growth), 7, 96)
        self.assertEqual(value["selected_candidate_order"], 1)
        self.assertEqual(value["maximum_eligible_mean_log_growth"], 1.)

    def test_no_eligible_is_execution_review_without_baseline_fallback(self):
        value = runner.select_training_candidate(candidates(counts=[8] * 9), 7, 96)
        self.assertIsNone(value["selected_candidate_order"])
        self.assertEqual(value["execution_status"], "execution_review")
        self.assertEqual(value["execution_review_reason"], "execution_family_no_admissible_policy")

    def test_risk_first_has_separate_identity_without_infinite_json_price(self):
        growth = [.01] * 8 + [.04]
        value = runner.select_training_candidate(candidates(growth=growth), 7, 96)
        self.assertEqual(value["selected_candidate_order"], 8)
        meta = runner.candidate_meta(8, 30)
        self.assertEqual(meta["price_kind"], "risk_first")
        self.assertIsNone(meta["annual_price"])
        self.assertAlmostEqual(runner.candidate_meta(4, 30)["original_H_total_price"], .9)

    def test_all_nine_candidates_and_correct_denominator_are_required(self):
        with self.assertRaises(ValueError): runner.select_training_candidate(candidates()[:-1], 7, 96)
        wrong = candidates(); wrong[3]["n_training"] = 24
        with self.assertRaises(ValueError): runner.select_training_candidate(wrong, 7, 96)


class PriceAxisAndReviewTests(unittest.TestCase):
    def test_frozen_numeric_kernel_integrates_with_price_adapter_and_gate_records(self):
        from experimental_execution.risk_priced_fan_rollout import simulate_prices
        from experimental_execution.fan_rollout import simulate_policy
        r = np.array([[[.06, .01, .003], [-.04, .02, .003], [.03, 0., .003]],
                      [[-.05, .02, .004], [.08, -.01, .004], [.02, .01, .004]],
                      [[.01, 0., .002], [.02, .01, .002], [-.02, .02, .002]],
                      [[.04, -.01, .003], [-.01, .02, .003], [.05, .01, .003]]])
        inflation = np.full((4, 3), .01); target = np.array([.3, .2, .5])
        family = simulate_prices(r, inflation, r, inflation, target, runner.PRICES, L=.2, horizon=3)
        old = simulate_policy(r, inflation, r, inflation, target, policy="growth_only_rollout", L=.2)
        runner.verify_price_zero(runner.extract_price(family, 0), old)
        summaries = []
        for order in range(9):
            arm = runner.extract_price(family, order)
            common = {"horizon": 3, "loss_tolerance": .2}
            rows = runner.financial_rows(arm, [1, 2, 3, 4], common)
            summaries.append(runner.training_summary(rows, order, 3))
            self.assertEqual(len(runner.gate_records(arm, [1, 2, 3, 4], common, order)), 12)
        selected = runner.select_training_candidate(summaries, 0, 4)
        self.assertIsNotNone(selected["selected_candidate_order"])

    def test_cpi_is_not_price_sliced_even_when_Hplus1_equals_price_count(self):
        simulation = {"real_wealth": np.arange(18.).reshape(2, 3, 3), "fees": np.arange(12.).reshape(2, 3, 2),
                      "holdings": np.arange(54.).reshape(2, 3, 3, 3), "cpi": np.arange(6.).reshape(2, 3),
                      "annual_prices": np.arange(3.), "targets": np.array([.2, .3, .5]), "loss_tolerances": np.array([.4])}
        arm = runner.extract_price(simulation, 1)
        self.assertEqual(arm["real_wealth"].shape, (2, 3))
        self.assertEqual(arm["holdings"].shape, (2, 3, 3))
        np.testing.assert_array_equal(arm["cpi"], simulation["cpi"])
        np.testing.assert_array_equal(arm["fees"], simulation["fees"][:, 1])

    def test_execution_review_keeps_original_risk_product_but_null_financials(self):
        common = {"risk_assignment": "Balanced", "risk_assigned": True, "assignment": "Review", "selected_execution_status": "execution_review",
                  "candidate_local_index": 117, "annual_price": None}
        rows = runner.null_rows([1, 2], common)
        self.assertEqual(rows[0]["risk_assignment"], "Balanced")
        self.assertTrue(rows[0]["risk_assigned"])
        self.assertFalse(rows[0]["assigned"])
        self.assertTrue(all(rows[0][key] is None for key in runner.FINANCIAL))

    def test_outer_evaluation_calls_only_the_preselected_single_price(self):
        calls = []
        def simulate(actual, inflation, training, train_inf, target, prices, **kwargs):
            calls.append((actual.copy(), training.copy(), prices.copy()))
            return {"real_wealth": np.ones((len(actual), 1, 3)) * 100, "cpi": np.ones((len(actual), 3)), "annual_prices": prices}
        actual = np.zeros((2, 2, 3)); training = np.zeros((4, 2, 3))
        self.assertIsNone(runner.evaluate_selected(simulate, actual, np.zeros((2, 2)), training, np.zeros((4, 2)), [.2, .3, .5], None, L=.4, horizon=2))
        self.assertEqual(len(calls), 0)
        runner.evaluate_selected(simulate, actual, np.zeros((2, 2)), training, np.zeros((4, 2)), [.2, .3, .5], 3, L=.4, horizon=2)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][2], [.01])
        # A changed held-out future cannot cause this evaluator to search or
        # alter the already selected training order.
        runner.evaluate_selected(simulate, actual + 7, np.zeros((2, 2)), training, np.zeros((4, 2)), [.2, .3, .5], 3, L=.4, horizon=2)
        np.testing.assert_array_equal(calls[1][2], calls[0][2])


class GateReportingTests(unittest.TestCase):
    def arm(self, *, restore=False, growth=.01, kh=0, kr=1, score=-.005):
        H = 5
        arm = {"actions": np.array([["restore", "restore" if restore else "hold", "hold", "hold", "hold"]]),
               "pending_at_decision": np.array([[False, False, True, True, True]]), "trades": np.array([[True, restore, False, False, False]]),
               "fees": np.zeros((1, H)), "moved": np.zeros((1, H)), "gate_reason": np.full((1, H), "synthetic", dtype="U20")}
        for key in ["qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "forecast_n", "predicted_score"]: arm[key] = np.full((1, H), np.nan)
        arm["qg_hold"][0, 1] = 0.; arm["qg_restore"][0, 1] = growth
        arm["qbad_hold_count"][0, 1] = kh; arm["qbad_restore_count"][0, 1] = kr
        arm["forecast_n"][0, 1] = 10; arm["predicted_score"][0, 1] = score
        return arm

    def test_gate_score_uses_original_H_and_single_terminal_probability(self):
        rows = runner.gate_records(self.arm(), [1], {"horizon": 5}, 4)
        self.assertAlmostEqual(rows[1]["computed_score_advantage"], -.005)
        self.assertEqual(rows[1]["action"], "hold")
        self.assertIsNone(rows[0]["qg_hold"])
        self.assertIsNone(rows[2]["qg_hold"])

    def test_risk_first_can_label_unpriced_growth_sacrifice_without_inf_score(self):
        rows = runner.gate_records(self.arm(restore=True, growth=-.01, kh=2, kr=1, score=np.nan), [1], {"horizon": 5}, 8)
        self.assertIsNone(rows[1]["computed_score_advantage"])
        self.assertTrue(rows[1]["restore_sacrifices_unpriced_growth"])
        self.assertEqual(rows[1]["runner_gate_reason"], "risk_first_lower_bad")

    def test_common_auto_pairing_does_not_impute_review_returns(self):
        common = {"scope": "outer_oof", "context": "outer_fold_1", "outer_fold": 1, "client": "C", "risk_assignment": "Balanced", "risk_assigned": True}
        rows = [*runner.null_rows([1], {**common, "policy": runner.LEARNER}), *runner.null_rows([1], {**common, "policy": "annual_then_pending"})]
        paired, summary = runner.paired_outputs(pd.DataFrame(rows))
        self.assertFalse(paired.common_auto.iloc[0]); self.assertEqual(summary.common_auto_n.iloc[0], 0)
        self.assertIsNone(paired.delta_log_growth.iloc[0])


if __name__ == "__main__": unittest.main()
