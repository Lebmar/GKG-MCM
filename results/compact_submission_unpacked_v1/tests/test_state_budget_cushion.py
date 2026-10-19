"""Constructed V3 mechanism/boundary checks; no supplied market data."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from safeinvest.core import execute_target
from experimental_execution import state_budget_execution as v2
from experimental_execution import state_budget_mitigation as v2b
from experimental_execution import state_budget_cushion_primary as primary
from experimental_execution import state_budget_cushion as v3


def state(*, holdings=(45., 5., 50.), peak=100., mdd=0., support=None,
          forecast=None, weights=None):
    support = np.ones((3, 3)) if support is None else np.asarray(support, float)
    forecast = support if forecast is None else np.asarray(forecast, float)
    weights = np.full((1, len(forecast)), 1 / len(forecast)) if weights is None else weights
    return (np.asarray(holdings, float), 1., peak, mdd, support, forecast, weights)


def decision(module, args=None, **kwargs):
    params = dict(target=[.45, .05, .5], L=.4, release_fraction=1.)
    params.update(kwargs)
    return module.decide_state_action(*(state() if args is None else args), **params)


def simulate(module, actual=None, training=None, **kwargs):
    actual = np.ones((2, 3, 3)) if actual is None else np.asarray(actual, float)
    training = np.ones((3, 3, 3)) if training is None else np.asarray(training, float)
    params = dict(target=[.45, .05, .5], L=.4, horizon=actual.shape[1],
                  training_ids=[1, 2, 3], release_fraction=1.)
    params.update(kwargs)
    return module.simulate_state_budget(actual - 1, np.zeros(actual.shape[:2]),
        training - 1, np.zeros(training.shape[:2]), **params)


class StateBudgetCushionTests(unittest.TestCase):
    def assert_shared_equal(self, actual, expected):
        for key, value in expected.items():
            np.testing.assert_equal(actual[key], value, err_msg=key)

    def test_formula_matches_drawdown_form_and_scale_invariance(self):
        for loss in (.1, .4, 1.):
            for dd in (0., loss / 4, loss / 2, min(loss, .99)):
                real = 100 * (1 - dd)
                expected = np.clip((loss - dd) / (loss * (1 - dd)), 0., 1.)
                result = primary.cushion_scale(real, 100., loss)
                self.assertAlmostEqual(result, expected, places=14)
                self.assertAlmostEqual(primary.cushion_scale(real * 17, 1700., loss), result, places=14)

    def test_scale_monotone_in_current_drawdown_and_boundary_clips(self):
        scales = [primary.cushion_scale(100 * (1 - d), 100., .4) for d in (0, .1, .25, .4, .5)]
        self.assertEqual(scales[0], 1.); self.assertEqual(scales[-1], 0.)
        self.assertTrue(np.all(np.diff(scales) <= 0))
        self.assertEqual(primary.cushion_scale(60., 100., .4), 0.)
        self.assertEqual(primary.cushion_scale(50., 100., .4), 0.)

    def test_zero_loss_boundary_is_explicit_without_division(self):
        self.assertEqual(primary.cushion_scale(100., 100., 0.), 1.)
        self.assertEqual(primary.cushion_scale(100 - 1e-11, 100., 0.), 0.)
        self.assertEqual(primary.cushion_scale(50., 100., 1.), 1.)
        for args in ((0, 100, .4), (100, 0, .4), (101, 100, .4), (100, 100, -1), (100, 100, np.nan)):
            with self.assertRaises(ValueError): primary.cushion_scale(*args)

    def test_at_peak_all_shared_primary_fields_match_each_original_family(self):
        support = np.array([[.7, .9, 1.01], [1.4, 1.02, 1.], [1.1, 1.2, .99]])
        weights = np.vstack((np.full(3, 1 / 3), [.6, .3, .1], np.eye(3)))
        args = state(support=support, weights=weights)
        for rho in (.5, .75, 1.):
            for rule in ("uniform", "ensemble_min"):
                old = decision(v2, args, release_fraction=rho, forecast_rule=rule)
                new = decision(primary, args, release_fraction=rho, forecast_rule=rule)
                self.assert_shared_equal(new, old)
                self.assertEqual(new["cushion_scale"], 1.)

    def test_initial_mandatory_original_target_has_exact_shared_witness(self):
        args = state(holdings=(0, 0, 100))
        old = decision(v2b, args, initial=True)
        new = decision(v3, args, initial=True)
        self.assert_shared_equal(new, old)
        self.assertEqual(new["selected_index"], 1)
        self.assertEqual(new["cushion_scale"], 1.)
        np.testing.assert_array_equal(new["proposal_target_weights"][0], [.45, .05, .5])

    def test_lower_sleeve_can_supply_safe_point_absent_from_original_family(self):
        # A constructed mechanism witness, not an observation from supplied markets.
        support = np.array([[.2, .4, .95], [1.2, 1.03, 1.01], [1.2, 1.03, 1.01]])
        args = state(peak=150., mdd=1 - 100 / 150, support=support)
        old = decision(v2, args); new = decision(v3, args)
        self.assertEqual(old["status"], "no_feasible_action")
        self.assertEqual(new["status"], "selected")
        self.assertTrue(new["support_feasible"]); self.assertFalse(new["mitigation_evaluated"])
        self.assertEqual(new["cushion_scale"], .25)
        self.assertGreaterEqual((support @ new["posttrade_holdings"]).min(), new["budget_floor"] - 1e-10)
        self.assertLess(new["proposal_target_weights"][3, 0], .25 * .45)

    def test_primary_partial_points_keep_exact_fee_and_floor_geometry(self):
        support = np.array([[.2, .4, .95], [1.2, 1.03, 1.01], [1.2, 1.03, 1.01]])
        args = state(peak=150., mdd=1 - 100 / 150, support=support)
        result = decision(primary, args); x = args[0]
        for j, proposal in enumerate(result["proposal_target_weights"]):
            endpoint, fee, moved = execute_target(x, proposal, .0025)
            for k in range(1 + 3 * j, 4 + 3 * j):
                q = result["candidate_lambda"][k]
                if not np.isfinite(q): continue
                y = result["candidate_posttrade"][k]
                np.testing.assert_allclose(y, x + q * (endpoint - x), atol=2e-14, rtol=0)
                self.assertAlmostEqual(result["candidate_fees"][k], q * fee, places=13)
                self.assertAlmostEqual(result["candidate_moved"][k], q * moved, places=12)
                self.assertAlmostEqual(y.sum() + q * fee, x.sum(), places=12)
                self.assertAlmostEqual(q * fee, .0025 * np.maximum(y - x, 0).sum(), places=13)
                if result["candidate_feasible"][k]:
                    self.assertGreaterEqual((support @ y).min(), result["budget_floor"] - 1e-10)

    def test_hold_is_not_capped_by_proposal_exposure(self):
        args = state(holdings=(70, 0, 30), peak=120., mdd=.2)
        result = decision(primary, args)
        self.assertEqual(result["selected_index"], 0)
        self.assertGreater(args[0][0] / args[0].sum(), result["proposal_target_weights"][:, 0].max())
        np.testing.assert_array_equal(result["posttrade_holdings"], args[0])
        self.assertEqual(result["fee"], 0.)

    def test_replaced_menu_can_forfeit_previous_higher_growth_points(self):
        support = np.tile([1.5, 1.2, 1.01], (3, 1))
        args = state(holdings=(0, 0, 100), peak=150., mdd=1 - 100 / 150, support=support)
        old = decision(v2, args); new = decision(primary, args)
        self.assertTrue(old["support_feasible"]); self.assertTrue(new["support_feasible"])
        self.assertGreater(old["pred_log_growth_selected"], new["pred_log_growth_selected"])
        self.assertGreater(old["posttrade_holdings"][0], new["posttrade_holdings"][0])

    def test_negative_real_bill_prevents_universal_cash_safety_claim(self):
        args = state(holdings=(40, 0, 50), peak=150., mdd=.4,
                     support=np.tile([.2, .4, .95], (3, 1)))
        result = decision(v3, args)
        self.assertEqual(result["cushion_scale"], 0.)
        np.testing.assert_array_equal(result["proposal_target_weights"], np.tile([0, 0, 1], (4, 1)))
        self.assertEqual(result["selected_index"], -1)
        self.assertFalse(result["support_feasible"]); self.assertTrue(result["guard_triggered"])
        self.assertTrue(result["mitigation_evaluated"])
        self.assertGreater(result["worst_real_shortfall_selected"], 0.)

    def test_clamped_original_floor_retained_for_tiny_negative_headroom(self):
        real = 60. - 2e-11
        args = state(holdings=(0, 0, real), peak=100., mdd=1 - real / 100)
        result = decision(primary, args, release_fraction=.5)
        self.assertEqual(result["cushion_scale"], 0.)
        self.assertEqual(result["budget_floor"], 60.)
        self.assertEqual(result["floor"], 60.)

    def test_mitigation_is_frozen_function_and_original_unscaled_nine_points(self):
        self.assertIs(v3.mitigate_support_failure, v2b.mitigate_support_failure)
        args = state(holdings=(40, 0, 50), peak=150., mdd=.4,
                     support=np.tile([.2, .4, .95], (3, 1)))
        new = decision(v3, args)
        old = v2b.mitigate_support_failure(args[0], args[1], args[4], target=[.45, .05, .5],
            budget_floor=new["budget_floor"], forecast_real_gross=args[5], forecast_weights=args[6],
            real_peak=args[2], running_mdd=args[3], L=.4)
        for key, value in old.items():
            if key.startswith("candidate_"):
                np.testing.assert_equal(new["mitigation_" + key], value, err_msg=key)
        self.assertEqual(new["mitigation_selected_index"], old["selected_index"])
        self.assertEqual(new["fee"], old["fee"])
        self.assertEqual(new["actual_decision_source"], "mitigation")
        self.assertEqual(new["selected_index"], -1)

    def test_initial_review_does_not_use_invalid_actual_market_or_fallback(self):
        training = np.tile([.2, .4, .6], (3, 3, 1))
        actual = np.full((2, 3, 3), np.nan)
        old = simulate(v2b, actual, training); new = simulate(v3, actual, training)
        self.assertEqual(new["status"], "preinvestment_review"); self.assertIsNone(new["arm"])
        self.assert_shared_equal(new["initial_witness"], old["initial_witness"])
        self.assertFalse(new["initial_witness"]["mitigation_evaluated"])
        self.assertEqual(new["preinvestment_review_n"], 2)

    def test_no_fee_flat_at_peak_paths_match_all_previous_financial_fields(self):
        before = v2b.simulate_state_budget
        old = simulate(v2b, cost_rate=0); new = simulate(v3, cost_rate=0)
        self.assertIs(v2b.simulate_state_budget, before)
        self.assert_shared_equal(new["arm"], old["arm"])
        np.testing.assert_array_equal(new["arm"]["cushion_scale"], np.ones((2, 3)))

    def test_one_mitigation_then_absorbing_hold_preserves_original_horizon(self):
        training = np.ones((3, 3, 3)); training[:, 1] = [.2, .4, .6]
        actual = training[:2].copy()
        arm = simulate(v3, actual, training)["arm"]
        np.testing.assert_array_equal(arm["actual_decision_source"][0], ["primary", "mitigation", "pending"])
        np.testing.assert_array_equal(arm["mitigation_evaluated"][0], [False, True, False])
        self.assertEqual(arm["real_wealth"].shape, (2, 4))
        self.assertTrue(np.isnan(arm["cushion_scale"][:, 2]).all())
        self.assertTrue(np.isnan(arm["proposal_target_weights"][:, 2]).all())
        self.assertFalse(arm["automatic_service"][:, 1:].any())
        self.assertTrue((arm["fees"][:, 2] == 0).all())
        self.assertTrue((arm["actions"][:, 2] == "hold").all())

    def test_actual_breach_never_rescued_by_cushion_proposal(self):
        actual = np.ones((2, 3, 3)); actual[:, 0] = .3
        old = simulate(v2b, actual); new = simulate(v3, actual)
        self.assert_shared_equal(new["arm"], old["arm"])
        self.assertFalse(new["arm"]["mitigation_evaluated"].any())
        self.assertTrue(np.isnan(new["arm"]["cushion_scale"][:, 1:]).all())

    def test_current_actual_future_and_ids_do_not_change_action_or_cushion(self):
        training = np.ones((3, 3, 3)); training[:, 1] = [.2, .4, .6]
        first = np.ones((2, 3, 3)); second = first.copy(); second[:, 1:] = [3., 1., 1.]
        a = simulate(v3, first, training, actual_ids=[2, 4])["arm"]
        b = simulate(v3, second, training, actual_ids=[80, 90])["arm"]
        for key in ("cushion_scale", "proposal_target_weights", "candidate_posttrade",
                    "candidate_member_advantage", "mitigation_candidate_posttrade",
                    "mitigation_selected_index", "posttrade_holdings", "fees"):
            np.testing.assert_equal(a[key][:, :2], b[key][:, :2], err_msg=key)

    def test_fixed_calendar_policy_does_not_use_later_training_year(self):
        a = np.ones((3, 3, 3)); b = a.copy(); b[:, 2] = [.1, .3, .8]
        first = simulate(v3, training=a)["arm"]; second = simulate(v3, training=b)["arm"]
        for key in ("cushion_scale", "proposal_target_weights", "candidate_posttrade", "fees"):
            np.testing.assert_equal(first[key][:, :2], second[key][:, :2], err_msg=key)

    def test_new_witnesses_are_saved_readonly_and_fees_charged_once(self):
        arm = simulate(v3)["arm"]
        self.assertEqual(arm["proposal_target_weights"].shape, (2, 3, 4, 3))
        np.testing.assert_allclose(arm["posttrade_holdings"].sum(axis=2) + arm["fees"],
                                   arm["pretrade_wealth"], atol=3e-14, rtol=0)
        for value in arm.values():
            if isinstance(value, np.ndarray): self.assertFalse(value.flags.writeable)


if __name__ == "__main__":
    unittest.main()
