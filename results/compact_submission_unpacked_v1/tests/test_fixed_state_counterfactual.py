import math
import unittest

import numpy as np

from experimental_execution.fixed_state_counterfactual import continue_from_state, compare_actions


class FixedStateCounterfactualTests(unittest.TestCase):
    def run_branch(self, action="hold", **changes):
        args = dict(holdings=[50., 30., 20.], cpi=1., peak=100., worst=0.,
                    target=[.3, .3, .4], loss=.4, returns=np.zeros((2, 3)),
                    inflation=np.zeros(2), year_index=0, original_horizon=2,
                    current_action=action, cost_rate=.0025)
        args.update(changes)
        return continue_from_state(**args)

    def test_one_self_financing_fee_and_no_terminal_fee(self):
        hold, restore = self.run_branch(), self.run_branch("restore")
        fee = .0025 * 20 / (1 + .0025 * .4)
        # Hold now and B next year pays this same single fee one year later.
        self.assertAlmostEqual(hold["nominal_fees"], fee, places=13)
        self.assertAlmostEqual(restore["nominal_fees"], fee, places=13)
        self.assertAlmostEqual(restore["total_log_growth"], math.log(1 - fee / 100), places=14)
        self.assertAlmostEqual(restore["fees"][1], 0., places=14)
        self.assertEqual(len(restore["fees"]), 2)
        self.assertAlmostEqual(restore["nominal_fees"], .0025 * restore["nominal_moved"], places=13)

    def test_first_crossing_is_absorbing_and_stops_future_restore(self):
        r = np.array([[0., 0., 0.], [-.6, 0., 0.], [1., 0., 0.]])
        b = self.run_branch(holdings=[100., 0., 0.], target=[.5, .4, .1],
                            returns=r, inflation=np.zeros(3), year_index=1,
                            original_horizon=3, cost_rate=0)
        self.assertEqual(b["actions"].tolist(), ["hold", "hold"])
        self.assertEqual(b["first_crossing_original_year"], 2)
        self.assertTrue(b["breach"])
        self.assertAlmostEqual(b["terminal_real"], 80.)
        self.assertAlmostEqual(b["history_mdd"], .6)

    def test_peak_and_historical_mdd_are_not_restarted(self):
        b = self.run_branch(holdings=[100., 0., 0.], peak=200., worst=.5, loss=.6,
                            target=[1., 0., 0.], returns=np.array([[9., 9., 9.], [.1, 0., 0.]]),
                            year_index=1, cost_rate=0)
        self.assertAlmostEqual(b["terminal_real"], 110.)
        self.assertAlmostEqual(b["history_mdd"], .5)
        self.assertAlmostEqual(b["real_peak"][-1], 200.)
        self.assertFalse(b["breach"])

    def test_joint_inflation_and_original_calendar_suffix(self):
        b = self.run_branch(holdings=[0., 0., 100.], target=[0., 0., 1.],
                            returns=np.zeros((2, 3)), inflation=np.array([9., .25]),
                            year_index=1, cost_rate=0)
        self.assertAlmostEqual(b["terminal_real"], 80.)
        self.assertAlmostEqual(b["history_mdd"], .2)
        self.assertAlmostEqual(b["total_log_growth"], math.log(.8))
        self.assertEqual(len(b["real_wealth"]), 2)

    def test_already_breached_or_inconsistent_state_is_not_reset(self):
        for changes in (dict(ever_breached=True), dict(worst=.41), dict(peak=90.),
                        dict(peak=200., worst=.1), dict(cpi=0.)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_branch(**changes)

    def test_original_horizon_and_market_shapes_must_match(self):
        for changes in (dict(year_index=2), dict(year_index=True), dict(original_horizon=3),
                        dict(returns=np.zeros((1, 3))), dict(inflation=np.array([0., -1.]))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_branch(**changes)

    def test_original_last_year_charges_current_fee_once_and_stops(self):
        b = self.run_branch("restore", holdings=[90., 0., 10.], returns=np.zeros((30, 3)),
                            inflation=np.zeros(30), year_index=29, original_horizon=30)
        fee = .0025 * 60 / (1 + .0025 * .7)
        self.assertEqual(len(b["fees"]), 1)
        self.assertEqual(len(b["real_wealth"]), 2)
        self.assertAlmostEqual(b["nominal_fees"], fee, places=13)
        self.assertAlmostEqual(b["terminal_real"], 100 - fee, places=13)
        self.assertIsNone(b["first_crossing_original_year"])

    def test_growth_score_and_event_have_distinct_units(self):
        p = {"hold": {"log": .2, "bad_fraction": .1},
             "restore": {"log": .21, "bad_fraction": .12}}
        realized = {"hold": {"total_log_growth": .1, "breach": False},
                    "restore": {"total_log_growth": .12, "breach": True}}
        out = compare_actions(p, realized, annual_price=.03, original_horizon=30, chosen_action="hold")
        self.assertAlmostEqual(out["predicted_score"], .01 - .9 * .02)
        self.assertAlmostEqual(out["realized_suffix_score"], .02 - .9)
        self.assertEqual(out["realized_suffix_delta_event"], 1)
        self.assertEqual(out["two_branch_realized_score_shortfall"], 0.)

    def test_score_zero_band_holds_and_better_branch_is_not_future_policy(self):
        p = {"hold": {"log": 0., "bad_fraction": 0.},
             "restore": {"log": 1e-13, "bad_fraction": 0.}}
        r = {"hold": {"total_log_growth": 0., "breach": False},
             "restore": {"total_log_growth": .1, "breach": False}}
        out = compare_actions(p, r, annual_price=0., original_horizon=30, chosen_action="hold")
        self.assertEqual(out["predicted_action"], "hold")
        self.assertFalse(out["score_action_agreement"])
        self.assertAlmostEqual(out["two_branch_realized_score_shortfall"], .1)


if __name__ == "__main__":
    unittest.main()
