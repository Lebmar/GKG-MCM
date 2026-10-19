"""Closed-form checks for saved-trace forensics, without market files."""
import math
import unittest

import numpy as np

from diagnostics.incremental_admission_risk import (
    ConsistencyChecks, distribution_summary, fractional_upper_cvar,
    peak_trough_diagnostic, select_event_witnesses, summarize_saved_path,
)


def known_account():
    # Initial bill-to-equity purchase, a 50% fall, then full recovery.
    real = np.array([[100., 50., 100.]])
    holdings = np.array([[[0., 0., 100.], [50., 0., 0.], [100., 0., 0.]]])
    weights = holdings / real[:, :, None]
    arm = dict(real_wealth=real, nominal_wealth=real.copy(), holdings=holdings, weights=weights,
               real_peak=np.array([[100., 100., 100.]]), current_drawdown=np.array([[0., .5, 0.]]),
               running_mdd=np.array([[0., .5, .5]]), ever_breached=np.array([[False, True, True]]),
               cpi=np.ones((1, 3)), posttrade_holdings=np.array([[[100., 0., 0.], [50., 0., 0.]]]),
               fees=np.zeros((1, 2)), moved=np.array([[100., 0.]]), pretrade_wealth=real[:, :-1],
               pretrade_tv_drift=np.array([[1., 0.]]), end_year_tv_drift=np.zeros((1, 2)),
               trades=np.array([[True, False]]), target_execution=np.array([[True, False]]),
               pending_at_decision=np.array([[False, True]]), actions=np.array([["restore", "hold"]]),
               first_crossing=np.array([1]), targets=np.array([1., 0., 0.]))
    row = dict(log_growth=0., cagr=0., terminal_real=100., mdd=.5, excess=.1, breach=True,
               nominal_fees=0., nominal_moved=100., trade_fraction=.5,
               target_execution_fraction=.5, first_crossing=1., terminal_equity=1., terminal_bond=0., terminal_bill=0.)
    return arm, row


class IncrementalAdmissionRiskTests(unittest.TestCase):
    def test_fractional_tail_has_exactly_ten_percent_mass(self):
        self.assertAlmostEqual(fractional_upper_cvar(np.arange(1., 25.)), 23.25)
        self.assertAlmostEqual(fractional_upper_cvar([.3]), .3)
        self.assertAlmostEqual(fractional_upper_cvar([1., 2., 3.], 1.), 2.)

    def test_tail_rejects_empty_nonfinite_and_invalid_probability(self):
        for values, fraction in [([], .1), ([np.nan], .1), ([[1.]], .1), ([1.], 0), ([1.], True), ([1.], 1.1)]:
            with self.subTest(values=values, fraction=fraction), self.assertRaises(ValueError):
                fractional_upper_cvar(values, fraction)

    def test_peak_recovery_does_not_clear_ever_breached(self):
        d = peak_trough_diagnostic([100., 50., 100.])
        self.assertEqual(d["first_crossing"], 1)
        self.assertEqual(d["recovery_endpoint"], 2)
        self.assertEqual(d["ever_breached"].tolist(), [False, True, True])
        self.assertEqual(d["running_mdd"].tolist(), [0., .5, .5])

    def test_interval_drawdown_uses_later_higher_peak(self):
        d = peak_trough_diagnostic([100., 80., 160., 64., 128.])
        self.assertAlmostEqual(d["mdd"], .6)
        self.assertEqual((d["mdd_peak_year"], d["mdd_trough_year"]), (2, 3))
        self.assertEqual(d["first_crossing"], 3)
        self.assertIsNone(d["recovery_endpoint"])
        self.assertEqual(d["post_crossing_endpoints_below_original_peak_n"], 1)

    def test_boundary_crossing_tolerance_and_last_endpoint(self):
        self.assertIsNone(peak_trough_diagnostic([100., 60.])["first_crossing"])
        d = peak_trough_diagnostic([100., 59.9999999])
        self.assertEqual(d["first_crossing"], 1)
        self.assertEqual(d["post_crossing_endpoints_below_original_peak_n"], 0)
        self.assertIsNone(d["recovery_endpoint"])

    def test_known_account_reconstructs_wealth_drift_and_next_decision_pending(self):
        arm, row = known_account(); checks = ConsistencyChecks()
        path, years = summarize_saved_path(arm, 0, row, policy="joint_tie_set_refit_learner", cost=0, checks=checks)
        self.assertGreater(checks.count, 70)
        self.assertEqual(path["pending_decision_count"], 1)
        self.assertEqual(path["actual_trade_count"], 1)
        self.assertEqual(path["recovery_endpoint"], 2)
        self.assertEqual([x["first_crossing_this_year"] for x in years], [True, False])
        self.assertTrue(years[1]["pending_at_decision"])

    def test_accounting_rejects_money_creation_and_reset_breach(self):
        for name in ("posttrade_holdings", "ever_breached"):
            arm, row = known_account()
            if name == "posttrade_holdings": arm[name][0, 0, 0] += 1
            else: arm[name][0, 2] = False
            with self.subTest(name=name), self.assertRaises(ValueError):
                summarize_saved_path(arm, 0, row, policy="joint_tie_set_refit_learner", cost=0)

    def test_real_drawdown_can_breach_with_constant_nominal_wealth(self):
        arm, row = known_account()
        arm.update(real_wealth=np.array([[100., 50., 50.]]), nominal_wealth=np.full((1, 3), 100.),
                   cpi=np.array([[1., 2., 2.]]), holdings=np.array([[[0., 0., 100.]] * 3]),
                   weights=np.array([[[0., 0., 1.]] * 3]),
                   posttrade_holdings=np.array([[[0., 0., 100.]] * 2]),
                   pretrade_wealth=np.full((1, 2), 100.), pretrade_tv_drift=np.zeros((1, 2)),
                   current_drawdown=np.array([[0., .5, .5]]), moved=np.zeros((1, 2)),
                   trades=np.zeros((1, 2), bool), targets=np.array([0., 0., 1.]))
        row.update(log_growth=math.log(.5) / 2, cagr=math.sqrt(.5) - 1, terminal_real=50.,
                   nominal_moved=0., trade_fraction=0., terminal_equity=0., terminal_bill=1.)
        path, _ = summarize_saved_path(arm, 0, row, policy="annual_then_pending", cost=0)
        self.assertTrue(path["breach"])
        self.assertAlmostEqual(path["mdd"], .5)
        self.assertTrue(path["right_censored_recovery"])

    def test_full_switch_fee_is_once_charged_before_first_return(self):
        arm, row = known_account()
        fee = 100 * .0025 / 1.0025
        scale = 1 - fee / 100
        arm["holdings"][0, 1:] *= scale
        arm["real_wealth"][0, 1:] *= scale
        arm["nominal_wealth"] = arm["real_wealth"].copy()
        arm["posttrade_holdings"] *= scale
        arm["pretrade_wealth"] = arm["real_wealth"][:, :-1]
        arm["fees"][0, 0] = fee
        arm["moved"][0, 0] = 100 - fee
        arm["current_drawdown"][0, 1:] = [1 - .5 * scale, 1 - scale]
        arm["running_mdd"][0, 1:] = 1 - .5 * scale
        row.update(log_growth=math.log(scale) / 2, cagr=math.sqrt(scale) - 1,
                   terminal_real=100 * scale, mdd=1 - .5 * scale, excess=.6 - .5 * scale,
                   nominal_fees=fee, nominal_moved=100 - fee)
        path, annual = summarize_saved_path(arm, 0, row, policy="joint_tie_set_refit_learner")
        self.assertAlmostEqual(path["nominal_fees"], fee)
        self.assertEqual(annual[1]["nominal_fee"], 0.)

    def test_observed_csv_boolean_required_and_no_invented_null_distribution(self):
        arm, row = known_account(); row["breach"] = 1.
        with self.assertRaises(ValueError):
            summarize_saved_path(arm, 0, row, policy="joint_tie_set_refit_learner", cost=0)
        with self.assertRaises(ValueError): distribution_summary([])

    def test_witnesses_include_all_breaches_and_minimum_original_nonbreach(self):
        rows = [dict(scenario_id=s, policy=p, breach=(s == 9 and p == "B"))
                for s in (14, 9, 3) for p in ("learner", "B", "EQ")]
        self.assertEqual(select_event_witnesses(rows), dict(any_arm_breach_ids=[9], minimum_all_arm_nonbreach_id=3, selected_scenario_ids=[3, 9]))
        with self.assertRaises(ValueError): select_event_witnesses(rows + [rows[0]])
        with self.assertRaises(ValueError): select_event_witnesses(rows[:-1])


if __name__ == "__main__":
    unittest.main()
