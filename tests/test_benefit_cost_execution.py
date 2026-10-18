"""Synthetic-only benefit/cost and whole-history accounting checks; no I/O."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
from experimental_execution.benefit_cost_execution import (
    BENEFIT_HORIZONS, InitialEconomicRejection, estimate_trade, simulate_benefit_cost)


def scalar_fee(x, w, rate):
    """Independent scalar bisection, not the production executor."""
    v = float(sum(x)); lo, hi = 0., rate * v
    for _ in range(100):
        fee = (lo + hi) / 2
        moved = sum(max((v - fee) * wi - xi, 0.) for xi, wi in zip(x, w))
        if fee > rate * moved:
            hi = fee
        else:
            lo = fee
    fee = (lo + hi) / 2
    return (v - fee) * np.asarray(w), fee


def scalar_branch(x, cp, peak, worst, path, inflation, year, h, loss):
    x = np.array(x, float).copy(); bad = False; logs = []; initial = sum(x) / cp
    # The reference start denominator is supplied separately for paid restore.
    for t in range(year, h):
        x *= 1 + path[t]; cp *= 1 + inflation[t]
        real = float(sum(x)) / cp; peak = max(peak, real)
        worst = max(worst, 1 - real / peak); bad |= worst - loss > 1e-12
        logs.append(np.log(real))
    return np.array(logs), bad, worst, initial


class BenefitCostExecutionTests(unittest.TestCase):
    def training(self, n=3, h=4):
        return np.tile([.12, .03, .01], (n, h, 1)), np.zeros((n, h))

    def quote(self, **changes):
        r, pi = self.training()
        args = dict(holdings=[40., 20., 40.], cpi=1., real_peak=100., running_mdd=0.,
            ever_breached=False, training_returns=r, training_inflation=pi,
            year_index=1, target=[.5, .2, .3], L=.4)
        args.update(changes)
        return estimate_trade(**args)

    def arm(self, **changes):
        r, pi = self.training()
        args = dict(actual_returns=np.zeros((2, 4, 3)), actual_inflation=np.zeros((2, 4)),
            training_returns=r, training_inflation=pi, target=[.5, .2, .3], L=.4)
        args.update(changes)
        return simulate_benefit_cost(**args)

    def test_two_fixed_estimands_only(self):
        self.assertEqual(BENEFIT_HORIZONS, ("one_year", "remaining_to_original_H"))
        for bad in (True, None, "annual", "remaining", 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.quote(benefit_horizon=bad)

    def test_exact_current_fee_and_decomposition_match_direct_paid_counterfactual(self):
        rng = np.random.default_rng(71031)
        r = rng.uniform(-.25, .3, (5, 4, 3)); pi = rng.uniform(-.02, .07, (5, 4))
        x = np.array([[40., 20., 40.], [10., 30., 60.]])
        for mode in BENEFIT_HORIZONS:
            a = self.quote(holdings=x, training_returns=r, training_inflation=pi, benefit_horizon=mode)
            post, fees = zip(*(scalar_fee(row, [.5, .2, .3], .0025) for row in x))
            np.testing.assert_allclose(a['posttrade_holdings'], post, atol=2e-14, rtol=0)
            np.testing.assert_allclose(a['quoted_nominal_fee'], fees, atol=1e-16, rtol=0)
            expected_penalty = -np.log1p(-np.array(fees) / 100)
            np.testing.assert_allclose(a['fee_log_penalty'], expected_penalty, atol=3e-19, rtol=0)
            np.testing.assert_array_equal(a['net_log_gain'], a['gross_log_gain'] - a['fee_log_penalty'][:, None])
            field = 'one_year_log_growth' if mode == 'one_year' else 'total_log_growth'
            direct = a['restore'][field] - a['hold'][field]
            np.testing.assert_allclose(a['net_log_gain'], direct, atol=2e-15, rtol=0)

    def test_full_H_risk_branches_match_independent_scalar_history(self):
        rng = np.random.default_rng(2208)
        r = rng.uniform(-.4, .4, (4, 5, 3)); pi = rng.uniform(-.04, .1, (4, 5))
        x = np.array([[40., 20., 40.], [20., 50., 30.]])
        a = self.quote(holdings=x, cpi=[1., 1.1], real_peak=[110., 100.],
            running_mdd=[.1, .15], training_returns=r, training_inflation=pi,
            horizon=4, year_index=1, L=.25)
        for i in range(2):
            for name in ('hold', 'restore'):
                start = x[i] if name == 'hold' else scalar_fee(x[i], [.5, .2, .3], .0025)[0]
                bads = []
                for j in range(4):
                    logs, bad, worst, _ = scalar_branch(start, [1., 1.1][i], [110., 100.][i],
                        [.1, .15][i], r[j], pi[j], 1, 4, .25)
                    expected = logs - np.log(100 / [1., 1.1][i])
                    self.assertAlmostEqual(a[name]['one_year_log_growth'][i, j], expected[0], places=14)
                    self.assertAlmostEqual(a[name]['total_log_growth'][i, j], expected[-1], places=14)
                    self.assertEqual(a[name]['ever_breached'][i, j], bad)
                    self.assertAlmostEqual(a[name]['running_mdd'][i, j], worst, places=14)
                    bads.append(bad)
                self.assertEqual(a['risk_'+name+'_breach_count'][i], sum(bads))

    def test_final_year_estimands_are_identical_and_keep_original_H(self):
        a = self.quote(year_index=3, benefit_horizon='one_year')
        b = self.quote(year_index=3, benefit_horizon='remaining_to_original_H')
        for key, value in a.items():
            if isinstance(value, np.ndarray):np.testing.assert_array_equal(value, b[key])
        self.assertEqual(a['horizon'], 4); self.assertEqual(a['benefit_periods'], 1)
        self.assertEqual(b['benefit_periods'], 1)

    def test_fee_paid_once_and_not_deducted_twice_from_direct_net(self):
        r = np.array([[[.004, 0., 0.]]])
        a = self.quote(holdings=[0, 0, 100], target=[1, 0, 0], year_index=0,
                       training_returns=r, training_inflation=[[0]], horizon=1)
        expected = np.log(1.004 / 1.0025)
        self.assertAlmostEqual(a['mean_net_log_gain'][0], expected, places=15)
        self.assertTrue(a['economic_pass'][0])
        self.assertLess(expected - a['fee_log_penalty'][0], 0)
        s = self.arm(actual_returns=r, actual_inflation=[[0]], training_returns=r,
                     training_inflation=[[0]], target=[1, 0, 0], horizon=1)
        self.assertAlmostEqual(s['terminal_real'][0], 100 * 1.004 / 1.0025, places=12)
        self.assertAlmostEqual(s['fees'][0, 0], 100 * .0025 / 1.0025, places=13)

    def test_positive_gross_below_cost_is_not_economically_admissible(self):
        r = np.array([[[.001, 0., 0.]]])
        a = self.quote(holdings=[0, 0, 100], target=[1, 0, 0], year_index=0,
                       training_returns=r, training_inflation=[[0]])
        self.assertGreater(a['mean_gross_log_gain'][0], 0)
        self.assertLess(a['mean_net_log_gain'][0], 0)
        self.assertFalse(a['recommend_restore'][0])

    def test_initial_nonpositive_gain_raises_explicit_review_evidence(self):
        r = np.zeros((3, 2, 3))
        for mode in BENEFIT_HORIZONS:
            with self.subTest(mode=mode), self.assertRaises(InitialEconomicRejection) as ctx:
                self.arm(actual_returns=np.zeros((2, 2, 3)), actual_inflation=np.zeros((2, 2)),
                    training_returns=r, training_inflation=np.zeros((3, 2)), benefit_horizon=mode)
            ex = ctx.exception
            self.assertIsInstance(ex, ValueError)
            self.assertEqual(ex.reason, 'mandatory_initial_restore_economic_nonpositive')
            np.testing.assert_array_equal(ex.failed_indices, [0, 1])
            np.testing.assert_array_equal(ex.estimate['economic_pass'], False)
            self.assertEqual(ex.estimate['year_index'], 0)
            self.assertFalse(ex.estimate['mean_net_log_gain'].flags.writeable)

    def test_remaining_holding_can_amortize_current_fee_unlike_one_year(self):
        r = np.tile([.0015, 0., 0.], (2, 2, 1)); pi = np.zeros((2, 2))
        args = dict(actual_returns=r[:1], actual_inflation=pi[:1], training_returns=r,
                    training_inflation=pi, target=[1, 0, 0])
        with self.assertRaises(InitialEconomicRejection):self.arm(**args, benefit_horizon='one_year')
        a = self.arm(**args, benefit_horizon='remaining_to_original_H')
        self.assertGreater(a['mean_net_log_gain'][0, 0], 0)
        np.testing.assert_array_equal(a['actions'][0], ['restore', 'hold'])
        self.assertAlmostEqual(a['terminal_real'][0], 100 * 1.0015**2 / 1.0025, places=12)

    def test_common_full_H_risk_veto_can_reject_positive_economic_gain(self):
        r = np.zeros((10, 1, 3)); r[:9, 0, 0] = .3; r[9, 0, 0] = -.6
        a = self.quote(holdings=[50, 0, 50], target=[.8, 0, .2], year_index=0,
                       training_returns=r, training_inflation=np.zeros((10, 1)), L=.4)
        self.assertGreater(a['mean_net_log_gain'][0], 0)
        self.assertEqual(a['risk_hold_breach_count'][0], 0)
        self.assertEqual(a['risk_restore_breach_count'][0], 1)
        self.assertFalse(a['risk_direction_pass'][0]); self.assertFalse(a['recommend_restore'][0])
        self.assertEqual(a['gate_reason'][0], 'full_H_risk_direction_veto_hold')

    def test_inception_records_estimate_but_does_not_apply_allbill_risk_veto(self):
        r = np.zeros((10, 1, 3)); r[:9, 0, 0] = .3; r[9, 0, 0] = -.6
        a = self.arm(actual_returns=np.zeros((1, 1, 3)), actual_inflation=[[0]],
            training_returns=r, training_inflation=np.zeros((10, 1)), target=[.8, 0, .2])
        self.assertTrue(a['economic_pass'][0, 0]); self.assertFalse(a['risk_direction_pass'][0, 0])
        self.assertTrue(a['estimate_available'][0, 0]); self.assertEqual(a['actions'][0, 0], 'restore')
        self.assertGreater(a['fees'][0, 0], 0)
        self.assertEqual(a['gate_reason'][0, 0], 'mandatory_initial_restore_economic_positive')

    def test_every_optional_restore_has_positive_net_and_risk_direction(self):
        r, pi = self.training(h=6)
        actual = np.zeros((2, 6, 3)); actual[:, ::2, 0] = -.25
        for mode in BENEFIT_HORIZONS:
            a = self.arm(actual_returns=actual, actual_inflation=np.zeros((2, 6)),
                training_returns=r, training_inflation=pi, benefit_horizon=mode, L=1.)
            optional = a['target_execution'].copy(); optional[:, 0] = False
            self.assertTrue(optional.any())
            self.assertTrue((a['mean_net_log_gain'][optional] > 1e-12).all())
            self.assertTrue((a['risk_restore_breach_count'][optional] <= a['risk_hold_breach_count'][optional]).all())
            np.testing.assert_array_equal(a['fees'][optional], a['quoted_nominal_fee'][optional])
            np.testing.assert_allclose(a['posttrade_holdings'].sum(axis=-1) + a['fees'],
                                       a['pretrade_wealth'], atol=2e-14, rtol=0)

    def test_zero_drift_is_zero_fee_zero_gain_and_optional_hold(self):
        a = self.arm()
        np.testing.assert_array_equal(a['actions'][:, 1:], 'hold')
        np.testing.assert_array_equal(a['quoted_nominal_fee'][:, 1:], 0.)
        np.testing.assert_array_equal(a['fees'][:, 1:], 0.)
        np.testing.assert_allclose(a['mean_net_log_gain'][:, 1:], 0., atol=2e-16, rtol=0)
        self.assertTrue((a['trade_count'] == 1).all())

    def test_breach_is_absorbing_with_no_estimates_or_fees_after_trigger(self):
        actual = np.array([[[-.8, 0, 0], [4.1, 0, 0], [0, 0, 0]]])
        r, pi = self.training(h=3)
        a = self.arm(actual_returns=actual, actual_inflation=np.zeros((1, 3)),
            training_returns=r, training_inflation=pi, target=[.75, .1, .15])
        np.testing.assert_array_equal(a['actions'][0], ['restore', 'hold', 'hold'])
        np.testing.assert_array_equal(a['ever_breached'][0], [False, True, True, True])
        self.assertEqual(a['first_crossing'][0], 1)
        self.assertTrue(np.isnan(a['mean_net_log_gain'][0, 1:]).all())
        np.testing.assert_array_equal(a['forecast_n'][0, 1:], -1)
        np.testing.assert_array_equal(a['benefit_periods'][0, 1:], -1)
        np.testing.assert_array_equal(a['estimate_available'][0], [True, False, False])
        np.testing.assert_array_equal(a['fees'][0, 1:], 0)
        self.assertTrue(a['recovered'][0]); self.assertEqual(a['recovery_time'][0], 1)

    def test_current_and_future_actual_returns_cannot_change_current_estimate_or_action(self):
        r, pi = self.training(h=5)
        actual = np.zeros((2, 5, 3)); changed = actual.copy(); changed[:, 2:, :] = [-.7, .4, .1]
        inflation = np.zeros((2, 5)); changed_pi = inflation.copy(); changed_pi[:, 2:] = .4
        a = self.arm(actual_returns=actual, actual_inflation=inflation, training_returns=r, training_inflation=pi)
        b = self.arm(actual_returns=changed, actual_inflation=changed_pi, training_returns=r, training_inflation=pi)
        for key in ('actions', 'posttrade_holdings', 'mean_gross_log_gain', 'mean_net_log_gain',
                    'risk_hold_breach_count', 'risk_restore_breach_count', 'fee_log_penalty'):
            np.testing.assert_array_equal(a[key][:, :3], b[key][:, :3])
        self.assertFalse(np.array_equal(a['terminal_real'], b['terminal_real']))

    def test_inherited_peak_CPI_MDD_and_full_H_risk_survive_later_recovery(self):
        r = np.zeros((1, 2, 3)); pi = np.array([[.25, -.2]])
        a = self.quote(holdings=[0, 0, 100], target=[0, 0, 1], year_index=0,
                       training_returns=r, training_inflation=pi, L=.1)
        self.assertAlmostEqual(a['hold']['total_log_growth'][0, 0], 0, places=14)
        self.assertEqual(a['risk_hold_breach_count'][0], 1)
        self.assertAlmostEqual(a['hold']['running_mdd'][0, 0], .2, places=14)
        old = self.quote(holdings=[0, 0, 100], target=[1, 0, 0], real_peak=200,
            running_mdd=.5, year_index=0, training_returns=np.array([[[-.3, 0, 0]]]),
            training_inflation=[[0]], L=.6, cost_rate=0)
        self.assertEqual(old['risk_restore_breach_count'][0], 1)

    def test_calendar_prefix_and_beyond_original_H_are_not_integrated(self):
        rng = np.random.default_rng(9315)
        r = rng.uniform(-.2, .2, (3, 5, 3)); pi = rng.uniform(0, .05, (3, 5))
        changed = r.copy(); changed[:, [0, 4], :] = .9
        changed_pi = pi.copy(); changed_pi[:, [0, 4]] = .8
        for mode in BENEFIT_HORIZONS:
            a = self.quote(training_returns=r, training_inflation=pi, horizon=4, benefit_horizon=mode)
            b = self.quote(training_returns=changed, training_inflation=changed_pi, horizon=4, benefit_horizon=mode)
            for key in ('gross_log_gain', 'net_log_gain', 'risk_hold_breach_count', 'risk_restore_breach_count'):
                np.testing.assert_array_equal(a[key], b[key])

    def test_gain_scale_and_inflation_invariance_do_not_erase_risk_history(self):
        a = self.quote()
        b = self.quote(holdings=np.array([40., 20., 40.]) * 17, real_peak=1700.)
        np.testing.assert_allclose(a['net_log_gain'], b['net_log_gain'], atol=3e-16, rtol=0)
        np.testing.assert_array_equal(a['risk_restore_breach_count'], b['risk_restore_breach_count'])
        c = self.quote(training_inflation=np.full((3, 4), .5))
        np.testing.assert_array_equal(a['net_log_gain'], c['net_log_gain'])
        self.assertGreater(c['risk_hold_breach_count'][0], a['risk_hold_breach_count'][0])

    def test_metrics_initial_fee_and_last_endpoint_no_terminal_trade(self):
        a = self.arm(horizon=1, L=0.)
        self.assertEqual(a['real_wealth'].shape, (2, 2)); self.assertEqual(a['fees'].shape, (2, 1))
        self.assertTrue((a['first_crossing'] == 1).all()); self.assertTrue((a['available_followup'] == 0).all())
        self.assertTrue((a['mdd'] > 0).all()); self.assertTrue((a['volatility'] == 0).all())
        np.testing.assert_array_equal(a['real_wealth'][:, 0], 100)
        np.testing.assert_array_equal(a['terminal_real'], 100 - a['nominal_fees'])

    def test_readonly_outputs_no_input_mutation_and_arm_helper_compatibility(self):
        r, pi = self.training(h=30); actual = np.zeros((2, 30, 3)); w = np.array([.5, .2, .3])
        a = self.arm(actual_returns=actual, actual_inflation=np.zeros((2, 30)), training_returns=r,
                     training_inflation=pi, target=w)
        for value in a.values():
            if isinstance(value, np.ndarray):self.assertFalse(value.flags.writeable)
        np.testing.assert_array_equal(actual, 0); np.testing.assert_array_equal(w, [.5, .2, .3])
        np.testing.assert_array_equal(r, np.tile([.12, .03, .01], (3, 30, 1)))
        import run_joint_liquidity_refit as joint
        rows = joint.arm_records(a, [1, 2], {'horizon':30, 'loss_tolerance':.4, 'reference_only':False})
        self.assertEqual(len(rows), 2)
        for i, row in enumerate(rows):self.assertAlmostEqual(row['log_growth'], a['log_growth'][i], places=14)

    def test_invalid_or_pending_states_are_rejected_not_reset(self):
        for change in (dict(year_index=True), dict(year_index=-1), dict(year_index=4),
                       dict(ever_breached=True), dict(running_mdd=.5), dict(real_peak=50),
                       dict(cpi=0), dict(horizon=0), dict(cost_rate=1), dict(target=[.5, .2, .2])):
            with self.subTest(change=change), self.assertRaises(ValueError):self.quote(**change)


if __name__ == '__main__':unittest.main()
