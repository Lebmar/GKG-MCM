"""Independent policy invariants using synthetic paths, never contest holdouts.

Causality comparisons preserve the contractual horizon: shortening the input
array would also change the legitimately known remaining investment horizon.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from safeinvest.core import execute_target
from safeinvest.policies import forecast_bank, projected_score, simulate, target_family


def deterministic_bank(equity_gross=1.2, years=30, size=7):
    """A frozen positive projection, with no relationship to test returns."""
    gross = np.ones((size, years, 3))
    gross[:, :, 0] = equity_gross
    return np.cumprod(gross, axis=1)


class ForecastTests(unittest.TestCase):
    def test_bank_preserves_joint_assets_inflation_and_contiguous_blocks(self):
        scene = np.arange(3)[:, None]
        period = np.arange(10)[None, :]
        equity = 1.001 + .04 * scene + .001 * period
        real_gross = np.stack([equity, equity**2, equity**3], axis=2)
        inflation = np.broadcast_to(.01 + .001 * period, equity.shape).copy()
        returns = real_gross * (1 + inflation[:, :, None]) - 1
        before_returns, before_inflation = returns.copy(), inflation.copy()
        bank = forecast_bank(returns, inflation, size=9, block_length=5, seed=29)
        increments = np.concatenate([bank[:, :1], bank[:, 1:] / bank[:, :-1]], axis=1)
        self.assertEqual(bank.shape, (9, 30, 3))
        assert_allclose(increments[:, :, 1], increments[:, :, 0]**2, atol=1e-13)
        assert_allclose(increments[:, :, 2], increments[:, :, 0]**3, atol=1e-13)
        for start in range(0, 30, 5):
            assert_allclose(np.diff(increments[:, start:start+5, 0], axis=1), .001, atol=1e-13)
        assert_array_equal(returns, before_returns)
        assert_array_equal(inflation, before_inflation)
        assert_array_equal(bank, forecast_bank(returns, inflation, size=9, block_length=5, seed=29))

    def test_all_actions_use_the_exact_same_frozen_bank(self):
        bank = deterministic_bank()
        before = bank.copy()
        returns, inflation = np.zeros((1, 4, 3)), np.zeros((1, 4))
        with patch('safeinvest.policies.projected_score', wraps=projected_score) as scored:
            simulate(returns, inflation, [.6, .15, .25], mode='dynamic',
                     bank=bank, cost_rate=0, epsilon=0, risk_aversion=0)
        self.assertGreater(len(scored.call_args_list), 3)
        for call in scored.call_args_list:
            self.assertIs(call.args[2], bank)
        assert_array_equal(bank, before)


class CausalityTests(unittest.TestCase):
    def test_future_return_and_inflation_perturbations_preserve_current_decisions(self):
        rng = np.random.default_rng(904)
        returns = rng.uniform(-.08, .15, (3, 8, 3))
        inflation = rng.uniform(0, .035, (3, 8))
        changed_returns, changed_inflation = returns.copy(), inflation.copy()
        cut = 3
        changed_returns[:, cut:, :] = [-.75, .12, .01]
        changed_inflation[:, cut:] = .2
        bank = deterministic_bank()
        for mode in ('annual', 'gate', 'dynamic'):
            with self.subTest(mode=mode):
                kwargs = dict(base_target=[.6, .15, .25], mode=mode,
                              bank=bank, review_threshold=.25)
                original = simulate(returns, inflation, **kwargs)
                perturbed = simulate(changed_returns, changed_inflation, **kwargs)
                # The decision at cut is made before the altered cut return.
                for key in ('weights', 'targets', 'costs', 'moved', 'trades', 'pending', 'benefit'):
                    assert_allclose(original[key][:, :cut+1], perturbed[key][:, :cut+1],
                                    atol=1e-13, err_msg=key)
                assert_allclose(original['wealth'][:, :cut+1], perturbed['wealth'][:, :cut+1])
                self.assertFalse(np.allclose(original['wealth'][:, cut+1],
                                            perturbed['wealth'][:, cut+1]))


class ProjectionTests(unittest.TestCase):
    def test_fee_is_in_log_wealth_once_and_in_year_end_drawdown(self):
        holdings = np.array([[0., 0., 100.]])
        post, cost, _ = execute_target(holdings, [.5, .25, .25])
        bank = np.ones((2, 30, 3))
        ratio = post.sum() / holdings.sum()
        score = projected_score(holdings, post, bank, 4, np.ones(1), np.zeros(1), 0)
        assert_allclose(score, [np.log(ratio)], atol=1e-14)
        risk_score = projected_score(holdings, post, bank, 4, np.ones(1), np.zeros(1), 4)
        assert_allclose(risk_score, [np.log(ratio)-4*float(cost[0])/100], atol=1e-14)
        self.assertLess(float(score[0]), 0.)

    def test_fee_recovered_before_year_end_is_not_an_extra_drawdown_observation(self):
        holdings = np.array([[0., 0., 100.]])
        post, _, _ = execute_target(holdings, [.5, .25, .25])
        # Every projected year end is above the actual pre-trade wealth peak.
        # The temporary fee deduction is not a separate annual observation.
        bank = np.full((3, 4, 3), 1.02)
        ratio = post.sum() / holdings.sum()
        score = projected_score(holdings, post, bank, 4, np.ones(1), np.zeros(1), 4)
        assert_allclose(score, [np.log(1.02*ratio)], atol=1e-14)

    def test_projection_continues_actual_historical_peak(self):
        # Existing history: 100 -> 120 -> 90. Future: 99 -> 69.3.
        holdings = np.array([[30., 30., 30.]])
        bank = np.array([[[1.1]*3, [.77]*3]])
        score = projected_score(holdings, holdings, bank, 2,
                                np.array([120/90]), np.array([.25]), 4)
        assert_allclose(score, [np.log(.77)-4*(1-69.3/120)], atol=1e-14)

    def test_recovery_cannot_erase_a_previously_realized_maximum_drawdown(self):
        holdings = np.array([[20., 30., 50.]])
        bank = np.full((2, 4, 3), 1.3)
        no_penalty = projected_score(holdings, holdings, bank, 4,
                                     np.ones(1), np.array([.4]), 0)
        penalty = projected_score(holdings, holdings, bank, 4,
                                  np.ones(1), np.array([.4]), 4)
        assert_allclose(penalty-no_penalty, [-1.6], atol=1e-14)

    def test_simulator_passes_historical_mdd_after_wealth_recovers(self):
        returns = np.broadcast_to(np.array([.2, -.25, .5, 0.])[None, :, None], (1, 4, 3)).copy()
        with patch('safeinvest.policies.projected_score', wraps=projected_score) as scored:
            simulate(returns, np.zeros((1, 4)), [0, 0, 1], mode='gate',
                     bank=np.ones((2, 30, 3)), epsilon=100.)
        last_period = [call for call in scored.call_args_list if call.args[3] == 1]
        self.assertTrue(last_period)
        for call in last_period:
            assert_allclose(call.args[4], [1.])
            assert_allclose(call.args[5], [.25])


class DecisionTests(unittest.TestCase):
    def test_global_target_bounds_local_step_and_bill_floor(self):
        base = np.array([.6, .15, .25])
        family = target_family(base, dynamic=True)
        self.assertTrue(np.all(family >= 0))
        assert_allclose(family.sum(axis=1), 1.)
        self.assertTrue(np.all(family[:, 2] >= .10))
        self.assertTrue(np.all(np.abs(family[:, 0]-base[0]) <= .10000000001))
        assert_allclose(family[:, 1]/family[:, 2], base[1]/base[2])
        result = simulate(np.zeros((1, 7, 3)), np.zeros((1, 7)), base,
                          mode='dynamic', bank=deterministic_bank(2.),
                          cost_rate=0, epsilon=0, risk_aversion=0)
        targets = result['targets'][0]
        self.assertTrue(np.any(np.abs(targets[1:]-targets[:-1]) > 1e-10))
        self.assertTrue(np.all(np.abs(np.diff(targets[:, 0])) <= .05000000001))
        self.assertTrue(np.all(np.abs(targets[:, 0]-base[0]) <= .10000000001))
        self.assertTrue(np.all(targets[:, 2] >= .10))
        assert_allclose(targets.sum(axis=1), 1.)

    def test_rejected_proposal_preserves_active_target(self):
        base = np.array([.6, .15, .25])
        returns = np.zeros((1, 5, 3)); returns[:, :, 0] = .03
        result = simulate(returns, np.zeros((1, 5)), base, mode='dynamic',
                          bank=deterministic_bank(2.), epsilon=100., risk_aversion=0)
        assert_allclose(result['targets'], np.broadcast_to(base, (1, 5, 3)))
        assert_array_equal(result['trades'][:, 1:], False)
        self.assertTrue(np.any(result['benefit'][:, 1:] > 0))
        self.assertFalse(np.allclose(result['weights'][:, -1], base))

    def test_unprofitable_rebalance_is_rejected_after_cost(self):
        returns = np.zeros((1, 2, 3)); returns[:, 0, 0] = 1.
        bank = deterministic_bank(.99)
        # Two-year prospects justify initial allocation, but the first-year
        # projection alone favors reducing equity after realized weight drift.
        bank[:, 1:, 0] = .99*2**np.arange(1, 30)
        common = dict(returns=returns, inflation=np.zeros((1, 2)),
                      base_target=[.5, 0, .5], mode='gate', bank=bank,
                      epsilon=0, risk_aversion=0)
        free = simulate(**common, cost_rate=0)
        costly = simulate(**common, cost_rate=.02)
        self.assertTrue(free['trades'][0, 1])
        self.assertFalse(costly['trades'][0, 1])
        self.assertEqual(float(costly['costs'][0, 1]), 0.)

    def test_initial_cost_gate_can_reject_an_unfavorable_allocation(self):
        returns = np.zeros((1, 3, 3))
        for mode in ('gate', 'dynamic'):
            with self.subTest(mode=mode):
                result = simulate(returns, np.zeros((1, 3)), [.6, .15, .25],
                                  mode=mode, bank=deterministic_bank(.8),
                                  epsilon=.002, risk_aversion=4)
                self.assertFalse(result['trades'][0, 0])
                self.assertLess(float(result['benefit'][0, 0]), 0.)
                assert_allclose(result['weights'][0, 0], [0, 0, 1])
                self.assertEqual(float(result['costs'][0, 0]), 0.)

    def test_accepted_later_trades_exceed_the_frozen_net_benefit_margin(self):
        returns = np.zeros((1, 6, 3)); returns[:, :, 0] = -.15
        epsilon = .002
        for mode in ('gate', 'dynamic'):
            with self.subTest(mode=mode):
                result = simulate(returns, np.zeros((1, 6)), [.6, .15, .25],
                                  mode=mode, bank=deterministic_bank(), epsilon=epsilon,
                                  risk_aversion=0)
                traded = result['trades'][:, 1:]
                self.assertTrue(np.any(traded))
                self.assertTrue(np.all(result['benefit'][:, 1:][traded] > epsilon))
                assert_allclose(result['costs'], .0025*result['moved'], atol=1e-13)


class ReviewTests(unittest.TestCase):
    def test_pending_preserves_holdings_but_continues_returns_and_scoring(self):
        base = np.array([.6, .15, .25])
        returns = np.array([[[-.6, 0., 0.], [.5, 0., .01],
                             [-.1, .02, .01], [1., .01, .01], [0., 0., 0.]]])
        result = simulate(returns, np.zeros((1, 5)), base, mode='dynamic',
                          bank=deterministic_bank(2.), review_threshold=.25,
                          epsilon=0, risk_aversion=0)
        assert_array_equal(result['pending'], [[False, True, True, True, True]])
        assert_array_equal(result['trades'][:, 1:], False)
        assert_array_equal(result['costs'][:, 1:], 0.)
        initial, _, _ = execute_target([0, 0, 100], base)
        marked = initial * np.cumprod(1+returns[0], axis=0)
        expected_wealth = np.r_[100., marked.sum(axis=1)]
        assert_allclose(result['wealth'][0], expected_wealth)
        self.assertGreater(result['wealth'][0, -1], 100.)
        self.assertEqual(result['pending'].shape[1], 5)
        self.assertTrue(result['pending'][0, -1])
        assert_allclose(result['targets'], np.broadcast_to(base, (1, 5, 3)))

    def test_purchasing_power_loss_can_trigger_review_without_nominal_loss(self):
        returns = np.zeros((1, 3, 3))
        inflation = np.array([[.4, 0., 0.]])
        result = simulate(returns, inflation, [0, 0, 1], mode='gate',
                          bank=np.ones((2, 30, 3)), review_threshold=.25)
        assert_allclose(result['wealth'], 100.)
        assert_array_equal(result['pending'], [[False, True, True]])


if __name__ == '__main__':
    unittest.main()
