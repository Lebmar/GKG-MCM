"""Synthetic financial/coupling tests, independent of any fitted market result."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from experimental_execution.review_pending import NUMERIC_TOL, simulate_paired


def independent_scalar(returns, inflation, target, loss, cost, pending):
    """Own scalar fee bisection and path accounting; never calls the executor."""
    x, cpi, peak, worst, ever, tau = np.array([0., 0., 100.]), 1., 100., 0., False, -1
    result = {key: [] for key in ("holdings", "nominal_wealth", "real_wealth", "real_peak",
                                  "current_drawdown", "running_mdd", "ever_breached", "fees", "moved", "actions")}
    for key, value in (("holdings", x.copy()), ("nominal_wealth", 100.), ("real_wealth", 100.),
                       ("real_peak", 100.), ("current_drawdown", 0.), ("running_mdd", 0.), ("ever_breached", False)):
        result[key].append(value)
    trigger_peak = trigger_drawdown = np.nan
    for year, (draw, infl) in enumerate(zip(returns, inflation)):
        restore = not (pending and tau >= 0)
        if restore:
            wealth = float(sum(x)); low, high = 0., wealth * cost
            for _ in range(100):
                fee = (low + high) / 2
                purchases = sum(max((wealth - fee) * float(w) - float(a), 0.) for a, w in zip(x, target))
                if fee > cost * purchases: high = fee
                else: low = fee
            fee = (low + high) / 2
            post = (wealth - fee) * target
        else:
            fee, post = 0., x.copy()
        moved = sum(max(float(a) - float(b), 0.) for a, b in zip(post, x))
        x = post * (1 + draw)
        nominal = float(sum(x)); cpi *= 1 + infl; real = nominal / cpi
        peak = max(peak, real); current = max(0., 1 - real / peak); worst = max(worst, current)
        ever = ever or worst - loss > 1e-12
        if tau < 0 and ever:
            tau = year + 1; trigger_peak = peak; trigger_drawdown = worst
        for key, value in (("holdings", x.copy()), ("nominal_wealth", nominal), ("real_wealth", real),
                           ("real_peak", peak), ("current_drawdown", current), ("running_mdd", worst),
                           ("ever_breached", ever), ("fees", fee), ("moved", moved),
                           ("actions", "restore" if restore else "hold")):
            result[key].append(value)
    result = {key: np.asarray(value) for key, value in result.items()}
    result["tau"], result["trigger_peak"], result["trigger_drawdown"] = tau, trigger_peak, trigger_drawdown
    return result


class PairAccountingTests(unittest.TestCase):
    def test_independent_scalar_paths_match_vectorized_two_arms(self):
        rng = np.random.default_rng(887)
        market = rng.uniform(-.3, .25, (5, 7, 3)); infl = rng.uniform(-.01, .05, (5, 7))
        targets = np.array([[.5, .3, .2], [.2, .2, .6], [1 / 3] * 3]); losses = [.1, .2, .3]
        result = simulate_paired(market, infl, targets, losses)
        for row in range(5):
            for col, (w, loss) in enumerate(zip(targets, losses)):
                for name, pending in (("annual", False), ("pending_hold", True)):
                    scalar = independent_scalar(market[row], infl[row], w, loss, .0025, pending)
                    arm = result[name]
                    for field in ("holdings", "nominal_wealth", "real_wealth", "real_peak", "current_drawdown",
                                  "running_mdd", "fees", "moved"):
                        np.testing.assert_allclose(arm[field][row, col], scalar[field], atol=2e-12, rtol=0)
                    np.testing.assert_array_equal(arm["ever_breached"][row, col], scalar["ever_breached"])
                    np.testing.assert_array_equal(arm["actions"][row, col], scalar["actions"])
                    self.assertEqual(arm["first_crossing"][row, col], scalar["tau"])
        self.assertTrue(result["coupling"]["same_first_crossing"])
        self.assertTrue(result["coupling"]["same_terminal_ever_breached"])
        self.assertEqual(result["coupling"]["max_prefix_balance_error"], 0)

    def test_twenty_money_transfer_costs_point05_exactly_once(self):
        target = [20 / 99.95, 0., 79.95 / 99.95]
        result = simulate_paired(np.zeros((1, 1, 3)), np.zeros((1, 1)), target, .1)
        for arm in (result["annual"], result["pending_hold"]):
            self.assertAlmostEqual(arm["moved"][0, 0, 0], 20, places=12)
            self.assertAlmostEqual(arm["fees"][0, 0, 0], .05, places=13)
            self.assertAlmostEqual(arm["nominal_wealth"][0, 0, 1], 99.95, places=12)
            np.testing.assert_allclose(arm["posttrade_holdings"][0, 0, 0], [20, 0, 79.95], atol=2e-13, rtol=0)
            self.assertEqual(arm["real_wealth"][0, 0, 0], 100)

    def test_both_arms_keep_crossing_year_restore_and_only_later_pending_holds(self):
        market = np.array([[[-.4, 0, 0], [1, 0, 0], [-.5, 0, 0]]])
        result = simulate_paired(market, np.zeros((1, 3)), [.5, .5, 0], .1)
        a, p = result["annual"], result["pending_hold"]
        self.assertEqual(a["first_crossing"][0, 0], 1)
        self.assertEqual(p["first_crossing"][0, 0], 1)
        np.testing.assert_array_equal(a["actions"][0, 0], ["restore"] * 3)
        np.testing.assert_array_equal(p["actions"][0, 0], ["restore", "hold", "hold"])
        np.testing.assert_array_equal(p["pending_at_decision"][0, 0], [False, True, True])
        self.assertGreater(a["fees"][0, 0, 1], 0)
        np.testing.assert_array_equal(p["fees"][0, 0, 1:], 0)
        np.testing.assert_array_equal(a["holdings"][..., :2, :], p["holdings"][..., :2, :])
        self.assertTrue(a["ever_breached"][0, 0, -1])
        self.assertTrue(p["ever_breached"][0, 0, -1])
        # Nominal asset balances still grow; pending does not freeze wealth.
        np.testing.assert_allclose(p["holdings"][0, 0, 2], p["holdings"][0, 0, 1] * [2, 1, 1], atol=0, rtol=0)
        # Lower cost can coexist with a deeper later drawdown.
        self.assertGreater(p["running_mdd"][0, 0, -1], a["running_mdd"][0, 0, -1])

    def test_self_financing_and_real_wealth_conservation(self):
        market = np.array([[[.1, -.1, .01], [-.3, .05, .01], [.2, -.04, .01], [-.1, .02, .01]]])
        inflation = np.array([[.02, .03, -.01, .02]])
        result = simulate_paired(market, inflation, [[.4, .4, .2], [.7, .1, .2]], [.05, .2])
        for arm in (result["annual"], result["pending_hold"]):
            np.testing.assert_allclose(arm["fees"], .0025 * arm["moved"], atol=5e-15, rtol=0)
            np.testing.assert_allclose(arm["posttrade_holdings"].sum(axis=-1) + arm["fees"], arm["pretrade_wealth"], atol=3e-13, rtol=0)
            np.testing.assert_allclose(arm["holdings"][..., 1:, :], arm["posttrade_holdings"] * (1 + market[:, None]), atol=0, rtol=0)
            np.testing.assert_allclose(arm["real_wealth"], arm["nominal_wealth"] / arm["cpi"][:, None, :], atol=0, rtol=0)
            np.testing.assert_allclose(arm["weights"].sum(axis=-1), 1, atol=3e-16, rtol=0)

    def test_zero_cost_and_identity_calls_do_not_invent_transfers(self):
        result = simulate_paired(np.zeros((1, 3, 3)), np.zeros((1, 3)), [0, 0, 1], .1, cost_rate=0)
        for arm in (result["annual"], result["pending_hold"]):
            np.testing.assert_array_equal(arm["real_wealth"], 100)
            np.testing.assert_array_equal(arm["fees"], 0)
            np.testing.assert_array_equal(arm["moved"], 0)
            np.testing.assert_array_equal(arm["trades"], False)
            np.testing.assert_array_equal(arm["target_execution"], True)


class CrossingAndRecoveryTests(unittest.TestCase):
    def test_no_crossing_means_identical_paths_and_not_applicable_recovery(self):
        result = simulate_paired(np.full((2, 4, 3), .1), np.zeros((2, 4)), [1 / 3] * 3, .2)
        a, p = result["annual"], result["pending_hold"]
        for key in ("holdings", "fees", "moved", "real_wealth", "actions"):
            np.testing.assert_array_equal(a[key], p[key])
        for arm in (a, p):
            np.testing.assert_array_equal(arm["first_crossing"], -1)
            np.testing.assert_array_equal(arm["available_followup"], -1)
            np.testing.assert_array_equal(arm["observed_followup"], -1)
            np.testing.assert_array_equal(arm["right_censored"], False)
            self.assertTrue(np.isnan(arm["drawdown_at_trigger"]).all())
            self.assertTrue(np.isnan(arm["peak_at_trigger"]).all())

    def test_crossing_at_h_is_recorded_with_zero_censored_followup_and_no_late_action(self):
        market = np.array([[[.1] * 3, [-.3] * 3]])
        result = simulate_paired(market, np.zeros((1, 2)), [1 / 3] * 3, .2)
        for arm in (result["annual"], result["pending_hold"]):
            self.assertEqual(arm["first_crossing"][0, 0], 2)
            self.assertEqual(arm["available_followup"][0, 0], 0)
            self.assertEqual(arm["observed_followup"][0, 0], 0)
            self.assertEqual(arm["recovery_time"][0, 0], -1)
            self.assertTrue(arm["right_censored"][0, 0])
            np.testing.assert_array_equal(arm["actions"], "restore")
        np.testing.assert_array_equal(result["annual"]["real_wealth"], result["pending_hold"]["real_wealth"])

    def test_inflation_crossing_recovery_updates_peak_but_never_clears_b_or_reenters(self):
        # Cash wealth100 becomes real83.33,125,113.64.  The original100 peak
        # is recovered after one period, while full-history breach stays true.
        result = simulate_paired(np.zeros((1, 3, 3)), [[.2, -1 / 3, .1]], [0, 0, 1], .1)
        for arm in (result["annual"], result["pending_hold"]):
            self.assertEqual(arm["first_crossing"][0, 0], 1)
            self.assertEqual(arm["peak_at_trigger"][0, 0], 100)
            self.assertEqual(arm["recovery_endpoint"][0, 0], 2)
            self.assertEqual(arm["recovery_time"][0, 0], 1)
            self.assertEqual(arm["available_followup"][0, 0], 2)
            self.assertEqual(arm["observed_followup"][0, 0], 1)
            self.assertFalse(arm["right_censored"][0, 0])
            self.assertTrue(arm["ever_breached"][0, 0, 2])
            self.assertEqual(arm["current_drawdown"][0, 0, 2], 0)
            self.assertAlmostEqual(arm["real_peak"][0, 0, 2], 125)
        np.testing.assert_array_equal(result["pending_hold"]["actions"][0, 0], ["restore", "hold", "hold"])

    def test_unrecovered_crossing_is_right_censored_not_fake_recovery(self):
        market = np.array([[[-.5] * 3, [0] * 3, [0] * 3]])
        result = simulate_paired(market, np.zeros((1, 3)), [0, 0, 1], .2)
        for arm in (result["annual"], result["pending_hold"]):
            self.assertEqual(arm["first_crossing"][0, 0], 1)
            self.assertFalse(arm["recovered"][0, 0])
            self.assertTrue(arm["right_censored"][0, 0])
            self.assertEqual(arm["recovery_time"][0, 0], -1)
            self.assertEqual(arm["available_followup"][0, 0], 2)
            self.assertEqual(arm["observed_followup"][0, 0], 2)

    def test_initial_allocation_fee_is_included_at_first_annual_endpoint(self):
        result = simulate_paired(np.zeros((1, 1, 3)), np.zeros((1, 1)), [1 / 3] * 3, 0)
        fee_fraction = (.0025 * 2 / 3) / (1 + .0025 * 2 / 3)
        for arm in (result["annual"], result["pending_hold"]):
            self.assertAlmostEqual(arm["fees"][0, 0, 0], 100 * fee_fraction, places=13)
            self.assertAlmostEqual(arm["running_mdd"][0, 0, 1], fee_fraction, places=14)
            self.assertEqual(arm["first_crossing"][0, 0], 1)

    def test_crossing_equality_uses_explicit_one_e_minus12_guard(self):
        for decline, expected in ((.2, -1), (.2 + .5e-12, -1), (.2 + 3e-12, 1)):
            result = simulate_paired(np.full((1, 1, 3), -decline), np.zeros((1, 1)), [0, 0, 1], .2)
            self.assertEqual(result["annual"]["first_crossing"][0, 0], expected)
            self.assertEqual(result["pending_hold"]["first_crossing"][0, 0], expected)

    def test_vector_loss_columns_and_horizon_prefix_are_separate(self):
        market = np.array([[[0] * 3, [-.2] * 3, [-.3] * 3]])
        targets = [[0, 0, 1], [0, 0, 1]]
        short = simulate_paired(market, np.zeros((1, 3)), targets, [.1, .3], horizon=2)
        long = simulate_paired(market, np.zeros((1, 3)), targets, [.1, .3], horizon=3)
        np.testing.assert_array_equal(short["annual"]["first_crossing"], [[2, -1]])
        np.testing.assert_array_equal(long["annual"]["first_crossing"], [[2, 3]])
        self.assertEqual(short["annual"]["real_wealth"].shape, (1, 2, 3))
        np.testing.assert_array_equal(short["annual"]["real_wealth"][..., 0], 100)


class InterfaceTests(unittest.TestCase):
    def test_input_copies_and_readonly_results(self):
        market = np.zeros((1, 2, 3)); infl = np.zeros((1, 2)); targets = np.array([[.5, .2, .3]])
        originals = market.copy(), infl.copy(), targets.copy()
        result = simulate_paired(market, infl, targets, .2)
        for value, expected in zip((market, infl, targets), originals):
            np.testing.assert_array_equal(value, expected)
        targets[:] = [0, 0, 1]
        np.testing.assert_array_equal(result["annual"]["targets"], [[.5, .2, .3]])
        for arm in (result["annual"], result["pending_hold"]):
            for array in arm.values():
                self.assertFalse(array.flags.writeable)

    def test_future_valid_suffix_does_not_change_previous_actions_or_states(self):
        market = np.array([[[-.3, 0, 0], [.5, -.1, .01], [-.1, .1, 0]]])
        changed = market.copy(); changed[:, 1:] = [[.9, -.4, .5], [-.8, .6, 1]]
        inflation = np.array([[.02, .03, .01]]); changed_inflation = inflation.copy(); changed_inflation[:, 1:] = .5
        before = simulate_paired(market, inflation, [.5, .2, .3], .1)
        after = simulate_paired(changed, changed_inflation, [.5, .2, .3], .1)
        for arm in ("annual", "pending_hold"):
            for field in ("holdings", "weights", "real_wealth", "real_peak", "running_mdd", "ever_breached"):
                axis = -2 if field in ("holdings", "weights") else -1
                np.testing.assert_array_equal(np.take(before[arm][field], [0, 1], axis=axis),
                                               np.take(after[arm][field], [0, 1], axis=axis))
            np.testing.assert_array_equal(before[arm]["actions"][..., :2], after[arm]["actions"][..., :2])

    def test_invalid_shapes_values_thresholds_and_horizon(self):
        base = dict(returns=np.zeros((1, 2, 3)), inflation=np.zeros((1, 2)), targets=[.5, .2, .3], loss_tolerances=.2)
        invalid = [dict(returns=np.empty((0, 2, 3)), inflation=np.empty((0, 2))),
                   dict(returns=np.zeros((1, 2, 2))), dict(inflation=np.zeros((1, 3))),
                   dict(returns=np.full((1, 2, 3), -1)), dict(inflation=np.full((1, 2), -1)),
                   dict(returns=np.full((1, 2, 3), np.nan)), dict(targets=[.5, .2, .2]),
                   dict(targets=[-.1, .6, .5]), dict(targets=[]), dict(loss_tolerances=[.1, .2]),
                   dict(loss_tolerances=np.nan), dict(loss_tolerances=-.1), dict(loss_tolerances=True),
                   dict(horizon=0), dict(horizon=3), dict(horizon=1.0), dict(horizon=True),
                   dict(cost_rate=-.1), dict(cost_rate=1), dict(cost_rate=False)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                simulate_paired(**(base | change))


if __name__ == "__main__":
    unittest.main()
