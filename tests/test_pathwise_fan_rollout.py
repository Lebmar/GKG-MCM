"""Synthetic finite-support shield checks, not an OOF risk certificate."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from experimental_execution.pathwise_fan_rollout import evaluate_actions, simulate_policy
from experimental_execution.review_pending import simulate_paired
from test_fan_rollout import scalar_fan, scalar_restore


class PathwiseShieldTests(unittest.TestCase):
    def test_growth_gate_uses_exact_zero_without_small_negative_debt(self):
        train = np.zeros((1, 1, 3)); train[0, 0, 0] = 5e-13
        result = evaluate_actions([0, 0, 100], 1, 100, 0, False, train, np.zeros((1, 1)),
            year_index=0, target=[1, 0, 0], L=.5, cost_rate=0)
        self.assertLess(result["min_log_advantage_hold"][0], 0)
        self.assertGreater(result["min_log_advantage_hold"][0], -1e-12)
        self.assertEqual(result["growth_violation_count"][0], 1)
        self.assertTrue(result["pathwise_recommend_restore"][0])
        self.assertEqual(result["delete_one_flip_count"][0], -1)

    def test_risk_violation_alone_blocks_higher_terminal_growth_hold(self):
        train = np.array([[[-.5, 0, 0], [2., 0, 0]]])
        result = evaluate_actions([100, 0, 0], 1, 100, 0, False, train, np.zeros((1, 2)),
            year_index=0, target=[0, 0, 1], L=.2, cost_rate=0)
        self.assertGreater(result["min_log_advantage_hold"][0], 0)
        self.assertEqual(result["growth_violation_count"][0], 0)
        self.assertEqual(result["breach_violation_count"][0], 1)
        self.assertEqual(result["union_violation_count"][0], 1)
        self.assertTrue(result["pathwise_recommend_restore"][0])

    def test_mean_pareto_counterexample_is_rejected_without_tuning(self):
        gross = np.array([[[2., 1., 1.], [.8, 1., 1.]], [[.5, 1., 1.], [1., .68, 1.]]])
        market, inflation, w = gross - 1, np.zeros((2, 2)), [.5, .4, .1]
        result = evaluate_actions([100, 40, 10], 1, 150, 0, False, market, inflation,
            year_index=1, target=w, L=.12, cost_rate=0)
        self.assertGreater(result["hold"]["mean_log_growth"][0], result["restore"]["mean_log_growth"][0])
        self.assertEqual(result["hold"]["breach_count"][0], result["restore"]["breach_count"][0])
        self.assertTrue(result["pathwise_recommend_restore"][0])
        self.assertEqual(result["growth_violation_count"][0], 1)
        self.assertEqual(result["breach_violation_count"][0], 1)
        self.assertEqual(result["union_violation_count"][0], 1)
        self.assertEqual(result["delete_one_flip_count"][0], 1)
        candidate = simulate_policy(market, inflation, market, inflation, w, L=.12, cost_rate=0)
        baseline = simulate_paired(market, inflation, w, .12, cost_rate=0)["pending_hold"]
        np.testing.assert_allclose(candidate["real_wealth"][:, -1], [135., 62.2], atol=3e-14, rtol=0)
        np.testing.assert_array_equal(candidate["ever_breached"][:, -1], baseline["ever_breached"][:, 0, -1])
        self.assertEqual(candidate["actions"][0, 1], "restore")

    def test_action_values_and_violations_match_independent_scalar_fan(self):
        rng = np.random.default_rng(7762)
        training = rng.uniform(-.4, .7, (4, 4, 3)); inflation = rng.uniform(-.01, .07, (4, 4))
        target = np.array([.3, .4, .3]); states = np.array([[70., 10., 20.], [20., 50., 30.]])
        result = evaluate_actions(states, [1., 1.1], [110., 100.], [.12, .15], False, training, inflation,
            year_index=1, target=target, L=.25)
        for i, (state, price, peak, worst) in enumerate(zip(states, [1, 1.1], [110, 100], [.12, .15])):
            hl, hb = scalar_fan(state, price, peak, worst, training, inflation, 1, target, .25, .0025, "hold")
            rl, rb = scalar_fan(state, price, peak, worst, training, inflation, 1, target, .25, .0025, "restore")
            np.testing.assert_allclose(result["hold"]["total_log_growth"][i], hl, atol=2e-15, rtol=0)
            np.testing.assert_allclose(result["restore"]["total_log_growth"][i], rl, atol=2e-15, rtol=0)
            np.testing.assert_array_equal(result["hold"]["ever_breached"][i], hb)
            np.testing.assert_array_equal(result["restore"]["ever_breached"][i], rb)
            self.assertEqual(result["growth_violation_count"][i], sum(hl < rl))
            self.assertEqual(result["breach_violation_count"][i], sum(hb & ~rb))
            self.assertEqual(result["union_violation_count"][i], sum((hl < rl) | (hb & ~rb)))

    def test_deletion_flips_match_explicit_deletions_of_same_state_fan(self):
        rng = np.random.default_rng(84)
        train = rng.uniform(-.4, .6, (4, 3, 3)); inflation = np.zeros((4, 3))
        args = ([60, 0, 40], 1, 100, 0, False)
        kwargs = dict(year_index=0, target=[.3, .2, .5], L=.3)
        full = evaluate_actions(*args, train, inflation, **kwargs)
        flips = 0
        for j in range(len(train)):
            sub = evaluate_actions(*args, np.delete(train, j, axis=0), np.delete(inflation, j, axis=0), **kwargs)
            flips += int(sub["pathwise_recommend_restore"][0] != full["pathwise_recommend_restore"][0])
        self.assertEqual(full["delete_one_flip_count"][0], flips)
        # With two different violating paths no one deletion can repair all.
        duplicate = np.zeros((2, 1, 3)); duplicate[:, 0, 0] = .1
        v = evaluate_actions([0, 0, 100], 1, 100, 0, False, duplicate, np.zeros((2, 1)),
            year_index=0, target=[1, 0, 0], L=.5, cost_rate=0)
        self.assertEqual(v["union_violation_count"][0], 2)
        self.assertEqual(v["delete_one_flip_count"][0], 0)

    def test_h1_to_h4_replay_dominates_B_on_every_retained_fan_path(self):
        rng = np.random.default_rng(93267)
        fans = [(rng.uniform(-.45, .6, (3, 4, 3)), rng.uniform(-.03, .08, (3, 4)), [.4, .35, .25], .25),
                (rng.uniform(-.2, .3, (4, 4, 3)), rng.uniform(0, .05, (4, 4)), [.2, .3, .5], .15),
                (np.array([[[.2, -.1, 0], [-.3, .2, .01], [.6, -.05, .01], [-.15, .1, .01]]]),
                 np.array([[.04, .02, -.02, .01]]), [.5, .2, .3], .2)]
        comparisons = 0
        for market, inflation, w, loss in fans:
            for h in range(1, 5):
                for cost in (0., .0025):
                    shield = simulate_policy(market, inflation, market, inflation, w, L=loss, horizon=h, cost_rate=cost)
                    base = simulate_paired(market, inflation, w, loss, horizon=h, cost_rate=cost)["pending_hold"]
                    # FP tolerance is ONLY an audit comparison, never a gate
                    # debt.  The gate itself uses exact computed >=0.
                    delta = np.log(shield["real_wealth"][:, -1]) - np.log(base["real_wealth"][:, 0, -1])
                    self.assertTrue(np.all(delta >= -3e-14))
                    self.assertTrue(np.all(shield["ever_breached"][:, -1] <= base["ever_breached"][:, 0, -1]))
                    active = shield["union_violation_count"] >= 0
                    hold = (shield["actions"] == "hold") & active
                    np.testing.assert_array_equal(shield["union_violation_count"][hold], 0)
                    self.assertTrue(np.all(shield["min_log_advantage_hold"][hold] >= 0))
                    comparisons += len(market)
        self.assertEqual(comparisons, 64)

    def test_strict_supported_improvement_and_fee_saving_are_possible(self):
        market = np.array([[[.5, 0, 0], [.02, .02, .02], [.03, .03, .03]],
                           [[0, .4, 0], [.02, .02, .02], [.03, .03, .03]]])
        pi = np.zeros((2, 3)); target = [.4, .3, .3]
        shield = simulate_policy(market, pi, market, pi, target, L=.5)
        base = simulate_paired(market, pi, target, .5)["pending_hold"]
        np.testing.assert_array_equal(shield["actions"], [["restore", "hold", "hold"]] * 2)
        self.assertTrue(np.all(shield["real_wealth"][:, -1] > base["real_wealth"][:, 0, -1]))
        self.assertTrue(np.all(shield["fees"].sum(axis=1) < base["fees"][:, 0].sum(axis=1)))

    def test_initial_mandatory_fee_and_actual_transition_replay(self):
        target = np.array([20 / 99.95, 0, 79.95 / 99.95])
        market = np.zeros((1, 3, 3)); inflation = np.array([[.25, -.2, .01]])
        out = simulate_policy(market, inflation, np.zeros((2, 3, 3)), np.zeros((2, 3)), target, L=.1)
        self.assertEqual(out["actions"][0, 0], "restore")
        self.assertAlmostEqual(out["fees"][0, 0], .05, places=13)
        self.assertAlmostEqual(out["moved"][0, 0], 20, places=12)
        self.assertEqual(out["first_crossing"][0], 1)
        np.testing.assert_array_equal(out["actions"][0, 1:], "hold")
        self.assertTrue(np.isnan(out["qg_hold"]).all())
        np.testing.assert_array_equal(out["union_violation_count"], -1)
        np.testing.assert_allclose(out["fees"], .0025 * out["moved"], atol=2e-15, rtol=0)
        np.testing.assert_allclose(out["posttrade_holdings"].sum(axis=-1) + out["fees"], out["pretrade_wealth"], atol=2e-13, rtol=0)
        np.testing.assert_allclose(out["holdings"][:, 1:], out["posttrade_holdings"] * (1 + market), atol=0, rtol=0)

    def test_current_actual_future_is_absent_from_gate(self):
        rng = np.random.default_rng(78)
        actual = rng.uniform(-.1, .2, (2, 4, 3)); inflation = np.zeros((2, 4))
        train = rng.uniform(-.3, .4, (3, 4, 3)); train_pi = np.zeros((3, 4))
        changed, changed_pi = actual.copy(), inflation.copy()
        changed[:, 2:] = [[-.9, 2, .1], [.5, -.6, 0]]; changed_pi[:, 2:] = [.3, -.1]
        a = simulate_policy(actual, inflation, train, train_pi, [.5, .3, .2], L=.5)
        b = simulate_policy(changed, changed_pi, train, train_pi, [.5, .3, .2], L=.5)
        for field in ("actions", "qg_hold", "qg_restore", "union_violation_count", "delete_one_flip_count"):
            np.testing.assert_array_equal(a[field][:, :3], b[field][:, :3])
        np.testing.assert_array_equal(a["holdings"][:, :3], b["holdings"][:, :3])

    def test_validation_and_readonly_contract(self):
        market = np.zeros((1, 2, 3)); inflation = np.zeros((1, 2))
        with self.assertRaises(ValueError):
            simulate_policy(market, inflation, market, inflation, [.4, .3, .3], L=.3, policy="growth_only_rollout")
        with self.assertRaises(ValueError):
            evaluate_actions([0, 0, 100], 1, 100, 0, True, market, inflation,
                year_index=0, target=[.4, .3, .3], L=.3)
        out = simulate_policy(market, inflation, market, inflation, [.4, .3, .3], L=.3)
        with self.assertRaises(ValueError): out["real_wealth"][0, 0] = 0
        values = evaluate_actions([0, 0, 100], 1, 100, 0, False, market, inflation,
            year_index=0, target=[.4, .3, .3], L=.3)
        with self.assertRaises(ValueError): values["union_violation_count"][0] = 10


if __name__ == "__main__":
    unittest.main()
