"""Independent small-path accounting and causality checks; no market data."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from experimental_execution.fan_rollout import evaluate_actions, simulate_policy
from experimental_execution.review_pending import simulate_paired


def scalar_restore(x, target, cost):
    """Independent100-step fee bisection, with no production fee/step calls."""
    wealth = sum(map(float, x))
    low, high = 0., cost * wealth
    for _ in range(100):
        fee = (low + high) / 2
        moved = sum(max((wealth - fee) * float(w) - float(a), 0.) for a, w in zip(x, target))
        if fee > cost * moved:
            high = fee
        else:
            low = fee
    fee = (low + high) / 2
    return (wealth - fee) * target, fee


def scalar_fan(x, price, peak, worst, training, inflation, year, target, loss, cost, action):
    """Each suffix independently carries history and its own absorbing breach."""
    start = sum(map(float, x)) / price
    logs, bads = [], []
    for path, prices in zip(training, inflation):
        a, cp, pk, dd, bad = x.copy(), float(price), float(peak), float(worst), False
        for t in range(year, len(path)):
            restore = action == "restore" if t == year else not bad
            if restore:
                a, _ = scalar_restore(a, target, cost)
            a = a * (1 + path[t])
            cp *= 1 + prices[t]
            real = sum(map(float, a)) / cp
            pk = max(pk, real)
            dd = max(dd, 1 - real / pk)
            bad = bad or dd - loss > 1e-12
        logs.append(np.log(real) - np.log(start))
        bads.append(bad)
    return np.array(logs), np.array(bads)


def scalar_policy(actual, inflation, training, train_inflation, target, loss, cost, policy):
    x, price, peak, worst, bad = np.array([0., 0., 100.]), 1., 100., 0., False
    out = {key: [] for key in ("nominal_wealth", "real_wealth", "real_peak", "running_mdd",
                               "holdings", "ever_breached", "fees", "actions", "qg_hold", "qg_restore")}
    for key, value in (("nominal_wealth", 100.), ("real_wealth", 100.), ("real_peak", 100.),
                       ("running_mdd", 0.), ("holdings", x.copy()), ("ever_breached", False)):
        out[key].append(value)
    for year in range(len(actual)):
        qh = qr = np.nan
        if year == 0:
            restore = True
        elif bad:
            restore = False
        else:
            hl, hb = scalar_fan(x, price, peak, worst, training, train_inflation, year, target, loss, cost, "hold")
            rl, rb = scalar_fan(x, price, peak, worst, training, train_inflation, year, target, loss, cost, "restore")
            qh, qr = hl.mean(), rl.mean()
            restore = qr - qh > 1e-12 if policy == "growth_only_rollout" else not (
                qh >= qr - 1e-12 and sum(hb) <= sum(rb))
        if restore:
            post, fee = scalar_restore(x, target, cost)
        else:
            post, fee = x, 0.
        x = post * (1 + actual[year])
        price *= 1 + inflation[year]
        nominal = sum(map(float, x)); real = nominal / price
        peak = max(peak, real); worst = max(worst, 1 - real / peak)
        bad = bad or worst - loss > 1e-12
        for key, value in (("nominal_wealth", nominal), ("real_wealth", real), ("real_peak", peak),
                           ("running_mdd", worst), ("holdings", x.copy()), ("ever_breached", bad),
                           ("fees", fee), ("actions", "restore" if restore else "hold"),
                           ("qg_hold", qh), ("qg_restore", qr)):
            out[key].append(value)
    return {key: np.asarray(value) for key, value in out.items()}


class FanRolloutTests(unittest.TestCase):
    def test_batched_action_scores_match_independent_scalar_full_suffixes(self):
        rng = np.random.default_rng(5741)
        train = rng.uniform(-.3, .3, (4, 4, 3)); infl = rng.uniform(-.02, .06, (4, 4))
        x = np.array([[45., 15., 40.], [20., 50., 30.]])
        prices = np.array([1.1, .95]); peaks = np.array([105., 120.]); worsts = [.15, .18]
        target = np.array([.4, .3, .3])
        for year in (0, 1, 3):
            result = evaluate_actions(x, prices, peaks, worsts, False, train, infl,
                                      year_index=year, target=target, L=.22)
            for i in range(2):
                for action in ("hold", "restore"):
                    log, bad = scalar_fan(x[i], prices[i], peaks[i], worsts[i], train, infl, year, target, .22, .0025, action)
                    np.testing.assert_allclose(result[action]["total_log_growth"][i], log, atol=2e-15, rtol=0)
                    np.testing.assert_array_equal(result[action]["ever_breached"][i], bad)
                    self.assertEqual(result[action]["breach_count"][i], sum(bad))
                    self.assertAlmostEqual(result[action]["mean_log_growth"][i], log.mean(), places=14)

    def test_current_twenty_transfer_fee_is_charged_once(self):
        target = [20 / 99.95, 0, 79.95 / 99.95]
        result = evaluate_actions([0, 0, 100], 1, 100, 0, False, np.zeros((1, 1, 3)), np.zeros((1, 1)),
                                  year_index=0, target=target, L=.5)
        self.assertEqual(result["hold"]["total_log_growth"].shape, (1, 1))
        self.assertAlmostEqual(result["restore"]["total_log_growth"][0, 0], np.log(99.95 / 100), places=14)
        self.assertEqual(result["hold"]["total_log_growth"][0, 0], 0)
        self.assertFalse(result["growth_recommend_restore"][0])
        for policy in ("growth_only_rollout", "pareto_hold_rollout"):
            h1 = simulate_policy(np.zeros((1, 1, 3)), np.zeros((1, 1)),
                np.zeros((1, 1, 3)), np.zeros((1, 1)), target, policy=policy, L=.0001)
            self.assertEqual(h1["actions"][0, 0], "restore")
            self.assertAlmostEqual(h1["fees"][0, 0], .05, places=13)
            self.assertAlmostEqual(h1["moved"][0, 0], 20, places=12)
            self.assertAlmostEqual(h1["real_wealth"][0, 1], 99.95, places=12)
            self.assertAlmostEqual(h1["running_mdd"][0, 1], .0005, places=14)
            self.assertEqual(h1["first_crossing"][0], 1)
            self.assertTrue(h1["right_censored"][0])
            self.assertTrue(np.isnan(h1["qg_hold"][0, 0]))

    def test_future_baseline_restoration_fee_is_included(self):
        training = np.array([[[.32, -.12, 0], [.32, -.12, 0]], [[-.12, .32, 0], [-.12, .32, 0]]])
        result = evaluate_actions([50, 50, 0], 1, 100, 0, False, training, np.zeros((2, 2)),
                                  year_index=0, target=[.5, .5, 0], L=.9)
        expected = 2 * np.log(1.1) + np.log1p(-.00025 / 1.00125)
        for action in ("hold", "restore"):
            np.testing.assert_allclose(result[action]["total_log_growth"], expected, atol=1e-15, rtol=0)
        self.assertLess(expected, 2 * np.log(1.1))

    def test_own_branch_breach_latches_before_future_baseline_action(self):
        train = np.array([[[-.8, 0, 0], [2, 0, 0], [-.2, 0, 0]]])
        infl = np.zeros((1, 3)); target = np.array([.5, 0, .5])
        result = evaluate_actions([20, 0, 80], 1, 100, 0, False, train, infl,
                                  year_index=0, target=target, L=.2)
        for action in ("hold", "restore"):
            log, bad = scalar_fan(np.array([20., 0, 80]), 1, 100, 0, train, infl, 0, target, .2, .0025, action)
            np.testing.assert_allclose(result[action]["total_log_growth"][0], log, atol=1e-15, rtol=0)
            np.testing.assert_array_equal(result[action]["ever_breached"][0], bad)
        self.assertFalse(result["hold"]["ever_breached"][0, 0])
        self.assertTrue(result["restore"]["ever_breached"][0, 0])

    def test_original_historical_peak_and_inflation_are_carried(self):
        train = np.array([[[-.3, 0, 0]]]); pi = np.zeros((1, 1))
        old = evaluate_actions([0, 0, 100], 1, 200, .5, False, train, pi,
                               year_index=0, target=[1, 0, 0], L=.6)
        fresh = evaluate_actions([0, 0, 100], 1, 100, 0, False, train, pi,
                                 year_index=0, target=[1, 0, 0], L=.6)
        self.assertTrue(old["restore"]["ever_breached"][0, 0])
        self.assertFalse(fresh["restore"]["ever_breached"][0, 0])
        recovered = evaluate_actions([0, 0, 100], 1, 100, 0, False, np.zeros((1, 2, 3)), [[.25, -.2]],
                                     year_index=0, target=[0, 0, 1], L=.1)
        self.assertAlmostEqual(recovered["hold"]["total_log_growth"][0, 0], 0, places=14)
        self.assertTrue(recovered["hold"]["ever_breached"][0, 0])

    def test_calendar_prefix_and_beyond_original_h_are_not_forecasted(self):
        train = np.array([[[.2, -.1, .01], [-.1, .3, .02], [.5, -.2, .03], [.1, .1, .1]]])
        inflation = np.array([[.03, .02, .04, .05]])
        changed, pi2 = train.copy(), inflation.copy()
        changed[:, [0, 3]] = [[[-.9, 2, 5], [4, -.9, 10]]]
        pi2[:, [0, 3]] = [4, 6]
        kwargs = dict(year_index=1, target=[.4, .4, .2], L=.3, horizon=3)
        a = evaluate_actions([40, 30, 30], 1, 100, 0, False, train, inflation, **kwargs)
        b = evaluate_actions([40, 30, 30], 1, 100, 0, False, changed, pi2, **kwargs)
        for action in ("hold", "restore"):
            np.testing.assert_array_equal(a[action]["total_log_growth"], b[action]["total_log_growth"])
            np.testing.assert_array_equal(a[action]["ever_breached"], b[action]["ever_breached"])

    def test_pareto_can_restore_at_predicted_growth_sacrifice(self):
        actual = np.array([[[3., 0, 0], [0, 0, 0]]])
        train = np.zeros((2, 2, 3)); train[:, 1, 0] = [2., -.5]
        infl, train_infl = np.zeros((1, 2)), np.zeros((2, 2))
        growth = simulate_policy(actual, infl, train, train_infl, [.5, 0, .5], policy="growth_only_rollout", L=.3)
        pareto = simulate_policy(actual, infl, train, train_infl, [.5, 0, .5], policy="pareto_hold_rollout", L=.3)
        self.assertEqual(growth["actions"][0, 1], "hold")
        self.assertEqual(pareto["actions"][0, 1], "restore")
        self.assertEqual(pareto["qbad_hold_count"][0, 1], 1)
        self.assertEqual(pareto["qbad_restore_count"][0, 1], 0)
        self.assertLess(pareto["predicted_growth_advantage_restore"][0, 1], 0)
        self.assertTrue(pareto["restoration_predicted_growth_sacrifice"][0, 1])
        self.assertEqual(pareto["delete_one_flip_count"][0, 1], 1)
        self.assertEqual(growth["delete_one_flip_count"][0, 1], 1)
        # Original-H annualization is an external reporting operation.  Q is
        # the single remaining period's TOTAL log factor without H division.
        expected, _ = scalar_fan(pareto["holdings"][0, 1].copy(), 1., pareto["real_peak"][0, 1], 0.,
            train, train_infl, 1, np.array([.5, 0, .5]), .3, .0025, "hold")
        self.assertAlmostEqual(pareto["qg_hold"][0, 1], expected.mean(), places=14)

    def test_delete_one_recommendations_match_explicit_fan_deletions(self):
        rng = np.random.default_rng(87)
        train = rng.uniform(-.4, .7, (5, 3, 3)); infl = rng.uniform(-.02, .1, (5, 3))
        state = ([70, 0, 30], 1, 100, 0, False)
        kw = dict(year_index=0, target=[.4, .2, .4], L=.25)
        full = evaluate_actions(*state, train, infl, **kw)
        counts = {rule: 0 for rule in ("growth", "pareto")}
        for deleted in range(len(train)):
            sub = evaluate_actions(*state, np.delete(train, deleted, axis=0), np.delete(infl, deleted, axis=0), **kw)
            for rule in counts:
                counts[rule] += int(sub[rule + "_recommend_restore"][0] != full[rule + "_recommend_restore"][0])
        for rule, count in counts.items():
            self.assertEqual(full[rule + "_delete_one_flip_count"][0], count)
        single = evaluate_actions(*state, train[:1], infl[:1], **kw)
        self.assertEqual(single["growth_delete_one_flip_count"][0], -1)
        self.assertEqual(single["pareto_delete_one_flip_count"][0], -1)

    def test_local_fan_pareto_rule_has_no_global_policy_improvement_theorem(self):
        # The year2 fan is unconditional and is recombined with a history
        # generated by just one of its paths.  This is an intentional model
        # limitation, not a reason to tune the gate after observing outcomes.
        gross = np.array([[[2., 1., 1.], [.8, 1., 1.]],
                          [[.5, 1., 1.], [1., .68, 1.]]])
        market, inflation, target = gross - 1, np.zeros((2, 2)), np.array([.5, .4, .1])
        values = evaluate_actions([100, 40, 10], 1, 150, 0, False, market, inflation,
                                  year_index=1, target=target, L=.12, cost_rate=0)
        np.testing.assert_allclose(np.exp(values["hold"]["total_log_growth"])[0], [130 / 150, 137.2 / 150], atol=1e-15, rtol=0)
        np.testing.assert_allclose(np.exp(values["restore"]["total_log_growth"])[0], [135 / 150, 130.8 / 150], atol=1e-15, rtol=0)
        self.assertEqual(values["hold"]["breach_count"][0], 1)
        self.assertEqual(values["restore"]["breach_count"][0], 1)
        self.assertGreater(values["hold"]["mean_log_growth"][0], values["restore"]["mean_log_growth"][0])
        self.assertFalse(values["pareto_recommend_restore"][0])
        baseline = simulate_paired(market, inflation, target, .12, cost_rate=0)["pending_hold"]
        candidate = simulate_policy(market, inflation, market, inflation, target,
                                    policy="pareto_hold_rollout", L=.12, cost_rate=0)
        np.testing.assert_allclose(baseline["real_wealth"][:, 0, -1], [135, 62.2], atol=3e-14, rtol=0)
        np.testing.assert_allclose(candidate["real_wealth"][:, -1], [130, 62.2], atol=3e-14, rtol=0)
        self.assertEqual(baseline["ever_breached"][:, 0, -1].sum(), 1)
        self.assertEqual(candidate["ever_breached"][:, -1].sum(), 2)
        baseline_mean_g = np.mean(np.log(baseline["real_wealth"][:, 0, -1] / 100)) / 2
        candidate_mean_g = np.mean(np.log(candidate["real_wealth"][:, -1] / 100)) / 2
        self.assertLess(candidate_mean_g, baseline_mean_g)

    def test_full_two_policies_match_independent_nested_scalar_replay(self):
        rng = np.random.default_rng(780)
        actual = rng.uniform(-.45, .6, (3, 5, 3)); inflation = rng.uniform(-.01, .08, (3, 5))
        training = rng.uniform(-.4, .5, (4, 5, 3)); train_pi = rng.uniform(-.02, .06, (4, 5))
        w = np.array([.4, .35, .25])
        for policy in ("growth_only_rollout", "pareto_hold_rollout"):
            result = simulate_policy(actual, inflation, training, train_pi, w, policy=policy, L=.25)
            for row in range(len(actual)):
                expected = scalar_policy(actual[row], inflation[row], training, train_pi, w, .25, .0025, policy)
                for key in ("nominal_wealth", "real_wealth", "real_peak", "running_mdd", "holdings", "fees", "qg_hold", "qg_restore"):
                    np.testing.assert_allclose(result[key][row], expected[key], atol=4e-13, rtol=0, equal_nan=True)
                for key in ("ever_breached", "actions"):
                    np.testing.assert_array_equal(result[key][row], expected[key])
            np.testing.assert_allclose(result["fees"], .0025 * result["moved"], atol=2e-15, rtol=0)
            np.testing.assert_allclose(result["posttrade_holdings"].sum(axis=-1) + result["fees"], result["pretrade_wealth"], atol=2e-13, rtol=0)

    def test_initial_and_pending_have_no_gate_and_recovery_does_not_reenter(self):
        actual = np.zeros((1, 3, 3)); inflation = np.array([[.25, -.2, .1]])
        result = simulate_policy(actual, inflation, np.zeros((2, 3, 3)), np.zeros((2, 3)), [0, 0, 1],
                                 policy="growth_only_rollout", L=.1)
        np.testing.assert_array_equal(result["actions"], [["restore", "hold", "hold"]])
        np.testing.assert_array_equal(result["gate_reason"], [["mandatory_initial_restore", "already_breached_pending_hold", "already_breached_pending_hold"]])
        self.assertTrue(np.isnan(result["qg_hold"]).all())
        np.testing.assert_array_equal(result["forecast_n"], -1)
        self.assertEqual(result["first_crossing"][0], 1)
        self.assertEqual(result["recovery_time"][0], 1)
        self.assertEqual(result["observed_followup"][0], 1)
        self.assertEqual(result["available_followup"][0], 2)
        self.assertFalse(result["right_censored"][0])
        np.testing.assert_array_equal(result["ever_breached"], [[False, True, True, True]])

    def test_identity_zero_fee_and_breach_at_last_endpoint(self):
        for cost in (0., .0025):
            out = simulate_policy(np.zeros((1, 2, 3)), [[0, .25]], np.zeros((1, 2, 3)), np.zeros((1, 2)),
                                  [0, 0, 1], policy="pareto_hold_rollout", L=.1, cost_rate=cost)
            np.testing.assert_array_equal(out["fees"], 0)
            np.testing.assert_array_equal(out["moved"], 0)
            self.assertEqual(out["first_crossing"][0], 2)
            self.assertTrue(out["right_censored"][0])
            self.assertEqual(out["available_followup"][0], 0)
            self.assertEqual(out["observed_followup"][0], 0)
            self.assertEqual(out["delete_one_flip_count"][0, 1], -1)

    def test_future_actual_returns_do_not_enter_an_earlier_gate(self):
        rng = np.random.default_rng(912)
        actual = rng.uniform(-.1, .2, (2, 4, 3)); inflation = np.zeros((2, 4))
        train = rng.uniform(-.3, .4, (3, 4, 3)); train_pi = np.zeros((3, 4))
        changed, changed_pi = actual.copy(), inflation.copy()
        changed[:, 2:] = np.array([[-.9, 2, .1], [.5, -.6, 0]])
        changed_pi[:, 2:] = [.3, -.1]
        for policy in ("growth_only_rollout", "pareto_hold_rollout"):
            kwargs = dict(target=[.5, .3, .2], policy=policy, L=.5)
            a = simulate_policy(actual, inflation, train, train_pi, **kwargs)
            b = simulate_policy(changed, changed_pi, train, train_pi, **kwargs)
            for key in ("actions", "qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count"):
                np.testing.assert_array_equal(a[key][:, :3], b[key][:, :3])
            np.testing.assert_array_equal(a["holdings"][:, :3], b["holdings"][:, :3])

    def test_inputs_are_unmodified_outputs_readonly_and_horizon_fresh(self):
        actual = np.zeros((1, 4, 3)); train = np.ones((2, 4, 3)) * .01
        inflation, train_pi = np.zeros((1, 4)), np.zeros((2, 4))
        saved = [a.copy() for a in (actual, train, inflation, train_pi)]
        out = simulate_policy(actual, inflation, train, train_pi, [.5, .2, .3],
                              policy="growth_only_rollout", L=.5, horizon=2)
        self.assertEqual(out["real_wealth"].shape, (1, 3))
        self.assertEqual(out["real_wealth"][0, 0], 100)
        self.assertEqual(out["gate_reason"][0, 0], "mandatory_initial_restore")
        self.assertFalse(out["triggered"][0]); self.assertEqual(out["available_followup"][0], -1)
        self.assertTrue(np.isnan(out["added_severity"][0]))
        for a, before in zip((actual, train, inflation, train_pi), saved):
            np.testing.assert_array_equal(a, before)
        with self.assertRaises(ValueError):
            out["real_wealth"][0, 0] = 0

    def test_validation_rejects_inconsistent_or_invalid_inputs(self):
        train, pi = np.zeros((1, 2, 3)), np.zeros((1, 2))
        kw = dict(year_index=0, target=[.4, .3, .3], L=.3)
        for state in (([0, 0, 100], 1, 99, 0, False), ([0, 0, 100], 1, 200, 0, False),
                      ([0, 0, 100], 1, 100, 0, True), ([0, 0, 100], 0, 100, 0, False),
                      ([0, 0, 1e-200], 1, 1e-210, 0, False)):
            with self.assertRaises(ValueError):
                evaluate_actions(*state, train, pi, **kw)
        for change in (dict(year_index=2), dict(year_index=True), dict(horizon=3), dict(horizon=0),
                       dict(L=True), dict(L=-.1), dict(cost_rate=1), dict(target=[.5, .5, .1])):
            with self.assertRaises(ValueError):
                evaluate_actions([0, 0, 100], 1, 100, 0, False, train, pi, **(kw | change))
        broken = train.copy(); broken[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            evaluate_actions([0, 0, 100], 1, 100, 0, False, broken, pi, **kw)
        with self.assertRaises(ValueError):
            simulate_policy(train, pi, train, pi, [.4, .3, .3], policy="unknown", L=.3)
        with self.assertRaises(ValueError):
            simulate_policy(train, pi, train[:, :1], pi[:, :1], [.4, .3, .3], policy="growth_only_rollout", L=.3)


if __name__ == "__main__":
    unittest.main()
