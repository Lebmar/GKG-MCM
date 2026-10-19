"""Independent bounded-target action/history tests on synthetic joint fans."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from experimental_execution.bounded_target_update import choose_actions, evaluate_actions, simulate_controllers
from experimental_execution.risk_priced_fan_rollout import DEFAULT_ANNUAL_PRICES, simulate_prices
from test_fan_rollout import scalar_fan, scalar_restore

TARGETS = np.array([[.2, .3, .5], [.5, .3, .2], [.75, .15, .1]])


def independent_action_fan(x, cp, peak, dd, train, pi, year, targets, current, loss, cost):
    logs, flags = [], []
    log, bad = scalar_fan(x, cp, peak, dd, train, pi, year, targets[current], loss, cost, "hold")
    logs.append(log); flags.append(bad)
    for target in targets:
        log, bad = scalar_fan(x, cp, peak, dd, train, pi, year, target, loss, cost, "restore")
        logs.append(log); flags.append(bad)
    return np.array(logs), np.array(flags)


def independent_choice(logs, bad, current, eta, H, enabled):
    qg, qb = logs.mean(axis=1), bad.sum(axis=1)
    available = [0, current + 1] if not enabled else list(range(4))
    priority = [0, current + 1, *[j for j in range(1, 4) if j != current + 1]]
    if np.isinf(eta):
        minimum = min(int(qb[j]) for j in available)
        valid = [j for j in available if qb[j] == minimum]
        if qb[0] == minimum: valid = [j for j in valid if qg[j] - qg[0] > 1e-12]
        if not valid: return 0, np.nan
        maximum = max(qg[j] for j in valid)
        return next(j for j in priority if j in valid and qg[j] >= maximum - 1e-12), np.nan
    scores = [float(qg[j] - qg[0] - H * eta * ((int(qb[j]) - int(qb[0])) / logs.shape[1])) for j in range(4)]
    valid = [j for j in available if scores[j] > 1e-12]
    if not valid: return 0, 0.
    maximum = max(scores[j] for j in valid)
    chosen = next(j for j in priority if j in valid and scores[j] >= maximum - 1e-12)
    return chosen, scores[chosen]


def independent_controller(actual, actual_pi, train, train_pi, targets, initial, eta, enabled, loss, cost):
    x, cp, peak, dd, bad, target = np.array([0., 0., 100.]), 1., 100., 0., False, initial
    H = len(actual)
    result = {key: [] for key in ("holdings", "real_wealth", "real_peak", "running_mdd", "ever_breached", "fees", "moved", "action_index", "current_target_before", "current_target_after", "target_updated", "qg_actions", "qbad_actions", "selected_score")}
    for key, value in (("holdings", x.copy()), ("real_wealth", 100.), ("real_peak", 100.), ("running_mdd", 0.), ("ever_breached", False)):
        result[key].append(value)
    for t in range(H):
        before = target; qg, qb, score = np.full(4, np.nan), np.full(4, -1), np.nan
        if t == 0:
            choice = initial + 1
        elif bad:
            choice = 0
        else:
            logs, flags = independent_action_fan(x, cp, peak, dd, train, train_pi, t, targets, target, loss, cost)
            qg, qb = logs.mean(axis=1), flags.sum(axis=1)
            choice, score = independent_choice(logs, flags, target, eta, H, enabled)
            if not enabled:
                missing = [j for j in range(1, 4) if j != target + 1]
                qg[missing], qb[missing] = np.nan, -1
        old = x.copy()
        if choice:
            target = choice - 1; post, fee = scalar_restore(x, targets[target], cost)
        else:
            post, fee = x, 0.
        moved = sum(max(float(a) - float(b), 0.) for a, b in zip(post, old))
        x = post * (1 + actual[t]); cp *= 1 + actual_pi[t]
        real = sum(map(float, x)) / cp; peak = max(peak, real); dd = max(dd, 1 - real / peak)
        bad = bad or dd - loss > 1e-12
        for key, value in (("holdings", x.copy()), ("real_wealth", real), ("real_peak", peak), ("running_mdd", dd),
            ("ever_breached", bad), ("fees", fee), ("moved", moved), ("action_index", choice),
            ("current_target_before", before), ("current_target_after", target), ("target_updated", t > 0 and before != target),
            ("qg_actions", qg), ("qbad_actions", qb), ("selected_score", score)):
            result[key].append(value)
    return {key: np.asarray(value) for key, value in result.items()}


class BoundedTargetTests(unittest.TestCase):
    def test_disabled_mode_exactly_reproduces_all_nine_parent_prices(self):
        rng = np.random.default_rng(8166)
        actual = rng.uniform(-.35, .4, (3, 5, 3)); pi = rng.uniform(-.02, .07, (3, 5))
        train = rng.uniform(-.4, .55, (4, 5, 3)); train_pi = rng.uniform(-.02, .07, (4, 5))
        expected = simulate_prices(actual, pi, train, train_pi, TARGETS[1], DEFAULT_ANNUAL_PRICES, L=.25)
        actual_out = simulate_controllers(actual, pi, train, train_pi, TARGETS, DEFAULT_ANNUAL_PRICES, L=.25, updates_enabled=False)
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                if value.dtype.kind == "f":
                    np.testing.assert_allclose(actual_out[key], value, atol=0, rtol=0, equal_nan=True, err_msg=key)
                else:
                    np.testing.assert_array_equal(actual_out[key], value, err_msg=key)
        np.testing.assert_array_equal(actual_out["current_target_index"], 1)
        np.testing.assert_array_equal(actual_out["target_updated"], False)

    def test_action_fans_use_current_hold_and_each_new_targets_own_baseline(self):
        rng = np.random.default_rng(938)
        train = rng.uniform(-.45, .65, (3, 4, 3)); pi = rng.uniform(-.02, .08, (3, 4))
        states = np.array([[65., 10., 25.], [10., 60., 30.]])
        out = evaluate_actions(states, [1., 1.05], [110., 105.], [.15, .2], False, train, pi,
            year_index=1, target_matrix=TARGETS, current_target_index=[0, 2], L=.3)
        for i, current in enumerate((0, 2)):
            logs, bad = independent_action_fan(states[i], [1., 1.05][i], [110., 105.][i], [.15, .2][i], train, pi, 1, TARGETS, current, .3, .0025)
            np.testing.assert_allclose(out["total_log_actions"][i], logs, atol=3e-15, rtol=0)
            np.testing.assert_array_equal(out["ever_breached_actions"][i], bad)
            np.testing.assert_allclose(out["qg_actions"][i], logs.mean(axis=1), atol=2e-15, rtol=0)
            np.testing.assert_array_equal(out["qbad_actions"][i], bad.sum(axis=1))
        # Holding exactly the same asset balances, but a different CURRENT
        # target, changes future B and can change true hold Q before final H.
        other = evaluate_actions(states[:1], 1., 110., .15, False, train, pi,
            year_index=1, target_matrix=TARGETS, current_target_index=2, L=.3)
        self.assertGreater(np.max(np.abs(out["total_log_actions"][0, 0] - other["total_log_actions"][0, 0])), 1e-6)
        np.testing.assert_array_equal(out["total_log_actions"][0, 1:], other["total_log_actions"][0, 1:])

    def test_finite_and_risk_first_whole_controllers_match_independent_oracle(self):
        rng = np.random.default_rng(1051)
        actual = rng.uniform(-.45, .6, (3, 4, 3)); pi = rng.uniform(-.02, .07, (3, 4))
        train = rng.uniform(-.4, .6, (3, 4, 3)); train_pi = rng.uniform(-.02, .08, (3, 4))
        prices = [0., .02, .3, np.inf]
        for enabled in (False, True):
            out = simulate_controllers(actual, pi, train, train_pi, TARGETS, prices, L=.25, updates_enabled=enabled)
            for i in range(len(actual)):
                for j, eta in enumerate(prices):
                    scalar = independent_controller(actual[i], pi[i], train, train_pi, TARGETS, 1, eta, enabled, .25, .0025)
                    for key in ("holdings", "real_wealth", "real_peak", "running_mdd", "fees", "moved", "qg_actions", "selected_score"):
                        np.testing.assert_allclose(out[key][i, j], scalar[key], atol=5e-13, rtol=0, equal_nan=True, err_msg=f"{i}/{j}/{key}")
                    for key in ("ever_breached", "action_index", "current_target_before", "current_target_after", "target_updated", "qbad_actions"):
                        np.testing.assert_array_equal(out[key][i, j], scalar[key], err_msg=key)
            np.testing.assert_allclose(out["fees"], .0025 * out["moved"], atol=3e-15, rtol=0)
            np.testing.assert_allclose(out["posttrade_holdings"].sum(axis=-1) + out["fees"], out["pretrade_wealth"], atol=2e-13, rtol=0)
            changed_action = out["action_index"][..., 1:] > 0
            finite = np.isfinite(np.asarray(prices))[None, :, None]
            self.assertTrue(np.all(out["selected_score"][..., 1:][changed_action & finite] > 1e-12))
            np.testing.assert_array_equal(out["current_target_index"][..., 1:], out["current_target_after"])

    def test_positive_filter_precedes_global_tie_priority(self):
        # Current Balanced restore.9e-12 is numerically tied with another
        # action1.1e-12 but fails the strict positive-benefit requirement.
        qg = [[0., 1.1e-12, .9e-12, -1.]]
        action, scores, _ = choose_actions(qg, np.zeros((1, 4), int), np.ones((1, 4), bool), 2, [0.], 30, [1])
        self.assertEqual(action[0], 1)
        self.assertGreater(scores[0, action[0]], 1e-12)
        lex, _, _ = choose_actions(qg, np.zeros((1, 4), int), np.ones((1, 4), bool), 2, [np.inf], 30, [1])
        self.assertEqual(lex[0], 1)

    def test_current_target_then_CBG_ties_and_risk_first_lexicography(self):
        available = np.ones((1, 4), bool)
        choice, _, _ = choose_actions([[0., .1, .1, .1]], np.zeros((1, 4), int), available, 2, [0.], 5, [1])
        self.assertEqual(choice[0], 2)  # current restore wins over lower C ID.
        choice, _, _ = choose_actions([[0., .1, .1, .1]], np.ones((1, 4), int), available, 2, [np.inf], 5, [1])
        self.assertEqual(choice[0], 2)  # same priority after lex risk minimization.
        choice, _, _ = choose_actions([[0., .1, -.1, .1]], np.zeros((1, 4), int), available, 2, [0.], 5, [1])
        self.assertEqual(choice[0], 1)  # other C before G.
        choice, scores, _ = choose_actions([[0., -2., 3., -1.]], np.array([[2, 0, 2, 1]]), available, 3, [np.inf], 30, [1])
        self.assertEqual(choice[0], 1)  # min risk beats growth, even negative.
        self.assertTrue(np.isnan(scores).all())
        choice, _, _ = choose_actions([[0., -.1, 1e-13, -.1]], np.zeros((1, 4), int), available, 3, [np.inf], 30, [1])
        self.assertEqual(choice[0], 0)  # equal risk + no positive growth ->hold.

    def test_probability_fraction_original_H_and_unsigned_risk_reduction(self):
        qg = [[0., -.005, -.2, -.2]]
        qb = np.array([[2, 1, 2, 2]], dtype=np.uint64)
        choice, scores, _ = choose_actions(qg, qb, np.ones((1, 4), bool), 2, [.01], 2, [1])
        self.assertEqual(choice[0], 1)
        self.assertAlmostEqual(scores[0, 1], .005, places=15)
        choice, scores, _ = choose_actions([[0., .05, -.2, -.2]], np.array([[0, 1, 0, 0]]),
            np.ones((1, 4), bool), 100, [.01], 10, [1])
        self.assertEqual(choice[0], 1); self.assertAlmostEqual(scores[0, 1], .049, places=15)

    def test_metadata_target_update_can_have_zero_money_moved(self):
        targets = np.array([[.25, 0, .75], [.5, 0, .5], [.75, 0, .25]])
        actual = np.zeros((1, 3, 3)); actual[0, 0, 0] = 2.
        train = np.zeros((1, 3, 3)); train[0, 2, 0] = 1.
        out = simulate_controllers(actual, np.zeros((1, 3)), train, np.zeros((1, 3)), targets, [0.], L=.4)
        self.assertEqual(out["action_index"][0, 0, 1], 3)
        self.assertEqual(out["current_target_before"][0, 0, 1], 1)
        self.assertEqual(out["current_target_after"][0, 0, 1], 2)
        self.assertTrue(out["target_updated"][0, 0, 1])
        self.assertTrue(out["target_execution"][0, 0, 1])
        self.assertFalse(out["trades"][0, 0, 1])
        self.assertEqual(out["moved"][0, 0, 1], 0)
        self.assertEqual(out["fees"][0, 0, 1], 0)
        self.assertGreater(out["selected_score"][0, 0, 1], 1e-12)
        np.testing.assert_array_equal(out["current_target_index"][0, 0], [1, 1, 2, 2])
        self.assertEqual(out["action_index"][0, 0, 2], 0)

    def test_initial_fee_final_trigger_and_pending_never_update_or_reenter(self):
        target = np.array([20 / 99.95, 0, 79.95 / 99.95])
        targets = np.array([[.1, .1, .8], target, [.7, .2, .1]])
        out = simulate_controllers(np.zeros((1, 1, 3)), np.zeros((1, 1)), np.zeros((2, 1, 3)), np.zeros((2, 1)), targets,
                                  [0., .1, np.inf], L=.0001)
        np.testing.assert_allclose(out["fees"], .05, atol=2e-15, rtol=0)
        np.testing.assert_allclose(out["moved"], 20, atol=2e-13, rtol=0)
        np.testing.assert_array_equal(out["first_crossing"], 1)
        np.testing.assert_array_equal(out["available_followup"], 0)
        np.testing.assert_array_equal(out["right_censored"], True)
        np.testing.assert_array_equal(out["target_updated"], False)
        self.assertTrue(np.isnan(out["qg_actions"]).all())
        pending = simulate_controllers(np.zeros((1, 3, 3)), [[.25, -.2, .1]], np.zeros((1, 3, 3)), np.zeros((1, 3)),
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]], [0., .1, np.inf], L=.1)
        np.testing.assert_array_equal(pending["current_target_index"], 1)
        np.testing.assert_array_equal(pending["target_updated"], False)
        np.testing.assert_array_equal(pending["action_index"][..., 1:], 0)
        np.testing.assert_array_equal(pending["ever_breached"][0], [[False, True, True, True]] * 3)
        np.testing.assert_array_equal(pending["recovery_time"], 1)

    def test_current_future_actual_returns_do_not_affect_an_earlier_gate(self):
        rng = np.random.default_rng(933)
        actual = rng.uniform(-.1, .2, (2, 4, 3)); pi = np.zeros((2, 4))
        train = rng.uniform(-.25, .35, (3, 4, 3)); train_pi = np.zeros((3, 4))
        changed, changed_pi = actual.copy(), pi.copy()
        changed[:, 2:] = [[-.9, 2, .1], [.5, -.6, 0]]; changed_pi[:, 2:] = [.3, -.1]
        a = simulate_controllers(actual, pi, train, train_pi, TARGETS, [0., .03, np.inf], L=.4)
        b = simulate_controllers(changed, changed_pi, train, train_pi, TARGETS, [0., .03, np.inf], L=.4)
        for key in ("action_index", "current_target_before", "current_target_after", "qg_actions", "qbad_actions", "selected_score"):
            np.testing.assert_array_equal(a[key][:, :, :3], b[key][:, :, :3])
        np.testing.assert_array_equal(a["holdings"][..., :3, :], b["holdings"][..., :3, :])

    def test_disabled_unavailable_actions_are_not_zero_risk_options(self):
        train = np.zeros((2, 2, 3))
        out = evaluate_actions([40, 30, 30], 1, 100, 0, False, train, np.zeros((2, 2)),
            year_index=0, target_matrix=TARGETS, current_target_index=1, updates_enabled=False, L=.4)
        np.testing.assert_array_equal(out["available_actions"], [[True, False, True, False]])
        np.testing.assert_array_equal(out["qbad_actions"][0, [1, 3]], -1)
        self.assertTrue(np.isnan(out["qg_actions"][0, [1, 3]]).all())
        choice, _, _ = choose_actions(out["qg_actions"], out["qbad_actions"], out["available_actions"], 2, [np.inf], 2, [1])
        self.assertIn(choice[0], [0, 2])

    def test_validation_immutability_and_readonly_contract(self):
        actual, train = np.zeros((1, 3, 3)), np.ones((2, 3, 3)) * .01
        pi, train_pi = np.zeros((1, 3)), np.zeros((2, 3)); targets = TARGETS.copy(); prices = np.array([0., .1, np.inf])
        inputs = [actual, pi, train, train_pi, targets, prices]; saved = [a.copy() for a in inputs]
        out = simulate_controllers(*inputs, L=.4, horizon=2)
        self.assertEqual(out["real_wealth"].shape, (1, 3, 3)); self.assertEqual(out["cpi"].shape, (1, 3))
        for a, original in zip(inputs, saved): np.testing.assert_array_equal(a, original)
        for value in out.values():
            if isinstance(value, np.ndarray): self.assertFalse(value.flags.writeable)
        for change in (dict(initial_target_index=True), dict(initial_target_index=3), dict(initial_target_index=1.),
                       dict(updates_enabled=1), dict(horizon=4), dict(horizon=True), dict(L=True)):
            with self.assertRaises(ValueError): simulate_controllers(*inputs, L=.4, **change) if "L" not in change else simulate_controllers(*inputs, **change)
        with self.assertRaises(ValueError): simulate_controllers(actual, pi, train, train_pi, [[1, 0, 0]], prices)
        with self.assertRaises(ValueError): choose_actions([[0, 0, 0, 0]], np.zeros((1, 4), int), [[False] * 4], 2, [0.], 3, [1])


if __name__ == "__main__": unittest.main()
