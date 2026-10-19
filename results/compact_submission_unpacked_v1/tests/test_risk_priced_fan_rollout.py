"""Independent synthetic tests for technical risk prices, never market tuning."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from experimental_execution.risk_priced_fan_rollout import priced_choices, simulate_prices
from experimental_execution.fan_rollout import simulate_policy
from test_fan_rollout import scalar_fan, scalar_restore


def independent_priced_policy(actual, inflation, train, train_pi, target, loss, cost, eta):
    """Own scalar path/fee oracle using no production step/choice/evaluator.

    scalar_fan/restore are independent100-step bisection references from the
    earlier synthetic tests, not wrappers around any numerical kernel.
    """
    original_h = len(actual)
    x, cp, peak, worst, bad, tau = np.array([0., 0., 100.]), 1., 100., 0., False, -1
    result = {key: [] for key in ("holdings", "nominal_wealth", "real_wealth", "real_peak",
        "current_drawdown", "running_mdd", "ever_breached", "cpi", "fees", "moved", "actions",
        "qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "predicted_score")}
    for key, value in (("holdings", x.copy()), ("nominal_wealth", 100.), ("real_wealth", 100.),
        ("real_peak", 100.), ("current_drawdown", 0.), ("running_mdd", 0.), ("ever_breached", False), ("cpi", 1.)):
        result[key].append(value)
    for t in range(original_h):
        qh = qr = score = np.nan; bh = br = -1
        if t == 0:
            restore = True
        elif bad:
            restore = False
        else:
            hl, hb = scalar_fan(x, cp, peak, worst, train, train_pi, t, target, loss, cost, "hold")
            rl, rb = scalar_fan(x, cp, peak, worst, train, train_pi, t, target, loss, cost, "restore")
            qh, qr, bh, br = hl.mean(), rl.mean(), int(hb.sum()), int(rb.sum())
            if np.isinf(eta):
                restore = br < bh or (br == bh and qr - qh > 1e-12)
            else:
                score = qr - qh - original_h * eta * ((br - bh) / len(train))
                restore = score > 1e-12
        pre = x.copy()
        if restore:
            post, fee = scalar_restore(x, target, cost)
        else:
            post, fee = x, 0.
        moved = sum(max(float(a) - float(b), 0.) for a, b in zip(post, pre))
        x = post * (1 + actual[t]); cp *= 1 + inflation[t]
        nominal = sum(map(float, x)); real = nominal / cp
        peak = max(peak, real); dd = max(1 - real / peak, 0.); worst = max(worst, dd)
        bad = bad or worst - loss > 1e-12
        if tau < 0 and bad: tau = t + 1
        for key, value in (("holdings", x.copy()), ("nominal_wealth", nominal), ("real_wealth", real),
            ("real_peak", peak), ("current_drawdown", dd), ("running_mdd", worst), ("ever_breached", bad),
            ("cpi", cp), ("fees", fee), ("moved", moved), ("actions", "restore" if restore else "hold"),
            ("qg_hold", qh), ("qg_restore", qr), ("qbad_hold_count", bh), ("qbad_restore_count", br),
            ("predicted_score", score)):
            result[key].append(value)
    result = {key: np.asarray(value) for key, value in result.items()}
    result["first_crossing"] = tau
    return result


class RiskPricedFanTests(unittest.TestCase):
    def test_zero_price_reproduces_every_available_growth_only_trace(self):
        rng = np.random.default_rng(822)
        actual = rng.uniform(-.35, .45, (3, 5, 3)); pi = rng.uniform(-.02, .06, (3, 5))
        train = rng.uniform(-.4, .5, (4, 5, 3)); train_pi = rng.uniform(-.02, .08, (4, 5))
        w = [.4, .35, .25]
        prior = simulate_policy(actual, pi, train, train_pi, w, L=.25, policy="growth_only_rollout")
        priced = simulate_prices(actual, pi, train, train_pi, w, [0., .03, np.inf], L=.25)
        checked = 0
        for key, original in prior.items():
            if (not isinstance(original, np.ndarray) or key not in priced
                    or key in ("targets", "loss_tolerances", "gate_reason")):
                continue
            value = priced[key] if key == "cpi" else priced[key][:, 0]
            if original.dtype.kind == "f":
                np.testing.assert_allclose(value, original, atol=0, rtol=0, equal_nan=True, err_msg=key)
            else:
                np.testing.assert_array_equal(value, original, err_msg=key)
            checked += 1
        self.assertGreaterEqual(checked, 35)
        self.assertEqual(priced["cpi"].shape, (3, 6))
        self.assertEqual(priced["real_wealth"].shape, (3, 3, 6))

    def test_finite_and_risk_first_full_policy_match_independent_scalar_replay(self):
        rng = np.random.default_rng(1058)
        actual = rng.uniform(-.5, .65, (3, 4, 3)); pi = rng.uniform(-.03, .08, (3, 4))
        train = rng.uniform(-.45, .7, (4, 4, 3)); train_pi = rng.uniform(-.02, .08, (4, 4))
        target, prices = np.array([.5, .3, .2]), np.array([0., .01, .3, np.inf])
        out = simulate_prices(actual, pi, train, train_pi, target, prices, L=.2)
        for i in range(len(actual)):
            for j, eta in enumerate(prices):
                scalar = independent_priced_policy(actual[i], pi[i], train, train_pi, target, .2, .0025, eta)
                for key in ("holdings", "nominal_wealth", "real_wealth", "real_peak", "current_drawdown",
                    "running_mdd", "fees", "moved", "qg_hold", "qg_restore", "predicted_score"):
                    np.testing.assert_allclose(out[key][i, j], scalar[key], atol=4e-13, rtol=0, equal_nan=True, err_msg=f"{i}/{j}/{key}")
                for key in ("actions", "ever_breached", "qbad_hold_count", "qbad_restore_count"):
                    np.testing.assert_array_equal(out[key][i, j], scalar[key])
                np.testing.assert_allclose(out["cpi"][i], scalar["cpi"], atol=0, rtol=0)
                self.assertEqual(out["first_crossing"][i, j], scalar["first_crossing"])
        np.testing.assert_allclose(out["fees"], .0025 * out["moved"], atol=3e-15, rtol=0)
        np.testing.assert_allclose(out["posttrade_holdings"].sum(axis=-1) + out["fees"], out["pretrade_wealth"], atol=2e-13, rtol=0)

    def test_score_uses_original_horizon_and_probability_fraction(self):
        decision, score, _ = priced_choices(0., .05, np.array(0), np.array(1), 100, .01, 10)
        self.assertAlmostEqual(float(score), .049, places=14)
        self.assertTrue(bool(decision))  # Counting1 instead of1/100 reverses this.
        decision, score, _ = priced_choices(0., .05, np.array(0), np.array(1), 2, .02, 10)
        self.assertAlmostEqual(float(score), -.05, places=14)
        self.assertFalse(bool(decision))  # H=1 would incorrectly restore.

    def test_last_decision_keeps_original_H_and_can_sacrifice_growth_for_risk(self):
        actual = np.array([[[3., 0, 0], [0, 0, 0], [0, 0, 0]]])
        train = np.zeros((2, 3, 3)); train[:, 2, 0] = [2., -.5]
        out = simulate_prices(actual, np.zeros((1, 3)), train, np.zeros((2, 3)), [.5, 0, .5], [0., .02, np.inf], L=.3, cost_rate=0)
        j, t = 1, 2
        delta_g = out["qg_restore"][0, j, t] - out["qg_hold"][0, j, t]
        delta_p = (out["qbad_restore_count"][0, j, t] - out["qbad_hold_count"][0, j, t]) / 2
        self.assertLess(delta_g, 0)
        self.assertEqual(delta_p, -.5)
        self.assertGreater(out["predicted_score"][0, j, t], 0)
        self.assertAlmostEqual(out["predicted_score"][0, j, t], delta_g - 3 * .02 * delta_p, places=14)
        self.assertLess(delta_g - .02 * delta_p, 0)  # remaining H1 gives wrong action.
        self.assertEqual(out["actions"][0, j, t], "restore")
        self.assertEqual(out["actions"][0, 0, t], "hold")
        self.assertTrue(out["restoration_predicted_growth_sacrifice"][0, j, t])

    def test_risk_first_uses_count_then_growth_then_hold(self):
        qh = np.zeros(6); qr = np.array([-1., 1., 1e-13, .1, 0., -.1])
        bh, br = np.array([2, 1, 1, 1, 1, 1]), np.array([1, 2, 1, 1, 1, 1])
        chosen, score, reasons = priced_choices(qh, qr, bh, br, 3, np.inf, 30)
        np.testing.assert_array_equal(chosen, [True, False, False, True, False, False])
        self.assertTrue(np.isnan(score).all())
        self.assertEqual(reasons[0], "risk_first_lower_predicted_risk")
        self.assertEqual(reasons[1], "risk_first_higher_predicted_risk")
        self.assertEqual(reasons[2], "risk_first_equal_risk_hold_or_tie")

    def test_unsigned_decreasing_risk_count_does_not_wrap(self):
        bh, br = np.array([2], dtype=np.uint64), np.array([1], dtype=np.uint64)
        decision, score, _ = priced_choices([0.], [-.005], bh, br, 2, [.01], 2)
        self.assertAlmostEqual(score[0], .005, places=15)
        self.assertTrue(decision[0])
        lex, lex_score, _ = priced_choices([0.], [-100.], bh, br, 2, [np.inf], 2)
        self.assertTrue(lex[0]); self.assertTrue(np.isnan(lex_score[0]))

    def test_current_and_future_Q_fees_are_not_double_charged(self):
        actual = np.array([[[.5, 0, 0], [0, 0, 0]]])
        train = np.zeros((2, 2, 3))
        out = simulate_prices(actual, np.zeros((1, 2)), train, np.zeros((2, 2)), [.5, 0, .5], [0., .1, np.inf], L=.5)
        np.testing.assert_allclose(out["fees"][0, :, 0], 50 * .0025 / (1 + .5 * .0025), atol=2e-15, rtol=0)
        expected_restore = np.log1p(-.1 * .0025 / (1 + .5 * .0025))
        np.testing.assert_allclose(out["qg_restore"][0, :, 1], expected_restore, atol=1e-15, rtol=0)
        np.testing.assert_allclose(out["qg_hold"][0, :, 1], 0, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(out["actions"][0, :, 1], "hold")

    def test_initial_cost_H1_crossing_and_no_fabricated_Q_or_followup(self):
        target = [20 / 99.95, 0, 79.95 / 99.95]
        out = simulate_prices(np.zeros((1, 1, 3)), np.zeros((1, 1)), np.zeros((2, 1, 3)), np.zeros((2, 1)),
                              target, [0., .1, np.inf], L=.0001)
        np.testing.assert_allclose(out["fees"], .05, atol=2e-15, rtol=0)
        np.testing.assert_allclose(out["moved"], 20, atol=2e-13, rtol=0)
        np.testing.assert_allclose(out["real_wealth"][..., -1], 99.95, atol=2e-13, rtol=0)
        np.testing.assert_allclose(out["running_mdd"][..., -1], .0005, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(out["first_crossing"], 1)
        np.testing.assert_array_equal(out["right_censored"], True)
        np.testing.assert_array_equal(out["available_followup"], 0)
        np.testing.assert_array_equal(out["observed_followup"], 0)
        self.assertTrue(np.isnan(out["qg_hold"]).all()); self.assertTrue(np.isnan(out["predicted_score"]).all())
        np.testing.assert_array_equal(out["forecast_n"], -1)

    def test_history_breach_and_recovery_are_absorbing_for_every_price(self):
        out = simulate_prices(np.zeros((1, 3, 3)), [[.25, -.2, .1]], np.zeros((2, 3, 3)), np.zeros((2, 3)),
                              [0, 0, 1], [0., .1, np.inf], L=.1)
        np.testing.assert_array_equal(out["actions"][0], [["restore", "hold", "hold"]] * 3)
        np.testing.assert_array_equal(out["ever_breached"][0], [[False, True, True, True]] * 3)
        np.testing.assert_array_equal(out["first_crossing"], 1)
        np.testing.assert_array_equal(out["recovery_time"], 1)
        np.testing.assert_array_equal(out["right_censored"], False)
        np.testing.assert_array_equal(out["available_followup"], 2)
        np.testing.assert_array_equal(out["observed_followup"], 1)
        self.assertTrue(np.isnan(out["qg_restore"]).all())
        self.assertEqual(out["cpi"].shape, (1, 4))

    def test_actual_current_and_future_returns_are_not_forecast_inputs(self):
        rng = np.random.default_rng(107)
        actual = rng.uniform(-.1, .3, (2, 4, 3)); pi = np.zeros((2, 4))
        train = rng.uniform(-.3, .4, (3, 4, 3)); train_pi = np.zeros((3, 4))
        changed, changed_pi = actual.copy(), pi.copy()
        changed[:, 2:] = [[-.9, 2, .1], [.5, -.6, 0]]; changed_pi[:, 2:] = [.3, -.1]
        a = simulate_prices(actual, pi, train, train_pi, [.5, .3, .2], [0., .02, np.inf], L=.5)
        b = simulate_prices(changed, changed_pi, train, train_pi, [.5, .3, .2], [0., .02, np.inf], L=.5)
        for field in ("actions", "qg_hold", "qg_restore", "qbad_hold_count", "qbad_restore_count", "predicted_score"):
            np.testing.assert_array_equal(a[field][..., :3], b[field][..., :3])
        np.testing.assert_array_equal(a["holdings"][..., :3, :], b["holdings"][..., :3, :])

    def test_inputs_immutable_outputs_readonly_and_original_H_truncation(self):
        actual = np.zeros((2, 4, 3)); pi = np.zeros((2, 4)); train = np.ones((3, 4, 3)) * .02
        train_pi = np.zeros((3, 4)); prices = np.array([0., .03, np.inf]); w = np.array([.4, .3, .3])
        arrays = [actual, pi, train, train_pi, prices, w]; copies = [x.copy() for x in arrays]
        out = simulate_prices(*arrays[:4], w, prices, L=.5, horizon=2)
        self.assertEqual(out["nominal_wealth"].shape, (2, 3, 3))
        self.assertEqual(out["cpi"].shape, (2, 3))
        np.testing.assert_array_equal(out["real_wealth"][..., 0], 100)
        for a, before in zip(arrays, copies): np.testing.assert_array_equal(a, before)
        for value in out.values():
            if isinstance(value, np.ndarray): self.assertFalse(value.flags.writeable)
        with self.assertRaises(ValueError): out["annual_prices"][0] = 9

    def test_invalid_N_H_counts_Q_and_prices_are_rejected(self):
        args = [0., .1, np.array(0), np.array(1), 2, .01, 3]
        for index, value in ((4, 0), (4, True), (4, 2.), (6, 0), (6, True), (6, 3.),
                             (0, np.nan), (3, np.array(3)), (2, np.array(.5)), (2, np.array(-1)),
                             (5, -1.), (5, np.nan), (5, -np.inf), (5, True)):
            changed = args.copy(); changed[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError): priced_choices(*changed)
        market, pi = np.zeros((1, 2, 3)), np.zeros((1, 2))
        for prices in ([], [0, 0], [[0, .1]], .1, [np.nan], [-1.], [-np.inf], [True, False]):
            with self.subTest(prices=prices), self.assertRaises(ValueError):
                simulate_prices(market, pi, market, pi, [.4, .3, .3], prices, L=.3)
        for h in (0, 3, True, 1.5):
            with self.assertRaises(ValueError): simulate_prices(market, pi, market, pi, [.4, .3, .3], [0.], L=.3, horizon=h)
        with self.assertRaises(ValueError): simulate_prices(market, pi, market[:0], pi[:0], [.4, .3, .3], [0.], L=.3)
        bad = market.copy(); bad[0, 0, 0] = -1
        with self.assertRaises(ValueError): simulate_prices(bad, pi, market, pi, [.4, .3, .3], [0.], L=.3)


if __name__ == "__main__": unittest.main()
