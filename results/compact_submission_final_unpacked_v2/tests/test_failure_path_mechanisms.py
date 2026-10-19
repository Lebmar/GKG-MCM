"""Synthetic fixed-account identities; no real market file is opened."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

from diagnostics.failure_path_mechanisms import (Checks, first_difference, prepare_account,
    annual_records, endpoint_records, paired_records, cross_stage_summary, Q_FLOAT_FIELDS, Q_COUNT_FIELDS)
from safeinvest.core import execute_target

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("failure_mechanisms_runner", ROOT / "scripts/diagnose_failure_path_mechanisms.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def model_account(returns, inflation, target, desired_actions, *, loss=.4, cost=.0025, fan_n=6, price=.03):
    """Artificial fixed action trace, including synthetic matching Q scores."""
    r, z = np.asarray(returns, float), np.asarray(inflation, float)
    h = len(r); x, cp, pk, dd, bad, tau = np.array([0., 0., 100.]), 1., 100., 0., False, -1
    names = ("holdings", "weights", "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd", "ever_breached", "cpi",
             "posttrade_holdings", "fees", "moved", "pending_at_decision", "target_execution", "actions", "trades")
    out = {name: [] for name in names}
    for name in Q_FLOAT_FIELDS:
        out[name] = np.full(h, np.nan)
    for name in Q_COUNT_FIELDS:
        out[name] = np.full(h, -1, dtype=np.int64)
    out["gate_reason"] = np.full(h, "", dtype="U48")
    def state():
        V = float(x.sum()); real = V / cp
        for name, value in (("holdings", x.copy()), ("weights", x / V), ("nominal_wealth", V), ("real_wealth", real),
                            ("real_peak", pk), ("current_drawdown", max(1 - real / pk, 0.)), ("running_mdd", dd), ("ever_breached", bad), ("cpi", cp)):
            out[name].append(value)
    state()
    for t in range(h):
        action = "restore" if t == 0 else "hold" if bad else desired_actions[t]
        if t and not bad:
            advantage = .01 if action == "restore" else -.01
            for name, value in (("qg_hold", 0.), ("qg_restore", advantage), ("predicted_score", advantage),
                                ("predicted_growth_advantage_restore", advantage), ("qbad_hold_count", 0), ("qbad_restore_count", 0),
                                ("forecast_n", fan_n), ("predicted_breach_count_delta_restore", 0)):
                out[name][t] = value
        post, fee, moved = execute_target(x, target, cost) if action == "restore" else (x.copy(), 0., 0.)
        for name, value in (("posttrade_holdings", post.copy()), ("fees", fee), ("moved", moved), ("pending_at_decision", bad),
                            ("target_execution", action == "restore"), ("actions", action), ("trades", moved > 1e-12)):
            out[name].append(value)
        x = post * (1 + r[t]); cp *= 1 + z[t]
        real = float(x.sum()) / cp; pk = max(pk, real); dd = max(dd, max(1 - real / pk, 0.)); bad = bad or dd - loss > 1e-12
        if tau < 0 and bad:
            tau = t + 1
        state()
    out = {name: np.asarray(value) for name, value in out.items()}
    out["first_crossing"] = np.asarray(tau)
    return out


class FailurePathMechanismsTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([[.4, -.1, .01], [-.3, .08, .01], [.5, -.03, .01], [-.15, .04, .01]])
        self.z = np.array([.02, .03, -.01, .01])
        self.w = np.array([.55, .3, .15])
        self.raw_B = runner.reconstruct_B(self.r, self.z, self.w)
        self.raw_M = model_account(self.r, self.z, self.w, ["restore", "hold", "restore", "hold"])

    def B(self):
        return prepare_account(self.raw_B, self.r, self.z, self.w)

    def M(self):
        return prepare_account(self.raw_M, self.r, self.z, self.w, is_model=True, annual_price=.03, fan_n=6)

    def test_first_difference_multidimensional(self):
        self.assertEqual(first_difference([[0, 1], [2, 3]], [[0, 1], [2, 4]], atol=.1, offset=1), 2)

    def test_first_difference_null_and_tolerance(self):
        self.assertIsNone(first_difference([1, 2], [1, 2 + 1e-11], atol=1e-10))

    def test_first_difference_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            first_difference([1, np.nan], [1, 2], atol=0)

    def test_first_difference_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            first_difference([1, 2], [[1, 2]], atol=0)

    def test_B_retains_pending_after_recovery(self):
        r = np.array([[-.6, 0, 0], [3., 0, 0], [0., 0, 0], [0., 0, 0]])
        raw = runner.reconstruct_B(r, np.zeros(4), [1., 0., 0.])
        b = prepare_account(raw, r, np.zeros(4), [1., 0., 0.])
        self.assertEqual(b["first_crossing_value"], 1)
        self.assertGreater(b["real_wealth"][2], b["real_peak"][0])
        self.assertEqual(b["actions"].tolist(), ["restore", "hold", "hold", "hold"])

    def test_initial_fee_is_exact_and_charged(self):
        r = np.zeros((2, 3)); raw = runner.reconstruct_B(r, np.zeros(2), [1, 0, 0])
        self.assertAlmostEqual(raw["fees"][0], 100 * .0025 / 1.0025, places=13)
        self.assertEqual(len(raw["fees"]), 2)
        self.assertAlmostEqual(raw["fees"][1], 0., places=13)

    def test_whole_peak_cannot_be_reset(self):
        raw = copy.deepcopy(self.raw_B); raw["real_peak"][2] = raw["real_wealth"][2]
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w)

    def test_running_mdd_cannot_be_reset(self):
        raw = copy.deepcopy(self.raw_B); raw["running_mdd"][-1] = 0.
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w)

    def test_cost_cannot_be_double_counted(self):
        raw = copy.deepcopy(self.raw_B); raw["fees"][0] *= 2
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w)

    def test_common_market_checks_holdings(self):
        changed = self.r.copy(); changed[2, 0] += .01
        with self.assertRaises(ValueError):
            prepare_account(self.raw_B, changed, self.z, self.w)

    def test_model_Q_initial_and_pending_are_null(self):
        raw = copy.deepcopy(self.raw_M); raw["qg_hold"][0] = 0.
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w, is_model=True, annual_price=.03, fan_n=6)

    def test_model_Q_score_uses_original_H(self):
        raw = copy.deepcopy(self.raw_M)
        # A risk-reducing restoration can sacrifice predicted growth.
        t = 2; raw["qbad_hold_count"][t] = 2; raw["qbad_restore_count"][t] = 1
        raw["predicted_breach_count_delta_restore"][t] = -1
        raw["qg_restore"][t] = -.005; raw["predicted_growth_advantage_restore"][t] = -.005
        raw["predicted_score"][t] = -.005 + 4 * .03 / 6
        prepare_account(raw, self.r, self.z, self.w, is_model=True, annual_price=.03, fan_n=6)
        raw["predicted_score"][t] = -.005 + 2 * .03 / 6
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w, is_model=True, annual_price=.03, fan_n=6)

    def test_model_Q_invalid_fan_is_rejected(self):
        raw = copy.deepcopy(self.raw_M); raw["forecast_n"][1] = 5
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w, is_model=True, annual_price=.03, fan_n=6)

    def test_model_Q_action_must_match_score(self):
        raw = copy.deepcopy(self.raw_M); raw["predicted_score"][1] = .1
        with self.assertRaises(ValueError):
            prepare_account(raw, self.r, self.z, self.w, is_model=True, annual_price=.03, fan_n=6)

    def test_B_Q_export_is_always_null(self):
        rows = annual_records(self.B())
        self.assertTrue(all(row[k] is None for row in rows for k in (*Q_FLOAT_FIELDS, *Q_COUNT_FIELDS)))
        self.assertTrue(all(not row["q_available"] for row in rows))

    def test_complete_endpoint_and_period_lengths(self):
        self.assertEqual(len(endpoint_records(self.M())), 5)
        self.assertEqual(len(annual_records(self.M())), 4)

    def test_exact_log_fee_exposure_decomposition(self):
        m, b = self.M(), self.B()
        rows, summary, _ = paired_records(m, b, self.r, self.z)
        for row in rows:
            self.assertAlmostEqual(row["annual_real_log_growth_gap"], row["log_fee_drag_gap"] + row["log_posttrade_gross_gap"], places=13)
        self.assertAlmostEqual(summary["terminal_log_real_wealth_ratio"], summary["total_log_fee_drag_gap"] + summary["total_log_posttrade_gross_gap"], places=13)

    def test_exact_nominal_two_alternative_decompositions(self):
        rows, _, _ = paired_records(self.M(), self.B(), self.r, self.z)
        for row in rows:
            self.assertAlmostEqual(row["nominal_wealth_gap_end"], row["nominal_wealth_gap_start"] - row["nominal_fee_gap"] + row["nominal_return_pnl_gap"], places=12)
            self.assertAlmostEqual(row["nominal_wealth_gap_end"], row["inherited_wealth_gap_at_B_growth"] + row["fee_gap_at_B_growth"] + row["exposure_gap_at_model_postcapital"], places=12)

    def test_current_DD_ratio_identity_is_not_running_MDD(self):
        rows, _, _ = paired_records(self.M(), self.B(), self.r, self.z)
        self.assertTrue(any(abs(row["real_peak_gap_end"]) > 1e-10 for row in rows))
        for row in rows:
            self.assertAlmostEqual(row["current_drawdown_gap_end"], row["current_DD_wealth_ratio_part"] + row["current_DD_peak_ratio_part"], places=13)

    def test_initial_same_action_and_first_later_difference(self):
        _, summary, _ = paired_records(self.M(), self.B(), self.r, self.z)
        self.assertEqual(summary["first_action_divergence_year"], 2)
        self.assertEqual(summary["first_posttrade_weights_divergence_year"], 2)
        self.assertEqual(summary["first_real_wealth_divergence_endpoint"], 2)

    def test_identical_paths_have_no_difference_or_witnesses(self):
        b = self.B(); _, summary, witnesses = paired_records(b, b, self.r, self.z)
        self.assertIsNone(summary["first_action_divergence_year"])
        for name in summary:
            if name.startswith("first_"):
                self.assertIsNone(summary[name])
        self.assertEqual(witnesses, [])

    def test_crossing_witnesses_include_pre_crossing_zero_endpoint(self):
        r = np.array([[-.6, 0, 0], [3., 0, 0], [0., 0, 0]])
        raw = runner.reconstruct_B(r, np.zeros(3), [1., 0., 0.])
        b = prepare_account(raw, r, np.zeros(3), [1., 0., 0.])
        _, _, witnesses = paired_records(b, b, r, np.zeros(3))
        self.assertEqual(sorted({x["endpoint"] for x in witnesses}), [0, 1])
        self.assertEqual(len(witnesses), 4)

    def test_original_pending_prohibition(self):
        r = np.array([[-.6, 0, 0], [3., 0, 0]])
        raw = runner.reconstruct_B(r, np.zeros(2), [1., 0., 0.]); raw["pending_at_decision"][1] = False
        with self.assertRaises(ValueError):
            prepare_account(raw, r, np.zeros(2), [1., 0., 0.])

    def test_pair_CPI_mismatch_is_rejected(self):
        m, b = self.M(), self.B(); b["cpi"][2] += .01
        with self.assertRaises(ValueError):
            paired_records(m, b, self.r, self.z)

    def test_protocol_not_ready_fails_before_loading(self):
        with self.assertRaises(RuntimeError):
            runner.validate_protocol({"execution_ready": False})

    def test_cross_stage_comparison_describes_both_fixed_accounts(self):
        m, b = self.M(), self.B()
        row = cross_stage_summary(m, b)
        self.assertEqual(row["first_action_divergence_year"], 2)
        self.assertEqual(row["full_mdd"], float(m["running_mdd"][-1]))
        self.assertEqual(row["inner_mdd"], float(b["running_mdd"][-1]))
        self.assertEqual(row["full_terminal_real"], float(m["real_wealth"][-1]))

    def test_unfrozen_or_foreign_inventory_is_rejected(self):
        p = dict(execution_ready=True, data=dict(scenario_ids=list(range(1, 121)), allow_final=False),
                 original_H=30, loss_tolerance=.4, cost_rate=.0025,
                 output_directory="results/model_review/failure_path_mechanisms_v1", inventory=[])
        with self.assertRaises(ValueError):
            runner.validate_protocol(p)


if __name__ == "__main__":
    unittest.main()
