"""Financial identities and causal synthetic ablation checks; no market reads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_fixed_execution_ablation", ROOT / "scripts/run_fixed_execution_ablation.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
from safeinvest_research.static import simulate_annual_targets


class FinancialAndCausalTests(unittest.TestCase):
    def market(self):
        returns = np.array([[[.1, -.03, .01], [-.2, .05, .02], [.3, .01, 0]],
                            [[-.1, .02, .01], [.2, -.04, 0], [.1, .01, .03]]])
        return returns, np.array([[.02, .04, -.01], [.01, .02, .03]]), np.array([[.5, .25, .25], [1., 0, 0], [0, 0, 1.]])

    def test_annual_matches_independent_static_engine_every_accounting_field(self):
        market, inflation, targets = self.market()
        actual = RUNNER.simulate_execution(market, inflation, targets, "annual_restore")
        reference = simulate_annual_targets(market, inflation, targets)
        for key in ["nominal_wealth", "real_wealth", "fees", "moved", "trades", "targets"]:
            assert_allclose(actual[key], reference[key], rtol=0, atol=1e-12)
        self.assertTrue(actual["target_execution"].all())
        assert_allclose(actual["fee_fraction"], actual["fees"] / actual["pretrade_wealth"], rtol=0, atol=0)

    def test_common_initial_fee_and_hold_closed_form_without_later_trades(self):
        market, inflation, targets = self.market()
        annual = RUNNER.simulate_execution(market, inflation, targets, "annual_restore")
        hold = RUNNER.simulate_execution(market, inflation, targets, "initial_then_hold")
        assert_allclose(annual["fees"][..., 0], hold["fees"][..., 0], atol=0, rtol=0)
        assert_allclose(annual["nominal_wealth"][..., :2], hold["nominal_wealth"][..., :2], atol=0, rtol=0)
        self.assertFalse(hold["trades"][..., 1:].any())
        self.assertFalse(hold["target_execution"][..., 1:].any())
        self.assertLess(RUNNER.verify_hold_closed_form(hold, market, inflation, .0025), 1e-11)
        # A complete switch from bills solves post=100/(1+c), not 100*(1-c).
        assert_allclose(hold["fees"][:, 1, 0], 100 * .0025 / 1.0025, atol=1e-13, rtol=0)
        self.assertTrue(np.all(hold["real_wealth"][..., 0] == 100))

    def test_known_end_weights_and_tv_drift_are_post_return_not_post_trade(self):
        market = np.array([[[1., 0, 0], [1., 0, 0]]])
        inflation = np.zeros((1, 2))
        target = [.5, 0, .5]
        annual = RUNNER.simulate_execution(market, inflation, target, "annual_restore")
        hold = RUNNER.simulate_execution(market, inflation, target, "initial_then_hold")
        assert_allclose(hold["end_weights"][0, 0], [[2 / 3, 0, 1 / 3], [.8, 0, .2]], atol=1e-14, rtol=0)
        assert_allclose(hold["end_year_tv_drift"][0, 0], [1 / 6, .3], atol=1e-14, rtol=0)
        assert_allclose(annual["end_year_tv_drift"][0, 0], [1 / 6, 1 / 6], atol=1e-14, rtol=0)
        self.assertAlmostEqual(RUNNER.horizon_values(hold, 2)["max_end_year_tv_drift"][0, 0], .3)
        self.assertAlmostEqual(RUNNER.horizon_values(hold, 2)["trade_fraction"][0, 0], .5)
        self.assertGreater(annual["fees"][0, 0, 1], 0)
        self.assertEqual(hold["fees"][0, 0, 1], 0)

    def test_future_returns_cannot_change_past_actions_or_current_pretrade_fee(self):
        market, inflation, targets = self.market()
        changed = market.copy()
        changed[:, 1:] = [[.7, -.5, .04], [-.8, .1, .01]]
        for policy in RUNNER.POLICIES:
            before = RUNNER.simulate_execution(market, inflation, targets, policy)
            after = RUNNER.simulate_execution(changed, inflation, targets, policy)
            assert_array_equal(before["nominal_wealth"][..., :2], after["nominal_wealth"][..., :2])
            assert_array_equal(before["fees"][..., :2], after["fees"][..., :2])
            assert_array_equal(before["moved"][..., :2], after["moved"][..., :2])
            assert_array_equal(before["pretrade_wealth"][..., :2], after["pretrade_wealth"][..., :2])

    def test_identical_asset_gross_has_no_restoration_transfer_or_drift(self):
        market = np.full((2, 4, 3), .02)
        infl = np.zeros((2, 4))
        target = [.4, .2, .4]
        annual = RUNNER.simulate_execution(market, infl, target, "annual_restore")
        hold = RUNNER.simulate_execution(market, infl, target, "initial_then_hold")
        assert_allclose(annual["real_wealth"], hold["real_wealth"], atol=1e-12, rtol=0)
        self.assertFalse(annual["trades"][..., 1:].any())
        assert_allclose(annual["end_year_tv_drift"], 0, atol=1e-14, rtol=0)
        self.assertEqual(RUNNER.horizon_values(annual, 4)["target_execution_fraction"][0, 0], 1)
        self.assertEqual(RUNNER.horizon_values(hold, 4)["target_execution_fraction"][0, 0], .25)

    def test_invalid_execution_inputs_fail_before_trading(self):
        market, inflation, targets = self.market()
        bad = market.copy(); bad[0, 0, 0] = -1
        for args in [(bad, inflation, targets, "annual_restore"),
                     (market, inflation, [.6, .6, 0], "annual_restore"),
                     (market, inflation, targets, "forecast_gate")]:
            with self.assertRaises(ValueError):
                RUNNER.simulate_execution(*args)


class PairingAndReferralTests(unittest.TestCase):
    def test_pathwise_quantile_difference_is_not_difference_of_marginal_quantiles(self):
        rows = []
        for policy, values in [("annual_restore", [0., .5]), ("initial_then_hold", [.5, 0.])]:
            for i, value in enumerate(values):
                rows.append({"context": "constructed", "scenario_id": i + 1, "policy": policy,
                             **{field: value if field == "cagr" else 0. for field in RUNNER.VALUE_FIELDS}})
        frame = pd.DataFrame(rows)
        paired = RUNNER.paired_differences(frame, ["context", "scenario_id"])
        summary = RUNNER.summarize_paired(paired, ["context"]).iloc[0]
        self.assertEqual(summary["mean_delta_cagr"], 0.)
        self.assertEqual(summary["p10_delta_cagr"], -.5)
        marginal_p10_difference = np.quantile([.5, 0], .1, method="inverted_cdf") - np.quantile([0, .5], .1, method="inverted_cdf")
        self.assertNotEqual(summary["p10_delta_cagr"], marginal_p10_difference)

    def test_frozen_review_is_null_in_both_branches_and_paired_summary(self):
        market = np.zeros((2, 2, 3)); inflation = np.zeros((2, 2))
        fit = {"assignments": {"A": "Review", "B": "Conservative"},
               "menu": [{"name": "Conservative", "candidate_index": 7, "weights": [.5, 0, .5]}]}
        profiles = [{"name": name, "horizon": 2, "loss_tolerance": .1} for name in ["A", "B"]]
        rows = []
        for policy in RUNNER.POLICIES:
            sim = RUNNER.simulate_execution(market, inflation, [.5, 0, .5], policy)
            rows.extend(RUNNER.client_rows({2: RUNNER.horizon_values(sim, 2)}, fit, profiles, [1, 2],
                                          context="full", scope="full", fold=None, policy=policy))
        frame = pd.DataFrame(rows)
        self.assertTrue(frame.loc[frame.client == "A", list(RUNNER.VALUE_FIELDS) + ["breach", "excess"]].isna().all().all())
        summary = RUNNER.summarize(frame, ["scope", "client", "policy"], clients=True)
        a = summary.loc[summary.client == "A"]
        self.assertTrue((a.review_paths == 2).all())
        self.assertTrue(a.breach_count.isna().all())
        paired = RUNNER.paired_differences(frame, ["context", "scenario_id", "client", "candidate_index"], clients=True)
        paired_summary = RUNNER.summarize_paired(paired, ["client"], clients=True)
        a = paired_summary.loc[paired_summary.client == "A"].iloc[0]
        self.assertEqual(a["assigned_pairs"], 0)
        self.assertEqual(a["review_pairs"], 2)
        self.assertTrue(pd.isna(a["new_breach_count"]))
        self.assertTrue(pd.isna(a["mean_delta_log_growth"]))

    def test_pairing_rejects_unmatched_paths_or_reassignment(self):
        rows = [{"scenario_id": scenario, "policy": policy, "assigned": True,
                 **{field: 0. for field in RUNNER.VALUE_FIELDS}, "excess": 0., "breach": False}
                for scenario in [1, 2] for policy in RUNNER.POLICIES]
        frame = pd.DataFrame(rows)
        with self.assertRaises(ValueError):
            RUNNER.paired_differences(frame.iloc[:-1], ["scenario_id"])
        frame.loc[(frame.scenario_id == 1) & (frame.policy == "initial_then_hold"), "assigned"] = False
        with self.assertRaises(ValueError):
            RUNNER.paired_differences(frame, ["scenario_id"], clients=True)


class SyntheticEndToEndTests(unittest.TestCase):
    def test_frozen_full_and_fold_pipeline_has_only_guarded_mocked_market_access(self):
        protocol = json.loads(RUNNER.primary.PROTOCOL.read_text())
        weights = RUNNER.primary.simplex_grid(.05)
        returns = np.broadcast_to(np.array([.02, .01, .003]), (120, 30, 3)).copy()
        inflation = np.full((120, 30), .005)
        ids = np.arange(1, 121)
        simulation = simulate_annual_targets(returns, inflation, weights)
        arrays = {"scenario_ids": ids, "weights": weights}; metrics = {}
        for horizon in protocol["horizons"]:
            metrics[horizon] = RUNNER.path_metrics(simulation["real_wealth"][..., :horizon + 1])
            arrays[f"H{horizon}__fees"] = simulation["fees"][..., :horizon].sum(axis=-1)
            arrays[f"H{horizon}__moved"] = simulation["moved"][..., :horizon].sum(axis=-1)
            arrays[f"H{horizon}__trade_fraction"] = simulation["trades"][..., :horizon].mean(axis=-1)
        candidates = [0, 50, 231]
        menu = [{"name": role, "candidate_index": index, "weights": weights[index].tolist()}
                for role, index in zip(["Conservative", "Balanced", "Growth"], candidates)]
        fit = {"alpha": .10, "liquidity_case": "risk_only", "menu": menu, "n_training": 120,
               "liquidity_floors": protocol["liquidity_cases"]["risk_only"],
               "assignments": {"A": "Review", "B": "Conservative", "C": "Balanced", "D": "Review", "E": "Growth"}}
        folds = []
        for item in protocol["validation"]["folds"]:
            folds.append({**copy.deepcopy(fit), **item, "n_training": 96})
            # Real primary fold fits record floors, not the full-fit-only case label.
            folds[-1].pop("liquidity_case")
        original_rows = []
        fields = {h: RUNNER.horizon_values(RUNNER.simulate_execution(returns, inflation, weights[candidates], "annual_restore"), h)
                  for h in protocol["horizons"]}
        for fold in folds:
            for profile in protocol["clients"]:
                assigned = fit["assignments"][profile["name"]]
                item = next((p for p in menu if p["name"] == assigned), None)
                for scenario in fold["test_ids"]:
                    index = scenario - 1
                    row = {"case": RUNNER.CASE, "fold": fold["fold"], "scenario_id": scenario,
                           "client": profile["name"], "assignment": assigned, "assigned": item is not None,
                           "candidate_index": None, "log_growth": None, "cagr": None, "mdd": None, "breach": None, "excess": None}
                    if item:
                        col = candidates.index(item["candidate_index"])
                        row.update({field: float(fields[profile["horizon"]][field][index, col]) for field in ["log_growth", "cagr", "mdd"]})
                        row["candidate_index"] = item["candidate_index"]
                        row["breach"] = row["mdd"] > profile["loss_tolerance"] + 1e-12
                        row["excess"] = max(row["mdd"] - profile["loss_tolerance"], 0.)
                    original_rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / "inputs"; out = base / "outputs"
            source.mkdir()
            for file, value in [("full_development_menus.json", {RUNNER.CASE: fit}), ("fit_manifest.json", {}),
                                ("validation_fold_menus.json", {RUNNER.CASE: folds}),
                                ("validation_manifest.json", {"sources": {"synthetic": "fixed"}})]:
                (source / file).write_text(json.dumps(value))
            pd.DataFrame(original_rows).to_csv(source / "validation_oof_paths.csv", index=False)
            cache = source / "cache" / f"{RUNNER.primary.cache_key(.05, .0025)}.npz"
            cache.parent.mkdir(); np.savez_compressed(cache, **arrays)
            digest = hashlib.sha256()
            for value in [ids, returns, inflation]: digest.update(np.ascontiguousarray(value).tobytes())
            cache.with_suffix(".json").write_text(json.dumps({"development_arrays_sha256": digest.hexdigest()}))
            with patch.object(RUNNER.primary, "OUT", source), \
                    patch.object(RUNNER.primary, "get_cache", return_value=(arrays, metrics)), \
                    patch.object(RUNNER.primary, "compatible_full_fit", return_value={RUNNER.CASE: fit}), \
                    patch.object(RUNNER.primary, "source_manifest", return_value={"synthetic": "fixed"}), \
                    patch.object(RUNNER, "input_manifest", return_value={"synthetic_inputs": "fixed"}), \
                    patch.object(RUNNER, "source_manifest", return_value={"synthetic": "fixed"}), \
                    patch.object(RUNNER, "load_paths", return_value=(returns, inflation, ids)) as loader, \
                    patch.object(sys, "argv", ["run_fixed_execution_ablation.py", "--out", str(out)]), \
                    patch("builtins.print"):
                RUNNER.main()
            self.assertEqual(list(loader.call_args.args[1]), list(range(1, 121)))
            self.assertIs(loader.call_args.kwargs["allow_final"], False)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["final_scenarios_loaded"])
            self.assertEqual(manifest["development_arrays_sha256"], digest.hexdigest())
            for file, expected in manifest["outputs_sha256"].items():
                self.assertEqual(RUNNER.sha(out / file), expected)
            checks = json.loads((out / "accounting_checks.json").read_text())
            self.assertTrue(checks["original_oof"]["annual_oof_matches"])
            self.assertEqual(checks["original_oof"]["review_records"], 240)
            paths = pd.read_csv(out / "client_path_metrics.csv")
            self.assertEqual(len(paths), 2400)
            self.assertTrue(paths.loc[~paths.assigned, list(RUNNER.VALUE_FIELDS)].isna().all().all())


if __name__ == "__main__":
    unittest.main()
