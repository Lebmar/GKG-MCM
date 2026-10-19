"""Synthetic adapter/specification/release gates, never original market data."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_evaluation_preparation", ROOT / "scripts/prepare_final_evaluation.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixtures():
    profiles = [dict(name=n, horizon=h, loss_tolerance=l, liquidity=q) for n, h, l, q in
                (("A", 5, .1, "high"), ("B", 15, .25, "medium"), ("C", 30, .4, "low"), ("D", 25, .15, "medium"), ("E", 6, .35, "high"))]
    registry = dict(current_C_execution=dict(final_status="execution_review", financial_outputs=None, H=30, L=.4,
        original_target=[.45, .05, .5], original_selected_eta=.03), safeguards=dict(automatic_target_updates=False),
        current_static_full120=dict(profile_Q1_assignments=dict(A="Review", B="Conservative", C="Balanced", D="Review", E="Growth"),
        portfolios=dict(Conservative=[.25, .3, .45], Balanced=[.45, .05, .5], Growth=[.65, .05, .3])))
    objects = m.object_specification(registry, profiles)
    p = dict(version="locked-final-evaluation-preparation-v1", development_execution_ready=False, final_execution_ready=False,
        development_ids=list(range(1, 121)), final_ids=list(range(121, 181)), development_allow_final=False,
        final_release_path=str(m.RELEASE), development_output_directory=str(m.DEV_OUT), final_output_directory=str(m.FINAL_OUT),
        object_specification=objects, object_specification_sha256=m.canonical_sha(objects),
        exact_reproduction_fields={name: [str(i) for i in range(n)] for name, n in
            (("unapproved_fixed_shadow", 48), ("same_target_B", 36), ("exact_thirds_annual", 36))},
        execution_sources_sha256={"synthetic_source.py": "1" * 64}, frozen_input_sha256={"synthetic_input.txt": "2" * 64})
    return registry, profiles, objects, p


def release_fixture(p):
    p = copy.deepcopy(p); p["development_execution_ready"] = True
    prep_sha, dev_sha = "a" * 64, "b" * 64
    release = dict(version="locked-final-evaluation-release-v1", final_execution_ready=True,
        preparation_protocol_sha256=prep_sha, execution_sources_sha256=p["execution_sources_sha256"],
        object_specification_sha256=p["object_specification_sha256"], development_gate_manifest_sha256=dev_sha,
        training_ids=list(range(1, 121)), evaluation_ids=list(range(121, 181)), output_directory=str(m.FINAL_OUT))
    dev = dict(status="passed", final_scenarios_loaded=False, preparation_protocol_sha256=prep_sha,
        object_specification_sha256=p["object_specification_sha256"], execution_sources_sha256=p["execution_sources_sha256"])
    audit = dict(status="passed", parent_manifest_sha256=dev_sha)
    return p, release, dev, audit, dict(preparation_sha256=prep_sha, development_manifest_sha256=dev_sha)


def synthetic_financial_rows(n):
    _, _, objects, _ = fixtures(); obj = objects["Q2_objects"][1]
    rows = []
    for i in range(1, n + 1):
        row = {k: obj[k] for k in m.GROUP_KEYS}
        row.update(scenario_id=i, scope="synthetic", execution_status=obj["execution_status"], case_status=obj["case_status"],
            q1_assignment="Balanced", liquidity="low", service_admitted=False, financial_available=True, q1_suitable=None)
        row.update({name: 0. for name in m.FINANCIAL})
        row.update(cagr=float(i), log_growth=float(np.log1p(i)), mdd=i / 100., volatility=.01, nominal_fees=.1,
                   trade_fraction=.2, breach=i > 40, first_crossing=30 if i > 40 else None)
        rows.append(row)
    return rows


class FinalEvaluationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.registry, self.profiles, self.objects, self.p = fixtures()

    def test_definition_preserves_four_Q2_and_five_own_profile_objects(self):
        m.validate_definition(self.p, self.registry, self.profiles)
        self.assertEqual(len(self.objects["Q2_objects"]), 4)
        self.assertEqual([x["horizon"] for x in self.objects["Q1_objects"]], [5, 15, 30, 25, 6])

    def test_reference_or_shadow_availability_is_not_service_admission(self):
        main, shadow, b, thirds = self.objects["Q2_objects"]
        self.assertFalse(main["financial_available"])
        self.assertTrue(shadow["financial_available"]); self.assertFalse(shadow["service_admitted"])
        self.assertIsNone(b["service_admitted"]); self.assertIsNone(thirds["service_admitted"])

    def test_both_draft_gates_are_closed(self):
        with self.assertRaises(RuntimeError):
            m.authorize_development(self.p)
        with self.assertRaises(RuntimeError):
            m.authorize_final(self.p, None, {}, {}, preparation_sha256="a", development_manifest_sha256="b")

    def test_changed_assignment_or_updates_cannot_become_another_model(self):
        bad = copy.deepcopy(self.registry); bad["safeguards"]["automatic_target_updates"] = True
        with self.assertRaises(ValueError):
            m.object_specification(bad, self.profiles)

    def test_missing_separate_final_release_blocks_before_any_loader(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(m, "preparation", return_value=(self.p, self.objects)), mock.patch("safeinvest.data.load_paths") as loader:
            with self.assertRaisesRegex(RuntimeError, "No separate final release"):
                m.final_evaluation(Path(tmp))
            loader.assert_not_called()
            self.assertFalse((Path(tmp) / m.FINAL_OUT).exists())

    def test_draft_development_gate_never_loads_or_creates_a_release(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(m, "preparation", return_value=(self.p, self.objects)), mock.patch("safeinvest.data.load_paths") as loader:
            with self.assertRaises(RuntimeError):
                m.development_gate(Path(tmp))
            loader.assert_not_called(); self.assertFalse((Path(tmp) / m.RELEASE).exists())

    def test_all_release_identity_bindings_are_required(self):
        args = release_fixture(self.p)
        m.authorize_final(*args[:4], **args[4])
        for field in ("preparation_protocol_sha256", "execution_sources_sha256", "object_specification_sha256", "development_gate_manifest_sha256", "training_ids", "evaluation_ids", "output_directory"):
            p, release, dev, audit, keywords = copy.deepcopy(args)
            release[field] = None
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                m.authorize_final(p, release, dev, audit, **keywords)

    def test_passed_dev_alone_and_wrong_audit_do_not_enable_final(self):
        p, release, dev, audit, kwargs = release_fixture(self.p)
        with self.assertRaises(RuntimeError):
            m.authorize_final(p, None, dev, audit, **kwargs)
        audit["parent_manifest_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            m.authorize_final(p, release, dev, audit, **kwargs)

    def test_final_release_cannot_enable_by_mutating_preparation_false_flag(self):
        p, release, dev, audit, kwargs = release_fixture(self.p)
        p["final_execution_ready"] = True
        with self.assertRaises(RuntimeError):
            m.authorize_final(p, release, dev, audit, **kwargs)

    def test_exact_array_checks_dtype_shape_values_and_paired_NULL(self):
        m.exact_array(np.array([1., np.nan]), np.array([1., np.nan]), "synthetic")
        for other in (np.array([1., 0.]), np.array([[1., np.nan]]), np.array([1., np.nan], dtype=np.float32)):
            with self.subTest(shape=other.shape), self.assertRaises(ValueError):
                m.exact_array(np.array([1., np.nan]), other, "synthetic")

    def test_main_Review_summary_preserves_population_and_NULL(self):
        rows = m.path_rows(None, [121, 180], self.objects["Q2_objects"][0], "synthetic")
        summary = m.summarize(rows)
        self.assertEqual(summary["original_n"], 2); self.assertEqual(summary["case_n"], 2)
        self.assertEqual(summary["service_automatic_n"], 0); self.assertEqual(summary["financial_n"], 0)
        self.assertEqual(summary["execution_review_n"], 2)
        self.assertTrue(all(summary[k] is None for k in m.SUMMARY_FINANCIAL))
        self.assertTrue(all(row[k] is None for row in rows for k in m.FINANCIAL))

    def test_Q1_Review_retains_own_H_L_liquidity_and_NULL(self):
        for index, horizon, loss in ((0, 5, .1), (3, 25, .15)):
            rows = m.path_rows(None, [121, 180], self.objects["Q1_objects"][index], "synthetic")
            summary = m.summarize(rows)
            self.assertEqual(summary["horizon"], horizon); self.assertEqual(summary["loss_tolerance"], loss)
            self.assertEqual(summary["q1_review_n"], 2); self.assertIsNone(summary["breach_count"])
            self.assertEqual(rows[0]["liquidity"], self.profiles[index]["liquidity"])

    def test_Q1_C_suitability_does_not_authorize_Q2_C_service(self):
        obj = self.objects["Q1_objects"][2]
        self.assertTrue(obj["q1_suitable"]); self.assertTrue(obj["financial_available"])
        self.assertEqual(obj["current_Q2_C_execution"], "execution_review")
        self.assertFalse(self.objects["Q2_objects"][0]["financial_available"])

    def test_cannot_substitute_a_physical_account_for_Review(self):
        with self.assertRaises(ValueError):
            m.path_rows({}, [121], self.objects["Q2_objects"][0], "synthetic")

    def test_summary_does_not_treat_NULL_as_zero(self):
        rows = m.path_rows(None, [121], self.objects["Q2_objects"][0], "synthetic")
        rows[0]["nominal_fees"] = 0.
        with self.assertRaises(ValueError):
            m.summarize(rows)

    def test_final60_quantiles_and_growth_units_are_fixed(self):
        summary = m.summarize(synthetic_financial_rows(60))
        self.assertEqual(summary["median_cagr"], 30.5); self.assertEqual(summary["p10_cagr"], 6.)
        self.assertEqual(summary["p90_mdd"], .54)
        self.assertAlmostEqual(summary["cvar90_mdd"], .575)
        self.assertNotEqual(summary["mean_log_growth"], summary["mean_cagr"])

    def test_fractional_tail_mass_not_above_interpolated_percentile(self):
        summary = m.summarize(synthetic_financial_rows(24))
        self.assertAlmostEqual(summary["cvar90_mdd"], (.24 + .23 + .4 * .22) / 2.4)

    def test_failed_attempt_is_retained_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fresh_attempt"
            def fail(state):
                m.update_attempt(out, state, "synthetic_final_parser_started", final_value_access_started=True, final_values_may_have_been_parsed=True)
                raise RuntimeError("synthetic forced failure; no market file opened")
            with self.assertRaises(RuntimeError):
                m.run_archived_attempt(out, {}, fail)
            receipt = m.readj(out / "manifest.json")
            self.assertEqual(receipt["status"], "failed"); self.assertTrue(receipt["final_values_may_have_been_parsed"])
            self.assertFalse(receipt["completed_final_array_load"]); self.assertTrue((out / "error.log").is_file())
            original = (out / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                m.run_archived_attempt(out, {}, fail)
            self.assertEqual((out / "manifest.json").read_bytes(), original)

    def test_singleton_price_and_target_axes_are_not_scenario_id_indexed(self):
        from experimental_execution.review_pending import simulate_paired
        r, z = np.zeros((2, 30, 3)), np.zeros((2, 30))
        raw = simulate_paired(r, z, [.45, .05, .5], .4, horizon=30)["pending_hold"]
        arm = m.squeeze_reference(raw, [.45, .05, .5])
        rows = m.path_rows(arm, [121, 180], self.objects["Q2_objects"][2], "synthetic")
        self.assertEqual([row["scenario_id"] for row in rows], [121, 180])
        self.assertAlmostEqual(sum(rows[0]["terminal_" + a + "_weight"] for a in m.ASSETS), 1.)

    def test_trade_fraction_denominator_includes_all_original_periods(self):
        from experimental_execution.review_pending import simulate_paired
        r, z = np.zeros((1, 30, 3)), np.zeros((1, 30))
        arm = m.squeeze_reference(simulate_paired(r, z, [.45, .05, .5], .4, horizon=30)["pending_hold"], [.45, .05, .5])
        rows = m.path_rows(arm, [180], self.objects["Q2_objects"][2], "synthetic")
        self.assertAlmostEqual(rows[0]["trade_fraction"], 1 / 30)

    def test_excess_within_original_breach_tolerance_is_zero(self):
        from experimental_execution.review_pending import simulate_paired
        # One-step wealth drop exceeds L by only5e-13; with zero costs this
        # remains equality under the frozen numerical breach convention.
        loss = .4; r = np.array([[[-loss - 5e-13, -loss - 5e-13, -loss - 5e-13]]])
        arm = m.squeeze_reference(simulate_paired(r, np.zeros((1, 1)), [1/3]*3, loss, horizon=1, cost_rate=0)["annual"], [1/3]*3)
        obj = copy.deepcopy(self.objects["Q2_objects"][3]); obj["horizon"] = 1
        row = m.path_rows(arm, [121], obj, "synthetic")[0]
        self.assertFalse(row["breach"]); self.assertEqual(row["excess"], 0.)
        self.assertEqual(row["volatility"], 0.)

    def test_exactthirds_keeps_restoring_after_crossing(self):
        from experimental_execution.review_pending import simulate_paired
        r = np.zeros((1, 3, 3)); r[:, 0, :] = -.6; z = np.zeros((1, 3))
        pair = simulate_paired(r, z, [1/3]*3, .4, horizon=3, cost_rate=0)
        self.assertTrue(pair["annual"]["target_execution"].all())
        self.assertEqual(pair["pending_hold"]["actions"].reshape(-1).tolist(), ["restore", "hold", "hold"])

    def test_future_actual_returns_cannot_change_earlier_fixed_fan_Q_or_action(self):
        from experimental_execution.risk_priced_fan_rollout import simulate_prices
        train, z = np.zeros((2, 3, 3)), np.zeros((2, 3))
        a = np.zeros((1, 3, 3)); c = a.copy(); c[:, 1:, :] = .2
        x = simulate_prices(a, np.zeros((1, 3)), train, z, [.45, .05, .5], [.03], L=.4, horizon=3)
        y = simulate_prices(c, np.zeros((1, 3)), train, z, [.45, .05, .5], [.03], L=.4, horizon=3)
        self.assertEqual(x["actions"][0, 0, 1], y["actions"][0, 0, 1])
        self.assertEqual(x["qg_hold"][0, 0, 1], y["qg_hold"][0, 0, 1])
        self.assertEqual(x["qg_restore"][0, 0, 1], y["qg_restore"][0, 0, 1])

    def test_Review_physical_trees_cannot_be_archived(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            m.save_trace_archive(Path(tmp) / "bad.npz", {"guarded_C_service": {}}, [121])

    def test_current_dev_output_bytes_are_checked_before_future_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.py"; source.write_text("synthetic\n")
            out = root / m.DEV_OUT; out.mkdir(parents=True); value = out / "value.txt"; value.write_text("frozen\n")
            sources = {"source.py": m.sha(source)}
            manifest = dict(sources=sources, sources_at_end=sources, input_sha256=sources, input_sha256_at_end=sources,
                            outputs_sha256={"value.txt": m.sha(value)})
            audited = {str(value.relative_to(root)): m.sha(value)}
            audit = dict(sources_and_artifacts_at_start=audited, sources_and_artifacts_at_end=audited)
            m.verify_development_evidence(root, manifest, audit)
            value.write_text("changed after independent audit\n")
            with self.assertRaisesRegex(RuntimeError, "development-gate output differs"):
                m.verify_development_evidence(root, manifest, audit)

    def test_complete_kernel_source_closure_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = copy.deepcopy(self.p); sources = {}
            for name in m.REQUIRED_SOURCES:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic source\n")
                sources[name] = m.sha(path)
            inputpath = root / "input.txt"; inputpath.write_text("synthetic input\n")
            p.update(execution_sources_sha256=sources, frozen_input_sha256={"input.txt": m.sha(inputpath)})
            m.freeze_snapshot(p, root)
            p["execution_sources_sha256"] = {k: v for k, v in sources.items() if k != "src/safeinvest/data.py"}
            with self.assertRaisesRegex(RuntimeError, "complete adapter/test/production-kernel"):
                m.freeze_snapshot(p, root)

    def test_all_synthetic_output_objects_use_local_row_order_and_preserve_Review(self):
        actual = np.zeros((2, 30, 3)); actual[:, :, 0] = .005
        inflation = np.zeros((2, 30))
        arms = m.simulate_locked(actual, inflation, actual.copy(), inflation.copy(), self.objects)
        self.assertEqual(len(arms), 6)
        self.assertNotIn("guarded_C_service", arms)
        q2, q1, summaries = m.evaluation_rows(arms, [121, 180], self.objects, "synthetic_no_market_file")
        self.assertEqual((len(q2), len(q1), len(summaries)), (8, 10, 9))
        main = next(r for r in summaries if r["object_id"] == "guarded_C_service")
        self.assertEqual(main["financial_n"], 0); self.assertIsNone(main["breach_count"])
        product = next(r for r in summaries if r["object_id"] == "Q1_C_static_product")
        self.assertEqual(product["financial_n"], 2); self.assertIsNone(product["service_automatic_n"])


if __name__ == "__main__":
    unittest.main()
