"""Synthetic joint-liquidity fitting, isolation, and referral semantics."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
from experimental_execution import joint_liquidity_static as joint
from safeinvest_research.static import simplex_grid


def synthetic():
    all_weights = simplex_grid(.05)
    allowed = np.flatnonzero(np.all(all_weights >= .05 - 1e-12, axis=1))
    w = all_weights[allowed]
    g, d = np.full((120, 172), .001), np.full((120, 172), .95)
    identified = []
    for target, growth, loss in [([.1, .2, .7], .01, .05),
                                 ([.5, .2, .3], .03, .15),
                                 ([.6, .1, .3], .049, .30),
                                 ([.8, .15, .05], .05, .30)]:
        i = np.flatnonzero(np.all(np.isclose(w, target, atol=1e-14, rtol=0), axis=1)).item()
        g[:, i], d[:, i] = growth, loss
        identified.append(i)
    m = {h: dict(log_growth=g.copy(), cagr=np.expm1(g), mdd=d.copy(),
                 volatility=np.zeros_like(g), terminal_real=100*np.exp(h*g))
         for h in [5, 6, 15, 25, 30]}
    return w, m, allowed, identified


class JointLiquidityTests(unittest.TestCase):
    def test_joint_fit_changes_menu_and_oracle_instead_of_post_filtering(self):
        w, m, _, (a, b, liquid_growth, illiquid_growth) = synthetic()
        old_floors = deepcopy(joint.base.frozen_nested.FLOORS)
        old = joint.base.frozen_nested.fit_gamma(m, w, joint.PROFILES, np.arange(72), 0.)
        new = joint.fit_gamma(m, w, joint.PROFILES, np.arange(72), 0.)
        self.assertEqual(old['selection']['identified_indices'], [a, b, illiquid_growth])
        self.assertEqual(new['selection']['identified_indices'], [a, b, liquid_growth])
        self.assertAlmostEqual(new['selection']['oracle_growth'][4], .049)
        self.assertAlmostEqual(old['selection']['oracle_growth'][4], .05)
        # Post-filtering the old menu would leave E no more than .03 growth.
        old_e_growth = max(m[6]['log_growth'][0, p['candidate_index']]
                           for p in old['menu'] if p['weights'][2] >= .30-1e-12)
        new_e = next(x for x in new['client_product_details'] if x['client']=='E' and x['chosen'])
        self.assertAlmostEqual(old_e_growth, .03)
        self.assertAlmostEqual(new_e['mean_log_growth'], .049)
        self.assertEqual(joint.base.frozen_nested.FLOORS, old_floors)
        self.assertEqual(old_floors, {'high': 0., 'medium': 0., 'low': 0.})

    def test_all_inner_fits_explicitly_receive_floors_and_unequal90_pooling_is_unchanged(self):
        w, m, allowed, _ = synthetic()
        ctx = joint.make_context(range(1, 91), seed=8702026101801)
        actual = joint.fit_menu
        seen = []
        def spy(metrics, weights, profiles, indices, **kwargs):
            seen.append((np.asarray(indices).copy(), deepcopy(kwargs)))
            return actual(metrics, weights, profiles, indices, **kwargs)
        with patch.object(joint, 'fit_menu', side_effect=spy):
            out = joint.calibrate_liquidity_static(m, w, joint.PROFILES, ctx,
                                                   candidate_original_indices=allowed)
        self.assertEqual(len(seen), 21)
        for gi, gamma in enumerate(joint.GAMMAS):
            for fi, fold in enumerate(ctx['inner_folds']):
                ix, kw = seen[4*gi+fi]
                np.testing.assert_array_equal(ix, np.asarray(fold['train_ids'])-1)
                self.assertEqual(kw, {'alpha': gamma, 'liquidity_floors': joint.LIQUIDITY_FLOORS})
        self.assertEqual([len(x['test_ids']) for x in ctx['inner_folds']], [23,23,22,22])
        np.testing.assert_array_equal(seen[-1][0], np.arange(90))
        self.assertEqual(out['refit']['selection']['selected_gamma'], 0.)
        for summary in out['inner_summaries']:
            rows = [r for r in out['inner_oof_paths']
                    if r['training_screen_gamma']==summary['training_screen_gamma']]
            self.assertEqual(len(rows), 450)
            self.assertEqual(summary, joint.base.frozen_nested.inner_summary(rows, summary['training_screen_gamma']))
        self.assertEqual(out['q2_case']['status'], 'not_balanced_case')
        self.assertIsNone(out['q2_case']['product'])
        for item in out['refit']['raw_fit']['menu']:
            self.assertEqual(item['candidate_original232_index'], int(allowed[item['candidate_index']]))

    def test_outside_training_nan_and_mutation_cannot_change_any_fitting_evidence(self):
        w, m, _, _ = synthetic()
        ctx = joint.make_context(range(10, 100), seed=51)
        original = deepcopy(m), w.copy(), deepcopy(ctx), deepcopy(joint.PROFILES)
        out = joint.calibrate_liquidity_static(m, w, joint.PROFILES, ctx)
        changed = deepcopy(m)
        excluded = np.array(sorted(set(range(120))-set(np.asarray(ctx['training_ids'])-1)))
        for fields in changed.values():
            for values in fields.values(): values[excluded] = np.nan
        self.assertEqual(out, joint.calibrate_liquidity_static(changed, w, joint.PROFILES, ctx))
        for h in m:
            for f in m[h]: np.testing.assert_array_equal(m[h][f], original[0][h][f])
        np.testing.assert_array_equal(w, original[1])
        self.assertEqual(ctx, original[2]); self.assertEqual(joint.PROFILES, original[3])

    def test_raw_feasibility_counts_and_chosen_floors_are_full_family_diagnostics(self):
        w, m, _, _ = synthetic()
        fitted = joint.fit_gamma(m, w, joint.PROFILES, np.arange(72), 0.)
        self.assertNotIn('alpha', fitted)
        self.assertEqual(fitted['training_screen_gamma'], 0.)
        for p in joint.PROFILES:
            diag = fitted['feasibility_diagnostics'][p['name']]
            risk = m[p['horizon']]['mdd'][0] <= p['loss_tolerance']+1e-12
            liquid = w[:,2] >= joint.LIQUIDITY_FLOORS[p['liquidity']]-1e-12
            self.assertEqual(diag['risk_feasible_n'], int(risk.sum()))
            self.assertEqual(diag['joint_feasible_n'], int((risk&liquid).sum()))
            for detail in fitted['client_product_details']:
                self.assertNotIn('sample_feasible', detail)
                if detail['client']==p['name'] and detail['chosen']:
                    self.assertTrue(liquid[detail['candidate_index']])
                    self.assertTrue(detail['training_screen_feasible'])

    def test_liquidity_intersection_referral_is_not_mislabeled_pure_risk(self):
        w, m, _, (_, _, _, illiquid) = synthetic()
        # A has a risk-feasible target, but it holds only5% bills.
        m[5]['mdd'][:] = .95
        m[5]['mdd'][:, illiquid] = .02
        ctx = joint.make_context(range(1, 73), seed=17)
        out = joint.calibrate_liquidity_static(m, w, joint.PROFILES, ctx)
        a = out['refit']['q1_review_details']['A']
        self.assertEqual(a['effective_assignment'], 'Review')
        self.assertEqual(a['q1_review_kind'], 'risk_liquidity_intersection_empty')
        self.assertEqual(a['feasibility_diagnostics']['risk_feasible_n'], 1)
        self.assertEqual(a['feasibility_diagnostics']['joint_feasible_n'], 0)

    def test_no_inner_evidence_guard_and_no_supported_gamma_remain_distinct(self):
        w, m, _, _ = synthetic()
        ctx = joint.make_context(range(1, 73), seed=1)
        actual = joint.fit_gamma
        def no_C_inner(metrics, weights, profiles, indices, gamma):
            fit = actual(metrics, weights, profiles, indices, gamma)
            if len(indices)<72: fit['assignments']['C'] = 'Review'
            return fit
        with patch.object(joint, 'fit_gamma', side_effect=no_C_inner):
            out = joint.calibrate_liquidity_static(m, w, joint.PROFILES, ctx)
        self.assertNotEqual(out['refit']['raw_assignments']['C'], 'Review')
        self.assertEqual(out['q2_case']['status'], 'risk_review')
        self.assertEqual(out['q2_case']['public_status'], 'q1_review')
        self.assertEqual(out['q2_case']['q1_review_kind'], 'no_inner_auto_evidence')
        failed = deepcopy(m)
        for fields in failed.values(): fields['mdd'][:] = .99
        with patch.object(joint, 'fit_gamma', wraps=actual) as fitter:
            unsupported = joint.calibrate_liquidity_static(failed, w, joint.PROFILES, ctx)
        self.assertEqual(fitter.call_count, 20)
        self.assertIsNone(unsupported['refit']['selection']['selected_gamma'])
        self.assertIsNone(unsupported['refit']['raw_fit'])
        self.assertEqual(unsupported['q2_case']['q1_review_kind'], 'calibration_unsupported')
        self.assertTrue(all(r['log_growth'] is None and r['breach'] is None
                            for r in unsupported['inner_oof_paths']))

    def test_exact_family_profiles_mapping_and_context_are_enforced(self):
        w, m, allowed, _ = synthetic()
        ctx = joint.make_context(range(1, 73), seed=1)
        for changed in [w[::-1], w[:4]]:
            with self.assertRaises(ValueError):
                joint.calibrate_liquidity_static(m, changed, joint.PROFILES, ctx)
        for mapping in [allowed[::-1], allowed.astype(float), np.full(172, True)]:
            with self.assertRaises(ValueError):
                joint.calibrate_liquidity_static(m, w, joint.PROFILES, ctx, candidate_original_indices=mapping)
        changed = deepcopy(joint.PROFILES); changed[0]['liquidity'] = 'low'
        with self.assertRaises(ValueError): joint.calibrate_liquidity_static(m,w,changed,ctx)
        changed = deepcopy(ctx); changed['inner_folds'][0]['test_ids'][0]=121
        with self.assertRaises(ValueError): joint.calibrate_liquidity_static(m,w,joint.PROFILES,changed)

    def test_feasibility_diagnostics_match_actual_screen_at_one_ulp_boundary(self):
        w, m, _, (a, _, _, _) = synthetic()
        # At this representable boundary addition and subtraction comparisons
        # differ. Diagnostics must report the actual empirical_risk mask.
        loss = joint.PROFILES[0]['loss_tolerance']
        value = loss + 1e-12
        self.assertFalse(value > loss + 1e-12)
        self.assertTrue(value - loss > 1e-12)
        m[5]['mdd'][:] = .95
        m[5]['mdd'][:,a] = value
        fit = joint.fit_gamma(m,w,joint.PROFILES,np.arange(72),0.)
        diag = fit['feasibility_diagnostics']['A']
        self.assertEqual(diag['risk_feasible_n'], 0)
        self.assertEqual(diag['joint_feasible_n'], fit['oracle_candidate_count']['A'])


if __name__ == '__main__': unittest.main()
