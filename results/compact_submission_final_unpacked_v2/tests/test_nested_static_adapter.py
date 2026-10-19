"""Synthetic-only training isolation and frozen static-adapter semantics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
from experimental_execution import nested_static_adapter as adapter
import run_nested_static_calibration as legacy
import run_positive_asset_specification as strict
from safeinvest_research.static import simplex_grid

PROFILES = [dict(name=n, horizon=h, loss_tolerance=l, liquidity=q)
            for n,h,l,q in [('A',5,.1,'high'),('B',15,.25,'medium'),('C',30,.4,'low'),
                            ('D',25,.15,'medium'),('E',6,.35,'high')]]


def synthetic():
    w = np.array([[.1,.2,.7],[.3,.3,.4],[.5,.3,.2],[.8,.1,.1]])
    g = np.tile([.01,.02,.03,.04], (120,1))
    d = np.tile([.02,.10,.20,.50], (120,1))
    fields = {h:dict(log_growth=g.copy(), cagr=np.expm1(g), mdd=d.copy(),
                     volatility=np.zeros_like(g), terminal_real=100*np.exp(h*g))
              for h in [5,6,15,25,30]}
    return w,fields


class PartitionTests(unittest.TestCase):
    def test_all_four_sizes_and_unequal90_folds_are_complete_complements(self):
        for n in [120,96,90,72]:
            ctx = adapter.make_context(range(1,n+1), seed=8702026101801, name=f'T{n}')
            self.assertEqual(adapter.validate_context(ctx),ctx)
            seen=[]
            for f in ctx['inner_folds']:
                self.assertFalse(set(f['train_ids']) & set(f['test_ids']))
                self.assertEqual(set(f['train_ids'])|set(f['test_ids']),set(range(1,n+1)))
                seen.extend(f['test_ids'])
            self.assertEqual(sorted(seen),list(range(1,n+1)))
            self.assertEqual(len(seen),len(set(seen)))
            if n==90:
                self.assertEqual([len(f['test_ids']) for f in ctx['inner_folds']],[23,23,22,22])
                self.assertEqual([len(f['train_ids']) for f in ctx['inner_folds']],[67,67,68,68])

    def test_seed_and_id_order_are_deterministic_but_seed_changes_fold_identity(self):
        a=adapter.make_context(range(1,91),seed=23)
        self.assertEqual(a,adapter.make_context(range(90,0,-1),seed=23))
        self.assertNotEqual(a['inner_folds'],adapter.make_context(range(1,91),seed=24)['inner_folds'])
        # Independent construction checks the exact declared RNG algorithm.
        chunks=np.array_split(np.random.default_rng(23).permutation(np.arange(1,91)),4)
        self.assertEqual(a['inner_folds'][0]['test_ids'],sorted(map(int,chunks[0])))

    def test_invalid_or_tampered_ids_seeds_and_folds_are_rejected(self):
        for ids in [list(range(1,72))+[71],list(range(1,72))+[121],
                    [True]+list(range(2,73)),[float(i) for i in range(1,73)],range(1,74)]:
            with self.assertRaises(ValueError): adapter.make_context(ids,seed=1)
        for seed in [True,-1,1.,2**64]:
            with self.assertRaises(ValueError): adapter.make_context(range(1,73),seed=seed)
        ctx=adapter.make_context(range(1,91),seed=1)
        bad=deepcopy(ctx); bad['inner_folds'][0]['test_ids'][0]=121
        with self.assertRaises(ValueError): adapter.validate_context(bad)
        bad=deepcopy(ctx); bad['inner_seed']=2
        with self.assertRaises(ValueError): adapter.validate_context(bad)
        with self.assertRaises(ValueError): adapter.make_context(range(1,73),seed=1,outer_test_ids=[1])

    def test_old_six_seeded_contexts_match_generic_fold_constructor_exactly(self):
        protocol=json.loads((ROOT/'docs/positive_asset_specification_protocol.json').read_text())
        for old in protocol['contexts']:
            self.assertEqual(adapter.validate_context(old),old)
            self.assertEqual(old,adapter.make_context(old['training_ids'],seed=old['inner_seed'],
                  name=old['name'],context_id=old['context_id'],outer_test_ids=old['outer_test_ids']))


class TrainingAndReproductionTests(unittest.TestCase):
    def test_entire_menu_gamma_and_admission_are_fitted_only_inside90(self):
        w,m=synthetic(); ctx=adapter.make_context(range(1,91),seed=8702026101801)
        seen=[]; actual=legacy.fit_gamma
        def spy(metrics,weights,profiles,indices,gamma):
            seen.append((tuple(indices),gamma))
            return actual(metrics,weights,profiles,indices,gamma)
        with patch.object(legacy,'fit_gamma',side_effect=spy):
            result=adapter.calibrate_static(m,w,PROFILES,ctx)
        self.assertEqual(len(seen),21)
        for gi,gamma in enumerate(adapter.GAMMAS):
            for fi,f in enumerate(ctx['inner_folds']):
                indices,used=seen[4*gi+fi]
                self.assertEqual(indices,tuple(i-1 for i in f['train_ids']))
                self.assertEqual(used,gamma)
        self.assertEqual(seen[-1][0],tuple(range(90)))
        self.assertEqual(result['refit']['selection']['selected_gamma'],0.)
        # Every original T path contributes once per gamma/profile despite unequal folds.
        self.assertTrue(all(s['total_path_profile_units']==90*5 for s in result['inner_summaries']))
        for summary in result['inner_summaries']:
            self.assertEqual(summary,legacy.inner_summary(
                [r for r in result['inner_oof_paths'] if r['training_screen_gamma']==summary['training_screen_gamma']],
                summary['training_screen_gamma']))

    def test_outside_training_values_nan_or_changed_cannot_change_output_and_inputs_immutable(self):
        w,m=synthetic(); ctx=adapter.make_context(range(10,100),seed=7)
        original=(deepcopy(m),w.copy(),deepcopy(ctx),deepcopy(PROFILES))
        result=adapter.calibrate_static(m,w,PROFILES,ctx)
        changed=deepcopy(m); excluded=np.array(sorted(set(range(1,121))-set(ctx['training_ids'])))-1
        for fields in changed.values():
            for a in fields.values(): a[excluded]=np.nan
        self.assertEqual(result,adapter.calibrate_static(changed,w,PROFILES,ctx))
        for h,fields in m.items():
            for k,a in fields.items(): np.testing.assert_array_equal(a,original[0][h][k])
        np.testing.assert_array_equal(w,original[1]); self.assertEqual(ctx,original[2]); self.assertEqual(PROFILES,original[3])

    def test_generic_and_frozen_calibration_exactly_match_each_old_context_on_synthetic_metrics(self):
        w,m=synthetic(); protocol=json.loads((ROOT/'docs/positive_asset_specification_protocol.json').read_text())
        for ctx in protocol['contexts']:
            refit,summaries,fits,rows=legacy.calibrate_context(m,w,PROFILES,ctx)
            result=adapter.calibrate_static(m,w,PROFILES,ctx)
            self.assertEqual(result['refit'],refit)
            self.assertEqual(result['inner_summaries'],summaries)
            self.assertEqual(result['inner_fold_fits'],fits)
            self.assertEqual(result['inner_oof_paths'],rows)

    def test_unsupported_all_review_never_defaults_gamma_or_does_refit(self):
        w,m=synthetic(); ctx=adapter.make_context(range(1,73),seed=1)
        fake=dict(menu=[],assignments={p['name']:'Review' for p in PROFILES})
        with patch.object(legacy,'fit_gamma',side_effect=lambda *a,**k:deepcopy(fake)) as fitter:
            result=adapter.calibrate_static(m,w,PROFILES,ctx)
        self.assertEqual(fitter.call_count,20)
        self.assertIsNone(result['refit']['selection']['selected_gamma'])
        self.assertIsNone(result['refit']['raw_fit'])
        self.assertEqual(result['q2_case']['status'],'risk_review')
        self.assertTrue(all(r['log_growth'] is None and r['breach'] is None for r in result['inner_oof_paths']))

    def test_zero_inner_evidence_admission_guard_is_preserved(self):
        w,m=synthetic(); ctx=adapter.make_context(range(1,73),seed=1)
        actual=legacy.fit_gamma
        def no_C_inner(metrics,weights,profiles,indices,gamma):
            fitted=actual(metrics,weights,profiles,indices,gamma)
            if len(indices)<72: fitted['assignments']['C']='Review'
            return fitted
        with patch.object(legacy,'fit_gamma',side_effect=no_C_inner):
            result=adapter.calibrate_static(m,w,PROFILES,ctx)
        self.assertNotEqual(result['refit']['raw_assignments']['C'],'Review')
        self.assertEqual(result['refit']['effective_assignments']['C'],'Review')
        c=next(x for x in result['refit']['admission_details'] if x['client']=='C')
        self.assertEqual(c['reason'],'no_inner_auto_evidence')
        self.assertEqual(result['q2_case']['status'],'risk_review')

    def test_strict172_preparation_mapping_and_no_input_mutation(self):
        w=simplex_grid(.05); base=np.arange(120*232,dtype=float).reshape(120,232)
        m={15:{k:base.copy() for k in adapter.STATIC_FIELDS}}
        allowed,small,fields=adapter.prepare_strict_family(w,m)
        self.assertEqual(len(allowed),172); self.assertTrue(np.all(small>=.05-1e-12))
        np.testing.assert_array_equal(fields[15]['mdd'],base[:,allowed])
        fields[15]['mdd'][0,0]=-999
        self.assertNotEqual(m[15]['mdd'][0,allowed[0]],-999)
        with self.assertRaises(ValueError): adapter.prepare_strict_family(w[::-1],m)
        with self.assertRaises(ValueError): adapter.prepare_strict_family(w,{15:{k:a[:119] for k,a in m[15].items()}})

    def test_legacy_wrapper_subsets_and_annotates_without_reading_results(self):
        # All candidate metrics equivalent -> small synthetic optimization,
        # while exercising the exact232 ->172 production family boundary.
        w=simplex_grid(.05); g=np.full((120,232),.02); d=np.full_like(g,.03)
        m={h:dict(log_growth=g.copy(),cagr=np.expm1(g),mdd=d.copy(),volatility=np.zeros_like(g),
                  terminal_real=100*np.exp(h*g)) for h in [5,6,15,25,30]}
        ctx=json.loads((ROOT/'docs/positive_asset_specification_protocol.json').read_text())['contexts'][0]
        result=adapter.reproduce_legacy_context(m,w,PROFILES,ctx)
        allowed,small,fields=strict.subset_family(w,m)
        refit,summaries,fits,rows=legacy.calibrate_context(fields,small,PROFILES,ctx)
        expected=strict.annotate_indices(dict(refit=refit,inner_summaries=summaries,
                           inner_fold_fits=fits,inner_oof_paths=rows),allowed)
        for key in expected: self.assertEqual(result[key],expected[key])
        self.assertTrue(all(p['candidate_original232_index']==int(allowed[p['candidate_index']])
                            for p in result['refit']['raw_fit']['menu']))

    def test_metric_shape_finiteness_and_mapping_failures(self):
        w,m=synthetic(); ctx=adapter.make_context(range(1,73),seed=1)
        bad=deepcopy(m); bad[30]['mdd'][0,0]=np.nan
        with self.assertRaises(ValueError): adapter.calibrate_static(bad,w,PROFILES,ctx)
        bad=deepcopy(m); bad[15]['cagr']=bad[15]['cagr'][:100]
        with self.assertRaises(ValueError): adapter.calibrate_static(bad,w,PROFILES,ctx)
        for mapping in [[0,1,2,232],[0,1,1,3],[0.,1.,2.,3.],[0,1,2]]:
            with self.assertRaises(ValueError): adapter.calibrate_static(m,w,PROFILES,ctx,candidate_original_indices=mapping)


class BalancedCaseTests(unittest.TestCase):
    def test_case_labels_preserve_full_Q1_assignment_and_financial_null_membership(self):
        products=[dict(name=n,candidate_index=i,weights=[.3,.3,.4])
                  for i,n in enumerate(['Conservative','Balanced','Growth'])]
        for label,status in [('Balanced','balanced_case'),('Review','risk_review'),
                             ('Conservative','not_balanced_case'),('Growth','not_balanced_case')]:
            refit=dict(effective_assignments={'C':label},raw_fit=dict(menu=products))
            before=deepcopy(refit); case=adapter.classify_balanced_case(refit)
            self.assertEqual(refit,before); self.assertEqual(case['status'],status)
            self.assertEqual(case['effective_assignment'],label)
            self.assertEqual(case['product'] is not None,label=='Balanced')
            self.assertEqual(case['in_balanced_case'],label=='Balanced')
            if label in ['Conservative','Growth']: self.assertIsNotNone(case['q1_product'])
            if case['product'] is not None:
                case['product']['weights'][0]=99
                self.assertEqual(refit,before)


if __name__=='__main__': unittest.main()
