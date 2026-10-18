"""Synthetic descriptive ledgers; no market data or new policy execution."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('refit_diagnostic_test',ROOT/'scripts/analyze_refit_admission_mechanisms.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class IntegerBudgetTests(unittest.TestCase):
    def test_integer_platform_and_reference_cap_are_distinct_from_beta_floor(self):
        counts={i:5+i%2 for i in range(9)};growth={i:.03 for i in range(9)}
        low=m.cap_ledger(96,.05,4,counts,growth)
        high=m.cap_ledger(96,.1,4,counts,growth)
        self.assertEqual((low['raw_beta_floor'],high['raw_beta_floor']),(4,9))
        self.assertEqual((low['effective_cap'],high['effective_cap']),(4,4))
        self.assertEqual(low['feasible_orders'],[]);self.assertEqual(high['feasible_orders'],[])
        self.assertFalse(high['minimum_count_attainable_under_reference'])
        self.assertEqual(high['minK_minus_cap'],1)

    def test_feasible_growth_ties_use_one_global_band_and_are_not_unique_eta(self):
        counts={i:0 for i in range(9)};growth={i:.01 for i in range(9)}
        growth[2]=.01+1.1e-12;growth[1]=.01+.5e-12
        r=m.cap_ledger(72,.025,9,counts,growth)
        self.assertEqual(r['growth_tied_feasible_orders'],[1,2])
        self.assertEqual(r['canonical_selected_order'],1)
        self.assertEqual(r['raw_beta_floor'],1)
        for beta in [False,.02]:
            with self.assertRaises(ValueError):m.cap_ledger(72,beta,9,counts,growth)


class EquivalenceTests(unittest.TestCase):
    def test_same_behavior_vector_groups_betas_but_changed_price_metadata_is_separate(self):
        rows=[dict(job_id='t1',scenario_id=i,assigned=True,selected_candidate_order=4,execution_status='automated') for i in (1,2)]
        other=[{**r,'selected_candidate_order':5} for r in rows]
        groups=m.equivalence_classes({'0.025':rows,'0.05':rows,'0.075':other},
                                     ['job_id','scenario_id','assigned','selected_candidate_order','execution_status'])
        self.assertEqual([x['members'] for x in groups],[['0.025','0.05'],['0.075']])

    def test_same_mean_is_not_pathwise_recorded_outcome_equivalence(self):
        a=[dict(scenario_id=1,log_growth=.01,mdd=.2),dict(scenario_id=2,log_growth=.03,mdd=.3)]
        b=[dict(scenario_id=1,log_growth=.03,mdd=.3),dict(scenario_id=2,log_growth=.01,mdd=.2)]
        self.assertAlmostEqual(np.mean([r['log_growth'] for r in a]),np.mean([r['log_growth'] for r in b]))
        result=m.recorded_pair_difference(a,b)
        self.assertFalse(result['recorded_outcome_equal'])
        self.assertEqual(result['differing_path_ids'],[1,2])
        self.assertEqual(result['field_difference_counts']['log_growth'],2)
        same=m.recorded_pair_difference(a,list(reversed(a)))
        self.assertTrue(same['recorded_outcome_equal'])


class IdentityLedgerTests(unittest.TestCase):
    def test_dynamic_event_ledger_counts_retained_flips_and_added_events(self):
        a={1:True,2:True,3:False,4:False}
        b={1:True,2:False,3:True,4:True,5:True,6:False}
        r=m.count_decomposition(a,b)
        self.assertEqual((r['inner_count'],r['refit_count'],r['count_delta']),(2,4,2))
        self.assertEqual(r['upward_flip_ids'],[3,4]);self.assertEqual(r['downward_flip_ids'],[2])
        self.assertEqual(r['added_event_ids'],[5])
        self.assertEqual(r['count_delta'],r['retained_net_flip_count']+r['refit_added_count'])
        with self.assertRaises(ValueError):m.count_decomposition({0:True},{0:True})
        with self.assertRaises(ValueError):m.count_decomposition({121:True},{121:True})
        with self.assertRaises(ValueError):m.count_decomposition({1:True},{2:True})

    def test_fixed_annual_cross_has_two_exact_but_order_dependent_ledgers(self):
        mdd=np.zeros((120,2));mdd[0,0]=.5;mdd[2,1]=.5
        r=m.annual_target_cross(mdd,[1,2],[1,2,3],0,1)
        self.assertEqual(r['total_count_delta'],0)
        self.assertEqual((r['target_first_on_retained'],r['addition_after_target']),(-1,1))
        self.assertEqual((r['addition_first_under_inner_target'],r['target_after_addition']),(0,0))
        self.assertEqual(r['target_by_added_paths_interaction'],1)
        self.assertEqual(r['T_inner_event_ids'],[1]);self.assertEqual(r['added_refit_event_ids'],[3])
        # Same target makes the target contrast exactly0, not a new fitted rule.
        same=m.annual_target_cross(mdd,[1,2],[1,2,3],0,0)
        self.assertEqual(same['target_first_on_retained'],0)
        with self.assertRaises(ValueError):m.annual_target_cross(mdd,[1,121],[1,2,121],0,1)

    def test_saved_training_group_requires_complete_ids_and_nonimputed_nulls(self):
        frame=pd.DataFrame([dict(scenario_id=1,assigned=True,breach=False,mdd=.2,log_growth=.02),
                            dict(scenario_id=2,assigned=True,breach=True,mdd=.5,log_growth=.01)])
        self.assertEqual(m.saved_count(frame,[1,2]),1)
        with self.assertRaises(ValueError):m.saved_count(frame,[1,3])
        bad=frame.copy();bad.loc[0,'scenario_id']=2
        with self.assertRaises(ValueError):m.saved_count(bad,[1,2])
        null=pd.DataFrame([dict(scenario_id=1,assigned=False,breach=None,mdd=None,log_growth=None)])
        self.assertIsNone(m.saved_count(null,[1]))
        null.loc[0,'mdd']=0.
        with self.assertRaises(ValueError):m.saved_count(null,[1])


if __name__=='__main__':unittest.main()
