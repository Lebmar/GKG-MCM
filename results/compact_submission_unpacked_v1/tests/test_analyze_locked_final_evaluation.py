"""Synthetic checks of paired populations, event flags and units only."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location('paired_final', Path(__file__).resolve().parents[1]/'scripts/analyze_locked_final_evaluation.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def frame(ids=(121,122,123,124)):
    return pd.DataFrame(dict(scenario_id=ids, log_growth=[.04,.01,.03,-.01], cagr=[.05,.02,.04,-.02],
        mdd=[.45,.2,.5,.1], nominal_fees=[1.,2.,3.,4.], trade_fraction=[.2,.3,.4,.5], breach=[True,False,True,False]))


class PairedFinalTests(unittest.TestCase):
    def test_events_keep_all_four_unfavorable_and_favorable_sets(self):
        left, right = frame(), frame()
        right.breach = [False, True, True, False]
        rows, summary = a.paired(left, right, 'test')
        self.assertEqual(len(rows),4)
        self.assertEqual(summary['event_original_ids'], dict(neither=[124],shadow_only=[121],reference_only=[122],both=[123]))
        self.assertEqual(sum(summary['event_counts'].values()),4)

    def test_ID_order_is_paired_by_identity_and_growth_stays_annual(self):
        left, right = frame(), frame()
        right.log_growth = [.02,.02,.03,-.02]
        rows, result = a.paired(left, right.iloc[::-1], 'test')
        self.assertEqual([r['scenario_id'] for r in rows],[121,122,123,124])
        delta=result['metric_differences']['log_growth']
        self.assertAlmostEqual(delta['mean'],.005)
        self.assertAlmostEqual(delta['median'],.005)
        self.assertEqual((delta['positive'],delta['negative'],delta['equality']),(2,1,1))

    def test_different_population_cannot_be_inner_joined_away(self):
        with self.assertRaisesRegex(ValueError,'identical'):
            a.paired(frame(),frame(ids=(121,122,123,125)),'test')

    def test_duplicate_ID_cannot_expand_a_pair(self):
        with self.assertRaisesRegex(ValueError,'unique'):
            a.paired(frame(ids=(121,121,123,124)),frame(),'test')

    def test_false_string_is_not_truthy_event(self):
        self.assertIs(a.flag('False'),False)
        self.assertIs(a.flag(np.bool_(False)),False)
        self.assertIs(a.flag('True'),True)
        with self.assertRaises(ValueError):
            a.flag(float('nan'))

    def test_nonfinite_reference_cannot_produce_an_optimistic_subset(self):
        right=frame();right.loc[1,'cagr']=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            a.paired(frame(),right,'test')

    def test_numeric_equality_is_retained_not_called_profit(self):
        right=frame();right.log_growth += .5e-12
        _,result=a.paired(frame(),right,'test')
        self.assertEqual(result['metric_differences']['log_growth']['equality'],4)


if __name__=='__main__':
    unittest.main()
