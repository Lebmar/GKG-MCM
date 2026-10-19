import importlib.util
from pathlib import Path
import unittest
import pandas as pd

spec=importlib.util.spec_from_file_location('final_guard',Path(__file__).resolve().parents[1]/'scripts/finalize_global_policy_admission.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class GlobalGuardTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(job_id='j',scenario_id=i,assigned=True,breach=i==1,candidate_local_index=1.,candidate_original232_index=2.,static_gamma=.075,q1_effective_assignment='Balanced') for i in range(1,21)])
    def test_same_U_excludes_refused_highrisk_reference_rows(self):
        a=self.rows();b=self.rows();a.loc[a.scenario_id>10,'assigned']=False;b.loc[b.scenario_id>10,'breach']=True
        g=m.inner_guard(a,b,list(range(1,21)));self.assertEqual((g['auto_inner_n'],g['k_B_same_U'],g['absolute_cap']),(10,1,1));self.assertTrue(g['eligible'])
    def test_relative_guard_can_reject_absolute_pass(self):
        a=self.rows();b=self.rows();b['breach']=False
        self.assertEqual(m.inner_guard(a,b,list(range(1,21)))['reasons'],['inner_risk_exceeds_same_U_B'])
    def test_empty_auto_is_not_zero_risk(self):
        a=self.rows();a['assigned']=False;g=m.inner_guard(a,self.rows(),list(range(1,21)))
        self.assertIsNone(g['k_inner']);self.assertFalse(g['eligible'])
    def test_mismatched_target_or_missing_original_ID_rejected(self):
        a=self.rows();b=self.rows();b.loc[0,'candidate_local_index']=3
        with self.assertRaises(ValueError):m.inner_guard(a,b,list(range(1,21)))
        with self.assertRaises(ValueError):m.inner_guard(a.iloc[:-1],self.rows(),list(range(1,21)))
    def test_risk_benefit_can_justify_growth_sacrifice_without_second_fee(self):
        d=m.justification(1.,.99,3,1,100,.03)
        self.assertTrue(d['recommend_restore']);self.assertTrue(d['strict_growth_sacrifice']);self.assertAlmostEqual(d['risk_adjusted_net_advantage'],.008)
    def test_growth_loss_without_risk_reduction_cannot_trade(self):
        self.assertFalse(m.justification(1.,.99,1,1,100,1.)['recommend_restore'])
    def test_zero_risk_price_and_ties(self):
        self.assertFalse(m.justification(1.,1.,3,0,100,0.)['recommend_restore'])
        self.assertTrue(m.justification(1.,1.01,3,3,100,0.)['recommend_restore'])
    def test_riskfirst_lexicographic_case(self):
        self.assertTrue(m.justification(1.,.5,3,2,100,None)['recommend_restore'])
        self.assertFalse(m.justification(1.,1.,3,3,100,None)['recommend_restore'])


if __name__=='__main__':unittest.main()
