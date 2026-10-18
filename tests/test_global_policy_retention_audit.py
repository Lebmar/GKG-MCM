"""Independent synthetic audit primitives; no result/market files are read."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/audit_global_policy_retention.py'
spec=importlib.util.spec_from_file_location('independent_global_retention_audit',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class GlobalRetentionIndependentAuditTests(unittest.TestCase):
    def rows(self,n=20):
        model=[];B=[]
        for sid in range(1,n+1):
            common=dict(scenario_id=sid,job_id='inner1' if sid<=n//2 else 'inner2',assigned=True,
                candidate_local_index=1,candidate_original232_index=2,static_gamma=.075,
                q1_effective_assignment='Balanced',case_status='balanced_case',horizon=30,
                loss_tolerance=.4,q1_candidate_local_index=1,q1_candidate_original232_index=2,
                selection_training_n=72,liquidity_policy_fingerprint='frozen')
            model.append({**common,'breach':sid==1})
            B.append({**common,'breach':sid==1,'policy':'annual_then_pending'})
        return model,B

    def test_fixed_same_U_relative_guard_with_absolute_equality(self):
        rows,B=self.rows();B[0]['breach']=False
        gate=m.independent_guard(rows,B,range(1,21))
        self.assertEqual(gate['k_inner'],1);self.assertEqual(gate['absolute_cap'],2)
        self.assertEqual(gate['reasons'],['inner_risk_exceeds_same_U_B']);self.assertFalse(gate['eligible'])

    def test_original_U_excludes_rejected_reference_risk(self):
        rows,B=self.rows()
        for r in rows[10:]:r.update(assigned=False,breach=None)
        for r in B[10:]:r['breach']=True
        gate=m.independent_guard(rows,B,range(1,21))
        self.assertEqual((gate['auto_inner_n'],gate['k_inner'],gate['k_B_same_U'],gate['absolute_cap']),(10,1,1,1))
        self.assertEqual(gate['auto_inner_ids'],list(range(1,11)));self.assertTrue(gate['eligible'])

    def test_empty_U_is_absent_risk_evidence_not_zero(self):
        rows,B=self.rows()
        for r in rows:r.update(assigned=False,breach=None)
        g=m.independent_guard(rows,B,range(1,21))
        self.assertEqual(g['auto_inner_n'],0);self.assertIsNone(g['k_inner'])
        self.assertIsNone(g['k_B_same_U']);self.assertIsNone(g['absolute_cap'])
        self.assertEqual(g['reasons'],['no_inner_auto_evidence']);self.assertFalse(g['eligible'])

    def test_missing_duplicate_foreign_ID_or_wrong_job_cannot_become_Review(self):
        rows,B=self.rows()
        for bad in (rows[:-1],[*rows[:-1],rows[0]]):
            with self.assertRaises(ValueError):m.independent_guard(bad,B,range(1,21))
        wrong=[dict(x) for x in B];wrong[0]['job_id']='another fit'
        with self.assertRaises(ValueError):m.independent_guard(rows,wrong,range(1,21))
        with self.assertRaises(ValueError):m.independent_guard(rows,B,range(2,22))

    def test_any_target_label_gamma_training_identity_or_wrong_B_policy_rejected(self):
        for field,value in [('candidate_local_index',3),('candidate_original232_index',7),
            ('static_gamma',.1),('q1_effective_assignment','Growth'),('horizon',20),
            ('loss_tolerance',.5),('selection_training_n',90),('policy','eta_0')]:
            rows,B=self.rows();B[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.independent_guard(rows,B,range(1,21))

    def test_unavailable_B_or_nonboolean_events_are_integrity_failures(self):
        rows,B=self.rows();B[0]['assigned']=False
        with self.assertRaises(ValueError):m.independent_guard(rows,B,range(1,21))
        rows,B=self.rows();rows[0]['breach']=1
        with self.assertRaises(ValueError):m.independent_guard(rows,B,range(1,21))
        rows,B=self.rows();rows[0].update(assigned=False,breach=True)
        with self.assertRaises(ValueError):m.independent_guard(rows,B,range(1,21))

    def test_absolute_count_cap_is_not_reset_or_replaced_with_B_count(self):
        rows,B=self.rows()
        for r in rows:r['breach']=r['scenario_id']<=3
        for r in B:r['breach']=r['scenario_id']<=4
        g=m.independent_guard(rows,B,range(1,21))
        self.assertEqual(g['reasons'],['inner_absolute_risk_exceeds_alpha'])

    def test_finite_netQ_score_uses_original_H_and_no_second_fee(self):
        g=m.independent_score(1.,.99,3,1,100,.03,30)
        self.assertAlmostEqual(g['risk_adjusted_net_advantage'],.008,places=14)
        self.assertTrue(g['recommend_restore']);self.assertTrue(g['strict_growth_sacrifice'])
        self.assertEqual(g['justification'],'risk_reduction_with_growth_cost')
        short=m.independent_score(1.,.99,3,1,100,.03,1)
        self.assertFalse(short['recommend_restore'])
        net=m.independent_score(0.,np.log(1.004/1.0025),0,0,100,0.)
        self.assertTrue(net['recommend_restore'])

    def test_risk_first_is_integer_lexicographic_without_infinity_arithmetic(self):
        self.assertTrue(m.independent_score(0.,-1.,3,2,10,None)['recommend_restore'])
        self.assertFalse(m.independent_score(0.,100.,2,3,10,None)['recommend_restore'])
        self.assertFalse(m.independent_score(0.,1e-12,2,2,10,None)['recommend_restore'])
        self.assertTrue(m.independent_score(0.,2e-12,2,2,10,None)['recommend_restore'])
        self.assertIsNone(m.independent_score(0.,-1.,3,2,10,None)['risk_adjusted_net_advantage'])

    def test_exhaustive_nonpositive_restore_requires_strict_risk_reduction(self):
        for eta in (*m.PRICES,):
            for kh in range(6):
                for kr in range(6):
                    for delta in [-.25,-1e-13,0.,1e-12,2e-12,.25]:
                        g=m.independent_score(0.,delta,kh,kr,10,eta)
                        if g['recommend_restore'] and delta<=1e-12:self.assertLess(kr,kh)
                        if eta==0 and delta<=1e-12:self.assertFalse(g['recommend_restore'])

    def test_counts_and_technical_parameter_validation(self):
        for bad in [True,0.,-1]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):m.independent_score(0.,1.,bad,0,10,.03)
        for eta in [True,-.1,np.nan,np.inf]:
            with self.subTest(eta=eta),self.assertRaises(ValueError):m.independent_score(0.,1.,0,0,10,eta)

    def test_strict_sacrifice_separates_numerical_tie(self):
        tie=m.independent_score(0.,-.5e-12,1,0,10,.03)
        strict=m.independent_score(0.,-2e-12,1,0,10,.03)
        self.assertTrue(tie['recommend_restore']);self.assertFalse(tie['strict_growth_sacrifice'])
        self.assertTrue(strict['strict_growth_sacrifice'])

    def synthetic_clients(self):
        out=[]
        for repeat in range(6):
            for policy in [m.POLICY,*m.REFS]:
                for sid in range(1,121):
                    active=policy!=m.POLICY or repeat==2 and sid==1
                    vals={field:None for field in m.FIN}
                    if active:vals.update(log_growth=.02,cagr=np.expm1(.02),mdd=.2,nominal_fees=.5,trade_fraction=.1,breach=False)
                    out.append(dict(context='fit'+str(repeat),repeat_id=repeat,policy=policy,scenario_id=sid,assigned=active,**vals))
        return out

    def test_zero_admission_summary_and_pairs_keep_NULL(self):
        rows=self.synthetic_clients();summary=m.independent_summary(rows);pairs=m.common_B_pairs(rows)
        self.assertEqual(len(summary),18);self.assertEqual(len(pairs),6)
        empty=[r for r in summary if r['policy']==m.POLICY and r['repeat_id']!=2]
        self.assertTrue(all(r['assigned_n']==0 and r['breach_count'] is None and r['mean_log_growth'] is None for r in empty))
        for row in pairs:
            if row['repeat_id']!=2:self.assertIsNone(row['model_breaches']);self.assertIsNone(row['mean_delta_log_growth'])
        self.assertEqual(pairs[2]['common_auto_n'],1)

    def test_common_pairs_do_not_include_other_B_outcomes(self):
        rows=self.synthetic_clients()
        for r in rows:
            if r['policy']=='annual_then_pending' and r['scenario_id']>1:r['breach']=True;r['log_growth']=-1.
        pairs=m.common_B_pairs(rows)
        self.assertEqual(pairs[2]['B_breaches'],0);self.assertEqual(pairs[2]['mean_delta_log_growth'],0.)

    def test_numeric_ledger_checks_fail_corruption_and_preserve_NULL_bool(self):
        c=m.Checks();c.value(np.nan,None,'null');c.value(True,True,'bool')
        with self.assertRaises(AssertionError):c.value(0,None,'not null')
        with self.assertRaises(AssertionError):c.value('True',True,'not observed boolean')
        with self.assertRaises(AssertionError):c.value(.1,.2,'corrupt score')


if __name__=='__main__':unittest.main()
