"""Synthetic mechanical-identity and failure checks; no data or artifact I/O."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_failure_path_mechanisms as a

class FailureMechanismAuditPureTests(unittest.TestCase):
    def test_self_financing_restores_charge_destination_amount_once(self):
        x=np.array([30.,40.,30.]);w=np.array([.45,.05,.5])
        post,fee,moved=a.independent_trade(x,w,.0025)
        self.assertAlmostEqual(fee,.0025*np.maximum(post-x,0).sum(),places=13)
        self.assertAlmostEqual(post.sum()+fee,100.,places=12)
        np.testing.assert_allclose(post/post.sum(),w,atol=1e-15)
        self.assertAlmostEqual(moved,np.maximum(post-x,0).sum(),places=12)
    def test_paired_three_term_and_log_identities(self):
        x=np.array([30.,40.,30.]);y=np.array([50.,20.,20.]);w=np.array([.45,.05,.5])
        pm,fm,_=a.independent_trade(x,w,.0025)
        pb,fb,_=a.independent_trade(y,w,.0025)
        z=a.paired_terms(x,y,pm,pb,fm,fb,[.8,1.04,1.02],endpoint_cpi=1.05)
        self.assertLess(abs(z['nominal_identity_residual']),1e-13)
        self.assertLess(abs(z['log_identity_residual']),1e-14)
        self.assertAlmostEqual(z['direct_asset_gap_recursion'],z['end_nominal_gap'],places=12)
        self.assertAlmostEqual(z['end_real_gap'],z['end_nominal_gap']/1.05,places=12)
    def test_hold_vs_restore_does_not_double_charge_fee(self):
        x=np.array([80.,10.,10.]);pm,fm,_=a.independent_trade(x,[.4,.2,.4],.0025)
        z=a.paired_terms(x,x,pm,x,fm,0,[.7,1.1,1.02],endpoint_cpi=1.03)
        self.assertLess(z['log_fee_survival_gap'],0)
        self.assertAlmostEqual(z['log_wealth_gap_increment'],z['log_fee_survival_gap']+z['log_exposure_gap'],places=13)
    def test_mismatched_posttrade_fee_cannot_pass_identity_input(self):
        with self.assertRaisesRegex(ValueError,'self-financing'):
            a.paired_terms([20,30,50],[20,30,50],[20,30,50],[20,30,50],.25,0,[1,1,1],endpoint_cpi=1)
    def test_first_actions_and_exposures_differ_separately(self):
        self.assertIsNone(a.first_difference(['restore','hold'],['restore','hold']))
        self.assertEqual(a.first_difference([[.4,.2,.4],[.4,.2,.4]],[[.45,.05,.5],[.45,.05,.5]]),1)
    def test_missing_states_are_not_numerical_divergence(self):
        with self.assertRaisesRegex(ValueError,'NULL'):
            a.first_difference([1,np.nan],[1,2])
    def test_endpoint_zero_and_threshold_equality(self):
        self.assertEqual(a.first_crossing([0,.4,.400000000002]),2)
        self.assertIsNone(a.first_crossing([0,.4,.4000000000001]))
        self.assertEqual(a.first_difference([0,1,2],[0,1,3],start_year=0),2)
    def test_running_mdd_reset_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'reset'):a.first_crossing([0,.45,.2])
    def test_pending_recovery_cannot_restart_automatic_trade(self):
        market=np.array([[-.5,0,0],[2,0,0],[0,0,0]])
        with self.assertRaisesRegex(AssertionError,'pending'):
            a.replay_recorded_actions(market,np.zeros(3),[1,0,0],['restore','hold','restore'],cost_rate=0)
    def test_own_B_keeps_initial_peak_and_actual_inflation(self):
        market=np.array([[-.2,0,0],[2,0,0],[0,0,0]])
        r=a.replay_own_B(market,[.5,0,0],[1,0,0],cost_rate=0)
        self.assertEqual(r['first_crossing'],1)
        self.assertEqual(list(r['actions']),['restore','hold','hold'])
        np.testing.assert_allclose(r['cpi'],[1,1.5,1.5,1.5])
        self.assertGreater(r['running_mdd'][-1],.4)
        self.assertGreater(r['real_wealth'][-1],r['real_peak'][1])
    def test_score_uses_original_H_and_total_log_without_extra_fee(self):
        r=a.independent_score(.1,.11,5,6,100,.03)
        self.assertAlmostEqual(r['score'],.001)
        self.assertEqual(r['action'],'restore')
    def test_risk_first_is_lexicographic_and_null_score(self):
        r=a.independent_score(.1,-.2,6,5,100,None)
        self.assertIsNone(r['score']);self.assertEqual(r['action'],'restore')
        self.assertEqual(a.independent_score(.1,.1,5,5,100,None)['action'],'hold')
    def test_integer_counts_required(self):
        with self.assertRaises(ValueError):a.independent_score(.1,.2,False,4,100,.03)
        with self.assertRaises(ValueError):a.independent_score(.1,.2,101,4,100,.03)
    def test_checks_do_not_accept_null_filled_with_zero(self):
        with self.assertRaisesRegex(AssertionError,'NULL'):a.Checks().scalar(0,None,'no crossing')

    def test_protocol_rejects_unfrozen_contract_before_artifact_read(self):
        with self.assertRaisesRegex(RuntimeError,'execution-ready'):a.validate_inventory({'execution_ready':False})
    def test_full_year_initial_restore_cannot_be_skipped(self):
        with self.assertRaisesRegex(ValueError,'initial restore'):
            a.replay_recorded_actions(np.zeros((2,3)),np.zeros(2),[.4,.2,.4],['hold','hold'])
    def test_never_crossed_and_end_breach_are_not_interchanged(self):
        self.assertIsNone(a.first_crossing([0,.1,.2]))
        self.assertEqual(a.first_crossing([0,.1,.5]),2)

if __name__=='__main__':unittest.main()
