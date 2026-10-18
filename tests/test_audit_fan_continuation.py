"""Pure fixed-state deletion/continuation checks; no market/result reads."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_fan_continuation as audit


class FanContinuationAuditTests(unittest.TestCase):
    def test_score_uses_original_horizon_and_already_net_Q(self):
        q=audit.recommendation(.1,.09,5,4,100,.1,30)
        self.assertAlmostEqual(q['score'],.02);self.assertTrue(q['restore'])
        self.assertFalse(audit.recommendation(.1,.09,5,4,100,.1,1)['restore'])
        self.assertTrue(audit.recommendation(.1,.10001,0,0,100,0.,30)['restore'])

    def test_risk_first_is_integer_lexicographic_and_score_NULL(self):
        q=audit.recommendation(10.,-10.,2,1,10,None,30)
        self.assertTrue(q['restore']);self.assertIsNone(q['score'])
        self.assertFalse(audit.recommendation(-10.,10.,1,2,10,None,30)['restore'])
        self.assertFalse(audit.recommendation(0.,0.,1,1,10,None,30)['restore'])

    def test_delete_whole_member_matches_explicit_small_hand_example(self):
        original,rows=audit.explicit_delete_one([0.,2.,4.],[1.,3.,5.],[False,True,True],[False,False,True],[4,8,12],0.,30)
        self.assertEqual([r['deleted_training_id'] for r in rows],[4,8,12])
        self.assertEqual([r['n_remaining'] for r in rows],[2,2,2])
        self.assertEqual([r['hold_Q'] for r in rows],[3.,2.,1.])
        self.assertEqual([r['hold_k'] for r in rows],[2,1,1]);self.assertTrue(original['restore'])

    def test_paired_probability_delete_bound_really_can_equal_two_over_N(self):
        original,rows=audit.explicit_delete_one(np.zeros(3),np.zeros(3),[True,False,False],[False,True,True],[1,2,3],None,30)
        change=rows[0]['delta_breach_probability']-original['delta_breach_probability']
        self.assertAlmostEqual(change,2/3);self.assertGreater(change,1/3)
        self.assertEqual(rows[0]['hold_k'],0);self.assertEqual(rows[0]['restore_k'],2)

    def test_deletion_has_a_decision_flip_without_refitting_state_or_eta(self):
        original,rows=audit.explicit_delete_one([0.,0.,0.],[-2.,1.,1.],[False]*3,[False]*3,[1,2,3],0.,30)
        self.assertFalse(original['restore']);self.assertTrue(rows[0]['action_flip']);self.assertTrue(rows[0]['restore'])
        self.assertFalse(rows[1]['action_flip']);self.assertFalse(rows[2]['action_flip'])

    def test_branch_suffix_matches_closed_form_single_current_fee(self):
        r=np.zeros((2,3));r[1,0]=.5
        out=audit.replay_suffix([0.,0.,100.],1.,100.,0.,[.5,.25,.25],r,np.zeros(2),1,first_action='restore')
        fee=100*.0025*.75/(1+.0025*.75)
        self.assertAlmostEqual(out['nominal_fees'],fee,places=13)
        self.assertAlmostEqual(out['terminal_real'],(100-fee)*1.25,places=12)
        self.assertAlmostEqual(out['total_log_growth'],math.log((100-fee)*1.25/100),places=13)
        self.assertEqual(out['periods'][0]['original_year'],2)

    def test_forecast_and_actual_suffix_branch_agree_on_same_market_member(self):
        train=np.zeros((1,3,3));train[0,1,0]=.4;train[0,2,:]=-.7
        pi=np.zeros((1,3));target=np.array([.4,.3,.3])
        fan=audit.scalar_fan([0.,0.,100.],1.,100.,0.,target,.4,train,pi,1)
        for action in ('hold','restore'):
            out=audit.replay_suffix([0.,0.,100.],1.,100.,0.,target,train[0],pi[0],1,first_action=action)
            self.assertAlmostEqual(out['total_log_growth'],fan[action]['mean'],places=13)
            self.assertEqual(out['ever_breached'],bool(fan[action]['bad'][0]))

    def test_history_CPI_peak_and_absorbing_hold_are_retained(self):
        r=np.zeros((3,3));pi=np.zeros(3);pi[1]=.25
        out=audit.replay_suffix([0.,0.,100.],2.,100.,.5,[.4,.3,.3],r,pi,1,first_action='hold',loss=.55)
        self.assertEqual(out['first_crossing_original_year'],2)
        self.assertEqual(out['periods'][1]['action'],'hold');self.assertTrue(out['periods'][1]['pending_at_decision'])
        self.assertAlmostEqual(out['terminal_real'],40.);self.assertAlmostEqual(out['terminal_mdd'],.6)

    def test_final_year_chosen_continuation_gap_zero_does_not_imply_forecast_error_zero(self):
        r=np.zeros((3,3));r[-1,0]=.5;target=[.4,.3,.3]
        branch=audit.replay_suffix([40.,30.,30.],1.,100.,0.,target,r,np.zeros(3),2,first_action='hold')
        saved=audit.replay_suffix([40.,30.,30.],1.,100.,0.,target,r,np.zeros(3),2,fixed_actions=['hold'])
        self.assertEqual(branch['total_log_growth'],saved['total_log_growth'])
        forecast=audit.scalar_fan([40.,30.,30.],1.,100.,0.,np.array(target),.4,np.zeros((2,3,3)),np.zeros((2,3)),2)
        self.assertNotAlmostEqual(forecast['hold']['mean'],branch['total_log_growth'])

    def test_pending_saved_restore_and_bad_input_shapes_rejected(self):
        args=([0.,0.,100.],2.,100.,.6,[.4,.3,.3],np.zeros((3,3)),np.zeros(3),1)
        with self.assertRaises(ValueError):audit.replay_suffix(*args,fixed_actions=['restore','hold'])
        with self.assertRaises(ValueError):audit.explicit_delete_one([0.],[0.],[False],[False],[1],0.)
        with self.assertRaises(ValueError):audit.recommendation(0.,0.,True,0,10,0.,30)
        with self.assertRaises(ValueError):audit.recommendation(0.,0.,0,0,10,float('inf'),30)

    def test_overwrite_guards_precede_any_prospective_numeric_reader(self):
        for name in ('independent_audit.json','audit_failure_receipt.json'):
            with tempfile.TemporaryDirectory() as folder:
                out=Path(folder);(out/name).write_text('{}')
                with patch.object(audit,'OUT',out),patch.object(audit,'readj',side_effect=AssertionError('No read')):
                    with self.assertRaises(RuntimeError):audit.main()


if __name__=='__main__':unittest.main()
