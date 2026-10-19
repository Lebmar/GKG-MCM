"""Exact lemma and fixed earliest-witness checks, entirely synthetic."""
from pathlib import Path
from copy import deepcopy
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_intervention_scale_equivalence as d
from audit_review_pending_execution import independent_trade
from audit_fan_rollout_execution import scalar_fan


def synthetic(k=1.):
    value=(100+np.arange(31,dtype=float))*k;weights=np.tile([.4,.2,.4],(31,1));flags=np.zeros(31,bool)
    a=dict(nominal_wealth=value,real_wealth=value.copy(),real_peak=value.copy(),cpi=np.ones(31),holdings=value[:,None]*weights,weights=weights,
        current_drawdown=np.zeros(31),running_mdd=np.zeros(31),ever_breached=flags,actions=np.array(['restore']+['hold']*29),
        policy_actions=np.array(['restore']+['hold']*29),forced=np.array([False,True]+[False]*28),pending_at_decision=np.zeros(30,bool),
        fees=np.zeros(30),moved=np.zeros(30),posttrade_holdings=value[:-1,None]*weights[:-1])
    a.update({field:np.zeros(30) for field in d.FLOAT_Q});a.update({field:np.zeros(30,dtype=int) for field in d.COUNT_Q});a['forecast_n'][:]=120
    return a

class ScaleEquivalenceLemmaTests(unittest.TestCase):
    def test_exact_self_financing_fee_is_positive_homogeneous(self):
        x=np.array([75.,15.,10.]);target=[.4,.2,.4]
        p,f,m=independent_trade(x,target,.0025)
        for k in (.05,2.,100.):
            pk,fk,mk=independent_trade(k*x,target,.0025)
            np.testing.assert_allclose(pk,k*p,atol=1e-11,rtol=2e-15)
            self.assertAlmostEqual(fk,k*f,places=12);self.assertAlmostEqual(mk,k*m,places=9)
    def test_old_independent_fan_logs_and_events_scale_with_original_peak(self):
        x=np.array([120.,30.,50.]);cp=1.4;peak=170.;worst=.2;target=[.45,.05,.5]
        market=np.array([[[.1,.02,.01],[-.4,.03,.01],[.5,-.1,.01]],[[.2,0,.01],[.15,-.03,.01],[-.5,.04,.01]]])
        inflation=np.array([[.01,.03,.04],[.02,.01,.02]])
        q=scalar_fan(x,cp,peak,worst,target,.4,market,inflation,1,cost_rate=.0025)
        scaled=scalar_fan(7.3*x,cp,7.3*peak,worst,target,.4,market,inflation,1,cost_rate=.0025)
        for action in ('hold','restore'):
            np.testing.assert_allclose(q[action]['logs'],scaled[action]['logs'],atol=1e-14,rtol=0)
            np.testing.assert_array_equal(q[action]['bad'],scaled[action]['bad'])
    def test_earliest_interior_witness_preserves_k_and_future(self):
        result,scan=d.diagnose(synthetic(),synthetic(1.2),2)
        self.assertEqual(result['earliest_interior_equivalence_endpoint'],2)
        self.assertAlmostEqual(result['scale_k'],1.2)
        self.assertEqual(result['status'],'scale_equivalence_verified')
        self.assertEqual(result['future_checks']['future_decision_periods'],28)
    def test_later_good_endpoint_does_not_replace_first_failed_candidate(self):
        c=synthetic();t=synthetic(1.2);t['weights'][3]=[.3,.3,.4]
        result,scan=d.diagnose(c,t,2)
        self.assertEqual(result['earliest_interior_equivalence_endpoint'],2)
        self.assertEqual(result['status'],'earliest_candidate_failed_future_verification')
        self.assertTrue(any(row['endpoint']>3 and row['is_candidate'] for row in scan))
    def test_terminal_only_match_is_not_a_future_witness(self):
        c=synthetic();t=synthetic(1.2);t['weights'][:30]=[.3,.3,.4]
        result,scan=d.diagnose(c,t,2)
        self.assertIsNone(result['earliest_interior_equivalence_endpoint'])
        self.assertTrue(result['terminal_state_equivalent']);self.assertIsNone(result['future_checks'])
    def test_equal_final_mdd_does_not_force_a_scale_witness(self):
        c=synthetic();t=synthetic(1.2);t['weights'][:]=[.3,.3,.4]
        result,scan=d.diagnose(c,t,2)
        self.assertTrue(result['terminal_mdd_equal_within_declared_tolerance'])
        self.assertIsNone(result['earliest_interior_equivalence_endpoint'])
    def test_changed_peak_or_running_mdd_blocks_sufficient_state(self):
        c=synthetic();t=synthetic(1.2)
        t['real_peak']*=1.1
        self.assertFalse(d.equivalent(d.state_errors(c,t,5)))
        t=synthetic(1.2);t['running_mdd'][5]=.01
        self.assertFalse(d.equivalent(d.state_errors(c,t,5)))
    def test_near_state_match_cannot_override_different_future_action(self):
        c=synthetic();t=synthetic(1.2);t['actions'][5]='restore'
        result,_=d.diagnose(c,t,2)
        self.assertEqual(result['status'],'earliest_candidate_failed_future_verification')
    def test_intervention_prestate_is_not_a_candidate(self):
        result,scan=d.diagnose(synthetic(),synthetic(1.2),4)
        self.assertEqual(scan[0]['endpoint'],4)
        self.assertEqual(result['earliest_interior_equivalence_endpoint'],4)
    def test_final_mdd_attainment_order_is_disclosed(self):
        result,_=d.diagnose(synthetic(),synthetic(1.2),2)
        self.assertEqual(result['control_final_mdd_first_attained_endpoint'],0)
        self.assertFalse(result['equivalence_strictly_before_both_final_mdd_attainments'])

if __name__=='__main__':unittest.main()
