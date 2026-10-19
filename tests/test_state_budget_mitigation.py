"""Synthetic V2b intervention checks; no supplied development/final markets."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"research")]
from safeinvest.core import execute_target
from experimental_execution import state_budget_execution as primary
from experimental_execution import state_budget_mitigation as m


def failed_state(**kwargs):
    support=np.array([[.2,.4,.6],[3.,1.,1.],[3.,1.,1.]])
    params=dict(target=[.45,.05,.5],L=.4,cost_rate=.0025,release_fraction=1.,forecast_rule="uniform")
    params.update(kwargs)
    return ([45.,5.,50.],1.,100.,0.,support,support,np.full((1,3),1/3)),params


def simulate(module,real_actual=None,real_training=None,**kwargs):
    actual=np.ones((2,3,3)) if real_actual is None else np.asarray(real_actual,float)
    training=np.ones((3,3,3)) if real_training is None else np.asarray(real_training,float)
    params=dict(target=[.45,.05,.5],L=.4,horizon=actual.shape[1],training_ids=[1,2,3],cost_rate=.0025,release_fraction=1.)
    params.update(kwargs)
    return module.simulate_state_budget(actual-1,np.zeros(actual.shape[:2]),training-1,np.zeros(training.shape[:2]),**params)


class StateBudgetMitigationTests(unittest.TestCase):
    def test_nine_points_and_exact_fee_postdollar_funding(self):
        x=np.array([55.,35.,10.]);target=np.array([.4,.1,.5]);support=np.array([[.2,.4,.6],[1.4,1.1,1.]])
        result=m.mitigate_support_failure(x,1.2,support,target=target,budget_floor=60.)
        self.assertEqual(result["candidate_posttrade"].shape,(9,3))
        np.testing.assert_array_equal(result["candidate_q"],[0,.5,1,.5,1,.5,1,.5,1])
        for j,alpha in enumerate(primary.TILTS):
            endpoint,fee,moved=execute_target(x,alpha*target+(1-alpha)*np.array([0,0,1]))
            for order,q in enumerate((.5,1.)):
                index=1+2*j+order
                np.testing.assert_allclose(result["candidate_posttrade"][index],x+q*(endpoint-x),atol=2e-14,rtol=0)
                self.assertAlmostEqual(result["candidate_fees"][index],q*fee,places=13)
                self.assertAlmostEqual(result["candidate_moved"][index],q*moved,places=12)
        np.testing.assert_allclose(result["candidate_posttrade"].sum(axis=1)+result["candidate_fees"],x.sum(),atol=2e-14,rtol=0)
        np.testing.assert_allclose(result["candidate_fees"],.0025*np.maximum(result["candidate_posttrade"]-x,0).sum(axis=1),atol=2e-15,rtol=0)

    def test_ranking_is_worst_nextwealth_not_mean_growth(self):
        support=np.array([[1.,1.,.9],[1.,1.,2.]])
        result=m.mitigate_support_failure([100.,0.,0.],1.,support,target=[0,0,1],cost_rate=0,budget_floor=90)
        self.assertGreater((support@result["candidate_posttrade"][2]).mean(),100.)
        self.assertEqual(result["selected_index"],0)

    def test_fee_can_remove_apparent_protective_improvement(self):
        support=np.tile([1.,1.,1.5],(3,1))
        free=m.mitigate_support_failure([100,0,0],1.,support,target=[0,0,1],cost_rate=0,budget_floor=120)
        costly=m.mitigate_support_failure([100,0,0],1.,support,target=[0,0,1],cost_rate=.99,budget_floor=120)
        self.assertGreater(free["selected_index"],0);self.assertEqual(costly["selected_index"],0)

    def test_global_tie_uses_hold_then_fee_proposal_and_q_order(self):
        constant=m.mitigate_support_failure([100,0,0],1.,np.ones((3,3)),target=[0,0,1],cost_rate=0,budget_floor=120)
        self.assertEqual(constant["selected_index"],0)
        # First row improves to100 at q=.5; second row caps all q at100.
        support=np.array([[.5,1.,1.5],[1.,1.,1.]])
        plateau=m.mitigate_support_failure([100,0,0],1.,support,target=[0,0,1],cost_rate=0,budget_floor=120)
        self.assertEqual(plateau["selected_index"],1);self.assertEqual(plateau["selected_q"],.5)

    def test_improvement_must_exceed_relative_tolerance(self):
        for advantage,expected in ((.5e-12,False),(2e-12,True)):
            support=np.tile([1.,1.,1.+advantage],(3,1))
            result=m.mitigate_support_failure([100,0,0],1.,support,target=[0,0,1],cost_rate=0,budget_floor=120)
            self.assertEqual(result["trade"],expected)

    def test_failed_primary_arrays_preserved_and_actual_forecast_describes_mitigation(self):
        args,kwargs=failed_state();old=primary.decide_state_action(*args,**kwargs);new=m.decide_state_action(*args,**kwargs)
        self.assertEqual(old["status"],"no_feasible_action")
        for key,value in old.items():
            if key.startswith(("candidate_","interval_")):
                np.testing.assert_array_equal(new[key],value)
        self.assertEqual(new["selected_index"],-1);self.assertFalse(new["support_feasible"])
        self.assertTrue(new["guard_triggered"]);self.assertTrue(new["mitigation_evaluated"])
        self.assertEqual(new["actual_decision_source"],"mitigation")
        self.assertEqual(new["primary_gate_reason"],old["gate_reason"])
        np.testing.assert_array_equal(new["primary_failed_hold_posttrade"],old["posttrade_holdings"])
        self.assertEqual(new["primary_failed_hold_score"],0.)
        self.assertLess(new["selected_score"],0.);self.assertTrue(new["growth_sacrifice"])
        self.assertGreater(new["worst_real_shortfall_selected"],0.)
        self.assertGreater(new["worst_real_shortfall_improvement"],0.)
        self.assertEqual(new["gate_reason"],"risk_mitigation_then_review")
        self.assertGreater(new["fee"],0.)

    def test_nine_forecast_witnesses_and_logfee_decomposition(self):
        args,kwargs=failed_state();support=args[4];weights=np.vstack((np.full(3,1/3),[.6,.2,.2],[1,0,0],[0,1,0],[0,0,1]))
        result=m.mitigate_support_failure(args[0],1.,support,target=kwargs["target"],budget_floor=60.,
            forecast_real_gross=support,forecast_weights=weights,real_peak=100.,running_mdd=0.,L=.4)
        x=np.asarray(args[0]);next_real=result["candidate_posttrade"]@support.T
        advantage=np.log(next_real/(support@x))@weights.T
        np.testing.assert_allclose(result["candidate_member_advantage"],advantage,atol=2e-16,rtol=0)
        logfee=np.log1p(-result["candidate_fees"]/x.sum())
        np.testing.assert_allclose(result["candidate_member_advantage"],result["candidate_member_gross_advantage"]+logfee[:,None],atol=8*np.finfo(float).eps,rtol=0)
        self.assertEqual(result["candidate_member_predicted_next_breach"].shape,(9,5))

    def test_primary_safe_policy_is_exactly_the_same_on_all_shared_fields(self):
        before=primary.simulate_state_budget
        old=simulate(primary);new=simulate(m)
        self.assertIs(primary.simulate_state_budget,before)
        for key,value in old["arm"].items():np.testing.assert_array_equal(new["arm"][key],value)
        self.assertFalse(new["arm"]["mitigation_evaluated"].any())
        self.assertTrue((new["arm"]["actual_decision_source"]=="primary").all())

    def test_first_failed_support_has_one_trade_then_absorbing_pending(self):
        training=np.ones((3,3,3));training[:,1]=[.2,.4,.6]
        actual=np.ones((2,3,3));actual[:,1]=[.2,.4,.6]
        old=simulate(primary,actual,training);new=simulate(m,actual,training);a,b=old["arm"],new["arm"]
        np.testing.assert_array_equal(a["holdings"][:,:2],b["holdings"][:,:2])
        np.testing.assert_array_equal(a["guard_pending"],b["guard_pending"])
        np.testing.assert_array_equal(a["automatic_service"],b["automatic_service"])
        np.testing.assert_array_equal(b["first_guard_trigger"],[2,2])
        np.testing.assert_array_equal(b["actual_decision_source"][0],["primary","mitigation","pending"])
        np.testing.assert_array_equal(b["mitigation_evaluated"][0],[False,True,False])
        self.assertTrue((b["actions"][:,2:]=="hold").all());self.assertFalse(b["support_feasible"][:,1].any())
        self.assertGreater(b["real_wealth"][0,2],a["real_wealth"][0,2])
        self.assertGreater(b["running_mdd"][0,2],.4)  # partial protection did not restore the client floor
        self.assertTrue(b["ever_breached"][:,2:].all());self.assertTrue((b["fees"][:,2:]==0).all())

    def test_initial_review_still_has_no_fallback_or_physical_account(self):
        training=np.tile([.2,.4,.6],(3,3,1));actual=np.full((2,3,3),np.nan)
        result=simulate(m,actual,training)
        self.assertEqual(result["status"],"preinvestment_review");self.assertIsNone(result["arm"])
        self.assertFalse(result["initial_witness"]["mitigation_evaluated"])

    def test_already_actual_breached_never_gets_new_mitigation_trade(self):
        training=np.ones((3,3,3));training[:,1]=[.2,.4,.6]
        actual=np.ones((2,3,3));actual[:,0]=.3
        old=simulate(primary,actual,training);new=simulate(m,actual,training)
        self.assertFalse(new["arm"]["mitigation_evaluated"].any())
        for key,value in old["arm"].items():np.testing.assert_array_equal(new["arm"][key],value)
        self.assertTrue((new["arm"]["actual_decision_source"][:,1:]=="pending").all())

    def test_no_improvement_means_hold_then_review_with_zero_extra_fee(self):
        training=np.ones((3,3,3));training[:,1]=.5
        old=simulate(primary,real_training=training);new=simulate(m,real_training=training)
        a=new["arm"];self.assertTrue(a["mitigation_evaluated"][:,1].all());self.assertFalse(a["mitigation_trade"].any())
        self.assertTrue((a["mitigation_selected_index"][:,1]==0).all());self.assertTrue((a["fees"][:,1:]==0).all())
        for key in ("holdings","real_wealth","running_mdd","fees","guard_pending"):
            np.testing.assert_array_equal(a[key],old["arm"][key])

    def test_realized_outside_support_can_make_protection_lose_growth(self):
        training=np.ones((3,3,3));training[:,1]=[.2,.4,.6]
        actual=np.ones((2,3,3));actual[:,1]=[3.,1.,1.]
        old=simulate(primary,actual,training);new=simulate(m,actual,training)
        self.assertLess(new["arm"]["real_wealth"][0,-1],old["arm"]["real_wealth"][0,-1])
        self.assertFalse(new["metadata"]["mitigation_floor_restoration_guarantee"])

    def test_actual_future_and_ids_do_not_change_first_mitigation_choice(self):
        training=np.ones((3,3,3));training[:,1]=[.2,.4,.6]
        a=np.ones((2,3,3));b=a.copy();b[:,1]=[3.,1.,1.]
        first=simulate(m,a,training,actual_ids=[2,4])["arm"];second=simulate(m,b,training,actual_ids=[80,90])["arm"]
        for key in ("mitigation_candidate_posttrade","mitigation_candidate_member_advantage","mitigation_selected_index","actual_previous_state"):
            np.testing.assert_array_equal(first[key][:,:2],second[key][:,:2])

    def test_all_new_arm_arrays_readonly_and_actual_fee_once(self):
        training=np.ones((3,3,3));training[:,1]=[.2,.4,.6]
        a=simulate(m,real_training=training)["arm"]
        np.testing.assert_allclose(a["posttrade_holdings"].sum(axis=2)+a["fees"],a["pretrade_wealth"],atol=3e-14,rtol=0)
        np.testing.assert_allclose(a["fees"],.0025*a["moved"],atol=3e-15,rtol=0)
        self.assertEqual(a["mitigation_candidate_member_advantage"].shape,(2,3,9,5))
        for value in a.values():
            if isinstance(value,np.ndarray):self.assertFalse(value.flags.writeable)


if __name__=="__main__":unittest.main()
