"""Synthetic accounting, filtration and information-set checks; no raw data."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from safeinvest.core import execute_target, metric_arrays
from experimental_execution import state_budget_execution as m


def action(x=(0., 0., 100.), *, support=None, growth=None, weights=None, **kwargs):
    support = np.ones((3, 3)) if support is None else np.asarray(support, float)
    growth = support if growth is None else np.asarray(growth, float)
    weights = np.full((1, len(growth)), 1./len(growth)) if weights is None else weights
    params = dict(target=[.45, .05, .5], L=.4, cost_rate=.0025, release_fraction=1., forecast_rule="uniform")
    params.update(kwargs)
    return m.decide_state_action(x, 1., 100., 0., support, growth, weights, **params)


def simulation(real_actual=None, *, actual_inflation=None, real_training=None, training_inflation=None, **kwargs):
    a = np.ones((2, 3, 3)) if real_actual is None else np.asarray(real_actual, float)
    t = np.ones((3, a.shape[1], 3)) if real_training is None else np.asarray(real_training, float)
    ai = np.zeros(a.shape[:2]) if actual_inflation is None else np.asarray(actual_inflation, float)
    ti = np.zeros(t.shape[:2]) if training_inflation is None else np.asarray(training_inflation, float)
    params = dict(target=[.45, .05, .5], L=.4, horizon=a.shape[1], training_ids=np.arange(1, len(t)+1), release_fraction=1.)
    params.update(kwargs)
    return m.simulate_state_budget(a*(1+ai[..., None])-1, ai, t*(1+ti[..., None])-1, ti, **params)


class StateBudgetExecutionTests(unittest.TestCase):
    def test_interval_intersects_both_slopes_and_keeps_interior_when_endpoints_fail(self):
        result = m.segment_interval([-.2, .8, .1], [1., -1., 0.])
        self.assertTrue(result["feasible"]); self.assertAlmostEqual(result["lo"], .2); self.assertAlmostEqual(result["hi"], .8)
        self.assertEqual((result["lower_member"], result["upper_member"]), (0, 1))
        self.assertFalse(np.all(np.array([-.2, .8, .1]) >= 0))
        self.assertFalse(np.all(np.array([-.2, .8, .1])+np.array([1., -1., 0.]) >= 0))

    def test_interval_detects_zero_slope_and_contradictory_bounds(self):
        for a, b in (([-.1], [0.]), ([-2.], [1.]), ([.2, -.7], [-1., 1.])):
            result = m.segment_interval(a, b)
            self.assertFalse(result["feasible"]); self.assertTrue(np.isnan(result["lo"]))
        result = m.segment_interval([0.], [0.])
        self.assertEqual((result["lo"], result["hi"]), (0., 1.))

    def test_exact_postdollar_segment_fee_and_funding_at_interior_point(self):
        x = np.array([55., 35., 10.]); w = np.array([.1, .1, .8]); rate=.0025
        endpoint, fee, moved = execute_target(x, w, rate)
        for q in (0., .37, 1.):
            post = x+q*(endpoint-x)
            self.assertAlmostEqual(post.sum()+q*fee, x.sum(), places=12)
            self.assertAlmostEqual(rate*np.maximum(post-x, 0).sum(), q*fee, places=12)
            replay, actual_fee, actual_moved = execute_target(x, post/post.sum(), rate)
            np.testing.assert_allclose(replay, post, atol=2e-14, rtol=0)
            self.assertAlmostEqual(float(actual_fee), q*float(fee), places=13)
            self.assertAlmostEqual(float(actual_moved), q*float(moved), places=12)

    def test_release_fraction_preserves_correct_state_dependent_headroom(self):
        x = np.array([0., 0., 90.]); support=np.ones((3,3))
        result=m.decide_state_action(x,1.,100.,.1,support,support,np.full((1,3),1/3),target=[0,0,1],L=.2,release_fraction=.5)
        self.assertAlmostEqual(result["floor"],80.); self.assertAlmostEqual(result["budget_floor"],85.)
        self.assertTrue(result["hold_feasible"])

    def test_negative_roundoff_headroom_never_lowers_original_client_floor(self):
        real=90.-8e-11; worst=1-real/100.
        support=np.ones((3,3))
        result=m.decide_state_action([0,0,real],1.,100.,worst,support,support,np.full((1,3),1/3),
            target=[0,0,1],L=.1,release_fraction=.5,cost_rate=0)
        self.assertLessEqual(worst-.1,m.TOL)
        self.assertEqual(result["floor"],90.); self.assertEqual(result["budget_floor"],90.)
        self.assertGreaterEqual(result["budget_floor"],result["floor"])

    def test_positive_nominal_bill_is_not_a_real_safe_asset(self):
        real = np.full((3,3),1.02/1.05)
        result=action(support=real,L=.02,target=[0,0,1],cost_rate=0,initial=True)
        self.assertEqual(result["status"],"initial_review")

    def test_exact_initial_fee_reverses_boundary_feasibility(self):
        real = np.full((3,3),.9)
        zero=action(support=real,L=.1,target=[1,0,0],cost_rate=0,initial=True)
        paid=action(support=real,L=.1,target=[1,0,0],cost_rate=.0025,initial=True)
        self.assertEqual(zero["status"],"selected"); self.assertEqual(paid["status"],"initial_review")

    def test_joint_support_retains_hedge_instead_of_inventing_marginal_crash(self):
        real=np.array([[.5,1.5,1],[1.5,.5,1],[1,1,1]])
        result=action(support=real,target=[.5,.5,0],L=.1,cost_rate=0,initial=True)
        self.assertEqual(result["status"],"selected")
        self.assertTrue(np.min(real@result["posttrade_holdings"])>=90.)

    def test_hold_priority_on_ties_and_no_fee_only_churn(self):
        for rate in (0.,.0025):
            result=action(cost_rate=rate)
            self.assertEqual(result["selected_index"],0); self.assertEqual(result["fee"],0.)
            self.assertEqual(result["gate_reason"],"support_safe_hold")

    def test_safe_positive_growth_trade_and_finite_menu_only(self):
        real=np.tile([1.2,1.05,1.],(3,1))
        result=action(support=real,target=[1,0,0],cost_rate=0)
        self.assertGreater(result["selected_score"],m.TOL); self.assertEqual(result["gate_reason"],"support_safe_growth")
        self.assertAlmostEqual(result["posttrade_holdings"][0],100.)
        for j in range(4):
            values=result["candidate_lambda"][1+3*j:4+3*j]
            if result["interval_feasible"][j]:
                lo,hi=result["interval_lo"][j],result["interval_hi"][j]
                np.testing.assert_array_equal(values,[lo,(lo+hi)/2,hi])

    def test_paired_ensemble_score_is_min_member_advantage_not_difference_of_minima(self):
        real=np.array([[1.3,.9,1.],[.8,1.4,1.],[1.2,1.1,1.]])
        weights=np.vstack((np.full(3,1/3),[.8,.1,.1],[1,0,0],[0,1,0],[0,0,1]))
        result=action(support=real,weights=weights,forecast_rule="ensemble_min",target=[.5,.5,0])
        for i in np.flatnonzero(np.isfinite(result["candidate_lambda"])):
            post=result["candidate_posttrade"][i]
            expected=weights@np.log((real@post)/(real@np.array([0.,0.,100.])))
            np.testing.assert_allclose(result["candidate_member_advantage"][i],expected,atol=2e-16,rtol=0)
            self.assertAlmostEqual(result["candidate_score"][i],float(expected.min()),places=14)
        self.assertGreaterEqual(result["selected_score"],0.)

    def test_fee_and_gross_exposure_decomposition_exact_for_every_finite_point(self):
        real=np.array([[1.3,.9,1.],[.8,1.4,1.],[1.2,1.1,1.]])
        result=action(support=real,target=[.4,.4,.2])
        valid=np.isfinite(result["candidate_lambda"])
        np.testing.assert_allclose(result["candidate_member_advantage"][valid],
            result["candidate_member_gross_advantage"][valid]+result["candidate_log_fee"][valid,None],atol=3e-16,rtol=0)

    def test_risk_protection_can_sacrifice_growth_but_improves_afterfee_floor(self):
        real=np.array([[.5,1.,1.],[2.,1.,1.],[2.,1.,1.]])
        result=action(x=[100.,0.,0.],support=real,target=[0,0,1],L=.1)
        self.assertFalse(result["hold_feasible"]); self.assertTrue(result["support_feasible"])
        self.assertEqual(result["gate_reason"],"risk_budget_protection")
        self.assertTrue(result["growth_sacrifice"]); self.assertLess(result["selected_score"],0)
        self.assertGreater(result["fee"],0); self.assertGreater(result["worst_real_shortfall_improvement"],39.9)
        self.assertLess(result["worst_real_shortfall_selected"],1e-11)

    def test_selected_score_global_band_and_ties_use_declared_fee_proposal_order(self):
        real=np.tile([1.001,.999,1.],(3,1))
        result=action(support=real,target=[.5,.5,0],cost_rate=0)
        valid=np.flatnonzero(result["candidate_feasible"])
        maximum=result["candidate_score"][valid].max()
        if maximum<=m.TOL:
            self.assertEqual(result["selected_index"],0)
        else:
            band=[int(i) for i in valid if result["candidate_score"][i]>=maximum-m.TOL and (i==0 or result["candidate_score"][i]>m.TOL)]
            expected=min(band,key=lambda i:(i!=0,result["candidate_fees"][i],result["candidate_proposal"][i],i))
            self.assertEqual(result["selected_index"],expected)

    def test_inconsistent_history_and_invalid_forecast_weights_are_rejected(self):
        ones=np.ones((3,3)); uniform=np.full((1,3),1/3)
        for peak,worst in ((99.,0.),(120.,0.),(100.,.5)):
            with self.assertRaises(ValueError):
                m.decide_state_action([0,0,100],1.,peak,worst,ones,ones,uniform,target=[0,0,1],L=.4)
        with self.assertRaises(ValueError):
            action(weights=np.array([[.5,.25,.25]]))

    def test_preinvestment_review_has_no_account_and_does_not_inspect_actual_values(self):
        train=np.full((3,2,3),-.5); pi=np.zeros((3,2))
        actual=np.full((2,2,3),np.nan)
        result=m.simulate_state_budget(actual,np.full((2,2),np.nan),train,pi,[.45,.05,.5],L=.1,horizon=2,
            training_ids=[1,2,3],actual_ids=[80,2])
        self.assertEqual(result["status"],"preinvestment_review"); self.assertIsNone(result["arm"])
        self.assertEqual(result["preinvestment_review_n"],2); self.assertFalse(result["financial_available"])
        self.assertFalse(result["metadata"]["initial_decision_uses_actual_market"])

    def test_initial_full_target_never_cash_substituted_or_scenario_selected(self):
        train=np.tile([.5,1.,1.],(3,2,1)); actual=np.ones((2,2,3))
        result=simulation(actual,real_training=train,target=[1,0,0],L=.1,actual_ids=[2,80])
        self.assertEqual(result["status"],"preinvestment_review")
        witness=result["initial_witness"]
        self.assertEqual(np.flatnonzero(np.isfinite(witness["candidate_lambda"])).tolist(),[0,1])
        self.assertEqual(witness["candidate_lambda"][1],1.)

    def test_guard_review_trigger_counts_current_period_and_preserves_funded_finance(self):
        train=np.ones((3,3,3)); train[:,1]=.5
        result=simulation(real_training=train,target=[0,0,1],L=.2,cost_rate=0)
        arm=result["arm"]; self.assertEqual(result["status"],"funded")
        np.testing.assert_array_equal(arm["real_wealth"],100*np.ones((2,4)))
        np.testing.assert_array_equal(arm["first_guard_trigger"],[2,2])
        np.testing.assert_array_equal(arm["was_pending_input"][0],[False,False,True])
        np.testing.assert_array_equal(arm["pending_at_decision"][0],[False,True,True])
        np.testing.assert_array_equal(arm["automatic_service"][0],[True,False,False])
        np.testing.assert_array_equal(arm["guard_pending"][0],[False,False,True,True])
        self.assertFalse(arm["ever_breached"].any()); self.assertEqual(arm["gate_reason"][0,1],"no_feasible_declared_proposal_pending")
        self.assertTrue(arm["gate_evaluated"][0,1]); self.assertFalse(arm["gate_evaluated"][0,2])

    def test_original_real_peak_cpi_breach_and_recovery_survive_pending(self):
        real=np.broadcast_to(np.array([1.5,.5,2.2,1.1])[None,:,None],(2,4,3)).copy()
        pi=np.broadcast_to([.05,.1,.02,.03],(2,4)).copy()
        train=np.ones((3,4,3))
        result=simulation(real,actual_inflation=pi,real_training=train,target=[1,0,0],L=.2,cost_rate=0)
        arm=result["arm"]
        np.testing.assert_allclose(arm["real_wealth"][0],[100,150,75,165,181.5],atol=1e-12,rtol=0)
        np.testing.assert_allclose(arm["cpi"][0,1:],np.cumprod(1+pi[0]),rtol=0,atol=2e-16)
        np.testing.assert_array_equal(arm["first_crossing"],[2,2]); np.testing.assert_array_equal(arm["recovery_time"],[1,1])
        self.assertTrue(arm["ever_breached"][:,2:].all()); self.assertTrue((arm["actions"][:,2:]=="hold").all())
        np.testing.assert_array_equal(arm["fees"][:,2:],0)
        metrics=metric_arrays(arm["nominal_wealth"],pi)
        np.testing.assert_allclose(metrics["mdd"],arm["running_mdd"][:,-1],atol=2e-16,rtol=0)

    def test_actual_future_or_scenario_labels_cannot_change_prior_actions(self):
        train=np.ones((3,3,3)); train[:,:,0]=1.05
        actual=np.ones((2,3,3)); alternative=actual.copy(); alternative[:,1]=[1.8,.7,1.1]
        first=simulation(actual,real_training=train,forecast_rule="ensemble_min",actual_ids=[11,2])["arm"]
        second=simulation(alternative,real_training=train,forecast_rule="ensemble_min",actual_ids=[90,4])["arm"]
        for name in ("actions","selected_lambda","selected_score","candidate_member_advantage","actual_previous_state"):
            np.testing.assert_array_equal(first[name][:,:2],second[name][:,:2])
        np.testing.assert_allclose(first["training_previous_state"][1,:,0],.05,rtol=0,atol=1e-15)

    def test_pooled_is_explicit_stress_not_silently_used_for_calendar_admission(self):
        train=np.ones((3,2,3)); train[:,1,0]=.4
        actual=np.ones((2,2,3))
        calendar=simulation(actual,real_training=train,target=[1,0,0],L=.2,support_mode="calendar")
        pooled=simulation(actual,real_training=train,target=[1,0,0],L=.2,support_mode="pooled")
        self.assertEqual(calendar["status"],"funded"); self.assertEqual(pooled["status"],"preinvestment_review")

    def test_exact_period_funding_no_double_fee_and_readonly_traces(self):
        train=np.tile(np.array([1.1,1.02,1.]),(3,3,1)); actual=np.broadcast_to(train[0],(2,3,3)).copy()
        arm=simulation(actual,real_training=train)["arm"]
        np.testing.assert_allclose(arm["posttrade_holdings"].sum(axis=2)+arm["fees"],arm["pretrade_wealth"],atol=3e-14,rtol=0)
        np.testing.assert_allclose(arm["fees"],.0025*arm["moved"],atol=3e-15,rtol=0)
        np.testing.assert_allclose(arm["holdings"][:,1:],arm["posttrade_holdings"]*actual,atol=2e-14,rtol=0)
        self.assertEqual(arm["candidate_member_advantage"].shape,(2,3,13,5))
        np.testing.assert_allclose(arm["realized_real_gross"],actual,atol=2e-16,rtol=0)
        np.testing.assert_allclose(arm["training_real_gross"],np.transpose(train,(1,0,2)),atol=2e-16,rtol=0)
        for value in arm.values():
            if isinstance(value,np.ndarray):
                self.assertFalse(value.flags.writeable)

    def test_forecasts_are_batched_once_per_active_year_not_per_account(self):
        with mock.patch.object(m,"build_forecast_ensemble",wraps=m.build_forecast_ensemble) as build:
            result=simulation()
        self.assertEqual(result["status"],"funded")
        self.assertEqual(build.call_count,4)  # one initial admission + three active years
        self.assertEqual([call.kwargs["n_actual"] for call in build.call_args_list],[1,2,2,2])


if __name__=="__main__":
    unittest.main()
