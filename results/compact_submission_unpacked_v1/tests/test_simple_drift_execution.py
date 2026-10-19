"""Synthetic checks only: no real data, fit, saved study or forecast is read."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
from experimental_execution.simple_drift_execution import (
    DEFAULT_THRESHOLDS,restoration_mask,simulate_drift)
from experimental_execution.review_pending import simulate_paired


class SimpleDriftExecutionTests(unittest.TestCase):
    def run_arm(self,threshold=.05,**changes):
        args=dict(returns=np.zeros((1,3,3)),inflation=np.zeros((1,3)),target=[.4,.3,.3],L=.4,cost_rate=.0025)
        args.update(changes)
        return simulate_drift(threshold=threshold,**args)

    def test_declared_three_candidate_family_only(self):
        self.assertEqual(DEFAULT_THRESHOLDS,(0.,.05,.1))
        for bad in (True,False,.025,-.05,.2,float('nan'),float('inf'),'0.05',None):
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.run_arm(threshold=bad)

    def test_threshold_zero_is_byte_exact_old_annual_pending_on_synthetic_paths(self):
        rng=np.random.default_rng(6188)
        r=rng.uniform(-.35,.32,(7,30,3));pi=rng.uniform(-.015,.07,(7,30));w=[.45,.05,.5]
        new=simulate_drift(r,pi,w,0.,L=.15)
        old=simulate_paired(r,pi,w,.15)['pending_hold']
        for key,value in old.items():
            if key in ('targets','loss_tolerances'):continue
            expected=value if key=='cpi' else value[:,0]
            with self.subTest(key=key):np.testing.assert_array_equal(new[key],expected)
        np.testing.assert_array_equal(new['decision_reason'][new['pending_at_decision']],'already_breached_pending_hold')

    def test_strict_threshold_equality_and_no_added_tolerance(self):
        tv=np.array([np.nextafter(.05,0),.05,np.nextafter(.05,1),.1])
        np.testing.assert_array_equal(restoration_mask(tv,np.zeros(4,bool),.05),[False,False,True,True])
        np.testing.assert_array_equal(restoration_mask(np.zeros(3),np.array([False,True,False]),0.),[True,False,True])
        np.testing.assert_array_equal(restoration_mask(np.zeros(3),np.array([False,True,False]),.1,initial=True),[True,False,True])

    def test_twenty_money_transfer_costs_point05_once_at_inception(self):
        w=[20/99.95,0,79.95/99.95]
        a=self.run_arm(target=w,returns=np.zeros((1,1,3)),inflation=np.zeros((1,1)))
        self.assertAlmostEqual(a['moved'][0,0],20,places=12)
        self.assertAlmostEqual(a['fees'][0,0],.05,places=13)
        self.assertAlmostEqual(a['terminal_real'][0],99.95,places=12)
        self.assertEqual(a['real_wealth'][0,0],100)
        self.assertEqual(a['trade_count'][0],1)
        self.assertEqual(a['volatility'][0],0)
        self.assertEqual(a['fees'].shape,(1,1))

    def test_zero_returns_charge_only_actual_initial_transfer(self):
        for delta in DEFAULT_THRESHOLDS:
            with self.subTest(delta=delta):
                a=self.run_arm(threshold=delta)
                self.assertGreater(a['fees'][0,0],0)
                np.testing.assert_array_equal(a['fees'][0,1:],0)
                np.testing.assert_array_equal(a['moved'][0,1:],0)
                self.assertEqual(a['trade_count'][0],1)
                self.assertEqual(a['target_execution_count'][0],3 if delta==0 else 1)
                self.assertEqual(a['decision_reason'][0,1],'annual_restore_threshold_zero' if delta==0 else 'tv_within_threshold_hold')

    def test_all_bills_identity_restores_are_not_money_transfers(self):
        for delta in DEFAULT_THRESHOLDS:
            a=self.run_arm(threshold=delta,target=[0,0,1])
            np.testing.assert_array_equal(a['real_wealth'],100)
            np.testing.assert_array_equal(a['fees'],0)
            np.testing.assert_array_equal(a['moved'],0)
            np.testing.assert_array_equal(a['trades'],False)
            self.assertEqual(a['trade_count'][0],0)
            self.assertEqual(a['target_execution_count'][0],3 if delta==0 else 1)

    def test_self_financing_cpi_and_period_fraction_identity(self):
        rng=np.random.default_rng(7102);r=rng.uniform(-.15,.2,(4,9,3));pi=rng.uniform(-.02,.04,(4,9))
        for delta in DEFAULT_THRESHOLDS:
            a=self.run_arm(threshold=delta,returns=r,inflation=pi,L=.3)
            np.testing.assert_allclose(a['posttrade_holdings'].sum(axis=-1)+a['fees'],a['pretrade_wealth'],atol=3e-13,rtol=0)
            np.testing.assert_allclose(a['fees'],.0025*a['moved'],atol=5e-15,rtol=0)
            np.testing.assert_array_equal(a['holdings'][:,1:,:],a['posttrade_holdings']*(1+r))
            np.testing.assert_array_equal(a['real_wealth'],a['nominal_wealth']/a['cpi'])
            np.testing.assert_array_equal(a['fee_fraction'],a['fees']/a['pretrade_wealth'])
            np.testing.assert_array_equal(a['moved_fraction'],a['moved']/a['pretrade_wealth'])
            np.testing.assert_allclose(a['weights'].sum(axis=-1),1,atol=3e-16,rtol=0)

    def test_actions_use_only_pre_return_state(self):
        r=np.array([[[.2,0,0],[.01,.02,.03],[-.2,.1,0]]])
        b=r.copy();b[:,1,:]=[-.8,.5,.9]
        a=self.run_arm(returns=r,cost_rate=0);c=self.run_arm(returns=b,cost_rate=0)
        np.testing.assert_array_equal(a['actions'][:,:2],c['actions'][:,:2])
        np.testing.assert_array_equal(a['pretrade_tv_drift'][:,:2],c['pretrade_tv_drift'][:,:2])
        self.assertNotEqual(a['terminal_real'][0],c['terminal_real'][0])

    def test_large_pre_return_drift_restores_but_small_drift_holds(self):
        r=np.array([[[1.,0,0],[0,0,0],[0,0,0]]])
        a=self.run_arm(returns=r,target=[.5,.5,0],cost_rate=0,threshold=.1)
        self.assertGreater(a['pretrade_tv_drift'][0,1],.1)
        np.testing.assert_array_equal(a['actions'][0],['restore','restore','hold'])
        np.testing.assert_array_equal(a['weights'][0,2],[.5,.5,0])

    def test_breach_is_absorbing_and_does_not_undo_crossing_trade(self):
        r=np.array([[[-.8,0,0],[3,0,0],[-.2,0,0]]])
        for delta in DEFAULT_THRESHOLDS:
            a=self.run_arm(threshold=delta,returns=r,target=[.75,.1,.15],L=.4)
            self.assertEqual(a['first_crossing'][0],1)
            np.testing.assert_array_equal(a['actions'][0],['restore','hold','hold'])
            np.testing.assert_array_equal(a['pending_at_decision'][0],[False,True,True])
            self.assertGreater(a['fees'][0,0],0)
            np.testing.assert_array_equal(a['fees'][0,1:],0)
            np.testing.assert_array_equal(a['ever_breached'][0],[False,True,True,True])
            np.testing.assert_array_equal(a['holdings'][0,2],a['holdings'][0,1]*[4,1,1])

    def test_inflation_recovery_updates_peak_without_reset_or_reentry(self):
        a=self.run_arm(threshold=.1,target=[0,0,1],inflation=np.array([[1.,-.75,0.]]))
        np.testing.assert_array_equal(a['real_wealth'][0],[100,50,200,200])
        np.testing.assert_array_equal(a['real_peak'][0],[100,100,200,200])
        np.testing.assert_array_equal(a['running_mdd'][0],[0,.5,.5,.5])
        np.testing.assert_array_equal(a['ever_breached'][0],[False,True,True,True])
        np.testing.assert_array_equal(a['actions'][0],['restore','hold','hold'])
        self.assertEqual(a['recovery_time'][0],1)
        self.assertEqual(a['peak_at_trigger'][0],100)
        self.assertEqual(a['current_drawdown'][0,-1],0)

    def test_loss_boundary_keeps_original_numeric_equality_convention(self):
        for drop,expected in ((.4,False),(.4+.5e-12,False),(.4+2e-12,True)):
            a=self.run_arm(returns=np.full((1,1,3),-drop),inflation=np.zeros((1,1)),target=[0,0,1],L=.4)
            self.assertEqual(bool(a['ever_breached'][0,-1]),expected)

    def test_crossing_at_original_horizon_has_zero_followup_no_terminal_fee(self):
        r=np.array([[[0.,0.,0.],[-.7,-.7,-.7]]])
        a=self.run_arm(returns=r,inflation=np.zeros((1,2)),target=[0,0,1])
        self.assertEqual(a['first_crossing'][0],2)
        self.assertEqual(a['available_followup'][0],0)
        self.assertTrue(a['right_censored'][0])
        self.assertEqual(a['recovery_time'][0],-1)
        self.assertEqual(a['fees'].shape,(1,2))
        self.assertEqual(a['nominal_fees'][0],0)

    def test_inception_fee_enters_annual_mdd_and_no_fee_instant_endpoint(self):
        a=self.run_arm(returns=np.zeros((1,1,3)),inflation=np.zeros((1,1)),L=0)
        self.assertEqual(a['real_wealth'].shape,(1,2))
        self.assertEqual(a['first_crossing'][0],1)
        self.assertGreater(a['mdd'][0],0)
        self.assertEqual(a['real_peak'][0,0],100)

    def test_no_terminal_restore_after_last_return_creates_drift(self):
        r=np.array([[[0,0,0],[1,0,0]]])
        a=self.run_arm(returns=r,inflation=np.zeros((1,2)),cost_rate=0)
        np.testing.assert_array_equal(a['actions'][0],['restore','hold'])
        self.assertGreater(a['end_year_tv_drift'][0,-1],.05)
        self.assertEqual(a['target_execution_count'][0],1)
        self.assertEqual(a['moved'][0,1],0)

    def test_arrays_are_readonly_and_inputs_unchanged(self):
        r=np.zeros((2,3,3));pi=np.zeros((2,3));w=np.array([.4,.3,.3])
        a=self.run_arm(returns=r,inflation=pi,target=w)
        for value in a.values():
            if isinstance(value,np.ndarray):self.assertFalse(value.flags.writeable)
        np.testing.assert_array_equal(r,0);np.testing.assert_array_equal(pi,0);np.testing.assert_array_equal(w,[.4,.3,.3])

    def test_original_horizon_slices_without_reset_and_validates_bad_inputs(self):
        a=self.run_arm(horizon=1)
        self.assertEqual(a['real_wealth'].shape,(1,2))
        for change in (dict(horizon=True),dict(horizon=0),dict(horizon=4),dict(L=True),dict(L=-.1),dict(cost_rate=True),
                       dict(cost_rate=1),dict(target=[.4,.3,.2]),dict(target=[-.1,.5,.6]),dict(target=[[.4,.3,.3]]),
                       dict(inflation=np.zeros((1,2))),dict(returns=np.zeros((1,3,2))),dict(returns=np.zeros((0,3,3))),
                       dict(inflation=np.full((1,3),-1)),dict(returns=np.full((1,3,3),float('inf')))):
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_arm(**change)

    def test_squeezed_arm_compatible_with_existing_joint_financial_and_trace_helpers(self):
        import run_joint_liquidity_refit as joint
        a=self.run_arm(returns=np.zeros((2,30,3)),inflation=np.zeros((2,30)))
        values=joint.arm_records(a,[1,2],{'horizon':30,'loss_tolerance':.4,'reference_only':False})
        self.assertEqual(len(values),2);self.assertTrue(all(v['assigned'] for v in values))
        for i,v in enumerate(values):
            self.assertAlmostEqual(v['log_growth'],a['log_growth'][i],places=14)
            self.assertAlmostEqual(v['nominal_fees'],a['nominal_fees'][i],places=14)
        archive={};groups={};joint.save_trace(archive,groups,'synthetic',a,[1,2],{},kind='simple_drift')
        self.assertEqual(groups['synthetic']['scenario_ids'],[1,2])
        np.testing.assert_array_equal(archive['synthetic__fee_fraction'],a['fee_fraction'])
