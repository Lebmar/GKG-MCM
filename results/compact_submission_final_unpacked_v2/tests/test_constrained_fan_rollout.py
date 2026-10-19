"""Pure synthetic checks of constrained action choice and causal accounting."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
from experimental_execution.constrained_fan_rollout import (
    LOCAL_ALPHA, constrained_choices, evaluate_actions, simulate_constrained)
from experimental_execution.fan_rollout import evaluate_actions as base_evaluate
from tests.test_fan_rollout import scalar_fan, scalar_restore


def scalar_policy(actual, inflation, train, train_infl, target, loss, cost):
    """Independent scalar fan/fee replay; it never calls the new gate/step."""
    x=np.array([0.,0.,100.]);cp=1.;peak=100.;worst=0.;bad=False
    records={name:[] for name in ('holdings','real_wealth','real_peak','running_mdd',
        'ever_breached','fees','actions','qg_hold','qg_restore','qbad_hold_count','qbad_restore_count','gate_reason')}
    for key,value in (('holdings',x.copy()),('real_wealth',100.),('real_peak',peak),
                       ('running_mdd',worst),('ever_breached',bad)):records[key].append(value)
    for t in range(len(actual)):
        qh=qr=np.nan;kh=kr=-1
        if t==0:
            hl,hb=scalar_fan(x,cp,peak,worst,train,train_infl,t,target,loss,cost,'hold')
            rl,rb=scalar_fan(x,cp,peak,worst,train,train_infl,t,target,loss,cost,'restore')
            qh,qr=float(hl.mean()),float(rl.mean());kh,kr=int(sum(hb)),int(sum(rb))
            restore=True;reason='mandatory_initial_restore'
        elif bad:restore=False;reason='already_breached_pending_hold'
        else:
            hl,hb=scalar_fan(x,cp,peak,worst,train,train_infl,t,target,loss,cost,'hold')
            rl,rb=scalar_fan(x,cp,peak,worst,train,train_infl,t,target,loss,cost,'restore')
            qh,qr=float(hl.mean()),float(rl.mean());kh,kr=int(sum(hb)),int(sum(rb))
            cap=math.floor(.1*len(train)+1e-12)
            feasible=[a for a,k in [('hold',kh),('restore',kr)] if k<=cap]
            if feasible:
                if feasible==['restore']:restore=True;reason='risk_constraint_repair'
                elif feasible==['hold']:restore=False;reason='hold'
                else:restore=qr-qh>1e-12;reason='growth_improvement' if restore else 'hold'
            else:
                if kr<kh:restore=True
                elif kh<kr:restore=False
                else:restore=qr-qh>1e-12
                reason='least_violation_restore' if restore else 'hold'
        post,fee=scalar_restore(x,target,cost) if restore else (x.copy(),0.)
        x=post*(1+actual[t]);cp*=1+inflation[t]
        real=float(sum(x))/cp;peak=max(peak,real);worst=max(worst,1-real/peak)
        bad=bad or worst-loss>1e-12
        for key,value in (('holdings',x.copy()),('real_wealth',real),('real_peak',peak),('running_mdd',worst),
            ('ever_breached',bad),('fees',fee),('actions','restore' if restore else 'hold'),
            ('qg_hold',qh),('qg_restore',qr),('qbad_hold_count',kh),('qbad_restore_count',kr),('gate_reason',reason)):
            records[key].append(value)
    return {name:np.asarray(values) for name,values in records.items()}


class ConstrainedFanRolloutTests(unittest.TestCase):
    def arm(self,**changes):
        args=dict(actual_returns=np.zeros((2,4,3)),actual_inflation=np.zeros((2,4)),
            training_returns=np.zeros((3,4,3)),training_inflation=np.zeros((3,4)),
            target=[.4,.3,.3],L=.4)
        args.update(changes);return simulate_constrained(**args)

    def test_fixed_count_caps_and_inclusive_feasibility(self):
        self.assertEqual(LOCAL_ALPHA,.1)
        for n,cap in [(1,0),(2,0),(9,0),(10,1),(24,2),(72,7),(96,9),(120,12)]:
            with self.subTest(n=n):
                a=constrained_choices(0.,.1,np.array(cap),np.array(cap),n)
                self.assertEqual(a['local_breach_cap'],cap)
                self.assertTrue(a['hold_model_feasible']);self.assertTrue(a['restore_model_feasible'])
                self.assertTrue(a['recommend_restore']);self.assertFalse(a['local_model_infeasible'])

    def test_feasible_actions_maximize_already_net_growth_and_tie_holds(self):
        delta=np.array([-.1,0.,.5e-12,1e-12,2e-12,.1])
        a=constrained_choices(np.zeros(6),delta,np.zeros(6,dtype=int),np.ones(6,dtype=int),10)
        np.testing.assert_array_equal(a['recommend_restore'],[False,False,False,False,True,True])
        np.testing.assert_array_equal(a['gate_reason'],['hold','hold','hold','hold','growth_improvement','growth_improvement'])
        np.testing.assert_array_equal(a['economic_improvement'],a['recommend_restore'])
        np.testing.assert_array_equal(a['risk_cost_justification'],False)

    def test_only_feasible_restore_can_sacrifice_growth_and_reports_cap_repair(self):
        a=constrained_choices(np.zeros(3),[-.2,0.,.2],np.array([2,2,2]),np.array([1,1,1]),10)
        np.testing.assert_array_equal(a['recommend_restore'],True)
        np.testing.assert_array_equal(a['gate_reason'],'risk_constraint_repair')
        np.testing.assert_array_equal(a['restoration_predicted_growth_sacrifice'],[True,False,False])
        np.testing.assert_array_equal(a['economic_improvement'],[False,False,True])
        np.testing.assert_array_equal(a['risk_cost_justification'],[True,True,False])
        np.testing.assert_array_equal(a['repair_meets_cap'],True)
        np.testing.assert_array_equal(a['local_model_infeasible'],False)

    def test_only_feasible_hold_wins_even_if_restore_has_much_higher_growth(self):
        a=constrained_choices(0.,100.,np.array(1),np.array(2),10)
        self.assertFalse(a['recommend_restore']);self.assertEqual(a['gate_reason'],'hold')
        self.assertTrue(a['hold_model_feasible']);self.assertFalse(a['restore_model_feasible'])

    def test_both_infeasible_minimum_count_then_net_growth_then_hold(self):
        a=constrained_choices(np.zeros(6),[-1.,1.,1.,0.,1e-12,-1.],
            np.array([3,2,3,3,3,2]),np.array([2,3,3,3,3,2]),10)
        np.testing.assert_array_equal(a['recommend_restore'],[True,False,True,False,False,False])
        np.testing.assert_array_equal(a['local_model_infeasible'],True)
        np.testing.assert_array_equal(a['gate_reason'],['least_violation_restore','hold','least_violation_restore','hold','hold','hold'])
        np.testing.assert_array_equal(a['repair_meets_cap'],False)
        np.testing.assert_array_equal(a['risk_cost_justification'],[True,False,False,False,False,False])

    def test_non_growth_justification_implies_strictly_lower_risk_count(self):
        kh,kr,delta=np.meshgrid(np.arange(11),np.arange(11),np.array([-.5,-1e-13,0.,1e-12,.5]),indexing='ij')
        a=constrained_choices(np.zeros(delta.shape),delta,kh,kr,10)
        risk=a['risk_cost_justification'];restore=a['recommend_restore']
        self.assertTrue(np.all(kr[risk]<kh[risk]))
        np.testing.assert_array_equal(a['economic_improvement']|risk,restore)
        self.assertFalse(np.any(a['economic_improvement']&risk))

    def test_unsigned_bad_counts_do_not_wrap_negative_risk_delta(self):
        a=constrained_choices(0.,-.2,np.array(2,dtype=np.uint64),np.array(1,dtype=np.uint64),10)
        self.assertEqual(float(a['predicted_breach_count_delta_restore']),-1.)
        self.assertTrue(a['recommend_restore'])

    def test_inputs_reject_noninteger_counts_or_invalid_fan(self):
        for kh,kr,n in [(1.,0,10),(True,False,10),(-1,0,10),(0,11,10),(0,0,True),(0,0,0),(0,0,2.)]:
            with self.subTest(kh=kh,kr=kr,n=n),self.assertRaises(ValueError):constrained_choices(0.,.1,np.asarray(kh),np.asarray(kr),n)
        with self.assertRaises(ValueError):constrained_choices(np.nan,0.,np.array(0),np.array(0),2)

    def test_branch_values_exactly_reuse_frozen_B_evaluator_with_no_extra_fee(self):
        train=np.array([[[0.,0.,0.],[.004,0.,0.]]]);pi=np.zeros((1,2))
        args=([0.,0.,100.],1.,100.,0.,False,train,pi)
        kw=dict(year_index=1,target=[1.,0.,0.],L=.4,horizon=2)
        a=evaluate_actions(*args,**kw);b=base_evaluate(*args,**kw)
        for action in ('hold','restore'):
            for field,value in b[action].items():np.testing.assert_array_equal(a[action][field],value)
        self.assertAlmostEqual(a['predicted_growth_advantage_restore'][0],np.log(1.004/1.0025),places=14)
        self.assertTrue(a['recommend_restore'][0]);self.assertEqual(a['gate_reason'][0],'growth_improvement')

    def test_full_replay_risk_repair_has_negative_net_advantage_and_fee(self):
        actual=np.array([[[3.,0.,0.],[0.,0.,0.]]])
        train=np.zeros((2,2,3));train[:,1,0]=[2.,-.5]
        a=self.arm(actual_returns=actual,actual_inflation=np.zeros((1,2)),
            training_returns=train,training_inflation=np.zeros((2,2)),target=[.5,0,.5],L=.3)
        self.assertEqual(a['actions'][0,1],'restore')
        self.assertEqual(a['gate_reason'][0,1],'risk_constraint_repair')
        self.assertEqual(a['qbad_hold_count'][0,1],1);self.assertEqual(a['qbad_restore_count'][0,1],0)
        self.assertLess(a['predicted_growth_advantage_restore'][0,1],-1e-12)
        self.assertTrue(a['restoration_predicted_growth_sacrifice'][0,1])
        self.assertTrue(a['risk_cost_justification'][0,1]);self.assertFalse(a['economic_improvement'][0,1])
        self.assertTrue(a['repair_meets_cap'][0,1]);self.assertGreater(a['fees'][0,1],0)

    def test_actual_paths_match_independent_scalar_fan_and_self_financing_replay(self):
        rng=np.random.default_rng(97514)
        train=rng.uniform(-.4,.5,(5,5,3));train_pi=rng.uniform(-.02,.07,(5,5))
        actual=rng.uniform(-.35,.45,(3,5,3));pi=rng.uniform(-.02,.05,(3,5));w=np.array([.5,.2,.3])
        a=self.arm(actual_returns=actual,actual_inflation=pi,training_returns=train,
            training_inflation=train_pi,target=w,L=.22)
        for i in range(3):
            expected=scalar_policy(actual[i],pi[i],train,train_pi,w,.22,.0025)
            for key,value in expected.items():
                if value.dtype.kind in 'USb':np.testing.assert_array_equal(a[key][i],value)
                else:np.testing.assert_allclose(a[key][i],value,atol=5e-13,rtol=0,equal_nan=True)
        np.testing.assert_allclose(a['posttrade_holdings'].sum(axis=-1)+a['fees'],a['pretrade_wealth'],atol=3e-14,rtol=0)
        np.testing.assert_allclose(a['fees'],.0025*a['moved'],atol=5e-16,rtol=0)
        np.testing.assert_array_equal(a['holdings'][:,1:],a['posttrade_holdings']*(1+actual))

    def test_mandatory_initial_trade_has_no_new_economic_gate(self):
        a=self.arm(horizon=1)
        np.testing.assert_array_equal(a['actions'],'restore')
        self.assertTrue((a['fees'][:,0]>0).all())
        np.testing.assert_array_equal(a['real_wealth'][:,0],100.)
        np.testing.assert_array_equal(a['estimate_available'],True)
        self.assertTrue(np.isfinite(a['qg_hold']).all())
        self.assertTrue((a['predicted_growth_advantage_restore']<0).all())
        np.testing.assert_array_equal(a['local_constraint_applied'],False)
        np.testing.assert_array_equal(a['quoted_nominal_fee'],a['fees'])
        for key in ('economic_improvement','risk_cost_justification','repair_meets_cap'):
            np.testing.assert_array_equal(a[key],False)

    def test_initial_fee_enters_original_MDD_and_final_endpoint_without_liquidation(self):
        a=self.arm(horizon=1,L=0.)
        np.testing.assert_array_equal(a['first_crossing'],1)
        np.testing.assert_array_equal(a['available_followup'],0)
        self.assertEqual(a['real_wealth'].shape,(2,2));self.assertEqual(a['fees'].shape,(2,1))
        np.testing.assert_array_equal(a['terminal_real'],100.-a['nominal_fees'])
        self.assertTrue((a['mdd']>0).all());np.testing.assert_array_equal(a['volatility'],0.)

    def test_zero_returns_have_no_optional_transfer_or_fake_positive_growth(self):
        a=self.arm()
        np.testing.assert_array_equal(a['actions'][:,1:],'hold')
        np.testing.assert_array_equal(a['fees'][:,1:],0.)
        np.testing.assert_array_equal(a['gate_reason'][:,1:],'hold')
        np.testing.assert_array_equal(a['trade_count'],1)

    def test_pending_is_absorbing_and_has_no_model_comparison(self):
        actual=np.array([[[-.8,0,0],[4.1,0,0],[0,0,0]]])
        a=self.arm(actual_returns=actual,actual_inflation=np.zeros((1,3)),
            training_returns=np.zeros((2,3,3)),training_inflation=np.zeros((2,3)),target=[.75,.1,.15])
        np.testing.assert_array_equal(a['actions'][0],['restore','hold','hold'])
        np.testing.assert_array_equal(a['ever_breached'][0],[False,True,True,True])
        np.testing.assert_array_equal(a['estimate_available'][0],[True,False,False])
        np.testing.assert_array_equal(a['forecast_n'][0],[2,-1,-1])
        np.testing.assert_array_equal(a['local_model_infeasible'][0],False)
        self.assertTrue(np.isnan(a['qg_hold'][0,1:]).all())
        self.assertTrue(a['recovered'][0]);self.assertEqual(a['recovery_time'][0],1)
        self.assertEqual(a['first_crossing'][0],1)

    def test_CPI_peak_and_old_breach_survive_inflation_recovery(self):
        a=self.arm(actual_returns=np.zeros((1,3,3)),actual_inflation=[[1.,-.75,0.]],
            training_returns=np.zeros((2,3,3)),training_inflation=np.zeros((2,3)),target=[0,0,1])
        np.testing.assert_array_equal(a['real_wealth'][0],[100,50,200,200])
        np.testing.assert_array_equal(a['real_peak'][0],[100,100,200,200])
        np.testing.assert_array_equal(a['running_mdd'][0],[0,.5,.5,.5])
        np.testing.assert_array_equal(a['pending_at_decision'][0],[False,True,True])

    def test_initial_Q_risk_and_recommendation_do_not_override_mandatory_configuration(self):
        train=np.array([[[-.8,0.,0.]]])
        a=self.arm(actual_returns=np.zeros((1,1,3)),actual_inflation=[[0.]],
            training_returns=train,training_inflation=[[0.]],target=[.75,.1,.15])
        self.assertEqual(a['qbad_hold_count'][0,0],0)
        self.assertEqual(a['qbad_restore_count'][0,0],1)
        self.assertFalse(a['restore_model_feasible'][0,0])
        self.assertLess(a['predicted_growth_advantage_restore'][0,0],0)
        self.assertEqual(a['actions'][0,0],'restore')
        self.assertFalse(a['local_constraint_applied'][0,0])
        self.assertFalse(a['risk_cost_justification'][0,0])

    def test_same_original_horizon_and_training_calendar_suffix_are_preserved(self):
        rng=np.random.default_rng(616)
        train=rng.uniform(-.3,.3,(3,5,3));pi=rng.uniform(-.02,.05,(3,5))
        changed=train.copy();changed[:,[0,4]]=.9;pi2=pi.copy();pi2[:,[0,4]]=.8
        args=([45,20,35],1.,110.,.1,False)
        kw=dict(year_index=1,target=[.4,.3,.3],L=.4,horizon=4)
        a=evaluate_actions(*args,train,pi,**kw);b=evaluate_actions(*args,changed,pi2,**kw)
        self.assertEqual(a['horizon'],4);self.assertEqual(a['year_index'],1)
        for key in ('hold','restore'):
            for field,value in a[key].items():np.testing.assert_array_equal(value,b[key][field])

    def test_actual_current_or_future_returns_do_not_enter_current_action(self):
        actual=np.zeros((2,5,3));changed=actual.copy();changed[:,2:]=[-.7,.5,.1]
        pi=np.zeros((2,5));pi2=pi.copy();pi2[:,2:]=.3
        train=np.tile([.1,.03,.01],(3,5,1));tip=np.zeros((3,5))
        a=self.arm(actual_returns=actual,actual_inflation=pi,training_returns=train,training_inflation=tip)
        b=self.arm(actual_returns=changed,actual_inflation=pi2,training_returns=train,training_inflation=tip)
        for key in ('actions','qg_hold','qg_restore','qbad_hold_count','qbad_restore_count',
                    'gate_reason','posttrade_holdings','local_model_infeasible'):
            np.testing.assert_array_equal(a[key][:,:3],b[key][:,:3])

    def test_readonly_outputs_inputs_unchanged_metrics_and_joint_helper_compatible(self):
        actual=np.zeros((2,30,3));pi=np.zeros((2,30));train=np.zeros((3,30,3));tip=np.zeros((3,30));w=np.array([.4,.3,.3])
        a=self.arm(actual_returns=actual,actual_inflation=pi,training_returns=train,training_inflation=tip,target=w)
        for value in a.values():
            if isinstance(value,np.ndarray):self.assertFalse(value.flags.writeable)
        np.testing.assert_array_equal(actual,0);np.testing.assert_array_equal(train,0);np.testing.assert_array_equal(w,[.4,.3,.3])
        import run_joint_liquidity_refit as joint
        rows=joint.arm_records(a,[1,2],{'horizon':30,'loss_tolerance':.4,'reference_only':False})
        self.assertEqual(len(rows),2)
        for i,row in enumerate(rows):
            self.assertAlmostEqual(row['log_growth'],a['log_growth'][i],places=14)
            self.assertAlmostEqual(row['nominal_fees'],a['nominal_fees'][i],places=14)
        for key in ('log_growth','mdd','breach','excess','nominal_fees','trade_count'):
            self.assertEqual(a[key].shape,(2,))


if __name__=='__main__':unittest.main()
