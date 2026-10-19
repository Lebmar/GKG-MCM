"""Synthetic controls for one forced action and unchanged policy feedback."""
import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np

from experimental_execution.failure_action_intervention import replay_intervention
from experimental_execution.risk_priced_fan_rollout import simulate_prices

_path=Path(__file__).resolve().parents[1]/'scripts/run_failure_action_intervention.py'
_spec=importlib.util.spec_from_file_location('intervention_runner_tests',_path)
runner=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(runner)


class InterventionTests(unittest.TestCase):
    def fixture(self):
        H = 5
        actual = np.array([[.20,.02,.01],[-.12,.03,.01],[.25,-.01,.01],[-.08,.02,.01],[.12,.02,.01]])
        train = np.stack([actual, actual*.6, actual*1.2])
        inflation = np.array([.02,.01,.03,.02,.01])
        train_pi = np.broadcast_to(inflation, (3,H)).copy()
        target = np.array([.4,.3,.3])
        original = simulate_prices(actual[None], inflation[None], train, train_pi, target, [.03], L=.4, horizon=H)
        return actual, inflation, train, train_pi, target, original

    def replay(self, data, t=1, **overrides):
        r, pi, tr, tp, w, old = data
        args = dict(holdings=old['holdings'][0,0,t], cpi=old['cpi'][0,t],
            real_peak=old['real_peak'][0,0,t], running_mdd=old['running_mdd'][0,0,t],
            year_index=t, forced_first_action=str(old['actions'][0,0,t]), horizon=len(r))
        args.update(overrides)
        return replay_intervention(r,pi,tr,tp,w,.03,**args)

    def test_original_action_identity_reproduces_all_future_states_Q_and_actions(self):
        data = self.fixture(); out = self.replay(data); old = data[-1]; t = 1
        for key in ('holdings','weights','nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd','ever_breached'):
            np.testing.assert_allclose(out[key],old[key][0,0,t:],atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(out['cpi'],old['cpi'][0,t:],atol=1e-14,rtol=0)
        for key in ('actions','policy_actions'):
            np.testing.assert_array_equal(out[key],old['actions'][0,0,t:])
        for key in ('qg_hold','qg_restore','qbad_hold_count','qbad_restore_count','forecast_n','predicted_score','fees','moved','posttrade_holdings'):
            np.testing.assert_allclose(out[key],old[key][0,0,t:],atol=1e-13,rtol=1e-13,equal_nan=True)

    def test_only_first_action_is_forced_then_state_is_replanned(self):
        H=4; r=np.zeros((H,3)); r[:,0]=.1
        tr=np.broadcast_to(r,(3,H,3)).copy(); z=np.zeros(H)
        out=replay_intervention(r,z,tr,np.zeros((3,H)),[.6,.2,.2],0.,
            holdings=[0,0,100],cpi=1.,real_peak=100.,running_mdd=0.,year_index=1,
            forced_first_action='hold',horizon=H)
        self.assertEqual(out['policy_actions'][0],'restore')
        self.assertEqual(out['actions'][0],'hold')
        self.assertEqual(out['actions'][1],'restore')
        np.testing.assert_array_equal(out['forced'],[True,False,False])

    def test_own_breach_holds_absorbingly_even_after_full_recovery(self):
        H=4;r=np.zeros((H,3));r[1,0]=-.7;r[2:,0]=3.
        out=replay_intervention(r,np.zeros(H),np.zeros((2,H,3)),np.zeros((2,H)),[1,0,0],0.,
            holdings=[0,0,100],cpi=1.,real_peak=100.,running_mdd=0.,year_index=1,
            forced_first_action='restore',horizon=H)
        self.assertEqual(out['first_crossing_original_year'],2)
        np.testing.assert_array_equal(out['actions'],['restore','hold','hold'])
        self.assertGreater(out['real_wealth'][-1],100)
        self.assertTrue(out['terminal_breach'])
        self.assertTrue(np.isnan(out['qg_hold'][1:]).all())
        np.testing.assert_array_equal(out['forecast_n'][1:],[-1,-1])

    def test_actual_next_return_is_not_an_input_to_its_action_values(self):
        data=self.fixture(); one=self.replay(data)
        altered=list(data); altered[0]=data[0].copy();altered[0][2:,0]=.8
        two=self.replay(tuple(altered))
        for key in ('actions','policy_actions','qg_hold','qg_restore','predicted_score'):
            if key in ('actions','policy_actions'):np.testing.assert_array_equal(one[key][:2],two[key][:2])
            else:np.testing.assert_allclose(one[key][:2],two[key][:2],rtol=0,atol=0)

    def test_original_clock_and_peak_are_not_reset(self):
        out=self.replay(self.fixture(),t=3)
        np.testing.assert_array_equal(out['original_year'],[4,5])
        self.assertEqual(out['original_horizon'],5)
        self.assertEqual(out['original_year_index'],3)
        self.assertGreaterEqual(out['running_mdd'][0],out['current_drawdown'][0])
        self.assertEqual(out['holdings'].shape,(3,3))

    def test_final_year_single_paid_restore_matches_analytic_fee(self):
        H=3;r=np.zeros((H,3));r[-1,0]=.5
        out=replay_intervention(r,np.zeros(H),np.zeros((2,H,3)),np.zeros((2,H)),[.5,.25,.25],0.,
            holdings=[0,0,100],cpi=2.,real_peak=60.,running_mdd=.2,year_index=2,
            forced_first_action='restore',horizon=H)
        fee=100*.0025*.75/(1+.0025*.75)
        self.assertAlmostEqual(out['fees'][0],fee,places=13)
        self.assertAlmostEqual(out['terminal_real'],(100-fee)*1.25/2,places=13)
        self.assertAlmostEqual(out['total_remaining_log_growth'],np.log((100-fee)*1.25/100),places=13)

    def test_risk_first_uses_separate_nan_score(self):
        r,pi,tr,tp,w,old=self.fixture();t=1
        out=replay_intervention(r,pi,tr,tp,w,None,holdings=old['holdings'][0,0,t],
            cpi=old['cpi'][0,t],real_peak=old['real_peak'][0,0,t],running_mdd=old['running_mdd'][0,0,t],
            year_index=t,forced_first_action='hold',horizon=5)
        self.assertTrue(np.isnan(out['predicted_score']).all())

    def test_rejects_inception_pending_and_changed_horizon(self):
        data=self.fixture()
        for values in ({'year_index':0},{'year_index':5},{'running_mdd':.5},{'horizon':6}):
            with self.assertRaises(ValueError):self.replay(data,**values)

    def test_output_state_arrays_are_readonly(self):
        out=self.replay(self.fixture())
        with self.assertRaises(ValueError):out['real_wealth'][0]=0

    def test_unfrozen_main_never_reads_numeric_parent_or_market(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'p.json';path.write_text(json.dumps({'execution_ready':False}))
            with patch.object(runner,'PROTOCOL',path),patch.object(runner,'snapshots',side_effect=AssertionError('no numeric read')):
                with self.assertRaises(RuntimeError):runner.main()

    def test_all_three_nominal_witness_roles_remain_on_no_intervention(self):
        item={'pair_id':'synthetic','scenario_id':35,'training_fan_n':90}
        rows=runner.witness_rows(item,'control',None,None)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['status']=='no_action_difference_skip' for r in rows))
        self.assertTrue(all(r['qg_hold'] is None for r in rows))

    def test_terminal_intervention_keeps_outside_and_duplicate_roles(self):
        item={'pair_id':'synthetic','scenario_id':35,'training_fan_n':90}
        a={key:np.zeros(30) for key in ('qg_hold','qg_restore','predicted_score')}
        a.update({key:np.zeros(30,dtype=np.int64) for key in ('qbad_hold_count','qbad_restore_count','forecast_n')})
        a.update(forced=np.zeros(30,bool),pending_at_decision=np.zeros(30,bool),policy_actions=np.full(30,'hold'),actions=np.full(30,'hold'))
        rows=runner.witness_rows(item,'treatment',a,30)
        self.assertEqual([r['status'] for r in rows],['active_Q','outside_original_horizon_skip','duplicate_year_alias'])
        self.assertEqual(rows[2]['alias_of'],rows[0]['slot_id'])
        self.assertIsNone(rows[1]['qg_hold'])

    def test_missing_frozen_closure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'p.json';path.write_text('{}')
            with patch.object(runner,'PROTOCOL',path):
                with self.assertRaises(RuntimeError):runner.snapshots({'execution_sources_sha256':{},'frozen_input_sha256':{}},runner.sha(path))


if __name__=='__main__':unittest.main()
