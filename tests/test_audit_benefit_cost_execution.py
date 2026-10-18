"""Synthetic independent branch/gate tests; no prospective numeric input."""
from pathlib import Path
from copy import deepcopy
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_benefit_cost_execution import independent_train_comparison, independent_benefit_action
import audit_benefit_cost_execution as audit


class IndependentBenefitCostTests(unittest.TestCase):
    def estimate(self, **changes):
        args = dict(holdings=[0., 0., 100.], cpi=1., peak=100., worst=0., target=[.5, .25, .25],
                    training_returns=np.zeros((1, 2, 3)), training_inflation=np.zeros((1, 2)),
                    year_index=0, horizon=2, economic_periods=1)
        args.update(changes)
        return independent_train_comparison(**args)

    def test_fee_gross_net_identity_matches_closed_form_current_purchase(self):
        r = np.array([[[.5, 0., 0.], [0., 0., 0.]]])
        e = self.estimate(training_returns=r)
        fee = 100 * .0025 * .75 / (1 + .0025 * .75)
        self.assertAlmostEqual(e["quoted_nominal_fee"], fee, places=13)
        self.assertAlmostEqual(e["fee_log_penalty"], -math.log1p(-fee / 100), places=14)
        self.assertAlmostEqual(e["mean_gross_log_gain"], math.log(1.25), places=14)
        self.assertAlmostEqual(e["mean_net_log_gain"], math.log(1.25) - e["fee_log_penalty"], places=14)

    def test_risk_uses_full_original_remaining_horizon_when_economic_horizon_is_one(self):
        r = np.array([[[.5, 0., 0.], [-.99, 0., 0.]]])
        e = self.estimate(training_returns=r)
        self.assertTrue(e["economic_pass"])
        self.assertEqual(e["benefit_periods"], 1)
        self.assertEqual((e["risk_hold_breach_count"], e["risk_restore_breach_count"]), (0, 1))
        self.assertFalse(e["risk_direction_pass"])
        self.assertGreater(e["restore_branches"][0]["running_mdd"][-1], .4)

    def test_economic_horizon_changes_economic_quantity_not_the_risk_horizon(self):
        r = np.array([[[.1, 0., 0.], [.1, 0., 0.]]])
        short = self.estimate(training_returns=r, economic_periods=1)
        long = self.estimate(training_returns=r, economic_periods=2)
        self.assertGreater(long["mean_net_log_gain"], short["mean_net_log_gain"])
        self.assertEqual(long["risk_hold_breach_count"], short["risk_hold_breach_count"])
        self.assertEqual(long["risk_restore_breach_count"], short["risk_restore_breach_count"])

    def test_actual_CPI_and_history_peak_are_not_normalized_away(self):
        r = np.array([[[9., 9., 9.], [0., 0., 0.]]]); pi = np.array([[9., .25]])
        e = self.estimate(cpi=2., peak=100., worst=.5, loss=.55, year_index=1,
                          training_returns=r, training_inflation=pi)
        self.assertEqual(e["risk_hold_breach_count"], 1)
        self.assertAlmostEqual(e["hold_branches"][0]["terminal_real"], 40.)
        self.assertAlmostEqual(e["hold_branches"][0]["running_mdd"][-1], .6)
        self.assertEqual(e["hold_branches"][0]["first_crossing_original_year"], 2)

    def test_inflation_cancels_from_paired_benefit_but_can_change_MDD_events(self):
        ordinary = self.estimate()
        inflated = self.estimate(training_inflation=np.array([[1., 0.]]))
        self.assertAlmostEqual(ordinary["mean_net_log_gain"], inflated["mean_net_log_gain"], places=14)
        self.assertEqual(ordinary["risk_hold_breach_count"], 0)
        self.assertEqual(inflated["risk_hold_breach_count"], 1)

    def test_every_training_scenario_is_evaluated_from_the_same_actual_state(self):
        r = np.zeros((3, 2, 3)); r[0, 0, 0] = .5; r[1, 0, 0] = -.5
        e = self.estimate(training_returns=r, training_inflation=np.zeros((3, 2)))
        self.assertEqual(e["forecast_n"], 3)
        self.assertEqual(len(e["sample_net_log_gain"]), 3)
        for branch in e["hold_branches"]:
            self.assertAlmostEqual(branch["real_wealth"][0], 100.)
            self.assertAlmostEqual(branch["real_peak"][0], 100.)

    def test_initial_positive_gain_is_required_but_initial_relative_risk_veto_is_excluded(self):
        action, reason = independent_benefit_action(year_index=0, ever_breached=False,
                                                   mean_net_log_gain=.01, hold_bad=0, restore_bad=1, forecast_n=1)
        self.assertEqual(action, "restore")
        self.assertEqual(reason, "mandatory_initial_restore_economic_positive")
        for value in (0., -1e-5, 1e-12):
            with self.subTest(value=value), self.assertRaises(ValueError):
                independent_benefit_action(year_index=0, ever_breached=False, mean_net_log_gain=value,
                                           hold_bad=0, restore_bad=0, forecast_n=1)

    def test_later_positive_gain_cannot_override_risk_veto_or_pending(self):
        self.assertEqual(independent_benefit_action(year_index=1, ever_breached=False,
                         mean_net_log_gain=.01, hold_bad=0, restore_bad=1, forecast_n=1)[0], "hold")
        self.assertEqual(independent_benefit_action(year_index=1, ever_breached=False,
                         mean_net_log_gain=.01, hold_bad=1, restore_bad=1, forecast_n=1)[0], "restore")
        self.assertEqual(independent_benefit_action(year_index=1, ever_breached=False,
                         mean_net_log_gain=0., hold_bad=1, restore_bad=0, forecast_n=1)[0], "hold")
        self.assertEqual(independent_benefit_action(year_index=1, ever_breached=True)[0], "hold")

    def test_original_calendar_and_risk_count_types_are_guarded(self):
        with self.assertRaises(ValueError): self.estimate(year_index=2)
        with self.assertRaises(ValueError): self.estimate(year_index=1, economic_periods=2)
        with self.assertRaises(ValueError): self.estimate(worst=.41)
        with self.assertRaises(ValueError):
            independent_benefit_action(year_index=1, ever_breached=False, mean_net_log_gain=.01,
                                       hold_bad=1., restore_bad=0, forecast_n=1)

    def test_independent_replay_matches_fee_paid_kernel_and_absorbing_history(self):
        # The production kernel is a fixture generator only; the auditor never
        # imports it and independently reconstructs every physical transition.
        sys.path[:0] = [str(audit.ROOT/'research'), str(audit.ROOT/'src')]
        from experimental_execution.benefit_cost_execution import simulate_benefit_cost
        training=np.zeros((3,30,3));training[:,:,0]=.03
        actual=np.zeros((1,30,3));actual[:,:,0]=.02;actual[:,3,:]=-.7
        pi=np.zeros((1,30));target=[.4,.3,.3]
        for mode in audit.METHODS.values():
            with self.subTest(mode=mode):
                arm=simulate_benefit_cost(actual,pi,training,np.zeros((3,30)),target,L=.4,horizon=30,cost_rate=.0025,benefit_horizon=mode)
                estimates={k:arm[k][0] for k in (*audit.QFLOATS,*audit.QCOUNTS,'economic_pass','risk_direction_pass','estimate_available','gate_reason','decision_reason')}
                replay=audit.independent_model_replay(actual[0],pi[0],target,training,np.zeros((3,30)),mode=mode,saved_estimates=estimates)
                independent=audit.independent_model_replay(actual[0],pi[0],target,training,np.zeros((3,30)),mode=mode)
                self.assertEqual(replay['status'],'complete');self.assertEqual(replay['account']['first_crossing'],4)
                self.assertTrue(replay['account']['pending_at_decision'][4:].all())
                saved={k:arm[k] if k=='targets' else arm[k][0] for k in replay['account'] if k in arm}
                audit.accounting.compare_saved_account(saved,replay['account'])
                audit.accounting.compare_saved_account(saved,independent['account'])
                for k in audit.accounting.METRICS:
                    audit.accounting.maybe(replay['metrics'][k],independent['metrics'][k],audit.AuditLedger(),'synthetic/metric')
                corrupt=deepcopy(estimates);corrupt['mean_net_log_gain'][0]+=.001
                with self.assertRaises(ValueError):
                    audit.independent_model_replay(actual[0],pi[0],target,training,np.zeros((3,30)),mode=mode,saved_estimates=corrupt)

    def test_success_and_failure_guards_precede_every_read(self):
        for name in ('independent_audit.json','audit_failure_receipt.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                out=Path(directory);(out/name).write_text('{}')
                with patch.object(audit,'OUT',out),patch.object(audit,'readj',side_effect=AssertionError('No reads allowed')):
                    with self.assertRaises(RuntimeError):audit.main()

    def test_synthetic_complete_pipeline_all_review_preserves_every_denominator(self):
        # No market/cache/results are read. The frozen split/protocol metadata
        # are permissible fixture inputs; the source runner only creates mocks.
        sys.path.insert(0,str(audit.ROOT/'tests'))
        from test_benefit_cost_pipeline import BenefitCostPipelineTests
        import run_benefit_cost_execution as runner
        p,b,r,pi,ids=BenefitCostPipelineTests.fixture('Review')
        frozen=json.loads(audit.PROTOCOL.read_text())
        for k in ('data','target_updates_enabled','original_H','loss_tolerance','cost_rate'):
            p[k]=frozen[k]
        parent=json.loads((audit.ROOT/'docs/refit_stability_protocol.json').read_text())
        full=json.loads((audit.ROOT/'docs/joint_liquidity_refit_protocol.json').read_text())
        out=runner.run_pipeline(p,b,r,pi,ids)
        allowed=list(range(5,22))+[23]+list(range(24,37))+[57]+list(range(58,198))
        self.assertEqual(len(allowed),172)
        weights=np.tile([1.,0.,0.],(232,1));weights[allowed]=[.3,.3,.4]
        cache={'weights':weights,'H30__mdd':np.zeros((120,232))}
        tables={name:audit.accounting.observed_bools(__import__('pandas').DataFrame(rows)) for name,rows in out['tables'].items()}
        result=audit.audit_artifacts(p,b,r,pi,ids,tables,out['choices'],out['groups'],out['arrays'],cache,parent=parent,full=full)
        self.assertEqual(result['counts']['training_rows'],20160)
        self.assertEqual(result['counts']['training_policy_witnesses'],260)
        self.assertEqual(result['counts']['initial_forecasts'],0)
        self.assertTrue(all(q['status']=='q1_review' for q in result['training_witness_inventory']))
        self.assertEqual(result['counts']['selected_trace_accounts'],720)
        corrupt={**tables,'client_paths':tables['client_paths'].copy()}
        index=corrupt['client_paths'].index[corrupt['client_paths'].policy.eq('benefit_cost_one_year')][0]
        corrupt['client_paths'].loc[index,'log_growth']=0.
        with self.assertRaises(ValueError):
            audit.audit_artifacts(p,b,r,pi,ids,corrupt,out['choices'],out['groups'],out['arrays'],cache,parent=parent,full=full)

    def test_synthetic_balanced_pipeline_checks_saved_forecasts_and_training_witnesses(self):
        sys.path.insert(0,str(audit.ROOT/'tests'))
        from test_benefit_cost_pipeline import BenefitCostPipelineTests
        import run_benefit_cost_execution as runner
        p,b,r,pi,ids=BenefitCostPipelineTests.fixture('Balanced')
        frozen=json.loads(audit.PROTOCOL.read_text())
        for k in ('data','target_updates_enabled','original_H','loss_tolerance','cost_rate'):
            p[k]=frozen[k]
        r[:]=0.;r[:,:,0]=.03
        parent=json.loads((audit.ROOT/'docs/refit_stability_protocol.json').read_text())
        full=json.loads((audit.ROOT/'docs/joint_liquidity_refit_protocol.json').read_text())
        engine=runner.simulate;arms={}
        def identical_paths(ar,ap,tr,target,policy):
            key=(len(tr[0]),tuple(target),policy)
            if key not in arms:arms[key]=engine(ar[:1],ap[:1],tr,target,policy)
            return {k:(v if k=='targets' else np.repeat(v,len(ar),axis=0)) if isinstance(v,np.ndarray) else v
                    for k,v in arms[key].items()}
        with patch.object(runner,'simulate',side_effect=identical_paths):
            out=runner.run_pipeline(p,b,r,pi,ids)
        allowed=list(range(5,22))+[23]+list(range(24,37))+[57]+list(range(58,198))
        weights=np.tile([1.,0.,0.],(232,1));weights[allowed]=[.3,.3,.4]
        weights[13]=[.2,.3,.5];weights[23]=[.4,.3,.3];weights[57]=[.7,.2,.1]
        cache={'weights':weights,'H30__mdd':np.zeros((120,232))}
        tables={name:audit.accounting.observed_bools(__import__('pandas').DataFrame(rows)) for name,rows in out['tables'].items()}
        # Identical synthetic returns deliberately allow memoizing the pure
        # independent estimator; no production audit computation is patched.
        original=audit.independent_train_comparison;memo={}
        def memoized(x,cp,peak,worst,target,tr,tpi,**kw):
            key=(tuple(x),cp,peak,worst,tuple(target),tr.shape,tpi.shape,tuple(sorted(kw.items())))
            if key not in memo:memo[key]=original(x,cp,peak,worst,target,tr,tpi,**kw)
            return memo[key]
        with patch.object(audit,'independent_train_comparison',side_effect=memoized):
            result=audit.audit_artifacts(p,b,r,pi,ids,tables,out['choices'],out['groups'],out['arrays'],cache,parent=parent,full=full)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['counts']['initial_forecasts'],260)
        self.assertEqual(result['counts']['training_policy_witnesses'],260)
        self.assertEqual(result['counts']['witness_forecast_states'],7800)
        self.assertEqual(result['counts']['selected_trace_accounts'],10440)
        self.assertEqual(result['counts']['independently_recomputed_subset_states'],1040)
        self.assertTrue(all(q['status']=='complete_independent_policy_replay' for q in result['training_witness_inventory']))


if __name__ == "__main__": unittest.main()
