"""Final auditor pure synthetic checks: never load real market/result values."""
from pathlib import Path
import sys, unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_final_locked_evaluation as a
from audit_review_pending_execution import independent_trade


def metadata():
    sources = {'synthetic.py': 's' * 64}; inputs = {'synthetic.csv': 'i' * 64}
    p = dict(final_execution_ready=False, development_execution_ready=True, development_ids=list(range(1,121)),
        final_ids=list(range(121,181)), execution_sources_sha256=sources, frozen_input_sha256=inputs,
        object_specification_sha256='o' * 64)
    release = dict(version='locked-final-evaluation-release-v1', final_execution_ready=True,
        preparation_protocol_sha256='p'*64, execution_sources_sha256=sources, object_specification_sha256='o'*64,
        development_gate_manifest_sha256='d'*64, development_gate_audit_sha256='a'*64,
        final_audit_plan_sha256='l'*64, training_ids=list(range(1,121)), evaluation_ids=list(range(121,181)),
        output_directory='results/final_locked_evaluation_v1')
    dev = dict(status='passed', final_scenarios_loaded=False, preparation_protocol_sha256='p'*64,
        object_specification_sha256='o'*64, sources=sources, sources_at_end=sources,
        input_sha256=inputs, input_sha256_at_end=inputs)
    audit = dict(status='passed', parent_manifest_sha256='d'*64)
    m = dict(status='complete', release_sha256='r'*64, preparation_protocol_sha256='p'*64,
        object_specification_sha256='o'*64, development_gate_manifest_sha256='d'*64,
        development_gate_audit_sha256='a'*64, training_ids_loaded=list(range(1,121)),
        evaluation_ids_loaded=list(range(121,181)), final_scenarios_loaded=True, final_value_access_started=True,
        final_values_may_have_been_parsed=True, completed_final_array_load=True, new_fits=0,
        selection_changed=False, admission_changed=False, current_C_service='execution_review',
        current_C_service_financials=None, Q2_path_rows=240, Q1_path_rows=300, summary_rows=9,
        sources=sources, sources_at_end=sources, input_sha256=inputs, input_sha256_at_end=inputs,
        outputs_sha256={name:'f'*64 for name in a.OUTPUTS})
    kw = dict(preparation_sha='p'*64, release_sha='r'*64, dev_sha='d'*64, dev_audit_sha='a'*64, plan_sha='l'*64)
    return p, release, m, dev, audit, kw


class FinalAuditPureTests(unittest.TestCase):
    def test_complete_authorization_metadata(self):
        p,r,m,d,u,kw=metadata(); a.authorization_metadata(p,r,m,d,u,**kw)
    def test_completed_dev_does_not_release_final(self):
        p,r,m,d,u,kw=metadata(); r['final_execution_ready']=False
        with self.assertRaisesRegex(ValueError,'release'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_plan_hash_is_bound(self):
        p,r,m,d,u,kw=metadata(); r['final_audit_plan_sha256']='x'*64
        with self.assertRaisesRegex(ValueError,'binding'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_incomplete_final_not_readable(self):
        p,r,m,d,u,kw=metadata();m['status']='incomplete'
        with self.assertRaisesRegex(ValueError,'Complete'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_changed_ending_source_rejected(self):
        p,r,m,d,u,kw=metadata();m['sources_at_end']={'changed.py':'z'*64}
        with self.assertRaisesRegex(ValueError,'snapshots'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_no_shadow_fill_current_C(self):
        p,r,m,d,u,kw=metadata();m['current_C_service_financials']={'terminal_real':100}
        with self.assertRaisesRegex(ValueError,'Complete'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_missing_NULL_metadata_key_rejected(self):
        p,r,m,d,u,kw=metadata();del m['current_C_service_financials']
        with self.assertRaisesRegex(ValueError,'Complete'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_six_outputs_no_silent_missing_tree(self):
        p,r,m,d,u,kw=metadata();del m['outputs_sha256']['evaluation_traces.npz']
        with self.assertRaisesRegex(ValueError,'six'):a.authorization_metadata(p,r,m,d,u,**kw)
    def test_template_metadata_not_scenario_indexed(self):
        template={'targets':np.array([.45,.05,.5]),'fees':np.zeros((120,30))}
        observed={'targets':np.array([.45,.05,.5]),'fees':np.zeros((60,30))}
        a.validate_layout(observed,template,list(template))
        observed['targets']=np.tile([.45,.05,.5],(60,1))
        with self.assertRaisesRegex(ValueError,'shape'):a.validate_layout(observed,template,list(template))
    def test_missing_field_not_accepted_by_layout(self):
        with self.assertRaisesRegex(ValueError,'inventory'):
            a.validate_layout({'fees':np.zeros((60,30))},{'fees':np.zeros((120,30))},['fees','qg_hold'])
    def test_dtype_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError,'dtype'):
            a.validate_layout({'fees':np.zeros((60,30),np.float32)},{'fees':np.zeros((120,30))},['fees'])
    def test_finite_gate_keeps_original_H(self):
        score,action=a.gate_algebra(1.,.995,9,8)
        self.assertAlmostEqual(score,.0025);self.assertEqual(action,'restore')
        self.assertEqual(a.gate_algebra(1.,.995,9,8,h=1)[1],'hold')
    def test_gate_tie_holds_and_counts_are_integral(self):
        self.assertEqual(a.gate_algebra(1.,1.,8,8)[1],'hold')
        with self.assertRaises(ValueError):a.gate_algebra(1.,1.,8.5,8)
        with self.assertRaises(ValueError):a.gate_algebra(np.nan,1.,8,8)
    def test_inactive_shadow_fields_are_NULL_sentinel_and_causal_action(self):
        arm={k:np.full((1,2),np.nan) for k in a.Q_FLOAT}
        arm.update({k:np.full((1,2),-1,dtype=np.int64) for k in a.Q_INT})
        arm.update(actions=np.array([['restore','hold']]),
            gate_reason=np.array([['mandatory_initial_restore','already_breached_pending_hold']]),
            restoration_predicted_growth_sacrifice=np.zeros((1,2),bool))
        self.assertEqual(a.check_shadow_decision(arm,0,0,False,a.Checks(),'initial'),'restore')
        self.assertEqual(a.check_shadow_decision(arm,0,1,True,a.Checks(),'pending'),'hold')
        arm['qg_hold'][0,1]=0.
        with self.assertRaisesRegex(AssertionError,'NULL'):a.check_shadow_decision(arm,0,1,True,a.Checks(),'pending')
    def test_active_shadow_risk_cost_algebra_and_sacrifice(self):
        arm={k:np.full((1,2),np.nan) for k in a.Q_FLOAT}
        arm.update({k:np.full((1,2),-1,dtype=np.int64) for k in a.Q_INT})
        for key,value in dict(qg_hold=1.,qg_restore=.995,predicted_score=.0025,
            predicted_growth_advantage_restore=-.005,qbad_hold_count=9,qbad_restore_count=8,
            forecast_n=120,predicted_breach_count_delta_restore=-1).items():arm[key][0,1]=value
        arm.update(actions=np.array([['restore','restore']]),gate_reason=np.array([['','finite_score_positive']]),
            restoration_predicted_growth_sacrifice=np.array([[False,True]]))
        self.assertEqual(a.check_shadow_decision(arm,0,1,False,a.Checks(),'active'),'restore')
        arm['forecast_n'][0,1]=60
        with self.assertRaisesRegex(AssertionError,'training'):a.check_shadow_decision(arm,0,1,False,a.Checks(),'active')
    def test_initial_transfer_self_finance_closed_form(self):
        post,fee,moved=independent_trade(np.array([0.,0.,100.]),[1.,0.,0.],.0025)
        self.assertAlmostEqual(fee,100*.0025/1.0025,places=13)
        self.assertAlmostEqual(fee,.0025*moved,places=13)
        self.assertAlmostEqual(sum(post)+fee,100.,places=13)
    def test_original_peak_inflation_and_pending_reference(self):
        market=np.array([[-.5,0.,0.],[1.,0.,0.],[1.,0.,0.]])
        annual=a.replay_reference(market,np.zeros(3),[.5,0.,.5],.2,arm='annual',cost_rate=0)
        pending=a.replay_reference(market,np.zeros(3),[.5,0.,.5],.2,arm='pending_hold',cost_rate=0)
        self.assertEqual(list(annual['actions']),['restore']*3)
        self.assertEqual(list(pending['actions']),['restore','hold','hold'])
        self.assertEqual(list(pending['real_wealth']),[100.,75.,100.,150.])
        self.assertEqual(pending['first_crossing'],1)
        inflation=a.replay_reference(np.zeros((2,3)),np.array([.1,.1]),[0.,0.,1.],.99,arm='annual',cost_rate=0)
        self.assertAlmostEqual(inflation['real_wealth'][-1],100/1.21)
        self.assertEqual(inflation['real_peak'][-1],100)
    def test_NULL_mask_corruption_rejected(self):
        with self.assertRaisesRegex(AssertionError,'NULLmask'):
            a.compare_array(np.array([0.]),np.array([np.nan]),a.Checks(),'synthetic')
    def test_independent_physical_reference_complete_35_fields(self):
        ref=a.replay_reference(np.zeros((2,3)),np.zeros(2),[.45,.05,.5],.4,arm='annual')
        self.assertEqual(len(set(ref)-a.COMMON),35)
        self.assertEqual(ref['holdings'].shape,(3,3))
        self.assertEqual(ref['fees'].shape,(2,))
    def test_pending_witness_slots_retained_without_replacement(self):
        arm={'actions':np.full((3,30),'hold')};refs={sid:{'ever_breached':np.ones(31,bool)} for sid in (121,150,180)}
        with patch.object(a,'scalar_fan',side_effect=AssertionError('must not call pending Q')):
            rows=a.fixed_witnesses(arm,refs,np.array([121,150,180]),None,None,a.Checks())
        self.assertEqual(len(rows),9)
        self.assertEqual({x['original_year'] for x in rows},{2,16,30})
        self.assertTrue(all(x['status']=='skipped_original_slot_pending' and x['replacement'] is None for x in rows))
    def test_excess_equality_band_and_no_terminal_fee(self):
        ref=a.replay_reference(np.array([[0.,0.,-.4-5e-13]]),np.zeros(1),[0.,0.,1.],.4,arm='annual',cost_rate=0)
        metrics=a.metrics_from_reference(ref,.4)
        self.assertFalse(metrics['breach']);self.assertEqual(metrics['excess'],0.)
        self.assertEqual(len(ref['fees']),1);self.assertEqual(metrics['volatility'],0.)
    def test_summary_keeps_Review_financial_NULLs(self):
        obj=dict(object_id='guarded_C_service',question='Q2',profile='C',horizon=30,loss_tolerance=.4,
            financial_scope='none_preinvestment_review',service_admitted=False,execution_status='execution_review',
            case_status='balanced_case',q1_assignment='Balanced',liquidity='low',financial_available=False)
        rows=[dict(a.metadata_row(obj,sid),**dict.fromkeys(a.FIN)) for sid in range(121,181)]
        summary=a.summary_expected(rows,obj)
        self.assertEqual(summary['execution_review_n'],60);self.assertEqual(summary['financial_n'],0)
        self.assertTrue(all(summary[x] is None for x in a.SUMMARY_FIN))
    def test_scalar_fan_inherits_original_peak_and_suffix(self):
        train=np.array([[[100.,100.,100.],[-.5,0.,0.]],[[100.,100.,100.],[0.,0.,0.]]])
        # The huge first return is past, so changing it cannot alter year2Q.
        q=a.scalar_fan([50.,0.,50.],1.,150.,.2,[.5,0.,.5],.4,train,np.zeros((2,2)),1,cost_rate=0)
        changed=train.copy();changed[:,0,:]=-0.99
        other=a.scalar_fan([50.,0.,50.],1.,150.,.2,[.5,0.,.5],.4,changed,np.zeros((2,2)),1,cost_rate=0)
        self.assertEqual(q['hold']['count'],1)
        for action in ('hold','restore'):np.testing.assert_array_equal(q[action]['logs'],other[action]['logs'])


if __name__=='__main__':unittest.main()
