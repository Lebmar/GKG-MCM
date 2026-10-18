"""Independent constrained-action, accounting, fake-I/O and pipeline fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research'),str(ROOT/'src'),str(ROOT/'tests')]
import audit_constrained_fan_execution as audit


class ConstrainedAuditTests(unittest.TestCase):
    def test_both_feasible_maximizes_netQ_and_retains_hold_at_global_tie(self):
        self.assertEqual(audit.independent_choice(.1,.2,0,1,20)['action'],'restore')
        for qr in (.1,.1+0.5e-12,.09):self.assertEqual(audit.independent_choice(.1,qr,0,1,20)['action'],'hold')
        self.assertEqual(audit.independent_choice(.1,.1+2e-12,0,1,20)['action'],'restore')

    def test_only_restore_feasible_can_sacrifice_net_growth_and_strictly_reduce_count(self):
        q=audit.independent_choice(.3,.2,3,2,20)
        self.assertEqual(q['action'],'restore');self.assertEqual(q['gate_reason'],'risk_constraint_repair')
        self.assertTrue(q['risk_cost_justification']);self.assertFalse(q['economic_improvement'])
        self.assertTrue(q['restoration_predicted_growth_sacrifice']);self.assertTrue(q['repair_meets_cap'])
        positive=audit.independent_choice(.1,.2,3,2,20)
        self.assertTrue(positive['economic_improvement']);self.assertFalse(positive['risk_cost_justification'])
        self.assertEqual(positive['gate_reason'],'risk_constraint_repair')

    def test_only_hold_feasible_refuses_even_higher_restore_growth(self):
        q=audit.independent_choice(0.,10.,2,3,20)
        self.assertEqual(q['action'],'hold');self.assertEqual(q['decision_detail'],'only_hold_model_feasible')

    def test_empty_feasible_set_orders_integer_violations_before_growth(self):
        q=audit.independent_choice(100.,-1.,5,4,20)
        self.assertTrue(q['local_model_infeasible']);self.assertFalse(q['repair_meets_cap'])
        self.assertEqual(q['gate_reason'],'least_violation_restore');self.assertTrue(q['risk_cost_justification'])
        hold=audit.independent_choice(0.,1.,3,4,20)
        self.assertEqual(hold['decision_detail'],'least_violation_hold')
        self.assertEqual(audit.independent_choice(0.,0.,3,3,20)['action'],'hold')
        self.assertEqual(audit.independent_choice(0.,1.,3,3,20)['action'],'restore')

    def test_exact_integer_cap_boundaries_and_invalid_inputs(self):
        self.assertEqual(audit.independent_choice(0.,1.,7,8,72)['local_breach_cap'],7)
        self.assertEqual(audit.independent_choice(0.,1.,7,8,72)['action'],'hold')
        for n,kh,kr in ((True,0,0),(0,0,0),(20,1.,0),(20,False,0),(20,-1,0),(20,0,21)):
            with self.subTest(n=n,kh=kh,kr=kr),self.assertRaises(ValueError):audit.independent_choice(0.,1.,kh,kr,n)
        with self.assertRaises(ValueError):audit.independent_choice(float('nan'),1.,0,0,20)

    def test_quoted_fee_is_a_physical_quote_not_a_second_Q_subtraction(self):
        # A positive direct net Q difference wins even if smaller than the
        # separately reported current fee: the Q already includes that cost.
        q=audit.independent_choice(.02,.020001,0,0,20)
        self.assertEqual(q['action'],'restore');self.assertTrue(q['economic_improvement'])

    @staticmethod
    def model_fixture():
        from experimental_execution.constrained_fan_rollout import simulate_constrained
        train=np.zeros((2,30,3));train[:,:,0]=.02;train[:,0,0]=-.2
        actual=np.zeros((1,30,3));actual[:,:,0]=.01;actual[:,2,:]=-.7
        pi=np.zeros((1,30));target=np.array([.4,.3,.3])
        arm=simulate_constrained(actual,pi,train,np.zeros((2,30)),target,L=.4,horizon=30,cost_rate=.0025)
        e={k:arm[k][0] for k in (*audit.QFLOATS,*audit.QCOUNTS,*audit.QBOOLS,*audit.QSTRINGS)}
        return arm,e,actual[0],pi[0],train,target

    def test_complete_saved_physics_and_initial_diagnostic_pending_semantics(self):
        arm,e,r,pi,train,target=self.model_fixture()
        expected,metric,extra=audit.replay_model(r,pi,target,len(train),e)
        saved={k:arm[k] if k=='targets' else arm[k][0] for k in expected if k in arm}
        audit.accounting.compare_saved_account(saved,expected)
        self.assertEqual(expected['first_crossing'],3);self.assertTrue(expected['pending_at_decision'][3:].all())
        self.assertTrue(e['estimate_available'][0]);self.assertFalse(e['local_constraint_applied'][0])
        self.assertLess(e['qg_restore'][0],e['qg_hold'][0])
        self.assertEqual(expected['actions'][0],'restore')
        self.assertEqual(len('mandatory_initial_restore_diagnostic_only'),41)
        self.assertEqual(e['decision_detail'][0],'mandatory_initial_restore_diagnostic_onl')
        for k in ('economic_improvement','risk_cost_justification','repair_meets_cap','restoration_predicted_growth_sacrifice'):self.assertFalse(e[k][0])
        for k,v in extra.items():self.assertEqual(arm[k][0],v)
        self.assertGreater(metric['mdd'],.4)

    def test_corrupt_action_classification_quote_pending_and_integer_counts_are_rejected(self):
        _,e,r,pi,train,target=self.model_fixture()
        for key,index,value in (('quoted_nominal_fee',0,1.),('risk_cost_justification',0,True),
             ('qg_hold',3,0.),('forecast_n',1,3),('gate_reason',1,'growth_improvement'),('qbad_hold_count',1,3)):
            with self.subTest(key=key):
                corrupt=deepcopy(e);corrupt[key][index]=value
                with self.assertRaises(ValueError):audit.replay_model(r,pi,target,len(train),corrupt)

    def test_overwrite_guards_precede_every_read(self):
        for name in ('independent_audit.json','audit_failure_receipt.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                out=Path(directory);(out/name).write_text('{}')
                with patch.object(audit,'OUT',out),patch.object(audit,'readj',side_effect=AssertionError('No read allowed')):
                    with self.assertRaises(RuntimeError):audit.main()

    @staticmethod
    def pipeline_fixture(label):
        from test_constrained_fan_pipeline import ConstrainedFanPipelineTests
        import run_constrained_fan_execution as runner
        p,b,r,pi,ids=ConstrainedFanPipelineTests.fixture(label)
        frozen=json.loads(audit.PROTOCOL.read_text())
        for k in ('data','target_updates_enabled','original_H','loss_tolerance','cost_rate'):p[k]=frozen[k]
        r[:]=0.
        allowed=list(range(5,22))+[23]+list(range(24,37))+[57]+list(range(58,198))
        weights=np.tile([1.,0.,0.],(232,1));weights[allowed]=[.3,.3,.4]
        weights[13]=[.2,.3,.5];weights[23]=[.4,.3,.3];weights[57]=[.7,.2,.1]
        cache={'weights':weights,'H30__mdd':np.zeros((120,232))}
        parent=json.loads((ROOT/'docs/refit_stability_protocol.json').read_text());full=json.loads((ROOT/'docs/joint_liquidity_refit_protocol.json').read_text())
        engine=runner.simulate;arms={}
        def identical(ar,ap,tr,target,policy):
            key=(len(tr[0]),tuple(target),policy)
            if key not in arms:arms[key]=engine(ar[:1],ap[:1],tr,target,policy)
            return {k:(v if k in ('targets','loss_tolerances') else np.repeat(v,len(ar),axis=0)) if isinstance(v,np.ndarray) else v for k,v in arms[key].items()}
        with patch.object(runner,'simulate',side_effect=identical):out=runner.run_pipeline(p,b,r,pi,ids)
        tables={k:audit.accounting.observed_bools(pd.DataFrame(v)) for k,v in out['tables'].items()}
        return p,b,r,pi,ids,tables,out,cache,parent,full

    def test_full_synthetic_Q1_review_preserves_all_unavailable_slots_and_NULLs(self):
        p,b,r,pi,ids,t,out,cache,parent,full=self.pipeline_fixture('Review')
        result=audit.audit_artifacts(p,b,r,pi,ids,t,out['choices'],out['groups'],out['arrays'],cache,parent=parent,full=full)
        self.assertEqual(result['counts']['training_rows'],10080);self.assertEqual(result['counts']['training_summaries'],130)
        self.assertEqual(result['counts']['selected_trace_accounts'],720)
        self.assertEqual(result['counts']['unavailable_forecast_slots'],272)
        self.assertEqual(result['counts']['independent_forecast_states'],0)
        corrupt={**t,'client_paths':t['client_paths'].copy()};i=corrupt['client_paths'].index[corrupt['client_paths'].policy.eq(audit.PRIMARY)][0]
        corrupt['client_paths'].loc[i,'log_growth']=0.
        with self.assertRaises(ValueError):audit.audit_artifacts(p,b,r,pi,ids,corrupt,out['choices'],out['groups'],out['arrays'],cache,parent=parent,full=full)

    def test_full_synthetic_admitted_pipeline_all_accounts_and_fixed_Q_slots(self):
        p,b,r,pi,ids,t,out,cache,parent,full=self.pipeline_fixture('Balanced')
        original=audit.scalar_fan;memo={}
        # Every fabricated market path is identically zero. Cache only in this
        # synthetic fixture by exact state and N; production audit is unpatched.
        def memoized(x,cp,peak,worst,target,loss,tr,tpi,year,**kw):
            key=(tuple(x),cp,peak,worst,tuple(target),loss,tr.shape,tpi.shape,year,tuple(sorted(kw.items())))
            if key not in memo:memo[key]=original(x,cp,peak,worst,target,loss,tr,tpi,year,**kw)
            return memo[key]
        with patch.object(audit,'scalar_fan',side_effect=memoized):
            result=audit.audit_artifacts(p,b,r,pi,ids,t,out['choices'],out['groups'],out['arrays'],cache,parent=parent,full=full)
        self.assertEqual(result['status'],'passed');self.assertEqual(result['counts']['selected_trace_accounts'],7200)
        self.assertEqual(result['counts']['model_trace_accounts'],3240)
        self.assertEqual(result['counts']['independent_forecast_states'],272)
        self.assertEqual(result['counts']['full120_paired_paths'],120)
        self.assertEqual(result['counts']['unavailable_forecast_slots'],0)


if __name__=='__main__':unittest.main()
