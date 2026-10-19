import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mitigation_runner',ROOT/'scripts/run_state_budget_mitigation.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def arm():
    return dict(real_wealth=np.full((2,31),100.),fees=np.zeros((2,30)),trades=np.zeros((2,30),bool),
        ever_breached=np.zeros((2,31),bool),first_crossing=np.full(2,-1),
        pending_at_decision=np.zeros((2,30),bool),guard_triggered=np.zeros((2,30),bool),
        guard_pending=np.zeros((2,31),bool),gate_reason=np.full((2,30),'support_safe_hold'))


class RunnerTests(unittest.TestCase):
    def test_normal_support_pass_is_not_a_support_review(self):
        rows=runner.rows_from_arm(arm(),[1,2],{})
        self.assertTrue(all(not r['support_review'] for r in rows))

    def test_funded_review_keeps_finance_excludes_trigger_year_from_auto(self):
        a=arm();a['guard_pending'][0,2:]=True;a['guard_triggered'][0,1]=True
        a['pending_at_decision'][0,2:]=True
        rows=runner.rows_from_arm(a,[1,2],{})
        self.assertTrue(rows[0]['assigned']);self.assertTrue(rows[0]['support_review'])
        self.assertAlmostEqual(rows[0]['automatic_fraction'],1/30)
        self.assertEqual(rows[0]['cagr'],0.)

    def test_null_finance_is_absent_not_cash(self):
        rows=runner.null_rows([1,2],{},'initial_review')
        self.assertIsNone(rows[0]['mdd']);self.assertFalse(rows[0]['assigned'])

    def test_original_partition_metadata_validates_without_markets(self):
        jobs=json.loads((ROOT/'docs/joint_liquidity_refit_protocol.json').read_text())['jobs']
        runner.validate_jobs(jobs)
        for bad in [121,True,1.5]:
            mutated=copy.deepcopy(jobs);mutated[0]['evaluation_ids'][0]=bad
            with self.assertRaises(RuntimeError):runner.validate_jobs(mutated)

    def test_initial_Q1_target_uses_own_training_fit(self):
        j=dict(stage='inner',training_ids=[1,2,3],evaluation_ids=[4])
        b=dict(refit=dict(training_ids=[1,2,5]),q2_case=dict(in_balanced_case=True,product=dict(weights=[.45,.05,.5])))
        with self.assertRaises(ValueError):runner.validate_job(j,b)
        b['refit']['training_ids']=[1,2,3]
        np.testing.assert_array_equal(runner.validate_job(j,b),[.45,.05,.5])

    def test_inner_archive_keeps_physical_and_selected_action_evidence(self):
        a=arm();a['candidate_posttrade']=np.zeros((2,30,13,3));a['selected_lambda']=np.zeros((2,30));a['mitigation_candidate_posttrade']=np.zeros((2,30,9,3))
        kept,omitted=runner.archive_arrays(a,'inner_candidate')
        self.assertIn('real_wealth',kept);self.assertIn('selected_lambda',kept)
        self.assertEqual(omitted,['candidate_posttrade','mitigation_candidate_posttrade'])
        full,dropped=runner.archive_arrays(a,'outer_selected')
        self.assertIn('candidate_posttrade',full);self.assertEqual(dropped,[])

    def test_complete_pipeline_exports_every_job_and_original_ID_with_synthetic_markets(self):
        p=json.loads((ROOT/'docs/state_budget_mitigation_protocol.json').read_text())
        jobs=json.loads((ROOT/p['job_protocol']).read_text())['jobs']
        def fake_sim(r,pi,train,trainpi,target,**kw):
            n=len(r);a=arm()
            for k,v in list(a.items()):a[k]=np.repeat(v[:1],n,axis=0)
            return dict(status='funded',financial_available=True,arm=a,initial_witness={'synthetic':True})
        with tempfile.TemporaryDirectory() as td:
            with patch.object(runner,'freeze_gate',return_value=(p,dict(source_sha256={},input_sha256={}),jobs)), \
                 patch.object(runner,'FREEZE',ROOT/'docs/state_budget_mitigation_protocol.json'), \
                 patch.object(runner,'load_paths',return_value=(np.zeros((120,30,3)),np.zeros((120,30)),np.arange(1,121))), \
                 patch('experimental_execution.state_budget_mitigation.simulate_state_budget',side_effect=fake_sim), \
                 patch('builtins.print'):
                runner.run(Path(td)/'synthetic')
            d=json.loads((Path(td)/'synthetic/manifest.json').read_text())
            self.assertEqual(d['status'],'complete');self.assertEqual(d['trace_groups'],240)
            self.assertEqual(d['path_rows'],6360);self.assertFalse(d['final_scenarios_loaded'])
            x=json.loads((Path(td)/'synthetic/selections.json').read_text())
            self.assertEqual(len(x),6);self.assertTrue(all(v['final_status']=='funded' for v in x.values()))

    def test_mitigation_trigger_reports_actual_trade_fee_and_ordinary_service(self):
        a=arm();a['gate_reason']=a['gate_reason'].astype('U50')
        a['gate_reason'][0,1]='risk_mitigation_then_review'
        a['fees'][0,1]=.12;a['trades'][0,1]=True
        a['guard_pending'][0,2:]=True;a['guard_triggered'][0,1]=True
        a['pending_at_decision'][0,2:]=True
        rows=runner.rows_from_arm(a,[1,2],{})
        self.assertEqual(rows[0]['mitigation_nominal_fees'],.12)
        self.assertAlmostEqual(rows[0]['mitigation_trade_fraction'],1/30)
        self.assertAlmostEqual(rows[0]['automatic_fraction'],1/30)
        total=runner.summarize_records(rows)
        self.assertAlmostEqual(total['mean_mitigation_fees'],.06)
        self.assertIsNone(runner.summarize_records(runner.null_rows([1],{},'initial_review'))['mean_mitigation_fees'])

    def test_failed_pipeline_preserves_failed_attempt_without_overwrite(self):
        p=json.loads((ROOT/'docs/state_budget_mitigation_protocol.json').read_text())
        jobs=json.loads((ROOT/p['job_protocol']).read_text())['jobs']
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'failed'
            with patch.object(runner,'freeze_gate',return_value=(p,dict(source_sha256={},input_sha256={}),jobs)), \
                 patch.object(runner,'FREEZE',ROOT/'docs/state_budget_mitigation_protocol.json'), \
                 patch.object(runner,'load_paths',side_effect=ValueError('synthetic parser failure')):
                with self.assertRaises(ValueError):runner.run(out)
                self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'failed')
                self.assertTrue((out/'failure.txt').exists())
                with self.assertRaises(RuntimeError):runner.run(out)


if __name__=='__main__':unittest.main()
