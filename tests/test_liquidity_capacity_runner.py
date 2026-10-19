"""Synthetic liquidity driver checks; no supplied markets or numeric archives."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('liquidity_capacity_runner',ROOT/'scripts/diagnose_liquidity_capacity.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def writej(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def fixture(root,*,funded_C=False):
    jobs=[];fits={};prior=[]
    for fold in range(1,6):
        evaluation=list(range(24*(fold-1)+1,24*fold+1))
        training=sorted(set(range(1,121))-set(evaluation));jid=f'outer{fold}'
        jobs.append(dict(job_id=jid,stage='refit',parent_context_id=fold,training_ids=training,evaluation_ids=evaluation))
        assignment={profile['name']:('Balanced' if funded_C and profile['name']=='C' else 'Review') for profile in runner.PROFILES}
        fits[jid]=dict(refit=dict(training_ids=training,effective_assignments=assignment,
            raw_fit=dict(menu=[dict(name='Balanced',weights=[.45,.05,.5])]) if funded_C else None))
        for sid in evaluation:
            for name,value in assignment.items():
                prior.append(dict(job_id=jid,scenario_id=sid,profile=name,effective_assignment=value,assigned=value!='Review'))
    paths=dict(jobs='synthetic/jobs.json',static_fits='synthetic/fits.json',q1_oof_paths='synthetic/prior.csv.gz')
    writej(root/paths['jobs'],dict(jobs=jobs));writej(root/paths['static_fits'],fits)
    pd.DataFrame(prior).to_csv(root/paths['q1_oof_paths'],index=False,compression=dict(method='gzip',mtime=0))
    protocol=dict(allow_final=False,development_ids=list(range(1,121)),**paths)
    plan=root/'docs/liquidity_capacity_diagnostic_protocol.json';writej(plan,protocol)
    freeze=root/'docs/liquidity_capacity_diagnostic_source_freeze.json'
    frozen=dict(execution_ready=True,protocol_sha256=runner.sha(plan),source_sha256={},input_sha256={})
    writej(freeze,frozen)
    return plan,freeze,paths


class LiquidityCapacityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.out=self.root/'results/liquidity_capacity_diagnostic_v1'

    def execute(self,plan,freeze,*,loader=None):
        market=np.zeros((120,30,3));inflation=np.zeros((120,30))
        with patch.object(runner,'ROOT',self.root),patch.object(runner,'PLAN',plan), \
             patch.object(runner,'FREEZE',freeze),patch.object(runner,'OUT',self.out), \
             patch.object(runner,'load_paths',return_value=(market,inflation,np.arange(1,121))) as parsed:
            runner.run()
        self.assertEqual(list(parsed.call_args.args[1]),list(range(1,121)))
        self.assertEqual(parsed.call_args.kwargs,{'allow_final':False})

    def test_allReview_retains600_NULL_paths_and_empty_annual_csv_schema(self):
        plan,freeze,_=fixture(self.root);self.execute(plan,freeze)
        paths=pd.read_csv(self.out/'path_capacity.csv.gz');annual=pd.read_csv(self.out/'annual_capacity.csv.gz')
        receipt=json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(len(paths),600);self.assertFalse(paths.assigned.any())
        self.assertTrue(paths[list(runner.FIELDS)].isna().all().all())
        self.assertEqual(len(annual),0);self.assertEqual(list(annual.columns),list(runner.ANNUAL_COLUMNS))
        self.assertEqual(receipt['status'],'complete');self.assertEqual(receipt['annual_rows'],0)
        summaries=json.loads((self.out/'summary.json').read_text())
        self.assertTrue(all(s['financial_n']==0 and s['review_n']==120 and s['mean_end_bill_share'] is None for s in summaries))

    def test_C_only_funded_keeps3600_ownH_years_and_other_profile_NULLs(self):
        plan,freeze,_=fixture(self.root,funded_C=True);self.execute(plan,freeze)
        paths=pd.read_csv(self.out/'path_capacity.csv.gz');annual=pd.read_csv(self.out/'annual_capacity.csv.gz')
        self.assertEqual(len(paths),600);self.assertEqual(int(paths.assigned.sum()),120)
        self.assertEqual(len(annual),3600);self.assertEqual(set(annual.profile),{'C'})
        self.assertTrue((annual.horizon==30).all());self.assertEqual(set(annual.period),set(range(1,31)))
        self.assertTrue(paths.loc[paths.profile!='C',list(runner.FIELDS)].isna().all().all())
        np.testing.assert_array_equal(annual.groupby('scenario_id').size().to_numpy(),np.full(120,30))
        np.testing.assert_allclose(annual.end_bill_share,.5,atol=1e-15,rtol=0)
        np.testing.assert_allclose(annual.hypothetical_available_bill_ratio+annual.hypothetical_liquidation_drag,1,atol=1e-15,rtol=0)
        receipt=json.loads((self.out/'manifest.json').read_text())
        self.assertFalse(receipt['hypothetical_liquidation_charged_to_strategy']);self.assertEqual(receipt['new_policy_runs'],0)
        self.assertFalse(receipt['withdrawals_assumed']);self.assertFalse(receipt['final_scenarios_loaded'])

    def test_wrong_previous_assignment_leaves_failed_receipt_and_no_overwrite(self):
        plan,freeze,paths=fixture(self.root,funded_C=True)
        prior=pd.read_csv(self.root/paths['q1_oof_paths']);prior.loc[0,'effective_assignment']='wrong'
        prior.to_csv(self.root/paths['q1_oof_paths'],index=False,compression=dict(method='gzip',mtime=0))
        with self.assertRaisesRegex(RuntimeError,'unchanged Q1 effective assignment'):self.execute(plan,freeze)
        receipt=json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(receipt['status'],'failed');self.assertTrue((self.out/'failure.txt').exists())
        with patch.object(runner,'ROOT',self.root),patch.object(runner,'PLAN',plan), \
             patch.object(runner,'FREEZE',freeze),patch.object(runner,'OUT',self.out),patch.object(runner,'load_paths') as parsed:
            with self.assertRaisesRegex(RuntimeError,'Refuse overwrite'):runner.run()
        parsed.assert_not_called()

    def test_missing_freeze_blocks_before_market_parser_or_attempt_creation(self):
        plan,freeze,_=fixture(self.root);freeze.unlink()
        with patch.object(runner,'ROOT',self.root),patch.object(runner,'PLAN',plan), \
             patch.object(runner,'FREEZE',freeze),patch.object(runner,'OUT',self.out),patch.object(runner,'load_paths') as parsed:
            with self.assertRaises(FileNotFoundError):runner.run()
        parsed.assert_not_called();self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()
