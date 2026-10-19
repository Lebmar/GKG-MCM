"""Synthetic saved-state pipeline and before-array integrity gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('guard_physical_feasibility_runner',ROOT/'scripts/diagnose_guard_physical_feasibility.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def put(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,bytes):p.write_bytes(value)
    else:m.writej(p,value)
    return m.sha(p)


def groups_and_arrays(*,one_guard=False):
    groups=[];arrays={}
    for context in range(6):
        ids=list(range(1,121)) if context==0 else list(range(24*(context-1)+1,24*context+1))
        training=list(range(1,121)) if context==0 else sorted(set(range(1,121))-set(ids))
        funded=one_guard and context==1;gid=f'c{context}__selected';n=len(ids)
        a={}
        if funded:
            a=dict(first_guard_trigger=np.full(n,-1,dtype=np.int64),guard_triggered=np.zeros((n,30),bool),
                holdings=np.broadcast_to(np.array([45.,5.,50.]),(n,31,3)).copy(),cpi=np.ones((n,31)),
                real_peak=np.full((n,31),100.),running_mdd=np.zeros((n,31)),
                training_real_gross=np.ones((30,len(training),3)),budget_floor=np.full((n,30),60.),
                was_pending_input=np.zeros((n,30),bool),selected_index=np.full((n,30),-1,dtype=np.int64),
                support_feasible=np.zeros((n,30),bool),candidate_feasible=np.zeros((n,30,13),bool),
                interval_feasible=np.zeros((n,30,4),bool))
            a['first_guard_trigger'][0]=2;a['guard_triggered'][0,1]=True
            a['training_real_gross'][1]=.5
            arrays.update({gid+'__'+key:value for key,value in a.items()})
        groups.append(dict(group_id=gid,kind='full_selected' if context==0 else 'outer_selected',
            job_id=f'job{context}',scenario_ids=ids,training_ids=training,target=[.45,.05,.5],
            candidate=dict(release_fraction=1.,forecast_rule='uniform'),array_names=list(a),
            status='funded' if funded else 'no_inner_feasible_candidate'))
    return groups,arrays


def freeze_fixture(root,*,one_guard=False):
    sources={name:put(root,name,b'# synthetic frozen source\n') for name in m.REQUIRED_SOURCES}
    closure={}
    for parent in m.PARENTS:
        folder=Path(parent['directory']);groups,arrays=groups_and_arrays(one_guard=one_guard and parent['version']=='V2b')
        before={parent['protocol']:put(root,parent['protocol'],{'frozen':'synthetic'}),
            parent['source_freeze']:put(root,parent['source_freeze'],{'frozen':'synthetic'})}
        put(root,folder/'trace_groups.json',groups)
        np.savez_compressed(root/folder/'traces.npz',**arrays)
        for name in m.PRODUCER_OUTPUTS-{'trace_groups.json','traces.npz'}:put(root,folder/name,b'synthetic bound artifact; not parsed\n')
        outputs={name:m.sha(root/folder/name) for name in m.PRODUCER_OUTPUTS}
        parent_sha=put(root,folder/'manifest.json',dict(status='complete',final_scenarios_loaded=False,new_final_reads=0,
            sources_and_inputs_at_start=before,sources_and_inputs_at_end=before,outputs_sha256=outputs))
        bound={**before,str(folder/'manifest.json'):parent_sha,**{str(folder/name):digest for name,digest in outputs.items()}}
        put(root,folder/'independent_audit.json',dict(status='passed',final_scenarios_loaded=False,parent_manifest_sha256=parent_sha,
            sources_and_artifacts_at_start=bound,sources_and_artifacts_at_end=bound))
        closure.update(m.parent_closure(root,parent))
    protocol=dict(version='guard-full-simplex-physical-feasibility-v1',execution_ready=True,development_ids=list(range(1,121)),
        allow_final=False,parents=list(m.PARENTS),scope_kinds=['outer_selected','full_selected'],
        solver_method=m.SOLVER_METHOD,solver_options=m.SOLVER_OPTIONS,certificate_tolerance=m.CERTIFICATE_TOL,
        comparison_tolerance=m.COMPARISON_TOL,expected_scipy_version=m.scipy.__version__,cost_rate=.0025,
        loss_tolerance=.4,original_horizon=30)
    plan_sha=put(root,m.PLAN,protocol)
    put(root,m.FREEZE,dict(execution_ready=True,protocol_sha256=plan_sha,source_sha256=sources,input_sha256=closure))


class GuardPhysicalFeasibilityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'output'

    def test_allNULL_selected_groups_preserve_inventory_and_empty_csv_header(self):
        freeze_fixture(self.root)
        with mock.patch.object(m,'solve_full_simplex') as solver:m.run(self.out,self.root)
        solver.assert_not_called()
        receipt=m.readj(self.out/'manifest.json');self.assertEqual(receipt['status'],'complete')
        self.assertEqual(receipt['selected_group_count'],12);self.assertEqual(receipt['first_guard_count'],0)
        frame=pd.read_csv(self.out/'guard_cases.csv.gz')
        self.assertEqual(list(frame.columns),list(m.CSV_FIELDS));self.assertEqual(len(frame),0)
        summaries=m.readj(self.out/'summary.json');self.assertEqual(len(summaries),4)
        self.assertTrue(all(s['physical_path_n']==0 and s['first_guard_n']==0 and s['LP_count']==0 for s in summaries))

    def test_one_firstguard_is_diagnosed_once_with_all_eight_lp_witnesses(self):
        freeze_fixture(self.root,one_guard=True);m.run(self.out,self.root)
        rows=pd.read_csv(self.out/'guard_cases.csv.gz')
        self.assertEqual(len(rows),1);self.assertEqual(rows.iloc[0].version,'V2b')
        self.assertEqual(rows.iloc[0].scenario_id,1);self.assertEqual(rows.iloc[0].period,2)
        self.assertEqual(rows.iloc[0].classification,'all_eight_numerical_optima_below_budget')
        self.assertEqual(rows.iloc[0].passed_certificate_count,8)
        import gzip
        with gzip.open(self.out/'case_inputs_and_lp_witnesses.json.gz','rt') as stream:cases=json.load(stream)
        self.assertEqual(len(cases),1);self.assertEqual(len(cases[0]['result']['patterns']),8)
        self.assertFalse(cases[0]['actual_future_used_in_LP'])
        np.testing.assert_array_equal(cases[0]['support_real_gross'],np.full((96,3),.5))
        receipt=m.readj(self.out/'manifest.json')
        self.assertFalse(receipt['raw_market_values_parsed']);self.assertFalse(receipt['account_action_changed'])
        self.assertFalse(receipt['exact_arithmetic_impossibility_proof']);self.assertEqual(receipt['LP_count'],8)

    def test_changed_parent_artifact_or_missing_freeze_blocks_before_np_load(self):
        for mode in ('artifact','freeze'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);freeze_fixture(root)
                if mode=='artifact':(root/m.PARENTS[0]['directory']/'traces.npz').write_bytes(b'changed')
                else:(root/m.FREEZE).unlink()
                with mock.patch.object(m.np,'load') as parser:
                    with self.assertRaises((RuntimeError,FileNotFoundError)):m.run(root/'out',root)
                parser.assert_not_called();self.assertFalse((root/'out').exists())

    def test_failed_diagnostic_preserves_partial_evidence_and_refuses_retry(self):
        freeze_fixture(self.root,one_guard=True)
        with mock.patch.object(m,'solve_full_simplex',side_effect=ArithmeticError('synthetic diagnostic failure')):
            with self.assertRaises(ArithmeticError):m.run(self.out,self.root)
        self.assertEqual(m.readj(self.out/'manifest.json')['status'],'failed')
        self.assertTrue((self.out/'cases_partial.json.gz').exists());self.assertTrue((self.out/'failure.txt').exists())
        with mock.patch.object(m,'freeze_gate') as gate,mock.patch.object(m.np,'load') as parser:
            with self.assertRaises(RuntimeError):m.run(self.out,self.root)
        gate.assert_not_called();parser.assert_not_called()


if __name__=='__main__':unittest.main()
