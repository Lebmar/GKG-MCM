"""Constructed markets and temporary release metadata only; no real CSV."""
import ast
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import submission_final_adapter as m


def registry():return m.compact.readj(ROOT/m.compact.REGISTRY)


def put(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,bytes):path.write_bytes(value)
    else:m.compact.writej(path,value)
    return m.compact.sha(path)


def gate_fixture(root):
    spec=m.producer.object_specification(registry())
    mapping={name:put(root,Path(name),b'# synthetic hashed source\n') for name in m.REQUIRED_RUNTIME}
    mapping[str(m.compact.REGISTRY)]=put(root,m.compact.REGISTRY,registry())
    mapping['data/SafeInvest_Synthetic_Data.csv']=put(root,Path('data/SafeInvest_Synthetic_Data.csv'),b'never parsed fixture\n')
    mapping[str(m.compact.EXPECTED)]=put(root,m.compact.EXPECTED,dict(version='compact-canonical-development-expectations-v1',object_specification=m.compact.specification(registry())))
    mapping[str(m.PROTOCOL)]=put(root,m.PROTOCOL,dict(source_only_test=True))
    proof=dict(version='compact-fixed-nonblind-final-expectations-v1',object_specification=spec,
        previously_exposed_nonblind=True,array_signatures={},output_bytes_sha256={name:'a'*64 for name in m.OUTPUTS-{'traces.npz'}},
        rows=dict(Q2=240,Q1=300,Q1_reference=300,total=840,summary=14,comparisons=8,trace_groups=14),financial_objects=12)
    mapping[str(m.EXPECTED)]=put(root,m.EXPECTED,proof)
    release=dict(version='compact-fixed-nonblind-final-expectation-release-v1',reproduction_ready=True,previously_exposed_nonblind=True,
        new_independent_validation=False,new_fits=0,new_selections=0,training_ids=m.TRAIN,evaluation_ids=m.ACTUAL,
        object_specification_sha256=m.compact.canonical_sha(spec),registry_sha256=mapping[str(m.compact.REGISTRY)],
        protocol_sha256=mapping[str(m.PROTOCOL)],expected_sha256=mapping[str(m.EXPECTED)],
        source_sha256={name:mapping[name] for name in m.REQUIRED_RUNTIME})
    mapping[str(m.EXPECTED_RELEASE)]=put(root,m.EXPECTED_RELEASE,release)
    meta=dict(version='compact-fixed-nonblind-final-addon-v1',status='complete',explicit_reproduction_ready=True,
        previously_exposed_nonblind=True,new_independent_validation=False,object_specification=spec,files_sha256=mapping,
        expected_release_sha256=mapping[str(m.EXPECTED_RELEASE)])
    final_sha=put(root,m.FINAL_BUNDLE,meta)
    common=dict(version='compact-technical-source-bundle-v1',status='complete',development_ready=True,final_access=False,
        object_specification=m.compact.specification(registry()),files_sha256={**mapping,str(m.FINAL_BUNDLE):final_sha})
    put(root,m.compact.BUNDLE,common)
    return spec,proof


class SubmissionFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.synthetic=Path(cls.tmp.name)/'constructed'
        with mock.patch.object(m.compact,'load_paths',side_effect=AssertionError('No supplied market reads')) as loader,\
                mock.patch.object(m.producer,'run',side_effect=AssertionError('Do not call historical entrypoint')) as run:
            m.run(output=cls.synthetic)
        loader.assert_not_called();run.assert_not_called()
        cls.proof=m.compact.readj(cls.synthetic/'reproduction_checks.json')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_default_is_real_kernel_full14_object_constructed_market(self):
        state=m.compact.readj(self.synthetic/'manifest.json')
        self.assertEqual(state['status'],'passed');self.assertTrue(state['market_is_constructed'])
        self.assertFalse(state['raw_market_values_parsed']);self.assertFalse(state['final_scenarios_loaded'])
        self.assertFalse(state['canonical_final_compared']);self.assertFalse(state['original_producer_run_called'])
        self.assertEqual(self.proof['rows']['total'],840);self.assertEqual(self.proof['financial_objects'],12)
        self.assertEqual(self.proof['rows']['summary'],14);self.assertEqual(self.proof['rows']['comparisons'],8)
        main=self.proof['array_signatures']['mitigation_main__candidate_posttrade']
        self.assertEqual(main['shape'],[60,30,13,3])

    def test_review_null_and_own_horizons_in_constructed_actual_outputs(self):
        import csv
        with gzip.open(self.synthetic/'Q1_paths.csv.gz','rt') as handle:rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),300)
        for row in rows:
            if row['profile'] in ('A','D'):
                self.assertEqual(row['assigned'],'False');self.assertTrue(all(row[k]=='' for k in m.producer.FIN))
        for profile,horizon in [('B',15),('C',30),('E',6)]:
            self.assertEqual(self.proof['array_signatures']['Q1_'+profile+'__real_wealth']['shape'],[60,horizon+1])
        self.assertEqual(m.compact.readj(self.synthetic/'severe_example.json')['scenario_id'],121)

    def test_complete_ten_artifact_bytes_and_all_arrays_are_checked(self):
        expected=copy.deepcopy(self.proof);m.compare_expected(self.proof,expected)
        self.assertEqual(set(expected['output_bytes_sha256']),m.OUTPUTS-{'traces.npz'})
        for field in ('array_signatures','output_bytes_sha256','rows','financial_objects'):
            bad=copy.deepcopy(expected);bad[field]={} if field!='financial_objects' else 11
            with self.assertRaises(AssertionError):m.compare_expected(self.proof,bad)

    def test_missing_final_manifest_stops_before_both_loaders_and_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);put(root,m.compact.REGISTRY,registry())
            with mock.patch.object(m.compact,'load_paths') as loader:
                with self.assertRaises(FileNotFoundError):m.run(reproduce_fixed_nonblind_final=True,root=root)
            loader.assert_not_called();self.assertFalse((root/'submission_final_reproduction_fixed_nonblind_final_v1').exists())

    def test_complete_noncyclic_manifest_and_release_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,_=gate_fixture(root)
            with mock.patch.object(m,'verify_module_origins'):
                common,proof=m.authorize(root,spec)
            meta=m.compact.readj(root/m.FINAL_BUNDLE)
            self.assertEqual(meta['files_sha256'],{k:v for k,v in common['files_sha256'].items() if k!=str(m.FINAL_BUNDLE)})
            self.assertTrue(proof['previously_exposed_nonblind']);self.assertFalse(common['final_access'])

    def test_changed_source_release_or_spec_never_reaches_loader(self):
        for change in ('source','release','spec'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as td:
                root=Path(td);gate_fixture(root)
                if change=='source':(root/'scripts/submission_final_adapter.py').write_bytes(b'changed source')
                elif change=='release':
                    value=m.compact.readj(root/m.EXPECTED_RELEASE);value['reproduction_ready']=False;put(root,m.EXPECTED_RELEASE,value)
                else:
                    value=registry();value['Q2']['target']=[.4,.1,.5];put(root,m.compact.REGISTRY,value)
                with mock.patch.object(m.compact,'load_paths') as loader:
                    with self.assertRaises((RuntimeError,ValueError)):m.run(reproduce_fixed_nonblind_final=True,root=root)
                loader.assert_not_called()

    def test_explicit_flag_has_exact_train_then_actual_ids_and_failure_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);gate_fixture(root);calls=[]
            def loader(path,ids,**kwargs):
                calls.append((list(ids),kwargs));n=len(list(ids));return np.zeros((n,30,3)),np.zeros((n,30)),np.asarray(list(ids))
            with mock.patch.object(m,'verify_module_origins'),mock.patch.object(m.compact,'load_paths',side_effect=loader),\
                    mock.patch.object(m,'produce',side_effect=ValueError('Deliberate post-load fixture failure')):
                with self.assertRaises(ValueError):m.run(reproduce_fixed_nonblind_final=True,root=root)
            self.assertEqual(calls,[(m.TRAIN,dict(allow_final=False)),(m.ACTUAL,dict(allow_final=True))])
            out=root/'submission_final_reproduction_fixed_nonblind_final_v1';state=m.compact.readj(out/'manifest.json')
            self.assertEqual(state['status'],'failed');self.assertTrue(state['final_scenarios_loaded']);self.assertTrue((out/'failure.txt').exists())
            with mock.patch.object(m.compact,'load_paths') as loader:
                with self.assertRaises(RuntimeError):m.run(reproduce_fixed_nonblind_final=True,root=root)
            loader.assert_not_called()

    def test_immutable_producer_helpers_are_used_without_run_or_parameter_search(self):
        text=Path(m.__file__).read_text();tree=ast.parse(text)
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        for name in ('evaluate_objects','path_rows','summary','compare','severe_example'):
            self.assertIn('producer.'+name,calls)
        self.assertNotIn('producer.run',calls);self.assertNotIn('select_inner',text);self.assertNotIn('fit_joint',text)
        self.assertEqual(calls.count('compact.load_paths'),2)


if __name__=='__main__':unittest.main()
