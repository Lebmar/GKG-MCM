"""Compact entrypoint synthetic and fixed-development guards only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compact_submission_adapter_test',ROOT/'scripts/submission_adapter.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def registry():
    return dict(development_ids=list(range(1,121)),Q2=dict(profile='C',strategic_product='Balanced',target=[.45,.05,.5],
        horizon=30,L=.4,cost_rate=.0025,selected_candidate_id='release_0.5_uniform',release_fraction=.5,
        forecast_rule='uniform',support_mode='calendar',service_alpha=.1,persistent_target_relearning=False))


def put(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,bytes):p.write_bytes(value)
    else:m.writej(p,value)
    return m.sha(p)


def compact_fixture(root):
    spec=m.specification(registry());put(root,m.REGISTRY,registry())
    arms={name:{'values':np.arange(240,dtype=float).reshape(120,2)} for name in m.OBJECTS}
    summaries={name:dict(original_n=120,mean_log_growth=.01) for name in m.OBJECTS}
    expected=dict(version='compact-canonical-development-expectations-v1',object_specification=spec,
        array_signatures=m.signatures(arms),summaries=summaries)
    files={name:put(root,Path(name),b'# synthetic frozen source\n') for name in m.REQUIRED_RUNTIME}
    for name,value in [(m.REGISTRY,registry()),(m.EXPECTED,expected),(Path('data/SafeInvest_Synthetic_Data.csv'),b'synthetic bytes; mocked loader\n')]:
        files[str(name)]=put(root,name,value)
    put(root,m.BUNDLE,dict(version='compact-technical-source-bundle-v1',status='complete',development_ready=True,
        final_access=False,object_specification=spec,files_sha256=files))
    return arms,summaries


class SubmissionAdapterTests(unittest.TestCase):
    def test_default_synthetic_uses_generated_arrays_not_real_csv(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);put(root,m.REGISTRY,registry());start=time.perf_counter()
            with mock.patch.object(m,'load_paths') as loader:m.run(root=root)
            loader.assert_not_called();self.assertLess(time.perf_counter()-start,1.)
            receipt=m.readj(root/'submission_reproduction_synthetic_v1/manifest.json')
            self.assertEqual(receipt['status'],'passed');self.assertFalse(receipt['raw_market_values_parsed'])
            self.assertFalse(receipt['canonical_development_compared']);self.assertFalse(receipt['final_scenarios_loaded'])

    def test_development_without_completed_manifest_or_changed_registry_blocks_loader(self):
        for mode in ('missing','draft','registry'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);compact_fixture(root)
                if mode=='missing':(root/m.BUNDLE).unlink()
                elif mode=='draft':
                    b=m.readj(root/m.BUNDLE);b['development_ready']=False;put(root,m.BUNDLE,b)
                else:
                    r=registry();r['Q2']['target']=[.4,.1,.5];put(root,m.REGISTRY,r)
                with mock.patch.object(m,'load_paths') as loader:
                    with self.assertRaises((RuntimeError,ValueError,FileNotFoundError)):m.run('development',root=root)
                loader.assert_not_called();self.assertFalse((root/'submission_reproduction_development_v1').exists())

    def test_explicit_development_loads_only120_and_compares_all_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);arms,summaries=compact_fixture(root)
            with mock.patch.object(m,'verify_loaded_modules'),mock.patch.object(m,'load_paths',return_value=(np.zeros((120,30,3)),np.zeros((120,30)),np.arange(1,121))) as loader, \
                 mock.patch.object(m,'evaluate_fixed',return_value=(arms,summaries)):
                m.run('development',root=root)
            self.assertEqual(list(loader.call_args.args[1]),list(range(1,121)));self.assertEqual(loader.call_args.kwargs,{'allow_final':False})
            receipt=m.readj(root/'submission_reproduction_development_v1/manifest.json')
            self.assertEqual(receipt['status'],'passed');self.assertTrue(receipt['canonical_development_compared'])
            self.assertEqual(receipt['scenario_ids'],list(range(1,121)));self.assertEqual(receipt['new_fits'],0);self.assertEqual(receipt['new_selections'],0)

    def test_array_digest_preserves_schema_values_and_signedzero_but_normalizes_nan_endian(self):
        value=np.array([1.,np.nan,-0.]);other=value.copy();other.view(np.uint64)[1]=0x7ff8000000000001
        self.assertEqual(m.array_signature(value),m.array_signature(other))
        self.assertEqual(m.array_signature(value),m.array_signature(value.astype('>f8')))
        for changed in (value.astype(np.float32),value.reshape(1,3),np.array([2.,np.nan,-0.]),np.array([1.,np.nan,0.])):
            self.assertNotEqual(m.array_signature(value),m.array_signature(changed))

    def test_mismatch_preserves_failed_attempt_and_no_final_mode_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);arms,summaries=compact_fixture(root);bad=copy.deepcopy(arms);bad['mitigation_main']['values'][0,0]+=1
            with mock.patch.object(m,'verify_loaded_modules'),mock.patch.object(m,'load_paths',return_value=(np.zeros((120,30,3)),np.zeros((120,30)),np.arange(1,121))), \
                 mock.patch.object(m,'evaluate_fixed',return_value=(bad,summaries)):
                with self.assertRaises(AssertionError):m.run('development',root=root)
            out=root/'submission_reproduction_development_v1';self.assertEqual(m.readj(out/'manifest.json')['status'],'failed')
            self.assertTrue((out/'failure.txt').exists())
            with mock.patch.object(m,'load_paths') as loader:
                with self.assertRaises(RuntimeError):m.run('development',root=root)
                with self.assertRaises(ValueError):m.run('final',root=root)
            loader.assert_not_called()

    def test_synthetic_financial_summary_matches_frozen_canonical_helper(self):
        registry_spec=m.specification(registry());market=np.zeros((2,30,3));pi=np.zeros((2,30));ids=np.array([101,102])
        arms,summaries=m.evaluate_fixed(market,pi,ids,np.zeros((3,30,3)),np.zeros((3,30)),[1,2,3],registry_spec)
        spec=importlib.util.spec_from_file_location('compact_frozen_financial_helper',ROOT/'scripts/run_state_budget_mitigation.py')
        canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical)
        for name in m.OBJECTS:
            expected=canonical.summarize_records(canonical.rows_from_arm(arms[name],ids,{}))
            m.compare_summary(summaries[name],expected)
        m.verify_loaded_modules(ROOT)

    def test_relative_paths_and_symlinks_are_not_valid_manifest_members(self):
        for name in ('.','','../x','/tmp/x','a/../b','a\\b','a//b'):
            with self.assertRaises(ValueError):m.safe_name(name)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'real').write_bytes(b'data');(root/'link').symlink_to(root/'real')
            with self.assertRaises(RuntimeError):m.verify_files(root,{'link':m.sha(root/'real')})


if __name__=='__main__':unittest.main()
