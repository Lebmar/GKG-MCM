"""Temporary synthetic canonical artifacts and tiny ZIPs; no real package."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compact_submission_builder_test',ROOT/'scripts/prepare_compact_submission.py')
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


def fixture(root,*,wrong_model=False):
    sources={name:put(root,Path(name),b'# synthetic source\n') for name in m.SOURCES}
    before={'synthetic/original.py':put(root,Path('synthetic/original.py'),b'# original synthetic evidence\n')}
    ids=list(range(1,121));groups=[];arrays={};summaries=[]
    for obj in m.adapter.OBJECTS:
        main=obj=='mitigation_main';gid='full_'+obj
        groups.append(dict(group_id=gid,kind='full_selected' if main else 'reference',job_id='c0_refit',policy='release_0.5_uniform' if main else obj,
            status='funded',scenario_ids=ids,training_ids=ids,target=[.45,.05,.5],
            candidate=dict(candidate_id='wrong' if wrong_model else 'release_0.5_uniform'),array_names=['real_wealth','fees']))
        arrays[gid+'__real_wealth']=np.full((120,31),100.);arrays[gid+'__fees']=np.zeros((120,30))
        summaries.append(dict(scope='full_selected' if main else 'full_training_reference',policy='state_budget_learner' if main else obj,
            original_n=120,financial_n=120,mean_log_growth=0.))
    put(root,m.CANON/'trace_groups.json',groups);np.savez_compressed(root/m.CANON/'traces.npz',**arrays)
    pd.DataFrame(summaries).to_csv(root/m.CANON/'summary.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    for name in ('path_metrics.csv.gz','selections.json','initial_witnesses.json'):put(root,m.CANON/name,b'bound fixture; not numerically parsed\n')
    outputs={p.name:m.sha(p) for p in (root/m.CANON).iterdir()}
    manifest_sha=put(root,m.CANON/'manifest.json',dict(status='complete',final_scenarios_loaded=False,new_final_reads=0,
        sources_and_inputs_at_start=before,sources_and_inputs_at_end=before,outputs_sha256=outputs))
    bound={**before,str(m.CANON/'manifest.json'):manifest_sha,**{str(m.CANON/name):digest for name,digest in outputs.items()}}
    put(root,m.CANON/'independent_audit.json',dict(status='passed',parent_manifest_sha256=manifest_sha,final_scenarios_loaded=False,
        sources_and_artifacts_at_start=bound,sources_and_artifacts_at_end=bound))
    for name in m.EXTRAS:put(root,Path(name),b'synthetic documentation/input bytes\n')
    put(root,m.adapter.REGISTRY,registry());put(root,Path('docs/DATA_NOTICE.md'),b'synthetic notice\n')
    plan=dict(version='compact-submission-preparation-v1',execution_ready=True,allow_final=False,development_ids=ids,
        maximum_archive_bytes=m.MAX_ZIP,maximum_source_bytes=m.MAX_SOURCE,object_specification=m.adapter.specification(registry()))
    plan_sha=put(root,m.PLAN,plan)
    inputs={name:m.sha(root/name) for name in m.REQUIRED_INPUTS};inputs.update(m.canonical_closure(root))
    put(root,m.FREEZE,dict(execution_ready=True,protocol_sha256=plan_sha,source_sha256=sources,input_sha256=inputs))


class CompactSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_draft_or_missing_freeze_blocks_before_npz_or_csv(self):
        fixture(self.root);(self.root/m.FREEZE).unlink()
        with mock.patch.object(m.np,'load') as npz,mock.patch.object(m.pd,'read_csv') as csv:
            with self.assertRaises(FileNotFoundError):m.prepare_expected(self.root)
        npz.assert_not_called();csv.assert_not_called();self.assertFalse((self.root/m.PREP).exists())

    def test_frozen_canonical_fullgroups_make_small_dtype_shape_digest_proof(self):
        fixture(self.root);m.prepare_expected(self.root)
        proof=m.readj(self.root/m.PREP/'expected_development.json');receipt=m.readj(self.root/m.PREP/'manifest.json')
        self.assertEqual(receipt['status'],'complete');self.assertEqual(receipt['object_count'],4);self.assertEqual(receipt['array_fields'],8)
        self.assertEqual(set(proof['array_signatures']),set(m.adapter.OBJECTS))
        self.assertEqual(proof['array_signatures']['mitigation_main']['real_wealth']['shape'],[120,31])
        self.assertEqual(proof['array_signatures']['mitigation_main']['real_wealth']['dtype'],'<f8')
        self.assertEqual(proof['summaries']['mitigation_main']['original_n'],120)
        self.assertFalse(receipt['final_scenarios_loaded']);self.assertEqual(receipt['new_policy_runs'],0)
        with self.assertRaises(RuntimeError):m.prepare_expected(self.root)

    def test_wrong_canonical_tuple_preserves_failed_preparation(self):
        fixture(self.root,wrong_model=True)
        with self.assertRaises(RuntimeError):m.prepare_expected(self.root)
        self.assertEqual(m.readj(self.root/m.PREP/'manifest.json')['status'],'failed')
        self.assertTrue((self.root/m.PREP/'failure.txt').exists())

    def test_zip_bytes_stable_and_size_gate_is_strict(self):
        payload={'scripts/example.py':b'print(1)\n','docs/example.md':b'fixture\n'}
        a=self.root/'a.zip';b=self.root/'b.zip';m.write_zip(a,payload);m.write_zip(b,payload)
        self.assertEqual(a.read_bytes(),b.read_bytes())
        with zipfile.ZipFile(a) as z:self.assertTrue(all(i.date_time==m.TIMESTAMP for i in z.infolist()))
        too_small=self.root/'small.zip'
        with self.assertRaises(RuntimeError):m.write_zip(too_small,payload,maximum_bytes=1)
        self.assertTrue(too_small.exists())  # Failure evidence is retained.

    def test_traversal_symlink_and_credential_pattern_rejected(self):
        with self.assertRaises(ValueError):m.write_zip(self.root/'bad.zip',{'../escape':b'x'})
        (self.root/'real.py').write_bytes(b'print(1)\n');(self.root/'link.py').symlink_to(self.root/'real.py')
        with self.assertRaises(RuntimeError):m.checked_file(self.root,'link.py')
        (self.root/'token.py').write_bytes(b'sk-'+b'x'*48)
        with self.assertRaises(RuntimeError):m.checked_file(self.root,'token.py')
        evil=self.root/'evil.zip'
        with zipfile.ZipFile(evil,'x') as z:z.writestr('../escape',b'x')
        with self.assertRaises(ValueError):m.extract_verified_zip(evil,self.root/'extract')
        self.assertFalse((self.root/'extract').exists())

    def test_allowlist_excludes_reports_results_and_secrets(self):
        fixture(self.root);m.prepare_expected(self.root)
        put(self.root,Path('report/extra.py'),b'# not package source\n');put(self.root,Path('results/huge.npz'),b'not copied')
        put(self.root,Path('docs/secrets.json'),{'secret':'not bundled'})
        paths,mapping=m.packing_inventory(self.root)
        self.assertNotIn('report/extra.py',mapping);self.assertNotIn('results/huge.npz',mapping);self.assertNotIn('docs/secrets.json',mapping)
        self.assertIn(str(m.adapter.EXPECTED),mapping);self.assertIn('scripts/submission_adapter.py',mapping)

    def test_build_needs_separate_release_then_verified_extract_has_same_manifest(self):
        fixture(self.root);m.prepare_expected(self.root)
        with self.assertRaises(FileNotFoundError):m.build(self.root)
        self.assertFalse((self.root/m.BUILD).exists())
        _,mapping=m.packing_inventory(self.root)
        release=dict(version='compact-submission-build-release-v1',build_ready=True,protocol_sha256=m.sha(self.root/m.PLAN),
            source_freeze_sha256=m.sha(self.root/m.FREEZE),preparation_manifest_sha256=m.sha(self.root/m.PREP/'manifest.json'),
            allowlist_without_release_sha256=m.adapter.canonical_sha(mapping))
        put(self.root,m.BUILD_RELEASE,release);m.build(self.root)
        receipt=m.readj(self.root/m.BUILD/'manifest.json');self.assertEqual(receipt['status'],'complete')
        self.assertLess(receipt['archive_bytes'],50_000_000);self.assertFalse(receipt['development_reproduction_executed'])
        destination=m.extract_verified_zip(self.root/m.ZIP,self.root/'fresh_unpack')
        compact=m.readj(destination/m.adapter.BUNDLE)
        self.assertFalse(compact['final_access']);self.assertTrue(compact['development_ready'])
        self.assertEqual(set(compact['files_sha256']),set(m.packing_inventory(self.root)[1]))
        with self.assertRaises(RuntimeError):m.extract_verified_zip(self.root/m.ZIP,destination)
        with self.assertRaises(RuntimeError):m.build(self.root)


if __name__=='__main__':unittest.main()
