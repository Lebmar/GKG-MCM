"""Temporary canonical arrays, fake audit metadata and tiny ZIPs only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import prepare_compact_final_submission as m
from test_submission_final_adapter import put,registry


def fixture(root,*,bad_review=False):
    sources={name:put(root,Path(name),b'# synthetic source\n') for name in m.SOURCES}
    for name in m.v1.EXTRAS:put(root,Path(name),b'synthetic package input\n')
    put(root,m.compact.REGISTRY,registry());put(root,Path(m.EXTRA),b'nonblind synthetic documentation\n')
    put(root,Path('docs/DATA_NOTICE.md'),b'synthetic data notice\n')
    expected_sha=put(root,m.v1.PREP/'expected_development.json',dict(version='compact-canonical-development-expectations-v1',object_specification=m.compact.specification(registry())))
    put(root,m.v1.PREP/'manifest.json',dict(status='complete',outputs_sha256={'expected_development.json':expected_sha}))
    spec=m.addon.producer.object_specification(registry());groups=[];arrays={}
    for obj in spec['objects']:
        available=obj['rule']!='Review';names=['real_wealth','fees'] if available else []
        if bad_review and obj['object_id']=='Q1_A':available=True;names=['real_wealth','fees']
        for key in names:
            arrays[obj['object_id']+'__'+key]=np.full((60,obj['horizon']+(key=='real_wealth')),100. if key=='real_wealth' else 0.)
        groups.append(dict(**obj,status='funded' if available else 'q1_review',financial_available=available,scenario_ids=m.addon.ACTUAL,
            training_ids=m.addon.TRAIN if obj['rule']=='mitigation' else None,array_names=names,omitted_diagnostic_arrays=[]))
    put(root,m.CANON/'trace_groups.json',groups);np.savez_compressed(root/m.CANON/'traces.npz',**arrays)
    for name in m.addon.OUTPUTS-{'traces.npz','trace_groups.json'}:put(root,m.CANON/name,b'canonical synthetic artifact bytes\n')
    outputs={name:m.compact.sha(root/m.CANON/name) for name in m.addon.OUTPUTS}
    manifest=dict(status='complete',scope='previously_exposed_nonblind',new_fits=0,new_selections=0,
        training_ids_loaded=m.addon.TRAIN,final_ids_loaded=m.addon.ACTUAL,sources_and_inputs_at_start=sources,sources_and_inputs_at_end=sources,
        outputs_sha256=outputs,Q2_rows=240,Q1_rows=300,Q1_reference_rows=300,summary_rows=14,comparisons=8,trace_groups=14)
    parent_sha=put(root,m.CANON/'manifest.json',manifest)
    bound={**sources,**{str(m.CANON/name):digest for name,digest in outputs.items()},str(m.CANON/'manifest.json'):parent_sha}
    put(root,m.CANON/'independent_audit.json',dict(status='passed',producer_manifest_sha256=parent_sha,sources_and_artifacts_at_start=bound,sources_and_artifacts_at_end=bound))
    replay_file=put(root,m.REPLAY/'comparison.json',dict(status='passed',fixture=True))
    put(root,m.REPLAY/'manifest.json',dict(status='passed',sources_and_original_artifacts_at_start=bound,
        sources_and_original_artifacts_at_end=bound,outputs_sha256={'comparison.json':replay_file}))
    plan=dict(version='compact-fixed-nonblind-final-preparation-v1',execution_ready=True,training_ids=m.addon.TRAIN,evaluation_ids=m.addon.ACTUAL,
        previously_exposed_nonblind=True,new_independent_validation=False,object_specification=spec)
    plan_sha=put(root,m.PLAN,plan)
    inputs={name:m.compact.sha(root/name) for name in m.REQUIRED_INPUTS};inputs.update(m.canonical_closure(root))
    put(root,m.FREEZE,dict(execution_ready=True,protocol_sha256=plan_sha,source_sha256=sources,input_sha256=inputs))


class CompactFinalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_missing_freeze_stops_before_canonical_npz(self):
        fixture(self.root);(self.root/m.FREEZE).unlink()
        with mock.patch.object(m.np,'load') as loader:
            with self.assertRaises(FileNotFoundError):m.prepare_expected(self.root)
        loader.assert_not_called();self.assertFalse((self.root/m.PREP).exists())

    def test_all14_groups_24_minimal_arrayfields_and_ten_bytes_outputs(self):
        fixture(self.root);m.prepare_expected(self.root)
        expected=m.compact.readj(self.root/m.PREP/'expected_nonblind_final.json');receipt=m.compact.readj(self.root/m.PREP/'manifest.json')
        self.assertEqual(receipt['status'],'complete');self.assertEqual(len(expected['array_signatures']),24)
        self.assertEqual(expected['financial_objects'],12);self.assertEqual(expected['rows']['total'],840)
        self.assertEqual(set(expected['output_bytes_sha256']),m.addon.OUTPUTS-{'traces.npz'})
        self.assertFalse(any(name.startswith('Q1_A__') or name.startswith('Q1_D__') for name in expected['array_signatures']))
        release=m.compact.readj(self.root/m.PREP/'expected_nonblind_final_release.json')
        self.assertTrue(release['previously_exposed_nonblind']);self.assertFalse(release['new_independent_validation'])
        self.assertEqual(release['expected_sha256'],m.compact.sha(self.root/m.PREP/'expected_nonblind_final.json'))
        with self.assertRaises(RuntimeError):m.prepare_expected(self.root)

    def test_review_financial_arrays_rejected_with_failed_attempt_preserved(self):
        fixture(self.root,bad_review=True)
        with self.assertRaises(RuntimeError):m.prepare_expected(self.root)
        self.assertEqual(m.compact.readj(self.root/m.PREP/'manifest.json')['status'],'failed')
        self.assertTrue((self.root/m.PREP/'failure.txt').exists())

    def test_wrong_independent_binding_rejected_before_numeric_parse(self):
        fixture(self.root);value=m.compact.readj(self.root/m.CANON/'independent_audit.json');value['producer_manifest_sha256']='0'*64
        put(self.root,m.CANON/'independent_audit.json',value)
        with mock.patch.object(m.np,'load') as loader:
            with self.assertRaises(RuntimeError):m.prepare_expected(self.root)
        loader.assert_not_called()

    def test_separate_build_activation_noncyclic_manifests_and_v1_untouched(self):
        fixture(self.root);m.prepare_expected(self.root)
        v1path=self.root/m.v1.ZIP;v1path.parent.mkdir(parents=True);v1path.write_bytes(b'preexisting immutable V1 fixture\n');old=v1path.read_bytes()
        with self.assertRaises(FileNotFoundError):m.build(self.root)
        _,mapping=m.packing_inventory(self.root)
        release=dict(version='compact-fixed-nonblind-final-build-release-v1',build_ready=True,
            protocol_sha256=m.compact.sha(self.root/m.PLAN),source_freeze_sha256=m.compact.sha(self.root/m.FREEZE),
            preparation_manifest_sha256=m.compact.sha(self.root/m.PREP/'manifest.json'),allowlist_without_release_sha256=m.compact.canonical_sha(mapping))
        put(self.root,m.BUILD_RELEASE,release);m.build(self.root)
        receipt=m.compact.readj(self.root/m.BUILD/'manifest.json');self.assertEqual(receipt['status'],'complete')
        self.assertLess(receipt['archive_bytes'],50_000_000);self.assertEqual(v1path.read_bytes(),old)
        unpack=m.v1.extract_verified_zip(self.root/m.ZIP,self.root/'unpacked')
        common=m.compact.readj(unpack/m.compact.BUNDLE);final=m.compact.readj(unpack/m.addon.FINAL_BUNDLE)
        self.assertEqual(final['files_sha256'],{k:v for k,v in common['files_sha256'].items() if k!=str(m.addon.FINAL_BUNDLE)})
        self.assertFalse(common['final_access']);self.assertTrue(final['explicit_reproduction_ready'])
        with mock.patch.object(m.addon,'verify_module_origins'):m.addon.authorize(unpack,m.addon.producer.object_specification(registry()))
        with self.assertRaises(RuntimeError):m.build(self.root)


if __name__=='__main__':unittest.main()
