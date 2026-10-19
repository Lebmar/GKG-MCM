"""Small isolated synthetic file tests; never run real diagnostic producers."""
import copy,gzip,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('last_mile_replay_test',ROOT/'scripts/reproduce_last_mile_diagnostics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(folder,*,mtime=1,array_dtype='f8'):
    folder.mkdir();data=b'scenario_id,value\n1,1.25\n2,NULL\n'
    with gzip.GzipFile(filename=str(folder/'table.csv.gz'),mode='wb',mtime=mtime) as f:f.write(data)
    m.writej(folder/'summary.json',dict(status='complete',values=[1.25,None]))
    np.savez_compressed(folder/'arrays.npz',value=np.array([1.,np.nan,-0.],dtype=array_dtype),labels=np.array(['a','b']))
    declared={p.name:m.sha(p) for p in folder.iterdir()}
    manifest=dict(status='complete',started_utc='2026-10-19T01:00:00+00:00',completed_utc='2026-10-19T01:01:00+00:00',outputs_sha256=declared,
                  sources_and_inputs_at_start={'input.txt':'a'*64},sources_and_inputs_at_end={'input.txt':'a'*64})
    m.writej(folder/'manifest.json',manifest);return declared


class LastMileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_both_diagnostic_outputs_and_audits_cannot_be_seeded(self):
        for p in m.PROFILES:
            for name in [p['original_output']+'/manifest.json',p['original_output']+'/independent_audit.json']:
                with self.assertRaises(RuntimeError):m.no_output_seed({name:'a'*64})
        m.no_output_seed({'results/state_budget_mitigation_v1/traces.npz':'a'*64,'results/support_failure_diagnostic_v1/failure.txt':'b'*64})

    def test_copy_only_explicit_hashes_without_existing_destination(self):
        source=self.root/'source';work=self.root/'work';source.mkdir();work.mkdir();(source/'a.txt').write_bytes(b'parent saved bytes')
        mapping={'a.txt':m.sha(source/'a.txt')};m.copy_dependencies(source,work,mapping)
        self.assertEqual((work/'a.txt').read_bytes(),b'parent saved bytes')
        self.assertFalse(any((work/p['original_output']).exists() for p in m.PROFILES))
        with self.assertRaises(RuntimeError):m.copy_dependencies(source,work,mapping)

    def test_relative_paths_and_conflicting_hashes_rejected(self):
        for name in ['.','','../escape','/tmp/x','a\\b','a//b']:
            with self.assertRaises(ValueError):m.relative(name)
        with self.assertRaises(ValueError):m.merge({'a':'a'*64},{'a':'b'*64})

    def test_gzip_only_header_mtime_and_UTC_changes_are_allowed(self):
        old=self.root/'old';new=self.root/'new';declared=fixture(old,mtime=1);fixture(new,mtime=2)
        stamp=m.readj(new/'manifest.json');stamp['started_utc']='2026-10-19T02:00:00+00:00';stamp['completed_utc']='2026-10-19T02:01:00+00:00';m.writej(new/'manifest.json',stamp)
        proof=m.compare_outputs(old,new,declared)
        self.assertEqual(proof['status'],'passed');self.assertTrue(proof['files']['table.csv.gz']['decompressed_text_bytes_equal'])
        self.assertEqual(proof['normalized_root_manifest_fields'],['started_utc','completed_utc','outputs_sha256/table.csv.gz'])
        self.assertEqual(proof['original_manifest']['outputs_sha256'],declared)

    def test_changed_table_or_summary_is_not_hidden_by_manifest_hash_update(self):
        for name in ['summary.json','table.csv.gz']:
            with self.subTest(name=name):
                old=self.root/('old_'+name);new=self.root/('new_'+name);declared=fixture(old);fixture(new)
                if name.endswith('.gz'):
                    with gzip.GzipFile(filename=str(new/name),mode='wb',mtime=1) as f:f.write(b'different text\n')
                else:m.writej(new/name,dict(values=[1.2500000000000002,None]))
                manifest=m.readj(new/'manifest.json');manifest['outputs_sha256'][name]=m.sha(new/name);m.writej(new/'manifest.json',manifest)
                with self.assertRaises(AssertionError):m.compare_outputs(old,new,declared)

    def test_npz_all_field_dtype_shape_and_value_exactness(self):
        a=self.root/'a.npz';b=self.root/'b.npz';np.savez(a,x=np.array([1.,np.nan]));np.savez(b,x=np.array([1.,np.nan]))
        self.assertTrue(m.compare_npz(a,b)['exact'])
        for values in [np.array([1.,np.nan],dtype='f4'),np.array([[1.,np.nan]]),np.array([2.,np.nan])]:
            np.savez(b,x=values)
            with self.assertRaises(AssertionError):m.compare_npz(a,b)
        np.savez(b,y=np.array([1.,np.nan]))
        with self.assertRaises(AssertionError):m.compare_npz(a,b)

    def test_manifest_other_fields_and_extra_outputs_never_ignored(self):
        old=self.root/'old';new=self.root/'new';declared=fixture(old);fixture(new)
        manifest=m.readj(new/'manifest.json');manifest['extra']='notallowed';m.writej(new/'manifest.json',manifest)
        with self.assertRaises(AssertionError):m.compare_outputs(old,new,declared)
        fixture3=self.root/'other';fixture(fixture3);(fixture3/'unexpected.txt').write_text('extra')
        with self.assertRaises(AssertionError):m.compare_outputs(old,fixture3,declared)

    def test_old_attempt_and_missing_activation_prevent_replay(self):
        (self.root/m.OUT).mkdir(parents=True)
        with mock.patch.object(m,'preflight') as gate,mock.patch.object(m.subprocess,'run') as child:
            with self.assertRaises(RuntimeError):m.run(self.root)
        gate.assert_not_called();child.assert_not_called()
        other=self.root/'missing';other.mkdir()
        with mock.patch.object(m.subprocess,'run') as child:
            with self.assertRaises(FileNotFoundError):m.run(other)
        child.assert_not_called();self.assertFalse((other/m.OUT).exists())

    def test_bootstrap_paths_and_python_flags_are_fixed_without_original_results(self):
        text=Path(m.__file__).read_text()
        self.assertIn("'-I','-B'",text);self.assertIn("('PYTHONPATH','PYTHONHOME')",text)
        self.assertNotIn('load_paths(',text);self.assertIn("'safeinvest.data' in sys.modules",m.BOOTSTRAP)
        self.assertIn("runpy.run_path(str(root/script),run_name='__main__')",m.BOOTSTRAP)


if __name__=='__main__':unittest.main()
