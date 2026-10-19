"""Pure constructed-file tests; no market loader or real producer execution."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import reproduce_technical_rebuild as r


class ReplayTests(unittest.TestCase):
    def test_unsafe_paths_rejected(self):
        for path in ('', '.', '../a', '/tmp/a', 'a/../b', 'a//b', './a'):
            with self.subTest(path=path),self.assertRaises(ValueError):r.relative(path)
        self.assertEqual(r.relative('results/a.npz'),Path('results/a.npz'))

    def test_hash_conflict_rejected(self):
        data={};r.merge(data,{'a':'a'*64})
        with self.assertRaises(ValueError):r.merge(data,{'a':'b'*64})
        for digest in ('A'*64,'x'*64,'a'*63):
            with self.assertRaises(ValueError):r.merge({}, {'a':digest})

    def test_complete_four_profile_plan_and_no_omission(self):
        p=json.loads((ROOT/r.PLAN).read_text());p.update(execution_ready=True,development_ids=list(range(1,121)))
        declared=copy.deepcopy(p['profiles'])
        self.assertEqual(r.validate_plan(p),declared)
        self.assertEqual([z['profile_id'] for z in declared[:4]],r.FIRST_PROFILES)
        p['profiles'].pop(0)
        with self.assertRaises(RuntimeError):r.validate_plan(p)

    def test_no_final_or_duplicate_profiles(self):
        p=json.loads((ROOT/r.PLAN).read_text());p.update(execution_ready=True,development_ids=list(range(1,121)))
        p['allow_final']=True
        with self.assertRaises(RuntimeError):r.validate_plan(p)
        p['allow_final']=False;p['profiles'][1]['original_output']=p['profiles'][0]['original_output']
        with self.assertRaises(ValueError):r.validate_plan(p)

    def test_explicit_fixed_development_loader_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'p.py'
            path.write_text('r,p,i=load_paths(file,range(1,121),allow_final=False)')
            r.loader_source_gate(path)
            for bad in ('r,p,i=load_paths(file,range(1,181),allow_final=False)',
                        'r,p,i=load_paths(file,range(1,121),allow_final=True)', 'r,p,i=load_paths(file,range(1,121))'):
                path.write_text(bad)
                with self.assertRaises(RuntimeError):r.loader_source_gate(path)

    def test_virtualenv_symlink_spelling_preserved(self):
        p={'script':'scripts/run_x.py','original_output':'results/x'}
        cmd=r.isolated_command('/original/.venv/bin/python','/fresh',p,'/evidence/runtime.json','/original/.venv')
        self.assertEqual(cmd[0],'/original/.venv/bin/python');self.assertEqual(cmd[1],'-I');self.assertEqual(cmd[2],'-c')
        with self.assertRaises(ValueError):r.isolated_command('.venv/bin/python','/fresh',p,'/receipt','/venv')

    def test_environment_isolation(self):
        env=r.clean_environment(dict(PYTHONPATH='/old',PYTHONHOME='/global',KEEP='yes'))
        self.assertNotIn('PYTHONPATH',env);self.assertNotIn('PYTHONHOME',env)
        self.assertEqual(env['PYTHONNOUSERSITE'],'1');self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1');self.assertEqual(env['KEEP'],'yes')

    def test_bootstrap_real_isolation_on_constructed_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'src/safeinvest').mkdir(parents=True);(root/'src/safeinvest/__init__.py').write_text('')
            for name in ('experimental_execution','safeinvest_research'):
                (root/'research'/name).mkdir(parents=True);(root/'research'/name/'marker.py').write_text('value=1')
            (root/'scripts').mkdir();script=root/'scripts/dummy.py'
            script.write_text('import experimental_execution.marker, safeinvest_research.marker\n')
            profile=dict(script='scripts/dummy.py',original_output='results/synthetic')
            receipt=root/'runtime.json';cmd=r.isolated_command(ROOT/'.venv/bin/python',root,profile,receipt,ROOT/'.venv')
            result=subprocess.run(cmd,cwd=root,env=r.clean_environment({}),capture_output=True,check=False)
            self.assertEqual(result.returncode,0,result.stderr.decode());meta=r.readj(receipt)
            self.assertEqual(meta['status'],'passed');self.assertEqual(meta['isolated'],1);self.assertEqual(meta['no_user_site'],1)
            self.assertIn('experimental_execution.marker',meta['package_origins_after'])
            self.assertFalse(list(root.rglob('__pycache__')))

    def test_only_two_root_timestamps_normalized(self):
        a=dict(started_utc='2026-10-19T01:00:00+00:00',completed_utc='2026-10-19T02:00:00+00:00',status='complete',nested={'started_utc':'keep'})
        b=copy.deepcopy(a);b['started_utc']='2026-10-19T03:00:00Z';b['completed_utc']='2026-10-19T04:00:00Z'
        self.assertEqual(r.normalize_manifest(a),r.normalize_manifest(b));self.assertIn('started_utc',r.normalize_manifest(a)['nested'])
        b['status']='different';self.assertNotEqual(r.normalize_manifest(a),r.normalize_manifest(b))
        a['completed_utc']='2026-10-19T02:00:00+08:00'
        with self.assertRaises(ValueError):r.normalize_manifest(a)

    def test_npz_inventory_dtype_shape_values_and_paired_NaN(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.npz',Path(folder)/'b.npz'
            np.savez(a,x=np.array([1.,np.nan]),ids=np.array([1,2],dtype='int64'))
            np.savez(b,x=np.array([1.,np.nan]),ids=np.array([1,2],dtype='int64'))
            self.assertTrue(r.compare_npz(a,b)['exact'])
            np.savez(b,x=np.array([1.,np.nan]),ids=np.array([1,2],dtype='int32'))
            self.assertFalse(r.compare_npz(a,b)['exact'])
            np.savez(b,x=np.array([[1.,np.nan]]),ids=np.array([1,2],dtype='int64'))
            self.assertFalse(r.compare_npz(a,b)['exact'])
            np.savez(b,x=np.array([np.nan,1.]),ids=np.array([1,2],dtype='int64'))
            self.assertFalse(r.compare_npz(a,b)['exact'])
            np.savez(b,x=np.array([1.,np.nan]))
            self.assertFalse(r.compare_npz(a,b)['exact'])

    def test_copy_forbids_own_outputs_and_checks_bytes(self):
        with tempfile.TemporaryDirectory() as orig,tempfile.TemporaryDirectory() as fresh:
            a,b=Path(orig),Path(fresh);(a/'file').write_text('ok')
            r.copy_dependencies(a,b,{'file':r.sha(a/'file')},'results/target')
            self.assertEqual((b/'file').read_text(),'ok')
            with self.assertRaises(RuntimeError):r.copy_dependencies(a,b,{'file':r.sha(a/'file')},'results/target')
            with self.assertRaises(RuntimeError):r.copy_dependencies(a,b,{'results/target/tree.npz':'a'*64},'results/target')

    def test_transitive_freeze_closure_and_target_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'docs').mkdir();(root/'scripts').mkdir()
            (root/'scripts/p.py').write_text('load_paths(file,range(1,121),allow_final=False)')
            r.writej(root/'docs/p.json',dict(development_ids=list(range(1,121)),allow_final=False))
            (root/'required').write_text('old predecessor identity')
            r.writej(root/'docs/parent_source_freeze.json',dict(source_sha256={'required':r.sha(root/'required')},input_sha256={'docs/p.json':r.sha(root/'docs/p.json')}))
            freeze=dict(execution_ready=True,protocol_sha256=r.sha(root/'docs/p.json'),
                source_sha256={'scripts/p.py':r.sha(root/'scripts/p.py')},input_sha256={'docs/parent_source_freeze.json':r.sha(root/'docs/parent_source_freeze.json')})
            r.writej(root/'docs/p_source_freeze.json',freeze)
            p=dict(script='scripts/p.py',protocol='docs/p.json',source_freeze='docs/p_source_freeze.json',original_output='results/target')
            self.assertIn('required',r.dependency_closure(root,p))
            freeze['input_sha256']['results/target/array.npz']='a'*64;r.writej(root/'docs/p_source_freeze.json',freeze)
            with self.assertRaises(RuntimeError):r.dependency_closure(root,p)

    def prepare_comparison(self,a,b):
        for folder in (a,b):
            folder.mkdir();(folder/'data.csv').write_text('x\n1\n')
        mapping={'data.csv':r.sha(a/'data.csv')}
        m=dict(status='complete',started_utc='2026-10-19T00:00:00Z',completed_utc='2026-10-19T01:00:00Z',outputs_sha256=mapping)
        r.writej(a/'manifest.json',m);m['completed_utc']='2026-10-19T02:00:00Z';r.writej(b/'manifest.json',m)
        return mapping

    def test_compare_all_outputs_and_extra_files(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b';mapping=self.prepare_comparison(a,b)
            self.assertEqual(r.compare_outputs(a,b,mapping)['status'],'passed')
            (b/'hidden.json').write_text('{}');self.assertEqual(r.compare_outputs(a,b,mapping)['status'],'differences')

    def test_bytes_and_root_hashes_not_relaxed(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b';mapping=self.prepare_comparison(a,b)
            (b/'data.csv').write_text('x\r\n1\r\n')
            self.assertEqual(r.compare_outputs(a,b,mapping)['status'],'differences')
            m=r.readj(b/'manifest.json');m['outputs_sha256']['data.csv']='a'*64;r.writej(b/'manifest.json',m)
            self.assertIn('root_manifest_output_hashes',[d['kind'] for d in r.compare_outputs(a,b,mapping)['differences']])

    def test_repeat_manifest_exact_no_timestamp_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b';mapping=self.prepare_comparison(a,b)
            for root in (a,b):(root/'repeat').mkdir()
            (a/'repeat/manifest.json').write_text('{"value":1}');(b/'repeat/manifest.json').write_text('{ "value": 1 }')
            mapping['repeat/manifest.json']=r.sha(a/'repeat/manifest.json')
            for root in (a,b):
                m=r.readj(root/'manifest.json');m['outputs_sha256']=mapping;r.writej(root/'manifest.json',m)
            self.assertEqual(r.compare_outputs(a,b,mapping)['status'],'differences')

    def test_failed_gate_retained_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'attempt'
            with patch.object(r,'ready',side_effect=RuntimeError('synthetic missing freeze')):
                with self.assertRaises(RuntimeError):r.run(out,root)
            state=r.readj(out/'manifest.json');self.assertEqual(state['status'],'failed');self.assertFalse(state['automatic_retry'])
            self.assertTrue((out/'failure.txt').is_file());self.assertEqual(state['profiles'],[])
            with self.assertRaises(RuntimeError):r.run(out,root)


if __name__=='__main__':unittest.main()
