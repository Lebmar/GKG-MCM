"""Constructed metadata/files/runtime only; no actual final output is read."""
from pathlib import Path
import copy,json,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import reproduce_mitigation_locked_final_evaluation as a


def plan():return dict(version='mitigation-locked-final-isolated-reproduction-plan-v1',execution_ready=False,training_ids=a.producer.TRAIN,evaluation_ids=a.producer.ACTUAL,producer_source=a.SCRIPT,producer_output_directory=str(a.producer.OUT),output_directory=str(a.OUT),normalized_root_manifest_fields=['started_utc','completed_utc'],python_executable='.venv/bin/python',previously_exposed_nonblind=True,expected_producer_manifest_sha256_source='reproduction_source_freeze.expected_producer_manifest_sha256')


class FinalReplayTests(unittest.TestCase):
    def test_fixed_single_source_plan_and_unknown_result_blocked(self):
        p=plan();a.validate_plan(p)
        with self.assertRaises(RuntimeError):a.metadata_inputs(ROOT,p,None)
        for key,value in [('execution_ready',True),('expected_producer_manifest_sha256_source',None),('training_ids',[1]),('evaluation_ids',list(range(1,61))),('producer_source','other.py'),('normalized_root_manifest_fields',['completed_utc'])]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(RuntimeError):a.validate_plan(q)

    def test_copy_refuses_own_final_output_dependency(self):
        with patch.object(a.producer,'preflight',return_value=({}, {}, {str(a.producer.OUT/'traces.npz'):'a'*64})),patch.object(a,'readj') as read:
            with self.assertRaises(RuntimeError):a.metadata_inputs(ROOT,plan(),'a'*64)
            read.assert_not_called()

    def test_release_metadata_is_not_bypassed_for_dependency_copy(self):
        with patch.object(a.producer,'preflight',side_effect=RuntimeError('missing release')),patch.object(a,'readj') as read:
            with self.assertRaises(RuntimeError):a.metadata_inputs(ROOT,plan(),'a'*64)
            read.assert_not_called()

    def test_cli_has_no_output_or_ready_override_and_shared_globals_unchanged(self):
        self.assertIn('sys.argv = [str(script), "--execute"]',a.BOOTSTRAP)
        self.assertIn('"--output", str(target)',a.shared.BOOTSTRAP)
        self.assertNotIn('"--output", str(target)',a.BOOTSTRAP)
        command=a.command(ROOT/'.venv/bin/python',Path('/fresh'),Path('/receipt'),ROOT/'.venv')
        self.assertEqual(command[0],str(ROOT/'.venv/bin/python'));self.assertEqual(command[1],'-I')
        with self.assertRaises(ValueError):a.command('.venv/bin/python','/fresh','/r','/v')

    def test_constructed_bootstrap_uses_fresh_domain_packages_and_originalvenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'src/safeinvest').mkdir(parents=True);(root/'src/safeinvest/__init__.py').write_text('')
            for package in ('experimental_execution','safeinvest_research'):
                (root/'research'/package).mkdir(parents=True);(root/'research'/package/'marker.py').write_text('value=1')
            (root/'scripts').mkdir();(root/a.SCRIPT).write_text('import sys\nimport experimental_execution.marker, safeinvest_research.marker\nassert sys.argv[1:]==["--execute"]\n')
            receipt=root/'runtime.json';cmd=a.command(ROOT/'.venv/bin/python',root,receipt,ROOT/'.venv')
            result=subprocess.run(cmd,cwd=root,env=a.shared.clean_environment({}),capture_output=True,check=False)
            self.assertEqual(result.returncode,0,result.stderr.decode());meta=a.readj(receipt)
            self.assertEqual(meta['status'],'passed');self.assertEqual(meta['isolated'],1);self.assertEqual(meta['no_user_site'],1)
            self.assertIn('experimental_execution.marker',meta['package_origins_after']);self.assertFalse(list(root.rglob('__pycache__')))

    def prepare_outputs(self,root):
        original,replay=root/'original',root/'replay';original.mkdir();replay.mkdir();outputs={}
        for name in ('Q2_paths.csv.gz','Q1_paths.csv.gz','Q1_thirds_references.csv.gz','summary.csv.gz','paired_paths.csv.gz','object_specification.json','trace_groups.json','initial_witness.json','comparisons.json','severe_example.json'):
            for folder in (original,replay):(folder/name).write_bytes(b'constructed\n')
            outputs[name]=a.sha(original/name)
        for folder in (original,replay):np.savez(folder/'traces.npz',x=np.array([1.,np.nan]),ids=np.array([121,180]))
        outputs['traces.npz']=a.sha(original/'traces.npz')
        m=dict(status='complete',scope='previously_exposed_nonblind',started_utc='2026-10-19T01:00:00Z',completed_utc='2026-10-19T01:01:00Z',outputs_sha256=outputs)
        a.writej(original/'manifest.json',m);m['completed_utc']='2026-10-19T02:00:00Z';a.writej(replay/'manifest.json',m)
        return original,replay,outputs

    def test_all_declared_final_outputs_exact_with_only_two_times_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig,rep,outputs=self.prepare_outputs(Path(tmp));result=a.shared.compare_outputs(orig,rep,outputs)
            self.assertEqual(result['status'],'passed');self.assertEqual(result['declared_files'],11)
            self.assertEqual(result['files']['traces.npz']['npz']['fields_checked'],2)
            m=a.readj(rep/'manifest.json');m['scope']='blind';a.writej(rep/'manifest.json',m)
            self.assertEqual(a.shared.compare_outputs(orig,rep,outputs)['status'],'differences')

    def test_missing_extra_bytes_dtype_shape_and_pairednan_differences_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig,rep,outputs=self.prepare_outputs(Path(tmp));(rep/'unexpected').write_text('x')
            self.assertEqual(a.shared.compare_outputs(orig,rep,outputs)['status'],'differences');(rep/'unexpected').unlink()
            (rep/'summary.csv.gz').write_bytes(b'changed')
            self.assertEqual(a.shared.compare_outputs(orig,rep,outputs)['status'],'differences')
            np.savez(rep/'traces.npz',x=np.array([[1.,np.nan]]),ids=np.array([121,180],dtype='int32'))
            self.assertFalse(a.shared.compare_npz(orig/'traces.npz',rep/'traces.npz')['exact'])

    def test_failed_attempt_retains_evidence_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'authorize',side_effect=RuntimeError('synthetic missing release')):
            root=Path(tmp)
            with self.assertRaises(RuntimeError):a.run(root)
            out=root/a.OUT;self.assertEqual(a.readj(out/'manifest.json')['status'],'failed');self.assertTrue((out/'failure.txt').is_file())
            with self.assertRaises(RuntimeError):a.run(root)


if __name__=='__main__':unittest.main()
