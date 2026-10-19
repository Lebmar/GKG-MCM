"""Synthetic closure, exact comparison, guarded subprocess and receipt tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('synthetic_nested_reproduction',PROJECT/'scripts/reproduce_nested_dynamic_stage.py')
repro=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(repro)

FAKE_STAGE='''
from pathlib import Path
import hashlib,json,time
import numpy as np
from safeinvest.data import load_paths
ROOT=Path(__file__).resolve().parents[1]
load_paths(ROOT/'data/fake.csv',range(1,121),allow_final=False)
out=ROOT/'results/model_review/nested_dynamic_calibration_v1';out.mkdir(parents=True)
p=json.loads((ROOT/'docs/nested_dynamic_calibration_protocol.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src={x:sha(ROOT/x) for x in p['execution_sources_sha256']}
src['docs/nested_dynamic_calibration_protocol.json']=sha(ROOT/'docs/nested_dynamic_calibration_protocol.json')
(out/'table.csv').write_text('scenario_id,value\\n1,2\\n')
np.savez_compressed(out/'actual_traces.npz',x=np.array([1.,np.nan]),label=np.array(['hold']))
np.savez_compressed(out/'validation_traces.npz',y=np.arange(3))
(out/'progress.json').write_text(json.dumps({'completed_jobs':[{'job_id':'job'+str(i),'seconds':time.perf_counter(),'case_status':'balanced_case','gamma':0.,'training_n':72,'evaluation_n':24} for i in range(30)]}))
m={'completed_utc':str(time.time()),'elapsed_seconds':time.perf_counter(),'sources':src,'sources_at_end':src,'input_sha256':p['frozen_input_sha256'],'input_sha256_at_end':p['frozen_input_sha256'],'final_scenarios_loaded':False,'scenario_ids_loaded':list(range(1,121)),'training_tasks':30,'outputs_sha256':{x.name:sha(x) for x in out.iterdir()}}
(out/'manifest.json').write_text(json.dumps(m))
'''


class ComparisonTests(unittest.TestCase):
    def test_progress_only_runtime_can_differ_not_identity_gamma_case_or_n(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a/progress.json';b=Path(t)/'b/progress.json';a.parent.mkdir();b.parent.mkdir()
            original={'completed_jobs':[dict(job_id='c0_inner1',seconds=1.,gamma=.025,training_n=90,evaluation_n=30,case_status='balanced_case')]}
            changed=deepcopy(original);changed['completed_jobs'][0]['seconds']=2.
            a.write_text(json.dumps(original));b.write_text(json.dumps(changed))
            self.assertIn('runtime',repro.compare_artifact(a,b)['comparison'])
            for key,value in [('job_id','other'),('gamma',.05),('training_n',72),('case_status','risk_review')]:
                bad=deepcopy(changed);bad['completed_jobs'][0][key]=value;b.write_text(json.dumps(bad))
                with self.assertRaises(AssertionError):repro.compare_artifact(a,b)

    def test_npz_all_named_arrays_exact_no_financial_tolerance(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.npz';b=Path(t)/'b.npz'
            np.savez(a,x=np.array([1.,np.nan]),y=np.array(['hold']))
            np.savez_compressed(b,x=np.array([1.,np.nan]),y=np.array(['hold']))
            self.assertEqual(repro.compare_artifact(a,b)['comparison'],'exact_named_arrays')
            np.savez_compressed(b,x=np.array([np.nextafter(1.,2.),np.nan]),y=np.array(['hold']))
            with self.assertRaises(AssertionError):repro.compare_artifact(a,b)

    def test_ast_import_closure_adds_transitive_relative_module_and_package_init(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'src/pkg').mkdir(parents=True);(root/'scripts').mkdir()
            (root/'scripts/entry.py').write_text('from pkg.a import value\n')
            (root/'src/pkg/__init__.py').write_text('')
            (root/'src/pkg/a.py').write_text('from .b import value\n')
            (root/'src/pkg/b.py').write_text('value=1\n')
            closure=repro.import_closure(root,['scripts/entry.py'])
            self.assertEqual(set(closure),{'scripts/entry.py','src/pkg/a.py','src/pkg/b.py','src/pkg/__init__.py'})
            with self.assertRaises(ValueError):repro.safe_file(root,'../escape')


class IsolatedReplaySmoke(unittest.TestCase):
    def make_fixture(self,root,forbidden=False):
        for p in ('scripts','tests','src/safeinvest','docs','data'):(root/p).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PROJECT/repro.SELF,root/repro.SELF)
        shutil.copyfile(PROJECT/repro.TEST,root/repro.TEST)
        (root/'requirements.txt').write_text('numpy>=2\npandas>=2\nscipy>=1\n')
        (root/'src/safeinvest/__init__.py').write_text('')
        (root/'src/safeinvest/data.py').write_text('def load_paths(path,scenario_ids,allow_final=False):\n return None\n')
        (root/'data/fake.csv').write_bytes(b'opaque-synthetic-fixture-not-market-data')
        script=FAKE_STAGE.replace('range(1,121),allow_final=False','[121],allow_final=True') if forbidden else FAKE_STAGE
        (root/repro.RUNNER).write_text(script)
        source={str(repro.RUNNER):repro.sha(root/repro.RUNNER),'src/safeinvest/data.py':repro.sha(root/'src/safeinvest/data.py')}
        p={'execution_ready':True,'execution_sources_sha256':source,
           'frozen_input_sha256':{'data/fake.csv':repro.sha(root/'data/fake.csv')}}
        (root/repro.PROTOCOL).write_text(json.dumps(p))
        result=subprocess.run([sys.executable,str(root/repro.RUNNER)],cwd=root,
            env={**__import__('os').environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_full_synthetic_subprocess_receipt_log_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_fixture(root)
            baseline=repro.sha(root/repro.STAGE/'manifest.json')
            out=root/'results/model_review/replay_receipt'
            receipt=repro.reproduce(root,out)
            self.assertEqual(receipt['status'],'passed');self.assertEqual(receipt['training_tasks'],30)
            self.assertEqual(receipt['source_input_artifact_sha256_start'],receipt['source_input_artifact_sha256_end'])
            self.assertEqual(baseline,repro.sha(root/repro.STAGE/'manifest.json'))
            self.assertTrue((out/'replay_log.txt').is_file())
            calls=json.loads((out/'loader_guard.json').read_text())['calls']
            self.assertEqual(calls,[dict(scenario_ids=list(range(1,121)),allow_final=False)])
            self.assertIn('src/safeinvest/__init__.py',receipt['closure_additions_beyond_stage_manifest'])
            self.assertIn('requirements.txt',receipt['closure_sha256'])
            self.assertIn('runtime',receipt['artifact_comparisons']['progress.json']['comparison'])
            with self.assertRaises(ValueError):repro.reproduce(root,out)

    def test_runtime_guard_rejects_final_ids_and_never_writes_passed_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_fixture(root,forbidden=True)
            out=root/'results/model_review/rejected_replay'
            with self.assertRaises(RuntimeError):repro.reproduce(root,out)
            self.assertTrue((out/'replay_log.txt').exists())
            self.assertIn('only permits', (out/'replay_log.txt').read_text())
            self.assertFalse((out/'manifest.json').exists())


if __name__=='__main__':unittest.main()
