"""Isolated byte replay of the final saved-data guard; no fitting or raw CSV."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=Path('results/model_review/global_policy_admission_v1')
PROTOCOL=Path('docs/global_policy_admission_protocol.json')
OUT=ROOT/'results/model_review/global_policy_admission_reproduction_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():raise RuntimeError('Never overwrite a replay receipt')
    p=json.loads((ROOT/PROTOCOL).read_text());m=json.loads((ROOT/STAGE/'manifest.json').read_text())
    a=json.loads((ROOT/STAGE/'independent_audit.json').read_text())
    if a['status']!='passed' or a['parent_manifest_sha256']!=sha(ROOT/STAGE/'manifest.json'):raise RuntimeError('Bound passed audit required')
    closure={**p['execution_sources_sha256'],**p['frozen_input_sha256'],str(PROTOCOL):sha(ROOT/PROTOCOL)}
    if any(sha(ROOT/f)!=v for f,v in closure.items()):raise RuntimeError('Frozen closure changed')
    if any(f.startswith('data/') for f in closure):raise RuntimeError('Saved-data guard must not need raw market data')
    before={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/STAGE).iterdir() if f.is_file()}
    self_sha=sha(__file__)
    with tempfile.TemporaryDirectory(prefix='safeinvest-final-guard-') as temp:
        root=Path(temp)
        for name,digest in closure.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            if sha(target)!=digest:raise RuntimeError('Closure copy differs')
        env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'}
        child=subprocess.run([sys.executable,str(root/'scripts/finalize_global_policy_admission.py')],cwd=root,env=env,capture_output=True,text=True)
        if child.returncode:raise RuntimeError('Isolated replay failed: '+child.stderr)
        copied_after={f:sha(root/f) for f in closure}
        if copied_after!=closure:raise RuntimeError('Copied source/input changed')
        outputs={}
        for name,digest in m['outputs_sha256'].items():
            if sha(ROOT/STAGE/name)!=digest or sha(root/STAGE/name)!=digest:raise RuntimeError('Replay bytes differ: '+name)
            outputs[name]=digest
        if json.loads((root/STAGE/'manifest.json').read_text())!=m:raise RuntimeError('Replay manifest differs')
    after={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/STAGE).iterdir() if f.is_file()}
    if before!=after or sha(__file__)!=self_sha:raise RuntimeError('Original output/reproducer source changed')
    OUT.mkdir(parents=True)
    receipt=dict(status='passed',sources={'scripts/reproduce_global_policy_admission.py':self_sha},
        parent_manifest_sha256=sha(ROOT/STAGE/'manifest.json'),closure_sha256=closure,copied_files_sha256_before=closure,copied_files_sha256_after=copied_after,
        original_archive_sha256_start=before,original_archive_sha256_end=after,byte_exact_output_sha256=outputs,
        replay_manifest_exact=True,new_fits=0,raw_market_csv_copied=False,market_data_loaded=False,final_scenarios_loaded=False,
        limitations=['Isolated same-source/input copy on current runtime, not a fresh Git clone or independent validation',
            'Saved development outputs are reused; no new market observations or risk certificate'])
    (OUT/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print('passed',len(outputs),'byte-exact artifacts')


if __name__=='__main__':main()
