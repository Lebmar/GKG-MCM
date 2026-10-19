"""Isolated exact replay of the frozen30-job joint liquidity/refit stage only.

The raw CSV is copied opaquely to retain its frozen SHA; it may contain180
scenarios. The subprocess installs an additional load_paths ID/allow_final
check and records every call. Only original development1..120 may be loaded.
No old experiment is rerun and the original study is never an output target.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGE=Path('results/model_review/joint_liquidity_refit_v1')
PROTOCOL=Path('docs/joint_liquidity_refit_protocol.json')
RUNNER=Path('scripts/run_joint_liquidity_refit.py')
SELF=Path('scripts/reproduce_joint_liquidity_stage.py')
TEST=Path('tests/test_reproduce_joint_liquidity_stage.py')
DEFAULT_OUT=Path('results/model_review/joint_liquidity_reproduction_v1')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_file(root,relative):
    root=Path(root).resolve(); rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Closure paths must be repository-relative')
    result=(root/rel).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f'Missing or escaping closure file: {relative}')
    return result


def _module_file(root,module):
    for base in ('src','research','scripts','tests'):
        stem=Path(base).joinpath(*module.split('.'))
        for candidate in (stem.with_suffix('.py'),stem/'__init__.py'):
            if (root/candidate).is_file(): return candidate
    return None


def import_closure(root,initial):
    """Resolve local static imports recursively; include package initializers.

    External imports use the current interpreter and are recorded by installed
    package versions. No pip installation or untracked repository helper is
    used. Namespace packages need no fabricated __init__.py.
    """
    root=Path(root).resolve(); found=set(map(str,initial)); todo=list(found)
    while todo:
        rel=Path(todo.pop()); file=safe_file(root,rel)
        if rel.suffix!='.py': continue
        module='.'.join(rel.with_suffix('').parts[1:])
        package=module if rel.name=='__init__.py' else module.rpartition('.')[0]
        modules=[]
        for node in ast.walk(ast.parse(file.read_text(),filename=str(rel))):
            if isinstance(node,ast.Import): modules.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parts=package.split('.') if package else []
                    if node.level>len(parts): continue
                    prefix='.'.join(parts[:len(parts)-node.level+1])
                    name='.'.join(x for x in (prefix,node.module or '') if x)
                else: name=node.module or ''
                if name: modules.append(name)
                modules.extend('.'.join(x for x in (name,a.name) if x) for a in node.names if a.name!='*')
        for name in modules:
            candidate=_module_file(root,name)
            if candidate is None: continue
            additions=[candidate]
            for parent in candidate.parents:
                init=parent/'__init__.py'
                if (root/init).is_file(): additions.append(init)
            for addition in additions:
                text=str(addition)
                if text not in found: found.add(text); todo.append(text)
    return {rel:sha(safe_file(root,rel)) for rel in sorted(found)}


def collect_closure(root,manifest,protocol):
    """Validate source/input hashes, then add explicitly hashed import closure."""
    root=Path(root).resolve(); declared={}
    for mapping in (manifest['sources'],protocol['frozen_input_sha256']):
        for rel,digest in mapping.items():
            if rel in declared and declared[rel]!=digest: raise ValueError('Conflicting frozen SHA: '+rel)
            if sha(safe_file(root,rel))!=digest: raise ValueError('Frozen file changed: '+rel)
            declared[rel]=digest
    for rel in (str(PROTOCOL),str(RUNNER),str(SELF),str(TEST),'requirements.txt'):
        declared[rel]=sha(safe_file(root,rel))
    closure=import_closure(root,declared)
    return closure,{k:v for k,v in closure.items() if k not in manifest['sources'] and k not in protocol['frozen_input_sha256']}


def compare_artifact(left,right):
    left,right=Path(left),Path(right)
    if sha(left)==sha(right): return {'comparison':'byte_exact','excluded_fields':[]}
    if left.name=='progress.json':
        a,b=json.loads(left.read_text()),json.loads(right.read_text())
        for value in (a,b):
            for job in value['completed_jobs']: job.pop('seconds',None)
        if a!=b: raise AssertionError('Progress differs beyond completed_jobs[*].seconds')
        return {'comparison':'exact_JSON_except_declared_runtime','excluded_fields':['completed_jobs[*].seconds']}
    if left.suffix=='.npz':
        with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
            if set(a.files)!=set(b.files): raise AssertionError('NPZ names differ: '+left.name)
            for key in a.files:
                x,y=a[key],b[key]
                if x.dtype!=y.dtype or x.shape!=y.shape: raise AssertionError('NPZ dtype/shape differs: '+key)
                equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
                if not equal: raise AssertionError('NPZ values differ: '+key)
            return {'comparison':'exact_named_arrays','array_count':len(a.files),
                    'excluded_fields':['ZIP container metadata only; no array tolerance']}
    raise AssertionError('Deterministic artifact bytes differ: '+left.name)


def compare_study(original,replayed):
    original,replayed=Path(original),Path(replayed)
    a=json.loads((original/'manifest.json').read_text()); b=json.loads((replayed/'manifest.json').read_text())
    for m in (a,b):
        if m['final_scenarios_loaded'] is not False or m['scenario_ids_loaded']!=list(range(1,121)) or m['training_tasks']!=30:
            raise AssertionError('Unexpected replay scope')
        if m['sources']!=m['sources_at_end'] or m['input_sha256']!=m['input_sha256_at_end']:
            raise AssertionError('Source/input changed during the stage')
    if set(a['outputs_sha256'])!=set(b['outputs_sha256']): raise AssertionError('Output artifact set changed')
    comparisons={}
    for rel in a['outputs_sha256']:
        left=safe_file(original,rel); right=safe_file(replayed,rel)
        if sha(left)!=a['outputs_sha256'][rel] or sha(right)!=b['outputs_sha256'][rel]:
            raise AssertionError('Artifact hash differs from its own manifest: '+rel)
        comparisons[rel]={**compare_artifact(left,right),'original_sha256':sha(left),'replay_sha256':sha(right)}
    # Output hash values were independently verified and artifacts compared
    # above, including permitted progress/NPZ-container differences.
    for m in (a,b):
        m.pop('completed_utc',None); m.pop('elapsed_seconds',None)
        for rel,result in comparisons.items():
            if result['comparison']!='byte_exact':
                m['outputs_sha256'][rel]='<independently-verified-exact-artifact>'
    if a!=b: raise AssertionError('Manifest differs beyond declared time/artifact-container metadata')
    return comparisons


LAUNCHER=r'''
import json, pathlib, runpy, sys
root=pathlib.Path.cwd()
sys.path[:0]=[str(root/'src'),str(root/'research'),str(root/'scripts')]
import safeinvest.data as data
original=data.load_paths
calls=[]
def guarded(path,scenario_ids,*args,**kwargs):
    ids=list(scenario_ids)
    allow=kwargs.get('allow_final',False)
    if args or allow is not False or ids!=list(range(1,121)):
        raise RuntimeError('Isolated replay only permits explicit development1..120 and allow_final=False')
    calls.append({'scenario_ids':ids,'allow_final':False})
    return original(path,ids,allow_final=False)
data.load_paths=guarded
runpy.run_path(str(root/'scripts/run_joint_liquidity_refit.py'),run_name='__main__')
if not calls: raise RuntimeError('Development-loader guard was never exercised')
(root/'replay_load_guard.json').write_text(json.dumps({'calls':calls,'final_scenarios_loaded':False}))
'''


def reproduce(root,output):
    root=Path(root).resolve(); output=Path(output).resolve()
    if output.exists(): raise ValueError('Preserve existing receipt; choose a new output')
    if output.is_relative_to(root) and output.parent!=root/'results/model_review':
        raise ValueError('Use a new top-level research receipt or an external directory')
    original=root/STAGE
    manifest=json.loads(safe_file(original,'manifest.json').read_text())
    protocol=json.loads(safe_file(root,PROTOCOL).read_text())
    if protocol.get('execution_ready') is not True: raise ValueError('Original stage protocol is not frozen')
    if manifest['sources']!=manifest['sources_at_end'] or manifest['input_sha256']!=manifest['input_sha256_at_end']:
        raise ValueError('Original stage was not source/input stable')
    closure,extra=collect_closure(root,manifest,protocol)
    baseline={str(STAGE/'manifest.json'):sha(original/'manifest.json'),
              **{str(STAGE/p):sha(safe_file(original,p)) for p in manifest['outputs_sha256']}}
    start={**closure,**baseline}; started=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='safeinvest-joint30-replay-') as temporary:
            copy_root=Path(temporary)
            for rel,digest in closure.items():
                target=copy_root/rel; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(safe_file(root,rel),target)
                if sha(target)!=digest: raise AssertionError('Copied closure hash mismatch: '+rel)
            if (copy_root/STAGE).exists(): raise AssertionError('Original outputs must not enter replay workspace')
            environment=dict(os.environ)
            environment['PYTHONPATH']=os.pathsep.join(str(copy_root/x) for x in ('src','research','scripts'))
            environment['PYTHONNOUSERSITE']='1'
            result=subprocess.run([sys.executable,'-c',LAUNCHER],cwd=copy_root,env=environment,capture_output=True,text=True)
            log=(result.stdout+result.stderr).replace(str(copy_root),'<isolated-copy>').replace(str(root),'<original-workspace>')
            output.mkdir(parents=True)
            (output/'replay_log.txt').write_text(log)
            if result.returncode: raise RuntimeError('Isolated stage failed; see replay_log.txt')
            comparisons=compare_study(original,copy_root/STAGE)
            guard=json.loads((copy_root/'replay_load_guard.json').read_text())
            (output/'loader_guard.json').write_text(json.dumps(guard,indent=2)+'\n')
        end={rel:sha(safe_file(root,rel)) for rel in start}
        if start!=end: raise AssertionError('Original source/input/artifact files changed during replay')
        receipt={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.perf_counter()-started,'stage':str(STAGE),'training_tasks':30,
            'baseline_manifest_sha256':start[str(STAGE/'manifest.json')],
            'closure_sha256':closure,'closure_additions_beyond_stage_manifest':extra,
            'source_input_artifact_sha256_start':start,'source_input_artifact_sha256_end':end,
            'artifact_comparisons':comparisons,'manifest_excluded_fields':['completed_utc','elapsed_seconds',
                'outputs_sha256 digest values only for normalized-progress/NPZ-container comparisons, after independent own-manifest verification'],
            'runtime':{'python':platform.python_version(),'implementation':platform.python_implementation(),
                'platform':platform.platform(),'packages':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy')}},
            'outputs_sha256':{name:sha(output/name) for name in ('replay_log.txt','loader_guard.json')},
            'final_scenarios_loaded':False,'scenario_ids':list(range(1,121)),
            'limitations':['Same-source deterministic replay, not an independent algorithm or independent market evidence',
                'The frozen raw CSV is copied opaquely, including all rows needed for its original SHA; guarded loading admits1..120 only',
                'AST closure covers local static imports; installed Python/NumPy/Pandas/SciPy runtime versions and tracked requirements are recorded',
                'No financial numerical tolerance; NPZ named-array exact equality permits only container metadata differences',
                'Progress job seconds and manifest timestamps/runtime vary; all other fields remain exact']}
        (output/'manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        return receipt
    except Exception:
        # Preserve a produced failure log, never mark an incomplete replay passed.
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/DEFAULT_OUT)
    args=parser.parse_args()
    result=reproduce(ROOT,args.output)
    print(json.dumps({'status':result['status'],'training_tasks':30,'output':str(args.output)}),flush=True)


if __name__=='__main__': main()
