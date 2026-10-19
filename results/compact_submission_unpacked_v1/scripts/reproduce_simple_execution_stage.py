"""Isolated exact replay of the frozen125-job simple execution diagnostic stage only.

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
STAGE=Path('results/model_review/simple_execution_comparison_v1')
PROTOCOL=Path('docs/simple_execution_comparison_protocol.json')
RUNNER=Path('scripts/run_simple_execution_comparison.py')
SELF=Path('scripts/reproduce_simple_execution_stage.py')
TEST=Path('tests/test_reproduce_simple_execution_stage.py')
DEFAULT_OUT=Path('results/model_review/simple_execution_reproduction_v1')


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
    if manifest['sources']!=protocol['execution_sources_sha256'] or manifest['input_sha256']!=protocol['frozen_input_sha256']:
        raise ValueError('Executed source/input set differs from frozen protocol')
    for mapping in (protocol['execution_sources_sha256'],protocol['frozen_input_sha256']):
        for rel,digest in mapping.items():
            if rel in declared and declared[rel]!=digest: raise ValueError('Conflicting frozen SHA: '+rel)
            if sha(safe_file(root,rel))!=digest: raise ValueError('Frozen file changed: '+rel)
            declared[rel]=digest
    if str(RUNNER) not in declared:raise ValueError('Frozen execution sources must include the runner')
    for rel in (str(PROTOCOL),):
        declared[rel]=sha(safe_file(root,rel))
    closure=import_closure(root,declared)
    return closure,{k:v for k,v in closure.items() if k not in manifest['sources'] and k not in protocol['frozen_input_sha256']}


def compare_artifact(left,right):
    left,right=Path(left),Path(right)
    byte_equal=sha(left)==sha(right)
    if byte_equal and left.suffix!='.npz': return {'comparison':'byte_exact','excluded_fields':[]}
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
                equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind=='f' else np.array_equal(x,y)
                if not equal: raise AssertionError('NPZ values differ: '+key)
            return {'comparison':'exact_named_arrays','array_count':len(a.files),'container_byte_exact':byte_equal,
                    'excluded_fields':['ZIP container metadata only; no array tolerance']}
    raise AssertionError('Deterministic artifact bytes differ: '+left.name)


def compare_study(original,replayed):
    original,replayed=Path(original),Path(replayed)
    a=json.loads((original/'manifest.json').read_text()); b=json.loads((replayed/'manifest.json').read_text())
    for m in (a,b):
        if (m.get('status')!='complete' or m['final_scenarios_loaded'] is not False
                or m['scenario_ids_loaded']!=list(range(1,121)) or m['training_tasks']!=125
                or m['outer_fits']!=25 or m['repeat_count']!=5 or m['candidate_count']!=3):
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
import hashlib, json, os, pathlib, runpy, sys
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
runpy.run_path(str(root/'scripts/run_simple_execution_comparison.py'),run_name='__main__')
if not calls: raise RuntimeError('Development-loader guard was never exercised')
(root/'replay_load_guard.json').write_text(json.dumps({'calls':calls,'final_scenarios_loaded':False}))
original_root=pathlib.Path(os.environ['SAFEINVEST_REPLAY_ORIGINAL_ROOT']).resolve()
local_imports={}
for name,module in tuple(sys.modules.items()):
    file=getattr(module,'__file__',None)
    if not file: continue
    path=pathlib.Path(file).resolve()
    if any(path.is_relative_to(original_root/folder) for folder in ('src','research','scripts','tests')):
        raise RuntimeError('Original project module escaped isolation: '+name)
    if path.is_relative_to(root) and path.is_file():
        local_imports[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'replay_runtime_imports.json').write_text(json.dumps({'local_imported_files_sha256':local_imports,'original_project_module_imported':False}))
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
    audit=json.loads(safe_file(original,'independent_audit.json').read_text())
    if manifest.get('status')!='complete' or audit.get('status')!='passed':raise ValueError('Completed stage and passed independent audit required')
    audited_manifest=audit.get('sources_and_artifacts_at_start',{}).get(str(STAGE/'manifest.json'),audit.get('parent_manifest_sha256'))
    if audited_manifest!=sha(original/'manifest.json'):raise ValueError('Audit does not bind current original manifest')
    if manifest.get('protocol_sha256')!=sha(root/PROTOCOL):raise ValueError('Original protocol snapshot differs')
    closure,extra=collect_closure(root,manifest,protocol)
    baseline={str(path.relative_to(root)):sha(path) for path in sorted(original.rglob('*')) if path.is_file()}
    tool_sources={str(SELF):sha(safe_file(root,SELF)),str(TEST):sha(safe_file(root,TEST))}
    start={**closure,**baseline,**tool_sources}; started=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='safeinvest-simple125-replay-') as temporary:
            copy_root=Path(temporary)
            for rel,digest in closure.items():
                target=copy_root/rel; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(safe_file(root,rel),target)
                if sha(target)!=digest: raise AssertionError('Copied closure hash mismatch: '+rel)
            if (copy_root/STAGE).exists(): raise AssertionError('Original outputs must not enter replay workspace')
            environment=dict(os.environ)
            environment['PYTHONPATH']=os.pathsep.join(str(copy_root/x) for x in ('src','research','scripts'))
            environment['PYTHONNOUSERSITE']='1'
            environment['SAFEINVEST_REPLAY_ORIGINAL_ROOT']=str(root)
            result=subprocess.run([sys.executable,'-c',LAUNCHER],cwd=copy_root,env=environment,capture_output=True,text=True)
            log=(result.stdout+result.stderr).replace(str(copy_root),'<isolated-copy>').replace(str(root),'<original-workspace>')
            output.mkdir(parents=True)
            (output/'replay_log.txt').write_text(log)
            if result.returncode: raise RuntimeError('Isolated stage failed; see replay_log.txt')
            comparisons=compare_study(original,copy_root/STAGE)
            guard=json.loads((copy_root/'replay_load_guard.json').read_text())
            (output/'loader_guard.json').write_text(json.dumps(guard,indent=2)+'\n')
            imported=json.loads((copy_root/'replay_runtime_imports.json').read_text())
            (output/'runtime_imports.json').write_text(json.dumps(imported,indent=2)+'\n')
            copied_after={rel:sha(safe_file(copy_root,rel)) for rel in closure}
            if closure!=copied_after:raise AssertionError('A copied source/input/import file changed during replay')
        original_after={str(path.relative_to(root)):sha(path) for path in sorted(original.rglob('*')) if path.is_file()}
        if baseline!=original_after:raise AssertionError('Original archive file set or SHA changed during replay')
        end={rel:sha(safe_file(root,rel)) for rel in start}
        if start!=end: raise AssertionError('Original source/input/artifact files changed during replay')
        receipt={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.perf_counter()-started,'stage':str(STAGE),'training_tasks':125,
            'baseline_manifest_sha256':start[str(STAGE/'manifest.json')],
            'baseline_independent_audit_sha256':start[str(STAGE/'independent_audit.json')],
            'upstream_independent_audit_passed_before_replay':True,
            'closure_sha256':closure,'closure_additions_beyond_stage_manifest':extra,
            'copied_files_sha256_before':closure,'copied_files_sha256_after':copied_after,
            'original_archive_sha256_start':baseline,'original_archive_sha256_end':original_after,
            'wrapper_test_sources_sha256':tool_sources,
            'source_input_artifact_sha256_start':start,'source_input_artifact_sha256_end':end,
            'artifact_comparisons':comparisons,'manifest_excluded_fields':['completed_utc','elapsed_seconds',
                'outputs_sha256 digest values only for normalized-progress/NPZ-container comparisons, after independent own-manifest verification'],
            'runtime':{'python':platform.python_version(),'implementation':platform.python_implementation(),
                'platform':platform.platform(),'interpreter':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','matplotlib')}},
            'outputs_sha256':{name:sha(output/name) for name in ('replay_log.txt','loader_guard.json','runtime_imports.json')},
            'final_scenarios_loaded':False,'scenario_ids':list(range(1,121)),
            'limitations':['Same-source deterministic replay, not an independent algorithm or independent market evidence',
                'The frozen raw CSV is copied opaquely, including all rows needed for its original SHA; guarded loading admits1..120 only',
                'AST closure covers local static imports; installed Python/NumPy/Pandas/SciPy runtime versions and tracked requirements are recorded',
                'No financial numerical tolerance; NPZ named-array exact equality permits only container metadata differences',
                'Progress job seconds and manifest timestamps/runtime vary; all other fields remain exact',
                'This is a source/input/import-closure copy, not a fresh Git clone, independent algorithm or new independent market sample']}
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
    print(json.dumps({'status':result['status'],'training_tasks':125,'output':str(args.output)}),flush=True)


if __name__=='__main__': main()
