"""Isolated same-source replay of two saved-data diagnostics, not main models."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,json,os,re,shutil,subprocess,sys,tempfile,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=Path('docs/last_mile_diagnostic_reproduction_protocol_v1.json')
FREEZE=Path('docs/last_mile_diagnostic_reproduction_source_freeze_v1.json')
ACTIVATION=Path('docs/last_mile_diagnostic_reproduction_activation_v1.json')
OUT=Path('results/technical_last_mile_reproduction_v1')
SOURCES=('scripts/reproduce_last_mile_diagnostics.py','tests/test_reproduce_last_mile_diagnostics.py')
PROFILES=(
    dict(profile_id='review_operation',script='scripts/diagnose_review_operation.py',flag='--run',
         protocol='docs/review_operation_diagnostic_protocol_v1.json',source_freeze='docs/review_operation_diagnostic_source_freeze_v1.json',
         activation='docs/review_operation_diagnostic_release_v1.json',original_output='results/review_operation_diagnostic_v1'),
    dict(profile_id='support_failure',script='scripts/diagnose_support_failure_v2.py',flag='--execute',
         protocol='docs/support_failure_diagnostic_protocol_v2.json',source_freeze='docs/support_failure_diagnostic_source_freeze_v2.json',
         activation='docs/support_failure_diagnostic_activation_v2.json',original_output='results/support_failure_diagnostic_v2'))
DOMAINS=('diagnostics','safeinvest','safeinvest_research','experimental_execution')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def relative(name):
    if not isinstance(name,str):raise ValueError('Canonical relative path required')
    p=Path(name)
    if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix()!=name:raise ValueError('Canonical relative path required')
    return p
def merge(result,mapping):
    for k,v in mapping.items():
        relative(k)
        if not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None or (k in result and result[k]!=v):raise ValueError('Exact nonconflicting SHA map required')
        result[k]=v
def current(root,mapping):
    root=Path(root).resolve()
    for name,digest in mapping.items():
        path=root/relative(name)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path)!=digest:raise RuntimeError('Immutable dependency changed: '+name)
        cursor=path.parent
        while cursor!=root:
            if cursor.is_symlink():raise RuntimeError('No copied symlink parent')
            cursor=cursor.parent
def no_output_seed(mapping):
    forbidden=[Path(p['original_output']) for p in PROFILES]+[OUT]
    for name in mapping:
        path=relative(name)
        if any(path==p or p in path.parents for p in forbidden):raise RuntimeError('Never seed either target output/audit or replay artifacts: '+name)


def dependency_closure(root,profile):
    """Only producer freezes and their required inputs, not auditor freezes."""
    root=Path(root);f=readj(root/profile['source_freeze']);release=readj(root/profile['activation'])
    if (f.get('protocol_sha256')!=sha(root/profile['protocol']) or release.get('execution_ready') is not True
            or release.get('protocol_sha256')!=sha(root/profile['protocol']) or release.get('source_freeze_sha256')!=sha(root/profile['source_freeze'])):
        raise RuntimeError('Unchanged original diagnostic activation required')
    result={};merge(result,f['source_sha256']);merge(result,f['input_sha256'])
    merge(result,{profile[k]:sha(root/profile[k]) for k in ('protocol','source_freeze','activation')})
    # Preserve regular package initializers if present, without importing them.
    for parent in ('research/diagnostics/__init__.py','src/safeinvest/__init__.py'):
        if (root/parent).is_file():merge(result,{parent:sha(root/parent)})
    if profile['script'] not in result:raise RuntimeError('Exact copied producer entrypoint required')
    no_output_seed(result);current(root,result)
    return dict(sorted(result.items()))


def original_evidence(root,profile):
    root=Path(root);folder=root/profile['original_output'];m=readj(folder/'manifest.json');audit=readj(folder/'independent_audit.json')
    if (m.get('status')!='complete' or m.get('final_scenarios_loaded') is not False or m['sources_and_inputs_at_start']!=m['sources_and_inputs_at_end']
            or m.get('protocol_sha256')!=sha(root/profile['protocol']) or m.get('source_freeze_sha256')!=sha(root/profile['source_freeze'])
            or audit.get('status')!='passed' or audit.get('producer_manifest_sha256')!=sha(folder/'manifest.json')):
        raise RuntimeError('Current passed original diagnostic/audit identity required')
    outputs={};merge(outputs,m['outputs_sha256'])
    if not outputs or 'manifest.json' in outputs:raise RuntimeError('Complete declared nonmanifest output inventory required')
    current(folder,outputs)
    return m,outputs,{str(Path(profile['original_output'])/name):sha(folder/name) for name in [*outputs,'manifest.json','independent_audit.json']}


def required_inputs(root=ROOT):
    """Metadata and byte hashes only, usable before activation by root."""
    dependencies={};originals={};profiles=[]
    for profile in PROFILES:
        dep=dependency_closure(root,profile);m,outputs,evidence=original_evidence(root,profile)
        merge(dependencies,dep);merge(originals,evidence)
        profiles.append(dict(profile=profile,dependencies=dep,original_manifest=m,declared_outputs=outputs))
    result={};merge(result,dependencies);merge(result,originals)
    return result,profiles


def preflight(root=ROOT):
    root=Path(root);p,f,a=readj(root/PLAN),readj(root/FREEZE),readj(root/ACTIVATION)
    if (p.get('version')!='last-mile-saved-diagnostic-isolated-reproduction-v1' or p.get('execution_ready') is not False or p.get('profiles')!=list(PROFILES)
            or p.get('python_executable')!='.venv/bin/python' or p.get('new_main_model_runs')!=0
            or f.get('execution_ready') is not True or f.get('protocol_sha256')!=sha(root/PLAN)
            or a.get('execution_ready') is not True or a.get('protocol_sha256')!=sha(root/PLAN) or a.get('source_freeze_sha256')!=sha(root/FREEZE)
            or not set(SOURCES)<=set(f.get('source_sha256',{}))):raise RuntimeError('Freeze and activate bounded diagnostic replay first')
    snapshot={};merge(snapshot,f['source_sha256']);merge(snapshot,f['input_sha256']);current(root,snapshot)
    needed,profiles=required_inputs(root)
    if any(snapshot.get(k)!=v for k,v in needed.items()):raise RuntimeError('Freeze full copied and comparison-only evidence closure')
    return snapshot,profiles


def copy_dependencies(root,workspace,mapping):
    root,workspace=Path(root),Path(workspace);no_output_seed(mapping)
    for name,digest in mapping.items():
        path=workspace/relative(name)
        if path.exists():raise RuntimeError('Never overwrite an isolated dependency')
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,path)
        if sha(path)!=digest:raise RuntimeError('Copied dependency hash differs')
    if any((workspace/p['original_output']).exists() for p in PROFILES):raise RuntimeError('Both diagnostic output roots must be absent before child execution')


BOOTSTRAP=r'''
import json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();script=sys.argv[2];flag=sys.argv[3]
sys.path[:0]=[str(root/'scripts'),str(root/'src'),str(root/'research')]
sys.argv=[str(root/script),flag]
try:
    runpy.run_path(str(root/script),run_name='__main__')
finally:
    origins={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in ('diagnostics','safeinvest','safeinvest_research','experimental_execution'):continue
        paths=[]
        if getattr(module,'__file__',None):paths.append(str(Path(module.__file__).resolve()))
        elif getattr(module,'__path__',None):paths.extend(str(Path(p).resolve()) for p in module.__path__)
        if not paths or any(not Path(p).is_relative_to(root) for p in paths):raise RuntimeError('Domain module escaped isolated root: '+name)
        origins[name]=paths
    if 'safeinvest.data' in sys.modules:raise RuntimeError('Raw market loader module must not be imported by these diagnostics')
    import numpy,pandas
    record=dict(python=sys.version,executable=sys.executable,numpy=numpy.__version__,pandas=pandas.__version__,module_origins=origins,
        raw_loader_module_imported=False,raw_CSV_may_be_byte_hashed=True,new_main_model_runs=0)
    (root/'__replay_runtime.json').write_text(json.dumps(record,indent=2)+'\n')
'''


def compare_npz(a,b):
    rows=[]
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        if len(set(x.files))!=len(x.files) or len(set(y.files))!=len(y.files) or set(x.files)!=set(y.files):raise AssertionError('NPZ field inventory differs')
        for name in sorted(x.files):
            u,v=x[name],y[name]
            equal=u.shape==v.shape and u.dtype==v.dtype and bool(np.array_equal(u,v,equal_nan=True) if u.dtype.kind in 'fc' else np.array_equal(u,v))
            rows.append(dict(field=name,dtype=u.dtype.str,shape=list(u.shape),scalars=u.size,exact=equal))
            if not equal:raise AssertionError('NPZ exact field differs: '+name)
    return dict(fields=len(rows),scalars=sum(r['scalars'] for r in rows),exact=True,field_details=rows)


def normalize_manifest(value):
    result=copy.deepcopy(value)
    for key in ('started_utc','completed_utc'):
        stamp=datetime.fromisoformat(result.pop(key))
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds()!=0:raise ValueError('Declared UTC timestamps required')
    return result


def compare_outputs(original,replay,declared):
    original,replay=Path(original),Path(replay);expected=set(declared)|{'manifest.json'}
    if {str(p.relative_to(replay)) for p in replay.rglob('*') if p.is_file()}!=expected:raise AssertionError('Complete replay file inventory differs')
    a,b=readj(original/'manifest.json'),readj(replay/'manifest.json')
    if b.get('status')!='complete' or set(b['outputs_sha256'])!=set(declared):raise AssertionError('Replay manifest incomplete')
    left,right=normalize_manifest(a),normalize_manifest(b);rows={};normalized=[]
    for name,digest in sorted(declared.items()):
        x,y=original/relative(name),replay/relative(name);xh,yh=sha(x),sha(y)
        if xh!=digest or b['outputs_sha256'][name]!=yh:raise AssertionError('Original or replay artifact hash differs from own manifest')
        row=dict(original_sha256=xh,replay_sha256=yh,container_bytes_equal=xh==yh)
        if name.endswith('.gz'):
            u,v=x.read_bytes(),y.read_bytes();text_equal=gzip.decompress(u)==gzip.decompress(v)
            mtime_only=(len(u)>=10 and len(v)>=10 and u[:4]+u[8:]==v[:4]+v[8:])
            row.update(decompressed_text_bytes_equal=text_equal,gzip_equal_except_mtime=mtime_only)
            if not text_equal or not mtime_only:raise AssertionError('Gzip text/body differs beyond allowed header mtime: '+name)
            if xh!=yh:
                left['outputs_sha256'][name]=right['outputs_sha256'][name]='VERIFIED_GZIP_MTIME_ONLY'
                normalized.append('outputs_sha256/'+name)
        elif name.endswith('.npz'):
            row['npz']=compare_npz(x,y)
            if xh!=yh:raise AssertionError('NPZ container hash change is outside allowed manifest normalization')
        elif xh!=yh:raise AssertionError('Non-gzip artifact bytes differ: '+name)
        rows[name]=row
    if left!=right:raise AssertionError('Root manifest differs outside UTC and verified gzip-mtime hashes')
    return dict(status='passed',files=rows,artifacts=len(rows),normalized_root_manifest_fields=['started_utc','completed_utc',*normalized],
        root_manifest_equal_after_declared_normalization=True,original_manifest=a,replay_manifest=b)


def run(root=ROOT):
    root=Path(root).resolve();output=root/OUT
    if output.exists():raise RuntimeError('Never overwrite any diagnostic replay attempt')
    snapshot,profiles=preflight(root);output.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_isolated_copy',started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/PLAN),
        source_freeze_sha256=sha(root/FREEZE),activation_sha256=sha(root/ACTIVATION),sources_and_original_artifacts_at_start=snapshot,
        new_main_model_runs=0,new_fits=0,new_selections=0,raw_market_parser_calls=0,final_market_values_parsed=False,
        scope='Same-source same-saved-input postprocessing reproducibility only; no new independent market validation',profiles=[])
    writej(output/'manifest.json',state);work=None
    try:
        python=(root/'.venv/bin/python').absolute()  # Preserve the venv symlink; never resolve it to global Python.
        if not python.is_file():raise RuntimeError('Original absolute venv Python required')
        for item in profiles:
            p=item['profile'];folder=output/p['profile_id'];folder.mkdir();(output/'workspaces').mkdir(exist_ok=True)
            work=Path(tempfile.mkdtemp(prefix=p['profile_id']+'-',dir=output/'workspaces'))
            writej(folder/'copied_dependencies.json',item['dependencies']);writej(folder/'invocation.json',dict(workspace=str(work),python=str(python),script=p['script'],flag=p['flag'],expected_output_seeded=False))
            copy_dependencies(root,work,item['dependencies'])
            env=dict(os.environ)
            for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
            env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'subprocess.log').open('w') as log:
                completed=subprocess.run([str(python),'-I','-B','-c',BOOTSTRAP,str(work),p['script'],p['flag']],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if completed.returncode:raise RuntimeError('Isolated child failed: '+p['profile_id'])
            shutil.copyfile(work/'__replay_runtime.json',folder/'runtime.json')
            comparison=compare_outputs(root/p['original_output'],work/p['original_output'],item['declared_outputs'])
            writej(folder/'comparison.json',comparison);current(work,item['dependencies'])
            state['profiles'].append(dict(profile_id=p['profile_id'],status='passed',comparison_sha256=sha(folder/'comparison.json'),artifacts=comparison['artifacts'],workspace_removed_after_pass=True))
            shutil.rmtree(work);work=None;writej(output/'manifest.json',state)
        current(root,snapshot)
        if sha(root/PLAN)!=state['protocol_sha256'] or sha(root/FREEZE)!=state['source_freeze_sha256'] or sha(root/ACTIVATION)!=state['activation_sha256']:raise RuntimeError('Replay activation changed')
        state.update(status='passed',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),sources_and_original_artifacts_at_end=snapshot,
            outputs_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'})
        writej(output/'manifest.json',state)
    except BaseException as exc:
        (output/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False,failed_workspace=str(work) if work else None)
        writej(output/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print('Source-only. Separate frozen activation required; no replay performed.')
