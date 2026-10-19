"""Prepare canonical expectations, then a separately activated compact ZIP.

This additive utility never fits/selects a policy or reads final return values.
Historical runners, sources and numerical archives remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import submission_adapter as adapter

PLAN=Path('docs/compact_submission_protocol.json')
FREEZE=Path('docs/compact_submission_source_freeze.json')
BUILD_RELEASE=Path('docs/compact_submission_build_release.json')
PREP=Path('results/compact_submission_preparation_v1')
BUILD=Path('results/compact_submission_build_v1')
CANON=Path('results/state_budget_mitigation_v1')
ZIP=Path('dist/Team87_Lebmar_Technical_Source_v1.zip')
MAX_ZIP=50_000_000
MAX_SOURCE=45_000_000
TIMESTAMP=(1980,1,1,0,0,0)
EXTRAS=('SUBMISSION_README.md','submission/AI_USE.md','submission/LICENSE-DATA-CC-BY-NC-SA-4.0.txt',
    'requirements-research-pinned.txt','data/SafeInvest_Synthetic_Data.csv')
REQUIRED_INPUTS={str(adapter.REGISTRY),str(PLAN),'docs/DATA_NOTICE.md',*EXTRAS}
SOURCES=adapter.REQUIRED_RUNTIME|{'scripts/prepare_compact_submission.py','tests/test_submission_adapter.py','tests/test_compact_submission.py'}
SECRET=re.compile(rb'(?:sk-[A-Za-z0-9_-]{35,}|ghp_[A-Za-z0-9]{36,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def sha(path):return adapter.sha(path)
def readj(path):return adapter.readj(path)
def writej(path,value):adapter.writej(path,value)


def canonical_closure(root=ROOT):
    """Metadata/byte hashes only; canonical NPZ/financial CSV not parsed."""
    root=Path(root);m=readj(root/CANON/'manifest.json');audit=readj(root/CANON/'independent_audit.json')
    before=m['sources_and_inputs_at_start'];bound=audit['sources_and_artifacts_at_start']
    if (m['status']!='complete' or m['final_scenarios_loaded'] is not False or m.get('new_final_reads')!=0
            or before!=m['sources_and_inputs_at_end'] or set(m['outputs_sha256'])!={'traces.npz','trace_groups.json','summary.csv.gz','path_metrics.csv.gz','selections.json','initial_witnesses.json'}
            or audit['status']!='passed'
            or audit['parent_manifest_sha256']!=sha(root/CANON/'manifest.json')
            or audit['final_scenarios_loaded'] is not False or bound!=audit['sources_and_artifacts_at_end']):
        raise RuntimeError('Unchanged completed audited canonical development evidence required')
    files={str(CANON/name):digest for name,digest in m['outputs_sha256'].items()}
    adapter.verify_files(root,before);adapter.verify_files(root,bound);adapter.verify_files(root,files)
    for name,digest in {**files,str(CANON/'manifest.json'):sha(root/CANON/'manifest.json')}.items():
        if bound.get(name)!=digest:raise RuntimeError('Canonical independent audit binding missing')
    return {**before,**bound,**files,str(CANON/'manifest.json'):sha(root/CANON/'manifest.json'),
        str(CANON/'independent_audit.json'):sha(root/CANON/'independent_audit.json')}


def preflight(root=ROOT):
    root=Path(root);p,f=readj(root/PLAN),readj(root/FREEZE)
    if (p.get('version')!='compact-submission-preparation-v1' or p.get('execution_ready') is not True
            or f.get('execution_ready') is not True or f.get('protocol_sha256')!=sha(root/PLAN)
            or p.get('allow_final') is not False or p.get('development_ids')!=list(range(1,121))
            or p.get('maximum_archive_bytes')!=MAX_ZIP or p.get('maximum_source_bytes')!=MAX_SOURCE
            or not SOURCES<=set(f.get('source_sha256',{})) or not REQUIRED_INPUTS<=set(f.get('input_sha256',{}))):raise RuntimeError('Freeze exact compact preparation sources/protocol before canonical numeric parse')
    adapter.verify_files(root,f['source_sha256']);adapter.verify_files(root,f['input_sha256'])
    frozen={**f['source_sha256'],**f['input_sha256']};closure=canonical_closure(root)
    if any(frozen.get(name)!=digest for name,digest in closure.items()):raise RuntimeError('Complete audited canonical transitive closure required')
    spec=adapter.specification(readj(root/adapter.REGISTRY))
    if p.get('object_specification')!=spec:raise RuntimeError('Fixed compact object specification changed')
    return p,f,frozen,spec


def prepare_expected(root=ROOT):
    root=Path(root)
    if (root/PREP).exists():raise RuntimeError('Never overwrite any expectation-preparation attempt')
    p,f,before,spec=preflight(root);(root/PREP).mkdir(parents=True,exist_ok=False)
    manifest=dict(status='incomplete',phase='before_canonical_numeric_parse',protocol_sha256=sha(root/PLAN),
        source_freeze_sha256=sha(root/FREEZE),sources_and_inputs_at_start=before,final_scenarios_loaded=False,new_policy_runs=0)
    writej(root/PREP/'manifest.json',manifest)
    try:
        groups=readj(root/CANON/'trace_groups.json')
        selected=[g for g in groups if g['kind']=='full_selected']
        if len(selected)!=1:raise RuntimeError('Exactly one canonical fullselected group required')
        main=selected[0]
        if (main.get('status')!='funded' or main['scenario_ids']!=list(range(1,121)) or main['training_ids']!=list(range(1,121))
                or main['candidate']['candidate_id']!=spec['main']['selected_candidate_id'] or main['target']!=spec['main']['target']):
            raise RuntimeError('Canonical fullselected is not the locked full120 model')
        chosen={'mitigation_main':main}
        for obj in adapter.OBJECTS[1:]:
            matching=[g for g in groups if g['kind']=='reference' and g['job_id']==main['job_id'] and g['policy']==obj]
            if len(matching)!=1 or matching[0]['scenario_ids']!=list(range(1,121)):raise RuntimeError('All canonical full reference groups required')
            chosen[obj]=matching[0]
        fingerprints={}
        with np.load(root/CANON/'traces.npz',allow_pickle=False) as archive:
            for obj,g in chosen.items():
                if not g['array_names']:raise RuntimeError('A fixed canonical object has no financial arrays')
                fingerprints[obj]={key:adapter.array_signature(archive[g['group_id']+'__'+key]) for key in sorted(g['array_names'])}
        table=pd.read_csv(root/CANON/'summary.csv.gz',float_precision='round_trip');summaries={}
        for obj in adapter.OBJECTS:
            scope,policy=('full_selected','state_budget_learner') if obj=='mitigation_main' else ('full_training_reference',obj)
            row=table[(table.scope==scope)&(table.policy==policy)]
            if len(row)!=1:raise RuntimeError('Exactly one canonical full summary per fixed object')
            summaries[obj]={key:adapter.clean(value) for key,value in row.iloc[0].to_dict().items() if key not in ('scope','policy')}
        expected=dict(version='compact-canonical-development-expectations-v1',object_specification=spec,
            fingerprint_convention='C order, little-endian dtype, canonical quietNaN, signedzero retained; dtype/shape separately bound',
            array_signatures=fingerprints,summaries=summaries,
            source_evidence=dict(canonical_manifest_sha256=sha(root/CANON/'manifest.json'),canonical_audit_sha256=sha(root/CANON/'independent_audit.json'),
                groups={obj:g['group_id'] for obj,g in chosen.items()}),no_refit=True,no_selection=True,final_access=False)
        writej(root/PREP/'expected_development.json',expected)
        after={name:sha(root/name) for name in before}
        if after!=before or sha(root/PLAN)!=manifest['protocol_sha256'] or sha(root/FREEZE)!=manifest['source_freeze_sha256']:
            raise RuntimeError('Original source/input/protocol/freeze changed during expectation preparation')
        manifest.update(status='complete',phase='complete',sources_and_inputs_at_end=after,
            outputs_sha256={'expected_development.json':sha(root/PREP/'expected_development.json')},
            object_count=4,array_fields=sum(len(v) for v in fingerprints.values()),new_fits=0,new_selections=0)
        writej(root/PREP/'manifest.json',manifest)
    except BaseException as exc:
        (root/PREP/'failure.txt').write_text(traceback.format_exc());manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(root/PREP/'manifest.json',manifest);raise


def checked_file(root,name):
    name=adapter.safe_name(name);path=root/name
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):raise RuntimeError('Package file missing/outside root')
    cursor=path
    while cursor!=root:
        if cursor.is_symlink():raise RuntimeError('No symlink package input: '+str(name))
        cursor=cursor.parent
    data=path.read_bytes()
    if SECRET.search(data):raise RuntimeError('Credential/private-key pattern found in candidate package file: '+str(name))
    return path


def packing_inventory(root=ROOT):
    root=Path(root).resolve();paths={}
    for folder in ('src','research','scripts','tests'):
        for path in sorted((root/folder).rglob('*.py')):
            name=path.relative_to(root).as_posix();paths[name]=checked_file(root,name)
    for path in sorted((root/'docs').rglob('*')):
        if path.suffix not in ('.json','.md') or not path.is_file():continue
        if path.name.lower() in ('secrets.json','credentials.json') or any(v.lower() in ('private','secrets') for v in path.relative_to(root).parts):continue
        name=path.relative_to(root).as_posix();paths[name]=checked_file(root,name)
    for name in EXTRAS:paths[name]=checked_file(root,name)
    if str(adapter.EXPECTED) in paths:raise RuntimeError('Duplicate virtual expectation member')
    paths[str(adapter.EXPECTED)]=checked_file(root,str(PREP/'expected_development.json'))
    if sum(path.stat().st_size for path in paths.values())>=MAX_SOURCE:raise RuntimeError('Source allowlist exceeds predeclared45MB uncompressed budget')
    return paths,{name:sha(path) for name,path in paths.items()}


def write_zip(path,payloads,*,maximum_bytes=MAX_ZIP):
    names=list(payloads)
    if len(names)!=len(set(names)):raise RuntimeError('Duplicate ZIP names')
    for name in names:adapter.safe_name(name)
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in sorted(payloads):
            info=zipfile.ZipInfo(name,TIMESTAMP);info.create_system=3;info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,payloads[name],compresslevel=9)
    if Path(path).stat().st_size>=maximum_bytes:raise RuntimeError('Archive must be strictly smaller than50,000,000 bytes')
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist())!=sorted(names):raise RuntimeError('ZIP member inventory changed')
        for info in archive.infolist():
            adapter.safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr>>16) or info.date_time!=TIMESTAMP:raise RuntimeError('ZIP link/timestamp rejected')
            if archive.read(info.filename)!=payloads[info.filename]:raise RuntimeError('ZIP payload mismatch')


def extract_verified_zip(path,destination):
    destination=Path(destination)
    if destination.exists():raise RuntimeError('Fresh extraction directory required')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in archive.infolist())>=MAX_SOURCE+1_000_000:raise RuntimeError('Duplicate/oversize extracted inventory')
        for info in archive.infolist():
            adapter.safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr>>16) or info.is_dir():raise RuntimeError('Only regular allowlisted ZIP members')
        manifest=json.loads(archive.read(str(adapter.BUNDLE)))
        if set(names)!=set(manifest['files_sha256'])|{str(adapter.BUNDLE)}:raise RuntimeError('Extraction manifest inventory differs')
        for name,digest in manifest['files_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise RuntimeError('Extracted file digest differs')
        destination.mkdir(parents=True,exist_ok=False)
        for name in names:
            target=destination/adapter.safe_name(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
    return destination


def build(root=ROOT):
    root=Path(root).resolve()
    if (root/BUILD).exists() or (root/ZIP).exists():raise RuntimeError('Never overwrite build attempt or existing ZIP')
    _,_,_,spec=preflight(root);prep=readj(root/PREP/'manifest.json');release=readj(root/BUILD_RELEASE)
    if (prep['status']!='complete' or prep['protocol_sha256']!=sha(root/PLAN) or prep['source_freeze_sha256']!=sha(root/FREEZE)
            or prep['sources_and_inputs_at_start']!=prep['sources_and_inputs_at_end']
            or prep['outputs_sha256']['expected_development.json']!=sha(root/PREP/'expected_development.json')
            or release.get('version')!='compact-submission-build-release-v1' or release.get('build_ready') is not True
            or release.get('protocol_sha256')!=sha(root/PLAN) or release.get('source_freeze_sha256')!=sha(root/FREEZE)
            or release.get('preparation_manifest_sha256')!=sha(root/PREP/'manifest.json')):
        raise RuntimeError('Separate current engineering build activation required')
    adapter.verify_files(root,prep['sources_and_inputs_at_end'])
    paths,mapping=packing_inventory(root)
    # The release binds every other member; its own hash is subsequently
    # included in the bundle manifest, avoiding a self-referential digest.
    without_release={k:v for k,v in mapping.items() if k!=str(BUILD_RELEASE)}
    if release.get('allowlist_without_release_sha256')!=adapter.canonical_sha(without_release):raise RuntimeError('Exact source-package allowlist changed')
    (root/BUILD).mkdir(parents=True,exist_ok=False);state=dict(status='incomplete',phase='before_zip',protocol_sha256=sha(root/PLAN),
        source_freeze_sha256=sha(root/FREEZE),build_release_sha256=sha(root/BUILD_RELEASE),new_policy_runs=0,final_scenarios_loaded=False)
    writej(root/BUILD/'manifest.json',state)
    try:
        bundle=dict(version='compact-technical-source-bundle-v1',status='complete',development_ready=True,final_access=False,
            object_specification=spec,files_sha256=mapping,preparation_manifest_sha256=sha(root/PREP/'manifest.json'),
            interpretation='Fixed source/model reproduction only; historical full engineering gates require omitted large archives.')
        payloads={name:path.read_bytes() for name,path in paths.items()}
        payloads[str(adapter.BUNDLE)]=(json.dumps(bundle,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        (root/ZIP).parent.mkdir(parents=True,exist_ok=True);write_zip(root/ZIP,payloads)
        if any(sha(paths[name])!=digest for name,digest in mapping.items()):raise RuntimeError('Package source changed during build')
        state.update(status='complete',phase='complete',archive=str(ZIP),archive_sha256=sha(root/ZIP),archive_bytes=(root/ZIP).stat().st_size,
            member_count=len(payloads),uncompressed_bytes=sum(len(v) for v in payloads.values()),development_reproduction_executed=False)
        writej(root/BUILD/'manifest.json',state)
    except BaseException as exc:
        (root/BUILD/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(root/BUILD/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group()
    g.add_argument('--prepare-expectations',action='store_true');g.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.prepare_expectations:prepare_expected()
    elif args.build:build()
    else:print(json.dumps(dict(source_only=True,packaged=False,market_values_parsed=False)))
