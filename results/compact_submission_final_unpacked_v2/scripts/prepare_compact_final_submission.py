"""Add fixed nonblind-final reproduction to a fresh V2 technical ZIP.

Preparation extracts existing audited output expectations; it does not execute
any strategy. Building is separately activated and never modifies the V1 ZIP.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_compact_submission as v1
import submission_final_adapter as addon

compact=addon.compact
PLAN=addon.PROTOCOL
FREEZE=Path('docs/compact_submission_final_source_freeze.json')
BUILD_RELEASE=Path('docs/compact_submission_final_build_release.json')
PREP=Path('results/compact_submission_final_preparation_v1')
BUILD=Path('results/compact_submission_final_build_v1')
CANON=Path('results/mitigation_locked_final_evaluation_v1')
REPLAY=Path('results/mitigation_locked_final_reproduction_v1')
ZIP=Path('dist/Team87_Lebmar_Technical_Source_v2.zip')
EXTRA='submission/FINAL_REPRODUCTION.md'
SOURCES=addon.REQUIRED_RUNTIME|{'scripts/prepare_compact_submission.py','scripts/prepare_compact_final_submission.py',
    'tests/test_submission_final_adapter.py','tests/test_compact_final_submission.py'}
REQUIRED_INPUTS={str(PLAN),str(compact.REGISTRY),'docs/DATA_NOTICE.md',EXTRA,
    str(v1.PREP/'manifest.json'),str(v1.PREP/'expected_development.json')}


def canonical_closure(root=ROOT):
    """Read metadata and byte hashes, never canonical arrays or financial rows."""
    root=Path(root);m=compact.readj(root/CANON/'manifest.json');a=compact.readj(root/CANON/'independent_audit.json')
    replay=compact.readj(root/REPLAY/'manifest.json')
    before=m['sources_and_inputs_at_start'];audit=a['sources_and_artifacts_at_start'];r=replay['sources_and_original_artifacts_at_start']
    if (m.get('status')!='complete' or before!=m['sources_and_inputs_at_end'] or set(m['outputs_sha256'])!=addon.OUTPUTS
            or m.get('scope')!='previously_exposed_nonblind' or m.get('new_fits')!=0 or m.get('new_selections')!=0
            or m.get('training_ids_loaded')!=addon.TRAIN or m.get('final_ids_loaded')!=addon.ACTUAL
            or a.get('status')!='passed' or audit!=a['sources_and_artifacts_at_end']
            or a.get('producer_manifest_sha256')!=compact.sha(root/CANON/'manifest.json')
            or replay.get('status')!='passed' or r!=replay['sources_and_original_artifacts_at_end']):
        raise RuntimeError('Completed frozen final producer, independent audit and isolated replay required')
    outputs={str(CANON/name):digest for name,digest in m['outputs_sha256'].items()}
    originals={**outputs,str(CANON/'manifest.json'):compact.sha(root/CANON/'manifest.json')}
    for name,digest in originals.items():
        if audit.get(name)!=digest or r.get(name)!=digest:raise RuntimeError('Current complete original output binding required')
    replay_outputs={str(REPLAY/name):digest for name,digest in replay['outputs_sha256'].items()}
    snapshot={}
    for mapping in (before,audit,r,originals,replay_outputs,
            {str(CANON/'independent_audit.json'):compact.sha(root/CANON/'independent_audit.json'),str(REPLAY/'manifest.json'):compact.sha(root/REPLAY/'manifest.json')}):
        addon.producer.merge(snapshot,mapping)
    compact.verify_files(root,snapshot)
    return snapshot


def preflight(root=ROOT):
    root=Path(root);p=compact.readj(root/PLAN);f=compact.readj(root/FREEZE)
    if (p.get('version')!='compact-fixed-nonblind-final-preparation-v1' or p.get('execution_ready') is not True
            or f.get('execution_ready') is not True or f.get('protocol_sha256')!=compact.sha(root/PLAN)
            or p.get('training_ids')!=addon.TRAIN or p.get('evaluation_ids')!=addon.ACTUAL
            or p.get('previously_exposed_nonblind') is not True or p.get('new_independent_validation') is not False
            or not SOURCES<=set(f.get('source_sha256',{})) or not REQUIRED_INPUTS<=set(f.get('input_sha256',{}))):
        raise RuntimeError('Freeze exact additive compact-final preparation before canonical numerical parse')
    snapshot={};addon.producer.merge(snapshot,f['source_sha256']);addon.producer.merge(snapshot,f['input_sha256'])
    compact.verify_files(root,snapshot)
    closure=canonical_closure(root)
    if any(snapshot.get(k)!=v for k,v in closure.items()):raise RuntimeError('Full passed canonical/audit/replay closure must be frozen')
    spec=addon.producer.object_specification(compact.readj(root/compact.REGISTRY))
    if p.get('object_specification')!=spec:raise RuntimeError('All14 frozen final objects required')
    dev=compact.readj(root/v1.PREP/'manifest.json')
    if dev.get('status')!='complete' or dev['outputs_sha256'].get('expected_development.json')!=compact.sha(root/v1.PREP/'expected_development.json'):
        raise RuntimeError('Keep existing completed V1 development proof unchanged')
    return spec,snapshot


def prepare_expected(root=ROOT):
    root=Path(root)
    if (root/PREP).exists():raise RuntimeError('Never overwrite any compact-final preparation attempt')
    spec,snapshot=preflight(root);(root/PREP).mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_canonical_numeric_parse',protocol_sha256=compact.sha(root/PLAN),
        source_freeze_sha256=compact.sha(root/FREEZE),sources_and_inputs_at_start=snapshot,
        new_policy_runs=0,new_fits=0,new_selections=0,previously_exposed_nonblind=True,new_independent_validation=False)
    compact.writej(root/PREP/'manifest.json',state)
    try:
        groups=compact.readj(root/CANON/'trace_groups.json');m=compact.readj(root/CANON/'manifest.json')
        if [g['object_id'] for g in groups]!=[o['object_id'] for o in spec['objects']]:raise RuntimeError('Complete original14 groups in fixed order required')
        names=[]
        for obj,g in zip(spec['objects'],groups):
            if any(g.get(k)!=v for k,v in obj.items()) or g['scenario_ids']!=addon.ACTUAL:raise RuntimeError('Canonical ownH/spec/identity differs')
            if obj['rule']=='Review' and (g['financial_available'] or g['array_names']):raise RuntimeError('Q1 A/D Review cannot have financial arrays')
            if obj['rule']!='Review' and (not g['financial_available'] or not g['array_names']):raise RuntimeError('All12 original funded objects must retain every array')
            names.extend(g['object_id']+'__'+key for key in g['array_names'])
        with np.load(root/CANON/'traces.npz',allow_pickle=False) as arrays:
            if set(arrays.files)!=set(names) or len(names)!=len(set(names)):raise RuntimeError('Every array from every financial object required')
            fingerprints={key:compact.array_signature(arrays[key]) for key in sorted(names)}
        expected=dict(version='compact-fixed-nonblind-final-expectations-v1',object_specification=spec,
            previously_exposed_nonblind=True,new_independent_validation=False,
            array_signatures=fingerprints,output_bytes_sha256={name:m['outputs_sha256'][name] for name in sorted(addon.OUTPUTS-{'traces.npz'})},
            rows=dict(Q2=240,Q1=300,Q1_reference=300,total=840,summary=14,comparisons=8,trace_groups=14),
            financial_objects=sum(g['financial_available'] for g in groups),
            canonical_manifest_sha256=compact.sha(root/CANON/'manifest.json'),canonical_independent_audit_sha256=compact.sha(root/CANON/'independent_audit.json'),
            isolated_replay_manifest_sha256=compact.sha(root/REPLAY/'manifest.json'))
        for field,count in [('Q2_rows',240),('Q1_rows',300),('Q1_reference_rows',300),('summary_rows',14),('comparisons',8),('trace_groups',14)]:
            if m.get(field)!=count:raise RuntimeError('Canonical complete inventory mismatch')
        compact.writej(root/PREP/'expected_nonblind_final.json',expected)
        release=dict(version='compact-fixed-nonblind-final-expectation-release-v1',reproduction_ready=True,
            previously_exposed_nonblind=True,new_independent_validation=False,new_fits=0,new_selections=0,
            training_ids=addon.TRAIN,evaluation_ids=addon.ACTUAL,object_specification_sha256=compact.canonical_sha(spec),
            registry_sha256=compact.sha(root/compact.REGISTRY),protocol_sha256=compact.sha(root/PLAN),
            expected_sha256=compact.sha(root/PREP/'expected_nonblind_final.json'),
            source_sha256={name:snapshot[name] for name in sorted(addon.REQUIRED_RUNTIME)},
            interpretation='Explicit fixed-table reproduction of previously exposed final scenarios; no independent additional sample or post-final adjustment.')
        compact.writej(root/PREP/'expected_nonblind_final_release.json',release)
        after={k:compact.sha(root/k) for k in snapshot}
        if after!=snapshot or compact.sha(root/PLAN)!=state['protocol_sha256'] or compact.sha(root/FREEZE)!=state['source_freeze_sha256']:
            raise RuntimeError('Original frozen evidence changed during extraction')
        state.update(status='complete',phase='complete',sources_and_inputs_at_end=after,array_fields=len(fingerprints),
            outputs_sha256={name:compact.sha(root/PREP/name) for name in ('expected_nonblind_final.json','expected_nonblind_final_release.json')})
        compact.writej(root/PREP/'manifest.json',state)
    except BaseException as exc:
        (root/PREP/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        compact.writej(root/PREP/'manifest.json',state);raise


def packing_inventory(root=ROOT):
    root=Path(root).resolve();paths,_=v1.packing_inventory(root)
    paths[EXTRA]=v1.checked_file(root,EXTRA)
    for member,source in [(addon.EXPECTED,PREP/'expected_nonblind_final.json'),(addon.EXPECTED_RELEASE,PREP/'expected_nonblind_final_release.json')]:
        if str(member) in paths:raise RuntimeError('Duplicate final virtual member')
        paths[str(member)]=v1.checked_file(root,str(source))
    if sum(p.stat().st_size for p in paths.values())>=v1.MAX_SOURCE:raise RuntimeError('Compact V2 source exceeds predeclared45MB')
    return paths,{k:compact.sha(p) for k,p in paths.items()}


def build(root=ROOT):
    root=Path(root).resolve()
    if (root/BUILD).exists() or (root/ZIP).exists():raise RuntimeError('Fresh V2 build/archive required; V1 is immutable')
    spec,_=preflight(root);prep=compact.readj(root/PREP/'manifest.json');release=compact.readj(root/BUILD_RELEASE)
    if (prep.get('status')!='complete' or prep.get('protocol_sha256')!=compact.sha(root/PLAN)
            or prep.get('source_freeze_sha256')!=compact.sha(root/FREEZE) or prep['sources_and_inputs_at_start']!=prep['sources_and_inputs_at_end']
            or release.get('version')!='compact-fixed-nonblind-final-build-release-v1' or release.get('build_ready') is not True
            or release.get('protocol_sha256')!=compact.sha(root/PLAN) or release.get('source_freeze_sha256')!=compact.sha(root/FREEZE)
            or release.get('preparation_manifest_sha256')!=compact.sha(root/PREP/'manifest.json')):
        raise RuntimeError('Separate exact completed V2 engineering build activation required')
    compact.verify_files(root,prep['sources_and_inputs_at_end'])
    for name,digest in prep['outputs_sha256'].items():
        if compact.sha(root/PREP/name)!=digest:raise RuntimeError('Prepared final expectation changed')
    paths,mapping=packing_inventory(root)
    if release.get('allowlist_without_release_sha256')!=compact.canonical_sha({k:v for k,v in mapping.items() if k!=str(BUILD_RELEASE)}):
        raise RuntimeError('Exact V2 payload allowlist changed')
    (root/BUILD).mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_zip',protocol_sha256=compact.sha(root/PLAN),source_freeze_sha256=compact.sha(root/FREEZE),
        build_release_sha256=compact.sha(root/BUILD_RELEASE),new_policy_runs=0,actual_final_replay_executed=False)
    compact.writej(root/BUILD/'manifest.json',state)
    try:
        final_meta=dict(version='compact-fixed-nonblind-final-addon-v1',status='complete',explicit_reproduction_ready=True,
            previously_exposed_nonblind=True,new_independent_validation=False,object_specification=spec,files_sha256=mapping,
            expected_release_sha256=mapping[str(addon.EXPECTED_RELEASE)])
        final_bytes=(json.dumps(final_meta,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        final_sha=hashlib.sha256(final_bytes).hexdigest()
        common=dict(version='compact-technical-source-bundle-v1',status='complete',development_ready=True,final_access=False,
            automatic_final_access=False,explicit_nonblind_final_addon=str(addon.FINAL_BUNDLE),
            object_specification=compact.specification(compact.readj(root/compact.REGISTRY)),
            files_sha256={**mapping,str(addon.FINAL_BUNDLE):final_sha},preparation_manifest_sha256=compact.sha(root/v1.PREP/'manifest.json'),
            interpretation='Fixed development and explicitly requested previously exposed final reproduction; old historical gates require omitted large archives.')
        payloads={name:path.read_bytes() for name,path in paths.items()}
        payloads[str(addon.FINAL_BUNDLE)]=final_bytes
        payloads[str(compact.BUNDLE)]=(json.dumps(common,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        (root/ZIP).parent.mkdir(parents=True,exist_ok=True);v1.write_zip(root/ZIP,payloads)
        if any(compact.sha(paths[k])!=digest for k,digest in mapping.items()):raise RuntimeError('V2 input changed during build')
        state.update(status='complete',phase='complete',archive=str(ZIP),archive_sha256=compact.sha(root/ZIP),archive_bytes=(root/ZIP).stat().st_size,
            member_count=len(payloads),uncompressed_bytes=sum(len(v) for v in payloads.values()))
        compact.writej(root/BUILD/'manifest.json',state)
    except BaseException as exc:
        (root/BUILD/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        compact.writej(root/BUILD/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--prepare-expectations',action='store_true');group.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.prepare_expectations:prepare_expected()
    elif args.build:build()
    else:print(json.dumps(dict(source_only=True,packaged=False,canonical_final_values_parsed=False)))
