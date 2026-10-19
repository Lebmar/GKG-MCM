"""Repeat the fixed, previously exposed final tables; default constructed data.

This additive entrypoint calls the frozen producer's pure helpers and unchanged
kernels. It never calls producer.run, fits a model, selects a candidate, or
replaces the historical producer's original evidence gates.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import submission_adapter as compact
import run_mitigation_locked_final_evaluation as producer

FINAL_BUNDLE=Path('submission_final_bundle_manifest.json')
EXPECTED=Path('submission/expected_nonblind_final.json')
EXPECTED_RELEASE=Path('submission/expected_nonblind_final_release.json')
PROTOCOL=Path('docs/compact_submission_final_protocol.json')
TRAIN=producer.TRAIN
ACTUAL=producer.ACTUAL
OUTPUTS={'traces.npz','Q2_paths.csv.gz','Q1_paths.csv.gz','Q1_thirds_references.csv.gz',
    'summary.csv.gz','paired_paths.csv.gz','object_specification.json','trace_groups.json',
    'initial_witness.json','comparisons.json','severe_example.json'}
REQUIRED_RUNTIME=compact.REQUIRED_RUNTIME|{'scripts/submission_final_adapter.py',
    'scripts/run_mitigation_locked_final_evaluation.py'}


def verify_module_origins(root):
    compact.verify_loaded_modules(root)
    for module,name in ((compact,'scripts/submission_adapter.py'),(producer,'scripts/run_mitigation_locked_final_evaluation.py')):
        if Path(module.__file__).resolve()!=(Path(root)/name).resolve():raise RuntimeError('Frozen helpers must load from this extracted package')


def authorize(root,spec):
    """All metadata/hash gates run before either real market loader call."""
    root=Path(root)
    registry=compact.readj(root/compact.REGISTRY)
    common,_=compact.verify_compact_manifest(root,compact.specification(registry))
    meta=compact.readj(root/FINAL_BUNDLE);release=compact.readj(root/EXPECTED_RELEASE)
    required=REQUIRED_RUNTIME|{str(FINAL_BUNDLE),str(EXPECTED),str(EXPECTED_RELEASE),str(PROTOCOL)}
    mapping=common['files_sha256'];payload_map={k:v for k,v in mapping.items() if k!=str(FINAL_BUNDLE)}
    if (not required<=set(mapping) or meta.get('version')!='compact-fixed-nonblind-final-addon-v1'
            or meta.get('status')!='complete' or meta.get('explicit_reproduction_ready') is not True
            or meta.get('previously_exposed_nonblind') is not True or meta.get('new_independent_validation') is not False
            or meta.get('object_specification')!=spec or meta.get('files_sha256')!=payload_map
            or meta.get('expected_release_sha256')!=compact.sha(root/EXPECTED_RELEASE)):
        raise RuntimeError('Exact completed compact final addon manifest required')
    if (release.get('version')!='compact-fixed-nonblind-final-expectation-release-v1'
            or release.get('reproduction_ready') is not True or release.get('previously_exposed_nonblind') is not True
            or release.get('new_independent_validation') is not False or release.get('new_fits')!=0 or release.get('new_selections')!=0
            or release.get('training_ids')!=TRAIN or release.get('evaluation_ids')!=ACTUAL
            or release.get('object_specification_sha256')!=compact.canonical_sha(spec)
            or release.get('registry_sha256')!=compact.sha(root/compact.REGISTRY)
            or release.get('protocol_sha256')!=compact.sha(root/PROTOCOL)
            or release.get('expected_sha256')!=compact.sha(root/EXPECTED)
            or not REQUIRED_RUNTIME<=set(release.get('source_sha256',{}))
            or any(mapping.get(k)!=v for k,v in release['source_sha256'].items())):
        raise RuntimeError('Separate fixed-source/spec/canonical expectation release required')
    expected=compact.readj(root/EXPECTED)
    if (expected.get('version')!='compact-fixed-nonblind-final-expectations-v1'
            or expected.get('object_specification')!=spec or expected.get('previously_exposed_nonblind') is not True
            or set(expected.get('output_bytes_sha256',{}))!=OUTPUTS-{'traces.npz'}):
        raise RuntimeError('All fixed final output expectations required')
    verify_module_origins(root)
    return common,expected


def produce(spec,actual,actual_pi,training,training_pi,output):
    """Same pure helpers and output ordering as the immutable producer."""
    output=Path(output)
    results,witness=producer.evaluate_objects(spec,actual,actual_pi,training,training_pi,
        engine=compact.simulate_state_budget,paired=compact.simulate_paired)
    rows=[];groups=[];arrays={}
    for obj in spec['objects']:
        result=results[obj['object_id']];rows+=producer.path_rows(obj,result,ACTUAL,metrics=compact.path_metrics)
        names=[]
        if result['arm'] is not None:
            for key,value in result['arm'].items():
                if isinstance(value,np.ndarray):arrays[obj['object_id']+'__'+key]=value;names.append(key)
        groups.append(dict(**obj,status=result['status'],financial_available=result['financial_available'],
            scenario_ids=ACTUAL,training_ids=TRAIN if obj['rule']=='mitigation' else None,array_names=names,omitted_diagnostic_arrays=[]))
    q2=[r for r in rows if r['question']=='Q2'];q1=[r for r in rows if r['question']=='Q1'];refs=[r for r in rows if r['question']=='Q1_reference']
    summaries=[producer.summary([r for r in rows if r['object_id']==obj['object_id']]) for obj in spec['objects']]
    comparisons=[];pairs=[]
    menus=[('mitigation_main',x) for x in ('same_target_B','same_target_annual','exact_thirds_annual')]
    menus += [('Q1_'+name,'Q1_'+name+'_thirds') for name,_,_,_ in producer.PROFILES]
    for model,reference in menus:
        value,paired=producer.compare([r for r in rows if r['object_id']==model],[r for r in rows if r['object_id']==reference])
        comparisons.append(dict(model_object=model,reference_object=reference,**value))
        pairs.extend(dict(model_object=model,reference_object=reference,**r) for r in paired)
    if (len(q2),len(q1),len(refs),len(summaries),len(comparisons),len(groups))!=(240,300,300,14,8,14):
        raise RuntimeError('Complete14-object/840-row inventory required')
    for filename,data in [('Q2_paths.csv.gz',q2),('Q1_paths.csv.gz',q1),('Q1_thirds_references.csv.gz',refs),('summary.csv.gz',summaries),('paired_paths.csv.gz',pairs)]:
        producer.write_csv(output/filename,data)
    for filename,value in [('object_specification.json',spec),('trace_groups.json',groups),('initial_witness.json',witness),
            ('comparisons.json',comparisons),('severe_example.json',producer.severe_example([r for r in q2 if r['object_id']=='mitigation_main'],results))]:
        producer.writej(output/filename,value)
    np.savez_compressed(output/'traces.npz',**arrays)
    return dict(array_signatures={key:compact.array_signature(value) for key,value in sorted(arrays.items())},
        output_bytes_sha256={name:compact.sha(output/name) for name in sorted(OUTPUTS-{'traces.npz'})},
        rows=dict(Q2=240,Q1=300,Q1_reference=300,total=840,summary=14,comparisons=8,trace_groups=14),
        financial_objects=sum(r['financial_available'] for r in results.values()))


def compare_expected(proof,expected):
    for name in ('array_signatures','output_bytes_sha256','rows','financial_objects'):
        if proof[name]!=expected[name]:raise AssertionError('Canonical fixed-final mismatch: '+name)


def run(*,reproduce_fixed_nonblind_final=False,output=None,root=ROOT):
    root=Path(root).resolve();mode='fixed_nonblind_final' if reproduce_fixed_nonblind_final else 'constructed_synthetic'
    output=Path(output or root/f'submission_final_reproduction_{mode}_v1').resolve()
    if output.exists():raise RuntimeError('Never overwrite any final-adapter reproduction attempt')
    spec=producer.object_specification(compact.readj(root/compact.REGISTRY))
    common=expected=None;before=None
    if reproduce_fixed_nonblind_final:
        common,expected=authorize(root,spec)
        before={str(compact.BUNDLE):compact.sha(root/compact.BUNDLE),str(FINAL_BUNDLE):compact.sha(root/FINAL_BUNDLE)}
    output.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_market_data',mode=mode,started_utc=datetime.now(timezone.utc).isoformat(),
        market_is_constructed=not reproduce_fixed_nonblind_final,raw_market_values_parsed=False,final_scenarios_loaded=False,
        training_ids=TRAIN,evaluation_ids=ACTUAL,new_fits=0,new_selections=0,post_final_parameter_changes=False,
        previously_exposed_nonblind=bool(reproduce_fixed_nonblind_final),new_independent_validation=False,
        original_producer_run_called=False,historical_evidence_gates_replaced=False)
    compact.writej(output/'manifest.json',state)
    try:
        if reproduce_fixed_nonblind_final:
            state.update(phase='explicit_guarded_market_load',raw_market_values_parsed=True);compact.writej(output/'manifest.json',state)
            train,train_pi,train_ids=compact.load_paths(root/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
            state.update(phase='explicit_final_market_load',final_scenarios_loaded=True);compact.writej(output/'manifest.json',state)
            actual,actual_pi,ids=compact.load_paths(root/'data/SafeInvest_Synthetic_Data.csv',range(121,181),allow_final=True)
            if train_ids.tolist()!=TRAIN or ids.tolist()!=ACTUAL:raise RuntimeError('Exact original training/final identity required')
        else:
            train=np.zeros((120,30,3));train_pi=np.zeros((120,30));actual=np.zeros((60,30,3));actual_pi=np.zeros((60,30))
        proof=produce(spec,actual,actual_pi,train,train_pi,output)
        if reproduce_fixed_nonblind_final:
            compare_expected(proof,expected);compact.verify_files(root,common['files_sha256']);compact.verify_files(root,before)
        compact.writej(output/'reproduction_checks.json',proof)
        state.update(status='passed',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            canonical_final_compared=bool(reproduce_fixed_nonblind_final),source_bundle_manifest_sha256=(before or {}).get(str(compact.BUNDLE)),
            final_addon_manifest_sha256=(before or {}).get(str(FINAL_BUNDLE)),array_fields=len(proof['array_signatures']),
            **proof['rows'],outputs_sha256={name:compact.sha(output/name) for name in sorted(OUTPUTS|{'reproduction_checks.json'})})
        compact.writej(output/'manifest.json',state)
    except BaseException as exc:
        (output/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        compact.writej(output/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduce-fixed-nonblind-final',action='store_true')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    run(reproduce_fixed_nonblind_final=args.reproduce_fixed_nonblind_final,output=args.output)
