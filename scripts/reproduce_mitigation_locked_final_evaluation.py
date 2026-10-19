"""Isolated same-source replay of the one released, nonblind V2b final.

Source preparation never reads actual final values. A separately frozen exact
producer manifest/output map must exist before execution. This adds no market
observations, blind validation, policy selection or parameter adjustment.
"""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,tempfile,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import reproduce_technical_rebuild as shared
import run_mitigation_locked_final_evaluation as producer
PLAN=Path('docs/mitigation_locked_final_reproduction_plan.json')
FREEZE=Path('docs/mitigation_locked_final_reproduction_source_freeze.json')
OUT=Path('results/mitigation_locked_final_reproduction_v1')
SCRIPT='scripts/run_mitigation_locked_final_evaluation.py'
OWN={'scripts/reproduce_mitigation_locked_final_evaluation.py','tests/test_mitigation_locked_final_reproduction.py'}
sha,readj,writej=shared.sha,shared.readj,shared.writej


def validate_plan(plan):
    if (plan.get('version')!='mitigation-locked-final-isolated-reproduction-plan-v1' or plan.get('execution_ready') is not False
        or plan.get('training_ids')!=producer.TRAIN or plan.get('evaluation_ids')!=producer.ACTUAL
        or plan.get('producer_source')!=SCRIPT or plan.get('producer_output_directory')!=str(producer.OUT)
        or plan.get('output_directory')!=str(OUT) or plan.get('normalized_root_manifest_fields')!=['started_utc','completed_utc']
        or plan.get('python_executable')!='.venv/bin/python' or plan.get('previously_exposed_nonblind') is not True
        or plan.get('expected_producer_manifest_sha256_source')!='reproduction_source_freeze.expected_producer_manifest_sha256'):
        raise RuntimeError('Separately frozen fixed single-producer replay plan required')


def metadata_inputs(root,plan,expected_producer_manifest_sha256):
    """Only producer release checks and complete artifact byte hashes, no array load."""
    root=Path(root);validate_plan(plan)
    if not isinstance(expected_producer_manifest_sha256,str) or len(expected_producer_manifest_sha256)!=64:
        raise RuntimeError('Root must bind the actual completed producer manifest SHA in replay sourcefreeze')
    definition,spec,dependencies=producer.preflight(root)
    forbidden=producer.OUT
    for name in dependencies:
        path=shared.relative(name)
        if path==forbidden or forbidden in path.parents or path==OUT or OUT in path.parents:
            raise RuntimeError('Never seed own final or replay outputs into fresh dependencies')
    folder=root/forbidden;manifest=readj(folder/'manifest.json')
    if (sha(folder/'manifest.json')!=expected_producer_manifest_sha256 or manifest.get('status')!='complete'
        or manifest.get('scope')!='previously_exposed_nonblind' or manifest.get('post_final_parameter_changes') is not False
        or manifest.get('new_fits')!=0 or manifest.get('new_selections')!=0
        or manifest.get('training_ids_loaded')!=producer.TRAIN or manifest.get('final_ids_loaded')!=producer.ACTUAL
        or manifest.get('sources_and_inputs_at_start')!=dependencies or manifest.get('sources_and_inputs_at_end')!=dependencies
        or manifest.get('protocol_sha256')!=sha(root/producer.PROTOCOL) or manifest.get('source_freeze_sha256')!=sha(root/producer.FREEZE)
        or manifest.get('release_sha256')!=sha(root/producer.RELEASE) or manifest.get('release_freeze_sha256')!=sha(root/producer.RELEASE_FREEZE)
        or manifest.get('object_specification_sha256')!=definition['object_specification_sha256']
        or (manifest.get('Q2_rows'),manifest.get('Q1_rows'),manifest.get('Q1_reference_rows'),manifest.get('summary_rows'),manifest.get('comparisons'),manifest.get('trace_groups'))!=(240,300,300,14,8,14)):
        raise RuntimeError('Complete released unchanged original producer identity required')
    outputs={};shared.merge(outputs,manifest['outputs_sha256'])
    if not outputs or 'manifest.json' in outputs:raise RuntimeError('Complete non-root output map required')
    shared.current(folder,outputs)
    inputs=dict(dependencies)
    shared.merge(inputs,{str(forbidden/'manifest.json'):sha(folder/'manifest.json')})
    shared.merge(inputs,{str(forbidden/name):digest for name,digest in outputs.items()})
    return dict(sorted(inputs.items())),dict(sorted(dependencies.items())),outputs


def authorize(root=ROOT):
    root=Path(root);plan,freeze=readj(root/PLAN),readj(root/FREEZE)
    validate_plan(plan)
    if (freeze.get('version')!='mitigation-locked-final-reproduction-source-freeze-v1' or freeze.get('execution_ready') is not True
        or freeze.get('plan_sha256')!=sha(root/PLAN) or not OWN<=set(freeze.get('source_sha256',{}))
        or freeze['source_sha256'].get('scripts/reproduce_technical_rebuild.py')!=sha(root/'scripts/reproduce_technical_rebuild.py')
        or freeze['source_sha256'].get(SCRIPT)!=sha(root/SCRIPT)):
        raise RuntimeError('Complete independent replay adapter/shared helper/producer source freeze required')
    shared.current(root,freeze['source_sha256']);shared.current(root,freeze['input_sha256'])
    expected,dependencies,outputs=metadata_inputs(root,plan,freeze.get('expected_producer_manifest_sha256'))
    if expected!=freeze['input_sha256']:raise RuntimeError('Freeze every original/dependency/output SHA exactly')
    return plan,freeze,dependencies,outputs


# The copied original producer has no output-override CLI. Use its frozen
# relative output under the fresh source root, without modifying any globals.
BOOTSTRAP=shared.BOOTSTRAP.replace('sys.argv = [str(script), "--execute", "--output", str(target)]','sys.argv = [str(script), "--execute"]')
if BOOTSTRAP==shared.BOOTSTRAP:raise RuntimeError('Expected frozen bootstrap argument contract differs')


def command(python,workspace,receipt,venv):
    python,workspace=Path(python),Path(workspace)
    if not python.is_absolute() or not workspace.is_absolute():raise ValueError('Absolute originalvenv and fresh workspace required')
    return [str(python),'-I','-c',BOOTSTRAP,str(workspace),str(workspace/SCRIPT),str(workspace/producer.OUT),str(receipt),str(venv)]


def run(root=ROOT):
    root=Path(root).absolute();out=root/OUT
    if out.exists():raise RuntimeError('Never overwrite or silently retry a replay attempt')
    out.mkdir(parents=True);workspace=None
    state=dict(status='incomplete',phase='before_replay_gates',started_utc=datetime.now(timezone.utc).isoformat(),scope='previously_exposed_nonblind',independent_additional_scenarios=0,automatic_retry=False)
    writej(out/'manifest.json',state)
    try:
        plan,freeze,dependencies,outputs=authorize(root)
        snapshot={};shared.merge(snapshot,freeze['source_sha256']);shared.merge(snapshot,freeze['input_sha256']);shared.merge(snapshot,{str(PLAN):sha(root/PLAN),str(FREEZE):sha(root/FREEZE)})
        state.update(phase='isolated_replay',plan_sha256=sha(root/PLAN),source_freeze_sha256=sha(root/FREEZE),sources_and_original_artifacts_at_start=snapshot)
        workspace=Path(tempfile.mkdtemp(prefix='gkg-mitigation-locked-final-'));state['workspace']=str(workspace);writej(out/'manifest.json',state)
        shared.copy_dependencies(root,workspace,dependencies,str(producer.OUT));writej(out/'copied_dependencies.json',dependencies)
        invocation=command(root/plan['python_executable'],workspace,out/'runtime.json',root/'.venv')
        writej(out/'invocation.json',dict(argv=invocation,cwd=str(workspace),removed_environment_keys=['PYTHONHOME','PYTHONPATH'],timeout_seconds=600))
        with (out/'producer.log').open('wb') as log:
            child=subprocess.run(invocation,cwd=workspace,env=shared.clean_environment(os.environ),stdout=log,stderr=subprocess.STDOUT,check=False,timeout=600)
        if child.returncode!=0 or readj(out/'runtime.json').get('status')!='passed':raise RuntimeError('Isolated unchanged producer/runtime failed')
        shared.current(root,dependencies);shared.current(workspace,dependencies)
        result=shared.compare_outputs(root/producer.OUT,workspace/producer.OUT,outputs);writej(out/'comparison.json',result)
        if result['status']!='passed':raise RuntimeError('Retain all replay differences')
        shared.current(root,snapshot);shutil.rmtree(workspace);workspace=None
        state.update(status='passed',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),workspace_removed_after_exact_comparison=True,sources_and_original_artifacts_at_end={name:sha(root/name) for name in snapshot},comparison_sha256=sha(out/'comparison.json'),declared_outputs=len(outputs),interpretation='Same released source/data/runtime reproduction only; not new independent final evidence, blind validation or parameter selection.',outputs_sha256={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.name!='manifest.json'})
        writej(out/'manifest.json',state)
    except BaseException as exc:
        (out/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',phase='failed_evidence_retained',error_type=type(exc).__name__,error=str(exc),retained_workspace=None if workspace is None else str(workspace),automatic_retry=False);writej(out/'manifest.json',state);raise
    print(json.dumps(dict(status='passed',manifest_sha256=sha(out/'manifest.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,no_actual_final_access=True,requires_frozen_complete_original=True)))
