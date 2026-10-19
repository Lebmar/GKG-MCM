"""Isolated same-source130-task replay; no additional market validation."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import reproduce_simple_execution_stage as engine


def compare_study(original,replayed):
    original,replayed=Path(original),Path(replayed)
    a=json.loads((original/'manifest.json').read_text());b=json.loads((replayed/'manifest.json').read_text())
    for m in (a,b):
        if (m.get('status')!='complete' or m['final_scenarios_loaded'] is not False
            or m['scenario_ids_loaded']!=list(range(1,121)) or m['training_tasks']!=130
            or m['primary_method']!='constrained_net_growth_rollout'):raise AssertionError('Unexpected replay scope')
        if m['sources']!=m['sources_at_end'] or m['input_sha256']!=m['input_sha256_at_end']:raise AssertionError('Source/input mutation')
    if set(a['outputs_sha256'])!=set(b['outputs_sha256']):raise AssertionError('Output set differs')
    compared={}
    for rel in a['outputs_sha256']:
        x=engine.safe_file(original,rel);y=engine.safe_file(replayed,rel)
        if engine.sha(x)!=a['outputs_sha256'][rel] or engine.sha(y)!=b['outputs_sha256'][rel]:raise AssertionError('Output SHA differs')
        compared[rel]={**engine.compare_artifact(x,y),'original_sha256':engine.sha(x),'replay_sha256':engine.sha(y)}
    for m in (a,b):
        m.pop('completed_utc',None);m.pop('elapsed_seconds',None)
        for rel,result in compared.items():
            if result['comparison']!='byte_exact':m['outputs_sha256'][rel]='<independently-verified-exact-artifact>'
    if a!=b:raise AssertionError('Manifest differs beyond declared runtime/container metadata')
    return compared


def configure():
    engine.STAGE=Path('results/model_review/constrained_fan_execution_v1')
    engine.PROTOCOL=Path('docs/constrained_fan_execution_protocol.json')
    engine.RUNNER=Path('scripts/run_constrained_fan_execution.py')
    engine.SELF=Path('scripts/reproduce_constrained_fan_execution.py')
    engine.TEST=Path('tests/test_reproduce_constrained_fan_execution.py')
    engine.DEFAULT_OUT=Path('results/model_review/constrained_fan_reproduction_v1')
    engine.LAUNCHER=engine.LAUNCHER.replace('scripts/run_simple_execution_comparison.py','scripts/run_constrained_fan_execution.py')
    engine.compare_study=compare_study


def reproduce(root,output):
    configure()
    dependency=Path(engine.__file__).resolve()
    before=engine.sha(dependency)
    engine.reproduce(root,output)
    after=engine.sha(dependency)
    if before!=after:raise RuntimeError('Shared isolated-replay engine changed')
    receipt=Path(output)/'manifest.json'
    m=json.loads(receipt.read_text())
    m['wrapper_engine_source_sha256']={str(dependency.relative_to(Path(root).resolve())):before}
    m['wrapper_engine_source_sha256_at_end']={str(dependency.relative_to(Path(root).resolve())):after}
    receipt.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    reproduce(ROOT,ROOT/Path('results/model_review/constrained_fan_reproduction_v1'))
