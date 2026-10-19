"""Compact fixed-model reproduction; default synthetic, no final-access CLI.

This new adapter does not disable or replace historical evidence gates. Its
explicit development mode reproduces one already locked model on IDs1-120,
without fitting, selection or a new independent-validation claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research')]
from safeinvest.data import load_paths
from experimental_execution.state_budget_mitigation import simulate_state_budget
from experimental_execution.review_pending import simulate_paired
from experimental_execution.state_budget_selection import summarize_records as base_summary
from safeinvest_research.metrics import path_metrics

REGISTRY=Path('docs/mitigation_main_model_registry.json')
BUNDLE=Path('submission_bundle_manifest.json')
EXPECTED=Path('submission/expected_development.json')
OBJECTS=('mitigation_main','same_target_B','same_target_annual','exact_thirds_annual')
REQUIRED_RUNTIME={'scripts/submission_adapter.py','src/safeinvest/__init__.py','src/safeinvest/core.py','src/safeinvest/data.py',
    'research/experimental_execution/state_budget_execution.py','research/experimental_execution/state_budget_mitigation.py',
    'research/experimental_execution/forecast_ensemble.py','research/experimental_execution/review_pending.py',
    'research/experimental_execution/state_budget_selection.py','research/safeinvest_research/metrics.py'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def readj(path):return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def writej(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def safe_name(name):
    if not isinstance(name,str):raise ValueError('Canonical relative bundle path required')
    p=Path(name)
    if (not name or not p.parts or p.is_absolute() or '..' in p.parts
            or '.' in p.parts or p.as_posix()!=name or '\\' in name):raise ValueError('Canonical relative bundle path required')
    return p


def verify_files(root,mapping):
    root=Path(root).resolve()
    for name,digest in mapping.items():
        p=safe_name(name);path=root/p
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path)!=digest:
            raise RuntimeError('Compact package file changed: '+name)
        cursor=path
        while cursor!=root:
            if cursor.is_symlink():raise RuntimeError('No symlink compact package members or parents')
            cursor=cursor.parent


def verify_loaded_modules(root):
    root=Path(root).resolve()
    modules={'safeinvest.core':'src/safeinvest/core.py','safeinvest.data':'src/safeinvest/data.py',
        'experimental_execution.state_budget_execution':'research/experimental_execution/state_budget_execution.py',
        'experimental_execution.state_budget_mitigation':'research/experimental_execution/state_budget_mitigation.py',
        'experimental_execution.forecast_ensemble':'research/experimental_execution/forecast_ensemble.py',
        'experimental_execution.review_pending':'research/experimental_execution/review_pending.py',
        'experimental_execution.state_budget_selection':'research/experimental_execution/state_budget_selection.py',
        'safeinvest_research.metrics':'research/safeinvest_research/metrics.py'}
    for name,path in modules.items():
        module=sys.modules.get(name)
        if module is None or Path(module.__file__).resolve()!=(root/path).resolve():
            raise RuntimeError('Import must resolve to the compact package: '+name)


def specification(registry):
    required=dict(profile='C',strategic_product='Balanced',target=[.45,.05,.5],horizon=30,L=.4,cost_rate=.0025,
        selected_candidate_id='release_0.5_uniform',release_fraction=.5,forecast_rule='uniform',support_mode='calendar',
        service_alpha=.1,persistent_target_relearning=False)
    if registry.get('development_ids')!=list(range(1,121)) or any(registry['Q2'].get(k)!=v for k,v in required.items()):
        raise ValueError('The already locked V2b specification may not change')
    return dict(version='compact-locked-V2b-four-objects-v1',main=required,objects=list(OBJECTS),
        training_ids=list(range(1,121)),evaluation_ids=list(range(1,121)),initial_wealth=100.,
        asset_order=['equity','bond','bill'],no_refit=True,no_selection=True,final_access=False)


def array_signature(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'biufUS':raise ValueError('Only explicit nonobject numerical/string arrays may be fingerprinted')
    a=np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'),copy=True))
    if a.dtype.kind=='f':a[np.isnan(a)]=np.nan
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def signatures(arms):
    return {name:{key:array_signature(value) for key,value in sorted(arms[name].items()) if isinstance(value,np.ndarray)} for name in OBJECTS}


def unwrap_reference(arm):
    return {k:(v if k=='cpi' else v[:,0]) for k,v in arm.items() if isinstance(v,np.ndarray)
        and k not in ('targets','loss_tolerances') and (k=='cpi' or v.ndim>=2)}


def summarize(arm,ids):
    metric=path_metrics(arm['real_wealth']);n,h=arm['fees'].shape;rows=[]
    for i,sid in enumerate(ids):
        reasons=np.asarray(arm.get('gate_reason',np.full((n,h),'')))[i]
        trigger=np.asarray(arm.get('guard_triggered',np.zeros((n,h),bool)))[i]
        review=bool(np.asarray(arm.get('guard_pending',np.zeros((n,h+1),bool)))[i].any())
        rows.append(dict(scenario_id=int(sid),assigned=True,**{k:float(v[i]) for k,v in metric.items()},
            nominal_fees=float(arm['fees'][i].sum()),trade_fraction=float(arm['trades'][i].mean()),
            breach=bool(arm['ever_breached'][i,-1]),automatic_fraction=float(1-np.mean(arm['pending_at_decision'][i]|trigger)),
            protection_trade_fraction=float(np.mean((reasons=='risk_budget_protection')|((reasons=='risk_mitigation_then_review')&arm['trades'][i]))),
            mitigation_trade_fraction=float(np.mean((reasons=='risk_mitigation_then_review')&arm['trades'][i])),
            mitigation_nominal_fees=float(arm['fees'][i][reasons=='risk_mitigation_then_review'].sum()),support_review=review))
    result=base_summary(rows)
    result['mean_mitigation_trade_fraction']=float(np.mean([r['mitigation_trade_fraction'] for r in rows]))
    result['mean_mitigation_fees']=float(np.mean([r['mitigation_nominal_fees'] for r in rows]))
    return result


def evaluate_fixed(returns,inflation,ids,training,training_pi,training_ids,spec):
    q=spec['main'];w=np.asarray(q['target'],float)
    # Canonical jobs use advanced-index copies of the loader views. Preserve
    # that C-order layout for exact field reproduction, without changing data.
    returns=np.ascontiguousarray(returns);inflation=np.ascontiguousarray(inflation)
    training=np.ascontiguousarray(training);training_pi=np.ascontiguousarray(training_pi)
    main=simulate_state_budget(returns,inflation,training,training_pi,w,L=q['L'],horizon=q['horizon'],cost_rate=q['cost_rate'],
        release_fraction=q['release_fraction'],forecast_rule=q['forecast_rule'],training_ids=training_ids,actual_ids=ids,support_mode='calendar')
    if not main['financial_available']:raise RuntimeError('Locked funded full-development reproduction unexpectedly became Review')
    paired=simulate_paired(returns,inflation,w,q['L'],horizon=q['horizon'],cost_rate=q['cost_rate'])
    thirds=simulate_paired(returns,inflation,np.ones(3)/3,q['L'],horizon=q['horizon'],cost_rate=q['cost_rate'])
    arms=dict(mitigation_main=main['arm'],same_target_B=unwrap_reference(paired['pending_hold']),
        same_target_annual=unwrap_reference(paired['annual']),exact_thirds_annual=unwrap_reference(thirds['annual']))
    return arms,{name:summarize(arms[name],ids) for name in OBJECTS}


def compare_summary(actual,expected):
    if set(actual)!=set(expected):raise AssertionError('Expected summary field inventory differs')
    for key,value in expected.items():
        if value is None:
            if actual[key] is not None:raise AssertionError('Expected summary NULL: '+key)
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            if not np.isfinite(actual[key]) or not np.isclose(actual[key],value,atol=1e-12,rtol=1e-12):raise AssertionError('Summary mismatch: '+key)
        elif actual[key]!=value:raise AssertionError('Summary mismatch: '+key)


def compare_expected(arms,summaries,expected,spec):
    if expected['object_specification']!=spec or expected['array_signatures']!=signatures(arms):
        raise AssertionError('Canonical full-development array schema/dtype/shape/value digest mismatch')
    if set(expected['summaries'])!=set(OBJECTS):raise AssertionError('All four canonical summaries required')
    for name in OBJECTS:compare_summary(summaries[name],expected['summaries'][name])


def verify_compact_manifest(root,spec):
    m=readj(root/BUNDLE)
    if (m.get('version')!='compact-technical-source-bundle-v1' or m.get('status')!='complete' or m.get('development_ready') is not True
            or m.get('final_access') is not False or m.get('object_specification')!=spec
            or not REQUIRED_RUNTIME|{str(REGISTRY),'data/SafeInvest_Synthetic_Data.csv',str(EXPECTED)}<=set(m.get('files_sha256',{}))):
        raise RuntimeError('Completed exact compact bundle manifest required before development parser')
    verify_files(root,m['files_sha256'])
    expected=readj(root/EXPECTED)
    if expected.get('version')!='compact-canonical-development-expectations-v1' or expected.get('object_specification')!=spec:
        raise RuntimeError('Frozen canonical expectation specification differs')
    return m,expected


def run(mode='synthetic',output=None,root=ROOT):
    root=Path(root).resolve()
    if mode not in ('synthetic','development'):raise ValueError('Only synthetic or development1-120 exists; no final mode')
    output=Path(output or root/f'submission_reproduction_{mode}_v1').resolve()
    if output.exists():raise RuntimeError('Never overwrite any existing reproduction attempt')
    spec=specification(readj(root/REGISTRY));manifest=expected=None;bundle_sha=None
    if mode=='development':
        manifest,expected=verify_compact_manifest(root,spec);verify_loaded_modules(root);bundle_sha=sha(root/BUNDLE)
    output.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',mode=mode,phase='before_data',started_utc=datetime.now(timezone.utc).isoformat(),
        object_specification=spec,raw_market_values_parsed=False,final_scenarios_loaded=False,new_fits=0,new_selections=0,
        historical_runner_gates_replaced=False,independent_validation=False)
    writej(output/'manifest.json',state)
    try:
        if mode=='development':
            state.update(phase='guarded_development_load',raw_market_values_parsed=True);writej(output/'manifest.json',state)
            market,pi,ids=load_paths(root/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
            if ids.tolist()!=list(range(1,121)):raise RuntimeError('Only original ordered development1-120 permitted')
            training,training_pi,training_ids=market,pi,ids
        else:
            market=np.zeros((2,30,3));pi=np.zeros((2,30));ids=np.array([101,102])
            training=np.zeros((3,30,3));training_pi=np.zeros((3,30));training_ids=np.array([1,2,3])
        arms,summaries=evaluate_fixed(market,pi,ids,training,training_pi,training_ids,spec)
        if mode=='development':
            compare_expected(arms,summaries,expected,spec);verify_files(root,manifest['files_sha256'])
            if sha(root/BUNDLE)!=bundle_sha:raise RuntimeError('Compact bundle manifest changed during reproduction')
        proof=dict(object_specification=spec,array_signatures=signatures(arms),summaries=summaries)
        writej(output/'reproduction_checks.json',proof)
        state.update(status='passed',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),scenario_ids=ids.tolist(),
            canonical_development_compared=mode=='development',array_fields=sum(len(v) for v in proof['array_signatures'].values()),
            source_bundle_manifest_sha256=bundle_sha,
            outputs_sha256={'reproduction_checks.json':sha(output/'reproduction_checks.json')})
        writej(output/'manifest.json',state)
    except BaseException as exc:
        (output/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(output/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group()
    g.add_argument('--synthetic',action='store_true');g.add_argument('--development',action='store_true')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    run('development' if args.development else 'synthetic',args.output)
