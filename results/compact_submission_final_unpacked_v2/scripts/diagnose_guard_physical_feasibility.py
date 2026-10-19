"""All selected first-guard states, exact-fee full-simplex diagnostic only.

Coefficients use audited predecision state and own training/calendar support.
Whole saved state members are read for indexing, without using future columns
or loading actual market-realization members as inputs to the diagnostic LP.
Root freezes this new protocol/source before any saved numerical array parse.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research')]
from diagnostics.guard_physical_feasibility import solve_full_simplex, SOLVER_METHOD, SOLVER_OPTIONS, CERTIFICATE_TOL, COMPARISON_TOL

PLAN=Path('docs/guard_physical_feasibility_protocol.json')
FREEZE=Path('docs/guard_physical_feasibility_source_freeze.json')
OUT=ROOT/'results/guard_physical_feasibility_v1'
PARENTS=(dict(version='V2b',directory='results/state_budget_mitigation_v1',protocol='docs/state_budget_mitigation_protocol.json',source_freeze='docs/state_budget_mitigation_source_freeze.json'),
         dict(version='V3',directory='results/state_budget_cushion_v1',protocol='docs/state_budget_cushion_protocol.json',source_freeze='docs/state_budget_cushion_source_freeze.json'))
PRODUCER_OUTPUTS={'path_metrics.csv.gz','summary.csv.gz','selections.json','trace_groups.json','initial_witnesses.json','traces.npz'}
REQUIRED_SOURCES={'research/diagnostics/guard_physical_feasibility.py','scripts/diagnose_guard_physical_feasibility.py',
    'tests/test_guard_physical_feasibility.py','tests/test_guard_physical_feasibility_runner.py'}
IDENTITY=('version','scope','job_id','group_id','scenario_id','account_row','period')
CSV_FIELDS=IDENTITY+('current_real_wealth','real_peak','running_mdd','budget_floor','release_fraction','cushion_scale',
    'classification','all_eight_numerically_resolved','passed_certificate_count','best_pattern_index',
    'hold_worst_next_real','best_worst_next_real','best_budget_margin_real','maximum_dual_objective_real','best_nominal_fee','comparison_band_real')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def readj(path):return json.loads(Path(path).read_text())


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def writej(path,value):Path(path).write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def writegz(path,value):
    payload=(json.dumps(jsonable(value),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    Path(path).write_bytes(gzip.compress(payload,mtime=0))


def verify_map(root,mapping):
    for name,digest in mapping.items():
        path=Path(name)
        if (path.is_absolute() or '..' in path.parts or path.as_posix()!=name
                or not isinstance(digest,str) or len(digest)!=64 or sha(root/path)!=digest):
            raise RuntimeError('Frozen guard diagnostic dependency changed: '+str(name))


def parent_closure(root,parent):
    """Read/hash metadata/bytes only, without np.load or CSV numeric parsing."""
    folder=Path(parent['directory']); manifest_path=folder/'manifest.json'; audit_path=folder/'independent_audit.json'
    m=readj(root/manifest_path);audit=readj(root/audit_path)
    before=m['sources_and_inputs_at_start'];bound=audit['sources_and_artifacts_at_start']
    if (m['status']!='complete' or m.get('final_scenarios_loaded') is not False
            or m.get('new_final_reads')!=0 or before!=m['sources_and_inputs_at_end']
            or set(m['outputs_sha256'])!=PRODUCER_OUTPUTS
            or audit['status']!='passed' or audit.get('final_scenarios_loaded') is not False
            or audit['parent_manifest_sha256']!=sha(root/manifest_path)
            or bound!=audit['sources_and_artifacts_at_end']):
        raise RuntimeError('Require current completed producer and passed bound independent audit')
    verify_map(root,before);verify_map(root,bound)
    outputs={str(folder/name):digest for name,digest in m['outputs_sha256'].items()}
    verify_map(root,outputs)
    required={str(manifest_path):sha(root/manifest_path),**outputs,
        parent['protocol']:sha(root/parent['protocol']),parent['source_freeze']:sha(root/parent['source_freeze'])}
    if any(bound.get(name)!=digest for name,digest in required.items()):
        raise RuntimeError('Independent audit omitted a current producer artifact/protocol binding')
    return {**before,**bound,**required,str(audit_path):sha(root/audit_path)}


def freeze_gate(root=ROOT):
    root=Path(root);p,f=readj(root/PLAN),readj(root/FREEZE)
    if (p.get('version')!='guard-full-simplex-physical-feasibility-v1' or p.get('execution_ready') is not True
            or f.get('execution_ready') is not True or f.get('protocol_sha256')!=sha(root/PLAN)
            or p.get('development_ids')!=list(range(1,121)) or p.get('allow_final') is not False
            or p.get('parents')!=list(PARENTS) or p.get('scope_kinds')!=['outer_selected','full_selected']
            or p.get('solver_method')!=SOLVER_METHOD or p.get('solver_options')!=SOLVER_OPTIONS
            or p.get('certificate_tolerance')!=CERTIFICATE_TOL or p.get('comparison_tolerance')!=COMPARISON_TOL
            or p.get('expected_scipy_version')!=scipy.__version__ or p.get('cost_rate')!=.0025
            or p.get('loss_tolerance')!=.4 or p.get('original_horizon')!=30):
        raise RuntimeError('Freeze unchanged diagnostic/solver/development-only protocol before saved-state access')
    if not REQUIRED_SOURCES<=set(f.get('source_sha256',{})):
        raise RuntimeError('Complete new diagnostic source/test closure required')
    verify_map(root,f['source_sha256']);verify_map(root,f['input_sha256'])
    frozen={**f['source_sha256'],**f['input_sha256']}
    for parent in PARENTS:
        closure=parent_closure(root,parent)
        if any(frozen.get(name)!=digest for name,digest in closure.items()):
            raise RuntimeError('New diagnostic freeze must include the complete producer/audit transitive closure')
    return p,f


def selected_inventory(groups):
    selected=[g for g in groups if g['kind'] in ('outer_selected','full_selected')]
    if (len(selected)!=6 or sum(g['kind']=='full_selected' for g in selected)!=1
            or len({g['group_id'] for g in selected})!=6):
        raise RuntimeError('Retain exactly one full selected and five outer selected metadata groups')
    for kind in ('outer_selected','full_selected'):
        ids=[sid for g in selected if g['kind']==kind for sid in g['scenario_ids']]
        if sorted(ids)!=list(range(1,121)) or any(type(sid) is not int for sid in ids):
            raise RuntimeError('Every original ID1-120 once per selected scope, including NULL groups')
    return selected


def first_guard_states(group,archive,*,version,horizon=30,L=.4):
    """Only predecision state and own training/calendar rows are accessed."""
    if not group['array_names']:return []
    n=len(group['scenario_ids']);prefix=group['group_id']+'__'
    def get(name,shape):
        if name not in group['array_names']:raise RuntimeError('Missing selected state witness: '+name)
        value=archive[prefix+name]
        if value.shape!=shape:raise RuntimeError('Selected state shape changed: '+name)
        return value
    tr=group['training_ids']
    if not tr or len(set(tr))!=len(tr) or any(type(i) is not int or not 1<=i<=120 for i in tr):
        raise RuntimeError('Only original own-training IDs1-120 may supply support')
    if group['kind']=='outer_selected' and set(tr)&set(group['scenario_ids']):raise RuntimeError('Outer support leakage')
    first=get('first_guard_trigger',(n,));trigger=get('guard_triggered',(n,horizon))
    if first.dtype.kind not in 'iu' or trigger.dtype.kind!='b':raise RuntimeError('Typed first-guard identity required')
    positions=np.flatnonzero(first>=0)
    if np.any((first!=-1)&((first<2)|(first>horizon))):raise RuntimeError('Funded first guard must follow original initial admission')
    if not np.array_equal(trigger.sum(axis=1),first>=0):raise RuntimeError('Exactly one guard event or none for each selected path')
    states=[]
    if not len(positions):return states
    x=get('holdings',(n,horizon+1,3));cp=get('cpi',(n,horizon+1))
    peak=get('real_peak',(n,horizon+1));mdd=get('running_mdd',(n,horizon+1))
    support=get('training_real_gross',(horizon,len(tr),3));floor=get('budget_floor',(n,horizon))
    prior_pending=get('was_pending_input',(n,horizon));selected=get('selected_index',(n,horizon))
    feasible=get('support_feasible',(n,horizon));candidates=get('candidate_feasible',(n,horizon,13))
    intervals=get('interval_feasible',(n,horizon,4))
    scales=get('cushion_scale',(n,horizon)) if version=='V3' else None
    rho=float(group['candidate']['release_fraction'])
    if rho not in (.5,.75,1.):raise RuntimeError('Unchanged six-family budget release required')
    for i in positions:
        t=int(first[i])-1;real=float(x[i,t].sum()/cp[i,t]);orig_floor=(1-L)*peak[i,t]
        expected=max(orig_floor,real-rho*(real-orig_floor))
        if (not trigger[i,t] or prior_pending[i,t] or selected[i,t]!=-1 or feasible[i,t] or candidates[i,t].any()
                or not np.isfinite(floor[i,t]) or abs(float(floor[i,t])-expected)>1e-10*real
                or mdd[i,t]-L>1e-12):
            raise RuntimeError('First guard must be the original active failed primary state with unchanged floor/history')
        states.append(dict(version=version,scope=group['kind'],job_id=group['job_id'],group_id=group['group_id'],
            scenario_id=int(group['scenario_ids'][i]),account_row=int(i),period=t+1,
            holdings=x[i,t].copy(),cpi=float(cp[i,t]),real_peak=float(peak[i,t]),running_mdd=float(mdd[i,t]),
            current_real_wealth=real,budget_floor=float(floor[i,t]),release_fraction=rho,
            cushion_scale=None if scales is None else float(scales[i,t]),training_ids=tr,
            support_real_gross=support[t].copy(),primary_feasible_point_count=0,
            primary_feasible_interval_count=int(intervals[i,t].sum()),actual_future_used_in_LP=False))
    return states


def run(output=OUT,root=ROOT):
    root=Path(root);output=Path(output).resolve()
    if output.exists():raise RuntimeError('Never overwrite any complete/failed/incomplete diagnostic attempt')
    p,f=freeze_gate(root);output.mkdir(parents=True,exist_ok=False)
    snapshot={**f['source_sha256'],**f['input_sha256']}
    manifest=dict(status='incomplete',phase='before_saved_array_parse',started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(root/PLAN),source_freeze_sha256=sha(root/FREEZE),sources_and_inputs_at_start=snapshot,
        raw_market_values_parsed=False,final_scenarios_loaded=False,new_policy_runs=0,account_action_changed=False)
    writej(output/'manifest.json',manifest);cases=[];inventory=[]
    try:
        for parent in PARENTS:
            folder=root/parent['directory'];groups=selected_inventory(readj(folder/'trace_groups.json'))
            manifest.update(phase='saved_state_diagnostic',current_version=parent['version']);writej(output/'manifest.json',manifest)
            with np.load(folder/'traces.npz',allow_pickle=False) as archive:
                for group in groups:
                    states=first_guard_states(group,archive,version=parent['version'])
                    inventory.append(dict(version=parent['version'],scope=group['kind'],group_id=group['group_id'],
                        job_id=group['job_id'],scenario_ids=group['scenario_ids'],status=group.get('status'),
                        physical_arm_present=bool(group['array_names']),first_guard_count=len(states)))
                    for state in states:
                        result=solve_full_simplex(state['holdings'],state['cpi'],state['support_real_gross'],
                            budget_floor=state['budget_floor'],cost_rate=p['cost_rate'])
                        cases.append(dict(**state,result=result))
        summaries=[]
        for version in ('V2b','V3'):
            for scope in ('outer_selected','full_selected'):
                cohort=[r for r in cases if r['version']==version and r['scope']==scope]
                groups=[g for g in inventory if g['version']==version and g['scope']==scope]
                classifications={name:sum(c['result']['classification']==name for c in cohort) for name in (
                    'numerical_physical_feasible_witness_above_budget','all_eight_numerical_optima_below_budget',
                    'numerical_boundary_unresolved','incomplete_solver_or_certificate_evidence','solver_or_certificate_unresolved')}
                summaries.append(dict(version=version,scope=scope,original_n=120,group_count=len(groups),
                    physical_path_n=sum(len(g['scenario_ids']) for g in groups if g['physical_arm_present']),
                    first_guard_n=len(cohort),LP_count=8*len(cohort),classification_counts=classifications,
                    all_eight_resolved_n=sum(c['result']['all_eight_numerically_resolved'] for c in cohort)))
        rows=[{**{k:c.get(k) for k in CSV_FIELDS},**{k:c['result'].get(k) for k in CSV_FIELDS if k not in c}} for c in cases]
        pd.DataFrame(rows,columns=CSV_FIELDS).to_csv(output/'guard_cases.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        writegz(output/'case_inputs_and_lp_witnesses.json.gz',cases);writej(output/'selected_group_inventory.json',inventory)
        writej(output/'summary.json',summaries)
        end={name:sha(root/name) for name in snapshot}
        if end!=snapshot or sha(root/PLAN)!=manifest['protocol_sha256'] or sha(root/FREEZE)!=manifest['source_freeze_sha256']:
            raise RuntimeError('Frozen diagnostic or original parent evidence changed')
        manifest.update(status='complete',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            sources_and_inputs_at_end=end,selected_group_count=len(inventory),first_guard_count=len(cases),LP_count=8*len(cases),
            exact_arithmetic_impossibility_proof=False,all_first_guards_preserved=True,new_final_reads=0,
            outputs_sha256={p.name:sha(p) for p in output.iterdir() if p.name!='manifest.json'})
        writej(output/'manifest.json',manifest)
    except BaseException as exc:
        writegz(output/'cases_partial.json.gz',cases);writej(output/'inventory_partial.json',inventory)
        (output/'failure.txt').write_text(traceback.format_exc())
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(output/'manifest.json',manifest);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    if args.execute:run(args.output)
    else:print(json.dumps(dict(source_only=True,raw_market_values_parsed=False,account_action_changed=False)))
