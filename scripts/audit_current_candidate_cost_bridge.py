"""Independent saved-evidence identity and arithmetic audit, after producer.

No new bridge producer, strategy, metric function or market parser is imported.
This is an openly subsequent verification, not blind numerical validation.
"""
from datetime import datetime,timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/model_review/current_candidate_cost_bridge_v1'
P=ROOT/'docs/current_candidate_cost_bridge_protocol.json'
J=ROOT/'results/model_review/joint_liquidity_refit_v1'
O=ROOT/'results/model_review/selected_controller_cost_stress_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as stream:return list(csv.DictReader(stream))
def value(v):
    return None if v=='' else True if v=='True' else False if v=='False' else float(v)


def main():
    receipt=OUT/'independent_audit.json'
    if receipt.exists():raise RuntimeError('Never overwrite an independent verification')
    p,m=readj(P),readj(OUT/'manifest.json')
    assert p['execution_ready'] and m['status']=='passed' and m['protocol_sha256']==sha(P)
    files={**p['input_sha256'],**p['source_sha256'],str(P.relative_to(ROOT)):sha(P),
        str((OUT/'manifest.json').relative_to(ROOT)):sha(OUT/'manifest.json'),
        str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))}
    files.update({str((OUT/f).relative_to(ROOT)):h for f,h in m['outputs_sha256'].items()})
    assert m['inputs_and_sources']=={**p['input_sha256'],**p['source_sha256']}
    for path,digest in files.items():assert sha(ROOT/path)==digest,path
    before={path:sha(ROOT/path) for path in files}
    for folder in (J,O):
        manifest,audit=readj(folder/'manifest.json'),readj(folder/'independent_audit.json')
        assert audit['status']=='passed'
        start=audit.get('sources_and_artifacts_at_start',audit.get('source_and_artifact_sha256_start',{}))
        end=audit.get('sources_and_artifacts_at_end',audit.get('source_and_artifact_sha256_end',{}))
        assert start and start==end
        expected=audit.get('parent_manifest_sha256',start.get(str((folder/'manifest.json').relative_to(ROOT))))
        assert expected==sha(folder/'manifest.json')
        for name in ('actual_traces.npz','client_paths.csv'):
            assert sha(folder/name)==manifest['outputs_sha256'][name]
    plan=readj(O/'frozen_controller_plan.json')['full_development']
    counts={};array_values=0
    mapping={'fixed_target_learner':'joint_tie_set_refit_learner','annual_then_pending':'annual_then_pending','equal_thirds_annual':'equal_thirds_annual'}
    with np.load(J/'actual_traces.npz',allow_pickle=False) as current,np.load(O/'actual_traces.npz',allow_pickle=False) as old:
        for historic,modern in mapping.items():
            a='c0__'+modern+'__';b='base__full_development__C__'+historic+'__'
            fields=sorted(k[len(a):] for k in current.files if k.startswith(a))
            assert fields==p['expected_trace_fields'][historic]
            assert np.all(old[b+'scenario_ids']==np.arange(1,121))
            for field in fields:
                left=current[a+field]
                if b+field in old.files:right=old[b+field]
                else:
                    assert field=='targets' and historic!='fixed_target_learner'
                    right=np.asarray(plan['menu'][1]['weights'] if historic=='annual_then_pending' else [1/3]*3,dtype=np.float64)
                assert left.shape==right.shape and left.dtype==right.dtype,(historic,field)
                if left.dtype.kind in 'fc':same=(left==right)|(np.isnan(left)&np.isnan(right))
                else:same=left==right
                assert bool(np.all(same)),(historic,field)
                array_values+=int(left.size)
            counts[historic]=len(fields)
    old_rows=rows(O/'client_paths.csv')
    original={(r['cost_label'],r['policy'],int(r['scenario_id'])):r for r in old_rows if r['context']=='full_development' and r['policy'] in mapping}
    output=rows(OUT/'research_paths.csv');summaries=rows(OUT/'research_summary.csv')
    assert len(output)==1800 and len(summaries)==15
    indexed={(r['cost_label'],r['research_policy'],int(r['scenario_id'])):r for r in output}
    expected={(rate['label'],policy,sid) for rate in p['rates'] for policy in mapping for sid in range(1,121)}
    assert set(indexed)==expected and len(indexed)==len(output)
    fields=('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','nominal_moved','trade_fraction','breach','excess','first_crossing')
    financial_checks=0
    for key,row in indexed.items():
        old=original[key]
        assert float(row['cost_rate'])==float(old['cost_rate'])
        for field in fields:
            assert value(row[field])==value(old[field]),(key,field)
            financial_checks+=1
    summary_index={(r['cost_label'],r['research_policy']):r for r in summaries}
    assert len(summary_index)==15
    max_error=0.
    for rate in p['rates']:
        for policy in mapping:
            cohort=[indexed[rate['label'],policy,sid] for sid in range(1,121)]
            vals=lambda field:sorted(float(value(row[field])) for row in cohort)
            growth,mdd=vals('cagr'),vals('mdd')
            expected_summary=dict(n=120,mean_log_growth=math.fsum(vals('log_growth'))/120,
                median_cagr=(growth[59]+growth[60])/2,p10_cagr=growth[11],median_mdd=(mdd[59]+mdd[60])/2,
                p90_mdd=mdd[107],breach_count=sum(int(value(row['breach'])) for row in cohort),
                mean_nominal_fees=math.fsum(vals('nominal_fees'))/120,
                mean_trade_fraction=math.fsum(vals('trade_fraction'))/120)
            actual=summary_index[rate['label'],policy]
            for field,number in expected_summary.items():
                error=abs(float(actual[field])-number);max_error=max(max_error,error)
                assert error<=2e-12,(rate['label'],policy,field,error)
    bridge=readj(OUT/'bridge.json')
    assert bridge['current_service']['status']=='execution_review' and bridge['current_service']['financial_outputs'] is None
    assert bridge['current_service']['stress_admission_recomputed'] is False
    after={path:sha(ROOT/path) for path in files};assert before==after
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(P),auditor_sha256=sha(Path(__file__)),
        sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,
        exact_array_fields_checked=counts,scalar_values_in_exact_arrays=array_values,
        research_rows_independently_matched_to_old_archive=1800,financial_scalar_checks=financial_checks,
        summary_groups_independently_recomputed=15,maximum_summary_absolute_error=max_error,
        audit_development_history='First metadata-only attempt stopped because the older cost audit uses source_and_artifact_sha256_start/end, rather than the later plural key. Explicit schema support and exact start/end binding were added before numerical comparisons. No frozen producer or evidence changed.',
        raw_market_rows_parsed=False,new_policy_replays=0,new_fits=0,final_scenarios_loaded=False,
        scope='Post-hoc independent exact saved-array and copied-row verification, ordered-statistic/compensated-sum arithmetic; not a new market or strategy test')
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','exact_array_fields_checked','scalar_values_in_exact_arrays','financial_scalar_checks','maximum_summary_absolute_error')}))


if __name__=='__main__':main()
