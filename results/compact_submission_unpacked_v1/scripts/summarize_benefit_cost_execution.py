"""Read-only summaries of prospectively chosen benefit-cost controllers."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
import run_simple_execution_comparison as h

STAGE=ROOT/'results/model_review/benefit_cost_execution_v1'
OUT=ROOT/'results/model_review/benefit_cost_summary_v1'
METHODS=('benefit_cost_one_year','benefit_cost_remaining_H')


def main():
    if OUT.exists():raise RuntimeError('Preserve existing summary')
    m=h.read_json(STAGE/'manifest.json');audit=h.read_json(STAGE/'independent_audit.json')
    if m['status']!='complete' or audit['status']!='passed':raise RuntimeError('Complete independently audited stage required')
    if audit['parent_manifest_sha256']!=h.sha(STAGE/'manifest.json'):raise RuntimeError('Audit manifest binding differs')
    start={str(p.relative_to(ROOT)):h.sha(p) for p in STAGE.iterdir() if p.is_file()}
    for rel,digest in m['outputs_sha256'].items():
        if h.sha(STAGE/rel)!=digest:raise RuntimeError('Output differs: '+rel)
    for rel,digest in {**m['sources'],**m['input_sha256']}.items():
        if h.sha(ROOT/rel)!=digest:raise RuntimeError('Source/input differs: '+rel)
    frame=pd.read_csv(STAGE/'client_paths.csv.gz');fits=pd.read_csv(STAGE/'fit_admission.csv.gz')
    summaries=[];pairs=[]
    for (repeat,policy),g in frame.groupby(['repeat_id','policy'],sort=True):
        a=g.loc[g.assigned];reference=policy not in METHODS
        if not g.loc[~g.assigned,list(h.FINANCIAL)].isna().all().all():raise RuntimeError('Review financials must be NULL')
        summaries.append(dict(repeat_id=int(repeat),policy=policy,scope='in_sample_full120' if repeat==0 else 'outer_oof',
            original_n=len(g),assigned_n=len(a),financial_null_n=len(g)-len(a),reference_only=reference,
            breach_count=int(a.breach.sum()) if len(a) else None,conditional_breach_probability=float(a.breach.mean()) if len(a) else None,
            **{out:float(a[col].mean()) if len(a) else None for out,col in [('mean_log_growth','log_growth'),('mean_mdd','mdd'),('mean_fees','nominal_fees'),('mean_trade_fraction','trade_fraction')]},
            p90_mdd=float(np.quantile(a.mdd,.9,method='inverted_cdf')) if len(a) else None))
        if not reference:
            b=frame.loc[(frame.repeat_id==repeat)&frame.policy.eq('annual_then_pending')]
            z=g.merge(b,on=['context','scenario_id'],suffixes=('_model','_B'),validate='one_to_one');z=z.loc[z.assigned_model&z.assigned_B]
            pairs.append(dict(repeat_id=int(repeat),policy=policy,scope='in_sample_full120' if repeat==0 else 'outer_oof',common_auto_n=len(z),
                model_breaches=int(z.breach_model.sum()) if len(z) else None,B_breaches=int(z.breach_B.sum()) if len(z) else None,
                **{'mean_delta_'+col:float((z[col+'_model']-z[col+'_B']).mean()) if len(z) else None for col in ['log_growth','mdd','nominal_fees','trade_fraction']}))
    decision=[]
    for category,groups in h.read_json(STAGE/'trace_groups.json').items():
        with np.load(STAGE/(category+'_traces.npz'),allow_pickle=False) as a:
            for key,g in groups.items():
                if g['metadata']['policy'] not in METHODS:continue
                available=a[key+'__estimate_available'];optional=available.copy();optional[:,0]=False
                restore=a[key+'__target_execution']&optional;net=a[key+'__mean_net_log_gain'];risk=a[key+'__risk_direction_pass'];reason=a[key+'__gate_reason']
                decision.append(dict(group=key,category=category,policy=g['metadata']['policy'],repeat_id=g['metadata']['repeat_id'],
                    scope=g['metadata']['scope'],paths=len(g['scenario_ids']),optional_decisions=int(optional.sum()),optional_restores=int(restore.sum()),
                    growth_nonpositive_holds=int(((reason=='estimated_net_nonpositive_hold')&optional).sum()),
                    positive_growth_risk_veto_holds=int(((reason=='full_H_risk_direction_veto_hold')&optional).sum()),
                    invalid_optional_restores=int((restore&((net<=1e-12)|~risk)).sum()),
                    minimum_estimated_net_restore=float(net[restore].min()) if restore.any() else None,
                    mean_quoted_fee_fraction_restore=float(a[key+'__quoted_fee_fraction'][restore].mean()) if restore.any() else None))
    if any(d['invalid_optional_restores'] for d in decision):raise RuntimeError('Optional trade violated its two predicates')
    OUT.mkdir(parents=True)
    h.write_csv_gz(OUT/'repeat_summary.csv.gz',summaries);h.write_csv_gz(OUT/'paired_B_summary.csv.gz',pairs);h.write_csv_gz(OUT/'decision_summary.csv.gz',decision)
    full=fits.loc[fits.repeat_id==0].where(pd.notna(fits),None).to_dict('records')
    # JSON conversion avoids pandas NaN as valid economic outcomes.
    import json
    h.write_json(OUT/'full_development_admission.json',json.loads(fits.loc[fits.repeat_id==0].to_json(orient='records')))
    after={str(p.relative_to(ROOT)):h.sha(p) for p in STAGE.iterdir() if p.is_file()}
    if start!=after:raise RuntimeError('Read-only parent changed')
    h.write_json(OUT/'manifest.json',dict(status='complete',parent_manifest_sha256=h.sha(STAGE/'manifest.json'),
        source_sha256={'scripts/summarize_benefit_cost_execution.py':h.sha(__file__)},
        artifacts_at_start=start,artifacts_at_end=after,outputs_sha256={f.name:h.sha(f) for f in OUT.iterdir() if f.is_file()},
        limitations=['Primary named before execution; no outcome-driven switch','Five repeated OOF results reuse original120 paths','Full-development outcome is in-sample, not validation','NULL risk for zero automatic coverage']))
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False));print(fits.loc[fits.repeat_id==0].to_string(index=False))


if __name__=='__main__':main()
