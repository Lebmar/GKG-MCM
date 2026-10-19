"""Own-horizon static target liquidity: drift and hypothetical annual liquidation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research')]
from safeinvest.data import load_paths
from diagnostics.liquidity_capacity import annual_target_liquidity
from diagnostics.q1_stability import PROFILES,FLOORS
PLAN=ROOT/'docs/liquidity_capacity_diagnostic_protocol.json'
FREEZE=ROOT/'docs/liquidity_capacity_diagnostic_source_freeze.json'
OUT=ROOT/'results/liquidity_capacity_diagnostic_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIELDS=('target_bill_share','mean_end_bill_share','minimum_end_bill_share','fraction_end_below_target_floor','max_hypothetical_liquidation_drag','mean_bill_real_loss_fraction')
COMMON_COLUMNS=('job_id','scenario_id','profile','horizon','liquidity','target_floor','effective_assignment','assigned')
PATH_COLUMNS=COMMON_COLUMNS+FIELDS
ANNUAL_COLUMNS=COMMON_COLUMNS+('period','target_bill_share','end_bill_share','end_below_target_floor',
    'hypothetical_liquidation_drag','hypothetical_available_bill_ratio','bill_real_return')


def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def authorize():
    p,f=json.loads(PLAN.read_text()),json.loads(FREEZE.read_text())
    if not f['execution_ready'] or f['protocol_sha256']!=sha(PLAN) or p['allow_final'] is not False or p['development_ids']!=list(range(1,121)):
        raise RuntimeError('Freeze complete development-only diagnostic before numeric access')
    for key in ('source_sha256','input_sha256'):
        for name,digest in f[key].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Frozen dependency changed: '+name)
    return p,f


def run():
    p,f=authorize()
    if OUT.exists():raise RuntimeError('Refuse overwrite')
    OUT.mkdir(parents=True)
    snapshot=dict(protocol_sha256=sha(PLAN),source_freeze_sha256=sha(FREEZE),sources_and_inputs={**f['source_sha256'],**f['input_sha256']})
    writej(OUT/'manifest.json',dict(status='running',phase='before_guarded_parser',**snapshot))
    try:
        r,pi,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
        jobs=json.loads((ROOT/p['jobs']).read_text())['jobs'];fits=json.loads((ROOT/p['static_fits']).read_text())
        original=pd.read_csv(ROOT/p['q1_oof_paths'],float_precision='round_trip')
        paths=[];periods=[]
        for job in jobs:
            if job['stage']!='refit' or job['parent_context_id']==0:continue
            if set(job['training_ids'])&set(job['evaluation_ids']):raise RuntimeError('Own outer whole-path split required')
            fit=fits[job['job_id']]['refit']
            if fit['training_ids']!=job['training_ids']:raise RuntimeError('Own training target mismatch')
            for profile in PROFILES:
                name=profile['name'];H=profile['horizon'];floor=FLOORS[profile['liquidity']]
                assignment=fit['effective_assignments'][name]
                menu=fit['raw_fit']['menu'] if fit['raw_fit'] else []
                product=next((row for row in menu if row['name']==assignment),None)
                for sid in job['evaluation_ids']:
                    prior=original[(original.job_id==job['job_id'])&(original.scenario_id==sid)&(original.profile==name)]
                    if len(prior)!=1 or bool(prior.iloc[0].assigned)!=(product is not None) or prior.iloc[0].effective_assignment!=assignment:
                        raise RuntimeError('Diagnostic must retain the unchanged Q1 effective assignment')
                    common=dict(job_id=job['job_id'],scenario_id=sid,profile=name,horizon=H,liquidity=profile['liquidity'],target_floor=floor,effective_assignment=assignment,assigned=product is not None)
                    if product is None:paths.append(dict(common,**dict.fromkeys(FIELDS)));continue
                    q=annual_target_liquidity(product['weights'],r[sid-1,:H],floor=floor)
                    bill_real=(1+r[sid-1,:H,2])/(1+pi[sid-1,:H])-1
                    paths.append(dict(common,target_bill_share=product['weights'][2],mean_end_bill_share=float(q['end_bill_share'].mean()),minimum_end_bill_share=float(q['end_bill_share'].min()),fraction_end_below_target_floor=float(q['below_target_floor'].mean()),max_hypothetical_liquidation_drag=float(q['hypothetical_liquidation_drag'].max()),mean_bill_real_loss_fraction=float(np.mean(bill_real<0))))
                    for t in range(H):periods.append(dict(common,period=t+1,target_bill_share=product['weights'][2],end_bill_share=float(q['end_bill_share'][t]),end_below_target_floor=bool(q['below_target_floor'][t]),hypothetical_liquidation_drag=float(q['hypothetical_liquidation_drag'][t]),hypothetical_available_bill_ratio=float(q['hypothetical_available_bill_ratio'][t]),bill_real_return=float(bill_real[t])))
        summaries=[]
        for profile in PROFILES:
            rows=[q for q in paths if q['profile']==profile['name']];funded=[q for q in rows if q['assigned']]
            if len(rows)!=120 or sorted(q['scenario_id'] for q in rows)!=list(range(1,121)):raise RuntimeError('Each profile requires original120 once')
            value=dict(profile=profile['name'],horizon=profile['horizon'],liquidity=profile['liquidity'],target_floor=FLOORS[profile['liquidity']],original_n=120,financial_n=len(funded),review_n=120-len(funded))
            keys=('minimum_target_bill_share','maximum_target_bill_share','mean_end_bill_share','minimum_end_bill_share','mean_fraction_end_below_target_floor','paths_with_any_end_below_target_floor','maximum_hypothetical_liquidation_drag','mean_bill_real_loss_fraction')
            value.update(dict.fromkeys(keys))
            if funded:value.update(minimum_target_bill_share=min(q['target_bill_share'] for q in funded),maximum_target_bill_share=max(q['target_bill_share'] for q in funded),mean_end_bill_share=float(np.mean([q['mean_end_bill_share'] for q in funded])),minimum_end_bill_share=min(q['minimum_end_bill_share'] for q in funded),mean_fraction_end_below_target_floor=float(np.mean([q['fraction_end_below_target_floor'] for q in funded])),paths_with_any_end_below_target_floor=sum(q['fraction_end_below_target_floor']>0 for q in funded),maximum_hypothetical_liquidation_drag=max(q['max_hypothetical_liquidation_drag'] for q in funded),mean_bill_real_loss_fraction=float(np.mean([q['mean_bill_real_loss_fraction'] for q in funded])))
            summaries.append(value)
        if len(paths)!=600 or len(periods)!=sum(q['horizon'] for q in PROFILES for v in paths if v['profile']==q['name'] and v['assigned']):raise RuntimeError('Own-H endpoint denominator')
        pd.DataFrame(paths,columns=PATH_COLUMNS).to_csv(OUT/'path_capacity.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        pd.DataFrame(periods,columns=ANNUAL_COLUMNS).to_csv(OUT/'annual_capacity.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        writej(OUT/'summary.json',summaries)
        if any(sha(ROOT/name)!=digest for name,digest in snapshot['sources_and_inputs'].items()):raise RuntimeError('Frozen diagnostic source/input changed')
        writej(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),**snapshot,path_rows=len(paths),annual_rows=len(periods),final_scenarios_loaded=False,new_policy_runs=0,new_fits=0,withdrawals_assumed=False,hypothetical_liquidation_charged_to_strategy=False,outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name!='manifest.json'}))
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc());writej(OUT/'manifest.json',dict(status='failed',error_type=type(error).__name__,error=str(error),**snapshot));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,development_only=True,withdrawals_assumed=False)))
