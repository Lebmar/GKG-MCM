"""Reproducible stages: develop -> freeze -> final. See docs/protocol.json."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse, hashlib, json, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from safeinvest.core import metric_arrays
from safeinvest.data import load_paths
from safeinvest.portfolios import evaluate_grid, rank_profile_candidates, CLIENTS, _wilson_upper
from safeinvest.policies import simulate, forecast_bank

P=json.loads((ROOT/'docs/protocol.json').read_text())
DATA=ROOT/'data/SafeInvest_Synthetic_Data.csv'

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def fingerprint():
    paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'docs/protocol.json',DATA]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def fit(returns,inf,out):
    grid=evaluate_grid(returns,inf)
    grid=grid[(grid[['w_equity','w_bond','w_bill']]>=P['minimum_each_asset']-1e-12).all(axis=1)]
    out.mkdir(parents=True,exist_ok=True);grid.to_csv(out/'grid.csv',index=False)
    portfolios={}
    for name,lam in P['risk_aversion'].items():
        ranked=rank_profile_candidates(grid,lam,P['portfolio_bill_floors'][name])
        ranked.to_csv(out/f'ranking_{name}.csv',index=False)
        row=ranked.iloc[0]
        portfolios[name]=[float(row.w_equity),float(row.w_bond),float(row.w_bill)]
    write_json(out/'portfolios.json',portfolios)
    return portfolios

def run_all(r,inf,portfolios,bank,cost=.0025):
    base=portfolios['Balanced'];kw=dict(bank=bank,cost_rate=cost,epsilon=P['benefit_margin'])
    return {
        'S0_equal':simulate(r,inf,[1/3]*3,'annual',cost_rate=cost),
        'S1_annual':simulate(r,inf,base,'annual',cost_rate=cost),
        'S2_gate':simulate(r,inf,base,'gate',**kw),
        'S3_dynamic':simulate(r,inf,base,'dynamic',**kw),
        'S2_review':simulate(r,inf,base,'gate',review_threshold=P['review_drawdown'],**kw),
        'S3_review':simulate(r,inf,base,'dynamic',review_threshold=P['review_drawdown'],**kw),
    }

def summary(name,sim,inf,cost=.0025):
    m=metric_arrays(sim['wealth'],inf)
    U=np.log(m['terminal_real']/100)-4*m['mdd']
    return dict(strategy=name,cost_rate=cost,n=len(inf),horizon=inf.shape[1],
                cagr_median=float(np.median(m['cagr'])),cagr_p10=float(np.quantile(m['cagr'],.10)),
                mdd_median=float(np.median(m['mdd'])),mdd_p90=float(np.quantile(m['mdd'],.90)),
                volatility_median=float(np.median(m['volatility'])),
                terminal_real_median=float(np.median(m['terminal_real'])),
                mean_utility=float(U.mean()),trade_fraction=float(sim['trades'].mean()),
                subsequent_trade_fraction=float(sim['trades'][:,1:].mean()),
                cost_mean=float(sim['costs'].sum(axis=1).mean()),
                review_fraction=float(sim['pending'].any(axis=1).mean()))

def paired(a,b,inf):
    ma=metric_arrays(a['wealth'],inf);mb=metric_arrays(b['wealth'],inf)
    ua=np.log(ma['terminal_real']/100)-4*ma['mdd'];ub=np.log(mb['terminal_real']/100)-4*mb['mdd']
    rng=np.random.default_rng(P['bootstrap_seed'])
    idx=rng.integers(len(inf),size=(P['bootstrap_replicates'],len(inf)))
    diffs={
      'mean_utility':(ua[idx]-ub[idx]).mean(axis=1),
      'median_cagr':np.median(ma['cagr'][idx],axis=1)-np.median(mb['cagr'][idx],axis=1),
      'p10_cagr':np.quantile(ma['cagr'][idx],.10,axis=1)-np.quantile(mb['cagr'][idx],.10,axis=1),
      'p90_mdd':np.quantile(ma['mdd'][idx],.90,axis=1)-np.quantile(mb['mdd'][idx],.90,axis=1)}
    return {name:{'estimate':float((ua-ub).mean()) if name=='mean_utility' else float(
                {'median_cagr':lambda:np.median(ma['cagr'])-np.median(mb['cagr']),
                 'p10_cagr':lambda:np.quantile(ma['cagr'],.10)-np.quantile(mb['cagr'],.10),
                 'p90_mdd':lambda:np.quantile(ma['mdd'],.90)-np.quantile(mb['mdd'],.90)}[name]()),
                'low95':float(np.quantile(values,.025)),'high95':float(np.quantile(values,.975))}
                for name,values in diffs.items()}

def save_runs(out,runs,inf,ids):
    rows=[summary(name,sim,inf) for name,sim in runs.items()]
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    arrays={'inflation':inf,'scenario_ids':ids}
    for name,sim in runs.items():
        arrays.update({f'{name}__{k}':v for k,v in sim.items()})
    np.savez_compressed(out/'paths.npz',**arrays)
    return rows

def onboarding(r,inf,portfolios,bank,selected,alpha=.1):
    rows=[];assignments={}
    for client in CLIENTS:
        feasible=[]
        for name,target in portfolios.items():
            H=client.horizon
            mode=('dynamic' if selected=='S3_review' else 'gate') if name=='Balanced' else 'annual'
            sim=simulate(r[:,:H],inf[:,:H],target,mode,bank=bank,
                         review_threshold=client.loss_tolerance if name=='Balanced' else None)
            m=metric_arrays(sim['wealth'],inf[:,:H]);k=int((m['mdd']>client.loss_tolerance).sum())
            liquidity=target[2]>=P['liquidity_bill_floors'][client.liquidity]-1e-12
            eligible=k/len(r)<=alpha+1e-12 and liquidity
            growth=float(np.median(m['cagr']))
            rows.append(dict(client=client.name,portfolio=name,horizon=H,tolerance=client.loss_tolerance,
                             w_equity=target[0],w_bond=target[1],w_bill=target[2],
                             breach_count=k,n=len(r),breach_probability=k/len(r),
                             breach_wilson_upper95=_wilson_upper(k,len(r)),
                             cagr_median=growth,mdd_p90=float(np.quantile(m['mdd'],.9)),
                             liquidity_ok=liquidity,eligible=eligible))
            if eligible:feasible.append((growth,name))
        assignments[client.name]=max(feasible)[1] if feasible else 'Human review'
    return pd.DataFrame(rows),assignments

def develop():
    out=ROOT/'results/development';out.mkdir(parents=True,exist_ok=True)
    r,inf,_=load_paths(DATA,range(1,81))
    portfolios=fit(r,inf,out/'fit80')
    bank=forecast_bank(r,inf,P['forecast_size'],P['forecast_block_length'],P['forecast_seed'])
    vr,vi,ids=load_paths(DATA,range(81,121))
    runs=run_all(vr,vi,portfolios,bank)
    rows=save_runs(out,runs,vi,ids)
    compare=paired(runs['S3_review'],runs['S2_review'],vi)
    enable=(compare['mean_utility']['low95']>0 and compare['p90_mdd']['estimate']<=.02
            and compare['p10_cagr']['estimate']>=-.0025)
    decision={'selected':'S3_review' if enable else 'S2_review','paired_dynamic_minus_gate':compare,
              'parameters_tuned_on_validation':False,'protocol_sha256':sha(ROOT/'docs/protocol.json'),
              'fingerprints':fingerprint()}
    write_json(out/'architecture_decision.json',decision)
    print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(decision,indent=2))

def freeze():
    out=ROOT/'results/frozen';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Frozen manifest exists; do not silently replace it.')
    decision=json.loads((ROOT/'results/development/architecture_decision.json').read_text())
    if decision['protocol_sha256']!=sha(ROOT/'docs/protocol.json'):raise RuntimeError('Protocol changed after validation.')
    if decision['fingerprints']!=fingerprint():raise RuntimeError('Model code changed after validation; rerun development first.')
    r,inf,ids=load_paths(DATA,range(1,121))
    portfolios=fit(r,inf,out)
    bank=forecast_bank(r,inf,P['forecast_size'],P['forecast_block_length'],P['forecast_seed'])
    np.save(out/'forecast_bank.npy',bank)
    onboard,assign=onboarding(r,inf,portfolios,bank,decision['selected'])
    onboard.to_csv(out/'onboarding.csv',index=False)
    sensitivities=[]
    for alpha in P['onboarding_alpha_sensitivity']:
        _,a=onboarding(r,inf,portfolios,bank,decision['selected'],alpha)
        sensitivities.append({'alpha':alpha,**a})
    pd.DataFrame(sensitivities).to_csv(out/'onboarding_sensitivity.csv',index=False)
    write_json(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
       selected=decision['selected'],portfolios=portfolios,assignments=assign,
       fingerprints=fingerprint(),forecast_sha256=sha(out/'forecast_bank.npy'),
       python=platform.python_version(),numpy=np.__version__,
       validation_decision=decision,final_policy_evaluation_run=False,
       prior_exposure_disclosure=P['prior_exposure_disclosure']))
    print(json.dumps({'portfolios':portfolios,'assignments':assign,'selected':decision['selected']},indent=2))

def final():
    frozen=ROOT/'results/frozen';out=ROOT/'results/final'
    manifest=json.loads((frozen/'manifest.json').read_text())
    if manifest['fingerprints']!=fingerprint():raise RuntimeError('Frozen source/config/data hash mismatch. Final gate closed.')
    if sha(frozen/'forecast_bank.npy')!=manifest['forecast_sha256']:raise RuntimeError('Forecast bank mismatch.')
    if (out/'evaluation_manifest.json').exists():raise RuntimeError('Final evaluation already exists; use archived evidence, not iterative tuning.')
    out.mkdir(parents=True,exist_ok=True)
    r,inf,ids=load_paths(DATA,range(121,181),allow_final=True)
    bank=np.load(frozen/'forecast_bank.npy');ports=manifest['portfolios']
    runs=run_all(r,inf,ports,bank);rows=save_runs(out,runs,inf,ids)
    selected=manifest['selected']
    comparisons={name:paired(runs[name],runs['S0_equal'],inf) for name in runs if name!='S0_equal'}
    comparisons['S3_review_minus_S2_review']=paired(runs['S3_review'],runs['S2_review'],inf)
    comparisons['S2_gate_minus_S1_annual']=paired(runs['S2_gate'],runs['S1_annual'],inf)
    comparisons['S2_review_minus_S2_gate']=paired(runs['S2_review'],runs['S2_gate'],inf)
    write_json(out/'paired_comparisons.json',comparisons)
    sensitive=[]
    for cost in P['cost_sensitivity']:
        if cost==P['cost_rate']:
            rr=runs
        else:rr=run_all(r,inf,ports,bank,cost)
        sensitive.extend(summary(name,rr[name],inf,cost) for name in ['S0_equal',selected])
    pd.DataFrame(sensitive).to_csv(out/'cost_sensitivity.csv',index=False)
    profiles,_=onboarding(r,inf,ports,bank,selected)
    profiles['frozen_assignment']=profiles.client.map(manifest['assignments'])
    profiles.to_csv(out/'onboarding_diagnostics.csv',index=False)
    m=metric_arrays(runs['S0_equal']['wealth'],inf);worst=int(ids[np.argmax(m['mdd'])])
    # Recompute 15-year policies at that horizon, not truncate a 30-year lookahead policy.
    short=run_all(r[:,:15],inf[:,:15],ports,bank)
    pd.DataFrame([summary(name,sim,inf[:,:15]) for name,sim in short.items()]).to_csv(out/'horizon15.csv',index=False)
    write_json(out/'evaluation_manifest.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
      frozen_manifest_sha256=sha(frozen/'manifest.json'),fingerprints=fingerprint(),
      selected=selected,scenario_ids=ids.tolist(),severe_scenario=worst,
      selection_changed_after_evaluation=False))
    print(pd.DataFrame(rows).to_string(index=False));print('Frozen policy:',selected,'; severe scenario:',worst)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','freeze','final'])
    args=parser.parse_args();globals()[args.stage]()
