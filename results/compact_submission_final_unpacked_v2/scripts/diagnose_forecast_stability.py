"""Audit numerical precision of V1 forecasts on fitting scenarios 1–80 only.

Synthetic decision states are specified before this diagnostic. No policy is
selected, no final data are read, and no existing archive is replaced.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from safeinvest.data import load_paths
from safeinvest.core import execute_target
from safeinvest.policies import forecast_bank,projected_score

r,inf,ids=load_paths(ROOT/'data/SafeInvest_Synthetic_Data.csv',range(1,81))
base=np.asarray(json.loads((ROOT/'results/development/fit80/portfolios.json').read_text())['Balanced'])
states={
 'initial_bills':{'weights':[0,0,1],'current_drawdown':0.,'past_mdd':0.},
 'equity_overweight_5pp':{'weights':(base+np.array([.05,-.025,-.025])).tolist(),'current_drawdown':0.,'past_mdd':0.},
 'equity_overweight_1pp_in_drawdown':{'weights':(base+np.array([.01,-.005,-.005])).tolist(),'current_drawdown':.10,'past_mdd':.15},
 'already_at_target':{'weights':base.tolist(),'current_drawdown':0.,'past_mdd':0.}
}
sizes=[256,1024,4096];seeds=list(range(1729,1741));horizons=[5,15,30]
values={(name,n,h):[] for name in states for n in sizes for h in horizons}
for n in sizes:
 for seed in seeds:
  bank=forecast_bank(r,inf,size=n,block_length=5,seed=seed)
  for name,state in states.items():
   x=np.array([state['weights']])*100
   post,fee,moved=execute_target(x,base)
   peak=np.array([1/(1-state['current_drawdown'])]);past=np.array([state['past_mdd']])
   for h in horizons:
    hold=projected_score(x,x,bank,h,peak,past,4.)
    trade=projected_score(x,post,bank,h,peak,past,4.)
    values[name,n,h].append(float((trade-hold)[0]))
rows=[]
for (name,n,h),v in values.items():
 a=np.asarray(v)
 rows.append({'state':name,'forecast_size':n,'remaining_years':h,
  'mean_net_utility_gain':float(a.mean()),'sd_across_random_seeds':float(a.std(ddof=1)),
  'min_gain':float(a.min()),'max_gain':float(a.max()),
  'seed_trade_fraction_at_v1_margin':float(np.mean(a>.002)),
  'sd_over_v1_margin':float(a.std(ddof=1)/.002)})
assert all(row['mean_net_utility_gain']==0 and row['sd_across_random_seeds']==0
           for row in rows if row['state']=='already_at_target')
out={'purpose':'Numerical precision audit of an unaccepted V1 modelling choice',
 'data_sha256':hashlib.sha256((ROOT/'data/SafeInvest_Synthetic_Data.csv').read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'zero_trade_control_passed':True,
 'scenarios':ids.tolist(),'base_target_from_fit80':base.tolist(),'synthetic_states':states,
 'seed_count':len(seeds),'seeds':seeds,'block_length':5,'v1_margin':.002,
 'not_done':['No forecasting model selected','No weights fitted anew','No 81–180 returns read',
             'No final policy replayed','No change to old model or report'],
 'interpretation':'Between-seed spread measures Monte Carlo numerical variability conditional on the same empirical model. It is not predictive uncertainty or a confidence interval for future market performance.',
 'rows':rows}
path=ROOT/'results/model_review/forecast_stability.json';path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(out,indent=2))
noncontrol=[x for x in rows if x['state']!='already_at_target']
for n in sizes:
 group=[x for x in noncontrol if x['forecast_size']==n]
 disagree=sum(0<x['seed_trade_fraction_at_v1_margin']<1 for x in group)
 print(json.dumps({'size':n,'states_horizons':len(group),'seed_disagreement_cases':disagree,
                  'largest_sd_over_margin':max(x['sd_over_v1_margin'] for x in group)}))
print('Saved',path.relative_to(ROOT))
