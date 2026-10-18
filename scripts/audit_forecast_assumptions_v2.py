"""Independent least-squares and table audit of the fixed fan diagnostic.

No import of its diagnostic module/runner. Original development paths only.
An augmented least-squares solve checks the archived normal-equation fit.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from safeinvest.data import load_paths

CHANNELS=('equity_real_log','bond_real_log','bill_real_log','inflation_log')
TARGETS=('next_year','remaining_to_original30_mean_annual')
OUT=ROOT/'results/model_review/forecast_assumptions_v2'
PROTOCOL=ROOT/'docs/forecast_assumption_diagnostic_protocol.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def independent_fit(x,y):
    xm=x.mean(axis=0);ym=y.mean(axis=0)
    xs=np.std(x-xm,axis=(0,1));ys=np.std(y-ym,axis=(0,1))
    xs[xs<=1e-15]=1.;ys[ys<=1e-15]=1.
    xx=((x-xm)/xs).reshape(-1,4);yy=((y-ym)/ys).reshape(-1,4)
    # Augmented system is equivalent to mean loss + fixed lambda1 ridge.
    a=np.concatenate((xx,np.sqrt(len(xx))*np.eye(4)),axis=0)
    b=np.concatenate((yy,np.zeros((4,4))),axis=0)
    beta=np.linalg.lstsq(a,b,rcond=None)[0]
    return xm,ym,xs,ys,beta


def audit_tables(returns,inflation,ids,folds,tables,findings):
    nchecks=0;max_abs_error=0.
    def check(condition,message):
        nonlocal nchecks
        nchecks+=1
        if not condition:raise AssertionError(message)
    def close(actual,expected,message):
        nonlocal max_abs_error
        aa,ee=float(actual),float(expected)
        check(np.isfinite(aa) and np.isfinite(ee),message+' finite')
        err=abs(aa-ee);max_abs_error=max(max_abs_error,err)
        check(err<=2e-11*max(1.,abs(ee)),message)
    def boolean(v):return v if isinstance(v,(bool,np.bool_)) else {'True':True,'False':False}[v]
    q=np.empty((len(ids),30,4))
    for i in range(len(ids)):
        for t in range(30):
            infl=np.log(1+inflation[i,t]);q[i,t,3]=infl
            for j in range(3):q[i,t,j]=np.log(1+returns[i,t,j])-infl
    pos={int(s):i for i,s in enumerate(ids)}
    annual={(int(r['fold']),int(r['scenario_id']),r['target'],r['channel'],int(r['decision_original_year'])):r for r in tables['annual_predictions']}
    losses={(int(r['fold']),int(r['scenario_id']),r['target'],r['channel']):r for r in tables['path_losses']}
    coeff={(int(r['fold']),r['target'],r['outcome'],r['lag_predictor']):r for r in tables['coefficients']}
    scale={(int(r['fold']),r['target'],r['channel']):r for r in tables['training_scales']}
    calendar={(int(r['fold']),r['target'],r['channel'],int(r['decision_original_year'])):r for r in tables['calendar_training_summary']}
    check(len(annual)==27840,'annual inventory unique')
    check(len(losses)==960,'path inventory unique')
    expected_losses={}
    for fold in folds:
        fid=int(fold['fold']);train=[pos[i] for i in fold['training_ids']];test=[pos[i] for i in fold['evaluation_ids']]
        x=q[:,:29,:]
        for target in TARGETS:
            y=q[:,1:,:].copy()
            if target!=TARGETS[0]:
                for t in range(29):y[:,t,:]=q[:,t+1:,:].mean(axis=1)
            xm,ym,xs,ys,beta=independent_fit(x[train],y[train])
            baseline=np.broadcast_to(ym,(24,29,4));pred=baseline+((x[test]-xm)/xs)@beta*ys
            quantile=np.quantile(y[train],[.1,.9],axis=0,method='linear')
            minimum=y[train].min(axis=0);maximum=y[train].max(axis=0)
            for j,ch in enumerate(CHANNELS):
                close(scale[fid,target,ch]['feature_residual_rms'],xs[j],'feature scale')
                close(scale[fid,target,ch]['outcome_residual_rms'],ys[j],'outcome scale')
                for k,predictor in enumerate(CHANNELS):close(coeff[fid,target,ch,predictor]['standardized_coefficient'],beta[k,j],'augmented least-squares coefficient')
                for t in range(29):
                    row=calendar[fid,target,ch,t+2]
                    for key,value in (('training_lag_mean',xm[t,j]),('training_outcome_mean',ym[t,j]),('training_outcome_q10',quantile[0,t,j]),('training_outcome_q90',quantile[1,t,j]),('training_outcome_min',minimum[t,j]),('training_outcome_max',maximum[t,j])):close(row[key],value,key)
                for a,sid in enumerate(fold['evaluation_ids']):
                    row=losses[fid,sid,target,ch]
                    yy=y[test[a],:,j];bb=baseline[a,:,j];ss=pred[a,:,j]
                    inside=(yy>=quantile[0,:,j])&(yy<=quantile[1,:,j])
                    outside=(yy<minimum[:,j])|(yy>maximum[:,j])
                    lagoutside=(x[test[a],:,j]<x[train,:,j].min(axis=0))|(x[test[a],:,j]>x[train,:,j].max(axis=0))
                    vals={'baseline_mse':np.mean((yy-bb)**2),'state_mse':np.mean((yy-ss)**2),
                          'baseline_standardized_mse':np.mean(((yy-bb)/ys[j])**2),'state_standardized_mse':np.mean(((yy-ss)/ys[j])**2),
                          'baseline_bias':np.mean(bb-yy),'state_bias':np.mean(ss-yy),
                          'interval_10_90_coverage':np.mean(inside),'outside_training_target_range_fraction':np.mean(outside),
                          'lag_outside_training_channel_range_fraction':np.mean(lagoutside)}
                    expected_losses[fid,sid,target,ch]=vals
                    for key,value in vals.items():close(row[key],value,key)
                    for t in range(29):
                        ar=annual[fid,sid,target,ch,t+2]
                        check(int(ar['last_observed_original_year'])==t+1,'temporal feature index')
                        check(int(ar['remaining_original_years'])==29-t,'remaining original horizon')
                        for key,value in (('actual_log_quantity',yy[t]),('calendar_mean_prediction',bb[t]),('fixed_state_prediction',ss[t]),('empirical_q10',quantile[0,t,j]),('empirical_q90',quantile[1,t,j])):close(ar[key],value,key)
                        check(boolean(ar['inside_empirical_interval'])==inside[t],'inclusive interval')
                        check(boolean(ar['outside_training_range'])==outside[t],'strict range')
    for name in ('fold_summary','summary'):
        for row in tables[name]:
            group=[v for (fid,sid,target,ch),v in expected_losses.items() if target==row['target'] and ch==row['channel'] and (name=='summary' or fid==int(row['fold']))]
            check(int(row['original_path_n'])==len(group),'summary path N')
            for key in group[0]:close(row[key],np.mean([v[key] for v in group]),'summary '+key)
            b=np.mean([v['baseline_standardized_mse'] for v in group]);s=np.mean([v['state_standardized_mse'] for v in group]);diff=np.array([v['state_standardized_mse']-v['baseline_standardized_mse'] for v in group])
            close(row['state_to_calendar_mse_ratio'],s/b,'aggregate MSE ratio')
            close(row['paired_standardized_loss_difference'],diff.mean(),'paired loss difference')
            for key,value in zip(('path_loss_difference_q10','path_loss_difference_q50','path_loss_difference_q90'),np.quantile(diff,[.1,.5,.9],method='linear'),strict=True):close(row[key],value,'loss quantile')
            close(row['path_fraction_state_lower_loss'],np.mean(diff<0),'lower loss path fraction')
    for row in findings['primary_vector_results']:
        d=[]
        for sid in range(1,121):
            group=[v for (fid,s,target,ch),v in expected_losses.items() if s==sid and target==row['target'] and ch in CHANNELS[:3]]
            check(len(group)==3,'primary three asset quantities only')
            d.append([np.mean([v['baseline_standardized_mse'] for v in group]),np.mean([v['state_standardized_mse'] for v in group])])
        d=np.array(d);close(row['state_to_calendar_mse_ratio'],d[:,1].mean()/d[:,0].mean(),'primary vector ratio')
        close(row['paired_mean_loss_difference'],np.mean(d[:,1]-d[:,0]),'primary vector paired difference')
        close(row['path_fraction_state_lower_loss'],np.mean(d[:,1]<d[:,0]),'primary vector lower fraction')
    return {'numeric_checks':nchecks,'maximum_absolute_difference':max_abs_error,
            'annual_prediction_rows':len(annual),'original_path_loss_rows':len(losses),
            'independent_solver':'augmented least squares via np.linalg.lstsq; original normal-equation module and runner never imported',
            'no_trading_policy_or_tail_probability_audited':True}


def main():
    destination=OUT/'independent_audit.json'
    if destination.exists():raise FileExistsError('Preserve existing audit')
    p=json.loads(PROTOCOL.read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    check_paths={**p['execution_sources_sha256'],**p['frozen_input_sha256'],**manifest['outputs_sha256']}
    before={name:sha(ROOT/name) for name in check_paths}
    if before!=check_paths:raise AssertionError('Source/input/output integrity before audit')
    if p['data']['scenario_ids']!=list(range(1,121)) or p['data']['allow_final'] is not False:raise AssertionError('Development-only guard')
    r,pi,ids=load_paths(ROOT/p['data']['path'],p['data']['scenario_ids'],allow_final=False)
    tables={name:list(csv.DictReader((OUT/f'{name}.csv').open())) for name in manifest['row_counts']}
    findings=json.loads((OUT/'findings.json').read_text())
    report=audit_tables(r,pi,ids,p['folds'],tables,findings)
    after={name:sha(ROOT/name) for name in check_paths}
    if before!=after:raise AssertionError('Source/input/output integrity after audit')
    report.update({'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
        'auditor_sha256':sha(Path(__file__)),'protocol_sha256':sha(PROTOCOL),
        'parent_manifest_sha256':sha(OUT/'manifest.json'),
        'sources_and_artifacts_at_start':before,'sources_and_artifacts_at_end':after,
        'scenario_ids_loaded':ids.tolist(),'final_scenarios_loaded':False,
        'interpretation':'Independent numerical algorithm check, not independent human peer review or statistical risk validation.'})
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources_and_artifacts_at_start','sources_and_artifacts_at_end','scenario_ids_loaded')},indent=2))


if __name__=='__main__':main()
