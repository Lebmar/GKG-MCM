"""Independent accounting/identity audit for the frozen11failure-path pairs.

No producer, policy/execution kernel, metric implementation or selector import.
Only previous independent fee/state references and the guarded development
loader (inside main after protocol checks) may be used.  Fixed saved actions
are replayed; this does not validate the conditional forecast law or identify
causal contributions to a breach.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_fan_rollout_execution import reference_actual
from audit_review_pending_execution import independent_trade, independent_metrics

PROTOCOL = ROOT / 'docs/failure_path_mechanism_protocol.json'
PLAN = ROOT / 'docs/failure_path_mechanism_audit_plan.json'
OUT = ROOT / 'results/model_review/failure_path_mechanisms_v1'
TOL = 1e-12
H, LOSS, COST = 30, .4, .0025
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def readj(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream: return json.load(stream)
    return json.loads(path.read_text())


def null(value):
    return value is None or isinstance(value, (float, np.floating)) and np.isnan(value)


def first_difference(a, b, *, start_year=1, atol=TOL):
    """Original-clock first difference; None if equal, not replacement year0."""
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or x.ndim < 1 or not len(x):
        raise ValueError('Identical nonempty yearly shapes required')
    if isinstance(start_year, bool) or not isinstance(start_year, int) or start_year < 0:
        raise ValueError('Nonnegative original starting year required')
    if not np.isfinite(atol) or atol < 0: raise ValueError('Finite nonnegative atol required')
    if x.dtype.kind in 'USb' or y.dtype.kind in 'USb':
        differ = x != y
    else:
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Unavailable/NULL states cannot be a numerical divergence')
        differ = np.abs(x - y) > atol
    if differ.ndim > 1: differ = np.any(differ, axis=tuple(range(1, differ.ndim)))
    changed = np.flatnonzero(differ)
    return int(changed[0]) + start_year if len(changed) else None


def first_crossing(running_mdd, loss=LOSS):
    values = np.asarray(running_mdd, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite original endpoint history including endpoint0 required')
    if values[0] != 0 or np.any(np.diff(values) < -TOL):
        raise ValueError('Running MDD starts at0 and cannot reset')
    events = np.flatnonzero(values - loss > TOL)
    return int(events[0]) if len(events) else None


def replay_recorded_actions(returns, inflation, target, actions, *, loss=LOSS, cost_rate=COST):
    """Full independent ledger of one saved M policy or genuine own-history B."""
    action_list = list(actions)
    if len(action_list) != len(returns) or not action_list or action_list[0] != 'restore':
        raise ValueError('Original full horizon and mandatory initial restore required')
    if any(a not in ('restore', 'hold') for a in action_list): raise ValueError('Unknown saved action')
    def fixed(t, x, cpi, peak, worst, bad):
        action = action_list[t]
        if bad and action != 'hold': raise AssertionError('Absorbing pending cannot trade or resume')
        return action
    return reference_actual(returns, inflation, target, loss, fixed, cost_rate=cost_rate)


def replay_own_B(returns, inflation, target, *, loss=LOSS, cost_rate=COST):
    return reference_actual(returns, inflation, target, loss,
        lambda t, x, cpi, peak, worst, bad: 'hold' if bad else 'restore', cost_rate=cost_rate)


def paired_terms(pre_model, pre_B, post_model, post_B, fee_model, fee_B, gross, *, endpoint_cpi):
    """Separate exact algebra with common calendar CPI; no causal attribution."""
    xm, xb, pm, pb, g = [np.asarray(v, float) for v in (pre_model, pre_B, post_model, post_B, gross)]
    if any(v.shape != (3,) for v in (xm, xb, pm, pb, g)):
        raise ValueError('Three asset balances and joint gross returns required')
    if not np.isfinite(np.concatenate((xm, xb, pm, pb, g))).all() or np.any(g <= 0):
        raise ValueError('Finite balances and positive gross required')
    vm, vb, wm, wb = map(float, (sum(xm), sum(xb), sum(pm), sum(pb)))
    fm, fb, cp = map(float, (fee_model, fee_B, endpoint_cpi))
    if min(vm, vb, wm, wb, cp) <= 0 or not np.isfinite([fm, fb, cp]).all() or min(fm, fb) < 0:
        raise ValueError('Positive account/CPI and nonnegative finite fees required')
    if abs(vm - fm - wm) > 3e-10 or abs(vb - fb - wb) > 3e-10:
        raise ValueError('Posttrade balances must be exactly self-financing after one fee')
    gm, gb = float(np.dot(pm / wm, g)), float(np.dot(pb / wb, g))
    prior_gap = vm - vb
    inherited = prior_gap * gb
    fee_term = (fb - fm) * gb
    exposure = wm * (gm - gb)
    end_m, end_b = float(np.dot(pm, g)), float(np.dot(pb, g))
    fee_log = math.log1p(-fm / vm) - math.log1p(-fb / vb)
    exposure_log = math.log(gm) - math.log(gb)
    prior_log = math.log(vm / vb)
    end_log = math.log(end_m / end_b)
    return dict(model_posttrade_gross=gm, B_posttrade_gross=gb,
        inherited_nominal_gap_contribution=inherited, fee_nominal_gap_contribution=fee_term,
        exposure_nominal_gap_contribution=exposure, end_nominal_gap=end_m-end_b,
        nominal_identity_residual=(end_m-end_b)-(inherited+fee_term+exposure),
        direct_asset_gap_recursion=prior_gap-(fm-fb)+float(np.dot(pm-pb,g-1)),
        end_real_gap=(end_m-end_b)/cp, inherited_real_gap_contribution=inherited/cp,
        fee_real_gap_contribution=fee_term/cp, exposure_real_gap_contribution=exposure/cp,
        prior_log_wealth_gap=prior_log, end_log_wealth_gap=end_log,
        log_wealth_gap_increment=end_log-prior_log, log_fee_survival_gap=fee_log,
        log_exposure_gap=exposure_log, log_identity_residual=(end_log-prior_log)-(fee_log+exposure_log))


def independent_score(qh, qr, kh, kr, n, eta, *, horizon=H):
    if (isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
        or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0<=k<=n for k in (kh, kr))):
        raise ValueError('Observed integer fan and event counts required')
    if not np.isfinite([qh, qr]).all(): raise ValueError('Finite net total remaining logs required')
    if eta is None:
        return dict(score=None, action='restore' if kr<kh or kr==kh and qr-qh>TOL else 'hold')
    if isinstance(eta, bool) or not np.isfinite(eta) or eta<0: raise ValueError('Technical nonnegative eta required')
    score=float(qr-qh-horizon*eta*(kr-kh)/n)
    return dict(score=score, action='restore' if score>TOL else 'hold')


class Checks:
    def __init__(self): self.n=0; self.maximum_errors={}
    def check(self, condition, label):
        self.n+=1
        if not condition: raise AssertionError(label)
    def close(self, actual, expected, label, *, atol=3e-10, rtol=3e-13):
        a,b=np.asarray(actual),np.asarray(expected)
        self.check(a.shape==b.shape,label+'/shape')
        if a.dtype.kind in 'bUS' or b.dtype.kind in 'bUS':
            self.check(np.array_equal(a,b),label+'/exact');return
        err=np.abs(a.astype(float)-b.astype(float));finite=np.isfinite(err)
        field=label.split('/')[-1]
        self.maximum_errors[field]=max(self.maximum_errors.get(field,0.),float(err[finite].max(initial=0)))
        self.check(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),label+'/numeric')
    def scalar(self, actual, expected, label):
        if null(expected): self.check(null(actual),label+'/NULL')
        elif isinstance(expected,(bool,np.bool_)): self.check(isinstance(actual,(bool,np.bool_)) and actual==expected,label+'/bool')
        elif isinstance(expected,str): self.check(actual==expected,label+'/text')
        else:self.close(actual,expected,label)



ASSETS=('equity','bond','bill')
STATE_FIELDS=('nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd','cpi','ever_breached')
Q_FLOAT_FIELDS=('qg_hold','qg_restore','predicted_score','predicted_growth_advantage_restore')
Q_COUNT_FIELDS=('qbad_hold_count','qbad_restore_count','forecast_n','predicted_breach_count_delta_restore')


def derived_reference(reference):
    r={k:np.asarray(v).copy() for k,v in reference.items()}
    r['posttrade_weights']=r['posttrade_holdings']/r['posttrade_holdings'].sum(axis=1)[:,None]
    r['pretrade_holdings']=r['holdings'][:-1]
    r['pretrade_weights']=r['weights'][:-1]
    r['real_log_growth']=np.log(r['real_wealth'][1:]/r['real_wealth'][:-1])
    r['first_crossing_value']=None if int(reference['first_crossing'])<0 else int(reference['first_crossing'])
    return r


def audit_endpoints(frame,ref,checks,label):
    checks.check(len(frame)==H+1 and sorted(frame.endpoint.tolist())==list(range(H+1)),label+'/31unique endpoints')
    for row in frame.sort_values('endpoint').to_dict('records'):
        t=int(row['endpoint'])
        for field in STATE_FIELDS:checks.scalar(row[field],ref[field][t],label+'/'+field)
        checks.scalar(row['remaining_threshold_margin'],LOSS-ref['running_mdd'][t],label+'/margin')
        for j,asset in enumerate(ASSETS):
            checks.scalar(row['holding_'+asset],ref['holdings'][t,j],label+'/holding_'+asset)
            checks.scalar(row['weight_'+asset],ref['weights'][t,j],label+'/weight_'+asset)


def audit_annual(frame,ref,source,is_model,eta,fan_n,market,checks,label):
    checks.check(len(frame)==H and sorted(frame.year.tolist())==list(range(1,H+1)),label+'/30unique original years')
    for row in frame.sort_values('year').to_dict('records'):
        t=int(row['year'])-1;active=is_model and t>0 and not bool(ref['pending_at_decision'][t])
        values=dict(action=str(ref['actions'][t]),nominal_fee=ref['fees'][t],nominal_moved=ref['moved'][t],
            fee_fraction=ref['fees'][t]/ref['nominal_wealth'][t],pending_at_decision=bool(ref['pending_at_decision'][t]),
            actual_trade=bool(ref['trades'][t]),target_execution=bool(ref['target_execution'][t]),
            gross_portfolio_factor=float(np.dot(ref['posttrade_weights'][t],1+market[t])),
            real_log_growth=ref['real_log_growth'][t],first_crossing_this_year=ref['first_crossing_value']==t+1,
            q_available=bool(active))
        for key,value in values.items():checks.scalar(row[key],value,label+'/'+key)
        for j,asset in enumerate(ASSETS):
            checks.scalar(row['posttrade_holding_'+asset],ref['posttrade_holdings'][t,j],label+'/post_holding_'+asset)
            checks.scalar(row['posttrade_weight_'+asset],ref['posttrade_weights'][t,j],label+'/post_weight_'+asset)
        if active:
            for field in (*Q_FLOAT_FIELDS,*Q_COUNT_FIELDS):checks.scalar(row[field],source[field][t],label+'/copied_'+field)
            kh,kr=int(source['qbad_hold_count'][t]),int(source['qbad_restore_count'][t])
            checks.check(int(source['forecast_n'][t])==fan_n,label+'/fan N')
            gate=independent_score(source['qg_hold'][t],source['qg_restore'][t],kh,kr,fan_n,eta)
            checks.scalar(row['predicted_score'],gate['score'],label+'/independent total-log score')
            checks.check(row['action']==gate['action'],label+'/independent action score')
            checks.scalar(row['predicted_growth_advantage_restore'],source['qg_restore'][t]-source['qg_hold'][t],label+'/delta logQ')
            checks.scalar(row['predicted_breach_count_delta_restore'],kr-kh,label+'/delta bad')
        else:
            for field in (*Q_FLOAT_FIELDS,*Q_COUNT_FIELDS):checks.check(null(row[field]),label+'/unavailable Q NULL/'+field)
        if is_model:checks.scalar(row['gate_reason'],str(source['gate_reason'][t]),label+'/copied gate reason')
        elif source is not None and 'gate_reason' in source:checks.scalar(row['gate_reason'],str(source['gate_reason'][t]),label+'/B copied gate reason')


def independent_summary(m,b,paired):
    summary=dict(first_action_divergence_year=first_difference(m['actions'],b['actions']),
        model_first_crossing=m['first_crossing_value'],B_first_crossing=b['first_crossing_value'],
        model_mdd=float(m['running_mdd'][-1]),B_mdd=float(b['running_mdd'][-1]),
        model_breach=bool(m['ever_breached'][-1]),B_breach=bool(b['ever_breached'][-1]),
        terminal_real_wealth_gap=float(m['real_wealth'][-1]-b['real_wealth'][-1]),
        terminal_log_real_wealth_ratio=math.log(float(m['real_wealth'][-1]/b['real_wealth'][-1])),
        mean_annual_log_growth_gap=float(np.mean(m['real_log_growth']-b['real_log_growth'])),
        total_nominal_fee_gap=float(sum(m['fees'])-sum(b['fees'])),
        total_log_fee_drag_gap=float(sum(row['log_fee_drag_gap'] for row in paired)),
        total_log_posttrade_gross_gap=float(sum(row['log_posttrade_gross_gap'] for row in paired)))
    specs={
      'first_pretrade_holdings_divergence_year':('pretrade_holdings',1e-10,1),
      'first_pretrade_weights_divergence_year':('pretrade_weights',1e-12,1),
      'first_nominal_fee_divergence_year':('fees',1e-10,1),
      'first_posttrade_holdings_divergence_year':('posttrade_holdings',1e-10,1),
      'first_posttrade_weights_divergence_year':('posttrade_weights',1e-12,1),
      'first_endpoint_holdings_divergence_endpoint':('holdings',1e-10,0),
      'first_nominal_wealth_divergence_endpoint':('nominal_wealth',1e-10,0),
      'first_real_wealth_divergence_endpoint':('real_wealth',1e-10,0),
      'first_real_peak_divergence_endpoint':('real_peak',1e-10,0),
      'first_current_drawdown_divergence_endpoint':('current_drawdown',1e-12,0),
      'first_running_mdd_divergence_endpoint':('running_mdd',1e-12,0)}
    for name,(field,tol,start) in specs.items():summary[name]=first_difference(m[field],b[field],start_year=start,atol=tol)
    return summary


def audit_pair_rows(frame,m,b,market,inflation,checks,label):
    checks.check(len(frame)==H and sorted(frame.year.tolist())==list(range(1,H+1)),label+'/330complete paired years')
    expected=[]
    for t in range(H):
        end=t+1;fm,fb=float(m['fees'][t]),float(b['fees'][t])
        z=paired_terms(m['holdings'][t],b['holdings'][t],m['posttrade_holdings'][t],b['posttrade_holdings'][t],
            fm,fb,1+market[t],endpoint_cpi=m['cpi'][end])
        checks.close(z['nominal_identity_residual'],0.,label+'/independent nominal residual')
        checks.close(z['log_identity_residual'],0.,label+'/independent log residual',atol=2e-12,rtol=0)
        checks.close(z['direct_asset_gap_recursion'],z['end_nominal_gap'],label+'/independent direct P&L recursion')
        rm,rb,pm,pb=[float(account[field][end]) for account,field in ((m,'real_wealth'),(b,'real_wealth'),(m,'real_peak'),(b,'real_peak'))]
        dd=float(m['current_drawdown'][end]-b['current_drawdown'][end])
        checks.close(dd,rb/pb-rm/pm,label+'/independent DD ratio',atol=2e-12,rtol=0)
        row=dict(year=end,model_action=str(m['actions'][t]),B_action=str(b['actions'][t]),
            action_differs=bool(m['actions'][t]!=b['actions'][t]),
            nominal_wealth_gap_start=float(m['nominal_wealth'][t]-b['nominal_wealth'][t]),
            nominal_wealth_gap_end=z['end_nominal_gap'],nominal_fee_gap=fm-fb,
            nominal_return_pnl_gap=float(np.dot(m['posttrade_holdings'][t]-b['posttrade_holdings'][t],market[t])),
            inherited_wealth_gap_at_B_growth=z['inherited_nominal_gap_contribution'],
            fee_gap_at_B_growth=z['fee_nominal_gap_contribution'],exposure_gap_at_model_postcapital=z['exposure_nominal_gap_contribution'],
            real_wealth_gap_end=rm-rb,real_peak_gap_end=pm-pb,
            log_real_wealth_ratio_start=z['prior_log_wealth_gap'],log_real_wealth_ratio_end=z['end_log_wealth_gap'],
            annual_real_log_growth_gap=float(m['real_log_growth'][t]-b['real_log_growth'][t]),
            log_fee_drag_gap=z['log_fee_survival_gap'],log_posttrade_gross_gap=z['log_exposure_gap'],
            current_drawdown_gap_end=dd,current_DD_wealth_ratio_part=(rb-rm)/pb,
            current_DD_peak_ratio_part=rm*(1/pb-1/pm),running_mdd_gap_end=float(m['running_mdd'][end]-b['running_mdd'][end]),
            model_first_crossing_this_year=m['first_crossing_value']==end,B_first_crossing_this_year=b['first_crossing_value']==end,
            inflation=float(inflation[t]),**{'market_return_'+asset:float(market[t,j]) for j,asset in enumerate(ASSETS)})
        actual=frame.loc[frame.year.eq(end)].iloc[0].to_dict()
        for key,value in row.items():checks.scalar(actual[key],value,label+'/'+key)
        expected.append(row)
    checks.close(sum(x['log_fee_drag_gap']+x['log_posttrade_gross_gap'] for x in expected),
        math.log(m['real_wealth'][-1]/b['real_wealth'][-1]),label+'/whole path telescoping log',atol=2e-12,rtol=0)
    return independent_summary(m,b,expected)


def audit_crossing_witnesses(frame,m,b,checks,label):
    endpoints=sorted({e for ref in (m,b) if ref['first_crossing_value'] is not None
                       for e in (ref['first_crossing_value']-1,ref['first_crossing_value'])})
    checks.check(len(frame)==2*len(endpoints),label+'/all own/B crossing and preceding witnesses')
    for endpoint in endpoints:
        for arm,ref in (('model',m),('B',b)):
            rows=frame.loc[frame.arm.eq(arm)&frame.endpoint.eq(endpoint)]
            checks.check(len(rows)==1,label+'/unique crossing witness')
            row=rows.iloc[0].to_dict();tau=ref['first_crossing_value']
            relation='no_crossing' if tau is None else 'crossing' if endpoint==tau else 'pre_crossing' if endpoint==tau-1 else 'other_arm_witness'
            checks.scalar(row['relation_to_own_crossing'],relation,label+'/relation')
            for field in STATE_FIELDS:checks.scalar(row[field],ref[field][endpoint],label+'/witness '+field)
            checks.scalar(row['remaining_threshold_margin'],LOSS-ref['running_mdd'][endpoint],label+'/witness margin')
            checks.scalar(row['preceding_action'],None if endpoint==0 else str(ref['actions'][endpoint-1]),label+'/preceding action')
            checks.scalar(row['next_action'],None if endpoint==H else str(ref['actions'][endpoint]),label+'/next action')
            checks.scalar(row['next_pending'],None if endpoint==H else bool(ref['pending_at_decision'][endpoint]),label+'/next pending')
            for j,asset in enumerate(ASSETS):
                checks.scalar(row['holding_'+asset],ref['holdings'][endpoint,j],label+'/witness holding_'+asset)
                checks.scalar(row['weight_'+asset],ref['weights'][endpoint,j],label+'/witness weight_'+asset)

def audit_cross_stage(items,refs,checks):
    by_pair={x['pair_id']:x for x in items}
    rows=readj(OUT/'cross_stage_summary.json')
    table=pd.read_csv(OUT/'cross_stage_summary.csv.gz',float_precision='round_trip')
    checks.check(len(rows)==4 and {x['scenario_id'] for x in rows}=={35,52,98,112},'4distinct cross-stage JSON identities')
    checks.check(len(table)==4 and set(table.scenario_id)=={35,52,98,112},'4distinct cross-stage CSV identities')
    lookup={x['scenario_id']:x for x in rows}
    for sid in (35,52,98,112):
        full,inner=by_pair['full_'+str(sid)],by_pair['inner_'+str(sid)]
        f,i=refs[full['pair_id']],refs[inner['pair_id']]
        expected=dict(scenario_id=sid,full_pair_id=full['pair_id'],inner_pair_id=inner['pair_id'],
            full_job_id=full['job_id'],inner_job_id=inner['job_id'],
            full_target=json.dumps(full['target'],separators=(',',':')),inner_target=json.dumps(inner['target'],separators=(',',':')),
            full_training_ids=json.dumps(full['training_ids'],separators=(',',':')),inner_training_ids=json.dumps(inner['training_ids'],separators=(',',':')),
            full_training_fan_n=full['training_fan_n'],inner_training_fan_n=inner['training_fan_n'],
            full_candidate_order=full['selected_candidate_order'],inner_candidate_order=inner['selected_candidate_order'],
            full_annual_price=PRICES[full['selected_candidate_order']],inner_annual_price=PRICES[inner['selected_candidate_order']],
            target_changed=full['target']!=inner['target'],fan_changed=full['training_ids']!=inner['training_ids'],
            price_changed=full['selected_candidate_order']!=inner['selected_candidate_order'],
            full_evaluation_in_training=sid in full['training_ids'],inner_evaluation_in_training=sid in inner['training_ids'])
        differences=independent_summary(f,i,[])
        expected.update({key:value for key,value in differences.items() if key.startswith('first_')})
        for stage,ref in (('full',f),('inner',i)):
            expected.update({stage+'_first_crossing':ref['first_crossing_value'],stage+'_mdd':float(ref['running_mdd'][-1]),
                stage+'_terminal_real':float(ref['real_wealth'][-1]),stage+'_mean_annual_log_growth':math.log(float(ref['real_wealth'][-1]/ref['real_wealth'][0]))/H,
                stage+'_nominal_fees':float(sum(ref['fees']))})
        csv=table.loc[table.scenario_id.eq(sid)].iloc[0].to_dict()
        for kind,row in (('JSON',lookup[sid]),('CSV',csv)):
            for field,value in expected.items():checks.scalar(row[field],value,str(sid)+'/cross-stage '+kind+'/'+field)


META_FIELDS=('pair_id','stage','scenario_id','folder','job_id','context','repeat_id','selected_candidate_order','selected_beta','training_fan_n')
EXPECTED_PAIRS={'full_35','full_52','full_98','full_112','inner_35','inner_52','inner_98','inner_112','outer_r2_52','outer_r4_112','outer_r5_112'}


def validate_inventory(p):
    if p.get('execution_ready') is not True:raise RuntimeError('Frozen execution-ready protocol required')
    if (p['data']['scenario_ids']!=list(range(1,121)) or p['data']['allow_final'] is not False
        or p['original_H']!=H or p['loss_tolerance']!=LOSS or p['cost_rate']!=COST
        or p['output_directory']!=str(OUT.relative_to(ROOT))):raise ValueError('Original guarded development/accounting contract required')
    items=p['inventory']
    if len(items)!=11 or {x['pair_id'] for x in items}!=EXPECTED_PAIRS:raise ValueError('Exactly11 frozen pair identities required')
    for x in items:
        stage=x['stage'];sid=x['scenario_id'];train=x['training_ids'];full=stage=='full_fixed_training';inner=stage=='selected_beta_inner'
        if stage not in ('full_fixed_training','selected_beta_inner','outer_oof'):raise ValueError('Unknown fit stage')
        if (isinstance(sid,bool) or not isinstance(sid,int) or sid not in (35,52,98,112)
            or len(set(train))!=len(train) or len(train)!=x['training_fan_n'] or len(train)!=(120 if full else 90 if inner else 96)
            or any(isinstance(i,bool) or not isinstance(i,int) or not 1<=i<=120 for i in train)):
            raise ValueError('Original unique development identities and own fan size required')
        if ((sid in train)!=full or (x['B_group'] is None)!=inner or (x['trace_scope']=='validation')!=inner
            or x['selected_candidate_order'] not in (4,5)
            or x['folder']!=('refit_stability_v1' if stage=='outer_oof' else 'joint_liquidity_refit_v1')):
            raise ValueError('Stage/source/training inclusion contract required')
    return items


def snapshot(p,m):
    paths={PROTOCOL,PLAN,OUT/'manifest.json',Path(__file__).resolve(),ROOT/'tests/test_audit_failure_path_mechanisms.py',
        ROOT/'scripts/audit_fan_rollout_execution.py',ROOT/'scripts/audit_review_pending_execution.py'}
    for key in ('execution_sources_sha256','frozen_input_sha256'):
        for rel,digest in p[key].items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts or len(digest)!=64:raise ValueError('Safe repository-relative SHA entries required')
            if sha(ROOT/rel)!=digest:raise AssertionError('Frozen protocol dependency changed: '+rel)
            paths.add(ROOT/rel)
    for rel,digest in m['outputs_sha256'].items():
        if Path(rel).name!=rel or sha(OUT/rel)!=digest:raise AssertionError('Producer output hash changed: '+rel)
        paths.add(OUT/rel)
    return {str(path.relative_to(ROOT)):sha(path) for path in sorted(paths)}


def source_account(archive,group,sid):
    index=group['scenario_ids'].index(sid);prefix=group['group_name']+'__'
    result={}
    for key in archive.files:
        if not key.startswith(prefix):continue
        field=key[len(prefix):];values=archive[key]
        result[field]=values.copy() if field in ('targets','annual_prices','loss_tolerances') else values[index].copy()
    if not result:raise ValueError('Frozen parent account missing')
    return result


def identity_check(item,jobs,bundles,selections,groups,checks):
    label=item['pair_id'];job=jobs[item['job_id']];bundle=bundles[item['job_id']]
    checks.check(job['training_ids']==item['training_ids'],label+'/own original training IDs')
    checks.check(item['scenario_id'] in job['evaluation_ids'] or item['stage']=='full_fixed_training' and item['scenario_id'] in job['training_ids'],label+'/own evaluation identity')
    checks.check(job['parent_context']==item['context'],label+'/job context')
    checks.check(bundle['context']['training_ids']==item['training_ids'],label+'/own static bundle fan IDs')
    product=bundle['q2_case']['product']
    checks.close(product['weights'],item['target'],label+'/own target weights',atol=0,rtol=0)
    checks.scalar(product['candidate_local_index'],item['target_local_id'],label+'/own local target ID')
    checks.scalar(product['candidate_original232_index'],item['target_original232_id'],label+'/own original232 target ID')
    checks.scalar(bundle['refit']['selection']['selected_gamma'],item['static_gamma'],label+'/own static gamma')
    selected=selections[item['context']]['methods']['joint_tie_set_refit_learner']
    checks.scalar(selected['selected_beta'],item['selected_beta'],label+'/selected beta of original program')
    price=bundle['training_price_selections'][str(item['selected_beta'])]
    checks.scalar(price['selected_candidate_order'],item['selected_candidate_order'],label+'/original own job selected order at beta')
    for role,key in (('model',item['model_group']),('B',item['B_group'])):
        if key is None:continue
        checks.check(key in groups,label+'/'+role+' original source group')
        group=groups[key];md=group['metadata'];checks.check(item['scenario_id'] in group['scenario_ids'],label+'/'+role+' source scenario')
        for field,expected in [('job_id',item['job_id']),('context',item['context']),('selection_training_n',item['training_fan_n']),
            ('candidate_local_index',item['target_local_id']),('candidate_original232_index',item['target_original232_id']),
            ('horizon',H),('loss_tolerance',LOSS),('static_gamma',item['static_gamma'])]:checks.scalar(md[field],expected,label+'/'+role+' '+field)
        checks.scalar(group['selected_candidate_order'],item['selected_candidate_order'] if role=='model' else None,label+'/'+role+' exact selected order')


def audit_B_cache(item,ref,cache,checks):
    rows=cache.loc[cache.job_id.eq(item['job_id'])&cache.scenario_id.eq(item['scenario_id'])]
    checks.check(len(rows)==1,item['pair_id']+'/one original same-own-job B cache')
    row=rows.iloc[0].to_dict()
    for field,expected in [('policy','annual_then_pending'),('selection_training_n',item['training_fan_n']),
        ('candidate_local_index',item['target_local_id']),('candidate_original232_index',item['target_original232_id']),
        ('horizon',H),('loss_tolerance',LOSS),('assigned',True)]:checks.scalar(row[field],expected,item['pair_id']+'/cached B '+field)
    metrics=independent_metrics(ref,LOSS)
    for field in ('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','nominal_moved',
        'terminal_equity','terminal_bond','terminal_bill','breach','excess','first_crossing'):
        checks.scalar(row[field],metrics[field],item['pair_id']+'/independently reconstructed vs cached B '+field)
    checks.scalar(row['trade_fraction'],float(np.mean(ref['trades'])),item['pair_id']+'/cached B trade fraction')
    checks.scalar(row['target_execution_fraction'],float(np.mean(ref['target_execution'])),item['pair_id']+'/cached B target execution fraction')


def run():
    receipt=OUT/'independent_audit.json'
    if receipt.exists():raise RuntimeError('Refuse to overwrite an independent audit receipt')
    p=readj(PROTOCOL);items=validate_inventory(p)
    m=readj(OUT/'manifest.json')
    if (m.get('status')!='complete' or m['protocol_sha256']!=sha(PROTOCOL)
        or m['sources']!=m['sources_at_end'] or m['input_sha256']!=m['input_sha256_at_end']
        or m['sources']!=p['execution_sources_sha256'] or m['input_sha256']!=p['frozen_input_sha256']):
        raise RuntimeError('Complete producer with frozen matching source/input/protocol hashes required before new numerical output')
    before=snapshot(p,m);checks=Checks()
    for key,expected in [('final_scenarios_loaded',False),('new_fits',0),('new_model_simulations',0),('new_forecast_evaluations',0),
        ('new_reference_simulations',4),('saved_model_accounts',11),('saved_B_accounts',7),('selection_changed',False),('admission_changed',False),('current_C_service_financials_added',False)]:checks.scalar(m[key],expected,'manifest/'+key)
    checks.check(m['original_market_scenario_ids_loaded']==list(range(1,121)),'manifest/original development identity scope')
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    market,inflation,ids=load_paths(ROOT/p['data']['path'],range(1,121),allow_final=False)
    digest=hashlib.sha256()
    for arr in (ids,market,inflation):digest.update(np.ascontiguousarray(arr).tobytes())
    checks.check(np.array_equal(ids,np.arange(1,121)) and digest.hexdigest()==m['development_array_sha256'],'guarded original data arrays')
    folders=('joint_liquidity_refit_v1','refit_stability_v1')
    jobs={};bundles={};selections={};groups={}
    for folder,proto in zip(folders,('joint_liquidity_refit_protocol.json','refit_stability_protocol.json')):
        parent=ROOT/'results/model_review'/folder;pm=readj(parent/'manifest.json');pa=readj(parent/'independent_audit.json')
        checks.check(pa['status']=='passed' and pm['sources']==pm['sources_at_end'] and pm['input_sha256']==pm['input_sha256_at_end'],folder+'/completed immutable parent audited')
        protocol=readj(ROOT/'docs'/proto);jobs.update({x['job_id']:x for x in protocol['jobs']})
        bundles.update(readj(parent/('static_refits.json.gz' if folder=='refit_stability_v1' else 'static_refits.json')))
        selections.update(readj(parent/'context_selections.json'));groups[folder]=readj(parent/'trace_groups.json')
    tables={name:pd.read_csv(OUT/(name+'.csv.gz'),float_precision='round_trip') for name in ('account_endpoints','annual_states','paired_annual','paired_summary','crossing_witnesses')}
    for name,count,keys in [('account_endpoints',682,['pair_id','arm','endpoint']),('annual_states',660,['pair_id','arm','year']),
        ('paired_annual',330,['pair_id','year']),('paired_summary',11,['pair_id'])]:
        checks.check(len(tables[name])==count and not tables[name].duplicated(keys).any(),name+'/complete distinct denominator')
        checks.check(set(tables[name].pair_id)==EXPECTED_PAIRS,name+'/fixed original pair inventory')
    json_summary=readj(OUT/'paired_summary.json');presence=readj(OUT/'source_trace_presence.json')
    checks.check(len(json_summary)==11 and {x['pair_id'] for x in json_summary}==EXPECTED_PAIRS,'11complete JSON summaries')
    checks.check(len(presence)==11 and {x['pair_id'] for x in presence}==EXPECTED_PAIRS,'11complete source-presence entries')
    summary_map={x['pair_id']:x for x in json_summary};presence_map={x['pair_id']:x for x in presence}
    cache=pd.read_csv(ROOT/'results/model_review/benefit_cost_execution_v1/inner_reference_paths.csv.gz',float_precision='round_trip')
    archives={};results=[];source_replayed=0;active_Q=0;pending_periods=0;model_references={}
    try:
        for item in items:
            label=item['pair_id'];sid=item['scenario_id'];folder=item['folder'];scope=item['trace_scope'];r=market[sid-1];z=inflation[sid-1]
            gs=groups[folder][scope];identity_check(item,jobs,bundles,selections,gs,checks)
            key=(folder,scope)
            if key not in archives:archives[key]=np.load(ROOT/'results/model_review'/folder/(scope+'_traces.npz'),allow_pickle=False)
            source={}
            for arm,group_name in (('model',item['model_group']),('B',item['B_group'])):
                if group_name is None:source[arm]=None;continue
                group={**gs[group_name],'group_name':group_name}
                source[arm]=source_account(archives[key],group,sid)
                checks.close(source[arm]['targets'],item['target'],label+'/'+arm+' target metadata',atol=0,rtol=0)
                if 'loss_tolerances' in source[arm]:checks.close(source[arm]['loss_tolerances'],[LOSS],label+'/'+arm+' original loss array metadata',atol=0,rtol=0)
            model=derived_reference(replay_recorded_actions(r,z,item['target'],source['model']['actions']))
            baseline=derived_reference(replay_own_B(r,z,item['target']))
            model_references[label]=model
            if item['B_group'] is None:audit_B_cache(item,baseline,cache,checks)
            for arm,ref in (('model',model),('B',baseline)):
                src=source[arm];source_name='new_same_own_target_annual_then_pending_production_fee_reference' if src is None else 'frozen_saved_trace'
                if src is not None:
                    for field in sorted(set(src)&set(ref)):
                        checks.close(src[field],ref[field],label+'/'+arm+'/independent parent ledger '+field)
                    source_replayed+=1
                for name in ('account_endpoints','annual_states'):
                    rows=tables[name].loc[tables[name].pair_id.eq(label)&tables[name].arm.eq(arm)]
                    for field in META_FIELDS:checks.check(rows[field].eq(item[field]).all(),label+'/'+name+'/'+arm+'/metadata '+field)
                    checks.check(rows.account_source.eq(source_name).all(),label+'/'+arm+'/source availability label')
                audit_endpoints(tables['account_endpoints'].loc[tables['account_endpoints'].pair_id.eq(label)&tables['account_endpoints'].arm.eq(arm)],ref,checks,label+'/'+arm)
                audit_annual(tables['annual_states'].loc[tables['annual_states'].pair_id.eq(label)&tables['annual_states'].arm.eq(arm)],ref,src,arm=='model',PRICES[item['selected_candidate_order']],item['training_fan_n'],r,checks,label+'/'+arm)
                pending_periods+=int(np.sum(ref['pending_at_decision']))
            active_Q+=int(np.sum(~model['pending_at_decision'][1:]))
            paired=tables['paired_annual'].loc[tables['paired_annual'].pair_id.eq(label)]
            summary= audit_pair_rows(paired,model,baseline,r,z,checks,label)
            csv_summary=tables['paired_summary'].loc[tables['paired_summary'].pair_id.eq(label)].iloc[0].to_dict()
            for location,record in [('JSON',summary_map[label]),('CSV',csv_summary)]:
                for field,value in summary.items():checks.scalar(record[field],value,label+'/'+location+' summary '+field)
                for field in META_FIELDS:checks.scalar(record[field],item[field],label+'/'+location+' summary metadata '+field)
            for name in ('paired_annual','crossing_witnesses'):
                rows=tables[name].loc[tables[name].pair_id.eq(label)]
                for field in META_FIELDS:checks.check(rows[field].eq(item[field]).all(),label+'/'+name+'/metadata '+field)
            audit_crossing_witnesses(tables['crossing_witnesses'].loc[tables['crossing_witnesses'].pair_id.eq(label)],model,baseline,checks,label)
            available=presence_map[label]
            for field in META_FIELDS:checks.scalar(available[field],item[field],label+'/availability metadata '+field)
            for field,value in dict(model_group=item['model_group'],B_group=item['B_group'],model_source='frozen_saved_trace',
                B_source='new_same_own_target_annual_then_pending_production_fee_reference' if source['B'] is None else 'frozen_saved_trace',
                model_first_crossing=model['first_crossing_value'],B_first_crossing=baseline['first_crossing_value']).items():checks.scalar(available[field],value,label+'/availability '+field)
            results.append(dict(pair_id=label,scenario_id=sid,job_id=item['job_id'],model_first_crossing=model['first_crossing_value'],B_first_crossing=baseline['first_crossing_value'],
                first_action_divergence_year=summary['first_action_divergence_year']))
    finally:
        for archive in archives.values():archive.close()
    audit_cross_stage(items,model_references,checks)
    after=snapshot(p,m);checks.check(before==after,'source/input/output/protocol immutable throughout audit')
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks_passed=checks.n,
        maximum_absolute_errors=checks.maximum_errors,maximum_numeric_absolute_error=max(checks.maximum_errors.values(),default=0.),
        parent_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(PLAN),
        auditor_sha256=sha(Path(__file__)),test_sha256=sha(ROOT/'tests/test_audit_failure_path_mechanisms.py'),
        pairs_checked=11,cross_stage_same_original_IDs_checked=4,accounts_independently_replayed=22,parent_source_accounts_independently_replayed=source_replayed,
        inner_B_accounts_independently_reconstructed_and_cached_endpoints_checked=4,annual_account_periods_checked=660,
        account_endpoints_checked=682,paired_annual_records_checked=330,active_saved_Q_score_algebra_checked=active_Q,
        pending_account_periods_checked=pending_periods,full_fan_Q_recomputed=0,final_scenarios_loaded=False,
        audit_development_history=['First auditor invocation stopped before receipt creation because a legacy B trace has no optional loss_tolerances NPZ array. Original loss was already checked against immutable group metadata. Auditor adapter was corrected to check the optional array when present; producer/protocol/results were unchanged.'],
        independent_method='Previous separate100-step fee bisection and scalar accounting; all model actions fixed from immutable parent traces, own-B actions recomputed from own history; new producer never imported.',
        inference_limits=['Mechanical realized-path accounting only; no new selection, admission, conditional-law validation, complete independent adaptive action optimization, or risk certification.',
            'First divergence and ordered wealth/fee/exposure decomposition do not identify causal forecast/eta/target/fan contributions.',
            'Original IDs repeat across fits;22accounts/11pairs are not22or11independent market samples.'],
        sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,pair_results=results)
    receipt.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','accounts_independently_replayed','paired_annual_records_checked','maximum_numeric_absolute_error')}))
    return result


if __name__ == '__main__': run()
