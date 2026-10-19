"""Independent all-member scalar-fan audit; never imports contrast producer.

The originalcase matching, sharedstate, fans, score algebra and archived
outcomes are rebuilt separately.  Every currentaction/forecastmember branch
uses the earlier independent fee/state scalar_fan. This is one fixed-state
model contrast, not population risk validation or completepolicy replay.
"""
from __future__ import annotations
from datetime import datetime,timezone
import gzip,hashlib,json,math
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_fan_rollout_execution import scalar_fan
PROTOCOL=ROOT/'docs/matched_parameter_fan_contrast_protocol.json'
PLAN=ROOT/'docs/matched_parameter_fan_contrast_audit_plan.json'
OUT=ROOT/'results/model_review/matched_parameter_fan_contrast_v1'
PRICES=(0.,.001,.003,.01,.03,.1,.3,1.,None)
KERNELS=('research/experimental_execution/risk_priced_fan_rollout.py','research/experimental_execution/fan_rollout.py','research/experimental_execution/review_pending.py','src/safeinvest/core.py')
STATE=('holdings','weights','nominal_wealth','real_wealth','cpi','real_peak','current_drawdown','running_mdd','ever_breached')
PREVIOUS=('posttrade_holdings','fees','moved','target_execution','pending_at_decision','trades')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(path.read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def null(v):return v is None or isinstance(v,(float,np.floating)) and np.isnan(v)


class Checks:
    def __init__(self):self.n=0;self.maxerr=0.
    def check(self,c,label):
        self.n+=1
        if not c:raise AssertionError(label)
    def value(self,a,b,label,atol=2e-12,rtol=0):
        if null(b):self.check(null(a),label+'/NULL');return
        if isinstance(b,(str,dict)) or isinstance(b,list) and any(isinstance(x,(str,dict)) for x in b):self.check(a==b,label);return
        x,y=np.asarray(a),np.asarray(b);self.check(x.shape==y.shape,label+'/shape')
        if x.dtype.kind in 'bUS' or y.dtype.kind in 'bUS':self.check(np.array_equal(x,y),label+'/exact');return
        d=np.abs(x.astype(float)-y.astype(float));self.maxerr=max(self.maxerr,float(d[np.isfinite(d)].max(initial=0)))
        self.check(np.allclose(x,y,atol=atol,rtol=rtol,equal_nan=True),label+'/numeric')


def metadata_cases(parent,manifests):
    out=[]
    for item in parent['inventory']:
        pm=manifests[item['folder']]
        out.append({**item,'original_H':parent['original_H'],'loss_tolerance':parent['loss_tolerance'],'cost_rate':parent['cost_rate'],
            'annual_eta':PRICES[item['selected_candidate_order']],'kernel_sha256':{x:pm['sources'][x] for x in KERNELS},'development_arrays_sha256':pm['development_arrays_sha256']})
    return out


def metadata_pairs(cases):
    fields=('scenario_id','target','annual_eta','original_H','loss_tolerance','cost_rate','kernel_sha256','development_arrays_sha256')
    return [dict(contrast_id=a['pair_id']+'__vs__'+b['pair_id'],left_case_id=a['pair_id'],right_case_id=b['pair_id'])
        for a,b in combinations(sorted(cases,key=lambda r:r['pair_id']),2) if all(a[k]==b[k] for k in fields) and a['training_ids']!=b['training_ids']]


def fixed_fans(left,right):
    full,outer,sid=left['training_ids'],right['training_ids'],left['scenario_id']
    if full!=list(range(1,121)) or len(outer)!=96 or len(set(outer))!=96 or sid not in full or sid in outer or not set(outer)<set(full):raise ValueError('Exact full120/one-self-delete119/ownouter96 bridge required')
    return [('full120',full,'left'),('full_without_actual119',[i for i in full if i!=sid],None),('outer96',outer,'right')]


def fixed_gate(qh,qr,kh,kr,n,eta):
    if any(isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<=n for k in (kh,kr)) or not np.isfinite([qh,qr,eta]).all():raise ValueError('Finite logs/eta and integer badcounts required')
    score=qr-qh-30*eta*(kr-kh)/n;positive=score>1e-12
    return dict(predicted_score=score,chosen_action='restore' if positive else 'hold',gate_reason='finite_score_positive' if positive else 'finite_score_nonpositive')


def extract(z,group,meta,sid):
    i=meta['scenario_ids'].index(sid);prefix=group+'__'
    return {key[len(prefix):]:(z[key].copy() if key[len(prefix):] in ('targets','annual_prices','loss_tolerances') else z[key][i].copy()) for key in z.files if key.startswith(prefix)}


def outcome(a):
    log=math.log(float(a['real_wealth'][-1]/a['real_wealth'][0]));tau=int(a['first_crossing'])
    return dict(terminal_real_wealth=float(a['real_wealth'][-1]),terminal_nominal_wealth=float(a['nominal_wealth'][-1]),total_net_log_real_growth=log,
        annual_log_real_growth=log/30,cagr=math.expm1(log/30),original_H_MDD=float(a['running_mdd'][-1]),ever_breached=bool(a['ever_breached'][-1]),
        first_crossing_year=None if tau<0 else tau,nominal_fees=float(sum(a['fees'])),trade_fraction_including_initial_and_pending=float(np.mean(a['trades'])))


def snapshot(p,m):
    result={str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL),str(PLAN.relative_to(ROOT)):sha(PLAN),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        'tests/test_audit_matched_parameter_fan_contrast.py':sha(ROOT/'tests/test_audit_matched_parameter_fan_contrast.py'),
        'scripts/audit_fan_rollout_execution.py':sha(ROOT/'scripts/audit_fan_rollout_execution.py'),'scripts/audit_review_pending_execution.py':sha(ROOT/'scripts/audit_review_pending_execution.py')}
    for name,digest in {**p['frozen_input_sha256'],**p['execution_sources_sha256']}.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT/name)!=digest:raise RuntimeError('Frozen fan diagnostic dependency changed: '+name)
        result[name]=digest
    for name,digest in m['output_sha256'].items():
        if Path(name).name!=name or sha(OUT/name)!=digest:raise RuntimeError('Frozen fan output changed: '+name)
        result[str((OUT/name).relative_to(ROOT))]=digest
    result[str((OUT/'manifest.json').relative_to(ROOT))]=sha(OUT/'manifest.json')
    return result


def run():
    receipt=OUT/'independent_audit.json'
    if receipt.exists():raise RuntimeError('Never overwrite passed independent fan audit')
    p=readj(PROTOCOL)
    if not p.get('execution_ready') or not p.get('execution_frozen_utc'):raise RuntimeError('Wait for frozen fan protocol')
    m=readj(OUT/'manifest.json')
    if m['status']!='passed' or m['protocol_sha256']!=sha(PROTOCOL) or m['sources_and_inputs_sha256']!={**p['frozen_input_sha256'],**p['execution_sources_sha256']}:raise RuntimeError('Complete passed producer with matching hashes required before new values')
    before=snapshot(p,m);c=Checks()
    c.check(p['data']['scenario_ids']==list(range(1,121)) and p['data']['allow_final'] is False and (p['original_H'],p['loss_tolerance'],p['cost_rate'])==(30,.4,.0025),'original guarded fan accounting')
    for key,value in [('metadata_selection_uses_outcomes',False),('new_actual_policy_replays',0),('new_fits',0),('final_scenarios_read',False)]:c.value(m[key],value,'manifest/'+key)
    parent=readj(ROOT/p['parent_protocol']);manifests={f:readj(ROOT/'results/model_review'/f/'manifest.json') for f in p['parent_folders']}
    cases=metadata_cases(parent,manifests);selected=metadata_pairs(cases)
    c.check(len(cases)==11 and selected==p['eligible_pairs']==m['eligible_pairs'],'all55metadata-pair matching')
    c.value(canonical(parent['inventory']),p['parent_inventory_canonical_sha256'],'unchanged11case canonical inventory')
    by_case={x['pair_id']:x for x in cases};groups={f:readj(ROOT/'results/model_review'/f/'trace_groups.json')['actual'] for f in p['parent_folders']}
    static={f:readj(ROOT/'results/model_review'/f/p['static_files'][f]) for f in p['parent_folders']}
    pairs=readj(OUT/'pairs.json');c.check(len(pairs)==len(selected),'all eligible pairs retained')
    pairmap={x['contrast_id']:x for x in pairs};qrows=pd.read_csv(OUT/'same_state_Q.csv',float_precision='round_trip');members=pd.read_csv(OUT/'fan_members.csv',float_precision='round_trip')
    c.check(len(qrows)==3*len(selected) and not qrows.duplicated(['contrast_id','fan_label']).any(),'all3distinct Qfans')
    sys.path.insert(0,str(ROOT/'src'))
    from safeinvest.data import load_paths
    r,pi,ids=load_paths(ROOT/p['data']['path'],range(1,121),allow_final=False)
    digest=hashlib.sha256()
    for array in (ids,r,pi):digest.update(np.ascontiguousarray(array).tobytes())
    c.check(np.array_equal(ids,np.arange(1,121)) and m['original_scenario_ids_loaded']==list(range(1,121)) and digest.hexdigest()==m['development_arrays_sha256'],'guarded original arrays')
    archives={f:np.load(ROOT/'results/model_review'/f/'actual_traces.npz',allow_pickle=False) for f in p['parent_folders']};expected_member_n=0;results=[]
    try:
      with np.load(OUT/'fan_branch_arrays.npz',allow_pickle=False) as saved:
       for identity in selected:
        pid=identity['contrast_id'];pair=pairmap[pid];left,right=by_case[identity['left_case_id']],by_case[identity['right_case_id']];accounts={}
        c.check(pair['left_case']==left and pair['right_case']==right,pid+'/complete own metadata')
        for side,item in [('left',left),('right',right)]:
            meta=groups[item['folder']][item['model_group']];bundle=static[item['folder']][item['job_id']]
            c.check(meta['metadata']['job_id']==item['job_id'] and meta['selected_candidate_order']==item['selected_candidate_order'],pid+'/'+side+' source group')
            c.check(bundle['refit']['training_ids']==item['training_ids'] and bundle['q2_case']['product']['weights']==item['target'] and bundle['training_price_selections'][str(item['selected_beta'])]['selected_candidate_order']==item['selected_candidate_order'],pid+'/'+side+' own fit/fan/target/price')
            a=extract(archives[item['folder']],item['model_group'],meta,item['scenario_id']);accounts[side]=a
            c.value(a['targets'],item['target'],pid+'/'+side+' target',atol=0);c.value(a['annual_prices'],[item['annual_eta']],pid+'/'+side+' eta',atol=0);c.value(a['loss_tolerances'],[.4],pid+'/'+side+' L',atol=0)
            for field,value in outcome(a).items():c.value(pair['original_full_horizon_outcomes'][side][field],value,pid+'/'+side+' source outcome '+field)
        A,B=accounts['left'],accounts['right'];different=np.flatnonzero(A['actions']!=B['actions']);t=int(different[0]) if len(different) else None
        c.value(pair['tau_year'],None if t is None else t+1,pid+'/original first difference')
        if t is None:raise AssertionError('No-disagreement output requires explicit empty-array audit path; do not manufacture active results')
        c.check(t>=1 and not A['ever_breached'][t] and not B['ever_breached'][t],pid+'/both active after inception')
        for field in STATE:
            c.value(A[field][:t+1],B[field][:t+1],pid+'/common prefix '+field,atol=2e-10,rtol=5e-13)
            c.value(pair['common_state'][field],A[field][t],pid+'/recorded exact common state '+field)
        for field in PREVIOUS:c.value(A[field][:t],B[field][:t],pid+'/previous '+field,atol=2e-10,rtol=5e-13)
        c.check(np.array_equal(A['actions'][:t],B['actions'][:t]),pid+'/previous actions exact')
        pathsha=hashlib.sha256()
        for x in (np.array([left['scenario_id']],dtype=np.int64),r[left['scenario_id']-1],pi[left['scenario_id']-1]):pathsha.update(np.ascontiguousarray(x).tobytes())
        c.value(pair['same_actual_original_market_path_sha256'],pathsha.hexdigest(),pid+'/actual market SHA')
        c.check(pair['pre_tau_state_identity']['status']=='passed' and pair['pre_tau_state_identity']['tau_year']==t+1,pid+'/prefix status')
        for field in ('terminal_real_wealth','total_net_log_real_growth','original_H_MDD','nominal_fees'):
            c.value(pair['original_outcome_right_minus_left'][field],outcome(B)[field]-outcome(A)[field],pid+'/original outcome delta '+field)
        for label,fan,side in fixed_fans(left,right):
            n=len(fan);expected_member_n+=n;index=np.asarray(fan)-1;values=scalar_fan(A['holdings'][t],float(A['cpi'][t]),float(A['real_peak'][t]),float(A['running_mdd'][t]),left['target'],.4,r[index],pi[index],t,cost_rate=.0025)
            qh,qr=values['hold']['mean'],values['restore']['mean'];kh,kr=values['hold']['count'],values['restore']['count'];choice=fixed_gate(qh,qr,kh,kr,n,left['annual_eta'])
            rows=qrows.loc[qrows.contrast_id.eq(pid)&qrows.fan_label.eq(label)];c.check(len(rows)==1,pid+'/'+label+' one Qrow');row=rows.iloc[0].to_dict()
            expected=dict(original_scenario_id=left['scenario_id'],year=t+1,fan_n=n,fan_ids_sha256=canonical(fan),actual_ID_in_fan=left['scenario_id'] in fan,original_H=30,annual_eta=left['annual_eta'],cost_rate=.0025,
                qg_hold=qh,qg_restore=qr,qbad_hold_count=kh,qbad_restore_count=kr,predicted_growth_advantage_restore=qr-qh,predicted_breach_count_delta_restore=kr-kh,availability='active_fixed_state',**choice)
            for field,value in expected.items():c.value(row[field],value,pid+'/'+label+'/Qrow '+field)
            mr=members.loc[members.contrast_id.eq(pid)&members.fan_label.eq(label)]
            c.check(mr.fan_scenario_id.tolist()==fan,pid+'/'+label+' exact335ordered members')
            c.check(mr.year.eq(t+1).all(),pid+'/'+label+' same original year')
            for action in ('hold','restore'):
                key=pid+'__'+label+'__'+action
                c.value(saved[key+'__total_log_growth'],values[action]['logs'][None,:],pid+'/'+label+'/'+action+' allnetlogs')
                c.value(saved[key+'__ever_breached'],values[action]['bad'][None,:],pid+'/'+label+'/'+action+' allbadflags')
                c.value(mr[action+'_total_log_growth'].to_numpy(),values[action]['logs'],pid+'/'+label+'/'+action+' membernetlogs')
                c.value(mr[action+'_ever_breached'].to_numpy(),values[action]['bad'],pid+'/'+label+'/'+action+' memberbadflags')
            if side is not None:
                old=accounts[side]
                for field in ('qg_hold','qg_restore','qbad_hold_count','qbad_restore_count','predicted_score','predicted_growth_advantage_restore','predicted_breach_count_delta_restore'):c.value(row[field],old[field][t],pid+'/'+side+' original saved Q '+field)
                c.value(row['chosen_action'],str(old['actions'][t]),pid+'/'+side+' original saved action');c.value(row['gate_reason'],str(old['gate_reason'][t]),pid+'/'+side+' original saved reason')
                c.value(pair['saved_reproduction'][side]['status'],'passed',pid+'/'+side+' original reproduction receipt')
            results.append(dict(contrast_id=pid,fan_label=label,year=t+1,fan_n=n,**choice,qg_hold=qh,qg_restore=qr,qbad_hold_count=kh,qbad_restore_count=kr))
    finally:
        for z in archives.values():z.close()
    c.check(len(members)==expected_member_n and m['fan_member_records']==expected_member_n and m['fixed_state_Q_rows']==len(qrows),'complete output branch denominators')
    after=snapshot(p,m);c.check(before==after,'all fan contrast source/input/protocol/output unchanged')
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks_passed=c.n,maximum_numeric_absolute_error=c.maxerr,
        producer_manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(PROTOCOL),audit_plan_sha256=sha(PLAN),auditor_sha256=sha(Path(__file__)),test_sha256=sha(ROOT/'tests/test_audit_matched_parameter_fan_contrast.py'),
        metadata_cases_compared=11,unordered_metadata_pairs_compared=55,eligible_pairs_checked=len(selected),fixed_states_checked=len(selected),Q_fans_independently_recomputed=len(results),
        paired_fan_members_independently_replayed=expected_member_n,action_fan_branches_independently_replayed=2*expected_member_n,new_actual_policy_replays=0,new_fits=0,final_scenarios_read=False,
        independent_method='Earlier independent scalar_fan with100step fee bisection; every branch terminal log and inherited-history event compared in both NPZ andCSV. New producer never imported.',
        inference_limits=p['limitations'],sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,fan_results=results)
    receipt.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','paired_fan_members_independently_replayed','action_fan_branches_independently_replayed','maximum_numeric_absolute_error')}));return result

if __name__=='__main__':run()
