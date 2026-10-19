"""All-object scalar audit of the separately released locked V2b evaluation.

No production execution/forecast/metric helper is imported. The independently
written frozen V2/V2b audit math is explicitly shared. Every main active
choice and every declared physical account is checked. No selection occurs.
"""
from __future__ import annotations
import argparse,ast,csv,gzip,hashlib,json,math,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_rebuild as base
import audit_state_budget_mitigation as previous
PROTOCOL=Path('docs/mitigation_locked_final_evaluation_protocol.json')
PRODUCER_FREEZE=Path('docs/mitigation_locked_final_evaluation_source_freeze.json')
RELEASE=Path('docs/mitigation_locked_final_evaluation_release.json')
ACTIVATION=Path('docs/mitigation_locked_final_evaluation_release_freeze.json')
REGISTRY=Path('docs/mitigation_main_model_registry.json')
PLAN=Path('docs/mitigation_locked_final_audit_plan.json')
AUDIT_FREEZE=Path('docs/mitigation_locked_final_audit_source_freeze.json')
OUT=Path('results/mitigation_locked_final_evaluation_v1')
ATTEMPT=Path('results/mitigation_locked_final_audit_attempt_v1')
TRAIN=list(range(1,121));ACTUAL=list(range(121,181));TOL=1e-12
PROFILES=(('A',5,.1,'high'),('B',15,.25,'medium'),('C',30,.4,'low'),('D',25,.15,'medium'),('E',6,.35,'high'))
FIN=previous.FIN
PAIR=('log_growth','cagr','mdd','nominal_fees','trade_fraction','automatic_fraction')
sha,readj,writej=base.sha,base.readj,base.writej
Checks=base.Checks
scalar_decision,scalar_forecasts=previous.scalar_decision,base.scalar_forecasts

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def physical(arm,r,pi,target,group,c):
    loss=group['L']
    label=group['group_id'];x=arm['holdings'];post=arm['posttrade_holdings'];n,h=len(r),r.shape[1]
    c.check(x.shape==(n,h+1,3) and post.shape==(n,h,3),label+'/complete original-H account')
    c.arr(x[:,0],np.tile([0.,0.,100.],(n,1)),label+'/initial100bills')
    c.arr(x[:,1:],post*(1+r),label+'/after-action actual joint nominal returns')
    cp=np.column_stack((np.ones(n),np.cumprod(1+pi,axis=1)))
    c.arr(arm['cpi'],cp,label+'/original CPI')
    wealth=x.sum(axis=2);real=wealth/cp;peak=np.maximum.accumulate(real,axis=1);dd=np.maximum(1-real/peak,0);mdd=np.maximum.accumulate(dd,axis=1)
    for key,value in [('nominal_wealth',wealth),('real_wealth',real),('real_peak',peak),('current_drawdown',dd),('running_mdd',mdd),('pretrade_wealth',wealth[:,:-1])]:c.arr(arm[key],value,label+'/'+key)
    c.arr(arm['weights'],x/wealth[:,:,None],label+'/weights');c.arr(arm['end_weights'],arm['weights'][:,1:],label+'/endweights')
    moved=np.maximum(post-x[:,:-1],0).sum(axis=2)
    c.arr(arm['moved'],moved,label+'/destination amount once');c.arr(arm['fees'],.0025*moved,label+'/implicit fee equation once')
    c.arr(post.sum(axis=2)+arm['fees'],wealth[:,:-1],label+'/self finance')
    c.arr(arm['fee_fraction'],arm['fees']/wealth[:,:-1],label+'/fee fraction')
    c.arr(arm['moved_fraction'],moved/wealth[:,:-1],label+'/moved fraction')
    c.arr(arm['trades'],moved>TOL,label+'/physical trade not target label')
    c.arr(arm['pretrade_tv_drift'],.5*np.abs(arm['weights'][:,:-1]-target).sum(axis=2),label+'/pretrade drift')
    c.arr(arm['end_year_tv_drift'],.5*np.abs(arm['weights'][:,1:]-target).sum(axis=2),label+'/end drift')
    bad=np.maximum.accumulate(mdd-loss>TOL,axis=1);c.arr(arm['ever_breached'],bad,label+'/absorbing original breach')
    first=np.where(bad.any(axis=1),bad.argmax(axis=1),-1);c.arr(arm['first_crossing'],first,label+'/first crossing')
    c.arr(arm['triggered'],first>=0,label+'/triggered')
    recovered=np.zeros(n,bool);recovery=np.full(n,-1,int)
    for i,tau in enumerate(first):
        if tau>=0:
            for key,value in [('drawdown_at_trigger',mdd[i,tau]),('peak_at_trigger',peak[i,tau]),('real_wealth_at_trigger',real[i,tau])]:c.value(arm[key][i],value,label+f'/{i}/'+key)
            end=next((e for e in range(tau+1,h+1) if real[i,e]>=peak[i,tau]),-1)
            recovery[i]=end;recovered[i]=end>=0
        else:
            for key in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):c.check(np.isnan(arm[key][i]),label+f'/{i}/'+key+' NULL')
    for key,value in [('recovery_endpoint',recovery),('recovered',recovered),('recovery_time',np.where(recovered,recovery-first,-1)),
            ('right_censored',(first>=0)&~recovered),('available_followup',np.where(first>=0,h-first,-1)),
            ('observed_followup',np.where(recovered,recovery-first,np.where(first>=0,h-first,-1)))]:c.arr(arm[key],value,label+'/'+key)
    severity=np.array([max(mdd[i,-1]-mdd[i,tau],0) if tau>=0 else np.nan for i,tau in enumerate(first)])
    c.arr(arm['added_severity'],severity,label+'/added severity')
    terminal=np.where(bad[:,-1],mdd[:,-1]-loss,0.) if group['kind']!='reference' else np.maximum(mdd[:,-1]-loss,0)
    c.arr(arm['terminal_excess'],terminal,label+'/terminal excess convention')
    if group['kind']=='reference':
        pending=bad[:,:-1] if group['policy']=='same_target_B' else np.zeros((n,h),bool)
        c.arr(arm['pending_at_decision'],pending,label+'/reference original rule')
        c.arr(arm['target_execution'],~pending,label+'/reference annual/pending executions')
        c.arr(arm['actions'],np.where(pending,'hold','restore'),label+'/reference actions')
        c.arr(post[pending],x[:,:-1][pending],label+'/reference pending hold')
        restore=~pending;c.arr(post[restore]/post[restore].sum(axis=1)[:,None],np.broadcast_to(target,post[restore].shape),label+'/reference target',atol=3e-12)
        return
    guard=arm['guard_pending'];new=arm['guard_triggered'];before=guard[:,:-1]|bad[:,:-1]
    c.arr(guard[:,1:],np.maximum.accumulate(new,axis=1),label+'/absorbing guard pending')
    c.arr(arm['was_pending_input'],before,label+'/pre-action pending')
    c.arr(arm['was_guard_pending_input'],guard[:,:-1],label+'/pre-action guard')
    c.arr(arm['pending_at_decision'],before|new,label+'/current guard included pending')
    c.arr(arm['guard_pending_at_decision'],guard[:,:-1]|new,label+'/current guard flag')
    c.arr(arm['automatic_service'],~(before|new),label+'/automatic service')
    c.arr(arm['gate_evaluated'],~before,label+'/only active evaluated')
    c.arr(new,(arm['selected_index']<0)&~before,label+'/new guard only evaluated no feasible')
    c.arr(arm['first_guard_trigger'],np.where(new.any(axis=1),new.argmax(axis=1)+1,-1),label+'/first guard')
    c.arr(post[before],x[:,:-1][before],label+'/pending exact physical hold')
    c.arr(arm['actions'][before],np.full(np.count_nonzero(before),'hold'),label+'/pending action')
    c.check(not arm['target_execution'][before].any(),label+'/pending no target execution')
    c.arr(arm['targets'],target,label+'/unchanged strategic target');c.arr(arm['loss_tolerances'],[loss],label+'/original loss tolerance')
    gross=(1+r)/(1+pi[:,:,None]);c.arr(arm['realized_real_gross'],gross,label+'/after-action realized real gross')
    previous=np.full((n,h,4),np.nan);previous[:,1:,:3]=gross[:,:-1]-1;previous[:,1:,3]=pi[:,:-1]
    c.arr(arm['actual_previous_state'],previous,label+'/actual prefix t-1 only')
    inactive=before
    inactive_float='floor budget_floor selected_score pred_log_growth_hold pred_log_growth_selected uniform_predicted_advantage conditional_predicted_advantage min_predicted_advantage current_log_fee gross_log_advantage_uniform gross_log_advantage_min worst_real_shortfall_hold worst_real_shortfall_selected worst_real_shortfall_improvement selected_lambda'.split()
    for key in inactive_float:c.check(np.isnan(arm[key][inactive]).all(),label+'/'+key+'/pending NULL')
    for key in ('selected_index','selected_proposal'):c.check(np.all(arm[key][inactive]==-1),label+'/'+key+'/pending sentinel')
    for key in ('forecast_n','forecast_member_count'):c.check(np.all(arm[key][inactive]==0),label+'/'+key+'/pending zero')
    for key in arm:
        if key.startswith('candidate_') or key.startswith('interval_'):
            values=arm[key][inactive]
            if arm[key].dtype.kind=='f':c.check(np.isnan(values).all(),label+'/'+key+'/pending NULL')
            elif arm[key].dtype.kind=='i':c.check(np.all(values==-1),label+'/'+key+'/pending sentinel')
            elif arm[key].dtype.kind=='b':c.check(not values.any(),label+'/'+key+'/pending false')
    for key in ('support_feasible','hold_feasible','growth_sacrifice'):c.check(not arm[key][inactive].any(),label+'/'+key+'/pending false')
    expectedreason=np.where(guard[:,:-1],'guard_pending_hold','already_breached_pending_hold')
    c.arr(arm['gate_reason'][inactive],expectedreason[inactive],label+'/pending reason')

def reference_schema(arm,group,n,train_n,c):
    h=group['horizon']
    full=base.PHYSICAL|base.EXTRAS
    omitted={k for k in full if k.startswith('candidate_') or k in ('training_previous_state','training_real_gross')}
    expected=base.PHYSICAL if group['kind']=='reference' else full-omitted if group['kind']=='inner_candidate' else full
    c.check(set(arm)==expected,group['group_id']+'/complete declared array inventory')
    c.check(set(group['array_names'])==set(arm) and len(group['array_names'])==len(arm),group['group_id']+'/array metadata inventory')
    c.check(set(group['omitted_diagnostic_arrays'])==(omitted if group['kind']=='inner_candidate' else set()),group['group_id']+'/exact deliberate omitted diagnostics')
    scalar=set('first_crossing triggered drawdown_at_trigger peak_at_trigger real_wealth_at_trigger recovery_endpoint recovery_time recovered right_censored available_followup observed_followup added_severity terminal_excess first_guard_trigger'.split())
    end=set('nominal_wealth real_wealth real_peak current_drawdown running_mdd ever_breached cpi guard_pending'.split())
    booleans=set('ever_breached target_execution pending_at_decision trades triggered recovered right_censored guard_pending guard_pending_at_decision was_pending_input was_guard_pending_input automatic_service support_feasible hold_feasible growth_sacrifice guard_triggered gate_evaluated interval_feasible candidate_feasible'.split())
    ints=set('first_crossing recovery_endpoint recovery_time available_followup observed_followup first_guard_trigger selected_index selected_proposal forecast_n forecast_member_count interval_lower_member interval_upper_member candidate_proposal candidate_worst_support_member'.split())
    for key,value in arm.items():
        shape=(n,h)
        if key in scalar:shape=(n,)
        elif key in end:shape=(n,h+1)
        elif key in ('holdings','weights'):shape=(n,h+1,3)
        elif key in ('posttrade_holdings','end_weights','realized_real_gross'):shape=(n,h,3)
        elif key=='targets':shape=(3,)
        elif key=='loss_tolerances':shape=(1,)
        elif key=='training_previous_state':shape=(h,train_n,4)
        elif key=='training_real_gross':shape=(h,train_n,3)
        elif key=='actual_previous_state' or key.startswith('interval_'):shape=(n,h,4)
        elif key=='candidate_posttrade':shape=(n,h,13,3)
        elif key.startswith('candidate_member_') or key=='candidate_predicted_next_breach':shape=(n,h,13,5)
        elif key.startswith('candidate_'):shape=(n,h,13)
        c.check(value.shape==shape,group['group_id']+'/'+key+'/shape')
        kind='b' if key in booleans else 'i' if key in ints else 'U' if key in ('actions','gate_reason') else 'f'
        c.check(value.dtype.kind==kind,group['group_id']+'/'+key+'/dtype')


def expected_objects():
    result=[];target=[.45,.05,.5]
    for oid,rule,w in [('mitigation_main','mitigation',target),('same_target_B','pending_hold',target),('same_target_annual','annual',target),('exact_thirds_annual','annual',[1/3]*3)]:
        result.append(dict(object_id=oid,question='Q2',profile='C',horizon=30,L=.4,liquidity='low',assignment='Balanced',target=w,rule=rule,
            role='locked_development_candidate' if rule=='mitigation' else 'hypothetical_reference',q1_suitable=True))
    assignments={'A':'Review','B':'Conservative','C':'Balanced','D':'Review','E':'Growth'}
    weights={'Conservative':[.25,.3,.45],'Balanced':target,'Growth':[.65,.05,.3]}
    for name,h,L,liquidity in PROFILES:
        assignment=assignments[name];available=assignment!='Review'
        result.append(dict(object_id='Q1_'+name,question='Q1',profile=name,horizon=h,L=L,liquidity=liquidity,assignment=assignment,target=weights[assignment] if available else None,
            rule='annual' if available else 'Review',role='static_product_reference' if available else 'Q1_Review',q1_suitable=available))
        result.append(dict(object_id='Q1_'+name+'_thirds',question='Q1_reference',profile=name,horizon=h,L=L,liquidity=liquidity,assignment='exact_thirds_reference',target=[1/3]*3,
            rule='annual',role='hypothetical_reference',q1_suitable=None))
    return result


def path_metrics(arm,i,loss):
    values=[float(v) for v in arm['real_wealth'][i]];h=len(values)-1
    annual=[values[t+1]/values[t]-1 for t in range(h)];mu=math.fsum(annual)/h
    mdd=max([0.]+[1-values[t]/values[s] for s in range(h) for t in range(s+1,h+1)])
    log=math.log(values[-1]/100)/h;reason=arm.get('gate_reason',np.full_like(arm['actions'],'',dtype='U64'))[i];mit=reason=='risk_mitigation_then_review'
    first=next((t for t in range(1,h+1) if float(arm['running_mdd'][i,t])-loss>TOL),None)
    return dict(log_growth=log,cagr=math.expm1(log),mdd=mdd,volatility=math.sqrt(math.fsum((v-mu)**2 for v in annual)/(h-1)) if h>1 else 0.,terminal_real=values[-1],
        nominal_fees=math.fsum(float(v) for v in arm['fees'][i]),trade_fraction=float(np.count_nonzero(arm['moved'][i]>TOL))/h,breach=mdd-loss>TOL,first_crossing=first,
        automatic_fraction=float(np.count_nonzero(~arm['pending_at_decision'][i]))/h,protection_trade_fraction=float(np.count_nonzero((reason=='risk_budget_protection')|(mit&arm['trades'][i])))/h,
        support_review=bool(np.any(arm.get('guard_pending',np.zeros((len(arm['real_wealth']),h+1),bool))[i])),
        mitigation_trade_fraction=float(np.count_nonzero(mit&arm['trades'][i]))/h,mitigation_nominal_fees=math.fsum(float(v) for v in arm['fees'][i][mit]))


def summary(rows,template=None):
    rows=list(rows);basis=rows[0] if rows else template
    if basis is None or len({r['scenario_id'] for r in rows})!=len(rows):raise ValueError('Unique same-object original population required')
    if any(r['object_id']!=basis['object_id'] for r in rows):raise ValueError('Do not pool horizons/objects')
    result={key:basis[key] for key in ('object_id','question','profile','horizon','L','liquidity','assignment','role','q1_suitable','scope')}
    funded=[r for r in rows if r['assigned']];n=len(funded)
    result.update(original_n=len(rows),financial_n=n,initial_review_n=len(rows)-n,initial_coverage=n/len(rows) if rows else None)
    result.update(dict.fromkeys('breach_count breach_fraction mean_log_growth median_cagr p10_cagr mean_mdd median_mdd p90_mdd cvar90_mdd mean_volatility mean_nominal_fees mean_trade_fraction mean_automatic_fraction funded_support_review_n mean_protection_trade_fraction mean_mitigation_trade_fraction mean_mitigation_nominal_fees'.split()))
    if not n:return result
    avg=lambda key:math.fsum(float(row[key]) for row in funded)/n
    losses=[r['mdd'] for r in funded];cagrs=[r['cagr'] for r in funded]
    result.update(breach_count=sum(r['breach'] for r in funded),breach_fraction=sum(r['breach'] for r in funded)/n,
        mean_log_growth=avg('log_growth'),median_cagr=base.median(cagrs),p10_cagr=base.quantile(cagrs,.1),mean_mdd=avg('mdd'),median_mdd=base.median(losses),p90_mdd=base.quantile(losses,.9),cvar90_mdd=base.cvar(losses),
        mean_volatility=avg('volatility'),mean_nominal_fees=avg('nominal_fees'),mean_trade_fraction=avg('trade_fraction'),mean_automatic_fraction=avg('automatic_fraction'),
        funded_support_review_n=sum(r['support_review'] for r in funded),mean_protection_trade_fraction=avg('protection_trade_fraction'),mean_mitigation_trade_fraction=avg('mitigation_trade_fraction'),mean_mitigation_nominal_fees=avg('mitigation_nominal_fees'))
    return result


def compare(model,reference):
    a={r['scenario_id']:r for r in model};b={r['scenario_id']:r for r in reference}
    if set(a)!=set(b) or len(a)!=len(model) or len(b)!=len(reference):raise ValueError('Exact unique same-ID cohort required')
    ids=sorted(i for i in a if a[i]['assigned'])
    if any(not b[i]['assigned'] for i in ids):raise ValueError('Same-ID reference unavailable')
    pairs=[dict(scenario_id=i,**{k:a[i][k]-b[i][k] for k in PAIR}) for i in ids]
    deltas={k:dict(mean=math.fsum(r[k] for r in pairs)/len(pairs) if pairs else None,median=base.median([r[k] for r in pairs]) if pairs else None,
        positive_n=sum(r[k]>TOL for r in pairs),negative_n=sum(r[k]<-TOL for r in pairs),numerical_tie_n=sum(abs(r[k])<=TOL for r in pairs)) for k in PAIR}
    partition={k:[] for k in ('neither','model_only','reference_only','both')}
    for i in ids:
        x,y=a[i]['breach'],b[i]['breach'];partition['both' if x and y else 'model_only' if x else 'reference_only' if y else 'neither'].append(i)
    return dict(model=summary(model),reference_all_original=summary(reference),same_funded_model=summary([a[i] for i in ids],a[next(iter(a))]),
        same_funded_reference=summary([b[i] for i in ids],b[next(iter(b))]),same_funded_ids=ids,initial_review_ids=sorted(set(a)-set(ids)),paired_differences=deltas,event_id_partition=partition),pairs


def csv_compare(path,expected,c,label):
    with gzip.open(path,'rt',newline='') as stream:
        reader=csv.DictReader(stream);actual=list(reader);columns=reader.fieldnames
    wanted=sorted({k for r in expected for k in r if k!='target'})
    c.check(columns==wanted,label+'/complete declared CSV fields');c.check(len(actual)==len(expected),label+'/all original rows')
    for i,(observed,target) in enumerate(zip(actual,expected)):
        for key in wanted:
            e=target[key];value=observed[key]
            if e is None:c.check(value=='',label+f'/{i}/'+key+'/NULL');continue
            if isinstance(e,(bool,np.bool_)):
                c.check(value in ('True','False'),label+f'/{i}/'+key+'/boolean');value=value=='True'
            elif isinstance(e,(int,float,np.number)):
                value=float(value)
                if isinstance(e,(int,np.integer)):c.check(value.is_integer(),label+f'/{i}/'+key+'/integer')
            c.value(value,e,label+f'/{i}/'+key)


def source_review(root):
    tree=ast.parse((Path(root)/'scripts/run_mitigation_locked_final_evaluation.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_paths']
    scopes={(ast.unparse(n.args[1]),{k.arg:ast.literal_eval(k.value) for k in n.keywords}.get('allow_final')) for n in calls}
    if len(calls)!=2 or scopes!={('range(1, 121)',False),('range(121, 181)',True)}:raise AssertionError('Fixed independent training/final loader scopes required')
    return dict(status='passed',scope='Source review of fixed loader calls and no search; numerical/account/decision checks are separate')


def preflight(root=ROOT):
    root=Path(root)
    for path in (RELEASE,ACTIVATION,AUDIT_FREEZE):
        if not (root/path).is_file():raise RuntimeError('Separate final release/hash activation/audit source freeze required before values')
    p,f,release,activation,a=map(readj,(root/PROTOCOL,root/PRODUCER_FREEZE,root/RELEASE,root/ACTIVATION,root/AUDIT_FREEZE))
    if (p.get('execution_ready') is not True or p.get('final_execution_ready') is not False or p.get('training_ids')!=TRAIN or p.get('evaluation_ids')!=ACTUAL or p.get('output_directory')!=str(OUT)
        or f.get('execution_ready') is not True or f.get('protocol_sha256')!=sha(root/PROTOCOL)):
        raise RuntimeError('Frozen preparation must preserve final-disabled gate and fixed IDs')
    if (release.get('final_execution_ready') is not True or release.get('protocol_sha256')!=sha(root/PROTOCOL) or release.get('source_freeze_sha256')!=sha(root/PRODUCER_FREEZE)
        or release.get('object_specification_sha256')!=p['object_specification_sha256'] or release.get('registry_sha256')!=p['registry_sha256']
        or release.get('training_ids')!=TRAIN or release.get('evaluation_ids')!=ACTUAL or release.get('output_directory')!=str(OUT)
        or release.get('previously_exposed_nonblind') is not True or release.get('adjust_after_final') is not False
        or release.get('technical_evidence_sha256')!=p['technical_evidence_sha256']):raise RuntimeError('Exact immutable nonblind final release required')
    if (activation.get('execution_ready') is not True or activation.get('release_sha256')!=sha(root/RELEASE) or activation.get('protocol_sha256')!=sha(root/PROTOCOL)
        or activation.get('source_freeze_sha256')!=sha(root/PRODUCER_FREEZE)):raise RuntimeError('Independent exact release SHA activation required')
    if not a.get('execution_ready') or a.get('plan_sha256')!=sha(root/PLAN):raise RuntimeError('Independent source/test/plan freeze required')
    required={'scripts/audit_mitigation_locked_final_evaluation.py','tests/test_audit_mitigation_locked_final_evaluation.py',str(PLAN),'scripts/audit_state_budget_rebuild.py','scripts/audit_state_budget_mitigation.py','src/safeinvest/data.py'}
    if not required<=set(a['source_sha256']):raise RuntimeError('Complete shared independent numerical source closure required')
    if f['input_sha256'].get(str(PLAN))!=sha(root/PLAN):raise RuntimeError('Producer source freeze must bind prospective audit plan')
    snapshot={};snapshot.update(f['source_sha256']);snapshot.update(f['input_sha256']);snapshot.update({str(path):sha(root/path) for path in (PROTOCOL,PRODUCER_FREEZE,RELEASE,ACTIVATION)})
    base.verify_map(snapshot,root);base.verify_map(a['source_sha256'],root)
    if digest(p['object_specification'])!=p['object_specification_sha256'] or p['object_specification']['objects']!=expected_objects():raise RuntimeError('Original14 frozen objects differ from independent fixed contract')
    for name,digest_value in p['technical_evidence_sha256'].items():
        if snapshot.get(name)!=digest_value:raise RuntimeError('Technical receipts absent from final closure')
        receipt=readj(root/name)
        if receipt.get('status') not in ('passed','complete'):raise RuntimeError('Prior technical evidence incomplete')
        for file,value in receipt.get('outputs_sha256',{}).items():snapshot[str(Path(name).parent/file)]=value
    base.verify_map(snapshot,root)
    m=readj(root/OUT/'manifest.json')
    required_outputs={'Q2_paths.csv.gz','Q1_paths.csv.gz','Q1_thirds_references.csv.gz','summary.csv.gz','paired_paths.csv.gz','object_specification.json','trace_groups.json','initial_witness.json','comparisons.json','severe_example.json','traces.npz'}
    if (m.get('status')!='complete' or m.get('sources_and_inputs_at_start')!=snapshot or m.get('sources_and_inputs_at_end')!=snapshot
        or m.get('protocol_sha256')!=sha(root/PROTOCOL) or m.get('source_freeze_sha256')!=sha(root/PRODUCER_FREEZE) or m.get('release_sha256')!=sha(root/RELEASE)
        or m.get('release_freeze_sha256')!=sha(root/ACTIVATION) or m.get('final_scenarios_loaded') is not True or m.get('training_ids_loaded')!=TRAIN or m.get('final_ids_loaded')!=ACTUAL
        or m.get('scope')!='previously_exposed_nonblind' or m.get('post_final_parameter_changes') is not False or m.get('new_fits')!=0 or m.get('new_selections')!=0
        or tuple(m.get(k) for k in ('Q2_rows','Q1_rows','Q1_reference_rows','summary_rows','comparisons','trace_groups'))!=(240,300,300,14,8,14)
        or set(m.get('outputs_sha256',{}))!=required_outputs):raise RuntimeError('Complete unchanged fixed final producer required')
    base.verify_map(m['outputs_sha256'],root/OUT)
    names=set(snapshot)|set(a['source_sha256'])|{str(PLAN),str(AUDIT_FREEZE),str(OUT/'manifest.json')}
    names|={str(OUT/name) for name in m['outputs_sha256']}
    return p,m,{name:sha(root/name) for name in sorted(names)}



def model_gates(arm,train,trainpi,ids,group,c):
    label=group['group_id'];n,h=arm['fees'].shape;active=arm['gate_evaluated'];mit=arm['mitigation_evaluated'];spec=group['candidate']
    gross=(1+train[:,:h])/(1+trainpi[:,:h,None])
    if 'training_real_gross' in arm:c.arr(arm['training_real_gross'],np.transpose(gross,(1,0,2)),label+'/own training gross')
    lag=np.full((h,len(ids),4),np.nan);lag[1:,:,:3]=np.transpose(gross[:,:-1]-1,(1,0,2));lag[1:,:,3]=trainpi[:,:-1].T
    if 'training_previous_state' in arm:c.arr(arm['training_previous_state'],lag,label+'/own previous-only training state')
    c.arr(arm['actual_decision_source'],np.where(~active,'pending',np.where(mit,'mitigation','primary')),label+'/actual branch source')
    c.arr(arm['primary_gate_reason'][~active],np.full(np.count_nonzero(~active),''),label+'/pending primary reason')
    for key in ('primary_selected_score','primary_pred_log_growth_selected'):c.check(np.isnan(arm[key][~active]).all(),label+'/'+key+'/pending NULL')
    c.check(np.all(mit.sum(axis=1)<=1) and not mit[:,0].any() and not mit[~active].any(),label+'/one funded active noninitial mitigation only')
    for key in previous.MIT_FIELDS:
        if key not in arm or key in ('actual_decision_source','primary_gate_reason','primary_selected_score','primary_pred_log_growth_selected','mitigation_evaluated','mitigation_trade'):continue
        values=arm[key][~mit]
        if arm[key].dtype.kind=='f':c.check(np.isnan(values).all(),label+'/'+key+'/nonmitigation NULL')
        elif arm[key].dtype.kind=='i':c.check(np.all(values==-1),label+'/'+key+'/nonmitigation sentinel')
    for key in ('mitigation_evaluated','mitigation_trade'):c.check(not arm[key][~active].any(),label+'/'+key+'/pending false')
    slots=np.argwhere(active);mit_n=0
    for i,t in slots:
        cp=float(arm['cpi'][i,t]);x=arm['holdings'][i,t];target=group['target']
        weights=scalar_forecasts(ids) if t==0 else scalar_forecasts(ids,lag[t],arm['actual_previous_state'][i,t])
        expected=scalar_decision(x,cp,arm['real_peak'][i,t],arm['running_mdd'][i,t],gross[:,t],gross[:,t],weights,
            target=target,release=spec['release_fraction'],rule=spec['forecast_rule'],initial=t==0)
        observed={};retained={};K=len(weights)
        for key,value in expected.items():
            source={'fee':'fees','action':'actions'}.get(key,key)
            if key=='status':continue
            if key=='real_wealth':actual=arm['real_wealth'][i,t]
            elif source in arm:
                actual=arm[source][i,t]
                if key.startswith(('candidate_member_','mitigation_candidate_member_')) or key=='candidate_predicted_next_breach':
                    c.check(np.isnan(actual[...,K:]).all(),label+f'/{i}/{t}/'+key+'/unused member padding NULL')
                    actual=actual[...,:K]
            else:continue
            observed[key]=actual;retained[key]=value
        previous.decision_check(observed,retained,c,label+f'/{i}/{t}/all_active_scalar',x=x,cp=cp,support=gross[:,t],target=target)
        c.value(arm['forecast_n'][i,t],len(ids),label+f'/{i}/{t}/training count')
        c.value(arm['forecast_member_count'][i,t],K,label+f'/{i}/{t}/member count')
        idx=expected['selected_index'];s=max(idx,0)
        c.value(arm['selected_proposal'][i,t],int(expected['candidate_proposal'][s]) if idx>=0 else -1,label+f'/{i}/{t}/primary proposal')
        c.value(arm['selected_lambda'][i,t],expected['candidate_lambda'][s] if idx>=0 else np.nan,label+f'/{i}/{t}/primary lambda')
        execution=expected['mitigation_selected_index']>0 if expected['mitigation_evaluated'] else idx>0
        c.value(arm['target_execution'][i,t],execution,label+f'/{i}/{t}/executed branch target label')
        if expected['mitigation_evaluated']:
            mit_n+=1;real=expected['real_wealth'];chosen=expected['mitigation_selected_index'];improvement=expected['mitigation_worst_next_real_improvement']
            c.check(not arm['support_feasible'][i,t] and arm['guard_triggered'][i,t] and not arm['automatic_service'][i,t],label+f'/{i}/{t}/mitigation not recertified')
            c.check(improvement>TOL*real if chosen>0 else expected['fee']==0 and expected['moved']==0,label+f'/{i}/{t}/strict afterfee improvement or free hold')
            c.mitigation_checks.append(dict(group_id=label,scenario_id=group['scenario_ids'][i],original_year=int(t+1),selected_index=int(chosen),
                trade=bool(expected['mitigation_trade']),nominal_fee=float(expected['fee']),worst_real_improvement=float(improvement),
                residual_real_shortfall=float(expected['worst_real_shortfall_selected']),predicted_paired_score=float(expected['selected_score']),support_certificate=False))
    c.check(mit_n==np.count_nonzero(mit),label+'/all fallback events independently enumerated')
    c.active_decision_groups.append(dict(group_id=label,active_decisions=len(slots),mitigation_decisions=mit_n,
        original_year_and_account_slots_sha256=hashlib.sha256(np.ascontiguousarray(slots).tobytes()).hexdigest()))


def audit_values(root,p,m,c):
    sys.path.insert(0,str(Path(root)/'src'))
    from safeinvest.data import load_paths
    train,train_pi,training_ids=load_paths(Path(root)/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
    actual,actual_pi,actual_ids=load_paths(Path(root)/'data/SafeInvest_Synthetic_Data.csv',range(121,181),allow_final=True)
    c.check(training_ids.tolist()==TRAIN and actual_ids.tolist()==ACTUAL,'exact original train/final identity')
    folder=Path(root)/OUT;spec=readj(folder/'object_specification.json')
    base.record_check(spec,p['object_specification'],c,'frozen object specification')
    objects=expected_objects();groups=readj(folder/'trace_groups.json');c.check(len(groups)==14 and len({g['object_id'] for g in groups})==14,'all14 unique original objects')
    c.mitigation_checks=[];c.active_decision_groups=[];rows=[];all_arms={};inventories={};counts=dict(objects=14,original_scenarios=60,physical_accounts=0,periods=0,endpoints=0,Review_rows=0)
    expected_archive=set();groups_by={g['object_id']:g for g in groups}
    gross=(1+train)/(1+train_pi[:,:,None]);initial=scalar_decision([0,0,100],1,100,0,gross[:,0],gross[:,0],scalar_forecasts(TRAIN),target=[.45,.05,.5],release=.5,rule='uniform',initial=True)
    previous.decision_check(readj(folder/'initial_witness.json'),initial,c,'independent initial witness',x=np.array([0.,0.,100.]),cp=1.,support=gross[:,0],target=[.45,.05,.5])
    with np.load(folder/'traces.npz',allow_pickle=False) as archive:
        for obj in objects:
            oid=obj['object_id'];g=groups_by[oid]
            for key,value in obj.items():base.record_check(g[key],value,c,oid+'/original metadata/'+key)
            c.check(g['scenario_ids']==ACTUAL and g['training_ids']==(TRAIN if obj['rule']=='mitigation' else None),oid+'/fixed identity')
            c.check(g['omitted_diagnostic_arrays']==[],oid+'/no thin final menus')
            availability=obj['rule']!='Review' and (obj['rule']!='mitigation' or initial['selected_index']>=0)
            c.check(g['financial_available'] is availability,oid+'/independent financial availability')
            status=('funded' if availability else 'preinvestment_review') if obj['rule']=='mitigation' else 'q1_review' if not availability else obj['role']
            c.check(g['status']==status,oid+'/execution status')
            arm={name:archive[oid+'__'+name] for name in g['array_names']};expected_archive|={oid+'__'+name for name in g['array_names']};all_arms[oid]=arm
            if not availability:
                c.check(not arm and not g['array_names'],oid+'/Review has no physical account');counts['Review_rows']+=60
            else:
                c.check(len(g['array_names'])==len(set(g['array_names'])),oid+'/unique array metadata')
                kind='full_selected' if obj['rule']=='mitigation' else 'reference';meta=dict(group_id=oid,kind=kind,horizon=obj['horizon'],L=obj['L'],target=obj['target'],array_names=g['array_names'],omitted_diagnostic_arrays=[],candidate=base.SPECS[0],scenario_ids=ACTUAL,
                    policy='same_target_B' if obj['rule']=='pending_hold' else 'annual')
                if kind=='reference':reference_schema(arm,meta,60,120,c)
                else:previous.schema(arm,meta,60,120,c)
                h=obj['horizon'];physical(arm,actual[:,:h],actual_pi[:,:h],np.array(obj['target']),meta,c)
                if kind!='reference':model_gates(arm,train,train_pi,TRAIN,meta,c)
                inventories[oid]={name:dict(dtype=str(value.dtype),shape=list(value.shape)) for name,value in arm.items()}
                counts['physical_accounts']+=60;counts['periods']+=60*h;counts['endpoints']+=60*(h+1)
            for i,sid in enumerate(ACTUAL):
                row=dict(**obj,scope='previously_exposed_nonblind',execution_status=status,assigned=availability,scenario_id=sid)
                row.update(path_metrics(arm,i,obj['L']) if availability else dict.fromkeys(FIN));rows.append(row)
        c.check(set(archive.files)==expected_archive and len(archive.files)==len(expected_archive),'complete unique NPZ field inventory')
    q2=[r for r in rows if r['question']=='Q2'];q1=[r for r in rows if r['question']=='Q1'];q1ref=[r for r in rows if r['question']=='Q1_reference']
    c.check((len(q2),len(q1),len(q1ref))==(240,300,300),'all840 original object/ID rows')
    for filename,value in [('Q2_paths.csv.gz',q2),('Q1_paths.csv.gz',q1),('Q1_thirds_references.csv.gz',q1ref)]:csv_compare(folder/filename,value,c,filename)
    summaries=[summary([r for r in rows if r['object_id']==obj['object_id']]) for obj in objects];csv_compare(folder/'summary.csv.gz',summaries,c,'all14 summaries')
    comparisons=[];pairs=[]
    compare_objects=[('mitigation_main',name) for name in ('same_target_B','same_target_annual','exact_thirds_annual')]+[('Q1_'+name,'Q1_'+name+'_thirds') for name,_,_,_ in PROFILES]
    for model,reference in compare_objects:
        value,paired=compare([r for r in rows if r['object_id']==model],[r for r in rows if r['object_id']==reference])
        comparisons.append(dict(model_object=model,reference_object=reference,**value));pairs.extend(dict(model_object=model,reference_object=reference,**r) for r in paired)
    base.record_check(readj(folder/'comparisons.json'),comparisons,c,'all8 complete comparisons/partitions')
    csv_compare(folder/'paired_paths.csv.gz',pairs,c,'all matched paired path rows')
    main=[r for r in q2 if r['object_id']=='mitigation_main' and r['assigned']]
    if not main:severe=dict(scenario_id=None,reason='No main financial account',traces=None)
    else:
        # Preserve the declared exact tie rule on the already verified saved
        # original-H MDD values, rather than silently redefining rounding ties.
        values=all_arms['mitigation_main']['running_mdd'][:,-1];worst=float(max(values));sid=min(ACTUAL[i] for i,v in enumerate(values) if v==worst);i=ACTUAL.index(sid)
        keys=('real_wealth','running_mdd','cpi','holdings','posttrade_holdings','fees','trades','guard_triggered','gate_reason','actual_decision_source')
        traces={name:{key:value[i] for key,value in all_arms[name].items() if key in keys} for name in ('mitigation_main','same_target_B','same_target_annual','exact_thirds_annual')}
        severe=dict(scenario_id=sid,rule='max fullH main MDD then min exact-tied originalID',main_mdd=worst,traces=traces,scope='Worst-case illustration, not a population estimate')
    base.record_check(readj(folder/'severe_example.json'),base.clean(severe),c,'prespecified severe example/full traces')
    counts.update(path_rows=len(rows),summary_rows=len(summaries),comparison_groups=len(comparisons),paired_rows=len(pairs),
        all_main_active_decisions=sum(g['active_decisions'] for g in c.active_decision_groups),mitigation_decisions=len(c.mitigation_checks))
    return dict(counts=counts,field_inventory=inventories,all_main_decision_groups=c.active_decision_groups,all_mitigation_events=c.mitigation_checks,
        numeric_binding_ties=c.numeric_binding_ties,severe_original_id=severe['scenario_id'])


def run(root=ROOT):
    root=Path(root);attempt=root/ATTEMPT;receipt=root/OUT/'independent_audit.json'
    if attempt.exists() or receipt.exists():raise RuntimeError('Never overwrite or silently retry a final audit attempt')
    attempt.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_release_and_hash_gates',started_utc=datetime.now(timezone.utc).isoformat(),final_value_access_started=False,automatic_retry=False)
    writej(attempt/'attempt.json',state)
    try:
        p,m,before=preflight(root);review=source_review(root);c=Checks()
        state.update(phase='released_fixed_final_audit',final_value_access_started=True);writej(attempt/'attempt.json',state)
        result=audit_values(root,p,m,c);after={name:sha(root/name) for name in before};c.check(before==after,'all source/input/output evidence unchanged')
        result.update(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks=c.n,maximum_absolute_numeric_error=c.max_error,
            plan_sha256=sha(root/PLAN),audit_source_freeze_sha256=sha(root/AUDIT_FREEZE),producer_manifest_sha256=sha(root/OUT/'manifest.json'),
            sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,source_review=review,training_ids_loaded=TRAIN,final_ids_loaded=ACTUAL,
            final_scenarios_loaded=True,new_fits=0,new_selections=0,post_final_parameter_changes=False,scope='previously_exposed_nonblind',
            shared_independent_engine='Frozen V2 fee/forecast/primary and V2b fallback/ledger math; all final main decisions recomputed. No production policy import.',
            limitation='One already-exposed final cohort, not fresh blind validation, real-future calibration, recursive-feasibility proof or policy retuning.')
        writej(receipt,result);state.update(status='passed',phase='complete',receipt_sha256=sha(receipt));writej(attempt/'attempt.json',state)
        print(json.dumps(dict(status='passed',checks=c.n,counts=result['counts'],receipt_sha256=sha(receipt))),flush=True)
    except BaseException as error:
        (attempt/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',phase='failure_retained',error_type=type(error).__name__,error=str(error),failure_sha256=sha(attempt/'failure.txt'))
        writej(attempt/'attempt.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,final_access_enabled=False,requires_independent_release_freeze=True)))

