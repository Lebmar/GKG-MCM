"""Independent saved-development Review-operation audit; no policy/fee-kernel import."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=Path('docs/review_operation_diagnostic_audit_plan_v1.json')
FREEZE=Path('docs/review_operation_diagnostic_audit_source_freeze_v1.json')
PARENT=Path('results/review_operation_diagnostic_v1')
OLD=Path('results/state_budget_mitigation_v1')
ATTEMPT=Path('results/review_operation_diagnostic_audit_attempt_v1')
RECEIPT=PARENT/'independent_audit.json'
ABS=1e-9
REL=5e-12
EPS=1e-12
H=30
L=.4
K=.0025
PHASES=('before_guard','guard_year','after_guard','no_guard')
FRACTIONS=(.25,.5,.75,1.)
FEES=(0.,.001,.0025,.005,.01)
REFS=('same_target_B','same_target_annual','exact_thirds_annual')
MAIN_FIELDS={'fees','moved','trades','holdings','posttrade_holdings','cpi','nominal_wealth','real_wealth','real_peak',
    'current_drawdown','running_mdd','first_guard_trigger','first_crossing','guard_triggered','guard_pending',
    'was_pending_input','was_guard_pending_input','automatic_service','gate_reason','realized_real_gross'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvrows(p):
    with gzip.open(p,'rt',newline='') as f:return list(csv.DictReader(f))
def verify(root,mapping):
    for name,digest in mapping.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name or sha(root/p)!=digest:raise AssertionError('Frozen audit dependency changed: '+name)


class Checks:
    def __init__(self):self.scalar_checks=0;self.max_absolute_error=0.
    def value(self,a,b,label=''):
        self.scalar_checks+=1
        if a is None or b is None:
            if a is not None or b is not None:raise AssertionError('NULL mismatch '+label)
        elif isinstance(b,(bool,np.bool_)):
            if not isinstance(a,(bool,np.bool_)) or bool(a)!=bool(b):raise AssertionError('Boolean mismatch '+label)
        elif isinstance(b,(str,bytes)):
            if a!=b:raise AssertionError('Text mismatch '+label)
        elif isinstance(b,(int,np.integer)):
            if a!=b:raise AssertionError('Integer mismatch '+label)
        else:
            a,b=float(a),float(b)
            if math.isnan(a) and math.isnan(b):return
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=REL,abs_tol=ABS):raise AssertionError('Numeric mismatch '+label+': '+str((a,b)))
            self.max_absolute_error=max(self.max_absolute_error,abs(a-b))
    def array(self,a,b,label='',exact=False):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Shape/dtype mismatch '+label+': '+str((a.shape,b.shape,a.dtype,b.dtype)))
        if exact or b.dtype.kind in 'biuUS':
            if not np.array_equal(a,b,equal_nan=b.dtype.kind in 'fc'):raise AssertionError('Exact array mismatch '+label)
            self.scalar_checks+=b.size;return
        for x,y in zip(a.flat,b.flat):self.value(x,y,label)
    def tree(self,a,b,label=''):
        if isinstance(b,dict):
            if set(a)!=set(b):raise AssertionError('Object keys mismatch '+label)
            for k in b:self.tree(a[k],b[k],label+'/'+k)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):raise AssertionError('List length mismatch '+label)
            for i,(x,y) in enumerate(zip(a,b)):self.tree(x,y,label+'/'+str(i))
        else:self.value(a,b,label)
    def table(self,actual,expected,keys,label):
        columns=set().union(*(x.keys() for x in expected));am={tuple(r[k] for k in keys):r for r in actual}
        em={tuple(str(r[k]) for k in keys):r for r in expected}
        if len(am)!=len(actual) or len(em)!=len(expected) or set(am)!=set(em):raise AssertionError('Exact row identity mismatch '+label)
        for index,row in em.items():
            if set(am[index])!=columns:raise AssertionError('CSV columns mismatch '+label)
            for field in columns:
                text=am[index][field];value=row.get(field)
                if value is None:parsed=None if text=='' else text
                elif isinstance(value,bool):parsed={'True':True,'False':False}.get(text,text)
                elif isinstance(value,(int,float,np.number)):
                    try:parsed=float(text)
                    except ValueError:raise AssertionError('Expected numeric cell '+label+'/'+field)
                else:parsed=text
                self.value(parsed,value,label+'/'+str(index)+'/'+field)


def fee_endpoint(holdings,target,cost=K):
    """Independent scalar bisection, no shared execute_target implementation."""
    x=[float(z) for z in holdings];w=[float(z) for z in target];v=math.fsum(x)
    if len(x)!=3 or len(w)!=3 or min(x)<0 or min(w)<0 or not math.isclose(math.fsum(w),1.,rel_tol=0,abs_tol=1e-12) or not 0<=cost<1:raise ValueError('Valid long-only target and fee required')
    low,high=v/(1+cost),v
    for _ in range(100):
        mid=(low+high)/2
        if mid+cost*math.fsum(max(mid*w[i]-x[i],0.) for i in range(3))>v:high=mid
        else:low=mid
    total=(low+high)/2;post=[total*z for z in w];moved=math.fsum(max(post[i]-x[i],0.) for i in range(3));fee=cost*moved
    return post,fee,moved


def financial(real,fees,trades,mdd_override=None):
    r=[float(z) for z in real];h=len(r)-1;peak=r[0];loss=0.
    for v in r:peak=max(peak,v);loss=max(loss,1-v/peak)
    if mdd_override is not None:loss=float(mdd_override)
    log=math.log(r[-1]/r[0])/h
    return dict(terminal_real=r[-1],log_growth=log,cagr=math.expm1(log),mdd=loss,breach=loss-L>EPS,
        nominal_fees=math.fsum(float(x) for x in fees),trade_fraction=sum(bool(x) for x in trades)/h)


def quantile(values,q):return sorted(values)[max(0,math.ceil(q*len(values))-1)]
def median(values):
    v=sorted(values);n=len(v)
    return v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2

def cvar90(values):
    v=sorted(values,reverse=True);tail=.1*len(v)
    return math.fsum(min(1.,max(0.,tail-i))*x for i,x in enumerate(v))/tail

def summary(rows,extra=False):
    n=len(rows);values=lambda k:[r[k] for r in rows]
    out=dict(original_n=n,breach_count=sum(r['breach'] for r in rows),mean_log_growth=math.fsum(values('log_growth'))/n,
        median_cagr=median(values('cagr')),p10_cagr=quantile(values('cagr'),.1),mean_mdd=math.fsum(values('mdd'))/n,p90_mdd=quantile(values('mdd'),.9),
        mean_nominal_fees=math.fsum(values('nominal_fees'))/n,mean_trade_fraction=math.fsum(values('trade_fraction'))/n)
    if extra:out.update(median_mdd=median(values('mdd')),cvar90_mdd=cvar90(values('mdd')))
    return out


def independent_market(reference):
    post=reference['posttrade_holdings'];after=reference['holdings'][:,1:];cp=reference['cpi'];g=np.empty_like(post);real=np.empty_like(post)
    if np.any(post<=0):raise AssertionError('Positive thirds reference required')
    for i in range(len(post)):
        for t in range(post.shape[1]):
            for k in range(3):g[i,t,k]=after[i,t,k]/post[i,t,k];real[i,t,k]=g[i,t,k]/(cp[i,t+1]/cp[i,t])
    return g,real


def static_reference(gross,cpi,target,rule):
    """Scalar-time independent ledger, including every frozen reference field."""
    g=np.asarray(gross);cp=np.asarray(cpi);n,h,_=g.shape;w=np.asarray(target,float);w=w/math.fsum(w)
    period=('fees','moved','pretrade_wealth','fee_fraction','moved_fraction','pretrade_tv_drift','end_year_tv_drift')
    a={k:np.zeros((n,h)) for k in period}
    a.update({k:np.zeros((n,h+1)) for k in ('nominal_wealth','real_wealth','real_peak','current_drawdown','running_mdd')})
    a.update(holdings=np.zeros((n,h+1,3)),weights=np.zeros((n,h+1,3)),posttrade_holdings=np.zeros((n,h,3)),
        ever_breached=np.zeros((n,h+1),bool),target_execution=np.zeros((n,h),bool),pending_at_decision=np.zeros((n,h),bool),
        actions=np.full((n,h),'hold',dtype='U7'),first_crossing=np.full(n,-1,dtype=np.int64),cpi=cp.copy())
    for name in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):a[name]=np.full(n,np.nan)
    for i in range(n):
        x=[0.,0.,100.];peak=100.;loss=0.;cross=-1
        a['holdings'][i,0]=x;a['weights'][i,0]=[0.,0.,1.]
        for key in ('nominal_wealth','real_wealth','real_peak'):a[key][i,0]=100.
        for t in range(h):
            v=math.fsum(x)
            post,fee,moved=fee_endpoint(x,w) if t==0 or rule=='annual' else (x.copy(),0.,0.)
            a['target_execution'][i,t]=t==0 or rule=='annual';a['actions'][i,t]='restore' if t==0 or rule=='annual' else 'hold'
            a['fees'][i,t]=fee;a['moved'][i,t]=moved;a['posttrade_holdings'][i,t]=post
            a['pretrade_wealth'][i,t]=v;a['fee_fraction'][i,t]=fee/v;a['moved_fraction'][i,t]=moved/v
            a['pretrade_tv_drift'][i,t]=.5*math.fsum(abs(x[k]/v-w[k]) for k in range(3))
            x=[post[k]*g[i,t,k] for k in range(3)];nominal=math.fsum(x);real=nominal/cp[i,t+1];peak=max(peak,real);dd=max(1-real/peak,0.);loss=max(loss,dd)
            if cross<0 and loss-L>EPS:
                cross=t+1
                for key,val in [('drawdown_at_trigger',loss),('peak_at_trigger',peak),('real_wealth_at_trigger',real)]:a[key][i]=val
            a['holdings'][i,t+1]=x;a['weights'][i,t+1]=[z/nominal for z in x]
            for key,val in [('nominal_wealth',nominal),('real_wealth',real),('real_peak',peak),('current_drawdown',dd),('running_mdd',loss),('ever_breached',loss-L>EPS)]:a[key][i,t+1]=val
            a['end_year_tv_drift'][i,t]=.5*math.fsum(abs(x[k]/nominal-w[k]) for k in range(3))
        a['first_crossing'][i]=cross
    triggered=a['first_crossing']>=0;recovered=np.zeros(n,bool);recovery=np.full(n,-1,dtype=np.int64)
    for i in range(n):
        if triggered[i]:
            for t in range(int(a['first_crossing'][i])+1,h+1):
                if a['real_wealth'][i,t]>=a['peak_at_trigger'][i]:recovered[i]=True;recovery[i]=t;break
    a.update(triggered=triggered,recovered=recovered,recovery_endpoint=recovery,
        recovery_time=np.where(recovered,recovery-a['first_crossing'],-1),right_censored=triggered&~recovered,
        available_followup=np.where(triggered,h-a['first_crossing'],-1),
        observed_followup=np.where(recovered,recovery-a['first_crossing'],np.where(triggered,h-a['first_crossing'],-1)),
        added_severity=np.where(triggered,np.maximum(a['running_mdd'][:,-1]-a['drawdown_at_trigger'],0),np.nan),
        terminal_excess=np.maximum(a['running_mdd'][:,-1]-L,0),trades=a['moved']>EPS,end_weights=a['weights'][:,1:].copy())
    return a


def cost_path(real,peak,running,holdings,cpi,tau,q):
    """Independent two-observation fee boundary, then project to annual indexes."""
    r=np.asarray(real);p=np.asarray(peak);m=np.asarray(running);h=len(r)-1;valid=0<tau<h
    out=r.copy();peaks=p.copy();mdds=m.copy();fee=0.
    if valid and q:
        out[tau:]=r[tau:]*(1-q);fee=q*math.fsum(holdings[tau])
        history=[]
        for t in range(h+1):
            if t==tau:history.append((t,float(r[t]),False))  # Preserve pre-debit endpoint observation.
            history.append((t,float(out[t]),True))
        peak_now=float(r[0]);max_loss=0.
        for t,wealth,project in history:
            peak_now=max(peak_now,wealth);max_loss=max(max_loss,1-wealth/peak_now)
            if project:peaks[t]=peak_now;mdds[t]=max_loss
    crossings=[t for t,x in enumerate(mdds) if x-L>EPS]
    return dict(real_wealth=out,real_peak=peaks,running_mdd=mdds,applicable=valid,review_fee_nominal=fee,
        review_fee_real=fee/float(cpi[tau]) if valid else 0.,debit_endpoint=tau if valid else None,
        first_crossing=crossings[0] if crossings else None,breach=bool(mdds[-1]-L>EPS))


def phases(a,ids,job,real_gross,c):
    n,h=a['fees'].shape;annual=[];parts=[];accounts=[]
    for i,sid in enumerate(ids):
        tau=int(a['first_guard_trigger'][i]);r=a['real_wealth'][i];local=[]
        c.array(a['realized_real_gross'][i],real_gross[i],'saved main jointrealgross')
        guard_indices=list(np.flatnonzero(a['guard_triggered'][i]));c.value(tau,int(guard_indices[0])+1 if guard_indices else -1,'guard year')
        c.value(len(guard_indices)<=1,True,'one guard')
        for t in range(h):
            old=a['holdings'][i,t];post=a['posttrade_holdings'][i,t];cp=float(a['cpi'][i,t]);fee=float(a['fees'][i,t]);v=math.fsum(old);postsum=math.fsum(post)
            c.value(postsum+fee,v,'main self-finance');weights=[float(z)/postsum for z in post]
            if bool(a['was_pending_input'][i,t]):
                c.value(fee,0.,'pending fee');c.value(bool(a['trades'][i,t]),False,'pending trade');c.array(post,old,'pending holdings')
            gain=[float(post[k])*(real_gross[i,t,k]-1)/cp for k in range(3)]
            log=math.log(float(r[t+1])/float(r[t]));logfee=math.log1p(-fee/v);logmarket=math.log(math.fsum(weights[k]*real_gross[i,t,k] for k in range(3)))
            c.value(logfee+logmarket,log,'annual log identity');c.value(-fee/cp+math.fsum(gain),float(r[t+1]-r[t]),'annual real-dollar identity')
            phase='no_guard' if tau<0 else 'before_guard' if t+1<tau else 'guard_year' if t+1==tau else 'after_guard'
            row=dict(job_id=job,scenario_id=int(sid),period=t+1,phase=phase,guard_year=None if tau<0 else tau,
                existing_pending=bool(a['was_pending_input'][i,t]),guard_triggered=bool(a['guard_triggered'][i,t]),guard_pending_before=bool(a['was_guard_pending_input'][i,t]),
                ordinary_service=bool(a['automatic_service'][i,t]),trade=bool(a['trades'][i,t]),nominal_fee=fee,real_fee_at_charge=fee/cp,
                net_log_growth=log,log_fee=logfee,real_portfolio_log_growth=logmarket,real_wealth_start=float(r[t]),real_wealth_end=float(r[t+1]),
                original_peak_end=float(a['real_peak'][i,t+1]),original_current_dd_end=float(a['current_drawdown'][i,t+1]),running_mdd_start=float(a['running_mdd'][i,t]),
                running_mdd_end=float(a['running_mdd'][i,t+1]),gate_reason=str(a['gate_reason'][i,t]))
            for k,asset in enumerate(('equity','bond','bill')):row[asset+'_weight']=weights[k];row[asset+'_real_gain']=gain[k]
            local.append(row);annual.append(row)
        c.value(math.fsum(x['net_log_growth'] for x in local),math.log(float(r[-1])/float(r[0])),'wholeH log identity')
        for phase in PHASES:
            z=[x for x in local if x['phase']==phase];means=lambda key:math.fsum(x[key] for x in z)/len(z) if z else None
            parts.append(dict(job_id=job,scenario_id=int(sid),phase=phase,periods=len(z),net_log_growth=math.fsum(x['net_log_growth'] for x in z),
                log_fee=math.fsum(x['log_fee'] for x in z),nominal_fees=math.fsum(x['nominal_fee'] for x in z),trades=sum(x['trade'] for x in z),
                mean_equity_weight=means('equity_weight'),mean_bond_weight=means('bond_weight'),mean_bill_weight=means('bill_weight'),
                running_mdd_start=z[0]['running_mdd_start'] if z else None,running_mdd_end=z[-1]['running_mdd_end'] if z else None,
                max_original_peak_drawdown=max(x['original_current_dd_end'] for x in z) if z else None))
        accounts.append(dict(job_id=job,scenario_id=int(sid),first_guard=None if tau<0 else tau,
            first_crossing=None if a['first_crossing'][i]<0 else int(a['first_crossing'][i]),post_guard_periods=h-tau if tau>=0 else None,
            review_debit_within_H=bool(0<tau<h),**financial(r,a['fees'][i],a['trades'][i])))
    return annual,parts,accounts


def break_even(main,reference,nominal,valid):
    if not valid:return dict(signed_fraction_threshold=None,max_nonnegative_fee_fraction=None,max_nominal_fee=None,zero_fee_already_worse=None)
    q=1-float(reference)/float(main)
    return dict(signed_fraction_threshold=q,max_nonnegative_fee_fraction=q if q>=0 else None,max_nominal_fee=q*nominal if q>=0 else None,zero_fee_already_worse=q<0)


def paired(static_rows,baselines):
    rows=[];summaries=[]
    for policy in sorted({r['policy'] for r in static_rows}):
        x={r['scenario_id']:r for r in static_rows if r['policy']==policy}
        for reference in ('main',*REFS):
            y={r['scenario_id']:r for r in baselines if r['policy']==reference};counts=dict(neither=0,static_only=0,reference_only=0,both=0);part=[]
            if set(x)!=set(range(1,121)) or set(y)!=set(x):raise AssertionError('Exact120 sameID pairing required')
            for sid in range(1,121):
                a,b=x[sid],y[sid];category='both' if a['breach'] and b['breach'] else 'static_only' if a['breach'] else 'reference_only' if b['breach'] else 'neither';counts[category]+=1
                row=dict(policy=policy,reference=reference,scenario_id=sid,**{'delta_'+k:a[k]-b[k] for k in ('log_growth','cagr','mdd','nominal_fees','trade_fraction')});part.append(row);rows.append(row)
            summaries.append(dict(policy=policy,reference=reference,paired_n=120,event_partition=counts,mean_differences={k:math.fsum(r['delta_'+k] for r in part)/120 for k in ('log_growth','cagr','mdd','nominal_fees','trade_fraction')}))
    return rows,summaries


def authorize(root):
    plan=readj(root/PLAN);freeze=readj(root/FREEZE)
    if plan.get('version')!='independent-review-operation-audit-v1' or plan.get('execution_ready') is not False or plan.get('allow_final') is not False or plan.get('development_ids')!=list(range(1,121)) or freeze.get('execution_ready') is not True or freeze.get('plan_sha256')!=sha(root/PLAN):raise RuntimeError('Root exact source/output audit freeze required before numerical parsing')
    if set(freeze.get('source_sha256',{}))!={'scripts/audit_review_operation.py','tests/test_audit_review_operation.py'}:raise RuntimeError('Freeze both independent source/tests')
    inputs=freeze['input_sha256'];verify(root,freeze['source_sha256']);verify(root,inputs)
    m=readj(root/PARENT/'manifest.json');old=readj(root/OLD/'manifest.json');a=readj(root/OLD/'independent_audit.json')
    if m.get('status')!='complete' or m.get('sources_and_inputs_at_start')!=m.get('sources_and_inputs_at_end') or m.get('raw_market_parser_called') is not False or m.get('final_scenarios_loaded') is not False or m.get('main_policy_changed') is not False or m.get('new_admission') is not False or m.get('new_fits')!=0 or m.get('new_selections')!=0 or sha(root/PARENT/'manifest.json')!=plan['producer_manifest_sha256']:raise RuntimeError('Complete immutable development-only diagnostic required')
    expected_outputs=set(plan['producer_outputs'])
    if set(m['outputs_sha256'])!=expected_outputs:raise RuntimeError('Every declared producer output required')
    required={str(PARENT/'manifest.json'):sha(root/PARENT/'manifest.json'),str(OLD/'manifest.json'):sha(root/OLD/'manifest.json'),str(OLD/'independent_audit.json'):sha(root/OLD/'independent_audit.json')}
    for key,digest in m['sources_and_inputs_at_start'].items():required[key]=digest
    for name,digest in m['outputs_sha256'].items():required[str(PARENT/name)]=digest
    for name,digest in old['outputs_sha256'].items():required[str(OLD/name)]=digest
    for key,digest in a['sources_and_artifacts_at_start'].items():
        if key in required and required[key]!=digest:raise RuntimeError('Conflicting old audit identity')
        required[key]=digest
    if a.get('status')!='passed' or a.get('parent_manifest_sha256')!=sha(root/OLD/'manifest.json') or a['sources_and_artifacts_at_start']!=a['sources_and_artifacts_at_end'] or old.get('status')!='complete':raise RuntimeError('Passed old ledger identity required')
    if any(inputs.get(k)!=v for k,v in required.items()):raise RuntimeError('Freeze full diagnostic outputs and exact old producer/audit closure before NPZ')
    verify(root,required)
    for name,key in [('docs/review_operation_diagnostic_protocol_v1.json','protocol_sha256'),('docs/review_operation_diagnostic_source_freeze_v1.json','source_freeze_sha256'),('docs/review_operation_diagnostic_release_v1.json','release_sha256')]:
        if inputs.get(name)!=m[key]:raise RuntimeError('Producer activation identity not frozen')
    return plan,freeze,m


def arm(archive,group,fields=None):
    return {k:archive[group['group_id']+'__'+k] for k in (group['array_names'] if fields is None else sorted(fields))}


def run(root=ROOT):
    root=Path(root)
    if (root/RECEIPT).exists() or (root/ATTEMPT).exists():raise RuntimeError('Never overwrite audit attempt or receipt')
    (root/ATTEMPT).mkdir(parents=True);state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),raw_market_parser_called=False,final_scenarios_loaded=False)
    writej(root/ATTEMPT/'manifest.json',state)
    try:
        plan,freeze,parent=authorize(root);snapshot={**freeze['source_sha256'],**freeze['input_sha256']};plan_sha=sha(root/PLAN);freeze_sha=sha(root/FREEZE)
        groups=readj(root/OLD/'trace_groups.json');selected=[g for g in groups if g['kind']=='outer_selected'];newgroups=readj(root/PARENT/'trace_groups.json')
        if len(selected)!=5 or sorted(s for g in selected for s in g['scenario_ids'])!=list(range(1,121)):raise AssertionError('Five fullOOF original120IDs exactly once')
        c=Checks();tables={k:[] for k in ['annual_phases','account_phases','original_accounts','review_cost_paths','review_cost_break_even','low_exposure_paths','baseline_paths']};expected_groups=[];checks=[];expected_keys=set()
        with np.load(root/OLD/'traces.npz',allow_pickle=False) as oldnpz,np.load(root/PARENT/'traces.npz',allow_pickle=False) as newnpz:
            def trace(prefix,name,expected,exact=False):
                key=prefix+'__'+name;expected_keys.add(key);c.array(newnpz[key],expected,key,exact=exact)
            for group in selected:
                ids=group['scenario_ids'];job=group['job_id'];train=group['training_ids'];target=group['target']
                if len(ids)!=24 or len(set(train))!=96 or set(ids)&set(train) or any(s not in range(1,121) for s in [*ids,*train]):raise AssertionError('Own96 and24 development IDs only')
                a=arm(oldnpz,group,MAIN_FIELDS)
                if a['fees'].shape!=(24,30):raise AssertionError('FullH30 required')
                refs={}
                for policy in REFS:
                    candidates=[r for r in groups if r['kind']=='reference' and r['job_id']==job and r['policy']==policy]
                    if len(candidates)!=1 or candidates[0]['scenario_ids']!=ids:raise AssertionError('Ownjob sameID references')
                    refs[policy]=arm(oldnpz,candidates[0])
                gross,rg=independent_market(refs['exact_thirds_annual']);annual,parts,accounts=phases(a,ids,job,rg,c)
                tables['annual_phases']+=annual;tables['account_phases']+=parts;tables['original_accounts']+=accounts
                prefix=job+'__source'
                for name,value in a.items():trace(prefix,name,value,True)
                expected_groups.append(dict(group_id=prefix,kind='saved_outer_selected',scenario_ids=ids,training_ids=train,target=target,array_names=sorted(a)))
                for i,sid in enumerate(ids):
                    tables['baseline_paths'].append(dict(policy='main',scenario_id=sid,job_id=job,**financial(a['real_wealth'][i],a['fees'][i],a['trades'][i])))
                    tau=int(a['first_guard_trigger'][i]);valid=0<tau<H
                    for policy,r in refs.items():
                        tables['baseline_paths'].append(dict(policy=policy,scenario_id=sid,job_id=job,**financial(r['real_wealth'][i],r['fees'][i],r['trades'][i])))
                        tables['review_cost_break_even'].append(dict(job_id=job,scenario_id=sid,reference=policy,guard_year=None if tau<0 else tau,applicable=valid,
                            **break_even(a['real_wealth'][i,-1],r['real_wealth'][i,-1],math.fsum(a['holdings'][i,tau]) if valid else 0,valid)))
                for q in FEES:
                    policy='review_fee_'+str(q);packed={k:[] for k in ('real_wealth','real_peak','running_mdd')}
                    for i,sid in enumerate(ids):
                        tau=int(a['first_guard_trigger'][i]);cost=cost_path(a['real_wealth'][i],a['real_peak'][i],a['running_mdd'][i],a['holdings'][i],a['cpi'][i],tau,q)
                        metrics=financial(cost['real_wealth'],a['fees'][i],a['trades'][i],cost['running_mdd'][-1]);metrics['nominal_fees']+=cost['review_fee_nominal']
                        tables['review_cost_paths'].append(dict(policy=policy,job_id=job,scenario_id=sid,fee_fraction=q,applicable=cost['applicable'],guard_year=None if tau<0 else tau,debit_endpoint=cost['debit_endpoint'],
                            review_fee_nominal=cost['review_fee_nominal'],review_fee_real=cost['review_fee_real'],first_crossing=cost['first_crossing'],**metrics))
                        for name in packed:packed[name].append(cost[name])
                    prefix=job+'__'+policy
                    for name,values in packed.items():trace(prefix,name,np.asarray(values),exact=q==0)
                    expected_groups.append(dict(group_id=prefix,kind='review_cost_sensitivity',scenario_ids=ids,fee_fraction=q,array_names=list(packed)))
                for fraction in FRACTIONS:
                    w=fraction*np.asarray(target)+(1-fraction)*np.array([0.,0.,1.])
                    for rule in ('hold','annual'):
                        policy='a'+str(fraction)+'_'+rule;r=static_reference(gross,a['cpi'],w,rule);prefix=job+'__'+policy
                        for name,value in r.items():trace(prefix,name,value)
                        expected_groups.append(dict(group_id=prefix,kind='hypothetical_static_reference',scenario_ids=ids,training_ids=train,target=w.tolist(),fraction=fraction,rule=rule,array_names=sorted(r),service_admitted=False))
                        if fraction==1 and rule=='annual':
                            maximum=0.
                            if set(r)!=set(refs['same_target_annual']):raise AssertionError('Every old annual reference field')
                            for name in r:
                                c.array(r[name],refs['same_target_annual'][name],'independent originalannual '+name)
                                actual=newnpz[prefix+'__'+name];old=refs['same_target_annual'][name]
                                if actual.dtype.kind not in 'bUSiu':
                                    finite=np.isfinite(actual)&np.isfinite(old)
                                    if finite.any():maximum=max(maximum,float(np.max(np.abs(actual[finite]-old[finite]))))
                            checks.append(dict(job_id=job,fields=len(r),maximum_absolute_difference=maximum,absolute_tolerance=ABS,relative_tolerance=REL))
                        for i,sid in enumerate(ids):tables['low_exposure_paths'].append(dict(policy=policy,job_id=job,scenario_id=sid,fraction=fraction,rule=rule,service_admitted=False,**financial(r['real_wealth'][i],r['fees'][i],r['trades'][i])))
            if set(newnpz.files)!=expected_keys:raise AssertionError('All and only declared new trace fields')
        c.tree(newgroups,expected_groups,'tracegroups');c.tree(readj(root/PARENT/'annual_reproduction_checks.json'),checks,'originalannual checkmetadata')
        producer_summaries=[];extra_summaries=[]
        for kind,key in [('original','baseline_paths'),('review_cost','review_cost_paths'),('low_exposure','low_exposure_paths')]:
            for policy in sorted({r['policy'] for r in tables[key]}):
                subset=[r for r in tables[key] if r['policy']==policy]
                if sorted(r['scenario_id'] for r in subset)!=list(range(1,121)):raise AssertionError('Every financial summary full120IDs')
                producer_summaries.append(dict(kind=kind,policy=policy,**summary(subset)));extra_summaries.append(dict(kind=kind,policy=policy,**summary(subset,True)))
        tables['financial_summary']=producer_summaries
        pair_rows,pair_summaries=paired(tables['low_exposure_paths'],tables['baseline_paths']);tables['low_exposure_paired_paths']=pair_rows
        keys={'annual_phases':('job_id','scenario_id','period'),'account_phases':('job_id','scenario_id','phase'),'original_accounts':('job_id','scenario_id'),
            'review_cost_paths':('policy','scenario_id'),'review_cost_break_even':('reference','scenario_id'),'low_exposure_paths':('policy','scenario_id'),
            'baseline_paths':('policy','scenario_id'),'financial_summary':('kind','policy'),'low_exposure_paired_paths':('policy','reference','scenario_id')}
        for name,expected in tables.items():c.table(csvrows(root/PARENT/(name+'.csv.gz')),expected,keys[name],name)
        c.tree(readj(root/PARENT/'low_exposure_paired_summary.json'),pair_summaries,'allpairedsummaries')
        actual_op=readj(root/PARENT/'operation_summary.json');phase_summary={}
        for phase in PHASES:
            z=[r for r in tables['annual_phases'] if r['phase']==phase]
            phase_summary[phase]=dict(periods=len(z),mean_equity_weight=math.fsum(r['equity_weight'] for r in z)/len(z) if z else None,
                nominal_fees=math.fsum(r['nominal_fee'] for r in z),total_log_growth=math.fsum(r['net_log_growth'] for r in z))
        accounts=tables['original_accounts'];expected_op=dict(original_n=120,guard_n=sum(r['first_guard'] is not None for r in accounts),no_guard_n=sum(r['first_guard'] is None for r in accounts),
            within_H_review_debits=sum(r['review_debit_within_H'] for r in accounts),phase_summary=phase_summary,
            cost_assumption='Unknown one-time fee scenarios only. Proportional asset debit after trigger-year market at endpoint tau, before decision tau+1, only tau<H. No service benefit assumed; original peak/CPI retained.',
            mdd_caution='Full-H risk is recalculated with original history; segment maxima are never added or treated as causal risk contributions.',
            reference_caution='Eight predeclared simple low-exposure references, not risk-matched/certified/selected products. All120 OOF IDs retained. Adaptive development diagnostic, no independent new sample.')
        c.tree(actual_op,expected_op,'operation_summary')
        verify(root,snapshot)
        if sha(root/PLAN)!=plan_sha or sha(root/FREEZE)!=freeze_sha:raise AssertionError('Audit sourcefreeze changed')
        result=dict(version='independent-review-operation-audit-v1',status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),producer_manifest_sha256=sha(root/PARENT/'manifest.json'),
            plan_sha256=plan_sha,source_freeze_sha256=freeze_sha,source_and_input_sha256_at_start=snapshot,source_and_input_sha256_at_end={k:sha(root/k) for k in snapshot},
            scalar_checks=c.scalar_checks,maximum_absolute_numeric_difference=c.max_absolute_error,absolute_tolerance=ABS,relative_tolerance=REL,
            counts={k:len(v) for k,v in tables.items()},trace_groups=len(expected_groups),trace_fields=len(expected_keys),paired_summaries=len(pair_summaries),
            originalannual_fields_per_fold=[r['fields'] for r in checks],extra_complete_financial_summaries=extra_summaries,
            extra_summary_definition='All17 full120 same-ID summaries, medianMDD plus fractional-mass CVaR90 added independently. Producer summary schema unchanged. Costcases retain originalpredebitpeak plus immediatefee observation.',
            scope='Independent arithmetic/ledger/ref reconstruction audit on existing development paths; not a new model evaluation, new sample, forecast/action oracle, preference estimate or selection.',
            raw_market_parser_called=False,final_scenarios_loaded=False,new_main_policy_rollouts=0,independent_static_reference_accounts_reconstructed=960,new_fits=0,new_selections=0,main_policy_changed=False,producer_outputs_modified=False)
        writej(root/RECEIPT,result);state.update(status='passed',completed_utc=result['completed_utc'],receipt_sha256=sha(root/RECEIPT));writej(root/ATTEMPT/'manifest.json',state)
        print(json.dumps(dict(status='passed',checks=c.scalar_checks,receipt=str(RECEIPT))))
    except BaseException as error:
        (root/ATTEMPT/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(error).__name__,error=str(error),automatic_retry=False);writej(root/ATTEMPT/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:print(json.dumps(dict(source_only=True,actual_audit_executed=False)))
