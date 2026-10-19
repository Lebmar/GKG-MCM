"""Independent saved-array audit; no producer, policy or raw market imports."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PLAN=Path('docs/support_failure_independent_audit_plan_v1.json')
FREEZE=Path('docs/support_failure_independent_audit_source_freeze_v1.json')
ACTIVATION=Path('docs/support_failure_independent_audit_activation_v1.json')
PARENT=Path('results/support_failure_diagnostic_v2')
CANON=Path('results/state_budget_mitigation_v1')
STRESS=Path('results/state_budget_mitigation_stress_v1')
ATTEMPT=Path('results/support_failure_independent_audit_attempt_v1')
RECEIPT=PARENT/'independent_audit.json'
SOURCES=('scripts/audit_support_failure.py','tests/test_audit_support_failure.py')
SETTINGS=('base','fee_0000','fee_00125','fee_005','fee_01','pooled_support','actual_inflation_plus_02')
COHORTS=('outer_OOF','full_resubstitution')
POLICIES=('state_budget_learner','same_target_B','same_target_annual','exact_thirds_annual')
CORNERS=('BB','BS','SB','SS')
MDD_PARTS=('MDD_CPI_first','MDD_nominal_after_CPI','MDD_nominal_first','MDD_CPI_after_nominal','MDD_interaction','MDD_total')
ATOL=3e-10
RTOL=5e-13


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def normal(v):
    if isinstance(v,np.generic):return normal(v.item())
    if isinstance(v,float) and math.isnan(v):return None
    if isinstance(v,str):
        if v=='True':return True
        if v=='False':return False
        if v[:1] in ('[','{'):
            try:return json.loads(v)
            except ValueError:pass
    return v
def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


class Checks:
    def __init__(self):self.count=0;self.maximum=0.;self.nonunique_binding_rows=0
    def same(self,a,b,*,atol=ATOL,rtol=RTOL):
        a,b=normal(a),normal(b)
        if isinstance(b,dict):
            if not isinstance(a,dict) or set(a)!=set(b):raise AssertionError('Dictionary inventory mismatch')
            for k in b:self.same(a[k],b[k],atol=atol,rtol=rtol)
        elif isinstance(b,(list,tuple)):
            if not isinstance(a,(list,tuple)) or len(a)!=len(b):raise AssertionError('Complete ordered list required')
            for x,y in zip(a,b):self.same(x,y,atol=atol,rtol=rtol)
        elif b is None:
            if a is not None:raise AssertionError('Financial NULL mismatch')
            self.count+=1
        elif isinstance(b,bool):
            if not isinstance(a,bool) or a!=b:raise AssertionError('Boolean event/flag mismatch')
            self.count+=1
        elif isinstance(b,(int,float,np.integer,np.floating)):
            if a is None or isinstance(a,bool) or not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(a),float(b),abs_tol=atol,rel_tol=rtol):
                raise AssertionError('Independent numerical mismatch: '+str((a,b)))
            self.maximum=max(self.maximum,abs(float(a)-float(b)));self.count+=1
        else:
            if a!=b:raise AssertionError('Identity/category mismatch: '+str((a,b)))
            self.count+=1
    def array(self,a,b,*,atol=ATOL,rtol=RTOL):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True):raise AssertionError('Complete saved array mismatch')
        valid=np.isfinite(a)&np.isfinite(b)
        if valid.any():self.maximum=max(self.maximum,float(np.max(np.abs(a[valid]-b[valid]))))
        self.count+=a.size


class Table:
    def __init__(self,path,keys):
        self.frame=pd.read_csv(path,float_precision='round_trip');self.keys=keys;self.rows={}
        for raw in self.frame.to_dict('records'):
            row={k:normal(v) for k,v in raw.items()};key=tuple(row[k] for k in keys)
            if key in self.rows:raise AssertionError('Duplicated complete population key')
            self.rows[key]=row
        self.seen=set()
    def check(self,check,expected):
        key=tuple(expected[k] for k in self.keys)
        if key not in self.rows or key in self.seen:raise AssertionError('Missing/repeated original population row')
        self.seen.add(key);row=self.rows[key]
        for name,value in expected.items():
            if name not in row:raise AssertionError('Missing declared output field: '+name)
            check.same(row[name],value)
    def complete(self):
        if self.seen!=set(self.rows):raise AssertionError('Unverified rows remain')


def metrics_one(real,L=.4):
    r=np.asarray(real,float)
    if r.ndim!=1 or len(r)<2 or np.any(r<=0) or not np.isfinite(r).all() or abs(r[0]-100)>3e-10:raise ValueError('Original positive complete path required')
    peaks=[];running=[];peak=float(r[0]);mdd=0.;first=-1
    for t,value in enumerate(r):
        peak=max(peak,float(value));mdd=max(mdd,1-float(value)/peak);peaks.append(peak);running.append(mdd)
        if first<0 and mdd-L>1e-12:first=t
    growth=math.log(float(r[-1])/100)/(len(r)-1)
    return dict(real=r,peak=peaks,running_mdd=running,mdd=mdd,first_crossing=first,breach=bool(mdd-L>1e-12),log_growth=growth,cagr=math.expm1(growth))


def corner_values(vb,cb,vs,cs):
    vb,cb,vs,cs=[np.asarray(x,float) for x in (vb,cb,vs,cs)]
    if any(x.shape!=vb.shape or np.any(x<=0) for x in (cb,vs,cs)) or vb.ndim!=1 or abs(cb[0]-1)>1e-14 or abs(cs[0]-1)>1e-14:raise ValueError('Aligned original CPI/nominal paths required')
    data={name:metrics_one(v/c) for name,v,c in [('BB',vb,cb),('BS',vb,cs),('SB',vs,cb),('SS',vs,cs)]}
    bb,bs,sb,ss=[data[k]['mdd'] for k in CORNERS]
    parts=dict(MDD_CPI_first=bs-bb,MDD_nominal_after_CPI=ss-bs,MDD_nominal_first=sb-bb,MDD_CPI_after_nominal=ss-sb,
        MDD_interaction=ss-sb-bs+bb,MDD_total=ss-bb)
    return data,np.log(vs/vb),-np.log(cs/cb),parts


def initial_values(target,rate,rho,training):
    w=np.asarray(target,float);s=np.asarray(training,float)
    denominator=1+float(rate)*(float(w[0])+float(w[1]));post=np.asarray([100*float(v)/denominator for v in w])
    fee=float(rate)*math.fsum(post[:2]);floor=60.;b=max(floor,100-float(rho)*(100-floor))
    values=np.asarray([[math.fsum(float(row[k])*float(post[k]) for k in range(3)) for row in year] for year in s.transpose(1,0,2)]).T
    return post,fee,b,values


def minima(training,post,cpi):
    s=np.asarray(training);y=np.asarray(post)
    values=np.sum(s*y[None,:],axis=1)/float(cpi)
    return float(values.min()),values


def dependencies(root=ROOT):
    """Metadata/byte hashes only; callable while preparing this audit freeze."""
    root=Path(root);m=readj(root/PARENT/'manifest.json')
    if m.get('status')!='complete' or m.get('final_scenarios_loaded') is not False or m.get('raw_csv_opened') is not False or m.get('new_policy_runs')!=0 or m['sources_and_inputs_at_start']!=m['sources_and_inputs_at_end']:
        raise RuntimeError('Completed unchanged saved-only diagnostic required')
    mapping={**m['sources_and_inputs_at_start'],str(PARENT/'manifest.json'):sha(root/PARENT/'manifest.json')}
    mapping.update({str(PARENT/name):digest for name,digest in m['outputs_sha256'].items()})
    for name in ('docs/support_failure_diagnostic_protocol_v2.json','docs/support_failure_diagnostic_source_freeze_v2.json','docs/support_failure_diagnostic_activation_v2.json'):
        mapping[name]=sha(root/name)
    for folder,audit_path in ((CANON,CANON/'independent_audit.json'),(STRESS,Path('results/state_budget_mitigation_stress_audit_v1/manifest.json'))):
        original=readj(root/folder/'manifest.json');audit=readj(root/audit_path);parent=str(folder/'manifest.json')
        if original.get('status')!='complete' or audit.get('status')!='passed' or original.get('final_scenarios_loaded') is not False or audit.get('final_scenarios_loaded') is not False:
            raise RuntimeError('Original development parents must remain complete and independently passed')
        if original['sources_and_inputs_at_start']!=original['sources_and_inputs_at_end']:raise RuntimeError('Original parent changed during execution')
        bindings=[audit.get('parent_manifest_sha256'),audit.get('producer_manifest_sha256')]
        for key in ('sources_and_inputs','sources_and_artifacts'):
            if key+'_at_start' in audit:
                if audit[key+'_at_start']!=audit[key+'_at_end']:raise RuntimeError('Original audit start/end mismatch')
                bindings.append(audit[key+'_at_start'].get(parent))
        if sha(root/parent) not in bindings or mapping.get(parent)!=sha(root/parent) or mapping.get(str(audit_path))!=sha(root/audit_path):raise RuntimeError('Complete current original audit binding required')
        for name,digest in original['outputs_sha256'].items():
            if mapping.get(str(folder/name))!=digest:raise RuntimeError('All original saved outputs must be frozen in audit inputs')
    for name,digest in mapping.items():
        p=Path(name)
        if not p.parts or p.is_absolute() or '..' in p.parts or p.as_posix()!=name or name.startswith('data/') or 'final' in name or sha(root/p)!=digest:raise RuntimeError('No changed/raw/final audit dependency')
    return mapping


def preflight(root=ROOT):
    root=Path(root);p,f,a=readj(root/PLAN),readj(root/FREEZE),readj(root/ACTIVATION)
    if (p.get('version')!='independent-support-failure-audit-v1' or p.get('execution_ready') is not False or p.get('allow_final') is not False
            or p.get('development_ids')!=list(range(1,121)) or f.get('protocol_sha256')!=sha(root/PLAN)
            or not set(SOURCES)<=set(f.get('source_sha256',{})) or a.get('execution_ready') is not True
            or a.get('protocol_sha256')!=sha(root/PLAN) or a.get('source_freeze_sha256')!=sha(root/FREEZE)):
        raise RuntimeError('Separately activated frozen independent audit required')
    snapshot={**f['source_sha256'],**f['input_sha256']}
    required=dependencies(root)
    if any(snapshot.get(k)!=v for k,v in required.items()):raise RuntimeError('Freeze every parent/derived consumed artifact')
    for name,digest in snapshot.items():
        if sha(root/name)!=digest:raise RuntimeError('Audit source/input changed')
    return snapshot


def load_arm(archive,g,keys):
    if g.get('financial_available') is False or not g['array_names']:raise AssertionError('No financial arm may be fabricated')
    if not set(keys)<=set(g['array_names']):raise AssertionError('Original saved fields missing')
    return {k:archive[g['group_id']+'__'+k] for k in keys}


def outer_reference_groups(groups):
    """Identify own outer jobs positively; inner/full distractors are irrelevant."""
    outer=[g for g in groups if g['kind']=='outer_selected'];jobs={g['job_id']:g for g in outer}
    if len(outer)!=5 or len(jobs)!=5:raise AssertionError('Exactly five distinct selected outer jobs required')
    refs=[g for g in groups if g['kind']=='reference' and g['policy']=='exact_thirds_annual' and g['job_id'] in jobs]
    if len(refs)!=5 or len({g['job_id'] for g in refs})!=5:raise AssertionError('Exactly one positive thirds reference per selected outer job required')
    for g in refs:
        if g['scenario_ids']!=jobs[g['job_id']]['scenario_ids']:raise AssertionError('Outer reference IDs must match its selected job')
    return refs


def reconstruct(cz,cgroups,check):
    nominal=np.zeros((120,30,3));inflation=np.zeros((120,30));seen=[]
    thirds=outer_reference_groups(cgroups)
    for g in thirds:
        a=load_arm(cz,g,['holdings','posttrade_holdings','cpi'])
        if np.any(a['posttrade_holdings']<=0):raise AssertionError('Positive per-asset denominator required')
        for i,sid in enumerate(g['scenario_ids']):
            if not 1<=sid<=120 or sid in seen:raise AssertionError('Original development identity duplicated or outside1-120')
            seen.append(sid)
            for t in range(30):
                inflation[sid-1,t]=a['cpi'][i,t+1]/a['cpi'][i,t]
                for k in range(3):nominal[sid-1,t,k]=a['holdings'][i,t+1,k]/a['posttrade_holdings'][i,t,k]
    if sorted(seen)!=list(range(1,121)) or np.any(nominal<=0) or np.any(inflation<=0):raise AssertionError('All120 original positive joint paths required')
    contexts={g['job_id']:g for g in cgroups if g['kind'] in ('outer_selected','full_selected')};training={}
    if set(contexts)!={'c0_refit','c1_refit','c2_refit','c3_refit','c4_refit','c5_refit'}:raise AssertionError('All six own-fit contexts required')
    for job,g in contexts.items():
        tr=np.asarray(g['training_ids'],int);ids=np.asarray(g['scenario_ids'],int)
        if job=='c0_refit':
            if tr.tolist()!=list(range(1,121)) or ids.tolist()!=list(range(1,121)):raise AssertionError('Full is separate resubstitution')
        elif len(tr)!=96 or len(ids)!=24 or set(tr)&set(ids):raise AssertionError('Own96/24 disjoint fold required')
        training[job]=nominal[tr-1]/inflation[tr-1,:,None]
        check.array(cz[g['group_id']+'__training_real_gross'],training[job].transpose(1,0,2),atol=3e-11)
        check.array(cz[g['group_id']+'__realized_real_gross'],nominal[ids-1]/inflation[ids-1,:,None],atol=3e-11)
    return nominal,inflation,contexts,training


def audit_initial(root,groups,witnesses,training,check):
    tables=[Table(root/PARENT/name,keys) for name,keys in [
        ('initial_support_contexts.csv.gz',('setting_id','job_id')),('initial_support_by_year.csv.gz',('job_id','training_year')),
        ('initial_support_members.csv.gz',('job_id','training_id','training_year'))]]
    witness={(w['setting_id'],w['context_id']):w['witness'] for w in witnesses}
    for g in groups:
        if g['policy']!='state_budget_learner':continue
        train=training[g['job_id']];setting=g['setting'];rho=g['selected_candidate']['release_fraction'];n,h,_=train.shape
        y,fee,b,values=initial_values(g['own_fit_target'],setting['cost_rate'],rho,train)
        cal=float(values[:,0].min());pool=float(values.min());declared=pool if setting['support_mode']=='pooled' else cal
        if pool>cal+ATOL:raise AssertionError('Pool must contain calendar support')
        check.same(g['financial_available'],bool(declared>=b-1e-10))
        w=witness[(g['setting_id'],g['context_id'])];check.array(w['candidate_posttrade'][1],y);check.same(w['budget_floor'],b)
        check.same(w['candidate_min_support_slack'][1],declared-b)
        tables[0].check(check,dict(setting_id=g['setting_id'],cohort=g['cohort'],context_id=g['context_id'],job_id=g['job_id'],
            training_n=n,evaluated_original_n=len(g['scenario_ids']),target=g['own_fit_target'],rho=rho,cost_rate=setting['cost_rate'],support_mode=setting['support_mode'],
            status=g['status'],financial_available=g['financial_available'],full_target_fee=fee,original_floor=60.,budget_floor=b,
            calendar_year1_min=cal,pooled_min=pool,calendar_year1_slack=cal-b,pooled_slack=pool-b,
            pooled_violating_training_rows=int(np.sum(values<b-1e-10)),calendar_violating_training_rows=int(np.sum(values[:,0]<b-1e-10))))
        if g['setting_id']!='base':continue
        for t in range(h):
            minimum=float(values[:,t].min());ids=[g['training_ids'][i] for i,value in enumerate(values[:,t]) if abs(value-minimum)<=1e-10]
            tables[1].check(check,dict(job_id=g['job_id'],cohort=g['cohort'],training_year=t+1,training_n=n,
                min_full_target_next_real=minimum,budget_floor=b,min_slack=minimum-b,violating_rows=int(np.sum(values[:,t]<b-1e-10)),
                minimum_member_original_ids=ids,negative_real_bill_rows=int(np.sum(train[:,t,2]<1)),min_real_bill_gross=float(train[:,t,2].min())))
            for i,sid in enumerate(g['training_ids']):
                tables[2].check(check,dict(job_id=g['job_id'],cohort=g['cohort'],training_id=sid,training_year=t+1,
                    equity_real_gross=float(train[i,t,0]),bond_real_gross=float(train[i,t,1]),bill_real_gross=float(train[i,t,2]),
                    full_target_next_real=float(values[i,t]),slack=float(values[i,t]-b),budget_floor=b,negative_real_bill=bool(train[i,t,2]<1)))
    for table in tables:table.complete()
    return dict(context_rows=len(tables[0].rows),year_rows=len(tables[1].rows),joint_member_rows=len(tables[2].rows))


def audit_corners(root,z,groups,check):
    paths=Table(root/PARENT/'inflation_four_corner_paths.csv.gz',('cohort','policy','job_id','scenario_id'))
    endpoints=Table(root/PARENT/'inflation_four_corner_endpoints.csv.gz',('cohort','policy','job_id','scenario_id','endpoint'))
    summaries=[];controls=0;corners_count=0
    keys=('nominal_wealth','cpi','real_wealth','holdings','posttrade_holdings','fees','running_mdd','first_crossing')
    for cohort in COHORTS:
        for policy in POLICIES:
            rows=[];bg=[g for g in groups if g['setting_id']=='base' and g['cohort']==cohort and g['policy']==policy]
            if sorted(s for g in bg for s in g['scenario_ids'])!=list(range(1,121)):raise AssertionError('No corner ID filtering or full/outer pooling')
            for g in bg:
                candidates=[x for x in groups if x['setting_id']=='actual_inflation_plus_02' and x['job_id']==g['job_id'] and x['policy']==policy]
                if len(candidates)!=1 or candidates[0]['scenario_ids']!=g['scenario_ids']:raise AssertionError('Exact same original-ID shift pairing required')
                shift=candidates[0];meta=dict(cohort=cohort,policy=policy,job_id=g['job_id'])
                if not g['financial_available'] or not shift['financial_available']:
                    for sid in g['scenario_ids']:
                        row={k:None for k in paths.frame.columns};row.update(**meta,scenario_id=sid,base_financial=g['financial_available'],shift_financial=shift['financial_available'])
                        paths.check(check,row)
                    continue
                b,s=load_arm(z,g,keys),load_arm(z,shift,keys)
                if policy in ('same_target_annual','exact_thirds_annual'):
                    for key in ('nominal_wealth','holdings','posttrade_holdings','fees'):
                        if not np.array_equal(b[key],s[key]):raise AssertionError('Annual/thirds nominal negative control must be exact')
                        controls+=b[key].size
                for i,sid in enumerate(g['scenario_ids']):
                    data,nominal,erosion,parts=corner_values(b['nominal_wealth'][i],b['cpi'][i],s['nominal_wealth'][i],s['cpi'][i])
                    for corner,old in [('BB',b),('SS',s)]:
                        check.array(data[corner]['real'],old['real_wealth'][i]);check.array(data[corner]['running_mdd'],old['running_mdd'][i])
                        check.same(data[corner]['first_crossing'],int(old['first_crossing'][i]),atol=0,rtol=0)
                    row=dict(**meta,scenario_id=sid,base_financial=True,shift_financial=True,
                        terminal_log_nominal_component=float(nominal[-1]),terminal_log_CPI_component=float(erosion[-1]),**parts)
                    for corner in CORNERS:
                        for field in ('mdd','first_crossing','breach','log_growth','cagr'):row[corner+'_'+field]=data[corner][field]
                    paths.check(check,row);rows.append(row);corners_count+=4
                    check.same(parts['MDD_CPI_first']+parts['MDD_nominal_after_CPI'],parts['MDD_total'])
                    check.same(parts['MDD_nominal_first']+parts['MDD_CPI_after_nominal'],parts['MDD_total'])
                    for t in range(31):
                        endpoint=dict(**meta,scenario_id=sid,endpoint=t,log_nominal_path_component=float(nominal[t]),log_CPI_component=float(erosion[t]))
                        check.same(math.log(data['SS']['real'][t]/data['BB']['real'][t]),float(nominal[t]+erosion[t]))
                        for corner in CORNERS:
                            for field in ('real','peak','running_mdd'):endpoint[corner+'_'+field]=float(data[corner][field][t])
                        endpoints.check(check,endpoint)
            stats=dict(cohort=cohort,policy=policy,original_n=120,paired_financial_n=len(rows),corners={},ordered_MDD_means={})
            for corner in CORNERS:stats['corners'][corner]=dict(breach_count=sum(r[corner+'_breach'] for r in rows) if rows else None,
                mean_mdd=math.fsum(r[corner+'_mdd'] for r in rows)/len(rows) if rows else None,
                median_cagr=float(np.median([r[corner+'_cagr'] for r in rows])) if rows else None)
            for part in MDD_PARTS:stats['ordered_MDD_means'][part]=math.fsum(r[part] for r in rows)/len(rows) if rows else None
            stats['event_quadrants']=[dict(**{k+'_breach':v for k,v in zip(CORNERS,bits)},original_ids=[r['scenario_id'] for r in rows if tuple(r[k+'_breach'] for k in CORNERS)==bits]) for bits in itertools.product((False,True),repeat=4)]
            summaries.append(stats)
    check.same(readj(root/PARENT/'inflation_four_corner_summary.json'),summaries)
    paths.complete();endpoints.complete()
    return dict(paired_paths=len(paths.rows),endpoints=len(endpoints.rows),independent_corner_paths=corners_count,nominal_control_scalars=controls)


def audit_support(root,z,groups,training,gross,inflation,check):
    table=Table(root/PARENT/'directional_support_periods.csv.gz',('setting_id','cohort','job_id','scenario_id','year'))
    records=[];keys=('holdings','posttrade_holdings','fees','cpi','real_wealth','real_peak','running_mdd','realized_real_gross','gate_evaluated',
        'automatic_service','guard_triggered','was_pending_input','support_feasible','budget_floor','worst_real_shortfall_selected','first_guard_trigger','first_crossing','actual_decision_source')
    for g in groups:
        if g['policy']!='state_budget_learner' or not g['financial_available']:continue
        a=load_arm(z,g,keys);tr=training[g['job_id']];pooled=tr.reshape(-1,3);rho=g['selected_candidate']['release_fraction']
        for i,sid in enumerate(g['scenario_ids']):
            check.array(a['cpi'][i,1:]/a['cpi'][i,:-1],inflation[sid-1]+g['setting']['actual_inflation_shift'],atol=3e-11)
            check.array(a['realized_real_gross'][i],gross[sid-1]/(inflation[sid-1]+g['setting']['actual_inflation_shift'])[:,None],atol=3e-11)
            for t in range(30):
                x=a['holdings'][i,t];y=a['posttrade_holdings'][i,t];fee=float(a['fees'][i,t]);c=float(a['cpi'][i,t]);cn=float(a['cpi'][i,t+1]);r=float(a['real_wealth'][i,t]);p=float(a['real_peak'][i,t])
                check.same(math.fsum(y)+fee,math.fsum(x));check.array(y*gross[sid-1,t],a['holdings'][i,t+1])
                real_gross=gross[sid-1,t]*c/cn;actual=math.fsum(float(y[k])*float(gross[sid-1,t,k])/cn for k in range(3))
                check.same(actual,float(a['real_wealth'][i,t+1]));check.same(math.fsum(x)/c,r)
                cal,cv=minima(tr[:,t],y,c);pool,pv=minima(pooled,y,c)
                declared=pool if g['setting']['support_mode']=='pooled' else cal;floor=.6*p;b=max(floor,r-rho*(r-floor))
                active=bool(a['gate_evaluated'][i,t])
                if active:
                    check.same(float(a['budget_floor'][i,t]),b);check.same(float(a['worst_real_shortfall_selected'][i,t]),max(b-declared,0.))
                key=(g['setting_id'],g['cohort'],g['job_id'],sid,t+1);saved=table.rows.get(key)
                if saved is None:raise AssertionError('Every funded original year required')
                try:ci=g['training_ids'].index(int(saved['calendar_worst_training_id']));pi=g['training_ids'].index(int(saved['pooled_worst_training_id']));pt=int(saved['pooled_worst_training_year'])-1
                except ValueError as exc:raise AssertionError('Binding row must be an own-training joint row') from exc
                if not 0<=pt<30:raise AssertionError('Binding year outside original training support')
                check.same(float(cv[ci]),cal);check.same(float(pv[pi*30+pt]),pool)
                check.nonunique_binding_rows+=int(np.sum(np.isclose(cv,cal,atol=ATOL,rtol=RTOL))>1 or np.sum(np.isclose(pv,pool,atol=ATOL,rtol=RTOL))>1)
                asset=[float(y[k])*(float(real_gross[k])-1)/c for k in range(3)]
                nominal_gain=math.fsum(float(y[k])*(float(gross[sid-1,t,k])-1)/cn for k in range(3));nominal_fee=-fee/cn;erosion=-math.fsum(x)*(cn/c-1)/cn
                change=actual-r;check.same(math.fsum(asset)-fee/c,change);check.same(nominal_gain+nominal_fee+erosion,change)
                row=dict(setting_id=g['setting_id'],cohort=g['cohort'],job_id=g['job_id'],scenario_id=sid,year=t+1,
                    gate_evaluated=active,ordinary_service=bool(a['automatic_service'][i,t]),current_guard=bool(a['guard_triggered'][i,t]),
                    prior_pending=bool(a['was_pending_input'][i,t]),support_feasible=bool(a['support_feasible'][i,t]),decision_source=str(a['actual_decision_source'][i,t]),
                    current_real=r,original_peak=p,offline_original_floor=floor,offline_budget_floor=b,calendar_directional_min=cal,pooled_directional_min=pool,
                    declared_directional_min=declared,actual_next_real=actual,calendar_worst_training_id=int(saved['calendar_worst_training_id']),
                    pooled_worst_training_id=int(saved['pooled_worst_training_id']),pooled_worst_training_year=pt+1,
                    directional_miss=bool(actual<declared-1e-12*r),budget_failure=bool(actual<b-1e-12*r),actual_bill_real_gross=float(real_gross[2]),
                    actual_bill_negative_real=bool(real_gross[2]<1),fee_nominal=fee,fee_current_real=-fee/c,
                    equity_real_PnL=asset[0],bond_real_PnL=asset[1],bill_real_PnL=asset[2],nominal_asset_gain_at_next_CPI=nominal_gain,
                    nominal_fee_at_next_CPI=nominal_fee,inflation_erosion=erosion,real_wealth_change=change,
                    first_guard_year=int(a['first_guard_trigger'][i]),first_crossing=int(a['first_crossing'][i]))
                table.check(check,row);records.append(row)
    summaries=[]
    for setting in SETTINGS:
        for cohort in COHORTS:
            gs=[g for g in groups if g['policy']=='state_budget_learner' and g['setting_id']==setting and g['cohort']==cohort]
            rr=[r for r in records if r['setting_id']==setting and r['cohort']==cohort];funded=sum(len(g['scenario_ids']) for g in gs if g['financial_available'])
            stats=dict(setting_id=setting,cohort=cohort,original_n=sum(len(g['scenario_ids']) for g in gs),financial_n=funded,account_periods=len(rr),initial_review_n=120-funded,phases=[])
            for phase,field in [('ordinary','ordinary_service'),('guard_mitigation','current_guard'),('prior_pending','prior_pending')]:
                part=[r for r in rr if r[field]];stats['phases'].append(dict(phase=phase,periods=len(part),distinct_original_ids=len({r['scenario_id'] for r in part}),
                    directional_miss_periods=sum(r['directional_miss'] for r in part),budget_failure_periods=sum(r['budget_failure'] for r in part),
                    negative_real_bill_periods=sum(r['actual_bill_negative_real'] for r in part),supported_safe_actual_budget_failures=sum(r['gate_evaluated'] and r['support_feasible'] and r['budget_failure'] for r in part)))
            if sum(x['periods'] for x in stats['phases'])!=len(rr):raise AssertionError('Service phase partition incomplete')
            summaries.append(stats)
    check.same(readj(root/PARENT/'directional_support_summary.json'),summaries);table.complete()
    return dict(periods=len(table.rows),support_summaries=len(summaries),binding_tie_periods=check.nonunique_binding_rows)


def audit_inventory(root,check):
    old=pd.read_csv(root/STRESS/'path_metrics.csv.gz',float_precision='round_trip');new=pd.read_csv(root/PARENT/'path_inventory.csv.gz',float_precision='round_trip')
    keys=['setting_id','cohort','policy','scenario_id']
    if set(old.columns)!=set(new.columns) or len(old)!=6720 or len(new)!=6720:raise AssertionError('Complete original inventory columns/6720 rows required')
    for frame in (old,new):
        if frame.duplicated(keys).any():raise AssertionError('Duplicate original case')
        for _,part in frame.groupby(keys[:-1]):
            if sorted(part.scenario_id)!=list(range(1,121)):raise AssertionError('All original IDs once per population')
    a=old.sort_values(keys).to_dict('records');b=new.sort_values(keys).to_dict('records')
    for x,y in zip(a,b):check.same(y,x)
    return dict(inventory_rows=len(old))


def run(root=ROOT):
    root=Path(root)
    if (root/RECEIPT).exists() or (root/ATTEMPT).exists():raise RuntimeError('Fresh audit attempt/receipt required')
    snapshot=preflight(root);(root/ATTEMPT).mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='before_saved_numerical_parse',started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(root/PLAN),source_freeze_sha256=sha(root/FREEZE),activation_sha256=sha(root/ACTIVATION),
        producer_manifest_sha256=sha(root/PARENT/'manifest.json'),sources_and_artifacts_at_start=snapshot,
        producer_imported=False,policy_kernel_imported=False,raw_csv_opened=False,final_scenarios_loaded=False,new_policy_runs=0,new_fits=0,new_selections=0)
    writej(root/ATTEMPT/'manifest.json',state)
    try:
        check=Checks();counts=audit_inventory(root,check)
        cg=readj(root/CANON/'trace_groups.json');sg=readj(root/STRESS/'trace_groups.json');w=readj(root/STRESS/'initial_witnesses.json')
        if len(sg)!=168 or len(w)!=42:raise AssertionError('Complete fixed setting/context inventory required')
        with np.load(root/CANON/'traces.npz',allow_pickle=False) as cz,np.load(root/STRESS/'traces.npz',allow_pickle=False) as sz:
            gross,inflation,contexts,training=reconstruct(cz,cg,check)
            for g in sg:
                if g['policy']=='state_budget_learner':
                    original=contexts[g['job_id']]
                    for left,right in [('training_ids','training_ids'),('scenario_ids','scenario_ids'),('own_fit_target','target'),('selected_candidate','candidate')]:check.same(g[left],original[right])
            counts.update(audit_initial(root,sg,w,training,check));counts.update(audit_corners(root,sz,sg,check));counts.update(audit_support(root,sz,sg,training,gross,inflation,check))
        after={name:sha(root/name) for name in snapshot}
        if after!=snapshot or sha(root/PLAN)!=state['protocol_sha256'] or sha(root/FREEZE)!=state['source_freeze_sha256'] or sha(root/ACTIVATION)!=state['activation_sha256']:
            raise RuntimeError('Frozen audit evidence changed')
        state.update(status='passed',phase='complete',completed_utc=datetime.now(timezone.utc).isoformat(),checks=check.count,
            maximum_absolute_numeric_difference=check.maximum,counts=counts,sources_and_artifacts_at_end=after,
            tolerance=dict(absolute=ATOL,relative=RTOL,gross_absolute=3e-11,event='MDD-L>1e-12',initial_floor_absolute=1e-10),
            binding_scope='Every directional bound and recorded own-training binding row is independently checked. Numerically tied minima do not identify a unique member; no decision/Q replay.',
            interpretation='Saved development accounting and complete fixed-stress table audit only; no new independent markets or causal MDD allocation.')
        writej(root/ATTEMPT/'manifest.json',state);writej(root/RECEIPT,state)
    except BaseException as exc:
        (root/ATTEMPT/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        writej(root/ATTEMPT/'manifest.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:print(json.dumps(dict(source_only=True,numerical_audit_executed=False)))
