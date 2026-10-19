"""Independent scalar audit of annual target-share liquidity postprocessing.

No production diagnostic/kernel is imported. No strategy is run or fitted;
only fixed own-fit weights and individual annual returns enter ratio math.
"""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=Path('docs/liquidity_capacity_diagnostic_protocol.json')
PRODUCER_FREEZE=Path('docs/liquidity_capacity_diagnostic_source_freeze.json')
PLAN=Path('docs/liquidity_capacity_diagnostic_audit_plan.json')
FREEZE=Path('docs/liquidity_capacity_diagnostic_audit_source_freeze.json')
OUT=Path('results/liquidity_capacity_diagnostic_v1')
ATTEMPT=Path('results/liquidity_capacity_diagnostic_audit_attempt_v1')
PROFILES=(('A',5,'high',.3),('B',15,'medium',.1),('C',30,'low',0.),('D',25,'medium',.1),('E',6,'high',.3))
TOL=1e-12
FIELDS=('target_bill_share','mean_end_bill_share','minimum_end_bill_share','fraction_end_below_target_floor','max_hypothetical_liquidation_drag','mean_bill_real_loss_fraction')
COMMON=('job_id','scenario_id','profile','horizon','liquidity','target_floor','effective_assignment','assigned')
PATH_COLUMNS=COMMON+FIELDS
ANNUAL_COLUMNS=COMMON+('period','target_bill_share','end_bill_share','end_below_target_floor','hypothetical_liquidation_drag','hypothetical_available_bill_ratio','bill_real_return')
SUMMARY_FIELDS=('minimum_target_bill_share','maximum_target_bill_share','mean_end_bill_share','minimum_end_bill_share','mean_fraction_end_below_target_floor','paths_with_any_end_below_target_floor','maximum_hypothetical_liquidation_drag','mean_bill_real_loss_fraction')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def readj(path):return json.loads(Path(path).read_text())
def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def relative(name):
    if not isinstance(name,str):raise ValueError('Relative immutable path required')
    p=Path(name)
    if not name or name=='.' or p.is_absolute() or '..' in p.parts or p.as_posix()!=name:raise ValueError('Canonical relative path required')
    return p


def verify(root,mapping):
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty immutable hash map required')
    for name,digest in mapping.items():
        if not isinstance(digest,str) or len(digest)!=64 or sha(Path(root)/relative(name))!=digest:raise RuntimeError('Changed frozen evidence: '+name)


class Checks:
    def __init__(self):self.n=0;self.maximum_error=0.
    def check(self,test,label):
        self.n+=1
        if not test:raise AssertionError(label)
    def equal(self,expected,actual,label):
        if expected is None:self.check(actual is None,label+'/NULL')
        elif isinstance(expected,(str,bool)):
            self.check(type(actual) is type(expected) and actual==expected,label+'/exact')
        elif isinstance(expected,(int,float)):
            self.check(isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual),label+'/finite')
            if isinstance(expected,int):self.check(actual==expected,label+'/integer')
            else:
                error=abs(actual-expected);self.maximum_error=max(self.maximum_error,error)
                self.check(error<=5e-13+5e-13*abs(expected),label+'/scalar')
        elif isinstance(expected,list):
            self.check(isinstance(actual,list) and len(actual)==len(expected),label+'/length')
            for i,(a,b) in enumerate(zip(expected,actual)):self.equal(a,b,label+f'/{i}')
        elif isinstance(expected,dict):
            self.check(isinstance(actual,dict) and set(actual)==set(expected),label+'/keys')
            for name,value in expected.items():self.equal(value,actual[name],label+'/'+name)
        else:raise TypeError(type(expected))


def scalar_annual(weights,returns,inflation,*,floor,cost=.0025):
    w=[float(x) for x in weights];r=[float(x) for x in returns];pi=float(inflation)
    if len(w)!=3 or len(r)!=3 or any(not math.isfinite(x) or x<0 for x in w) or abs(math.fsum(w)-1)>TOL:
        raise ValueError('Three nonnegative simplex weights required')
    if any(not math.isfinite(x) or x<=-1 for x in [*r,pi]) or not 0<=cost<1 or not 0<=floor<=1 or w[2]<floor-TOL:
        raise ValueError('Positive joint gross and original normative floor required')
    dollars=[w[i]*(1+r[i]) for i in range(3)];total=math.fsum(dollars)
    bill=dollars[2]/total
    drag=cost/(1+cost)*(1-bill)
    return dict(target_bill_share=w[2],end_bill_share=bill,end_below_target_floor=bill<floor-TOL,
        hypothetical_liquidation_drag=drag,hypothetical_available_bill_ratio=1-drag,bill_real_return=(1+r[2])/(1+pi)-1)


def scalar_path(annual):
    if not annual:raise ValueError('Assigned original horizon must have endpoints')
    n=len(annual)
    return dict(target_bill_share=annual[0]['target_bill_share'],mean_end_bill_share=math.fsum(r['end_bill_share'] for r in annual)/n,
        minimum_end_bill_share=min(r['end_bill_share'] for r in annual),fraction_end_below_target_floor=sum(r['end_below_target_floor'] for r in annual)/n,
        max_hypothetical_liquidation_drag=max(r['hypothetical_liquidation_drag'] for r in annual),mean_bill_real_loss_fraction=sum(r['bill_real_return']<0 for r in annual)/n)


def scalar_summary(profile,rows):
    name,h,liquidity,floor=profile;funded=[r for r in rows if r['assigned']];n=len(funded)
    result=dict(profile=name,horizon=h,liquidity=liquidity,target_floor=floor,original_n=len(rows),financial_n=n,review_n=len(rows)-n,
        **dict.fromkeys(SUMMARY_FIELDS))
    if n:result.update(minimum_target_bill_share=min(r['target_bill_share'] for r in funded),maximum_target_bill_share=max(r['target_bill_share'] for r in funded),
        mean_end_bill_share=math.fsum(r['mean_end_bill_share'] for r in funded)/n,minimum_end_bill_share=min(r['minimum_end_bill_share'] for r in funded),
        mean_fraction_end_below_target_floor=math.fsum(r['fraction_end_below_target_floor'] for r in funded)/n,
        paths_with_any_end_below_target_floor=sum(r['fraction_end_below_target_floor']>0 for r in funded),
        maximum_hypothetical_liquidation_drag=max(r['max_hypothetical_liquidation_drag'] for r in funded),
        mean_bill_real_loss_fraction=math.fsum(r['mean_bill_real_loss_fraction'] for r in funded)/n)
    return result


def csv_rows(path,columns=None):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='') as stream:
        reader=csv.DictReader(stream)
        if columns is not None and reader.fieldnames!=list(columns):raise AssertionError('Declared CSV header: '+str(path))
        return list(reader)


def decoded(value,expected):
    if expected is None:
        if value!='':raise AssertionError('Review scalar must be empty NULL')
        return None
    if isinstance(expected,bool):
        if value not in ('True','False'):raise AssertionError('Genuine CSV bool required')
        return value=='True'
    if isinstance(expected,int):
        numeric=float(value)
        if not math.isfinite(numeric) or numeric!=int(numeric):raise AssertionError('Integer metadata required')
        return int(numeric)
    if isinstance(expected,float):return float(value)
    return value


def compare_rows(expected,actual,keys,columns,c,label):
    c.check(len(actual)==len(expected),label+'/complete row count')
    lookup={tuple(row[key] for key in keys):row for row in expected};c.check(len(lookup)==len(expected),label+'/expected unique keys')
    seen=set()
    for row in actual:
        key=tuple(decoded(row[name],0) if name in ('scenario_id','period') else row[name] for name in keys)
        c.check(key in lookup and key not in seen,label+'/unique original row identity');seen.add(key)
        target=lookup[key];c.check(set(row)==set(columns) and set(target)==set(columns),label+'/complete fields')
        for name,value in target.items():c.equal(value,decoded(row[name],value),label+'/'+str(key)+'/'+name)
    c.check(seen==set(lookup),label+'/all row keys retained')


def source_review(root):
    text=(Path(root)/'scripts/diagnose_liquidity_capacity.py').read_text();tree=ast.parse(text)
    loaders=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_paths']
    if len(loaders)!=1 or ast.unparse(loaders[0].args[1])!='range(1, 121)' or {k.arg:ast.literal_eval(k.value) for k in loaders[0].keywords}.get('allow_final') is not False:
        raise AssertionError('Only unchanged development loader permitted')
    if 'execute_target' in text or 'simulate_' in text:raise AssertionError('Liquidity diagnostic must not run or charge a policy')
    return dict(status='passed',scope='Source identity/no path-charge or strategy call; independent endpoint ratios, not a new investment simulation')


def authorize(root=ROOT):
    root=Path(root);p,f,m,a=readj(root/PROTOCOL),readj(root/PRODUCER_FREEZE),readj(root/OUT/'manifest.json'),readj(root/FREEZE)
    if (not a.get('execution_ready') or a.get('plan_sha256')!=sha(root/PLAN) or not f.get('execution_ready')
        or f.get('protocol_sha256')!=sha(root/PROTOCOL) or p.get('development_ids')!=list(range(1,121)) or p.get('allow_final') is not False):
        raise RuntimeError('Separately frozen development-only producer and independent auditor required')
    required={'scripts/audit_liquidity_capacity.py','tests/test_audit_liquidity_capacity.py',str(PLAN),'src/safeinvest/data.py'}
    if not required<=set(a.get('source_sha256',{})):raise RuntimeError('Bind full audit source/test/plan/loader')
    verify(root,a['source_sha256']);verify(root,f['source_sha256']);verify(root,f['input_sha256'])
    closure={**f['source_sha256'],**f['input_sha256']}
    if (m.get('status')!='complete' or m.get('protocol_sha256')!=sha(root/PROTOCOL) or m.get('source_freeze_sha256')!=sha(root/PRODUCER_FREEZE)
        or m.get('sources_and_inputs')!=closure or m.get('final_scenarios_loaded') is not False or m.get('path_rows')!=600
        or any(m.get(k)!=0 for k in ('new_policy_runs','new_fits'))
        or any(m.get(k) is not False for k in ('withdrawals_assumed','hypothetical_liquidation_charged_to_strategy'))
        or set(m.get('outputs_sha256',{}))!={'path_capacity.csv.gz','annual_capacity.csv.gz','summary.json'}):
        raise RuntimeError('Complete unchanged mathematical producer with all three output hashes required')
    verify(root/OUT,m['outputs_sha256'])
    parent=readj(root/p['parent_Q1_manifest']);audit=readj(root/p['parent_Q1_independent_audit']);ledger=Path(p['q1_oof_paths'])
    if (parent.get('status')!='complete' or audit.get('status')!='passed' or parent.get('final_scenarios_loaded') is not False
        or audit.get('producer_manifest_sha256')!=sha(root/p['parent_Q1_manifest'])
        or audit.get('producer_outputs_at_start',{}).get(ledger.name)!=sha(root/ledger)
        or parent.get('outputs_sha256',{}).get(ledger.name)!=sha(root/ledger)):
        raise RuntimeError('Passed unchanged parent Q1 own-horizon audit and ledger required')
    names=set(closure)|set(a['source_sha256'])|{str(PROTOCOL),str(PRODUCER_FREEZE),str(PLAN),str(FREEZE),str(OUT/'manifest.json')}
    names|={str(OUT/name) for name in m['outputs_sha256']}
    return p,m,{name:sha(root/name) for name in sorted(names)}


def audit_values(root,p,m,c):
    sys.path.insert(0,str(Path(root)/'src'))
    from safeinvest.data import load_paths
    market,inflation,ids=load_paths(Path(root)/'data/SafeInvest_Synthetic_Data.csv',range(1,121),allow_final=False)
    c.equal(list(range(1,121)),ids.tolist(),'exact development identities')
    jobs=[j for j in readj(Path(root)/p['jobs'])['jobs'] if j['stage']=='refit' and j['parent_context_id']!=0]
    c.check(len(jobs)==5 and len({j['job_id'] for j in jobs})==5,'five distinct outer refits')
    c.equal(list(range(1,121)),sorted(sid for j in jobs for sid in j['evaluation_ids']),'all outer original IDs once')
    fits=readj(Path(root)/p['static_fits']);prior=csv_rows(Path(root)/p['q1_oof_paths'])
    prior_map={(row['job_id'],int(row['scenario_id']),row['profile']):row for row in prior}
    c.check(len(prior)==600 and len(prior_map)==600,'complete unique original Q1 ledger')
    paths=[];annual=[]
    for job in jobs:
        tr,ev=job['training_ids'],job['evaluation_ids'];fit=fits[job['job_id']]['refit']
        c.check(len(tr)==96 and len(ev)==24 and not set(tr)&set(ev) and set(tr)|set(ev)==set(range(1,121)),'own96/24 partition')
        c.equal(tr,fit['training_ids'],'own training fit identity')
        c.equal(ev,fit['outer_test_ids'],'own fit evaluation identity')
        for profile in PROFILES:
            name,h,liquidity,floor=profile;assignment=fit['effective_assignments'][name]
            menu=fit['raw_fit']['menu'] if fit['raw_fit'] else [];products=[row for row in menu if row['name']==assignment]
            c.check(len(products)<=1,'unique saved product name');product=products[0] if products else None
            c.check((product is None)==(assignment=='Review'),'only declared Review lacks a financial product')
            for sid in ev:
                old=prior_map[(job['job_id'],sid,name)]
                c.equal(assignment,old['effective_assignment'],'unchanged Q1 assignment');c.equal(product is not None,decoded(old['assigned'],True),'unchanged Q1 funding')
                c.equal(h,decoded(old['horizon'],0),'own original profile horizon')
                common=dict(job_id=job['job_id'],scenario_id=sid,profile=name,horizon=h,liquidity=liquidity,target_floor=floor,effective_assignment=assignment,assigned=product is not None)
                if product is None:paths.append(dict(common,**dict.fromkeys(FIELDS)));continue
                values=[]
                for t in range(h):
                    value=scalar_annual(product['weights'],market[sid-1,t],inflation[sid-1,t],floor=floor)
                    c.equal(1.,value['hypothetical_available_bill_ratio']+value['hypothetical_liquidation_drag'],'hypothetical endpoint funding only')
                    values.append(value);annual.append(dict(common,period=t+1,**value))
                paths.append(dict(common,**scalar_path(values)))
    c.check(len(paths)==600,'all600 metadata/path records')
    expected_annual=sum(row['horizon'] for row in paths if row['assigned']);c.equal(expected_annual,len(annual),'assigned own-H annual denominator')
    c.equal(expected_annual,m['annual_rows'],'producer annual denominator')
    for profile in PROFILES:c.equal(list(range(1,121)),sorted(row['scenario_id'] for row in paths if row['profile']==profile[0]),profile[0]+'/original120')
    compare_rows(paths,csv_rows(Path(root)/OUT/'path_capacity.csv.gz',PATH_COLUMNS),('job_id','scenario_id','profile'),PATH_COLUMNS,c,'all paths')
    compare_rows(annual,csv_rows(Path(root)/OUT/'annual_capacity.csv.gz',ANNUAL_COLUMNS),('job_id','scenario_id','profile','period'),ANNUAL_COLUMNS,c,'all annual endpoints')
    summaries=[scalar_summary(profile,[row for row in paths if row['profile']==profile[0]]) for profile in PROFILES]
    c.equal(summaries,readj(Path(root)/OUT/'summary.json'),'all five profile summaries')
    return dict(path_rows=len(paths),assigned_paths=sum(row['assigned'] for row in paths),review_paths=sum(not row['assigned'] for row in paths),annual_rows=len(annual),
        profiles=[dict(profile=profile[0],original_n=120,assigned_n=sum(row['assigned'] for row in paths if row['profile']==profile[0]),
                       annual_n=sum(row['horizon'] for row in paths if row['profile']==profile[0] and row['assigned'])) for profile in PROFILES])


def run(root=ROOT):
    root=Path(root);attempt=root/ATTEMPT;receipt=root/OUT/'independent_audit.json'
    if attempt.exists() or receipt.exists():raise RuntimeError('Never overwrite or silently retry an audit attempt')
    attempt.mkdir(parents=True,exist_ok=False)
    state=dict(status='incomplete',phase='before_hash_gates',started_utc=datetime.now(timezone.utc).isoformat(),market_value_access_started=False,automatic_retry=False)
    writej(attempt/'attempt.json',state)
    try:
        p,m,before=authorize(root);c=Checks();source=source_review(root)
        state.update(phase='guarded_development_values',market_value_access_started=True);writej(attempt/'attempt.json',state)
        counts=audit_values(root,p,m,c);after={name:sha(root/name) for name in before};c.equal(before,after,'all frozen source/input/output hashes unchanged')
        result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks=c.n,maximum_absolute_numeric_error=c.maximum_error,counts=counts,
            plan_sha256=sha(root/PLAN),audit_source_freeze_sha256=sha(root/FREEZE),producer_manifest_sha256=sha(root/OUT/'manifest.json'),
            sources_and_artifacts_at_start=before,sources_and_artifacts_at_end=after,source_review=source,
            development_ids_loaded=list(range(1,121)),final_scenarios_loaded=False,new_fits=0,new_policy_runs=0,hypothetical_liquidation_charged_to_strategy=False,
            scope='All-row scalar endpoint capacity postprocessing and metadata audit. No strategy simulation, liability schedule, Q2 admission or future guarantee.')
        writej(receipt,result);state.update(status='passed',phase='complete',receipt_sha256=sha(receipt));writej(attempt/'attempt.json',state)
        print(json.dumps(dict(status='passed',checks=c.n,counts=counts,receipt_sha256=sha(receipt))),flush=True)
    except BaseException as error:
        (attempt/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',phase='failure_retained',error_type=type(error).__name__,error=str(error),failure_sha256=sha(attempt/'failure.txt'))
        writej(attempt/'attempt.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,independent_scalar_audit=True,new_policy_runs=0,market_values_parsed=False)))
