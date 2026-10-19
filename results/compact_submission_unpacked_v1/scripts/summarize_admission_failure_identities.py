"""Post-hoc identity ledger from existing audited tables, never new modeling.

The event differences were inspected before this script was written. This
is subsequent evidence organization, not a blind/prospective test. It loads
no raw market returns, fits nothing, changes no policy/admission, excludes
no influential path and refuses to overwrite its own output archive.
"""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/model_review/admission_failure_identities_v1'
POLICY='joint_tie_set_refit_learner'
GUARDED='global_policy_with_same_U_guard'
REFERENCE='annual_then_pending'
PRICES=(0.,.001,.003,.01,.03,.1,.3,1.,None)
INPUTS=(
 'docs/main_model_registry.json','docs/final_model_convergence_judgment.json',
 'docs/joint_liquidity_refit_protocol.json','docs/refit_stability_protocol.json',
 'results/model_review/joint_liquidity_refit_v1/manifest.json',
 'results/model_review/joint_liquidity_refit_v1/independent_audit.json',
 'results/model_review/joint_liquidity_refit_v1/context_selections.json',
 'results/model_review/joint_liquidity_refit_v1/static_refits.json',
 'results/model_review/joint_liquidity_refit_v1/training_candidate_paths.csv',
 'results/model_review/joint_liquidity_refit_v1/inner_validation_paths.csv',
 'results/model_review/joint_liquidity_refit_v1/client_paths.csv',
 'results/model_review/refit_stability_v1/manifest.json',
 'results/model_review/refit_stability_v1/independent_audit.json',
 'results/model_review/refit_stability_v1/context_selections.json',
 'results/model_review/refit_stability_v1/static_refits.json.gz',
 'results/model_review/benefit_cost_execution_v1/manifest.json',
 'results/model_review/benefit_cost_execution_v1/independent_audit.json',
 'results/model_review/benefit_cost_execution_v1/inner_reference_paths.csv.gz',
 'results/model_review/global_policy_admission_v1/manifest.json',
 'results/model_review/global_policy_admission_v1/independent_audit.json',
 'results/model_review/global_policy_admission_v1/client_paths.csv.gz')
METRICS=('breach','mdd','first_crossing','log_growth','nominal_fees','trade_fraction')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):
 p=Path(path)
 if p.suffix=='.gz':
  with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
 return json.loads(p.read_text())
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x) for x in v]
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.bool_,)):return bool(v)
 if isinstance(v,(np.floating,float)):return None if not np.isfinite(v) else float(v)
 return v
def writej(path,value):Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsons(value):return json.dumps(clean(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)
def flag(v):
 if isinstance(v,(bool,np.bool_)):return bool(v)
 if isinstance(v,str) and v in ('True','False'):return v=='True'
 raise ValueError('Observed event/admission requires a boolean, not a coerced NULL/string')
def snapshots():return {p:sha(ROOT/p) for p in INPUTS}
def source_snapshot():return {str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))}


def validate_parent(folder):
 manifest=readj(ROOT/folder/'manifest.json');audit=readj(ROOT/folder/'independent_audit.json')
 if manifest.get('status','complete')!='complete' or audit['status']!='passed':raise ValueError('Complete parent and passed audit required')
 if manifest['sources']!=manifest['sources_at_end'] or manifest['input_sha256']!=manifest['input_sha256_at_end']:raise ValueError('Parent start/end snapshot differs')
 if manifest.get('final_scenarios_loaded') is not False:raise ValueError('No final scenarios in source parent')
 msha=sha(ROOT/folder/'manifest.json')
 if 'parent_manifest_sha256' in audit:
  if audit['parent_manifest_sha256']!=msha:raise ValueError('Audit is not bound to current parent')
 else:
  key=folder+'/manifest.json'
  if audit['sources_and_artifacts_at_start'].get(key)!=msha:raise ValueError('Legacy audit manifest snapshot differs')
 # Validate every used source artifact in this parent against its manifest.
 for rel in INPUTS:
  path=Path(rel)
  if path.parent!=Path(folder) or path.name in ('manifest.json','independent_audit.json'):continue
  candidates=[manifest['outputs_sha256'].get(path.name),manifest['outputs_sha256'].get(rel)]
  if sha(ROOT/path) not in candidates:raise ValueError('Used artifact differs from frozen parent output: '+rel)
 return manifest,audit


def table(rel):
 f=pd.read_csv(ROOT/rel)
 if not f.scenario_id.between(1,120).all():raise ValueError('Only original development IDs permitted')
 return f

def indexed(frame,expected):
 if frame.scenario_id.duplicated().any() or sorted(map(int,frame.scenario_id))!=sorted(expected):raise ValueError('Missing/duplicate/foreign identity')
 return frame.set_index('scenario_id',drop=False)

def meta(job,bundle):
 if bundle['refit']['training_ids']!=job['training_ids']:raise ValueError('Own cached training identity changed')
 expected=[] if job['job_id']=='c0_refit' else job['evaluation_ids']
 if bundle['refit']['outer_test_ids']!=expected:raise ValueError('Static holdout vs dynamic in-sample identity changed')
 product=bundle['q2_case']['product']
 return dict(job_id=job['job_id'],training_ids=job['training_ids'],evaluation_ids=job['evaluation_ids'],
  training_fan_n=len(job['training_ids']),target=product['weights'] if product else None,
  target_local_id=product['candidate_index'] if product else None,
  target_original232_id=product['candidate_original232_index'] if product else None,
  static_gamma=bundle['refit']['selection']['selected_gamma'])

def verify_row(row,metadata):
 if int(row.selection_training_n)!=metadata['training_fan_n']:raise ValueError('Row fan/train size differs from own job')
 if flag(row.assigned):
  if int(row.candidate_local_index)!=metadata['target_local_id'] or int(row.candidate_original232_index)!=metadata['target_original232_id']:raise ValueError('Recorded target identity mismatch')

def arm_fields(row,prefix):
 assigned=flag(row.assigned);out={prefix+'_available':assigned}
 for k in METRICS:
  value=row[k]
  if not assigned:
   if not pd.isna(value):raise ValueError('Unserved model outcome must remain NULL')
   out[prefix+'_'+k]=None
  elif k=='breach':out[prefix+'_'+k]=flag(value)
  else:out[prefix+'_'+k]=None if pd.isna(value) else float(value)
 return out

def event_class(left,right):
 if left is None or right is None:return None
 return 'both_breach' if left and right else 'model_only' if left else 'B_only' if right else 'neither_breach'

def transition(full,inner):
 if full is None or inner is None:return None
 return 'both_breach' if full and inner else 'full_only' if full else 'inner_only' if inner else 'neither_breach'

def matrix(rows,left,right,mode='within'):
 observed=[x for x in rows if x[left] is not None and x[right] is not None]
 classify=event_class if mode=='within' else transition
 names=('both_breach','model_only','B_only','neither_breach') if mode=='within' else ('both_breach','full_only','inner_only','neither_breach')
 return dict(n=len(observed),left_events=sum(x[left] for x in observed),right_events=sum(x[right] for x in observed),
  cells={name:dict(n=sum(classify(x[left],x[right])==name for x in observed),scenario_ids=[x['scenario_id'] for x in observed if classify(x[left],x[right])==name]) for name in names})


def build():
 jp=readj(ROOT/'docs/joint_liquidity_refit_protocol.json');rp=readj(ROOT/'docs/refit_stability_protocol.json')
 jb=readj(ROOT/'results/model_review/joint_liquidity_refit_v1/static_refits.json');rb=readj(ROOT/'results/model_review/refit_stability_v1/static_refits.json.gz')
 selection=readj(ROOT/'results/model_review/joint_liquidity_refit_v1/context_selections.json')['full_development']['methods'][POLICY]
 if selection['execution_status']!='automated':raise ValueError('This ledger expects the originally selected full-development controller')
 beta=float(selection['selected_beta']);order=int(selection['selected_candidate_order'])
 jobs={j['job_id']:j for j in jp['jobs'][:5]};fm=meta(jobs['c0_refit'],jb['c0_refit']);nids=list(range(1,121))
 train=table('results/model_review/joint_liquidity_refit_v1/training_candidate_paths.csv')
 full=indexed(train.loc[train.job_id.eq('c0_refit')&train.execution_candidate_order.eq(order)],nids)
 old=table('results/model_review/joint_liquidity_refit_v1/client_paths.csv');fc=old.loc[old.context.eq('full_development')]
 fullB=indexed(fc.loc[fc.policy.eq(REFERENCE)],nids)
 # Source the nine training events from training_candidate_paths, not from
 # the guarded NULL service rows. Check archived old full120 replay agrees.
 client=indexed(fc.loc[fc.policy.eq(POLICY)],nids)
 for metric in ('mdd','log_growth','nominal_fees'):
  if not np.allclose(full[metric],client[metric],atol=1e-12,rtol=0):raise ValueError('Full training vs old full in-sample replay differ')
 if [flag(x) for x in full.breach]!=[flag(x) for x in client.breach]:raise ValueError('Full training event identities differ from archived replay')
 inner=table('results/model_review/joint_liquidity_refit_v1/inner_validation_paths.csv')
 inner=indexed(inner.loc[inner.context.eq('full_development')&inner.training_beta.eq(beta)],nids)
 B=table('results/model_review/benefit_cost_execution_v1/inner_reference_paths.csv.gz');innerB=indexed(B.loc[B.context.eq('full_development')],nids)
 fullrows=[]
 for sid in nids:
  fr=full.loc[sid];br=fullB.loc[sid];ir=inner.loc[sid];ib=innerB.loc[sid]
  if ir.job_id!=ib.job_id:raise ValueError('Inner B must be same-own-job and original ID')
  im=meta(jobs[ir.job_id],jb[ir.job_id]);inside=int(ir.selected_candidate_order) if flag(ir.assigned) else None
  if flag(ir.assigned) and jb[ir.job_id]['training_price_selections'][str(beta)]['selected_candidate_order']!=inside:raise ValueError('Original selected beta price was changed')
  for row,metadata in [(fr,fm),(br,fm),(ir,im),(ib,im)]:verify_row(row,metadata)
  if sid in im['training_ids'] or sid not in im['evaluation_ids']:raise ValueError('Inner path not held out of its own training')
  row=dict(ledger_kind='full_training_vs_selected_beta_inner',scenario_id=sid,repeat_id=0,context='full_development',inner_job_id=ir.job_id,
   current_full_C_service='execution_review',full_values_scope='old_selected_training_support_not_current_service_returns',
   selected_beta=beta,full_eta_kind='risk_first' if order==8 else 'finite',full_eta=PRICES[order],full_order=order,
   inner_eta_kind=None if inside is None else 'risk_first' if inside==8 else 'finite',inner_eta=None if inside is None else PRICES[inside],inner_order=inside,
   full_target=jsons(fm['target']),inner_target=jsons(im['target']),full_static_gamma=fm['static_gamma'],inner_static_gamma=im['static_gamma'],
   full_training_fan_n=fm['training_fan_n'],inner_training_fan_n=im['training_fan_n'],
   full_training_ids=jsons(fm['training_ids']),inner_training_ids=jsons(im['training_ids']),
   target_changed=fm['target']!=im['target'],eta_changed=inside!=order,training_fan_membership_changed=fm['training_ids']!=im['training_ids'],
   same_original_market_ID=True,full_fit_includes_evaluation_ID=True,inner_fit_includes_evaluation_ID=False,
   **arm_fields(fr,'full_model'),**arm_fields(br,'full_B'),**arm_fields(ir,'inner_model'),**arm_fields(ib,'inner_B'))
  row.update(model_full_to_inner_event=transition(row['full_model_breach'],row['inner_model_breach']),
   B_full_to_inner_event=transition(row['full_B_breach'],row['inner_B_breach']),
   full_model_vs_same_target_B=event_class(row['full_model_breach'],row['full_B_breach']),
   inner_model_vs_same_target_B=event_class(row['inner_model_breach'],row['inner_B_breach']))
  fullrows.append(row)
 outer=table('results/model_review/global_policy_admission_v1/client_paths.csv.gz');outer=outer.loc[outer.scope.eq('outer_oof')]
 left=outer.loc[outer.policy.eq(GUARDED)];right=outer.loc[outer.policy.eq(REFERENCE)]
 if len(left)!=600 or len(right)!=600:raise ValueError('All five original120 denominators required')
 keys=['repeat_id','context','scenario_id']
 if left.duplicated(keys).any() or right.duplicated(keys).any():raise ValueError('Outer identity duplicate')
 rr=right.set_index(keys,drop=False);rjobs={j['job_id']:j for j in rp['jobs']};selections=readj(ROOT/'results/model_review/refit_stability_v1/context_selections.json')
 outerrows=[]
 for _,mr in left.sort_values(keys).iterrows():
  br=rr.loc[(mr.repeat_id,mr.context,mr.scenario_id)]
  if mr.job_id!=br.job_id:raise ValueError('Outer reference job differs')
  om=meta(rjobs[mr.job_id],rb[mr.job_id]);verify_row(mr,om);verify_row(br,om)
  choice=selections[mr.context]['methods'][POLICY];o=choice['selected_candidate_order'];oldauto=choice['execution_status']=='automated'
  if flag(mr.assigned) and not oldauto:raise ValueError('Previously refused policy was rescued')
  row=dict(ledger_kind='outer_model_vs_same_target_B',scenario_id=int(mr.scenario_id),repeat_id=int(mr.repeat_id),context=mr.context,outer_job_id=mr.job_id,
   outer_final_status=mr.execution_status,outer_original_status=choice['execution_status'],outer_scope='outer_oof',
   outer_beta=choice['selected_beta'],outer_eta_kind=None if o is None else 'risk_first' if o==8 else 'finite',outer_eta=None if o is None else PRICES[o],outer_order=o,
   outer_target=jsons(om['target']),outer_static_gamma=om['static_gamma'],outer_training_fan_n=om['training_fan_n'],outer_training_ids=jsons(om['training_ids']),
   outer_B_reference_only=True,
   **arm_fields(mr,'outer_model'),**arm_fields(br,'outer_B'))
  row['common_auto']=row['outer_model_available'] and row['outer_B_available']
  row['outer_event_class']=event_class(row['outer_model_breach'],row['outer_B_breach']) if row['common_auto'] else None
  for metric in ('mdd','log_growth','nominal_fees'):
   row['outer_delta_'+metric]=row['outer_model_'+metric]-row['outer_B_'+metric] if row['common_auto'] else None
  outerrows.append(row)
 summary=dict(post_hoc=True,purpose='Existing-result identity diagnosis, not new validation or model selection',
  selected_beta=beta,unchanged_full_eta=PRICES[order],full_model_source='training_candidate_paths.csv:c0_refit/original_selected_candidate_order',
  full_model_training_breach_IDs=[x['scenario_id'] for x in fullrows if x['full_model_breach']],
  full_model_to_inner=matrix(fullrows,'full_model_breach','inner_model_breach','transition'),
  full_B_to_inner=matrix(fullrows,'full_B_breach','inner_B_breach','transition'),
  full_model_vs_B=matrix(fullrows,'full_model_breach','full_B_breach'),inner_model_vs_B=matrix(fullrows,'inner_model_breach','inner_B_breach'),
  inner_component_groups=[dict(job_id=jid,n=sum(x['inner_job_id']==jid for x in fullrows),target=jsons(jb[jid]['q2_case']['product']['weights']),
                              eta_orders=sorted({x['inner_order'] for x in fullrows if x['inner_job_id']==jid}),training_fan_n=len(jobs[jid]['training_ids'])) for jid in jobs if jid!='c0_refit'],
  outer_by_repeat=[],outer_by_fit=[],outer_model_only_occurrences=[],outer_B_only_occurrences=[],
  limits=['Same original120IDs, but full-training fixed policy and4inner learned policies differ in targets, fans and sometimeseta;9vs10is not a validation error of one fixed model.',
          'Inner10vsB8is an actual paired two-event difference on these stored paths, not just a changed denominator; it is not a statistical population-risk proof.',
          'Outer groups condition on training-fit admission; reference outcomes outside common-auto are hypothetical, model financials remainNULL.',
          'Repeated original IDs across fits are dependent. Influential IDs are descriptive evidence, never grounds to delete outliers or change gates.',
          'No raw returns, fits, new policies, predictive calibration test or causal attribution to target versus fan versus action rule is performed.'])
 for groupkey,label in [('repeat_id','outer_by_repeat'),('context','outer_by_fit')]:
  for value in sorted({x[groupkey] for x in outerrows}):
   group=[x for x in outerrows if x[groupkey]==value];common=[x for x in group if x['common_auto']]
   mat=matrix(common,'outer_model_breach','outer_B_breach')
   if not common:mat.update(left_events=None,right_events=None)
   summary[label].append(dict(group=value,original_n=len(group),common_auto_n=len(common),paired_events=mat,
    conditional_model_risk=mat['left_events']/len(common) if common else None,
    conditional_B_risk=mat['right_events']/len(common) if common else None))
 for row in outerrows:
  if row['outer_event_class'] in ('model_only','B_only'):
   key='outer_model_only_occurrences' if row['outer_event_class']=='model_only' else 'outer_B_only_occurrences'
   summary[key].append({k:row[k] for k in ('repeat_id','context','scenario_id')})
 summary['outer_model_only_distinct_original_IDs']=sorted({x['scenario_id'] for x in summary['outer_model_only_occurrences']})
 return fullrows+outerrows,summary


def main():
 if OUT.exists():raise RuntimeError('Do not overwrite an existing diagnosis archive')
 before=snapshots();sources=source_snapshot()
 for folder in ('results/model_review/joint_liquidity_refit_v1','results/model_review/refit_stability_v1','results/model_review/benefit_cost_execution_v1','results/model_review/global_policy_admission_v1'):validate_parent(folder)
 registry=readj(ROOT/'docs/main_model_registry.json')
 for p,h in registry['evidence_sha256'].items():
  if p in before and before[p]!=h:raise ValueError('Registry evidence changed: '+p)
 rows,summary=build()
 if len(rows)!=720:raise ValueError('Require120full-inner and600outer identity rows')
 if snapshots()!=before or source_snapshot()!=sources:raise RuntimeError('Evidence/source changed during diagnosis')
 OUT.mkdir(parents=True)
 (OUT/'ledger.csv.gz').write_bytes(gzip.compress(pd.DataFrame(rows).to_csv(index=False,lineterminator='\n').encode(),mtime=0))
 writej(OUT/'summary.json',summary)
 if snapshots()!=before or source_snapshot()!=sources:raise RuntimeError('Evidence changed before receipt')
 writej(OUT/'manifest.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
  post_hoc_existing_evidence_only=True,source=sources,sources_at_end=source_snapshot(),input_sha256=before,input_sha256_at_end=snapshots(),
  outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},row_counts=dict(ledger=len(rows),full_inner120=120,outer_original600=600),
  new_fits=0,policy_changes=0,raw_market_loaded=False,final_scenarios_loaded=False,
  runtime=dict(pandas=pd.__version__,numpy=np.__version__),limitations=summary['limits']))
 print(json.dumps({'status':'complete','rows':len(rows),'out':str(OUT.relative_to(ROOT)),'manifest_sha256':sha(OUT/'manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
