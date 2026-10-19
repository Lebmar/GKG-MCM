"""Read-only current-contract data reconciliation plus recorded semantic review.

No rawmarket load, forecast, fit, execution or report creation.  The reviewed
contract is bound by its exact SHA; a new contract needs a new review.  This
receipt is an automated/AI computational and semantic review, not external
human peer review or a financial safety certificate.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'docs/current_model_contract_2026_10_19.json'
EXPECTED='5f47d9a7b3720e2642bd7767cd3fb126e7df9ee9b7d759a7ecd24ba6e5d124af'
RECEIPT=ROOT/'docs/current_model_contract_2026_10_19_independent_review.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(rel):return json.loads((ROOT/rel).read_text())
def run():
    if RECEIPT.exists():raise RuntimeError('Never overwrite a reviewed-contract receipt')
    if sha(CONTRACT)!=EXPECTED:raise RuntimeError('Wait for the exact revised draft to be reviewed')
    p=json.loads(CONTRACT.read_text());checks=0;maxerr=0.
    def check(c,label):
        nonlocal checks
        checks+=1
        if not c:raise AssertionError(label)
    def same(a,b,label):
        nonlocal maxerr
        if b is None:check(pd.isna(a),label+'/NULL')
        elif isinstance(b,(str,bool)):check(a==b,label+'/exact')
        else:
            err=abs(float(a)-float(b));maxerr=max(maxerr,err);check(err<=2e-12,label+'/numeric')
    inputs={str(CONTRACT.relative_to(ROOT)):sha(CONTRACT),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for rel,digest in p['evidence_sha256'].items():check(sha(ROOT/rel)==digest,'contract evidence SHA/'+rel);inputs[rel]=digest
    extra=['results/model_review/global_policy_admission_v1/client_paths.csv.gz','results/model_review/global_policy_admission_v1/trade_estimates.csv.gz',
        'results/model_review/benefit_cost_execution_v1/training_summary.csv.gz','results/model_review/constrained_fan_execution_v1/training_summary.csv.gz',
        'results/model_review/joint_liquidity_refit_v1/q1_context_assignments.csv']
    for rel in extra:inputs[rel]=sha(ROOT/rel)
    registry=read('docs/main_model_registry.json')
    check(p['Q1_product_and_liquidity']['current_full120']==registry['current_static_full120'],'Q1 unchanged authoritative registry')
    check(p['current_C_service']==registry['current_C_execution'],'C unchanged authoritative registry including NULLs')
    s=pd.read_csv(ROOT/p['Q2_evaluation_contract']['existing_table'],float_precision='round_trip')
    check(len(s)==p['Q2_evaluation_contract']['rows']==18,'18current summary rows')
    check(not s.duplicated(['repeat_id','policy']).any() and set(s.repeat_id)==set(range(6)) and len(set(s.policy))==3,'6contexts times3distinct policies')
    check(set(p['Q2_evaluation_contract']['required_columns'])<=set(s.columns),'all required Q2columns present')
    policy='global_policy_with_same_U_guard';ss=s.loc[s.policy.eq(policy)].sort_values('repeat_id')
    observed=p['Q2_evaluation_contract']['observed_limits']
    check(ss.loc[ss.repeat_id>0,'assigned_n'].tolist()==observed['guarded_outer_auto_per_repeat'],'OOF admitted denominators')
    actual=[None if pd.isna(x) else int(x) for x in ss.loc[ss.repeat_id>0,'breach_count']]
    check(actual==observed['guarded_outer_breaches_per_repeat'],'OOF event counts with zero-coverage NULLs')
    zero=ss.loc[ss.assigned_n.eq(0)]
    for field in ['breach_count','conditional_breach_probability','mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','mean_fees','mean_trade_fraction']:
        check(zero[field].isna().all(),'zero service financial NULL/'+field)
    cp=pd.read_csv(ROOT/extra[0],float_precision='round_trip');model=cp.loc[cp.policy.eq(policy)&cp.repeat_id.gt(0)&cp.assigned.eq(True)]
    check(len(model)==observed['automatic_scenario_repeat_records']==96,'96automatic scenario-repeat records')
    check(model.scenario_id.nunique()==observed['distinct_original_ids_in_these_records']==74,'74distinct originalIDs')
    B=cp.loc[cp.policy.eq('annual_then_pending')]
    Bcounts=[]
    for repeat in range(1,6):
        m=model.loc[model.repeat_id.eq(repeat)]
        if not len(m):Bcounts.append(None);continue
        pair=m.merge(B.loc[B.repeat_id.eq(repeat)],on=['context','scenario_id'],suffixes=('_M','_B'),validate='one_to_one')
        check(len(pair)==len(m) and pair.assigned_B.eq(True).all(),'sameautomatic B cohort/'+str(repeat))
        Bcounts.append(int(pair.breach_B.eq(True).sum()))
    check(Bcounts==observed['same_outer_A_B_events'],'sameouterautomatic B event counts')
    t=pd.read_csv(ROOT/extra[1],float_precision='round_trip');tr=t.loc[t.final_guard_admitted.eq(True)&t.active_optional.eq(True)&t.restore.eq(True)]
    check(len(tr)==observed['qualified_record_optional_restores']==678,'678qualified optional restores')
    check(int(tr.justification.eq('net_growth_improvement').sum())==observed['predicted_net_growth_restores']==376,'376predicted net growth restores')
    check(int(tr.justification.eq('risk_reduction_with_growth_cost').sum())==observed['predicted_risk_reduction_with_growth_cost_restores']==302,'302risk reduction with growth cost restores')
    positive=pd.read_csv(ROOT/extra[2]);positive=positive.loc[positive.job_id.eq('c0_refit')]
    check(len(positive)==2 and positive.breach_count.eq(31).all() and positive.n_training.eq(120).all(),'both positivefinancialbenefit trial variants31/120')
    check(observed['positive_financial_benefit_trial_full_training']['breaches']==31,'correct31trial label')
    constrained=pd.read_csv(ROOT/extra[3]);constrained=constrained.loc[constrained.job_id.eq('c0_refit')]
    check(len(constrained)==1 and int(constrained.breach_count.iloc[0])==25 and int(constrained.n_training.iloc[0])==120,'localconstrained25/120')
    prices=pd.read_csv(ROOT/'results/model_review/joint_liquidity_refit_v1/training_candidate_summary.csv')
    eta0=prices.loc[prices.job_id.eq('c0_refit')&prices.execution_candidate_order.eq(0)]
    check(len(eta0)==1 and int(eta0.breach_count.iloc[0])==observed['retained_eta0_full_training']['breaches']==37,'retained eta0 distinct37/120')
    q1=pd.read_csv(ROOT/extra[4]);full=q1.loc[q1.context.eq('full_development')]
    expected_H={'A':5,'B':15,'C':30,'D':25,'E':6};expected_L={'A':.1,'B':.25,'C':.4,'D':.15,'E':.35}
    for client in 'ABCDE':
        row=full.loc[full.client.eq(client)].iloc[0]
        same(row.horizon,expected_H[client],'ownH '+client);same(row.loss_tolerance,expected_L[client],'ownL '+client)
        same(row.effective_assignment,registry['current_static_full120']['profile_Q1_assignments'][client],'Q1assignment '+client)
    endpoint_file='results/model_review/failure_path_mechanisms_v1/account_endpoints.csv.gz';annual_file='results/model_review/failure_path_mechanisms_v1/annual_states.csv.gz'
    endpoints=pd.read_csv(ROOT/endpoint_file,float_precision='round_trip');annual=pd.read_csv(ROOT/annual_file,float_precision='round_trip');fieldcount=0
    for values,frame,key in [(p['severe_example']['endpoints'],endpoints,'endpoint'),(p['severe_example']['actions'],annual,'year')]:
        check(len(values)==8,'8severe '+key+' records')
        for record in values:
            rows=frame.loc[frame.pair_id.eq('outer_r5_112')&frame.arm.eq(record['arm'])&frame[key].eq(record[key])]
            check(len(rows)==1,'unique severe '+key+' identity');row=rows.iloc[0]
            for field,value in record.items():same(row[field],value,'severe '+record['arm']+'/'+str(record[key])+'/'+field);fieldcount+=1
    text=p['customer_explanation']['english'];words=len(text.split());sentences=[x.strip() for x in text.split('.') if x.strip()]
    check(words==p['customer_explanation']['word_count']==44 and words<=p['customer_explanation']['maximum_words']==50,'44word explanation <=50')
    check(len(sentences)==3,'three explanation sentences')
    check('First two sentences' in p['customer_explanation']['purpose'],'preinvestment versus conditionalpostbreach sentence scope')
    check(sha(CONTRACT)==EXPECTED and all(sha(ROOT/rel)==digest for rel,digest in inputs.items()),'contract and all review inputs unchanged')
    result=dict(version='current-model-contract-independent-readonly-review-v1',status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        contract_sha256=EXPECTED,reviewer_source_sha256=sha(Path(__file__)),checks_passed=checks,maximum_numeric_absolute_error=maxerr,
        original_problem_pdf_pages_read=3,original_problem_text_extraction='Local pypdf; actual3pages read before semantic review',raw_market_data_loaded=False,new_fits=0,new_policy_simulations=0,final_scenarios_read=False,
        source_and_contract_sha256=inputs,summary_rows_reconciled=18,severe_endpoint_records_reconciled=8,severe_annual_action_records_reconciled=8,severe_record_fields_reconciled=fieldcount,
        customer_explanation_words=44,customer_explanation_sentences=3,
        semantic_review=dict(no_blocking_inconsistency=True,conclusion='Current contract agrees with original ProblemA and unchanged authoritative registry; remaining autonomousC/final-evaluation limitations remain explicit.',
            closed_ambiguities=['The8finite-price candidates and separate ninth lexicographic riskfirst are distinguished.','Ordinary netreal annual volatility ddof1 is separate from annualendpoint MDD and fractional-tail CVaR90.',
                'Q1 ownH static annualproduct reference continues after Lcrossing; it is separate from Q2 B annualuntilpending and from executionapproval.',
                'Full120 self-including training resubstitution is not prospective heldout evaluation; futureactual returns are excluded from live/heldout information sets.',
                'Positivefinancialbenefit trial31/120 is distinguished from retainedeta0 growthonly37/120; no oldregistry edited.',
                'Actualtradefraction uses moved>1e-12 over original30periods including inception andpending, rather than restorelabels alone.',
                'Severe outer_r5_112 has the same year26 target but different inheritedwealth/realpeak; year27+pending preserves exposure andoriginalhistory despite partial recovery.',
                'First2customer sentences explain currentpreinvestmentReview; finalconditional sentence explains exposed pending investments. No adviser reply or loss guarantee is invented.',
                'Known-cost evidence bridge is shadowcandidate sensitivity with fixedparameter/admission boundary, not empirically verified reapprovalrobustness.',
                'Q2 final121–180 evaluation/report/package remain unfinished and explicitly deferred; no newblindness or deliverysuccess is asserted.']),
        boundaries=['Review is a problem-allowed decision; it does not demonstrate effective reliable autonomousCinvesting or complete final-evaluation duties.',
            'This AI computational/semantic review is not external human peer review, a populationrisk certificate, or forecastlaw calibration.',
            'Only the exact bound draft is reviewed; future contract edits require a new review rather than reusing this receipt.'])
    RECEIPT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','contract_sha256','maximum_numeric_absolute_error','severe_record_fields_reconciled','customer_explanation_words')}));return result
if __name__=='__main__':run()
