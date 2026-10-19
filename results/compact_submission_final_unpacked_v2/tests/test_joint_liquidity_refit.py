"""Pure helper integration tests for joint-liquidity two-learner research.

Only declared protocol identities are read; no market inputs or saved outcomes.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
import run_joint_liquidity_refit as runner


def family(n, prices):
    p = len(prices)
    return {'real_wealth':np.full((n,p,31),100.), 'cpi':np.ones((n,31)),
        'target':np.array([.4,.3,.3]), 'annual_prices':np.asarray(prices).copy(),
        'fees':np.zeros((n,p,30)), 'moved':np.zeros((n,p,30)),
        'weights':np.broadcast_to([.4,.3,.3],(n,p,31,3)).copy(),
        'trades':np.zeros((n,p,30),bool),'target_execution':np.zeros((n,p,30),bool),
        'ever_breached':np.zeros((n,p,31),bool),'first_crossing':np.full((n,p),-1,dtype=int)}


def financial(value=.03):
    return {**dict.fromkeys(runner.FINANCIAL,0.),'log_growth':value,
        'cagr':float(np.expm1(value)),'mdd':.2,'breach':False,'excess':0.,
        'nominal_fees':.5,'trade_fraction':.2,'first_crossing':None}


def row(sid, status, *, policy=None, value=.03, fold=1):
    assignment = {'automated':'Balanced','execution_review':'Balanced',
                  'q1_review':'Review','not_balanced_case':'Growth'}[status]
    common = dict(scope='outer_oof',policy=policy or runner.LEARNERS[0],scenario_id=sid,
        context='toy',dynamic_inner_fold=fold,q1_effective_assignment=assignment,
        case_status=runner.case_of(assignment),execution_status=status,execution_reason=None)
    if status=='automated': return {**common,'assigned':True,**financial(value)}
    return runner.null_records([sid],common,status,status)[0]


class PriceTraceTests(unittest.TestCase):
    def test_single_price_metadata_and_exact_training_fan_with_deduplicated_orders(self):
        f = family(2,runner.priced.PRICES); old=deepcopy(f)
        for order in (0,4,8):
            arm=runner.truthful_extract(f,order)
            np.testing.assert_array_equal(arm['annual_prices'],[runner.priced.PRICES[order]])
            self.assertEqual(arm['real_wealth'].shape,(2,31))
            archive,groups={},{}
            runner.save_trace(archive,groups,'same',arm,[8,9],{'horizon':30},order)
            np.testing.assert_array_equal(archive['same__annual_prices'],[runner.priced.PRICES[order]])
        for key in old: np.testing.assert_array_equal(old[key],f[key])
        actual=np.zeros((2,30,3));infl=np.zeros((2,30))
        train=np.full((3,30,3),.01);ti=np.zeros((3,30));target=np.array([.4,.3,.3]);calls=[]
        def engine(a,i,t,tinfl,w,prices,**kwargs):
            self.assertIs(a,actual);self.assertIs(i,infl);self.assertIs(t,train)
            self.assertIs(tinfl,ti);self.assertIs(w,target)
            self.assertEqual(kwargs,dict(L=.4,cost_rate=.0025,horizon=30))
            self.assertEqual(len(prices),1);calls.append(prices[0]);return family(2,prices)
        arms=runner.evaluate_unique_orders(engine,actual,infl,train,ti,target,[4,4,None,8])
        self.assertEqual(list(arms),[4,8]);self.assertEqual(calls,[.03,np.inf])
        self.assertEqual(runner.evaluate_unique_orders(engine,actual,infl,train,ti,target,[None]),{})
        self.assertEqual(len(calls),2)

    def test_invalid_selected_orders_never_start_an_evaluation(self):
        def forbidden(*args,**kwargs): self.fail('Invalid selection was simulated')
        dummy=np.zeros((1,30,3));infl=np.zeros((1,30))
        for value in (-1,9,True,1.5):
            with self.assertRaises(ValueError):
                runner.evaluate_unique_orders(forbidden,dummy,infl,dummy,infl,np.ones(3)/3,[value])


class JointCaseTests(unittest.TestCase):
    def test_refit_aware_can_use_only_the_top_inner_tie_set_not_arbitrary_eligible_betas(self):
        bs=[]
        for beta in runner.BETAS:
            auto=96 if beta>=.05 else 0
            bs.append(dict(training_beta=beta,original_n=96,balanced_case_n=96,q1_review_n=0,
                not_balanced_case_n=0,case_auto_n=auto,case_execReview_n=96-auto,
                breach_count=6 if auto else None,conditional_mean_log_growth=.03 if auto else None,
                eligible=bool(auto)))
        train=[dict(execution_candidate_order=i,n_training=96,breach_count=6,
                    mean_log_growth=.04 if i==4 else .03) for i in range(9)]
        selections={str(b):runner.price_selection(train,6,96,b) for b in runner.BETAS}
        methods=runner.final_choices('Balanced',bs,train,selections,6,96)
        canonical,aware=[methods[m] for m in runner.LEARNERS]
        self.assertEqual(canonical['selected_beta'],.05)
        self.assertEqual(canonical['execution_status'],'execution_review')
        self.assertEqual(aware['selected_beta'],.075);self.assertEqual(aware['selected_candidate_order'],4)
        self.assertEqual(aware['execution_status'],'automated')
        # A lower-growth but eligible refit-feasible beta outside E cannot save
        # admission. This tests evidence sharing without changing old tie rules.
        for summary in bs:
            if summary['training_beta']>.05:summary['conditional_mean_log_growth']=.02
        failed=runner.final_choices('Balanced',bs,train,selections,6,96)
        self.assertTrue(all(x['execution_status']=='execution_review' for x in failed.values()))
        for assignment,status in [('Review','q1_review'),('Growth','not_balanced_case')]:
            classified=runner.final_choices(assignment,bs,[],selections,None,96)
            self.assertTrue(all(x['execution_status']==status and x['selected_candidate_order'] is None
                                for x in classified.values()))

    def test_original_case_auto_and_q1_review_denominators_are_distinct(self):
        rows=[row(i+1,('automated','execution_review','q1_review','not_balanced_case')[i//30]) for i in range(120)]
        s=runner.performance_summary(rows)[0]
        self.assertEqual([s[k] for k in ('original_n','balanced_case_n','case_auto_n','case_execReview_n',
                                        'q1_review_n','not_balanced_case_n')],[120,60,30,30,30,30])
        self.assertEqual(s['financial_null_paths'],90);self.assertEqual(s['assigned_paths'],30)
        self.assertAlmostEqual(s['mean_log_growth'],.03);self.assertEqual(s['breach_count'],0)

    def test_case_labels_duplicate_ids_or_nonauto_financials_are_rejected(self):
        for bad in [[row(1,'automated'),row(1,'automated')],
                    [{**row(1,'q1_review'),'case_status':'risk_review'}],
                    [{**row(1,'execution_review'),'mdd':0.}],
                    [{**row(1,'q1_review'),'execution_status':'risk_review'}]]:
            with self.assertRaises(ValueError):runner.joint_counts(bad)

    def test_pooled_unequal_inner_folds_score_one_original_path_each(self):
        rows=[]
        for i in range(90):
            r=row(i+1,'automated',fold=1 if i<23 else 2 if i<46 else 3 if i<68 else 4)
            r.update(training_beta=.05,breach=i<9,mdd=.5 if i<9 else .2,
                     log_growth=.02 if i<23 else .04)
            rows.append(r)
        out=runner.joint_beta_summary(rows,.05,list(range(1,91)))
        self.assertEqual(out['case_auto_n'],90);self.assertEqual(out['breach_count'],9)
        self.assertTrue(out['eligible']);self.assertEqual(out['auto_inner_folds'],4)
        self.assertAlmostEqual(out['conditional_mean_log_growth'],(23*.02+67*.04)/90)
        rows[9]['breach']=True;rows[9]['mdd']=.5
        self.assertFalse(runner.joint_beta_summary(rows,.05,list(range(1,91)))['eligible'])

    def test_zero_auto_risk_is_null_not_zero_and_new_review_case_is_preserved(self):
        rows=[{**row(i+1,'q1_review'),'training_beta':0.} for i in range(24)]
        out=runner.joint_beta_summary(rows,0.,list(range(1,25)))
        self.assertFalse(out['eligible']);self.assertEqual(out['q1_review_n'],24)
        self.assertIsNone(out['breach_count']);self.assertIsNone(out['conditional_breach_probability'])
        self.assertIsNone(out['conditional_mean_log_growth'])

    def test_reference_availability_never_becomes_service_admission(self):
        arm=runner.truthful_extract(family(2,runner.priced.PRICES),4)
        rows=runner.arm_records(arm,[1,2],dict(scope='outer_oof',context='toy',policy='equal_thirds_annual',
            reference_only=True,case_status='q1_review',q1_effective_assignment='Review',horizon=30,loss_tolerance=.4))
        self.assertTrue(all(r['hypothetical_reference'] and r['execution_status']=='hypothetical_reference' for r in rows))
        out=runner.performance_summary(rows)[0]
        self.assertEqual(out['reference_available_n'],2);self.assertEqual(out['case_reference_available_n'],0)
        self.assertIsNone(out['case_auto_n']);self.assertIsNone(out['case_execReview_n'])

    def test_two_learners_and_references_compare_common_auto_only(self):
        rows=[]
        for i in range(4):
            for k,learner in enumerate(runner.LEARNERS):
                rows.append(row(i+1,'automated' if i<=k else 'execution_review',policy=learner,value=.03+.01*k))
            for policy in ('annual_then_pending','equal_thirds_annual'):
                rows.append({**row(i+1,'automated',policy=policy,value=.02),'reference_only':True})
        pairs,summaries=runner.paired_outputs(rows)
        self.assertEqual(len(pairs),20);self.assertEqual(len(summaries),5)
        cross=next(s for s in summaries if s['reference_policy']==runner.LEARNERS[0])
        self.assertFalse(cross['reference_is_hypothetical'])
        self.assertEqual(cross['learner_auto_n'],2);self.assertEqual(cross['reference_available_n'],1)
        self.assertEqual(cross['common_auto_n'],1);self.assertAlmostEqual(cross['mean_delta_log_growth'],.01)
        self.assertTrue(all(r['delta_log_growth'] is None for r in pairs if not r['common_auto']))

    def test_q1_financials_keep_own_profile_horizon_and_review_nulls(self):
        profiles=[dict(name=n,horizon=h,loss_tolerance=l,liquidity=q) for n,h,l,q in
                  [('A',5,.1,'high'),('B',15,.25,'medium'),('C',30,.4,'low'),
                   ('D',25,.15,'medium'),('E',6,.35,'high')]]
        metrics={h:{key:np.full((120,1),h*.001 if key=='log_growth' else .01)
                    for key in ('log_growth','cagr','mdd','volatility','terminal_real')}
                 for h in (5,6,15,25,30)}
        effective={p['name']:'Review' if p['name']=='A' else 'Balanced' for p in profiles}
        product=dict(name='Balanced',candidate_index=0,candidate_original232_index=73,weights=[.3,.3,.4])
        review={p['name']:dict(q1_review_kind='no_inner_auto_evidence' if p['name']=='A' else None,
                    q1_review_reasons=['no_inner_auto_evidence'] if p['name']=='A' else [],
                    feasibility_diagnostics=dict(risk_feasible_n=172,liquidity_feasible_n=100,
                        joint_feasible_n=100,menu_joint_feasible_n=1,
                        floor=runner.LIQUIDITY_FLOORS[p['liquidity']],training_n=72,
                        training_screen_gamma=.05)) for p in profiles}
        bundle=dict(refit=dict(raw_fit=dict(menu=[product]),selection=dict(selected_gamma=.05),
            raw_assignments=effective,effective_assignments=effective,q1_review_details=review))
        assignments,rows=runner.q1_records(bundle,profiles,metrics,[119,120],3,'toy')
        self.assertEqual(len(assignments),5);self.assertEqual(len(rows),10)
        # Real adapter diagnostics repeat floor: export must merge rather than
        # passing duplicate dict() keyword arguments (the attempt1 regression).
        for item in assignments:
            profile=next(p for p in profiles if p['name']==item['client'])
            self.assertEqual(item['floor'],runner.LIQUIDITY_FLOORS[profile['liquidity']])
            self.assertEqual(item['training_n'],72);self.assertEqual(item['training_screen_gamma'],.05)
            self.assertEqual(item['risk_feasible_n'],172);self.assertEqual(item['joint_feasible_n'],100)
        self.assertTrue(all(r['log_growth'] is None for r in rows if r['client']=='A'))
        self.assertEqual({r['log_growth'] for r in rows if r['client']=='C'},{.03})
        self.assertEqual({r['log_growth'] for r in rows if r['client']=='E'},{.006})
        self.assertEqual({r['horizon'] for r in rows if r['client']=='E'},{6})
        summary=runner.q1_summary(rows)
        self.assertEqual(len(summary),5)
        a=next(r for r in summary if r['client']=='A')
        self.assertEqual(a['original_n'],2);self.assertEqual(a['annual_assigned_n'],0)
        self.assertIsNone(a['conditional_breach_probability']);self.assertIsNone(a['mean_log_growth'])


class ProtocolTests(unittest.TestCase):
    def test_declared_floor_family_and_all_four_inner_tasks_must_precede_refit(self):
        protocol=json.loads(runner.PROTOCOL.read_text())
        contexts=json.loads((ROOT/'docs/positive_asset_specification_protocol.json').read_text())['contexts']
        self.assertEqual(len(runner.validate_protocol(protocol,contexts)),30)
        bad=deepcopy(protocol);bad['liquidity_policy']['floors']['high']=.5
        with self.assertRaises(ValueError):runner.validate_protocol(bad,contexts)
        bad=deepcopy(protocol);bad['jobs'][0],bad['jobs'][4]=bad['jobs'][4],bad['jobs'][0]
        with self.assertRaises(ValueError):runner.validate_protocol(bad,contexts)


if __name__=='__main__':unittest.main()
