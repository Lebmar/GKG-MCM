"""Synthetic-only integration checks for the complete-process runner helpers.

No raw data, cached market outcomes or result files are read; the frozen
protocol files supply declared task identities only.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
import run_nested_dynamic_calibration as runner
from experimental_execution.nested_dynamic_selection import final_status


def fake_family(n, prices):
    p=len(prices); h=runner.H
    return {'real_wealth':np.full((n,p,h+1),100.),
            'cpi':np.ones((n,h+1)), 'target':np.array([.4,.3,.3]),
            'annual_prices':np.asarray(prices).copy(),
            'fees':np.zeros((n,p,h)), 'moved':np.zeros((n,p,h)),
            'weights':np.broadcast_to([.4,.3,.3],(n,p,h+1,3)).copy(),
            'trades':np.zeros((n,p,h),bool),'target_execution':np.zeros((n,p,h),bool),
            'ever_breached':np.zeros((n,p,h+1),bool),
            'first_crossing':np.full((n,p),-1,dtype=int)}


def fake_financial(value=.03):
    return {**dict.fromkeys(runner.FINANCIAL,0.),'log_growth':value,
            'cagr':float(np.expm1(value)),'mdd':.2,'breach':False,'excess':0.,
            'nominal_fees':.5,'trade_fraction':.2,'first_crossing':None}


class PriceAndTraceIntegrationTests(unittest.TestCase):
    def test_truthful_single_price_extract_and_archive_do_not_store_family_catalog(self):
        family=fake_family(2,runner.priced.PRICES)
        before=deepcopy(family)
        for order in (0,4,8):
            arm=runner.truthful_extract(family,order)
            self.assertEqual(arm['real_wealth'].shape,(2,31))
            self.assertEqual(arm['annual_prices'].shape,(1,))
            self.assertEqual(arm['annual_prices'][0],runner.priced.PRICES[order])
            np.testing.assert_array_equal(arm['cpi'],family['cpi'])
            archive={}; groups={}
            runner.save_trace(archive,groups,'toy',arm,[10,11],{'horizon':30},order)
            np.testing.assert_array_equal(archive['toy__annual_prices'],[runner.priced.PRICES[order]])
            self.assertEqual(groups['toy']['selected_candidate_order'],order)
        for k in family: np.testing.assert_array_equal(family[k],before[k])

    def test_unique_selected_orders_use_one_real_price_and_only_the_supplied_training_fan(self):
        actual=np.zeros((2,30,3)); infl=np.zeros((2,30))
        train=np.full((3,30,3),.01); train_infl=np.zeros((3,30)); target=np.array([.4,.3,.3])
        calls=[]
        def simulator(a,i,t,ti,w,prices,**kw):
            self.assertIs(a,actual); self.assertIs(i,infl)
            self.assertIs(t,train); self.assertIs(ti,train_infl); self.assertIs(w,target)
            self.assertEqual(len(prices),1)
            self.assertEqual(kw,dict(L=.4,cost_rate=.0025,horizon=30))
            calls.append(float(prices[0])); return fake_family(len(a),prices)
        arms=runner.evaluate_unique_orders(simulator,actual,infl,train,train_infl,target,[4,None,0,4])
        self.assertEqual(list(arms),[0,4]); self.assertEqual(calls,[0.,.03])
        for order,arm in arms.items():
            np.testing.assert_array_equal(arm['annual_prices'],[runner.priced.PRICES[order]])
        self.assertEqual(runner.evaluate_unique_orders(simulator,actual,infl,train,train_infl,target,[None]),{})
        self.assertEqual(len(calls),2)


class CaseAndDenominatorIntegrationTests(unittest.TestCase):
    def test_final_case_priority_and_all_financial_nulls_are_preserved(self):
        for case in ['risk_review','not_balanced_case']:
            self.assertEqual(final_status(case,None,None),(case,case))
            rows=runner.null_records([1,2],{'case_status':case},case,case)
            self.assertTrue(all(not r['assigned'] and r['execution_status']==case for r in rows))
            self.assertTrue(all(r[k] is None for r in rows for k in runner.FINANCIAL))
        self.assertEqual(final_status('balanced_case',None,None),('execution_review','no_eligible_inner_beta'))
        self.assertEqual(final_status('balanced_case',.025,None),('execution_review','no_admissible_training_price'))
        self.assertEqual(final_status('balanced_case',.025,0),('automated',None))

    def test_original120_case60_and_automatic30_are_distinct_with_review_not_zero_loss(self):
        rows=[]
        for i in range(120):
            status=('automated','execution_review','risk_review','not_balanced_case')[i//30]
            assignment={'automated':'Balanced','execution_review':'Balanced','risk_review':'Review',
                        'not_balanced_case':'Conservative'}[status]
            common=dict(scope='outer_oof',policy=runner.LEARNER,scenario_id=i+1,context='toy',
                        q1_effective_assignment=assignment,
                        case_status='balanced_case' if status in ('automated','execution_review') else status,
                        execution_status=status,execution_reason=None)
            if status=='automated': rows.append({**common,'assigned':True,**fake_financial()})
            else: rows.extend(runner.null_records([i+1],common,status,status))
        s=runner.performance_summary(rows)[0]
        self.assertEqual([s[k] for k in ('original_n','balanced_case_n','case_auto_n','case_execReview_n',
                                      'risk_review_n','not_balanced_case_n')],[120,60,30,30,30,30])
        self.assertEqual(s['financial_null_paths'],90)
        self.assertEqual(s['assigned_paths'],30)
        self.assertAlmostEqual(s['mean_log_growth'],.03)
        self.assertEqual(s['breach_count'],0)

    def test_references_have_hypothetical_status_and_availability_not_service_auto_counts(self):
        arm=runner.truthful_extract(fake_family(2,runner.priced.PRICES),4)
        rows=runner.arm_records(arm,[1,2],dict(scope='outer_oof',context='toy',
            policy='equal_thirds_annual',reference_only=True,case_status='not_balanced_case',
            q1_effective_assignment='Conservative',horizon=30,loss_tolerance=.4))
        self.assertTrue(all(r['hypothetical_reference'] and r['execution_status']=='hypothetical_reference'
                            and r['assigned'] for r in rows))
        s=runner.performance_summary(rows)[0]
        self.assertEqual(s['original_n'],2)
        self.assertEqual(s['reference_available_n'],2)
        self.assertEqual(s['case_reference_available_n'],0)
        self.assertIsNone(s['case_auto_n']); self.assertIsNone(s['case_execReview_n'])
        self.assertTrue(s['reference_only'])

    def test_paired_hypothetical_reference_keeps_all_originals_but_never_fills_learner_null(self):
        rows=[]
        for i in range(4):
            auto=i==0
            common=dict(scope='outer_oof',context='toy',scenario_id=i+1,case_status='balanced_case')
            learner={**common,'policy':runner.LEARNER,'assigned':auto,
                     **(fake_financial(.03) if auto else dict.fromkeys(runner.FINANCIAL))}
            rows.append(learner)
            for p in ('annual_then_pending','equal_thirds_annual'):
                rows.append({**common,'policy':p,'reference_only':True,'assigned':True,**fake_financial(.02)})
        pairs,summaries=runner.paired_outputs(rows)
        self.assertEqual(len(pairs),8)
        self.assertTrue(all(x['delta_log_growth'] is None for x in pairs if not x['common_auto']))
        self.assertTrue(all(x['original_pairs']==4 and x['learner_auto_n']==1 and
                            x['reference_available_n']==4 and x['common_auto_n']==1 for x in summaries))
        for x in summaries: self.assertAlmostEqual(x['mean_delta_log_growth'],.01)


class ProtocolOrderingTests(unittest.TestCase):
    def test_protocol_rejects_refit_before_its_four_inner_validation_jobs(self):
        p=json.loads(runner.PROTOCOL.read_text())
        legacy=json.loads((ROOT/'docs/positive_asset_specification_protocol.json').read_text())['contexts']
        runner.validate_protocol(p,legacy)
        bad=deepcopy(p)
        refit=next(j for j in bad['jobs'] if j['parent_context_id']==0 and j['stage']=='refit')
        bad['jobs'].remove(refit); bad['jobs'].insert(0,refit)
        with self.assertRaises(ValueError): runner.validate_protocol(bad,legacy)


if __name__=='__main__': unittest.main()
