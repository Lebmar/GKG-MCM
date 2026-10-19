import copy
import unittest

from experimental_execution.state_budget_selection import select_inner,summarize_records,CANDIDATES


def records(n=20, mdd=.1, growth=.03, funded=True):
    return [dict(scenario_id=i+1,assigned=funded,breach=mdd>.4,log_growth=growth,
        cagr=.03,mdd=mdd,volatility=.1,nominal_fees=.4,trade_fraction=.25,
        support_review=False,automatic_fraction=1.,protection_trade_fraction=0.) for i in range(n)]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ref=records(mdd=.2)
        self.rows={s['candidate_id']:records(growth=.03+.001*s['order']) for s in CANDIDATES}

    def test_selects_by_declared_complete_inner_evidence(self):
        self.assertEqual(select_inner(self.rows,self.ref)['selected_candidate_id'],CANDIDATES[-1]['candidate_id'])

    def test_zero_financials_null_not_zero_risk(self):
        for s in CANDIDATES: self.rows[s['candidate_id']]=records(funded=False)
        x=select_inner(self.rows,self.ref)
        self.assertEqual(x['status'],'execution_review')
        self.assertIsNone(x['evidence'][0]['summary']['breach_fraction'])

    def test_tail_risk_not_hidden_by_low_event_count(self):
        self.rows[CANDIDATES[-1]['candidate_id']][-1]['mdd']=.39
        x=select_inner(self.rows,self.ref)
        self.assertFalse(x['evidence'][-1]['eligible'])
        self.assertIn('inner_cvar90_mdd_worse_than_same_cohort_B',x['evidence'][-1]['reasons'])

    def test_event_cap_is_financial_denominator(self):
        r=self.rows[CANDIDATES[0]['candidate_id']]
        for row in r[:3]: row['breach']=True
        self.assertFalse(select_inner(self.rows,self.ref)['evidence'][0]['eligible'])

    def test_post_investment_review_retains_financial_and_loss(self):
        for row in self.rows[CANDIDATES[-1]['candidate_id']]: row['support_review']=True
        x=select_inner(self.rows,self.ref)
        self.assertNotEqual(x['selected_candidate_id'],CANDIDATES[-1]['candidate_id'])
        self.assertEqual(x['evidence'][-1]['summary']['financial_n'],20)

    def test_same_cohort_references_not_unconditional_substitute(self):
        self.rows[CANDIDATES[0]['candidate_id']][0]['assigned']=False
        self.ref[0]['mdd']=.9
        self.assertEqual(select_inner(self.rows,self.ref)['evidence'][0]['same_cohort_B']['financial_n'],19)

    def test_missing_and_duplicate_ids_rejected(self):
        x=copy.deepcopy(self.rows);x[CANDIDATES[0]['candidate_id']].pop()
        with self.assertRaises(ValueError):select_inner(x,self.ref)
        with self.assertRaises(ValueError):select_inner(self.rows,self.ref+[self.ref[0]])
        with self.assertRaises(ValueError):summarize_records([self.ref[0],self.ref[0]])

    def test_true_ties_use_predeclared_order(self):
        x={s['candidate_id']:records() for s in CANDIDATES}
        self.assertEqual(select_inner(x,self.ref)['selected_candidate_id'],CANDIDATES[0]['candidate_id'])

    def test_nonbalanced_and_Q1_review_are_not_execution_refusals(self):
        rows=records(3,funded=False)
        rows[0]['review_reason']='not_balanced_case';rows[1]['review_reason']='q1_review'
        s=summarize_records(rows)
        self.assertEqual(s['not_balanced_n'],1);self.assertEqual(s['q1_review_n'],1)
        self.assertEqual(s['initial_execution_review_n'],1)


if __name__=='__main__':unittest.main()
