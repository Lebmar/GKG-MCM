import importlib.util
from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('refit_stability_summary',ROOT/'scripts/summarize_refit_stability.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RefitStabilitySummaryTests(unittest.TestCase):
    def test_eight_synthetic_denominator_null_pairing_boundaries(self):
        self.assertEqual(m.self_test(),{'status':'passed','synthetic_boundary_cases':8,'market_or_result_files_opened':False})

    @staticmethod
    def fixture(assignment='Balanced'):
        product={'weights':[.4,.25,.35]}
        case={'effective_assignment':assignment,'public_status':m.case_of(assignment),'q1_product':product if assignment!='Review' else None,
              'product':product if assignment=='Balanced' else None,'q1_review_kind':None,'q1_review_reasons':[]}
        E=[.05,.075,.1];inner={'top_E':E,'maximum_auto_coverage':96,'maximum_mean_log_growth_at_coverage':.03}
        c={'repeat_id':1,'outer_fold':1,'training_ids':list(range(1,97)),'evaluation_ids':list(range(97,121)),
           'case':case,'static_gamma':.075,'methods':{
            m.LEARNERS[0]:{'selected_beta':.05,'selected_candidate_order':None,'execution_status':'execution_review',
                'execution_reason':'no_admissible_training_price','selection_evidence':{'inner_sets':inner}},
            m.LEARNERS[1]:{'selected_beta':.075,'selected_candidate_order':0,'execution_status':'automated',
                'execution_reason':None,'selection_evidence':{'inner_sets':inner,'refit_attempts':[{'training_beta':.05},{'training_beta':.075}]}}}}
        bundle={'reference_training_breach_count':9 if assignment=='Balanced' else None,'training_price_selections':{
            '.05':{},'0.05':{'selected_candidate_order':None},'0.075':{'selected_candidate_order':0},'0.1':{'selected_candidate_order':0}}}
        stats=pd.DataFrame([{'job_id':'c101_refit','execution_candidate_order':i,'breach_count':6+i//3 if assignment=='Balanced' else None} for i in range(9)])
        fs=pd.DataFrame([{'context':'example','policy':p,'original_n':24,'breach_count':None} for p in m.LEARNERS])
        return {'example':c},{'c101_refit':bundle},stats,fs

    def test_all_top_set_ledgers_distinguish_actual_attempts(self):
        context,margins=m.context_tables(*self.fixture())
        self.assertEqual(len(context),2);self.assertEqual(len(margins),6)
        canonical=margins.loc[margins.policy.eq(m.LEARNERS[0])]
        aware=margins.loc[margins.policy.eq(m.LEARNERS[1])]
        self.assertEqual(int(canonical.algorithm_attempted.sum()),1)
        self.assertEqual(int(aware.algorithm_attempted.sum()),2)
        self.assertEqual(int(aware.executed_beta.sum()),1)
        self.assertEqual(aware.loc[aware.executed_beta,'training_beta'].iloc[0],.075)
        self.assertEqual(aware.loc[aware.executed_beta,'beta_selected_training_slack'].iloc[0],1)
        self.assertFalse(bool(aware.loc[aware.training_beta.eq(.1),'algorithm_attempted'].iloc[0]))

    def test_outside_case_preserves_inner_set_without_inventing_training_evidence(self):
        args=list(self.fixture('Growth'))
        c=args[0]['example']
        for p in m.LEARNERS:
            c['methods'][p]['execution_status']='not_balanced_case';c['methods'][p]['selected_candidate_order']=None
        context,margins=m.context_tables(*args)
        self.assertTrue(margins.empty)
        self.assertIn('training_breach_cap',margins.columns)
        self.assertTrue(context.minimum_nine_training_breach_count.isna().all())
        self.assertFalse(context.refit_margin_available.any())

    def test_outer_margin_cannot_silently_change_training_denominator(self):
        stats=self.fixture()[2]
        with self.assertRaises(ValueError):m.training_margin(stats,9,72,.075,0)
