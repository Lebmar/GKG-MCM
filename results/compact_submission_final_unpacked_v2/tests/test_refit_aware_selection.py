"""Pure training evidence selector tests; no saved market results loaded."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research')]
from experimental_execution import refit_aware_selection as m


def inner(spec=None,n=100):
    spec=spec or {b:(100,5,.03) for b in m.BETAS}
    rows=[]
    for b in m.BETAS:
        a,k,g=spec[b]
        rows.append(dict(training_beta=b,original_n=n,balanced_case_n=n,case_auto_n=a,case_execReview_n=n-a,
            risk_review_n=0,not_balanced_case_n=0,breach_count=k,conditional_mean_log_growth=g,
            eligible=bool(a) and k<=int(np.floor(.1*a+1e-12))))
    return rows


def train(k=6,n=96):
    return [dict(execution_candidate_order=i,n_training=n,breach_count=k,
                 mean_log_growth=.03+i*.001) for i in range(9)]


class TopSetTests(unittest.TestCase):
    def test_E_not_all_eligible_and_not_old_minimum_tie_collapse(self):
        rows=inner({0.:(80,0,.2),.025:(100,5,.03),.05:(100,5,.03+.5e-12),
                    .075:(100,5,.02),.1:(100,20,.2)})
        e=m.top_inner_set(rows)
        self.assertEqual(e['eligible_betas'],[0.,.025,.05,.075])
        self.assertEqual(e['top_E'],[.025,.05])

    def test_global_growth_band_has_no_pairwise_tolerance_chain(self):
        rows=inner({0.:(100,5,.03),.025:(100,5,.03+.75e-12),.05:(100,5,.03+1.5e-12),
                    .075:(100,5,.01),.1:(100,5,.01)})
        self.assertEqual(m.top_inner_set(rows)['top_E'],[.025,.05])

    def test_zero_evidence_or_no_risk_eligible_is_empty_not_zero_risk(self):
        zero=inner({b:(0,None,None) for b in m.BETAS})
        self.assertEqual(m.top_inner_set(zero)['top_E'],[])
        bad=inner({b:(100,11,.03) for b in m.BETAS})
        self.assertEqual(m.top_inner_set(bad)['eligible_betas'],[])

    def test_invalid_betas_denominators_counts_and_saved_risk_flags_rejected(self):
        invalid=[]
        for change in ['beta','boolbeta','n','floatn','auton','wrongrisk','zerofilled']:
            rows=inner()
            if change=='beta':rows[0]['training_beta']=.2
            if change=='boolbeta':rows[0]['training_beta']=False
            if change=='n':rows[0]['original_n']=99
            if change=='floatn':rows[0]['original_n']=100.
            if change=='auton':rows[0]['case_auto_n']=101
            if change=='wrongrisk':rows[0]['eligible']=False
            if change=='zerofilled':rows[0].update(case_auto_n=0,case_execReview_n=100,breach_count=0)
            invalid.append(rows)
        for rows in invalid:
            with self.assertRaises(ValueError):m.top_inner_set(rows)


class RefitRuleTests(unittest.TestCase):
    def test_first_refit_feasible_beta_in_E_only_uses_original_price_rule(self):
        rows=inner({b:(100,5,.03) if b in [.05,.075,.1] else (0,None,None) for b in m.BETAS})
        before=deepcopy((rows,train()))
        result=m.select_refit_aware('Balanced',rows,train(),6,96)
        self.assertEqual(result['inner_sets']['top_E'],[.05,.075,.1])
        self.assertEqual(result['selected_beta'],.075)
        self.assertEqual(result['selected_candidate_order'],8)
        self.assertEqual([x['training_beta'] for x in result['refit_attempts']],[.05,.075])
        self.assertEqual((rows,train()),before)
        self.assertFalse(result['financial_outcomes_evaluated'])

    def test_beta_outside_E_cannot_rescue_even_if_eligible_and_refit_feasible(self):
        rows=inner({b:(100,5,.03) if b==.05 else (100,5,.02) for b in m.BETAS})
        result=m.select_refit_aware('Balanced',rows,train(),6,96)
        self.assertEqual(result['inner_sets']['top_E'],[.05])
        self.assertEqual(result['execution_status'],'execution_review')
        self.assertEqual([x['training_beta'] for x in result['refit_attempts']],[.05])
        self.assertIsNone(result['selected_beta'])

    def test_all_E_refit_fail_does_not_change_cap_or_fallback_to_reference(self):
        result=m.select_refit_aware('Balanced',inner(),train(k=8),7,96)
        self.assertEqual(result['execution_status'],'execution_review')
        self.assertIsNone(result['selected_candidate_order'])
        self.assertTrue(all(x['training_breach_cap']<=7 for x in result['refit_attempts']))
        empty=m.select_refit_aware('Balanced',inner({b:(0,None,None) for b in m.BETAS}),train(),6,96)
        self.assertEqual(empty['execution_review_reason'],'no_top_inner_beta')
        self.assertEqual(empty['refit_attempts'],[])

    def test_Q1_case_priority_does_not_evaluate_invalid_execution_inputs(self):
        for label,status in [('Review','risk_review'),('Conservative','not_balanced_case'),('Growth','not_balanced_case')]:
            with patch.object(m,'price_selection',side_effect=AssertionError('Should not select')):
                result=m.select_refit_aware(label,None,None,None,None)
            self.assertEqual(result['execution_status'],status)
            self.assertIsNone(result['selected_beta']);self.assertEqual(result['refit_attempts'],[])
        with self.assertRaises(ValueError):m.select_refit_aware('Unknown',None,None,None,None)

    def test_refit_training_denominator_and_candidate_family_validation(self):
        for n,kB,summary in [(96.,6,train()),(0,0,train()),(96,97,train()),(96,6,train()[:-1])]:
            with self.assertRaises(ValueError):m.select_refit_aware('Balanced',inner(),summary,kB,n)
        summaries=train();summaries[0]['n_training']=72
        with self.assertRaises(ValueError):m.select_refit_aware('Balanced',inner(),summaries,6,96)
        summaries=train();summaries[0]['breach_count']=6.
        with self.assertRaises(ValueError):m.select_refit_aware('Balanced',inner(),summaries,6,96)

    def test_behavior_classes_are_separate_from_top_E(self):
        records=[]
        for b in m.BETAS:
            for sid in [1,2]:
                records.append(dict(training_beta=b,scenario_id=sid,job_id='inner',case_status='balanced_case',
                    execution_status='automated',assigned=True,selected_candidate_order=4))
        classes=m.inner_behavior_classes(records)
        self.assertEqual(classes[0]['betas'],list(m.BETAS))
        rows=inner({b:(100,5,.03) if b==.05 else (100,5,.02) for b in m.BETAS})
        self.assertEqual(m.top_inner_set(rows)['top_E'],[.05])
        invalid=deepcopy(records);invalid[0]['scenario_id']=121
        with self.assertRaises(ValueError):m.inner_behavior_classes(invalid)


if __name__=='__main__':unittest.main()
