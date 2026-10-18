import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out

a=module('forecast_assumption_auditor','scripts/audit_forecast_assumptions_v2.py')
r=module('forecast_assumption_original','scripts/diagnose_forecast_assumptions_v2.py')


class ForecastAssumptionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(2881)
        cls.market=rng.normal(.025,.04,(120,30,3));cls.inflation=rng.normal(.02,.01,(120,30))
        cls.folds=[];ids=list(range(1,121))
        for k in range(5):
            test=ids[k*24:(k+1)*24]
            cls.folds.append({'fold':k+1,'training_ids':sorted(set(ids)-set(test)),'evaluation_ids':test})
        cls.tables=r.compute_diagnostic(cls.market,cls.inflation,ids,cls.folds)
        cls.tables['fold_summary']=r.aggregate(cls.tables['path_losses'],by_fold=True)
        cls.tables['summary']=r.aggregate(cls.tables['path_losses'],by_fold=False)
        cls.findings={'primary_vector_results':r.vector_summary(cls.tables['path_losses'],cls.tables['fold_summary'])}

    def test_independent_augmented_solver_and_complete_table_audit(self):
        report=a.audit_tables(self.market,self.inflation,range(1,121),self.folds,self.tables,self.findings)
        self.assertGreater(report['numeric_checks'],400000)
        self.assertLess(report['maximum_absolute_difference'],1e-12)

    def test_corrupt_future_prediction_is_detected(self):
        old=self.tables['annual_predictions'][0]['fixed_state_prediction']
        self.tables['annual_predictions'][0]['fixed_state_prediction']=old+.001
        try:
            with self.assertRaises(AssertionError):a.audit_tables(self.market,self.inflation,range(1,121),self.folds,self.tables,self.findings)
        finally:self.tables['annual_predictions'][0]['fixed_state_prediction']=old

    def test_primary_vector_cannot_include_inflation_or_wrong_path_weight(self):
        old=self.findings['primary_vector_results'][0]['state_to_calendar_mse_ratio']
        self.findings['primary_vector_results'][0]['state_to_calendar_mse_ratio']=old+.01
        try:
            with self.assertRaises(AssertionError):a.audit_tables(self.market,self.inflation,range(1,121),self.folds,self.tables,self.findings)
        finally:self.findings['primary_vector_results'][0]['state_to_calendar_mse_ratio']=old
