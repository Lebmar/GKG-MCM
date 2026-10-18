import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('forecast_assumption_runner',ROOT/'scripts/diagnose_forecast_assumptions_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ForecastAssumptionRunnerTests(unittest.TestCase):
    def test_nonempty_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"archive"
            m.ensure_fresh_output(p)
            p.mkdir();m.ensure_fresh_output(p)
            marker=p/"sentinel.txt";marker.write_text("preserve")
            with self.assertRaises(FileExistsError):m.ensure_fresh_output(p)
            self.assertEqual(marker.read_text(),"preserve")

    @staticmethod
    def folds():
        ids=list(range(1,121));out=[]
        for k in range(5):
            e=ids[k*24:(k+1)*24]
            out.append({'fold':k+1,'training_ids':sorted(set(ids)-set(e)),'evaluation_ids':e})
        return out

    def test_full_synthetic_inventory_and_path_first_aggregation(self):
        rng=np.random.default_rng(287)
        r=rng.normal(.03,.04,(120,30,3));pi=rng.normal(.02,.01,(120,30))
        tables=m.compute_diagnostic(r,pi,range(1,121),self.folds())
        self.assertEqual(len(tables['annual_predictions']),120*29*4*2)
        self.assertEqual(len(tables['path_losses']),120*4*2)
        self.assertEqual(len(tables['coefficients']),5*2*4*4)
        self.assertEqual(len(m.aggregate(tables['path_losses'],by_fold=True)),40)
        v=m.vector_summary(tables['path_losses'],[])
        self.assertEqual(len(v),2)
        self.assertEqual(v[0]['original_path_n'],120)
        self.assertEqual(len(v[0]['fold_results']),5)
        subset=[a for a in tables['annual_predictions'] if a['scenario_id']==1 and a['channel']=='equity_real_log' and a['target']=='next_year']
        row=next(a for a in tables['path_losses'] if a['scenario_id']==1 and a['channel']=='equity_real_log' and a['target']=='next_year')
        self.assertAlmostEqual(row['baseline_mse'],np.mean([(a['actual_log_quantity']-a['calendar_mean_prediction'])**2 for a in subset]))

    def test_held_out_future_is_scoring_only_at_first_decision(self):
        rng=np.random.default_rng(299)
        r=rng.normal(.02,.03,(120,30,3));pi=rng.normal(.02,.01,(120,30))
        a=m.compute_diagnostic(r,pi,range(1,121),self.folds())
        r[:24,1:]+=.25;pi[:24,1:]+=.04
        b=m.compute_diagnostic(r,pi,range(1,121),self.folds())
        select=lambda tab:[x for x in tab['annual_predictions'] if x['fold']==1 and x['decision_original_year']==2]
        aa,bb=select(a),select(b)
        self.assertEqual(len(aa),24*4*2)
        for x,y in zip(aa,bb,strict=True):
            self.assertEqual(x['calendar_mean_prediction'],y['calendar_mean_prediction'])
            self.assertEqual(x['fixed_state_prediction'],y['fixed_state_prediction'])
        self.assertNotEqual(aa[0]['actual_log_quantity'],bb[0]['actual_log_quantity'])
