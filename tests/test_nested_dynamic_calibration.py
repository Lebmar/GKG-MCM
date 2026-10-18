"""Runner admission/identity/selected-price tests with synthetic inputs only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"research"),str(ROOT/"src")]
import run_nested_dynamic_calibration as runner


class SelectedValidationTests(unittest.TestCase):
    def test_protocol_preserves_all_thirty_tasks_and_refit_order(self):
        protocol=json.loads((ROOT/'docs/nested_dynamic_calibration_protocol.json').read_text())
        contexts=json.loads((ROOT/'docs/positive_asset_specification_protocol.json').read_text())['contexts']
        self.assertEqual(len(runner.validate_protocol(protocol,contexts)),30)
        bad=deepcopy(protocol); bad['jobs'][0],bad['jobs'][4]=bad['jobs'][4],bad['jobs'][0]
        with self.assertRaises(ValueError): runner.validate_protocol(bad,contexts)
        bad=deepcopy(protocol); bad['jobs'][0]['training_ids'][0]=bad['jobs'][0]['evaluation_ids'][0]
        with self.assertRaises(ValueError): runner.validate_protocol(bad,contexts)

    def test_unique_preselected_prices_only_and_no_calls_for_referral(self):
        calls=[]
        def fake(actual,infl,train,train_infl,target,prices,**kw):
            calls.append(prices.copy())
            return {"real_wealth":np.full((len(actual),1,31),100.),"cpi":np.ones((len(actual),31)),"annual_prices":prices.copy()}
        actual=np.zeros((2,30,3)); inf=np.zeros((2,30)); train=np.zeros((4,30,3)); tinf=np.zeros((4,30))
        result=runner.evaluate_unique_orders(fake,actual,inf,train,tinf,[.4,.2,.4],[None,4,4,8,None])
        self.assertEqual(sorted(result),[4,8]); self.assertEqual(len(calls),2)
        np.testing.assert_array_equal(calls[0],[.03]); self.assertTrue(np.isposinf(calls[1][0]))
        self.assertEqual(runner.evaluate_unique_orders(fake,actual,inf,train,tinf,None,[None]),{})
        self.assertEqual(len(calls),2)
        runner.evaluate_unique_orders(fake,actual+7,inf,train,tinf,[.4,.2,.4],[4])
        np.testing.assert_array_equal(calls[-1],[.03])

    def test_actual_single_price_metadata_not_legacy_family_catalog(self):
        family={"real_wealth":np.ones((2,9,31)),"cpi":np.ones((2,31)),"annual_prices":runner.priced.PRICES.copy()}
        extracted=runner.truthful_extract(family,4)
        self.assertEqual(extracted["real_wealth"].shape,(2,31))
        np.testing.assert_array_equal(extracted["annual_prices"],[.03])
        self.assertEqual(family["annual_prices"].shape,(9,))

    def test_null_rows_keep_other_product_distinct_from_review(self):
        rows=runner.null_records([1,2],{"q1_effective_assignment":"Growth","policy":runner.LEARNER},"not_balanced_case","not_balanced_case")
        self.assertFalse(rows[0]["assigned"])
        self.assertTrue(all(rows[0][k] is None for k in runner.FINANCIAL))
        self.assertEqual(rows[0]["q1_effective_assignment"],"Growth")


class SummaryTests(unittest.TestCase):
    def test_zero_auto_financial_outcomes_are_null_and_reference_not_imputed(self):
        common={"scope":"outer_oof","context":"outer_fold_1","horizon":30,"loss_tolerance":.4,
                "case_status":"balanced_case","q1_effective_assignment":"Balanced"}
        rows=runner.null_records([1,2],{**common,"policy":runner.LEARNER},"execution_review","no_eligible_inner_beta")
        s=runner.performance_summary(rows)[0]
        self.assertEqual(s["original_n"],2); self.assertEqual(s["case_execReview_n"],2)
        self.assertIsNone(s["breach_count"]); self.assertIsNone(s["median_cagr"])
        for policy in ["annual_then_pending","equal_thirds_annual"]:
            rows+=runner.null_records([1,2],{**common,"policy":policy},"execution_review","synthetic")
        p,s=runner.paired_outputs(rows)
        self.assertTrue(all(not x["common_auto"] for x in p))
        self.assertTrue(all(x["common_auto_n"]==0 for x in s))
        self.assertTrue(all(x["delta_log_growth"] is None for x in p))

    def test_metric_quantiles_pool_paths_not_fold_quantiles(self):
        rows=[]
        for sid,mdd,cagr in [(1,.2,.01),(2,.3,.02),(3,.5,.1)]:
            rows.append({"scope":"outer_oof","policy":runner.LEARNER,"scenario_id":sid,
                "q1_effective_assignment":"Balanced","case_status":"balanced_case","execution_status":"automated",
                "assigned":True,"log_growth":np.log1p(cagr),"cagr":cagr,"mdd":mdd,"breach":mdd>.4,
                "nominal_fees":.1,"trade_fraction":.2,"excess":max(0,mdd-.4)})
        s=runner.performance_summary(rows)[0]
        self.assertEqual(s["median_cagr"],.02); self.assertEqual(s["p90_mdd"],.5)
        self.assertEqual(s["case_auto_n"],3); self.assertEqual(s["breach_count"],1)


if __name__=="__main__": unittest.main()
