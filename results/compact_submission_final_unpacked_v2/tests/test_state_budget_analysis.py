"""Pure/synthetic saved-analysis checks: no market files and no policy runs."""
from pathlib import Path
import copy
import csv
import gzip
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import analyze_state_budget_rebuild as analysis


def path_row(sid, *, assigned=True, mdd=.2, growth=.03, breach=False, reason=None):
    row = dict(scenario_id=sid,assigned=assigned,review_reason=reason)
    if not assigned:
        return dict(row,**dict.fromkeys(analysis.FINANCIAL))
    return dict(row,log_growth=growth,cagr=math.expm1(growth),mdd=mdd,volatility=.1,terminal_real=110.,
        nominal_fees=.4,trade_fraction=.2,breach=breach,first_crossing=None,automatic_fraction=.8,
        protection_trade_fraction=.1,support_review=False)


class StateBudgetAnalysisTests(unittest.TestCase):
    def test_quantile_median_and_fractional_tail(self):
        self.assertEqual(analysis.median([4,1,3,2]),2.5)
        self.assertEqual(analysis.quantile(list(range(1,26)),.1),3)
        self.assertEqual(analysis.quantile(list(range(1,26)),.9),23)
        self.assertEqual(analysis.cvar90(list(range(1,26))),(25+24+.5*23)/2.5)
        self.assertIsNone(analysis.cvar90([]))

    def test_comparison_retains_all_references_and_exact_invested_cohort(self):
        model = [path_row(1,mdd=.1),path_row(2,assigned=False,reason="not_balanced_case"),path_row(3,mdd=.3)]
        ref = [path_row(1,mdd=.15),path_row(2,mdd=.9),path_row(3,mdd=.25)]
        result,pairs = analysis.compare_cohorts(model,ref)
        self.assertEqual(result["reference_all_original"]["financial_n"],3)
        self.assertEqual(result["matched_reference"]["financial_n"],2)
        self.assertAlmostEqual(result["matched_reference"]["mean_mdd"],.2)
        self.assertEqual(result["model_original"]["not_balanced_n"],1)
        self.assertEqual(result["model_original"]["initial_review_n"],0)
        self.assertEqual([r["scenario_id"] for r in pairs],[1,3])

    def test_zero_coverage_finance_null_not_zero(self):
        result,pairs = analysis.compare_cohorts([path_row(1,assigned=False)], [path_row(1)])
        self.assertEqual(result["model_original"]["initial_coverage"],0)
        self.assertIsNone(result["matched_model"]["breach_fraction"])
        self.assertIsNone(result["paired_differences"]["log_growth"]["mean"])
        self.assertEqual(pairs,[])
        bad = path_row(1,assigned=False);bad["mdd"] = 0.
        with self.assertRaises(ValueError):analysis.cohort_summary([bad])

    def test_q1_review_nonbalanced_and_execution_review_are_separate(self):
        rows = [path_row(1,assigned=False,reason="q1_review"),
                path_row(2,assigned=False,reason="not_balanced_case"),
                path_row(3,assigned=False,reason="preinvestment_review")]
        result = analysis.cohort_summary(rows)
        self.assertEqual(result["initial_noninvested_n"],3)
        self.assertEqual(result["initial_review_n"],2)
        self.assertEqual(result["not_balanced_n"],1)
        self.assertEqual(result["q1_review_n"],1)
        self.assertEqual(result["execution_review_n"],1)

    def test_all_four_event_partitions_and_paired_signs(self):
        model = [path_row(i+1,breach=i in (1,3),growth=.04) for i in range(4)]
        ref = [path_row(i+1,breach=i in (2,3),growth=.03) for i in range(4)]
        result,_ = analysis.compare_cohorts(model,ref)
        self.assertEqual(result["event_id_partition"],dict(neither=[1],model_only=[2],reference_only=[3],both=[4]))
        self.assertEqual(result["paired_differences"]["log_growth"]["positive_n"],4)

    def test_identity_and_unavailable_reference_fail_closed(self):
        for rows in ([path_row(1),path_row(1)],[path_row(121)],[path_row(1.5)]):
            with self.assertRaises(ValueError):analysis.ids_map(rows)
        with self.assertRaises(ValueError):analysis.compare_cohorts([path_row(1)],[path_row(2)])
        with self.assertRaises(ValueError):analysis.compare_cohorts([path_row(1)],[path_row(1,assigned=False)])

    def test_reconstruct_joint_market_under_costs_and_inflation(self):
        post = np.array([[[32.,33.,34.],[40.,35.,30.]]])
        nominal = np.array([[[1.1,.9,1.01],[.8,1.05,1.02]]])
        pi = np.array([[.03,.08]])
        holdings = np.concatenate((np.array([[[0.,0.,100.]]]),post*nominal),axis=1)
        cp = np.column_stack((np.ones(1),np.cumprod(1+pi,axis=1)))
        gross,recovered_pi = analysis.reconstruct_joint_market(dict(posttrade_holdings=post,holdings=holdings,cpi=cp))
        np.testing.assert_allclose(gross,nominal/(1+pi[:,:,None]),atol=3e-16,rtol=0)
        np.testing.assert_allclose(recovered_pi,pi,atol=1e-16,rtol=0)
        bad = post.copy();bad[0,0,0] = 0
        with self.assertRaises(ValueError):
            analysis.reconstruct_joint_market(dict(posttrade_holdings=bad,holdings=holdings,cpi=cp))

    def market(self):
        market = {
            1:np.array([[1.1,1.02,1.01],[1.2,1.04,1.02]]),
            2:np.array([[.9,1.04,.99],[.8,1.08,1.01]]),
            3:np.array([[1.02,.98,1.],[1.05,.96,1.]]),
            4:np.array([[1.08,1.01,1.005],[1.12,1.02,1.01]]),
        }
        pi = {1:np.array([.02,.03]),2:np.array([.05,.06]),3:np.array([.01,.02]),4:np.array([.025,.04])}
        return market,pi

    def test_market_forecasts_have_t0_uniform_and_independent_gaussian_oracle(self):
        market,pi = self.market()
        rows = analysis.forecast_market_rows([1,2,3],[4],market,pi,job_id="synthetic")
        self.assertEqual(len(rows),2)
        self.assertIsNone(rows[0]["shrink_conditional_standardized_mse"])
        training = [list(market[s][0]-1)+[pi[s][0]] for s in (1,2,3)]
        actual = list(market[4][0]-1)+[pi[4][0]]
        means = [math.fsum(r[c] for r in training)/3 for c in range(4)]
        scales = [max(math.sqrt(math.fsum((r[c]-means[c])**2 for r in training)/3),1e-8) for c in range(4)]
        kernels = [math.exp(-.5*math.fsum(((actual[c]-r[c])/scales[c])**2 for c in range(4))/4) for r in training]
        weights = [.5/3+.5*k/math.fsum(kernels) for k in kernels]
        expected = math.fsum(w*(market[s][1,0]-1) for w,s in zip(weights,(1,2,3)))
        self.assertAlmostEqual(rows[1]["shrink_conditional_pred_equity"],expected,places=15)
        center = math.fsum(market[s][1,0]-1 for s in (1,2,3))/3
        scale = math.sqrt(math.fsum((market[s][1,0]-1-center)**2 for s in (1,2,3))/3)
        self.assertAlmostEqual(rows[1]["training_scale_equity"],scale,places=15)

    def test_market_diagnostic_requires_true_development_oof(self):
        market,pi = self.market()
        for train,actual in (([1,2,3],[3]),([1,1,2],[4]),([1,2,121],[4])):
            with self.assertRaises(ValueError):analysis.forecast_market_rows(train,actual,market,pi,job_id="x")

    def test_scenario_equal_average_is_not_annual_row_weighting(self):
        rows = [dict(scenario_id=1,error=0.)]*3+[dict(scenario_id=2,error=4.)]
        self.assertEqual(analysis.scenario_equal(rows,"error"),2.)
        self.assertIsNone(analysis.scenario_equal([],"error"))

    def policy_arm(self):
        gross = np.array([[[1.1,1.,.98],[.9,1.,1.01]]])
        x0 = np.array([0.,0.,100.]);p0 = np.array([50.,0.,49.9]);x1=p0*gross[0,0];x2=x1*gross[0,1]
        wealth = np.array([[100.,x1.sum(),x2.sum()]])
        peak = np.maximum.accumulate(wealth,axis=1)
        mdd = np.maximum.accumulate(1-wealth/peak,axis=1)
        base = dict(gate_evaluated=np.array([[True,True]]),selected_index=np.array([[1,-1]]),
            forecast_member_count=np.array([[1,5]]),cpi=np.ones((1,3)),holdings=np.array([[x0,x1,x2]]),
            posttrade_holdings=np.array([[p0,x1]]),realized_real_gross=gross,real_wealth=wealth,
            gate_reason=np.array([["mandatory_initial_full_target_support_pass","no_feasible_support_action_pending"]]),
            trades=np.array([[True,False]]),budget_floor=np.array([[80.,90.]]),
            selected_score=np.array([[.005,0.]]),
            support_feasible=np.array([[True,False]]),running_mdd=mdd,
            candidate_min_support_slack=np.full((1,2,13),np.nan),
            candidate_member_log_growth=np.full((1,2,13,5),np.nan),
            candidate_member_advantage=np.full((1,2,13,5),np.nan),
            candidate_predicted_next_breach=np.full((1,2,13,5),np.nan))
        for t,k,members in ((0,1,1),(1,0,5)):
            base["candidate_min_support_slack"][0,t,k] = 20.
            base["candidate_member_log_growth"][0,t,k,:members] = .01
            base["candidate_member_advantage"][0,t,k,:members] = .005
            base["candidate_predicted_next_breach"][0,t,k,:members] = .2
        return base

    def test_policy_calibration_is_after_fee_paired_and_referral_is_separate(self):
        arm = self.policy_arm()
        rows = analysis.policy_forecast_rows(arm,[4],job_id="x")
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0]["phase"],"inception")
        self.assertAlmostEqual(rows[0]["realized_paired_benefit"],math.log(arm["real_wealth"][0,1]/98.))
        self.assertEqual({r["phase"] for r in rows[1:]},{"referral"})
        self.assertTrue(all(r["directional_support_miss"] for r in rows[1:]))
        self.assertFalse(any(r["supported_safe_floor_failure"] for r in rows[1:]))
        self.assertAlmostEqual(rows[0]["brier"],.04)

    def test_pending_has_no_new_prediction_and_corrupt_actual_trace_rejected(self):
        arm = self.policy_arm();arm["gate_evaluated"][:]=False
        self.assertEqual(analysis.policy_forecast_rows(arm,[4],job_id="x"),[])
        arm = self.policy_arm();arm["realized_real_gross"][0,0,0] += .1
        with self.assertRaises(ValueError):analysis.policy_forecast_rows(arm,[4],job_id="x")

    def test_support_tolerance_scales_with_real_wealth_below_one(self):
        arm = self.policy_arm()
        arm["gate_evaluated"][0,0] = False
        factor = .5/arm["holdings"][0,1].sum()
        arm["holdings"][0,1:] *= factor
        arm["posttrade_holdings"][0,1] *= factor
        arm["real_wealth"][0,1:] *= factor
        arm["running_mdd"][0,1:] = 1-arm["real_wealth"][0,1:]/100.
        actual_next = arm["real_wealth"][0,2]
        arm["budget_floor"][0,1] = actual_next+7.5e-13
        arm["candidate_min_support_slack"][0,1,0] = 0.
        arm["selected_index"][0,1] = 0
        arm["support_feasible"][0,1] = True
        arm["gate_reason"][0,1] = "support_safe_hold"
        rows = analysis.policy_forecast_rows(arm,[4],job_id="tiny-real-wealth",L=.9999)
        self.assertEqual(len(rows),5)
        self.assertTrue(all(r["directional_support_miss"] for r in rows))
        self.assertTrue(all(r["supported_safe_floor_failure"] for r in rows))

    def test_prediction_summary_empty_and_fixed_bins(self):
        empty = analysis.prediction_summaries([],[])
        self.assertTrue(all(r["brier"] is None for r in empty["policy"]))
        rows = analysis.policy_forecast_rows(self.policy_arm(),[4],job_id="x")
        summary = analysis.prediction_summaries([],rows)
        bins = [r for r in summary["reliability_bins"] if r["forecast"]=="uniform" and r["phase"]=="inception"]
        self.assertEqual([r["rows"] for r in bins],[1,0,0])
        self.assertIsNone(bins[1]["scenario_equal_actual"])
        decision = [r for r in summary["decision"] if r["phase"] == "referral"]
        self.assertEqual(decision[0]["rows"],1)

    def test_analysis_not_ready_stops_before_numerical_output_read(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp);plan=folder/"plan.json";freeze=folder/"freeze.json"
            plan.write_text('{}');freeze.write_text(json.dumps(dict(execution_ready=False,protocol_sha256=analysis.sha(plan))))
            with patch.object(analysis,"PLAN",plan),patch.object(analysis,"FREEZE",freeze),patch.object(analysis.np,"load") as loader:
                with self.assertRaises(RuntimeError):analysis.run(folder/"never-read",folder/"never-write")
                loader.assert_not_called()
            self.assertFalse((folder/"never-write").exists())

    def test_zero_financial_csv_still_has_declared_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"empty.csv.gz"
            analysis.write_csv(p,[],analysis.POLICY_FIELDS)
            with gzip.open(p,"rt",newline="") as handle:
                table = csv.DictReader(handle)
                self.assertEqual(table.fieldnames,list(analysis.POLICY_FIELDS))
                self.assertEqual(list(table),[])


if __name__ == "__main__":unittest.main()
