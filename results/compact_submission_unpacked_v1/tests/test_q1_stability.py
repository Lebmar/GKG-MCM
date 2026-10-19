"""Synthetic tests for individual-set diagnostics and unchanged-menu Q1 OOF."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"research"),str(ROOT/"scripts")]
from diagnostics.q1_stability import individual_set,profile_metadata,own_horizon_rows,gamma_objective_band,PROFILES
import diagnose_q1_stability as driver


def bundle(train, *, raw="Balanced",effective="Balanced"):
    product=dict(name="Balanced",candidate_index=1,candidate_original232_index=9,
        weights=[.3,.3,.4],identified_by_profiles=True,reference_cvar90_mdd=.2)
    return dict(refit=dict(training_ids=list(train),selection=dict(selected_gamma=.1),
        raw_fit=dict(menu=[product]),raw_assignments={p["name"]:raw for p in PROFILES},
        effective_assignments={p["name"]:effective for p in PROFILES},
        admission_details=[dict(client=p["name"],reason="accepted" if effective!="Review" else "no_inner_auto_evidence") for p in PROFILES],
        q1_review_details={}),inner_summaries=[])


class Q1StabilityTests(unittest.TestCase):
    def setUp(self):
        self.w=np.array([[.2,.3,.5],[.3,.3,.4],[.6,.3,.1]])
        self.g=np.tile([.04,.039,.05],(6,1));self.d=np.full((6,3),.2)
        self.job=dict(job_id="c1_refit",parent_context_id=1,stage="refit",training_ids=[1,2,3],evaluation_ids=[4,5,6])

    def individual(self,**kw):
        args=dict(L=.4,liquidity_floor=.3,gamma=.1,raw_menu_index=1);args.update(kw)
        return individual_set(self.g,self.d,self.w,[1,2,3],**args)

    def test_fixed_epsilon_is_in_annual_log_growth_and_complete_set(self):
        result,members=self.individual()
        self.assertEqual(result["epsilon_annual_log_growth"],.001)
        self.assertEqual(result["feasible_indices"],[0,1])
        self.assertEqual(result["near_optimal_indices"],[0,1])
        self.assertEqual([r["candidate_local_index"] for r in members],[0,1])
        self.assertAlmostEqual(result["raw_menu_regret_if_feasible"],.001)
        self.assertAlmostEqual(result["total_variation_diameter"],.1)
        self.assertEqual(result["weight_min"],[.2,.3,.4])
        self.assertEqual(result["weight_max"],[.3,.3,.5])

    def test_growth_band_does_not_include_materially_inferior_target(self):
        self.g[:,1]=.0389
        result,_=self.individual()
        self.assertEqual(result["near_optimal_indices"],[0])
        self.assertFalse(result["raw_menu_in_near_optimal_set"])

    def test_feasible_individual_optimum_is_not_a_replacement_joint_menu(self):
        result,members=self.individual(liquidity_floor=0)
        self.assertEqual(result["near_optimal_indices"],[2])
        self.assertEqual(result["raw_menu_index"],1)
        self.assertEqual(members[0]["weights"],[.6,.3,.1])

    def test_only_nominated_training_rows_affect_individual_set(self):
        before,_=self.individual()
        self.g[3:]=np.inf;self.d[3:]=np.nan
        after,_=self.individual()
        self.assertEqual(before,after)

    def test_selected_gamma_unavailable_differs_from_empty_feasible_set(self):
        unavailable,members=self.individual(gamma=None)
        empty,_=self.individual(liquidity_floor=.9)
        self.assertEqual(unavailable["status"],"no_selected_gamma")
        self.assertIsNone(unavailable["feasible_n"])
        self.assertEqual(members,[])
        self.assertEqual(empty["status"],"empty_feasible_set")
        self.assertEqual(empty["feasible_n"],0)
        self.assertEqual(empty["near_optimal_indices"],[])

    def test_gamma_count_granularity_and_strict_event_boundary(self):
        self.d[0,0]=.4+2e-12
        result,_=self.individual()
        self.assertEqual(result["event_count_cap"],0)
        self.assertEqual(result["feasible_indices"],[1])
        self.d[0,0]=.4+5e-13
        result,_=self.individual()
        self.assertEqual(result["feasible_indices"],[0,1])

    def test_final_duplicate_noninteger_ids_and_bad_metrics_rejected(self):
        for ids in ([121],[1,1],[1.5]):
            with self.assertRaises(ValueError):
                individual_set(self.g,self.d,self.w,ids,L=.4,liquidity_floor=.3,gamma=.1)
        self.g[0,0]=np.nan
        with self.assertRaises(ValueError):self.individual()

    def test_gamma_objective_band_is_not_selected_gamma_uniqueness(self):
        b=dict(inner_summaries=[
            dict(eligible=True,raw_auto_path_profile_units=10,raw_auto_unit_mean_log_growth=.03,training_screen_gamma=0.),
            dict(eligible=True,raw_auto_path_profile_units=10,raw_auto_unit_mean_log_growth=.03+5e-13,training_screen_gamma=.025),
            dict(eligible=True,raw_auto_path_profile_units=9,raw_auto_unit_mean_log_growth=.1,training_screen_gamma=.1)])
        self.assertEqual(gamma_objective_band(b),[0.,.025])

    def fields(self):
        return {key:np.full((6,3),value) for key,value in
            [("log_growth",.03),("cagr",.0304545),("mdd",.2),("volatility",.1),
             ("terminal_real",120.),("fees",.5),("trade_fraction",1.)]}

    def test_raw_effective_metadata_and_own_horizon_review_null(self):
        b=bundle([1,2,3],effective="Review");p=PROFILES[0]
        meta=profile_metadata(self.job,b,p)
        self.assertEqual(meta["raw_assignment"],"Balanced")
        self.assertEqual(meta["effective_assignment"],"Review")
        self.assertEqual(meta["horizon"],5)
        rows,refs=own_horizon_rows(self.job,b,p,self.fields(),thirds_index=0)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(not r["assigned"] and r["mdd"] is None and r["log_growth"] is None for r in rows))
        self.assertTrue(all(r["assigned"] and r["horizon"]==5 for r in refs))

    def test_own_horizon_values_are_fixed_product_not_heldout_optimization(self):
        fields=self.fields();fields["log_growth"][3:,2]=99.
        rows,_=own_horizon_rows(self.job,bundle([1,2,3]),PROFILES[2],fields,thirds_index=0)
        self.assertTrue(all(r["candidate_local_index"]==1 and r["log_growth"]==.03 for r in rows))
        self.assertTrue(all(r["horizon"]==30 for r in rows))

    def test_oof_overlap_and_wrong_fit_metadata_rejected(self):
        bad=copy.deepcopy(self.job);bad["evaluation_ids"]=[3,4,5]
        with self.assertRaises(ValueError):own_horizon_rows(bad,bundle([1,2,3]),PROFILES[2],self.fields(),thirds_index=0)
        with self.assertRaises(ValueError):profile_metadata(self.job,bundle([2,3,4]),PROFILES[2])

    def test_summary_zero_coverage_and_matching_reference_denominator(self):
        model,refs=own_horizon_rows(self.job,bundle([1,2,3],effective="Review"),PROFILES[2],self.fields(),thirds_index=0)
        result=driver.matched_profile_summary(model,refs)
        self.assertEqual(result["model"]["financial_n"],0)
        self.assertIsNone(result["model"]["breach_fraction"])
        self.assertEqual(result["reference_all_original"]["financial_n"],3)
        self.assertEqual(result["same_assigned_reference"]["financial_n"],0)
        self.assertIsNone(result["paired_mean_log_growth"])

    def test_cache_values_not_read_without_frozen_release(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"protocol";f=Path(temp)/"freeze"
            p.write_text('{}');f.write_text(json.dumps(dict(execution_ready=False,protocol_sha256=driver.sha(p))))
            with patch.object(driver,"PROTOCOL",p),patch.object(driver,"FREEZE",f),patch.object(driver.np,"load") as loader:
                with self.assertRaises(RuntimeError):driver.authorize()
                loader.assert_not_called()


if __name__=="__main__":unittest.main()
