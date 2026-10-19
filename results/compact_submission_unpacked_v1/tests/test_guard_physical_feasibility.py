"""Constructed LP geometry/certificate tests, no supplied market arrays."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'src')]
from diagnostics import guard_physical_feasibility as m
from safeinvest.core import execute_target


class GuardPhysicalFeasibilityTests(unittest.TestCase):
    def test_eight_closed_patterns_cover_exact_executor_transitions(self):
        x=np.array([35.,15.,50.]);rate=.0025
        for target in ([0,0,1],[1,0,0],[.2,.6,.2],[.35,.15,.5]):
            y,fee,_=execute_target(x,target,rate)
            covered=[]
            for mask in m.BUY_PATTERNS:
                lp=m.build_pattern_lp(x,1,np.ones((2,3)),cost_rate=rate,buy_mask=mask)
                q=np.r_[y/x.sum(),0.]
                if np.all(q>=lp['lower']-1e-13) and np.all(q<=lp['upper']+1e-13):
                    covered.append(mask)
                    np.testing.assert_allclose(lp['A_eq']@q,lp['b_eq'],atol=2e-15,rtol=0)
            self.assertTrue(covered)
            self.assertAlmostEqual(y.sum()+fee,x.sum(),places=12)
        self.assertEqual(len(set(m.BUY_PATTERNS)),8)
        # Hold is on the closed boundary of every region, so a solver's
        # infeasible status for one region is missing numerical evidence.
        for mask in m.BUY_PATTERNS:
            lp=m.build_pattern_lp(x,1,np.ones((2,3)),cost_rate=rate,buy_mask=mask)
            q=np.r_[x/x.sum(),1.]
            self.assertTrue(np.all(q>=lp['lower']) and np.all(q<=lp['upper']))
            np.testing.assert_allclose(lp['A_eq']@q,lp['b_eq'],atol=2e-15,rtol=0)

    def test_uniform_bad_support_has_global_shortfall_with_all_dual_witnesses(self):
        q=m.solve_full_simplex([45,5,50],1,np.full((3,3),.5),budget_floor=60)
        self.assertEqual(q['classification'],'all_eight_numerical_optima_below_budget')
        self.assertEqual(q['passed_certificate_count'],8)
        self.assertAlmostEqual(q['best_worst_next_real'],50.,places=12)
        self.assertFalse(q['exact_arithmetic_impossibility_proof'])
        self.assertTrue(all(max(p['residuals'].values())<=m.CERTIFICATE_TOL for p in q['patterns']))

    def test_hedging_can_use_zero_asset_simplex_boundary(self):
        support=np.array([[2.,.2,.1],[.2,2.,.1]])
        q=m.solve_full_simplex([0,0,100],1,support,budget_floor=105,cost_rate=0)
        self.assertEqual(q['classification'],'numerical_physical_feasible_witness_above_budget')
        self.assertAlmostEqual(q['best_worst_next_real'],110.,places=11)
        np.testing.assert_allclose(q['best_posttrade_holdings'],[50,50,0],atol=2e-12,rtol=0)
        self.assertFalse(q['account_action_changed'])

    def test_full_bill_endpoint_fee_matches_closed_form_once(self):
        q=m.solve_full_simplex([100,0,0],1,[[.1,.1,1.]],budget_floor=90)
        self.assertAlmostEqual(q['best_worst_next_real'],100/1.0025,places=11)
        y=q['best_posttrade_holdings'];fee=q['best_nominal_fee']
        self.assertAlmostEqual(y.sum()+fee,100.,places=11)
        self.assertAlmostEqual(fee,.0025*np.maximum(y-[100,0,0],0).sum(),places=12)

    def test_fees_can_remove_apparent_floor_recovery(self):
        free=m.solve_full_simplex([100,0,0],1,[[1.,1.,1.001]],budget_floor=100.05,cost_rate=0)
        costly=m.solve_full_simplex([100,0,0],1,[[1.,1.,1.001]],budget_floor=100.05,cost_rate=.99)
        self.assertEqual(free['classification'],'numerical_physical_feasible_witness_above_budget')
        self.assertEqual(costly['classification'],'all_eight_numerical_optima_below_budget')
        self.assertAlmostEqual(costly['best_worst_next_real'],100.,places=10)

    def test_currency_and_cpi_scaling_do_not_change_physical_decision(self):
        x=np.array([35.,15.,50.]);support=np.array([[.5,.9,1.],[1.4,1.05,.99]])
        q=m.solve_full_simplex(x,1,support,budget_floor=80)
        other=m.solve_full_simplex(17*x,17,support,budget_floor=80)
        self.assertEqual(q['classification'],other['classification'])
        self.assertAlmostEqual(q['best_worst_next_real'],other['best_worst_next_real'],places=12)
        np.testing.assert_allclose(17*q['best_posttrade_holdings'],other['best_posttrade_holdings'],atol=5e-12,rtol=0)
        self.assertAlmostEqual(17*q['best_nominal_fee'],other['best_nominal_fee'],places=11)

    def test_equality_window_remains_explicitly_unresolved(self):
        for floor in (100.,100.+.5e-6,100.-.5e-6):
            q=m.solve_full_simplex([45,5,50],1,[[1,1,1]],budget_floor=floor)
            self.assertEqual(q['classification'],'numerical_boundary_unresolved')
        q=m.solve_full_simplex([45,5,50],1,[[1,1,1]],budget_floor=100.+2e-6)
        self.assertEqual(q['classification'],'all_eight_numerical_optima_below_budget')

    def test_family_infeasible_need_not_be_full_simplex_infeasible(self):
        x=np.array([45.,5.,50.]);w=np.array([.45,.05,.5])
        support=np.array([[.2,.4,.95],[1.2,1.03,1.01],[1.2,1.03,1.01]])
        # Every endpoint and hold is below90 on the same first row, so each
        # of the four affine segments is below90 there, regardless of q.
        for a in (1,.75,.5,.25):
            y,_,_=execute_target(x,a*w+(1-a)*np.array([0,0,1]))
            self.assertLess(support[0]@y,90.)
        self.assertLess(support[0]@x,90.)
        q=m.solve_full_simplex(x,1,support,budget_floor=90)
        self.assertEqual(q['classification'],'numerical_physical_feasible_witness_above_budget')

    def test_success_flag_does_not_hide_bad_primal_or_true_funding(self):
        lp=m.build_pattern_lp([45,5,50],1,[[1,1,1]],buy_mask=[True,False,False])
        res=m.linprog(lp['c'],A_ub=lp['A_ub'],b_ub=lp['b_ub'],A_eq=lp['A_eq'],b_eq=lp['b_eq'],
            bounds=list(zip(lp['lower'],lp['upper'])),method='highs',options=m.SOLVER_OPTIONS)
        self.assertTrue(m.check_solution(lp,res)['certificate_passed'])
        bad=copy.deepcopy(res);bad.x[0]+=.001
        certificate=m.check_solution(lp,bad)
        self.assertFalse(certificate['certificate_passed']);self.assertTrue(bad.success)
        self.assertGreater(certificate['residuals']['true_self_finance_residual_abs'],m.CERTIFICATE_TOL)

    def test_wrong_dual_marginal_is_rejected_even_when_primal_unchanged(self):
        lp=m.build_pattern_lp([100,0,0],1,[[.1,.1,1]],buy_mask=[False,False,True])
        res=m.linprog(lp['c'],A_ub=lp['A_ub'],b_ub=lp['b_ub'],A_eq=lp['A_eq'],b_eq=lp['b_eq'],
            bounds=list(zip(lp['lower'],lp['upper'])),method='highs',options=m.SOLVER_OPTIONS)
        bad=copy.deepcopy(res);bad.ineqlin.marginals[0]-=.01
        certificate=m.check_solution(lp,bad)
        self.assertFalse(certificate['certificate_passed'])
        self.assertGreater(certificate['residuals']['dual_stationarity_max_abs'],m.CERTIFICATE_TOL)

    def test_one_nonoptimal_pattern_blocks_global_shortfall_claim(self):
        original=m.linprog;calls=[]
        def incomplete(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(1)
            if len(calls)==1:result.status=1;result.success=False;result.message='synthetic limit'
            return result
        with mock.patch.object(m,'linprog',side_effect=incomplete):
            q=m.solve_full_simplex([45,5,50],1,[[.5,.5,.5]],budget_floor=60)
        self.assertEqual(q['passed_certificate_count'],7)
        self.assertEqual(q['classification'],'incomplete_solver_or_certificate_evidence')
        self.assertFalse(q['all_eight_numerically_resolved'])

    def test_all_solver_errors_are_retained_without_impossibility_claim(self):
        with mock.patch.object(m,'linprog',side_effect=ArithmeticError('synthetic solver failure')):
            q=m.solve_full_simplex([45,5,50],1,[[.5,.5,.5]],budget_floor=60)
        self.assertEqual(q['classification'],'solver_or_certificate_unresolved')
        self.assertEqual(len(q['patterns']),8)
        self.assertTrue(all(p['error_type']=='ArithmeticError' for p in q['patterns']))

    def test_input_support_and_holdings_are_not_mutated_or_given_actual_future(self):
        x=np.array([45.,5.,50.]);a=np.ones((3,3));before=(x.copy(),a.copy())
        m.solve_full_simplex(x,1,a,budget_floor=90)
        np.testing.assert_array_equal(x,before[0]);np.testing.assert_array_equal(a,before[1])
        with self.assertRaises(TypeError):m.solve_full_simplex(x,1,a,budget_floor=90,actual_future=[0,0,0])
        for args,kwargs in [(([0,0,0],1,a),{}),((x,0,a),{}),(([1e-308,0,0],1e308,a),{}),
                            ((x,1,np.zeros((3,3))),{}),((x,1,a),{'cost_rate':1})]:
            with self.assertRaises(ValueError):m.solve_full_simplex(*args,budget_floor=90,**kwargs)


if __name__=='__main__':unittest.main()
