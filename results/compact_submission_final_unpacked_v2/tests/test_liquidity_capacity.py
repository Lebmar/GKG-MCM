import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT/'src')]
from diagnostics.liquidity_capacity import bill_liquidation,annual_target_liquidity
from safeinvest.core import execute_target


class LiquidityCapacityTests(unittest.TestCase):
    def test_destination_fee_is_paid_once_and_matches_exact_trade_executor(self):
        x=np.array([[20.,30.,50.],[0,0,100.],[100,0,0.]])
        q=bill_liquidation(x)
        post,fee,moved=execute_target(x,[[0,0,1]],.0025)
        np.testing.assert_allclose(q['available_bills'],post[:,2],atol=1e-12,rtol=0)
        np.testing.assert_allclose(q['nominal_fee'],fee,atol=1e-12,rtol=0)
        np.testing.assert_allclose(q['transferred'],moved,atol=1e-12,rtol=0)
        np.testing.assert_allclose(q['available_bills']+q['nominal_fee'],100)

    def test_zero_fee_and_existing_bills_preserve_all_wealth(self):
        self.assertEqual(bill_liquidation([20,30,50],0)['available_bills'],100.)
        self.assertEqual(bill_liquidation([0,0,100])['nominal_fee'],0.)

    def test_proportions_are_unit_invariant_and_drift_is_not_target_failure(self):
        q=annual_target_liquidity([.65,.05,.3],[[.2,0,0],[-.2,0,0]],floor=.3)
        self.assertLess(q['end_bill_share'][0],.3);self.assertGreater(q['end_bill_share'][1],.3)
        np.testing.assert_array_equal(q['below_target_floor'],[True,False])
        np.testing.assert_allclose(q['hypothetical_available_bill_ratio']+q['hypothetical_liquidation_drag'],1)

    def test_formula_is_not_double_sided_turnover(self):
        q=bill_liquidation([20,30,50])
        self.assertAlmostEqual(q['nominal_fee'],.0025*50/1.0025)
        self.assertLess(q['nominal_fee'],.0025*50)

    def test_liquidation_is_only_a_hypothetical_ratio_not_a_path_deduction(self):
        r=np.array([[.1,0,0.]])
        before=r.copy();annual_target_liquidity([.4,.2,.4],r,floor=.3)
        np.testing.assert_array_equal(r,before)

    def test_invalid_shapes_markets_or_target_floor_rejected(self):
        for x in ([0,0,0],[-1,0,100],[1,2]):
            with self.assertRaises(ValueError):bill_liquidation(x)
        with self.assertRaises(ValueError):annual_target_liquidity([.6,.3,.1],[[0,0,0]],floor=.3)
        with self.assertRaises(ValueError):annual_target_liquidity([.6,.3,.1],[[-1,0,0]],floor=0)


if __name__=='__main__':unittest.main()
