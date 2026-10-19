"""Independent saved-array/summary/NULL boundary helpers; no artifact I/O."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_final_evaluation_development_gate as a

class DevelopmentGateAuditPureTests(unittest.TestCase):
    def test_inverted_CDF_quantile_uses_original_empirical_atom(self):
        self.assertEqual(a.empirical_quantile(range(1,25),.1),3)
        self.assertEqual(a.empirical_quantile(range(1,25),.9),22)
    def test_even_median_differs_from_inverse_CDF_p50(self):
        self.assertEqual(a.standard_median([1,2,4,8]),3)
        self.assertEqual(a.empirical_quantile([1,2,4,8],.5),2)
    def test_fractional_CVaR_includes_boundary_mass(self):
        self.assertAlmostEqual(a.fractional_cvar90(range(1,25)),(24+23+.4*22)/2.4)
    def test_inventory_rejects_silent_missing_Qfield(self):
        with self.assertRaisesRegex(ValueError,'inventory'):
            a.exact_inventory({'wealth':np.array([100.])},{'wealth':np.array([100.]),'Q':np.array([np.nan])},['wealth','Q'])
    def test_exact_NaNmask_is_not_zero_fill(self):
        with self.assertRaisesRegex(ValueError,'NULLmask'):
            a.exact_inventory({'Q':np.array([0.])},{'Q':np.array([np.nan])},['Q'])
    def test_exact_dtype_not_tolerance_conversion(self):
        with self.assertRaisesRegex(ValueError,'dtype'):
            a.exact_inventory({'x':np.array([1.],dtype=np.float32)},{'x':np.array([1.],dtype=np.float64)},['x'])
    def test_rejected_financials_NULL_notzero(self):
        row={key:None for key in a.FIN};a.require_financial_NULL(row)
        row['nominal_fees']=0
        with self.assertRaisesRegex(ValueError,'NULL'):a.require_financial_NULL(row)
    def test_excess_numerical_equality_zero_and_original_H(self):
        arm=dict(real_wealth=np.array([[100.,60.-5e-11]]),fees=np.zeros((1,1)),moved=np.zeros((1,1)),
            trades=np.zeros((1,1),bool),target_execution=np.ones((1,1),bool),weights=np.array([[[0.,0.,1.],[0.,0.,1.]]]))
        p=a.independent_path(arm,0,.4)
        self.assertFalse(p['breach']);self.assertEqual(p['excess'],0)
        self.assertEqual(p['volatility'],0);self.assertEqual(p['target_execution_fraction'],1)
    def test_trade_fraction_uses_physical_transfer(self):
        arm=dict(real_wealth=np.array([[100.,100.,100.]]),fees=np.zeros((1,2)),moved=np.zeros((1,2)),
            trades=np.zeros((1,2),bool),target_execution=np.ones((1,2),bool),weights=np.array([[[0.,0.,1.]]*3]))
        p=a.independent_path(arm,0,.4)
        self.assertEqual(p['trade_fraction'],0);self.assertEqual(p['target_execution_fraction'],1)
    def test_nonfinite_quantile_is_rejected(self):
        with self.assertRaises(ValueError):a.empirical_quantile([0,np.nan],.1)

if __name__=='__main__':unittest.main()
