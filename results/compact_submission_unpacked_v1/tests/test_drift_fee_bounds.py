"""Analytic transfer-fee bounds and a drift-only benefit counterexample."""
import unittest
import numpy as np
from safeinvest.core import execute_target


class DriftFeeBounds(unittest.TestCase):
    def test_destination_transfer_bounds_over_simplexes(self):
        rng=np.random.default_rng(87020261018)
        weights=rng.dirichlet(np.ones(3),size=2000)
        targets=rng.dirichlet(np.ones(3),size=2000)
        wealth=np.exp(rng.uniform(-5,8,2000))
        drift=.5*np.abs(weights-targets).sum(axis=1)
        for rate in (0.,.0025,.1,.9):
            _,fees,moved=execute_target(wealth[:,None]*weights,targets,rate)
            transfer=moved/wealth
            self.assertTrue(np.all(transfer>=drift/(1+rate)-1e-12))
            self.assertTrue(np.all(transfer<=drift+1e-12))
            np.testing.assert_allclose(fees/wealth,rate*transfer,atol=1e-12,rtol=0)

    def test_fee_bound_has_units_of_basis_points(self):
        rate=.0025
        for band,upper_bp in ((.05,1.25),(.1,2.5)):
            self.assertAlmostEqual(rate*band*10000,upper_bp)
            self.assertLess(rate*band/(1+rate)*10000,upper_bp)

    def test_zero_drift_has_zero_actual_transfer(self):
        weights=np.array([[.4,.2,.4],[0.,0.,1.],[1.,0.,0.]])
        _,fees,moved=execute_target(100*weights,weights,.0025)
        np.testing.assert_array_equal(fees,np.zeros(3))
        np.testing.assert_array_equal(moved,np.zeros(3))

    def test_same_drift_and_fee_can_have_opposite_net_benefit(self):
        current=np.array([60.,30.,10.]);target=np.array([.5,.3,.2])
        post,fee,_=execute_target(current,target,.0025)
        self.assertGreater(fee,0.)
        self.assertAlmostEqual(.5*np.abs(current/100-target).sum(),.1)
        benefits=[]
        for gross in (np.array([1.2,1.,1.]),np.array([.8,1.,1.])):
            benefits.append(np.log(post@gross/(current@gross)))
        self.assertLess(benefits[0],0.)
        self.assertGreater(benefits[1],0.)


if __name__=='__main__':unittest.main()
