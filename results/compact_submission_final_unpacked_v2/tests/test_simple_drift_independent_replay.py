"""Synthetic independent fee/action/state replay; no market file is read."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
from experimental_execution.simple_drift_execution import simulate_drift
from experimental_execution.review_pending import simulate_paired
from audit_fan_rollout_execution import reference_actual
from audit_review_pending_execution import independent_metrics


class SimpleDriftIndependentReplayTests(unittest.TestCase):
    def synthetic_market(self):
        rng=np.random.default_rng(870202610181337)
        returns=rng.uniform(-.28,.30,(6,30,3))
        inflation=rng.uniform(-.015,.055,(6,30))
        # Explicit early loss and recovery exercise absorbing pending paths.
        returns[0,0,:]=-.6;returns[0,1,:]=2.
        return returns,inflation

    def test_all_bands_match_separate_bisection_and_physical_state_replay(self):
        returns,inflation=self.synthetic_market()
        target=np.array([.45,.05,.5]);loss=.15
        for delta in (0.,.05,.10):
            arm=simulate_drift(returns,inflation,target,delta,L=loss)
            for i in range(len(returns)):
                def action(t,x,cpi,peak,worst,bad):
                    if t==0:return 'restore'
                    if bad:return 'hold'
                    if delta==0.:return 'restore'
                    tv=sum(abs(x/sum(x)-target))/2
                    return 'restore' if tv>delta else 'hold'
                ref=reference_actual(returns[i],inflation[i],target,loss,action,cost_rate=.0025)
                for field,value in ref.items():
                    with self.subTest(delta=delta,scenario=i,field=field):
                        actual=np.asarray(arm[field][i]);expected=np.asarray(value)
                        if expected.dtype.kind in 'bUiS':np.testing.assert_array_equal(actual,expected)
                        else:np.testing.assert_allclose(actual,expected,atol=3e-12,rtol=3e-13,equal_nan=True)
                metrics=independent_metrics(ref,loss)
                for field in ('log_growth','cagr','mdd','volatility','terminal_real','nominal_fees','nominal_moved','breach','excess'):
                    with self.subTest(delta=delta,scenario=i,metric=field):
                        np.testing.assert_allclose(arm[field][i],metrics[field],atol=3e-12,rtol=3e-13)

    def test_delta_zero_target_and_loss_metadata_match_frozen_pending_B(self):
        returns,inflation=self.synthetic_market();target=[.45,.15,.4]
        arm=simulate_drift(returns,inflation,target,0.,L=.4)
        old=simulate_paired(returns,inflation,target,.4)['pending_hold']
        np.testing.assert_array_equal(arm['targets'],old['targets'][0])
        np.testing.assert_array_equal(arm['loss_tolerances'],old['loss_tolerances'])
        self.assertTrue(np.all(arm['pending_at_decision'][0,1:]))
        self.assertEqual(arm['first_crossing'][0],1)


if __name__=='__main__':unittest.main()
