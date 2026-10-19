import numpy as np
import unittest
from experimental_execution.calendar_state_diagnostic import (
    market_log_variables, diagnostic_arrays, fit_calendar_state,
    predict_calendar_state, per_path_losses, validate_folds)


class CalendarStateDiagnosticTests(unittest.TestCase):
    def test_original_calendar_and_remaining_horizon(self):
        z=np.arange(2*30*4,dtype=float).reshape(2,30,4)
        x,y=diagnostic_arrays(z)
        np.testing.assert_array_equal(x[:,0],z[:,0])
        np.testing.assert_array_equal(y['next_year'][:,0],z[:,1])
        np.testing.assert_array_equal(y['remaining_to_original30_mean_annual'][:,-1],z[:,-1])
        np.testing.assert_allclose(y['remaining_to_original30_mean_annual'][:,0],z[:,1:].mean(axis=1))


    def test_real_logs_share_actual_inflation_and_positive_gross(self):
        r=np.full((2,30,3),.1); p=np.full((2,30),.03)
        z=market_log_variables(r,p)
        np.testing.assert_allclose(z[...,:3],np.log(1.1/1.03))
        np.testing.assert_allclose(z[...,3],np.log(1.03))
        r[0,0,0]=-1
        with self.assertRaises(ValueError):market_log_variables(r,p)


    def test_fixed_ridge_train_only_prediction_and_calendar_mean(self):
        rng=np.random.default_rng(234)
        x=rng.normal(size=(96,29,4)); calendar=np.arange(29)[None,:,None]*.1
        y=2*x+calendar
        m=fit_calendar_state(x,y)
        b,s=predict_calendar_state(m,x)
        np.testing.assert_allclose(b.mean(axis=0),y.mean(axis=0))
        assert np.mean((y-s)**2)<.3*np.mean((y-b)**2)
        assert m['penalty']==1 and m['training_paths']==96
        # Evaluation does not pass any future target or modify the frozen fit.
        before={k:v.copy() for k,v in m.items() if isinstance(v,np.ndarray)}
        predict_calendar_state(m,np.full((24,29,4),1e3))
        for k,v in before.items():np.testing.assert_array_equal(v,m[k])


    def test_calendar_centering_does_not_learn_calendar_as_state(self):
        x=np.broadcast_to(np.arange(29)[None,:,None],(8,29,4)).copy()
        y=3*x+7
        m=fit_calendar_state(x,y);b,s=predict_calendar_state(m,x+100)
        np.testing.assert_array_equal(m['standardized_coefficients'],np.zeros((4,4)))
        np.testing.assert_array_equal(s,b)
        np.testing.assert_array_equal(m['x_scale'],np.ones(4))


    def test_collapse_years_to_original_path_losses(self):
        y=np.zeros((2,29,4));b=np.ones_like(y);s=np.zeros_like(y);s[0,0]=29
        a=per_path_losses(y,b,s,np.ones(4)*2)
        assert a['state_mse'].shape==(2,4)
        np.testing.assert_array_equal(a['state_mse'][0],np.full(4,29))
        np.testing.assert_array_equal(a['baseline_standardized_mse'],np.full((2,4),.25))


    def test_complete_path_folds_and_no_final(self):
        ids=list(range(1,121));folds=[]
        for k in range(5):
            e=ids[k*24:(k+1)*24];folds.append({'training_ids':sorted(set(ids)-set(e)),'evaluation_ids':e})
        validate_folds(folds,ids)
        folds[0]['evaluation_ids'][0]=121
        with self.assertRaises(ValueError):validate_folds(folds,ids)


    def test_reject_invalid_penalties(self):
        for penalty in [True,0,-1,float('inf'),float('nan')]:
            with self.subTest(penalty=penalty), self.assertRaises(ValueError):
                fit_calendar_state(np.ones((2,29,4)),np.ones((2,29,4)),penalty=penalty)
