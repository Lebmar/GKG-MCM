"""Independent audit references, using synthetic paths and closed forms only."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('audit_fixed_state_synthetic',ROOT/'scripts/audit_fixed_state_forecast_diagnostic.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class IndependentReferenceTests(unittest.TestCase):
    def test_closed_forms_original_clock_history_and_own_pending(self):
        self.assertEqual(len(m.self_checks()),4)

    def test_suffix_only_one_year_pays_current_fee_and_no_terminal_liquidation(self):
        r=np.zeros((3,3));r[2,0]=.004
        b=m.independent_branch([0.,0.,100.],1.,100.,0.,[1.,0.,0.],.4,r,np.zeros(3),
            year_index=2,original_horizon=3,current_action='restore')
        wealth=100/1.0025*1.004
        self.assertAlmostEqual(b['terminal_real'],wealth,places=12)
        self.assertEqual(len(b['fees']),1);self.assertEqual(len(b['real_wealth']),2)
        self.assertAlmostEqual(b['nominal_fees'],100*.0025/1.0025,places=13)
        self.assertEqual(b['actions'].tolist(),['restore'])
        self.assertEqual(b['original_horizon'],3)

    def test_current_hold_then_reference_is_not_permanent_hold_or_reset_worst(self):
        r=np.zeros((2,3))
        b=m.independent_branch([60.,0.,40.],1.,100.,.19,[.5,0.,.5],.4,r,np.zeros(2),
            year_index=0,original_horizon=2,current_action='hold',cost_rate=0.)
        self.assertEqual(b['actions'].tolist(),['hold','restore'])
        self.assertEqual(b['running_mdd'].tolist(),[.19,.19,.19])
        np.testing.assert_array_equal(b['posttrade_holdings'],[[60.,0.,40.],[50.,0.,50.]])

    def test_summary_retains_decision_state_and_unique_path_denominators_separately(self):
        common=dict(analysis_scope='outer_oof',context='c',job_id='j',group='g',scenario_id=1,
            in_training_fan=False,chosen_action='hold',predicted_growth_sign=0,realized_suffix_growth_sign=1,
            growth_sign_agreement=False,score_action_agreement=True,realized_hold_event=False,realized_restore_event=True,
            delta_log_residual=.1,score_residual=-.2,two_branch_realized_score_shortfall=.3,
            predicted_hold_bad_fraction=.1,predicted_restore_bad_fraction=.2)
        rows=[dict(common,year=t) for t in (2,16,30)]
        s=next(x for x in m.independent_summary(rows) if x['level']=='analysis_scope')
        self.assertEqual((s['active_state_n'],s['unique_scenario_n'],s['unique_job_n'],s['unique_group_n']),(3,1,1,1))
        self.assertEqual(s['realized_restore_event_n'],3)
        self.assertAlmostEqual(s['mean_abs_score_residual'],.2)

    def test_everbreached_or_bad_original_clock_is_rejected_not_reset(self):
        args=([0.,0.,100.],1.,100.,0.,[.3,.3,.4],.4,np.zeros((2,3)),np.zeros(2))
        with self.assertRaises(ValueError):m.independent_branch(*args,year_index=0,original_horizon=2,current_action='hold',ever_breached=True)
        with self.assertRaises(ValueError):m.independent_branch(*args,year_index=2,original_horizon=2,current_action='hold')


if __name__=='__main__':unittest.main()
