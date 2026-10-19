"""Constructed mathematical/source tests; no raw data or producer imports."""
import ast
import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_cushion as a

TARGET=np.array([.45,.05,.5])


def decision(real=80.,support=None,initial=False):
    s=np.ones((3,3)) if support is None else np.asarray(support)
    return a.scalar_decision([real,0.,0.],1.,100.,1-real/100,s,s,np.full((1,3),1/3),
        target=TARGET,release=.5,rule='uniform',initial=initial)


def small_arm():
    h=3;gross=np.tile(np.array([[1.,1.,1.],[.2,.75,.75],[1.,1.,1.]])[None],(3,1,1))
    ids=[1,2,3];x=np.array([100.,0.,0.]);results=[]
    for t in range(2):
        weights=a.scalar_forecasts(ids) if t==0 else a.scalar_forecasts(ids,np.zeros((3,4)),np.zeros(4))
        results.append(a.scalar_decision(x,1.,100.,0.,gross[:,t],gross[:,t],weights,target=TARGET,release=.5,rule='uniform',initial=t==0))
    arm={}
    for key,value in results[0].items():
        if key in ('status','real_wealth'):continue
        name={'fee':'fees','action':'actions'}.get(key,key);value=np.asarray(value);shape=value.shape
        if key.startswith(('candidate_member_','mitigation_candidate_member_')) or key=='candidate_predicted_next_breach':shape=shape[:-1]+(5,)
        kind=value.dtype.kind;dtype='U64' if kind in 'US' else value.dtype
        fill=False if kind=='b' else -1 if kind in 'iu' else '' if kind in 'US' else np.nan
        arm[name]=np.full((1,h)+shape,fill,dtype=dtype)
        for t,z in enumerate(results):
            v=np.asarray(z[key])
            if key.startswith(('candidate_member_','mitigation_candidate_member_')) or key=='candidate_predicted_next_breach':arm[name][0,t,...,:v.shape[-1]]=v
            else:arm[name][0,t]=v
    arm.update(holdings=np.tile(x,(1,h+1,1)),cpi=np.ones((1,h+1)),real_wealth=np.full((1,h+1),100.),
        real_peak=np.full((1,h+1),100.),running_mdd=np.zeros((1,h+1)),gate_evaluated=np.array([[True,True,False]]),
        actual_previous_state=np.zeros((1,h,4)),forecast_n=np.array([[3,3,0]]),forecast_member_count=np.array([[1,5,0]]),
        selected_proposal=np.full((1,h),-1,int),selected_lambda=np.full((1,h),np.nan),target_execution=np.zeros((1,h),bool),
        automatic_service=np.array([[True,False,False]]))
    arm['actual_decision_source'][0,2]='pending';arm['primary_gate_reason'][0,2]=''
    for t,z in enumerate(results):
        idx=z['selected_index'];s=max(idx,0)
        if idx>=0:arm['selected_proposal'][0,t]=z['candidate_proposal'][s];arm['selected_lambda'][0,t]=z['candidate_lambda'][s]
        arm['target_execution'][0,t]=z['mitigation_selected_index']>0 if z['mitigation_evaluated'] else idx>0
    group=dict(group_id='synthetic',candidate=dict(release_fraction=.5,forecast_rule='uniform'),target=TARGET,scenario_ids=[4])
    return arm,gross-1,np.zeros((3,h)),ids,group


def check_arm(arm,r,pi,ids,group):
    c=a.Checks();c.mitigation_checks=[];c.active_decision_groups=[];a.model_gates(arm,r,pi,ids,group,c);return c


class CushionAuditTests(unittest.TestCase):
    def test_drawdown_identity_and_units(self):
        for R in (60.,70.,80.,90.,100.):
            m=a.cushion_scale(R,100.,.4)
            self.assertAlmostEqual(m,max(0.,(R-60)/(.4*R)),places=14)
            self.assertAlmostEqual(m,a.cushion_scale(R*1000,100000.,.4),places=14)
    def test_boundaries_and_monotonicity(self):
        self.assertEqual(a.cushion_scale(100,100,0),1);self.assertEqual(a.cushion_scale(100-1e-11,100,0),0)
        self.assertEqual(a.cushion_scale(10,100,1),1);self.assertEqual(a.cushion_scale(60-1e-11,100,.4),0)
        self.assertEqual([a.cushion_scale(x,100,.4) for x in (60,70,80,90,100)],sorted(a.cushion_scale(x,100,.4) for x in (60,70,80,90,100)))
        with self.assertRaises(ValueError):a.cushion_scale(101,100,.4)
    def test_at_peak_all_shared_fields_match_prior_independent(self):
        s=np.tile([1.1,.98,1.01],(3,1));weights=np.full((1,3),1/3)
        for release in (.5,.75,1.):
            for rule in ('uniform','ensemble_min'):
                kw=dict(target=TARGET,release=release,rule=rule)
                old=a.previous.scalar_decision([45,5,50],1,100,0,s,s,weights,**kw)
                new=a.scalar_decision([45,5,50],1,100,0,s,s,weights,**kw)
                for key in old:np.testing.assert_equal(new[key],old[key],err_msg=key)
    def test_initial_original_target_unchanged(self):
        s=np.ones((3,3));weights=np.full((1,3),1/3)
        old=a.previous.scalar_decision([0,0,100],1,100,0,s,s,weights,target=TARGET,release=.5,rule='uniform',initial=True)
        new=a.scalar_decision([0,0,100],1,100,0,s,s,weights,target=TARGET,release=.5,rule='uniform',initial=True)
        for key in old:np.testing.assert_equal(new[key],old[key])
        self.assertEqual(new['cushion_scale'],1);np.testing.assert_array_equal(new['proposal_target_weights'][0],TARGET)
    def test_scaled_destinations_and_fee_exact_once(self):
        z=decision();m=z['cushion_scale']
        for j,b in enumerate((1,.75,.5,.25)):
            np.testing.assert_allclose(z['proposal_target_weights'][j],b*m*TARGET+(1-b*m)*np.array([0,0,1]),rtol=0,atol=1e-15)
        for y,f in zip(z['candidate_posttrade'],z['candidate_fees']):
            if np.isfinite(f):self.assertAlmostEqual(sum(y)+f,80,places=12)
    def test_fallback_unscaled_and_never_recertified(self):
        s=np.tile([.2,.75,.75],(3,1));z=decision(support=s)
        self.assertTrue(z['mitigation_evaluated']);self.assertEqual(z['selected_index'],-1);self.assertFalse(z['support_feasible'])
        m=a.scalar_mitigation([80,0,0],1,s,target=TARGET,budget=z['budget_floor'],growth=s,weights=np.full((1,3),1/3),peak=100,worst=.2)
        np.testing.assert_array_equal(z['mitigation_candidate_posttrade'],m['candidate_posttrade'])
    def test_negative_real_bill_can_leave_zero_scale_infeasible(self):
        z=decision(real=60,support=np.full((3,3),.9))
        self.assertEqual(z['cushion_scale'],0);self.assertFalse(z['support_feasible']);self.assertGreater(z['worst_real_shortfall_selected'],0)
    def test_scaled_interval_binding_tie_uses_new_endpoint(self):
        z=decision(support=np.tile([.5,1.2,1.01],(3,1)));observed=copy.deepcopy(z)
        changed=0
        for key in ('interval_lower_member','interval_upper_member'):
            for j,k in enumerate(observed[key]):
                if z['interval_feasible'][j] and k>=0:observed[key][j]=(k+1)%3;changed+=1
        self.assertGreater(changed,0)
        c=a.Checks();a.decision_check(observed,z,c,'scaledtie',x=[80,0,0],cp=1,support=np.tile([.5,1.2,1.01],(3,1)),target=TARGET)
        self.assertEqual(len(c.numeric_binding_ties),changed)
    def test_false_binding_tie_rejected(self):
        z=decision(support=np.array([[.5,1.2,1.01],[.7,1.2,1.01],[.9,1.2,1.01]]));observed=copy.deepcopy(z)
        observed['candidate_worst_support_member'][0]=2
        with self.assertRaises(AssertionError):a.decision_check(observed,z,a.Checks(),'badtie',x=[80,0,0],cp=1,support=np.array([[.5,1.2,1.01],[.7,1.2,1.01],[.9,1.2,1.01]]),target=TARGET)
    def test_scalar_forecast_uses_declared_stricter_tolerance(self):
        z=decision();observed=copy.deepcopy(z);observed['selected_score']+=1e-10
        with self.assertRaises(AssertionError):a.decision_check(observed,z,a.Checks(),'forecasttol',x=[80,0,0],cp=1,support=np.ones((3,3)),target=TARGET)
    def test_every_active_decision_checked_with_pending_slots(self):
        c=check_arm(*small_arm());self.assertEqual(c.active_decision_groups[0]['active_decisions'],2)
        self.assertEqual(len(c.mitigation_checks),1)
    def test_corrupted_active_cushion_rejected(self):
        values=list(small_arm());values[0]['cushion_scale'][0,1]=.5
        with self.assertRaises(AssertionError):check_arm(*values)
    def test_pending_cushion_must_be_NULL(self):
        values=list(small_arm());values[0]['cushion_scale'][0,2]=0
        with self.assertRaises(AssertionError):check_arm(*values)
    def test_thin_group_still_recomputes_all_selected_decisions(self):
        values=list(small_arm());values[0]={k:v for k,v in values[0].items() if not k.startswith(('candidate_','mitigation_candidate_'))}
        c=check_arm(*values);self.assertEqual(c.active_decision_groups[0]['active_decisions'],2)
    def test_inventory_adds_two_only(self):
        self.assertEqual(len(a.previous.PRIMARY_FIELDS|a.previous.MIT_FIELDS|a.CUSHION_FIELDS),122)
        fields=a.previous.PRIMARY_FIELDS|a.previous.MIT_FIELDS|a.CUSHION_FIELDS
        omitted={k for k in fields if k.startswith(('candidate_','mitigation_candidate_')) or k in ('training_previous_state','training_real_gross')}
        self.assertEqual(len(fields-omitted),94)
    def test_no_production_or_unscaled_gate_import(self):
        text=Path(a.__file__).read_text();tree=ast.parse(text)
        modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('experimental_execution' in m or 'safeinvest.core' in m for m in modules))
        self.assertNotIn('base.model_gates(',text);self.assertNotIn('previous.model_gates(',text);self.assertNotIn('base.binding_check(',text)


if __name__=='__main__':unittest.main()
