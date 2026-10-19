"""Only constructed arrays and metadata; no raw/final market is parsed."""
from pathlib import Path
import ast,copy,json,math,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research')]
import run_mitigation_locked_final_evaluation as a
from safeinvest_research.metrics import path_metrics


def registry():return json.loads((ROOT/a.REGISTRY).read_text())

def gates():
    p=dict(version='mitigation-locked-final-evaluation-v1',execution_ready=True,final_execution_ready=False,training_ids=a.TRAIN,evaluation_ids=a.ACTUAL,output_directory=str(a.OUT),object_specification_sha256='o'*64,registry_sha256='r'*64,technical_evidence_sha256={'receipt.json':'e'*64})
    f=dict(execution_ready=True,protocol_sha256='p'*64)
    r=dict(version='mitigation-locked-final-evaluation-release-v1',final_execution_ready=True,protocol_sha256='p'*64,source_freeze_sha256='f'*64,object_specification_sha256=p['object_specification_sha256'],registry_sha256=p['registry_sha256'],training_ids=a.TRAIN,evaluation_ids=a.ACTUAL,output_directory=str(a.OUT),technical_evidence_sha256=p['technical_evidence_sha256'],previously_exposed_nonblind=True,adjust_after_final=False)
    activation=dict(version='mitigation-locked-final-release-hash-freeze-v1',execution_ready=True,release_sha256='s'*64,protocol_sha256='p'*64,source_freeze_sha256='f'*64)
    return p,f,r,activation


def arm(n,h):
    wealth=np.tile(100*1.01**np.arange(h+1),(n,1))
    holdings=np.repeat((wealth/3)[:,:,None],3,axis=2)
    return dict(real_wealth=wealth,nominal_wealth=wealth,holdings=holdings,posttrade_holdings=holdings[:,:-1].copy(),cpi=np.ones((n,h+1)),running_mdd=np.zeros((n,h+1)),fees=np.zeros((n,h)),moved=np.zeros((n,h)),trades=np.zeros((n,h),bool),pending_at_decision=np.zeros((n,h),bool),target_execution=np.ones((n,h),bool),first_crossing=np.full(n,-1),ever_breached=np.zeros((n,h+1),bool),gate_reason=np.full((n,h),'support_safe_hold',dtype='U64'),guard_triggered=np.zeros((n,h),bool),guard_pending=np.zeros((n,h+1),bool),actual_decision_source=np.full((n,h),'primary'))


class LockedFinalTests(unittest.TestCase):
    def test_all_frozen_objects_ownH_null_and_reference(self):
        spec=a.object_specification(registry());objects=spec['objects']
        self.assertEqual(len(objects),14);self.assertEqual(sum(x['question']=='Q2' for x in objects),4)
        self.assertEqual([(x['profile'],x['horizon']) for x in objects if x['question']=='Q1'],[(p,h) for p,h,_,_ in a.PROFILES])
        self.assertEqual([x['profile'] for x in objects if x['rule']=='Review'],['A','D'])
        self.assertEqual(sum(x['question']=='Q1_reference' for x in objects),5)
        self.assertFalse(spec['post_final_adjustment'])

    def test_registry_parameter_target_or_postfinal_change_rejected(self):
        for key,value in [('target',[.4,.1,.5]),('release_fraction',.75),('forecast_rule','ensemble_min'),('horizon',29),('persistent_target_relearning',True)]:
            bad=registry();bad['Q2'][key]=value
            with self.assertRaises(ValueError):a.object_specification(bad)
        bad=registry();bad['final_evaluation_contract']['adjust_after_final']=True
        with self.assertRaises(ValueError):a.object_specification(bad)

    def test_two_layer_release_exact_success(self):
        p,f,r,x=gates();a.authorize_release(p,f,r,x,protocol_sha='p'*64,freeze_sha='f'*64,release_sha='s'*64)

    def test_prep_never_enables_final_without_fresh_release_and_hash(self):
        for index,key,value in [(0,'final_execution_ready',True),(1,'execution_ready',False),(2,'final_execution_ready',False),(3,'release_sha256','changed')]:
            items=list(gates());items[index][key]=value
            with self.assertRaises(RuntimeError):a.authorize_release(*items,protocol_sha='p'*64,freeze_sha='f'*64,release_sha='s'*64)

    def test_release_cannot_change_ids_spec_evidence_or_nonblind_status(self):
        for key,value in [('training_ids',list(range(2,122))),('evaluation_ids',list(range(1,61))),('object_specification_sha256','bad'),('technical_evidence_sha256',{}),('previously_exposed_nonblind',False),('adjust_after_final',True)]:
            p,f,r,x=gates();r[key]=value
            with self.assertRaises(RuntimeError):a.authorize_release(p,f,r,x,protocol_sha='p'*64,freeze_sha='f'*64,release_sha='s'*64)

    def test_missing_release_stops_before_reading_numeric_metadata_or_import(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,'readj') as read,patch.object(a.np,'load') as loader:
            with self.assertRaises(RuntimeError):a.preflight(tmp)
            read.assert_not_called();loader.assert_not_called()

    def test_dependency_hash_changes_are_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source';p.write_text('one');mapping={'source':a.sha(p)};a.verify(tmp,mapping);p.write_text('two')
            with self.assertRaises(RuntimeError):a.verify(tmp,mapping)
            with self.assertRaises(ValueError):a.merge({}, {'../escape':'a'*64})

    def test_evaluation_never_refits_or_reselects_and_keeps_exact_argument_ids(self):
        spec=a.object_specification(registry());calls=[]
        def engine(r,pi,tr,tp,target,**kw):
            calls.append(('main',target,kw));return dict(arm=arm(60,kw['horizon']),status='funded',financial_available=True,initial_witness={'status':'selected'})
        def paired(r,pi,target,L,**kw):
            calls.append(('reference',target,dict(kw,L=L)));x=arm(60,kw['horizon']);wrapped={k:v if k=='cpi' else v[:,None] for k,v in x.items()};return dict(annual=wrapped,pending_hold=wrapped)
        result,w=a.evaluate_objects(spec,np.zeros((60,30,3)),np.zeros((60,30)),np.zeros((120,30,3)),np.zeros((120,30)),engine=engine,paired=paired)
        self.assertEqual(len(calls),12);self.assertEqual(len(result),14)
        main=calls[0][2];self.assertEqual(main['training_ids'],a.TRAIN);self.assertEqual(main['actual_ids'],a.ACTUAL)
        self.assertEqual((main['release_fraction'],main['forecast_rule'],main['support_mode']),(.5,'uniform','calendar'))
        self.assertIsNone(result['Q1_A']['arm']);self.assertIsNone(result['Q1_D']['arm'])
        self.assertEqual(result['Q1_E']['arm']['real_wealth'].shape,(60,7))

    def test_all840_rows_14_summaries_preserve_original_final_labels(self):
        spec=a.object_specification(registry());rows=[]
        for obj in spec['objects']:
            available=obj['rule']!='Review';result=dict(financial_available=available,status='funded' if available else 'q1_review',arm=arm(60,obj['horizon']) if available else None)
            group=a.path_rows(obj,result,a.ACTUAL,metrics=path_metrics);rows+=group
            self.assertEqual(sorted(x['scenario_id'] for x in group),a.ACTUAL)
            self.assertEqual(a.summary(group)['original_n'],60)
            if not available:self.assertTrue(all(x[k] is None for x in group for k in a.FIN))
        self.assertEqual(len(rows),840);self.assertEqual(sum(r['question']=='Q2' for r in rows),240)
        self.assertEqual(sum(r['question']=='Q1' for r in rows),300);self.assertEqual(sum(r['question']=='Q1_reference' for r in rows),300)

    def test_wrong_ownH_or_final_id_rejected(self):
        obj=next(x for x in a.object_specification(registry())['objects'] if x['object_id']=='Q1_E')
        result=dict(arm=arm(60,30),status='reference',financial_available=True)
        with self.assertRaises(ValueError):a.path_rows(obj,result,a.ACTUAL,metrics=path_metrics)
        result['arm']=arm(60,6)
        with self.assertRaises(ValueError):a.path_rows(obj,result,list(range(1,61)),metrics=path_metrics)

    def test_ordinary_service_excludes_mitigation_trigger_and_pending(self):
        obj=a.object_specification(registry())['objects'][0];x=arm(60,30)
        x['guard_triggered'][:,5]=True;x['pending_at_decision'][:,5:]=True;x['guard_pending'][:,6:]=True
        x['gate_reason'][:,5]='risk_mitigation_then_review';x['trades'][:,5]=True;x['fees'][:,5]=.3
        rows=a.path_rows(obj,dict(arm=x,status='funded',financial_available=True),a.ACTUAL,metrics=path_metrics)
        self.assertAlmostEqual(rows[0]['automatic_fraction'],5/30);self.assertTrue(rows[0]['support_review'])
        self.assertAlmostEqual(rows[0]['mitigation_trade_fraction'],1/30);self.assertEqual(rows[0]['mitigation_nominal_fees'],.3)

    def test_null_comparison_has_no_financial_counterfactual(self):
        objects=a.object_specification(registry())['objects'];aobj=next(x for x in objects if x['object_id']=='Q1_A');ref=next(x for x in objects if x['object_id']=='Q1_A_thirds')
        rows=a.path_rows(aobj,dict(arm=None,status='q1_review',financial_available=False),a.ACTUAL,metrics=path_metrics)
        rr=a.path_rows(ref,dict(arm=arm(60,5),status='ref',financial_available=True),a.ACTUAL,metrics=path_metrics)
        result,pairs=a.compare(rows,rr)
        self.assertEqual(pairs,[]);self.assertEqual(result['reference_all_original']['financial_n'],60)
        self.assertEqual(result['same_funded_reference']['financial_n'],0);self.assertIsNone(result['paired_differences']['mdd']['mean'])

    def test_severe_rule_uses_originalID_and_exact_tie(self):
        rows=[dict(scenario_id=s,assigned=True,mdd=.2) for s in a.ACTUAL]
        results={name:dict(arm=arm(60,30)) for name in ('mitigation_main','same_target_B','same_target_annual','exact_thirds_annual')}
        self.assertEqual(a.severe_example(rows,results)['scenario_id'],121)
        rows[-1]['mdd']=.21;self.assertEqual(a.severe_example(rows,results)['scenario_id'],180)
        self.assertIsNone(a.severe_example([dict(scenario_id=121,assigned=False)],{})['scenario_id'])

    def test_actual_kernel_import_smoke_without_loading_or_simulating(self):
        sys.path.insert(0,str(ROOT/'src'))
        from safeinvest.data import load_paths
        from experimental_execution.review_pending import simulate_paired
        from experimental_execution.state_budget_mitigation import simulate_state_budget
        import inspect
        for fn in (load_paths,simulate_paired,simulate_state_budget):self.assertTrue(callable(fn))
        self.assertIn('horizon',inspect.signature(simulate_paired).parameters)
        self.assertIn('cost_rate',inspect.signature(simulate_paired).parameters)
        self.assertIn('actual_ids',inspect.signature(simulate_state_budget).parameters)

    def test_real_kernels_constructed_zero_market_end_to_end(self):
        sys.path.insert(0,str(ROOT/'src'))
        from experimental_execution.review_pending import simulate_paired
        from experimental_execution.state_budget_mitigation import simulate_state_budget
        spec=a.object_specification(registry())
        results,witness=a.evaluate_objects(spec,np.zeros((60,30,3)),np.zeros((60,30)),np.zeros((120,30,3)),np.zeros((120,30)),engine=simulate_state_budget,paired=simulate_paired)
        self.assertEqual(sum(z['arm'] is not None for z in results.values()),12)
        main=results['mitigation_main']['arm'];self.assertEqual(main['holdings'].shape,(60,31,3))
        self.assertEqual(main['candidate_posttrade'].shape,(60,30,13,3))
        self.assertEqual(main['mitigation_candidate_posttrade'].shape,(60,30,9,3))
        self.assertEqual(main['training_real_gross'].shape,(30,120,3))
        fee=100*.0025*.5/(1+.0025*.5)
        np.testing.assert_allclose(main['fees'][:,0],fee,rtol=0,atol=1e-12)
        rows=[];summaries=[]
        for obj in spec['objects']:
            result=results[obj['object_id']];group=a.path_rows(obj,result,a.ACTUAL,metrics=path_metrics)
            rows+=group;summaries.append(a.summary(group))
            if result['arm'] is not None:
                self.assertEqual(result['arm']['real_wealth'].shape,(60,obj['horizon']+1))
                self.assertEqual(result['arm']['fees'].shape,(60,obj['horizon']))
                risky=1-obj['target'][2];expected=100*.0025*risky/(1+.0025*risky)
                np.testing.assert_allclose(result['arm']['fees'][:,0],expected,rtol=0,atol=1e-12)
            else:
                self.assertIn(obj['profile'],('A','D'));self.assertTrue(all(r[k] is None for r in group for k in a.FIN))
        self.assertEqual((len(rows),len(summaries)),(840,14));self.assertEqual(witness['selected_index'],1)
        mainrows=[r for r in rows if r['object_id']=='mitigation_main']
        self.assertEqual(a.severe_example(mainrows,results)['scenario_id'],121)
        self.assertTrue(all(r['assigned'] and not r['support_review'] for r in mainrows))

    def test_source_has_exact_loader_calls_after_release_and_no_selection(self):
        text=Path(a.__file__).read_text();tree=ast.parse(text)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_paths']
        self.assertEqual(len(calls),2);self.assertEqual([ast.unparse(c.args[1]) for c in calls],['range(1, 121)','range(121, 181)'])
        self.assertEqual([{k.arg:ast.literal_eval(k.value) for k in c.keywords}['allow_final'] for c in calls],[False,True])
        body=ast.get_source_segment(text,run);self.assertLess(body.index('preflight(root)'),body.index('from safeinvest.data import load_paths'))
        self.assertNotIn('select_inner',text);self.assertNotIn('fit_joint',text)


if __name__=='__main__':unittest.main()
