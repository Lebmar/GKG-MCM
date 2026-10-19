"""Pure constructed metadata/byte/ZIP checks; never reexecutes research."""
from pathlib import Path
import ast,copy,hashlib,json,sys,tempfile,unittest,zipfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import verify_technical_completion_checkpoint as a


def put(root,name,text='saved'):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);return p

def dump(root,name,obj):return put(root,name,json.dumps(obj))


class CompletionTests(unittest.TestCase):
    def test_old_checkpoint_exact_map_no_old_checker_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);put(root,'old/a','a');put(root,'old/b','b');put(root,'old/tests','Ran 2 tests in 1.0s\n\nOK\n')
            old=dict(status='passed',distinct_hashed_files=2,checked_current_file_sha256={name:a.raw_sha(root/name) for name in ('old/a','old/b')},unit_tests=dict(passed=2,log_path='old/tests',log_sha256=a.raw_sha(root/'old/tests')))
            dump(root,'old/checkpoint.json',old);ledger=a.Ledger(root)
            result=a.verify_old(root,ledger,Path('old/checkpoint.json'),expected_count=2)
            self.assertTrue(result['all_prior_hashes_unchanged']);self.assertFalse(result['old_checks_rerun'])
            put(root,'old/a','changed')
            with self.assertRaises(AssertionError):ledger.close()

    def test_ledger_rejects_changed_source_digest_and_endwindow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);put(root,'source.py','before');ledger=a.Ledger(root);digest=ledger.touch('source.py')
            with self.assertRaises(AssertionError):ledger.touch('source.py','0'*64)
            put(root,'source.py','after')
            with self.assertRaises(AssertionError):ledger.close()
            self.assertNotEqual(digest,a.raw_sha(root/'source.py'))

    def test_metadata_lineage_hashes_root_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);put(root,'src/a.py','code');put(root,'result/output.bin','opaque')
            sources={'src/a.py':a.raw_sha(root/'src/a.py')};outputs={'output.bin':a.raw_sha(root/'result/output.bin')}
            dump(root,'result/manifest.json',dict(status='complete',sources_and_inputs_at_start=sources,sources_and_inputs_at_end=sources,outputs_sha256=outputs))
            ledger=a.Ledger(root);a.verify_receipt(root,ledger,'result/manifest.json');self.assertEqual(len(ledger.close()),3)
            bad=a.readj(root/'result/manifest.json');bad['sources_and_inputs_at_end']={};dump(root,'result/manifest.json',bad)
            with self.assertRaises(AssertionError):a.verify_receipt(root,a.Ledger(root),'result/manifest.json')

    def test_full_test_archive_count_is_parsed_not_hardcoded(self):
        x=a.parse_test_log('test_a ... ok\n\nRan 1327 tests in 12.345s\n\nOK\n')
        self.assertEqual((x['tests_run'],x['successful_non_skipped'],x['skipped']),(1327,1327,0))
        y=a.parse_test_log('Ran 20 tests in 0.2s\n\nOK (skipped=2)\n')
        self.assertEqual(y['successful_non_skipped'],18)
        for text in ('Ran 20 tests in 1.0s\nFAILED (failures=1)','Ran 2 tests in 1s\nOK\nRan 3 tests in 1s\nOK','OK','Ran 20 tests in 1s\nOK\nother job'):
            with self.assertRaises(AssertionError):a.parse_test_log(text)

    def test_strict_decimal_50MB_limit(self):
        self.assertEqual(a.strict_zip_size(49_999_999),49_999_999)
        for n in (0,-1,50_000_000,50_000_001,True,1.5):
            with self.assertRaises(AssertionError):a.strict_zip_size(n)

    def bundle(self,root):
        payload=b'print("fixed")\n';manifest=dict(status='complete',version='compact-technical-source-bundle-v1',development_ready=True,final_access=False,files_sha256={'src/a.py':hashlib.sha256(payload).hexdigest()},object_specification={'fixed':True})
        raw=json.dumps(manifest).encode();archive=root/'dist/source.zip';archive.parent.mkdir()
        with zipfile.ZipFile(archive,'w') as z:z.writestr('src/a.py',payload);z.writestr('submission_bundle_manifest.json',raw)
        put(root,'unpacked/src/a.py',payload.decode());put(root,'unpacked/submission_bundle_manifest.json',raw.decode())
        dump(root,'build/manifest.json',dict(status='complete',archive='dist/source.zip',archive_sha256=a.raw_sha(archive),archive_bytes=archive.stat().st_size,member_count=2))
        put(root,'replay/reproduction_checks.json','{"status":"passed"}')
        dump(root,'replay/manifest.json',dict(status='passed',mode='development',canonical_development_compared=True,scenario_ids=list(range(1,121)),final_scenarios_loaded=False,new_fits=0,new_selections=0,source_bundle_manifest_sha256=hashlib.sha256(raw).hexdigest(),outputs_sha256={'reproduction_checks.json':a.raw_sha(root/'replay/reproduction_checks.json')}))
        return dict(archive='dist/source.zip',build_receipt='build/manifest.json',unpacked_directory='unpacked',development_replay_receipt='replay/manifest.json')

    def test_compact_archive_and_unpacked_development_receipt_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.bundle(root);ledger=a.Ledger(root);out=a.verify_bundle(root,ledger,spec)
            self.assertTrue(out['strictly_below_50000000']);self.assertTrue(out['development_replay_passed']);ledger.close()
            bad=a.readj(root/'replay/manifest.json');bad['final_scenarios_loaded']=True;dump(root,'replay/manifest.json',bad)
            with self.assertRaises(AssertionError):a.verify_bundle(root,a.Ledger(root),spec)

    def test_changed_unpacked_source_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.bundle(root);put(root,'unpacked/src/a.py','changed')
            with self.assertRaises(AssertionError):a.verify_bundle(root,a.Ledger(root),spec)

    def companion(self,root):
        files={};figures={}
        for i in range(1,10):
            name=f'{i:02d}.png';p=put(root,'assets/'+name,'image'+str(i));files['assets/'+name]=a.raw_sha(p);figures[name]=dict(sha256=a.raw_sha(p),actual_view_image=True)
        dump(root,'nb.ipynb',dict(cells=[dict(cell_type='code',execution_count=i,outputs=[]) for i in (1,2)]))
        put(root,'nb.html','<html>saved</html>');put(root,'builder.py','no execution')
        put(root,'input.csv','opaque bytes only');dump(root,'assets/source.json',dict(sources={'input.csv':a.raw_sha(root/'input.csv')}))
        dump(root,'assets/qa.json',dict(figures=figures,top_to_bottom_no_errors=True))
        put(root,'screens/wide.png','wide');put(root,'screens/narrow.png','narrow')
        dump(root,'browser.json',dict(status='passed',html_path='nb.html',wide_view_passed=True,narrow_view_passed=True,wide_actual_viewport_width=1440,narrow_requested_viewport_width=430,narrow_actual_viewport_width=500,actual_430px_tested=False,html_sha256=a.raw_sha(root/'nb.html'),screenshot_sha256={x:a.raw_sha(root/x) for x in ('screens/wide.png','screens/narrow.png')}))
        dump(root,'assets/build.json',dict(status='executed',notebook='nb.ipynb',notebook_sha256=a.raw_sha(root/'nb.ipynb'),html='nb.html',html_sha256=a.raw_sha(root/'nb.html'),builder_sha256=a.raw_sha(root/'builder.py'),figure_count=9,new_policy_runs=0,raw_market_reads=0,artifact_sha256=files))
        return dict(build_manifest='assets/build.json',builder_source='builder.py',source_manifest='assets/source.json',visual_qa='assets/qa.json',root_browser_qa='browser.json',expected_wide_viewport_width=1440,expected_narrow_viewport_width=500)

    def test_companion_all9_executed_and_measured_viewport_truthful(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.companion(root);out=a.verify_notebook(root,a.Ledger(root),spec)
            self.assertEqual(out['figure_count'],9);self.assertEqual(out['narrow_actual_viewport_width'],500);self.assertEqual(out['narrow_requested_viewport_width'],430)
            spec['expected_narrow_viewport_width']=430
            with self.assertRaises(AssertionError):a.verify_notebook(root,a.Ledger(root),spec)

    def test_notebook_error_or_unexecuted_cells_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.companion(root)
            nb=a.readj(root/'nb.ipynb');nb['cells'][1]['execution_count']=None;dump(root,'nb.ipynb',nb)
            build=a.readj(root/'assets/build.json');build['notebook_sha256']=a.raw_sha(root/'nb.ipynb');dump(root,'assets/build.json',build)
            with self.assertRaises(AssertionError):a.verify_notebook(root,a.Ledger(root),spec)

    def reading(self,root):
        put(root,str(a.FINAL/'manifest.json'),'opaque fixed producer');put(root,str(a.FINAL/'independent_audit.json'),'opaque passed audit')
        put(root,'reading.html','current page');put(root,'reading_builder.py','current source')
        dump(root,'historical_review.json',dict(status='historical',sources_and_artifacts_at_start={'reading.html':'0'*64,'reading_builder.py':'0'*64}))
        files={}
        for name in ('01','02','03'):
            put(root,'reading_assets/'+name+'.png','reviewed image '+name);files[name+'.png']=a.raw_sha(root/('reading_assets/'+name+'.png'))
        put(root,'reading_assets/all60_guard_timing.csv','opaque timing csv');files['all60_guard_timing.csv']=a.raw_sha(root/'reading_assets/all60_guard_timing.csv')
        dump(root,'reading_assets/build.json',dict(status='built_pending_visual_QA',source_sha256=a.raw_sha(root/'reading_builder.py'),page='reading.html',page_sha256=a.raw_sha(root/'reading.html'),sources_sha256={str(a.FINAL/'manifest.json'):a.raw_sha(root/a.FINAL/'manifest.json')},output_sha256=files,figures=['01','02','03'],raw_market_reads=0,new_policy_runs=0,new_fits=0,scope='previously_exposed_nonblind'))
        current=['reading.html','reading_builder.py','reading_assets/build.json','historical_review.json']
        mapping={x:a.raw_sha(root/x) for x in current}
        dump(root,'addendum.json',dict(version='mitigation-final-interpretation-timing-addendum-v1',status='passed_saved_timing_presentation_addendum',historical_review='historical_review.json',historical_review_sha256=a.raw_sha(root/'historical_review.json'),historical_review_unchanged=True,parent_manifest_sha256=a.raw_sha(root/a.FINAL/'manifest.json'),independent_audit_sha256=a.raw_sha(root/a.FINAL/'independent_audit.json'),current_builder_sha256=a.raw_sha(root/'reading_builder.py'),current_page_sha256=a.raw_sha(root/'reading.html'),timing_csv_sha256=files['all60_guard_timing.csv'],all60_original_ids_exact=True,all60_timing_csv_fields_match_saved_arrays=True,all_three_PNG_bytes_unchanged_from_historical_review=True,unchanged_PNG_sha256={k:v for k,v in files.items() if k.endswith('.png')},historical_snapshot_paths_that_have_since_changed={'reading.html':dict(historical_sha256='0'*64,current_sha256=a.raw_sha(root/'reading.html'))},sources_and_artifacts_at_start=mapping,sources_and_artifacts_at_end=mapping,raw_market_reads=0,new_model_runs=0,new_fits=0,new_selection=0,producer_outputs_changed=False,frozen_sources_modified=False))
        put(root,'reading_wide.png','wide');put(root,'reading_narrow.png','narrow')
        dump(root,'reading_browser.json',dict(status='passed',html_path='reading.html',html_sha256=a.raw_sha(root/'reading.html'),wide_view_passed=True,narrow_view_passed=True,wide_actual_viewport_width=1440,narrow_actual_viewport_width=500,narrow_requested_viewport_width=430,actual_430px_tested=False,screenshot_sha256={x:a.raw_sha(root/x) for x in ('reading_wide.png','reading_narrow.png')}))
        return dict(build_manifest='reading_assets/build.json',builder_source='reading_builder.py',timing_addendum='addendum.json',root_browser_qa='reading_browser.json',expected_wide_viewport_width=1440,expected_narrow_viewport_width=500)

    def test_current_reading_addendum_keeps_old_review_opaque(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.reading(root);ledger=a.Ledger(root);out=a.verify_final_reading(root,ledger,spec)
            self.assertTrue(out['historical_snapshot_not_used_as_current_source']);self.assertEqual(out['figure_count'],3);ledger.close()
            self.assertNotEqual(a.readj(root/'historical_review.json')['sources_and_artifacts_at_start']['reading.html'],a.raw_sha(root/'reading.html'))

    def test_reading_stale_addendum_or_browser_image_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.reading(root);put(root,'reading_narrow.png','modified screenshot')
            with self.assertRaises(AssertionError):a.verify_final_reading(root,a.Ledger(root),spec)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.reading(root);d=a.readj(root/'addendum.json');d['current_page_sha256']='0'*64;dump(root,'addendum.json',d)
            with self.assertRaises(AssertionError):a.verify_final_reading(root,a.Ledger(root),spec)

    def test_narrow_requested430_is_not_claimed_as_tested430(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=self.companion(root);d=a.readj(root/'browser.json');d['actual_430px_tested']=True;dump(root,'browser.json',d)
            with self.assertRaises(AssertionError):a.verify_notebook(root,a.Ledger(root),spec)

    def test_missing_release_gate_and_failure_nooverwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dump(root,str(a.PLAN),dict(version='technical-completion-metadata-checkpoint-v1',execution_ready=False));dump(root,str(a.FREEZE),dict(execution_ready=False))
            with self.assertRaises(RuntimeError):a.run(root)
            self.assertEqual(a.readj(root/a.OUT/'manifest.json')['status'],'failed')
            with self.assertRaises(RuntimeError):a.run(root)
            self.assertFalse((root/a.OUT/'verification.json').exists())

    def test_anchor_inventory_includes_log_bundle_qa_incident_and_old_checkpoint(self):
        p=dict(unit_test_log='full_tests.txt',companion=dict(build_manifest='build.json',source_manifest='sources.json',visual_qa='qa.json',root_browser_qa='browser.json',builder_source='builder.py'),final_reading=dict(build_manifest='reading_build.json',timing_addendum='addendum.json',builder_source='reading_builder.py',root_browser_qa='reading_browser.json'),bundle=dict(archive='source.zip',build_receipt='bundle.json',development_replay_receipt='replay.json'),extra_selected_files=['README.md'])
        names=a.required_anchor_paths(p)
        for name in (str(a.OLD),str(a.INCIDENT),'full_tests.txt','source.zip','browser.json','README.md','reading_build.json','addendum.json','reading_browser.json'):self.assertIn(name,names)

    def test_source_never_imports_or_runs_market_notebook_test_or_old_checker(self):
        text=Path(a.__file__).read_text();tree=ast.parse(text)
        modules=[(n.module or '') for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(x.startswith(('numpy','pandas','safeinvest','experimental_execution','nbclient','verify_final_evaluation_checkpoint','verify_research_checkpoint','subprocess')) for x in modules))
        self.assertNotIn('np.load',text);self.assertNotIn('read_csv',text);self.assertNotIn('unittest.main',text)


if __name__=='__main__':unittest.main()
