"""Metadata/byte-only completion checkpoint; no tests or research jobs rerun.

Preserves the old1055-file checkpoint and the precommit incident. Reads saved
JSON/notebook metadata, logs, ZIP member bytes and hashes only. Never imports a
market loader, numerical engine, notebook executor or old checkpoint runner.
"""
from __future__ import annotations
import argparse,hashlib,json,re,stat,sys,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=Path('docs/technical_completion_checkpoint_plan.json')
FREEZE=Path('docs/technical_completion_checkpoint_source_freeze.json')
OUT=Path('results/technical_completion_checkpoint_v1')
OLD=Path('results/final_locked_checkpoint_v1/verification.json')
REGISTRY=Path('docs/mitigation_main_model_registry.json')
FINAL_PROTOCOL=Path('docs/mitigation_locked_final_evaluation_protocol.json')
FINAL=Path('results/mitigation_locked_final_evaluation_v1')
FINAL_REPLAY=Path('results/mitigation_locked_final_reproduction_v1')
DEV_REPLAY=Path('results/technical_rebuild_reproduction_v1')
INCIDENT=Path('docs/mitigation_final_precommit_incident.json')
HASH_KEYS={'source_sha256','input_sha256','sources_sha256','development_evidence_sha256','technical_evidence_sha256','sources_and_inputs_at_start','sources_and_inputs_at_end','sources_and_artifacts_at_start','sources_and_artifacts_at_end','audit_sources_and_inputs','sources_and_original_artifacts_at_start','sources_and_original_artifacts_at_end','outputs_sha256','output_sha256','artifact_sha256','sources','analysis_source_sha256','producer_outputs_at_start','producer_outputs_at_end','analysis_input_sha256','source_input_sha256','file_sha256'}
PAIRS=(('sources_and_inputs_at_start','sources_and_inputs_at_end'),('sources_and_artifacts_at_start','sources_and_artifacts_at_end'),('sources_and_original_artifacts_at_start','sources_and_original_artifacts_at_end'),('producer_outputs_at_start','producer_outputs_at_end'))
HEX=re.compile(r'[0-9a-f]{64}')


def relative(name):
    p=Path(name)
    if not isinstance(name,str) or not name or p.is_absolute() or '..' in p.parts or p.as_posix()!=name or name=='.':raise ValueError('Canonical relative path required')
    return p

def raw_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def readj(path):return json.loads(Path(path).read_text())
def writej(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


class Ledger:
    def __init__(self,root):self.root=Path(root).resolve();self.start={};self.checks=0
    def touch(self,name,expected=None):
        name=str(relative(str(name)));path=self.root/name
        if not path.is_file() or not path.resolve().is_relative_to(self.root):raise RuntimeError('Required local file missing/escaped: '+name)
        digest=self.start.get(name) or raw_sha(path)
        if expected is not None and (not isinstance(expected,str) or HEX.fullmatch(expected) is None or digest!=expected):raise AssertionError('Immutable hash mismatch: '+name)
        if name in self.start and self.start[name]!=digest:raise AssertionError('File changed during checkpoint: '+name)
        self.start.setdefault(name,digest);self.checks+=1;return digest
    def mapping(self,mapping,prefix=Path('.')):
        if not isinstance(mapping,dict):raise ValueError('Hash mapping required')
        for name,digest in mapping.items():self.touch(str(prefix/relative(name)),digest)
    def close(self):
        end={name:raw_sha(self.root/name) for name in self.start}
        if end!=self.start:raise AssertionError('Selected source/output bytes changed during checkpoint')
        return dict(sorted(end.items()))


def verify_old(root,ledger,path=OLD,expected_count=1055):
    ledger.touch(path);old=readj(Path(root)/path)
    values=old['checked_current_file_sha256']
    if old['status']!='passed' or old['distinct_hashed_files']!=expected_count or len(values)!=expected_count:raise AssertionError('Complete prior1055 checkpoint required')
    ledger.mapping(values)
    ledger.touch(old['unit_tests']['log_path'],old['unit_tests']['log_sha256'])
    return dict(path=str(path),sha256=ledger.start[str(path)],old_hashed_files=len(values),old_tests=old['unit_tests']['passed'],old_checks_rerun=False,all_prior_hashes_unchanged=True)


def walk_records(value):
    if isinstance(value,dict):
        yield value
        for v in value.values():yield from walk_records(v)
    elif isinstance(value,list):
        for v in value:yield from walk_records(v)


def verify_receipt(root,ledger,name,*,statuses=('complete','passed')):
    ledger.touch(name);data=readj(Path(root)/name)
    if data.get('status') not in statuses:raise AssertionError('Required receipt not complete/passed: '+str(name))
    for record in walk_records(data):
        for before,after in PAIRS:
            if before in record and after in record and record[before]!=record[after]:raise AssertionError('Recorded start/end evidence mismatch: '+str(name))
        for key in HASH_KEYS:
            mapping=record.get(key)
            if not isinstance(mapping,dict) or not mapping:continue
            if not all(isinstance(v,str) and HEX.fullmatch(v) for v in mapping.values()):continue
            for file,digest in mapping.items():
                p=relative(file);candidates=[p,Path(name).parent/p]
                if key.startswith('output'):candidates.reverse()
                found=next((x for x in candidates if (Path(root)/x).is_file()),None)
                if found is None:raise AssertionError('Bound receipt artifact missing: '+str(name)+'/'+file)
                ledger.touch(found,digest)
    return data


def parse_test_log(text):
    matches=list(re.finditer(r'^Ran (\d+) tests? in ([0-9.]+)s\s*$',text,re.M))
    if len(matches)!=1:raise AssertionError('Exactly one full unittest summary required')
    last=next((x.strip() for x in reversed(text.splitlines()) if x.strip()),'')
    terminal=re.fullmatch(r'OK(?: \(skipped=(\d+)\))?',last)
    if not terminal:raise AssertionError('Full unittest archive must end in successful OK')
    if re.search(r'^FAILED(?:\s|\(|$)',text,re.M):raise AssertionError('Failed test evidence in full archive')
    ran=int(matches[0].group(1));skipped=int(terminal.group(1) or 0)
    if ran<=0 or skipped>ran:raise AssertionError('Invalid unittest counts')
    return dict(tests_run=ran,skipped=skipped,successful_non_skipped=ran-skipped,seconds=float(matches[0].group(2)),status='passed')


def verify_final(root,ledger,protocol):
    main=verify_receipt(root,ledger,FINAL/'manifest.json',statuses=('complete',))
    audit=verify_receipt(root,ledger,FINAL/'independent_audit.json',statuses=('passed',))
    replay=verify_receipt(root,ledger,FINAL_REPLAY/'manifest.json',statuses=('passed',))
    comp=readj(Path(root)/FINAL_REPLAY/'comparison.json');ledger.touch(FINAL_REPLAY/'comparison.json')
    if (main['protocol_sha256']!=ledger.touch(FINAL_PROTOCOL) or main['scope']!='previously_exposed_nonblind' or main['post_final_parameter_changes'] is not False
        or main['new_fits']!=0 or main['new_selections']!=0 or main['training_ids_loaded']!=list(range(1,121)) or main['final_ids_loaded']!=list(range(121,181))
        or (main['Q2_rows'],main['Q1_rows'],main['Q1_reference_rows'],main['summary_rows'],main['comparisons'],main['trace_groups'])!=(240,300,300,14,8,14)
        or len(main['outputs_sha256'])!=11):raise AssertionError('Complete fixed final object/ID contract required')
    if audit['producer_manifest_sha256']!=ledger.touch(FINAL/'manifest.json') or audit['post_final_parameter_changes'] is not False or audit['scope']!='previously_exposed_nonblind':raise AssertionError('Independent final audit does not bind unchanged producer')
    counts=audit['counts']
    if (counts['objects'],counts['original_scenarios'],counts['physical_accounts'],counts['periods'],counts['Review_rows'],counts['path_rows'],counts['summary_rows'],counts['comparison_groups'])!=(14,60,720,15120,120,840,14,8):raise AssertionError('Complete independent ownH/NULL accounting audit required')
    if (comp['status']!='passed' or comp['differences'] or len(comp['files'])!=11 or not comp['root_manifest_equal_after_two_timestamps']
        or comp['root_manifest_normalized_fields']!=['started_utc','completed_utc']):raise AssertionError('All11 isolated final comparisons required')
    if any(not x['bytes_equal'] or x.get('npz',{}).get('exact',True) is not True for x in comp['files'].values()):raise AssertionError('Final artifacts not byte/logically exact')
    if replay.get('independent_additional_scenarios')!=0:raise AssertionError('Replay is not new independent evidence')
    release=readj(Path(root)/'docs/mitigation_locked_final_evaluation_release.json');activation=readj(Path(root)/'docs/mitigation_locked_final_evaluation_release_freeze.json')
    if protocol['final_execution_ready'] is not False or release['final_execution_ready'] is not True or activation['release_sha256']!=ledger.touch('docs/mitigation_locked_final_evaluation_release.json'):raise AssertionError('Release/hash activation linkage changed')
    ledger.touch('docs/mitigation_locked_final_evaluation_release_freeze.json');ledger.touch('docs/mitigation_locked_final_evaluation_source_freeze.json')
    return dict(producer_manifest_sha256=ledger.touch(FINAL/'manifest.json'),audit_sha256=ledger.touch(FINAL/'independent_audit.json'),replay_manifest_sha256=ledger.touch(FINAL_REPLAY/'manifest.json'),objects=14,artifacts=11,scope='previously_exposed_nonblind',new_fits=0,new_selections=0)


def verify_dev_replays(root,ledger):
    m=verify_receipt(root,ledger,DEV_REPLAY/'manifest.json',statuses=('passed',))
    expected={'v2_canonical','v2b_canonical','v3_canonical','v2_stability','v2b_stability','v3_stability'}
    if len(m['profiles'])!=6 or {p['profile_id'] for p in m['profiles']}!=expected or any(p['status']!='passed' for p in m['profiles']):raise AssertionError('All six development replay profiles must pass')
    for row in m['profiles']:
        path=DEV_REPLAY/row['profile_id']/'comparison.json';ledger.touch(path,row['comparison_sha256']);comp=readj(Path(root)/path)
        if comp['status']!='passed' or comp['differences'] or not comp['root_manifest_equal_after_two_timestamps']:raise AssertionError('Complete development replay comparison failed')
    return dict(profiles=sorted(expected),count=6,independent_additional_scenarios=0)


def verify_notebook(root,ledger,spec):
    build=verify_receipt(root,ledger,spec['build_manifest'],statuses=('executed',));nbpath=build['notebook'];htmlpath=build['html']
    ledger.touch(nbpath,build['notebook_sha256']);ledger.touch(htmlpath,build['html_sha256']);ledger.touch(spec['builder_source'],build['builder_sha256'])
    if build['figure_count']!=9 or build.get('new_policy_runs')!=0 or build.get('raw_market_reads')!=0:raise AssertionError('All8+LP figures, saved-only executed companion required')
    notebook=readj(Path(root)/nbpath);cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    if not cells or [c['execution_count'] for c in cells]!=list(range(1,len(cells)+1)) or any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[])):raise AssertionError('Notebook not executed top-to-bottom without errors')
    source=readj(Path(root)/spec['source_manifest']);ledger.touch(spec['source_manifest']);ledger.mapping(source['sources'])
    qa=readj(Path(root)/spec['visual_qa']);ledger.touch(spec['visual_qa'])
    if len(qa['figures'])!=9 or qa.get('top_to_bottom_no_errors') is not True:raise AssertionError('Actual9figure QA required')
    for name,info in qa['figures'].items():
        if info.get('actual_view_image') is not True:raise AssertionError('Figure not visually viewed')
        ledger.touch(Path(spec['visual_qa']).parent/name,info['sha256'])
    browser=verify_browser_qa(root,ledger,spec,htmlpath)
    return dict(notebook=nbpath,html=htmlpath,figure_count=9,code_cells=len(cells),top_to_bottom_no_errors=True,actual_images_viewed=True,root_browser_wide_and_narrow=True,narrow_actual_viewport_width=browser['narrow_actual_viewport_width'],narrow_requested_viewport_width=browser.get('narrow_requested_viewport_width'))


def verify_browser_qa(root,ledger,spec,htmlpath):
    browser=readj(Path(root)/spec['root_browser_qa']);ledger.touch(spec['root_browser_qa'])
    if (browser.get('status')!='passed' or browser.get('html_path')!=htmlpath or browser.get('wide_view_passed') is not True
        or browser.get('narrow_view_passed') is not True or browser.get('narrow_actual_viewport_width')!=spec['expected_narrow_viewport_width']
        or browser.get('wide_actual_viewport_width')!=spec['expected_wide_viewport_width'] or browser.get('html_sha256')!=ledger.touch(htmlpath)):
        raise AssertionError('Current wide/measured-narrow HTML review required')
    if browser.get('narrow_requested_viewport_width')!=browser['narrow_actual_viewport_width'] and browser.get('actual_430px_tested') is not False:raise AssertionError('Do not claim requested430 was actual500 viewport')
    screenshots=browser.get('screenshot_sha256')
    if not isinstance(screenshots,dict) or len(screenshots)!=2:raise AssertionError('Bind both actual browser screenshots')
    ledger.mapping(screenshots)
    return browser


def verify_final_reading(root,ledger,spec):
    build=verify_receipt(root,ledger,spec['build_manifest'],statuses=('built_pending_visual_QA',))
    page=build['page'];assets=Path(spec['build_manifest']).parent
    ledger.touch(page,build['page_sha256']);ledger.touch(spec['builder_source'],build['source_sha256'])
    if len(build['figures'])!=3 or any(build.get(x)!=0 for x in ('raw_market_reads','new_policy_runs','new_fits')) or build.get('scope')!='previously_exposed_nonblind':raise AssertionError('Saved-only three-figure final reading contract required')
    addendum=verify_receipt(root,ledger,spec['timing_addendum'],statuses=('passed_saved_timing_presentation_addendum',))
    if (addendum.get('version')!='mitigation-final-interpretation-timing-addendum-v1' or addendum.get('historical_review_unchanged') is not True
        or addendum.get('parent_manifest_sha256')!=ledger.touch(FINAL/'manifest.json') or addendum.get('independent_audit_sha256')!=ledger.touch(FINAL/'independent_audit.json')
        or addendum.get('current_builder_sha256')!=ledger.touch(spec['builder_source']) or addendum.get('current_page_sha256')!=ledger.touch(page)
        or addendum.get('timing_csv_sha256')!=ledger.touch(assets/'all60_guard_timing.csv')
        or addendum.get('all60_original_ids_exact') is not True or addendum.get('all60_timing_csv_fields_match_saved_arrays') is not True
        or addendum.get('all_three_PNG_bytes_unchanged_from_historical_review') is not True
        or any(addendum.get(x)!=0 for x in ('raw_market_reads','new_model_runs','new_fits','new_selection'))
        or addendum.get('producer_outputs_changed') is not False or addendum.get('frozen_sources_modified') is not False):raise AssertionError('Current timing addendum does not bind unchanged saved final output/presentation')
    # Earlier review is opaque archival evidence. Its earlier HTML/builder hashes
    # are intentionally not requirements on the current presentation files.
    ledger.touch(addendum['historical_review'],addendum['historical_review_sha256'])
    png=addendum.get('unchanged_PNG_sha256',{})
    if len(png)!=3 or set(png)!={x+'.png' for x in build['figures']}:raise AssertionError('All three previously viewed final PNGs required')
    ledger.mapping(png,prefix=assets)
    for name,versions in addendum.get('historical_snapshot_paths_that_have_since_changed',{}).items():ledger.touch(name,versions['current_sha256'])
    browser=verify_browser_qa(root,ledger,spec,page)
    return dict(page=page,figure_count=3,current_build_manifest_sha256=ledger.touch(spec['build_manifest']),timing_addendum_sha256=ledger.touch(spec['timing_addendum']),historical_review_sha256=addendum['historical_review_sha256'],historical_snapshot_not_used_as_current_source=True,current_browser_wide_and_narrow=True,narrow_actual_viewport_width=browser['narrow_actual_viewport_width'],narrow_requested_viewport_width=browser.get('narrow_requested_viewport_width'),scope='previously_exposed_nonblind')


def strict_zip_size(size):
    if type(size) is not int or not 0<size<50_000_000:raise AssertionError('Archive must be strictly below50,000,000bytes')
    return size


def verify_bundle(root,ledger,spec):
    build=verify_receipt(root,ledger,spec['build_receipt'],statuses=('complete',));archive=relative(spec['archive']);digest=ledger.touch(archive)
    size=strict_zip_size((Path(root)/archive).stat().st_size)
    if build['archive']!=str(archive) or build['archive_sha256']!=digest or build['archive_bytes']!=size:raise AssertionError('Compact ZIP must be strictly below50,000,000bytes and exactly bound')
    with zipfile.ZipFile(Path(root)/archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(set(names))!=len(names) or any(i.is_dir() or stat.S_ISLNK(i.external_attr>>16) for i in infos):raise AssertionError('No duplicate/directory/symlink ZIP members')
        for name in names:relative(name)
        raw=z.read('submission_bundle_manifest.json');manifest=json.loads(raw)
        if manifest.get('status')!='complete' or manifest.get('version')!='compact-technical-source-bundle-v1' or manifest.get('development_ready') is not True or manifest.get('final_access') is not False:raise AssertionError('Source bundle boundary/schema changed')
        files=manifest['files_sha256']
        if set(names)!=set(files)|{'submission_bundle_manifest.json'} or build['member_count']!=len(names):raise AssertionError('Complete compact member inventory required')
        for name,expected in files.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise AssertionError('ZIP member changed: '+name)
            ledger.touch(Path(spec['unpacked_directory'])/relative(name),expected)
        unpacked_manifest=Path(spec['unpacked_directory'])/'submission_bundle_manifest.json';manifest_sha=hashlib.sha256(raw).hexdigest();ledger.touch(unpacked_manifest,manifest_sha)
    replay=verify_receipt(root,ledger,spec['development_replay_receipt'],statuses=('passed',))
    if (replay.get('mode')!='development' or replay.get('canonical_development_compared') is not True or replay.get('scenario_ids')!=list(range(1,121)) or replay.get('final_scenarios_loaded') is not False or replay.get('new_fits')!=0 or replay.get('new_selections')!=0 or replay.get('source_bundle_manifest_sha256')!=manifest_sha):raise AssertionError('Unpacked same-fixed-model development replay not complete')
    return dict(archive=str(archive),archive_sha256=digest,archive_bytes=size,strictly_below_50000000=True,member_count=len(names),development_replay_passed=True,final_access=False)


def required_anchor_paths(plan):
    c,b,r=plan['companion'],plan['bundle'],plan['final_reading']
    return sorted({str(OLD),str(REGISTRY),str(FINAL_PROTOCOL),str(INCIDENT),str(FINAL/'manifest.json'),str(FINAL/'independent_audit.json'),str(FINAL_REPLAY/'manifest.json'),str(DEV_REPLAY/'manifest.json'),plan['unit_test_log'],c['build_manifest'],c['source_manifest'],c['visual_qa'],c['root_browser_qa'],c['builder_source'],r['build_manifest'],r['timing_addendum'],r['builder_source'],r['root_browser_qa'],b['archive'],b['build_receipt'],b['development_replay_receipt'],*plan.get('extra_selected_files',[])})


def authorize(root):
    p,f=readj(Path(root)/PLAN),readj(Path(root)/FREEZE)
    if p.get('version')!='technical-completion-metadata-checkpoint-v1' or p.get('execution_ready') is not True or f.get('execution_ready') is not True or f.get('plan_sha256')!=raw_sha(Path(root)/PLAN):raise RuntimeError('Freeze complete checkpoint source/input/plan after all receipts/log/package ready')
    required={'scripts/verify_technical_completion_checkpoint.py','tests/test_technical_completion_checkpoint.py'}
    if not required<=set(f.get('source_sha256',{})) or not isinstance(p.get('expected_unit_test_count'),int) or p['expected_unit_test_count']<1:raise RuntimeError('Complete checkpoint source/full test count required')
    if not set(required_anchor_paths(p))<=set(f['input_sha256']):raise RuntimeError('Bind every checkpoint receipt/log/companion/bundle anchor before execution')
    ledger=Ledger(root);ledger.mapping(f['source_sha256']);ledger.mapping(f['input_sha256']);ledger.touch(PLAN);ledger.touch(FREEZE)
    return p,ledger


def run(root=ROOT):
    root=Path(root);out=root/OUT
    if out.exists():raise RuntimeError('Never overwrite a completion checkpoint attempt')
    out.mkdir(parents=True);state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),scope='Metadata and byte hashes only; no market parsing, model, notebook execution, test rerun or package replay')
    writej(out/'manifest.json',state)
    try:
        plan,ledger=authorize(root);old=verify_old(root,ledger)
        registry=readj(root/REGISTRY);ledger.touch(REGISTRY,plan['registry_sha256']);ledger.mapping(registry['sources_sha256']);ledger.mapping(registry['development_evidence_sha256'])
        protocol=readj(root/FINAL_PROTOCOL);ledger.touch(FINAL_PROTOCOL)
        receipts=protocol['technical_evidence_sha256']
        if len(receipts)!=plan['expected_technical_receipt_count']:raise AssertionError('All frozen technical receipts, including LP witness audit, required')
        for name,digest in receipts.items():ledger.touch(name,digest);verify_receipt(root,ledger,name)
        final=verify_final(root,ledger,protocol);development=verify_dev_replays(root,ledger)
        incident=readj(root/INCIDENT);ledger.touch(INCIDENT,plan['precommit_incident_sha256']);ledger.mapping(incident['file_sha256'])
        if incident.get('source_release_hash_freeze_preceded_actual_market_parse') is not True or incident.get('git_commit_preceded_actual_market_parse') is not False or incident.get('no_model_or_parameter_change') is not True or incident.get('no_numerical_retry') is not True:raise AssertionError('Preserve accurate precommit incident')
        notebook=verify_notebook(root,ledger,plan['companion']);reading=verify_final_reading(root,ledger,plan['final_reading']);bundle=verify_bundle(root,ledger,plan['bundle'])
        ledger.touch(plan['unit_test_log']);tests=parse_test_log((root/plan['unit_test_log']).read_text())
        if tests['tests_run']!=plan['expected_unit_test_count']:raise AssertionError('Full archived unittest count differs')
        for path in plan.get('extra_selected_files',[]):ledger.touch(path)
        end=ledger.close();(out/'unit_tests.txt').write_bytes((root/plan['unit_test_log']).read_bytes())
        if raw_sha(out/'unit_tests.txt')!=ledger.start[plan['unit_test_log']]:raise AssertionError('Test archive copy changed')
        result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),scope=state['scope'],runtime_python=sys.version,old_checkpoint=old,registry_sha256=ledger.start[str(REGISTRY)],technical_receipts=len(receipts),final=final,development_replays=development,companion=notebook,final_reading=reading,compact_bundle=bundle,unit_tests=dict(**tests,archived_log_path=str(OUT/'unit_tests.txt'),log_sha256=raw_sha(out/'unit_tests.txt')),distinct_hashed_files=len(end),hash_checks=ledger.checks,checked_current_file_sha256=end,selected_source_and_output_sha256_at_start=dict(sorted(ledger.start.items())),selected_source_and_output_sha256_at_end=end,precommit_incident=dict(path=str(INCIDENT),sha256=ledger.start[str(INCIDENT)],source_hash_freeze_preceded_parse=True,git_commit_preceded_parse=False),old_checkpoint_overwritten=False,old_checks_rerun=False,new_market_parses=0,new_model_runs=0,new_test_runs=0,new_notebook_executions=0,prior_final_exposure='Previously exposed nonblind; hashes/audit/replay do not restore blind validity or prove future prediction.')
        writej(out/'verification.json',result);state.update(status='passed',completed_utc=result['completed_utc'],verification_sha256=raw_sha(out/'verification.json'));writej(out/'manifest.json',state)
    except BaseException as exc:
        (out/'failure.txt').write_text(traceback.format_exc());state.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_retry=False);writej(out/'manifest.json',state);raise
    print(json.dumps(dict(status='passed',hashed_files=len(end),tests=tests['tests_run'],output=str(out))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:run()
    else:print(json.dumps(dict(source_only=True,market_parse=False,new_model_run=False,old_checks_rerun=False)))
